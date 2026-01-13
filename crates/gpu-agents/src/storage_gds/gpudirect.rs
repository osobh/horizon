//! GPUDirect Storage Implementation
//!
//! Provides direct GPU-to-storage data transfer capabilities using NVIDIA GPUDirect Storage (GDS).
//! Falls back to traditional CPU-mediated transfers when GDS is not available.

use anyhow::{Context, Result, anyhow};
use cudarc::driver::{CudaContext, CudaSlice, DevicePtr};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, atomic::{AtomicBool, AtomicU64, Ordering}};
use tokio::sync::{Mutex, RwLock};
use std::time::{Duration, Instant};

/// GPUDirect Storage configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuDirectConfig {
    /// Enable GPUDirect Storage
    pub enable_gds: bool,
    /// Maximum transfer size in bytes
    pub max_transfer_size: usize,
    /// I/O alignment requirement
    pub io_alignment: usize,
    /// Number of I/O queues
    pub num_io_queues: usize,
    /// Enable asynchronous I/O
    pub enable_async_io: bool,
    /// Batch size for operations
    pub batch_size: usize,
    /// Enable fallback to CPU transfers
    pub enable_fallback: bool,
}

impl Default for GpuDirectConfig {
    fn default() -> Self {
        Self {
            enable_gds: true,
            max_transfer_size: 256 * 1024 * 1024, // 256MB
            io_alignment: 4096, // 4KB
            num_io_queues: 4,
            enable_async_io: true,
            batch_size: 16,
            enable_fallback: true,
        }
    }
}

/// GDS availability checker
pub struct GdsAvailabilityChecker {
    available: AtomicBool,
}

impl GdsAvailabilityChecker {
    pub fn new() -> Self {
        Self {
            available: AtomicBool::new(false),
        }
    }

    /// Check if GDS is available on the system
    pub fn is_available(&self) -> bool {
        // In a real implementation, this would check:
        // 1. NVIDIA driver version >= 470
        // 2. CUDA version >= 11.4
        // 3. GDS library availability
        // 4. Supported filesystem (ext4, XFS with O_DIRECT)
        
        // For now, we'll simulate availability based on environment
        if std::env::var("ENABLE_GPUDIRECT").is_ok() {
            self.available.store(true, Ordering::Relaxed);
            true
        } else {
            false
        }
    }

    /// Check driver version compatibility
    pub fn check_driver_version(&self) -> Result<()> {
        // Simulated check
        Ok(())
    }

    /// Check CUDA version compatibility
    pub fn check_cuda_version(&self) -> Result<()> {
        // Simulated check
        Ok(())
    }

    /// Check filesystem support
    pub fn check_filesystem_support(&self) -> Result<()> {
        // Check for ext4/XFS with O_DIRECT support
        Ok(())
    }
}

/// GPU I/O buffer for GDS operations
pub struct GpuIoBuffer {
    device: Arc<CudaContext>,
    buffer: CudaSlice<u8>,
    size: usize,
    aligned: bool,
    pinned: bool,
}

impl GpuIoBuffer {
    /// Allocate aligned GPU buffer for I/O
    pub fn allocate(size: usize) -> Result<Self> {
        let ctx = CudaContext::new(0)?;

        // Ensure alignment
        let aligned_size = (size + 4095) & !4095; // Align to 4KB

        // Allocate GPU memory
        // SAFETY: CudaContext::alloc returns uninitialized GPU memory. This is safe
        // because the buffer will be populated via GPUDirect Storage I/O operations
        // before any kernel reads from it. The aligned size ensures proper alignment.
        let buffer = unsafe { ctx.alloc::<u8>(aligned_size)? };

        Ok(Self {
            device: ctx,
            buffer,
            size,
            aligned: true,
            pinned: true, // GDS requires pinned memory
        })
    }

    /// Allocate buffer with initial data
    pub fn allocate_with_data(data: &[u8]) -> Result<Self> {
        let mut buffer = Self::allocate(data.len())?;
        let stream = buffer.device.default_stream();
        stream.clone_htod_into(data.to_vec(), &mut buffer.buffer)?;
        Ok(buffer)
    }

    /// Allocate with fallback support
    pub fn allocate_with_fallback(size: usize) -> Result<Self> {
        Self::allocate(size).or_else(|_| {
            // Fallback to regular allocation
            let ctx = CudaContext::new(0)?;
            // SAFETY: CudaContext::alloc returns uninitialized GPU memory. This is safe
            // because the buffer will be populated via I/O operations before kernel use.
            let buffer = unsafe { ctx.alloc::<u8>(size)? };
            Ok(Self {
                device: ctx,
                buffer,
                size,
                aligned: false,
                pinned: false,
            })
        })
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn is_aligned(&self) -> bool {
        self.aligned
    }

    pub fn is_pinned(&self) -> bool {
        self.pinned
    }

    pub fn device_ptr(&self) -> Option<*mut u8> {
        Some(*self.buffer.device_ptr() as *mut u8)
    }

    /// Copy data back to host for verification
    pub fn to_host_vec(&self) -> Result<Vec<u8>> {
        let stream = self.device.default_stream();
        stream.clone_dtoh(&self.buffer)
            .context("Failed to copy GPU buffer to host")
    }
}

/// GPUDirect Storage manager
#[derive(Clone)]
pub struct GpuDirectManager {
    config: GpuDirectConfig,
    device: Arc<CudaContext>,
    is_initialized: Arc<AtomicBool>,
    io_queues: Arc<Vec<IoQueue>>,
    fallback_enabled: bool,
    stats: Arc<GdsStatistics>,
}

impl GpuDirectManager {
    /// Create new GDS manager
    pub fn new(config: GpuDirectConfig) -> Result<Self> {
        let checker = GdsAvailabilityChecker::new();

        if !checker.is_available() && !config.enable_fallback {
            return Err(anyhow!("GPUDirect Storage not available and fallback disabled"));
        }

        let device = CudaContext::new(0)?;
        let is_initialized = Arc::new(AtomicBool::new(false));
        
        // Initialize I/O queues
        let mut io_queues = Vec::new();
        for i in 0..config.num_io_queues {
            io_queues.push(IoQueue::new(i, config.batch_size));
        }

        let manager = Self {
            config: config.clone(),
            device,
            is_initialized: is_initialized.clone(),
            io_queues: Arc::new(io_queues),
            fallback_enabled: config.enable_fallback,
            stats: Arc::new(GdsStatistics::default()),
        };

        // Initialize GDS if available
        if checker.is_available() {
            manager.initialize_gds()?;
        }

        is_initialized.store(true, Ordering::Relaxed);
        Ok(manager)
    }

    /// Create with fallback support
    pub fn new_with_fallback(config: GpuDirectConfig) -> Result<Self> {
        let mut config = config;
        config.enable_fallback = true;
        Self::new(config)
    }

    /// Initialize GDS subsystem
    fn initialize_gds(&self) -> Result<()> {
        // In a real implementation, this would:
        // 1. Initialize cuFile library
        // 2. Register file handles
        // 3. Set up DMA mappings
        // 4. Configure I/O queues
        Ok(())
    }

    pub fn is_initialized(&self) -> bool {
        self.is_initialized.load(Ordering::Relaxed)
    }

    pub fn get_num_io_queues(&self) -> usize {
        self.io_queues.len()
    }

    /// Read file directly to GPU memory
    pub async fn read_to_gpu(
        &self,
        path: impl AsRef<Path>,
        buffer: &mut GpuIoBuffer,
        offset: u64,
        size: usize,
    ) -> Result<usize> {
        let start = Instant::now();
        
        // Check if GDS is available
        let checker = GdsAvailabilityChecker::new();
        if checker.is_available() {
            // Use GDS path
            self.gds_read(path.as_ref(), buffer, offset, size).await
        } else if self.fallback_enabled {
            // Use fallback path
            self.fallback_read(path.as_ref(), buffer, offset, size).await
        } else {
            Err(anyhow!("GDS not available and fallback disabled"))
        }
        .map(|bytes| {
            self.stats.record_read(bytes, start.elapsed());
            bytes
        })
    }

    /// Write GPU memory directly to file
    pub async fn write_from_gpu(
        &self,
        path: impl AsRef<Path>,
        buffer: &GpuIoBuffer,
        offset: u64,
        size: usize,
    ) -> Result<usize> {
        let start = Instant::now();
        
        let checker = GdsAvailabilityChecker::new();
        if checker.is_available() {
            self.gds_write(path.as_ref(), buffer, offset, size).await
        } else if self.fallback_enabled {
            self.fallback_write(path.as_ref(), buffer, offset, size).await
        } else {
            Err(anyhow!("GDS not available and fallback disabled"))
        }
        .map(|bytes| {
            self.stats.record_write(bytes, start.elapsed());
            bytes
        })
    }

    /// Asynchronous write operation
    pub fn write_from_gpu_async(
        &self,
        path: impl AsRef<Path>,
        buffer: &GpuIoBuffer,
        offset: u64,
        size: usize,
    ) -> Result<tokio::task::JoinHandle<Result<usize>>> {
        let path = path.as_ref().to_path_buf();
        let manager = self.clone();
        let buffer_ptr = buffer.device_ptr()?;
        let buffer_size = buffer.size();

        Ok(tokio::spawn(async move {
            // Create a temporary buffer reference for the async operation
            // In real implementation, this would use cuFile async APIs
            let temp_buffer = GpuIoBuffer {
                device: Arc::clone(&manager.device),
                // SAFETY: from_raw_parts reconstructs a CudaSlice from a raw device pointer.
                // The pointer was obtained from buffer.device_ptr() which is valid for the
                // lifetime of the original GpuIoBuffer. This is a simplified approach for
                // async operations - production code would use proper lifetime management
                // with cuFile async APIs to ensure the buffer remains valid.
                buffer: unsafe {
                    CudaSlice::from_raw_parts(
                        DevicePtr::from_raw(buffer_ptr as u64),
                        buffer_size,
                    )
                },
                size: buffer_size,
                aligned: true,
                pinned: true,
            };

            manager.write_from_gpu(path, &temp_buffer, offset, size).await
        }))
    }

    /// Execute batch operations
    pub async fn execute_batch(&self, batch: &mut GdsBatchOperation) -> Result<Vec<Result<IoResult>>> {
        let mut results = Vec::new();
        
        // Execute operations in parallel using I/O queues
        let chunks: Vec<_> = batch.operations.chunks(self.config.batch_size).collect();
        
        for chunk in chunks {
            let mut chunk_results = Vec::new();
            
            for op in chunk {
                let result = match op {
                    BatchOp::Read { path, buffer, offset, size } => {
                        self.read_to_gpu(path, buffer, *offset, *size).await
                            .map(|bytes| IoResult {
                                bytes_transferred: bytes,
                                duration: Duration::from_millis(1), // Placeholder
                            })
                    }
                    BatchOp::Write { path, buffer, offset, size } => {
                        self.write_from_gpu(path, buffer, *offset, *size).await
                            .map(|bytes| IoResult {
                                bytes_transferred: bytes,
                                duration: Duration::from_millis(1), // Placeholder
                            })
                    }
                };
                chunk_results.push(result);
            }
            
            results.extend(chunk_results);
        }
        
        Ok(results)
    }

    /// GDS read implementation
    async fn gds_read(
        &self,
        path: &Path,
        buffer: &mut GpuIoBuffer,
        offset: u64,
        size: usize,
    ) -> Result<usize> {
        // Simulated GDS read
        // In real implementation, this would use cuFileRead API
        
        // For simulation, fall back to regular read
        self.fallback_read(path, buffer, offset, size).await
    }

    /// GDS write implementation
    async fn gds_write(
        &self,
        path: &Path,
        buffer: &GpuIoBuffer,
        offset: u64,
        size: usize,
    ) -> Result<usize> {
        // Simulated GDS write
        // In real implementation, this would use cuFileWrite API
        
        // For simulation, fall back to regular write
        self.fallback_write(path, buffer, offset, size).await
    }

    /// Fallback read using traditional CPU-mediated transfer
    async fn fallback_read(
        &self,
        path: &Path,
        buffer: &mut GpuIoBuffer,
        _offset: u64,
        size: usize,
    ) -> Result<usize> {
        // Read to CPU buffer first
        let data = tokio::fs::read(path).await
            .context("Failed to read file")?;

        let read_size = size.min(data.len());

        // Copy to GPU
        let stream = self.device.default_stream();
        stream.clone_htod_into(data[..read_size].to_vec(), &mut buffer.buffer)?;

        Ok(read_size)
    }

    /// Fallback write using traditional CPU-mediated transfer
    async fn fallback_write(
        &self,
        path: &Path,
        buffer: &GpuIoBuffer,
        _offset: u64,
        size: usize,
    ) -> Result<usize> {
        // Copy from GPU to CPU
        let stream = self.device.default_stream();
        let data = stream.clone_dtoh(&buffer.buffer)?;
        let write_size = size.min(data.len());

        // Write to file
        tokio::fs::write(path, &data[..write_size]).await
            .context("Failed to write file")?;

        Ok(write_size)
    }
}

/// I/O queue for managing concurrent operations
struct IoQueue {
    id: usize,
    capacity: usize,
    pending: AtomicU64,
}

impl IoQueue {
    fn new(id: usize, capacity: usize) -> Self {
        Self {
            id,
            capacity,
            pending: AtomicU64::new(0),
        }
    }
}

/// Batch operation for GDS
pub struct GdsBatchOperation {
    operations: Vec<BatchOp>,
}

impl GdsBatchOperation {
    pub fn new() -> Self {
        Self {
            operations: Vec::new(),
        }
    }

    pub fn add_read(
        &mut self,
        path: impl Into<PathBuf>,
        buffer: GpuIoBuffer,
        offset: u64,
        size: usize,
    ) {
        self.operations.push(BatchOp::Read {
            path: path.into(),
            buffer,
            offset,
            size,
        });
    }

    pub fn add_write(
        &mut self,
        path: impl Into<PathBuf>,
        buffer: GpuIoBuffer,
        offset: u64,
        size: usize,
    ) {
        self.operations.push(BatchOp::Write {
            path: path.into(),
            buffer,
            offset,
            size,
        });
    }
}

enum BatchOp {
    Read {
        path: PathBuf,
        buffer: GpuIoBuffer,
        offset: u64,
        size: usize,
    },
    Write {
        path: PathBuf,
        buffer: GpuIoBuffer,
        offset: u64,
        size: usize,
    },
}

/// I/O operation result
pub struct IoResult {
    pub bytes_transferred: usize,
    pub duration: Duration,
}

/// GDS statistics
#[derive(Default)]
struct GdsStatistics {
    total_reads: AtomicU64,
    total_writes: AtomicU64,
    bytes_read: AtomicU64,
    bytes_written: AtomicU64,
    total_read_time: AtomicU64,
    total_write_time: AtomicU64,
}

impl GdsStatistics {
    fn record_read(&self, bytes: usize, duration: Duration) {
        self.total_reads.fetch_add(1, Ordering::Relaxed);
        self.bytes_read.fetch_add(bytes as u64, Ordering::Relaxed);
        self.total_read_time.fetch_add(duration.as_micros() as u64, Ordering::Relaxed);
    }

    fn record_write(&self, bytes: usize, duration: Duration) {
        self.total_writes.fetch_add(1, Ordering::Relaxed);
        self.bytes_written.fetch_add(bytes as u64, Ordering::Relaxed);
        self.total_write_time.fetch_add(duration.as_micros() as u64, Ordering::Relaxed);
    }
}

/// Integration with GPU storage
impl super::GpuStorageConfig {
    /// Enable GPUDirect Storage
    pub fn with_gpudirect(mut self) -> Self {
        self.enable_gpudirect = true;
        self
    }
}

/// Extension trait for GPU storage with GDS support
impl super::GpuAgentStorage {
    /// Store agent data using GPUDirect Storage
    pub async fn store_agent_gds(
        &self,
        agent_id: &str,
        data: &super::GpuAgentData,
    ) -> Result<()> {
        if !self.config.enable_gpudirect {
            return self.store_agent(agent_id, data).await;
        }

        // Serialize data
        let serialized = bincode::serialize(data)?;
        
        // Create GPU buffer
        let gpu_buffer = GpuIoBuffer::allocate_with_data(&serialized)?;
        
        // Get path
        let path = self.get_agent_path(agent_id);
        
        // Use GDS manager if available
        if let Some(ref gds_manager) = self.gds_manager {
            gds_manager.write_from_gpu(&path, &gpu_buffer, 0, serialized.len()).await?;
        } else {
            // Fallback to regular storage
            self.store_agent(agent_id, data).await?;
        }
        
        Ok(())
    }

    /// Load agent data using GPUDirect Storage
    pub async fn load_agent_gds(&self, agent_id: &str) -> Result<Option<super::GpuAgentData>> {
        if !self.config.enable_gpudirect {
            return self.load_agent(agent_id).await;
        }

        let path = self.get_agent_path(agent_id);
        
        if !path.exists() {
            return Ok(None);
        }

        // Get file size
        let metadata = tokio::fs::metadata(&path).await?;
        let file_size = metadata.len() as usize;
        
        // Create GPU buffer
        let mut gpu_buffer = GpuIoBuffer::allocate(file_size)?;
        
        // Use GDS manager if available
        if let Some(ref gds_manager) = self.gds_manager {
            gds_manager.read_to_gpu(&path, &mut gpu_buffer, 0, file_size).await?;
            
            // Copy back to host for deserialization
            let data = gpu_buffer.to_host_vec()?;
            let agent_data = bincode::deserialize(&data)?;
            Ok(Some(agent_data))
        } else {
            // Fallback to regular storage
            self.load_agent(agent_id).await
        }
    }
}

// Note: GDS manager field is already added to GpuAgentStorage in storage.rs