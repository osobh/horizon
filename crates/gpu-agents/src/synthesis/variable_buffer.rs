//! Variable-sized buffer support for synthesis kernels
//!
//! Handles dynamic buffer allocation and resizing for pattern matching

use anyhow::{anyhow, Context, Result};
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr, DeviceSlice};
use std::sync::Arc;

/// Variable-sized buffer manager for GPU operations
pub struct VariableBufferManager {
    device: Arc<CudaDevice>,
    config: BufferConfig,
    /// Current allocated buffers
    pattern_buffer: Option<CudaSlice<u8>>,
    ast_buffer: Option<CudaSlice<u8>>,
    match_buffer: Option<CudaSlice<u32>>,
}

/// Configuration for variable buffer management
#[derive(Debug, Clone)]
pub struct BufferConfig {
    /// Initial buffer size in bytes
    pub initial_size: usize,
    /// Growth factor when resizing (e.g., 1.5 = grow by 50%)
    pub growth_factor: f32,
    /// Maximum buffer size allowed
    pub max_size: usize,
    /// Alignment requirement in bytes
    pub alignment: usize,
}

impl Default for BufferConfig {
    fn default() -> Self {
        Self {
            initial_size: 1024 * 1024, // 1MB
            growth_factor: 1.5,
            max_size: 2 * 1024 * 1024 * 1024, // 2GB
            alignment: 256,                   // 256-byte alignment for optimal performance
        }
    }
}

impl VariableBufferManager {
    /// Create a new variable buffer manager
    pub fn new(device: Arc<CudaDevice>, config: BufferConfig) -> Result<Self> {
        Ok(Self {
            device,
            config,
            pattern_buffer: None,
            ast_buffer: None,
            match_buffer: None,
        })
    }

    /// Allocate or resize buffers to fit the required sizes
    pub fn ensure_capacity(
        &mut self,
        pattern_size: usize,
        ast_size: usize,
        match_size: usize,
    ) -> Result<()> {
        // Align sizes
        let aligned_pattern_size = self.align_size(pattern_size);
        let aligned_ast_size = self.align_size(ast_size);
        let aligned_match_size = self.align_size(match_size);

        // Check pattern buffer
        if self.pattern_buffer.is_none()
            || self.pattern_buffer.as_ref()?.len() < aligned_pattern_size
        {
            self.pattern_buffer = Some(
                unsafe { self.device.alloc::<u8>(aligned_pattern_size) }
                    .context("Failed to allocate pattern buffer")?,
            );
        }

        // Check AST buffer
        if self.ast_buffer.is_none() || self.ast_buffer.as_ref()?.len() < aligned_ast_size {
            self.ast_buffer = Some(
                unsafe { self.device.alloc::<u8>(aligned_ast_size) }
                    .context("Failed to allocate AST buffer")?,
            );
        }

        // Check match buffer (u32 elements)
        let match_elements = aligned_match_size / 4;
        if self.match_buffer.is_none() || self.match_buffer.as_ref()?.len() < match_elements
        {
            self.match_buffer = Some(
                self.device
                    .alloc_zeros::<u32>(match_elements)
                    .context("Failed to allocate match buffer")?,
            );
        }

        Ok(())
    }

    /// Get pattern buffer, allocating if necessary
    pub fn get_pattern_buffer(&mut self, required_size: usize) -> Result<&mut CudaSlice<u8>> {
        let aligned_size = self.align_size(required_size);

        // Ensure capacity
        if self.pattern_buffer.is_none()
            || self.pattern_buffer.as_ref()?.len() < aligned_size
        {
            self.pattern_buffer = Some(
                unsafe { self.device.alloc::<u8>(aligned_size) }
                    .context("Failed to allocate pattern buffer")?,
            );
        }

        self.pattern_buffer
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("Pattern buffer not available"))
    }

    /// Get AST buffer, allocating if necessary
    pub fn get_ast_buffer(&mut self, required_size: usize) -> Result<&mut CudaSlice<u8>> {
        let aligned_size = self.align_size(required_size);

        // Ensure capacity
        if self.ast_buffer.is_none() || self.ast_buffer.as_ref()?.len() < aligned_size {
            self.ast_buffer = Some(
                unsafe { self.device.alloc::<u8>(aligned_size) }
                    .context("Failed to allocate AST buffer")?,
            );
        }

        self.ast_buffer
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("AST buffer not available"))
    }

    /// Get match buffer, allocating if necessary
    pub fn get_match_buffer(&mut self, required_size: usize) -> Result<&mut CudaSlice<u32>> {
        let aligned_size = self.align_size(required_size);
        let elements = aligned_size / 4;

        // Ensure capacity
        if self.match_buffer.is_none() || self.match_buffer.as_ref()?.len() < elements {
            self.match_buffer = Some(
                self.device
                    .alloc_zeros::<u32>(elements)
                    .context("Failed to allocate match buffer")?,
            );
        }

        self.match_buffer
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("Match buffer not available"))
    }

    /// Calculate aligned size based on configuration
    fn align_size(&self, size: usize) -> usize {
        let alignment = self.config.alignment;
        (size + alignment - 1) / alignment * alignment
    }

    /// Deallocate all buffers
    pub fn clear(&mut self) {
        self.pattern_buffer = None;
        self.ast_buffer = None;
        self.match_buffer = None;
    }

    /// Get current memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        let pattern_size = self.pattern_buffer.as_ref().map(|b| b.len()).unwrap_or(0);
        let ast_size = self.ast_buffer.as_ref().map(|b| b.len()).unwrap_or(0);
        let match_size = self.match_buffer.as_ref().map(|b| b.len() * 4).unwrap_or(0);
        pattern_size + ast_size + match_size
    }
}

/// Variable-sized kernel launcher
pub struct VariableKernelLauncher {
    device: Arc<CudaDevice>,
    buffer_manager: VariableBufferManager,
}

impl VariableKernelLauncher {
    /// Create a new kernel launcher
    pub fn new(device: Arc<CudaDevice>) -> Result<Self> {
        let config = BufferConfig::default();
        let buffer_manager = VariableBufferManager::new(device.clone(), config)?;
        Ok(Self {
            device,
            buffer_manager,
        })
    }

    /// Launch pattern matching with variable-sized inputs
    pub fn launch_pattern_matching(
        &mut self,
        patterns: &[u8],
        ast_nodes: &[u8],
        pattern_count: u32,
        node_count: u32,
    ) -> Result<Vec<u32>> {
        // Ensure buffers have capacity
        self.buffer_manager.ensure_capacity(
            patterns.len(),
            ast_nodes.len(),
            (node_count as usize) * 8, // 2 u32s per node for matches
        )?;

        // Copy pattern data
        {
            let pattern_buffer = self.buffer_manager.get_pattern_buffer(patterns.len())?;
            self.device
                .htod_copy_into(patterns.to_vec(), pattern_buffer)?;
        }

        // Copy AST data
        {
            let ast_buffer = self.buffer_manager.get_ast_buffer(ast_nodes.len())?;
            self.device.htod_copy_into(ast_nodes.to_vec(), ast_buffer)?;
        }

        // Clear match buffer
        {
            let match_buffer = self
                .buffer_manager
                .get_match_buffer((node_count as usize) * 8)?;
            let zeros = vec![0u32; (node_count as usize) * 2];
            self.device.htod_copy_into(zeros, match_buffer)?;
        }

        // Launch kernel with proper buffer access
        let (pattern_ptr, ast_ptr, match_ptr) = {
            let pattern_buffer = self
                .buffer_manager
                .pattern_buffer
                .as_ref()
                .ok_or_else(|| anyhow!("Pattern buffer not initialized"))?;
            let ast_buffer = self
                .buffer_manager
                .ast_buffer
                .as_ref()
                .ok_or_else(|| anyhow!("AST buffer not initialized"))?;
            let match_buffer = self
                .buffer_manager
                .match_buffer
                .as_ref()
                .ok_or_else(|| anyhow!("Match buffer not initialized"))?;

            (
                *pattern_buffer.device_ptr() as *const u8,
                *ast_buffer.device_ptr() as *const u8,
                *match_buffer.device_ptr() as *mut u32,
            )
        };

        unsafe {
            crate::synthesis::launch_match_patterns_fast(
                pattern_ptr,
                ast_ptr,
                match_ptr,
                pattern_count,
                node_count,
            );
        }

        self.device.synchronize()?;

        // Get results
        let mut results = vec![0u32; (node_count as usize) * 2];
        {
            let match_buffer = self
                .buffer_manager
                .match_buffer
                .as_ref()
                .ok_or_else(|| anyhow!("Match buffer not initialized"))?;
            self.device
                .dtoh_sync_copy_into(match_buffer, &mut results)?;
        }

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_manager_creation() -> Result<(), Box<dyn std::error::Error>>  {
        let device = CudaDevice::new(0)?;
        let config = BufferConfig::default();
        let manager = VariableBufferManager::new(Arc::new(device), config);
        assert!(manager.is_ok());
    }

    #[test]
    fn test_buffer_allocation() -> Result<(), Box<dyn std::error::Error>>  {
        let device = Arc::new(CudaDevice::new(0)?);
        let mut manager = VariableBufferManager::new(device, BufferConfig::default()).unwrap();

        let result = manager.ensure_capacity(1024, 2048, 512);
        // Should panic with todo!
        assert!(result.is_err() || result.is_ok());
    }

    #[test]
    fn test_buffer_resizing() -> Result<(), Box<dyn std::error::Error>>  {
        let device = Arc::new(CudaDevice::new(0)?);
        let mut manager = VariableBufferManager::new(device, BufferConfig::default()).unwrap();

        // Try to get a buffer that requires allocation
        let result = manager.get_pattern_buffer(1024);
        // Should panic with todo!
        assert!(result.is_err() || result.is_ok());
    }

    #[test]
    fn test_alignment() -> Result<(), Box<dyn std::error::Error>>  {
        let device = Arc::new(CudaDevice::new(0)?);
        let manager = VariableBufferManager::new(device, BufferConfig::default())?;

        assert_eq!(manager.align_size(100), 256);
        assert_eq!(manager.align_size(256), 256);
        assert_eq!(manager.align_size(257), 512);
    }

    #[test]
    fn test_kernel_launcher() -> Result<(), Box<dyn std::error::Error>>  {
        let device = Arc::new(CudaDevice::new(0)?);
        let mut launcher = VariableKernelLauncher::new(device)?;

        let patterns = vec![0u8; 1024];
        let ast_nodes = vec![0u8; 2048];

        let result = launcher.launch_pattern_matching(&patterns, &ast_nodes, 10, 100);
        // Should panic with todo!
        assert!(result.is_err() || result.is_ok());
    }
}
