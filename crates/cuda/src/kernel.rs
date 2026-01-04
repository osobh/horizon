//! CUDA kernel loading and compilation

use crate::error::{CudaError, CudaResult};
use crate::stream::Stream;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

/// Kernel compilation options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompileOptions {
    /// Target architecture (e.g., "sm_90")
    pub arch: String,
    /// Optimization level (0-3)
    pub opt_level: u8,
    /// Enable fast math
    pub fast_math: bool,
    /// Maximum registers per thread
    pub max_registers: Option<u32>,
    /// Additional compiler flags
    pub extra_flags: Vec<String>,
}

impl Default for CompileOptions {
    fn default() -> Self {
        Self {
            arch: "sm_80".to_string(), // Default to Ampere
            opt_level: 3,
            fast_math: true,
            max_registers: None,
            extra_flags: vec![],
        }
    }
}

/// Compiled kernel metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelMetadata {
    /// Kernel name
    pub name: String,
    /// Source type (CUDA C, PTX, etc.)
    pub source_type: SourceType,
    /// Compilation options used
    pub compile_options: CompileOptions,
    /// Register usage
    pub registers_used: u32,
    /// Shared memory usage
    pub shared_memory: usize,
    /// Constant memory usage
    pub constant_memory: usize,
    /// Local memory usage
    pub local_memory: usize,
    /// Maximum threads per block
    pub max_threads: u32,
}

/// Source code type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SourceType {
    /// CUDA C/C++ source
    CudaC,
    /// PTX assembly
    Ptx,
    /// CUBIN binary
    Cubin,
}

/// Kernel source code variants
#[derive(Debug, Clone)]
pub enum KernelSource {
    /// PTX assembly code
    Ptx(String),
    /// CUBIN binary data
    Cubin(Vec<u8>),
    /// CUDA C/C++ source code
    CudaC(String),
}

/// Kernel launch configuration
#[derive(Debug, Clone)]
pub struct LaunchConfig {
    /// Grid dimensions (x, y, z)
    pub grid_dim: (u32, u32, u32),
    /// Block dimensions (x, y, z)
    pub block_dim: (u32, u32, u32),
    /// Shared memory size in bytes
    pub shared_memory_bytes: usize,
    /// CUDA stream for asynchronous execution
    pub stream: Option<Arc<Stream>>,
}

impl Default for LaunchConfig {
    fn default() -> Self {
        Self {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_memory_bytes: 0,
            stream: None,
        }
    }
}

/// Launch configuration builder
pub struct LaunchConfigBuilder {
    config: LaunchConfig,
}

impl Default for LaunchConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl LaunchConfigBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            config: LaunchConfig::default(),
        }
    }

    /// Set grid dimensions
    pub fn grid_dim(mut self, x: u32, y: u32, z: u32) -> Self {
        self.config.grid_dim = (x, y, z);
        self
    }

    /// Set block dimensions
    pub fn block_dim(mut self, x: u32, y: u32, z: u32) -> Self {
        self.config.block_dim = (x, y, z);
        self
    }

    /// Set shared memory size
    pub fn shared_memory(mut self, bytes: usize) -> Self {
        self.config.shared_memory_bytes = bytes;
        self
    }

    /// Set stream
    pub fn stream(mut self, stream: Arc<Stream>) -> Self {
        self.config.stream = Some(stream);
        self
    }

    /// Build the configuration
    pub fn build(self) -> LaunchConfig {
        self.config
    }
}

/// Kernel arguments
#[derive(Debug, Clone)]
pub enum KernelArg {
    /// Scalar value
    Scalar(i32),
    /// Pointer argument
    Pointer(*mut u8),
}

/// Kernel attributes
#[derive(Debug, Clone)]
pub struct KernelAttributes {
    /// Maximum threads per block
    pub max_threads_per_block: i32,
    /// Shared memory per block
    pub shared_memory_bytes: i32,
    /// Constant memory size
    pub const_memory_bytes: i32,
    /// Local memory per thread
    pub local_memory_bytes: i32,
    /// Number of registers used
    pub num_registers: i32,
    /// PTX version
    pub ptx_version: i32,
    /// Binary version
    pub binary_version: i32,
}

/// Occupancy information
#[derive(Debug, Clone)]
pub struct OccupancyInfo {
    /// Number of active blocks
    pub active_blocks: i32,
    /// Number of active warps
    pub active_warps: i32,
    /// Number of active threads
    pub active_threads: i32,
    /// Occupancy percentage
    pub occupancy_percentage: f32,
}

/// Kernel profiling information
#[derive(Debug, Clone)]
pub struct ProfileInfo {
    /// Execution duration in nanoseconds
    pub duration_ns: u64,
    /// Memory transferred in bytes
    pub memory_transferred_bytes: i64,
}

/// Extended compilation options (for tests)
#[derive(Debug, Clone)]
pub struct CompilationOptions {
    /// Target architecture
    pub arch: String,
    /// Optimization level
    pub opt_level: u8,
    /// Enable debug info
    pub debug_info: bool,
    /// Verbose output
    pub verbose: bool,
    /// Max registers per thread
    pub max_registers: Option<u32>,
    /// Preprocessor defines
    pub defines: Vec<(String, String)>,
    /// Include paths
    pub include_paths: Vec<PathBuf>,
}

impl Default for CompilationOptions {
    fn default() -> Self {
        Self {
            arch: "sm_80".to_string(),
            opt_level: 3,
            debug_info: false,
            verbose: false,
            max_registers: None,
            defines: vec![],
            include_paths: vec![],
        }
    }
}

/// A compiled CUDA kernel
#[derive(Clone)]
pub struct Kernel {
    /// Unique kernel ID
    pub id: Uuid,
    /// Kernel metadata
    pub metadata: KernelMetadata,
    /// Compiled code (PTX or CUBIN)
    compiled_code: Vec<u8>,
    /// Whether this is a mock kernel
    is_mock: bool,
}

impl Kernel {
    /// Create a new kernel (mainly for testing)
    pub fn new(_name: String, code: Vec<u8>, metadata: KernelMetadata) -> Self {
        Self {
            id: Uuid::new_v4(),
            metadata,
            compiled_code: code,
            is_mock: cfg!(cuda_mock),
        }
    }

    /// Get kernel name
    pub fn name(&self) -> &str {
        &self.metadata.name
    }

    /// Get compiled code
    pub fn code(&self) -> &[u8] {
        &self.compiled_code
    }

    /// Check if kernel is mock
    pub fn is_mock(&self) -> bool {
        self.is_mock
    }

    /// Launch the kernel
    pub fn launch(&self, config: &LaunchConfig, _args: Vec<KernelArg>) -> CudaResult<()> {
        // Validate launch configuration
        if config.grid_dim.0 == 0 || config.grid_dim.1 == 0 || config.grid_dim.2 == 0 {
            return Err(CudaError::InvalidValue {
                parameter: "grid dimensions".to_string(),
            });
        }
        if config.block_dim.0 == 0 || config.block_dim.1 == 0 || config.block_dim.2 == 0 {
            return Err(CudaError::InvalidValue {
                parameter: "block dimensions".to_string(),
            });
        }

        #[cfg(cuda_mock)]
        {
            // Mock implementation - just validate and return success
            Ok(())
        }

        #[cfg(not(cuda_mock))]
        {
            // Real implementation would use cuLaunchKernel
            Err(CudaError::MockModeError)
        }
    }

    /// Calculate occupancy for given block size and dynamic shared memory
    pub fn calculate_occupancy(
        &self,
        block_size: u32,
        _dynamic_shared_mem: usize,
    ) -> CudaResult<OccupancyInfo> {
        #[cfg(cuda_mock)]
        {
            // Mock calculation
            let warps_per_block = block_size.div_ceil(32);
            let max_blocks_per_sm = 32; // Typical for modern GPUs
            let active_blocks = max_blocks_per_sm.min(2048 / block_size);
            let active_warps = active_blocks * warps_per_block;
            let active_threads = active_blocks * block_size;
            let max_warps_per_sm = 64; // Typical for modern GPUs
            let occupancy_percentage = (active_warps as f32 / max_warps_per_sm as f32) * 100.0;

            Ok(OccupancyInfo {
                active_blocks: active_blocks as i32,
                active_warps: active_warps as i32,
                active_threads: active_threads as i32,
                occupancy_percentage,
            })
        }

        #[cfg(not(cuda_mock))]
        {
            // Real implementation would use cuOccupancyMaxActiveBlocksPerMultiprocessor
            Err(CudaError::MockModeError)
        }
    }

    /// Get kernel attributes
    pub fn get_attributes(&self) -> CudaResult<KernelAttributes> {
        #[cfg(cuda_mock)]
        {
            Ok(KernelAttributes {
                max_threads_per_block: self.metadata.max_threads as i32,
                shared_memory_bytes: self.metadata.shared_memory as i32,
                const_memory_bytes: self.metadata.constant_memory as i32,
                local_memory_bytes: self.metadata.local_memory as i32,
                num_registers: self.metadata.registers_used as i32,
                ptx_version: 70,    // PTX 7.0
                binary_version: 80, // SM 8.0
            })
        }

        #[cfg(not(cuda_mock))]
        {
            // Real implementation would use cuFuncGetAttribute
            Err(CudaError::MockModeError)
        }
    }

    /// Get last profiling info (mock only)
    #[cfg(cuda_mock)]
    pub fn get_last_profile_info(&self) -> Option<ProfileInfo> {
        // In a real implementation, this would return actual profiling data
        Some(ProfileInfo {
            duration_ns: 1_000_000,                // 1ms
            memory_transferred_bytes: 1024 * 1024, // 1MB
        })
    }

    /// Get last profiling info (real implementation)
    #[cfg(not(cuda_mock))]
    pub fn get_last_profile_info(&self) -> Option<ProfileInfo> {
        None
    }
}

/// A CUDA module containing multiple kernels
pub struct KernelModule {
    /// Module ID
    pub id: Uuid,
    /// Module name
    pub name: String,
    /// Loaded kernels
    kernels: Arc<RwLock<HashMap<String, Arc<Kernel>>>>,
    /// Module handle (would be CUmodule in real implementation)
    #[allow(dead_code)]
    handle: Option<u64>,
}

impl KernelModule {
    /// Create a new kernel module
    pub fn new(name: String) -> Self {
        Self {
            id: Uuid::new_v4(),
            name,
            kernels: Arc::new(RwLock::new(HashMap::new())),
            handle: None,
        }
    }

    /// Create module from KernelSource
    pub fn from_source(source: KernelSource) -> CudaResult<Self> {
        let name = match &source {
            KernelSource::Ptx(_) => "ptx_module",
            KernelSource::Cubin(_) => "cubin_module",
            KernelSource::CudaC(_) => "cuda_c_module",
        }
        .to_string();

        Ok(Self {
            id: Uuid::new_v4(),
            name,
            kernels: Arc::new(RwLock::new(HashMap::new())),
            handle: None,
        })
    }

    /// Load module from PTX file
    pub fn from_ptx_file(path: PathBuf) -> CudaResult<Self> {
        #[cfg(cuda_mock)]
        {
            // Mock implementation - just check if file exists
            if !path.exists() {
                return Err(CudaError::FileNotFound {
                    path: path.to_string_lossy().to_string(),
                });
            }
            Self::from_source(KernelSource::Ptx("mock_ptx".to_string()))
        }

        #[cfg(not(cuda_mock))]
        {
            Err(CudaError::MockModeError)
        }
    }

    /// Load module from CUBIN file
    pub fn from_cubin_file(path: PathBuf) -> CudaResult<Self> {
        #[cfg(cuda_mock)]
        {
            // Mock implementation - just check if file exists
            if !path.exists() {
                return Err(CudaError::FileNotFound {
                    path: path.to_string_lossy().to_string(),
                });
            }
            Self::from_source(KernelSource::Cubin(vec![]))
        }

        #[cfg(not(cuda_mock))]
        {
            Err(CudaError::MockModeError)
        }
    }

    /// Compile CUDA C source code
    pub fn compile_cuda_source(_source: &str, _options: &CompilationOptions) -> CudaResult<Self> {
        #[cfg(cuda_mock)]
        {
            let module = Self::from_source(KernelSource::CudaC(_source.to_string()))?;
            Ok(module)
        }

        #[cfg(not(cuda_mock))]
        {
            Err(CudaError::MockModeError)
        }
    }

    /// Get kernel metadata (for tests)
    pub fn metadata(&self) -> KernelMetadata {
        KernelMetadata {
            name: self.name.clone(),
            source_type: SourceType::Ptx,
            compile_options: CompileOptions::default(),
            registers_used: 32,
            shared_memory: 1024,
            constant_memory: 0,
            local_memory: 0,
            max_threads: 1024,
        }
    }

    /// Get kernel by name (sync version for tests)
    pub fn get_kernel(&self, name: &str) -> CudaResult<Arc<Kernel>> {
        #[cfg(cuda_mock)]
        {
            // For mock mode, create kernel on demand if it doesn't exist
            let kernels = self.kernels.blocking_read();
            if let Some(kernel) = kernels.get(name) {
                return Ok(kernel.clone());
            }
            drop(kernels);

            // Create mock kernel
            let metadata = KernelMetadata {
                name: name.to_string(),
                source_type: SourceType::Ptx,
                compile_options: CompileOptions::default(),
                registers_used: 32,
                shared_memory: 1024,
                constant_memory: 0,
                local_memory: 0,
                max_threads: 1024,
            };

            let kernel = Arc::new(Kernel::new(name.to_string(), vec![], metadata));

            let mut kernels = self.kernels.blocking_write();
            kernels.insert(name.to_string(), kernel.clone());

            Ok(kernel)
        }

        #[cfg(not(cuda_mock))]
        {
            let kernels = self.kernels.blocking_read();
            kernels
                .get(name)
                .cloned()
                .ok_or_else(|| CudaError::KernelNotFound {
                    name: name.to_string(),
                })
        }
    }

    /// Load kernel from file
    pub async fn load_from_file<P: AsRef<Path>>(
        &self,
        path: P,
        kernel_name: &str,
        options: CompileOptions,
    ) -> CudaResult<Arc<Kernel>> {
        let path = path.as_ref();

        // Read source file
        let source =
            tokio::fs::read_to_string(path)
                .await
                .map_err(|e| CudaError::CompilationError {
                    message: format!("Failed to read source file: {e}"),
                })?;

        // Determine source type from extension
        let source_type = match path.extension().and_then(|s| s.to_str()) {
            Some("cu") | Some("cuh") => SourceType::CudaC,
            Some("ptx") => SourceType::Ptx,
            Some("cubin") => SourceType::Cubin,
            _ => {
                return Err(CudaError::CompilationError {
                    message: "Unknown source file type".to_string(),
                })
            }
        };

        self.compile_kernel(kernel_name, &source, source_type, options)
            .await
    }

    /// Compile kernel from source
    pub async fn compile_kernel(
        &self,
        kernel_name: &str,
        source: &str,
        source_type: SourceType,
        options: CompileOptions,
    ) -> CudaResult<Arc<Kernel>> {
        #[cfg(cuda_mock)]
        {
            self.compile_mock_kernel(kernel_name, source, source_type, options)
                .await
        }

        #[cfg(not(cuda_mock))]
        {
            self.compile_real_kernel(kernel_name, source, source_type, options)
                .await
        }
    }

    /// Compile mock kernel for testing
    #[cfg(cuda_mock)]
    async fn compile_mock_kernel(
        &self,
        kernel_name: &str,
        _source: &str,
        source_type: SourceType,
        options: CompileOptions,
    ) -> CudaResult<Arc<Kernel>> {
        // Simulate compilation delay
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

        // Create mock metadata
        let metadata = KernelMetadata {
            name: kernel_name.to_string(),
            source_type,
            compile_options: options,
            registers_used: 32,
            shared_memory: 1024,
            constant_memory: 0,
            local_memory: 0,
            max_threads: 1024,
        };

        // Create mock compiled code
        let compiled_code = format!("MOCK_PTX_FOR_{kernel_name}").into_bytes();

        let kernel = Arc::new(Kernel::new(
            kernel_name.to_string(),
            compiled_code,
            metadata,
        ));

        // Store in module
        let mut kernels = self.kernels.write().await;
        kernels.insert(kernel_name.to_string(), kernel.clone());

        Ok(kernel)
    }

    /// Compile real kernel using NVRTC
    #[cfg(not(cuda_mock))]
    async fn compile_real_kernel(
        &self,
        kernel_name: &str,
        source: &str,
        source_type: SourceType,
        options: CompileOptions,
    ) -> CudaResult<Arc<Kernel>> {
        use std::io::Write;
        use std::process::Command;
        use tempfile::NamedTempFile;

        match source_type {
            SourceType::CudaC => {
                // Write source to temporary file
                let mut source_file =
                    NamedTempFile::new().map_err(|e| CudaError::CompilationError {
                        message: format!("Failed to create temp file: {e}"),
                    })?;

                source_file.write_all(source.as_bytes()).map_err(|e| {
                    CudaError::CompilationError {
                        message: format!("Failed to write source: {e}"),
                    }
                })?;

                let source_path = source_file.path();

                // Create output file
                let output_file =
                    NamedTempFile::new().map_err(|e| CudaError::CompilationError {
                        message: format!("Failed to create output file: {e}"),
                    })?;

                // Build nvcc command
                let mut cmd = Command::new("nvcc");
                cmd.arg("-ptx")
                    .arg("-o")
                    .arg(output_file.path())
                    .arg(format!("-arch={}", options.arch));

                if options.fast_math {
                    cmd.arg("-use_fast_math");
                }

                cmd.arg(format!("-O{}", options.opt_level));

                if let Some(max_regs) = options.max_registers {
                    cmd.arg(format!("-maxrregcount={}", max_regs));
                }

                for flag in &options.extra_flags {
                    cmd.arg(flag);
                }

                cmd.arg(source_path);

                // Run compilation
                let output = cmd.output().map_err(|e| CudaError::CompilationError {
                    message: format!("Failed to run nvcc: {e}"),
                })?;

                if !output.status.success() {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    return Err(CudaError::CompilationError {
                        message: format!("Compilation failed: {stderr}"),
                    });
                }

                // Read compiled PTX
                let compiled_code = tokio::fs::read(output_file.path()).await.map_err(|e| {
                    CudaError::CompilationError {
                        message: format!("Failed to read compiled code: {e}"),
                    }
                })?;

                // Create metadata (would extract from real compilation)
                let metadata = KernelMetadata {
                    name: kernel_name.to_string(),
                    source_type,
                    compile_options: options,
                    registers_used: 32, // Placeholder
                    shared_memory: 0,
                    constant_memory: 0,
                    local_memory: 0,
                    max_threads: 1024,
                };

                let kernel = Arc::new(Kernel::new(
                    kernel_name.to_string(),
                    compiled_code,
                    metadata,
                ));

                let mut kernels = self.kernels.write().await;
                kernels.insert(kernel_name.to_string(), kernel.clone());

                Ok(kernel)
            }
            SourceType::Ptx => {
                // PTX is already compiled, just validate and store
                let metadata = KernelMetadata {
                    name: kernel_name.to_string(),
                    source_type,
                    compile_options: options,
                    registers_used: 32,
                    shared_memory: 0,
                    constant_memory: 0,
                    local_memory: 0,
                    max_threads: 1024,
                };

                let kernel = Arc::new(Kernel::new(
                    kernel_name.to_string(),
                    source.as_bytes().to_vec(),
                    metadata,
                ));

                let mut kernels = self.kernels.write().await;
                kernels.insert(kernel_name.to_string(), kernel.clone());

                Ok(kernel)
            }
            SourceType::Cubin => Err(CudaError::CompilationError {
                message: "CUBIN loading not yet implemented".to_string(),
            }),
        }
    }

    /// Get kernel by name (async version)
    pub async fn get_kernel_async(&self, name: &str) -> CudaResult<Arc<Kernel>> {
        let kernels = self.kernels.read().await;
        kernels
            .get(name)
            .cloned()
            .ok_or_else(|| CudaError::KernelNotFound {
                name: name.to_string(),
            })
    }

    /// List all kernels in module
    pub async fn list_kernels(&self) -> Vec<String> {
        let kernels = self.kernels.read().await;
        kernels.keys().cloned().collect()
    }

    /// Get module name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get kernel count
    pub async fn kernel_count(&self) -> usize {
        let kernels = self.kernels.read().await;
        kernels.len()
    }

    /// Get kernel by name (non-async version for tests)
    pub fn get_kernel_sync(&self, name: &str) -> Option<Arc<Kernel>> {
        let kernels = self.kernels.blocking_read();
        kernels.get(name).cloned()
    }

    /// Add a kernel to the module
    pub fn add_kernel(&mut self, kernel: Kernel) {
        let mut kernels = self.kernels.blocking_write();
        kernels.insert(kernel.name().to_string(), Arc::new(kernel));
    }

    /// Get all kernel names
    pub fn kernel_names(&self) -> Vec<String> {
        let kernels = self.kernels.blocking_read();
        kernels.keys().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compile_options_default() {
        let options = CompileOptions::default();
        assert_eq!(options.arch, "sm_80");
        assert_eq!(options.opt_level, 3);
        assert!(options.fast_math);
        assert!(options.max_registers.is_none());
        assert!(options.extra_flags.is_empty());
    }

    #[tokio::test]
    async fn test_kernel_module_creation() {
        let module = KernelModule::new("test_module".to_string());
        assert_eq!(module.name, "test_module");
        assert!(module.list_kernels().await.is_empty());
    }

    #[tokio::test]
    async fn test_compile_cuda_kernel() {
        let module = KernelModule::new("test_module".to_string());

        let source = r#"
        extern "C" __global__ void vector_add(float* a, float* b, float* c, int n) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                c[idx] = a[idx] + b[idx];
            }
        }
        "#;

        let options = CompileOptions::default();
        let kernel = module
            .compile_kernel("vector_add", source, SourceType::CudaC, options)
            .await
            .unwrap();

        assert_eq!(kernel.name(), "vector_add");
        assert_eq!(kernel.metadata.source_type, SourceType::CudaC);

        #[cfg(cuda_mock)]
        {
            assert!(kernel.is_mock());
            let code = String::from_utf8_lossy(kernel.code());
            assert!(code.contains("MOCK_PTX"));
        }
    }

    #[tokio::test]
    async fn test_compile_ptx_kernel() {
        let module = KernelModule::new("test_module".to_string());

        let ptx_source = r#"
        .version 7.0
        .target sm_80
        .address_size 64
        
        .visible .entry kernel_entry() {
            ret;
        }
        "#;

        let options = CompileOptions::default();
        let kernel = module
            .compile_kernel("kernel_entry", ptx_source, SourceType::Ptx, options)
            .await
            .unwrap();

        assert_eq!(kernel.name(), "kernel_entry");
        assert_eq!(kernel.metadata.source_type, SourceType::Ptx);
    }

    #[tokio::test]
    async fn test_get_kernel() {
        let module = KernelModule::new("test_module".to_string());

        // Compile a kernel
        let source = "extern \"C\" __global__ void test_kernel() {}";
        let options = CompileOptions::default();

        module
            .compile_kernel("test_kernel", source, SourceType::CudaC, options)
            .await?;

        // Get kernel
        let kernel = module.get_kernel("test_kernel").await.unwrap();
        assert_eq!(kernel.name(), "test_kernel");

        // Try to get non-existent kernel
        let result = module.get_kernel("nonexistent").await;
        assert!(matches!(result, Err(CudaError::KernelNotFound { .. })));
    }

    #[tokio::test]
    async fn test_list_kernels() {
        let module = KernelModule::new("test_module".to_string());

        // Initially empty
        assert!(module.list_kernels().await.is_empty());

        // Add kernels
        let options = CompileOptions::default();
        module
            .compile_kernel(
                "kernel1",
                "extern \"C\" __global__ void kernel1() {}",
                SourceType::CudaC,
                options.clone(),
            )
            .await
            .unwrap();

        module
            .compile_kernel(
                "kernel2",
                "extern \"C\" __global__ void kernel2() {}",
                SourceType::CudaC,
                options,
            )
            .await
            .unwrap();

        let kernels = module.list_kernels().await;
        assert_eq!(kernels.len(), 2);
        assert!(kernels.contains(&"kernel1".to_string()));
        assert!(kernels.contains(&"kernel2".to_string()));
    }
}
