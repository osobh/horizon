//! GPU kernel management for query acceleration
//!
//! This module handles loading, compiling, and managing GPU kernels
//! for accelerated graph operations.

use crate::error::KnowledgeGraphResult;
use stratoswarm_cuda::{
    kernel::{CompileOptions, KernelMetadata, SourceType},
    Kernel,
};
use std::collections::HashMap;

/// GPU kernel manager for graph operations
pub struct GpuKernelManager {
    /// Available kernels indexed by name
    kernels: HashMap<String, Kernel>,
    /// Whether GPU is available
    gpu_available: bool,
}

impl GpuKernelManager {
    /// Create a new GPU kernel manager
    pub async fn new(gpu_enabled: bool) -> KnowledgeGraphResult<Self> {
        let mut manager = Self {
            kernels: HashMap::new(),
            gpu_available: gpu_enabled,
        };

        if gpu_enabled {
            manager.load_all_kernels().await?;
        }

        Ok(manager)
    }

    /// Load all available GPU kernels
    async fn load_all_kernels(&mut self) -> KnowledgeGraphResult<()> {
        self.load_path_finding_kernel().await?;
        self.load_neighborhood_kernel().await?;
        self.load_pagerank_kernel().await?;
        Ok(())
    }

    /// Load path finding kernel
    async fn load_path_finding_kernel(&mut self) -> KnowledgeGraphResult<()> {
        let kernel = Kernel::new(
            "graph_path_kernel".to_string(),
            vec![], // Empty code for mock implementation
            KernelMetadata {
                name: "graph_path_kernel".to_string(),
                source_type: SourceType::Ptx,
                compile_options: CompileOptions::default(),
                registers_used: 32,
                shared_memory: 1024,
                constant_memory: 0,
                local_memory: 0,
                max_threads: 256,
            },
        );

        self.kernels.insert("path_finding".to_string(), kernel);
        Ok(())
    }

    /// Load neighborhood kernel
    async fn load_neighborhood_kernel(&mut self) -> KnowledgeGraphResult<()> {
        let kernel = Kernel::new(
            "graph_neighborhood_kernel".to_string(),
            vec![], // Empty code for mock implementation
            KernelMetadata {
                name: "graph_neighborhood_kernel".to_string(),
                source_type: SourceType::Ptx,
                compile_options: CompileOptions::default(),
                registers_used: 24,
                shared_memory: 512,
                constant_memory: 0,
                local_memory: 0,
                max_threads: 128,
            },
        );

        self.kernels.insert("neighborhood".to_string(), kernel);
        Ok(())
    }

    /// Load PageRank kernel
    async fn load_pagerank_kernel(&mut self) -> KnowledgeGraphResult<()> {
        let kernel = Kernel::new(
            "graph_pagerank_kernel".to_string(),
            vec![], // Empty code for mock implementation
            KernelMetadata {
                name: "graph_pagerank_kernel".to_string(),
                source_type: SourceType::Ptx,
                compile_options: CompileOptions::default(),
                registers_used: 48,
                shared_memory: 2048,
                constant_memory: 0,
                local_memory: 0,
                max_threads: 512,
            },
        );

        self.kernels.insert("pagerank".to_string(), kernel);
        Ok(())
    }

    /// Get a kernel by name
    pub fn get_kernel(&self, name: &str) -> Option<&Kernel> {
        self.kernels.get(name)
    }

    /// Check if GPU is available
    pub fn is_gpu_available(&self) -> bool {
        self.gpu_available
    }

    /// Check if a specific kernel is available
    pub fn has_kernel(&self, name: &str) -> bool {
        self.kernels.contains_key(name)
    }

    /// Get all available kernel names
    pub fn available_kernels(&self) -> Vec<String> {
        self.kernels.keys().cloned().collect()
    }

    /// Get kernel metadata
    pub fn get_kernel_metadata(&self, name: &str) -> Option<&KernelMetadata> {
        self.kernels.get(name).map(|k| &k.metadata)
    }

    /// Reload a specific kernel
    pub async fn reload_kernel(&mut self, name: &str) -> KnowledgeGraphResult<bool> {
        match name {
            "path_finding" => {
                self.load_path_finding_kernel().await?;
                Ok(true)
            }
            "neighborhood" => {
                self.load_neighborhood_kernel().await?;
                Ok(true)
            }
            "pagerank" => {
                self.load_pagerank_kernel().await?;
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    /// Get kernel statistics
    pub fn get_kernel_stats(&self, name: &str) -> Option<KernelStats> {
        self.get_kernel_metadata(name).map(|metadata| KernelStats {
            name: metadata.name.clone(),
            registers_used: metadata.registers_used,
            shared_memory: metadata.shared_memory,
            max_threads: metadata.max_threads,
            source_type: metadata.source_type.clone(),
        })
    }

    /// Validate all kernels are properly loaded
    pub fn validate_kernels(&self) -> KnowledgeGraphResult<()> {
        let expected_kernels = ["path_finding", "neighborhood", "pagerank"];

        for kernel_name in &expected_kernels {
            if !self.has_kernel(kernel_name) {
                return Err(crate::error::KnowledgeGraphError::Other(format!(
                    "Required kernel '{}' is not loaded",
                    kernel_name
                )));
            }
        }

        Ok(())
    }
}

/// Kernel performance statistics
#[derive(Debug, Clone)]
pub struct KernelStats {
    /// Kernel name
    pub name: String,
    /// Number of registers used
    pub registers_used: u32,
    /// Shared memory usage in bytes
    pub shared_memory: usize,
    /// Maximum threads per block
    pub max_threads: u32,
    /// Source type
    pub source_type: SourceType,
}

impl KernelStats {
    /// Calculate kernel occupancy estimate
    pub fn estimated_occupancy(&self) -> f32 {
        // Simple occupancy estimation based on resource usage
        let register_occupancy = 1.0 - (self.registers_used as f32 / 255.0).min(1.0);
        let memory_occupancy = 1.0 - (self.shared_memory as f32 / 49152.0).min(1.0);

        (register_occupancy + memory_occupancy) / 2.0
    }

    /// Check if kernel is resource-intensive
    pub fn is_resource_intensive(&self) -> bool {
        self.registers_used > 32 || self.shared_memory > 1024
    }
}

/// GPU kernel execution context
pub struct KernelExecutionContext {
    /// Grid dimensions
    pub grid_dim: (u32, u32, u32),
    /// Block dimensions
    pub block_dim: (u32, u32, u32),
    /// Shared memory size
    pub shared_memory: usize,
}

impl KernelExecutionContext {
    /// Create a new execution context
    pub fn new(grid_dim: (u32, u32, u32), block_dim: (u32, u32, u32)) -> Self {
        Self {
            grid_dim,
            block_dim,
            shared_memory: 0,
        }
    }

    /// Set shared memory size
    pub fn with_shared_memory(mut self, size: usize) -> Self {
        self.shared_memory = size;
        self
    }

    /// Calculate total threads
    pub fn total_threads(&self) -> u32 {
        self.grid_dim.0
            * self.grid_dim.1
            * self.grid_dim.2
            * self.block_dim.0
            * self.block_dim.1
            * self.block_dim.2
    }

    /// Validate execution parameters
    pub fn validate(&self) -> bool {
        // Basic validation of execution parameters
        self.grid_dim.0 > 0
            && self.grid_dim.1 > 0
            && self.grid_dim.2 > 0
            && self.block_dim.0 > 0
            && self.block_dim.1 > 0
            && self.block_dim.2 > 0
            && self.block_dim.0 * self.block_dim.1 * self.block_dim.2 <= 1024
    }
}

impl Default for KernelExecutionContext {
    fn default() -> Self {
        Self::new((1, 1, 1), (256, 1, 1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_gpu_kernel_manager_creation() {
        // Test CPU-only creation
        let cpu_manager = GpuKernelManager::new(false).await;
        assert!(cpu_manager.is_ok());

        let manager = cpu_manager?;
        assert!(!manager.is_gpu_available());
        assert_eq!(manager.available_kernels().len(), 0);

        // Test GPU-enabled creation
        let gpu_manager = GpuKernelManager::new(true).await;
        assert!(gpu_manager.is_ok());

        let manager = gpu_manager.unwrap();
        assert!(manager.is_gpu_available());
        assert_eq!(manager.available_kernels().len(), 3);
    }

    #[tokio::test]
    async fn test_kernel_availability() {
        let mut manager = GpuKernelManager::new(true).await.unwrap();

        assert!(manager.has_kernel("path_finding"));
        assert!(manager.has_kernel("neighborhood"));
        assert!(manager.has_kernel("pagerank"));
        assert!(!manager.has_kernel("nonexistent"));

        let available = manager.available_kernels();
        assert!(available.contains(&"path_finding".to_string()));
        assert!(available.contains(&"neighborhood".to_string()));
        assert!(available.contains(&"pagerank".to_string()));
    }

    #[tokio::test]
    async fn test_kernel_metadata() {
        let manager = GpuKernelManager::new(true).await.unwrap();

        let metadata = manager.get_kernel_metadata("path_finding");
        assert!(metadata.is_some());

        let metadata = metadata?;
        assert_eq!(metadata.name, "graph_path_kernel");
        assert_eq!(metadata.registers_used, 32);
        assert_eq!(metadata.shared_memory, 1024);
    }

    #[tokio::test]
    async fn test_kernel_stats() {
        let manager = GpuKernelManager::new(true).await.unwrap();

        let stats = manager.get_kernel_stats("path_finding");
        assert!(stats.is_some());

        let stats = stats?;
        assert_eq!(stats.name, "graph_path_kernel");
        assert!(stats.estimated_occupancy() > 0.0);
        assert!(stats.estimated_occupancy() <= 1.0);
    }

    #[tokio::test]
    async fn test_kernel_reload() {
        let mut manager = GpuKernelManager::new(true).await.unwrap();

        let result = manager.reload_kernel("path_finding").await;
        assert!(result.is_ok());
        assert!(result?);

        let result = manager.reload_kernel("nonexistent").await;
        assert!(result.is_ok());
        assert!(!result?);
    }

    #[tokio::test]
    async fn test_kernel_validation() {
        let manager = GpuKernelManager::new(true).await.unwrap();
        let result = manager.validate_kernels();
        assert!(result.is_ok());

        let cpu_manager = GpuKernelManager::new(false).await?;
        let result = cpu_manager.validate_kernels();
        assert!(result.is_err());
    }

    #[test]
    fn test_kernel_stats_occupancy() {
        let stats = KernelStats {
            name: "test".to_string(),
            registers_used: 16,
            shared_memory: 512,
            max_threads: 256,
            source_type: SourceType::Ptx,
        };

        let occupancy = stats.estimated_occupancy();
        assert!(occupancy > 0.0 && occupancy <= 1.0);
        assert!(!stats.is_resource_intensive());

        let intensive_stats = KernelStats {
            name: "intensive".to_string(),
            registers_used: 64,
            shared_memory: 2048,
            max_threads: 512,
            source_type: SourceType::Ptx,
        };

        assert!(intensive_stats.is_resource_intensive());
    }

    #[test]
    fn test_execution_context() {
        let context = KernelExecutionContext::new((2, 2, 1), (256, 1, 1)).with_shared_memory(1024);

        assert_eq!(context.grid_dim, (2, 2, 1));
        assert_eq!(context.block_dim, (256, 1, 1));
        assert_eq!(context.shared_memory, 1024);
        assert_eq!(context.total_threads(), 1024);
        assert!(context.validate());

        // Test invalid context
        let invalid_context = KernelExecutionContext::new((1, 1, 1), (2048, 1, 1));
        assert!(!invalid_context.validate());
    }

    #[test]
    fn test_default_execution_context() {
        let context = KernelExecutionContext::default();
        assert!(context.validate());
        assert_eq!(context.total_threads(), 256);
    }
}
