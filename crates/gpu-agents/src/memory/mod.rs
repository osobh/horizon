//! 5-Tier Memory Manager for GPU Agents
//!
//! Manages memory across GPU→CPU→NVMe→SSD→HDD tiers with automatic migration,
//! compression, and CUDA Unified Memory integration.

pub mod compression;
pub mod migration;
pub mod migration_optimization;
pub mod page_table;
pub mod tier_manager;
pub mod unified_memory;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod migration_performance_test;

use anyhow::Result;
use cudarc::driver::CudaDevice;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

// Re-export main types
pub use compression::{compress_lz4, compress_zstd, decompress_lz4, decompress_zstd};
pub use migration::{MigrationEngine, MigrationPolicy};
pub use page_table::{PageId, PageInfo, PageTable};
pub use tier_manager::{TierConfig, TierManager, TierStatistics};
pub use unified_memory::UnifiedMemory;

/// Memory tier levels from fastest to slowest
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TierLevel {
    Gpu = 0,  // Tier 1: GPU memory (32GB)
    Cpu = 1,  // Tier 2: CPU memory (96GB)
    Nvme = 2, // Tier 3: NVMe storage (5.5TB)
    Ssd = 3,  // Tier 4: SSD storage (4.5TB)
    Hdd = 4,  // Tier 5: HDD storage (3.7TB)
}

impl TierLevel {
    /// Get compression algorithm for this tier
    pub fn compression_algorithm(&self) -> CompressionAlgorithm {
        match self {
            TierLevel::Gpu | TierLevel::Cpu => CompressionAlgorithm::None,
            TierLevel::Nvme => CompressionAlgorithm::Lz4,
            TierLevel::Ssd => CompressionAlgorithm::Zstd(3), // Balanced
            TierLevel::Hdd => CompressionAlgorithm::Zstd(9), // High compression
        }
    }

    /// Get tier name as string
    pub fn name(&self) -> &'static str {
        match self {
            TierLevel::Gpu => "GPU",
            TierLevel::Cpu => "CPU",
            TierLevel::Nvme => "NVMe",
            TierLevel::Ssd => "SSD",
            TierLevel::Hdd => "HDD",
        }
    }

    /// Get typical access latency for this tier
    pub fn latency_ns(&self) -> u64 {
        match self {
            TierLevel::Gpu => 100,       // ~100ns GPU memory access
            TierLevel::Cpu => 100,       // ~100ns CPU memory access
            TierLevel::Nvme => 20_000,   // ~20μs NVMe access
            TierLevel::Ssd => 100_000,   // ~100μs SSD access
            TierLevel::Hdd => 5_000_000, // ~5ms HDD access
        }
    }
}

/// Compression algorithms supported by tiers
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CompressionAlgorithm {
    None,
    Lz4,
    Zstd(i32), // Compression level
}

/// Configuration for tier storage paths
#[derive(Debug, Clone)]
pub struct StorageConfig {
    pub nvme_path: PathBuf, // /nvme/exorust
    pub ssd_path: PathBuf,  // /ssd/exorust
    pub hdd_path: PathBuf,  // /hdd/exorust
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            nvme_path: PathBuf::from("/nvme/exorust"),
            ssd_path: PathBuf::from("/ssd/exorust"),
            hdd_path: PathBuf::from("/hdd/exorust"),
        }
    }
}

/// Page metadata for tracking
#[derive(Debug, Clone)]
pub struct PageMetadata {
    pub id: PageId,
    pub tier: TierLevel,
    pub size: usize,
    pub compressed: bool,
    pub compression_ratio: f32,
    pub access_count: u64,
    pub last_access: Instant,
    pub dirty: bool,
}

/// Migration request for async processing
#[derive(Debug, Clone)]
pub struct MigrationRequest {
    pub page_id: PageId,
    pub source_tier: TierLevel,
    pub target_tier: TierLevel,
    pub priority: MigrationPriority,
    pub deadline: Option<Instant>,
}

/// Migration priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MigrationPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

/// Result of a migration operation
#[derive(Debug)]
pub struct MigrationResult {
    pub page_id: PageId,
    pub success: bool,
    pub duration_us: u64,
    pub bytes_moved: usize,
    pub compression_saved: usize,
}

/// Memory manager metrics
#[derive(Debug, Default)]
pub struct MemoryMetrics {
    pub total_pages: u64,
    pub total_bytes: u64,
    pub migrations_completed: u64,
    pub migrations_failed: u64,
    pub average_migration_time_us: u64,
    pub compression_ratio: f32,
    pub tier_utilization: [f32; 5],
}

/// Main 5-tier memory manager
pub struct MemoryManager {
    device: Arc<CudaDevice>,
    tier_manager: TierManager,
    migration_engine: MigrationEngine,
    metrics: MemoryMetrics,
}

impl MemoryManager {
    /// Create new memory manager with default configuration
    pub fn new(device: Arc<CudaDevice>) -> Result<Self> {
        let config = TierConfig::default();
        let tier_manager = TierManager::new(Arc::clone(&device), config)?;
        let migration_engine = MigrationEngine::new(Arc::clone(&device))?;

        Ok(Self {
            device,
            tier_manager,
            migration_engine,
            metrics: MemoryMetrics::default(),
        })
    }

    /// Allocate memory for an agent (1MB total)
    pub fn allocate_agent_memory(&mut self) -> Result<AgentMemory> {
        // Allocate 256B core memory in GPU
        let core_page = self.tier_manager.allocate_page(TierLevel::Gpu)?;

        // Allocate 256KB x 4 specialized memory in CPU initially
        let mut specialized_pages = Vec::new();
        for _ in 0..4 {
            let pages_needed = 256 * 1024 / 4096; // 64 pages per 256KB
            let mut pages = Vec::new();
            for _ in 0..pages_needed {
                pages.push(self.tier_manager.allocate_page(TierLevel::Cpu)?);
            }
            specialized_pages.push(pages);
        }

        Ok(AgentMemory {
            core_page,
            working_memory: specialized_pages[0].clone(),
            episodic_memory: specialized_pages[1].clone(),
            semantic_memory: specialized_pages[2].clone(),
            procedural_memory: specialized_pages[3].clone(),
        })
    }

    /// Get current memory metrics
    pub fn get_metrics(&self) -> &MemoryMetrics {
        &self.metrics
    }

    /// Trigger manual garbage collection
    pub fn garbage_collect(&mut self) -> Result<usize> {
        self.tier_manager.garbage_collect()
    }
}

/// Agent memory allocation across tiers
#[derive(Debug)]
pub struct AgentMemory {
    pub core_page: PageId,              // 256B in GPU
    pub working_memory: Vec<PageId>,    // 256KB
    pub episodic_memory: Vec<PageId>,   // 256KB
    pub semantic_memory: Vec<PageId>,   // 256KB
    pub procedural_memory: Vec<PageId>, // 256KB
}

// External CUDA kernel functions
extern "C" {
    pub fn launch_page_marking(pages: *const u8, marks: *mut u32, num_pages: u32);

    pub fn launch_compression_kernel(
        input: *const u8,
        output: *mut u8,
        input_size: u32,
        output_size: *mut u32,
        algorithm: u32,
    );

    pub fn launch_migration_kernel(
        source: *const u8,
        destination: *mut u8,
        size: u32,
        source_tier: u32,
        dest_tier: u32,
    );
}
