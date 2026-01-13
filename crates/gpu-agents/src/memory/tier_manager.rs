//! Tier Manager Implementation
//!
//! Orchestrates memory allocation and management across 5 tiers

use super::{PageId, PageInfo, PageTable, TierLevel};
use anyhow::{Context, Result};
use cudarc::driver::CudaContext;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

/// Configuration for tier management
#[derive(Debug, Clone)]
pub struct TierConfig {
    pub gpu_capacity_mb: usize,
    pub cpu_capacity_mb: usize,
    pub nvme_path: PathBuf,
    pub ssd_path: PathBuf,
    pub hdd_path: PathBuf,
    pub page_size_kb: usize,
    pub migration_batch_size: usize,
    pub enable_compression: bool,
    pub prefetch_distance: usize,
}

impl Default for TierConfig {
    fn default() -> Self {
        Self {
            gpu_capacity_mb: 32 * 1024, // 32GB
            cpu_capacity_mb: 96 * 1024, // 96GB
            nvme_path: PathBuf::from("/nvme/exorust"),
            ssd_path: PathBuf::from("/ssd/exorust"),
            hdd_path: PathBuf::from("/hdd/exorust"),
            page_size_kb: 4, // 4KB pages
            migration_batch_size: 1024,
            enable_compression: true,
            prefetch_distance: 16,
        }
    }
}

/// Statistics for tier utilization
#[derive(Debug, Default, Clone)]
pub struct TierStatistics {
    pub gpu_pages_allocated: u64,
    pub cpu_pages_allocated: u64,
    pub nvme_pages_allocated: u64,
    pub ssd_pages_allocated: u64,
    pub hdd_pages_allocated: u64,
    pub total_pages_allocated: u64,
    pub gpu_utilization_percent: f32,
    pub cpu_utilization_percent: f32,
    pub nvme_utilization_percent: f32,
    pub ssd_utilization_percent: f32,
    pub hdd_utilization_percent: f32,
    pub total_migrations: u64,
    pub failed_migrations: u64,
}

/// Manages memory across all 5 tiers
pub struct TierManager {
    device: Arc<CudaContext>,
    config: TierConfig,

    // GPU memory management
    gpu_allocator: Arc<Mutex<GpuAllocator>>,

    // CPU memory management
    cpu_allocator: Arc<Mutex<CpuAllocator>>,

    // Storage tier paths
    nvme_path: PathBuf,
    ssd_path: PathBuf,
    hdd_path: PathBuf,

    // Page tracking
    page_table: Arc<Mutex<PageTable>>,
    next_page_id: Arc<Mutex<u64>>,

    // Statistics
    stats: Arc<Mutex<TierStatistics>>,
}

impl TierManager {
    /// Create new tier manager
    #[must_use = "ignoring the Result may hide tier manager initialization errors"]
    pub fn new(device: Arc<CudaContext>, config: TierConfig) -> Result<Self> {
        // Ensure storage directories exist
        fs::create_dir_all(&config.nvme_path)?;
        fs::create_dir_all(&config.ssd_path)?;
        fs::create_dir_all(&config.hdd_path)?;

        // Create sub-directories for organization
        fs::create_dir_all(config.nvme_path.join("pages"))?;
        fs::create_dir_all(config.ssd_path.join("pages"))?;
        fs::create_dir_all(config.hdd_path.join("pages"))?;

        let gpu_pages = config.gpu_capacity_mb * 1024 / config.page_size_kb;
        let cpu_pages = config.cpu_capacity_mb * 1024 / config.page_size_kb;

        Ok(Self {
            device: Arc::clone(&device),
            gpu_allocator: Arc::new(Mutex::new(GpuAllocator::new(Arc::clone(&device), gpu_pages)?)),
            cpu_allocator: Arc::new(Mutex::new(CpuAllocator::new(cpu_pages)?)),
            nvme_path: config.nvme_path.clone(),
            ssd_path: config.ssd_path.clone(),
            hdd_path: config.hdd_path.clone(),
            page_table: Arc::new(Mutex::new(PageTable::new(10_000_000))), // 10M pages max
            next_page_id: Arc::new(Mutex::new(1)),
            stats: Arc::new(Mutex::new(TierStatistics::default())),
            config,
        })
    }

    /// Get tier capacity in bytes
    pub fn get_tier_capacity(&self, tier: TierLevel) -> u64 {
        let page_size = self.config.page_size_kb as u64 * 1024;
        match tier {
            TierLevel::Gpu => self.config.gpu_capacity_mb as u64 * 1024 * 1024,
            TierLevel::Cpu => self.config.cpu_capacity_mb as u64 * 1024 * 1024,
            TierLevel::Nvme => 5500u64 * 1024 * 1024 * 1024, // 5.5TB
            TierLevel::Ssd => 4500u64 * 1024 * 1024 * 1024,  // 4.5TB
            TierLevel::Hdd => 3700u64 * 1024 * 1024 * 1024,  // 3.7TB
        }
    }

    /// Allocate a page in specified tier
    #[must_use = "PageId must be stored to free the allocated memory later"]
    pub fn allocate_page(&mut self, tier: TierLevel) -> Result<PageId> {
        let page_id = {
            let mut next_id = self.next_page_id.lock().unwrap();
            let id = PageId::new(*next_id);
            *next_id += 1;
            id
        };

        let page_size = self.config.page_size_kb * 1024;
        let address = match tier {
            TierLevel::Gpu => {
                let mut allocator = self.gpu_allocator.lock().unwrap();
                allocator.allocate(page_size)?
            }
            TierLevel::Cpu => {
                let mut allocator = self.cpu_allocator.lock().unwrap();
                allocator.allocate(page_size)?
            }
            TierLevel::Nvme | TierLevel::Ssd | TierLevel::Hdd => {
                // For storage tiers, return file-based address
                self.allocate_storage_page(tier, page_id)?
            }
        };

        // Update page table
        let page_info = PageInfo {
            tier,
            address,
            dirty: false,
            access_count: 0,
            last_access: std::time::Instant::now(),
        };

        {
            let mut page_table = self.page_table.lock().unwrap();
            page_table.insert(page_id, page_info)?;
        }

        // Update statistics
        {
            let mut stats = self.stats.lock().unwrap();
            match tier {
                TierLevel::Gpu => stats.gpu_pages_allocated += 1,
                TierLevel::Cpu => stats.cpu_pages_allocated += 1,
                TierLevel::Nvme => stats.nvme_pages_allocated += 1,
                TierLevel::Ssd => stats.ssd_pages_allocated += 1,
                TierLevel::Hdd => stats.hdd_pages_allocated += 1,
            }
            stats.total_pages_allocated += 1;
        }

        Ok(page_id)
    }

    /// Free a page
    pub fn free_page(&mut self, page_id: PageId) -> Result<()> {
        let page_info = {
            let mut page_table = self.page_table.lock().unwrap();
            page_table.remove(page_id)?
        };

        match page_info.tier {
            TierLevel::Gpu => {
                let mut allocator = self.gpu_allocator.lock().unwrap();
                allocator.free(page_info.address)?;
            }
            TierLevel::Cpu => {
                let mut allocator = self.cpu_allocator.lock().unwrap();
                allocator.free(page_info.address)?;
            }
            TierLevel::Nvme | TierLevel::Ssd | TierLevel::Hdd => {
                self.free_storage_page(page_info.tier, page_id)?;
            }
        }

        // Update statistics
        {
            let mut stats = self.stats.lock().unwrap();
            match page_info.tier {
                TierLevel::Gpu => stats.gpu_pages_allocated -= 1,
                TierLevel::Cpu => stats.cpu_pages_allocated -= 1,
                TierLevel::Nvme => stats.nvme_pages_allocated -= 1,
                TierLevel::Ssd => stats.ssd_pages_allocated -= 1,
                TierLevel::Hdd => stats.hdd_pages_allocated -= 1,
            }
            stats.total_pages_allocated -= 1;
        }

        Ok(())
    }

    /// Get page tier
    pub fn get_page_tier(&self, page_id: PageId) -> Option<TierLevel> {
        let page_table = self.page_table.lock().ok()?;
        page_table.lookup(page_id).map(|info| info.tier)
    }

    /// Write data to a page
    pub fn write_page(&mut self, page_id: PageId, data: &[u8]) -> Result<()> {
        let page_info = {
            let mut page_table = self.page_table.lock().unwrap();
            let info = page_table.lookup_mut(page_id).context("Page not found")?;
            info.dirty = true;
            info.clone()
        };

        match page_info.tier {
            TierLevel::Gpu => {
                // Write to GPU memory
                let gpu_allocator = self.gpu_allocator.lock().unwrap();
                gpu_allocator.write(page_info.address, data)?;
            }
            TierLevel::Cpu => {
                // Write to CPU memory
                let mut cpu_allocator = self.cpu_allocator.lock().unwrap();
                cpu_allocator.write(page_info.address, data)?;
            }
            TierLevel::Nvme | TierLevel::Ssd | TierLevel::Hdd => {
                // Write to storage
                self.write_storage_page(page_info.tier, page_id, data)?;
            }
        }

        Ok(())
    }

    /// Read data from a page
    pub fn read_page(&self, page_id: PageId) -> Result<Vec<u8>> {
        let page_info = {
            let mut page_table = self.page_table.lock().unwrap();
            page_table.access_page(page_id);
            page_table
                .lookup(page_id)
                .context("Page not found")?
                .clone()
        };

        let data = match page_info.tier {
            TierLevel::Gpu => {
                let gpu_allocator = self.gpu_allocator.lock().unwrap();
                gpu_allocator.read(page_info.address, self.config.page_size_kb * 1024)?
            }
            TierLevel::Cpu => {
                let cpu_allocator = self.cpu_allocator.lock().unwrap();
                cpu_allocator.read(page_info.address, self.config.page_size_kb * 1024)?
            }
            TierLevel::Nvme | TierLevel::Ssd | TierLevel::Hdd => {
                self.read_storage_page(page_info.tier, page_id)?
            }
        };

        Ok(data)
    }

    /// Migrate page to different tier
    pub fn migrate_page(&mut self, page_id: PageId, target_tier: TierLevel) -> Result<()> {
        // Read page data
        let data = self.read_page(page_id)?;

        // Get current tier
        let source_tier = self.get_page_tier(page_id).context("Page not found")?;

        if source_tier == target_tier {
            return Ok(()); // Already in target tier
        }

        // Allocate in target tier
        let new_page = self.allocate_page(target_tier)?;

        // Write data to new location
        self.write_page(new_page, &data)?;

        // Update page table to point to new location
        {
            let mut page_table = self.page_table.lock().unwrap();
            // Get new address first to avoid borrowing conflicts
            let new_address = page_table
                .lookup(new_page)
                .context("New page not found")?
                .address;

            if let Some(info) = page_table.lookup_mut(page_id) {
                info.tier = target_tier;
                info.address = new_address;
            }
            // Remove temporary page entry
            page_table.remove(new_page)?;
        }

        // Free old location
        self.free_page(page_id)?;

        // Update statistics
        {
            let mut stats = self.stats.lock().unwrap();
            stats.total_migrations += 1;
        }

        Ok(())
    }

    /// Batch migrate multiple pages
    pub fn batch_migrate(&mut self, page_ids: &[PageId], target_tier: TierLevel) -> Result<()> {
        for &page_id in page_ids {
            self.migrate_page(page_id, target_tier)?;
        }
        Ok(())
    }

    /// Access a page (updates LRU)
    pub fn access_page(&self, page_id: PageId) {
        let mut page_table = self.page_table.lock().unwrap();
        page_table.access_page(page_id);
    }

    /// Predict and prefetch pages
    pub fn predict_and_prefetch(&mut self, accessed_page: PageId) -> Result<()> {
        // Simple sequential prefetching for now
        let page_num = accessed_page.as_u64();

        for i in 1..=self.config.prefetch_distance {
            let prefetch_id = PageId::new(page_num + i as u64);

            // Check if page exists and is in lower tier
            if let Some(current_tier) = self.get_page_tier(prefetch_id) {
                if current_tier > TierLevel::Gpu {
                    // Promote to GPU tier
                    self.migrate_page(prefetch_id, TierLevel::Gpu).ok();
                }
            }
        }

        Ok(())
    }

    /// Get statistics
    pub fn get_statistics(&self) -> TierStatistics {
        let stats = self.stats.lock().unwrap();
        let mut result = (*stats).clone();

        // Calculate utilization percentages
        let page_size = self.config.page_size_kb as f32 * 1024.0;
        result.gpu_utilization_percent = (stats.gpu_pages_allocated as f32 * page_size)
            / (self.config.gpu_capacity_mb as f32 * 1024.0 * 1024.0)
            * 100.0;
        result.cpu_utilization_percent = (stats.cpu_pages_allocated as f32 * page_size)
            / (self.config.cpu_capacity_mb as f32 * 1024.0 * 1024.0)
            * 100.0;

        // Storage utilization would need actual disk usage checks
        result.nvme_utilization_percent = (stats.nvme_pages_allocated as f32 * page_size)
            / (5500.0 * 1024.0 * 1024.0 * 1024.0)
            * 100.0;
        result.ssd_utilization_percent = (stats.ssd_pages_allocated as f32 * page_size)
            / (4500.0 * 1024.0 * 1024.0 * 1024.0)
            * 100.0;
        result.hdd_utilization_percent = (stats.hdd_pages_allocated as f32 * page_size)
            / (3700.0 * 1024.0 * 1024.0 * 1024.0)
            * 100.0;

        result
    }

    /// Garbage collect unused pages
    ///
    /// This performs several cleanup tasks:
    /// 1. Removes pages with zero access count and no recent access
    /// 2. Coalesces free memory regions in GPU/CPU allocators
    /// 3. Cleans up orphaned storage files
    /// 4. Demotes cold pages to lower tiers
    pub fn garbage_collect(&mut self) -> Result<usize> {
        let mut collected_count = 0;
        let now = std::time::Instant::now();
        let stale_threshold = std::time::Duration::from_secs(300); // 5 minutes

        // Phase 1: Identify and collect stale pages
        let stale_pages: Vec<PageId> = {
            let page_table = self.page_table.lock().unwrap();
            page_table
                .iter_pages()
                .filter(|(_, info)| {
                    // Page is stale if:
                    // - Has zero access count AND hasn't been accessed in stale_threshold
                    // - OR is marked as dirty but in a cold tier without recent access
                    let age = now.duration_since(info.last_access);
                    (info.access_count == 0 && age > stale_threshold)
                        || (info.dirty && info.tier > TierLevel::Ssd && age > stale_threshold * 2)
                })
                .map(|(id, _)| *id)
                .collect()
        };

        // Free stale pages
        for page_id in stale_pages {
            if self.free_page(page_id).is_ok() {
                collected_count += 1;
            }
        }

        // Phase 2: Coalesce free memory regions in GPU allocator
        {
            let mut gpu_alloc = self.gpu_allocator.lock().unwrap();
            Self::coalesce_free_list(&mut gpu_alloc.free_list);
        }

        // Phase 3: Coalesce free memory regions in CPU allocator
        {
            let mut cpu_alloc = self.cpu_allocator.lock().unwrap();
            Self::coalesce_cpu_free_list(&mut cpu_alloc.free_list);
        }

        // Phase 4: Clean up orphaned storage files
        collected_count += self.cleanup_orphaned_storage_files()?;

        // Phase 5: Demote cold GPU pages to CPU (if GPU utilization is high)
        let stats = self.get_statistics();
        if stats.gpu_utilization_percent > 80.0 {
            let cold_gpu_pages: Vec<PageId> = {
                let page_table = self.page_table.lock().unwrap();
                let mut pages: Vec<_> = page_table
                    .iter_pages()
                    .filter(|(_, info)| info.tier == TierLevel::Gpu)
                    .map(|(id, info)| (*id, info.access_count, info.last_access))
                    .collect();

                // Sort by access count (ascending) then by last access (oldest first)
                pages.sort_by(|a, b| {
                    a.1.cmp(&b.1).then_with(|| b.2.cmp(&a.2))
                });

                // Take bottom 10% for demotion
                let demote_count = pages.len() / 10;
                pages.into_iter().take(demote_count).map(|(id, _, _)| id).collect()
            };

            for page_id in cold_gpu_pages {
                if self.migrate_page(page_id, TierLevel::Cpu).is_ok() {
                    collected_count += 1;
                }
            }
        }

        Ok(collected_count)
    }

    /// Coalesce adjacent free regions in GPU allocator free list
    fn coalesce_free_list(free_list: &mut Vec<(u64, usize)>) {
        if free_list.len() < 2 {
            return;
        }

        // Sort by address
        free_list.sort_by_key(|(addr, _)| *addr);

        let mut i = 0;
        while i < free_list.len() - 1 {
            let (addr1, size1) = free_list[i];
            let (addr2, size2) = free_list[i + 1];

            // Check if regions are adjacent
            if addr1 + size1 as u64 == addr2 {
                // Merge regions
                free_list[i] = (addr1, size1 + size2);
                free_list.remove(i + 1);
            } else {
                i += 1;
            }
        }
    }

    /// Coalesce adjacent free regions in CPU allocator free list
    fn coalesce_cpu_free_list(free_list: &mut Vec<(usize, usize)>) {
        if free_list.len() < 2 {
            return;
        }

        // Sort by offset
        free_list.sort_by_key(|(offset, _)| *offset);

        let mut i = 0;
        while i < free_list.len() - 1 {
            let (offset1, size1) = free_list[i];
            let (offset2, size2) = free_list[i + 1];

            // Check if regions are adjacent
            if offset1 + size1 == offset2 {
                // Merge regions
                free_list[i] = (offset1, size1 + size2);
                free_list.remove(i + 1);
            } else {
                i += 1;
            }
        }
    }

    /// Clean up storage files that don't correspond to any tracked page
    fn cleanup_orphaned_storage_files(&self) -> Result<usize> {
        let mut cleaned = 0;

        // Get all tracked page IDs
        let tracked_pages: std::collections::HashSet<u64> = {
            let page_table = self.page_table.lock().unwrap();
            page_table.iter_pages().map(|(id, _)| id.as_u64()).collect()
        };

        // Check each storage tier
        for (tier, base_path) in [
            (TierLevel::Nvme, &self.nvme_path),
            (TierLevel::Ssd, &self.ssd_path),
            (TierLevel::Hdd, &self.hdd_path),
        ] {
            let pages_dir = base_path.join("pages");
            if !pages_dir.exists() {
                continue;
            }

            if let Ok(entries) = fs::read_dir(&pages_dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.extension().map(|e| e == "page").unwrap_or(false) {
                        // Extract page ID from filename
                        if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                            if let Ok(page_id) = u64::from_str_radix(stem, 16) {
                                // Check if this page is tracked
                                if !tracked_pages.contains(&page_id) {
                                    // Orphaned file - remove it
                                    if fs::remove_file(&path).is_ok() {
                                        cleaned += 1;
                                        tracing::debug!(
                                            "Removed orphaned page file: {:?}",
                                            path
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(cleaned)
    }

    /// Get storage paths
    pub fn nvme_path(&self) -> &PathBuf {
        &self.nvme_path
    }
    pub fn ssd_path(&self) -> &PathBuf {
        &self.ssd_path
    }
    pub fn hdd_path(&self) -> &PathBuf {
        &self.hdd_path
    }

    // Private helper methods

    fn allocate_storage_page(&self, tier: TierLevel, page_id: PageId) -> Result<u64> {
        // Return page ID as address for file-based storage
        Ok(page_id.as_u64())
    }

    fn free_storage_page(&self, tier: TierLevel, page_id: PageId) -> Result<()> {
        let path = self.get_storage_path(tier, page_id);
        if path.exists() {
            fs::remove_file(path)?;
        }
        Ok(())
    }

    fn write_storage_page(&self, tier: TierLevel, page_id: PageId, data: &[u8]) -> Result<()> {
        let path = self.get_storage_path(tier, page_id);
        let mut file = File::create(path)?;

        // Apply compression if enabled
        if self.config.enable_compression {
            let compressed = match tier.compression_algorithm() {
                super::CompressionAlgorithm::None => data.to_vec(),
                super::CompressionAlgorithm::Lz4 => super::compress_lz4(data)?,
                super::CompressionAlgorithm::Zstd(level) => super::compress_zstd(data, level)?,
            };
            file.write_all(&compressed)?;
        } else {
            file.write_all(data)?;
        }

        Ok(())
    }

    fn read_storage_page(&self, tier: TierLevel, page_id: PageId) -> Result<Vec<u8>> {
        let path = self.get_storage_path(tier, page_id);
        let mut file = File::open(path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;

        // Decompress if needed
        if self.config.enable_compression {
            let decompressed = match tier.compression_algorithm() {
                super::CompressionAlgorithm::None => buffer,
                super::CompressionAlgorithm::Lz4 => {
                    super::decompress_lz4(&buffer, self.config.page_size_kb * 1024)?
                }
                super::CompressionAlgorithm::Zstd(_) => {
                    super::decompress_zstd(&buffer, self.config.page_size_kb * 1024)?
                }
            };
            Ok(decompressed)
        } else {
            Ok(buffer)
        }
    }

    fn get_storage_path(&self, tier: TierLevel, page_id: PageId) -> PathBuf {
        let base = match tier {
            TierLevel::Nvme => &self.nvme_path,
            TierLevel::Ssd => &self.ssd_path,
            TierLevel::Hdd => &self.hdd_path,
            _ => panic!("Invalid storage tier"),
        };

        base.join("pages")
            .join(format!("{:016x}.page", page_id.as_u64()))
    }
}

/// GPU memory allocator
struct GpuAllocator {
    device: Arc<CudaContext>,
    free_list: Vec<(u64, usize)>, // (address, size)
    allocated: HashMap<u64, usize>,
    total_size: usize,
    next_address: u64,
}

impl GpuAllocator {
    fn new(device: Arc<CudaContext>, num_pages: usize) -> Result<Self> {
        let page_size = 4096;
        let total_size = num_pages * page_size;

        Ok(Self {
            device,
            free_list: vec![(0, total_size)],
            allocated: HashMap::new(),
            total_size,
            next_address: 0,
        })
    }

    fn allocate(&mut self, size: usize) -> Result<u64> {
        // Simple first-fit allocation
        for i in 0..self.free_list.len() {
            let (addr, free_size) = self.free_list[i];
            if free_size >= size {
                // Remove from free list
                self.free_list.remove(i);

                // Add remainder back to free list if any
                if free_size > size {
                    self.free_list.push((addr + size as u64, free_size - size));
                }

                // Track allocation
                self.allocated.insert(addr, size);

                return Ok(addr);
            }
        }

        anyhow::bail!("GPU memory full")
    }

    fn free(&mut self, address: u64) -> Result<()> {
        let size = self
            .allocated
            .remove(&address)
            .context("Invalid GPU address")?;

        // Add back to free list (simple, no coalescing for now)
        self.free_list.push((address, size));

        Ok(())
    }

    fn write(&self, address: u64, data: &[u8]) -> Result<()> {
        // In real implementation, would write to GPU memory
        // For now, this is a placeholder
        Ok(())
    }

    fn read(&self, address: u64, size: usize) -> Result<Vec<u8>> {
        // In real implementation, would read from GPU memory
        // For now, return zeros
        Ok(vec![0u8; size])
    }
}

/// CPU memory allocator
struct CpuAllocator {
    memory: Vec<u8>,
    free_list: Vec<(usize, usize)>,          // (offset, size)
    allocated: HashMap<u64, (usize, usize)>, // address -> (offset, size)
}

impl CpuAllocator {
    fn new(num_pages: usize) -> Result<Self> {
        let page_size = 4096;
        let total_size = num_pages * page_size;

        Ok(Self {
            memory: vec![0u8; total_size],
            free_list: vec![(0, total_size)],
            allocated: HashMap::new(),
        })
    }

    fn allocate(&mut self, size: usize) -> Result<u64> {
        // Simple first-fit allocation
        for i in 0..self.free_list.len() {
            let (offset, free_size) = self.free_list[i];
            if free_size >= size {
                // Remove from free list
                self.free_list.remove(i);

                // Add remainder back to free list if any
                if free_size > size {
                    self.free_list.push((offset + size, free_size - size));
                }

                // Track allocation
                let address = offset as u64;
                self.allocated.insert(address, (offset, size));

                return Ok(address);
            }
        }

        anyhow::bail!("CPU memory full")
    }

    fn free(&mut self, address: u64) -> Result<()> {
        let (offset, size) = self
            .allocated
            .remove(&address)
            .context("Invalid CPU address")?;

        // Add back to free list
        self.free_list.push((offset, size));

        // Clear memory
        for i in offset..offset + size {
            self.memory[i] = 0;
        }

        Ok(())
    }

    fn write(&mut self, address: u64, data: &[u8]) -> Result<()> {
        let (offset, size) = self
            .allocated
            .get(&address)
            .context("Invalid CPU address")?;

        if data.len() > *size {
            anyhow::bail!("Data too large for allocated page");
        }

        self.memory[*offset..*offset + data.len()].copy_from_slice(data);
        Ok(())
    }

    fn read(&self, address: u64, size: usize) -> Result<Vec<u8>> {
        let (offset, alloc_size) = self
            .allocated
            .get(&address)
            .context("Invalid CPU address")?;

        if size > *alloc_size {
            anyhow::bail!("Read size exceeds allocated size");
        }

        Ok(self.memory[*offset..*offset + size].to_vec())
    }
}
