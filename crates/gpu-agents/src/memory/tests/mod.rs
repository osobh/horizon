//! Tests for 5-Tier Memory Manager
//!
//! Tests tier management, page migration, and CUDA Unified Memory integration

use super::*;
use cudarc::driver::CudaDevice;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

// Test tier configuration creation
#[test]
fn test_tier_config_creation() {
    let config = TierConfig {
        gpu_capacity_mb: 32 * 1024, // 32GB
        cpu_capacity_mb: 96 * 1024, // 96GB
        nvme_path: PathBuf::from("/nvme/exorust"),
        ssd_path: PathBuf::from("/ssd/exorust"),
        hdd_path: PathBuf::from("/hdd/exorust"),
        page_size_kb: 4,            // 4KB pages
        migration_batch_size: 1024, // 1024 pages per batch
        enable_compression: true,
        prefetch_distance: 16, // Prefetch 16 pages ahead
    };

    assert_eq!(config.gpu_capacity_mb, 32768);
    assert_eq!(config.cpu_capacity_mb, 98304);
    assert_eq!(config.page_size_kb, 4);
    assert!(config.enable_compression);
}

// Test tier manager initialization
#[test]
fn test_tier_manager_creation() {
    let Ok(device) = CudaDevice::new(0) else {
        println!("No GPU available, skipping test");
        return;
    };
    let device = Arc::new(device);

    let config = TierConfig::default();
    let tier_manager = TierManager::new(device, config).expect("Failed to create tier manager");

    // Check tier capacities
    assert_eq!(
        tier_manager.get_tier_capacity(TierLevel::Gpu),
        32 * 1024 * 1024 * 1024
    );
    assert_eq!(
        tier_manager.get_tier_capacity(TierLevel::Cpu),
        96 * 1024 * 1024 * 1024
    );

    // Check tier paths exist
    assert!(tier_manager.nvme_path().exists());
    assert!(tier_manager.ssd_path().exists());
    assert!(tier_manager.hdd_path().exists());
}

// Test page allocation across tiers
#[test]
fn test_page_allocation() {
    let Ok(device) = CudaDevice::new(0) else {
        println!("No GPU available, skipping test");
        return;
    };
    let device = Arc::new(device);

    let config = TierConfig::default();
    let mut tier_manager = TierManager::new(device, config).expect("Failed to create tier manager");

    // Allocate pages in GPU tier
    let page_id = tier_manager
        .allocate_page(TierLevel::Gpu)
        .expect("Failed to allocate GPU page");
    assert!(page_id.is_valid());

    // Verify page is in GPU tier
    assert_eq!(tier_manager.get_page_tier(page_id), Some(TierLevel::Gpu));

    // Free the page
    tier_manager
        .free_page(page_id)
        .expect("Failed to free page");
    assert_eq!(tier_manager.get_page_tier(page_id), None);
}

// Test CUDA Unified Memory wrapper
#[test]
fn test_unified_memory() {
    let Ok(device) = CudaDevice::new(0) else {
        println!("No GPU available, skipping test");
        return;
    };
    let device = Arc::new(device);

    // Create unified memory allocation
    let size = 1024 * 1024; // 1MB
    let unified_mem =
        UnifiedMemory::<u8>::new(device.clone(), size).expect("Failed to create unified memory");

    assert_eq!(unified_mem.size(), size);
    assert!(unified_mem.device_ptr().is_valid());
    assert!(unified_mem.host_ptr().is_valid());

    // Test prefetch to GPU
    unified_mem
        .prefetch_to_gpu()
        .expect("Failed to prefetch to GPU");

    // Test prefetch to CPU
    unified_mem
        .prefetch_to_cpu()
        .expect("Failed to prefetch to CPU");
}

// Test page table with LRU eviction
#[test]
fn test_page_table_lru() {
    let max_pages = 1000;
    let mut page_table = PageTable::new(max_pages);

    // Fill page table
    for i in 0..max_pages {
        let page_id = PageId::new(i as u64);
        page_table
            .insert(
                page_id,
                PageInfo {
                    tier: TierLevel::Gpu,
                    address: i as u64 * 4096,
                    dirty: false,
                    access_count: 0,
                    last_access: Instant::now(),
                },
            )
            .expect("Failed to insert page");
    }

    // Access some pages to update LRU
    for i in 0..10 {
        let page_id = PageId::new(i as u64);
        page_table.access_page(page_id);
    }

    // Insert new page should evict LRU page
    let new_page = PageId::new(max_pages as u64);
    page_table
        .insert(
            new_page,
            PageInfo {
                tier: TierLevel::Gpu,
                address: max_pages as u64 * 4096,
                dirty: false,
                access_count: 0,
                last_access: Instant::now(),
            },
        )
        .expect("Failed to insert new page");

    // Check that frequently accessed pages are still present
    for i in 0..10 {
        let page_id = PageId::new(i as u64);
        assert!(page_table.lookup(page_id).is_some());
    }
}

// Test page migration between tiers
#[test]
fn test_page_migration() {
    let Ok(device) = CudaDevice::new(0) else {
        println!("No GPU available, skipping test");
        return;
    };
    let device = Arc::new(device);

    let config = TierConfig::default();
    let mut tier_manager = TierManager::new(device, config).expect("Failed to create tier manager");

    // Allocate page in GPU tier
    let page_id = tier_manager
        .allocate_page(TierLevel::Gpu)
        .expect("Failed to allocate page");

    // Write some data to the page
    let test_data = vec![42u8; 4096];
    tier_manager
        .write_page(page_id, &test_data)
        .expect("Failed to write page");

    // Migrate page to CPU tier
    let start = Instant::now();
    tier_manager
        .migrate_page(page_id, TierLevel::Cpu)
        .expect("Failed to migrate page");
    let migration_time = start.elapsed();

    // Verify migration completed in <1ms
    assert!(
        migration_time.as_millis() < 1,
        "Migration took {:?}, expected <1ms",
        migration_time
    );

    // Verify page is now in CPU tier
    assert_eq!(tier_manager.get_page_tier(page_id), Some(TierLevel::Cpu));

    // Read page and verify data
    let read_data = tier_manager
        .read_page(page_id)
        .expect("Failed to read page");
    assert_eq!(read_data, test_data);
}

// Test batch page migration
#[test]
fn test_batch_migration() {
    let Ok(device) = CudaDevice::new(0) else {
        println!("No GPU available, skipping test");
        return;
    };
    let device = Arc::new(device);

    let config = TierConfig::default();
    let mut tier_manager = TierManager::new(device, config).expect("Failed to create tier manager");

    // Allocate multiple pages
    let num_pages = 100;
    let mut page_ids = Vec::new();

    for _ in 0..num_pages {
        let page_id = tier_manager
            .allocate_page(TierLevel::Gpu)
            .expect("Failed to allocate page");
        page_ids.push(page_id);
    }

    // Batch migrate to CPU
    let start = Instant::now();
    tier_manager
        .batch_migrate(&page_ids, TierLevel::Cpu)
        .expect("Failed to batch migrate");
    let migration_time = start.elapsed();

    // Verify batch migration is efficient
    let time_per_page = migration_time.as_micros() / num_pages as u128;
    assert!(
        time_per_page < 100,
        "Batch migration too slow: {}μs per page",
        time_per_page
    );

    // Verify all pages migrated
    for page_id in &page_ids {
        assert_eq!(tier_manager.get_page_tier(*page_id), Some(TierLevel::Cpu));
    }
}

// Test tier-specific compression
#[test]
fn test_compression() {
    // Create test data with good compression potential
    let mut test_data = vec![0u8; 4096];
    for i in 0..1024 {
        test_data[i * 4] = (i % 256) as u8;
    }

    // Test LZ4 compression for NVMe tier
    let compressed = compress_lz4(&test_data).expect("LZ4 compression failed");
    assert!(compressed.len() < test_data.len());

    let decompressed =
        decompress_lz4(&compressed, test_data.len()).expect("LZ4 decompression failed");
    assert_eq!(decompressed, test_data);

    // Test ZSTD compression for SSD/HDD tiers
    let compressed = compress_zstd(&test_data, 3).expect("ZSTD compression failed");
    assert!(compressed.len() < test_data.len());

    let decompressed =
        decompress_zstd(&compressed, test_data.len()).expect("ZSTD decompression failed");
    assert_eq!(decompressed, test_data);
}

// Test predictive prefetching
#[test]
fn test_predictive_prefetch() {
    let Ok(device) = CudaDevice::new(0) else {
        println!("No GPU available, skipping test");
        return;
    };
    let device = Arc::new(device);

    let config = TierConfig::default();
    let mut tier_manager = TierManager::new(device, config).expect("Failed to create tier manager");

    // Create access pattern
    let mut page_ids = Vec::new();
    for i in 0..100 {
        let page_id = tier_manager
            .allocate_page(TierLevel::Cpu)
            .expect("Failed to allocate page");
        page_ids.push(page_id);
    }

    // Sequential access pattern
    for i in 0..10 {
        tier_manager.access_page(page_ids[i]);
    }

    // Trigger predictive prefetch
    tier_manager
        .predict_and_prefetch(page_ids[9])
        .expect("Prefetch failed");

    // Verify next pages are prefetched to GPU
    for i in 10..26 {
        // Prefetch distance is 16
        if let Some(tier) = tier_manager.get_page_tier(page_ids[i]) {
            // Pages should be promoted to higher tier
            assert!(tier as u8 <= TierLevel::Gpu as u8);
        }
    }
}

// Test memory pressure handling
#[test]
fn test_memory_pressure() {
    let Ok(device) = CudaDevice::new(0) else {
        println!("No GPU available, skipping test");
        return;
    };
    let device = Arc::new(device);

    // Create config with small GPU capacity for testing
    let mut config = TierConfig::default();
    config.gpu_capacity_mb = 100; // 100MB only

    let mut tier_manager = TierManager::new(device, config).expect("Failed to create tier manager");

    // Allocate pages until GPU is full
    let page_size = 4096;
    let pages_per_mb = 1024 * 1024 / page_size;
    let max_pages = 100 * pages_per_mb;

    let mut allocated_pages = Vec::new();
    for _ in 0..max_pages {
        match tier_manager.allocate_page(TierLevel::Gpu) {
            Ok(page_id) => allocated_pages.push(page_id),
            Err(_) => break, // GPU full
        }
    }

    // Try to allocate one more page - should trigger eviction
    let new_page = tier_manager
        .allocate_page(TierLevel::Gpu)
        .expect("Failed to allocate page under pressure");

    // Verify some pages were evicted to lower tier
    let mut evicted_count = 0;
    for page_id in &allocated_pages {
        if let Some(tier) = tier_manager.get_page_tier(*page_id) {
            if tier != TierLevel::Gpu {
                evicted_count += 1;
            }
        }
    }

    assert!(
        evicted_count > 0,
        "No pages were evicted under memory pressure"
    );
}

// Test tier statistics and monitoring
#[test]
fn test_tier_statistics() {
    let Ok(device) = CudaDevice::new(0) else {
        println!("No GPU available, skipping test");
        return;
    };
    let device = Arc::new(device);

    let config = TierConfig::default();
    let mut tier_manager = TierManager::new(device, config).expect("Failed to create tier manager");

    // Allocate pages across different tiers
    for _ in 0..10 {
        tier_manager.allocate_page(TierLevel::Gpu).ok();
    }
    for _ in 0..20 {
        tier_manager.allocate_page(TierLevel::Cpu).ok();
    }
    for _ in 0..30 {
        tier_manager.allocate_page(TierLevel::Nvme).ok();
    }

    // Get tier statistics
    let stats = tier_manager.get_statistics();

    assert_eq!(stats.gpu_pages_allocated, 10);
    assert_eq!(stats.cpu_pages_allocated, 20);
    assert_eq!(stats.nvme_pages_allocated, 30);
    assert_eq!(stats.total_pages_allocated, 60);

    // Check utilization percentages
    assert!(stats.gpu_utilization_percent > 0.0);
    assert!(stats.cpu_utilization_percent > 0.0);
    assert!(stats.nvme_utilization_percent > 0.0);
}
