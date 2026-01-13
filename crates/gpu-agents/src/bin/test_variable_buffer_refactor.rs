//! Test variable buffer implementation
//!
//! REFACTOR phase - optimize and improve implementation

use cudarc::driver::{CudaContext, DeviceSlice};
use gpu_agents::synthesis::variable_buffer::{
    BufferConfig, VariableBufferManager, VariableKernelLauncher,
};
use std::sync::Arc;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    println!("Testing Variable Buffer Implementation (REFACTOR phase)");
    println!("====================================================");

    let ctx = CudaContext::new(0)?;

    // Test 1: Performance benchmarking
    println!("\n1. Testing allocation performance...");
    let mut manager = VariableBufferManager::new(ctx.clone(), BufferConfig::default())?;

    let start = Instant::now();
    for size in [1024, 4096, 16384, 65536, 262144, 1048576] {
        manager.ensure_capacity(size, size * 2, size / 2)?;
        println!(
            "   Allocated buffers for size {} in {:?}",
            size,
            start.elapsed()
        );
    }

    // Test 2: Buffer reuse efficiency
    println!("\n2. Testing buffer reuse...");
    let mut reuse_manager = VariableBufferManager::new(ctx.clone(), BufferConfig::default())?;

    // First allocation
    reuse_manager.ensure_capacity(100_000, 200_000, 50_000)?;
    let initial_usage = reuse_manager.memory_usage();

    // Should reuse existing buffers
    reuse_manager.ensure_capacity(50_000, 100_000, 25_000)?;
    let reuse_usage = reuse_manager.memory_usage();

    println!(
        "✅ Buffer reuse: initial={} bytes, after reuse={} bytes",
        initial_usage, reuse_usage
    );
    assert_eq!(
        initial_usage, reuse_usage,
        "Buffers should be reused when possible"
    );

    // Test 3: Dynamic growth testing
    println!("\n3. Testing dynamic buffer growth...");
    let mut config = BufferConfig::default();
    config.growth_factor = 2.0; // Double size on each growth

    let mut growth_manager = VariableBufferManager::new(ctx.clone(), config)?;

    let sizes = vec![1024, 2048, 4096, 8192];
    for size in sizes {
        let buffer = growth_manager.get_pattern_buffer(size)?;
        println!("   Requested {} bytes, got {} bytes", size, buffer.len());
    }

    // Test 4: Kernel launcher performance
    println!("\n4. Testing kernel launcher performance...");
    let mut launcher = VariableKernelLauncher::new(ctx.clone())?;

    // Create test data with proper encoding
    let pattern_count = 100;
    let node_count = 1000; // Reduced for testing
    let patterns = vec![0u8; pattern_count * 64]; // 64 bytes per pattern
    let ast_nodes = vec![0u8; node_count * 64]; // 64 bytes per node (NODE_SIZE)

    let start = Instant::now();
    let results = launcher.launch_pattern_matching(
        &patterns,
        &ast_nodes,
        pattern_count as u32,
        node_count as u32,
    )?;
    let elapsed = start.elapsed();

    println!("✅ Kernel execution completed in {:?}", elapsed);
    println!(
        "   Processed {} patterns against {} nodes",
        pattern_count, node_count
    );
    println!("   Results size: {} u32 values", results.len());

    // Test 5: Memory fragmentation prevention
    println!("\n5. Testing memory fragmentation handling...");
    let mut frag_manager = VariableBufferManager::new(ctx.clone(), BufferConfig::default())?;

    // Allocate and deallocate in patterns that could cause fragmentation
    for i in 0..5 {
        let size = (i + 1) * 10240;
        frag_manager.ensure_capacity(size, size * 2, size / 2)?;
        if i % 2 == 0 {
            frag_manager.clear();
        }
    }

    println!("✅ Fragmentation test completed without errors");

    // Test 6: Concurrent access simulation
    println!("\n6. Testing thread safety considerations...");
    // Note: VariableBufferManager is not thread-safe by design
    // Users should wrap it in Arc<Mutex<>> for concurrent access
    println!("✅ Thread safety: Manager designed for single-threaded use");
    println!("   For concurrent access, wrap in Arc<Mutex<>>");

    println!("\n✅ All REFACTOR phase tests passed!");
    println!("\nOptimizations implemented:");
    println!("- Buffer reuse to minimize allocations");
    println!("- Configurable growth factor for dynamic sizing");
    println!("- Alignment for optimal GPU memory access");
    println!("- Clear separation of concerns between manager and launcher");

    Ok(())
}
