//! Test the safe synthesis kernel with debug output

use cudarc::driver::{CudaContext, DevicePtr};
use gpu_agents::synthesis::launch_match_patterns_safe;

fn main() -> anyhow::Result<()> {
    println!("🧪 Safe Synthesis Kernel Test");
    println!("============================");

    // Initialize CUDA
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    println!("✅ CUDA device initialized");

    // Test 1: Empty data (should complete quickly)
    println!("\n1. Testing with zero nodes...");
    {
        let pattern_buffer = stream.alloc_zeros::<u8>(1)?;
        let ast_buffer = stream.alloc_zeros::<u8>(1)?;
        let match_buffer = stream.alloc_zeros::<u32>(1)?;

        // SAFETY: All buffers are valid device pointers from alloc_zeros.
        // Using 0 patterns and 0 nodes for empty data test.
        let (pattern_ptr, _pattern_guard) = pattern_buffer.device_ptr(&stream);
        let (ast_ptr, _ast_guard) = ast_buffer.device_ptr(&stream);
        let (match_ptr, _match_guard) = match_buffer.device_ptr(&stream);
        unsafe {
            launch_match_patterns_safe(
                pattern_ptr as *const u8,
                ast_ptr as *const u8,
                match_ptr as *mut u32,
                0,
                0,
            );
        }

        println!("   ✅ Completed (no hang with 0 nodes)");
    }

    // Test 2: Single node
    println!("\n2. Testing with 1 pattern, 1 node...");
    {
        let pattern_buffer = stream.alloc_zeros::<u8>(40)?;
        let ast_buffer = stream.alloc_zeros::<u8>(40)?;
        let match_buffer = stream.alloc_zeros::<u32>(2)?;

        println!("   📊 Allocated buffers");
        println!("   🚀 Launching safe kernel...");

        // SAFETY: All buffers are valid device pointers from alloc_zeros.
        // Parameters (1 pattern, 1 node) match allocation sizes.
        let (pattern_ptr, _pattern_guard) = pattern_buffer.device_ptr(&stream);
        let (ast_ptr, _ast_guard) = ast_buffer.device_ptr(&stream);
        let (match_ptr, _match_guard) = match_buffer.device_ptr(&stream);
        unsafe {
            launch_match_patterns_safe(
                pattern_ptr as *const u8,
                ast_ptr as *const u8,
                match_ptr as *mut u32,
                1,
                1,
            );
        }

        println!("   ✅ Kernel returned successfully");

        // Check results
        let results: Vec<u32> = stream.clone_dtoh(&match_buffer)?;
        println!(
            "   📊 Results: node_id={}, match_flag={}",
            results[0], results[1]
        );
    }

    // Test 3: Multiple nodes
    println!("\n3. Testing with 1 pattern, 100 nodes...");
    {
        let pattern_buffer = stream.alloc_zeros::<u8>(40)?;
        let ast_buffer = stream.alloc_zeros::<u8>(40 * 100)?;
        let match_buffer = stream.alloc_zeros::<u32>(200)?;

        println!("   🚀 Launching safe kernel...");

        // SAFETY: All buffers are valid device pointers from alloc_zeros.
        // Parameters (1 pattern, 100 nodes) match allocation sizes.
        let (pattern_ptr, _pattern_guard) = pattern_buffer.device_ptr(&stream);
        let (ast_ptr, _ast_guard) = ast_buffer.device_ptr(&stream);
        let (match_ptr, _match_guard) = match_buffer.device_ptr(&stream);
        unsafe {
            launch_match_patterns_safe(
                pattern_ptr as *const u8,
                ast_ptr as *const u8,
                match_ptr as *mut u32,
                1,
                100,
            );
        }

        println!("   ✅ Kernel returned successfully");

        // Check first few results
        let slice = match_buffer.slice(0..6);
        let results: Vec<u32> = stream.clone_dtoh(&slice)?;
        println!("   📊 First 3 results:");
        for i in 0..3 {
            println!(
                "      - node_id={}, match_flag={}",
                results[i * 2],
                results[i * 2 + 1]
            );
        }
    }

    println!("\n✅ All tests completed successfully!");
    Ok(())
}
