//! Test CUDA kernel linking and execution

use cudarc::driver::{CudaContext, DevicePtr};
use gpu_agents::consensus::{launch_aggregate_votes, Vote};
use gpu_agents::synthesis::launch_match_patterns;

fn main() -> anyhow::Result<()> {
    println!("🧪 CUDA Kernel Linkage Test");
    println!("===========================");

    // Initialize CUDA
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    println!("✅ CUDA device initialized");

    // Test 1: Consensus kernel (known to work)
    println!("\n1. Testing consensus kernel (should work)...");
    {
        let vote_buffer = stream.alloc_zeros::<Vote>(10)?;
        let count_buffer = stream.alloc_zeros::<u32>(10)?;

        // SAFETY: vote_buffer and count_buffer are valid device pointers from alloc_zeros.
        // Parameters (10 votes, 2 proposals) match allocation sizes.
        unsafe {
            let (vote_ptr, _guard1) = vote_buffer.device_ptr(&stream);
            let (count_ptr, _guard2) = count_buffer.device_ptr(&stream);
            launch_aggregate_votes(
                vote_ptr as *const Vote,
                count_ptr as *mut u32,
                10,
                2,
            );
        }

        match stream.synchronize() {
            Ok(_) => println!("   ✅ Consensus kernel executed successfully"),
            Err(e) => println!("   ❌ Consensus kernel failed: {}", e),
        }
    }

    // Test 2: Synthesis kernel with minimal data
    println!("\n2. Testing synthesis kernel with minimal data...");
    {
        let pattern_buffer = stream.alloc_zeros::<u8>(40)?; // One node
        let ast_buffer = stream.alloc_zeros::<u8>(40)?; // One node
        let match_buffer = stream.alloc_zeros::<u32>(2)?; // One match result

        println!("   📊 Launching kernel with 1 pattern, 1 node...");
        // SAFETY: All buffers are valid device pointers from alloc_zeros.
        // Parameters (1 pattern, 1 node) match minimal allocation sizes.
        unsafe {
            let (pattern_ptr, _g1) = pattern_buffer.device_ptr(&stream);
            let (ast_ptr, _g2) = ast_buffer.device_ptr(&stream);
            let (match_ptr, _g3) = match_buffer.device_ptr(&stream);
            launch_match_patterns(
                pattern_ptr as *const u8,
                ast_ptr as *const u8,
                match_ptr as *mut u32,
                1,
                1,
            );
        }

        println!("   ⏱️  Synchronizing (this might hang)...");
        match stream.synchronize() {
            Ok(_) => {
                println!("   ✅ Synthesis kernel executed successfully");

                // Check results
                let results: Vec<u32> = stream.clone_dtoh(&match_buffer)?;
                println!("   📊 Results: [{}, {}]", results[0], results[1]);
            }
            Err(e) => println!("   ❌ Synthesis kernel failed: {}", e),
        }
    }

    println!("\n✅ Test complete!");
    Ok(())
}
