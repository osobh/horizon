//! Test CUDA kernel linking and execution

use cudarc::driver::{CudaDevice, DevicePtr};
use gpu_agents::consensus::{launch_aggregate_votes, Vote};
use gpu_agents::synthesis::launch_match_patterns;
use std::sync::Arc;

fn main() -> anyhow::Result<()> {
    println!("🧪 CUDA Kernel Linkage Test");
    println!("===========================");

    // Initialize CUDA
    let device = Arc::new(CudaDevice::new(0)?);
    println!("✅ CUDA device initialized");

    // Test 1: Consensus kernel (known to work)
    println!("\n1. Testing consensus kernel (should work)...");
    {
        let vote_buffer = device.alloc_zeros::<Vote>(10)?;
        let count_buffer = device.alloc_zeros::<u32>(10)?;

        unsafe {
            launch_aggregate_votes(
                *vote_buffer.device_ptr() as *const Vote,
                *count_buffer.device_ptr() as *mut u32,
                10,
                2,
            );
        }

        match device.synchronize() {
            Ok(_) => println!("   ✅ Consensus kernel executed successfully"),
            Err(e) => println!("   ❌ Consensus kernel failed: {}", e),
        }
    }

    // Test 2: Synthesis kernel with minimal data
    println!("\n2. Testing synthesis kernel with minimal data...");
    {
        let pattern_buffer = device.alloc_zeros::<u8>(40)?; // One node
        let ast_buffer = device.alloc_zeros::<u8>(40)?; // One node
        let match_buffer = device.alloc_zeros::<u32>(2)?; // One match result

        println!("   📊 Launching kernel with 1 pattern, 1 node...");
        unsafe {
            launch_match_patterns(
                *pattern_buffer.device_ptr() as *const u8,
                *ast_buffer.device_ptr() as *const u8,
                *match_buffer.device_ptr() as *mut u32,
                1,
                1,
            );
        }

        println!("   ⏱️  Synchronizing (this might hang)...");
        match device.synchronize() {
            Ok(_) => {
                println!("   ✅ Synthesis kernel executed successfully");

                // Check results
                let mut results = vec![0u32; 2];
                device.dtoh_sync_copy_into(&match_buffer, &mut results)?;
                println!("   📊 Results: [{}, {}]", results[0], results[1]);
            }
            Err(e) => println!("   ❌ Synthesis kernel failed: {}", e),
        }
    }

    println!("\n✅ Test complete!");
    Ok(())
}
