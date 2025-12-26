//! Test the original synthesis kernel to see where it hangs

use cudarc::driver::{CudaDevice, DevicePtr};
use gpu_agents::synthesis::launch_match_patterns;
use std::sync::Arc;

fn main() -> anyhow::Result<()> {
    println!("🧪 Original Synthesis Kernel Test");
    println!("=================================");

    // Initialize CUDA
    let device = Arc::new(CudaDevice::new(0)?);
    println!("✅ CUDA device initialized");

    // Test 1: Empty data (should complete quickly)
    println!("\n1. Testing with zero nodes...");
    {
        let pattern_buffer = device.alloc_zeros::<u8>(1)?;
        let ast_buffer = device.alloc_zeros::<u8>(1)?;
        let match_buffer = device.alloc_zeros::<u32>(1)?;

        unsafe {
            launch_match_patterns(
                *pattern_buffer.device_ptr() as *const u8,
                *ast_buffer.device_ptr() as *const u8,
                *match_buffer.device_ptr() as *mut u32,
                0,
                0,
            );
        }

        println!("   ✅ Completed (no hang with 0 nodes)");
    }

    // Test 2: Single node
    println!("\n2. Testing with 1 pattern, 1 node...");
    {
        let pattern_buffer = device.alloc_zeros::<u8>(40)?;
        let ast_buffer = device.alloc_zeros::<u8>(40)?;
        let match_buffer = device.alloc_zeros::<u32>(2)?;

        println!("   📊 Allocated buffers");
        println!("   🚀 Launching original kernel...");
        println!("   ⏱️  This might hang at cudaDeviceSynchronize()...");

        unsafe {
            launch_match_patterns(
                *pattern_buffer.device_ptr() as *const u8,
                *ast_buffer.device_ptr() as *const u8,
                *match_buffer.device_ptr() as *mut u32,
                1,
                1,
            );
        }

        println!("   ✅ Kernel returned successfully!");

        // Check results
        let mut results = vec![0u32; 2];
        device.dtoh_sync_copy_into(&match_buffer, &mut results)?;
        println!(
            "   📊 Results: node_id={}, match_flag={}",
            results[0], results[1]
        );
    }

    println!("\n✅ Test completed without hanging!");
    Ok(())
}
