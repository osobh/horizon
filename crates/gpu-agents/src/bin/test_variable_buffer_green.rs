//! Test variable buffer implementation
//!
//! GREEN phase - tests should pass with basic implementation

use cudarc::driver::{CudaDevice, DeviceSlice};
use gpu_agents::synthesis::variable_buffer::{BufferConfig, VariableBufferManager};
use std::sync::Arc;

fn main() -> anyhow::Result<()> {
    println!("Testing Variable Buffer Implementation (GREEN phase)");
    println!("==================================================");

    let device = CudaDevice::new(0)?;

    // Test 1: Buffer allocation
    println!("\n1. Testing buffer allocation...");
    let mut manager = VariableBufferManager::new(device, BufferConfig::default())?;

    manager.ensure_capacity(1024, 2048, 512)?;
    println!("✅ Buffer allocation succeeded");
    println!("   Memory usage: {} bytes", manager.memory_usage());

    // Test 2: Buffer resizing
    println!("\n2. Testing buffer resizing...");
    let pattern_buffer = manager.get_pattern_buffer(4096)?;
    println!(
        "✅ Got pattern buffer with size: {} bytes",
        pattern_buffer.len()
    );

    let ast_buffer = manager.get_ast_buffer(8192)?;
    println!("✅ Got AST buffer with size: {} bytes", ast_buffer.len());

    // Test 3: Alignment
    println!("\n3. Testing buffer alignment...");
    let small_buffer = manager.get_pattern_buffer(100)?;
    println!(
        "✅ Small buffer (100 bytes requested) has aligned size: {} bytes",
        small_buffer.len()
    );
    assert_eq!(
        small_buffer.len() % 256,
        0,
        "Buffer should be aligned to 256 bytes"
    );
    // Note: CUDA may have minimum allocation sizes (e.g., 4096 bytes)

    // Test 4: Memory usage tracking
    println!("\n4. Testing memory usage tracking...");
    let usage_before = manager.memory_usage();
    manager.ensure_capacity(10240, 20480, 5120)?;
    let usage_after = manager.memory_usage();
    println!(
        "✅ Memory usage before: {} bytes, after: {} bytes",
        usage_before, usage_after
    );

    // Test 5: Buffer clearing
    println!("\n5. Testing buffer clearing...");
    manager.clear();
    let usage_cleared = manager.memory_usage();
    println!("✅ Memory usage after clear: {} bytes", usage_cleared);
    assert_eq!(usage_cleared, 0, "Memory usage should be 0 after clearing");

    println!("\n✅ All tests passed in GREEN phase!");

    Ok(())
}
