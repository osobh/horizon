//! Integration tests for gpu_dma_lock kernel module
//!
//! Tests that verify the module works correctly when integrated with the kernel

#![cfg(test)]

use std::fs;
use std::process::Command;
use std::thread;
use std::time::Duration;

/// Test module loading and unloading
#[test]
#[ignore] // Requires root and actual kernel environment
fn test_module_load_unload() {
    // Check if running as root
    if !is_root() {
        eprintln!("Skipping test: requires root privileges");
        return;
    }

    // Build the module first
    assert!(build_module().is_ok());

    // Load module
    let output = Command::new("insmod")
        .arg("gpu_dma_lock.ko")
        .output()
        .expect("Failed to load module");

    assert!(output.status.success(), "Module load failed: {:?}", output);

    // Verify module is loaded
    let lsmod = Command::new("lsmod").output().expect("Failed to run lsmod");

    let lsmod_str = String::from_utf8_lossy(&lsmod.stdout);
    assert!(lsmod_str.contains("gpu_dma_lock"));

    // Verify /proc entries exist
    assert!(fs::metadata("/proc/swarm/gpu").is_ok());

    // Unload module
    let output = Command::new("rmmod")
        .arg("gpu_dma_lock")
        .output()
        .expect("Failed to unload module");

    assert!(
        output.status.success(),
        "Module unload failed: {:?}",
        output
    );
}

/// Test CUDA interception
#[test]
#[ignore] // Requires CUDA environment
fn test_cuda_interception() {
    if !is_cuda_available() {
        eprintln!("Skipping test: CUDA not available");
        return;
    }

    // Load module with CUDA interception enabled
    load_module_with_params("intercept_cuda=1");

    // Run simple CUDA allocation
    let test_program = r#"
        #include <cuda_runtime.h>
        #include <stdio.h>
        
        int main() {
            void *ptr;
            size_t size = 1024 * 1024 * 100; // 100MB
            
            cudaError_t err = cudaMalloc(&ptr, size);
            if (err != cudaSuccess) {
                printf("Allocation failed: %s\n", cudaGetErrorString(err));
                return 1;
            }
            
            cudaFree(ptr);
            return 0;
        }
    "#;

    // Compile and run test program
    compile_cuda_test(test_program, "test_alloc");
    let output = Command::new("./test_alloc")
        .output()
        .expect("Failed to run test program");

    assert!(output.status.success());

    // Check if allocation was tracked
    let proc_output =
        fs::read_to_string("/proc/swarm/gpu/0/allocations").expect("Failed to read proc file");

    assert!(proc_output.contains("100M"));

    unload_module();
}

/// Test memory quota enforcement
#[test]
#[ignore] // Requires kernel module environment
fn test_quota_enforcement() {
    load_module();

    // Set quota for test agent
    fs::write("/proc/swarm/gpu/agents/1000/quota", "1073741824") // 1GB
        .expect("Failed to set quota");

    // Try to allocate within quota
    let result = allocate_gpu_memory(1000, 512 << 20); // 512MB
    assert!(result.is_ok());

    // Try to allocate more, should succeed
    let result = allocate_gpu_memory(1000, 256 << 20); // 256MB
    assert!(result.is_ok());

    // Try to exceed quota, should fail
    let result = allocate_gpu_memory(1000, 512 << 20); // 512MB more
    assert!(result.is_err());

    // Check statistics
    let stats =
        fs::read_to_string("/proc/swarm/gpu/agents/1000/stats").expect("Failed to read stats");

    assert!(stats.contains("allocated: 768M"));
    assert!(stats.contains("quota: 1024M"));

    unload_module();
}

/// Test DMA access control
#[test]
#[ignore] // Requires kernel module and hardware access
fn test_dma_access_control() {
    load_module();

    // Grant DMA access to agent
    fs::write(
        "/proc/swarm/gpu/agents/2000/dma_permissions",
        "0x100000000-0x200000000:rw",
    )
    .expect("Failed to set DMA permissions");

    // Test valid access
    let result = test_dma_access(2000, 0x150000000, "read");
    assert!(result.is_ok());

    // Test invalid access (outside range)
    let result = test_dma_access(2000, 0x250000000, "read");
    assert!(result.is_err());

    // Test unauthorized agent
    let result = test_dma_access(2001, 0x150000000, "read");
    assert!(result.is_err());

    unload_module();
}

/// Test multi-GPU support
#[test]
#[ignore] // Requires multiple GPUs
fn test_multi_gpu_support() {
    let gpu_count = get_gpu_count();
    if gpu_count < 2 {
        eprintln!("Skipping test: requires multiple GPUs");
        return;
    }

    load_module();

    // Verify all GPUs are detected
    for i in 0..gpu_count {
        let proc_path = format!("/proc/swarm/gpu/{}/info", i);
        assert!(fs::metadata(&proc_path).is_ok());

        let info = fs::read_to_string(&proc_path).expect("Failed to read GPU info");

        assert!(info.contains("device_id"));
        assert!(info.contains("total_memory"));
    }

    // Test allocation on different GPUs
    for i in 0..gpu_count {
        let result = allocate_gpu_memory_on_device(3000 + i, 1 << 30, i); // 1GB
        assert!(result.is_ok());
    }

    // Verify allocations are tracked per device
    for i in 0..gpu_count {
        let stats = fs::read_to_string(format!("/proc/swarm/gpu/{}/stats", i))
            .expect("Failed to read stats");

        assert!(stats.contains("allocated: 1024M"));
    }

    unload_module();
}

/// Test concurrent allocations
#[test]
#[ignore] // Requires kernel module
fn test_concurrent_operations() {
    load_module();

    let threads = 10;
    let allocations_per_thread = 100;

    let handles: Vec<_> = (0..threads)
        .map(|t| {
            thread::spawn(move || {
                for i in 0..allocations_per_thread {
                    let agent_id = 4000 + t * 1000 + i;
                    let size = 10 << 20; // 10MB
                    let _ = allocate_gpu_memory(agent_id, size);
                    thread::sleep(Duration::from_micros(10));
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }

    // Verify no corruption in tracking
    let global_stats =
        fs::read_to_string("/proc/swarm/gpu/stats").expect("Failed to read global stats");

    let total_allocs: usize = global_stats
        .lines()
        .find(|l| l.starts_with("total_allocations:"))
        .and_then(|l| l.split(':').nth(1))
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(0);

    assert_eq!(total_allocs, threads * allocations_per_thread);

    unload_module();
}

/// Test GPU context isolation
#[test]
#[ignore] // Requires kernel module and GPU
fn test_gpu_context_isolation() {
    if !is_cuda_available() {
        eprintln!("Skipping test: CUDA not available");
        return;
    }

    load_module();

    // Create contexts for different agents
    create_gpu_context(5000, 0);
    create_gpu_context(5001, 0);

    // Verify contexts are isolated
    let isolation_info =
        fs::read_to_string("/proc/swarm/gpu/0/contexts").expect("Failed to read context info");

    assert!(isolation_info.contains("agent_5000: context_1"));
    assert!(isolation_info.contains("agent_5001: context_2"));
    assert!(isolation_info.contains("isolated: true"));

    unload_module();
}

/// Test memory pressure handling
#[test]
#[ignore] // Requires kernel module
fn test_memory_pressure() {
    load_module();

    // Set pressure thresholds
    fs::write("/proc/swarm/gpu/0/pressure_warning", "80").expect("Failed to set warning threshold");
    fs::write("/proc/swarm/gpu/0/pressure_critical", "95")
        .expect("Failed to set critical threshold");

    // Allocate memory to create pressure
    let device_memory = get_gpu_memory(0);
    let allocation_size = (device_memory as f64 * 0.85) as usize; // 85% of total

    allocate_gpu_memory_on_device(6000, allocation_size, 0).unwrap();

    // Check pressure status
    let pressure =
        fs::read_to_string("/proc/swarm/gpu/0/pressure").expect("Failed to read pressure");

    assert!(pressure.contains("level: warning"));
    assert!(pressure.contains("usage: 85%"));

    // Allocate more to reach critical
    allocate_gpu_memory_on_device(6001, (device_memory as f64 * 0.11) as usize, 0).unwrap();

    let pressure =
        fs::read_to_string("/proc/swarm/gpu/0/pressure").expect("Failed to read pressure");

    assert!(pressure.contains("level: critical"));

    unload_module();
}

// Helper functions

fn is_root() -> bool {
    unsafe { libc::geteuid() == 0 }
}

fn is_cuda_available() -> bool {
    Command::new("nvidia-smi")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn build_module() -> std::io::Result<()> {
    Command::new("make").current_dir("..").status().map(|_| ())
}

fn load_module() {
    load_module_with_params("")
}

fn load_module_with_params(params: &str) {
    let mut cmd = Command::new("insmod");
    cmd.arg("../gpu_dma_lock.ko");
    if !params.is_empty() {
        cmd.arg(params);
    }

    let output = cmd.output().expect("Failed to load module");
    if !output.status.success() {
        panic!(
            "Module load failed: {:?}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

fn unload_module() {
    Command::new("rmmod")
        .arg("gpu_dma_lock")
        .output()
        .expect("Failed to unload module");
}

fn get_gpu_count() -> u32 {
    Command::new("nvidia-smi")
        .arg("--query-gpu=count")
        .arg("--format=csv,noheader")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(0)
}

fn get_gpu_memory(device: u32) -> usize {
    let output = Command::new("nvidia-smi")
        .arg("--id")
        .arg(device.to_string())
        .arg("--query-gpu=memory.total")
        .arg("--format=csv,noheader,nounits")
        .output()
        .expect("Failed to query GPU memory");

    String::from_utf8_lossy(&output.stdout)
        .trim()
        .parse::<usize>()
        .unwrap_or(0)
        << 20 // Convert MB to bytes
}

fn allocate_gpu_memory(agent_id: u64, size: usize) -> Result<(), String> {
    let cmd = format!("alloc {} {} 0", agent_id, size);
    fs::write("/proc/swarm/gpu/ctl", cmd).map_err(|e| e.to_string())
}

fn allocate_gpu_memory_on_device(agent_id: u64, size: usize, device: u32) -> Result<(), String> {
    let cmd = format!("alloc {} {} {}", agent_id, size, device);
    fs::write("/proc/swarm/gpu/ctl", cmd).map_err(|e| e.to_string())
}

fn test_dma_access(agent_id: u64, addr: usize, mode: &str) -> Result<(), String> {
    let cmd = format!("dma_test {} {} {}", agent_id, addr, mode);
    fs::write("/proc/swarm/gpu/ctl", cmd).map_err(|e| e.to_string())
}

fn create_gpu_context(agent_id: u64, device: u32) {
    let cmd = format!("create_context {} {}", agent_id, device);
    fs::write("/proc/swarm/gpu/ctl", cmd).expect("Failed to create context");
}

fn compile_cuda_test(source: &str, name: &str) {
    fs::write(format!("{}.cu", name), source).expect("Failed to write source");

    Command::new("nvcc")
        .arg("-o")
        .arg(name)
        .arg(format!("{}.cu", name))
        .status()
        .expect("Failed to compile CUDA test");
}

#[cfg(test)]
mod mock_tests {
    use super::*;

    /// Mock test for environments without kernel access
    #[test]
    fn test_mock_allocation_tracking() {
        // This test runs without requiring actual kernel module
        println!("Running mock allocation tracking test");

        // Simulate allocation tracking
        let mut allocations = std::collections::HashMap::new();

        // Track some allocations
        allocations.insert((1000, 0), 1 << 30); // Agent 1000, 1GB on GPU0
        allocations.insert((1001, 1), 2 << 30); // Agent 1001, 2GB on GPU1

        // Verify tracking
        assert_eq!(allocations.get(&(1000, 0)), Some(&(1 << 30)));
        assert_eq!(allocations.get(&(1001, 1)), Some(&(2 << 30)));

        // Calculate total
        let total: usize = allocations.values().sum();
        assert_eq!(total, 3 << 30);
    }
}
