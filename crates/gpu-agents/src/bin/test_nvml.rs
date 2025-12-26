//! Test NVML initialization

use nvml_wrapper::Nvml;

fn main() {
    println!("Testing NVML initialization...");

    match Nvml::init() {
        Ok(nvml) => {
            println!("✓ NVML initialized successfully!");

            // Try to get device count
            match nvml.device_count() {
                Ok(count) => println!("✓ Found {} NVIDIA GPU(s)", count),
                Err(e) => println!("✗ Failed to get device count: {}", e),
            }

            // Try to get first device
            match nvml.device_by_index(0) {
                Ok(device) => {
                    println!("✓ Got device 0");

                    // Get device name
                    match device.name() {
                        Ok(name) => println!("  Device name: {}", name),
                        Err(e) => println!("  Failed to get name: {}", e),
                    }

                    // Get utilization
                    match device.utilization_rates() {
                        Ok(util) => println!("  GPU utilization: {}%", util.gpu),
                        Err(e) => println!("  Failed to get utilization: {}", e),
                    }
                }
                Err(e) => println!("✗ Failed to get device 0: {}", e),
            }
        }
        Err(e) => {
            println!("✗ Failed to initialize NVML: {}", e);
            println!("  This might be due to:");
            println!("  - Missing NVIDIA driver");
            println!("  - Insufficient permissions");
            println!("  - NVML library not found");
        }
    }
}
