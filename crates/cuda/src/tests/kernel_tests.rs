//! Tests for CUDA kernel module

use crate::error::CudaResult;
use crate::kernel::*;
use std::path::PathBuf;

#[test]
fn test_kernel_source_types() {
    // Test PTX source
    let ptx_source = KernelSource::Ptx(
        r#"
        .version 7.0
        .target sm_70
        .address_size 64
    "#
        .to_string(),
    );

    match ptx_source {
        KernelSource::Ptx(s) => assert!(s.contains(".version")),
        _ => panic!("Expected PTX source"),
    }

    // Test CUBIN source
    let cubin_data = vec![0xDE, 0xAD, 0xBE, 0xEF];
    let cubin_source = KernelSource::Cubin(cubin_data.clone());

    match cubin_source {
        KernelSource::Cubin(data) => assert_eq!(data, cubin_data),
        _ => panic!("Expected CUBIN source"),
    }

    // Test CUDA C source
    let cuda_c_source = KernelSource::CudaC(
        r#"
        __global__ void vector_add(float* a, float* b, float* c, int n) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                c[idx] = a[idx] + b[idx];
            }
        }
    "#
        .to_string(),
    );

    match cuda_c_source {
        KernelSource::CudaC(s) => assert!(s.contains("__global__")),
        _ => panic!("Expected CUDA C source"),
    }
}

#[test]
fn test_kernel_module_metadata() {
    let metadata = KernelMetadata {
        name: "test_module".to_string(),
        source_type: SourceType::Ptx,
        compile_options: CompileOptions::default(),
        registers_used: 255,
        shared_memory: 49152,
        constant_memory: 0,
        local_memory: 0,
        max_threads: 1024,
    };

    assert_eq!(metadata.name, "test_module");
    assert_eq!(metadata.source_type, SourceType::Ptx);
    assert_eq!(metadata.shared_memory, 49152);
}

#[cfg(cuda_mock)]
#[test]
fn test_mock_kernel_module_creation() {
    let source = KernelSource::Ptx("mock ptx".to_string());
    let module = KernelModule::from_source(source).unwrap();

    let metadata = module.metadata();
    assert!(!metadata.name.is_empty());
    assert_eq!(metadata.source_type, SourceType::Ptx);
}

#[cfg(cuda_mock)]
#[test]
fn test_mock_kernel_loading() {
    let source = KernelSource::CudaC(
        r#"
        __global__ void test_kernel(int* data) {
            data[threadIdx.x] = threadIdx.x * 2;
        }
    "#
        .to_string(),
    );

    let module = KernelModule::from_source(source)?;
    let kernel = module.get_kernel("test_kernel").unwrap();

    assert_eq!(kernel.name(), "test_kernel");
}

#[cfg(cuda_mock)]
#[test]
fn test_mock_kernel_launch() {
    let module = KernelModule::from_source(KernelSource::Ptx("".to_string())).unwrap();
    let kernel = module.get_kernel("mock_kernel").unwrap();

    let config = LaunchConfig {
        grid_dim: (256, 1, 1),
        block_dim: (128, 1, 1),
        shared_memory_bytes: 0,
        stream: None,
    };

    // In mock mode, launch should succeed
    assert!(kernel.launch(&config, vec![]).is_ok());
}

#[test]
fn test_launch_config_creation() {
    let config = LaunchConfig {
        grid_dim: (100, 1, 1),
        block_dim: (256, 1, 1),
        shared_memory_bytes: 1024,
        stream: None,
    };

    assert_eq!(config.grid_dim, (100, 1, 1));
    assert_eq!(config.block_dim, (256, 1, 1));
    assert_eq!(config.shared_memory_bytes, 1024);
    assert!(config.stream.is_none());
}

#[test]
fn test_launch_config_validation() {
    // Test various launch configurations
    let configs = vec![
        LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_memory_bytes: 0,
            stream: None,
        },
        LaunchConfig {
            grid_dim: (65535, 65535, 65535),
            block_dim: (1024, 1, 1),
            shared_memory_bytes: 49152,
            stream: None,
        },
        LaunchConfig {
            grid_dim: (100, 200, 1),
            block_dim: (32, 32, 1),
            shared_memory_bytes: 16384,
            stream: None,
        },
    ];

    for config in configs {
        // Verify dimensions are positive
        assert!(config.grid_dim.0 > 0);
        assert!(config.grid_dim.1 > 0);
        assert!(config.grid_dim.2 > 0);
        assert!(config.block_dim.0 > 0);
        assert!(config.block_dim.1 > 0);
        assert!(config.block_dim.2 > 0);
    }
}

#[cfg(cuda_mock)]
#[test]
fn test_kernel_argument_types() {
    let module = KernelModule::from_source(KernelSource::Ptx("".to_string())).unwrap();
    let kernel = module.get_kernel("test_kernel").unwrap();

    // Test various argument types
    let args = vec![
        KernelArg::Scalar(42i32),
        KernelArg::Scalar(3.14f32),
        KernelArg::Scalar(true),
        KernelArg::Pointer(0x1000 as *mut u8),
    ];

    let config = LaunchConfig::default();
    assert!(kernel.launch(&config, args).is_ok());
}

#[cfg(cuda_mock)]
#[test]
fn test_kernel_module_from_file() {
    // Test loading from PTX file
    let result = KernelModule::from_ptx_file(PathBuf::from("nonexistent.ptx"));
    assert!(result.is_err()); // File doesn't exist

    // Test loading from CUBIN file
    let result = KernelModule::from_cubin_file(PathBuf::from("nonexistent.cubin"));
    assert!(result.is_err()); // File doesn't exist
}

#[test]
fn test_kernel_source_size() {
    use std::mem::size_of_val;

    let ptx = KernelSource::Ptx("small".to_string());
    let cubin = KernelSource::Cubin(vec![0; 100]);
    let cuda_c = KernelSource::CudaC("code".to_string());

    // Verify enum doesn't have excessive overhead
    assert!(size_of_val(&ptx) < 100);
    assert!(size_of_val(&cubin) < 200);
    assert!(size_of_val(&cuda_c) < 100);
}

#[cfg(cuda_mock)]
#[test]
fn test_kernel_occupancy_calculation() {
    let module = KernelModule::from_source(KernelSource::Ptx("".to_string())).unwrap();
    let kernel = module.get_kernel("test_kernel").unwrap();

    // Test occupancy calculation
    let block_size = 256;
    let dynamic_shared_mem = 1024;

    let occupancy = kernel.calculate_occupancy(block_size, dynamic_shared_mem)?;

    assert!(occupancy.active_blocks > 0);
    assert!(occupancy.active_warps > 0);
    assert!(occupancy.active_threads > 0);
    assert!(occupancy.occupancy_percentage > 0.0);
    assert!(occupancy.occupancy_percentage <= 100.0);
}

#[cfg(cuda_mock)]
#[test]
fn test_kernel_attributes() {
    let module = KernelModule::from_source(KernelSource::Ptx("".to_string())).unwrap();
    let kernel = module.get_kernel("test_kernel").unwrap();

    let attrs = kernel.get_attributes().unwrap();

    assert!(attrs.max_threads_per_block > 0);
    assert!(attrs.shared_memory_bytes >= 0);
    assert!(attrs.const_memory_bytes >= 0);
    assert!(attrs.local_memory_bytes >= 0);
    assert!(attrs.num_registers >= 0);
    assert!(attrs.ptx_version > 0);
    assert!(attrs.binary_version > 0);
}

#[cfg(cuda_mock)]
#[test]
fn test_kernel_cache() {
    let module = KernelModule::from_source(KernelSource::Ptx("".to_string())).unwrap();

    // Getting the same kernel multiple times should return the same instance
    let kernel1 = module.get_kernel("cached_kernel").unwrap();
    let kernel2 = module.get_kernel("cached_kernel")?;

    assert_eq!(kernel1.name(), kernel2.name());
}

#[test]
fn test_launch_config_builder() {
    let config = LaunchConfigBuilder::new()
        .grid_dim(100, 1, 1)
        .block_dim(256, 1, 1)
        .shared_memory(4096)
        .build();

    assert_eq!(config.grid_dim, (100, 1, 1));
    assert_eq!(config.block_dim, (256, 1, 1));
    assert_eq!(config.shared_memory_bytes, 4096);
}

#[cfg(cuda_mock)]
#[test]
fn test_kernel_profiling_info() {
    let module = KernelModule::from_source(KernelSource::Ptx("".to_string())).unwrap();
    let kernel = module.get_kernel("test_kernel").unwrap();

    let config = LaunchConfig::default();
    kernel.launch(&config, vec![])?;

    // In a real implementation, we'd get profiling info
    let profile = kernel.get_last_profile_info();
    assert!(profile.is_some());

    let info = profile.unwrap();
    assert!(info.duration_ns > 0);
    assert!(info.memory_transferred_bytes >= 0);
}

#[test]
fn test_kernel_compilation_options() {
    let options = CompilationOptions {
        arch: "sm_80".to_string(),
        opt_level: 3,
        debug_info: false,
        verbose: false,
        max_registers: Some(64),
        defines: vec![("BLOCK_SIZE".to_string(), "256".to_string())],
        include_paths: vec![PathBuf::from("/usr/local/cuda/include")],
    };

    assert_eq!(options.arch, "sm_80");
    assert_eq!(options.opt_level, 3);
    assert!(!options.debug_info);
    assert_eq!(options.max_registers, Some(64));
    assert_eq!(options.defines.len(), 1);
}

#[cfg(cuda_mock)]
#[test]
fn test_runtime_compilation() {
    let cuda_source = r#"
        extern "C" __global__ void dynamic_kernel(float* data, int n) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                data[idx] = data[idx] * 2.0f;
            }
        }
    "#;

    let options = CompilationOptions::default();
    let module = KernelModule::compile_cuda_source(cuda_source, &options).unwrap();
    let kernel = module.get_kernel("dynamic_kernel").unwrap();

    assert_eq!(kernel.name(), "dynamic_kernel");
}

#[cfg(cuda_mock)]
#[test]
fn test_kernel_error_handling() {
    let module = KernelModule::from_source(KernelSource::Ptx("".to_string())).unwrap();

    // Test getting non-existent kernel
    let result = module.get_kernel("nonexistent_kernel");
    assert!(result.is_err());

    // Test invalid launch configuration
    let kernel = module.get_kernel("test_kernel")?;
    let invalid_config = LaunchConfig {
        grid_dim: (0, 0, 0), // Invalid: zero dimensions
        block_dim: (0, 0, 0),
        shared_memory_bytes: 0,
        stream: None,
    };

    let result = kernel.launch(&invalid_config, vec![]);
    assert!(result.is_err());
}

#[test]
fn test_kernel_argument_serialization() {
    let args = vec![
        KernelArg::Scalar(42i32),
        KernelArg::Scalar(3.14f64),
        KernelArg::Pointer(0x1000 as *mut u8),
    ];

    // Verify arguments can be properly sized
    for arg in args {
        match arg {
            KernelArg::Scalar(v) => assert_eq!(std::mem::size_of_val(&v), 4),
            KernelArg::Scalar(v) => assert_eq!(std::mem::size_of_val(&v), 8),
            KernelArg::Pointer(p) => {
                assert_eq!(std::mem::size_of_val(&p), std::mem::size_of::<usize>())
            }
        }
    }
}

#[cfg(cuda_mock)]
#[test]
fn test_kernel_module_lifecycle() {
    // Test module creation, kernel loading, and cleanup
    {
        let module = KernelModule::from_source(KernelSource::Ptx("".to_string())).unwrap();
        let _kernel = module.get_kernel("test_kernel").unwrap();
        // Module and kernel dropped here
    }

    // Should be able to create new module
    let _module2 = KernelModule::new(KernelSource::Ptx("".to_string()))?;
}

#[cfg(cuda_mock)]
#[test]
fn test_concurrent_kernel_launches() {
    use std::sync::Arc;
    use std::thread;

    let module = Arc::new(KernelModule::new(KernelSource::Ptx("".to_string())).unwrap());
    let kernel = Arc::new(module.get_kernel("test_kernel")?);

    let handles: Vec<_> = (0..10)
        .map(|i| {
            let kernel_clone = kernel.clone();
            thread::spawn(move || {
                let config = LaunchConfig {
                    grid_dim: (i + 1, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_memory_bytes: 0,
                    stream: None,
                };
                kernel_clone.launch(&config, vec![])
            })
        })
        .collect();

    for handle in handles {
        assert!(handle.join().unwrap().is_ok());
    }
}
