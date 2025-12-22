//! Tests for CUDA code compilation module

use crate::compiler::*;
use crate::error::SynthesisResult;

#[test]
    fn test_cuda_compiler_creation() -> Result<(), Box<dyn std::error::Error>> {
    let compiler = CudaCompiler::new().unwrap();
    assert!(compiler.is_initialized());
}

#[test]
    fn test_compiler_options() -> Result<(), Box<dyn std::error::Error>> {
    let mut options = CompilerOptions::default();
    assert_eq!(options.optimization_level, OptimizationLevel::Balanced);
    assert_eq!(options.target_arch, "sm_70");
    assert!(!options.debug_symbols);
    
    options.optimization_level = OptimizationLevel::Aggressive;
    options.debug_symbols = true;
    options.target_arch = "sm_86".to_string();
    
    assert_eq!(options.optimization_level, OptimizationLevel::Aggressive);
    assert!(options.debug_symbols);
    assert_eq!(options.target_arch, "sm_86");
}

#[test]
    fn test_compilation_target() -> Result<(), Box<dyn std::error::Error>> {
    let targets = vec![
        CompilationTarget::PTX,
        CompilationTarget::CUBIN,
        CompilationTarget::SASS,
    ];
    
    for target in targets {
        let compiler = CudaCompiler::with_target(target)?;
        assert_eq!(compiler.get_target(), target);
    }
}

#[test]
    fn test_simple_kernel_compilation() -> Result<(), Box<dyn std::error::Error>> {
    let compiler = CudaCompiler::new().unwrap();
    
    let kernel_source = r#"
        extern "C" __global__ void vector_add(float* a, float* b, float* c, int n) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                c[idx] = a[idx] + b[idx];
            }
        }
    "#;
    
    let result = compiler.compile_source(kernel_source, &CompilerOptions::default());
    assert!(result.is_ok());
    
    let binary = result.unwrap();
    assert!(!binary.is_empty());
    assert!(binary.len() > 100); // Reasonable size check
}

#[test]
    fn test_matrix_multiplication_compilation() -> Result<(), Box<dyn std::error::Error>> {
    let compiler = CudaCompiler::new().unwrap();
    
    let kernel_source = r#"
        extern "C" __global__ void matrix_mul(float* A, float* B, float* C, 
                                              int M, int N, int K) {
            int row = blockIdx.y * blockDim.y + threadIdx.y;
            int col = blockIdx.x * blockDim.x + threadIdx.x;
            
            if (row < M && col < N) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++) {
                    sum += A[row * K + k] * B[k * N + col];
                }
                C[row * N + col] = sum;
            }
        }
    "#;
    
    let options = CompilerOptions {
        optimization_level: OptimizationLevel::Aggressive,
        target_arch: "sm_80".to_string(),
        debug_symbols: false,
        include_paths: vec![],
        defines: vec![],
        compile_flags: vec!["-use_fast_math".to_string()],
    };
    
    let result = compiler.compile_source(kernel_source, &options);
    assert!(result.is_ok());
}

#[test]
    fn test_shared_memory_kernel_compilation() -> Result<(), Box<dyn std::error::Error>> {
    let compiler = CudaCompiler::new().unwrap();
    
    let kernel_source = r#"
        extern "C" __global__ void shared_memory_reduce(float* input, float* output, int n) {
            extern __shared__ float sdata[];
            
            int tid = threadIdx.x;
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            
            sdata[tid] = (idx < n) ? input[idx] : 0.0f;
            __syncthreads();
            
            for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    sdata[tid] += sdata[tid + s];
                }
                __syncthreads();
            }
            
            if (tid == 0) {
                output[blockIdx.x] = sdata[0];
            }
        }
    "#;
    
    let result = compiler.compile_source(kernel_source, &CompilerOptions::default());
    assert!(result.is_ok());
}

#[test]
    fn test_template_kernel_compilation() -> Result<(), Box<dyn std::error::Error>> {
    let compiler = CudaCompiler::new().unwrap();
    
    let kernel_source = r#"
        template<typename T>
        __global__ void template_add(T* a, T* b, T* c, int n) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                c[idx] = a[idx] + b[idx];
            }
        }
        
        // Explicit instantiation
        template __global__ void template_add<float>(float*, float*, float*, int);
        template __global__ void template_add<double>(double*, double*, double*, int);
    "#;
    
    let result = compiler.compile_source(kernel_source, &CompilerOptions::default());
    assert!(result.is_ok());
}

#[test]
    fn test_compilation_with_includes() -> Result<(), Box<dyn std::error::Error>> {
    let compiler = CudaCompiler::new().unwrap();
    
    let kernel_source = r#"
        #include <cuda_runtime.h>
        #include <device_launch_parameters.h>
        
        extern "C" __global__ void cuda_runtime_kernel(float* data, int n) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                data[idx] = sqrtf(data[idx]);
            }
        }
    "#;
    
    let options = CompilerOptions {
        include_paths: vec!["/usr/local/cuda/include".to_string()],
        ..CompilerOptions::default()
    };
    
    let result = compiler.compile_source(kernel_source, &options);
    assert!(result.is_ok());
}

#[test]
    fn test_compilation_with_defines() -> Result<(), Box<dyn std::error::Error>> {
    let compiler = CudaCompiler::new().unwrap();
    
    let kernel_source = r#"
        #ifdef BLOCK_SIZE
            #define THREADS BLOCK_SIZE
        #else
            #define THREADS 256
        #endif
        
        extern "C" __global__ void defined_kernel(float* data, int n) {
            int idx = blockIdx.x * THREADS + threadIdx.x;
            if (idx < n) {
                data[idx] *= 2.0f;
            }
        }
    "#;
    
    let options = CompilerOptions {
        defines: vec![("BLOCK_SIZE".to_string(), "512".to_string())],
        ..CompilerOptions::default()
    };
    
    let result = compiler.compile_source(kernel_source, &options);
    assert!(result.is_ok());
}

#[test]
    fn test_optimization_levels() -> Result<(), Box<dyn std::error::Error>> {
    let compiler = CudaCompiler::new().unwrap();
    
    let kernel_source = r#"
        extern "C" __global__ void optimization_test(float* data, int n) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                float x = data[idx];
                float result = x * x * x + 2.0f * x * x + x + 1.0f;
                data[idx] = result;
            }
        }
    "#;
    
    let optimization_levels = vec![
        OptimizationLevel::Debug,
        OptimizationLevel::Fast,
        OptimizationLevel::Balanced,
        OptimizationLevel::Aggressive,
        OptimizationLevel::Size,
    ];
    
    for opt_level in optimization_levels {
        let options = CompilerOptions {
            optimization_level: opt_level,
            ..CompilerOptions::default()
        };
        
        let result = compiler.compile_source(kernel_source, &options);
        assert!(result.is_ok());
        
        let binary = result.unwrap();
        assert!(!binary.is_empty());
    }
}

#[test]
    fn test_compute_capability_targeting() -> Result<(), Box<dyn std::error::Error>> {
    let compiler = CudaCompiler::new().unwrap();
    
    let kernel_source = r#"
        extern "C" __global__ void capability_test(float* data, int n) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                data[idx] = __fmaf_rn(data[idx], 2.0f, 1.0f);
            }
        }
    "#;
    
    let architectures = vec!["sm_35", "sm_50", "sm_60", "sm_70", "sm_75", "sm_80", "sm_86"];
    
    for arch in architectures {
        let options = CompilerOptions {
            target_arch: arch.to_string(),
            ..CompilerOptions::default()
        };
        
        let result = compiler.compile_source(kernel_source, &options);
        assert!(result.is_ok());
    }
}

#[test]
    fn test_compilation_error_handling() -> Result<(), Box<dyn std::error::Error>> {
    let compiler = CudaCompiler::new().unwrap();
    
    let invalid_sources = vec![
        "", // Empty source
        "invalid C++ code !!!",
        "extern \"C\" __global__ void incomplete_kernel(", // Syntax error
        "__global__ void wrong_declaration() { undefined_function(); }", // Undefined function
    ];
    
    for source in invalid_sources {
        let result = compiler.compile_source(source, &CompilerOptions::default());
        assert!(result.is_err());
    }
}

#[test]
    fn test_compilation_warnings() -> Result<(), Box<dyn std::error::Error>> {
    let compiler = CudaCompiler::new().unwrap();
    
    let warning_source = r#"
        extern "C" __global__ void warning_kernel(float* data, int n) {
            int unused_variable = 42;
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                data[idx] = 1.0f;
            }
        }
    "#;
    
    let options = CompilerOptions {
        compile_flags: vec!["-Xcompiler".to_string(), "-Wall".to_string()],
        ..CompilerOptions::default()
    };
    
    let result = compiler.compile_source(warning_source, &options);
    assert!(result.is_ok());
    
    // Should have compilation info including warnings
    let info = compiler.get_last_compilation_info().unwrap();
    assert!(!info.warnings.is_empty());
}

#[test]
    fn test_ptx_generation() -> Result<(), Box<dyn std::error::Error>> {
    let compiler = CudaCompiler::with_target(CompilationTarget::PTX).unwrap();
    
    let kernel_source = r#"
        extern "C" __global__ void ptx_kernel(int* data, int n) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                data[idx] = idx;
            }
        }
    "#;
    
    let result = compiler.compile_source(kernel_source, &CompilerOptions::default());
    assert!(result.is_ok());
    
    let ptx_code = result.unwrap();
    assert!(ptx_code.starts_with(b".version"));
}

#[test]
    fn test_sass_generation() -> Result<(), Box<dyn std::error::Error>> {
    let compiler = CudaCompiler::with_target(CompilationTarget::SASS).unwrap();
    
    let kernel_source = r#"
        extern "C" __global__ void sass_kernel(float* data, int n) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                data[idx] = data[idx] * data[idx];
            }
        }
    "#;
    
    let result = compiler.compile_source(kernel_source, &CompilerOptions::default());
    assert!(result.is_ok());
    
    let sass_code = result.unwrap();
    assert!(!sass_code.is_empty());
}

#[test]
    fn test_multiple_kernels_compilation() -> Result<(), Box<dyn std::error::Error>> {
    let compiler = CudaCompiler::new().unwrap();
    
    let kernel_source = r#"
        extern "C" __global__ void kernel1(float* a, float* b, int n) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                b[idx] = a[idx] + 1.0f;
            }
        }
        
        extern "C" __global__ void kernel2(float* a, float* b, int n) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                b[idx] = a[idx] * 2.0f;
            }
        }
        
        extern "C" __device__ float helper_function(float x) {
            return x * x + 1.0f;
        }
        
        extern "C" __global__ void kernel3(float* data, int n) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                data[idx] = helper_function(data[idx]);
            }
        }
    "#;
    
    let result = compiler.compile_source(kernel_source, &CompilerOptions::default());
    assert!(result.is_ok());
}

#[test]
    fn test_compiler_caching() -> Result<(), Box<dyn std::error::Error>> {
    let compiler = CudaCompiler::new().unwrap();
    
    let kernel_source = r#"
        extern "C" __global__ void cached_kernel(int* data, int n) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                data[idx] = idx * idx;
            }
        }
    "#;
    
    // First compilation
    let start = std::time::Instant::now();
    let result1 = compiler.compile_source(kernel_source, &CompilerOptions::default());
    let first_duration = start.elapsed();
    assert!(result1.is_ok());
    
    // Second compilation (should be cached)
    let start = std::time::Instant::now();
    let result2 = compiler.compile_source(kernel_source, &CompilerOptions::default());
    let second_duration = start.elapsed();
    assert!(result2.is_ok());
    
    // Second should be faster
    assert!(second_duration <= first_duration);
    
    // Results should be identical
    assert_eq!(result1.unwrap(), result2.unwrap());
}

#[test]
    fn test_compilation_from_file() -> Result<(), Box<dyn std::error::Error>> {
    let compiler = CudaCompiler::new().unwrap();
    
    let kernel_file = "/tmp/test_kernel.cu";
    let kernel_source = r#"
        extern "C" __global__ void file_kernel(double* data, int n) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                data[idx] = sqrt(data[idx]);
            }
        }
    "#;
    
    std::fs::write(kernel_file, kernel_source).unwrap();
    
    let result = compiler.compile_file(kernel_file, &CompilerOptions::default());
    assert!(result.is_ok());
    
    std::fs::remove_file(kernel_file).unwrap();
}

#[test]
    fn test_linker_integration() -> Result<(), Box<dyn std::error::Error>> {
    let compiler = CudaCompiler::new().unwrap();
    
    let kernel_source1 = r#"
        extern "C" __device__ float math_helper(float x) {
            return x * x + 2.0f * x + 1.0f;
        }
    "#;
    
    let kernel_source2 = r#"
        extern "C" __device__ float math_helper(float x);
        
        extern "C" __global__ void linked_kernel(float* data, int n) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                data[idx] = math_helper(data[idx]);
            }
        }
    "#;
    
    let obj1 = compiler.compile_source(kernel_source1, &CompilerOptions::default()).unwrap();
    let obj2 = compiler.compile_source(kernel_source2, &CompilerOptions::default()).unwrap();
    
    let linked_result = compiler.link_objects(&[obj1, obj2]);
    assert!(linked_result.is_ok());
}

#[test]
    fn test_cross_compilation() -> Result<(), Box<dyn std::error::Error>> {
    let compiler = CudaCompiler::new().unwrap();
    
    let kernel_source = r#"
        extern "C" __global__ void cross_compile_kernel(long long* data, int n) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                data[idx] = (long long)idx * idx;
            }
        }
    "#;
    
    let targets = vec!["sm_35", "sm_60", "sm_75"];
    
    for target in targets {
        let options = CompilerOptions {
            target_arch: target.to_string(),
            ..CompilerOptions::default()
        };
        
        let result = compiler.compile_source(kernel_source, &options);
        assert!(result.is_ok());
    }
}

#[test]
    fn test_compiler_reset() -> Result<(), Box<dyn std::error::Error>> {
    let mut compiler = CudaCompiler::new().unwrap();
    
    // Compile something to populate caches
    let kernel_source = "extern \"C\" __global__ void reset_test() {}";
    let _ = compiler.compile_source(kernel_source, &CompilerOptions::default());
    
    // Reset compiler state
    compiler.reset();
    
    // Should still be able to compile
    let result = compiler.compile_source(kernel_source, &CompilerOptions::default());
    assert!(result.is_ok());
}

#[test]
    fn test_compiler_info() -> Result<(), Box<dyn std::error::Error>> {
    let compiler = CudaCompiler::new().unwrap();
    
    let info = compiler.get_compiler_info().unwrap();
    assert!(!info.version.is_empty());
    assert!(!info.supported_architectures.is_empty());
    assert!(info.max_threads_per_block > 0);
}

#[test]
    fn test_concurrent_compilation() -> Result<(), Box<dyn std::error::Error>> {
    use std::sync::Arc;
    use std::thread;
    
    let compiler = Arc::new(CudaCompiler::new().unwrap());
    let mut handles = Vec::new();
    
    for i in 0..4 {
        let compiler_clone = compiler.clone();
        let handle = thread::spawn(move || {
            let kernel_source = format!(r#"
                extern "C" __global__ void concurrent_kernel_{}(float* data, int n) {{
                    int idx = blockIdx.x * blockDim.x + threadIdx.x;
                    if (idx < n) {{
                        data[idx] = data[idx] + {};
                    }}
                }}
            "#, i, i);
            
            compiler_clone.compile_source(&kernel_source, &CompilerOptions::default())
        });
        handles.push(handle);
    }
    
    for handle in handles {
        let result = handle.join().map_err(|_| "Thread join error")?;
        assert!(result.is_ok());
    }
}