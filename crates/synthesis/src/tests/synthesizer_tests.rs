//! Tests for main synthesizer module

use crate::synthesizer::*;
use crate::error::SynthesisResult;

#[test]
    fn test_kernel_synthesizer_creation() -> Result<(), Box<dyn std::error::Error>> {
    let synthesizer = KernelSynthesizer::new().unwrap();
    assert!(synthesizer.is_initialized());
}

#[test]
    fn test_synthesizer_configuration() -> Result<(), Box<dyn std::error::Error>> {
    let config = SynthesizerConfig {
        default_optimization_level: OptimizationLevel::Aggressive,
        enable_auto_tuning: true,
        cache_compiled_kernels: true,
        max_generation_attempts: 5,
        target_architectures: vec!["sm_70".to_string(), "sm_80".to_string()],
        enable_fallback_strategies: true,
        profiling_enabled: true,
    };
    
    let synthesizer = KernelSynthesizer::with_config(config.clone()).unwrap();
    let retrieved_config = synthesizer.get_config();
    
    assert_eq!(retrieved_config.default_optimization_level, config.default_optimization_level);
    assert_eq!(retrieved_config.enable_auto_tuning, config.enable_auto_tuning);
    assert_eq!(retrieved_config.target_architectures, config.target_architectures);
}

#[test]
    fn test_simple_kernel_synthesis() -> Result<(), Box<dyn std::error::Error>> {
    let synthesizer = KernelSynthesizer::new().unwrap();
    
    let synthesis_spec = KernelSynthesisSpec {
        name: "vector_add".to_string(),
        goal: "Add two vectors element-wise".to_string(),
        input_types: vec![DataType::Float32, DataType::Float32],
        output_types: vec![DataType::Float32],
        input_shapes: vec![vec![1000], vec![1000]],
        output_shapes: vec![vec![1000]],
        algorithm_hints: vec![AlgorithmHint::ElementWise],
        performance_requirements: PerformanceRequirements {
            max_execution_time_ms: 10.0,
            min_throughput_gflops: 100.0,
            max_memory_usage_mb: 64.0,
            target_occupancy: 0.75,
        },
        constraints: KernelConstraints::default(),
    };
    
    let result = synthesizer.synthesize_kernel(&synthesis_spec).unwrap();
    
    assert!(result.success);
    assert!(!result.kernel_code.is_empty());
    assert!(!result.host_code.is_empty());
    assert_eq!(result.kernel_name, "vector_add");
    assert!(result.launch_parameters.grid_dim.0 > 0);
    assert!(result.launch_parameters.block_dim.0 > 0);
}

#[test]
    fn test_matrix_multiplication_synthesis() -> Result<(), Box<dyn std::error::Error>> {
    let synthesizer = KernelSynthesizer::new().unwrap();
    
    let synthesis_spec = KernelSynthesisSpec {
        name: "matrix_multiply".to_string(),
        goal: "Multiply two matrices A (512x512) and B (512x512)".to_string(),
        input_types: vec![DataType::Float32, DataType::Float32],
        output_types: vec![DataType::Float32],
        input_shapes: vec![vec![512, 512], vec![512, 512]],
        output_shapes: vec![vec![512, 512]],
        algorithm_hints: vec![AlgorithmHint::MatrixOperation, AlgorithmHint::UseSharedMemory],
        performance_requirements: PerformanceRequirements {
            max_execution_time_ms: 50.0,
            min_throughput_gflops: 500.0,
            max_memory_usage_mb: 256.0,
            target_occupancy: 0.80,
        },
        constraints: KernelConstraints {
            max_shared_memory_bytes: 48 * 1024,
            max_registers_per_thread: 64,
            ..KernelConstraints::default()
        },
    };
    
    let result = synthesizer.synthesize_kernel(&synthesis_spec).unwrap();
    
    assert!(result.success);
    assert!(result.kernel_code.contains("matrix") || result.kernel_code.contains("mul"));
    assert!(result.kernel_code.contains("__shared__")); // Should use shared memory
    assert!(result.estimated_performance.throughput_gflops >= synthesis_spec.performance_requirements.min_throughput_gflops * 0.8);
}

#[test]
    fn test_reduction_synthesis() -> Result<(), Box<dyn std::error::Error>> {
    let synthesizer = KernelSynthesizer::new().unwrap();
    
    let synthesis_spec = KernelSynthesisSpec {
        name: "array_sum".to_string(),
        goal: "Sum all elements in an array of 1 million floats".to_string(),
        input_types: vec![DataType::Float32],
        output_types: vec![DataType::Float32],
        input_shapes: vec![vec![1000000]],
        output_shapes: vec![vec![1]],
        algorithm_hints: vec![AlgorithmHint::Reduction, AlgorithmHint::UseSharedMemory],
        performance_requirements: PerformanceRequirements {
            max_execution_time_ms: 5.0,
            min_throughput_gflops: 50.0,
            max_memory_usage_mb: 32.0,
            target_occupancy: 0.70,
        },
        constraints: KernelConstraints::default(),
    };
    
    let result = synthesizer.synthesize_kernel(&synthesis_spec).unwrap();
    
    assert!(result.success);
    assert!(result.kernel_code.contains("sum") || result.kernel_code.contains("reduce"));
    assert!(result.kernel_code.contains("__shared__"));
    assert!(result.kernel_code.contains("__syncthreads"));
}

#[test]
    fn test_convolution_synthesis() -> Result<(), Box<dyn std::error::Error>> {
    let synthesizer = KernelSynthesizer::new().unwrap();
    
    let synthesis_spec = KernelSynthesisSpec {
        name: "conv2d".to_string(),
        goal: "Apply 3x3 convolution to 256x256 image".to_string(),
        input_types: vec![DataType::Float32, DataType::Float32], // image and filter
        output_types: vec![DataType::Float32],
        input_shapes: vec![vec![256, 256], vec![3, 3]],
        output_shapes: vec![vec![254, 254]], // Valid convolution output
        algorithm_hints: vec![AlgorithmHint::Convolution, AlgorithmHint::UseTiling],
        performance_requirements: PerformanceRequirements {
            max_execution_time_ms: 20.0,
            min_throughput_gflops: 200.0,
            max_memory_usage_mb: 128.0,
            target_occupancy: 0.75,
        },
        constraints: KernelConstraints::default(),
    };
    
    let result = synthesizer.synthesize_kernel(&synthesis_spec).unwrap();
    
    assert!(result.success);
    assert!(result.kernel_code.contains("conv") || result.kernel_code.contains("filter"));
    
    // Should use tiling for efficient memory access
    assert!(result.kernel_code.contains("tile") || result.kernel_code.contains("__shared__"));
}

#[test]
    fn test_sorting_synthesis() -> Result<(), Box<dyn std::error::Error>> {
    let synthesizer = KernelSynthesizer::new().unwrap();
    
    let synthesis_spec = KernelSynthesisSpec {
        name: "radix_sort".to_string(),
        goal: "Sort array of 100000 integers using radix sort".to_string(),
        input_types: vec![DataType::UInt32],
        output_types: vec![DataType::UInt32],
        input_shapes: vec![vec![100000]],
        output_shapes: vec![vec![100000]],
        algorithm_hints: vec![AlgorithmHint::Sorting, AlgorithmHint::UseSharedMemory],
        performance_requirements: PerformanceRequirements {
            max_execution_time_ms: 15.0,
            min_throughput_gflops: 80.0,
            max_memory_usage_mb: 96.0,
            target_occupancy: 0.80,
        },
        constraints: KernelConstraints::default(),
    };
    
    let result = synthesizer.synthesize_kernel(&synthesis_spec).unwrap();
    
    assert!(result.success);
    assert!(result.kernel_code.contains("sort") || result.kernel_code.contains("radix"));
    
    // Radix sort typically uses multiple passes
    assert!(result.host_code.contains("for") || result.host_code.contains("pass"));
}

#[test]
    fn test_auto_tuning() -> Result<(), Box<dyn std::error::Error>> {
    let config = SynthesizerConfig {
        enable_auto_tuning: true,
        ..SynthesizerConfig::default()
    };
    let synthesizer = KernelSynthesizer::with_config(config)?;
    
    let synthesis_spec = KernelSynthesisSpec {
        name: "auto_tuned_kernel".to_string(),
        goal: "Element-wise operation with auto-tuning".to_string(),
        input_types: vec![DataType::Float32],
        output_types: vec![DataType::Float32],
        input_shapes: vec![vec![50000]],
        output_shapes: vec![vec![50000]],
        algorithm_hints: vec![AlgorithmHint::ElementWise],
        performance_requirements: PerformanceRequirements {
            max_execution_time_ms: 5.0,
            min_throughput_gflops: 100.0,
            max_memory_usage_mb: 32.0,
            target_occupancy: 0.85,
        },
        constraints: KernelConstraints::default(),
    };
    
    let result = synthesizer.synthesize_kernel(&synthesis_spec).unwrap();
    
    assert!(result.success);
    assert!(result.auto_tuning_results.is_some());
    
    let tuning_results = result.auto_tuning_results.unwrap();
    assert!(!tuning_results.tested_configurations.is_empty());
    assert!(tuning_results.best_configuration.block_size > 0);
    assert!(tuning_results.performance_improvement_factor >= 1.0);
}

#[test]
    fn test_multiple_data_types() -> Result<(), Box<dyn std::error::Error>> {
    let synthesizer = KernelSynthesizer::new().unwrap();
    
    let data_types = vec![
        (DataType::Float32, "float"),
        (DataType::Float64, "double"),
        (DataType::Int32, "int"),
        (DataType::UInt32, "unsigned int"),
    ];
    
    for (data_type, expected_in_code) in data_types {
        let synthesis_spec = KernelSynthesisSpec {
            name: format!("type_test_{:?}", data_type),
            goal: "Test different data types".to_string(),
            input_types: vec![data_type],
            output_types: vec![data_type],
            input_shapes: vec![vec![1000]],
            output_shapes: vec![vec![1000]],
            algorithm_hints: vec![AlgorithmHint::ElementWise],
            performance_requirements: PerformanceRequirements::default(),
            constraints: KernelConstraints::default(),
        };
        
        let result = synthesizer.synthesize_kernel(&synthesis_spec).unwrap();
        
        assert!(result.success);
        assert!(result.kernel_code.contains(expected_in_code));
    }
}

#[test]
    fn test_multi_dimensional_arrays() -> Result<(), Box<dyn std::error::Error>> {
    let synthesizer = KernelSynthesizer::new().unwrap();
    
    let synthesis_spec = KernelSynthesisSpec {
        name: "tensor_operation".to_string(),
        goal: "Process 3D tensor (64x64x32)".to_string(),
        input_types: vec![DataType::Float32],
        output_types: vec![DataType::Float32],
        input_shapes: vec![vec![64, 64, 32]],
        output_shapes: vec![vec![64, 64, 32]],
        algorithm_hints: vec![AlgorithmHint::ElementWise],
        performance_requirements: PerformanceRequirements::default(),
        constraints: KernelConstraints::default(),
    };
    
    let result = synthesizer.synthesize_kernel(&synthesis_spec).unwrap();
    
    assert!(result.success);
    
    // Should handle 3D indexing
    let has_3d_indexing = result.kernel_code.contains("blockIdx.x") && 
                         result.kernel_code.contains("blockIdx.y") &&
                         (result.kernel_code.contains("blockIdx.z") || result.kernel_code.contains("z"));
    assert!(has_3d_indexing);
}

#[test]
    fn test_performance_constraints() -> Result<(), Box<dyn std::error::Error>> {
    let synthesizer = KernelSynthesizer::new().unwrap();
    
    let synthesis_spec = KernelSynthesisSpec {
        name: "constrained_kernel".to_string(),
        goal: "Kernel with strict performance constraints".to_string(),
        input_types: vec![DataType::Float32],
        output_types: vec![DataType::Float32],
        input_shapes: vec![vec![10000]],
        output_shapes: vec![vec![10000]],
        algorithm_hints: vec![AlgorithmHint::ElementWise],
        performance_requirements: PerformanceRequirements {
            max_execution_time_ms: 1.0, // Very strict
            min_throughput_gflops: 500.0,
            max_memory_usage_mb: 16.0,
            target_occupancy: 0.90,
        },
        constraints: KernelConstraints {
            max_shared_memory_bytes: 1024, // Very limited
            max_registers_per_thread: 16,
            ..KernelConstraints::default()
        },
    };
    
    let result = synthesizer.synthesize_kernel(&synthesis_spec);
    
    // Should either succeed with optimized kernel or fail gracefully with explanation
    match result {
        Ok(r) => {
            assert!(r.success);
            assert!(r.estimated_performance.execution_time_ms <= synthesis_spec.performance_requirements.max_execution_time_ms * 1.1);
        }
        Err(_) => {
            // Acceptable if constraints are too strict
        }
    }
}

#[test]
    fn test_fallback_strategies() -> Result<(), Box<dyn std::error::Error>> {
    let config = SynthesizerConfig {
        enable_fallback_strategies: true,
        max_generation_attempts: 3,
        ..SynthesizerConfig::default()
    };
    let synthesizer = KernelSynthesizer::with_config(config)?;
    
    let challenging_spec = KernelSynthesisSpec {
        name: "challenging_kernel".to_string(),
        goal: "Extremely challenging synthesis request".to_string(),
        input_types: vec![DataType::Float64],
        output_types: vec![DataType::Float64],
        input_shapes: vec![vec![1000000]],
        output_shapes: vec![vec![1000000]],
        algorithm_hints: vec![AlgorithmHint::Custom("impossible_algorithm".to_string())],
        performance_requirements: PerformanceRequirements {
            max_execution_time_ms: 0.001, // Impossible constraint
            min_throughput_gflops: 10000.0,
            max_memory_usage_mb: 1.0,
            target_occupancy: 1.0,
        },
        constraints: KernelConstraints::default(),
    };
    
    let result = synthesizer.synthesize_kernel(&challenging_spec).unwrap();
    
    // Should provide fallback solution even if optimal solution impossible
    if !result.success {
        assert!(!result.fallback_solutions.is_empty());
        assert!(!result.fallback_explanation.is_empty());
    }
}

#[test]
    fn test_template_based_synthesis() -> Result<(), Box<dyn std::error::Error>> {
    let synthesizer = KernelSynthesizer::new().unwrap();
    
    let synthesis_spec = KernelSynthesisSpec {
        name: "template_kernel".to_string(),
        goal: "Use predefined template for matrix multiplication".to_string(),
        input_types: vec![DataType::Float32, DataType::Float32],
        output_types: vec![DataType::Float32],
        input_shapes: vec![vec![128, 128], vec![128, 128]],
        output_shapes: vec![vec![128, 128]],
        algorithm_hints: vec![
            AlgorithmHint::MatrixOperation, 
            AlgorithmHint::UseTemplate("optimized_gemm".to_string())
        ],
        performance_requirements: PerformanceRequirements::default(),
        constraints: KernelConstraints::default(),
    };
    
    let result = synthesizer.synthesize_kernel(&synthesis_spec).unwrap();
    
    assert!(result.success);
    assert!(result.template_info.is_some());
    
    let template_info = result.template_info.unwrap();
    assert_eq!(template_info.template_name, "optimized_gemm");
    assert!(!template_info.instantiation_parameters.is_empty());
}

#[test]
    fn test_batch_synthesis() -> Result<(), Box<dyn std::error::Error>> {
    let synthesizer = KernelSynthesizer::new().unwrap();
    
    let batch_specs = (0..3).map(|i| KernelSynthesisSpec {
        name: format!("batch_kernel_{}", i),
        goal: format!("Batch synthesis test {}", i),
        input_types: vec![DataType::Float32],
        output_types: vec![DataType::Float32],
        input_shapes: vec![vec![1000 * (i + 1)]],
        output_shapes: vec![vec![1000 * (i + 1)]],
        algorithm_hints: vec![AlgorithmHint::ElementWise],
        performance_requirements: PerformanceRequirements::default(),
        constraints: KernelConstraints::default(),
    }).collect();
    
    let results = synthesizer.synthesize_batch(batch_specs).unwrap();
    
    assert_eq!(results.len(), 3);
    for result in results {
        assert!(result.success);
        assert!(!result.kernel_code.is_empty());
    }
}

#[test]
    fn test_code_generation_strategies() -> Result<(), Box<dyn std::error::Error>> {
    let synthesizer = KernelSynthesizer::new().unwrap();
    
    let generation_strategies = vec![
        CodeGenerationStrategy::TemplateInstantiation,
        CodeGenerationStrategy::PatternMatching,
        CodeGenerationStrategy::OptimizationGuided,
        CodeGenerationStrategy::HybridApproach,
    ];
    
    for strategy in generation_strategies {
        let synthesis_spec = KernelSynthesisSpec {
            name: format!("strategy_test_{:?}", strategy),
            goal: "Test different code generation strategies".to_string(),
            input_types: vec![DataType::Float32],
            output_types: vec![DataType::Float32],
            input_shapes: vec![vec![1000]],
            output_shapes: vec![vec![1000]],
            algorithm_hints: vec![AlgorithmHint::ElementWise],
            performance_requirements: PerformanceRequirements::default(),
            constraints: KernelConstraints::default(),
        };
        
        let result = synthesizer.synthesize_with_strategy(&synthesis_spec, strategy).unwrap();
        
        assert!(result.success);
        assert_eq!(result.generation_strategy, strategy);
        assert!(!result.kernel_code.is_empty());
    }
}

#[test]
    fn test_synthesis_caching() -> Result<(), Box<dyn std::error::Error>> {
    let config = SynthesizerConfig {
        cache_compiled_kernels: true,
        ..SynthesizerConfig::default()
    };
    let synthesizer = KernelSynthesizer::with_config(config)?;
    
    let synthesis_spec = KernelSynthesisSpec {
        name: "cache_test".to_string(),
        goal: "Test synthesis caching".to_string(),
        input_types: vec![DataType::Float32],
        output_types: vec![DataType::Float32],
        input_shapes: vec![vec![1000]],
        output_shapes: vec![vec![1000]],
        algorithm_hints: vec![AlgorithmHint::ElementWise],
        performance_requirements: PerformanceRequirements::default(),
        constraints: KernelConstraints::default(),
    };
    
    // First synthesis
    let start_1 = std::time::Instant::now();
    let result_1 = synthesizer.synthesize_kernel(&synthesis_spec).unwrap();
    let duration_1 = start_1.elapsed();
    
    // Second synthesis (should use cache)
    let start_2 = std::time::Instant::now();
    let result_2 = synthesizer.synthesize_kernel(&synthesis_spec).unwrap();
    let duration_2 = start_2.elapsed();
    
    assert!(result_1.success && result_2.success);
    assert_eq!(result_1.kernel_code, result_2.kernel_code);
    
    // Second synthesis should be faster due to caching
    assert!(duration_2 <= duration_1);
}

#[test]
    fn test_profiling_integration() -> Result<(), Box<dyn std::error::Error>> {
    let config = SynthesizerConfig {
        profiling_enabled: true,
        ..SynthesizerConfig::default()
    };
    let synthesizer = KernelSynthesizer::with_config(config)?;
    
    let synthesis_spec = KernelSynthesisSpec {
        name: "profiling_test".to_string(),
        goal: "Test profiling integration".to_string(),
        input_types: vec![DataType::Float32],
        output_types: vec![DataType::Float32],
        input_shapes: vec![vec![10000]],
        output_shapes: vec![vec![10000]],
        algorithm_hints: vec![AlgorithmHint::ElementWise],
        performance_requirements: PerformanceRequirements::default(),
        constraints: KernelConstraints::default(),
    };
    
    let result = synthesizer.synthesize_kernel(&synthesis_spec).unwrap();
    
    assert!(result.success);
    assert!(result.profiling_data.is_some());
    
    let profiling = result.profiling_data.unwrap();
    assert!(profiling.synthesis_time.as_millis() > 0);
    assert!(profiling.compilation_time.as_millis() > 0);
    assert!(!profiling.stage_breakdown.is_empty());
}

#[test]
    fn test_error_recovery() -> Result<(), Box<dyn std::error::Error>> {
    let synthesizer = KernelSynthesizer::new().unwrap();
    
    let invalid_spec = KernelSynthesisSpec {
        name: "".to_string(), // Invalid empty name
        goal: "".to_string(), // Invalid empty goal
        input_types: vec![],  // Invalid empty inputs
        output_types: vec![], // Invalid empty outputs
        input_shapes: vec![],
        output_shapes: vec![],
        algorithm_hints: vec![],
        performance_requirements: PerformanceRequirements::default(),
        constraints: KernelConstraints::default(),
    };
    
    let result = synthesizer.synthesize_kernel(&invalid_spec);
    
    // Should fail gracefully with meaningful error
    assert!(result.is_err());
    
    // Synthesizer should still be functional after error
    let valid_spec = KernelSynthesisSpec {
        name: "recovery_test".to_string(),
        goal: "Test error recovery".to_string(),
        input_types: vec![DataType::Float32],
        output_types: vec![DataType::Float32],
        input_shapes: vec![vec![100]],
        output_shapes: vec![vec![100]],
        algorithm_hints: vec![AlgorithmHint::ElementWise],
        performance_requirements: PerformanceRequirements::default(),
        constraints: KernelConstraints::default(),
    };
    
    let recovery_result = synthesizer.synthesize_kernel(&valid_spec).unwrap();
    assert!(recovery_result.success);
}

#[test]
    fn test_synthesis_statistics() -> Result<(), Box<dyn std::error::Error>> {
    let synthesizer = KernelSynthesizer::new().unwrap();
    
    // Perform multiple synthesis operations
    for i in 0..5 {
        let synthesis_spec = KernelSynthesisSpec {
            name: format!("stats_test_{}", i),
            goal: "Collect statistics".to_string(),
            input_types: vec![DataType::Float32],
            output_types: vec![DataType::Float32],
            input_shapes: vec![vec![100]],
            output_shapes: vec![vec![100]],
            algorithm_hints: vec![AlgorithmHint::ElementWise],
            performance_requirements: PerformanceRequirements::default(),
            constraints: KernelConstraints::default(),
        };
        
        let _result = synthesizer.synthesize_kernel(&synthesis_spec);
    }
    
    let stats = synthesizer.get_synthesis_statistics().unwrap();
    
    assert!(stats.total_syntheses >= 5);
    assert!(stats.successful_syntheses <= stats.total_syntheses);
    assert!(stats.average_synthesis_time.as_millis() > 0);
    assert!(stats.cache_hit_rate >= 0.0 && stats.cache_hit_rate <= 1.0);
    assert!(!stats.most_common_algorithms.is_empty());
}

#[test]
    fn test_concurrent_synthesis() -> Result<(), Box<dyn std::error::Error>> {
    use std::sync::Arc;
    use std::thread;
    
    let synthesizer = Arc::new(KernelSynthesizer::new().unwrap());
    let mut handles = Vec::new();
    
    for i in 0..4 {
        let synthesizer_clone = synthesizer.clone();
        let handle = thread::spawn(move || {
            let synthesis_spec = KernelSynthesisSpec {
                name: format!("concurrent_test_{}", i),
                goal: "Test concurrent synthesis".to_string(),
                input_types: vec![DataType::Float32],
                output_types: vec![DataType::Float32],
                input_shapes: vec![vec![1000]],
                output_shapes: vec![vec![1000]],
                algorithm_hints: vec![AlgorithmHint::ElementWise],
                performance_requirements: PerformanceRequirements::default(),
                constraints: KernelConstraints::default(),
            };
            
            synthesizer_clone.synthesize_kernel(&synthesis_spec)
        });
        handles.push(handle);
    }
    
    for handle in handles {
        let result = handle.join().map_err(|_| "Thread join error")?.ok();
        assert!(result.success);
    }
}