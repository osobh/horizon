//! Tests for synthesis improvement and optimization module

use crate::improvement::*;
use crate::error::SynthesisResult;

#[test]
    fn test_improvement_engine_creation() -> Result<(), Box<dyn std::error::Error>> {
    let engine = ImprovementEngine::new().unwrap();
    assert!(engine.is_initialized());
}

#[test]
    fn test_improvement_strategy_registration() -> Result<(), Box<dyn std::error::Error>> {
    let mut engine = ImprovementEngine::new().unwrap();
    
    let strategies = vec![
        ImprovementStrategy::MemoryCoalescing,
        ImprovementStrategy::LoopUnrolling,
        ImprovementStrategy::SharedMemoryOptimization,
        ImprovementStrategy::RegisterOptimization,
        ImprovementStrategy::OccupancyMaximization,
    ];
    
    for strategy in strategies {
        engine.register_strategy(strategy).unwrap();
    }
    
    let registered = engine.get_registered_strategies();
    assert_eq!(registered.len(), 5);
}

#[test]
    fn test_performance_analysis() -> Result<(), Box<dyn std::error::Error>> {
    let engine = ImprovementEngine::new().unwrap();
    
    let kernel_code = r#"
        __global__ void simple_kernel(float* data, int n) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                data[idx] = data[idx] * 2.0f;
            }
        }
    "#;
    
    let performance_info = PerformanceInfo {
        execution_time_ms: 5.2,
        memory_bandwidth_gbps: 450.0,
        compute_utilization: 0.65,
        memory_utilization: 0.80,
        occupancy: 0.75,
        register_usage: 32,
        shared_memory_usage: 0,
        warp_efficiency: 0.90,
    };
    
    let analysis = engine.analyze_performance(kernel_code, &performance_info).unwrap();
    
    assert!(!analysis.bottlenecks.is_empty());
    assert!(!analysis.improvement_opportunities.is_empty());
    assert!(analysis.overall_score >= 0.0 && analysis.overall_score <= 1.0);
}

#[test]
    fn test_memory_coalescing_improvement() -> Result<(), Box<dyn std::error::Error>> {
    let engine = ImprovementEngine::new().unwrap();
    
    let uncoalesced_kernel = r#"
        __global__ void uncoalesced_transpose(float* input, float* output, int width, int height) {
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;
            
            if (x < width && y < height) {
                // Uncoalesced memory access pattern
                output[x * height + y] = input[y * width + x];
            }
        }
    "#;
    
    let improvement_request = ImprovementRequest {
        original_code: uncoalesced_kernel.to_string(),
        target_metrics: TargetMetrics {
            min_memory_bandwidth: 600.0,
            min_compute_utilization: 0.80,
            max_execution_time_ms: 3.0,
            min_occupancy: 0.80,
        },
        constraints: ImprovementConstraints {
            preserve_functionality: true,
            max_shared_memory_bytes: 48 * 1024,
            max_registers_per_thread: 64,
            target_architecture: "sm_80".to_string(),
        },
    };
    
    let result = engine.improve_kernel(&improvement_request).unwrap();
    
    assert!(result.success);
    assert!(!result.improved_code.is_empty());
    assert!(!result.applied_optimizations.is_empty());
    assert!(result.performance_gain > 0.0);
    
    // Should contain shared memory optimization
    assert!(result.improved_code.contains("__shared__"));
}

#[test]
    fn test_loop_unrolling_improvement() -> Result<(), Box<dyn std::error::Error>> {
    let engine = ImprovementEngine::new().unwrap();
    
    let loop_kernel = r#"
        __global__ void loop_kernel(float* data, int n) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                float sum = 0.0f;
                for (int i = 0; i < 8; i++) {
                    sum += data[idx + i * n];
                }
                data[idx] = sum;
            }
        }
    "#;
    
    let improvement_request = ImprovementRequest {
        original_code: loop_kernel.to_string(),
        target_metrics: TargetMetrics {
            min_compute_utilization: 0.90,
            ..TargetMetrics::default()
        },
        constraints: ImprovementConstraints::default(),
    };
    
    let result = engine.improve_kernel(&improvement_request).unwrap();
    
    assert!(result.success);
    
    // Should apply loop unrolling
    let unrolling_applied = result.applied_optimizations.iter()
        .any(|opt| matches!(opt, AppliedOptimization::LoopUnrolling { .. }));
    assert!(unrolling_applied);
}

#[test]
    fn test_occupancy_optimization() -> Result<(), Box<dyn std::error::Error>> {
    let engine = ImprovementEngine::new().unwrap();
    
    let low_occupancy_kernel = r#"
        __global__ void low_occupancy_kernel(double* data, int n) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                double local_array[1024]; // Large local memory usage
                for (int i = 0; i < 1024; i++) {
                    local_array[i] = data[idx] + i;
                }
                data[idx] = local_array[idx % 1024];
            }
        }
    "#;
    
    let improvement_request = ImprovementRequest {
        original_code: low_occupancy_kernel.to_string(),
        target_metrics: TargetMetrics {
            min_occupancy: 0.75,
            ..TargetMetrics::default()
        },
        constraints: ImprovementConstraints::default(),
    };
    
    let result = engine.improve_kernel(&improvement_request).unwrap();
    
    if result.success {
        // Should suggest register reduction or shared memory usage
        let has_memory_optimization = result.applied_optimizations.iter()
            .any(|opt| matches!(opt, 
                AppliedOptimization::SharedMemoryOptimization { .. } |
                AppliedOptimization::RegisterOptimization { .. }
            ));
        assert!(has_memory_optimization);
    }
}

#[test]
    fn test_warp_efficiency_improvement() -> Result<(), Box<dyn std::error::Error>> {
    let engine = ImprovementEngine::new().unwrap();
    
    let divergent_kernel = r#"
        __global__ void divergent_kernel(int* data, int n) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                if (data[idx] % 2 == 0) {
                    data[idx] = data[idx] * 2 + 5;
                } else {
                    data[idx] = data[idx] * 3 + 7;
                }
            }
        }
    "#;
    
    let improvement_request = ImprovementRequest {
        original_code: divergent_kernel.to_string(),
        target_metrics: TargetMetrics {
            min_warp_efficiency: 0.95,
            ..TargetMetrics::default()
        },
        constraints: ImprovementConstraints::default(),
    };
    
    let result = engine.improve_kernel(&improvement_request).unwrap();
    
    if result.success {
        // Should suggest branch reduction techniques
        let has_branch_optimization = result.applied_optimizations.iter()
            .any(|opt| matches!(opt, AppliedOptimization::BranchOptimization { .. }));
        assert!(has_branch_optimization);
    }
}

#[test]
    fn test_automatic_improvement_pipeline() -> Result<(), Box<dyn std::error::Error>> {
    let mut engine = ImprovementEngine::new().unwrap();
    
    // Register multiple improvement strategies
    engine.register_strategy(ImprovementStrategy::MemoryCoalescing).unwrap();
    engine.register_strategy(ImprovementStrategy::LoopUnrolling)?;
    engine.register_strategy(ImprovementStrategy::OccupancyMaximization)?;
    
    let original_kernel = r#"
        __global__ void multi_issue_kernel(float* matrix, int width, int height) {
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;
            
            if (x < width && y < height) {
                float sum = 0.0f;
                for (int k = 0; k < 4; k++) {
                    // Non-coalesced access + small loop
                    sum += matrix[k * width * height + y * width + x];
                }
                matrix[y * width + x] = sum;
            }
        }
    "#;
    
    let pipeline_config = ImprovementPipelineConfig {
        max_iterations: 5,
        convergence_threshold: 0.01,
        enable_aggressive_optimizations: true,
        preserve_numerical_accuracy: true,
    };
    
    let improvement_request = ImprovementRequest {
        original_code: original_kernel.to_string(),
        target_metrics: TargetMetrics {
            min_memory_bandwidth: 700.0,
            min_compute_utilization: 0.85,
            min_occupancy: 0.80,
            min_warp_efficiency: 0.90,
            max_execution_time_ms: 2.0,
        },
        constraints: ImprovementConstraints::default(),
    };
    
    let result = engine.run_improvement_pipeline(&improvement_request, &pipeline_config).unwrap();
    
    assert!(result.success);
    assert!(!result.applied_optimizations.is_empty());
    assert!(result.iterations_performed > 0);
    assert!(result.performance_gain >= 0.0);
}

#[test]
    fn test_constraint_validation() -> Result<(), Box<dyn std::error::Error>> {
    let engine = ImprovementEngine::new().unwrap();
    
    let kernel_code = r#"
        __global__ void constraint_test(float* data, int n) {
            __shared__ float sdata[1024];
            int idx = threadIdx.x;
            
            sdata[idx] = data[idx];
            __syncthreads();
            
            data[idx] = sdata[idx] * 2.0f;
        }
    "#;
    
    let constraints = ImprovementConstraints {
        preserve_functionality: true,
        max_shared_memory_bytes: 2048, // 2KB limit
        max_registers_per_thread: 32,
        target_architecture: "sm_70".to_string(),
    };
    
    let validation_result = engine.validate_constraints(kernel_code, &constraints).unwrap();
    
    assert!(validation_result.is_valid);
    assert!(validation_result.shared_memory_usage <= constraints.max_shared_memory_bytes);
    assert!(validation_result.register_usage <= constraints.max_registers_per_thread);
}

#[test]
    fn test_performance_prediction() -> Result<(), Box<dyn std::error::Error>> {
    let engine = ImprovementEngine::new().unwrap();
    
    let kernel_code = r#"
        __global__ void prediction_kernel(float* a, float* b, float* c, int n) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                c[idx] = a[idx] + b[idx];
            }
        }
    "#;
    
    let device_info = DeviceInfo {
        compute_capability: (8, 0),
        max_threads_per_block: 1024,
        shared_memory_per_block: 48 * 1024,
        max_registers_per_block: 65536,
        memory_bandwidth_gbps: 900.0,
        peak_flops: 19500.0e9,
        warp_size: 32,
    };
    
    let predicted_performance = engine.predict_performance(kernel_code, &device_info).unwrap();
    
    assert!(predicted_performance.estimated_execution_time_ms > 0.0);
    assert!(predicted_performance.estimated_memory_bandwidth > 0.0);
    assert!(predicted_performance.estimated_occupancy >= 0.0 && predicted_performance.estimated_occupancy <= 1.0);
    assert!(predicted_performance.confidence_score >= 0.0 && predicted_performance.confidence_score <= 1.0);
}

#[test]
    fn test_optimization_history() -> Result<(), Box<dyn std::error::Error>> {
    let mut engine = ImprovementEngine::new().unwrap();
    
    let kernel_code = "simple kernel code";
    let improvement_request = ImprovementRequest {
        original_code: kernel_code.to_string(),
        target_metrics: TargetMetrics::default(),
        constraints: ImprovementConstraints::default(),
    };
    
    // Perform multiple improvements
    for i in 0..3 {
        let mut request = improvement_request.clone();
        request.original_code = format!("{} iteration {}", kernel_code, i);
        let _ = engine.improve_kernel(&request);
    }
    
    let history = engine.get_optimization_history().unwrap();
    assert_eq!(history.len(), 3);
    
    for entry in history {
        assert!(!entry.timestamp.is_zero());
        assert!(!entry.original_code.is_empty());
    }
}

#[test]
    fn test_benchmark_comparison() -> Result<(), Box<dyn std::error::Error>> {
    let engine = ImprovementEngine::new().unwrap();
    
    let original_kernel = r#"
        __global__ void original_kernel(float* data, int n) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                data[idx] = data[idx] * 2.0f;
            }
        }
    "#;
    
    let optimized_kernel = r#"
        __global__ void optimized_kernel(float* data, int n) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                float4 val = reinterpret_cast<float4*>(data)[idx/4];
                val.x *= 2.0f; val.y *= 2.0f; val.z *= 2.0f; val.w *= 2.0f;
                reinterpret_cast<float4*>(data)[idx/4] = val;
            }
        }
    "#;
    
    let benchmark_config = BenchmarkConfig {
        input_sizes: vec![1024, 10240, 102400],
        iterations_per_size: 10,
        warmup_iterations: 3,
        measure_memory_transfers: true,
        measure_kernel_execution: true,
    };
    
    let comparison = engine.benchmark_kernels(
        original_kernel, 
        optimized_kernel, 
        &benchmark_config
    ).unwrap();
    
    assert!(!comparison.results.is_empty());
    assert!(comparison.overall_speedup >= 0.0);
    
    for result in comparison.results {
        assert!(result.original_time_ms > 0.0);
        assert!(result.optimized_time_ms > 0.0);
        assert!(result.input_size > 0);
    }
}

#[test]
    fn test_adaptive_optimization() -> Result<(), Box<dyn std::error::Error>> {
    let mut engine = ImprovementEngine::new().unwrap();
    
    // Enable adaptive optimization
    engine.enable_adaptive_optimization(true);
    
    let kernel_code = r#"
        __global__ void adaptive_kernel(double* data, int n) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                data[idx] = sqrt(data[idx] * data[idx] + 1.0);
            }
        }
    "#;
    
    let runtime_feedback = RuntimeFeedback {
        actual_execution_time_ms: 8.5,
        actual_memory_bandwidth_gbps: 420.0,
        actual_occupancy: 0.60,
        bottleneck_analysis: BottleneckType::MemoryBound,
        device_utilization: 0.70,
    };
    
    let adaptive_request = AdaptiveImprovementRequest {
        code: kernel_code.to_string(),
        runtime_feedback,
        previous_optimizations: vec![
            AppliedOptimization::LoopUnrolling { factor: 2 },
        ],
    };
    
    let result = engine.adaptive_improve(&adaptive_request).unwrap();
    
    if result.success {
        // Should suggest memory-focused optimizations for memory-bound kernel
        let has_memory_opt = result.applied_optimizations.iter()
            .any(|opt| matches!(opt, 
                AppliedOptimization::MemoryCoalescing { .. } |
                AppliedOptimization::SharedMemoryOptimization { .. }
            ));
        assert!(has_memory_opt);
    }
}

#[test]
    fn test_multi_objective_optimization() -> Result<(), Box<dyn std::error::Error>> {
    let engine = ImprovementEngine::new().unwrap();
    
    let kernel_code = r#"
        __global__ void multi_objective_kernel(float* input, float* output, int n) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                float result = 0.0f;
                for (int i = 0; i < 16; i++) {
                    result += input[idx + i * n] * (i + 1);
                }
                output[idx] = result;
            }
        }
    "#;
    
    let objectives = MultiObjectiveConfig {
        primary_objective: OptimizationObjective::Performance,
        secondary_objectives: vec![
            (OptimizationObjective::EnergyEfficiency, 0.3),
            (OptimizationObjective::CodeSize, 0.2),
        ],
        constraint_weights: vec![
            (ConstraintType::SharedMemory, 0.8),
            (ConstraintType::RegisterUsage, 0.6),
        ],
    };
    
    let improvement_request = ImprovementRequest {
        original_code: kernel_code.to_string(),
        target_metrics: TargetMetrics::default(),
        constraints: ImprovementConstraints::default(),
    };
    
    let result = engine.multi_objective_optimize(&improvement_request, &objectives).unwrap();
    
    if result.success {
        assert!(!result.pareto_solutions.is_empty());
        assert!(result.recommended_solution_index < result.pareto_solutions.len());
        
        for solution in &result.pareto_solutions {
            assert!(solution.performance_score >= 0.0);
            assert!(solution.energy_score >= 0.0);
            assert!(solution.size_score >= 0.0);
        }
    }
}

#[test]
    fn test_optimization_rollback() -> Result<(), Box<dyn std::error::Error>> {
    let mut engine = ImprovementEngine::new().unwrap();
    
    let original_code = "original kernel code";
    let improvement_request = ImprovementRequest {
        original_code: original_code.to_string(),
        target_metrics: TargetMetrics::default(),
        constraints: ImprovementConstraints::default(),
    };
    
    // Perform optimization
    let result = engine.improve_kernel(&improvement_request).unwrap();
    let optimization_id = result.optimization_id;
    
    // Rollback optimization
    let rollback_result = engine.rollback_optimization(&optimization_id).unwrap();
    assert!(rollback_result.success);
    assert_eq!(rollback_result.restored_code, original_code);
}

#[test]
    fn test_custom_optimization_strategies() -> Result<(), Box<dyn std::error::Error>> {
    let mut engine = ImprovementEngine::new().unwrap();
    
    let custom_strategy = CustomOptimizationStrategy {
        name: "CustomVectorization".to_string(),
        description: "Custom vectorization strategy".to_string(),
        implementation: Box::new(|code| {
            // Simple custom optimization: replace float with float4
            Ok(CustomOptimizationResult {
                optimized_code: code.replace("float", "float4"),
                estimated_speedup: 2.0,
                applied_changes: vec!["Vectorization".to_string()],
            })
        }),
        applicability_checker: Box::new(|code| {
            code.contains("float") && !code.contains("float4")
        }),
    };
    
    engine.register_custom_strategy(custom_strategy).unwrap();
    
    let kernel_code = r#"
        __global__ void custom_kernel(float* data, int n) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                data[idx] *= 2.0f;
            }
        }
    "#;
    
    let improvement_request = ImprovementRequest {
        original_code: kernel_code.to_string(),
        target_metrics: TargetMetrics::default(),
        constraints: ImprovementConstraints::default(),
    };
    
    let result = engine.improve_kernel(&improvement_request).unwrap();
    
    if result.success {
        assert!(result.improved_code.contains("float4"));
        
        let has_custom_opt = result.applied_optimizations.iter()
            .any(|opt| matches!(opt, AppliedOptimization::Custom { name, .. } if name == "CustomVectorization"));
        assert!(has_custom_opt);
    }
}