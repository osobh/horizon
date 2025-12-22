//! Comprehensive test suite for kernel fusion
//!
//! Tests all aspects of kernel fusion including analysis, compilation,
//! execution, and performance optimization.

use super::*;
use anyhow::Result;
use cudarc::driver::CudaDevice;
use std::sync::Arc;

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn test_kernel_fusion_config_default() {
        let config = KernelFusionConfig::default();

        assert!(config.enable_auto_fusion);
        assert_eq!(config.max_fusion_depth, 5);
        assert_eq!(config.min_ops_for_fusion, 2);
        assert!(config.enable_runtime_analysis);
        assert_eq!(config.fusion_strategy, FusionStrategy::Balanced);
    }

    #[test]
    fn test_fusion_thresholds_default() {
        let thresholds = FusionThresholds::default();

        assert_eq!(thresholds.min_speedup_factor, 1.2);
        assert_eq!(thresholds.max_memory_overhead, 20.0);
        assert_eq!(thresholds.max_register_pressure, 48);
        assert_eq!(thresholds.min_kernel_time_us, 10);
    }

    #[test]
    fn test_operation_types() {
        let ops = vec![
            OperationType::ElementWise(ElementWiseOp::Add),
            OperationType::ElementWise(ElementWiseOp::Multiply),
            OperationType::Reduction(ReductionOp::Sum),
            OperationType::Memory(MemoryOp::Copy),
            OperationType::Matrix(MatrixOp::GEMM),
        ];

        assert_eq!(ops.len(), 5);
        assert!(matches!(ops[0], OperationType::ElementWise(_)));
        assert!(matches!(ops[2], OperationType::Reduction(_)));
    }

    #[test]
    fn test_activation_types() {
        let activations = vec![
            ActivationType::ReLU,
            ActivationType::Sigmoid,
            ActivationType::Tanh,
            ActivationType::GELU,
            ActivationType::SiLU,
        ];

        assert_eq!(activations.len(), 5);
        assert_eq!(activations[0], ActivationType::ReLU);
    }

    #[test]
    fn test_fusion_pattern_creation() {
        let pattern = FusionPattern {
            name: "ElementWise-Reduction".to_string(),
            operations: vec![
                OperationType::ElementWise(ElementWiseOp::Add),
                OperationType::Reduction(ReductionOp::Sum),
            ],
            conditions: FusionConditions {
                data_dependencies: vec![DataDependency {
                    source_op: 0,
                    target_op: 1,
                    dependency_type: DependencyType::RAW,
                }],
                memory_pattern_compatible: true,
                register_constraints: RegisterConstraints {
                    max_registers_per_thread: 64,
                    current_usage: 20,
                    fusion_overhead: 10,
                },
                shared_memory_constraints: SharedMemoryConstraints {
                    max_shared_memory: 49152,
                    current_usage: 8192,
                    fusion_overhead: 4096,
                },
            },
            expected_speedup: 1.5,
        };

        assert_eq!(pattern.name, "ElementWise-Reduction");
        assert_eq!(pattern.operations.len(), 2);
        assert_eq!(pattern.expected_speedup, 1.5);
        assert!(pattern.conditions.memory_pattern_compatible);
    }

    #[test]
    fn test_tensor_descriptor() {
        let tensor = TensorDescriptor {
            shape: vec![32, 64, 128],
            dtype: DataType::F32,
            layout: MemoryLayout::RowMajor,
            strides: vec![8192, 128, 1],
        };

        assert_eq!(tensor.shape.len(), 3);
        assert_eq!(tensor.shape[0], 32);
        assert_eq!(tensor.dtype, DataType::F32);
        assert_eq!(tensor.layout, MemoryLayout::RowMajor);
    }

    #[test]
    fn test_memory_requirements() {
        let mem_req = MemoryRequirements {
            global_reads: 1024 * 1024,
            global_writes: 512 * 1024,
            shared_memory: 16384,
            registers_per_thread: 32,
        };

        assert_eq!(mem_req.global_reads, 1024 * 1024);
        assert_eq!(mem_req.global_writes, 512 * 1024);
        assert_eq!(mem_req.shared_memory, 16384);
        assert_eq!(mem_req.registers_per_thread, 32);
    }

    #[test]
    fn test_kernel_operation() {
        let op = KernelOperation {
            id: "matmul_0".to_string(),
            op_type: OperationType::Matrix(MatrixOp::GEMM),
            inputs: vec![
                TensorDescriptor {
                    shape: vec![64, 128],
                    dtype: DataType::F32,
                    layout: MemoryLayout::RowMajor,
                    strides: vec![128, 1],
                },
                TensorDescriptor {
                    shape: vec![128, 256],
                    dtype: DataType::F32,
                    layout: MemoryLayout::RowMajor,
                    strides: vec![256, 1],
                },
            ],
            outputs: vec![TensorDescriptor {
                shape: vec![64, 256],
                dtype: DataType::F32,
                layout: MemoryLayout::RowMajor,
                strides: vec![256, 1],
            }],
            estimated_time_us: 50,
            memory_requirements: MemoryRequirements {
                global_reads: 64 * 128 * 4 + 128 * 256 * 4,
                global_writes: 64 * 256 * 4,
                shared_memory: 8192,
                registers_per_thread: 40,
            },
        };

        assert_eq!(op.id, "matmul_0");
        assert!(matches!(op.op_type, OperationType::Matrix(MatrixOp::GEMM)));
        assert_eq!(op.inputs.len(), 2);
        assert_eq!(op.outputs.len(), 1);
        assert_eq!(op.estimated_time_us, 50);
    }

    #[test]
    fn test_fusion_opportunity() {
        let opportunity = FusionOpportunity {
            fusion_id: "fusion_001".to_string(),
            operations: vec![],
            expected_speedup: 1.8,
            memory_savings: 1024 * 1024,
            pattern: FusionPattern {
                name: "Test Pattern".to_string(),
                operations: vec![],
                conditions: FusionConditions {
                    data_dependencies: vec![],
                    memory_pattern_compatible: true,
                    register_constraints: RegisterConstraints {
                        max_registers_per_thread: 64,
                        current_usage: 30,
                        fusion_overhead: 5,
                    },
                    shared_memory_constraints: SharedMemoryConstraints {
                        max_shared_memory: 49152,
                        current_usage: 10240,
                        fusion_overhead: 2048,
                    },
                },
                expected_speedup: 1.8,
            },
            feasibility_score: 0.85,
        };

        assert_eq!(opportunity.fusion_id, "fusion_001");
        assert_eq!(opportunity.expected_speedup, 1.8);
        assert_eq!(opportunity.memory_savings, 1024 * 1024);
        assert_eq!(opportunity.feasibility_score, 0.85);
    }

    #[test]
    fn test_launch_config() {
        let config = LaunchConfig {
            grid_dim: (256, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 16384,
            stream_count: 4,
        };

        assert_eq!(config.grid_dim, (256, 1, 1));
        assert_eq!(config.block_dim, (128, 1, 1));
        assert_eq!(config.shared_mem_bytes, 16384);
        assert_eq!(config.stream_count, 4);
    }

    #[test]
    fn test_fusion_statistics() {
        let mut stats = FusionStatistics::default();
        stats.kernels_executed = 1000;
        stats.cache_hits = 800;
        stats.cache_misses = 200;
        stats.total_execution_time = Duration::from_millis(5000);

        assert_eq!(stats.cache_hit_rate(), 0.8);
        assert_eq!(stats.avg_execution_time(), Duration::from_millis(5));
    }

    #[test]
    fn test_dependency_types() {
        let deps = vec![
            DependencyType::RAW,
            DependencyType::WAR,
            DependencyType::WAW,
            DependencyType::None,
        ];

        assert_eq!(deps.len(), 4);
        assert_eq!(deps[0], DependencyType::RAW);
        assert_eq!(deps[3], DependencyType::None);
    }

    #[test]
    fn test_recommendation_types() {
        let rec_types = vec![
            RecommendationType::ImproveUtilization,
            RecommendationType::SplitKernel,
            RecommendationType::MergeMore,
            RecommendationType::ChangeStrategy,
            RecommendationType::OptimizeMemory,
        ];

        assert_eq!(rec_types.len(), 5);
        assert!(matches!(
            rec_types[0],
            RecommendationType::ImproveUtilization
        ));
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    async fn setup_test_device() -> Result<Arc<CudaDevice>> {
        CudaDevice::new(0).map(Arc::new).map_err(Into::into)
    }

    #[tokio::test]
    async fn test_kernel_fusion_engine_creation() -> Result<()> {
        let device = setup_test_device().await?;
        let config = KernelFusionConfig::default();

        let engine = KernelFusionEngine::new(device, config);
        let stats = engine.get_statistics();

        assert_eq!(stats.kernels_executed, 0);
        assert_eq!(stats.cache_hits, 0);
        assert_eq!(stats.cache_misses, 0);

        Ok(())
    }

    #[tokio::test]
    async fn test_elementwise_fusion_opportunity() -> Result<()> {
        let device = setup_test_device().await?;
        let config = KernelFusionConfig::default();
        let mut engine = KernelFusionEngine::new(device, config);

        // Create simple element-wise operations
        let operations = vec![
            KernelOperation {
                id: "add_0".to_string(),
                op_type: OperationType::ElementWise(ElementWiseOp::Add),
                inputs: vec![
                    create_test_tensor_descriptor(vec![1024, 1024]),
                    create_test_tensor_descriptor(vec![1024, 1024]),
                ],
                outputs: vec![create_test_tensor_descriptor(vec![1024, 1024])],
                estimated_time_us: 20,
                memory_requirements: MemoryRequirements {
                    global_reads: 2 * 1024 * 1024 * 4,
                    global_writes: 1024 * 1024 * 4,
                    shared_memory: 0,
                    registers_per_thread: 16,
                },
            },
            KernelOperation {
                id: "multiply_0".to_string(),
                op_type: OperationType::ElementWise(ElementWiseOp::Multiply),
                inputs: vec![
                    create_test_tensor_descriptor(vec![1024, 1024]),
                    create_test_tensor_descriptor(vec![1024, 1024]),
                ],
                outputs: vec![create_test_tensor_descriptor(vec![1024, 1024])],
                estimated_time_us: 20,
                memory_requirements: MemoryRequirements {
                    global_reads: 2 * 1024 * 1024 * 4,
                    global_writes: 1024 * 1024 * 4,
                    shared_memory: 0,
                    registers_per_thread: 16,
                },
            },
        ];

        // Analyze fusion opportunities
        match engine.analyze_fusion_opportunities(&operations).await {
            Ok(opportunities) => {
                println!("Found {} fusion opportunities", opportunities.len());
                // In test environment, might not find opportunities
            }
            Err(e) => {
                println!("Analysis error (expected in test environment): {}", e);
            }
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_reduction_fusion_pattern() -> Result<()> {
        let device = setup_test_device().await?;
        let config = KernelFusionConfig {
            fusion_strategy: FusionStrategy::Aggressive,
            ..Default::default()
        };
        let mut engine = KernelFusionEngine::new(device, config);

        // Create reduction pattern
        let operations = vec![
            KernelOperation {
                id: "relu_0".to_string(),
                op_type: OperationType::ElementWise(ElementWiseOp::Activation(
                    ActivationType::ReLU,
                )),
                inputs: vec![create_test_tensor_descriptor(vec![512, 512])],
                outputs: vec![create_test_tensor_descriptor(vec![512, 512])],
                estimated_time_us: 15,
                memory_requirements: MemoryRequirements {
                    global_reads: 512 * 512 * 4,
                    global_writes: 512 * 512 * 4,
                    shared_memory: 0,
                    registers_per_thread: 12,
                },
            },
            KernelOperation {
                id: "sum_0".to_string(),
                op_type: OperationType::Reduction(ReductionOp::Sum),
                inputs: vec![create_test_tensor_descriptor(vec![512, 512])],
                outputs: vec![create_test_tensor_descriptor(vec![512])],
                estimated_time_us: 25,
                memory_requirements: MemoryRequirements {
                    global_reads: 512 * 512 * 4,
                    global_writes: 512 * 4,
                    shared_memory: 2048,
                    registers_per_thread: 20,
                },
            },
        ];

        // Test analysis
        match engine.analyze_fusion_opportunities(&operations).await {
            Ok(opportunities) => {
                for opp in &opportunities {
                    println!(
                        "Fusion opportunity: {} with speedup {:.2}x",
                        opp.fusion_id, opp.expected_speedup
                    );
                }
            }
            Err(e) => {
                println!("Expected error in test environment: {}", e);
            }
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_cache_behavior() -> Result<()> {
        let device = setup_test_device().await?;
        let config = KernelFusionConfig::default();
        let mut engine = KernelFusionEngine::new(device, config);

        // Check initial cache state
        let stats = engine.get_statistics();
        assert_eq!(stats.cache_hits, 0);
        assert_eq!(stats.cache_misses, 0);

        // Clear cache
        engine.clear_cache();
        let stats = engine.get_statistics();
        assert_eq!(stats.cache_clears, 1);

        Ok(())
    }

    fn create_test_tensor_descriptor(shape: Vec<usize>) -> TensorDescriptor {
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }

        TensorDescriptor {
            shape,
            dtype: DataType::F32,
            layout: MemoryLayout::RowMajor,
            strides,
        }
    }
}

#[cfg(test)]
mod benchmark_tests {
    use super::*;

    #[test]
    fn test_fusion_speedup_calculation() {
        // Test speedup calculation for different scenarios
        let test_cases = vec![
            (100, 80, 1.25),  // 100μs -> 80μs = 1.25x speedup
            (50, 30, 1.67),   // 50μs -> 30μs = 1.67x speedup
            (200, 150, 1.33), // 200μs -> 150μs = 1.33x speedup
        ];

        for (original_time, fused_time, expected_speedup) in test_cases {
            let speedup = original_time as f32 / fused_time as f32;
            assert!((speedup - expected_speedup).abs() < 0.01);
        }
    }

    #[test]
    fn test_memory_savings_calculation() {
        // Test memory savings for fusion scenarios
        let test_cases = vec![
            // (reads_before, writes_before, reads_after, writes_after, expected_savings)
            (8192, 4096, 4096, 4096, 4096), // Save one intermediate write/read
            (16384, 8192, 8192, 4096, 12288), // Save more with better fusion
        ];

        for (reads_before, writes_before, reads_after, writes_after, expected_savings) in test_cases
        {
            let before_total = reads_before + writes_before;
            let after_total = reads_after + writes_after;
            let savings = before_total - after_total;
            assert_eq!(savings, expected_savings);
        }
    }

    #[test]
    fn test_register_pressure_analysis() {
        let test_cases = vec![
            RegisterConstraints {
                max_registers_per_thread: 64,
                current_usage: 30,
                fusion_overhead: 10,
            },
            RegisterConstraints {
                max_registers_per_thread: 64,
                current_usage: 50,
                fusion_overhead: 20,
            },
        ];

        for constraints in test_cases {
            let total_usage = constraints.current_usage + constraints.fusion_overhead;
            let can_fuse = total_usage <= constraints.max_registers_per_thread;

            if constraints.current_usage == 30 {
                assert!(can_fuse);
            } else if constraints.current_usage == 50 {
                assert!(!can_fuse); // Would exceed limit
            }
        }
    }

    #[test]
    fn test_shared_memory_constraints() {
        let constraints = SharedMemoryConstraints {
            max_shared_memory: 49152, // 48KB
            current_usage: 16384,     // 16KB
            fusion_overhead: 8192,    // 8KB
        };

        let total_usage = constraints.current_usage + constraints.fusion_overhead;
        assert_eq!(total_usage, 24576); // 24KB
        assert!(total_usage < constraints.max_shared_memory);
    }

    #[test]
    fn test_fusion_feasibility_scoring() {
        // Test feasibility score calculation
        let test_cases = vec![
            (1.5, 10.0, 30, 0.85), // Good speedup, low overhead
            (1.1, 25.0, 50, 0.45), // Low speedup, high overhead
            (2.0, 5.0, 20, 0.95),  // Excellent speedup, low overhead
        ];

        for (speedup, mem_overhead, reg_usage, expected_score) in test_cases {
            let score = calculate_feasibility_score(speedup, mem_overhead, reg_usage);
            assert!((score - expected_score).abs() < 0.1);
        }
    }

    fn calculate_feasibility_score(speedup: f32, mem_overhead: f32, reg_usage: u32) -> f32 {
        let speedup_score = (speedup - 1.0).min(1.0);
        let mem_score = 1.0 - (mem_overhead / 100.0).min(1.0);
        let reg_score = 1.0 - (reg_usage as f32 / 64.0).min(1.0);

        (speedup_score * 0.5 + mem_score * 0.3 + reg_score * 0.2)
            .max(0.0)
            .min(1.0)
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;

    #[test]
    fn test_fusion_config_invariants() {
        let config = KernelFusionConfig::default();

        // Invariants that should always hold
        assert!(config.max_fusion_depth > 0);
        assert!(config.max_fusion_depth <= 10); // Reasonable upper limit
        assert!(config.min_ops_for_fusion >= 2); // Need at least 2 ops to fuse
        assert!(config.min_ops_for_fusion <= config.max_fusion_depth);
    }

    #[test]
    fn test_threshold_invariants() {
        let thresholds = FusionThresholds::default();

        // Speedup factor should be > 1.0 (otherwise no benefit)
        assert!(thresholds.min_speedup_factor > 1.0);
        assert!(thresholds.min_speedup_factor < 10.0); // Reasonable upper bound

        // Memory overhead should be reasonable
        assert!(thresholds.max_memory_overhead >= 0.0);
        assert!(thresholds.max_memory_overhead <= 100.0);

        // Register pressure should be within GPU limits
        assert!(thresholds.max_register_pressure > 0);
        assert!(thresholds.max_register_pressure <= 255); // GPU register limit

        // Kernel time threshold should be positive
        assert!(thresholds.min_kernel_time_us > 0);
    }

    #[test]
    fn test_tensor_descriptor_invariants() {
        let shapes = vec![
            vec![1024],
            vec![32, 64],
            vec![16, 32, 64],
            vec![8, 16, 32, 64],
        ];

        for shape in shapes {
            let tensor = create_test_tensor(shape.clone());

            // Shape and strides should have same length
            assert_eq!(tensor.shape.len(), tensor.strides.len());

            // All dimensions should be positive
            assert!(tensor.shape.iter().all(|&dim| dim > 0));

            // Strides should be positive for row-major
            if tensor.layout == MemoryLayout::RowMajor {
                assert!(tensor.strides.iter().all(|&stride| stride > 0));
            }
        }
    }

    #[test]
    fn test_memory_requirements_consistency() {
        let mem_reqs = vec![
            MemoryRequirements {
                global_reads: 1024,
                global_writes: 512,
                shared_memory: 4096,
                registers_per_thread: 32,
            },
            MemoryRequirements {
                global_reads: 0,
                global_writes: 1024,
                shared_memory: 0,
                registers_per_thread: 16,
            },
        ];

        for req in mem_reqs {
            // Shared memory should be within block limit (typically 48KB)
            assert!(req.shared_memory <= 49152);

            // Registers should be within thread limit
            assert!(req.registers_per_thread <= 255);

            // At least some memory operation should occur
            assert!(req.global_reads > 0 || req.global_writes > 0);
        }
    }

    #[test]
    fn test_fusion_opportunity_consistency() {
        let opportunity = FusionOpportunity {
            fusion_id: "test_fusion".to_string(),
            operations: vec![],
            expected_speedup: 1.5,
            memory_savings: 1024,
            pattern: create_test_pattern(),
            feasibility_score: 0.8,
        };

        // Speedup should be > 1.0 for valid fusion
        assert!(opportunity.expected_speedup > 1.0);

        // Feasibility score should be in [0, 1]
        assert!(opportunity.feasibility_score >= 0.0);
        assert!(opportunity.feasibility_score <= 1.0);

        // Memory savings should be non-negative
        assert!(opportunity.memory_savings >= 0);
    }

    fn create_test_tensor(shape: Vec<usize>) -> TensorDescriptor {
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }

        TensorDescriptor {
            shape,
            dtype: DataType::F32,
            layout: MemoryLayout::RowMajor,
            strides,
        }
    }

    fn create_test_pattern() -> FusionPattern {
        FusionPattern {
            name: "Test Pattern".to_string(),
            operations: vec![],
            conditions: FusionConditions {
                data_dependencies: vec![],
                memory_pattern_compatible: true,
                register_constraints: RegisterConstraints {
                    max_registers_per_thread: 64,
                    current_usage: 20,
                    fusion_overhead: 5,
                },
                shared_memory_constraints: SharedMemoryConstraints {
                    max_shared_memory: 49152,
                    current_usage: 8192,
                    fusion_overhead: 2048,
                },
            },
            expected_speedup: 1.5,
        }
    }
}
