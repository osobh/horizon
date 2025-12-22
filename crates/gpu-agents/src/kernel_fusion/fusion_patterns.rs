//! Kernel fusion patterns module
//!
//! Defines common fusion patterns and optimization strategies for
//! efficient kernel combination.

use super::*;
use anyhow::{anyhow, Result};
use std::collections::HashMap;

/// Fusion pattern library
pub struct FusionPatternLibrary {
    patterns: HashMap<String, FusionPattern>,
    custom_patterns: Vec<FusionPattern>,
}

impl FusionPatternLibrary {
    /// Create new pattern library with builtin patterns
    pub fn new() -> Self {
        let mut library = Self {
            patterns: HashMap::new(),
            custom_patterns: Vec::new(),
        };

        // Register builtin patterns
        library.register_builtin_patterns();

        library
    }

    /// Register all builtin fusion patterns
    fn register_builtin_patterns(&mut self) {
        // Element-wise fusion patterns
        self.register_elementwise_patterns();

        // Reduction fusion patterns
        self.register_reduction_patterns();

        // Matrix operation patterns
        self.register_matrix_patterns();

        // Memory optimization patterns
        self.register_memory_patterns();

        // Complex fusion patterns
        self.register_complex_patterns();
    }

    /// Register element-wise patterns
    fn register_elementwise_patterns(&mut self) {
        // Binary operation fusion
        self.add_pattern(FusionPattern {
            name: "BinaryOpChain".to_string(),
            operations: vec![
                OperationType::ElementWise(ElementWiseOp::Add),
                OperationType::ElementWise(ElementWiseOp::Multiply),
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
                    current_usage: 16,
                    fusion_overhead: 4,
                },
                shared_memory_constraints: SharedMemoryConstraints {
                    max_shared_memory: 49152,
                    current_usage: 0,
                    fusion_overhead: 0,
                },
            },
            expected_speedup: 1.8,
        });

        // Activation chain
        self.add_pattern(FusionPattern {
            name: "ActivationChain".to_string(),
            operations: vec![
                OperationType::ElementWise(ElementWiseOp::Add),
                OperationType::ElementWise(ElementWiseOp::Activation(ActivationType::ReLU)),
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
                    current_usage: 12,
                    fusion_overhead: 2,
                },
                shared_memory_constraints: SharedMemoryConstraints {
                    max_shared_memory: 49152,
                    current_usage: 0,
                    fusion_overhead: 0,
                },
            },
            expected_speedup: 1.6,
        });

        // Multi-activation fusion
        self.add_pattern(FusionPattern {
            name: "MultiActivation".to_string(),
            operations: vec![
                OperationType::ElementWise(ElementWiseOp::Activation(ActivationType::ReLU)),
                OperationType::ElementWise(ElementWiseOp::Activation(ActivationType::Sigmoid)),
            ],
            conditions: FusionConditions {
                data_dependencies: vec![],
                memory_pattern_compatible: true,
                register_constraints: RegisterConstraints {
                    max_registers_per_thread: 64,
                    current_usage: 20,
                    fusion_overhead: 6,
                },
                shared_memory_constraints: SharedMemoryConstraints {
                    max_shared_memory: 49152,
                    current_usage: 0,
                    fusion_overhead: 0,
                },
            },
            expected_speedup: 1.4,
        });
    }

    /// Register reduction patterns
    fn register_reduction_patterns(&mut self) {
        // Map-reduce pattern
        self.add_pattern(FusionPattern {
            name: "MapReduce".to_string(),
            operations: vec![
                OperationType::ElementWise(ElementWiseOp::Multiply),
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
                    current_usage: 24,
                    fusion_overhead: 8,
                },
                shared_memory_constraints: SharedMemoryConstraints {
                    max_shared_memory: 49152,
                    current_usage: 4096,
                    fusion_overhead: 2048,
                },
            },
            expected_speedup: 2.0,
        });

        // Softmax pattern (max + exp + sum + div)
        self.add_pattern(FusionPattern {
            name: "Softmax".to_string(),
            operations: vec![
                OperationType::Reduction(ReductionOp::Max),
                OperationType::ElementWise(ElementWiseOp::Subtract),
                OperationType::ElementWise(ElementWiseOp::Activation(ActivationType::Sigmoid)), // exp approximation
                OperationType::Reduction(ReductionOp::Sum),
                OperationType::ElementWise(ElementWiseOp::Divide),
            ],
            conditions: FusionConditions {
                data_dependencies: vec![
                    DataDependency {
                        source_op: 0,
                        target_op: 1,
                        dependency_type: DependencyType::RAW,
                    },
                    DataDependency {
                        source_op: 1,
                        target_op: 2,
                        dependency_type: DependencyType::RAW,
                    },
                    DataDependency {
                        source_op: 2,
                        target_op: 3,
                        dependency_type: DependencyType::RAW,
                    },
                    DataDependency {
                        source_op: 3,
                        target_op: 4,
                        dependency_type: DependencyType::RAW,
                    },
                ],
                memory_pattern_compatible: true,
                register_constraints: RegisterConstraints {
                    max_registers_per_thread: 64,
                    current_usage: 32,
                    fusion_overhead: 12,
                },
                shared_memory_constraints: SharedMemoryConstraints {
                    max_shared_memory: 49152,
                    current_usage: 8192,
                    fusion_overhead: 4096,
                },
            },
            expected_speedup: 3.0,
        });

        // Layer normalization pattern
        self.add_pattern(FusionPattern {
            name: "LayerNorm".to_string(),
            operations: vec![
                OperationType::Reduction(ReductionOp::Mean),
                OperationType::ElementWise(ElementWiseOp::Subtract),
                OperationType::ElementWise(ElementWiseOp::Power),
                OperationType::Reduction(ReductionOp::Mean),
                OperationType::ElementWise(ElementWiseOp::Add), // epsilon
                OperationType::ElementWise(ElementWiseOp::Power), // sqrt
                OperationType::ElementWise(ElementWiseOp::Divide),
            ],
            conditions: FusionConditions {
                data_dependencies: vec![], // Complex dependencies
                memory_pattern_compatible: true,
                register_constraints: RegisterConstraints {
                    max_registers_per_thread: 64,
                    current_usage: 40,
                    fusion_overhead: 16,
                },
                shared_memory_constraints: SharedMemoryConstraints {
                    max_shared_memory: 49152,
                    current_usage: 8192,
                    fusion_overhead: 4096,
                },
            },
            expected_speedup: 2.5,
        });
    }

    /// Register matrix operation patterns
    fn register_matrix_patterns(&mut self) {
        // GEMM + Bias + Activation
        self.add_pattern(FusionPattern {
            name: "GemmBiasActivation".to_string(),
            operations: vec![
                OperationType::Matrix(MatrixOp::GEMM),
                OperationType::ElementWise(ElementWiseOp::Add), // bias
                OperationType::ElementWise(ElementWiseOp::Activation(ActivationType::ReLU)),
            ],
            conditions: FusionConditions {
                data_dependencies: vec![
                    DataDependency {
                        source_op: 0,
                        target_op: 1,
                        dependency_type: DependencyType::RAW,
                    },
                    DataDependency {
                        source_op: 1,
                        target_op: 2,
                        dependency_type: DependencyType::RAW,
                    },
                ],
                memory_pattern_compatible: true,
                register_constraints: RegisterConstraints {
                    max_registers_per_thread: 64,
                    current_usage: 48,
                    fusion_overhead: 8,
                },
                shared_memory_constraints: SharedMemoryConstraints {
                    max_shared_memory: 49152,
                    current_usage: 16384,
                    fusion_overhead: 4096,
                },
            },
            expected_speedup: 1.4,
        });

        // Batched GEMM fusion
        self.add_pattern(FusionPattern {
            name: "BatchedGemmFusion".to_string(),
            operations: vec![
                OperationType::Matrix(MatrixOp::BatchedGEMM),
                OperationType::Matrix(MatrixOp::BatchedGEMM),
            ],
            conditions: FusionConditions {
                data_dependencies: vec![],
                memory_pattern_compatible: true,
                register_constraints: RegisterConstraints {
                    max_registers_per_thread: 64,
                    current_usage: 56,
                    fusion_overhead: 4,
                },
                shared_memory_constraints: SharedMemoryConstraints {
                    max_shared_memory: 49152,
                    current_usage: 32768,
                    fusion_overhead: 8192,
                },
            },
            expected_speedup: 1.3,
        });

        // Convolution + pooling
        self.add_pattern(FusionPattern {
            name: "ConvPooling".to_string(),
            operations: vec![
                OperationType::Matrix(MatrixOp::Convolution),
                OperationType::Matrix(MatrixOp::Pooling),
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
                    current_usage: 44,
                    fusion_overhead: 12,
                },
                shared_memory_constraints: SharedMemoryConstraints {
                    max_shared_memory: 49152,
                    current_usage: 16384,
                    fusion_overhead: 8192,
                },
            },
            expected_speedup: 1.7,
        });
    }

    /// Register memory optimization patterns
    fn register_memory_patterns(&mut self) {
        // Transpose + GEMM
        self.add_pattern(FusionPattern {
            name: "TransposeGemm".to_string(),
            operations: vec![
                OperationType::Memory(MemoryOp::Transpose),
                OperationType::Matrix(MatrixOp::GEMM),
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
                    current_usage: 40,
                    fusion_overhead: 8,
                },
                shared_memory_constraints: SharedMemoryConstraints {
                    max_shared_memory: 49152,
                    current_usage: 16384,
                    fusion_overhead: 0,
                },
            },
            expected_speedup: 1.5,
        });

        // Reshape + operation
        self.add_pattern(FusionPattern {
            name: "ReshapeOp".to_string(),
            operations: vec![
                OperationType::Memory(MemoryOp::Reshape),
                OperationType::ElementWise(ElementWiseOp::Multiply),
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
                    current_usage: 16,
                    fusion_overhead: 4,
                },
                shared_memory_constraints: SharedMemoryConstraints {
                    max_shared_memory: 49152,
                    current_usage: 0,
                    fusion_overhead: 0,
                },
            },
            expected_speedup: 1.9,
        });

        // Gather-scatter optimization
        self.add_pattern(FusionPattern {
            name: "GatherScatter".to_string(),
            operations: vec![
                OperationType::Memory(MemoryOp::Gather),
                OperationType::ElementWise(ElementWiseOp::Multiply),
                OperationType::Memory(MemoryOp::Scatter),
            ],
            conditions: FusionConditions {
                data_dependencies: vec![
                    DataDependency {
                        source_op: 0,
                        target_op: 1,
                        dependency_type: DependencyType::RAW,
                    },
                    DataDependency {
                        source_op: 1,
                        target_op: 2,
                        dependency_type: DependencyType::RAW,
                    },
                ],
                memory_pattern_compatible: true,
                register_constraints: RegisterConstraints {
                    max_registers_per_thread: 64,
                    current_usage: 28,
                    fusion_overhead: 12,
                },
                shared_memory_constraints: SharedMemoryConstraints {
                    max_shared_memory: 49152,
                    current_usage: 8192,
                    fusion_overhead: 4096,
                },
            },
            expected_speedup: 2.2,
        });
    }

    /// Register complex fusion patterns
    fn register_complex_patterns(&mut self) {
        // Attention pattern (simplified)
        self.add_pattern(FusionPattern {
            name: "AttentionBlock".to_string(),
            operations: vec![
                OperationType::Matrix(MatrixOp::GEMM),               // Q
                OperationType::Matrix(MatrixOp::GEMM),               // K
                OperationType::Matrix(MatrixOp::GEMM),               // V
                OperationType::ElementWise(ElementWiseOp::Multiply), // QK^T
                OperationType::ElementWise(ElementWiseOp::Divide),   // scale
                OperationType::Reduction(ReductionOp::Max),          // softmax part 1
                OperationType::ElementWise(ElementWiseOp::Subtract),
                OperationType::ElementWise(ElementWiseOp::Activation(ActivationType::Sigmoid)),
                OperationType::Reduction(ReductionOp::Sum), // softmax part 2
                OperationType::ElementWise(ElementWiseOp::Divide),
                OperationType::Matrix(MatrixOp::GEMM), // attention * V
            ],
            conditions: FusionConditions {
                data_dependencies: vec![], // Complex dependencies
                memory_pattern_compatible: true,
                register_constraints: RegisterConstraints {
                    max_registers_per_thread: 64,
                    current_usage: 56,
                    fusion_overhead: 8,
                },
                shared_memory_constraints: SharedMemoryConstraints {
                    max_shared_memory: 49152,
                    current_usage: 32768,
                    fusion_overhead: 16384,
                },
            },
            expected_speedup: 2.8,
        });

        // Residual block pattern
        self.add_pattern(FusionPattern {
            name: "ResidualBlock".to_string(),
            operations: vec![
                OperationType::Matrix(MatrixOp::Convolution),
                OperationType::ElementWise(ElementWiseOp::Add), // bias
                OperationType::ElementWise(ElementWiseOp::Activation(ActivationType::ReLU)),
                OperationType::Matrix(MatrixOp::Convolution),
                OperationType::ElementWise(ElementWiseOp::Add), // bias
                OperationType::ElementWise(ElementWiseOp::Add), // residual
                OperationType::ElementWise(ElementWiseOp::Activation(ActivationType::ReLU)),
            ],
            conditions: FusionConditions {
                data_dependencies: vec![], // Complex dependencies with skip connection
                memory_pattern_compatible: true,
                register_constraints: RegisterConstraints {
                    max_registers_per_thread: 64,
                    current_usage: 52,
                    fusion_overhead: 12,
                },
                shared_memory_constraints: SharedMemoryConstraints {
                    max_shared_memory: 49152,
                    current_usage: 24576,
                    fusion_overhead: 12288,
                },
            },
            expected_speedup: 2.2,
        });
    }

    /// Add a pattern to the library
    pub fn add_pattern(&mut self, pattern: FusionPattern) {
        self.patterns.insert(pattern.name.clone(), pattern);
    }

    /// Get a pattern by name
    pub fn get_pattern(&self, name: &str) -> Option<&FusionPattern> {
        self.patterns.get(name)
    }

    /// Get all patterns
    pub fn get_all_patterns(&self) -> Vec<&FusionPattern> {
        self.patterns.values().collect()
    }

    /// Add custom pattern
    pub fn add_custom_pattern(&mut self, pattern: FusionPattern) {
        self.custom_patterns.push(pattern.clone());
        self.add_pattern(pattern);
    }

    /// Find patterns matching operations
    pub fn find_matching_patterns(&self, operations: &[KernelOperation]) -> Vec<&FusionPattern> {
        let mut matches = Vec::new();

        for pattern in self.patterns.values() {
            if self.operations_match_pattern(operations, pattern) {
                matches.push(pattern);
            }
        }

        // Sort by expected speedup (highest first)
        matches.sort_by(|a, b| {
            b.expected_speedup
                .partial_cmp(&a.expected_speedup)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        matches
    }

    /// Check if operations match a pattern
    fn operations_match_pattern(
        &self,
        operations: &[KernelOperation],
        pattern: &FusionPattern,
    ) -> bool {
        if operations.len() != pattern.operations.len() {
            return false;
        }

        // Check operation types match
        for (op, pattern_op) in operations.iter().zip(&pattern.operations) {
            if !self.op_type_matches(&op.op_type, pattern_op) {
                return false;
            }
        }

        true
    }

    /// Check if operation type matches pattern
    fn op_type_matches(&self, op: &OperationType, pattern: &OperationType) -> bool {
        match (op, pattern) {
            (OperationType::ElementWise(a), OperationType::ElementWise(b)) => {
                self.elementwise_op_matches(a, b)
            }
            (OperationType::Reduction(a), OperationType::Reduction(b)) => a == b,
            (OperationType::Memory(a), OperationType::Memory(b)) => a == b,
            (OperationType::Matrix(a), OperationType::Matrix(b)) => a == b,
            (OperationType::Custom(a), OperationType::Custom(b)) => a == b,
            _ => false,
        }
    }

    /// Check if element-wise operations match
    fn elementwise_op_matches(&self, op: &ElementWiseOp, pattern: &ElementWiseOp) -> bool {
        match (op, pattern) {
            // Exact matches
            (ElementWiseOp::Add, ElementWiseOp::Add) => true,
            (ElementWiseOp::Multiply, ElementWiseOp::Multiply) => true,
            (ElementWiseOp::Subtract, ElementWiseOp::Subtract) => true,
            (ElementWiseOp::Divide, ElementWiseOp::Divide) => true,
            (ElementWiseOp::Power, ElementWiseOp::Power) => true,

            // Activation matches (any activation matches any pattern activation)
            (ElementWiseOp::Activation(_), ElementWiseOp::Activation(_)) => true,

            _ => false,
        }
    }

    /// Analyze pattern effectiveness
    pub fn analyze_pattern_effectiveness(
        &self,
        pattern: &FusionPattern,
        hardware_config: &HardwareConfig,
    ) -> PatternEffectiveness {
        // Calculate various effectiveness metrics
        let memory_efficiency = self.calculate_memory_efficiency(pattern, hardware_config);
        let compute_efficiency = self.calculate_compute_efficiency(pattern, hardware_config);
        let overall_score = self.calculate_overall_score(pattern, hardware_config);

        PatternEffectiveness {
            pattern_name: pattern.name.clone(),
            memory_efficiency,
            compute_efficiency,
            overall_score,
            recommendations: self.generate_recommendations(pattern, hardware_config),
        }
    }

    /// Calculate memory efficiency
    fn calculate_memory_efficiency(
        &self,
        pattern: &FusionPattern,
        hardware_config: &HardwareConfig,
    ) -> f32 {
        let shared_mem_usage = pattern.conditions.shared_memory_constraints.current_usage
            + pattern.conditions.shared_memory_constraints.fusion_overhead;

        let shared_mem_efficiency =
            1.0 - (shared_mem_usage as f32 / hardware_config.max_shared_memory as f32);

        shared_mem_efficiency.max(0.0).min(1.0)
    }

    /// Calculate compute efficiency
    fn calculate_compute_efficiency(
        &self,
        pattern: &FusionPattern,
        hardware_config: &HardwareConfig,
    ) -> f32 {
        let register_usage = pattern.conditions.register_constraints.current_usage
            + pattern.conditions.register_constraints.fusion_overhead;

        let register_efficiency =
            1.0 - (register_usage as f32 / hardware_config.max_registers_per_thread as f32);

        register_efficiency.max(0.0).min(1.0)
    }

    /// Calculate overall effectiveness score
    fn calculate_overall_score(
        &self,
        pattern: &FusionPattern,
        hardware_config: &HardwareConfig,
    ) -> f32 {
        let memory_eff = self.calculate_memory_efficiency(pattern, hardware_config);
        let compute_eff = self.calculate_compute_efficiency(pattern, hardware_config);
        let speedup_factor = pattern.expected_speedup.min(3.0) / 3.0;

        // Weighted average
        (memory_eff * 0.3 + compute_eff * 0.3 + speedup_factor * 0.4)
            .max(0.0)
            .min(1.0)
    }

    /// Generate recommendations for pattern usage
    fn generate_recommendations(
        &self,
        pattern: &FusionPattern,
        hardware_config: &HardwareConfig,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Check shared memory usage
        let shared_mem_usage = pattern.conditions.shared_memory_constraints.current_usage
            + pattern.conditions.shared_memory_constraints.fusion_overhead;

        if shared_mem_usage > hardware_config.max_shared_memory * 80 / 100 {
            recommendations.push(format!(
                "High shared memory usage ({} bytes). Consider reducing fusion depth.",
                shared_mem_usage
            ));
        }

        // Check register pressure
        let register_usage = pattern.conditions.register_constraints.current_usage
            + pattern.conditions.register_constraints.fusion_overhead;

        if register_usage > hardware_config.max_registers_per_thread * 75 / 100 {
            recommendations.push(format!(
                "High register pressure ({} registers). May limit occupancy.",
                register_usage
            ));
        }

        // Check expected speedup
        if pattern.expected_speedup < 1.2 {
            recommendations
                .push("Low expected speedup. Consider if fusion overhead is worth it.".to_string());
        }

        recommendations
    }
}

/// Hardware configuration for pattern analysis
#[derive(Debug, Clone)]
pub struct HardwareConfig {
    pub max_shared_memory: usize,
    pub max_registers_per_thread: u32,
    pub warp_size: u32,
    pub max_threads_per_block: u32,
    pub sm_count: u32,
}

impl Default for HardwareConfig {
    fn default() -> Self {
        // Default to A100 configuration
        Self {
            max_shared_memory: 49152,               // 48KB
            max_registers_per_thread: 65536 / 1024, // 64 registers
            warp_size: 32,
            max_threads_per_block: 1024,
            sm_count: 108,
        }
    }
}

/// Pattern effectiveness analysis results
#[derive(Debug, Clone)]
pub struct PatternEffectiveness {
    pub pattern_name: String,
    pub memory_efficiency: f32,
    pub compute_efficiency: f32,
    pub overall_score: f32,
    pub recommendations: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pattern_library_creation() {
        let library = FusionPatternLibrary::new();

        // Should have builtin patterns
        assert!(!library.patterns.is_empty());

        // Check specific patterns exist
        assert!(library.get_pattern("BinaryOpChain").is_some());
        assert!(library.get_pattern("MapReduce").is_some());
        assert!(library.get_pattern("GemmBiasActivation").is_some());
    }

    #[test]
    fn test_pattern_matching() {
        let library = FusionPatternLibrary::new();

        // Create test operations
        let ops = vec![
            KernelOperation {
                id: "add".to_string(),
                op_type: OperationType::ElementWise(ElementWiseOp::Add),
                inputs: vec![],
                outputs: vec![],
                estimated_time_us: 10,
                memory_requirements: MemoryRequirements {
                    global_reads: 1024,
                    global_writes: 1024,
                    shared_memory: 0,
                    registers_per_thread: 8,
                },
            },
            KernelOperation {
                id: "mul".to_string(),
                op_type: OperationType::ElementWise(ElementWiseOp::Multiply),
                inputs: vec![],
                outputs: vec![],
                estimated_time_us: 10,
                memory_requirements: MemoryRequirements {
                    global_reads: 1024,
                    global_writes: 1024,
                    shared_memory: 0,
                    registers_per_thread: 8,
                },
            },
        ];

        let matches = library.find_matching_patterns(&ops);
        assert!(!matches.is_empty());
        assert_eq!(matches[0].name, "BinaryOpChain");
    }

    #[test]
    fn test_pattern_effectiveness() {
        let library = FusionPatternLibrary::new();
        let hardware = HardwareConfig::default();

        if let Some(pattern) = library.get_pattern("MapReduce") {
            let effectiveness = library.analyze_pattern_effectiveness(pattern, &hardware);

            assert!(effectiveness.memory_efficiency >= 0.0);
            assert!(effectiveness.memory_efficiency <= 1.0);
            assert!(effectiveness.compute_efficiency >= 0.0);
            assert!(effectiveness.compute_efficiency <= 1.0);
            assert!(effectiveness.overall_score >= 0.0);
            assert!(effectiveness.overall_score <= 1.0);
        }
    }

    #[test]
    fn test_custom_pattern() {
        let mut library = FusionPatternLibrary::new();

        let custom_pattern = FusionPattern {
            name: "CustomPattern".to_string(),
            operations: vec![
                OperationType::ElementWise(ElementWiseOp::Add),
                OperationType::ElementWise(ElementWiseOp::Add),
            ],
            conditions: FusionConditions {
                data_dependencies: vec![],
                memory_pattern_compatible: true,
                register_constraints: RegisterConstraints {
                    max_registers_per_thread: 64,
                    current_usage: 16,
                    fusion_overhead: 4,
                },
                shared_memory_constraints: SharedMemoryConstraints {
                    max_shared_memory: 49152,
                    current_usage: 0,
                    fusion_overhead: 0,
                },
            },
            expected_speedup: 1.5,
        };

        library.add_custom_pattern(custom_pattern);

        assert!(library.get_pattern("CustomPattern").is_some());
        assert_eq!(library.custom_patterns.len(), 1);
    }
}
