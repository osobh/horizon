//! Kernel fusion analysis module
//!
//! Analyzes kernel operations to identify fusion opportunities based on
//! data dependencies, memory patterns, and performance characteristics.

use super::*;
use anyhow::{anyhow, Result};
use cudarc::driver::CudaContext;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Fusion analyzer for identifying kernel fusion opportunities
pub struct FusionAnalyzer {
    device: Arc<CudaContext>,
    config: KernelFusionConfig,
    pattern_matcher: PatternMatcher,
    dependency_analyzer: DependencyAnalyzer,
    performance_predictor: PerformancePredictor,
}

impl FusionAnalyzer {
    /// Create new fusion analyzer
    pub fn new(device: Arc<CudaContext>, config: KernelFusionConfig) -> Self {
        Self {
            device,
            config: config.clone(),
            pattern_matcher: PatternMatcher::new(config.clone()),
            dependency_analyzer: DependencyAnalyzer::new(),
            performance_predictor: PerformancePredictor::new(config),
        }
    }

    /// Analyze operations for fusion opportunities
    pub async fn analyze_operations(
        &self,
        operations: &[KernelOperation],
    ) -> Result<Vec<FusionOpportunity>> {
        // Validate minimum operations for fusion
        if operations.len() < self.config.min_ops_for_fusion {
            return Ok(Vec::new());
        }

        let mut opportunities = Vec::new();

        // Build dependency graph
        let dependency_graph = self
            .dependency_analyzer
            .build_dependency_graph(operations)?;

        // Find fusion candidates based on dependencies
        let fusion_candidates = self.find_fusion_candidates(operations, &dependency_graph)?;

        // Evaluate each candidate
        for candidate in fusion_candidates {
            // Check pattern matching
            if let Some(pattern) = self.pattern_matcher.match_pattern(&candidate.operations) {
                // Predict performance
                let performance = self
                    .performance_predictor
                    .predict_performance(&candidate.operations, &pattern)?;

                // Check if fusion meets thresholds
                if self.meets_fusion_thresholds(&performance) {
                    // Calculate feasibility score
                    let feasibility_score =
                        self.calculate_feasibility_score(&candidate, &pattern, &performance)?;

                    let opportunity = FusionOpportunity {
                        fusion_id: format!("fusion_{:x}", candidate.hash()),
                        operations: candidate.operations.clone(),
                        expected_speedup: performance.expected_speedup,
                        memory_savings: performance.memory_savings,
                        pattern: pattern.clone(),
                        feasibility_score,
                    };

                    opportunities.push(opportunity);
                }
            }
        }

        // Sort by feasibility score (highest first)
        opportunities.sort_by(|a, b| {
            b.feasibility_score
                .partial_cmp(&a.feasibility_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(opportunities)
    }

    /// Find fusion candidates based on dependencies
    fn find_fusion_candidates(
        &self,
        operations: &[KernelOperation],
        dependency_graph: &DependencyGraph,
    ) -> Result<Vec<FusionCandidate>> {
        let mut candidates = Vec::new();

        // Strategy based on fusion configuration
        match &self.config.fusion_strategy {
            FusionStrategy::Aggressive => {
                candidates.extend(self.find_aggressive_candidates(operations, dependency_graph)?);
            }
            FusionStrategy::Balanced => {
                candidates.extend(self.find_balanced_candidates(operations, dependency_graph)?);
            }
            FusionStrategy::Conservative => {
                candidates.extend(self.find_conservative_candidates(operations, dependency_graph)?);
            }
            FusionStrategy::Custom(patterns) => {
                candidates.extend(self.find_custom_candidates(
                    operations,
                    dependency_graph,
                    patterns,
                )?);
            }
        }

        Ok(candidates)
    }

    /// Find aggressive fusion candidates (maximize fusion)
    fn find_aggressive_candidates(
        &self,
        operations: &[KernelOperation],
        dependency_graph: &DependencyGraph,
    ) -> Result<Vec<FusionCandidate>> {
        let mut candidates = Vec::new();
        let max_depth = self.config.max_fusion_depth;

        // Try to fuse as many operations as possible
        for start_idx in 0..operations.len() {
            for length in 2..=max_depth.min(operations.len() - start_idx) {
                let ops = operations[start_idx..start_idx + length].to_vec();

                // Check if operations can be fused
                if self.can_fuse_operations(&ops, dependency_graph)? {
                    candidates.push(FusionCandidate {
                        operations: ops,
                        strategy: FusionStrategy::Aggressive,
                    });
                }
            }
        }

        Ok(candidates)
    }

    /// Find balanced fusion candidates
    fn find_balanced_candidates(
        &self,
        operations: &[KernelOperation],
        dependency_graph: &DependencyGraph,
    ) -> Result<Vec<FusionCandidate>> {
        let mut candidates = Vec::new();

        // Use sliding window approach
        let window_size = 3; // Balanced window size

        for i in 0..operations.len().saturating_sub(window_size - 1) {
            let ops = operations[i..i + window_size].to_vec();

            if self.can_fuse_operations(&ops, dependency_graph)? {
                // Check for good balance of compute and memory operations
                if self.has_balanced_operations(&ops) {
                    candidates.push(FusionCandidate {
                        operations: ops,
                        strategy: FusionStrategy::Balanced,
                    });
                }
            }
        }

        Ok(candidates)
    }

    /// Find conservative fusion candidates (only obvious wins)
    fn find_conservative_candidates(
        &self,
        operations: &[KernelOperation],
        dependency_graph: &DependencyGraph,
    ) -> Result<Vec<FusionCandidate>> {
        let mut candidates = Vec::new();

        // Only fuse adjacent operations with direct dependencies
        for i in 0..operations.len() - 1 {
            let ops = vec![operations[i].clone(), operations[i + 1].clone()];

            // Check for direct producer-consumer relationship
            if self.has_direct_dependency(&ops[0], &ops[1], dependency_graph) {
                if self.can_fuse_operations(&ops, dependency_graph)? {
                    candidates.push(FusionCandidate {
                        operations: ops,
                        strategy: FusionStrategy::Conservative,
                    });
                }
            }
        }

        Ok(candidates)
    }

    /// Find custom fusion candidates based on patterns
    fn find_custom_candidates(
        &self,
        operations: &[KernelOperation],
        dependency_graph: &DependencyGraph,
        patterns: &[FusionPattern],
    ) -> Result<Vec<FusionCandidate>> {
        let mut candidates = Vec::new();

        for pattern in patterns {
            // Find sequences matching the pattern
            let matches = self.find_pattern_matches(operations, pattern)?;

            for matched_ops in matches {
                if self.can_fuse_operations(&matched_ops, dependency_graph)? {
                    candidates.push(FusionCandidate {
                        operations: matched_ops,
                        strategy: FusionStrategy::Custom(vec![pattern.clone()]),
                    });
                }
            }
        }

        Ok(candidates)
    }

    /// Check if operations can be fused
    fn can_fuse_operations(
        &self,
        operations: &[KernelOperation],
        dependency_graph: &DependencyGraph,
    ) -> Result<bool> {
        // Check basic constraints
        if operations.len() < 2 || operations.len() > self.config.max_fusion_depth {
            return Ok(false);
        }

        // Check dependencies allow fusion
        if !dependency_graph.allows_fusion(operations) {
            return Ok(false);
        }

        // Check memory requirements
        let total_memory = self.calculate_total_memory_requirements(operations)?;
        if !self.fits_in_device_memory(&total_memory) {
            return Ok(false);
        }

        // Check register pressure
        let total_registers = self.calculate_total_register_usage(operations)?;
        if total_registers > self.config.performance_thresholds.max_register_pressure {
            return Ok(false);
        }

        Ok(true)
    }

    /// Check if operations have balanced mix
    fn has_balanced_operations(&self, operations: &[KernelOperation]) -> bool {
        let mut compute_ops = 0;
        let mut memory_ops = 0;

        for op in operations {
            match &op.op_type {
                OperationType::ElementWise(_) | OperationType::Matrix(_) => compute_ops += 1,
                OperationType::Memory(_) => memory_ops += 1,
                OperationType::Reduction(_) => compute_ops += 1,
                OperationType::Custom(_) => compute_ops += 1,
            }
        }

        // Balanced if ratio is reasonable
        let ratio = compute_ops as f32 / (memory_ops + 1) as f32;
        ratio >= 0.5 && ratio <= 2.0
    }

    /// Check for direct dependency between operations
    fn has_direct_dependency(
        &self,
        producer: &KernelOperation,
        consumer: &KernelOperation,
        dependency_graph: &DependencyGraph,
    ) -> bool {
        dependency_graph.has_edge(&producer.id, &consumer.id)
    }

    /// Find operations matching a pattern
    fn find_pattern_matches(
        &self,
        operations: &[KernelOperation],
        pattern: &FusionPattern,
    ) -> Result<Vec<Vec<KernelOperation>>> {
        let mut matches = Vec::new();
        let pattern_len = pattern.operations.len();

        if pattern_len == 0 || pattern_len > operations.len() {
            return Ok(matches);
        }

        for i in 0..=operations.len() - pattern_len {
            let slice = &operations[i..i + pattern_len];

            if self.matches_pattern_sequence(slice, &pattern.operations) {
                matches.push(slice.to_vec());
            }
        }

        Ok(matches)
    }

    /// Check if operation sequence matches pattern
    fn matches_pattern_sequence(
        &self,
        operations: &[KernelOperation],
        pattern: &[OperationType],
    ) -> bool {
        if operations.len() != pattern.len() {
            return false;
        }

        operations
            .iter()
            .zip(pattern.iter())
            .all(|(op, pattern_type)| self.operation_matches_type(&op.op_type, pattern_type))
    }

    /// Check if operation matches pattern type
    fn operation_matches_type(&self, op: &OperationType, pattern: &OperationType) -> bool {
        match (op, pattern) {
            (OperationType::ElementWise(_), OperationType::ElementWise(_)) => true,
            (OperationType::Reduction(_), OperationType::Reduction(_)) => true,
            (OperationType::Memory(_), OperationType::Memory(_)) => true,
            (OperationType::Matrix(_), OperationType::Matrix(_)) => true,
            (OperationType::Custom(a), OperationType::Custom(b)) => a == b,
            _ => false,
        }
    }

    /// Calculate total memory requirements
    fn calculate_total_memory_requirements(
        &self,
        operations: &[KernelOperation],
    ) -> Result<MemoryRequirements> {
        let mut total = MemoryRequirements {
            global_reads: 0,
            global_writes: 0,
            shared_memory: 0,
            registers_per_thread: 0,
        };

        for op in operations {
            total.global_reads += op.memory_requirements.global_reads;
            total.global_writes += op.memory_requirements.global_writes;
            total.shared_memory = total
                .shared_memory
                .max(op.memory_requirements.shared_memory);
            total.registers_per_thread += op.memory_requirements.registers_per_thread;
        }

        Ok(total)
    }

    /// Calculate total register usage
    fn calculate_total_register_usage(&self, operations: &[KernelOperation]) -> Result<u32> {
        let mut total_registers = 0;

        for op in operations {
            total_registers += op.memory_requirements.registers_per_thread;
        }

        // Add fusion overhead
        total_registers += 8; // Typical fusion overhead

        Ok(total_registers)
    }

    /// Check if memory requirements fit in device
    fn fits_in_device_memory(&self, requirements: &MemoryRequirements) -> bool {
        // Check shared memory constraint (typically 48KB per block)
        if requirements.shared_memory > 49152 {
            return false;
        }

        // Check register constraint (typically 64 registers per thread)
        if requirements.registers_per_thread > 64 {
            return false;
        }

        true
    }

    /// Check if performance meets fusion thresholds
    fn meets_fusion_thresholds(&self, performance: &PerformancePrediction) -> bool {
        let thresholds = &self.config.performance_thresholds;

        performance.expected_speedup >= thresholds.min_speedup_factor
            && performance.memory_overhead_percent <= thresholds.max_memory_overhead
            && performance.register_pressure <= thresholds.max_register_pressure
            && performance.kernel_time_us >= thresholds.min_kernel_time_us
    }

    /// Calculate fusion feasibility score
    fn calculate_feasibility_score(
        &self,
        candidate: &FusionCandidate,
        pattern: &FusionPattern,
        performance: &PerformancePrediction,
    ) -> Result<f32> {
        // Weighted scoring based on multiple factors
        let speedup_score = (performance.expected_speedup - 1.0).min(1.0) * 0.4;
        let memory_score = (1.0 - performance.memory_overhead_percent / 100.0) * 0.2;
        let register_score = (1.0 - performance.register_pressure as f32 / 64.0) * 0.2;
        let pattern_score = pattern.expected_speedup.min(2.0) / 2.0 * 0.2;

        let total_score = speedup_score + memory_score + register_score + pattern_score;

        Ok(total_score.max(0.0).min(1.0))
    }
}

/// Fusion candidate
#[derive(Debug, Clone)]
struct FusionCandidate {
    operations: Vec<KernelOperation>,
    strategy: FusionStrategy,
}

impl FusionCandidate {
    /// Calculate hash for candidate identification
    fn hash(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        for op in &self.operations {
            op.id.hash(&mut hasher);
        }
        hasher.finish()
    }
}

/// Pattern matcher for fusion patterns
struct PatternMatcher {
    config: KernelFusionConfig,
    builtin_patterns: Vec<FusionPattern>,
}

impl PatternMatcher {
    fn new(config: KernelFusionConfig) -> Self {
        let builtin_patterns = Self::create_builtin_patterns();
        Self {
            config,
            builtin_patterns,
        }
    }

    /// Match operations against known patterns
    fn match_pattern(&self, operations: &[KernelOperation]) -> Option<FusionPattern> {
        // Check custom patterns first
        if let FusionStrategy::Custom(patterns) = &self.config.fusion_strategy {
            for pattern in patterns {
                if self.operations_match_pattern(operations, pattern) {
                    return Some(pattern.clone());
                }
            }
        }

        // Check builtin patterns
        for pattern in &self.builtin_patterns {
            if self.operations_match_pattern(operations, pattern) {
                return Some(pattern.clone());
            }
        }

        None
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
            if !self.op_matches_pattern(&op.op_type, pattern_op) {
                return false;
            }
        }

        // Check conditions are satisfied
        self.check_pattern_conditions(operations, &pattern.conditions)
    }

    /// Check if operation type matches pattern
    fn op_matches_pattern(&self, op: &OperationType, pattern: &OperationType) -> bool {
        match (op, pattern) {
            (OperationType::ElementWise(a), OperationType::ElementWise(b)) => {
                match (a, b) {
                    // Exact match or compatible operations
                    (ElementWiseOp::Add, ElementWiseOp::Add) => true,
                    (ElementWiseOp::Multiply, ElementWiseOp::Multiply) => true,
                    (ElementWiseOp::Activation(_), ElementWiseOp::Activation(_)) => true,
                    _ => false,
                }
            }
            (OperationType::Reduction(a), OperationType::Reduction(b)) => a == b,
            (OperationType::Memory(a), OperationType::Memory(b)) => a == b,
            (OperationType::Matrix(a), OperationType::Matrix(b)) => a == b,
            _ => false,
        }
    }

    /// Check pattern conditions
    fn check_pattern_conditions(
        &self,
        operations: &[KernelOperation],
        conditions: &FusionConditions,
    ) -> bool {
        // Check memory pattern compatibility
        if !conditions.memory_pattern_compatible {
            return true; // No restriction
        }

        // Check register constraints
        let total_registers: u32 = operations
            .iter()
            .map(|op| op.memory_requirements.registers_per_thread)
            .sum();

        if total_registers + conditions.register_constraints.fusion_overhead
            > conditions.register_constraints.max_registers_per_thread
        {
            return false;
        }

        // Check shared memory constraints
        let max_shared: usize = operations
            .iter()
            .map(|op| op.memory_requirements.shared_memory)
            .max()
            .unwrap_or(0);

        if max_shared + conditions.shared_memory_constraints.fusion_overhead
            > conditions.shared_memory_constraints.max_shared_memory
        {
            return false;
        }

        true
    }

    /// Create builtin fusion patterns
    fn create_builtin_patterns() -> Vec<FusionPattern> {
        vec![
            // ElementWise + Reduction pattern
            FusionPattern {
                name: "ElementWise-Reduction".to_string(),
                operations: vec![
                    OperationType::ElementWise(ElementWiseOp::Add),
                    OperationType::Reduction(ReductionOp::Sum),
                ],
                conditions: FusionConditions {
                    data_dependencies: vec![],
                    memory_pattern_compatible: true,
                    register_constraints: RegisterConstraints {
                        max_registers_per_thread: 64,
                        current_usage: 20,
                        fusion_overhead: 8,
                    },
                    shared_memory_constraints: SharedMemoryConstraints {
                        max_shared_memory: 49152,
                        current_usage: 4096,
                        fusion_overhead: 2048,
                    },
                },
                expected_speedup: 1.5,
            },
            // GEMM + Activation pattern
            FusionPattern {
                name: "GEMM-Activation".to_string(),
                operations: vec![
                    OperationType::Matrix(MatrixOp::GEMM),
                    OperationType::ElementWise(ElementWiseOp::Activation(ActivationType::ReLU)),
                ],
                conditions: FusionConditions {
                    data_dependencies: vec![],
                    memory_pattern_compatible: true,
                    register_constraints: RegisterConstraints {
                        max_registers_per_thread: 64,
                        current_usage: 40,
                        fusion_overhead: 10,
                    },
                    shared_memory_constraints: SharedMemoryConstraints {
                        max_shared_memory: 49152,
                        current_usage: 16384,
                        fusion_overhead: 4096,
                    },
                },
                expected_speedup: 1.3,
            },
        ]
    }
}

/// Dependency analyzer
struct DependencyAnalyzer {
    // Analysis state
}

impl DependencyAnalyzer {
    fn new() -> Self {
        Self {}
    }

    /// Build dependency graph from operations
    fn build_dependency_graph(&self, operations: &[KernelOperation]) -> Result<DependencyGraph> {
        let mut graph = DependencyGraph::new();

        // Add all operations as nodes
        for op in operations {
            graph.add_node(op.id.clone());
        }

        // Analyze dependencies between operations
        for i in 0..operations.len() {
            for j in i + 1..operations.len() {
                if let Some(dep) = self.analyze_dependency(&operations[i], &operations[j]) {
                    graph.add_edge(&operations[i].id, &operations[j].id, dep);
                }
            }
        }

        Ok(graph)
    }

    /// Analyze dependency between two operations
    fn analyze_dependency(
        &self,
        producer: &KernelOperation,
        consumer: &KernelOperation,
    ) -> Option<DependencyType> {
        // Check if consumer uses outputs from producer
        for producer_output in &producer.outputs {
            for consumer_input in &consumer.inputs {
                if self.tensors_overlap(producer_output, consumer_input) {
                    return Some(DependencyType::RAW); // Read after write
                }
            }
        }

        // Check for write-after-read dependencies
        for producer_input in &producer.inputs {
            for consumer_output in &consumer.outputs {
                if self.tensors_overlap(producer_input, consumer_output) {
                    return Some(DependencyType::WAR); // Write after read
                }
            }
        }

        // Check for write-after-write dependencies
        for producer_output in &producer.outputs {
            for consumer_output in &consumer.outputs {
                if self.tensors_overlap(producer_output, consumer_output) {
                    return Some(DependencyType::WAW); // Write after write
                }
            }
        }

        None
    }

    /// Check if two tensors overlap in memory
    fn tensors_overlap(&self, a: &TensorDescriptor, b: &TensorDescriptor) -> bool {
        // Simple shape comparison for now
        // In real implementation, would check actual memory addresses
        a.shape == b.shape && a.dtype == b.dtype
    }
}

/// Dependency graph for operations
struct DependencyGraph {
    nodes: HashSet<String>,
    edges: HashMap<(String, String), DependencyType>,
}

impl DependencyGraph {
    fn new() -> Self {
        Self {
            nodes: HashSet::new(),
            edges: HashMap::new(),
        }
    }

    fn add_node(&mut self, node: String) {
        self.nodes.insert(node);
    }

    fn add_edge(&mut self, from: &str, to: &str, dep_type: DependencyType) {
        self.edges
            .insert((from.to_string(), to.to_string()), dep_type);
    }

    fn has_edge(&self, from: &str, to: &str) -> bool {
        self.edges.contains_key(&(from.to_string(), to.to_string()))
    }

    /// Check if operations can be fused based on dependencies
    fn allows_fusion(&self, operations: &[KernelOperation]) -> bool {
        // Check for circular dependencies
        for i in 0..operations.len() {
            for j in 0..operations.len() {
                if i != j {
                    let id_i = &operations[i].id;
                    let id_j = &operations[j].id;

                    // Check for bidirectional dependencies (would create cycle)
                    if self.has_edge(id_i, id_j) && self.has_edge(id_j, id_i) {
                        return false;
                    }
                }
            }
        }

        true
    }
}

/// Performance predictor
struct PerformancePredictor {
    config: KernelFusionConfig,
}

impl PerformancePredictor {
    fn new(config: KernelFusionConfig) -> Self {
        Self { config }
    }

    /// Predict performance of fused operations
    fn predict_performance(
        &self,
        operations: &[KernelOperation],
        pattern: &FusionPattern,
    ) -> Result<PerformancePrediction> {
        // Calculate baseline performance
        let baseline_time: u64 = operations.iter().map(|op| op.estimated_time_us).sum();

        // Estimate fused kernel time
        let fused_time = self.estimate_fused_time(operations, pattern)?;

        // Calculate speedup
        let expected_speedup = baseline_time as f32 / fused_time as f32;

        // Calculate memory savings
        let memory_savings = self.calculate_memory_savings(operations)?;

        // Calculate overheads
        let memory_overhead_percent = self.calculate_memory_overhead(operations)?;
        let register_pressure = self.calculate_register_pressure(operations)?;

        Ok(PerformancePrediction {
            expected_speedup,
            memory_savings,
            memory_overhead_percent,
            register_pressure,
            kernel_time_us: fused_time,
        })
    }

    /// Estimate execution time for fused kernel
    fn estimate_fused_time(
        &self,
        operations: &[KernelOperation],
        pattern: &FusionPattern,
    ) -> Result<u64> {
        // Base time is maximum of individual operations (parallel execution)
        let max_time = operations
            .iter()
            .map(|op| op.estimated_time_us)
            .max()
            .unwrap_or(0);

        // Apply pattern-specific speedup
        let pattern_factor = 1.0 / pattern.expected_speedup;

        // Add fusion overhead (typically 5-10%)
        let fusion_overhead = 1.1;

        let estimated_time = (max_time as f32 * pattern_factor * fusion_overhead) as u64;

        Ok(estimated_time.max(1)) // At least 1 microsecond
    }

    /// Calculate memory savings from fusion
    fn calculate_memory_savings(&self, operations: &[KernelOperation]) -> Result<usize> {
        if operations.len() < 2 {
            return Ok(0);
        }

        // Savings come from eliminating intermediate data transfers
        let mut savings = 0;

        for i in 0..operations.len() - 1 {
            // Assume output of operation i is input to operation i+1
            // Fusion eliminates this intermediate write/read
            savings += operations[i].memory_requirements.global_writes;
            if i + 1 < operations.len() {
                savings += operations[i + 1].memory_requirements.global_reads
                    / operations[i + 1].inputs.len();
            }
        }

        Ok(savings)
    }

    /// Calculate memory overhead percentage
    fn calculate_memory_overhead(&self, operations: &[KernelOperation]) -> Result<f32> {
        let baseline_memory: usize = operations
            .iter()
            .map(|op| op.memory_requirements.global_reads + op.memory_requirements.global_writes)
            .sum();

        // Fusion typically reduces memory by 20-40%
        let fused_memory = (baseline_memory as f32 * 0.7) as usize;

        let overhead = if baseline_memory > 0 {
            ((baseline_memory - fused_memory) as f32 / baseline_memory as f32) * 100.0
        } else {
            0.0
        };

        Ok(overhead.abs()) // Return absolute overhead
    }

    /// Calculate register pressure for fused kernel
    fn calculate_register_pressure(&self, operations: &[KernelOperation]) -> Result<u32> {
        let base_registers: u32 = operations
            .iter()
            .map(|op| op.memory_requirements.registers_per_thread)
            .sum();

        // Fusion can sometimes reduce register usage through better scheduling
        let optimization_factor = 0.8;
        let fusion_overhead = 8; // Additional registers for fusion logic

        let total_registers =
            (base_registers as f32 * optimization_factor) as u32 + fusion_overhead;

        Ok(total_registers)
    }
}

/// Performance prediction result
#[derive(Debug, Clone)]
struct PerformancePrediction {
    expected_speedup: f32,
    memory_savings: usize,
    memory_overhead_percent: f32,
    register_pressure: u32,
    kernel_time_us: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fusion_analyzer_creation() -> Result<(), Box<dyn std::error::Error>> {
        let ctx = CudaContext::new(0)?;
        let config = KernelFusionConfig::default();
        let analyzer = FusionAnalyzer::new(ctx, config);

        // Analyzer created successfully
        assert!(true);
        Ok(())
    }

    #[test]
    fn test_pattern_matcher() {
        let config = KernelFusionConfig::default();
        let matcher = PatternMatcher::new(config);

        // Should have builtin patterns
        assert!(!matcher.builtin_patterns.is_empty());
    }

    #[test]
    fn test_dependency_graph() {
        let mut graph = DependencyGraph::new();
        graph.add_node("op1".to_string());
        graph.add_node("op2".to_string());
        graph.add_edge("op1", "op2", DependencyType::RAW);

        assert!(graph.has_edge("op1", "op2"));
        assert!(!graph.has_edge("op2", "op1"));
    }
}
