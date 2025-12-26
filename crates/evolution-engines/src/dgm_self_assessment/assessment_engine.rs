//! Main self-assessment engine implementation

use super::config::*;
use super::types::*;
use crate::error::EvolutionEngineResult;
use parking_lot::RwLock;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use uuid::Uuid;

/// DGM self-assessment system (renamed to avoid conflict with existing module)
pub struct SelfAssessmentEngine {
    /// Engine ID
    pub id: String,
    /// Current generation
    pub current_generation: u32,
    /// Configuration
    config: SelfAssessmentConfig,
    /// Modification history
    modification_history: Arc<RwLock<VecDeque<SelfModification>>>,
    /// Benchmark results history
    benchmark_history: Arc<RwLock<VecDeque<BenchmarkResults>>>,
    /// Pattern effectiveness tracking
    pattern_effectiveness: Arc<RwLock<HashMap<String, PatternEffectiveness>>>,
    /// Lineage tracking
    lineage_graph: Arc<RwLock<LineageGraph>>,
    /// Current assessment report
    current_assessment: Arc<RwLock<Option<AssessmentReport>>>,
    /// Performance tracker
    pub performance_tracker: PerformanceTracker,
    /// Lineage tracker
    pub lineage_tracker: LineageTracker,
}

/// Performance tracking component
#[derive(Debug)]
pub struct PerformanceTracker {
    /// Current generation count
    pub generation_count: u32,
    /// Performance history
    pub performance_history: VecDeque<f64>,
    /// Metrics
    pub metrics: PerformanceMetrics,
}

/// Lineage tracking component  
#[derive(Debug)]
pub struct LineageTracker {
    /// Lineage graph
    pub lineage_graph: HashMap<String, LineageNode>,
    /// Performance trends
    pub performance_trends: Vec<f64>,
}

impl SelfAssessmentEngine {
    /// Create new self-assessment system
    pub fn new(config: SelfAssessmentConfig) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            current_generation: 0,
            config,
            modification_history: Arc::new(RwLock::new(VecDeque::new())),
            benchmark_history: Arc::new(RwLock::new(VecDeque::new())),
            pattern_effectiveness: Arc::new(RwLock::new(HashMap::new())),
            lineage_graph: Arc::new(RwLock::new(LineageGraph::new())),
            current_assessment: Arc::new(RwLock::new(None)),
            performance_tracker: PerformanceTracker::new(),
            lineage_tracker: LineageTracker::new(),
        }
    }

    /// Record a self-modification
    pub fn record_modification(&self, modification: SelfModification) {
        let mut history = self.modification_history.write();
        history.push_back(modification.clone());

        // Maintain window size
        while history.len() > self.config.history_window {
            history.pop_front();
        }

        // Update lineage graph
        self.lineage_graph.write().add_modification(&modification);
    }

    /// Record benchmark results
    pub fn record_benchmark(&self, results: BenchmarkResults) {
        let mut history = self.benchmark_history.write();
        history.push_back(results);

        // Maintain window size
        while history.len() > self.config.history_window {
            history.pop_front();
        }
    }

    /// Update pattern effectiveness
    pub fn update_pattern_effectiveness(
        &self,
        pattern_id: &str,
        successful: bool,
        fitness_improvement: f64,
        generation: u32,
    ) {
        let mut effectiveness = self.pattern_effectiveness.write();

        let entry = effectiveness
            .entry(pattern_id.to_string())
            .or_insert(PatternEffectiveness {
                pattern_id: pattern_id.to_string(),
                total_applications: 0,
                successful_applications: 0,
                avg_fitness_improvement: 0.0,
                last_assessed: generation,
            });

        entry.total_applications += 1;
        if successful {
            entry.successful_applications += 1;
        }

        // Update running average
        let n = entry.total_applications as f64;
        entry.avg_fitness_improvement =
            (entry.avg_fitness_improvement * (n - 1.0) + fitness_improvement) / n;
        entry.last_assessed = generation;
    }

    /// Perform comprehensive self-assessment
    pub async fn perform_assessment(
        &mut self,
        generation: u32,
    ) -> EvolutionEngineResult<AssessmentReport> {
        self.current_generation = generation;

        let improvement_capability = self.assess_improvement_capability();
        let exploration_stats = self.assess_exploration_stats();
        let recent_performance = self.get_recent_performance();
        let top_modifications = self.get_top_modifications();
        let recommendations =
            self.generate_recommendations(&improvement_capability, &exploration_stats);

        let report = AssessmentReport {
            assessment_id: Uuid::new_v4().to_string(),
            generation,
            improvement_capability,
            exploration_stats,
            recent_performance,
            top_modifications,
            recommendations,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            assessment_score: 0.7, // Default score
        };

        *self.current_assessment.write() = Some(report.clone());
        Ok(report)
    }

    /// Generate assessment report (simplified public method)
    pub async fn generate_assessment_report(&mut self) -> EvolutionEngineResult<AssessmentReport> {
        self.perform_assessment(self.current_generation).await
    }

    /// Assess improvement capability
    fn assess_improvement_capability(&self) -> ImprovementCapability {
        let history = self.modification_history.read();

        if history.is_empty() {
            return ImprovementCapability {
                modification_success_rate: 0.0,
                avg_performance_gain: 0.0,
                beneficial_pattern_discovery_rate: 0.0,
                exploration_effectiveness: 0.0,
                exploitation_effectiveness: 0.0,
            };
        }

        let total_modifications = history.len() as f64;
        let successful_modifications = history
            .iter()
            .filter(|m| m.successful.unwrap_or(false))
            .count() as f64;

        let performance_gains: Vec<f64> = history
            .iter()
            .filter_map(|m| {
                m.performance_after
                    .map(|after| after - m.performance_before)
            })
            .collect();

        let avg_performance_gain = if !performance_gains.is_empty() {
            performance_gains.iter().sum::<f64>() / performance_gains.len() as f64
        } else {
            0.0
        };

        let pattern_effectiveness = self.pattern_effectiveness.read();
        let beneficial_patterns = pattern_effectiveness
            .values()
            .filter(|p| p.avg_fitness_improvement > self.config.min_improvement_threshold)
            .count() as f64;
        let total_patterns = pattern_effectiveness.len().max(1) as f64;

        ImprovementCapability {
            modification_success_rate: successful_modifications / total_modifications,
            avg_performance_gain,
            beneficial_pattern_discovery_rate: beneficial_patterns / total_patterns,
            exploration_effectiveness: self.calculate_exploration_effectiveness(),
            exploitation_effectiveness: self.calculate_exploitation_effectiveness(),
        }
    }

    /// Assess exploration statistics
    fn assess_exploration_stats(&self) -> ExplorationStats {
        let lineage = self.lineage_graph.read();
        let unique_lineages = lineage.count_unique_lineages();
        let diversity_score = lineage.calculate_diversity_score();
        let stepping_stone_effectiveness = lineage.calculate_stepping_stone_effectiveness();
        let dead_end_ratio = lineage.calculate_dead_end_ratio();

        ExplorationStats {
            archive_size: lineage.nodes.len(),
            unique_lineages,
            diversity_score,
            stepping_stone_effectiveness,
            dead_end_ratio,
        }
    }

    /// Get recent performance data
    fn get_recent_performance(&self) -> Vec<BenchmarkResults> {
        let history = self.benchmark_history.read();
        history.iter().rev().take(10).cloned().collect()
    }

    /// Get top performing modifications
    fn get_top_modifications(&self) -> Vec<SelfModification> {
        let history = self.modification_history.read();
        let mut modifications: Vec<_> = history.iter().cloned().collect();

        modifications.sort_by(|a, b| {
            let gain_a = a
                .performance_after
                .map(|after| after - a.performance_before)
                .unwrap_or(0.0);
            let gain_b = b
                .performance_after
                .map(|after| after - b.performance_before)
                .unwrap_or(0.0);
            gain_b
                .partial_cmp(&gain_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        modifications.into_iter().take(5).collect()
    }

    /// Generate improvement recommendations
    fn generate_recommendations(
        &self,
        capability: &ImprovementCapability,
        exploration: &ExplorationStats,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        if capability.modification_success_rate < 0.5 {
            recommendations.push("Consider more conservative modification strategies".to_string());
        }

        if capability.avg_performance_gain < self.config.min_improvement_threshold {
            recommendations.push("Focus on higher-impact modifications".to_string());
        }

        if exploration.diversity_score < 0.5 {
            recommendations.push("Increase exploration to maintain diversity".to_string());
        }

        if exploration.dead_end_ratio > 0.7 {
            recommendations.push("Reduce dead-end exploration paths".to_string());
        }

        if capability.exploration_effectiveness < 0.4 {
            recommendations.push("Improve exploration strategies".to_string());
        }

        recommendations
    }

    /// Calculate exploration effectiveness
    fn calculate_exploration_effectiveness(&self) -> f64 {
        let lineage = self.lineage_graph.read();
        let total_nodes = lineage.nodes.len();
        if total_nodes == 0 {
            return 0.0;
        }

        let beneficial_nodes = lineage
            .nodes
            .values()
            .filter(|node| node.is_stepping_stone)
            .count();

        beneficial_nodes as f64 / total_nodes as f64
    }

    /// Calculate exploitation effectiveness
    fn calculate_exploitation_effectiveness(&self) -> f64 {
        let pattern_effectiveness = self.pattern_effectiveness.read();
        if pattern_effectiveness.is_empty() {
            return 0.0;
        }

        let effective_patterns = pattern_effectiveness
            .values()
            .filter(|p| p.successful_applications as f64 / p.total_applications.max(1) as f64 > 0.6)
            .count();

        effective_patterns as f64 / pattern_effectiveness.len() as f64
    }
}

impl PerformanceTracker {
    pub fn new() -> Self {
        Self {
            generation_count: 0,
            performance_history: VecDeque::new(),
            metrics: PerformanceMetrics::default(),
        }
    }
}

impl LineageTracker {
    pub fn new() -> Self {
        Self {
            lineage_graph: HashMap::new(),
            performance_trends: Vec::new(),
        }
    }
}

impl LineageGraph {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: HashMap::new(),
        }
    }

    pub fn add_modification(&mut self, modification: &SelfModification) {
        let child_node = LineageNode {
            agent_id: modification.child_id.clone(),
            generation: modification.generation,
            performance: modification
                .performance_after
                .unwrap_or(modification.performance_before),
            modification_type: modification.modification_type.clone(),
            is_stepping_stone: modification.successful.unwrap_or(false),
        };

        self.nodes.insert(modification.child_id.clone(), child_node);

        self.edges
            .entry(modification.parent_id.clone())
            .or_insert_with(Vec::new)
            .push(modification.child_id.clone());
    }

    pub fn count_unique_lineages(&self) -> usize {
        // Count root nodes (nodes with no incoming edges)
        let has_parent: std::collections::HashSet<_> = self.edges.values().flatten().collect();
        self.nodes
            .keys()
            .filter(|node_id| !has_parent.contains(node_id))
            .count()
    }

    pub fn calculate_diversity_score(&self) -> f64 {
        if self.nodes.is_empty() {
            return 0.0;
        }

        // Simple diversity based on performance variance
        let performances: Vec<f64> = self.nodes.values().map(|n| n.performance).collect();
        let mean = performances.iter().sum::<f64>() / performances.len() as f64;
        let variance = performances.iter().map(|p| (p - mean).powi(2)).sum::<f64>()
            / performances.len() as f64;

        variance.sqrt().min(1.0)
    }

    pub fn calculate_stepping_stone_effectiveness(&self) -> f64 {
        if self.nodes.is_empty() {
            return 0.0;
        }

        let stepping_stones = self.nodes.values().filter(|n| n.is_stepping_stone).count();

        stepping_stones as f64 / self.nodes.len() as f64
    }

    pub fn calculate_dead_end_ratio(&self) -> f64 {
        if self.nodes.is_empty() {
            return 0.0;
        }

        let dead_ends = self
            .nodes
            .keys()
            .filter(|node_id| !self.edges.contains_key(*node_id))
            .count();

        dead_ends as f64 / self.nodes.len() as f64
    }
}
