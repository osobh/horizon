//! Data types for self-assessment functionality

use super::config::SelfAssessmentConfig;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Self-modification record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfModification {
    /// Modification ID
    pub id: String,
    /// Generation when applied
    pub generation: u32,
    /// Parent agent ID
    pub parent_id: String,
    /// Child agent ID
    pub child_id: String,
    /// Type of modification
    pub modification_type: ModificationType,
    /// Description of the change
    pub description: String,
    /// Performance before modification
    pub performance_before: f64,
    /// Performance after modification
    pub performance_after: Option<f64>,
    /// Success indicator
    pub successful: Option<bool>,
}

/// Types of self-modifications
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModificationType {
    /// Tool enhancement
    ToolEnhancement,
    /// Workflow improvement
    WorkflowImprovement,
    /// Parameter adjustment
    ParameterAdjustment,
    /// Architecture change
    ArchitectureChange,
    /// Pattern application
    PatternApplication(String),
    /// Random mutation
    RandomMutation,
}

/// Performance benchmark results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResults {
    /// Benchmark name
    pub name: String,
    /// Tasks attempted
    pub tasks_attempted: u32,
    /// Tasks succeeded
    pub tasks_succeeded: u32,
    /// Average completion time
    pub avg_completion_time: f64,
    /// Code quality score
    pub code_quality_score: f64,
    /// Timestamp
    pub timestamp: u64,
}

/// Self-improvement capability assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImprovementCapability {
    /// Self-modification success rate
    pub modification_success_rate: f64,
    /// Average performance gain per modification
    pub avg_performance_gain: f64,
    /// Discovery rate of beneficial patterns
    pub beneficial_pattern_discovery_rate: f64,
    /// Exploration effectiveness
    pub exploration_effectiveness: f64,
    /// Exploitation effectiveness
    pub exploitation_effectiveness: f64,
}

/// Archive exploration statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplorationStats {
    /// Total agents in archive
    pub archive_size: usize,
    /// Number of unique lineages
    pub unique_lineages: usize,
    /// Diversity score
    pub diversity_score: f64,
    /// Stepping stone effectiveness
    pub stepping_stone_effectiveness: f64,
    /// Dead-end ratio
    pub dead_end_ratio: f64,
}

/// Comprehensive assessment report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssessmentReport {
    /// Report ID
    pub assessment_id: String,
    /// Generation of assessment
    pub generation: u32,
    /// Overall improvement capability
    pub improvement_capability: ImprovementCapability,
    /// Exploration statistics
    pub exploration_stats: ExplorationStats,
    /// Recent benchmark performance
    pub recent_performance: Vec<BenchmarkResults>,
    /// Top performing modifications
    pub top_modifications: Vec<SelfModification>,
    /// Recommendations for improvement
    pub recommendations: Vec<String>,
    /// Timestamp of assessment
    pub timestamp: u64,
    /// Assessment score (0.0 to 1.0)
    pub assessment_score: f64,
}

/// Pattern effectiveness metrics
#[derive(Debug, Clone)]
pub(crate) struct PatternEffectiveness {
    /// Pattern ID
    pub pattern_id: String,
    /// Total applications
    pub total_applications: u32,
    /// Successful applications
    pub successful_applications: u32,
    /// Average fitness improvement
    pub avg_fitness_improvement: f64,
    /// Last assessment generation
    pub last_assessed: u32,
}

/// Node in the lineage graph
#[derive(Debug, Clone)]
pub(crate) struct LineageNode {
    /// Agent ID
    pub agent_id: String,
    /// Generation created
    pub generation: u32,
    /// Performance score
    pub performance: f64,
    /// Modification that created this agent
    pub modification_type: ModificationType,
    /// Whether this lineage led to improvements
    pub is_stepping_stone: bool,
}

/// Lineage graph for tracking agent evolution
#[derive(Debug, Clone)]
pub(crate) struct LineageGraph {
    /// Nodes: agent_id -> agent metadata
    pub nodes: HashMap<String, LineageNode>,
    /// Edges: parent_id -> [child_ids]
    pub edges: HashMap<String, Vec<String>>,
}

/// Performance metrics for self-assessment
#[derive(Debug, Clone, Default)]
pub struct PerformanceMetrics {
    /// Fitness improvement over time
    pub fitness_improvement: f64,
    /// Rate of convergence
    pub convergence_rate: f64,
    /// Success rate of modifications
    pub modification_success_rate: f64,
    /// Diversity score
    pub diversity_score: f64,
    /// Resource efficiency
    pub resource_efficiency: f64,
}

/// Assessment criteria for evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AssessmentCriteria {
    /// Fitness improvement over generations
    FitnessImprovement,
    /// Ability to maintain diversity
    DiversityMaintenance,
    /// Speed of convergence to optimal solutions
    ConvergenceSpeed,
    /// Efficient use of computational resources
    ResourceEfficiency,
    /// Adaptability to new environments
    AdaptabilityScore,
}

/// DGM Self-Assessment system for evaluating agent evolution and self-modification capabilities
///
/// This struct implements the core self-assessment functionality described in the Darwin Gödel Machine paper.
/// It tracks agent performance, evaluates modifications, and generates assessment reports.
#[derive(Debug)]
pub struct DgmSelfAssessment {
    /// Configuration for self-assessment
    config: SelfAssessmentConfig,
    /// Performance metrics over generations
    performance_history: Vec<(u32, f64)>, // (generation, avg_fitness)
    /// Tracked modifications
    modifications: Vec<SelfModification>,
    /// Current generation being assessed
    current_generation: u32,
    /// Performance metrics
    performance_metrics: PerformanceMetrics,
}

impl DgmSelfAssessment {
    /// Create a new DGM self-assessment instance
    ///
    /// # Arguments
    /// * `config` - Configuration for the assessment system
    ///
    /// # Returns
    /// A new DgmSelfAssessment instance
    pub fn new(config: SelfAssessmentConfig) -> Self {
        Self {
            config,
            performance_history: Vec::new(),
            modifications: Vec::new(),
            current_generation: 0,
            performance_metrics: PerformanceMetrics::default(),
        }
    }

    /// Evaluate a population of agents and assess their performance
    ///
    /// # Arguments
    /// * `agents` - The agents to evaluate
    ///
    /// # Returns
    /// Assessment results for the agent population
    pub fn evaluate_agents<T>(&mut self, agents: &[T]) -> AssessmentReport
    where
        T: crate::traits::Evolvable,
    {
        // Calculate population statistics
        let agent_count = agents.len();
        let estimated_avg_fitness = if agent_count > 0 {
            // Since we can't async evaluate here, use a placeholder approach
            // In real implementation, this would be passed in or calculated differently
            0.5 // Placeholder average fitness
        } else {
            0.0
        };

        // Track this generation's performance
        self.current_generation += 1;
        self.track_generation_performance(self.current_generation, estimated_avg_fitness);

        // Create assessment based on population characteristics
        let report = self.create_assessment_report(self.current_generation);

        tracing::info!(
            "Evaluated {} agents for generation {}, avg fitness: {:.3}",
            agent_count,
            self.current_generation,
            estimated_avg_fitness
        );

        report
    }

    /// Track performance for a generation
    ///
    /// # Arguments
    /// * `generation` - The generation number
    /// * `avg_fitness` - Average fitness for this generation
    pub fn track_generation_performance(&mut self, generation: u32, avg_fitness: f64) {
        self.current_generation = generation;
        self.performance_history.push((generation, avg_fitness));

        // Update performance metrics
        self.update_performance_metrics();
    }

    /// Get current performance metrics
    ///
    /// # Returns
    /// Current performance metrics
    pub fn get_performance_metrics(&self) -> &PerformanceMetrics {
        &self.performance_metrics
    }

    /// Generate a comprehensive assessment report
    ///
    /// # Arguments
    /// * `generation` - The generation to generate report for
    ///
    /// # Returns
    /// A comprehensive assessment report
    pub fn generate_assessment_report(&mut self, generation: u32) -> AssessmentReport {
        self.current_generation = generation;
        self.update_performance_metrics();
        self.create_assessment_report(generation)
    }

    /// Track a self-modification
    ///
    /// # Arguments
    /// * `modification` - The modification to track
    pub fn track_modification(&mut self, modification: SelfModification) {
        self.modifications.push(modification);
        self.update_performance_metrics();
    }

    /// Get modification statistics
    ///
    /// # Returns
    /// Statistics about tracked modifications
    pub fn get_modification_statistics(&self) -> Vec<String> {
        let mut stats = Vec::new();

        if self.modifications.is_empty() {
            stats.push("No modifications tracked yet".to_string());
            return stats;
        }

        let total = self.modifications.len();
        let successful = self
            .modifications
            .iter()
            .filter(|m| m.successful.unwrap_or(false))
            .count();
        let success_rate = successful as f64 / total as f64;

        stats.push(format!("Total modifications: {}", total));
        stats.push(format!(
            "Successful modifications: {} ({:.1}%)",
            successful,
            success_rate * 100.0
        ));
        stats.push(format!("Unique lineages: {}", self.count_unique_lineages()));

        stats
    }

    /// Perform comprehensive assessment for a generation
    ///
    /// # Arguments
    /// * `generation` - The generation to assess
    ///
    /// # Returns
    /// A comprehensive assessment report
    pub async fn perform_assessment(
        &self,
        generation: u32,
    ) -> crate::error::EvolutionEngineResult<AssessmentReport> {
        Ok(self.create_assessment_report(generation))
    }

    /// Update pattern effectiveness tracking
    ///
    /// # Arguments
    /// * `pattern_id` - ID of the pattern
    /// * `success` - Whether the pattern was successful
    /// * `fitness_improvement` - The fitness improvement achieved
    /// * `generation` - The current generation
    pub fn update_pattern_effectiveness(
        &self,
        pattern_id: &str,
        success: bool,
        fitness_improvement: f64,
        generation: u32,
    ) {
        // Create a modification record for this pattern application
        let modification = SelfModification {
            id: format!("pattern_{}_{}", pattern_id, generation),
            generation,
            parent_id: format!("pattern_{}", pattern_id),
            child_id: format!("result_{}", generation),
            modification_type: ModificationType::PatternApplication(pattern_id.to_string()),
            description: format!(
                "Applied pattern {} with fitness improvement {:.4}",
                pattern_id, fitness_improvement
            ),
            performance_before: 0.0, // Would be filled from actual context
            performance_after: Some(fitness_improvement),
            successful: Some(success),
        };

        // Note: In a real implementation, this would need mutable access to track modifications
        // For now, this serves as documentation of the intended behavior
        tracing::info!(
            "Pattern {} effectiveness update: success={}, improvement={:.4}, generation={}",
            pattern_id,
            success,
            fitness_improvement,
            generation
        );
    }

    /// Record a benchmark result
    ///
    /// # Arguments
    /// * `benchmark` - The benchmark results to record
    pub fn record_benchmark(&self, benchmark: BenchmarkResults) {
        // Log the benchmark for tracking
        tracing::info!(
            "Recording benchmark: {} - attempted: {}, succeeded: {}, quality: {:.3}",
            benchmark.name,
            benchmark.tasks_attempted,
            benchmark.tasks_succeeded,
            benchmark.code_quality_score
        );

        // Note: In a mutable implementation, this would be stored in internal state
        // For now, we just log the information
    }

    /// Record a self-modification
    ///
    /// # Arguments
    /// * `modification` - The modification to record
    pub fn record_modification(&self, modification: SelfModification) {
        tracing::info!(
            "Recording modification: {} (Gen {}): {} -> {}",
            modification.id,
            modification.generation,
            modification.parent_id,
            modification.child_id
        );

        // Note: In a mutable implementation, this would be stored with track_modification
        // This method serves as a logging interface for external recording
    }

    /// Check if assessment should be performed for this generation
    ///
    /// # Arguments
    /// * `generation` - The current generation
    ///
    /// # Returns
    /// True if assessment should be performed
    pub fn should_assess(&self, generation: u32) -> bool {
        // Perform assessment every N generations based on config
        generation % self.config.assessment_interval == 0 && generation > 0
    }

    /// Get the current assessment state
    ///
    /// # Returns
    /// The current assessment report if available
    pub fn get_current_assessment(&self) -> Option<AssessmentReport> {
        if self.performance_history.is_empty() {
            return None;
        }

        Some(self.create_assessment_report(self.current_generation))
    }

    /// Update internal performance metrics based on performance history
    fn update_performance_metrics(&mut self) {
        if self.performance_history.len() < 2 {
            return;
        }

        let recent_performances: Vec<f64> = self
            .performance_history
            .iter()
            .rev()
            .take(10)
            .map(|(_, fitness)| *fitness)
            .collect();

        // Calculate fitness improvement
        let first = recent_performances.last().unwrap_or(&0.0);
        let last = recent_performances.first().unwrap_or(&0.0);
        self.performance_metrics.fitness_improvement = last - first;

        // Calculate convergence rate (change in fitness over last few generations)
        self.performance_metrics.convergence_rate = if recent_performances.len() > 1 {
            let sum_diff: f64 = recent_performances
                .windows(2)
                .map(|w| (w[0] - w[1]).abs())
                .sum();
            sum_diff / (recent_performances.len() - 1) as f64
        } else {
            0.0
        };

        // Calculate modification success rate
        if !self.modifications.is_empty() {
            let successful = self
                .modifications
                .iter()
                .filter(|m| m.successful.unwrap_or(false))
                .count();
            self.performance_metrics.modification_success_rate =
                successful as f64 / self.modifications.len() as f64;
        }

        // Calculate diversity score (variation in recent fitness values)
        if !recent_performances.is_empty() {
            let mean = recent_performances.iter().sum::<f64>() / recent_performances.len() as f64;
            let variance = recent_performances
                .iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f64>()
                / recent_performances.len() as f64;
            self.performance_metrics.diversity_score = variance.sqrt();
        }

        // Resource efficiency (placeholder - would be calculated from actual resource usage)
        self.performance_metrics.resource_efficiency = 0.8;
    }

    /// Create an assessment report for the given generation
    fn create_assessment_report(&self, generation: u32) -> AssessmentReport {
        use std::time::{SystemTime, UNIX_EPOCH};

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        AssessmentReport {
            assessment_id: format!("dgm_assessment_{}", generation),
            generation,
            improvement_capability: self.calculate_improvement_capability(),
            exploration_stats: self.calculate_exploration_stats(),
            recent_performance: self.get_recent_benchmarks(),
            top_modifications: self.get_top_modifications(),
            recommendations: self.generate_recommendations(),
            timestamp,
            assessment_score: self.calculate_overall_score(),
        }
    }

    /// Calculate improvement capability metrics
    fn calculate_improvement_capability(&self) -> ImprovementCapability {
        ImprovementCapability {
            modification_success_rate: self.performance_metrics.modification_success_rate,
            avg_performance_gain: self.performance_metrics.fitness_improvement / 10.0,
            beneficial_pattern_discovery_rate: 0.7, // Placeholder
            exploration_effectiveness: 0.8,         // Placeholder
            exploitation_effectiveness: 0.6,        // Placeholder
        }
    }

    /// Calculate exploration statistics
    fn calculate_exploration_stats(&self) -> ExplorationStats {
        ExplorationStats {
            archive_size: self.modifications.len(),
            unique_lineages: self.count_unique_lineages(),
            diversity_score: self.performance_metrics.diversity_score,
            stepping_stone_effectiveness: 0.75, // Placeholder
            dead_end_ratio: 0.2,                // Placeholder
        }
    }

    /// Count unique lineages from modifications
    fn count_unique_lineages(&self) -> usize {
        use std::collections::HashSet;
        let mut unique_parents: HashSet<String> = HashSet::new();
        for modification in &self.modifications {
            unique_parents.insert(modification.parent_id.clone());
        }
        unique_parents.len().max(1)
    }

    /// Get recent benchmark results
    fn get_recent_benchmarks(&self) -> Vec<BenchmarkResults> {
        // Return recent performance as benchmark results
        self.performance_history
            .iter()
            .rev()
            .take(5)
            .map(|(generation, fitness)| BenchmarkResults {
                name: format!("Generation {}", generation),
                tasks_attempted: 100, // Placeholder
                tasks_succeeded: (fitness * 100.0) as u32,
                avg_completion_time: 1.0,
                code_quality_score: *fitness,
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
            })
            .collect()
    }

    /// Get top performing modifications
    fn get_top_modifications(&self) -> Vec<SelfModification> {
        let mut modifications = self.modifications.clone();
        modifications.sort_by(|a, b| {
            b.performance_after
                .unwrap_or(0.0)
                .partial_cmp(&a.performance_after.unwrap_or(0.0))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        modifications.into_iter().take(5).collect()
    }

    /// Generate improvement recommendations
    fn generate_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();

        if self.performance_metrics.fitness_improvement < 0.01 {
            recommendations
                .push("Consider increasing exploration rate to discover new solutions".to_string());
        }

        if self.performance_metrics.modification_success_rate < 0.5 {
            recommendations
                .push("Review modification strategies - success rate is below optimal".to_string());
        }

        if self.performance_metrics.diversity_score < 0.1 {
            recommendations.push(
                "Population diversity is low - consider diversity preservation mechanisms"
                    .to_string(),
            );
        }

        if recommendations.is_empty() {
            recommendations
                .push("Performance is optimal - maintain current parameters".to_string());
        }

        recommendations
    }

    /// Calculate overall assessment score
    fn calculate_overall_score(&self) -> f64 {
        let improvement_score = (self.performance_metrics.fitness_improvement + 1.0) / 2.0;
        let success_score = self.performance_metrics.modification_success_rate;
        let diversity_score = (self.performance_metrics.diversity_score * 2.0).min(1.0);
        let efficiency_score = self.performance_metrics.resource_efficiency;

        (improvement_score + success_score + diversity_score + efficiency_score) / 4.0
    }
}
