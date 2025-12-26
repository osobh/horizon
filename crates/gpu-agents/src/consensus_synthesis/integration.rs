//! Integration layer for Consensus and Synthesis modules
//!
//! Coordinates voting-based approval for synthesis operations

use crate::consensus::voting::GpuVoting;
use crate::evolution::engine_adapter::{ConsensusWeights, EvolutionEngineAdapter, PopulationStats};
use crate::knowledge::graph_adapter::{
    ConsensusOutcome, ConsensusPattern, KnowledgeGraphAdapter, SimilarPattern,
    SynthesisPerformanceMetrics,
};
use crate::synthesis::cross_crate_adapter::{SynthesisCrateAdapter, SynthesisMetrics};
use crate::synthesis::{AstNode, GpuSynthesisModule, SynthesisTask};
use anyhow::{anyhow, Context, Result};
use cudarc::driver::{CudaDevice, CudaSlice};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Configuration for consensus-synthesis integration
#[derive(Debug, Clone)]
pub struct IntegrationConfig {
    pub max_concurrent_tasks: usize,
    pub voting_timeout: Duration,
    pub min_voters: usize,
    pub retry_attempts: u32,
    pub conflict_resolution_strategy: ConflictStrategy,
}

impl Default for IntegrationConfig {
    fn default() -> Self {
        Self {
            max_concurrent_tasks: 100,
            voting_timeout: Duration::from_secs(10),
            min_voters: 3,
            retry_attempts: 3,
            conflict_resolution_strategy: ConflictStrategy::FirstWins,
        }
    }
}

/// Conflict resolution strategies
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConflictStrategy {
    FirstWins,
    HighestVoteWins,
    MergeIfPossible,
    RejectBoth,
}

/// Result of a consensus-synthesis workflow
#[derive(Debug, Clone)]
pub struct WorkflowResult {
    pub task_id: u64,
    pub consensus_achieved: bool,
    pub vote_percentage: f32,
    pub synthesis_result: Option<String>,
    pub execution_time: Duration,
    pub participating_nodes: Vec<u32>,
}

/// Task workflow status
#[derive(Debug, Clone, PartialEq)]
pub enum WorkflowStatus {
    Pending,
    VotingInProgress,
    ConsensusAchieved,
    ConsensusFailed,
    Executing,
    Completed,
    Failed,
}

/// Internal task tracking
struct TrackedTask {
    task: SynthesisTask,
    status: WorkflowStatus,
    votes: HashMap<u32, bool>, // node_id -> vote
    submitted_at: Instant,
    completed_at: Option<Instant>,
    result: Option<String>,
}

/// Main integration engine
pub struct ConsensusSynthesisEngine {
    device: Arc<CudaDevice>,
    config: IntegrationConfig,
    voting_engine: Option<GpuVoting>,
    synthesis_module: Option<GpuSynthesisModule>,
    // Cross-crate integration adapters
    synthesis_adapter: Option<SynthesisCrateAdapter>,
    evolution_adapter: Option<EvolutionEngineAdapter>,
    knowledge_adapter: Option<KnowledgeGraphAdapter>,
    tasks: Arc<Mutex<HashMap<u64, TrackedTask>>>,
    next_task_id: Arc<Mutex<u64>>,
    vote_buffer: Option<CudaSlice<u32>>,
    result_buffer: Option<CudaSlice<u8>>,
}

impl ConsensusSynthesisEngine {
    /// Create new integration engine
    pub fn new(device: Arc<CudaDevice>, config: IntegrationConfig) -> Result<Self> {
        // Initialize voting engine
        let voting_engine = GpuVoting::new(device.clone(), 1000)?;

        // Initialize synthesis module
        let synthesis_module = GpuSynthesisModule::new(device.clone(), 10000)?;

        // Allocate GPU buffers
        let vote_buffer = unsafe { device.alloc::<u32>(1000) }.ok();
        let result_buffer = unsafe { device.alloc::<u8>(1024 * 1024) }.ok(); // 1MB

        Ok(Self {
            device,
            config,
            voting_engine: Some(voting_engine),
            synthesis_module: Some(synthesis_module),
            // Initialize cross-crate adapters as None (will be initialized lazily)
            synthesis_adapter: None,
            evolution_adapter: None,
            knowledge_adapter: None,
            tasks: Arc::new(Mutex::new(HashMap::new())),
            next_task_id: Arc::new(Mutex::new(1)),
            vote_buffer,
            result_buffer,
        })
    }

    /// Submit a synthesis task for consensus approval
    pub fn submit_synthesis_task(&self, task: SynthesisTask) -> Result<u64> {
        let mut tasks = self.tasks.lock()?;
        let mut next_id = self.next_task_id.lock()?;

        let task_id = *next_id;
        *next_id += 1;

        let tracked = TrackedTask {
            task,
            status: WorkflowStatus::Pending,
            votes: HashMap::new(),
            submitted_at: Instant::now(),
            completed_at: None,
            result: None,
        };

        tasks.insert(task_id, tracked);

        Ok(task_id)
    }

    /// Collect votes from nodes for a task
    pub fn collect_votes(&self, task_id: u64, node_ids: &[u32]) -> Result<HashMap<u32, bool>> {
        let mut tasks = self.tasks.lock()?;

        let task = tasks
            .get_mut(&task_id)
            .ok_or_else(|| anyhow!("Task {} not found", task_id))?;

        if task.status != WorkflowStatus::Pending {
            return Err(anyhow!("Task {} is not in pending state", task_id));
        }

        task.status = WorkflowStatus::VotingInProgress;

        // Simulate voting (in real implementation, would use GPU voting)
        for &node_id in node_ids {
            // Simple voting logic: approve if task name hash is even
            let vote = task
                .task
                .pattern
                .value
                .as_ref()
                .map(|v| v.len() % 2 == 0)
                .unwrap_or(true);

            task.votes.insert(node_id, vote);
        }

        Ok(task.votes.clone())
    }

    /// Execute synthesis if consensus threshold is met
    pub fn execute_if_consensus(&self, task_id: u64, threshold: f32) -> Result<WorkflowResult> {
        let mut tasks = self.tasks.lock()?;

        let task = tasks
            .get_mut(&task_id)
            .ok_or_else(|| anyhow!("Task {} not found", task_id))?;

        // Calculate vote percentage
        let total_votes = task.votes.len() as f32;
        let positive_votes = task.votes.values().filter(|&&v| v).count() as f32;
        let vote_percentage = if total_votes > 0.0 {
            positive_votes / total_votes
        } else {
            0.0
        };

        let consensus_achieved = vote_percentage >= threshold;

        if consensus_achieved {
            task.status = WorkflowStatus::ConsensusAchieved;

            // Execute synthesis
            task.status = WorkflowStatus::Executing;

            // Simulate synthesis execution
            let result = format!(
                "fn {}() {{}}",
                task.task
                    .pattern
                    .value
                    .as_ref()
                    .unwrap_or(&"unnamed".to_string())
            );

            task.result = Some(result.clone());
            task.status = WorkflowStatus::Completed;
            task.completed_at = Some(Instant::now());

            Ok(WorkflowResult {
                task_id,
                consensus_achieved: true,
                vote_percentage,
                synthesis_result: Some(result),
                execution_time: task.completed_at.unwrap() - task.submitted_at,
                participating_nodes: task.votes.keys().copied().collect(),
            })
        } else {
            task.status = WorkflowStatus::ConsensusFailed;

            Ok(WorkflowResult {
                task_id,
                consensus_achieved: false,
                vote_percentage,
                synthesis_result: None,
                execution_time: Instant::now() - task.submitted_at,
                participating_nodes: task.votes.keys().copied().collect(),
            })
        }
    }

    /// Run complete workflow: submit, vote, execute
    pub fn run_workflow(
        &self,
        task: SynthesisTask,
        node_ids: &[u32],
        threshold: f32,
        timeout: Duration,
    ) -> Result<WorkflowResult> {
        let start = Instant::now();

        // Submit task
        let task_id = self.submit_synthesis_task(task)?;

        // Check timeout
        if start.elapsed() > timeout {
            return Err(anyhow!("Workflow timed out"));
        }

        // Collect votes
        self.collect_votes(task_id, node_ids)?;

        // Check timeout again
        if start.elapsed() > timeout {
            return Err(anyhow!("Workflow timed out"));
        }

        // Execute if consensus
        self.execute_if_consensus(task_id, threshold)
    }

    /// Process multiple tasks in parallel
    pub fn process_tasks_parallel(
        &self,
        tasks: Vec<SynthesisTask>,
        node_ids: &[u32],
        threshold: f32,
    ) -> Result<Vec<WorkflowResult>> {
        let mut results = Vec::new();

        // Submit all tasks
        let task_ids: Vec<u64> = tasks
            .into_iter()
            .map(|task| self.submit_synthesis_task(task))
            .collect::<Result<Vec<_>>>()?;

        // Vote on all tasks
        for &task_id in &task_ids {
            self.collect_votes(task_id, node_ids)?;
        }

        // Execute all that achieve consensus
        for task_id in task_ids {
            let result = self.execute_if_consensus(task_id, threshold)?;
            results.push(result);
        }

        Ok(results)
    }

    /// Resolve conflicts between tasks
    pub fn resolve_conflicts(&self, tasks: Vec<SynthesisTask>) -> Result<Vec<SynthesisTask>> {
        let mut resolved = Vec::new();
        let mut seen_resources: HashMap<String, u32> = HashMap::new();

        for task in tasks {
            let resource = task.pattern.value.clone().unwrap_or_default();

            match self.config.conflict_resolution_strategy {
                ConflictStrategy::FirstWins => {
                    if !seen_resources.contains_key(&resource) {
                        seen_resources.insert(resource.clone(), 1);
                        resolved.push(task);
                    }
                }
                ConflictStrategy::HighestVoteWins => {
                    // For now, use resource count as vote proxy (later tasks win)
                    if let Some(count) = seen_resources.get_mut(&resource) {
                        *count += 1;
                        // Remove previous version and add new one
                        resolved.retain(|t| t.pattern.value.as_ref() != Some(&resource));
                        resolved.push(task);
                    } else {
                        seen_resources.insert(resource.clone(), 1);
                        resolved.push(task);
                    }
                }
                ConflictStrategy::MergeIfPossible => {
                    // Would need merge logic
                    resolved.push(task);
                }
                ConflictStrategy::RejectBoth => {
                    if !seen_resources.contains_key(&resource) {
                        seen_resources.insert(resource.clone(), 1);
                        resolved.push(task);
                    } else {
                        // Remove previously added task with same resource
                        resolved.retain(|t| t.pattern.value.as_ref() != Some(&resource));
                    }
                }
            }
        }

        Ok(resolved)
    }

    /// Get current status of all tasks
    pub fn get_task_statuses(&self) -> HashMap<u64, (WorkflowStatus, Option<String>)> {
        let tasks = self.tasks.lock()?;
        tasks
            .iter()
            .map(|(&id, task)| (id, (task.status.clone(), task.result.clone())))
            .collect()
    }

    /// Clean up completed tasks older than specified duration
    pub fn cleanup_old_tasks(&self, older_than: Duration) -> Result<(), Box<dyn std::error::Error>>  {
        let mut tasks = self.tasks.lock()?;
        let now = Instant::now();

        tasks.retain(|_, task| {
            if let Some(completed_at) = task.completed_at {
                now - completed_at < older_than
            } else {
                true
            }
        });
    }

    /// Initialize cross-crate integration adapters
    pub async fn initialize_cross_crate_integration(&mut self) -> Result<()> {
        // Initialize synthesis adapter
        let synthesis_adapter = SynthesisCrateAdapter::new(self.device.clone())
            .context("Failed to initialize synthesis adapter")?;
        self.synthesis_adapter = Some(synthesis_adapter);

        // Initialize evolution adapter
        let evolution_adapter = EvolutionEngineAdapter::new(self.device.clone())
            .await
            .context("Failed to initialize evolution adapter")?;
        self.evolution_adapter = Some(evolution_adapter);

        // Initialize knowledge graph adapter
        let knowledge_adapter = KnowledgeGraphAdapter::new(self.device.clone())
            .await
            .context("Failed to initialize knowledge graph adapter")?;
        self.knowledge_adapter = Some(knowledge_adapter);

        Ok(())
    }

    /// Use independent synthesis crate for goal transformation
    pub async fn synthesize_from_goal(&mut self, goal: &str) -> Result<String> {
        let adapter = self
            .synthesis_adapter
            .as_mut()
            .ok_or_else(|| anyhow!("Synthesis adapter not initialized"))?;

        let kernel_id = adapter
            .synthesize_optimized_kernel(goal, 1000.0)
            .await
            .context("Failed to synthesize kernel from goal")?;

        // Store result in knowledge graph
        if let Some(knowledge_adapter) = &mut self.knowledge_adapter {
            let task = adapter.goal_to_synthesis_task(goal).await?;
            let metrics = SynthesisPerformanceMetrics {
                throughput: 1000.0,
                latency_ms: 5.0,
                accuracy: 0.95,
                resource_usage: 0.7,
            };

            let _pattern_id = knowledge_adapter
                .store_synthesis_pattern(goal, &task, &metrics)
                .await
                .context("Failed to store synthesis pattern")?;
        }

        Ok(kernel_id)
    }

    /// Use evolution engines to optimize consensus weights
    pub async fn optimize_consensus_with_evolution(&mut self) -> Result<ConsensusWeights> {
        // Extract needed data before borrowing evolution_adapter mutably
        let device = self.device.clone();
        let config = self.config.clone();

        let evolution_adapter = self
            .evolution_adapter
            .as_mut()
            .ok_or_else(|| anyhow!("Evolution adapter not initialized"))?;

        // Create a temporary reference structure instead of passing self
        let temp_engine = ConsensusSynthesisEngine {
            device,
            config,
            voting_engine: None,
            synthesis_module: None,
            synthesis_adapter: None,
            evolution_adapter: None,
            knowledge_adapter: None,
            tasks: Arc::new(Mutex::new(HashMap::new())),
            next_task_id: Arc::new(Mutex::new(1)),
            vote_buffer: None,
            result_buffer: None,
        };

        let weights = evolution_adapter
            .optimize_consensus_weights(&temp_engine)
            .await
            .context("Failed to optimize consensus weights")?;

        Ok(weights)
    }

    /// Use knowledge graph to find similar successful patterns
    pub async fn find_similar_synthesis_patterns(
        &mut self,
        goal: &str,
    ) -> Result<Vec<SimilarPattern>> {
        let knowledge_adapter = self
            .knowledge_adapter
            .as_mut()
            .ok_or_else(|| anyhow!("Knowledge graph adapter not initialized"))?;

        let patterns = knowledge_adapter
            .find_similar_patterns(goal, 0.8)
            .await
            .context("Failed to find similar patterns")?;

        Ok(patterns)
    }

    /// Use evolution engines to improve synthesis quality
    pub async fn evolve_synthesis_quality(&mut self) -> Result<PopulationStats> {
        let evolution_adapter = self
            .evolution_adapter
            .as_mut()
            .ok_or_else(|| anyhow!("Evolution adapter not initialized"))?;

        // Run hybrid evolution to improve synthesis agents
        let _metrics = evolution_adapter
            .evolve_hybrid()
            .await
            .context("Failed to evolve synthesis quality")?;

        // Get population statistics
        let stats = evolution_adapter
            .get_population_stats()
            .await
            .context("Failed to get population stats")?;

        Ok(stats)
    }

    /// Store consensus decision outcome in knowledge graph
    pub async fn store_consensus_decision(
        &mut self,
        decision_id: &str,
        context: &str,
        success_rate: f64,
        voting_strategy: &str,
        participant_count: usize,
    ) -> Result<String> {
        let knowledge_adapter = self
            .knowledge_adapter
            .as_mut()
            .ok_or_else(|| anyhow!("Knowledge graph adapter not initialized"))?;

        let outcome = ConsensusOutcome {
            success_rate,
            voting_strategy: voting_strategy.to_string(),
            participant_count,
            decision_quality: success_rate * 0.9, // Approximation
        };

        let node_id = knowledge_adapter
            .store_consensus_outcome(decision_id, context, &outcome)
            .await
            .context("Failed to store consensus outcome")?;

        Ok(node_id)
    }

    /// Check if cross-crate integration is initialized
    pub fn is_cross_crate_integration_enabled(&self) -> bool {
        self.synthesis_adapter.is_some()
            && self.evolution_adapter.is_some()
            && self.knowledge_adapter.is_some()
    }

    /// Get evolution metrics if available
    pub fn get_evolution_metrics(
        &self,
    ) -> Option<&stratoswarm_evolution_engines::metrics::EvolutionMetrics> {
        self.evolution_adapter
            .as_ref()
            .map(|adapter| adapter.get_metrics())
    }

    /// Get CUDA device reference for multi-region integration
    pub fn get_device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    /// Initialize cross-crate integrations for E2E workflows
    pub async fn initialize_cross_crate_integrations(&mut self) -> Result<()> {
        // Initialize synthesis adapter
        if self.synthesis_adapter.is_none() {
            let adapter = SynthesisCrateAdapter::new(self.device.clone())
                .context("Failed to create synthesis adapter")?;
            self.synthesis_adapter = Some(adapter);
        }

        // Initialize evolution adapter
        if self.evolution_adapter.is_none() {
            let adapter = EvolutionEngineAdapter::new(self.device.clone())
                .await
                .context("Failed to create evolution adapter")?;
            self.evolution_adapter = Some(adapter);
        }

        // Initialize knowledge graph adapter
        if self.knowledge_adapter.is_none() {
            let adapter = KnowledgeGraphAdapter::new(self.device.clone())
                .await
                .context("Failed to create knowledge graph adapter")?;
            self.knowledge_adapter = Some(adapter);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_integration_config_default() {
        let config = IntegrationConfig::default();
        assert_eq!(config.max_concurrent_tasks, 100);
        assert_eq!(config.min_voters, 3);
    }
}
