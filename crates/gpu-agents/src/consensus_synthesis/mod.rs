//! Missing integration components between consensus and synthesis
//!
//! Provides the glue layer for coordinating consensus decisions with synthesis operations

pub mod integration;

pub use integration::{ConflictStrategy, WorkflowStatus};

// Main types are already public in this module

use anyhow::{anyhow, Result};
use cudarc::driver::{CudaContext, CudaSlice, DeviceSlice};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use crate::consensus::{voting::GpuVoting, Proposal, Vote};
use crate::synthesis::{pattern::GpuPatternMatcher, NodeType, Pattern};

/// Consensus-driven synthesis engine
pub struct ConsensusSynthesisEngine {
    ctx: Arc<CudaContext>,
    voting_engine: Option<GpuVoting>,
    pattern_matcher: Option<GpuPatternMatcher>,
    decision_buffer: Option<CudaSlice<u32>>,
    synthesis_queue: Vec<SynthesisTask>,
    metrics: EngineMetrics,
}

#[derive(Debug, Clone)]
pub struct SynthesisTask {
    pub id: u32,
    pub proposal_id: u32,
    pub pattern: Pattern,
    pub consensus_threshold: f32,
    pub status: TaskStatus,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TaskStatus {
    Pending,
    AwaitingConsensus,
    Approved,
    Synthesizing,
    Completed,
    Rejected,
}

#[derive(Debug, Default)]
pub struct EngineMetrics {
    pub tasks_submitted: u64,
    pub tasks_approved: u64,
    pub tasks_rejected: u64,
    pub consensus_rounds: u64,
    pub synthesis_operations: u64,
    pub avg_consensus_time_us: f64,
    pub avg_synthesis_time_us: f64,
}

impl ConsensusSynthesisEngine {
    /// Create new consensus-synthesis engine
    pub fn new(ctx: Arc<CudaContext>) -> Result<Self> {
        let stream = ctx.default_stream();
        // Allocate decision buffer for consensus results
        // SAFETY: alloc returns uninitialized memory. decision_buffer will be written
        // by consensus voting before any reads during run_consensus_round().
        let decision_buffer = unsafe { stream.alloc::<u32>(1024) }
            .map_err(|e| anyhow!("Failed to allocate decision buffer: {}", e))?;

        Ok(Self {
            ctx,
            voting_engine: None,
            pattern_matcher: None,
            decision_buffer: Some(decision_buffer),
            synthesis_queue: Vec::new(),
            metrics: EngineMetrics::default(),
        })
    }

    /// Initialize voting engine
    pub fn init_voting(&mut self, num_agents: usize) -> Result<()> {
        self.voting_engine = Some(GpuVoting::new(Arc::clone(&self.ctx), num_agents)?);
        Ok(())
    }

    /// Initialize pattern matcher
    pub fn init_synthesis(&mut self) -> Result<()> {
        self.pattern_matcher = Some(GpuPatternMatcher::new(Arc::clone(&self.ctx), 1024)?);
        Ok(())
    }

    /// Submit synthesis task for consensus approval
    pub fn submit_synthesis_task(&mut self, pattern: Pattern, threshold: f32) -> Result<u32> {
        let task_id = self.synthesis_queue.len() as u32;

        let task = SynthesisTask {
            id: task_id,
            proposal_id: task_id, // For simplicity
            pattern,
            consensus_threshold: threshold,
            status: TaskStatus::Pending,
        };

        self.synthesis_queue.push(task);
        self.metrics.tasks_submitted += 1;

        Ok(task_id)
    }

    /// Run consensus on pending synthesis tasks
    pub fn run_consensus_round(&mut self) -> Result<Vec<u32>> {
        if self.voting_engine.is_none() {
            return Err(anyhow!("Voting engine not initialized"));
        }

        let start = Instant::now();
        let mut approved_tasks = Vec::new();

        // Process each pending task
        for task in &mut self.synthesis_queue {
            if task.status != TaskStatus::Pending {
                continue;
            }

            task.status = TaskStatus::AwaitingConsensus;

            // Create proposal for this synthesis task
            let proposal = Proposal {
                id: task.proposal_id,
                proposer_id: 0,
                value: task.id,
                round: 1,
            };

            // Run voting (simplified - in real implementation would collect votes)
            let voting_engine = self.voting_engine.as_ref().ok_or_else(|| anyhow::anyhow!("Voting engine not initialized"))?;

            // Create votes for the proposal
            let votes = vec![
                Vote {
                    agent_id: 0,
                    proposal_id: proposal.id,
                    value: 1, // Yes vote
                    timestamp: 0,
                };
                80
            ]; // 80% approval

            let vote_results = voting_engine.aggregate_votes(&votes, 2)?;

            // For now, simulate approval based on threshold
            let approval_rate = 0.8; // Simulated
            if approval_rate >= task.consensus_threshold {
                task.status = TaskStatus::Approved;
                approved_tasks.push(task.id);
                self.metrics.tasks_approved += 1;
            } else {
                task.status = TaskStatus::Rejected;
                self.metrics.tasks_rejected += 1;
            }
        }

        self.metrics.consensus_rounds += 1;
        self.metrics.avg_consensus_time_us = (self.metrics.avg_consensus_time_us
            * (self.metrics.consensus_rounds - 1) as f64
            + start.elapsed().as_micros() as f64)
            / self.metrics.consensus_rounds as f64;

        Ok(approved_tasks)
    }

    /// Execute approved synthesis tasks
    pub fn execute_synthesis(&mut self) -> Result<Vec<u32>> {
        if self.pattern_matcher.is_none() {
            return Err(anyhow!("Pattern matcher not initialized"));
        }

        let start = Instant::now();
        let mut completed_tasks = Vec::new();

        for task in &mut self.synthesis_queue {
            if task.status != TaskStatus::Approved {
                continue;
            }

            task.status = TaskStatus::Synthesizing;

            // Simulate pattern matching execution
            // In a real implementation, this would use the pattern matcher
            task.status = TaskStatus::Completed;
            completed_tasks.push(task.id);
            self.metrics.synthesis_operations += 1;
        }

        self.metrics.avg_synthesis_time_us = (self.metrics.avg_synthesis_time_us
            * self.metrics.synthesis_operations.saturating_sub(1) as f64
            + start.elapsed().as_micros() as f64)
            / self.metrics.synthesis_operations as f64;

        Ok(completed_tasks)
    }

    /// Get metrics
    pub fn get_metrics(&self) -> &EngineMetrics {
        &self.metrics
    }

    /// Get task status
    pub fn get_task_status(&self, task_id: u32) -> Option<TaskStatus> {
        self.synthesis_queue
            .iter()
            .find(|t| t.id == task_id)
            .map(|t| t.status.clone())
    }
}

/// Template registry for reusable synthesis patterns
pub struct TemplateRegistry {
    templates: HashMap<String, SynthesisTemplate>,
    usage_stats: HashMap<String, TemplateStats>,
}

#[derive(Debug, Clone)]
pub struct SynthesisTemplate {
    pub name: String,
    pub pattern: Pattern,
    pub required_consensus: f32,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Default)]
struct TemplateStats {
    usage_count: u64,
    success_count: u64,
    avg_consensus_time_us: f64,
    avg_synthesis_time_us: f64,
}

impl TemplateRegistry {
    pub fn new() -> Self {
        Self {
            templates: HashMap::new(),
            usage_stats: HashMap::new(),
        }
    }

    /// Register a new template
    pub fn register(&mut self, name: &str, pattern: Pattern, consensus: f32) -> Result<()> {
        if self.templates.contains_key(name) {
            return Err(anyhow!("Template {} already exists", name));
        }

        let template = SynthesisTemplate {
            name: name.to_string(),
            pattern,
            required_consensus: consensus,
            metadata: HashMap::new(),
        };

        self.templates.insert(name.to_string(), template);
        self.usage_stats
            .insert(name.to_string(), TemplateStats::default());

        Ok(())
    }

    /// Get template by name
    pub fn get(&self, name: &str) -> Option<&SynthesisTemplate> {
        self.templates.get(name)
    }

    /// Update usage statistics
    pub fn record_usage(
        &mut self,
        name: &str,
        success: bool,
        consensus_time: f64,
        synthesis_time: f64,
    ) {
        if let Some(stats) = self.usage_stats.get_mut(name) {
            stats.usage_count += 1;
            if success {
                stats.success_count += 1;
            }
            stats.avg_consensus_time_us =
                (stats.avg_consensus_time_us * (stats.usage_count - 1) as f64 + consensus_time)
                    / stats.usage_count as f64;
            stats.avg_synthesis_time_us =
                (stats.avg_synthesis_time_us * (stats.usage_count - 1) as f64 + synthesis_time)
                    / stats.usage_count as f64;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_creation() -> Result<(), Box<dyn std::error::Error>> {
        let ctx = CudaContext::new(0)?;
        let engine = ConsensusSynthesisEngine::new(ctx);
        assert!(engine.is_ok());
        Ok(())
    }

    #[test]
    fn test_template_registry() {
        let mut registry = TemplateRegistry::new();
        let pattern = Pattern {
            node_type: NodeType::Function,
            children: vec![],
            value: Some("test".to_string()),
        };

        assert!(registry.register("test_template", pattern, 0.75).is_ok());
        assert!(registry.get("test_template").is_some());
    }
}
