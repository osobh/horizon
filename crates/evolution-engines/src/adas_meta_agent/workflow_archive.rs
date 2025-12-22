//! WorkflowArchive implementation for storing and managing discovered workflows

use super::types::*;
use crate::error::EvolutionEngineResult;
use std::collections::HashMap;

impl WorkflowArchive {
    /// Create a new empty workflow archive
    pub fn new() -> Self {
        Self {
            workflows: HashMap::new(),
            best_workflow_id: None,
            performance_history: Vec::new(),
        }
    }

    /// Add a workflow to the archive
    pub fn add_workflow(
        &mut self,
        workflow: DiscoveredWorkflow,
        iteration: usize,
    ) -> EvolutionEngineResult<()> {
        let entry = ArchiveEntry {
            workflow_id: workflow.id.clone(),
            iteration,
            timestamp: chrono::Utc::now().to_rfc3339(),
            performance: workflow.performance_metrics.clone(),
            mutation_type: MutationType::AddRole, // Default for simplicity
        };

        self.performance_history.push(entry);
        self.workflows.insert(workflow.id.clone(), workflow);
        Ok(())
    }

    /// Update the best workflow ID based on current performance
    pub fn update_best_workflow(&mut self) -> EvolutionEngineResult<()> {
        let best_id = self
            .workflows
            .iter()
            .max_by(|a, b| {
                a.1.performance_metrics
                    .success_rate
                    .partial_cmp(&b.1.performance_metrics.success_rate)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(id, _)| id.clone());

        self.best_workflow_id = best_id;
        Ok(())
    }

    /// Get the best workflow if it exists
    pub fn get_best_workflow(&self) -> Option<&DiscoveredWorkflow> {
        self.best_workflow_id
            .as_ref()
            .and_then(|id| self.workflows.get(id))
    }

    /// Get the top N workflows by performance
    pub fn get_top_workflows(
        &self,
        count: usize,
    ) -> EvolutionEngineResult<Vec<DiscoveredWorkflow>> {
        let mut workflows: Vec<_> = self.workflows.values().collect();
        workflows.sort_by(|a, b| {
            b.performance_metrics
                .success_rate
                .partial_cmp(&a.performance_metrics.success_rate)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(workflows.into_iter().take(count).cloned().collect())
    }

    /// Check if the archive has converged (performance improvements have plateaued)
    pub fn has_converged(&self) -> bool {
        // Simple convergence check: if last 5 iterations haven't improved significantly
        if self.performance_history.len() < 10 {
            return false;
        }

        let recent = &self.performance_history[self.performance_history.len() - 5..];
        let oldest_recent = recent.first()?.performance.success_rate;
        let newest_recent = recent.last()?.performance.success_rate;

        (newest_recent - oldest_recent).abs() < 0.01
    }
}
