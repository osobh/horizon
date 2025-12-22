//! Meta-agent search functionality for ADAS

use super::engine::AdasEngine;
use crate::{
    adas_meta_agent::DiscoveredWorkflow,
    error::{EvolutionEngineError, EvolutionEngineResult},
};

impl AdasEngine {
    /// Perform Meta Agent Search for discovering optimal agentic systems
    pub async fn meta_agent_search(
        &mut self,
        task_description: &str,
    ) -> EvolutionEngineResult<DiscoveredWorkflow> {
        while !self.meta_agent.should_terminate() {
            let iteration_workflows = self.meta_agent.search_iteration(task_description).await?;
            self.discovered_workflows.extend(iteration_workflows);
        }

        // Return the best discovered workflow
        self.meta_agent.get_best_workflow().cloned().ok_or_else(|| {
            EvolutionEngineError::SearchError {
                message: "No workflows discovered during meta agent search".to_string(),
            }
        })
    }
}
