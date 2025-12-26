//! UniversalAgent - The single, evolvable agent implementation

pub mod handlers;

use async_trait::async_trait;
use chrono::Utc;
use std::pin::Pin;
use std::future::Future;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::dna::{AgentDNA, Modification, ModificationSource, ModificationType, SkillExecution};
use crate::error::{Result, UniversalAgentError};
use crate::learning::{LearningEngine, LearningImprovement};

// Re-export from horizon-agents-core
pub use horizon_agents_core::{
    Agent, AgentConfig, AgentError, AgentRequest, AgentResponse, AgentState, AutonomyLevel,
    HealthStatus, Lifecycle, SafetyThresholds,
};

/// Unique identifier for agent instances
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AgentId(pub Uuid);

impl AgentId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for AgentId {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for UniversalAgent
#[derive(Debug, Clone)]
pub struct UniversalAgentConfig {
    /// Name for this agent instance
    pub name: String,
    /// Whether to enable learning/upskilling
    pub learning_enabled: bool,
    /// Safety thresholds
    pub safety_thresholds: SafetyThresholds,
}

impl Default for UniversalAgentConfig {
    fn default() -> Self {
        Self {
            name: "universal-agent".to_string(),
            learning_enabled: true,
            safety_thresholds: SafetyThresholds::default(),
        }
    }
}

impl UniversalAgentConfig {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Default::default()
        }
    }

    pub fn with_learning_enabled(mut self, enabled: bool) -> Self {
        self.learning_enabled = enabled;
        self
    }
}

/// The Universal Agent - a single agent implementation that adapts to any role via DNA
pub struct UniversalAgent {
    /// Unique instance ID
    pub id: AgentId,
    /// The DNA that defines this agent's capabilities
    dna: Arc<RwLock<AgentDNA>>,
    /// Current lifecycle state
    lifecycle: Lifecycle,
    /// Learning engine for DNA evolution
    learning_engine: LearningEngine,
    /// Configuration
    config: UniversalAgentConfig,
}

impl UniversalAgent {
    /// Create a new UniversalAgent from DNA
    pub async fn from_dna(dna: AgentDNA, config: UniversalAgentConfig) -> Result<Self> {
        // Validate DNA first
        dna.validate().map_err(|errors| {
            UniversalAgentError::ValidationFailed(errors.join("; "))
        })?;

        let learning_rate = dna.behavior.learning_rate;

        Ok(Self {
            id: AgentId::new(),
            dna: Arc::new(RwLock::new(dna)),
            lifecycle: Lifecycle::new(),
            learning_engine: LearningEngine::new(learning_rate),
            config,
        })
    }

    /// Create from a template DNA
    pub async fn efficiency_hunter(config: UniversalAgentConfig) -> Result<Self> {
        Self::from_dna(AgentDNA::efficiency_hunter_template(), config).await
    }

    /// Create from a template DNA
    pub async fn cost_optimizer(config: UniversalAgentConfig) -> Result<Self> {
        Self::from_dna(AgentDNA::cost_optimizer_template(), config).await
    }

    /// Create from a template DNA
    pub async fn capacity_planner(config: UniversalAgentConfig) -> Result<Self> {
        Self::from_dna(AgentDNA::capacity_planner_template(), config).await
    }

    /// Create from a template DNA
    pub async fn scheduler_optimizer(config: UniversalAgentConfig) -> Result<Self> {
        Self::from_dna(AgentDNA::scheduler_optimizer_template(), config).await
    }

    /// Create from a template DNA
    pub async fn orchestrator(config: UniversalAgentConfig) -> Result<Self> {
        Self::from_dna(AgentDNA::orchestrator_template(), config).await
    }

    /// Execute a skill by ID
    pub async fn execute_skill(
        &self,
        skill_id: &str,
        request: AgentRequest,
    ) -> Result<AgentResponse> {
        let dna = self.dna.read().await;

        let skill = dna
            .skills
            .get_skill(skill_id)
            .ok_or_else(|| UniversalAgentError::SkillNotFound(skill_id.to_string()))?;

        // Check prerequisites
        for prereq in &skill.prerequisites {
            if dna.skills.get_skill(prereq).is_none() {
                return Err(UniversalAgentError::PrerequisitesNotMet {
                    skill_id: skill_id.to_string(),
                    missing: vec![prereq.clone()],
                });
            }
        }

        let start_time = Instant::now();

        // Execute based on skill type
        let result = match &skill.execution {
            SkillExecution::Builtin { handler_id } => {
                self.execute_builtin(handler_id, request.clone()).await
            }
            SkillExecution::LearnedWorkflow { .. } => {
                // TODO: Implement workflow execution
                Ok(AgentResponse::new(request.id, "Workflow execution not yet implemented".to_string()))
            }
            SkillExecution::PatternBased { .. } => {
                // TODO: Implement pattern execution
                Ok(AgentResponse::new(request.id, "Pattern execution not yet implemented".to_string()))
            }
            SkillExecution::LLMPowered { .. } => {
                // TODO: Implement LLM execution
                Ok(AgentResponse::new(request.id, "LLM execution not yet implemented".to_string()))
            }
            SkillExecution::Composite { sub_skills, .. } => {
                self.execute_composite(sub_skills, request.clone()).await
            }
        };

        let execution_time = start_time.elapsed();

        // Learn from execution if enabled
        if self.config.learning_enabled {
            let success = result.is_ok();
            drop(dna); // Release read lock before acquiring write

            if let Ok(learning_result) = self
                .learning_engine
                .analyze_execution(skill_id, success, execution_time)
                .await
            {
                if let Some(improvement) = learning_result.suggested_improvement {
                    self.apply_learning(skill_id, improvement).await?;
                }
            }
        }

        result
    }

    /// Execute a builtin handler
    async fn execute_builtin(
        &self,
        handler_id: &str,
        request: AgentRequest,
    ) -> Result<AgentResponse> {
        // Use the handlers module
        handlers::execute(handler_id, request).await
    }

    /// Execute composite skills (boxed to avoid infinite future size from recursion)
    fn execute_composite<'a>(
        &'a self,
        sub_skills: &'a [String],
        request: AgentRequest,
    ) -> Pin<Box<dyn Future<Output = Result<AgentResponse>> + Send + 'a>> {
        Box::pin(async move {
            let mut combined_content = Vec::new();

            for skill_id in sub_skills {
                let sub_request = AgentRequest::new(request.content.clone());
                match self.execute_skill(skill_id, sub_request).await {
                    Ok(response) => combined_content.push(response.content),
                    Err(e) => combined_content.push(format!("Error in {}: {}", skill_id, e)),
                }
            }

            Ok(AgentResponse::new(request.id, combined_content.join("\n\n")))
        })
    }

    /// Find the best skill for a given task
    pub async fn find_skill_for_task(&self, task_description: &str) -> Option<String> {
        let dna = self.dna.read().await;
        let task_lower = task_description.to_lowercase();

        // Simple keyword matching (could be enhanced with ML)
        for (skill_id, skill) in &dna.skills.skills {
            let skill_lower = skill.name.to_lowercase();
            let desc_lower = skill.description.to_lowercase();

            if task_lower.contains(&skill_lower)
                || skill_lower.contains(&task_lower)
                || desc_lower.contains(&task_lower)
            {
                return Some(skill_id.clone());
            }
        }

        // Return first skill as fallback
        dna.skills.skills.keys().next().cloned()
    }

    /// Apply a learning improvement to the DNA
    async fn apply_learning(
        &self,
        skill_id: &str,
        improvement: LearningImprovement,
    ) -> Result<()> {
        let mut dna = self.dna.write().await;

        match improvement {
            LearningImprovement::SkillUpgrade {
                new_proficiency,
                performance_delta,
            } => {
                // Extract skill info first to avoid borrow conflicts
                let skill_info = if let Some(skill) = dna.skills.get_skill_mut(skill_id) {
                    let old_level = skill.proficiency_level;
                    skill.proficiency_level =
                        (skill.proficiency_level + new_proficiency).min(10);
                    Some((old_level, skill.proficiency_level))
                } else {
                    None
                };

                if let Some((old_level, new_level)) = skill_info {
                    dna.lineage.modification_chain.push(
                        Modification::new(
                            ModificationType::SkillUpgraded {
                                skill_id: skill_id.to_string(),
                                from_level: old_level,
                                to_level: new_level,
                            },
                            format!(
                                "Skill {} upgraded from level {} to {}",
                                skill_id, old_level, new_level
                            ),
                            ModificationSource::Learning {
                                episode_id: Uuid::new_v4().to_string(),
                            },
                        )
                        .with_performance_delta(performance_delta),
                    );

                    dna.version = dna.version.increment_patch();
                    dna.updated_at = Utc::now();
                }
            }
            LearningImprovement::BehaviorAdjustment { parameter, delta } => {
                dna.adjust_behavior(
                    &parameter,
                    delta,
                    ModificationSource::Learning {
                        episode_id: Uuid::new_v4().to_string(),
                    },
                );
            }
            LearningImprovement::NewSkillDiscovered { skill } => {
                dna.add_skill(skill);
                dna.version = dna.version.increment_minor();
                dna.updated_at = Utc::now();
            }
        }

        Ok(())
    }

    /// Get current DNA (read-only snapshot)
    pub async fn get_dna(&self) -> AgentDNA {
        self.dna.read().await.clone()
    }

    /// Get capabilities for service discovery
    pub async fn get_capabilities(&self) -> Vec<crate::dna::Capability> {
        self.dna.read().await.capabilities.clone()
    }

    /// Check if agent has a specific capability
    pub async fn has_capability(&self, capability_id: &str) -> bool {
        self.dna.read().await.has_capability(capability_id)
    }
}

#[async_trait]
impl Agent for UniversalAgent {
    fn name(&self) -> &str {
        &self.config.name
    }

    async fn init(&mut self) -> std::result::Result<(), AgentError> {
        self.lifecycle
            .transition_to(AgentState::Initializing)
            .map_err(|e| AgentError::InitializationFailed(e.to_string()))?;

        // Initialization logic here

        self.lifecycle
            .transition_to(AgentState::Ready)
            .map_err(|e| AgentError::InitializationFailed(e.to_string()))?;

        Ok(())
    }

    async fn process(
        &self,
        request: AgentRequest,
    ) -> std::result::Result<AgentResponse, AgentError> {
        if !self.lifecycle.is_operational() {
            return Err(AgentError::NotInitialized);
        }

        // Find best skill for task
        if let Some(skill_id) = self.find_skill_for_task(&request.content).await {
            self.execute_skill(&skill_id, request)
                .await
                .map_err(|e| AgentError::ExecutionFailed(e.to_string()))
        } else {
            Ok(AgentResponse::new(request.id, "No suitable skill found for this task".to_string()))
        }
    }

    async fn health(&self) -> std::result::Result<HealthStatus, AgentError> {
        if self.lifecycle.is_operational() {
            Ok(HealthStatus::Healthy)
        } else {
            Ok(HealthStatus::Unhealthy {
                reason: format!("Agent state: {:?}", self.lifecycle.state()),
            })
        }
    }

    async fn shutdown(&mut self) -> std::result::Result<(), AgentError> {
        self.lifecycle
            .transition_to(AgentState::ShuttingDown)
            .map_err(|e| AgentError::ExecutionFailed(e.to_string()))?;

        // Cleanup logic here

        self.lifecycle
            .transition_to(AgentState::Shutdown)
            .map_err(|e| AgentError::ExecutionFailed(e.to_string()))?;

        Ok(())
    }

    fn autonomy_level(&self) -> AutonomyLevel {
        // This is a sync method, so we use block_in_place
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                self.dna.read().await.behavior.autonomy_level
            })
        })
    }

    fn set_autonomy_level(
        &mut self,
        level: AutonomyLevel,
    ) -> std::result::Result<(), AgentError> {
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                let mut dna = self.dna.write().await;
                // TODO: Validate transition
                dna.behavior.autonomy_level = level;
                Ok(())
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_create_efficiency_hunter() {
        let config = UniversalAgentConfig::new("test-agent");
        let agent = UniversalAgent::efficiency_hunter(config).await.unwrap();

        let dna = agent.get_dna().await;
        assert_eq!(dna.name, "efficiency-hunter");
        assert!(agent.has_capability("efficiency_analysis").await);
    }

    #[tokio::test]
    async fn test_find_skill_for_task() {
        let config = UniversalAgentConfig::new("test-agent");
        let agent = UniversalAgent::efficiency_hunter(config).await.unwrap();

        let skill = agent.find_skill_for_task("detect idle resources").await;
        assert!(skill.is_some());
    }

    #[tokio::test]
    async fn test_agent_lifecycle() {
        let config = UniversalAgentConfig::new("test-agent");
        let mut agent = UniversalAgent::efficiency_hunter(config).await.unwrap();

        assert!(agent.init().await.is_ok());
        assert!(agent.health().await.unwrap().is_healthy());
        assert!(agent.shutdown().await.is_ok());
    }
}
