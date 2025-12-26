//! MetaAgent implementation for ADAS search

use super::types::*;
use crate::error::{EvolutionEngineError, EvolutionEngineResult};
use std::collections::HashMap;
use uuid::Uuid;

impl MetaAgent {
    /// Create a new Meta Agent with default seed agents
    pub fn new(max_iterations: usize) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            archive: WorkflowArchive::new(),
            seed_agents: Self::create_default_seed_agents(),
            rewriting_policies: Self::create_default_rewriting_policies(),
            current_iteration: 0,
            max_iterations,
        }
    }

    /// Create the 7 default seed agents as mentioned in the ADAS paper
    pub fn create_default_seed_agents() -> Vec<SeedAgentTemplate> {
        vec![
            SeedAgentTemplate {
                name: "Task Analyzer".to_string(),
                domain: "analysis".to_string(),
                base_role: AgentRole {
                    name: "Task Analyzer".to_string(),
                    responsibility: "Analyze task requirements and decompose into subtasks"
                        .to_string(),
                    policy: "Always break down complex tasks systematically".to_string(),
                    prompt_template:
                        "You are a task analyzer. Break down the following task: {task}".to_string(),
                    tool_usage: vec![
                        "task_decomposer".to_string(),
                        "requirement_analyzer".to_string(),
                    ],
                    input_schema: serde_json::json!({"task": "string"}),
                    output_schema: serde_json::json!({"subtasks": "array", "requirements": "array"}),
                },
                adaptability_score: 0.8,
            },
            SeedAgentTemplate {
                name: "Solution Planner".to_string(),
                domain: "planning".to_string(),
                base_role: AgentRole {
                    name: "Solution Planner".to_string(),
                    responsibility: "Create detailed execution plans for tasks".to_string(),
                    policy: "Generate comprehensive, step-by-step plans".to_string(),
                    prompt_template:
                        "You are a solution planner. Create a plan for: {requirements}".to_string(),
                    tool_usage: vec![
                        "plan_generator".to_string(),
                        "resource_estimator".to_string(),
                    ],
                    input_schema: serde_json::json!({"requirements": "array"}),
                    output_schema: serde_json::json!({"plan": "object", "timeline": "array"}),
                },
                adaptability_score: 0.9,
            },
            SeedAgentTemplate {
                name: "Executor".to_string(),
                domain: "execution".to_string(),
                base_role: AgentRole {
                    name: "Executor".to_string(),
                    responsibility: "Execute plans and generate outputs".to_string(),
                    policy: "Follow plans precisely while adapting to constraints".to_string(),
                    prompt_template: "You are an executor. Execute this plan: {plan}".to_string(),
                    tool_usage: vec![
                        "action_executor".to_string(),
                        "output_generator".to_string(),
                    ],
                    input_schema: serde_json::json!({"plan": "object"}),
                    output_schema: serde_json::json!({"result": "object", "status": "string"}),
                },
                adaptability_score: 0.7,
            },
            SeedAgentTemplate {
                name: "Quality Checker".to_string(),
                domain: "validation".to_string(),
                base_role: AgentRole {
                    name: "Quality Checker".to_string(),
                    responsibility: "Validate outputs against requirements and constraints"
                        .to_string(),
                    policy: "Ensure all outputs meet quality standards".to_string(),
                    prompt_template:
                        "You are a quality checker. Validate: {output} against {requirements}"
                            .to_string(),
                    tool_usage: vec!["validator".to_string(), "quality_assessor".to_string()],
                    input_schema: serde_json::json!({"output": "object", "requirements": "array"}),
                    output_schema: serde_json::json!({"valid": "boolean", "issues": "array"}),
                },
                adaptability_score: 0.8,
            },
            SeedAgentTemplate {
                name: "Coordinator".to_string(),
                domain: "coordination".to_string(),
                base_role: AgentRole {
                    name: "Coordinator".to_string(),
                    responsibility: "Coordinate between multiple agents and manage workflow"
                        .to_string(),
                    policy: "Optimize collaboration and resolve conflicts".to_string(),
                    prompt_template: "You are a coordinator. Manage the workflow: {workflow}"
                        .to_string(),
                    tool_usage: vec![
                        "workflow_manager".to_string(),
                        "conflict_resolver".to_string(),
                    ],
                    input_schema: serde_json::json!({"workflow": "object"}),
                    output_schema: serde_json::json!({"coordination_plan": "object"}),
                },
                adaptability_score: 0.9,
            },
            SeedAgentTemplate {
                name: "Resource Manager".to_string(),
                domain: "resources".to_string(),
                base_role: AgentRole {
                    name: "Resource Manager".to_string(),
                    responsibility: "Manage computational and memory resources efficiently"
                        .to_string(),
                    policy: "Optimize resource usage while meeting performance requirements"
                        .to_string(),
                    prompt_template: "You are a resource manager. Optimize resources for: {task}"
                        .to_string(),
                    tool_usage: vec![
                        "resource_optimizer".to_string(),
                        "performance_monitor".to_string(),
                    ],
                    input_schema: serde_json::json!({"task": "object", "constraints": "object"}),
                    output_schema: serde_json::json!({"allocation": "object", "optimization": "object"}),
                },
                adaptability_score: 0.6,
            },
            SeedAgentTemplate {
                name: "Feedback Analyzer".to_string(),
                domain: "learning".to_string(),
                base_role: AgentRole {
                    name: "Feedback Analyzer".to_string(),
                    responsibility: "Analyze performance feedback and suggest improvements"
                        .to_string(),
                    policy: "Learn from failures and optimize future performance".to_string(),
                    prompt_template:
                        "You are a feedback analyzer. Analyze: {feedback} and suggest improvements"
                            .to_string(),
                    tool_usage: vec![
                        "feedback_processor".to_string(),
                        "improvement_suggester".to_string(),
                    ],
                    input_schema: serde_json::json!({"feedback": "object", "performance": "object"}),
                    output_schema: serde_json::json!({"analysis": "object", "suggestions": "array"}),
                },
                adaptability_score: 0.8,
            },
        ]
    }

    /// Create default rewriting policies
    pub fn create_default_rewriting_policies() -> Vec<RewritingPolicy> {
        vec![
            RewritingPolicy {
                name: "Performance-Based Prompt Rewriting".to_string(),
                target_component: RewriteTarget::AgentPrompt,
                strategy: RewriteStrategy::PerformanceFeedback,
                conditions: vec![RewriteCondition {
                    metric: "success_rate".to_string(),
                    threshold: 0.8,
                    comparison: ComparisonOperator::LessThan,
                }],
            },
            RewritingPolicy {
                name: "Tool Usage Optimization".to_string(),
                target_component: RewriteTarget::ToolUsage,
                strategy: RewriteStrategy::EfficiencyOptimization,
                conditions: vec![RewriteCondition {
                    metric: "execution_time".to_string(),
                    threshold: 10.0,
                    comparison: ComparisonOperator::GreaterThan,
                }],
            },
            RewritingPolicy {
                name: "Coordination Flow Enhancement".to_string(),
                target_component: RewriteTarget::CoordinationFlow,
                strategy: RewriteStrategy::DiversityEnhancement,
                conditions: vec![RewriteCondition {
                    metric: "constraint_satisfaction".to_string(),
                    threshold: 0.7,
                    comparison: ComparisonOperator::LessThan,
                }],
            },
        ]
    }

    /// Perform one iteration of the Meta Agent Search
    pub async fn search_iteration(
        &mut self,
        task_description: &str,
    ) -> EvolutionEngineResult<Vec<DiscoveredWorkflow>> {
        self.current_iteration += 1;

        // Generate candidate workflows from seed agents or archive
        let mut candidates = if self.current_iteration == 1 {
            self.generate_initial_workflows(task_description).await?
        } else {
            self.generate_variant_workflows(task_description).await?
        };

        // Evaluate each candidate workflow
        for candidate in &mut candidates {
            candidate.performance_metrics =
                self.evaluate_workflow(candidate, task_description).await?;

            // Add to archive
            self.archive
                .add_workflow(candidate.clone(), self.current_iteration)?;
        }

        // Apply rewriting policies to improve candidates
        let improved_candidates = self.apply_rewriting_policies(&candidates).await?;

        // Update archive with improved candidates
        for improved in &improved_candidates {
            self.archive
                .add_workflow(improved.clone(), self.current_iteration)?;
        }

        // Select best candidates for next iteration
        self.archive.update_best_workflow()?;

        Ok(improved_candidates)
    }

    /// Generate initial workflows from seed agents
    async fn generate_initial_workflows(
        &self,
        task_description: &str,
    ) -> EvolutionEngineResult<Vec<DiscoveredWorkflow>> {
        let mut workflows = Vec::new();

        // Create workflows by combining different seed agents
        for i in 0..5 {
            // Generate 5 initial workflows
            let selected_seeds = self.select_seeds_for_task(task_description, 3 + i % 3)?; // 3-5 agents per workflow
            let workflow = self
                .create_workflow_from_seeds(&selected_seeds, task_description)
                .await?;
            workflows.push(workflow);
        }

        Ok(workflows)
    }

    /// Generate variant workflows from existing archive
    async fn generate_variant_workflows(
        &self,
        task_description: &str,
    ) -> EvolutionEngineResult<Vec<DiscoveredWorkflow>> {
        let mut variants = Vec::new();

        // Get best workflows from archive
        let best_workflows = self.archive.get_top_workflows(3)?;

        for workflow in best_workflows {
            // Apply different mutation strategies
            let variant1 = self
                .mutate_workflow(&workflow, MutationType::AddRole, task_description)
                .await?;
            let variant2 = self
                .mutate_workflow(&workflow, MutationType::ReorderExecution, task_description)
                .await?;
            let variant3 = self
                .mutate_workflow(&workflow, MutationType::RefinePrompts, task_description)
                .await?;

            variants.extend(vec![variant1, variant2, variant3]);
        }

        Ok(variants)
    }

    /// Select seed agents appropriate for the task
    fn select_seeds_for_task(
        &self,
        _task_description: &str,
        count: usize,
    ) -> EvolutionEngineResult<Vec<&SeedAgentTemplate>> {
        // Simple selection based on adaptability score and task relevance
        let mut selected = Vec::new();
        let mut sorted_seeds = self.seed_agents.iter().collect::<Vec<_>>();
        sorted_seeds.sort_by(|a, b| {
            b.adaptability_score
                .partial_cmp(&a.adaptability_score)
                .unwrap()
        });

        // Always include task analyzer as first agent
        if let Some(analyzer) = sorted_seeds.iter().find(|s| s.name == "Task Analyzer") {
            selected.push(*analyzer);
        }

        // Add other high-scoring agents
        for seed in sorted_seeds.iter().take(count) {
            if selected.len() < count && selected.iter().all(|s| s.name != seed.name) {
                selected.push(*seed);
            }
        }

        if selected.is_empty() {
            return Err(EvolutionEngineError::InitializationError {
                message: "No suitable seed agents found".to_string(),
            });
        }

        Ok(selected)
    }

    /// Create a workflow from selected seed agents
    async fn create_workflow_from_seeds(
        &self,
        seeds: &[&SeedAgentTemplate],
        task_description: &str,
    ) -> EvolutionEngineResult<DiscoveredWorkflow> {
        let workflow_id = Uuid::new_v4().to_string();

        // Create agent roles from seeds
        let agent_roles: Vec<AgentRole> = seeds
            .iter()
            .map(|seed| {
                let mut role = seed.base_role.clone();
                // Adapt the role to the specific task
                role.prompt_template = role.prompt_template.replace("{task}", task_description);
                role
            })
            .collect();

        // Create coordination structure
        let coordination_structure = CoordinationStructure {
            execution_order: agent_roles.iter().map(|r| r.name.clone()).collect(),
            communication_graph: self.create_communication_graph(&agent_roles),
            information_flow: self.create_information_flow(&agent_roles),
            coordination_strategy: CoordinationStrategy::Sequential,
        };

        // Generate code implementation
        let code_implementation = self
            .generate_workflow_code(&agent_roles, &coordination_structure)
            .await?;

        Ok(DiscoveredWorkflow {
            id: workflow_id.clone(),
            name: format!("Workflow_{}", workflow_id[..8].to_string()),
            agent_roles,
            coordination_structure,
            code_implementation,
            performance_metrics: PerformanceMetrics::default(),
            generation: self.current_iteration,
            parent_workflow_ids: vec![],
        })
    }

    /// Create communication graph between agents
    fn create_communication_graph(&self, agents: &[AgentRole]) -> HashMap<String, Vec<String>> {
        let mut graph = HashMap::new();

        for (i, agent) in agents.iter().enumerate() {
            let mut connections = Vec::new();

            // Connect to next agent in sequence
            if i < agents.len() - 1 {
                connections.push(agents[i + 1].name.clone());
            }

            // Coordinators connect to all agents
            if agent.name.contains("Coordinator") {
                for other in agents {
                    if other.name != agent.name {
                        connections.push(other.name.clone());
                    }
                }
            }

            graph.insert(agent.name.clone(), connections);
        }

        graph
    }

    /// Create information flow between agents
    fn create_information_flow(&self, agents: &[AgentRole]) -> Vec<InformationFlow> {
        let mut flows = Vec::new();

        for i in 0..agents.len().saturating_sub(1) {
            flows.push(InformationFlow {
                from_agent: agents[i].name.clone(),
                to_agent: agents[i + 1].name.clone(),
                data_type: "processed_data".to_string(),
                transformation: None,
            });
        }

        flows
    }

    /// Generate workflow code implementation
    async fn generate_workflow_code(
        &self,
        agents: &[AgentRole],
        coordination: &CoordinationStructure,
    ) -> EvolutionEngineResult<String> {
        let mut code = String::new();

        code.push_str("def forward(self, taskInfo):\n");
        code.push_str("    \"\"\"\n");
        code.push_str("    Auto-generated workflow implementation\n");
        code.push_str("    \"\"\"\n");
        code.push_str("    results = {}\n\n");

        // Generate code for each agent in execution order
        for agent_name in &coordination.execution_order {
            if let Some(agent) = agents.iter().find(|a| a.name == *agent_name) {
                code.push_str(&format!("    # {} execution\n", agent.name));
                code.push_str(&format!(
                    "    {}_result = self.execute_agent('{}', taskInfo, results)\n",
                    agent.name.to_lowercase().replace(' ', "_"),
                    agent.name
                ));
                code.push_str(&format!(
                    "    results['{}'] = {}_result\n\n",
                    agent.name,
                    agent.name.to_lowercase().replace(' ', "_")
                ));
            }
        }

        code.push_str("    return self.format_final_output(results)\n");

        Ok(code)
    }

    /// Evaluate a workflow's performance
    async fn evaluate_workflow(
        &self,
        workflow: &DiscoveredWorkflow,
        _task_description: &str,
    ) -> EvolutionEngineResult<PerformanceMetrics> {
        // Simulate workflow evaluation (in real implementation, this would run the workflow)
        let complexity_penalty = workflow.agent_roles.len() as f64 * 0.1;
        let coordination_bonus = match workflow.coordination_structure.coordination_strategy {
            CoordinationStrategy::Parallel => 0.2,
            CoordinationStrategy::Pipeline => 0.15,
            CoordinationStrategy::Consensus => 0.1,
            _ => 0.0,
        };

        Ok(PerformanceMetrics {
            success_rate: (0.7 + coordination_bonus - complexity_penalty * 0.1).clamp(0.0, 1.0),
            average_execution_time: 5.0 + complexity_penalty,
            constraint_satisfaction: (0.8 - complexity_penalty * 0.05).clamp(0.0, 1.0),
            output_quality: (0.75 + coordination_bonus * 0.5).clamp(0.0, 1.0),
            resource_efficiency: (0.8 - complexity_penalty * 0.1).clamp(0.0, 1.0),
            robustness: (0.7 + coordination_bonus * 0.3).clamp(0.0, 1.0),
        })
    }

    /// Apply rewriting policies to improve workflows
    async fn apply_rewriting_policies(
        &self,
        workflows: &[DiscoveredWorkflow],
    ) -> EvolutionEngineResult<Vec<DiscoveredWorkflow>> {
        let mut improved = Vec::new();

        for workflow in workflows {
            let mut improved_workflow = workflow.clone();

            for policy in &self.rewriting_policies {
                if self.should_apply_policy(policy, &workflow.performance_metrics) {
                    improved_workflow = self
                        .apply_rewriting_policy(policy, improved_workflow)
                        .await?;
                }
            }

            improved.push(improved_workflow);
        }

        Ok(improved)
    }

    /// Check if a rewriting policy should be applied
    fn should_apply_policy(&self, policy: &RewritingPolicy, metrics: &PerformanceMetrics) -> bool {
        for condition in &policy.conditions {
            let metric_value = match condition.metric.as_str() {
                "success_rate" => metrics.success_rate,
                "execution_time" => metrics.average_execution_time,
                "constraint_satisfaction" => metrics.constraint_satisfaction,
                "output_quality" => metrics.output_quality,
                "resource_efficiency" => metrics.resource_efficiency,
                "robustness" => metrics.robustness,
                _ => continue,
            };

            let condition_met = match condition.comparison {
                ComparisonOperator::LessThan => metric_value < condition.threshold,
                ComparisonOperator::GreaterThan => metric_value > condition.threshold,
                ComparisonOperator::Equal => (metric_value - condition.threshold).abs() < 0.01,
                ComparisonOperator::NotEqual => (metric_value - condition.threshold).abs() >= 0.01,
            };

            if condition_met {
                return true;
            }
        }
        false
    }

    /// Apply a specific rewriting policy
    async fn apply_rewriting_policy(
        &self,
        policy: &RewritingPolicy,
        mut workflow: DiscoveredWorkflow,
    ) -> EvolutionEngineResult<DiscoveredWorkflow> {
        match policy.target_component {
            RewriteTarget::AgentPrompt => {
                // Improve agent prompts based on performance feedback
                for role in &mut workflow.agent_roles {
                    role.prompt_template =
                        format!("{} Focus on accuracy and efficiency.", role.prompt_template);
                }
            }
            RewriteTarget::ToolUsage => {
                // Optimize tool usage
                for role in &mut workflow.agent_roles {
                    role.tool_usage.push("performance_optimizer".to_string());
                }
            }
            RewriteTarget::CoordinationFlow => {
                // Improve coordination strategy
                workflow.coordination_structure.coordination_strategy =
                    CoordinationStrategy::Pipeline;
            }
            RewriteTarget::RoleDefinition => {
                // Refine role responsibilities
                for role in &mut workflow.agent_roles {
                    role.responsibility = format!(
                        "{} with enhanced performance monitoring",
                        role.responsibility
                    );
                }
            }
            RewriteTarget::CodeImplementation => {
                // Add error handling and optimization to code
                workflow.code_implementation = workflow.code_implementation.replace(
                    "def forward(self, taskInfo):",
                    "def forward(self, taskInfo):\n    # Enhanced with error handling and optimization"
                );
            }
        }

        // Update workflow ID to reflect changes
        workflow.id = Uuid::new_v4().to_string();
        workflow.generation = self.current_iteration;

        Ok(workflow)
    }

    /// Mutate a workflow with a specific mutation type
    async fn mutate_workflow(
        &self,
        workflow: &DiscoveredWorkflow,
        mutation_type: MutationType,
        _task_description: &str,
    ) -> EvolutionEngineResult<DiscoveredWorkflow> {
        let mut mutated = workflow.clone();
        mutated.id = Uuid::new_v4().to_string();
        mutated.parent_workflow_ids = vec![workflow.id.clone()];

        match mutation_type {
            MutationType::AddRole => {
                // Add a new agent role
                if let Some(seed) = self
                    .seed_agents
                    .iter()
                    .find(|s| !mutated.agent_roles.iter().any(|r| r.name == s.name))
                {
                    mutated.agent_roles.push(seed.base_role.clone());
                }
            }
            MutationType::RemoveRole => {
                // Remove a non-essential agent role
                if mutated.agent_roles.len() > 2 {
                    mutated.agent_roles.pop();
                }
            }
            MutationType::ModifyRole => {
                // Modify an existing role's responsibilities
                if let Some(role) = mutated.agent_roles.first_mut() {
                    role.responsibility =
                        format!("{} with enhanced capabilities", role.responsibility);
                }
            }
            MutationType::ReorderExecution => {
                // Reorder execution sequence
                if mutated.coordination_structure.execution_order.len() > 1 {
                    mutated.coordination_structure.execution_order.reverse();
                }
            }
            MutationType::ChangeCoordination => {
                // Change coordination strategy
                mutated.coordination_structure.coordination_strategy =
                    match workflow.coordination_structure.coordination_strategy {
                        CoordinationStrategy::Sequential => CoordinationStrategy::Parallel,
                        CoordinationStrategy::Parallel => CoordinationStrategy::Pipeline,
                        CoordinationStrategy::Pipeline => CoordinationStrategy::Consensus,
                        _ => CoordinationStrategy::Sequential,
                    };
            }
            MutationType::RefinePrompts => {
                // Refine agent prompts
                for role in &mut mutated.agent_roles {
                    role.prompt_template = format!("Enhanced: {}", role.prompt_template);
                }
            }
            MutationType::UpdateToolUsage => {
                // Update tool usage patterns
                for role in &mut mutated.agent_roles {
                    role.tool_usage.push("advanced_analyzer".to_string());
                }
            }
        }

        Ok(mutated)
    }

    /// Check if the search should terminate
    pub fn should_terminate(&self) -> bool {
        self.current_iteration >= self.max_iterations || self.archive.has_converged()
    }

    /// Get the best discovered workflow
    pub fn get_best_workflow(&self) -> Option<&DiscoveredWorkflow> {
        self.archive.get_best_workflow()
    }
}
