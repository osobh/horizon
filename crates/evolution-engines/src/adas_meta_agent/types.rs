//! Data structures for ADAS Meta Agent Search

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Meta-Agent that coordinates the ADAS search process
#[derive(Debug, Clone)]
pub struct MetaAgent {
    /// Unique identifier for the meta-agent
    pub id: String,
    /// Archive of discovered workflows
    pub archive: WorkflowArchive,
    /// Collection of seed agent templates
    pub seed_agents: Vec<SeedAgentTemplate>,
    /// Rewriting policies for evolution
    pub rewriting_policies: Vec<RewritingPolicy>,
    /// Current iteration counter
    pub current_iteration: usize,
    /// Maximum iterations allowed
    pub max_iterations: usize,
}

/// Archive of discovered workflows and their performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowArchive {
    /// Collection of discovered workflows
    pub workflows: HashMap<String, DiscoveredWorkflow>,
    /// ID of the best performing workflow
    pub best_workflow_id: Option<String>,
    /// Historical performance data
    pub performance_history: Vec<ArchiveEntry>,
}

/// A discovered agentic workflow with its configuration and performance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveredWorkflow {
    /// Unique identifier
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Agent roles in the workflow
    pub agent_roles: Vec<AgentRole>,
    /// How agents coordinate
    pub coordination_structure: CoordinationStructure,
    /// Implementation code
    pub code_implementation: String,
    /// Performance metrics
    pub performance_metrics: PerformanceMetrics,
    /// Evolution generation
    pub generation: usize,
    /// Parent workflow IDs for lineage tracking
    pub parent_workflow_ids: Vec<String>,
}

/// Individual agent role within a workflow
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentRole {
    /// Agent name
    pub name: String,
    /// What the agent is responsible for
    pub responsibility: String,
    /// Agent's policy/behavior
    pub policy: String,
    /// Template for agent prompts
    pub prompt_template: String,
    /// Tools the agent can use
    pub tool_usage: Vec<String>,
    /// Expected input schema
    pub input_schema: serde_json::Value,
    /// Expected output schema
    pub output_schema: serde_json::Value,
}

/// How agents coordinate and communicate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinationStructure {
    /// Agent execution order
    pub execution_order: Vec<String>,
    /// Communication graph: agent -> agents it communicates with
    pub communication_graph: HashMap<String, Vec<String>>,
    /// Information flow patterns
    pub information_flow: Vec<InformationFlow>,
    /// Coordination strategy used
    pub coordination_strategy: CoordinationStrategy,
}

/// Information flow between agents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InformationFlow {
    /// Source agent
    pub from_agent: String,
    /// Destination agent
    pub to_agent: String,
    /// Type of data being transferred
    pub data_type: String,
    /// Optional transformation applied
    pub transformation: Option<String>,
}

/// Different coordination strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CoordinationStrategy {
    /// Sequential execution
    Sequential,
    /// Parallel execution
    Parallel,
    /// Pipeline processing
    Pipeline,
    /// Hierarchical structure
    Hierarchical,
    /// Negotiation-based
    Negotiation,
    /// Consensus-driven
    Consensus,
}

/// Performance metrics for a discovered workflow
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Success rate (0.0 to 1.0)
    pub success_rate: f64,
    /// Average execution time in seconds
    pub average_execution_time: f64,
    /// How well constraints are satisfied (0.0 to 1.0)
    pub constraint_satisfaction: f64,
    /// Quality of output (0.0 to 1.0)
    pub output_quality: f64,
    /// Resource utilization efficiency (0.0 to 1.0)
    pub resource_efficiency: f64,
    /// Robustness to variations (0.0 to 1.0)
    pub robustness: f64,
}

/// Archive entry tracking workflow evolution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchiveEntry {
    /// Associated workflow ID
    pub workflow_id: String,
    /// Iteration when created
    pub iteration: usize,
    /// When it was created
    pub timestamp: String,
    /// Performance at time of creation
    pub performance: PerformanceMetrics,
    /// Type of mutation applied
    pub mutation_type: MutationType,
}

/// Types of mutations applied to workflows
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MutationType {
    /// Add a new agent role
    AddRole,
    /// Remove an agent role
    RemoveRole,
    /// Modify existing role
    ModifyRole,
    /// Change execution order
    ReorderExecution,
    /// Change coordination strategy
    ChangeCoordination,
    /// Refine prompt templates
    RefinePrompts,
    /// Update tool usage
    UpdateToolUsage,
}

/// Pre-written seed agent templates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeedAgentTemplate {
    /// Template name
    pub name: String,
    /// Domain of expertise
    pub domain: String,
    /// Base agent role
    pub base_role: AgentRole,
    /// How adaptable this template is
    pub adaptability_score: f64,
}

/// Rewriting policies for the meta-agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RewritingPolicy {
    /// Policy name
    pub name: String,
    /// What component to target
    pub target_component: RewriteTarget,
    /// How to rewrite
    pub strategy: RewriteStrategy,
    /// Conditions for applying this policy
    pub conditions: Vec<RewriteCondition>,
}

/// What can be rewritten
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RewriteTarget {
    /// Agent prompt templates
    AgentPrompt,
    /// Tool usage patterns
    ToolUsage,
    /// Coordination flow
    CoordinationFlow,
    /// Role definitions
    RoleDefinition,
    /// Code implementation
    CodeImplementation,
}

/// Rewriting strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RewriteStrategy {
    /// Based on performance feedback
    PerformanceFeedback,
    /// Fix constraint violations
    ConstraintViolationFix,
    /// Enhance diversity
    DiversityEnhancement,
    /// Optimize for efficiency
    EfficiencyOptimization,
    /// Improve robustness
    RobustnessImprovement,
}

/// Condition for applying rewrite policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RewriteCondition {
    /// Metric to check
    pub metric: String,
    /// Threshold value
    pub threshold: f64,
    /// Comparison operator
    pub comparison: ComparisonOperator,
}

/// Comparison operators for conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComparisonOperator {
    /// Less than threshold
    LessThan,
    /// Greater than threshold
    GreaterThan,
    /// Equal to threshold
    Equal,
    /// Not equal to threshold
    NotEqual,
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            success_rate: 0.5,
            average_execution_time: 10.0,
            constraint_satisfaction: 0.5,
            output_quality: 0.5,
            resource_efficiency: 0.5,
            robustness: 0.5,
        }
    }
}
