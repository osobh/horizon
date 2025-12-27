//! Integration test to verify ADAS meta agent module structure works correctly

use stratoswarm_evolution_engines::adas_meta_agent::{
    CoordinationStrategy, MetaAgent, MutationType, PerformanceMetrics,
    RewriteStrategy, RewriteTarget, WorkflowArchive,
};

#[test]
fn test_adas_meta_agent_module_imports() {
    // Test that we can import all types from the module
    let meta_agent = MetaAgent::new(10);
    assert!(!meta_agent.id.is_empty());
    assert_eq!(meta_agent.max_iterations, 10);
    assert_eq!(meta_agent.current_iteration, 0);

    // Test archive functionality
    let archive = WorkflowArchive::new();
    assert!(archive.workflows.is_empty());
    assert!(archive.best_workflow_id.is_none());

    // Test performance metrics
    let metrics = PerformanceMetrics::default();
    assert_eq!(metrics.success_rate, 0.5);
    assert_eq!(metrics.average_execution_time, 10.0);
}

#[test]
fn test_coordination_strategy_types() {
    // Test all coordination strategies are accessible
    let strategies = vec![
        CoordinationStrategy::Sequential,
        CoordinationStrategy::Parallel,
        CoordinationStrategy::Pipeline,
        CoordinationStrategy::Hierarchical,
        CoordinationStrategy::Negotiation,
        CoordinationStrategy::Consensus,
    ];
    assert_eq!(strategies.len(), 6);
}

#[test]
fn test_rewrite_targets_and_strategies() {
    // Test mutation and rewrite types
    let targets = vec![
        RewriteTarget::AgentPrompt,
        RewriteTarget::ToolUsage,
        RewriteTarget::CoordinationFlow,
        RewriteTarget::RoleDefinition,
        RewriteTarget::CodeImplementation,
    ];
    assert_eq!(targets.len(), 5);

    let mutation_types = vec![
        MutationType::AddRole,
        MutationType::RemoveRole,
        MutationType::ModifyRole,
        MutationType::ReorderExecution,
        MutationType::ChangeCoordination,
        MutationType::RefinePrompts,
        MutationType::UpdateToolUsage,
    ];
    assert_eq!(mutation_types.len(), 7);
}

#[tokio::test]
async fn test_meta_agent_workflow_operations() {
    let meta_agent = MetaAgent::new(5);

    // Test seed agents creation
    let seed_agents = meta_agent.seed_agents.clone();
    assert!(!seed_agents.is_empty());

    // Test rewriting policies
    let policies = meta_agent.rewriting_policies.clone();
    assert!(!policies.is_empty());

    // Test workflow archive operations
    let workflow_count = meta_agent.archive.workflows.len();
    assert_eq!(workflow_count, 0); // Should start empty
}
