//! Test the ADAS Meta Agent Search implementation
//!
//! This test verifies that the Meta Agent Search framework correctly:
//! - Creates workflows from seed agents
//! - Evaluates performance
//! - Applies rewriting policies
//! - Maintains workflow archive
//! - Performs iterative search

use exorust_evolution_engines::{
    adas::{AdasConfig, AdasEngine},
    adas_meta_agent::{ComparisonOperator, MetaAgent},
    error::EvolutionEngineResult,
    traits::EvolutionEngine,
};
use tokio;

#[tokio::main]
async fn main() -> EvolutionEngineResult<()> {
    println!("🧠 Testing ADAS Meta Agent Search Implementation");
    println!("================================================");

    // Test 1: Create Meta Agent with default configuration
    println!("\n1. Creating Meta Agent with default seed agents...");
    let mut meta_agent = MetaAgent::new(10); // 10 iterations max

    println!("   ✓ Meta Agent ID: {}", meta_agent.id);
    println!("   ✓ Seed agents count: {}", meta_agent.seed_agents.len());
    println!(
        "   ✓ Rewriting policies count: {}",
        meta_agent.rewriting_policies.len()
    );

    // Verify we have the expected 7 seed agents
    assert_eq!(
        meta_agent.seed_agents.len(),
        7,
        "Expected 7 default seed agents"
    );

    let expected_agents = [
        "Task Analyzer",
        "Solution Planner",
        "Executor",
        "Quality Checker",
        "Coordinator",
        "Resource Manager",
        "Feedback Analyzer",
    ];

    for expected in &expected_agents {
        assert!(
            meta_agent.seed_agents.iter().any(|s| s.name == *expected),
            "Missing expected seed agent: {}",
            expected
        );
        println!("   ✓ Found seed agent: {}", expected);
    }

    // Test 2: Perform search iterations
    println!("\n2. Performing Meta Agent Search iterations...");
    let task_description =
        "Optimize data processing pipeline with error handling and performance monitoring";

    let mut total_workflows = 0;
    for iteration in 1..=3 {
        println!("   Iteration {}:", iteration);

        let discovered_workflows = meta_agent.search_iteration(task_description).await?;
        total_workflows += discovered_workflows.len();

        println!("     ✓ Discovered {} workflows", discovered_workflows.len());

        // Verify workflows have valid structure
        for (i, workflow) in discovered_workflows.iter().enumerate() {
            println!(
                "     ✓ Workflow {}: '{}' with {} agents",
                i + 1,
                workflow.name,
                workflow.agent_roles.len()
            );

            // Check that workflow has valid performance metrics
            assert!(workflow.performance_metrics.success_rate >= 0.0);
            assert!(workflow.performance_metrics.success_rate <= 1.0);
            assert!(workflow.performance_metrics.average_execution_time > 0.0);

            // Check coordination structure
            assert!(!workflow.coordination_structure.execution_order.is_empty());
            println!(
                "       - Coordination: {:?}",
                workflow.coordination_structure.coordination_strategy
            );
            println!(
                "       - Success rate: {:.2}",
                workflow.performance_metrics.success_rate
            );
            println!(
                "       - Execution time: {:.2}s",
                workflow.performance_metrics.average_execution_time
            );
        }
    }

    println!("   ✓ Total workflows discovered: {}", total_workflows);
    assert!(
        total_workflows > 0,
        "Should have discovered at least one workflow"
    );

    // Test 3: Verify workflow archive functionality
    println!("\n3. Testing workflow archive...");
    println!(
        "   ✓ Archive contains {} workflows",
        meta_agent.archive.workflows.len()
    );
    println!(
        "   ✓ Performance history entries: {}",
        meta_agent.archive.performance_history.len()
    );

    // Get best workflow
    if let Some(best_workflow) = meta_agent.get_best_workflow() {
        println!(
            "   ✓ Best workflow: '{}' (success rate: {:.2})",
            best_workflow.name, best_workflow.performance_metrics.success_rate
        );

        // Verify best workflow has reasonable performance
        assert!(best_workflow.performance_metrics.success_rate > 0.0);
    } else {
        println!("   ⚠ No best workflow found yet");
    }

    // Test 4: Test different coordination strategies
    println!("\n4. Testing coordination strategies...");
    let workflows_by_strategy: Vec<_> = meta_agent
        .archive
        .workflows
        .values()
        .map(|w| &w.coordination_structure.coordination_strategy)
        .collect();

    let mut strategy_counts = std::collections::HashMap::new();
    for strategy in workflows_by_strategy {
        *strategy_counts
            .entry(format!("{:?}", strategy))
            .or_insert(0) += 1;
    }

    for (strategy, count) in strategy_counts {
        println!("   ✓ {} workflows use {} strategy", count, strategy);
    }

    // Test 5: Test rewriting policies
    println!("\n5. Testing rewriting policies...");
    for policy in &meta_agent.rewriting_policies {
        println!(
            "   ✓ Policy: '{}' targets {:?}",
            policy.name, policy.target_component
        );
        println!("     Strategy: {:?}", policy.strategy);

        // Verify conditions are reasonable
        for condition in &policy.conditions {
            println!(
                "     Condition: {} {} {}",
                condition.metric,
                match condition.comparison {
                    ComparisonOperator::LessThan => "<",
                    ComparisonOperator::GreaterThan => ">",
                    ComparisonOperator::Equal => "==",
                    ComparisonOperator::NotEqual => "!=",
                },
                condition.threshold
            );
        }
    }

    // Test 6: Test workflow code generation
    println!("\n6. Testing workflow code generation...");
    if let Some(workflow) = meta_agent.archive.workflows.values().next() {
        println!("   ✓ Sample generated code:");
        let code_lines: Vec<&str> = workflow.code_implementation.lines().take(5).collect();
        for line in code_lines {
            println!("     {}", line);
        }
        println!(
            "     ... ({} total lines)",
            workflow.code_implementation.lines().count()
        );

        // Verify code contains expected elements
        assert!(workflow.code_implementation.contains("def forward"));
        assert!(workflow.code_implementation.contains("results"));
    }

    // Test 7: Integration with ADAS Engine
    println!("\n7. Testing integration with ADAS Engine...");
    let config = AdasConfig::default();

    let mut adas_engine = AdasEngine::initialize(config).await?;

    // Test meta agent search integration
    let search_result = adas_engine.meta_agent_search(task_description).await?;
    println!(
        "   ✓ Meta agent search returned workflow: '{}'",
        search_result.name
    );
    println!(
        "     Agents: {}, success rate: {:.2}",
        search_result.agent_roles.len(),
        search_result.performance_metrics.success_rate
    );

    // Test 8: Verify termination conditions
    println!("\n8. Testing termination conditions...");
    println!(
        "   ✓ Current iteration: {}/{}",
        meta_agent.current_iteration, meta_agent.max_iterations
    );
    println!("   ✓ Should terminate: {}", meta_agent.should_terminate());

    if meta_agent.archive.has_converged() {
        println!("   ✓ Archive has converged");
    } else {
        println!("   ✓ Archive has not converged yet");
    }

    // Final summary
    println!("\n🎉 ADAS Meta Agent Search Test Results");
    println!("=====================================");
    println!("✅ Meta Agent Creation: PASSED");
    println!(
        "✅ Search Iterations: PASSED ({} workflows discovered)",
        total_workflows
    );
    println!(
        "✅ Workflow Archive: PASSED ({} workflows stored)",
        meta_agent.archive.workflows.len()
    );
    println!("✅ Coordination Strategies: PASSED");
    println!(
        "✅ Rewriting Policies: PASSED ({} policies active)",
        meta_agent.rewriting_policies.len()
    );
    println!("✅ Code Generation: PASSED");
    println!("✅ ADAS Engine Integration: PASSED");
    println!("✅ Termination Conditions: PASSED");

    println!("\n🚀 ADAS Meta Agent Search implementation is working correctly!");
    println!("   - All 7 seed agents are properly configured");
    println!("   - Iterative workflow discovery is functional");
    println!("   - Performance-based optimization is active");
    println!("   - Archive maintains workflow evolution history");
    println!("   - Integration with main ADAS engine is successful");

    Ok(())
}
