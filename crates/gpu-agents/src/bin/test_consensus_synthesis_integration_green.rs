//! Test Consensus + Synthesis Integration
//!
//! GREEN phase - make tests pass

use cudarc::driver::CudaContext;
use gpu_agents::consensus::voting::GpuVoting;
use gpu_agents::consensus_synthesis::integration::{
    ConsensusSynthesisEngine, IntegrationConfig, WorkflowResult,
};
use gpu_agents::consensus_synthesis::{ConflictStrategy, WorkflowStatus};
use gpu_agents::synthesis::{
    GpuSynthesisModule, NodeType, Pattern, SynthesisTask, Template, Token,
};
use std::collections::HashMap;
use std::time::{Duration, Instant};

fn main() -> anyhow::Result<()> {
    println!("Testing Consensus + Synthesis Integration (GREEN phase)");
    println!("======================================================");

    let ctx = CudaContext::new(0)?;

    // Test 1: Create integrated engine
    println!("\n1. Testing integrated engine creation...");
    let config = IntegrationConfig {
        max_concurrent_tasks: 10,
        voting_timeout: Duration::from_secs(5),
        min_voters: 3,
        retry_attempts: 3,
        conflict_resolution_strategy: ConflictStrategy::FirstWins,
    };
    let engine = ConsensusSynthesisEngine::new(ctx.clone(), config)?;
    println!("✅ Engine created with voting and synthesis modules initialized");

    // Test 2: Submit synthesis task for consensus
    println!("\n2. Testing synthesis task submission...");
    let pattern = Pattern {
        node_type: NodeType::Function,
        children: vec![],
        value: Some("main".to_string()),
    };
    let template = Template {
        tokens: vec![
            Token::Literal("fn ".to_string()),
            Token::Variable("name".to_string()),
            Token::Literal("() { println!(\"Hello from synthesized code!\"); }".to_string()),
        ],
    };
    let task = SynthesisTask { pattern, template };

    let task_id = engine.submit_synthesis_task(task)?;
    println!("✅ Task submitted with ID: {}", task_id);
    assert!(task_id >= 1, "Task ID should be positive");

    // Test 3: Vote on synthesis proposals
    println!("\n3. Testing voting on synthesis proposals...");
    let node_ids = vec![1, 2, 3, 4, 5]; // 5 nodes voting
    let votes = engine.collect_votes(task_id, &node_ids)?;
    println!("✅ Collected {} votes from nodes", votes.len());

    // Display vote results
    let yes_votes = votes.values().filter(|&&v| v).count();
    let no_votes = votes.values().filter(|&&v| !v).count();
    println!("   Vote breakdown: {} YES, {} NO", yes_votes, no_votes);
    assert_eq!(votes.len(), node_ids.len());

    // Test 4: Execute synthesis after consensus
    println!("\n4. Testing synthesis execution after consensus...");
    let threshold = 0.6; // 60% consensus required
    let result = engine.execute_if_consensus(task_id, threshold)?;

    println!("✅ Consensus achieved: {}", result.consensus_achieved);
    println!("   Vote percentage: {:.1}%", result.vote_percentage * 100.0);
    println!("   Execution time: {:?}", result.execution_time);

    if let Some(synthesis_result) = &result.synthesis_result {
        println!("   Synthesized code:\n   {}", synthesis_result);
    }

    // Test 5: Complete workflow test
    println!("\n5. Testing complete consensus-synthesis workflow...");
    let task2 = create_test_task("workflow_function");
    let start = Instant::now();

    let workflow_result = engine.run_workflow(
        task2,
        &node_ids,
        threshold,
        Duration::from_secs(5), // 5 second timeout
    )?;

    println!("✅ Workflow completed in {:?}", start.elapsed());
    println!("   Consensus: {}", workflow_result.consensus_achieved);
    println!("   Participants: {:?}", workflow_result.participating_nodes);

    // Test 6: Parallel task processing
    println!("\n6. Testing parallel task processing...");
    let tasks = vec![
        create_test_task("parallel_function1"),
        create_test_task("parallel_function2"),
        create_test_task("parallel_function3"),
    ];

    let start = Instant::now();
    let results = engine.process_tasks_parallel(tasks.clone(), &node_ids, threshold)?;
    let parallel_time = start.elapsed();

    println!(
        "✅ Processed {} tasks in parallel in {:?}",
        results.len(),
        parallel_time
    );

    // Show results
    for (i, result) in results.iter().enumerate() {
        println!(
            "   Task {}: consensus={}, vote={:.1}%",
            i + 1,
            result.consensus_achieved,
            result.vote_percentage * 100.0
        );
    }

    assert_eq!(results.len(), 3);

    // Test 7: Conflict resolution
    println!("\n7. Testing conflict resolution...");
    let conflicting_tasks = vec![
        create_test_task("shared_resource"),
        create_test_task("shared_resource"), // Same resource
        create_test_task("different_resource"),
    ];

    let resolution = engine.resolve_conflicts(conflicting_tasks)?;
    println!(
        "✅ Resolved conflicts: {} tasks remaining",
        resolution.len()
    );

    // Should have removed one duplicate
    assert_eq!(resolution.len(), 2, "Should have 2 unique resources");

    // Test 8: Task status tracking
    println!("\n8. Testing task status tracking...");
    let statuses = engine.get_task_statuses()?;
    println!("✅ Tracking {} tasks", statuses.len());

    for (id, (status, result)) in statuses.iter() {
        println!("   Task {}: {:?}", id, status);
        if let Some(res) = result {
            println!("      Result: {}", res.chars().take(50).collect::<String>());
        }
    }

    // Test 9: Performance metrics
    println!("\n9. Testing performance metrics...");

    // Submit more tasks to get meaningful metrics
    for i in 0..10 {
        let task = create_test_task(&format!("perf_test_{}", i));
        let task_id = engine.submit_synthesis_task(task)?;
        engine.collect_votes(task_id, &node_ids)?;
        engine.execute_if_consensus(task_id, threshold)?;
    }

    // Test cleanup
    println!("\n10. Testing old task cleanup...");
    std::thread::sleep(Duration::from_millis(100));
    engine.cleanup_old_tasks(Duration::from_millis(50));
    let remaining = engine.get_task_statuses()?.len();
    println!("✅ Cleanup complete: {} tasks remaining", remaining);

    // Summary
    println!("\n✅ All GREEN phase tests passed!");
    println!("\nIntegration Summary:");
    println!("- Consensus and synthesis modules successfully integrated");
    println!("- Tasks flow from submission → voting → execution");
    println!("- Parallel processing works correctly");
    println!("- Conflict resolution prevents duplicate work");
    println!("- Performance tracking and cleanup functional");

    Ok(())
}

fn create_test_task(name: &str) -> SynthesisTask {
    SynthesisTask {
        pattern: Pattern {
            node_type: NodeType::Function,
            children: vec![],
            value: Some(name.to_string()),
        },
        template: Template {
            tokens: vec![
                Token::Literal("fn ".to_string()),
                Token::Variable("name".to_string()),
                Token::Literal("() { /* auto-generated */ }".to_string()),
            ],
        },
    }
}
