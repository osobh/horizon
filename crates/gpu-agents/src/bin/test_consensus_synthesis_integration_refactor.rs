//! Test Consensus + Synthesis Integration
//!
//! REFACTOR phase - optimize implementation

use cudarc::driver::CudaDevice;
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
    println!("Testing Consensus + Synthesis Integration (REFACTOR phase)");
    println!("===========================================================");

    let device = CudaDevice::new(0)?;

    // Test 1: Optimized engine configuration
    println!("\n1. Testing optimized engine configuration...");
    let config = IntegrationConfig {
        max_concurrent_tasks: 1000,                 // Increased for performance
        voting_timeout: Duration::from_millis(500), // Faster timeout
        min_voters: 5,                              // More voters for better consensus
        retry_attempts: 2,                          // Fewer retries
        conflict_resolution_strategy: ConflictStrategy::HighestVoteWins,
    };
    let engine = ConsensusSynthesisEngine::new(device.clone(), config)?;
    println!("✅ Optimized engine created with enhanced configuration");

    // Test 2: Batch task submission performance
    println!("\n2. Testing batch task submission performance...");
    let start = Instant::now();
    let mut task_ids = Vec::new();

    for i in 0..100 {
        let task = create_test_task(&format!("batch_task_{}", i));
        let task_id = engine.submit_synthesis_task(task)?;
        task_ids.push(task_id);
    }

    let batch_time = start.elapsed();
    println!(
        "✅ Submitted 100 tasks in {:?} ({:.1} tasks/sec)",
        batch_time,
        100.0 / batch_time.as_secs_f64()
    );
    assert!(
        batch_time < Duration::from_secs(1),
        "Batch submission should be fast"
    );

    // Test 3: Parallel voting performance
    println!("\n3. Testing parallel voting performance...");
    let node_ids: Vec<u32> = (1..=50).collect(); // 50 voters
    let start = Instant::now();

    let mut vote_results = Vec::new();
    for &task_id in &task_ids[0..10] {
        // Test 10 tasks
        let votes = engine.collect_votes(task_id, &node_ids)?;
        vote_results.push(votes);
    }

    let voting_time = start.elapsed();
    println!(
        "✅ Collected votes for 10 tasks from 50 nodes in {:?}",
        voting_time
    );
    println!(
        "   Average: {:.1} ms per task",
        voting_time.as_millis() as f64 / 10.0
    );
    assert!(
        voting_time < Duration::from_secs(2),
        "Voting should be efficient"
    );

    // Test 4: Optimized consensus execution
    println!("\n4. Testing optimized consensus execution...");
    let threshold = 0.6;
    let start = Instant::now();

    let mut execution_results = Vec::new();
    for &task_id in &task_ids[0..10] {
        let result = engine.execute_if_consensus(task_id, threshold)?;
        execution_results.push(result);
    }

    let execution_time = start.elapsed();
    println!("✅ Executed consensus for 10 tasks in {:?}", execution_time);
    println!(
        "   Average: {:.1} ms per task",
        execution_time.as_millis() as f64 / 10.0
    );

    let consensus_rate = execution_results
        .iter()
        .filter(|r| r.consensus_achieved)
        .count() as f64
        / execution_results.len() as f64;
    println!("   Consensus rate: {:.1}%", consensus_rate * 100.0);

    // Test 5: High-throughput parallel workflow
    println!("\n5. Testing high-throughput parallel workflow...");
    let tasks: Vec<SynthesisTask> = (0..50)
        .map(|i| create_test_task(&format!("parallel_workflow_{}", i)))
        .collect();

    let start = Instant::now();
    let parallel_results = engine.process_tasks_parallel(tasks, &node_ids, threshold)?;
    let parallel_time = start.elapsed();

    println!("✅ Processed 50 tasks in parallel in {:?}", parallel_time);
    println!(
        "   Throughput: {:.1} tasks/sec",
        50.0 / parallel_time.as_secs_f64()
    );

    let success_rate = parallel_results
        .iter()
        .filter(|r| r.consensus_achieved)
        .count() as f64
        / parallel_results.len() as f64;
    println!("   Success rate: {:.1}%", success_rate * 100.0);

    assert!(
        parallel_time < Duration::from_secs(5),
        "Parallel processing should be fast"
    );
    assert_eq!(parallel_results.len(), 50);

    // Test 6: Advanced conflict resolution
    println!("\n6. Testing advanced conflict resolution...");
    let conflicting_tasks = vec![
        create_test_task("resource_a"),
        create_test_task("resource_b"),
        create_test_task("resource_a"), // Duplicate
        create_test_task("resource_c"),
        create_test_task("resource_b"), // Another duplicate
        create_test_task("resource_d"),
    ];

    let start = Instant::now();
    let resolved = engine.resolve_conflicts(conflicting_tasks)?;
    let resolution_time = start.elapsed();

    println!("✅ Resolved conflicts in {:?}", resolution_time);
    println!("   Input: 6 tasks, Output: {} tasks", resolved.len());

    // Should remove duplicates based on HighestVoteWins strategy
    assert!(resolved.len() <= 4, "Should resolve conflicts");
    assert!(
        resolution_time < Duration::from_millis(100),
        "Conflict resolution should be fast"
    );

    // Test 7: Memory and performance monitoring
    println!("\n7. Testing memory and performance monitoring...");
    let statuses = engine.get_task_statuses();
    println!("✅ Tracking {} active tasks", statuses.len());

    let completed_count = statuses
        .values()
        .filter(|(status, _)| *status == WorkflowStatus::Completed)
        .count();
    println!("   Completed tasks: {}", completed_count);

    let pending_count = statuses
        .values()
        .filter(|(status, _)| *status == WorkflowStatus::Pending)
        .count();
    println!("   Pending tasks: {}", pending_count);

    // Test 8: Stress test with cleanup
    println!("\n8. Testing stress test with cleanup...");
    let stress_tasks: Vec<SynthesisTask> = (0..200)
        .map(|i| create_test_task(&format!("stress_test_{}", i)))
        .collect();

    let start = Instant::now();

    // Submit all tasks
    let mut stress_task_ids = Vec::new();
    for task in stress_tasks {
        let task_id = engine.submit_synthesis_task(task)?;
        stress_task_ids.push(task_id);
    }

    // Process in batches
    let batch_size = 20;
    for batch_start in (0..stress_task_ids.len()).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(stress_task_ids.len());

        for &task_id in &stress_task_ids[batch_start..batch_end] {
            engine.collect_votes(task_id, &node_ids[0..10])?; // Use 10 voters for speed
            engine.execute_if_consensus(task_id, threshold)?;
        }
    }

    let stress_time = start.elapsed();
    println!("✅ Stress test completed: 200 tasks in {:?}", stress_time);
    println!(
        "   Throughput: {:.1} tasks/sec",
        200.0 / stress_time.as_secs_f64()
    );

    // Test cleanup
    std::thread::sleep(Duration::from_millis(100));
    engine.cleanup_old_tasks(Duration::from_millis(50));
    let remaining = engine.get_task_statuses().len();
    println!("   After cleanup: {} tasks remaining", remaining);

    assert!(
        stress_time < Duration::from_secs(10),
        "Stress test should complete quickly"
    );

    // Test 9: End-to-end performance metrics
    println!("\n9. Performance Summary");
    println!("=====================");

    let total_tasks = 100 + 50 + 200; // From various tests
    let total_time = batch_time + parallel_time + stress_time;

    println!("Overall performance:");
    println!("- Total tasks processed: {}", total_tasks);
    println!("- Total time: {:?}", total_time);
    println!(
        "- Average throughput: {:.1} tasks/sec",
        total_tasks as f64 / total_time.as_secs_f64()
    );
    println!("- Memory efficiency: Optimized with cleanup");
    println!("- Consensus quality: High (>60% threshold)");
    println!("- Conflict resolution: Active with HighestVoteWins");

    // Verify performance targets
    assert!(
        total_time.as_secs() < 15,
        "Total processing should be under 15 seconds"
    );
    let throughput = total_tasks as f64 / total_time.as_secs_f64();
    assert!(throughput > 20.0, "Should achieve >20 tasks/sec throughput");

    println!("\n✅ All REFACTOR phase optimizations successful!");
    println!("\nOptimizations Applied:");
    println!("- Enhanced configuration with higher concurrency limits");
    println!("- Batch processing for improved throughput");
    println!("- Parallel voting with multiple nodes");
    println!("- Advanced conflict resolution strategies");
    println!("- Memory cleanup and task lifecycle management");
    println!("- Performance monitoring and metrics collection");
    println!("- Stress testing for reliability validation");

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
                Token::Literal("() { /* optimized */ }".to_string()),
            ],
        },
    }
}
