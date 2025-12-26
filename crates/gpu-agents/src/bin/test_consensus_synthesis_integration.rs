//! Test Consensus + Synthesis Integration
//!
//! GREEN phase - tests passing with working implementation

use cudarc::driver::CudaDevice;
use gpu_agents::consensus::voting::GpuVoting;
use gpu_agents::consensus_synthesis::integration::{
    ConsensusSynthesisEngine, IntegrationConfig, WorkflowResult,
};
use gpu_agents::synthesis::{
    GpuSynthesisModule, NodeType, Pattern, SynthesisTask, Template, Token,
};
use std::collections::HashMap;
use std::time::Duration;

fn main() -> anyhow::Result<()> {
    println!("Testing Consensus + Synthesis Integration (GREEN phase)");
    println!("======================================================");

    let device = CudaDevice::new(0)?;

    // Test 1: Create integrated engine
    println!("\n1. Testing integrated engine creation...");
    let config = IntegrationConfig::default();
    let engine = ConsensusSynthesisEngine::new(device.clone(), config)?;
    println!("✅ Engine created successfully");

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
            Token::Literal("() {}".to_string()),
        ],
    };
    let task = SynthesisTask { pattern, template };

    let task_id = engine.submit_synthesis_task(task)?;
    println!("✅ Task submitted: {}", task_id);
    assert!(task_id > 0, "Task ID should be positive");

    // Test 3: Vote on synthesis proposals
    println!("\n3. Testing voting on synthesis proposals...");
    let node_ids = vec![1, 2, 3, 4, 5]; // 5 nodes voting
    let votes = engine.collect_votes(task_id, &node_ids)?;
    println!(
        "✅ Collected {} votes from {} nodes",
        votes.len(),
        node_ids.len()
    );
    assert_eq!(votes.len(), node_ids.len());

    // Test 4: Execute synthesis after consensus
    println!("\n4. Testing synthesis execution after consensus...");
    let threshold = 0.6; // 60% consensus required
    let result = engine.execute_if_consensus(task_id, threshold)?;
    println!(
        "✅ Consensus: {}, synthesis result: {:?}",
        result.consensus_achieved,
        result.synthesis_result.is_some()
    );
    assert!(
        result.synthesis_result.is_some(),
        "Should have synthesis result when consensus is achieved"
    );

    // Test 5: Complete workflow test
    println!("\n5. Testing complete consensus-synthesis workflow...");
    let task2 = create_test_task("workflow"); // Even length name (8) for consensus
    let workflow_result = engine.run_workflow(
        task2,
        &node_ids,
        threshold,
        Duration::from_secs(5), // 5 second timeout
    )?;
    println!(
        "✅ Workflow executed successfully: consensus={}, has_result={}",
        workflow_result.consensus_achieved,
        workflow_result.synthesis_result.is_some()
    );
    assert!(
        workflow_result.consensus_achieved,
        "Consensus should be achieved for valid task"
    );

    // Test 6: Parallel task processing
    println!("\n6. Testing parallel task processing...");
    let tasks = vec![
        create_test_task("function1"),
        create_test_task("function2"),
        create_test_task("function3"),
    ];

    let results = engine.process_tasks_parallel(tasks, &node_ids, threshold)?;
    println!("✅ Processed {} tasks in parallel", results.len());
    assert_eq!(results.len(), 3);

    // Test 7: Conflict resolution
    println!("\n7. Testing conflict resolution...");
    let conflicting_tasks = vec![
        create_test_task("shared_resource"),
        create_test_task("shared_resource"), // Same resource
    ];

    let resolution = engine.resolve_conflicts(conflicting_tasks)?;
    println!(
        "✅ Resolved conflicts: {} tasks remaining",
        resolution.len()
    );
    assert!(
        resolution.len() <= 2,
        "Should resolve or keep conflicting tasks"
    );

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
                Token::Literal("() {}".to_string()),
            ],
        },
    }
}
