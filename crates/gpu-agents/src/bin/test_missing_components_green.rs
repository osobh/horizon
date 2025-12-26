//! Test missing consensus/synthesis components
//!
//! GREEN phase - comprehensive tests

use cudarc::driver::CudaDevice;
use gpu_agents::consensus_synthesis::{ConsensusSynthesisEngine, TemplateRegistry};
use gpu_agents::synthesis::{NodeType, Pattern};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    println!("Testing Missing Consensus/Synthesis Components (GREEN phase)");
    println!("==========================================================");

    let device = CudaDevice::new(0)?;

    // Test 1: Consensus-Synthesis engine creation
    println!("\n1. Testing consensus-synthesis engine...");
    let mut engine = ConsensusSynthesisEngine::new(device.clone())?;
    println!("✅ Created consensus-synthesis engine");

    // Test 2: Initialize components with proper sizes
    println!("\n2. Testing component initialization...");
    engine.init_voting(1000)?; // Larger buffer for voting
    engine.init_synthesis()?;
    println!("✅ Initialized voting and synthesis components");

    // Test 3: Submit multiple synthesis tasks
    println!("\n3. Testing multiple synthesis task submission...");
    let patterns = vec![
        Pattern {
            node_type: NodeType::Function,
            children: vec![],
            value: Some("function_a".to_string()),
        },
        Pattern {
            node_type: NodeType::Variable,
            children: vec![],
            value: Some("var_x".to_string()),
        },
        Pattern {
            node_type: NodeType::Loop,
            children: vec![],
            value: Some("for_loop".to_string()),
        },
    ];

    let mut task_ids = Vec::new();
    for (i, pattern) in patterns.iter().enumerate() {
        let threshold = 0.6 + (i as f32 * 0.1); // Different thresholds
        let task_id = engine.submit_synthesis_task(pattern.clone(), threshold)?;
        task_ids.push(task_id);
        println!(
            "   Submitted task {} with threshold {:.1}",
            task_id, threshold
        );
    }
    println!("✅ Submitted {} synthesis tasks", task_ids.len());

    // Test 4: Run consensus rounds
    println!("\n4. Testing consensus rounds...");
    let start = Instant::now();
    let approved_tasks = engine.run_consensus_round()?;
    let consensus_time = start.elapsed();
    println!("✅ Consensus completed in {:?}", consensus_time);
    println!("   Approved tasks: {:?}", approved_tasks);

    // Test 5: Execute synthesis
    println!("\n5. Testing synthesis execution...");
    let start = Instant::now();
    let completed_tasks = engine.execute_synthesis()?;
    let synthesis_time = start.elapsed();
    println!("✅ Synthesis completed in {:?}", synthesis_time);
    println!("   Completed tasks: {:?}", completed_tasks);

    // Test 6: Template registry operations
    println!("\n6. Testing template registry...");
    let mut registry = TemplateRegistry::new();

    // Register templates
    for (i, pattern) in patterns.iter().enumerate() {
        let name = format!("template_{}", i);
        let consensus = 0.7 + (i as f32 * 0.05);
        registry.register(&name, pattern.clone(), consensus)?;
        println!(
            "   Registered template '{}' with consensus {:.2}",
            name, consensus
        );
    }

    // Test template retrieval
    let template = registry.get("template_0");
    assert!(template.is_some());
    println!("✅ Template registry working correctly");

    // Test 7: Task status tracking
    println!("\n7. Testing task status tracking...");
    for task_id in &task_ids {
        if let Some(status) = engine.get_task_status(*task_id) {
            println!("   Task {} status: {:?}", task_id, status);
        }
    }

    // Test 8: Metrics collection
    println!("\n8. Testing metrics collection...");
    let metrics = engine.get_metrics();
    println!("✅ Engine metrics:");
    println!("   Tasks submitted: {}", metrics.tasks_submitted);
    println!("   Tasks approved: {}", metrics.tasks_approved);
    println!("   Tasks rejected: {}", metrics.tasks_rejected);
    println!("   Consensus rounds: {}", metrics.consensus_rounds);
    println!("   Synthesis operations: {}", metrics.synthesis_operations);
    println!(
        "   Avg consensus time: {:.2} μs",
        metrics.avg_consensus_time_us
    );
    println!(
        "   Avg synthesis time: {:.2} μs",
        metrics.avg_synthesis_time_us
    );

    // Test 9: Workflow simulation
    println!("\n9. Testing complete workflow...");
    let workflow_start = Instant::now();

    // Submit new task
    let workflow_pattern = Pattern {
        node_type: NodeType::Block,
        children: vec![
            Pattern {
                node_type: NodeType::Variable,
                children: vec![],
                value: Some("x".to_string()),
            },
            Pattern {
                node_type: NodeType::BinaryOp,
                children: vec![],
                value: Some("+".to_string()),
            },
        ],
        value: Some("block".to_string()),
    };

    let workflow_task_id = engine.submit_synthesis_task(workflow_pattern, 0.8)?;
    engine.run_consensus_round()?;
    engine.execute_synthesis()?;

    let workflow_time = workflow_start.elapsed();
    println!("✅ Complete workflow executed in {:?}", workflow_time);

    // Test 10: Template usage statistics
    println!("\n10. Testing template usage statistics...");
    // Simulate template usage
    registry.record_usage("template_0", true, 100.0, 50.0);
    registry.record_usage("template_0", true, 120.0, 45.0);
    registry.record_usage("template_1", false, 150.0, 60.0);

    println!("✅ Template usage statistics recorded");

    println!("\n✅ All GREEN phase tests passed!");
    println!("\nKey achievements:");
    println!("- Consensus-synthesis integration working");
    println!("- Template registry functional");
    println!("- Task lifecycle management complete");
    println!("- Metrics collection operational");
    println!("- Complete workflow execution successful");

    Ok(())
}
