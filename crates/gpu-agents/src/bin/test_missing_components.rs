//! Test to find and implement missing consensus/synthesis components
//!
//! RED phase - identify missing components

use cudarc::driver::CudaDevice;
use gpu_agents::consensus_synthesis::{ConsensusSynthesisEngine, TemplateRegistry};
use gpu_agents::synthesis::{NodeType, Pattern};

fn main() -> anyhow::Result<()> {
    println!("Finding and Testing Missing Consensus/Synthesis Components (RED phase)");
    println!("===================================================================");

    let device = CudaDevice::new(0)?;

    // Test 1: Consensus-Synthesis engine creation
    println!("\n1. Testing consensus-synthesis engine...");
    let mut engine = ConsensusSynthesisEngine::new(device.clone())?;
    println!("✅ Created consensus-synthesis engine");

    // Test 2: Initialize voting and synthesis components
    println!("\n2. Testing component initialization...");
    engine.init_voting(100)?;
    engine.init_synthesis()?;
    println!("✅ Initialized voting and synthesis components");

    // Test 3: Submit synthesis task for consensus
    println!("\n3. Testing synthesis task submission...");
    let pattern = Pattern {
        node_type: NodeType::Function,
        children: vec![],
        value: Some("test_function".to_string()),
    };
    let task_id = engine.submit_synthesis_task(pattern.clone(), 0.75)?;
    println!("✅ Submitted synthesis task with ID: {}", task_id);

    // Test 4: Run consensus round
    println!("\n4. Testing consensus round...");
    let approved_tasks = engine.run_consensus_round()?;
    println!("✅ Consensus approved {} tasks", approved_tasks.len());

    // Test 5: Execute synthesis
    println!("\n5. Testing synthesis execution...");
    let completed_tasks = engine.execute_synthesis()?;
    println!("✅ Completed {} synthesis tasks", completed_tasks.len());

    // Test 6: Template registry
    println!("\n6. Testing template registry...");
    let mut registry = TemplateRegistry::new();
    registry.register("function_template", pattern.clone(), 0.8)?;
    println!("✅ Registered synthesis template");

    // Test 7: Get metrics
    println!("\n7. Testing metrics collection...");
    let metrics = engine.get_metrics();
    println!("✅ Metrics collected:");
    println!("   Tasks submitted: {}", metrics.tasks_submitted);
    println!("   Tasks approved: {}", metrics.tasks_approved);
    println!("   Consensus rounds: {}", metrics.consensus_rounds);

    Ok(())
}
