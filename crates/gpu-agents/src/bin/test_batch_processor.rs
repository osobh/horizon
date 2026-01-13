//! Test batch processor implementation

use cudarc::driver::CudaContext;
use gpu_agents::synthesis::batch_processor::{BatchConfig, BatchProcessor};
use gpu_agents::synthesis::{AstNode, NodeType, Pattern};

fn main() -> anyhow::Result<()> {
    println!("Testing Batch Processor");
    println!("======================");

    // Test 1: Create processor
    let ctx = CudaContext::new(0)?;
    let config = BatchConfig::default();
    let processor = BatchProcessor::new(ctx, config)?;
    println!("✅ Processor created");

    // Test 2: Try single batch (should panic with todo!)
    let patterns = vec![Pattern {
        node_type: NodeType::Variable,
        children: vec![],
        value: Some("x".to_string()),
    }];

    let asts = vec![AstNode {
        node_type: NodeType::Variable,
        children: vec![],
        value: Some("x".to_string()),
    }];

    println!("Testing single batch processing...");
    match processor.process_single_batch(&patterns, &asts) {
        Ok(_) => println!("✅ Single batch processed"),
        Err(e) => println!("❌ Expected todo! error: {}", e),
    }

    Ok(())
}
