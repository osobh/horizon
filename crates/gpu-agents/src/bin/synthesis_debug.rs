//! Synthesis System Debug Tool
//!
//! Isolates where the synthesis system hangs

use cudarc::driver::CudaContext;
use gpu_agents::synthesis::{
    AstNode, GpuSynthesisModule, NodeType, Pattern, SynthesisTask, Template, Token,
};
use std::sync::Arc;

fn main() -> anyhow::Result<()> {
    println!("🔍 Synthesis System Debug");
    println!("========================");

    // Step 1: Create CUDA device
    println!("\n1. Creating CUDA device...");
    let ctx = CudaContext::new(0)?;
    println!("   ✅ Device created");

    // Step 2: Create synthesis module with minimal size
    println!("\n2. Creating synthesis module (max_nodes=10)...");
    let synthesis = match GpuSynthesisModule::new(ctx.clone(), 10) {
        Ok(s) => {
            println!("   ✅ Module created");
            s
        }
        Err(e) => {
            println!("   ❌ Failed to create module: {}", e);
            return Err(e);
        }
    };

    // Step 3: Create minimal pattern
    println!("\n3. Creating minimal pattern...");
    let pattern = Pattern {
        node_type: NodeType::Variable,
        children: vec![],
        value: Some("test".to_string()),
    };
    println!("   ✅ Pattern created");

    // Step 4: Create minimal AST
    println!("\n4. Creating minimal AST...");
    let ast = AstNode {
        node_type: NodeType::Variable,
        children: vec![],
        value: Some("test".to_string()),
    };
    println!("   ✅ AST created");

    // Step 5: Create minimal template
    println!("\n5. Creating minimal template...");
    let template = Template {
        tokens: vec![Token::Literal("result".to_string())],
    };
    println!("   ✅ Template created");

    // Step 6: Create synthesis task
    println!("\n6. Creating synthesis task...");
    let task = SynthesisTask {
        pattern: pattern.clone(),
        template: template.clone(),
    };
    println!("   ✅ Task created");

    // Step 7: Test pattern matching only
    println!("\n7. Testing pattern matcher initialization...");
    {
        use gpu_agents::synthesis::pattern::GpuPatternMatcher;
        match GpuPatternMatcher::new(ctx.clone(), 10) {
            Ok(_) => println!("   ✅ Pattern matcher created"),
            Err(e) => println!("   ❌ Pattern matcher failed: {}", e),
        }
    }

    // Step 8: Test template expander only
    println!("\n8. Testing template expander initialization...");
    {
        use gpu_agents::synthesis::template::GpuTemplateExpander;
        match GpuTemplateExpander::new(ctx.clone(), 10) {
            Ok(_) => println!("   ✅ Template expander created"),
            Err(e) => println!("   ❌ Template expander failed: {}", e),
        }
    }

    // Step 9: Test AST transformer only
    println!("\n9. Testing AST transformer initialization...");
    {
        use gpu_agents::synthesis::ast::GpuAstTransformer;
        match GpuAstTransformer::new(ctx.clone(), 10) {
            Ok(_) => println!("   ✅ AST transformer created"),
            Err(e) => println!("   ❌ AST transformer failed: {}", e),
        }
    }

    // Step 10: Test synthesis with timeout
    println!("\n10. Testing synthesis (this might hang)...");
    println!("    ⏱️  Starting synthesis...");

    let start = std::time::Instant::now();
    match synthesis.synthesize(&task, &ast) {
        Ok(result) => {
            let elapsed = start.elapsed();
            println!(
                "   ✅ Synthesis completed in {:.2}ms",
                elapsed.as_secs_f64() * 1000.0
            );
            println!("   📄 Result: {}", result);
        }
        Err(e) => {
            let elapsed = start.elapsed();
            println!(
                "   ❌ Synthesis failed after {:.2}ms: {}",
                elapsed.as_secs_f64() * 1000.0,
                e
            );
        }
    }

    println!("\n✅ Debug complete!");

    Ok(())
}
