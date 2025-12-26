//! Test synthesis encoding to find the issue

use cudarc::driver::CudaDevice;
use gpu_agents::synthesis::pattern::GpuPatternMatcher;
use gpu_agents::synthesis::{AstNode, NodeType, Pattern};
use std::sync::Arc;

fn main() -> anyhow::Result<()> {
    println!("🧪 Synthesis Encoding Test");
    println!("==========================");

    // Initialize CUDA
    let device = CudaDevice::new(0)?;
    println!("✅ CUDA device initialized");

    // Create pattern matcher
    let matcher = GpuPatternMatcher::new(device, 100)?;
    println!("✅ Pattern matcher created");

    // Test 1: Simple pattern and AST
    println!("\n1. Testing simple pattern matching...");
    {
        let pattern = Pattern {
            node_type: NodeType::Variable,
            children: vec![],
            value: Some("x".to_string()),
        };

        let ast = AstNode {
            node_type: NodeType::Variable,
            children: vec![],
            value: Some("x".to_string()),
        };

        println!("   📊 Pattern: Variable 'x' with no children");
        println!("   📊 AST: Variable 'x' with no children");
        println!("   🚀 Matching...");

        match matcher.match_pattern(&pattern, &ast) {
            Ok(matches) => {
                println!("   ✅ Match completed! Found {} matches", matches.len());
                for (i, m) in matches.iter().enumerate() {
                    println!("      Match {}: node_id={}", i, m.node_id);
                }
            }
            Err(e) => println!("   ❌ Match failed: {}", e),
        }
    }

    // Test 2: Pattern with children
    println!("\n2. Testing pattern with children...");
    {
        let pattern = Pattern {
            node_type: NodeType::Function,
            children: vec![Pattern {
                node_type: NodeType::Variable,
                children: vec![],
                value: Some("x".to_string()),
            }],
            value: Some("f".to_string()),
        };

        let ast = AstNode {
            node_type: NodeType::Function,
            children: vec![AstNode {
                node_type: NodeType::Variable,
                children: vec![],
                value: Some("x".to_string()),
            }],
            value: Some("f".to_string()),
        };

        println!("   📊 Pattern: Function 'f' with Variable 'x' child");
        println!("   📊 AST: Function 'f' with Variable 'x' child");
        println!("   🚀 Matching...");

        match matcher.match_pattern(&pattern, &ast) {
            Ok(matches) => {
                println!("   ✅ Match completed! Found {} matches", matches.len());
                for (i, m) in matches.iter().enumerate() {
                    println!("      Match {}: node_id={}", i, m.node_id);
                }
            }
            Err(e) => println!("   ❌ Match failed: {}", e),
        }
    }

    // Test 3: Complex nested pattern
    println!("\n3. Testing complex nested pattern...");
    {
        let pattern = Pattern {
            node_type: NodeType::BinaryOp,
            children: vec![
                Pattern {
                    node_type: NodeType::Variable,
                    children: vec![],
                    value: Some("a".to_string()),
                },
                Pattern {
                    node_type: NodeType::Variable,
                    children: vec![],
                    value: Some("b".to_string()),
                },
            ],
            value: Some("+".to_string()),
        };

        let ast = AstNode {
            node_type: NodeType::BinaryOp,
            children: vec![
                AstNode {
                    node_type: NodeType::Variable,
                    children: vec![],
                    value: Some("a".to_string()),
                },
                AstNode {
                    node_type: NodeType::Variable,
                    children: vec![],
                    value: Some("b".to_string()),
                },
            ],
            value: Some("+".to_string()),
        };

        println!("   📊 Pattern: BinaryOp '+' with Variables 'a' and 'b'");
        println!("   📊 AST: BinaryOp '+' with Variables 'a' and 'b'");
        println!("   🚀 Matching...");

        match matcher.match_pattern(&pattern, &ast) {
            Ok(matches) => {
                println!("   ✅ Match completed! Found {} matches", matches.len());
                for (i, m) in matches.iter().enumerate() {
                    println!("      Match {}: node_id={}", i, m.node_id);
                }
            }
            Err(e) => println!("   ❌ Match failed: {}", e),
        }
    }

    println!("\n✅ All tests completed!");
    Ok(())
}
