//! Tests for GPU Synthesis Module
//!
//! Tests pattern matching, template expansion, and AST transformations on GPU

use super::*;
use cudarc::driver::CudaContext;
use std::sync::Arc;

// Test basic pattern matching on GPU
#[test]
fn test_gpu_pattern_matching() {
    // Skip if no GPU available
    let Ok(device) = CudaContext::new(0) else {
        println!("No GPU available, skipping test");
        return;
    };
    let device = Arc::new(device);

    // Create pattern matcher
    let matcher = GpuPatternMatcher::new(device.clone(), 1024).expect("Failed to create matcher");

    // Define a simple pattern: function($x) { return $x + 1; }
    let pattern = Pattern {
        node_type: NodeType::Function,
        children: vec![
            Pattern {
                node_type: NodeType::Variable,
                children: vec![],
                value: Some("$x".to_string()),
            },
            Pattern {
                node_type: NodeType::Block,
                children: vec![Pattern {
                    node_type: NodeType::Return,
                    children: vec![Pattern {
                        node_type: NodeType::BinaryOp,
                        children: vec![
                            Pattern {
                                node_type: NodeType::Variable,
                                children: vec![],
                                value: Some("$x".to_string()),
                            },
                            Pattern {
                                node_type: NodeType::Literal,
                                children: vec![],
                                value: Some("1".to_string()),
                            },
                        ],
                        value: Some("+".to_string()),
                    }],
                    value: None,
                }],
                value: None,
            },
        ],
        value: Some("increment".to_string()),
    };

    // Create AST to match against
    let ast = AstNode {
        node_type: NodeType::Function,
        children: vec![
            AstNode {
                node_type: NodeType::Variable,
                children: vec![],
                value: Some("y".to_string()),
            },
            AstNode {
                node_type: NodeType::Block,
                children: vec![AstNode {
                    node_type: NodeType::Return,
                    children: vec![AstNode {
                        node_type: NodeType::BinaryOp,
                        children: vec![
                            AstNode {
                                node_type: NodeType::Variable,
                                children: vec![],
                                value: Some("y".to_string()),
                            },
                            AstNode {
                                node_type: NodeType::Literal,
                                children: vec![],
                                value: Some("1".to_string()),
                            },
                        ],
                        value: Some("+".to_string()),
                    }],
                    value: None,
                }],
                value: None,
            },
        ],
        value: Some("increment".to_string()),
    };

    // Match pattern against AST
    let matches = matcher
        .match_pattern(&pattern, &ast)
        .expect("Pattern matching failed");

    // Should find one match with binding $x -> y
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].bindings.get("$x"), Some(&"y".to_string()));
}

// Test template expansion on GPU
#[test]
fn test_gpu_template_expansion() {
    let Ok(device) = CudaContext::new(0) else {
        println!("No GPU available, skipping test");
        return;
    };
    let device = Arc::new(device);

    // Create template expander
    let expander =
        GpuTemplateExpander::new(device.clone(), 1024).expect("Failed to create expander");

    // Define template: "function optimize_$name($x) { return $x * 2; }"
    let template = Template {
        tokens: vec![
            Token::Literal("function optimize_".to_string()),
            Token::Variable("$name".to_string()),
            Token::Literal("(".to_string()),
            Token::Variable("$x".to_string()),
            Token::Literal(") { return ".to_string()),
            Token::Variable("$x".to_string()),
            Token::Literal(" * 2; }".to_string()),
        ],
    };

    // Define bindings
    let bindings = vec![
        ("$name".to_string(), "add".to_string()),
        ("$x".to_string(), "value".to_string()),
    ]
    .into_iter()
    .collect();

    // Expand template
    let result = expander
        .expand_template(&template, &bindings)
        .expect("Template expansion failed");

    assert_eq!(result, "function optimize_add(value) { return value * 2; }");
}

// Test AST transformation on GPU
#[test]
fn test_gpu_ast_transformation() {
    let Ok(device) = CudaContext::new(0) else {
        println!("No GPU available, skipping test");
        return;
    };
    let device = Arc::new(device);

    // Create AST transformer
    let transformer =
        GpuAstTransformer::new(device.clone(), 1024).expect("Failed to create transformer");

    // Create input AST
    let ast = AstNode {
        node_type: NodeType::Function,
        children: vec![
            AstNode {
                node_type: NodeType::Variable,
                children: vec![],
                value: Some("x".to_string()),
            },
            AstNode {
                node_type: NodeType::Block,
                children: vec![AstNode {
                    node_type: NodeType::Return,
                    children: vec![AstNode {
                        node_type: NodeType::BinaryOp,
                        children: vec![
                            AstNode {
                                node_type: NodeType::Variable,
                                children: vec![],
                                value: Some("x".to_string()),
                            },
                            AstNode {
                                node_type: NodeType::Literal,
                                children: vec![],
                                value: Some("1".to_string()),
                            },
                        ],
                        value: Some("+".to_string()),
                    }],
                    value: None,
                }],
                value: None,
            },
        ],
        value: Some("increment".to_string()),
    };

    // Define transformation rule: x + 1 -> x++
    let rule = TransformRule {
        pattern: Pattern {
            node_type: NodeType::BinaryOp,
            children: vec![
                Pattern {
                    node_type: NodeType::Variable,
                    children: vec![],
                    value: Some("$var".to_string()),
                },
                Pattern {
                    node_type: NodeType::Literal,
                    children: vec![],
                    value: Some("1".to_string()),
                },
            ],
            value: Some("+".to_string()),
        },
        replacement: AstNode {
            node_type: NodeType::UnaryOp,
            children: vec![AstNode {
                node_type: NodeType::Variable,
                children: vec![],
                value: Some("$var".to_string()),
            }],
            value: Some("++".to_string()),
        },
    };

    // Apply transformation
    let result = transformer
        .transform_ast(&ast, &rule)
        .expect("AST transformation failed");

    // Check that x + 1 was transformed to x++
    assert_eq!(
        result.children[1].children[0].children[0].node_type,
        NodeType::UnaryOp
    );
    assert_eq!(
        result.children[1].children[0].children[0].value,
        Some("++".to_string())
    );
}

// Test parallel pattern matching performance
#[test]
fn test_gpu_pattern_matching_performance() {
    let Ok(device) = CudaContext::new(0) else {
        println!("No GPU available, skipping test");
        return;
    };
    let device = Arc::new(device);

    let matcher = GpuPatternMatcher::new(device.clone(), 10_000).expect("Failed to create matcher");

    // Create a large AST forest (many independent trees)
    let forest_size = 1000;
    let ast_forest: Vec<AstNode> = (0..forest_size)
        .map(|i| AstNode {
            node_type: NodeType::Function,
            children: vec![],
            value: Some(format!("func_{}", i)),
        })
        .collect();

    // Simple pattern to match all functions
    let pattern = Pattern {
        node_type: NodeType::Function,
        children: vec![],
        value: None,
    };

    let start = std::time::Instant::now();
    let matches = matcher
        .match_pattern_parallel(&pattern, &ast_forest)
        .expect("Parallel matching failed");
    let elapsed = start.elapsed();

    assert_eq!(matches.len(), forest_size);
    println!("Matched {} patterns in {:?} on GPU", forest_size, elapsed);

    // Should be significantly faster than sequential
    assert!(
        elapsed.as_millis() < 100,
        "GPU pattern matching took {}ms, expected <100ms",
        elapsed.as_millis()
    );
}

// Test synthesis pipeline integration
#[test]
fn test_synthesis_pipeline() {
    let Ok(device) = CudaContext::new(0) else {
        println!("No GPU available, skipping test");
        return;
    };
    let device = Arc::new(device);

    // Create synthesis module
    let synthesis =
        GpuSynthesisModule::new(device, 1024).expect("Failed to create synthesis module");

    // Define synthesis task: optimize all increment functions
    let task = SynthesisTask {
        pattern: Pattern {
            node_type: NodeType::Function,
            children: vec![
                Pattern {
                    node_type: NodeType::Variable,
                    children: vec![],
                    value: Some("$param".to_string()),
                },
                Pattern {
                    node_type: NodeType::Block,
                    children: vec![Pattern {
                        node_type: NodeType::Return,
                        children: vec![Pattern {
                            node_type: NodeType::BinaryOp,
                            children: vec![
                                Pattern {
                                    node_type: NodeType::Variable,
                                    children: vec![],
                                    value: Some("$param".to_string()),
                                },
                                Pattern {
                                    node_type: NodeType::Literal,
                                    children: vec![],
                                    value: Some("1".to_string()),
                                },
                            ],
                            value: Some("+".to_string()),
                        }],
                        value: None,
                    }],
                    value: None,
                },
            ],
            value: Some("$name".to_string()),
        },
        template: Template {
            tokens: vec![
                Token::Literal("function optimized_".to_string()),
                Token::Variable("$name".to_string()),
                Token::Literal("(".to_string()),
                Token::Variable("$param".to_string()),
                Token::Literal(") { return ++".to_string()),
                Token::Variable("$param".to_string()),
                Token::Literal("; }".to_string()),
            ],
        },
    };

    // Input code AST
    let input_ast = AstNode {
        node_type: NodeType::Function,
        children: vec![
            AstNode {
                node_type: NodeType::Variable,
                children: vec![],
                value: Some("x".to_string()),
            },
            AstNode {
                node_type: NodeType::Block,
                children: vec![AstNode {
                    node_type: NodeType::Return,
                    children: vec![AstNode {
                        node_type: NodeType::BinaryOp,
                        children: vec![
                            AstNode {
                                node_type: NodeType::Variable,
                                children: vec![],
                                value: Some("x".to_string()),
                            },
                            AstNode {
                                node_type: NodeType::Literal,
                                children: vec![],
                                value: Some("1".to_string()),
                            },
                        ],
                        value: Some("+".to_string()),
                    }],
                    value: None,
                }],
                value: None,
            },
        ],
        value: Some("increment".to_string()),
    };

    // Run synthesis
    let result = synthesis
        .synthesize(&task, &input_ast)
        .expect("Synthesis failed");

    // Should generate optimized code
    assert!(result.contains("optimized_increment"));
    assert!(result.contains("++x"));
}
