//! Tests for multi-hop reasoning in knowledge graphs
//!
//! Tests reasoning capabilities including:
//! - Multi-hop query execution
//! - Inference rule application
//! - Logical reasoning chains
//! - Probabilistic reasoning

use super::reasoning::*;
use crate::knowledge::{KnowledgeEdge, KnowledgeNode};
use anyhow::Result;
use cudarc::driver::CudaDevice;
use std::sync::Arc;

// Helper to create test device
fn create_test_device() -> Result<Arc<CudaDevice>> {
    Ok(CudaDevice::new(0)?)
}

// Helper to create test knowledge base
fn create_test_knowledge_base() -> Vec<KnowledgeNode> {
    vec![
        KnowledgeNode {
            id: 1,
            content: "Bird".to_string(),
            node_type: "concept".to_string(),
            embedding: vec![0.1; 128],
        },
        KnowledgeNode {
            id: 2,
            content: "Can fly".to_string(),
            node_type: "property".to_string(),
            embedding: vec![0.2; 128],
        },
        KnowledgeNode {
            id: 3,
            content: "Penguin".to_string(),
            node_type: "concept".to_string(),
            embedding: vec![0.3; 128],
        },
        KnowledgeNode {
            id: 4,
            content: "Cannot fly".to_string(),
            node_type: "property".to_string(),
            embedding: vec![0.4; 128],
        },
        KnowledgeNode {
            id: 5,
            content: "Animal".to_string(),
            node_type: "concept".to_string(),
            embedding: vec![0.5; 128],
        },
    ]
}

// =============================================================================
// Unit Tests
// =============================================================================

#[test]
fn test_inference_rule_creation() {
    let rule = InferenceRule {
        id: 1,
        name: "Transitivity".to_string(),
        pattern: RulePattern::Transitive {
            relation_type: "is_a".to_string(),
        },
        confidence: 0.9,
        priority: 10,
    };

    assert_eq!(rule.name, "Transitivity");
    assert_eq!(rule.confidence, 0.9);
    assert_eq!(rule.priority, 10);
}

#[test]
fn test_reasoning_engine_creation() -> Result<()> {
    let device = create_test_device()?;
    let engine = ReasoningEngine::new(device, 1000)?;

    assert_eq!(engine.rule_count(), 0);
    assert_eq!(engine.fact_count(), 0);

    Ok(())
}

#[test]
fn test_add_inference_rules() -> Result<()> {
    let device = create_test_device()?;
    let mut engine = ReasoningEngine::new(device, 1000)?;

    // Add transitivity rule
    let rule1 = InferenceRule {
        id: 1,
        name: "Transitivity".to_string(),
        pattern: RulePattern::Transitive {
            relation_type: "is_a".to_string(),
        },
        confidence: 0.95,
        priority: 10,
    };

    // Add inheritance rule
    let rule2 = InferenceRule {
        id: 2,
        name: "Property Inheritance".to_string(),
        pattern: RulePattern::PropertyInheritance {
            relation_type: "is_a".to_string(),
            property_type: "has_property".to_string(),
        },
        confidence: 0.85,
        priority: 8,
    };

    engine.add_rule(rule1)?;
    engine.add_rule(rule2)?;

    assert_eq!(engine.rule_count(), 2);

    Ok(())
}

#[test]
fn test_add_facts() -> Result<()> {
    let device = create_test_device()?;
    let mut engine = ReasoningEngine::new(device, 1000)?;

    // Add facts
    let fact1 = LogicalFact {
        subject: 3, // Penguin
        predicate: "is_a".to_string(),
        object: FactObject::Node(1), // Bird
        confidence: 1.0,
    };

    let fact2 = LogicalFact {
        subject: 1, // Bird
        predicate: "is_a".to_string(),
        object: FactObject::Node(5), // Animal
        confidence: 1.0,
    };

    engine.add_fact(fact1)?;
    engine.add_fact(fact2)?;

    assert_eq!(engine.fact_count(), 2);

    Ok(())
}

#[test]
fn test_simple_reasoning_query() -> Result<()> {
    let device = create_test_device()?;
    let mut engine = ReasoningEngine::new(device, 1000)?;

    // Setup knowledge base
    engine.add_fact(LogicalFact {
        subject: 3, // Penguin
        predicate: "is_a".to_string(),
        object: FactObject::Node(1), // Bird
        confidence: 1.0,
    })?;

    engine.add_fact(LogicalFact {
        subject: 1, // Bird
        predicate: "has_property".to_string(),
        object: FactObject::Node(2), // Can fly
        confidence: 0.9,
    })?;

    // Query: What properties does a penguin have?
    let query = ReasoningQuery {
        query_type: QueryType::WhatProperties { entity_id: 3 },
        max_hops: 2,
        min_confidence: 0.7,
        include_inferred: true,
    };

    let results = engine.reason(query)?;

    assert!(!results.conclusions.is_empty());
    // Should infer that penguin might be able to fly (through bird)

    Ok(())
}

#[test]
fn test_multi_hop_reasoning() -> Result<()> {
    let device = create_test_device()?;
    let mut engine = ReasoningEngine::new(device, 1000)?;

    // Add transitivity rule
    engine.add_rule(InferenceRule {
        id: 1,
        name: "Transitivity".to_string(),
        pattern: RulePattern::Transitive {
            relation_type: "is_a".to_string(),
        },
        confidence: 0.95,
        priority: 10,
    })?;

    // Setup chain: Penguin -> Bird -> Animal
    engine.add_fact(LogicalFact {
        subject: 3, // Penguin
        predicate: "is_a".to_string(),
        object: FactObject::Node(1), // Bird
        confidence: 1.0,
    })?;

    engine.add_fact(LogicalFact {
        subject: 1, // Bird
        predicate: "is_a".to_string(),
        object: FactObject::Node(5), // Animal
        confidence: 1.0,
    })?;

    // Query: Is penguin an animal?
    let query = ReasoningQuery {
        query_type: QueryType::IsRelated {
            subject_id: 3,
            predicate: "is_a".to_string(),
            object_id: 5,
        },
        max_hops: 3,
        min_confidence: 0.8,
        include_inferred: true,
    };

    let results = engine.reason(query)?;

    assert!(!results.conclusions.is_empty());
    assert_eq!(results.conclusions[0].object, FactObject::Node(5));
    assert!(results.conclusions[0].confidence > 0.9); // High confidence through transitivity

    Ok(())
}

#[test]
fn test_contradiction_detection() -> Result<()> {
    let device = create_test_device()?;
    let mut engine = ReasoningEngine::new(device, 1000)?;

    // Add conflicting facts
    engine.add_fact(LogicalFact {
        subject: 3, // Penguin
        predicate: "has_property".to_string(),
        object: FactObject::Node(2), // Can fly
        confidence: 0.3,
    })?;

    engine.add_fact(LogicalFact {
        subject: 3, // Penguin
        predicate: "has_property".to_string(),
        object: FactObject::Node(4), // Cannot fly
        confidence: 0.9,
    })?;

    // Add contradiction detection rule
    engine.add_rule(InferenceRule {
        id: 1,
        name: "Contradiction Detection".to_string(),
        pattern: RulePattern::Contradiction {
            predicate: "has_property".to_string(),
            conflicting_values: vec![2, 4], // Can fly vs Cannot fly
        },
        confidence: 1.0,
        priority: 20,
    })?;

    // Check for contradictions
    let contradictions = engine.find_contradictions()?;

    assert!(!contradictions.is_empty());
    assert_eq!(contradictions[0].entity_id, 3);
    assert!(contradictions[0].confidence_difference > 0.5);

    Ok(())
}

#[test]
fn test_probabilistic_reasoning() -> Result<()> {
    let device = create_test_device()?;
    let mut engine = ReasoningEngine::new(device, 1000)?;

    // Add probabilistic facts
    engine.add_fact(LogicalFact {
        subject: 1, // Bird
        predicate: "has_property".to_string(),
        object: FactObject::Node(2), // Can fly
        confidence: 0.85,            // Most birds can fly
    })?;

    engine.add_fact(LogicalFact {
        subject: 3, // Penguin
        predicate: "is_a".to_string(),
        object: FactObject::Node(1), // Bird
        confidence: 1.0,
    })?;

    // Add exception
    engine.add_fact(LogicalFact {
        subject: 3, // Penguin
        predicate: "has_property".to_string(),
        object: FactObject::Node(4), // Cannot fly
        confidence: 0.95,            // High confidence exception
    })?;

    // Query with probabilistic reasoning
    let query = ReasoningQuery {
        query_type: QueryType::WhatProperties { entity_id: 3 },
        max_hops: 2,
        min_confidence: 0.0, // Include all conclusions
        include_inferred: true,
    };

    let results = engine.reason(query)?;

    // Should have both properties with different confidences
    assert!(results.conclusions.len() >= 2);

    // Find the "cannot fly" conclusion
    let cannot_fly = results
        .conclusions
        .iter()
        .find(|c| c.object == FactObject::Node(4))
        .unwrap();

    assert!(cannot_fly.confidence > 0.9);

    Ok(())
}

#[test]
fn test_reasoning_chain_explanation() -> Result<()> {
    let device = create_test_device()?;
    let mut engine = ReasoningEngine::new(device, 1000)?;

    // Setup reasoning chain
    engine.add_fact(LogicalFact {
        subject: 3, // Penguin
        predicate: "lives_in".to_string(),
        object: FactObject::Value("Antarctica".to_string()),
        confidence: 0.95,
    })?;

    engine.add_fact(LogicalFact {
        subject: 6, // Antarctica (as a node)
        predicate: "has_climate".to_string(),
        object: FactObject::Value("cold".to_string()),
        confidence: 1.0,
    })?;

    // Add climate adaptation rule
    engine.add_rule(InferenceRule {
        id: 1,
        name: "Climate Adaptation".to_string(),
        pattern: RulePattern::Custom {
            description: "Animals adapt to their climate".to_string(),
        },
        confidence: 0.8,
        priority: 5,
    })?;

    // Query with explanation
    let query = ReasoningQuery {
        query_type: QueryType::Explain {
            fact: LogicalFact {
                subject: 3,
                predicate: "adapted_to".to_string(),
                object: FactObject::Value("cold".to_string()),
                confidence: 0.0,
            },
        },
        max_hops: 3,
        min_confidence: 0.5,
        include_inferred: true,
    };

    let results = engine.reason(query)?;

    assert!(!results.explanation_chains.is_empty());
    let chain = &results.explanation_chains[0];
    assert!(chain.steps.len() >= 2);
    assert!(chain.total_confidence > 0.7);

    Ok(())
}

#[test]
fn test_analogical_reasoning() -> Result<()> {
    let device = create_test_device()?;
    let mut engine = ReasoningEngine::new(device, 1000)?;

    // Setup analogy: Bird:Fly :: Fish:?
    engine.add_fact(LogicalFact {
        subject: 1, // Bird
        predicate: "can_do".to_string(),
        object: FactObject::Value("fly".to_string()),
        confidence: 0.9,
    })?;

    engine.add_fact(LogicalFact {
        subject: 1, // Bird
        predicate: "lives_in".to_string(),
        object: FactObject::Value("air".to_string()),
        confidence: 0.95,
    })?;

    engine.add_fact(LogicalFact {
        subject: 7, // Fish
        predicate: "lives_in".to_string(),
        object: FactObject::Value("water".to_string()),
        confidence: 0.95,
    })?;

    // Add analogical reasoning rule
    engine.add_rule(InferenceRule {
        id: 1,
        name: "Movement Analogy".to_string(),
        pattern: RulePattern::Analogy {
            source_domain: "air".to_string(),
            target_domain: "water".to_string(),
            relation_mapping: vec![("fly".to_string(), "swim".to_string())],
        },
        confidence: 0.85,
        priority: 7,
    })?;

    // Query: What can fish do?
    let query = ReasoningQuery {
        query_type: QueryType::WhatCan { entity_id: 7 },
        max_hops: 2,
        min_confidence: 0.7,
        include_inferred: true,
    };

    let results = engine.reason(query)?;

    // Should infer that fish can swim
    assert!(!results.conclusions.is_empty());
    let swim_conclusion = results.conclusions.iter().find(|c| {
        if let FactObject::Value(v) = &c.object {
            v == "swim"
        } else {
            false
        }
    });

    assert!(swim_conclusion.is_some());

    Ok(())
}

// =============================================================================
// Integration Tests
// =============================================================================

#[test]
fn test_complex_reasoning_scenario() -> Result<()> {
    let device = create_test_device()?;
    let mut engine = ReasoningEngine::new(device, 1000)?;

    // Build knowledge base about animals
    let facts = vec![
        // Taxonomy
        LogicalFact {
            subject: 3, // Penguin
            predicate: "is_a".to_string(),
            object: FactObject::Node(1), // Bird
            confidence: 1.0,
        },
        LogicalFact {
            subject: 1, // Bird
            predicate: "is_a".to_string(),
            object: FactObject::Node(5), // Animal
            confidence: 1.0,
        },
        // Properties
        LogicalFact {
            subject: 1, // Bird
            predicate: "has_property".to_string(),
            object: FactObject::Value("feathers".to_string()),
            confidence: 0.95,
        },
        LogicalFact {
            subject: 5, // Animal
            predicate: "needs".to_string(),
            object: FactObject::Value("food".to_string()),
            confidence: 1.0,
        },
        // Specific properties
        LogicalFact {
            subject: 3, // Penguin
            predicate: "lives_in".to_string(),
            object: FactObject::Value("Antarctica".to_string()),
            confidence: 0.9,
        },
    ];

    for fact in facts {
        engine.add_fact(fact)?;
    }

    // Add reasoning rules
    engine.add_rule(InferenceRule {
        id: 1,
        name: "Transitivity".to_string(),
        pattern: RulePattern::Transitive {
            relation_type: "is_a".to_string(),
        },
        confidence: 0.95,
        priority: 10,
    })?;

    engine.add_rule(InferenceRule {
        id: 2,
        name: "Property Inheritance".to_string(),
        pattern: RulePattern::PropertyInheritance {
            relation_type: "is_a".to_string(),
            property_type: "has_property".to_string(),
        },
        confidence: 0.85,
        priority: 8,
    })?;

    engine.add_rule(InferenceRule {
        id: 3,
        name: "Need Inheritance".to_string(),
        pattern: RulePattern::PropertyInheritance {
            relation_type: "is_a".to_string(),
            property_type: "needs".to_string(),
        },
        confidence: 0.9,
        priority: 9,
    })?;

    // Complex query: What do we know about penguins?
    let query = ReasoningQuery {
        query_type: QueryType::WhatProperties { entity_id: 3 },
        max_hops: 3,
        min_confidence: 0.6,
        include_inferred: true,
    };

    let results = engine.reason(query)?;

    // Should infer multiple properties
    assert!(results.conclusions.len() >= 3);

    // Check for inherited properties
    let has_feathers = results.conclusions.iter().any(|c| {
        if let FactObject::Value(v) = &c.object {
            v == "feathers"
        } else {
            false
        }
    });

    let needs_food = results.conclusions.iter().any(|c| {
        if let FactObject::Value(v) = &c.object {
            v == "food"
        } else {
            false
        }
    });

    assert!(has_feathers);
    assert!(needs_food);

    Ok(())
}

// =============================================================================
// Performance Tests
// =============================================================================

#[test]
#[ignore] // Run with --ignored for performance testing
fn test_reasoning_performance() -> Result<()> {
    let device = create_test_device()?;
    let mut engine = ReasoningEngine::new(device, 10000)?;

    // Add many facts
    let start = std::time::Instant::now();

    for i in 0..1000 {
        engine.add_fact(LogicalFact {
            subject: i,
            predicate: "is_a".to_string(),
            object: FactObject::Node(i + 1),
            confidence: 0.9,
        })?;
    }

    let fact_time = start.elapsed();

    // Add rules
    engine.add_rule(InferenceRule {
        id: 1,
        name: "Transitivity".to_string(),
        pattern: RulePattern::Transitive {
            relation_type: "is_a".to_string(),
        },
        confidence: 0.95,
        priority: 10,
    })?;

    // Perform reasoning
    let reason_start = std::time::Instant::now();

    let query = ReasoningQuery {
        query_type: QueryType::IsRelated {
            subject_id: 0,
            predicate: "is_a".to_string(),
            object_id: 999,
        },
        max_hops: 1000,
        min_confidence: 0.0,
        include_inferred: true,
    };

    let results = engine.reason(query)?;
    let reason_time = reason_start.elapsed();

    println!("Reasoning performance:");
    println!("  Add 1000 facts: {:?}", fact_time);
    println!("  Reasoning query: {:?}", reason_time);
    println!("  Conclusions found: {}", results.conclusions.len());

    assert!(fact_time.as_secs() < 1);
    assert!(reason_time.as_secs() < 5);

    Ok(())
}

// =============================================================================
// GPU Kernel Tests
// =============================================================================

#[test]
fn test_gpu_reasoning_kernels() -> Result<()> {
    let device = create_test_device()?;
    let mut engine = ReasoningEngine::new(device, 1000)?;

    // Add test data
    for i in 0..100 {
        engine.add_fact(LogicalFact {
            subject: i,
            predicate: "connected_to".to_string(),
            object: FactObject::Node((i + 1) % 100),
            confidence: 0.8,
        })?;
    }

    // Upload to GPU
    engine.sync_to_gpu()?;

    // Test GPU reasoning kernel
    let gpu_results = engine.gpu_multi_hop_reasoning(0, 50, 5)?;

    assert!(!gpu_results.is_empty());

    Ok(())
}
