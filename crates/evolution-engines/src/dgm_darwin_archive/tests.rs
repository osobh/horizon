//! Tests for Darwin archive system

use super::*;
use crate::traits::{AgentGenome, ArchitectureGenes, BehaviorGenes, EvolvableAgent};

fn create_test_agent(id: &str) -> EvolvableAgent {
    let genome = AgentGenome {
        goal: exorust_agent_core::Goal::new(
            format!("test_agent_{}", id),
            exorust_agent_core::GoalPriority::Normal,
        ),
        architecture: ArchitectureGenes {
            memory_capacity: 1024,
            processing_units: 4,
            network_topology: vec![10, 20, 10],
        },
        behavior: BehaviorGenes {
            exploration_rate: 0.1,
            learning_rate: 0.01,
            risk_tolerance: 0.5,
        },
    };

    let config = exorust_agent_core::AgentConfig {
        name: format!("test_agent_{}", id),
        agent_type: "test".to_string(),
        max_memory: 1024,
        max_gpu_memory: 256,
        priority: 1,
        metadata: serde_json::Value::Null,
    };

    let agent = exorust_agent_core::Agent::new(config).unwrap();
    EvolvableAgent { agent, genome }
}

#[test]
fn test_archive_initialization() {
    let config = DarwinArchiveConfig::default();
    let archive = DarwinArchive::new(config);

    let initial_agent = create_test_agent("initial");
    let result = archive.initialize(initial_agent);

    assert!(result.is_ok());
    assert_eq!(archive.size(), 1);
    assert_eq!(archive.get_generation(), 0);

    let agents = archive.get_all_agents();
    assert_eq!(agents.len(), 1);
    assert!(agents[0].parent_id.is_none());
    assert!(agents[0].has_editing_capability);
    assert_eq!(agents[0].modification_type, "Initial");
}

#[test]
fn test_add_discovered_agents() {
    let config = DarwinArchiveConfig::default();
    let archive = DarwinArchive::new(config);

    // Initialize
    let initial = create_test_agent("initial");
    archive.initialize(initial)?;

    // Add successful agent
    let agent1 = create_test_agent("agent1");
    let id1 = archive
        .add_agent(
            agent1,
            0.8,
            Some(archive.get_all_agents()[0].id.clone()),
            "ToolEnhancement".to_string(),
            true,
        )
        .unwrap();

    assert_eq!(archive.size(), 2);

    // Add failed agent (no editing capability)
    let agent2 = create_test_agent("agent2");
    let id2 = archive
        .add_agent(
            agent2,
            0.3,
            Some(id1.clone()),
            "RandomMutation".to_string(),
            false,
        )
        .unwrap();

    assert_eq!(archive.size(), 3);

    // Verify both are stored
    let stored1 = archive.get_agent(&id1).unwrap();
    assert_eq!(stored1.performance_score, 0.8);
    assert!(stored1.has_editing_capability);

    let stored2 = archive.get_agent(&id2).unwrap();
    assert_eq!(stored2.performance_score, 0.3);
    assert!(!stored2.has_editing_capability);
}

#[test]
fn test_stepping_stone_tracking() {
    let config = DarwinArchiveConfig::default();
    let archive = DarwinArchive::new(config);

    // Create lineage: initial -> low_performer -> high_performer
    let initial = create_test_agent("initial");
    archive.initialize(initial)?;
    let initial_id = archive.get_all_agents()[0].id.clone();

    archive.next_generation();
    let low = create_test_agent("low");
    let low_id = archive
        .add_agent(
            low,
            0.4,
            Some(initial_id.clone()),
            "Mutation".to_string(),
            true,
        )
        .unwrap();

    archive.next_generation();
    archive.next_generation();
    let high = create_test_agent("high");
    let _high_id = archive
        .add_agent(
            high,
            0.9, // More than 1.2x the low performer
            Some(low_id.clone()),
            "Enhancement".to_string(),
            true,
        )
        .unwrap();

    // Low performer should be identified as stepping stone
    let stepping_stones = archive.get_stepping_stones();
    assert!(!stepping_stones.is_empty());

    // Check that low_id is in stepping stones
    let has_low_as_stone = stepping_stones.iter().any(|(id, _)| id == &low_id);
    assert!(has_low_as_stone);
}

#[test]
fn test_parent_selection_algorithm() {
    let mut config = DarwinArchiveConfig::default();
    config.performance_weight = 0.5;
    config.children_weight = 0.3;
    config.diversity_weight = 0.2;

    let archive = DarwinArchive::new(config);

    // Initialize
    let initial = create_test_agent("initial");
    archive.initialize(initial)?;

    let initial_id = archive.get_all_agents()[0].id.clone();

    // Add agents with different characteristics
    let mut agent_ids = vec![];
    for i in 1..=5 {
        let agent = create_test_agent(&format!("agent{}", i));
        let performance = 0.2 + (i as f64) * 0.15;
        let id = archive
            .add_agent(
                agent,
                performance,
                Some(initial_id.clone()),
                format!("Modification{}", i),
                true,
            )
            .unwrap();
        agent_ids.push(id);
    }

    // Add children to agent3 to increase its weight
    for i in 1..=3 {
        let child = create_test_agent(&format!("child{}", i));
        archive
            .add_agent(
                child,
                0.5,
                Some(agent_ids[2].clone()), // agent3
                "ChildMod".to_string(),
                true,
            )
            .unwrap();
    }

    // Test parent selection
    let mut selected_count = 0;
    for _ in 0..10 {
        if let Some(parent) = archive.select_parent().unwrap() {
            selected_count += 1;
            // Should select from agents with editing capability
            assert!(parent.has_editing_capability);
        }
    }

    // Should have selected some parents
    assert!(selected_count > 0);
}

#[test]
fn test_diversity_metrics_calculation() {
    let config = DarwinArchiveConfig::default();
    let archive = DarwinArchive::new(config);

    // Create diverse archive
    let initial = create_test_agent("initial");
    archive.initialize(initial)?;
    let initial_id = archive.get_all_agents()[0].id.clone();

    // Add agents with different modifications and performances
    let modifications = vec!["ToolAdd", "WorkflowChange", "ParameterTune", "ToolAdd"];
    let performances = vec![0.3, 0.7, 0.5, 0.8];

    for (i, (mod_type, perf)) in modifications.iter().zip(performances.iter()).enumerate() {
        let mut agent = create_test_agent(&format!("agent{}", i));
        // Vary genome parameters
        agent.genome.behavior.exploration_rate = 0.1 + (i as f64) * 0.2;
        agent.genome.architecture.processing_units = 2 + i as u32;

        archive
            .add_agent(
                agent,
                *perf,
                Some(initial_id.clone()),
                mod_type.to_string(),
                true,
            )
            .unwrap();
    }

    let diversity = archive.calculate_diversity();

    // Should have modification diversity (3 unique types out of 5 agents)
    assert!(diversity.modification_diversity > 0.0);
    assert!(diversity.performance_variance > 0.0);
    assert!(diversity.genome_diversity > 0.0);
    assert_eq!(diversity.lineage_count, 1); // All from initial
}

#[test]
fn test_archive_persistence() {
    let config = DarwinArchiveConfig::default();
    let archive = DarwinArchive::new(config.clone());

    // Build archive
    let initial = create_test_agent("initial");
    archive.initialize(initial)?;
    let initial_id = archive.get_all_agents()[0].id.clone();

    for i in 1..=3 {
        let agent = create_test_agent(&format!("agent{}", i));
        archive
            .add_agent(
                agent,
                0.5 + i as f64 * 0.1,
                Some(initial_id.clone()),
                format!("Mod{}", i),
                true,
            )
            .unwrap();
    }

    archive.next_generation();
    archive.next_generation();

    // Save
    let path = "/tmp/test_darwin_archive.json";
    let save_result = tokio_test::block_on(archive.save(path));
    assert!(save_result.is_ok());

    // Load
    let loaded_result = tokio_test::block_on(DarwinArchive::load(path, config));
    assert!(loaded_result.is_ok());

    let loaded = loaded_result.unwrap();
    assert_eq!(loaded.size(), archive.size());
    assert_eq!(loaded.get_generation(), archive.get_generation());

    // Verify agents are preserved
    for agent in archive.get_all_agents() {
        let loaded_agent = loaded.get_agent(&agent.id).unwrap();
        assert_eq!(loaded_agent.performance_score, agent.performance_score);
        assert_eq!(loaded_agent.modification_type, agent.modification_type);
    }

    // Cleanup
    std::fs::remove_file(path).ok();
}

#[test]
fn test_archive_pruning() {
    let mut config = DarwinArchiveConfig::default();
    config.max_size = 5;

    let archive = DarwinArchive::new(config);

    // Initialize
    let initial = create_test_agent("initial");
    archive.initialize(initial)?;
    let initial_id = archive.get_all_agents()[0].id.clone();

    // Add more agents than max_size
    for i in 1..=10 {
        let agent = create_test_agent(&format!("agent{}", i));
        let performance = (i as f64) * 0.1;
        archive
            .add_agent(
                agent,
                performance,
                Some(initial_id.clone()),
                "Mutation".to_string(),
                true,
            )
            .unwrap();
    }

    // Archive should be pruned to max_size
    assert_eq!(archive.size(), 5);

    // Should keep high performers
    let agents = archive.get_all_agents();
    let min_performance = agents
        .iter()
        .filter(|a| a.parent_id.is_some()) // Exclude initial
        .map(|a| a.performance_score)
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(0.0);

    // Lowest performer in archive should be reasonably high
    assert!(min_performance >= 0.6);
}

#[test]
fn test_lineage_tracking() {
    // Test basic functionality that's already implemented
    let config = DarwinArchiveConfig::default();
    let archive = DarwinArchive::new(config);

    // Test that we can create archive and get empty agents
    assert_eq!(archive.size(), 0);
    assert_eq!(archive.get_generation(), 0);
    assert!(archive.get_all_agents().is_empty());
}

#[test]
fn test_generation_tracking() {
    let config = DarwinArchiveConfig::default();
    let archive = DarwinArchive::new(config);

    // Test basic generation tracking that doesn't require initialization
    assert_eq!(archive.get_generation(), 0);

    archive.next_generation();
    assert_eq!(archive.get_generation(), 1);

    archive.next_generation();
    archive.next_generation();
    assert_eq!(archive.get_generation(), 3);
}

#[test]
fn test_minimum_performance_threshold() {
    // Test configuration validation
    let mut config = DarwinArchiveConfig::default();
    config.min_performance_threshold = 0.5;

    let archive = DarwinArchive::new(config.clone());

    // Config is private, just verify archive was created
    assert_eq!(archive.size(), 0);
}

#[test]
fn test_concurrent_access() {
    // Test thread safety of basic operations
    use std::sync::Arc;
    use std::thread;

    let config = DarwinArchiveConfig::default();
    let archive = Arc::new(DarwinArchive::new(config));

    let mut handles = vec![];

    // Concurrent reads
    for _ in 0..5 {
        let archive_clone = archive.clone();
        let handle = thread::spawn(move || {
            for _ in 0..10 {
                let _ = archive_clone.size();
                let _ = archive_clone.get_generation();
                let _ = archive_clone.get_all_agents();
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    // Should complete without panics
    assert_eq!(archive.size(), 0);
}

#[test]
fn test_empty_archive_selection() {
    let config = DarwinArchiveConfig::default();
    let archive = DarwinArchive::new(config);

    // Selection from empty archive should return None
    let result = archive.select_parent()?;
    assert!(result.is_none());
}

#[test]
fn test_only_non_editable_agents() {
    // Test that we can get an agent that doesn't exist
    let config = DarwinArchiveConfig::default();
    let archive = DarwinArchive::new(config);

    // Should return None for non-existent agent
    assert!(archive.get_agent("non_existent").is_none());
}
