//! # Time-Travel Debugger Example
//!
//! This example demonstrates how to use the time-travel debugger to debug
//! an agent's execution, showing all major features including snapshots,
//! event recording, time navigation, breakpoints, and state comparison.

use chrono::{Duration, Utc};
use serde_json::json;
use std::collections::HashMap;
use stratoswarm_time_travel_debugger::*;
use uuid::Uuid;

#[tokio::main]
async fn main() -> error::Result<()> {
    // Initialize logging for better visibility
    tracing_subscriber::fmt::init();

    println!("🚀 Starting Time-Travel Debugger Example");
    println!("==========================================");

    // Step 1: Create the debugger with custom configuration
    let debugger = TimeDebugger::builder()
        .with_snapshot_config(SnapshotConfig {
            max_snapshots: 50,
            compression_enabled: true,
            diff_threshold: 0.1,
            cleanup_interval: std::time::Duration::from_secs(60),
        })
        .with_event_log_config(event_log::EventLogConfig {
            max_events_per_agent: 200,
            enable_causality_tracking: true,
            enable_compression: false,
            flush_interval: std::time::Duration::from_secs(30),
            enable_persistence: false,
        })
        .with_max_navigation_history(25)
        .build();

    // Start background tasks
    debugger.start_background_tasks().await?;

    // Step 2: Create a debug session for our agent
    let agent_id = Uuid::new_v4();
    println!("\n📊 Creating debug session for agent: {}", agent_id);

    let session_id = debugger.create_session(agent_id).await?;
    let mut session = debugger.get_session(session_id).await?;

    // Step 3: Start the session and set metadata
    session.start().await?;
    session
        .update_metadata(
            Some("AI Agent Learning Demo".to_string()),
            Some("Demonstrating agent learning through exploration and interaction".to_string()),
            Some(vec![
                "learning".to_string(),
                "exploration".to_string(),
                "demo".to_string(),
            ]),
        )
        .await?;

    println!("✅ Debug session started: {}", session_id);

    // Step 4: Simulate agent execution with detailed state tracking
    println!("\n🤖 Simulating Agent Execution");
    println!("==============================");

    // Initial state - agent just spawned
    let initial_state = json!({
        "agent_id": agent_id.to_string(),
        "health": 100,
        "energy": 100,
        "position": {"x": 0, "y": 0, "z": 0},
        "inventory": [],
        "skills": {
            "exploration": 1,
            "combat": 1,
            "crafting": 1
        },
        "knowledge": {
            "discovered_locations": [],
            "known_resources": [],
            "encountered_entities": []
        },
        "goals": ["explore_environment", "gather_resources", "learn_skills"],
        "current_goal": "explore_environment",
        "memory": {
            "recent_actions": [],
            "important_events": []
        },
        "stats": {
            "steps_taken": 0,
            "items_collected": 0,
            "skills_learned": 0,
            "decisions_made": 0
        }
    });

    let snapshot1 = session
        .take_snapshot(
            initial_state.clone(),
            2048,
            create_metadata("initial_spawn", "Agent initial state after spawning"),
        )
        .await?;

    println!("📸 Initial snapshot taken: {}", snapshot1);

    // Event 1: Agent starts exploring
    let explore_event = session
        .record_event(
            EventType::DecisionMade,
            json!({
                "decision": "start_exploration",
                "reasoning": "Need to learn about the environment",
                "confidence": 0.8,
                "alternatives_considered": ["stay_put", "random_movement"],
                "selected_strategy": "systematic_exploration"
            }),
            create_metadata("decision", "Agent decides to explore systematically"),
            None,
        )
        .await?;

    println!("📝 Recorded exploration decision: {}", explore_event);

    // Simulate some exploration movement
    for step in 1..=5 {
        let move_event = session
            .record_event(
                EventType::ActionExecution,
                json!({
                    "action": "move",
                    "direction": match step % 4 {
                        1 => "north",
                        2 => "east",
                        3 => "south",
                        0 => "west",
                        _ => "north"
                    },
                    "distance": 1,
                    "energy_cost": 2,
                    "step_number": step
                }),
                create_metadata("movement", &format!("Exploration step {}", step)),
                Some(explore_event), // All movements are caused by exploration decision
            )
            .await?;

        // Update state after movement
        let mut new_state = initial_state.clone();
        new_state["position"]["x"] = json!(step % 3);
        new_state["position"]["y"] = json!(step / 3);
        new_state["energy"] = json!(100 - step * 2);
        new_state["stats"]["steps_taken"] = json!(step);

        // Add to memory
        new_state["memory"]["recent_actions"] = json!([format!(
            "moved_{}",
            match step % 4 {
                1 => "north",
                2 => "east",
                3 => "south",
                0 => "west",
                _ => "north",
            }
        )]);

        if step % 2 == 0 {
            let snapshot = session
                .take_snapshot(
                    new_state,
                    2048 + step * 100,
                    create_metadata(
                        "exploration",
                        &format!("State after {} exploration steps", step),
                    ),
                )
                .await?;
            println!("📸 Exploration snapshot {}: {}", step, snapshot);
        }

        println!("🚶 Movement step {} completed: {}", step, move_event);
    }

    // Event: Agent discovers a resource
    let discovery_event = session
        .record_event(
            EventType::Custom("resource_discovery".to_string()),
            json!({
                "resource_type": "crystal",
                "resource_id": Uuid::new_v4(),
                "location": {"x": 2, "y": 1, "z": 0},
                "rarity": "uncommon",
                "estimated_value": 50,
                "discovery_method": "visual_scan"
            }),
            create_metadata("discovery", "Agent discovers valuable crystal"),
            None,
        )
        .await?;

    println!("💎 Resource discovered: {}", discovery_event);

    // Event: Agent makes decision about the resource
    let resource_decision = session
        .record_event(
            EventType::DecisionMade,
            json!({
                "decision": "collect_resource",
                "reasoning": "Crystal appears valuable and safe to collect",
                "confidence": 0.9,
                "risk_assessment": "low",
                "expected_benefit": "high"
            }),
            create_metadata("decision", "Agent decides to collect the crystal"),
            Some(discovery_event),
        )
        .await?;

    // Event: Agent collects the resource
    let collect_event = session
        .record_event(
            EventType::ActionExecution,
            json!({
                "action": "collect",
                "target": "crystal",
                "success": true,
                "time_taken": 3.5,
                "energy_cost": 5,
                "skill_used": "crafting"
            }),
            create_metadata("collection", "Agent successfully collects crystal"),
            Some(resource_decision),
        )
        .await?;

    // State after collection
    let post_collection_state = json!({
        "agent_id": agent_id.to_string(),
        "health": 100,
        "energy": 85, // Energy used for movement and collection
        "position": {"x": 2, "y": 1, "z": 0},
        "inventory": [
            {
                "type": "crystal",
                "rarity": "uncommon",
                "value": 50,
                "collected_at": Utc::now().to_string()
            }
        ],
        "skills": {
            "exploration": 2, // Skill improved through use
            "combat": 1,
            "crafting": 2 // Skill improved through collection
        },
        "knowledge": {
            "discovered_locations": [{"x": 2, "y": 1, "z": 0, "type": "resource_site"}],
            "known_resources": ["crystal"],
            "encountered_entities": []
        },
        "goals": ["explore_environment", "gather_resources", "learn_skills"],
        "current_goal": "gather_resources", // Goal changed
        "memory": {
            "recent_actions": ["moved_north", "discovered_crystal", "collected_crystal"],
            "important_events": ["first_resource_discovery"]
        },
        "stats": {
            "steps_taken": 5,
            "items_collected": 1,
            "skills_learned": 0,
            "decisions_made": 2
        }
    });

    let collection_snapshot = session
        .take_snapshot(
            post_collection_state,
            3072,
            create_metadata("post_collection", "State after collecting first resource"),
        )
        .await?;

    println!("📸 Post-collection snapshot: {}", collection_snapshot);

    // Event: Unexpected encounter with another entity
    let encounter_event = session
        .record_event(
            EventType::Custom("entity_encounter".to_string()),
            json!({
                "entity_type": "friendly_trader",
                "entity_id": Uuid::new_v4(),
                "location": {"x": 2, "y": 1, "z": 0},
                "initial_disposition": "neutral",
                "interaction_available": true,
                "threat_level": "none"
            }),
            create_metadata("encounter", "Agent encounters a friendly trader"),
            None,
        )
        .await?;

    // Event: Agent makes decision about interaction
    let interaction_decision = session
        .record_event(
            EventType::DecisionMade,
            json!({
                "decision": "initiate_trade",
                "reasoning": "Trader appears friendly and might offer valuable exchange",
                "confidence": 0.7,
                "risk_assessment": "minimal",
                "potential_outcomes": ["beneficial_trade", "information_gain", "skill_development"]
            }),
            create_metadata("decision", "Agent decides to trade with entity"),
            Some(encounter_event),
        )
        .await?;

    // Event: Trading interaction
    let trade_event = session
        .record_event(
            EventType::ActionExecution,
            json!({
                "action": "trade",
                "items_given": ["crystal"],
                "items_received": ["advanced_tool", "skill_book"],
                "satisfaction": "high",
                "relationship_change": "+positive",
                "new_knowledge_gained": true
            }),
            create_metadata(
                "trade",
                "Agent successfully trades crystal for tools and knowledge",
            ),
            Some(interaction_decision),
        )
        .await?;

    // Final state after trading
    let final_state = json!({
        "agent_id": agent_id.to_string(),
        "health": 100,
        "energy": 80,
        "position": {"x": 2, "y": 1, "z": 0},
        "inventory": [
            {
                "type": "advanced_tool",
                "quality": "high",
                "durability": 100,
                "special_abilities": ["efficient_collection", "enhanced_crafting"]
            },
            {
                "type": "skill_book",
                "skill": "advanced_exploration",
                "knowledge_value": 75
            }
        ],
        "skills": {
            "exploration": 2,
            "combat": 1,
            "crafting": 2,
            "trading": 1 // New skill acquired
        },
        "knowledge": {
            "discovered_locations": [
                {"x": 2, "y": 1, "z": 0, "type": "resource_site"},
                {"x": 2, "y": 1, "z": 0, "type": "trader_meeting_point"}
            ],
            "known_resources": ["crystal"],
            "encountered_entities": ["friendly_trader"],
            "trade_relationships": [{"entity": "friendly_trader", "status": "positive"}]
        },
        "goals": ["explore_environment", "gather_resources", "learn_skills"],
        "current_goal": "learn_skills", // Goal evolved again
        "memory": {
            "recent_actions": ["collected_crystal", "met_trader", "completed_trade"],
            "important_events": ["first_resource_discovery", "first_successful_trade"]
        },
        "stats": {
            "steps_taken": 5,
            "items_collected": 1,
            "skills_learned": 1,
            "decisions_made": 3
        }
    });

    let final_snapshot = session
        .take_snapshot(
            final_state,
            4096,
            create_metadata("final", "Final state after trading interaction"),
        )
        .await?;

    println!("📸 Final snapshot: {}", final_snapshot);

    // Step 5: Set up breakpoints for debugging
    println!("\n🎯 Setting up Breakpoints");
    println!("=========================");

    let decision_bp = session
        .create_breakpoint(
            BreakpointCondition::OnEventType(EventType::DecisionMade),
            create_metadata("breakpoint", "Break on all decision points"),
        )
        .await?;
    println!("🎯 Decision breakpoint created: {}", decision_bp);

    let energy_bp = session
        .create_breakpoint(
            BreakpointCondition::OnStateCondition {
                field_path: "energy".to_string(),
                expected_value: json!(85),
            },
            create_metadata("breakpoint", "Break when energy reaches 85"),
        )
        .await?;
    println!("🎯 Energy threshold breakpoint created: {}", energy_bp);

    let skill_bp = session
        .create_breakpoint(
            BreakpointCondition::OnStateCondition {
                field_path: "skills.trading".to_string(),
                expected_value: json!(1),
            },
            create_metadata("breakpoint", "Break when trading skill is acquired"),
        )
        .await?;
    println!("🎯 Skill acquisition breakpoint created: {}", skill_bp);

    // Step 6: Time navigation demonstration
    println!("\n⏰ Time Navigation Demo");
    println!("======================");

    // Navigate to beginning
    let start_pos = session.navigate_to_event_index(0).await?;
    println!(
        "⏪ Navigated to start: Event {} at {}",
        start_pos.event_index, start_pos.timestamp
    );

    // Step forward through key events
    let positions = vec![
        session
            .step(NavigationDirection::Forward, StepSize::Event)
            .await?,
        session
            .step(NavigationDirection::Forward, StepSize::Custom(3))
            .await?,
        session
            .step(NavigationDirection::Forward, StepSize::Event)
            .await?,
    ];

    for (i, pos) in positions.iter().enumerate() {
        println!(
            "⏩ Step {}: Event {} at {}",
            i + 1,
            pos.event_index,
            pos.timestamp
        );
    }

    // Navigate to specific time
    let mid_time = Utc::now() - Duration::minutes(1);
    let time_pos = session.navigate_to_time(mid_time).await?;
    println!(
        "🕐 Time navigation to {}: Event {}",
        mid_time, time_pos.event_index
    );

    // Navigate backward
    let back_pos = session
        .step(NavigationDirection::Backward, StepSize::Custom(2))
        .await?;
    println!(
        "⏪ Stepped back 2 events: Now at event {}",
        back_pos.event_index
    );

    // Step 7: State comparison and analysis
    println!("\n🔍 State Analysis");
    println!("================");

    // Compare initial and final states
    let comparison = session
        .compare_snapshots(snapshot1, final_snapshot, None)
        .await?;
    println!("📊 Comparison between initial and final states:");
    println!("   - Total changes: {}", comparison.summary.total_changes);
    println!("   - Additions: {}", comparison.summary.additions);
    println!("   - Modifications: {}", comparison.summary.modifications);
    println!("   - Removals: {}", comparison.summary.removals);
    println!(
        "   - Similarity score: {:.2}",
        comparison.summary.similarity_score
    );

    // Show some specific changes
    println!("\n📝 Key Changes:");
    for (i, change) in comparison.changes.iter().take(5).enumerate() {
        println!(
            "   {}. {}: {:?} -> {:?}",
            i + 1,
            change.path,
            change.old_value,
            change.new_value
        );
    }

    // Generate visual diff
    let diff = debugger
        .comparator
        .generate_visual_diff(&comparison, DiffFormat::Unified)
        .await?;
    println!("\n📄 Unified Diff (truncated):");
    for line in diff.lines().take(10) {
        println!("   {}", line);
    }

    // Step 8: Event analysis and causality
    println!("\n🔗 Event Analysis");
    println!("=================");

    // Analyze causality chain
    let chain = debugger.event_log.get_causality_chain(trade_event).await?;
    println!("🔗 Causality chain for trade event:");
    for (i, event) in chain.iter().enumerate() {
        println!(
            "   {}. {} -> {:?} at {}",
            i + 1,
            event.id,
            event.event_type,
            event.timestamp
        );
    }

    // Get events caused by discovery
    let caused = debugger
        .event_log
        .get_caused_events(discovery_event)
        .await?;
    println!("\n➡️  Events caused by resource discovery:");
    for event in caused.iter() {
        println!("   - {} -> {:?}", event.id, event.event_type);
    }

    // Replay events from a specific point
    println!("\n🔄 Event Replay:");
    let mut replay_count = 0;
    debugger
        .event_log
        .replay_from_time(agent_id, positions[1].timestamp, |event| {
            replay_count += 1;
            println!(
                "   Replaying: {:?} -> {}",
                event.event_type,
                event.event_data["action"].as_str().unwrap_or("N/A")
            );
            Ok(())
        })
        .await?;
    println!("   📊 Replayed {} events", replay_count);

    // Step 9: Session statistics and export
    println!("\n📈 Session Statistics");
    println!("====================");

    let stats = session.get_statistics().await?;
    println!("📊 Session Statistics:");
    println!("   - Session ID: {}", stats.session_id);
    println!("   - Agent ID: {}", stats.agent_id);
    println!("   - Duration: {:?}", stats.session_duration);
    println!("   - Snapshots: {}", stats.snapshot_count);
    println!("   - Events: {}", stats.total_events);
    println!("   - Memory usage: {} bytes", stats.total_memory_usage);
    println!("   - Breakpoints: {}", stats.breakpoint_count);
    println!("   - Navigation steps: {}", stats.navigation_steps);

    // Store some custom debug data
    session.set_custom_data(
        "debug_notes".to_string(),
        json!({
            "interesting_patterns": [
                "Agent showed good decision-making progression",
                "Trading interaction was handled optimally",
                "Skill development followed expected trajectory"
            ],
            "performance_metrics": {
                "decisions_per_minute": stats.total_events as f64 / stats.session_duration.num_minutes() as f64,
                "state_changes": comparison.summary.total_changes,
                "efficiency_score": 0.87
            },
            "recommendations": [
                "Consider implementing more sophisticated resource evaluation",
                "Add risk assessment for entity encounters",
                "Improve memory management for longer sessions"
            ]
        }),
    ).await?;

    let debug_notes = session.get_custom_data("debug_notes").await?;
    println!(
        "\n📝 Debug Notes: {}",
        serde_json::to_string_pretty(&debug_notes.unwrap())?
    );

    // Export the session
    println!("\n💾 Exporting Session");
    println!("===================");

    let exported = session.export_session(ExportFormat::Json).await?;
    println!("✅ Session exported: {} bytes", exported.len());

    // Step 10: System-wide statistics
    let system_stats = debugger.get_system_statistics().await?;
    println!("\n🌐 System Statistics");
    println!("===================");
    println!("📊 Overall System Stats:");
    println!("   - Total snapshots: {}", system_stats.total_snapshots);
    println!("   - Total events: {}", system_stats.total_events);
    println!(
        "   - Memory usage: {} bytes",
        system_stats.total_memory_usage
    );
    println!("   - Active sessions: {}", system_stats.active_sessions);
    println!("   - Total agents: {}", system_stats.total_agents);

    // Step 11: Cleanup
    println!("\n🧹 Cleanup");
    println!("==========");

    session.complete().await?;
    println!("✅ Session completed");

    debugger.stop_background_tasks().await?;
    println!("✅ Background tasks stopped");

    println!("\n🎉 Time-Travel Debugger Example Complete!");
    println!("==========================================");
    println!("This example demonstrated:");
    println!("✓ Session creation and management");
    println!("✓ State snapshot recording");
    println!("✓ Event logging with causality");
    println!("✓ Time navigation and breakpoints");
    println!("✓ State comparison and analysis");
    println!("✓ Pattern detection and insights");
    println!("✓ Export/import capabilities");
    println!("✓ Performance monitoring");

    Ok(())
}

/// Helper function to create metadata for events and snapshots
fn create_metadata(category: &str, description: &str) -> HashMap<String, String> {
    let mut metadata = HashMap::new();
    metadata.insert("category".to_string(), category.to_string());
    metadata.insert("description".to_string(), description.to_string());
    metadata.insert("timestamp".to_string(), Utc::now().to_string());
    metadata
}
