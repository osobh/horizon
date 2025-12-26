use serde_json::json;
use std::collections::HashMap;
use stratoswarm_time_travel_debugger::*;
use uuid::Uuid;

/// Test complete end-to-end debugging workflow
#[tokio::test]
async fn test_complete_debugging_workflow() {
    let debugger = TimeDebugger::builder()
        .with_max_navigation_history(50)
        .with_snapshot_config(SnapshotConfig {
            max_snapshots: 100,
            compression_enabled: true,
            diff_threshold: 0.1,
            cleanup_interval: std::time::Duration::from_secs(60),
        })
        .with_event_log_config(event_log::EventLogConfig {
            max_events_per_agent: 1000,
            enable_causality_tracking: true,
            enable_compression: false,
            flush_interval: std::time::Duration::from_secs(30),
            enable_persistence: false,
        })
        .build();

    debugger.start_background_tasks().await.unwrap();

    let agent_id = Uuid::new_v4();
    let session_id = debugger.create_session(agent_id).await.unwrap();
    let mut session = debugger.get_session(session_id).await.unwrap();

    // Phase 1: Start debugging session and set up initial state
    session.start().await.unwrap();
    session
        .update_metadata(
            Some("Integration Test Session".to_string()),
            Some("Testing complete workflow".to_string()),
            Some(vec!["integration".to_string(), "test".to_string()]),
        )
        .await
        .unwrap();

    // Phase 2: Simulate agent execution with snapshots and events
    let mut snapshots = Vec::new();
    let mut events = Vec::new();

    // Initial state
    let initial_snapshot = session
        .take_snapshot(
            json!({
                "health": 100,
                "energy": 100,
                "position": {"x": 0, "y": 0},
                "inventory": [],
                "level": 1,
                "experience": 0
            }),
            2048,
            [("phase".to_string(), "initial".to_string())]
                .into_iter()
                .collect(),
        )
        .await
        .unwrap();
    snapshots.push(initial_snapshot);

    // Event 1: Agent starts moving
    let move_event = session
        .record_event(
            EventType::ActionExecution,
            json!({"action": "move", "direction": "north", "distance": 5}),
            [("phase".to_string(), "movement".to_string())]
                .into_iter()
                .collect(),
            None,
        )
        .await
        .unwrap();
    events.push(move_event);

    // State after movement
    let move_snapshot = session
        .take_snapshot(
            json!({
                "health": 98, // Slightly reduced
                "energy": 95,  // Energy used for movement
                "position": {"x": 0, "y": 5},
                "inventory": [],
                "level": 1,
                "experience": 0
            }),
            2048,
            [("phase".to_string(), "post_movement".to_string())]
                .into_iter()
                .collect(),
        )
        .await
        .unwrap();
    snapshots.push(move_snapshot);

    // Event 2: Agent finds an item
    let find_event = session
        .record_event(
            EventType::StateChange,
            json!({"type": "item_found", "item": "health_potion"}),
            [("phase".to_string(), "discovery".to_string())]
                .into_iter()
                .collect(),
            Some(move_event), // Caused by movement
        )
        .await
        .unwrap();
    events.push(find_event);

    // Event 3: Agent collects the item
    let collect_event = session
        .record_event(
            EventType::ActionExecution,
            json!({"action": "collect", "item": "health_potion"}),
            [("phase".to_string(), "collection".to_string())]
                .into_iter()
                .collect(),
            Some(find_event), // Caused by finding the item
        )
        .await
        .unwrap();
    events.push(collect_event);

    // State after collection
    let collect_snapshot = session
        .take_snapshot(
            json!({
                "health": 98,
                "energy": 94, // Slight energy cost to collect
                "position": {"x": 0, "y": 5},
                "inventory": ["health_potion"],
                "level": 1,
                "experience": 10 // Experience gained
            }),
            2048,
            [("phase".to_string(), "post_collection".to_string())]
                .into_iter()
                .collect(),
        )
        .await
        .unwrap();
    snapshots.push(collect_snapshot);

    // Event 4: Agent encounters enemy
    let enemy_event = session
        .record_event(
            EventType::Custom("enemy_encounter".to_string()),
            json!({"enemy_type": "goblin", "enemy_level": 1}),
            [("phase".to_string(), "combat".to_string())]
                .into_iter()
                .collect(),
            None,
        )
        .await
        .unwrap();
    events.push(enemy_event);

    // Event 5: Agent makes decision to fight
    let decision_event = session
        .record_event(
            EventType::DecisionMade,
            json!({"decision": "fight", "reasoning": "enemy is weak"}),
            [("phase".to_string(), "decision".to_string())]
                .into_iter()
                .collect(),
            Some(enemy_event),
        )
        .await
        .unwrap();
    events.push(decision_event);

    // Event 6: Combat action
    let combat_event = session
        .record_event(
            EventType::ActionExecution,
            json!({"action": "attack", "target": "goblin", "damage_dealt": 25}),
            [("phase".to_string(), "combat_action".to_string())]
                .into_iter()
                .collect(),
            Some(decision_event),
        )
        .await
        .unwrap();
    events.push(combat_event);

    // State after combat
    let combat_snapshot = session
        .take_snapshot(
            json!({
                "health": 85, // Took some damage
                "energy": 85, // Energy used in combat
                "position": {"x": 0, "y": 5},
                "inventory": ["health_potion"],
                "level": 1,
                "experience": 35, // More experience gained
                "combat_stats": {
                    "enemies_defeated": 1,
                    "damage_dealt": 25,
                    "damage_taken": 13
                }
            }),
            2048,
            [("phase".to_string(), "post_combat".to_string())]
                .into_iter()
                .collect(),
        )
        .await
        .unwrap();
    snapshots.push(combat_snapshot);

    // Event 7: Agent uses health potion
    let heal_event = session
        .record_event(
            EventType::ActionExecution,
            json!({"action": "use_item", "item": "health_potion", "health_restored": 20}),
            [("phase".to_string(), "healing".to_string())]
                .into_iter()
                .collect(),
            None,
        )
        .await
        .unwrap();
    events.push(heal_event);

    // Final state
    let final_snapshot = session
        .take_snapshot(
            json!({
                "health": 100, // Fully healed
                "energy": 85,
                "position": {"x": 0, "y": 5},
                "inventory": [], // Potion consumed
                "level": 2, // Leveled up
                "experience": 5, // Experience reset after level up
                "combat_stats": {
                    "enemies_defeated": 1,
                    "damage_dealt": 25,
                    "damage_taken": 13
                }
            }),
            2048,
            [("phase".to_string(), "final".to_string())]
                .into_iter()
                .collect(),
        )
        .await
        .unwrap();
    snapshots.push(final_snapshot);

    // Phase 3: Set up breakpoints for debugging
    let health_breakpoint = session
        .create_breakpoint(
            BreakpointCondition::OnStateCondition {
                field_path: "health".to_string(),
                expected_value: json!(85),
            },
            [("type".to_string(), "health_threshold".to_string())]
                .into_iter()
                .collect(),
        )
        .await
        .unwrap();

    let combat_breakpoint = session
        .create_breakpoint(
            BreakpointCondition::OnEventType(EventType::Custom("enemy_encounter".to_string())),
            [("type".to_string(), "enemy_encounter".to_string())]
                .into_iter()
                .collect(),
        )
        .await
        .unwrap();

    let _decision_breakpoint = session
        .create_breakpoint(
            BreakpointCondition::OnEventType(EventType::DecisionMade),
            [("type".to_string(), "decision_point".to_string())]
                .into_iter()
                .collect(),
        )
        .await
        .unwrap();

    // Phase 4: Time navigation and analysis

    // Navigate to the beginning
    let start_position = session.navigate_to_event_index(0).await.unwrap();
    assert_eq!(start_position.event_index, 0);

    // Step through events one by one
    let mut positions = vec![start_position];
    for i in 1..events.len() {
        let position = session
            .step(NavigationDirection::Forward, StepSize::Event)
            .await
            .unwrap();
        assert_eq!(position.event_index, i);
        positions.push(position);
    }

    // Navigate backward
    let back_position = session
        .step(NavigationDirection::Backward, StepSize::Custom(3))
        .await
        .unwrap();
    assert_eq!(back_position.event_index, events.len() - 4);

    // Navigate to specific time
    let mid_time = positions[positions.len() / 2].timestamp;
    let time_position = session.navigate_to_time(mid_time).await.unwrap();
    assert!(time_position.timestamp >= mid_time);

    // Phase 5: State comparison and analysis

    // Compare initial and final states
    let initial_final_comparison = session
        .compare_snapshots(snapshots[0], snapshots[snapshots.len() - 1], None)
        .await
        .unwrap();

    assert!(!initial_final_comparison.changes.is_empty());
    assert!(initial_final_comparison.summary.total_changes > 0);

    // Verify specific changes
    let health_changes: Vec<_> = initial_final_comparison
        .changes
        .iter()
        .filter(|c| c.path == "health")
        .collect();
    assert!(health_changes.is_empty()); // Health should be same (100 -> 100)

    let level_changes: Vec<_> = initial_final_comparison
        .changes
        .iter()
        .filter(|c| c.path == "level")
        .collect();
    assert_eq!(level_changes.len(), 1);
    assert_eq!(level_changes[0].old_value, Some(json!(1)));
    assert_eq!(level_changes[0].new_value, Some(json!(2)));

    // Compare pre and post combat states
    let combat_comparison = session
        .compare_snapshots(
            snapshots[2], // Before combat
            snapshots[3], // After combat
            None,
        )
        .await
        .unwrap();

    let health_change = combat_comparison
        .changes
        .iter()
        .find(|c| c.path == "health")
        .unwrap();
    assert_eq!(health_change.old_value, Some(json!(98)));
    assert_eq!(health_change.new_value, Some(json!(85)));

    // Timeline comparison
    let timeline_comparisons = debugger
        .comparator
        .compare_timeline(
            &[
                debugger
                    .snapshot_manager
                    .get_snapshot(snapshots[0])
                    .await
                    .unwrap(),
                debugger
                    .snapshot_manager
                    .get_snapshot(snapshots[1])
                    .await
                    .unwrap(),
                debugger
                    .snapshot_manager
                    .get_snapshot(snapshots[2])
                    .await
                    .unwrap(),
            ],
            None,
        )
        .await
        .unwrap();

    assert_eq!(timeline_comparisons.len(), 2);

    // Pattern analysis
    let pattern_analysis = debugger
        .comparator
        .analyze_patterns(&timeline_comparisons)
        .await
        .unwrap();

    assert!(!pattern_analysis.volatile_fields.is_empty());
    assert!(!pattern_analysis.similarity_trend.is_empty());

    // Phase 6: Event replay and causality analysis

    // Test causality chain
    let causality_chain = debugger
        .event_log
        .get_causality_chain(collect_event)
        .await
        .unwrap();

    assert!(causality_chain.len() >= 2); // Should include find_event and collect_event
    assert!(causality_chain.iter().any(|e| e.id == find_event));
    assert!(causality_chain.iter().any(|e| e.id == collect_event));

    // Test caused events
    let caused_events = debugger
        .event_log
        .get_caused_events(enemy_event)
        .await
        .unwrap();

    assert!(caused_events.iter().any(|e| e.id == decision_event));

    // Replay events from a specific time
    let replay_start_time = positions[2].timestamp; // Start from third event
    let mut replayed_events = Vec::new();

    let replay_count = debugger
        .event_log
        .replay_from_time(agent_id, replay_start_time, |event| {
            replayed_events.push(event.clone());
            Ok(())
        })
        .await
        .unwrap();

    assert!(replay_count >= 5); // Should replay at least 5 events
    assert!(!replayed_events.is_empty());

    // Phase 7: Breakpoint testing

    // Check if breakpoints would trigger
    let combat_state_snapshot = debugger
        .snapshot_manager
        .get_snapshot(snapshots[3])
        .await
        .unwrap();
    let combat_event_data = debugger.event_log.get_event(combat_event).await.unwrap();

    let triggered_breakpoints = debugger
        .navigator
        .check_breakpoints(session_id, &combat_event_data, Some(&combat_state_snapshot))
        .await
        .unwrap();

    // Health breakpoint should trigger (health is 85)
    assert!(triggered_breakpoints.contains(&health_breakpoint));

    // Test enemy encounter breakpoint
    let enemy_event_data = debugger.event_log.get_event(enemy_event).await.unwrap();
    let enemy_triggered = debugger
        .navigator
        .check_breakpoints(session_id, &enemy_event_data, None)
        .await
        .unwrap();

    assert!(enemy_triggered.contains(&combat_breakpoint));

    // Phase 8: Session management and export/import

    // Get session statistics
    let stats = session.get_statistics().await.unwrap();
    assert_eq!(stats.agent_id, agent_id);
    assert_eq!(stats.snapshot_count, snapshots.len());
    assert_eq!(stats.total_events, events.len());
    assert!(stats.breakpoint_count >= 3); // Our breakpoints + auto breakpoints
    assert!(stats.navigation_steps > 0);

    // Store custom data
    session
        .set_custom_data(
            "test_metadata".to_string(),
            json!({"test_run": "integration", "events_processed": events.len()}),
        )
        .await
        .unwrap();

    let custom_data = session.get_custom_data("test_metadata").await.unwrap();
    assert_eq!(
        custom_data,
        Some(json!({"test_run": "integration", "events_processed": events.len()}))
    );

    // Export session
    let exported_data = session.export_session(ExportFormat::Json).await.unwrap();
    assert!(!exported_data.is_empty());

    // Import session (create new session from exported data)
    let imported_session = session::DebugSession::import_session(
        &exported_data,
        ExportFormat::Json,
        debugger.snapshot_manager.clone(),
        debugger.event_log.clone(),
        debugger.navigator.clone(),
        debugger.comparator.clone(),
    )
    .await
    .unwrap();

    assert_eq!(imported_session.metadata.agent_id, agent_id);
    assert_eq!(
        imported_session.metadata.name,
        Some("Integration Test Session".to_string())
    );

    // Phase 9: System-wide statistics
    let system_stats = debugger.get_system_statistics().await.unwrap();
    assert!(system_stats.total_snapshots >= snapshots.len());
    assert!(system_stats.total_events >= events.len());
    // The imported session is not registered with the session manager, so counts may vary
    assert!(system_stats.total_sessions >= 1);

    // Phase 10: Cleanup
    session.complete().await.unwrap();
    assert_eq!(session.get_state(), SessionState::Completed);

    debugger.stop_background_tasks().await.unwrap();

    // Verify session is completed
    let final_stats = debugger.get_system_statistics().await.unwrap();
    assert_eq!(final_stats.active_sessions, 0);
    // Total sessions might vary due to import/export operations
    assert!(final_stats.total_sessions >= 1);

    println!("Integration test completed successfully!");
    println!("- Processed {} snapshots", snapshots.len());
    println!("- Recorded {} events", events.len());
    println!("- Created {} breakpoints", 3);
    println!("- Performed {} navigation steps", stats.navigation_steps);
    println!(
        "- Total memory usage: {} bytes",
        system_stats.total_memory_usage
    );
}

/// Test concurrent debugging sessions
#[tokio::test]
async fn test_concurrent_sessions() {
    let debugger = TimeDebugger::new();
    debugger.start_background_tasks().await.unwrap();

    let agent1_id = Uuid::new_v4();
    let agent2_id = Uuid::new_v4();

    // Create two concurrent sessions
    let session1_id = debugger.create_session(agent1_id).await.unwrap();
    let session2_id = debugger.create_session(agent2_id).await.unwrap();

    let mut session1 = debugger.get_session(session1_id).await.unwrap();
    let mut session2 = debugger.get_session(session2_id).await.unwrap();

    // Start both sessions
    session1.start().await.unwrap();
    session2.start().await.unwrap();

    // Run concurrent operations
    let handles = vec![
        tokio::spawn(async move {
            // Session 1 operations
            for i in 0..10 {
                session1
                    .take_snapshot(json!({"agent": 1, "step": i}), 1024, HashMap::new())
                    .await
                    .unwrap();

                session1
                    .record_event(
                        EventType::StateChange,
                        json!({"step": i}),
                        HashMap::new(),
                        None,
                    )
                    .await
                    .unwrap();

                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
            session1
        }),
        tokio::spawn(async move {
            // Session 2 operations
            for i in 0..10 {
                session2
                    .take_snapshot(json!({"agent": 2, "step": i}), 2048, HashMap::new())
                    .await
                    .unwrap();

                session2
                    .record_event(
                        EventType::ActionExecution,
                        json!({"step": i}),
                        HashMap::new(),
                        None,
                    )
                    .await
                    .unwrap();

                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
            session2
        }),
    ];

    // Wait for both to complete
    let mut handles = handles.into_iter();
    let handle1 = handles.next().unwrap();
    let handle2 = handles.next().unwrap();
    let (mut session1, mut session2) = tokio::try_join!(handle1, handle2).unwrap();

    // Verify both sessions have their data
    let stats1 = session1.get_statistics().await.unwrap();
    let stats2 = session2.get_statistics().await.unwrap();

    assert_eq!(stats1.agent_id, agent1_id);
    assert_eq!(stats2.agent_id, agent2_id);
    assert_eq!(stats1.snapshot_count, 10);
    assert_eq!(stats2.snapshot_count, 10);
    assert_eq!(stats1.total_events, 10);
    assert_eq!(stats2.total_events, 10);

    // Complete sessions
    session1.complete().await.unwrap();
    session2.complete().await.unwrap();

    debugger.stop_background_tasks().await.unwrap();
}

/// Test error handling and recovery
#[tokio::test]
async fn test_error_handling() {
    let debugger = TimeDebugger::new();
    let agent_id = Uuid::new_v4();

    // Test session operations on non-existent session
    let fake_session_id = Uuid::new_v4();
    let result = debugger.get_session(fake_session_id).await;
    assert!(matches!(
        result,
        Err(TimeDebuggerError::SessionNotFound { .. })
    ));

    // Create session but don't start it
    let session_id = debugger.create_session(agent_id).await.unwrap();
    let session = debugger.get_session(session_id).await.unwrap();

    // Operations should fail on inactive session
    let result = session.take_snapshot(json!({}), 1024, HashMap::new()).await;
    assert!(matches!(
        result,
        Err(TimeDebuggerError::ConcurrentAccess { .. })
    ));

    let result = session
        .record_event(EventType::StateChange, json!({}), HashMap::new(), None)
        .await;
    assert!(matches!(
        result,
        Err(TimeDebuggerError::ConcurrentAccess { .. })
    ));

    // Test navigation on non-existent session
    let result = debugger
        .navigator
        .get_current_position(fake_session_id)
        .await;
    assert!(matches!(
        result,
        Err(TimeDebuggerError::SessionNotFound { .. })
    ));

    // Test snapshot operations
    let fake_snapshot_id = Uuid::new_v4();
    let result = debugger
        .snapshot_manager
        .get_snapshot(fake_snapshot_id)
        .await;
    assert!(matches!(
        result,
        Err(TimeDebuggerError::SnapshotNotFound { .. })
    ));

    // Test event operations
    let fake_event_id = Uuid::new_v4();
    let result = debugger.event_log.get_event(fake_event_id).await;
    assert!(matches!(result, Err(TimeDebuggerError::EmptyEventLog)));

    // Test breakpoint operations
    let fake_breakpoint_id = Uuid::new_v4();
    let result = debugger
        .navigator
        .remove_breakpoint(fake_breakpoint_id)
        .await;
    assert!(matches!(
        result,
        Err(TimeDebuggerError::BreakpointNotFound { .. })
    ));
}

/// Test performance with large datasets
#[tokio::test]
async fn test_performance_large_dataset() {
    let debugger = TimeDebugger::builder()
        .with_snapshot_config(SnapshotConfig {
            max_snapshots: 1000,
            compression_enabled: true,
            diff_threshold: 0.05,
            cleanup_interval: std::time::Duration::from_secs(1),
        })
        .with_event_log_config(event_log::EventLogConfig {
            max_events_per_agent: 5000,
            enable_causality_tracking: true,
            enable_compression: true,
            flush_interval: std::time::Duration::from_secs(1),
            enable_persistence: false,
        })
        .build();

    debugger.start_background_tasks().await.unwrap();

    let agent_id = Uuid::new_v4();
    let session_id = debugger.create_session(agent_id).await.unwrap();
    let mut session = debugger.get_session(session_id).await.unwrap();

    session.start().await.unwrap();

    let start_time = std::time::Instant::now();
    let num_operations = 100;

    // Create large dataset
    for i in 0usize..num_operations {
        // Take snapshot every 10 operations
        if i % 10 == 0 {
            session
                .take_snapshot(
                    json!({
                        "iteration": i,
                        "data": {
                            "values": (0..100).collect::<Vec<i32>>(),
                            "nested": {
                                "deep": {
                                    "value": i,
                                    "timestamp": chrono::Utc::now().timestamp()
                                }
                            }
                        }
                    }),
                    (1024 + i * 10) as u64,
                    [("iteration".to_string(), i.to_string())]
                        .into_iter()
                        .collect(),
                )
                .await
                .unwrap();
        }

        // Record events
        session
            .record_event(
                if i % 3 == 0 {
                    EventType::StateChange
                } else {
                    EventType::ActionExecution
                },
                json!({
                    "iteration": i,
                    "operation": format!("op_{}", i),
                    "data_size": i * 10,
                    "complexity": i % 5
                }),
                [("batch".to_string(), (i / 50).to_string())]
                    .into_iter()
                    .collect(),
                if i > 0 && i % 7 == 0 {
                    // Add some causality relationships
                    debugger
                        .event_log
                        .get_agent_events(agent_id, None, None)
                        .await
                        .ok()
                        .and_then(|events| events.get(i.saturating_sub(7)).map(|e| e.id))
                } else {
                    None
                },
            )
            .await
            .unwrap();
    }

    let creation_time = start_time.elapsed();
    println!(
        "Created {} operations in {:?}",
        num_operations, creation_time
    );

    // Test navigation performance
    let nav_start = std::time::Instant::now();

    // Navigate through time
    for i in 0..std::cmp::min(20, num_operations) {
        session.navigate_to_event_index(i).await.unwrap();
    }

    let nav_time = nav_start.elapsed();
    println!("Navigation through 20 positions took {:?}", nav_time);

    // Test comparison performance
    let comp_start = std::time::Instant::now();

    let snapshots = debugger
        .snapshot_manager
        .get_agent_snapshots(agent_id)
        .await
        .unwrap();
    if snapshots.len() >= 2 {
        let comparison = debugger
            .comparator
            .compare_snapshots(&snapshots[0], &snapshots[snapshots.len() - 1], None)
            .await
            .unwrap();

        println!("Comparison found {} changes", comparison.changes.len());
    }

    let comp_time = comp_start.elapsed();
    println!("State comparison took {:?}", comp_time);

    // Test query performance
    let query_start = std::time::Instant::now();

    let all_events = debugger
        .event_log
        .get_agent_events(agent_id, None, None)
        .await
        .unwrap();
    let state_change_events = debugger
        .event_log
        .get_events_by_type(agent_id, EventType::StateChange)
        .await
        .unwrap();

    let query_time = query_start.elapsed();
    println!(
        "Event queries took {:?} ({} total events, {} state changes)",
        query_time,
        all_events.len(),
        state_change_events.len()
    );

    // Get final statistics
    let stats = session.get_statistics().await.unwrap();
    let system_stats = debugger.get_system_statistics().await.unwrap();

    println!("Final statistics:");
    println!("- Snapshots: {}", stats.snapshot_count);
    println!("- Events: {}", stats.total_events);
    println!("- Memory usage: {} bytes", system_stats.total_memory_usage);
    println!("- Session duration: {:?}", stats.session_duration);

    // Cleanup
    session.complete().await.unwrap();
    debugger.stop_background_tasks().await.unwrap();

    // Performance assertions
    assert!(creation_time < std::time::Duration::from_secs(5));
    assert!(nav_time < std::time::Duration::from_secs(1));
    assert!(comp_time < std::time::Duration::from_secs(1));
    assert!(query_time < std::time::Duration::from_secs(1));
}
