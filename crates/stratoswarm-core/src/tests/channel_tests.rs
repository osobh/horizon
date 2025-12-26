//! Comprehensive integration tests for channel infrastructure.
//!
//! These tests verify all channel functionality including creation, message passing,
//! broadcast, request/response patterns, backpressure, and shutdown behavior.

use crate::channels::{
    messages::{
        CostMessage, EfficiencyMessage, EvolutionMessage, GovernorMessage, GpuCommand,
        KnowledgeMessage, SchedulerMessage, SelectionStrategy, SystemEvent,
    },
    patterns::{request_with_timeout, Request},
    registry::ChannelRegistry,
};
use bytes::Bytes;
use std::time::Duration;
use uuid::Uuid;

#[tokio::test]
async fn test_channel_creation_and_message_passing() {
    let registry = ChannelRegistry::new();

    // Get a GPU command sender
    let gpu_tx = registry.gpu_sender();

    // Get a receiver
    let mut gpu_rx = registry.subscribe_gpu();

    // Send a GPU command
    let kernel_id = "test_kernel".to_string();
    let cmd = GpuCommand::LaunchKernel {
        kernel_id: kernel_id.clone(),
        grid_dim: (1, 1, 1),
        block_dim: (256, 1, 1),
        params: Bytes::from(vec![1, 2, 3, 4]),
    };

    gpu_tx.send(cmd.clone()).await.unwrap();

    // Receive the command
    let received = gpu_rx.recv().await.unwrap();

    match received {
        GpuCommand::LaunchKernel {
            kernel_id: recv_id,
            grid_dim,
            block_dim,
            params,
        } => {
            assert_eq!(recv_id, kernel_id);
            assert_eq!(grid_dim, (1, 1, 1));
            assert_eq!(block_dim, (256, 1, 1));
            assert_eq!(params, Bytes::from(vec![1, 2, 3, 4]));
        }
        _ => panic!("Unexpected command type"),
    }
}

#[tokio::test]
async fn test_evolution_messages() {
    let registry = ChannelRegistry::new();

    let evo_tx = registry.evolution_sender();
    let mut evo_rx = registry.subscribe_evolution();

    let msg = EvolutionMessage::Step { generation: 42 };
    evo_tx.send(msg).await.unwrap();

    let received = evo_rx.recv().await.unwrap();
    match received {
        EvolutionMessage::Step { generation } => {
            assert_eq!(generation, 42);
        }
        _ => panic!("Unexpected message type"),
    }
}

#[tokio::test]
async fn test_broadcast_to_multiple_subscribers() {
    let registry = ChannelRegistry::new();

    // Create multiple subscribers
    let mut sub1 = registry.subscribe_events();
    let mut sub2 = registry.subscribe_events();
    let mut sub3 = registry.subscribe_events();

    let event_tx = registry.event_sender();

    // Broadcast an event
    let agent_id = Uuid::new_v4();
    let event = SystemEvent::AgentSpawned {
        agent_id,
        agent_type: "TestAgent".to_string(),
        timestamp: 1000,
    };

    event_tx.send(event.clone()).unwrap();

    // All subscribers should receive the event
    let recv1 = sub1.recv().await.unwrap();
    let recv2 = sub2.recv().await.unwrap();
    let recv3 = sub3.recv().await.unwrap();

    // Verify all received the same event
    for recv in [recv1, recv2, recv3] {
        match recv {
            SystemEvent::AgentSpawned {
                agent_id: recv_id,
                agent_type,
                timestamp,
            } => {
                assert_eq!(recv_id, agent_id);
                assert_eq!(agent_type, "TestAgent");
                assert_eq!(timestamp, 1000);
            }
            _ => panic!("Unexpected event type"),
        }
    }
}

#[tokio::test]
async fn test_request_response_pattern_success() {
    let registry = ChannelRegistry::new();
    let cost_tx = registry.cost_sender();
    let cost_rx = registry.subscribe_cost();

    // Spawn a handler that responds to cost queries
    tokio::spawn(async move {
        loop {
            let request = cost_rx.lock().await.recv().await;
            if let Some(request) = request {
                match request.payload {
                    CostMessage::QueryCost => {
                        let response = CostMessage::CostUpdate {
                            total_cents: 1234,
                            breakdown: vec![("gpu".to_string(), 1000), ("cpu".to_string(), 234)],
                        };
                        request.respond(response).unwrap();
                    }
                    _ => {}
                }
            } else {
                break;
            }
        }
    });

    // Send a request with timeout
    let response = request_with_timeout(
        &cost_tx,
        CostMessage::QueryCost,
        Duration::from_secs(5),
    )
    .await
    .unwrap();

    match response {
        CostMessage::CostUpdate {
            total_cents,
            breakdown,
        } => {
            assert_eq!(total_cents, 1234);
            assert_eq!(breakdown.len(), 2);
        }
        _ => panic!("Unexpected response type"),
    }
}

#[tokio::test]
async fn test_request_response_timeout() {
    let registry = ChannelRegistry::new();
    let efficiency_tx = registry.efficiency_sender();
    let _efficiency_rx = registry.subscribe_efficiency();

    // Don't handle the request, let it timeout
    let result = request_with_timeout(
        &efficiency_tx,
        EfficiencyMessage::QueryMetrics,
        Duration::from_millis(100),
    )
    .await;

    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        crate::error::ChannelError::Timeout { .. }
    ));
}

#[tokio::test]
async fn test_backpressure_on_bounded_channels() {
    let registry = ChannelRegistry::new();
    let gpu_tx = registry.gpu_sender();

    // GPU channel has buffer size of 100
    // Try to send 150 messages without receiving
    let mut send_tasks = Vec::new();

    for i in 0..150 {
        let tx = gpu_tx.clone();
        let task = tokio::spawn(async move {
            let cmd = GpuCommand::Synchronize {
                stream_id: Some(i),
            };
            tx.send(cmd).await
        });
        send_tasks.push(task);
    }

    // First 100 should succeed immediately
    // Remaining 50 should block until we start receiving

    let mut gpu_rx = registry.subscribe_gpu();

    // Start receiving to unblock senders
    tokio::spawn(async move {
        while gpu_rx.recv().await.is_ok() {
            // Just consume messages
        }
    });

    // All sends should eventually complete
    for task in send_tasks {
        task.await.unwrap().unwrap();
    }
}

#[tokio::test]
async fn test_graceful_shutdown_on_channel_close() {
    let registry = ChannelRegistry::new();
    let scheduler_tx = registry.scheduler_sender();
    let scheduler_rx = registry.subscribe_scheduler();

    // Send a message
    let task_id = Uuid::new_v4();
    let (resp_tx, _resp_rx) = tokio::sync::oneshot::channel();
    let msg = Request::new(
        SchedulerMessage::QueryStatus { task_id },
        crate::channels::patterns::Responder::new(resp_tx),
    );
    scheduler_tx.send(msg).await.unwrap();

    // Receive it
    let received = scheduler_rx.lock().await.recv().await;
    assert!(received.is_some());

    // Drop all senders
    drop(scheduler_tx);
    drop(registry);

    // Receiver should detect closed channel
    let result = scheduler_rx.lock().await.recv().await;
    assert!(result.is_none());
}

#[tokio::test]
async fn test_all_channel_types() {
    let registry = ChannelRegistry::new();

    // Test GPU channel
    {
        let tx = registry.gpu_sender();
        let mut rx = registry.subscribe_gpu();
        tx.send(GpuCommand::Synchronize { stream_id: None })
            .await
            .unwrap();
        assert!(rx.recv().await.is_ok());
    }

    // Test Evolution channel
    {
        let tx = registry.evolution_sender();
        let mut rx = registry.subscribe_evolution();
        tx.send(EvolutionMessage::GetBest { count: 5 })
            .await
            .unwrap();
        assert!(rx.recv().await.is_ok());
    }

    // Test Cost channel
    {
        let tx = registry.cost_sender();
        let rx = registry.subscribe_cost();
        tx.send(Request::new(
            CostMessage::QueryCost,
            crate::channels::patterns::Responder::new(tokio::sync::oneshot::channel().0),
        ))
        .await
        .unwrap();
        assert!(rx.lock().await.recv().await.is_some());
    }

    // Test Efficiency channel
    {
        let tx = registry.efficiency_sender();
        let rx = registry.subscribe_efficiency();
        tx.send(Request::new(
            EfficiencyMessage::QueryMetrics,
            crate::channels::patterns::Responder::new(tokio::sync::oneshot::channel().0),
        ))
        .await
        .unwrap();
        assert!(rx.lock().await.recv().await.is_some());
    }

    // Test Scheduler channel
    {
        let tx = registry.scheduler_sender();
        let rx = registry.subscribe_scheduler();
        tx.send(Request::new(
            SchedulerMessage::QueryStatus {
                task_id: Uuid::new_v4(),
            },
            crate::channels::patterns::Responder::new(tokio::sync::oneshot::channel().0),
        ))
        .await
        .unwrap();
        assert!(rx.lock().await.recv().await.is_some());
    }

    // Test Governor channel
    {
        let tx = registry.governor_sender();
        let rx = registry.subscribe_governor();
        tx.send(Request::new(
            GovernorMessage::QueryAvailable,
            crate::channels::patterns::Responder::new(tokio::sync::oneshot::channel().0),
        ))
        .await
        .unwrap();
        assert!(rx.lock().await.recv().await.is_some());
    }

    // Test Knowledge channel
    {
        let tx = registry.knowledge_sender();
        let rx = registry.subscribe_knowledge();
        tx.send(Request::new(
            KnowledgeMessage::Retrieve {
                key: "test".to_string(),
            },
            crate::channels::patterns::Responder::new(tokio::sync::oneshot::channel().0),
        ))
        .await
        .unwrap();
        assert!(rx.lock().await.recv().await.is_some());
    }

    // Test Events broadcast
    {
        let tx = registry.event_sender();
        let mut rx = registry.subscribe_events();
        tx.send(SystemEvent::Error {
            message: "test".to_string(),
            source: "test".to_string(),
            timestamp: 1000,
        })
        .unwrap();
        assert!(rx.recv().await.is_ok());
    }
}

#[tokio::test]
async fn test_zero_copy_bytes_transfer() {
    let registry = ChannelRegistry::new();
    let gpu_tx = registry.gpu_sender();
    let mut gpu_rx = registry.subscribe_gpu();

    // Create a large buffer
    let data = vec![42u8; 1024 * 1024]; // 1MB
    let bytes = Bytes::from(data);
    let original_ptr = bytes.as_ptr();

    // Send via channel
    let cmd = GpuCommand::TransferToDevice {
        buffer_id: "test_buffer".to_string(),
        data: bytes,
        offset: 0,
    };

    gpu_tx.send(cmd).await.unwrap();

    // Receive and verify zero-copy (same pointer)
    let received = gpu_rx.recv().await.unwrap();
    match received {
        GpuCommand::TransferToDevice { data, .. } => {
            assert_eq!(data.len(), 1024 * 1024);
            assert_eq!(data.as_ptr(), original_ptr);
            assert_eq!(data[0], 42);
        }
        _ => panic!("Unexpected command type"),
    }
}

#[tokio::test]
async fn test_evolution_selection_strategies() {
    let registry = ChannelRegistry::new();
    let evo_tx = registry.evolution_sender();
    let mut evo_rx = registry.subscribe_evolution();

    let strategies = vec![
        SelectionStrategy::Tournament { size: 3 },
        SelectionStrategy::Roulette,
        SelectionStrategy::Rank,
        SelectionStrategy::Elitist,
    ];

    for strategy in strategies {
        let msg = EvolutionMessage::Selection {
            strategy,
            count: 10,
        };
        evo_tx.send(msg).await.unwrap();

        let received = evo_rx.recv().await.unwrap();
        match received {
            EvolutionMessage::Selection {
                strategy: recv_strategy,
                count,
            } => {
                assert_eq!(count, 10);
                // Verify strategy matches (simplified check)
                match (strategy, recv_strategy) {
                    (SelectionStrategy::Tournament { .. }, SelectionStrategy::Tournament { .. }) => {}
                    (SelectionStrategy::Roulette, SelectionStrategy::Roulette) => {}
                    (SelectionStrategy::Rank, SelectionStrategy::Rank) => {}
                    (SelectionStrategy::Elitist, SelectionStrategy::Elitist) => {}
                    _ => panic!("Strategy mismatch"),
                }
            }
            _ => panic!("Unexpected message type"),
        }
    }
}

#[tokio::test]
async fn test_concurrent_senders_and_receivers() {
    let registry = ChannelRegistry::new();

    // Spawn multiple senders
    let mut sender_tasks = Vec::new();
    for i in 0..10 {
        let tx = registry.gpu_sender();
        let task = tokio::spawn(async move {
            for j in 0..100 {
                let cmd = GpuCommand::Synchronize {
                    stream_id: Some(i * 100 + j),
                };
                tx.send(cmd).await.unwrap();
            }
        });
        sender_tasks.push(task);
    }

    // Spawn multiple receivers
    let mut receiver_tasks = Vec::new();
    for _ in 0..5 {
        let mut rx = registry.subscribe_gpu();
        let task = tokio::spawn(async move {
            let mut count = 0;
            while count < 1000 {
                if rx.recv().await.is_ok() {
                    count += 1;
                } else {
                    break;
                }
            }
            count
        });
        receiver_tasks.push(task);
    }

    // Wait for all senders
    for task in sender_tasks {
        task.await.unwrap();
    }

    // Drop senders to signal completion
    drop(registry);

    // Verify all messages were received
    // With broadcast, each receiver gets ALL messages
    let mut total_received = 0;
    for task in receiver_tasks {
        total_received += task.await.unwrap();
    }

    // Each of 5 receivers should get all 1000 messages
    assert_eq!(total_received, 1000 * 5);
}

#[tokio::test]
async fn test_system_events_all_types() {
    let registry = ChannelRegistry::new();
    let tx = registry.event_sender();
    let mut rx = registry.subscribe_events();

    let events = vec![
        SystemEvent::AgentSpawned {
            agent_id: Uuid::new_v4(),
            agent_type: "TestAgent".to_string(),
            timestamp: 1000,
        },
        SystemEvent::FitnessImproved {
            individual_id: Uuid::new_v4(),
            old_fitness: 0.5,
            new_fitness: 0.8,
            timestamp: 2000,
        },
        SystemEvent::GpuUtilization {
            device_id: 0,
            utilization: 75.5,
            timestamp: 3000,
        },
        SystemEvent::MemoryPressure {
            usage_percent: 85.0,
            available_bytes: 1024 * 1024 * 1024,
            timestamp: 4000,
        },
        SystemEvent::Error {
            message: "Test error".to_string(),
            source: "test".to_string(),
            timestamp: 5000,
        },
    ];

    for event in events {
        tx.send(event).unwrap();
        assert!(rx.recv().await.is_ok());
    }
}
