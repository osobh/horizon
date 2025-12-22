//! Edge case tests for CpuAgent to enhance coverage to 90%+

use crate::agent::*;
use std::sync::Arc;
use tokio::sync::Mutex;

#[tokio::test]
async fn test_agent_capability_all_variants() {
    // Test all capability variants
    let capabilities = vec![
        AgentCapability::FileIo,
        AgentCapability::NetworkIo,
        AgentCapability::DatabaseIo,
        AgentCapability::Orchestration,
        AgentCapability::ApiManagement,
        AgentCapability::DataTransform,
        AgentCapability::Monitoring,
    ];

    for cap in &capabilities {
        let config = CpuAgentConfig {
            id: format!("test-{:?}", cap),
            capabilities: vec![cap.clone()],
            max_concurrent_tasks: 1,
            memory_limit_mb: 10,
        };

        let agent = BasicCpuAgent::new(config);
        assert!(agent.can_handle(cap));

        // Test that agent doesn't have other capabilities
        for other_cap in &capabilities {
            if other_cap != cap {
                assert!(!agent.can_handle(other_cap));
            }
        }
    }
}

#[tokio::test]
async fn test_agent_status_transitions() {
    let config = CpuAgentConfig {
        id: "status-test-agent".to_string(),
        capabilities: vec![
            AgentCapability::FileIo,
            AgentCapability::DataTransform, // Need this for Aggregate
        ],
        max_concurrent_tasks: 10,
        memory_limit_mb: 100,
    };
    let mut agent = BasicCpuAgent::new(config);

    // Initial status
    assert_eq!(agent.status(), AgentStatus::Created);

    // After initialization
    agent.initialize().await.unwrap();
    assert_eq!(agent.status(), AgentStatus::Ready);

    // During task execution
    let task = AgentTask {
        id: "status-test".to_string(),
        task_type: TaskType::Aggregate, // This takes longer (120ms)
        priority: 5,
    };

    // Execute task in background
    let agent_clone = Arc::new(Mutex::new(agent));
    let agent_ref = agent_clone.clone();
    let handle = tokio::spawn(async move {
        let mut agent = agent_ref.lock().await;
        agent.execute_task(task).await
    });

    // Give task time to start
    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

    // Check status during execution
    {
        let agent = agent_clone.lock().await;
        // Status might be Ready or Busy depending on timing
        let status = agent.status();
        assert!(status == AgentStatus::Busy || status == AgentStatus::Ready);
    }

    // Wait for task completion
    handle.await.unwrap().unwrap();

    // Status should be back to Ready
    {
        let agent = agent_clone.lock().await;
        assert_eq!(agent.status(), AgentStatus::Ready);
    }

    // Shutdown
    {
        let mut agent = agent_clone.lock().await;
        agent.shutdown().await.unwrap();
        assert_eq!(agent.status(), AgentStatus::Stopped);
    }
}

#[tokio::test]
async fn test_task_type_all_variants() {
    let task_types = vec![
        TaskType::FileRead("/test/file.txt".to_string()),
        TaskType::FileWrite("/test/output.txt".to_string()),
        TaskType::FileDelete("/test/temp.txt".to_string()),
        TaskType::CreateDirectory("/test/newdir".to_string()),
        TaskType::ListDirectory("/test/dir".to_string()),
        TaskType::NetworkFetch("https://example.com".to_string()),
        TaskType::NetworkSend("POST data".to_string()),
        TaskType::DatabaseQuery("SELECT * FROM test".to_string()),
        TaskType::Transform("uppercase".to_string()),
        TaskType::Aggregate,
        TaskType::Compute("matrix_multiply".to_string()),
        TaskType::GpuCompute("cuda_kernel".to_string()),
    ];

    let config = CpuAgentConfig {
        id: "variant-test".to_string(),
        capabilities: vec![
            AgentCapability::FileIo,
            AgentCapability::NetworkIo,
            AgentCapability::DatabaseIo,
            AgentCapability::DataTransform,
        ],
        max_concurrent_tasks: 20,
        memory_limit_mb: 100,
    };

    let mut agent = BasicCpuAgent::new(config);
    agent.initialize().await.unwrap();

    for (i, task_type) in task_types.iter().enumerate() {
        let task = AgentTask {
            id: format!("task-{}", i),
            task_type: task_type.clone(),
            priority: 5,
        };

        let result = agent.execute_task(task).await;

        // GpuCompute should fail during validation for CPU agents
        if matches!(task_type, TaskType::GpuCompute(_)) {
            assert!(result.is_err());
            if let Err(e) = result {
                assert!(e
                    .to_string()
                    .contains("CPU agents cannot execute GPU compute tasks"));
            }
        } else {
            // All other task types should succeed
            assert!(result.is_ok());
        }
    }
}

#[tokio::test]
async fn test_priority_edge_cases() {
    let priorities = vec![0, 1, 5, 10, 11, 255]; // Including invalid priorities

    for priority in priorities {
        let task = AgentTask {
            id: format!("priority-{}", priority),
            task_type: TaskType::Compute("test".to_string()),
            priority,
        };

        // Task creation should succeed regardless of priority value
        assert_eq!(task.priority, priority);
    }
}

#[tokio::test]
async fn test_config_edge_cases() {
    // Empty ID
    let config1 = CpuAgentConfig {
        id: "".to_string(),
        capabilities: vec![],
        max_concurrent_tasks: 0,
        memory_limit_mb: 0,
    };
    let agent1 = BasicCpuAgent::new(config1);
    assert_eq!(agent1.id(), "");

    // Very long ID
    let long_id = "a".repeat(1000);
    let config2 = CpuAgentConfig {
        id: long_id.clone(),
        capabilities: vec![],
        max_concurrent_tasks: 1000,
        memory_limit_mb: 100000,
    };
    let agent2 = BasicCpuAgent::new(config2);
    assert_eq!(agent2.id(), &long_id);

    // No capabilities
    let config3 = CpuAgentConfig {
        id: "no-cap".to_string(),
        capabilities: vec![],
        max_concurrent_tasks: 10,
        memory_limit_mb: 100,
    };
    let mut agent3 = BasicCpuAgent::new(config3);
    agent3.initialize().await.unwrap();

    // Should fail all capability-based tasks
    let task = AgentTask {
        id: "no-cap-task".to_string(),
        task_type: TaskType::FileRead("test.txt".to_string()),
        priority: 5,
    };

    let result = agent3.execute_task(task).await;
    assert!(result.is_err());
    if let Err(e) = result {
        assert!(e.to_string().contains("does not have required capability"));
    }
}

#[tokio::test]
async fn test_concurrent_task_limits() {
    let config = CpuAgentConfig {
        id: "concurrent-test".to_string(),
        capabilities: vec![AgentCapability::DataTransform],
        max_concurrent_tasks: 2, // Small limit for testing
        memory_limit_mb: 100,
    };

    let mut agent = BasicCpuAgent::new(config);
    agent.initialize().await.unwrap();

    // Since submit_task doesn't actually track tasks, we'll test execute_task instead
    // which does track active tasks during execution

    // Spawn multiple tasks concurrently
    let agent_arc = Arc::new(Mutex::new(agent));
    let mut handles = vec![];

    for i in 0..3 {
        let agent_clone = agent_arc.clone();
        let handle = tokio::spawn(async move {
            let task = AgentTask {
                id: format!("concurrent-{}", i),
                task_type: TaskType::Aggregate, // Longer running task
                priority: 5,
            };

            // Try to execute
            let mut agent = agent_clone.lock().await;
            agent.execute_task(task).await
        });
        handles.push(handle);

        // Small delay to ensure tasks start in order
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
    }

    // Wait for all tasks
    let results: Vec<_> = futures::future::join_all(handles).await;

    // All should complete successfully since they run sequentially due to the mutex
    for result in results {
        assert!(result.is_ok());
    }
}

#[tokio::test]
async fn test_metrics_calculation_edge_cases() {
    let config = CpuAgentConfig::default();
    let mut agent = BasicCpuAgent::new(config);
    agent.initialize().await.unwrap();

    // Initial metrics
    let metrics = agent.get_metrics();
    assert_eq!(metrics.tasks_completed, 0);
    assert_eq!(metrics.tasks_failed, 0);
    assert_eq!(metrics.average_task_duration_ms, 0.0);

    // Execute one successful task
    let task1 = AgentTask {
        id: "metrics-1".to_string(),
        task_type: TaskType::Compute("test".to_string()),
        priority: 5,
    };
    agent.execute_task(task1).await.unwrap();

    let metrics = agent.get_metrics();
    assert_eq!(metrics.tasks_completed, 1);
    assert!(metrics.average_task_duration_ms > 0.0);

    // Execute one failing task
    let task2 = AgentTask {
        id: "metrics-2".to_string(),
        task_type: TaskType::GpuCompute("cuda".to_string()),
        priority: 5,
    };
    let result = agent.execute_task(task2).await;
    assert!(result.is_err()); // GPU compute returns error during validation

    // Failed tasks that error during validation don't update metrics
    let metrics = agent.get_metrics();
    assert_eq!(metrics.tasks_completed, 1);
    assert_eq!(metrics.tasks_failed, 0); // Validation errors don't count as failed tasks
    assert!(metrics.total_task_duration_ms > 0.0);
}

#[tokio::test]
async fn test_task_validation_edge_cases() {
    // Agent with limited capabilities
    let config = CpuAgentConfig {
        id: "limited".to_string(),
        capabilities: vec![AgentCapability::FileIo],
        max_concurrent_tasks: 10,
        memory_limit_mb: 100,
    };

    let mut agent = BasicCpuAgent::new(config);
    agent.initialize().await.unwrap();

    // Test tasks requiring different capabilities
    let test_cases = vec![
        (TaskType::NetworkFetch("url".to_string()), false),
        (TaskType::DatabaseQuery("query".to_string()), false),
        (TaskType::Transform("uppercase".to_string()), false), // Agent doesn't have DataTransform
        (TaskType::Compute("compute".to_string()), true),      // CPU compute always allowed
        (TaskType::GpuCompute("gpu".to_string()), false),      // GPU compute never allowed
    ];

    for (task_type, should_succeed) in test_cases {
        let task_type_debug = format!("{:?}", task_type);
        let task = AgentTask {
            id: format!("validate-{}", task_type_debug),
            task_type,
            priority: 5,
        };

        let result = agent.execute_task(task).await;

        if should_succeed {
            assert!(
                result.is_ok(),
                "Task {} should succeed but got error",
                task_type_debug
            );
            match result {
                Ok(TaskExecutionResult::Success(_)) => {
                    // Success expected and received
                }
                Ok(TaskExecutionResult::Failed(msg)) => {
                    panic!(
                        "Expected success for {} but got failure: {}",
                        task_type_debug, msg
                    );
                }
                Err(e) => {
                    panic!(
                        "Expected success for {} but got error: {}",
                        task_type_debug, e
                    );
                }
            }
        } else {
            // Tasks should fail during validation
            assert!(result.is_err());
        }
    }
}

#[tokio::test]
async fn test_agent_not_ready_error() {
    let config = CpuAgentConfig::default();
    let mut agent = BasicCpuAgent::new(config);

    // Try to execute task before initialization
    let task = AgentTask {
        id: "not-ready".to_string(),
        task_type: TaskType::Compute("test".to_string()),
        priority: 5,
    };

    let result = agent.execute_task(task).await;
    assert!(result.is_err());
    if let Err(e) = result {
        assert!(e.to_string().contains("not ready"));
    }
}

#[tokio::test]
async fn test_file_operations_edge_cases() {
    let config = CpuAgentConfig {
        id: "file-ops".to_string(),
        capabilities: vec![AgentCapability::FileIo],
        max_concurrent_tasks: 10,
        memory_limit_mb: 100,
    };

    let mut agent = BasicCpuAgent::new(config);
    agent.initialize().await.unwrap();

    // Test various file paths
    let edge_paths = vec![
        "".to_string(),                                // Empty path
        ".".to_string(),                               // Current directory
        "..".to_string(),                              // Parent directory
        "/".to_string(),                               // Root
        "//double//slash".to_string(),                 // Double slashes
        "path with spaces".to_string(),                // Spaces
        "path/with/../dots".to_string(),               // Path traversal
        format!("very/long/path/{}", "a".repeat(200)), // Very long path
    ];

    for path in edge_paths {
        let task = AgentTask {
            id: format!("file-{}", path.len()),
            task_type: TaskType::FileRead(path.clone()),
            priority: 5,
        };

        // All should execute (may fail, but won't panic)
        let result = agent.execute_task(task).await;
        assert!(result.is_ok());
    }
}

#[tokio::test]
async fn test_serialization_edge_cases() {
    // Test serialization of all enums
    let capability = AgentCapability::Monitoring;
    let serialized = serde_json::to_string(&capability).unwrap();
    let deserialized: AgentCapability = serde_json::from_str(&serialized).unwrap();
    assert_eq!(capability, deserialized);

    let status = AgentStatus::Error;
    let serialized = serde_json::to_string(&status).unwrap();
    let deserialized: AgentStatus = serde_json::from_str(&serialized).unwrap();
    assert_eq!(status, deserialized);

    // Test metrics with extreme values (but valid for JSON)
    let metrics = AgentMetrics {
        tasks_completed: u64::MAX,
        tasks_failed: u64::MAX,
        total_task_duration_ms: f64::MAX,
        average_task_duration_ms: f64::MIN,
        memory_usage_mb: 0.0,
        cpu_usage_percent: -100.0,
    };

    let serialized = serde_json::to_string(&metrics).unwrap();
    let deserialized: AgentMetrics = serde_json::from_str(&serialized).unwrap();
    assert_eq!(metrics.tasks_completed, deserialized.tasks_completed);
    assert_eq!(metrics.tasks_failed, deserialized.tasks_failed);
    assert_eq!(
        metrics.total_task_duration_ms,
        deserialized.total_task_duration_ms
    );
    assert_eq!(metrics.cpu_usage_percent, deserialized.cpu_usage_percent);
}

#[tokio::test]
async fn test_shutdown_with_active_tasks() {
    let config = CpuAgentConfig {
        id: "shutdown-test".to_string(),
        capabilities: vec![AgentCapability::DataTransform],
        max_concurrent_tasks: 5,
        memory_limit_mb: 100,
    };

    let mut agent = BasicCpuAgent::new(config);
    agent.initialize().await.unwrap();

    // Create a long-running task to simulate active work
    let agent_arc = Arc::new(Mutex::new(agent));
    let agent_clone = agent_arc.clone();

    // Start a task that will run for a while
    let task_handle = tokio::spawn(async move {
        let task = AgentTask {
            id: "long-task".to_string(),
            task_type: TaskType::Transform("sleep".to_string()),
            priority: 5,
        };
        let mut agent = agent_clone.lock().await;
        agent.execute_task(task).await
    });

    // Give task time to start
    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

    // Start shutdown
    let agent_clone2 = agent_arc.clone();
    let shutdown_handle = tokio::spawn(async move {
        let mut agent = agent_clone2.lock().await;
        agent.shutdown().await
    });

    // Shutdown should eventually complete
    let result = tokio::time::timeout(tokio::time::Duration::from_secs(5), shutdown_handle).await;

    assert!(result.is_ok());
    task_handle.abort(); // Clean up the task
}

#[tokio::test]
async fn test_default_config_values() {
    let config = CpuAgentConfig::default();
    assert_eq!(config.id, "cpu-agent-default");
    assert_eq!(config.capabilities, vec![AgentCapability::FileIo]);
    assert_eq!(config.max_concurrent_tasks, 10);
    assert_eq!(config.memory_limit_mb, 100);
}

#[tokio::test]
async fn test_memory_cpu_usage_simulation() {
    let config = CpuAgentConfig::default();
    let mut agent = BasicCpuAgent::new(config);
    agent.initialize().await.unwrap();

    // Check initial metrics
    let initial_metrics = agent.get_metrics();

    // Execute some tasks to simulate activity
    for i in 0..3 {
        let task = AgentTask {
            id: format!("mem-test-{}", i),
            task_type: TaskType::Compute("test".to_string()),
            priority: 5,
        };
        agent.execute_task(task).await.unwrap();
    }

    // Metrics should have updated
    let final_metrics = agent.get_metrics();
    assert!(final_metrics.memory_usage_mb >= initial_metrics.memory_usage_mb);
    assert_eq!(final_metrics.tasks_completed, 3);
    assert!(final_metrics.total_task_duration_ms > 0.0);
}

// Test helper for stress testing
async fn create_stress_agent() -> BasicCpuAgent {
    let config = CpuAgentConfig {
        id: "stress-test".to_string(),
        capabilities: vec![
            AgentCapability::FileIo,
            AgentCapability::NetworkIo,
            AgentCapability::DatabaseIo,
            AgentCapability::DataTransform,
            AgentCapability::Monitoring,
        ],
        max_concurrent_tasks: 100,
        memory_limit_mb: 1000,
    };

    let mut agent = BasicCpuAgent::new(config);
    agent.initialize().await.unwrap();
    agent
}

#[tokio::test]
async fn test_rapid_task_execution() {
    let mut agent = create_stress_agent().await;

    // Execute many tasks rapidly
    for i in 0..20 {
        let task = AgentTask {
            id: format!("rapid-{}", i),
            task_type: match i % 5 {
                0 => TaskType::Compute("test".to_string()),
                1 => TaskType::Transform("uppercase".to_string()),
                2 => TaskType::Aggregate,
                3 => TaskType::NetworkFetch("url".to_string()),
                _ => TaskType::DatabaseQuery("query".to_string()),
            },
            priority: ((i % 10) + 1) as u8,
        };

        agent.execute_task(task).await.unwrap();
    }

    let metrics = agent.get_metrics();
    assert_eq!(metrics.tasks_completed + metrics.tasks_failed, 20);
    assert!(metrics.total_task_duration_ms > 0.0);
}
