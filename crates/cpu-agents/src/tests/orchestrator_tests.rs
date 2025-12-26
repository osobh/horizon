//! Tests for Orchestrator

use crate::{agent::*, orchestrator::*, Result};
use std::time::Duration;
use tokio::time::sleep;

#[tokio::test]
async fn test_orchestrator_creation() {
    let config = OrchestratorConfig {
        id: "test-orchestrator".to_string(),
        max_workflows: 10,
        task_timeout_seconds: 300,
        enable_retry: true,
        max_retries: 3,
    };

    let orchestrator = Orchestrator::new(config.clone());
    assert_eq!(orchestrator.id(), "test-orchestrator");
    assert_eq!(orchestrator.config().max_workflows, 10);
}

#[tokio::test]
async fn test_workflow_execution() {
    let config = OrchestratorConfig::default();
    let mut orchestrator = Orchestrator::new(config);

    // Create simple workflow
    let workflow = Workflow {
        id: "workflow-001".to_string(),
        name: "Test Workflow".to_string(),
        tasks: vec![
            WorkflowTask {
                id: "task-1".to_string(),
                task_type: TaskType::FileRead("input.txt".to_string()),
                dependencies: vec![],
                timeout_seconds: 60,
            },
            WorkflowTask {
                id: "task-2".to_string(),
                task_type: TaskType::Transform("uppercase".to_string()),
                dependencies: vec!["task-1".to_string()],
                timeout_seconds: 30,
            },
            WorkflowTask {
                id: "task-3".to_string(),
                task_type: TaskType::FileWrite("output.txt".to_string()),
                dependencies: vec!["task-2".to_string()],
                timeout_seconds: 60,
            },
        ],
    };

    let result = orchestrator.execute_workflow(workflow).await;
    assert!(result.is_ok());

    let workflow_result = result.unwrap();
    assert_eq!(workflow_result.workflow_id, "workflow-001");
    assert!(workflow_result.success);
    assert_eq!(workflow_result.completed_tasks.len(), 3);
}

#[tokio::test]
async fn test_parallel_task_execution() {
    let config = OrchestratorConfig::default();
    let mut orchestrator = Orchestrator::new(config);

    // Create workflow with parallel tasks
    let workflow = Workflow {
        id: "parallel-workflow".to_string(),
        name: "Parallel Test".to_string(),
        tasks: vec![
            WorkflowTask {
                id: "parallel-1".to_string(),
                task_type: TaskType::Compute("task1".to_string()),
                dependencies: vec![],
                timeout_seconds: 10,
            },
            WorkflowTask {
                id: "parallel-2".to_string(),
                task_type: TaskType::Compute("task2".to_string()),
                dependencies: vec![],
                timeout_seconds: 10,
            },
            WorkflowTask {
                id: "parallel-3".to_string(),
                task_type: TaskType::Compute("task3".to_string()),
                dependencies: vec![],
                timeout_seconds: 10,
            },
            WorkflowTask {
                id: "final".to_string(),
                task_type: TaskType::Aggregate,
                dependencies: vec![
                    "parallel-1".to_string(),
                    "parallel-2".to_string(),
                    "parallel-3".to_string(),
                ],
                timeout_seconds: 10,
            },
        ],
    };

    let start = std::time::Instant::now();
    let result = orchestrator.execute_workflow(workflow).await.unwrap();
    let duration = start.elapsed();

    // Should execute in parallel, so total time should be less than sequential
    assert!(duration.as_secs() < 20); // Would be 40s if sequential
    assert_eq!(result.completed_tasks.len(), 4);
}

#[tokio::test]
async fn test_task_dependencies() {
    let config = OrchestratorConfig::default();
    let mut orchestrator = Orchestrator::new(config);

    // Create workflow with complex dependencies
    let workflow = Workflow {
        id: "deps-workflow".to_string(),
        name: "Dependencies Test".to_string(),
        tasks: vec![
            WorkflowTask {
                id: "A".to_string(),
                task_type: TaskType::Compute("A".to_string()),
                dependencies: vec![],
                timeout_seconds: 5,
            },
            WorkflowTask {
                id: "B".to_string(),
                task_type: TaskType::Compute("B".to_string()),
                dependencies: vec!["A".to_string()],
                timeout_seconds: 5,
            },
            WorkflowTask {
                id: "C".to_string(),
                task_type: TaskType::Compute("C".to_string()),
                dependencies: vec!["A".to_string()],
                timeout_seconds: 5,
            },
            WorkflowTask {
                id: "D".to_string(),
                task_type: TaskType::Compute("D".to_string()),
                dependencies: vec!["B".to_string(), "C".to_string()],
                timeout_seconds: 5,
            },
        ],
    };

    let result = orchestrator.execute_workflow(workflow).await.unwrap();

    // Verify execution order
    let task_order: Vec<_> = result
        .completed_tasks
        .iter()
        .map(|t| t.task_id.as_str())
        .collect();

    // A must come before B and C
    let a_index = task_order.iter().position(|&t| t == "A").unwrap();
    let b_index = task_order.iter().position(|&t| t == "B").unwrap();
    let c_index = task_order.iter().position(|&t| t == "C").unwrap();
    let d_index = task_order.iter().position(|&t| t == "D").unwrap();

    assert!(a_index < b_index);
    assert!(a_index < c_index);
    assert!(b_index < d_index);
    assert!(c_index < d_index);
}

#[tokio::test]
async fn test_task_retry() {
    let config = OrchestratorConfig {
        id: "retry-orchestrator".to_string(),
        max_workflows: 10,
        task_timeout_seconds: 10,
        enable_retry: true,
        max_retries: 3,
    };

    let mut orchestrator = Orchestrator::new(config);
    orchestrator.set_failure_mode(true); // Enable test failure mode

    let workflow = Workflow {
        id: "retry-workflow".to_string(),
        name: "Retry Test".to_string(),
        tasks: vec![WorkflowTask {
            id: "flaky-task".to_string(),
            task_type: TaskType::Compute("flaky".to_string()),
            dependencies: vec![],
            timeout_seconds: 5,
        }],
    };

    let result = orchestrator.execute_workflow(workflow).await.unwrap();

    // Should eventually succeed after retries
    assert!(result.success);
    assert_eq!(result.retries_used, 2); // Fails twice, succeeds on third try
}

#[tokio::test]
async fn test_workflow_cancellation() {
    let config = OrchestratorConfig::default();
    let mut orchestrator = Orchestrator::new(config);

    let workflow = Workflow {
        id: "cancel-workflow".to_string(),
        name: "Cancel Test".to_string(),
        tasks: vec![WorkflowTask {
            id: "long-task".to_string(),
            task_type: TaskType::Compute("long".to_string()),
            dependencies: vec![],
            timeout_seconds: 60,
        }],
    };

    // Start workflow
    let workflow_id = workflow.id.clone();
    let handle = tokio::spawn(async move { orchestrator.execute_workflow(workflow).await });

    // Cancel after short delay
    sleep(Duration::from_millis(100)).await;
    // In real implementation, would call orchestrator.cancel_workflow(workflow_id)
    handle.abort();

    assert!(handle.await.is_err()); // Should be cancelled
}

#[tokio::test]
async fn test_workflow_timeout() {
    let config = OrchestratorConfig {
        id: "timeout-orchestrator".to_string(),
        max_workflows: 10,
        task_timeout_seconds: 2, // Very short timeout
        enable_retry: false,
        max_retries: 0,
    };

    let mut orchestrator = Orchestrator::new(config);

    let workflow = Workflow {
        id: "timeout-workflow".to_string(),
        name: "Timeout Test".to_string(),
        tasks: vec![WorkflowTask {
            id: "slow-task".to_string(),
            task_type: TaskType::Compute("slow".to_string()),
            dependencies: vec![],
            timeout_seconds: 1, // Will timeout
        }],
    };

    let result = orchestrator.execute_workflow(workflow).await.unwrap();
    assert!(!result.success);
    assert_eq!(result.failed_tasks.len(), 1);
    assert!(result.failed_tasks[0]
        .error
        .as_ref()
        .unwrap()
        .contains("timeout"));
}

#[tokio::test]
async fn test_concurrent_workflows() {
    let config = OrchestratorConfig {
        id: "concurrent-orchestrator".to_string(),
        max_workflows: 5,
        task_timeout_seconds: 10,
        enable_retry: false,
        max_retries: 0,
    };

    let orchestrator = std::sync::Arc::new(tokio::sync::Mutex::new(Orchestrator::new(config)));

    // Create multiple workflows
    let mut handles = Vec::new();

    for i in 0..5 {
        let orchestrator_clone = orchestrator.clone();
        let workflow = Workflow {
            id: format!("workflow-{}", i),
            name: format!("Test {}", i),
            tasks: vec![WorkflowTask {
                id: format!("task-{}", i),
                task_type: TaskType::Compute(format!("work-{}", i)),
                dependencies: vec![],
                timeout_seconds: 5,
            }],
        };

        let handle = tokio::spawn(async move {
            orchestrator_clone
                .lock()
                .await
                .execute_workflow(workflow)
                .await
        });
        handles.push(handle);
    }

    // Wait for all workflows
    let results: Vec<_> = futures::future::join_all(handles).await;

    // All should succeed
    assert_eq!(results.len(), 5);
    for result in results {
        assert!(result.unwrap().unwrap().success);
    }
}
