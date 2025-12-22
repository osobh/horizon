//! Tests for CpuAgent trait and implementations

use crate::{agent::*, Result};
use async_trait::async_trait;
use std::sync::Arc;
use tokio::sync::Mutex;

#[tokio::test]
async fn test_cpu_agent_creation() {
    let config = CpuAgentConfig {
        id: "test-agent-001".to_string(),
        capabilities: vec![AgentCapability::FileIo, AgentCapability::NetworkIo],
        max_concurrent_tasks: 10,
        memory_limit_mb: 100,
    };

    let agent = MockCpuAgent::new(config.clone());
    assert_eq!(agent.id(), "test-agent-001");
    assert_eq!(agent.capabilities().len(), 2);
    assert!(agent.can_handle(&AgentCapability::FileIo));
    // CPU agents don't have GPU compute capability by design
}

#[tokio::test]
async fn test_agent_lifecycle() {
    let config = CpuAgentConfig::default();
    let mut agent = MockCpuAgent::new(config);

    // Test initialization
    agent.initialize().await.expect("Failed to initialize");
    assert_eq!(agent.status(), AgentStatus::Ready);

    // Test task execution
    let task = AgentTask {
        id: "task-001".to_string(),
        task_type: TaskType::FileRead("test.txt".to_string()),
        priority: 5,
    };

    let result = agent
        .execute_task(task)
        .await
        .expect("Failed to execute task");
    assert!(matches!(result, TaskExecutionResult::Success(_)));

    // Test shutdown
    agent.shutdown().await.expect("Failed to shutdown");
    assert_eq!(agent.status(), AgentStatus::Stopped);
}

#[tokio::test]
async fn test_agent_resource_limits() {
    let config = CpuAgentConfig {
        id: "resource-test".to_string(),
        capabilities: vec![AgentCapability::FileIo],
        max_concurrent_tasks: 2,
        memory_limit_mb: 50,
    };

    let mut agent = MockCpuAgent::new(config);
    agent.initialize().await.unwrap();

    // Submit multiple tasks
    let task1 = create_test_task("task-1");
    let task2 = create_test_task("task-2");
    let task3 = create_test_task("task-3");

    // First two should succeed
    assert!(agent.submit_task(task1).await.is_ok());
    assert!(agent.submit_task(task2).await.is_ok());

    // Third should fail due to limit
    assert!(agent.submit_task(task3).await.is_err());
}

#[tokio::test]
async fn test_agent_error_handling() {
    let config = CpuAgentConfig::default();
    let mut agent = MockCpuAgent::new(config);
    agent.initialize().await.unwrap();

    // Create task that will fail
    let task = AgentTask {
        id: "fail-task".to_string(),
        task_type: TaskType::FileRead("/nonexistent/file.txt".to_string()),
        priority: 1,
    };

    let result = agent.execute_task(task).await;
    assert!(result.is_ok()); // Task execution returns Ok with error result

    match result.unwrap() {
        TaskExecutionResult::Failed(err) => {
            assert!(err.contains("not found"));
        }
        _ => panic!("Expected task to fail"),
    }
}

#[tokio::test]
async fn test_agent_metrics() {
    let config = CpuAgentConfig::default();
    let mut agent = MockCpuAgent::new(config);
    agent.initialize().await.unwrap();

    // Execute some tasks
    for i in 0..5 {
        let task = create_test_task(&format!("task-{}", i));
        agent.execute_task(task).await.unwrap();
    }

    let metrics = agent.get_metrics();
    assert_eq!(metrics.tasks_completed, 5);
    assert_eq!(metrics.tasks_failed, 0);
    assert!(metrics.average_task_duration_ms > 0.0);
}

// Mock implementation for testing
struct MockCpuAgent {
    config: CpuAgentConfig,
    status: Arc<Mutex<AgentStatus>>,
    active_tasks: Arc<Mutex<Vec<String>>>,
    metrics: Arc<Mutex<AgentMetrics>>,
}

impl MockCpuAgent {
    fn new(config: CpuAgentConfig) -> Self {
        Self {
            config,
            status: Arc::new(Mutex::new(AgentStatus::Created)),
            active_tasks: Arc::new(Mutex::new(Vec::new())),
            metrics: Arc::new(Mutex::new(AgentMetrics::default())),
        }
    }
}

#[async_trait]
impl CpuAgent for MockCpuAgent {
    fn id(&self) -> &str {
        &self.config.id
    }

    fn capabilities(&self) -> &[AgentCapability] {
        &self.config.capabilities
    }

    fn can_handle(&self, capability: &AgentCapability) -> bool {
        self.config.capabilities.contains(capability)
    }

    fn status(&self) -> AgentStatus {
        futures::executor::block_on(async { *self.status.lock().await })
    }

    async fn initialize(&mut self) -> Result<()> {
        *self.status.lock().await = AgentStatus::Ready;
        Ok(())
    }

    async fn execute_task(&mut self, task: AgentTask) -> Result<TaskExecutionResult> {
        let start = std::time::Instant::now();

        // Simulate task execution with a small delay to ensure measurable time
        tokio::time::sleep(tokio::time::Duration::from_millis(5)).await;

        // Simulate task execution
        let result = match &task.task_type {
            TaskType::FileRead(path) => {
                if path.contains("nonexistent") {
                    TaskExecutionResult::Failed("File not found".to_string())
                } else {
                    TaskExecutionResult::Success("Mock file content".to_string())
                }
            }
            _ => TaskExecutionResult::Success("Mock result".to_string()),
        };

        // Update metrics
        let mut metrics = self.metrics.lock().await;
        match &result {
            TaskExecutionResult::Success(_) => metrics.tasks_completed += 1,
            TaskExecutionResult::Failed(_) => metrics.tasks_failed += 1,
        }
        metrics.total_task_duration_ms += start.elapsed().as_millis() as f64;
        metrics.average_task_duration_ms = metrics.total_task_duration_ms
            / (metrics.tasks_completed + metrics.tasks_failed) as f64;

        Ok(result)
    }

    async fn submit_task(&self, task: AgentTask) -> Result<()> {
        let mut active = self.active_tasks.lock().await;
        if active.len() >= self.config.max_concurrent_tasks {
            return Err(crate::CpuAgentError::TaskError(
                "Maximum concurrent tasks reached".to_string(),
            ));
        }
        active.push(task.id);
        Ok(())
    }

    async fn shutdown(&mut self) -> Result<()> {
        *self.status.lock().await = AgentStatus::Stopped;
        Ok(())
    }

    fn get_metrics(&self) -> AgentMetrics {
        futures::executor::block_on(async { self.metrics.lock().await.clone() })
    }
}

fn create_test_task(id: &str) -> AgentTask {
    AgentTask {
        id: id.to_string(),
        task_type: TaskType::FileRead("test.txt".to_string()),
        priority: 5,
    }
}
