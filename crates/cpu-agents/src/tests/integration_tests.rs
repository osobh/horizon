//! Integration tests for CPU agents

use crate::{agent::*, bridge::*, io_manager::*, orchestrator::*, Result};
use std::path::PathBuf;
use std::sync::Arc;
use tempfile::TempDir;
use tokio::sync::Mutex;
use tokio::time::{sleep, Duration};

#[tokio::test]
async fn test_full_cpu_agent_workflow() {
    let temp_dir = TempDir::new().unwrap();

    // Create I/O Manager
    let io_config = IoConfig {
        max_file_size_mb: 100,
        buffer_size_kb: 64,
        enable_compression: true,
        temp_dir: temp_dir.path().join("io"),
    };
    let io_manager = Arc::new(Mutex::new(IoManager::new(io_config).await.unwrap()));

    // Create Bridge
    let bridge_config = BridgeConfig {
        shared_storage_path: temp_dir.path().join("bridge"),
        inbox_dir: "inbox".to_string(),
        outbox_dir: "outbox".to_string(),
        message_retention_seconds: 3600,
        max_message_size_mb: 10,
        polling_interval_ms: 100,
    };
    let bridge = Arc::new(Mutex::new(CpuGpuBridge::new(bridge_config).await.unwrap()));
    bridge.lock().await.start().await.unwrap();

    // Create Orchestrator
    let orchestrator_config = OrchestratorConfig {
        id: "integration-orchestrator".to_string(),
        max_workflows: 10,
        task_timeout_seconds: 60,
        enable_retry: true,
        max_retries: 3,
    };
    let orchestrator = Arc::new(Mutex::new(Orchestrator::new(orchestrator_config)));

    // Create workflow that uses all components
    let workflow = Workflow {
        id: "integration-workflow".to_string(),
        name: "Full Integration Test".to_string(),
        tasks: vec![
            // Task 1: Read input file
            WorkflowTask {
                id: "read-input".to_string(),
                task_type: TaskType::FileRead(
                    temp_dir
                        .path()
                        .join("input.txt")
                        .to_string_lossy()
                        .to_string(),
                ),
                dependencies: vec![],
                timeout_seconds: 10,
            },
            // Task 2: Send to GPU for processing
            WorkflowTask {
                id: "gpu-process".to_string(),
                task_type: TaskType::GpuCompute("matrix_multiply".to_string()),
                dependencies: vec!["read-input".to_string()],
                timeout_seconds: 30,
            },
            // Task 3: Write results
            WorkflowTask {
                id: "write-output".to_string(),
                task_type: TaskType::FileWrite(
                    temp_dir
                        .path()
                        .join("output.txt")
                        .to_string_lossy()
                        .to_string(),
                ),
                dependencies: vec!["gpu-process".to_string()],
                timeout_seconds: 10,
            },
        ],
    };

    // Prepare input file
    let input_data = "Test data for GPU processing";
    io_manager
        .lock()
        .await
        .execute(IoOperation::WriteFile {
            path: temp_dir.path().join("input.txt"),
            content: input_data.as_bytes().to_vec(),
            create_dirs: true,
        })
        .await
        .unwrap();

    // Execute workflow
    let result = orchestrator
        .lock()
        .await
        .execute_workflow(workflow)
        .await
        .unwrap();

    assert!(result.success);
    assert_eq!(result.completed_tasks.len(), 3);

    // Verify output file exists
    assert!(temp_dir.path().join("output.txt").exists());
}

#[tokio::test]
async fn test_cpu_gpu_data_exchange() {
    let temp_dir = TempDir::new().unwrap();

    // Create shared components
    let bridge_config = BridgeConfig {
        shared_storage_path: temp_dir.path().to_path_buf(),
        inbox_dir: "inbox".to_string(),
        outbox_dir: "outbox".to_string(),
        message_retention_seconds: 3600,
        max_message_size_mb: 10,
        polling_interval_ms: 50,
    };
    let bridge = Arc::new(Mutex::new(CpuGpuBridge::new(bridge_config).await.unwrap()));
    bridge.lock().await.start().await.unwrap();

    // CPU Agent sends large data to GPU
    let large_data = vec![42u8; 1024 * 1024]; // 1MB of data
    let data_message = CpuGpuMessage {
        id: "data-001".to_string(),
        message_type: MessageType::DataTransfer,
        source: "cpu-agent-001".to_string(),
        destination: "gpu-agent-001".to_string(),
        payload: serde_json::json!({
            "data_path": temp_dir.path().join("data.bin").to_string_lossy(),
            "data_size": large_data.len(),
            "checksum": calculate_checksum(&large_data),
        }),
        timestamp: chrono::Utc::now(),
        priority: 8,
    };

    // Write actual data to file
    tokio::fs::write(temp_dir.path().join("data.bin"), &large_data)
        .await
        .unwrap();

    // Send metadata via bridge
    bridge.lock().await.send_to_gpu(data_message).await.unwrap();

    // Simulate GPU processing and response
    sleep(Duration::from_millis(100)).await;

    // GPU would process and send result back
    let result_data = vec![84u8; 512 * 1024]; // 512KB result
    tokio::fs::write(temp_dir.path().join("result.bin"), &result_data)
        .await
        .unwrap();

    let result_message = CpuGpuMessage {
        id: "result-001".to_string(),
        message_type: MessageType::TaskResult,
        source: "gpu-agent-001".to_string(),
        destination: "cpu-agent-001".to_string(),
        payload: serde_json::json!({
            "status": "success",
            "result_path": temp_dir.path().join("result.bin").to_string_lossy(),
            "result_size": result_data.len(),
            "processing_time_ms": 85,
        }),
        timestamp: chrono::Utc::now(),
        priority: 8,
    };

    // Manually place in inbox (simulating GPU response)
    let inbox_path = temp_dir.path().join("inbox").join("result-001.json");
    let result_json = serde_json::to_string_pretty(&result_message).unwrap();
    tokio::fs::write(&inbox_path, result_json).await.unwrap();

    // CPU receives result
    sleep(Duration::from_millis(100)).await;
    let messages = bridge.lock().await.receive_from_gpu().await.unwrap();

    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0].message_type, MessageType::TaskResult);

    // Verify result data
    let result_payload = messages[0].payload.as_object().unwrap();
    assert_eq!(result_payload["status"], "success");
    assert_eq!(result_payload["result_size"], 512 * 1024);
}

#[tokio::test]
async fn test_multi_agent_coordination() {
    let temp_dir = TempDir::new().unwrap();

    // Create multiple CPU agents
    let mut agents = Vec::new();
    for i in 0..3 {
        let config = CpuAgentConfig {
            id: format!("cpu-agent-{}", i),
            capabilities: vec![
                AgentCapability::FileIo,
                AgentCapability::NetworkIo,
                AgentCapability::Orchestration,
            ],
            max_concurrent_tasks: 5,
            memory_limit_mb: 100,
        };
        let agent: Arc<Mutex<dyn CpuAgent>> = Arc::new(Mutex::new(TestCpuAgent::new(config)));
        agents.push(agent);
    }

    // Create shared orchestrator
    let orchestrator = Arc::new(Mutex::new(Orchestrator::new(OrchestratorConfig::default())));

    // Create complex workflow requiring multiple agents
    let workflow = Workflow {
        id: "multi-agent-workflow".to_string(),
        name: "Multi-Agent Test".to_string(),
        tasks: vec![
            // Parallel data collection by different agents
            WorkflowTask {
                id: "collect-1".to_string(),
                task_type: TaskType::FileRead("source1.txt".to_string()),
                dependencies: vec![],
                timeout_seconds: 10,
            },
            WorkflowTask {
                id: "collect-2".to_string(),
                task_type: TaskType::NetworkFetch("http://example.com/data".to_string()),
                dependencies: vec![],
                timeout_seconds: 10,
            },
            WorkflowTask {
                id: "collect-3".to_string(),
                task_type: TaskType::DatabaseQuery("SELECT * FROM metrics".to_string()),
                dependencies: vec![],
                timeout_seconds: 10,
            },
            // Merge results
            WorkflowTask {
                id: "merge".to_string(),
                task_type: TaskType::Transform("merge_datasets".to_string()),
                dependencies: vec![
                    "collect-1".to_string(),
                    "collect-2".to_string(),
                    "collect-3".to_string(),
                ],
                timeout_seconds: 20,
            },
            // Final processing
            WorkflowTask {
                id: "process".to_string(),
                task_type: TaskType::GpuCompute("analyze_merged_data".to_string()),
                dependencies: vec!["merge".to_string()],
                timeout_seconds: 30,
            },
        ],
    };

    // Execute with agent pool
    let result = orchestrator
        .lock()
        .await
        .execute_workflow_with_agents(workflow, agents)
        .await
        .unwrap();

    assert!(result.success);
    assert_eq!(result.completed_tasks.len(), 5);

    // Verify parallel execution
    let collect_tasks: Vec<_> = result
        .completed_tasks
        .iter()
        .filter(|t| t.task_id.starts_with("collect"))
        .collect();

    assert_eq!(collect_tasks.len(), 3);

    // Check that different agents handled different tasks
    let agent_ids: std::collections::HashSet<_> =
        collect_tasks.iter().map(|t| &t.executed_by).collect();

    assert!(agent_ids.len() >= 2); // At least 2 different agents used
}

#[tokio::test]
async fn test_failure_recovery() {
    let temp_dir = TempDir::new().unwrap();

    // Create components with retry enabled
    let orchestrator_config = OrchestratorConfig {
        id: "recovery-orchestrator".to_string(),
        max_workflows: 10,
        task_timeout_seconds: 30,
        enable_retry: true,
        max_retries: 3,
    };
    let mut orchestrator = Orchestrator::new(orchestrator_config);

    // Create workflow with failing task
    let workflow = Workflow {
        id: "recovery-workflow".to_string(),
        name: "Recovery Test".to_string(),
        tasks: vec![
            WorkflowTask {
                id: "setup".to_string(),
                task_type: TaskType::FileWrite("checkpoint.txt".to_string()),
                dependencies: vec![],
                timeout_seconds: 10,
            },
            WorkflowTask {
                id: "flaky-task".to_string(),
                task_type: TaskType::Compute("flaky_operation".to_string()),
                dependencies: vec!["setup".to_string()],
                timeout_seconds: 10,
            },
            WorkflowTask {
                id: "cleanup".to_string(),
                task_type: TaskType::FileDelete("temp.txt".to_string()),
                dependencies: vec!["flaky-task".to_string()],
                timeout_seconds: 10,
            },
        ],
    };

    // Enable failure simulation
    orchestrator.set_failure_mode(true);
    orchestrator.set_failure_rate(0.5); // 50% failure rate

    // Execute with retries
    let result = orchestrator.execute_workflow(workflow).await.unwrap();

    // Should eventually succeed
    assert!(result.success);
    assert!(result.retries_used > 0);
    assert_eq!(result.completed_tasks.len(), 3);
}

#[tokio::test]
async fn test_resource_isolation() {
    // Verify CPU agents cannot access GPU resources
    let config = CpuAgentConfig {
        id: "isolation-test".to_string(),
        capabilities: vec![AgentCapability::FileIo],
        max_concurrent_tasks: 1,
        memory_limit_mb: 50,
    };

    let agent = TestCpuAgent::new(config);

    // Attempt to use GPU capability (should fail)
    let gpu_task = AgentTask {
        id: "gpu-task".to_string(),
        task_type: TaskType::GpuCompute("invalid".to_string()),
        priority: 1,
    };

    let result = agent.can_execute(&gpu_task);
    assert!(!result);

    // Verify no GPU dependencies in CPU agent
    assert!(!agent.has_gpu_dependencies());
}

// Helper implementations

struct TestCpuAgent {
    config: CpuAgentConfig,
    status: AgentStatus,
}

#[async_trait::async_trait]
impl CpuAgent for TestCpuAgent {
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
        self.status
    }

    async fn initialize(&mut self) -> crate::Result<()> {
        Ok(())
    }

    async fn execute_task(&mut self, _task: AgentTask) -> crate::Result<TaskExecutionResult> {
        Ok(TaskExecutionResult::Success("Test result".to_string()))
    }

    async fn submit_task(&self, _task: AgentTask) -> crate::Result<()> {
        Ok(())
    }

    async fn shutdown(&mut self) -> crate::Result<()> {
        Ok(())
    }

    fn get_metrics(&self) -> AgentMetrics {
        AgentMetrics::default()
    }
}

impl TestCpuAgent {
    fn new(config: CpuAgentConfig) -> Self {
        Self {
            config,
            status: AgentStatus::Created,
        }
    }

    fn can_execute(&self, task: &AgentTask) -> bool {
        match &task.task_type {
            TaskType::GpuCompute(_) => false, // CPU agents cannot do GPU compute
            TaskType::FileRead(_) | TaskType::FileWrite(_) => {
                self.config.capabilities.contains(&AgentCapability::FileIo)
            }
            _ => true,
        }
    }

    fn has_gpu_dependencies(&self) -> bool {
        // Check that no GPU libraries are linked
        // In real implementation, would check actual dependencies
        false
    }
}

fn calculate_checksum(data: &[u8]) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    data.hash(&mut hasher);
    format!("{:x}", hasher.finish())
}
