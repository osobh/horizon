# CPU Agents API Reference

## Table of Contents

1. [Core Types](#core-types)
2. [Agent API](#agent-api)
3. [I/O Manager API](#io-manager-api)
4. [Orchestrator API](#orchestrator-api)
5. [Bridge API](#bridge-api)
6. [Task Management](#task-management)
7. [Error Handling](#error-handling)
8. [Usage Examples](#usage-examples)

## Core Types

### CpuAgent Trait

```rust
#[async_trait]
pub trait CpuAgent: Send + Sync {
    fn id(&self) -> AgentId;
    fn agent_type(&self) -> AgentType;
    async fn execute_task(&mut self, task: Task) -> Result<TaskResult>;
    async fn handle_message(&mut self, message: AgentMessage) -> Result<()>;
    fn get_capabilities(&self) -> Vec<Capability>;
    fn get_metrics(&self) -> AgentMetrics;
}
```

### AgentId

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AgentId(pub Uuid);

impl AgentId {
    pub fn new() -> Self
    pub fn from_string(s: &str) -> Result<Self>
}
```

### AgentType

```rust
#[derive(Debug, Clone, PartialEq)]
pub enum AgentType {
    IoManager,
    Orchestrator,
    DataLoader,
    ApiGateway,
    Custom(String),
}
```

### Task

```rust
#[derive(Debug, Clone)]
pub struct Task {
    pub id: Uuid,
    pub task_type: TaskType,
    pub priority: Priority,
    pub payload: Vec<u8>,
    pub dependencies: Vec<Uuid>,
    pub timeout: Option<Duration>,
    pub created_at: SystemTime,
}
```

### TaskType

```rust
#[derive(Debug, Clone, PartialEq)]
pub enum TaskType {
    FileRead { path: PathBuf },
    FileWrite { path: PathBuf, data: Vec<u8> },
    NetworkRequest { url: String, method: HttpMethod },
    DataTransform { operation: TransformOp },
    GpuSubmission { job: GpuJob },
    Custom(String),
}
```

## Agent API

### BaseAgent

```rust
pub struct BaseAgent {
    pub fn new(agent_type: AgentType) -> Self
    pub fn with_capabilities(mut self, capabilities: Vec<Capability>) -> Self
    pub async fn start(&mut self) -> Result<()>
    pub async fn stop(&mut self) -> Result<()>
    pub fn is_running(&self) -> bool
}
```

### AgentBuilder

```rust
pub struct AgentBuilder {
    pub fn new() -> Self
    pub fn with_type(mut self, agent_type: AgentType) -> Self
    pub fn with_capability(mut self, capability: Capability) -> Self
    pub fn with_task_handler<F>(mut self, handler: F) -> Self
        where F: Fn(Task) -> Result<TaskResult> + Send + Sync + 'static
    pub fn build(self) -> Result<Box<dyn CpuAgent>>
}
```

## I/O Manager API

### IoManager

```rust
pub struct IoManager {
    pub fn new(config: IoConfig) -> Result<Self>
    pub async fn read_file(&self, path: &Path) -> Result<Vec<u8>>
    pub async fn write_file(&self, path: &Path, data: &[u8]) -> Result<()>
    pub async fn read_dir(&self, path: &Path) -> Result<Vec<DirEntry>>
    pub async fn watch_file(&self, path: &Path, callback: FileWatchCallback) -> Result<WatchHandle>
    pub fn get_stats(&self) -> IoStats
}
```

### IoConfig

```rust
#[derive(Debug, Clone)]
pub struct IoConfig {
    pub buffer_size: usize,
    pub max_concurrent_ops: usize,
    pub enable_caching: bool,
    pub cache_size_mb: usize,
    pub compression: Option<CompressionType>,
}
```

### BatchIoManager

```rust
pub struct BatchIoManager {
    pub fn new(io_manager: Arc<IoManager>) -> Self
    pub async fn batch_read(&self, paths: Vec<PathBuf>) -> Result<Vec<Result<Vec<u8>>>>
    pub async fn batch_write(&self, operations: Vec<WriteOperation>) -> Result<Vec<Result<()>>>
    pub fn set_batch_size(&mut self, size: usize)
}
```

## Orchestrator API

### Orchestrator

```rust
pub struct Orchestrator {
    pub fn new(config: OrchestratorConfig) -> Result<Self>
    pub async fn submit_workflow(&self, workflow: Workflow) -> Result<WorkflowId>
    pub async fn get_workflow_status(&self, id: WorkflowId) -> Result<WorkflowStatus>
    pub async fn cancel_workflow(&self, id: WorkflowId) -> Result<()>
    pub async fn list_active_workflows(&self) -> Result<Vec<WorkflowSummary>>
}
```

### Workflow

```rust
#[derive(Debug, Clone)]
pub struct Workflow {
    pub id: WorkflowId,
    pub name: String,
    pub stages: Vec<WorkflowStage>,
    pub dependencies: HashMap<StageId, Vec<StageId>>,
    pub timeout: Option<Duration>,
    pub retry_policy: RetryPolicy,
}
```

### WorkflowStage

```rust
#[derive(Debug, Clone)]
pub struct WorkflowStage {
    pub id: StageId,
    pub name: String,
    pub tasks: Vec<Task>,
    pub parallel: bool,
    pub on_failure: FailureAction,
}
```

### WorkflowBuilder

```rust
pub struct WorkflowBuilder {
    pub fn new(name: &str) -> Self
    pub fn add_stage(mut self, stage: WorkflowStage) -> Self
    pub fn add_dependency(mut self, from: StageId, to: StageId) -> Self
    pub fn with_timeout(mut self, timeout: Duration) -> Self
    pub fn with_retry_policy(mut self, policy: RetryPolicy) -> Self
    pub fn build(self) -> Result<Workflow>
}
```

## Bridge API

### CpuGpuBridge

```rust
pub struct CpuGpuBridge {
    pub fn new() -> Self
    pub async fn submit_gpu_job(&self, job: GpuJob) -> Result<JobId>
    pub async fn get_job_result(&self, id: JobId) -> Result<Option<JobResult>>
    pub async fn cancel_job(&self, id: JobId) -> Result<()>
    pub fn get_pending_jobs(&self) -> Vec<JobId>
    pub fn get_bridge_stats(&self) -> BridgeStats
}
```

### GpuJob

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuJob {
    pub id: JobId,
    pub job_type: GpuJobType,
    pub input_data: Vec<u8>,
    pub priority: Priority,
    pub max_gpu_memory: Option<usize>,
    pub timeout: Option<Duration>,
}
```

### MessagePassing

```rust
pub struct MessageBus {
    pub fn new(config: MessageConfig) -> Result<Self>
    pub async fn send(&self, target: AgentId, message: AgentMessage) -> Result<()>
    pub async fn broadcast(&self, message: AgentMessage) -> Result<()>
    pub async fn subscribe(&self, topic: &str) -> Result<MessageReceiver>
    pub async fn publish(&self, topic: &str, message: AgentMessage) -> Result<()>
}
```

### AgentMessage

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentMessage {
    pub id: Uuid,
    pub from: AgentId,
    pub to: Option<AgentId>,
    pub message_type: MessageType,
    pub payload: Vec<u8>,
    pub timestamp: SystemTime,
}
```

## Task Management

### TaskQueue

```rust
pub struct TaskQueue {
    pub fn new(capacity: usize) -> Self
    pub async fn submit(&self, task: Task) -> Result<()>
    pub async fn get_next(&self) -> Option<Task>
    pub fn pending_count(&self) -> usize
    pub fn set_priority_comparator<F>(&mut self, comparator: F)
        where F: Fn(&Task, &Task) -> Ordering + Send + Sync + 'static
}
```

### TaskScheduler

```rust
pub struct TaskScheduler {
    pub fn new(config: SchedulerConfig) -> Result<Self>
    pub async fn schedule(&self, task: Task) -> Result<ScheduleHandle>
    pub async fn cancel(&self, handle: ScheduleHandle) -> Result<()>
    pub fn get_scheduled_tasks(&self) -> Vec<ScheduledTask>
    pub async fn run_scheduler(&self) -> Result<()>
}
```

### TaskResult

```rust
#[derive(Debug, Clone)]
pub enum TaskResult {
    Success { output: Vec<u8>, metrics: TaskMetrics },
    Failure { error: String, can_retry: bool },
    Cancelled,
    Timeout,
}
```

## Error Handling

### CpuAgentError

```rust
#[derive(Debug, thiserror::Error)]
pub enum CpuAgentError {
    #[error("I/O operation failed: {0}")]
    IoError(String),
    
    #[error("Task execution failed: {0}")]
    TaskError(String),
    
    #[error("Bridge communication error: {0}")]
    BridgeError(String),
    
    #[error("Workflow error: {0}")]
    WorkflowError(String),
    
    #[error("Resource limit exceeded: {0}")]
    ResourceError(String),
    
    #[error("Invalid configuration: {0}")]
    ConfigError(String),
}
```

## Usage Examples

### Creating a Custom CPU Agent

```rust
use cpu_agents::{AgentBuilder, Task, TaskResult, Capability};

let agent = AgentBuilder::new()
    .with_type(AgentType::Custom("DataProcessor".to_string()))
    .with_capability(Capability::FileIO)
    .with_capability(Capability::DataTransform)
    .with_task_handler(|task| {
        match task.task_type {
            TaskType::DataTransform { operation } => {
                // Process data
                Ok(TaskResult::Success {
                    output: processed_data,
                    metrics: TaskMetrics::default(),
                })
            }
            _ => Err(CpuAgentError::TaskError("Unsupported task".to_string()))
        }
    })
    .build()?;

agent.start().await?;
```

### File I/O Operations

```rust
use cpu_agents::{IoManager, IoConfig};

let config = IoConfig {
    buffer_size: 8192,
    max_concurrent_ops: 100,
    enable_caching: true,
    cache_size_mb: 128,
    compression: Some(CompressionType::Lz4),
};

let io_manager = IoManager::new(config)?;

// Read file
let data = io_manager.read_file(Path::new("/data/input.bin")).await?;

// Batch operations
let batch_manager = BatchIoManager::new(Arc::new(io_manager));
let results = batch_manager.batch_read(vec![
    PathBuf::from("/data/file1.bin"),
    PathBuf::from("/data/file2.bin"),
]).await?;
```

### Workflow Orchestration

```rust
use cpu_agents::{Orchestrator, WorkflowBuilder, WorkflowStage};

let orchestrator = Orchestrator::new(Default::default())?;

let workflow = WorkflowBuilder::new("DataPipeline")
    .add_stage(WorkflowStage {
        id: StageId::new(),
        name: "Load Data".to_string(),
        tasks: vec![load_task],
        parallel: false,
        on_failure: FailureAction::Retry { max_attempts: 3 },
    })
    .add_stage(WorkflowStage {
        id: StageId::new(),
        name: "Process Data".to_string(),
        tasks: vec![process_task1, process_task2],
        parallel: true,
        on_failure: FailureAction::Continue,
    })
    .with_timeout(Duration::from_secs(300))
    .build()?;

let workflow_id = orchestrator.submit_workflow(workflow).await?;

// Monitor progress
loop {
    let status = orchestrator.get_workflow_status(workflow_id).await?;
    match status {
        WorkflowStatus::Completed => break,
        WorkflowStatus::Failed(error) => return Err(error),
        _ => tokio::time::sleep(Duration::from_secs(1)).await,
    }
}
```

### CPU-GPU Communication

```rust
use cpu_agents::{CpuGpuBridge, GpuJob, GpuJobType};

let bridge = CpuGpuBridge::new();

// Submit GPU job
let gpu_job = GpuJob {
    id: JobId::new(),
    job_type: GpuJobType::Evolution { generations: 100 },
    input_data: agent_data,
    priority: Priority::High,
    max_gpu_memory: Some(1024 * 1024 * 1024), // 1GB
    timeout: Some(Duration::from_secs(60)),
};

let job_id = bridge.submit_gpu_job(gpu_job).await?;

// Wait for result
let result = loop {
    if let Some(result) = bridge.get_job_result(job_id).await? {
        break result;
    }
    tokio::time::sleep(Duration::from_millis(100)).await;
};

println!("GPU job completed: {:?}", result);
```

### Message Passing

```rust
use cpu_agents::{MessageBus, AgentMessage, MessageType};

let message_bus = MessageBus::new(Default::default())?;

// Subscribe to topic
let mut receiver = message_bus.subscribe("agent_updates").await?;

// Send message
let message = AgentMessage {
    id: Uuid::new_v4(),
    from: my_agent_id,
    to: Some(target_agent_id),
    message_type: MessageType::DataReady,
    payload: data.to_vec(),
    timestamp: SystemTime::now(),
};

message_bus.send(target_agent_id, message).await?;

// Receive messages
while let Some(msg) = receiver.recv().await {
    println!("Received message: {:?}", msg);
}
```

## Resource Isolation

**Important**: CPU agents must NEVER directly access GPU resources. All GPU operations must go through the CpuGpuBridge using the job submission mechanism.

```rust
// ❌ WRONG - Direct GPU access
let device = CudaDevice::new(0)?; // This will fail at compile time

// ✅ CORRECT - Through bridge
let job = GpuJob { /* ... */ };
let result = bridge.submit_gpu_job(job).await?;
```