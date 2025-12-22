# Shared Storage API Reference

## Table of Contents

1. [Overview](#overview)
2. [Core Types](#core-types)
3. [Storage Manager API](#storage-manager-api)
4. [Job Management API](#job-management-api)
5. [Zero-Copy Access API](#zero-copy-access-api)
6. [Data Flow Patterns](#data-flow-patterns)
7. [Error Handling](#error-handling)
8. [Usage Examples](#usage-examples)

## Overview

The Shared Storage system provides efficient communication between CPU and GPU agents through a job-based interface with zero-copy data access capabilities.

### Key Features

- **Job-based communication**: Asynchronous task submission and retrieval
- **Priority scheduling**: Critical, High, Normal, Low priorities
- **Zero-copy access**: Memory-mapped files for large data
- **Automatic cleanup**: TTL-based job expiration
- **Fault tolerance**: Retry mechanisms and error recovery

## Core Types

### SharedStorageConfig

```rust
#[derive(Debug, Clone)]
pub struct SharedStorageConfig {
    pub base_path: PathBuf,              // Default: /nvme/gpu/shared/
    pub max_job_size: usize,             // Default: 100MB
    pub cleanup_interval: Duration,       // Default: 60s
    pub job_ttl: Duration,               // Default: 300s
    pub enable_compression: bool,         // Default: true
    pub compression_level: i32,          // Default: 3
}
```

### AgentJob

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentJob {
    pub id: Uuid,
    pub job_type: JobType,
    pub priority: Priority,
    pub source_agent: AgentId,
    pub target_agent: Option<AgentId>,
    pub payload: Vec<u8>,
    pub metadata: HashMap<String, String>,
    pub created_at: SystemTime,
    pub updated_at: SystemTime,
    pub ttl: Duration,
    pub status: JobStatus,
}
```

### JobType

```rust
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum JobType {
    Compute { operation: String },
    DataTransfer { size: usize },
    ModelUpdate { version: u32 },
    Query { query_type: String },
    Control { command: String },
    Custom(String),
}
```

### JobStatus

```rust
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum JobStatus {
    Pending,
    Processing { started_at: SystemTime },
    Completed { result: JobResult },
    Failed { error: String, retry_count: u32 },
    Cancelled,
}
```

### Priority

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Priority {
    Critical = 0,
    High = 1,
    Normal = 2,
    Low = 3,
}
```

## Storage Manager API

### SharedStorageManager

```rust
pub struct SharedStorageManager {
    pub async fn new(config: SharedStorageConfig) -> Result<Self>
    pub async fn submit_job(&self, job: AgentJob) -> Result<Uuid>
    pub async fn get_job(&self, id: Uuid) -> Result<Option<AgentJob>>
    pub async fn update_job_status(&self, id: Uuid, status: JobStatus) -> Result<()>
    pub async fn get_pending_jobs(&self, limit: usize) -> Result<Vec<AgentJob>>
    pub async fn get_jobs_for_agent(&self, agent_id: AgentId) -> Result<Vec<AgentJob>>
    pub async fn cleanup_expired_jobs(&self) -> Result<usize>
    pub fn get_stats(&self) -> StorageStats
}
```

### StorageStats

```rust
#[derive(Debug, Clone)]
pub struct StorageStats {
    pub total_jobs: usize,
    pub pending_jobs: usize,
    pub processing_jobs: usize,
    pub completed_jobs: usize,
    pub failed_jobs: usize,
    pub storage_size_bytes: u64,
    pub avg_job_size_bytes: usize,
}
```

## Job Management API

### JobBuilder

```rust
pub struct JobBuilder {
    pub fn new(job_type: JobType) -> Self
    pub fn with_priority(mut self, priority: Priority) -> Self
    pub fn with_payload(mut self, payload: Vec<u8>) -> Self
    pub fn with_target(mut self, target: AgentId) -> Self
    pub fn with_metadata(mut self, key: String, value: String) -> Self
    pub fn with_ttl(mut self, ttl: Duration) -> Self
    pub fn build(self) -> Result<AgentJob>
}
```

### JobQueue

```rust
pub struct JobQueue {
    pub fn new() -> Self
    pub fn push(&self, job: AgentJob) -> Result<()>
    pub fn pop(&self) -> Option<AgentJob>
    pub fn peek(&self) -> Option<AgentJob>
    pub fn len(&self) -> usize
    pub fn get_by_priority(&self, priority: Priority) -> Vec<AgentJob>
}
```

### JobScheduler

```rust
pub struct JobScheduler {
    pub fn new(manager: Arc<SharedStorageManager>) -> Self
    pub async fn schedule_job(&self, job: AgentJob, delay: Duration) -> Result<Uuid>
    pub async fn schedule_recurring(&self, job: AgentJob, interval: Duration) -> Result<ScheduleId>
    pub async fn cancel_schedule(&self, id: ScheduleId) -> Result<()>
}
```

## Zero-Copy Access API

### MmapHandle

```rust
pub struct MmapHandle {
    pub fn new(path: PathBuf) -> Result<Self>
    pub fn as_slice(&self) -> &[u8]
    pub fn len(&self) -> usize
    pub fn offset(&self) -> usize
}
```

### ZeroCopyAccess

```rust
pub struct ZeroCopyAccess {
    pub fn create(path: PathBuf, size: usize) -> Result<Self>
    pub fn open(path: PathBuf) -> Result<Self>
    pub fn as_slice(&self) -> &[u8]
    pub fn as_mut_slice(&mut self) -> &mut [u8]
    pub fn resize(&mut self, new_size: usize) -> Result<()>
    pub fn sync(&self) -> Result<()>
}
```

### LargeDataHandler

```rust
pub struct LargeDataHandler {
    pub fn new(storage_manager: Arc<SharedStorageManager>) -> Self
    pub async fn store_large_data(&self, data: &[u8]) -> Result<DataHandle>
    pub async fn retrieve_large_data(&self, handle: DataHandle) -> Result<MmapHandle>
    pub async fn stream_data<F>(&self, handle: DataHandle, chunk_size: usize, processor: F) -> Result<()>
        where F: Fn(&[u8]) -> Result<()>
}
```

## Data Flow Patterns

### ComputePattern

```rust
pub struct ComputePattern {
    pub fn new() -> Self
    pub async fn submit_compute_job(&self, input: ComputeInput) -> Result<JobId>
    pub async fn get_compute_result(&self, job_id: JobId) -> Result<Option<ComputeResult>>
}

pub struct ComputeInput {
    pub algorithm: String,
    pub parameters: HashMap<String, Value>,
    pub input_data: Vec<f32>,
}
```

### StreamingPattern

```rust
pub struct StreamingPattern {
    pub fn new(config: StreamConfig) -> Self
    pub async fn start_stream(&self, source: DataSource) -> Result<StreamHandle>
    pub async fn process_chunk(&self, handle: StreamHandle, chunk: DataChunk) -> Result<()>
    pub async fn end_stream(&self, handle: StreamHandle) -> Result<StreamResult>
}
```

### BatchPattern

```rust
pub struct BatchPattern {
    pub fn new(batch_size: usize) -> Self
    pub async fn add_to_batch(&self, item: BatchItem) -> Result<()>
    pub async fn process_batch(&self) -> Result<BatchResult>
    pub fn set_batch_processor<F>(&mut self, processor: F)
        where F: Fn(Vec<BatchItem>) -> Result<BatchResult>
}
```

### PipelinePattern

```rust
pub struct PipelinePattern {
    pub fn new() -> Self
    pub fn add_stage<F>(&mut self, name: &str, processor: F) -> &mut Self
        where F: Fn(PipelineData) -> Result<PipelineData>
    pub async fn execute(&self, input: PipelineData) -> Result<PipelineData>
}
```

### ScatterGatherPattern

```rust
pub struct ScatterGatherPattern {
    pub fn new(num_workers: usize) -> Self
    pub async fn scatter<T>(&self, data: Vec<T>, processor: ProcessorFn<T>) -> Result<Vec<JobId>>
    pub async fn gather<R>(&self, job_ids: Vec<JobId>) -> Result<Vec<R>>
}
```

## Error Handling

### SharedStorageError

```rust
#[derive(Debug, thiserror::Error)]
pub enum SharedStorageError {
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),
    
    #[error("Serialization error: {0}")]
    SerializationError(String),
    
    #[error("Job not found: {0}")]
    JobNotFound(Uuid),
    
    #[error("Storage limit exceeded: {0}")]
    StorageLimitExceeded(String),
    
    #[error("Invalid job state: {0}")]
    InvalidJobState(String),
    
    #[error("Zero-copy access error: {0}")]
    ZeroCopyError(String),
}
```

## Usage Examples

### Basic Job Submission

```rust
use shared_storage::{SharedStorageManager, JobBuilder, JobType, Priority};

let manager = SharedStorageManager::new(SharedStorageConfig::default()).await?;

// Create and submit job
let job = JobBuilder::new(JobType::Compute { operation: "matrix_multiply".to_string() })
    .with_priority(Priority::High)
    .with_payload(input_data)
    .with_metadata("input_size".to_string(), "1024x1024".to_string())
    .build()?;

let job_id = manager.submit_job(job).await?;

// Poll for result
loop {
    if let Some(job) = manager.get_job(job_id).await? {
        match job.status {
            JobStatus::Completed { result } => {
                println!("Job completed: {:?}", result);
                break;
            }
            JobStatus::Failed { error, .. } => {
                return Err(anyhow!("Job failed: {}", error));
            }
            _ => {
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        }
    }
}
```

### Zero-Copy Large Data Transfer

```rust
use shared_storage::{LargeDataHandler, ZeroCopyAccess};

let handler = LargeDataHandler::new(manager.clone());

// Store large data
let large_data = vec![0u8; 100_000_000]; // 100MB
let handle = handler.store_large_data(&large_data).await?;

// CPU agent reads with zero-copy
let mmap = handler.retrieve_large_data(handle.clone()).await?;
let data_slice = mmap.as_slice();

// Process in chunks without loading all into memory
handler.stream_data(handle, 1024 * 1024, |chunk| {
    // Process 1MB chunks
    process_chunk(chunk)?;
    Ok(())
}).await?;
```

### Pipeline Processing

```rust
use shared_storage::{PipelinePattern, PipelineData};

let mut pipeline = PipelinePattern::new();

pipeline
    .add_stage("decode", |data| {
        let decoded = decode_data(data)?;
        Ok(decoded)
    })
    .add_stage("transform", |data| {
        let transformed = apply_transforms(data)?;
        Ok(transformed)
    })
    .add_stage("encode", |data| {
        let encoded = encode_data(data)?;
        Ok(encoded)
    });

let result = pipeline.execute(input_data).await?;
```

### Scatter-Gather Pattern

```rust
use shared_storage::ScatterGatherPattern;

let scatter_gather = ScatterGatherPattern::new(8); // 8 workers

// Scatter work across multiple GPU agents
let chunks = split_data_into_chunks(large_dataset, 8);
let job_ids = scatter_gather.scatter(chunks, |chunk| {
    // Each chunk processed by different GPU agent
    gpu_process(chunk)
}).await?;

// Gather results
let results: Vec<ProcessedData> = scatter_gather.gather(job_ids).await?;
let final_result = merge_results(results);
```

### Workflow with Shared Storage

```rust
use shared_storage::{SharedStorageManager, JobBuilder};
use cpu_agents::Orchestrator;

// CPU agent submits GPU job
let gpu_job = JobBuilder::new(JobType::Compute { operation: "train_model".to_string() })
    .with_priority(Priority::High)
    .with_payload(training_data)
    .build()?;

let job_id = storage_manager.submit_job(gpu_job).await?;

// GPU agent processes job (in separate process)
// ...

// CPU agent retrieves result
let completed_job = wait_for_completion(&storage_manager, job_id).await?;
if let JobStatus::Completed { result } = completed_job.status {
    let model = deserialize_model(result.data)?;
    save_model(model).await?;
}
```

### Resource Cleanup

```rust
use shared_storage::SharedStorageManager;

// Automatic cleanup task
let manager = storage_manager.clone();
tokio::spawn(async move {
    loop {
        match manager.cleanup_expired_jobs().await {
            Ok(count) => {
                if count > 0 {
                    println!("Cleaned up {} expired jobs", count);
                }
            }
            Err(e) => eprintln!("Cleanup error: {}", e),
        }
        tokio::time::sleep(Duration::from_secs(60)).await;
    }
});
```