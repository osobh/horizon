//! Job submission pipeline for efficient CPU→GPU→CPU workflows
//! 
//! Provides high-level abstractions for common job submission patterns
//! and handles batching, retry logic, and result aggregation.

use super::*;
use std::sync::Arc;
use std::time::{Duration, Instant};
use std::collections::VecDeque;
use tokio::sync::{Mutex, RwLock, mpsc};
use tokio::task::JoinHandle;
use anyhow::{Result, anyhow};

/// Job submission pipeline for high-throughput processing
pub struct JobSubmissionPipeline {
    storage_manager: Arc<SharedStorageManager>,
    batch_config: BatchConfig,
    retry_config: RetryConfig,
    pending_jobs: Arc<RwLock<HashMap<Uuid, PipelineJob>>>,
    job_queue: Arc<Mutex<VecDeque<QueuedJob>>>,
    batch_processor: Option<JoinHandle<()>>,
    result_aggregator: Option<JoinHandle<()>>,
    is_running: Arc<AtomicBool>,
    stats: Arc<PipelineStats>,
}

impl JobSubmissionPipeline {
    /// Create new job submission pipeline
    pub fn new(storage_manager: Arc<SharedStorageManager>) -> Self {
        Self {
            storage_manager,
            batch_config: BatchConfig::default(),
            retry_config: RetryConfig::default(),
            pending_jobs: Arc::new(RwLock::new(HashMap::new())),
            job_queue: Arc::new(Mutex::new(VecDeque::new())),
            batch_processor: None,
            result_aggregator: None,
            is_running: Arc::new(AtomicBool::new(false)),
            stats: Arc::new(PipelineStats::default()),
        }
    }

    /// Configure batching behavior
    pub fn with_batch_config(mut self, config: BatchConfig) -> Self {
        self.batch_config = config;
        self
    }

    /// Configure retry behavior
    pub fn with_retry_config(mut self, config: RetryConfig) -> Self {
        self.retry_config = config;
        self
    }

    /// Start the pipeline
    pub async fn start(&mut self) -> Result<()> {
        self.is_running.store(true, Ordering::Relaxed);

        // Start batch processor
        let storage_manager = self.storage_manager.clone();
        let job_queue = self.job_queue.clone();
        let pending_jobs = self.pending_jobs.clone();
        let batch_config = self.batch_config.clone();
        let is_running = self.is_running.clone();
        let stats = self.stats.clone();

        self.batch_processor = Some(tokio::spawn(async move {
            Self::batch_processor_loop(
                storage_manager,
                job_queue,
                pending_jobs,
                batch_config,
                is_running,
                stats,
            ).await;
        }));

        // Start result aggregator
        let storage_manager_clone = self.storage_manager.clone();
        let pending_jobs_clone = self.pending_jobs.clone();
        let retry_config = self.retry_config.clone();
        let is_running_clone = self.is_running.clone();

        self.result_aggregator = Some(tokio::spawn(async move {
            Self::result_aggregator_loop(
                storage_manager_clone,
                pending_jobs_clone,
                retry_config,
                is_running_clone,
            ).await;
        }));

        Ok(())
    }

    /// Stop the pipeline
    pub async fn stop(&mut self) -> Result<()> {
        self.is_running.store(false, Ordering::Relaxed);

        if let Some(handle) = self.batch_processor.take() {
            handle.abort();
        }

        if let Some(handle) = self.result_aggregator.take() {
            handle.abort();
        }

        Ok(())
    }

    /// Submit a single job
    pub async fn submit_job(
        &self,
        source_agent: usize,
        request: JobRequest,
    ) -> Result<Uuid> {
        let job_id = Uuid::new_v4();
        let queued_job = QueuedJob {
            id: job_id,
            source_agent,
            request,
            submitted_at: Instant::now(),
            priority_score: self.calculate_priority_score(&request),
        };

        // Add to queue
        self.job_queue.lock().await.push_back(queued_job);
        self.stats.jobs_queued.fetch_add(1, Ordering::Relaxed);

        Ok(job_id)
    }

    /// Submit multiple jobs as a batch
    pub async fn submit_batch(
        &self,
        source_agent: usize,
        requests: Vec<JobRequest>,
    ) -> Result<Vec<Uuid>> {
        let mut job_ids = vec![];
        let mut queue = self.job_queue.lock().await;

        for request in requests {
            let job_id = Uuid::new_v4();
            let queued_job = QueuedJob {
                id: job_id,
                source_agent,
                request,
                submitted_at: Instant::now(),
                priority_score: self.calculate_priority_score(&request),
            };

            queue.push_back(queued_job);
            job_ids.push(job_id);
        }

        self.stats.jobs_queued.fetch_add(job_ids.len() as u64, Ordering::Relaxed);
        Ok(job_ids)
    }

    /// Wait for job completion
    pub async fn wait_for_job(&self, job_id: Uuid, timeout: Duration) -> Result<JobResult> {
        let start = Instant::now();
        
        while start.elapsed() < timeout {
            let pending = self.pending_jobs.read().await;
            if let Some(pipeline_job) = pending.get(&job_id) {
                if let Some(result) = &pipeline_job.result {
                    return Ok(result.clone());
                }
            }

            tokio::time::sleep(Duration::from_millis(10)).await;
        }

        Err(anyhow!("Job {} timed out after {:?}", job_id, timeout))
    }

    /// Wait for batch completion
    pub async fn wait_for_batch(
        &self,
        job_ids: Vec<Uuid>,
        timeout: Duration,
    ) -> Result<Vec<JobResult>> {
        let start = Instant::now();
        let mut results = vec![];
        let mut remaining_ids = job_ids.into_iter().collect::<std::collections::HashSet<_>>();

        while !remaining_ids.is_empty() && start.elapsed() < timeout {
            let pending = self.pending_jobs.read().await;
            let mut completed = vec![];

            for job_id in &remaining_ids {
                if let Some(pipeline_job) = pending.get(job_id) {
                    if let Some(result) = &pipeline_job.result {
                        results.push(result.clone());
                        completed.push(*job_id);
                    }
                }
            }

            for job_id in completed {
                remaining_ids.remove(&job_id);
            }

            if !remaining_ids.is_empty() {
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
        }

        if !remaining_ids.is_empty() {
            Err(anyhow!("Batch timed out with {} incomplete jobs", remaining_ids.len()))
        } else {
            Ok(results)
        }
    }

    /// Get pipeline statistics
    pub fn get_stats(&self) -> PipelineStats {
        PipelineStats {
            jobs_queued: self.stats.jobs_queued.load(Ordering::Relaxed),
            jobs_submitted: self.stats.jobs_submitted.load(Ordering::Relaxed),
            jobs_completed: self.stats.jobs_completed.load(Ordering::Relaxed),
            jobs_failed: self.stats.jobs_failed.load(Ordering::Relaxed),
            jobs_retried: self.stats.jobs_retried.load(Ordering::Relaxed),
            batches_processed: self.stats.batches_processed.load(Ordering::Relaxed),
            avg_batch_size: self.stats.avg_batch_size.load(Ordering::Relaxed) as f64 / 100.0,
            avg_processing_time_ms: self.stats.avg_processing_time_ms.load(Ordering::Relaxed) as f64 / 100.0,
            queue_depth: 0, // Would need to lock to get current depth
        }
    }

    /// Calculate priority score for job ordering
    fn calculate_priority_score(&self, request: &JobRequest) -> u64 {
        match request.priority {
            JobPriority::Critical => 1000,
            JobPriority::High => 800,
            JobPriority::Normal => 500,
            JobPriority::Low => 200,
            JobPriority::Background => 100,
        }
    }

    /// Batch processor loop
    async fn batch_processor_loop(
        storage_manager: Arc<SharedStorageManager>,
        job_queue: Arc<Mutex<VecDeque<QueuedJob>>>,
        pending_jobs: Arc<RwLock<HashMap<Uuid, PipelineJob>>>,
        batch_config: BatchConfig,
        is_running: Arc<AtomicBool>,
        stats: Arc<PipelineStats>,
    ) {
        while is_running.load(Ordering::Relaxed) {
            let batch = {
                let mut queue = job_queue.lock().await;
                let batch_size = queue.len().min(batch_config.max_batch_size);
                
                if batch_size == 0 {
                    tokio::time::sleep(batch_config.batch_timeout).await;
                    continue;
                }

                // Sort by priority and take batch
                let mut jobs: Vec<_> = queue.drain(..batch_size).collect();
                jobs.sort_by_key(|job| std::cmp::Reverse(job.priority_score));
                jobs
            };

            if !batch.is_empty() {
                let batch_size = batch.len();
                let batch_start = Instant::now();

                // Submit batch to storage
                for queued_job in batch {
                    let target_agent = match queued_job.request.target {
                        TargetAgent::Gpu(id) => AgentId::GpuAgent(id),
                        TargetAgent::Cpu(id) => AgentId::CpuAgent(id),
                        _ => continue, // Skip unsupported targets for now
                    };

                    let job = AgentJob {
                        id: queued_job.id,
                        job_type: queued_job.request.job_type,
                        source_agent: AgentId::CpuAgent(queued_job.source_agent),
                        target_agent,
                        data: queued_job.request.data,
                        priority: queued_job.request.priority,
                        created_at: Utc::now(),
                        expires_at: None,
                        metadata: queued_job.request.metadata,
                    };

                    match storage_manager.submit_job(job).await {
                        Ok(_) => {
                            // Track pending job
                            let pipeline_job = PipelineJob {
                                id: queued_job.id,
                                submitted_at: batch_start,
                                retry_count: 0,
                                result: None,
                            };
                            pending_jobs.write().await.insert(queued_job.id, pipeline_job);
                            stats.jobs_submitted.fetch_add(1, Ordering::Relaxed);
                        }
                        Err(e) => {
                            eprintln!("Failed to submit job {}: {}", queued_job.id, e);
                            stats.jobs_failed.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                }

                stats.batches_processed.fetch_add(1, Ordering::Relaxed);
                let batch_time_ms = batch_start.elapsed().as_millis() as u64;
                stats.avg_batch_size.store((batch_size * 100) as u64, Ordering::Relaxed);
                stats.avg_processing_time_ms.store(batch_time_ms * 100, Ordering::Relaxed);
            }

            tokio::time::sleep(batch_config.batch_timeout).await;
        }
    }

    /// Result aggregator loop
    async fn result_aggregator_loop(
        storage_manager: Arc<SharedStorageManager>,
        pending_jobs: Arc<RwLock<HashMap<Uuid, PipelineJob>>>,
        retry_config: RetryConfig,
        is_running: Arc<AtomicBool>,
    ) {
        while is_running.load(Ordering::Relaxed) {
            // Check for completed jobs and timeouts
            let mut to_retry = vec![];
            let mut to_complete = vec![];

            {
                let mut pending = pending_jobs.write().await;
                let now = Instant::now();

                for (job_id, pipeline_job) in pending.iter_mut() {
                    // Check for timeout
                    if now.duration_since(pipeline_job.submitted_at) > retry_config.job_timeout {
                        if pipeline_job.retry_count < retry_config.max_retries {
                            pipeline_job.retry_count += 1;
                            pipeline_job.submitted_at = now;
                            to_retry.push(*job_id);
                        } else {
                            // Mark as failed
                            pipeline_job.result = Some(JobResult {
                                original_job_id: *job_id,
                                data: b"Job timed out".to_vec(),
                                source_agent: AgentId::GpuAgent(0), // Placeholder
                                status: JobStatus::Failed,
                                retry_count: pipeline_job.retry_count,
                                processing_time_ms: retry_config.job_timeout.as_millis() as u64,
                                metadata: [("error".to_string(), "timeout".to_string())].into_iter().collect(),
                            });
                            to_complete.push(*job_id);
                        }
                    }
                }
            }

            // Handle retries
            for job_id in to_retry {
                // In a real implementation, would resubmit the job
                println!("Retrying job {}", job_id);
            }

            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }
}

/// Batch processing configuration
#[derive(Clone)]
pub struct BatchConfig {
    pub max_batch_size: usize,
    pub batch_timeout: Duration,
    pub priority_based_batching: bool,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 32,
            batch_timeout: Duration::from_millis(100),
            priority_based_batching: true,
        }
    }
}

/// Retry configuration
#[derive(Clone)]
pub struct RetryConfig {
    pub max_retries: u32,
    pub retry_delay: Duration,
    pub job_timeout: Duration,
    pub exponential_backoff: bool,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            retry_delay: Duration::from_millis(100),
            job_timeout: Duration::from_secs(30),
            exponential_backoff: true,
        }
    }
}

/// Job in the processing queue
struct QueuedJob {
    id: Uuid,
    source_agent: usize,
    request: JobRequest,
    submitted_at: Instant,
    priority_score: u64,
}

/// Pipeline job tracking
struct PipelineJob {
    id: Uuid,
    submitted_at: Instant,
    retry_count: u32,
    result: Option<JobResult>,
}

/// Pipeline statistics
#[derive(Default)]
pub struct PipelineStats {
    pub jobs_queued: AtomicU64,
    pub jobs_submitted: AtomicU64,
    pub jobs_completed: AtomicU64,
    pub jobs_failed: AtomicU64,
    pub jobs_retried: AtomicU64,
    pub batches_processed: AtomicU64,
    pub avg_batch_size: AtomicU64, // * 100 for precision
    pub avg_processing_time_ms: AtomicU64, // * 100 for precision
    pub queue_depth: u64,
}

/// Workflow builder for complex multi-stage jobs
pub struct WorkflowBuilder {
    stages: Vec<WorkflowStage>,
    dependencies: HashMap<String, Vec<String>>,
}

impl WorkflowBuilder {
    /// Create new workflow builder
    pub fn new() -> Self {
        Self {
            stages: vec![],
            dependencies: HashMap::new(),
        }
    }

    /// Add a stage to the workflow
    pub fn add_stage(mut self, stage: WorkflowStage) -> Self {
        self.stages.push(stage);
        self
    }

    /// Add a dependency between stages
    pub fn add_dependency(mut self, stage: String, depends_on: String) -> Self {
        self.dependencies.entry(stage).or_insert_with(Vec::new).push(depends_on);
        self
    }

    /// Build and execute the workflow
    pub async fn execute(
        self,
        pipeline: &JobSubmissionPipeline,
        input_data: Vec<u8>,
    ) -> Result<Vec<u8>> {
        // Topological sort of stages based on dependencies
        let execution_order = self.topological_sort()?;
        
        let mut stage_results = HashMap::new();
        let mut current_data = input_data;

        for stage_name in execution_order {
            let stage = self.stages.iter()
                .find(|s| s.name == stage_name)
                .ok_or_else(|| anyhow!("Stage {} not found", stage_name))?;

            // Prepare input data from dependencies
            let stage_input = if let Some(deps) = self.dependencies.get(&stage_name) {
                let mut combined_input = current_data.clone();
                for dep in deps {
                    if let Some(dep_result) = stage_results.get(dep) {
                        combined_input.extend_from_slice(dep_result);
                    }
                }
                combined_input
            } else {
                current_data.clone()
            };

            // Execute stage
            let job_request = JobRequest {
                job_type: stage.job_type,
                target: stage.target,
                data: stage_input,
                priority: stage.priority,
                metadata: stage.metadata.clone(),
            };

            let job_id = pipeline.submit_job(stage.source_agent, job_request).await?;
            let result = pipeline.wait_for_job(job_id, Duration::from_secs(30)).await?;

            stage_results.insert(stage_name, result.data.clone());
            current_data = result.data;
        }

        Ok(current_data)
    }

    /// Topological sort of workflow stages
    fn topological_sort(&self) -> Result<Vec<String>> {
        let mut visited = std::collections::HashSet::new();
        let mut result = vec![];
        let mut temp_visited = std::collections::HashSet::new();

        for stage in &self.stages {
            if !visited.contains(&stage.name) {
                self.dfs_visit(&stage.name, &mut visited, &mut temp_visited, &mut result)?;
            }
        }

        result.reverse();
        Ok(result)
    }

    fn dfs_visit(
        &self,
        stage: &str,
        visited: &mut std::collections::HashSet<String>,
        temp_visited: &mut std::collections::HashSet<String>,
        result: &mut Vec<String>,
    ) -> Result<()> {
        if temp_visited.contains(stage) {
            return Err(anyhow!("Circular dependency detected involving stage {}", stage));
        }

        if !visited.contains(stage) {
            temp_visited.insert(stage.to_string());

            if let Some(deps) = self.dependencies.get(stage) {
                for dep in deps {
                    self.dfs_visit(dep, visited, temp_visited, result)?;
                }
            }

            temp_visited.remove(stage);
            visited.insert(stage.to_string());
            result.push(stage.to_string());
        }

        Ok(())
    }
}

/// Workflow stage definition
pub struct WorkflowStage {
    pub name: String,
    pub source_agent: usize,
    pub job_type: JobType,
    pub target: TargetAgent,
    pub priority: JobPriority,
    pub metadata: HashMap<String, String>,
}

impl Default for WorkflowBuilder {
    fn default() -> Self {
        Self::new()
    }
}