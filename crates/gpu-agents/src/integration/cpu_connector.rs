//! CPU agent connector for shared storage integration
//! 
//! Handles CPU-side job submission, result polling, and workflow coordination
//! while maintaining strict resource isolation.

use super::*;
use cpu_agents::{CpuAgent, IoManager, Orchestrator};
use dashmap::DashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, RwLock};
use tokio::task::JoinHandle;
use anyhow::{Result, anyhow};

/// CPU agent connector for shared storage
pub struct CpuAgentConnector {
    cpu_id: usize,
    storage_manager: Arc<SharedStorageManager>,
    poll_interval: Duration,
    is_running: Arc<AtomicBool>,
    stats: Arc<CpuConnectorStats>,
    cpu_agent: Arc<Mutex<Option<Arc<CpuAgent>>>>,
    polling_handle: Option<JoinHandle<()>>,
    pending_jobs: Arc<DashMap<Uuid, PendingJob>>,
    stream_producers: Arc<DashMap<Uuid, StreamProducer>>,
    stream_consumers: Arc<DashMap<Uuid, StreamConsumer>>,
}

impl CpuAgentConnector {
    /// Create new CPU agent connector
    pub fn new(
        cpu_id: usize,
        storage_manager: Arc<SharedStorageManager>,
        poll_interval: Duration,
    ) -> Self {
        Self {
            cpu_id,
            storage_manager,
            poll_interval,
            is_running: Arc::new(AtomicBool::new(true)),
            stats: Arc::new(CpuConnectorStats::default()),
            cpu_agent: Arc::new(Mutex::new(None)),
            polling_handle: None,
            pending_jobs: Arc::new(RwLock::new(HashMap::new())),
            stream_producers: Arc::new(RwLock::new(HashMap::new())),
            stream_consumers: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Get CPU ID
    pub fn cpu_id(&self) -> usize {
        self.cpu_id
    }

    /// Check if connector is running
    pub fn is_running(&self) -> bool {
        self.is_running.load(Ordering::Relaxed)
    }

    /// Start the connector
    pub async fn start(&mut self) -> Result<()> {
        // Initialize CPU agent if not already done
        if self.cpu_agent.lock().await.is_none() {
            let cpu_agent = Arc::new(CpuAgent::new(self.cpu_id)?);
            *self.cpu_agent.lock().await = Some(cpu_agent);
        }

        // Start polling thread for results
        let is_running = self.is_running.clone();
        let storage_manager = self.storage_manager.clone();
        let poll_interval = self.poll_interval;
        let cpu_id = self.cpu_id;
        let stats = self.stats.clone();
        let pending_jobs = self.pending_jobs.clone();

        self.polling_handle = Some(tokio::spawn(async move {
            while is_running.load(Ordering::Relaxed) {
                // Poll for results and completed jobs
                let results = match storage_manager.poll_jobs(
                    AgentId::CpuAgent(cpu_id),
                    10, // Max batch size
                ).await {
                    Ok(results) => results,
                    Err(e) => {
                        eprintln!("CPU {} result polling error: {}", cpu_id, e);
                        continue;
                    }
                };

                if !results.is_empty() {
                    stats.results_received.fetch_add(results.len() as u64, Ordering::Relaxed);
                    
                    // Update pending jobs
                    let mut pending = pending_jobs.write().await;
                    for result in results {
                        if let Some(original_id_str) = result.metadata.get("original_job_id") {
                            if let Ok(original_id) = Uuid::parse_str(original_id_str) {
                                if let Some(mut pending_job) = pending.remove(&original_id) {
                                    pending_job.result = Some(JobResult {
                                        original_job_id: original_id,
                                        data: result.data,
                                        source_agent: result.source_agent,
                                        status: if result.job_type == JobType::Error {
                                            JobStatus::Failed
                                        } else {
                                            JobStatus::Completed
                                        },
                                        retry_count: 0,
                                        processing_time_ms: result.metadata
                                            .get("processing_time_ms")
                                            .and_then(|s| s.parse().ok())
                                            .unwrap_or(0),
                                        metadata: result.metadata,
                                    });
                                    pending.insert(original_id, pending_job);
                                }
                            }
                        }
                    }
                }

                tokio::time::sleep(poll_interval).await;
            }
        }));

        Ok(())
    }

    /// Shutdown the connector
    pub async fn shutdown(&mut self) -> Result<()> {
        self.is_running.store(false, Ordering::Relaxed);
        
        if let Some(handle) = self.polling_handle.take() {
            handle.abort();
        }

        // Close all streams
        let stream_producers = self.stream_producers.write().await;
        for (_, producer) in stream_producers.iter() {
            producer.close().await?;
        }

        let stream_consumers = self.stream_consumers.write().await;
        for (_, consumer) in stream_consumers.iter() {
            consumer.close().await?;
        }

        Ok(())
    }

    /// Submit a job
    pub async fn submit_job(&self, request: JobRequest) -> Result<Uuid> {
        let job_id = Uuid::new_v4();
        let target_agent = match request.target {
            TargetAgent::Gpu(id) => AgentId::GpuAgent(id),
            TargetAgent::Cpu(id) => AgentId::CpuAgent(id),
            TargetAgent::Auto => {
                // Let load balancer decide
                return Err(anyhow!("Auto targeting not implemented"));
            }
            TargetAgent::AllGpus => {
                // Broadcast to all GPUs
                return Err(anyhow!("Broadcast not implemented"));
            }
            TargetAgent::AllCpus => {
                // Broadcast to all CPUs
                return Err(anyhow!("Broadcast not implemented"));
            }
        };

        let job = AgentJob {
            id: job_id,
            job_type: request.job_type,
            source_agent: AgentId::CpuAgent(self.cpu_id),
            target_agent,
            data: request.data,
            priority: request.priority,
            created_at: Utc::now(),
            expires_at: None,
            metadata: request.metadata,
        };

        // Track pending job
        let pending_job = PendingJob {
            id: job_id,
            submitted_at: Instant::now(),
            result: None,
        };
        self.pending_jobs.write().await.insert(job_id, pending_job);

        // Submit to storage
        let submitted_id = self.storage_manager.submit_job(job).await?;
        self.stats.jobs_submitted.fetch_add(1, Ordering::Relaxed);
        self.stats.bytes_sent.fetch_add(job.data.len() as u64, Ordering::Relaxed);

        Ok(submitted_id)
    }

    /// Submit a compute job
    pub async fn submit_compute_job(
        &self,
        data: Vec<u8>,
        target: TargetAgent,
        priority: JobPriority,
    ) -> Result<Uuid> {
        self.submit_job(JobRequest {
            job_type: JobType::ComputeRequest,
            target,
            data,
            priority,
            metadata: Default::default(),
        }).await
    }

    /// Submit an evolution job
    pub async fn submit_evolution_job(
        &self,
        task: EvolutionTask,
        target: TargetAgent,
        priority: JobPriority,
    ) -> Result<Uuid> {
        let data = bincode::serialize(&task)?;
        self.submit_job(JobRequest {
            job_type: JobType::Custom(EVOLUTION_JOB_TYPE),
            target,
            data,
            priority,
            metadata: Default::default(),
        }).await
    }

    /// Poll for results
    pub async fn poll_results(&self, max_count: usize) -> Result<Vec<JobResult>> {
        let mut results = vec![];
        let mut pending = self.pending_jobs.write().await;
        let mut to_remove = vec![];

        for (job_id, pending_job) in pending.iter() {
            if let Some(result) = &pending_job.result {
                results.push(result.clone());
                to_remove.push(*job_id);
                
                if results.len() >= max_count {
                    break;
                }
            }
        }

        // Remove completed jobs
        for job_id in to_remove {
            pending.remove(&job_id);
        }

        Ok(results)
    }

    /// Get connector statistics
    pub async fn get_stats(&self) -> Result<ConnectorStats> {
        Ok(ConnectorStats {
            jobs_submitted: self.stats.jobs_submitted.load(Ordering::Relaxed),
            results_received: self.stats.results_received.load(Ordering::Relaxed),
            bytes_sent: self.stats.bytes_sent.load(Ordering::Relaxed),
            pending_jobs: self.pending_jobs.read().await.len() as u64,
            ..Default::default()
        })
    }

    /// Get health status
    pub async fn get_health(&self) -> Result<AgentHealth> {
        Ok(AgentHealth {
            is_healthy: self.is_running(),
            last_heartbeat: Instant::now(),
            error_count: 0,
            recovered_from_crash: false,
        })
    }

    /// Get load information
    pub async fn get_load_info(&self) -> Result<CpuLoadInfo> {
        let stats = self.get_stats().await?;
        
        Ok(CpuLoadInfo {
            jobs_submitted: stats.jobs_submitted,
            cpu_utilization: self.get_cpu_utilization().await?,
            io_operations: self.stats.io_operations.load(Ordering::Relaxed),
            queue_depth: stats.pending_jobs as usize,
        })
    }

    /// Get CPU utilization
    async fn get_cpu_utilization(&self) -> Result<f32> {
        // In a real implementation, this would query system CPU usage
        // For now, estimate based on job submission rate
        let jobs_per_sec = self.stats.jobs_submitted.load(Ordering::Relaxed) as f32 / 
                          self.stats.uptime_seconds.load(Ordering::Relaxed) as f32;
        Ok((jobs_per_sec / 50.0).min(1.0)) // Assume 50 jobs/sec = 100% utilization
    }

    /// Setup stream producer
    pub async fn setup_stream_producer(
        &self,
        stream_id: Uuid,
        config: StreamConfig,
    ) -> Result<()> {
        let producer = StreamProducer::new(
            stream_id,
            config,
            self.storage_manager.clone(),
            self.cpu_id,
        );
        self.stream_producers.write().await.insert(stream_id, producer);
        Ok(())
    }

    /// Setup stream consumer
    pub async fn setup_stream_consumer(
        &self,
        stream_id: Uuid,
        config: StreamConfig,
    ) -> Result<()> {
        let consumer = StreamConsumer::new(stream_id, config);
        self.stream_consumers.write().await.insert(stream_id, consumer);
        Ok(())
    }

    /// Submit stream data
    pub async fn submit_stream_data(
        &self,
        stream_id: Uuid,
        data: Vec<u8>,
    ) -> Result<()> {
        let producers = self.stream_producers.read().await;
        if let Some(producer) = producers.get(&stream_id) {
            producer.send_data(data).await
        } else {
            Err(anyhow!("Stream producer {} not found", stream_id))
        }
    }

    /// Poll stream results
    pub async fn poll_stream_results(
        &self,
        stream_id: Uuid,
        max_count: usize,
    ) -> Result<Vec<StreamResult>> {
        let consumers = self.stream_consumers.read().await;
        if let Some(consumer) = consumers.get(&stream_id) {
            consumer.poll_results(max_count).await
        } else {
            Err(anyhow!("Stream consumer {} not found", stream_id))
        }
    }

    /// Close a stream
    pub async fn close_stream(&self, stream_id: Uuid) -> Result<()> {
        if let Some(producer) = self.stream_producers.write().await.remove(&stream_id) {
            producer.close().await?;
        }
        if let Some(consumer) = self.stream_consumers.write().await.remove(&stream_id) {
            consumer.close().await?;
        }
        Ok(())
    }
}

/// CPU connector statistics
#[derive(Default)]
pub struct CpuConnectorStats {
    pub jobs_submitted: AtomicU64,
    pub results_received: AtomicU64,
    pub bytes_sent: AtomicU64,
    pub io_operations: AtomicU64,
    pub uptime_seconds: AtomicU64,
}

/// Pending job tracking
struct PendingJob {
    id: Uuid,
    submitted_at: Instant,
    result: Option<JobResult>,
}

/// Stream producer for continuous data
struct StreamProducer {
    stream_id: Uuid,
    config: StreamConfig,
    storage_manager: Arc<SharedStorageManager>,
    cpu_id: usize,
    chunks_sent: AtomicU64,
}

impl StreamProducer {
    fn new(
        stream_id: Uuid,
        config: StreamConfig,
        storage_manager: Arc<SharedStorageManager>,
        cpu_id: usize,
    ) -> Self {
        Self {
            stream_id,
            config,
            storage_manager,
            cpu_id,
            chunks_sent: AtomicU64::new(0),
        }
    }

    async fn send_data(&self, data: Vec<u8>) -> Result<()> {
        // Split data into chunks if needed
        let chunks = if data.len() > self.config.chunk_size {
            data.chunks(self.config.chunk_size)
                .map(|chunk| chunk.to_vec())
                .collect()
        } else {
            vec![data]
        };

        for chunk in chunks {
            let job = AgentJob {
                id: Uuid::new_v4(),
                job_type: JobType::StreamData,
                source_agent: AgentId::CpuAgent(self.cpu_id),
                target_agent: AgentId::GpuAgent(0), // Default target
                data: chunk,
                priority: JobPriority::Normal,
                created_at: Utc::now(),
                expires_at: None,
                metadata: [
                    ("stream_id".to_string(), self.stream_id.to_string()),
                    ("chunk_id".to_string(), self.chunks_sent.load(Ordering::Relaxed).to_string()),
                ].into_iter().collect(),
            };

            self.storage_manager.submit_job(job).await?;
            self.chunks_sent.fetch_add(1, Ordering::Relaxed);
        }

        Ok(())
    }

    async fn close(&self) -> Result<()> {
        // Send end-of-stream marker
        let eos_job = AgentJob {
            id: Uuid::new_v4(),
            job_type: JobType::StreamData,
            source_agent: AgentId::CpuAgent(self.cpu_id),
            target_agent: AgentId::GpuAgent(0),
            data: vec![],
            priority: JobPriority::High,
            created_at: Utc::now(),
            expires_at: None,
            metadata: [
                ("stream_id".to_string(), self.stream_id.to_string()),
                ("end_of_stream".to_string(), "true".to_string()),
            ].into_iter().collect(),
        };

        self.storage_manager.submit_job(eos_job).await?;
        Ok(())
    }
}

/// Stream consumer for continuous results
struct StreamConsumer {
    stream_id: Uuid,
    config: StreamConfig,
    results: Arc<Mutex<Vec<StreamResult>>>,
}

impl StreamConsumer {
    fn new(stream_id: Uuid, config: StreamConfig) -> Self {
        Self {
            stream_id,
            config,
            results: Arc::new(Mutex::new(Vec::new())),
        }
    }

    async fn poll_results(&self, max_count: usize) -> Result<Vec<StreamResult>> {
        let mut results = self.results.lock().await;
        let to_return = results.drain(..results.len().min(max_count)).collect();
        Ok(to_return)
    }

    async fn close(&self) -> Result<()> {
        // Clear any remaining results
        self.results.lock().await.clear();
        Ok(())
    }
}

/// Stream processing result
#[derive(Clone)]
pub struct StreamResult {
    pub stream_id: Uuid,
    pub chunk_id: u64,
    pub data: Vec<u8>,
    pub timestamp: Instant,
}

/// Evolution task definition
#[derive(serde::Serialize, serde::Deserialize)]
pub enum EvolutionTask {
    Optimize {
        population: Vec<Vec<f32>>,
        fitness_function: FitnessFunction,
        generations: usize,
    },
    Evaluate {
        individuals: Vec<Vec<f32>>,
        fitness_function: FitnessFunction,
    },
}

/// Fitness function types
#[derive(serde::Serialize, serde::Deserialize)]
pub enum FitnessFunction {
    Sphere,
    Rastrigin,
    Rosenbrock,
    Custom(String),
}

/// Evolution result
#[derive(serde::Serialize, serde::Deserialize)]
pub struct EvolutionResult {
    pub best_fitness: f64,
    pub best_individual: Vec<f32>,
    pub generations_computed: usize,
    pub convergence_history: Vec<f64>,
}

// Constants for custom job types
const EVOLUTION_JOB_TYPE: u32 = 5001;

// Extension traits for CpuAgent
impl CpuAgent {
    /// Create new CPU agent
    fn new(id: usize) -> Result<Self> {
        // Placeholder implementation
        Ok(CpuAgent { id })
    }
}

// Placeholder struct for CpuAgent
pub struct CpuAgent {
    id: usize,
}