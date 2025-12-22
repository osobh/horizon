//! GPU agent connector for shared storage integration
//! 
//! Handles GPU-side job polling, processing, and result submission
//! while maintaining strict resource isolation.

use super::*;
use crate::GpuAgent;
use cudarc::driver::{CudaDevice, CudaStream};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, RwLock};
use tokio::task::JoinHandle;
use anyhow::{Result, anyhow};

/// GPU agent connector for shared storage
pub struct GpuAgentConnector {
    gpu_id: usize,
    storage_manager: Arc<SharedStorageManager>,
    poll_interval: Duration,
    is_running: Arc<AtomicBool>,
    stats: Arc<GpuConnectorStats>,
    gpu_agent: Arc<Mutex<Option<Arc<GpuAgent>>>>,
    polling_handle: Option<JoinHandle<()>>,
    stream_handlers: Arc<RwLock<HashMap<Uuid, StreamHandler>>>,
}

impl GpuAgentConnector {
    /// Create new GPU agent connector
    pub fn new(
        gpu_id: usize,
        storage_manager: Arc<SharedStorageManager>,
        poll_interval: Duration,
    ) -> Self {
        Self {
            gpu_id,
            storage_manager,
            poll_interval,
            is_running: Arc::new(AtomicBool::new(true)),
            stats: Arc::new(GpuConnectorStats::default()),
            gpu_agent: Arc::new(Mutex::new(None)),
            polling_handle: None,
            stream_handlers: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Get GPU ID
    pub fn gpu_id(&self) -> usize {
        self.gpu_id
    }

    /// Check if connector is running
    pub fn is_running(&self) -> bool {
        self.is_running.load(Ordering::Relaxed)
    }

    /// Start the connector
    pub async fn start(&mut self) -> Result<()> {
        // Initialize GPU agent if not already done
        if self.gpu_agent.lock().await.is_none() {
            let device = CudaDevice::new(self.gpu_id)?;
            let gpu_agent = Arc::new(GpuAgent::new(Arc::new(device)));
            *self.gpu_agent.lock().await = Some(gpu_agent);
        }

        // Start polling thread
        let is_running = self.is_running.clone();
        let storage_manager = self.storage_manager.clone();
        let poll_interval = self.poll_interval;
        let gpu_id = self.gpu_id;
        let stats = self.stats.clone();
        let gpu_agent = self.gpu_agent.clone();
        let stream_handlers = self.stream_handlers.clone();

        self.polling_handle = Some(tokio::spawn(async move {
            while is_running.load(Ordering::Relaxed) {
                // Poll for new jobs
                let jobs = match storage_manager.poll_jobs(
                    AgentId::GpuAgent(gpu_id),
                    10, // Max batch size
                ).await {
                    Ok(jobs) => jobs,
                    Err(e) => {
                        eprintln!("GPU {} polling error: {}", gpu_id, e);
                        continue;
                    }
                };

                if !jobs.is_empty() {
                    stats.jobs_received.fetch_add(jobs.len() as u64, Ordering::Relaxed);
                    
                    // Process jobs
                    for job in jobs {
                        let start = Instant::now();
                        
                        match Self::process_job(&gpu_agent, &job, &stream_handlers).await {
                            Ok(result) => {
                                // Submit result back
                                if let Err(e) = storage_manager.submit_job(
                                    AgentJob {
                                        id: Uuid::new_v4(),
                                        job_type: JobType::ComputeResult,
                                        source_agent: AgentId::GpuAgent(gpu_id),
                                        target_agent: job.source_agent,
                                        data: result,
                                        priority: job.priority,
                                        created_at: Utc::now(),
                                        expires_at: None,
                                        metadata: [
                                            ("original_job_id".to_string(), job.id.to_string()),
                                            ("processing_time_ms".to_string(), 
                                             start.elapsed().as_millis().to_string()),
                                        ].into_iter().collect(),
                                    }
                                ).await {
                                    eprintln!("GPU {} result submission error: {}", gpu_id, e);
                                }
                                
                                stats.jobs_processed.fetch_add(1, Ordering::Relaxed);
                                stats.total_processing_time_us.fetch_add(
                                    start.elapsed().as_micros() as u64,
                                    Ordering::Relaxed
                                );
                            }
                            Err(e) => {
                                eprintln!("GPU {} job processing error: {}", gpu_id, e);
                                stats.jobs_failed.fetch_add(1, Ordering::Relaxed);
                                
                                // Submit error result
                                let _ = storage_manager.submit_job(
                                    AgentJob {
                                        id: Uuid::new_v4(),
                                        job_type: JobType::Error,
                                        source_agent: AgentId::GpuAgent(gpu_id),
                                        target_agent: job.source_agent,
                                        data: e.to_string().into_bytes(),
                                        priority: job.priority,
                                        created_at: Utc::now(),
                                        expires_at: None,
                                        metadata: [
                                            ("original_job_id".to_string(), job.id.to_string()),
                                            ("error".to_string(), e.to_string()),
                                        ].into_iter().collect(),
                                    }
                                ).await;
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
        let stream_handlers = self.stream_handlers.write().await;
        for (_, handler) in stream_handlers.iter() {
            handler.close().await?;
        }

        Ok(())
    }

    /// Process a single job
    async fn process_job(
        gpu_agent: &Arc<Mutex<Option<Arc<GpuAgent>>>>,
        job: &AgentJob,
        stream_handlers: &Arc<RwLock<HashMap<Uuid, StreamHandler>>>,
    ) -> Result<Vec<u8>> {
        let agent = gpu_agent.lock().await;
        let agent = agent.as_ref().ok_or_else(|| anyhow!("GPU agent not initialized"))?;

        match &job.job_type {
            JobType::ComputeRequest => {
                // General compute request
                agent.process_compute(&job.data).await
            }
            JobType::StreamData => {
                // Streaming data processing
                if let Some(stream_id_str) = job.metadata.get("stream_id") {
                    let stream_id = Uuid::parse_str(stream_id_str)?;
                    let handlers = stream_handlers.read().await;
                    
                    if let Some(handler) = handlers.get(&stream_id) {
                        handler.process_chunk(&job.data).await
                    } else {
                        Err(anyhow!("Stream {} not found", stream_id))
                    }
                } else {
                    Err(anyhow!("Stream ID not provided"))
                }
            }
            JobType::Custom(type_id) => {
                // Custom job types
                agent.process_custom(*type_id, &job.data).await
            }
            _ => Err(anyhow!("Unsupported job type for GPU: {:?}", job.job_type)),
        }
    }

    /// Submit a result
    pub async fn submit_result(
        &self,
        original_job_id: Uuid,
        data: Vec<u8>,
        target: AgentId,
    ) -> Result<Uuid> {
        let job = AgentJob {
            id: Uuid::new_v4(),
            job_type: JobType::ComputeResult,
            source_agent: AgentId::GpuAgent(self.gpu_id),
            target_agent: target,
            data,
            priority: JobPriority::Normal,
            created_at: Utc::now(),
            expires_at: None,
            metadata: [
                ("original_job_id".to_string(), original_job_id.to_string()),
            ].into_iter().collect(),
        };

        self.storage_manager.submit_job(job).await
    }

    /// Get connector statistics
    pub async fn get_stats(&self) -> Result<ConnectorStats> {
        let jobs_processed = self.stats.jobs_processed.load(Ordering::Relaxed);
        let avg_processing_time_us = if jobs_processed > 0 {
            self.stats.total_processing_time_us.load(Ordering::Relaxed) / jobs_processed
        } else {
            0
        };

        Ok(ConnectorStats {
            jobs_received: self.stats.jobs_received.load(Ordering::Relaxed),
            jobs_processed,
            jobs_failed: self.stats.jobs_failed.load(Ordering::Relaxed),
            bytes_processed: self.stats.bytes_processed.load(Ordering::Relaxed),
            avg_processing_time_us,
            compute_utilization: self.get_gpu_utilization().await?,
            memory_utilization: self.get_memory_utilization().await?,
            evolution_generations_computed: self.stats.evolution_generations.load(Ordering::Relaxed),
            knowledge_queries_processed: self.stats.knowledge_queries.load(Ordering::Relaxed),
            stream_chunks_processed: self.stats.stream_chunks.load(Ordering::Relaxed),
            frames_processed: self.stats.frames_processed.load(Ordering::Relaxed),
        })
    }

    /// Get GPU utilization
    async fn get_gpu_utilization(&self) -> Result<f32> {
        // In a real implementation, this would query NVML
        // For now, estimate based on job processing rate
        let jobs_per_sec = self.stats.jobs_processed.load(Ordering::Relaxed) as f32 / 
                          self.stats.uptime_seconds.load(Ordering::Relaxed) as f32;
        Ok((jobs_per_sec / 100.0).min(1.0)) // Assume 100 jobs/sec = 100% utilization
    }

    /// Get memory utilization
    async fn get_memory_utilization(&self) -> Result<f32> {
        // In a real implementation, this would query CUDA memory info
        // For now, return a placeholder
        Ok(0.75) // 75% utilization
    }

    /// Get health status
    pub async fn get_health(&self) -> Result<AgentHealth> {
        Ok(AgentHealth {
            is_healthy: self.is_running(),
            last_heartbeat: Instant::now(),
            error_count: self.stats.jobs_failed.load(Ordering::Relaxed) as u32,
            recovered_from_crash: false,
        })
    }

    /// Get load information
    pub async fn get_load_info(&self) -> Result<GpuLoadInfo> {
        let stats = self.get_stats().await?;
        
        Ok(GpuLoadInfo {
            jobs_processed: stats.jobs_processed,
            compute_utilization: stats.compute_utilization,
            memory_utilization: stats.memory_utilization,
            queue_depth: self.storage_manager.get_queue_depth(
                AgentId::GpuAgent(self.gpu_id)
            ).await?,
        })
    }

    /// Setup stream processor
    pub async fn setup_stream_processor(
        &self,
        stream_id: Uuid,
        config: StreamConfig,
    ) -> Result<()> {
        let handler = StreamHandler::new(stream_id, config, self.gpu_id);
        self.stream_handlers.write().await.insert(stream_id, handler);
        Ok(())
    }

    /// Close a stream
    pub async fn close_stream(&self, stream_id: Uuid) -> Result<()> {
        if let Some(handler) = self.stream_handlers.write().await.remove(&stream_id) {
            handler.close().await?;
        }
        Ok(())
    }

    /// Simulate crash for testing
    #[cfg(test)]
    pub fn simulate_crash(&self) -> Result<()> {
        self.is_running.store(false, Ordering::Relaxed);
        Ok(())
    }
}

/// GPU connector statistics
#[derive(Default)]
pub struct GpuConnectorStats {
    pub jobs_received: AtomicU64,
    pub jobs_processed: AtomicU64,
    pub jobs_failed: AtomicU64,
    pub bytes_processed: AtomicU64,
    pub total_processing_time_us: AtomicU64,
    pub evolution_generations: AtomicU64,
    pub knowledge_queries: AtomicU64,
    pub stream_chunks: AtomicU64,
    pub frames_processed: AtomicU64,
    pub uptime_seconds: AtomicU64,
}

/// Connector statistics
#[derive(Clone, Default)]
pub struct ConnectorStats {
    pub jobs_received: u64,
    pub jobs_processed: u64,
    pub jobs_failed: u64,
    pub bytes_processed: u64,
    pub avg_processing_time_us: u64,
    pub compute_utilization: f32,
    pub memory_utilization: f32,
    pub evolution_generations_computed: u64,
    pub knowledge_queries_processed: u64,
    pub stream_chunks_processed: u64,
    pub frames_processed: u64,
}

/// Stream handler for continuous data processing
struct StreamHandler {
    stream_id: Uuid,
    config: StreamConfig,
    gpu_id: usize,
    buffer: Arc<Mutex<Vec<u8>>>,
    processed_chunks: AtomicU64,
}

impl StreamHandler {
    fn new(stream_id: Uuid, config: StreamConfig, gpu_id: usize) -> Self {
        Self {
            stream_id,
            config,
            gpu_id,
            buffer: Arc::new(Mutex::new(Vec::with_capacity(config.window_size))),
            processed_chunks: AtomicU64::new(0),
        }
    }

    async fn process_chunk(&self, data: &[u8]) -> Result<Vec<u8>> {
        let mut buffer = self.buffer.lock().await;
        buffer.extend_from_slice(data);

        // Process when buffer reaches chunk size
        if buffer.len() >= self.config.chunk_size {
            let chunk = buffer.drain(..self.config.chunk_size).collect::<Vec<_>>();
            self.processed_chunks.fetch_add(1, Ordering::Relaxed);
            
            // Simulate processing (in real implementation, would use GPU kernels)
            Ok(chunk.into_iter().map(|b| b.wrapping_add(1)).collect())
        } else {
            Ok(vec![]) // Not enough data yet
        }
    }

    async fn close(&self) -> Result<()> {
        // Process any remaining data
        let buffer = self.buffer.lock().await;
        if !buffer.is_empty() {
            // Handle remaining data
        }
        Ok(())
    }
}

// Extension traits for GpuAgent
impl GpuAgent {
    /// Process general compute request
    async fn process_compute(&self, data: &[u8]) -> Result<Vec<u8>> {
        // Simulate compute processing
        // In real implementation, would use GPU kernels
        Ok(data.iter().map(|&b| b.wrapping_mul(2)).collect())
    }

    /// Process custom job type
    async fn process_custom(&self, type_id: u32, data: &[u8]) -> Result<Vec<u8>> {
        match type_id {
            1001 => self.process_knowledge_graph_build(data).await,
            1002 => self.process_knowledge_graph_query(data).await,
            2001 => self.process_climate_init(data).await,
            2002 => self.process_climate_timestep(data).await,
            3001 => self.process_ml_data_load(data).await,
            3002 => self.process_ml_model_init(data).await,
            _ => Err(anyhow!("Unknown custom job type: {}", type_id)),
        }
    }

    async fn process_knowledge_graph_build(&self, data: &[u8]) -> Result<Vec<u8>> {
        // Build knowledge graph chunk
        Ok(vec![1, 2, 3]) // Placeholder
    }

    async fn process_knowledge_graph_query(&self, data: &[u8]) -> Result<Vec<u8>> {
        // Process knowledge graph query
        Ok(vec![4, 5, 6]) // Placeholder
    }

    async fn process_climate_init(&self, data: &[u8]) -> Result<Vec<u8>> {
        // Initialize climate simulation
        Ok(vec![7, 8, 9]) // Placeholder
    }

    async fn process_climate_timestep(&self, data: &[u8]) -> Result<Vec<u8>> {
        // Process climate timestep
        Ok(vec![10, 11, 12]) // Placeholder
    }

    async fn process_ml_data_load(&self, data: &[u8]) -> Result<Vec<u8>> {
        // Load ML training data
        Ok(vec![13, 14, 15]) // Placeholder
    }

    async fn process_ml_model_init(&self, data: &[u8]) -> Result<Vec<u8>> {
        // Initialize ML model
        Ok(vec![16, 17, 18]) // Placeholder
    }
}