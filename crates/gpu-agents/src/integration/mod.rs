//! Integration module for connecting GPU and CPU agents via shared storage
//! 
//! This module provides the glue layer that enables CPU and GPU agents to
//! communicate efficiently through the shared storage system while maintaining
//! strict resource isolation.

use shared_storage::*;
use crate::GpuAgent;
use cpu_agents::{CpuAgent, IoManager, Orchestrator};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};
use std::collections::HashMap;
use tokio::sync::{Mutex, RwLock};
use tokio::task::JoinHandle;
use anyhow::{Result, anyhow};
use serde::{Serialize, Deserialize};
use uuid::Uuid;

mod gpu_connector;
mod cpu_connector;
mod pipeline;
mod data_flow;
mod isolation;

pub use gpu_connector::*;
pub use cpu_connector::*;
pub use pipeline::*;
pub use data_flow::*;
pub use isolation::*;

/// Integration configuration
#[derive(Clone)]
pub struct IntegrationConfig {
    /// Shared storage manager
    pub storage_manager: Arc<SharedStorageManager>,
    /// GPU polling interval
    pub gpu_poll_interval: Duration,
    /// CPU polling interval
    pub cpu_poll_interval: Duration,
    /// Maximum batch size for job processing
    pub max_batch_size: usize,
    /// Enable GPUDirect storage if available
    pub enable_gpu_direct: bool,
}

impl Default for IntegrationConfig {
    fn default() -> Self {
        Self {
            storage_manager: Arc::new(SharedStorageManager::default()),
            gpu_poll_interval: Duration::from_millis(50),
            cpu_poll_interval: Duration::from_millis(100),
            max_batch_size: 32,
            enable_gpu_direct: false,
        }
    }
}

/// Target agent for job submission
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TargetAgent {
    /// Specific GPU agent
    Gpu(usize),
    /// Specific CPU agent
    Cpu(usize),
    /// Let load balancer decide
    Auto,
    /// All GPU agents (for broadcast)
    AllGpus,
    /// All CPU agents (for broadcast)
    AllCpus,
}

/// Job request from integration layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobRequest {
    /// Job type
    pub job_type: JobType,
    /// Target agent
    pub target: TargetAgent,
    /// Job data
    pub data: Vec<u8>,
    /// Priority
    pub priority: JobPriority,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

/// Job result from integration layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobResult {
    /// Original job ID
    pub original_job_id: Uuid,
    /// Result data
    pub data: Vec<u8>,
    /// Source agent
    pub source_agent: AgentId,
    /// Job status
    pub status: JobStatus,
    /// Retry count if any
    pub retry_count: u32,
    /// Processing time in milliseconds
    pub processing_time_ms: u64,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

/// Integration manager coordinating CPU and GPU agents
pub struct IntegrationManager {
    config: IntegrationConfig,
    gpu_connectors: Arc<RwLock<HashMap<usize, Arc<GpuAgentConnector>>>>,
    cpu_connectors: Arc<RwLock<HashMap<usize, Arc<CpuAgentConnector>>>>,
    is_running: Arc<AtomicBool>,
    stats: Arc<IntegrationStats>,
    monitor_handle: Option<JoinHandle<()>>,
    isolation_verifier: Option<Arc<ResourceIsolationVerifier>>,
    load_balancer: Option<Arc<LoadBalancer>>,
    fault_tolerance: Option<Arc<FaultToleranceManager>>,
}

impl IntegrationManager {
    /// Create new integration manager
    pub fn new(config: IntegrationConfig) -> Self {
        Self {
            config,
            gpu_connectors: Arc::new(RwLock::new(HashMap::new())),
            cpu_connectors: Arc::new(RwLock::new(HashMap::new())),
            is_running: Arc::new(AtomicBool::new(false)),
            stats: Arc::new(IntegrationStats::default()),
            monitor_handle: None,
            isolation_verifier: None,
            load_balancer: None,
            fault_tolerance: None,
        }
    }

    /// Register a GPU agent
    pub async fn register_gpu_agent(&mut self, gpu_id: usize) -> Result<Arc<GpuAgentConnector>> {
        let connector = Arc::new(GpuAgentConnector::new(
            gpu_id,
            self.config.storage_manager.clone(),
            self.config.gpu_poll_interval,
        ));
        
        self.gpu_connectors.write().await.insert(gpu_id, connector.clone());
        self.stats.gpu_agents_registered.fetch_add(1, Ordering::Relaxed);
        
        Ok(connector)
    }

    /// Register a CPU agent
    pub async fn register_cpu_agent(&mut self, cpu_id: usize) -> Result<Arc<CpuAgentConnector>> {
        let connector = Arc::new(CpuAgentConnector::new(
            cpu_id,
            self.config.storage_manager.clone(),
            self.config.cpu_poll_interval,
        ));
        
        self.cpu_connectors.write().await.insert(cpu_id, connector.clone());
        self.stats.cpu_agents_registered.fetch_add(1, Ordering::Relaxed);
        
        Ok(connector)
    }

    /// Start integration services
    pub async fn start(&mut self) -> Result<()> {
        self.is_running.store(true, Ordering::Relaxed);
        
        // Start all connectors
        let gpu_connectors = self.gpu_connectors.read().await;
        for connector in gpu_connectors.values() {
            connector.start().await?;
        }
        
        let cpu_connectors = self.cpu_connectors.read().await;
        for connector in cpu_connectors.values() {
            connector.start().await?;
        }
        
        // Start monitoring
        self.start_monitoring();
        
        Ok(())
    }

    /// Shutdown integration services
    pub async fn shutdown(&mut self) -> Result<()> {
        self.is_running.store(false, Ordering::Relaxed);
        
        // Stop all connectors
        let gpu_connectors = self.gpu_connectors.read().await;
        for connector in gpu_connectors.values() {
            connector.shutdown().await?;
        }
        
        let cpu_connectors = self.cpu_connectors.read().await;
        for connector in cpu_connectors.values() {
            connector.shutdown().await?;
        }
        
        // Stop monitoring
        if let Some(handle) = self.monitor_handle.take() {
            handle.abort();
        }
        
        Ok(())
    }

    /// Enable isolation monitoring
    pub fn enable_isolation_monitoring(&mut self) -> Result<()> {
        self.isolation_verifier = Some(Arc::new(ResourceIsolationVerifier::new()));
        if let Some(verifier) = &self.isolation_verifier {
            verifier.start_monitoring()?;
        }
        Ok(())
    }

    /// Get isolation report
    pub fn get_isolation_report(&self) -> Result<IsolationReport> {
        if let Some(verifier) = &self.isolation_verifier {
            verifier.generate_report()
        } else {
            Err(anyhow!("Isolation monitoring not enabled"))
        }
    }

    /// Enable load balancing
    pub fn enable_load_balancing(&mut self, config: LoadBalancingConfig) -> Result<()> {
        self.load_balancer = Some(Arc::new(LoadBalancer::new(config)));
        Ok(())
    }

    /// Enable fault tolerance
    pub fn enable_fault_tolerance(&mut self, config: FaultToleranceConfig) -> Result<()> {
        self.fault_tolerance = Some(Arc::new(FaultToleranceManager::new(config)));
        Ok(())
    }

    /// Create a stream for continuous data processing
    pub async fn create_stream(
        &self,
        producer_id: usize,
        processor_id: usize,
        consumer_id: usize,
        config: StreamConfig,
    ) -> Result<Uuid> {
        let stream_id = Uuid::new_v4();
        
        // Setup stream in connectors
        if let Some(cpu) = self.cpu_connectors.read().await.get(&producer_id) {
            cpu.setup_stream_producer(stream_id, config.clone()).await?;
        }
        
        if let Some(gpu) = self.gpu_connectors.read().await.get(&processor_id) {
            gpu.setup_stream_processor(stream_id, config.clone()).await?;
        }
        
        if let Some(cpu) = self.cpu_connectors.read().await.get(&consumer_id) {
            cpu.setup_stream_consumer(stream_id, config).await?;
        }
        
        Ok(stream_id)
    }

    /// Close a stream
    pub async fn close_stream(&self, stream_id: Uuid) -> Result<()> {
        // Close stream in all connectors
        for connector in self.gpu_connectors.read().await.values() {
            connector.close_stream(stream_id).await?;
        }
        
        for connector in self.cpu_connectors.read().await.values() {
            connector.close_stream(stream_id).await?;
        }
        
        Ok(())
    }

    /// Get agent health status
    pub async fn get_agent_health_status(&self) -> Result<HealthStatus> {
        let mut status = HealthStatus::default();
        
        for (id, connector) in self.gpu_connectors.read().await.iter() {
            status.gpu_agents.insert(*id, connector.get_health().await?);
        }
        
        for (id, connector) in self.cpu_connectors.read().await.iter() {
            status.cpu_agents.insert(*id, connector.get_health().await?);
        }
        
        Ok(status)
    }

    /// Get load statistics
    pub async fn get_load_statistics(&self) -> Result<LoadStatistics> {
        let mut stats = LoadStatistics::default();
        
        for (id, connector) in self.gpu_connectors.read().await.iter() {
            let load = connector.get_load_info().await?;
            stats.gpu_loads.insert(*id, load);
        }
        
        for (id, connector) in self.cpu_connectors.read().await.iter() {
            let load = connector.get_load_info().await?;
            stats.cpu_loads.insert(*id, load);
        }
        
        Ok(stats)
    }

    /// Set real-time constraints
    pub fn set_realtime_constraints(&mut self, config: RealtimeConfig) -> Result<()> {
        // Configure real-time scheduling
        // This would involve OS-specific calls in a real implementation
        self.config.gpu_poll_interval = Duration::from_micros(
            (1_000_000 / config.target_fps as u64) / 2
        );
        self.config.cpu_poll_interval = Duration::from_micros(
            1_000_000 / config.target_fps as u64
        );
        Ok(())
    }

    /// Start monitoring thread
    fn start_monitoring(&mut self) {
        let is_running = self.is_running.clone();
        let stats = self.stats.clone();
        let gpu_connectors = self.gpu_connectors.clone();
        let cpu_connectors = self.cpu_connectors.clone();
        
        self.monitor_handle = Some(tokio::spawn(async move {
            while is_running.load(Ordering::Relaxed) {
                // Update statistics
                let mut total_jobs = 0u64;
                let mut total_bytes = 0u64;
                
                for connector in gpu_connectors.read().await.values() {
                    let conn_stats = connector.get_stats().await.unwrap_or_default();
                    total_jobs += conn_stats.jobs_processed;
                    total_bytes += conn_stats.bytes_processed;
                }
                
                for connector in cpu_connectors.read().await.values() {
                    let conn_stats = connector.get_stats().await.unwrap_or_default();
                    total_jobs += conn_stats.jobs_submitted;
                    total_bytes += conn_stats.bytes_sent;
                }
                
                stats.total_jobs_processed.store(total_jobs, Ordering::Relaxed);
                stats.total_bytes_transferred.store(total_bytes, Ordering::Relaxed);
                
                tokio::time::sleep(Duration::from_secs(1)).await;
            }
        }));
    }
}

/// Integration statistics
#[derive(Default)]
pub struct IntegrationStats {
    pub gpu_agents_registered: AtomicU64,
    pub cpu_agents_registered: AtomicU64,
    pub total_jobs_processed: AtomicU64,
    pub total_bytes_transferred: AtomicU64,
    pub isolation_violations: AtomicU64,
}

/// Health status of agents
#[derive(Default)]
pub struct HealthStatus {
    pub gpu_agents: HashMap<usize, AgentHealth>,
    pub cpu_agents: HashMap<usize, AgentHealth>,
}

/// Individual agent health
#[derive(Clone)]
pub struct AgentHealth {
    pub is_healthy: bool,
    pub last_heartbeat: Instant,
    pub error_count: u32,
    pub recovered_from_crash: bool,
}

/// Load statistics
#[derive(Default)]
pub struct LoadStatistics {
    pub gpu_loads: HashMap<usize, GpuLoadInfo>,
    pub cpu_loads: HashMap<usize, CpuLoadInfo>,
}

/// GPU load information
#[derive(Clone)]
pub struct GpuLoadInfo {
    pub jobs_processed: u64,
    pub compute_utilization: f32,
    pub memory_utilization: f32,
    pub queue_depth: usize,
}

/// CPU load information
#[derive(Clone)]
pub struct CpuLoadInfo {
    pub jobs_submitted: u64,
    pub cpu_utilization: f32,
    pub io_operations: u64,
    pub queue_depth: usize,
}

/// Stream configuration
#[derive(Clone)]
pub struct StreamConfig {
    pub chunk_size: usize,
    pub window_size: usize,
    pub processing_interval: Duration,
}

/// Real-time configuration
pub struct RealtimeConfig {
    pub target_fps: u32,
    pub max_latency_ms: u32,
    pub frame_buffer_size: usize,
}

/// Load balancing configuration
pub struct LoadBalancingConfig {
    pub strategy: LoadBalancingStrategy,
    pub rebalance_interval: Duration,
    pub load_threshold: f32,
}

/// Load balancing strategy
#[derive(Clone, Copy)]
pub enum LoadBalancingStrategy {
    RoundRobin,
    LeastLoaded,
    WeightedRandom,
    Adaptive,
}

/// Fault tolerance configuration
pub struct FaultToleranceConfig {
    pub max_retries: u32,
    pub retry_delay: Duration,
    pub heartbeat_interval: Duration,
    pub agent_timeout: Duration,
}

/// Load balancer
pub struct LoadBalancer {
    config: LoadBalancingConfig,
    current_gpu: AtomicU64,
}

impl LoadBalancer {
    fn new(config: LoadBalancingConfig) -> Self {
        Self {
            config,
            current_gpu: AtomicU64::new(0),
        }
    }
}

/// Fault tolerance manager
pub struct FaultToleranceManager {
    config: FaultToleranceConfig,
}

impl FaultToleranceManager {
    fn new(config: FaultToleranceConfig) -> Self {
        Self { config }
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod integration_tests;

#[cfg(test)]
mod e2e_tests;

#[cfg(test)]
mod workflow_test;