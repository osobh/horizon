//! SLAI Bridge for Horizon Integration
//!
//! Provides GPU detection, fair-share scheduling, and job management
//! via the SLAI embedded runtime. Uses conditional compilation to
//! support both real SLAI and mock implementations.
//!
//! # Features
//!
//! - Multi-vendor GPU detection (Apple Metal, NVIDIA CUDA, AMD ROCm)
//! - Fair-share scheduler with tenant management
//! - Job queue with priority levels
//! - Real-time scheduler statistics

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

// Re-export types for Tauri commands
#[cfg(feature = "embedded-slai")]
use slai::{
    EmbeddedSlai, FairShareInfo as SlaiFairShareInfo, SchedulerStats as SlaiSchedulerStats,
    GpuInfo as SlaiGpuInfo, Job, JobPriority, Tenant,
};

/// GPU information for frontend.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GpuInfo {
    /// GPU index
    pub index: u32,
    /// GPU name/model
    pub name: String,
    /// Vendor (nvidia, amd, apple, intel)
    pub vendor: String,
    /// Total memory in bytes
    pub memory_total: u64,
    /// Available memory in bytes
    pub memory_available: u64,
    /// Number of compute units
    pub compute_units: u32,
    /// Is this the primary GPU
    pub is_primary: bool,
}

/// Scheduler statistics for frontend.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SchedulerStats {
    /// Total number of GPUs
    pub total_gpus: usize,
    /// Available (unallocated) GPUs
    pub available_gpus: usize,
    /// Jobs waiting in queue
    pub queued_jobs: usize,
    /// Currently running jobs
    pub running_jobs: usize,
    /// Completed jobs in history
    pub completed_jobs: usize,
    /// Registered tenants
    pub tenant_count: usize,
}

/// Fair-share allocation info for a tenant.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FairShareInfo {
    /// Tenant name
    pub tenant_name: String,
    /// Priority weight (1-1000)
    pub priority_weight: u32,
    /// Maximum GPUs allowed
    pub max_gpus: u32,
    /// Currently allocated GPUs
    pub current_gpus: u32,
    /// Number of queued jobs
    pub queued_jobs: usize,
    /// Number of running jobs
    pub running_jobs: usize,
}

/// Tenant information for frontend.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TenantInfo {
    /// Tenant ID
    pub id: String,
    /// Tenant name
    pub name: String,
    /// Priority weight
    pub priority_weight: u32,
    /// Maximum GPUs allowed
    pub max_gpus: u32,
    /// Maximum concurrent jobs
    pub max_concurrent_jobs: u32,
    /// Currently allocated GPUs
    pub current_gpus: u32,
    /// Status (active, suspended)
    pub status: String,
}

/// Job information for frontend.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct JobInfo {
    /// Job ID
    pub id: String,
    /// Job name
    pub name: String,
    /// Tenant ID
    pub tenant_id: String,
    /// Job status
    pub status: String,
    /// Priority level
    pub priority: String,
    /// GPUs requested
    pub gpus_requested: u32,
    /// Submitted timestamp (unix epoch)
    pub submitted_at: u64,
    /// Started timestamp (if running)
    pub started_at: Option<u64>,
    /// Assigned GPU indices (if running)
    pub assigned_gpus: Vec<u32>,
}

/// Job list with queued and running jobs.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct JobList {
    /// Jobs waiting in queue
    pub queued: Vec<JobInfo>,
    /// Currently running jobs
    pub running: Vec<JobInfo>,
    /// Recently completed jobs
    pub completed: Vec<JobInfo>,
}

/// Job submission request.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct JobRequest {
    /// Job name
    pub name: String,
    /// Tenant ID
    pub tenant_id: String,
    /// Number of GPUs required
    pub gpus: u32,
    /// Priority level (low, normal, high, critical)
    pub priority: String,
}

/// SLAI Bridge for Horizon integration.
pub struct SlaiBridge {
    #[cfg(feature = "embedded-slai")]
    slai: Arc<RwLock<Option<EmbeddedSlai>>>,
    #[cfg(not(feature = "embedded-slai"))]
    mock_state: Arc<RwLock<MockSlaiState>>,
}

#[cfg(not(feature = "embedded-slai"))]
struct MockSlaiState {
    gpus: Vec<GpuInfo>,
    tenants: Vec<TenantInfo>,
    queued_jobs: Vec<JobInfo>,
    running_jobs: Vec<JobInfo>,
    completed_jobs: Vec<JobInfo>,
    next_job_id: u64,
}

#[cfg(not(feature = "embedded-slai"))]
impl Default for MockSlaiState {
    fn default() -> Self {
        Self {
            gpus: vec![
                GpuInfo {
                    index: 0,
                    name: "Apple M3 Max".to_string(),
                    vendor: "apple".to_string(),
                    memory_total: 48 * 1024 * 1024 * 1024,
                    memory_available: 40 * 1024 * 1024 * 1024,
                    compute_units: 40,
                    is_primary: true,
                },
            ],
            tenants: vec![
                TenantInfo {
                    id: "ml-team".to_string(),
                    name: "ML Research Team".to_string(),
                    priority_weight: 100,
                    max_gpus: 8,
                    max_concurrent_jobs: 10,
                    current_gpus: 0,
                    status: "active".to_string(),
                },
                TenantInfo {
                    id: "inference".to_string(),
                    name: "Inference Production".to_string(),
                    priority_weight: 200,
                    max_gpus: 4,
                    max_concurrent_jobs: 20,
                    current_gpus: 0,
                    status: "active".to_string(),
                },
            ],
            queued_jobs: Vec::new(),
            running_jobs: Vec::new(),
            completed_jobs: Vec::new(),
            next_job_id: 1,
        }
    }
}

impl SlaiBridge {
    /// Create a new SLAI bridge.
    pub fn new() -> Self {
        #[cfg(feature = "embedded-slai")]
        {
            Self {
                slai: Arc::new(RwLock::new(None)),
            }
        }
        #[cfg(not(feature = "embedded-slai"))]
        {
            Self {
                mock_state: Arc::new(RwLock::new(MockSlaiState::default())),
            }
        }
    }

    /// Initialize the SLAI runtime.
    pub async fn initialize(&self) -> Result<(), String> {
        #[cfg(feature = "embedded-slai")]
        {
            tracing::info!("Initializing SLAI embedded runtime...");
            match EmbeddedSlai::new() {
                Ok(slai) => {
                    // Detect GPUs on initialization
                    if let Err(e) = slai.detect_and_configure().await {
                        tracing::warn!("GPU detection failed: {}", e);
                    }
                    *self.slai.write().await = Some(slai);
                    tracing::info!("SLAI embedded runtime initialized");
                    Ok(())
                }
                Err(e) => {
                    tracing::error!("Failed to initialize SLAI: {}", e);
                    Err(format!("SLAI initialization failed: {}", e))
                }
            }
        }
        #[cfg(not(feature = "embedded-slai"))]
        {
            tracing::info!("SLAI bridge initialized (mock mode)");
            Ok(())
        }
    }

    /// Detect and return available GPUs.
    pub async fn detect_gpus(&self) -> Vec<GpuInfo> {
        #[cfg(feature = "embedded-slai")]
        {
            let guard = self.slai.read().await;
            if let Some(ref slai) = *guard {
                match slai.detect_gpus() {
                    Ok(gpus) => {
                        return gpus.into_iter().map(|g| GpuInfo {
                            index: g.index as u32,
                            name: g.name,
                            vendor: format!("{:?}", g.vendor).to_lowercase(),
                            memory_total: g.memory_total,
                            memory_available: g.memory_available,
                            compute_units: g.compute_units,
                            is_primary: g.is_primary,
                        }).collect();
                    }
                    Err(e) => {
                        tracing::error!("GPU detection failed: {}", e);
                    }
                }
            }
            Vec::new()
        }
        #[cfg(not(feature = "embedded-slai"))]
        {
            self.mock_state.read().await.gpus.clone()
        }
    }

    /// Get scheduler statistics.
    pub async fn get_scheduler_stats(&self) -> SchedulerStats {
        #[cfg(feature = "embedded-slai")]
        {
            let guard = self.slai.read().await;
            if let Some(ref slai) = *guard {
                let stats = slai.get_stats().await;
                return SchedulerStats {
                    total_gpus: stats.total_gpus,
                    available_gpus: stats.available_gpus,
                    queued_jobs: stats.queued_jobs,
                    running_jobs: stats.running_jobs,
                    completed_jobs: stats.completed_jobs,
                    tenant_count: stats.tenant_count,
                };
            }
            SchedulerStats::default()
        }
        #[cfg(not(feature = "embedded-slai"))]
        {
            let state = self.mock_state.read().await;
            SchedulerStats {
                total_gpus: state.gpus.len(),
                available_gpus: state.gpus.len().saturating_sub(state.running_jobs.len()),
                queued_jobs: state.queued_jobs.len(),
                running_jobs: state.running_jobs.len(),
                completed_jobs: state.completed_jobs.len(),
                tenant_count: state.tenants.len(),
            }
        }
    }

    /// Get fair-share allocation info per tenant.
    pub async fn get_fair_share(&self) -> HashMap<String, FairShareInfo> {
        #[cfg(feature = "embedded-slai")]
        {
            let guard = self.slai.read().await;
            if let Some(ref slai) = *guard {
                let fair_share = slai.get_fair_share().await;
                return fair_share.into_iter().map(|(k, v)| {
                    (k, FairShareInfo {
                        tenant_name: v.tenant_name,
                        priority_weight: v.priority_weight,
                        max_gpus: v.max_gpus,
                        current_gpus: v.current_gpus,
                        queued_jobs: v.queued_jobs,
                        running_jobs: v.running_jobs,
                    })
                }).collect();
            }
            HashMap::new()
        }
        #[cfg(not(feature = "embedded-slai"))]
        {
            let state = self.mock_state.read().await;
            state.tenants.iter().map(|t| {
                let queued = state.queued_jobs.iter().filter(|j| j.tenant_id == t.id).count();
                let running = state.running_jobs.iter().filter(|j| j.tenant_id == t.id).count();
                (t.id.clone(), FairShareInfo {
                    tenant_name: t.name.clone(),
                    priority_weight: t.priority_weight,
                    max_gpus: t.max_gpus,
                    current_gpus: running as u32,
                    queued_jobs: queued,
                    running_jobs: running,
                })
            }).collect()
        }
    }

    /// List all registered tenants.
    pub async fn list_tenants(&self) -> Vec<TenantInfo> {
        #[cfg(feature = "embedded-slai")]
        {
            let guard = self.slai.read().await;
            if let Some(ref slai) = *guard {
                let tenants = slai.list_tenants().await;
                return tenants.into_iter().map(|t| TenantInfo {
                    id: t.id.clone(),
                    name: t.name.clone(),
                    priority_weight: t.priority_weight,
                    max_gpus: t.quota.max_gpus,
                    max_concurrent_jobs: t.quota.max_concurrent_jobs,
                    current_gpus: 0, // Updated via fair_share
                    status: format!("{:?}", t.status).to_lowercase(),
                }).collect();
            }
            Vec::new()
        }
        #[cfg(not(feature = "embedded-slai"))]
        {
            self.mock_state.read().await.tenants.clone()
        }
    }

    /// Create a new tenant.
    pub async fn create_tenant(&self, name: String, max_gpus: u32, max_concurrent_jobs: u32) -> Result<TenantInfo, String> {
        #[cfg(feature = "embedded-slai")]
        {
            let guard = self.slai.read().await;
            if let Some(ref slai) = *guard {
                match slai.create_tenant(&name, max_gpus, max_concurrent_jobs).await {
                    Ok(tenant) => {
                        return Ok(TenantInfo {
                            id: tenant.id.clone(),
                            name: tenant.name.clone(),
                            priority_weight: tenant.priority_weight,
                            max_gpus: tenant.quota.max_gpus,
                            max_concurrent_jobs: tenant.quota.max_concurrent_jobs,
                            current_gpus: 0,
                            status: format!("{:?}", tenant.status).to_lowercase(),
                        });
                    }
                    Err(e) => return Err(format!("Failed to create tenant: {}", e)),
                }
            }
            Err("SLAI not initialized".to_string())
        }
        #[cfg(not(feature = "embedded-slai"))]
        {
            let mut state = self.mock_state.write().await;
            let id = name.to_lowercase().replace(' ', "-");
            let tenant = TenantInfo {
                id: id.clone(),
                name: name.clone(),
                priority_weight: 100,
                max_gpus,
                max_concurrent_jobs,
                current_gpus: 0,
                status: "active".to_string(),
            };
            state.tenants.push(tenant.clone());
            Ok(tenant)
        }
    }

    /// List all jobs (queued, running, completed).
    pub async fn list_jobs(&self) -> JobList {
        #[cfg(feature = "embedded-slai")]
        {
            let guard = self.slai.read().await;
            if let Some(ref slai) = *guard {
                let queued = slai.list_queued_jobs().await.into_iter().map(|j| self.job_to_info(&j, None)).collect();
                let running = slai.list_running_jobs().await.into_iter().map(|r| {
                    self.job_to_info(&r.job, Some(r.assigned_gpus.clone()))
                }).collect();
                let completed = slai.list_completed_jobs().await.into_iter().take(20).map(|j| self.job_to_info(&j, None)).collect();
                return JobList { queued, running, completed };
            }
            JobList::default()
        }
        #[cfg(not(feature = "embedded-slai"))]
        {
            let state = self.mock_state.read().await;
            JobList {
                queued: state.queued_jobs.clone(),
                running: state.running_jobs.clone(),
                completed: state.completed_jobs.iter().take(20).cloned().collect(),
            }
        }
    }

    #[cfg(feature = "embedded-slai")]
    fn job_to_info(&self, job: &Job, assigned_gpus: Option<Vec<u32>>) -> JobInfo {
        JobInfo {
            id: job.id.clone(),
            name: job.name.clone(),
            tenant_id: job.tenant_id.clone(),
            status: format!("{:?}", job.status).to_lowercase(),
            priority: format!("{:?}", job.priority).to_lowercase(),
            gpus_requested: job.resources.gpus,
            submitted_at: job.created_at,
            started_at: job.started_at,
            assigned_gpus: assigned_gpus.unwrap_or_else(|| job.assigned_gpus.clone()),
        }
    }

    /// Submit a new job.
    pub async fn submit_job(&self, request: JobRequest) -> Result<String, String> {
        #[cfg(feature = "embedded-slai")]
        {
            let guard = self.slai.read().await;
            if let Some(ref slai) = *guard {
                let priority = match request.priority.as_str() {
                    "low" => JobPriority::Low,
                    "high" => JobPriority::High,
                    "critical" => JobPriority::Critical,
                    _ => JobPriority::Normal,
                };
                match slai.submit_simple_job(&request.name, &request.tenant_id, request.gpus, priority).await {
                    Ok(job_id) => {
                        // Try to schedule immediately
                        let _ = slai.schedule_next().await;
                        return Ok(job_id);
                    }
                    Err(e) => return Err(format!("Job submission failed: {}", e)),
                }
            }
            Err("SLAI not initialized".to_string())
        }
        #[cfg(not(feature = "embedded-slai"))]
        {
            let mut state = self.mock_state.write().await;
            let job_id = format!("JOB-{:04}", state.next_job_id);
            state.next_job_id += 1;

            let job = JobInfo {
                id: job_id.clone(),
                name: request.name,
                tenant_id: request.tenant_id,
                status: "queued".to_string(),
                priority: request.priority,
                gpus_requested: request.gpus,
                submitted_at: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                started_at: None,
                assigned_gpus: Vec::new(),
            };
            state.queued_jobs.push(job);
            Ok(job_id)
        }
    }

    /// Cancel a job by ID.
    pub async fn cancel_job(&self, job_id: &str) -> Result<(), String> {
        #[cfg(feature = "embedded-slai")]
        {
            let guard = self.slai.read().await;
            if let Some(ref slai) = *guard {
                match slai.cancel_job(job_id).await {
                    Ok(_) => return Ok(()),
                    Err(e) => return Err(format!("Failed to cancel job: {}", e)),
                }
            }
            Err("SLAI not initialized".to_string())
        }
        #[cfg(not(feature = "embedded-slai"))]
        {
            let mut state = self.mock_state.write().await;
            // Remove from queued
            if let Some(pos) = state.queued_jobs.iter().position(|j| j.id == job_id) {
                let mut job = state.queued_jobs.remove(pos);
                job.status = "cancelled".to_string();
                state.completed_jobs.insert(0, job);
                return Ok(());
            }
            // Remove from running
            if let Some(pos) = state.running_jobs.iter().position(|j| j.id == job_id) {
                let mut job = state.running_jobs.remove(pos);
                job.status = "cancelled".to_string();
                state.completed_jobs.insert(0, job);
                return Ok(());
            }
            Err(format!("Job {} not found", job_id))
        }
    }

    /// Schedule next pending job (for demo/testing).
    pub async fn schedule_next(&self) -> Option<JobInfo> {
        #[cfg(feature = "embedded-slai")]
        {
            let guard = self.slai.read().await;
            if let Some(ref slai) = *guard {
                if let Some(running) = slai.schedule_next().await {
                    return Some(self.job_to_info(&running.job, Some(running.assigned_gpus)));
                }
            }
            None
        }
        #[cfg(not(feature = "embedded-slai"))]
        {
            let mut state = self.mock_state.write().await;
            if let Some(mut job) = state.queued_jobs.pop() {
                job.status = "running".to_string();
                job.started_at = Some(
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_secs()
                );
                job.assigned_gpus = vec![0]; // Mock: assign GPU 0
                state.running_jobs.push(job.clone());
                return Some(job);
            }
            None
        }
    }
}

impl Default for SlaiBridge {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for SchedulerStats {
    fn default() -> Self {
        Self {
            total_gpus: 0,
            available_gpus: 0,
            queued_jobs: 0,
            running_jobs: 0,
            completed_jobs: 0,
            tenant_count: 0,
        }
    }
}

impl Default for JobList {
    fn default() -> Self {
        Self {
            queued: Vec::new(),
            running: Vec::new(),
            completed: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_bridge_creation() {
        let bridge = SlaiBridge::new();
        let _ = bridge.initialize().await;
        let stats = bridge.get_scheduler_stats().await;
        assert!(stats.total_gpus >= 0);
    }

    #[tokio::test]
    async fn test_list_tenants() {
        let bridge = SlaiBridge::new();
        let _ = bridge.initialize().await;
        let tenants = bridge.list_tenants().await;
        // Mock mode has default tenants
        #[cfg(not(feature = "embedded-slai"))]
        assert!(!tenants.is_empty());
    }

    #[tokio::test]
    async fn test_submit_job_mock() {
        let bridge = SlaiBridge::new();
        let _ = bridge.initialize().await;

        #[cfg(not(feature = "embedded-slai"))]
        {
            let job_id = bridge.submit_job(JobRequest {
                name: "test-job".to_string(),
                tenant_id: "ml-team".to_string(),
                gpus: 1,
                priority: "normal".to_string(),
            }).await.expect("Should submit");

            assert!(job_id.starts_with("JOB-"));

            let jobs = bridge.list_jobs().await;
            assert_eq!(jobs.queued.len(), 1);
        }
    }
}
