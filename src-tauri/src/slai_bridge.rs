//! SLAI Bridge
//!
//! Integrates SLAI GPU scheduling with Horizon using hpc-channels.
//!
//! ## Feature Flags
//!
//! - `embedded-slai`: When enabled, uses the real SLAI library for GPU detection
//!   and multi-tenant scheduling. When disabled, provides mock data for development.
//!
//! ## Architecture
//!
//! The SLAI bridge wraps either:
//! - Real `GpuDetector` and `FairShareScheduler` from SLAI (with embedded-slai)
//! - Mock GPU and job data with simulated behavior (without embedded-slai)
//!
//! Both implementations expose the same API to the Tauri commands.

use dashmap::DashMap;
use hpc_channels::channels;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;

// Import real SLAI types when feature is enabled
#[cfg(feature = "embedded-slai")]
use slai::{
    gpu::{GpuDetector, GpuInfo as SlaiGpuInfo},
    tenant::{FairShareScheduler, Job as SlaiJob, JobPriority as SlaiJobPriority, Tenant as SlaiTenant},
};

/// Bridge to the SLAI scheduling system.
pub struct SlaiBridge {
    /// GPU information cache
    gpus: Arc<RwLock<Vec<GpuInfo>>>,
    /// Active jobs (lock-free concurrent access)
    jobs: Arc<DashMap<String, SchedulerJob>>,
    /// Tenants (lock-free concurrent access)
    tenants: Arc<DashMap<String, Tenant>>,
    /// Job counter for generating IDs
    job_counter: AtomicU64,
    /// Tenant counter
    tenant_counter: AtomicU64,
    /// Real SLAI scheduler (when embedded-slai feature is enabled)
    #[cfg(feature = "embedded-slai")]
    scheduler: Arc<RwLock<Option<FairShareScheduler>>>,
    #[cfg(feature = "embedded-slai")]
    detector: Arc<RwLock<Option<GpuDetector>>>,
}

/// GPU information for the frontend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInfo {
    pub index: u32,
    pub name: String,
    pub vendor: String,
    pub memory_total_mb: u64,
    pub memory_used_mb: u64,
    pub memory_utilization: f64,
    pub compute_utilization: f64,
    pub temperature_c: Option<f64>,
    pub power_watts: Option<f64>,
    pub status: GpuStatus,
}

/// GPU status.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum GpuStatus {
    Available,
    Busy,
    Reserved,
    Offline,
    Error,
}

/// A scheduler job.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerJob {
    pub id: String,
    pub name: String,
    pub tenant_id: String,
    pub status: JobStatus,
    pub priority: JobPriority,
    pub gpus_requested: u32,
    pub assigned_gpus: Vec<u32>,
    pub created_at: u64,
    pub started_at: Option<u64>,
    pub completed_at: Option<u64>,
    pub estimated_duration_secs: Option<u64>,
    pub progress: Option<f64>,
    pub error_message: Option<String>,
}

/// Job status.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum JobStatus {
    Pending,
    Queued,
    Running,
    Completed,
    Failed,
    Cancelled,
}

/// Job priority.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum JobPriority {
    Critical,
    High,
    Normal,
    Low,
}

impl Default for JobPriority {
    fn default() -> Self {
        JobPriority::Normal
    }
}

/// A tenant (user/team) in the scheduler.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tenant {
    pub id: String,
    pub name: String,
    pub status: TenantStatus,
    pub priority_weight: f64,
    pub max_gpus: u32,
    pub max_concurrent_jobs: u32,
    pub current_gpu_usage: u32,
    pub current_job_count: u32,
    pub fair_share: f64,
    pub created_at: u64,
}

/// Tenant status.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum TenantStatus {
    Active,
    Suspended,
}

/// Job submission request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubmitJobRequest {
    pub name: String,
    pub tenant_id: String,
    pub gpus_requested: u32,
    pub priority: Option<JobPriority>,
    pub estimated_duration_secs: Option<u64>,
}

/// Tenant creation request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateTenantRequest {
    pub name: String,
    pub priority_weight: Option<f64>,
    pub max_gpus: Option<u32>,
    pub max_concurrent_jobs: Option<u32>,
}

/// Scheduler summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerSummary {
    pub total_gpus: u32,
    pub available_gpus: u32,
    pub busy_gpus: u32,
    pub total_jobs: usize,
    pub running_jobs: usize,
    pub queued_jobs: usize,
    pub completed_jobs: usize,
    pub failed_jobs: usize,
    pub total_tenants: usize,
    pub active_tenants: usize,
}

/// Fair share allocation per tenant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FairShareAllocation {
    pub tenant_id: String,
    pub tenant_name: String,
    pub allocated_share: f64,
    pub current_usage: f64,
    pub gpu_count: u32,
}

impl SlaiBridge {
    /// Create a new SLAI bridge.
    pub fn new() -> Self {
        let bridge = Self {
            gpus: Arc::new(RwLock::new(Vec::new())),
            jobs: Arc::new(DashMap::new()),
            tenants: Arc::new(DashMap::new()),
            job_counter: AtomicU64::new(1),
            tenant_counter: AtomicU64::new(1),
            #[cfg(feature = "embedded-slai")]
            scheduler: Arc::new(RwLock::new(None)),
            #[cfg(feature = "embedded-slai")]
            detector: Arc::new(RwLock::new(None)),
        };

        // Initialize with mock data for demo
        tokio::spawn({
            let bridge_gpus = Arc::clone(&bridge.gpus);
            let bridge_jobs = Arc::clone(&bridge.jobs);
            let bridge_tenants = Arc::clone(&bridge.tenants);
            async move {
                Self::initialize_mock_data(bridge_gpus, bridge_jobs, bridge_tenants).await;
            }
        });

        bridge
    }

    /// Initialize mock data for demonstration.
    async fn initialize_mock_data(
        gpus: Arc<RwLock<Vec<GpuInfo>>>,
        jobs: Arc<DashMap<String, SchedulerJob>>,
        tenants: Arc<DashMap<String, Tenant>>,
    ) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Mock GPUs
        let mock_gpus = vec![
            GpuInfo {
                index: 0,
                name: "NVIDIA A100-SXM4-80GB".to_string(),
                vendor: "NVIDIA".to_string(),
                memory_total_mb: 81920,
                memory_used_mb: 45000,
                memory_utilization: 0.55,
                compute_utilization: 0.78,
                temperature_c: Some(62.0),
                power_watts: Some(320.0),
                status: GpuStatus::Busy,
            },
            GpuInfo {
                index: 1,
                name: "NVIDIA A100-SXM4-80GB".to_string(),
                vendor: "NVIDIA".to_string(),
                memory_total_mb: 81920,
                memory_used_mb: 12000,
                memory_utilization: 0.15,
                compute_utilization: 0.0,
                temperature_c: Some(38.0),
                power_watts: Some(65.0),
                status: GpuStatus::Available,
            },
            GpuInfo {
                index: 2,
                name: "NVIDIA A100-SXM4-80GB".to_string(),
                vendor: "NVIDIA".to_string(),
                memory_total_mb: 81920,
                memory_used_mb: 78000,
                memory_utilization: 0.95,
                compute_utilization: 0.92,
                temperature_c: Some(71.0),
                power_watts: Some(385.0),
                status: GpuStatus::Busy,
            },
            GpuInfo {
                index: 3,
                name: "NVIDIA A100-SXM4-80GB".to_string(),
                vendor: "NVIDIA".to_string(),
                memory_total_mb: 81920,
                memory_used_mb: 0,
                memory_utilization: 0.0,
                compute_utilization: 0.0,
                temperature_c: Some(35.0),
                power_watts: Some(55.0),
                status: GpuStatus::Available,
            },
        ];
        *gpus.write().await = mock_gpus;

        // Mock tenants
        let mock_tenants = vec![
            Tenant {
                id: "tenant-research".to_string(),
                name: "ML Research Team".to_string(),
                status: TenantStatus::Active,
                priority_weight: 1.0,
                max_gpus: 8,
                max_concurrent_jobs: 10,
                current_gpu_usage: 2,
                current_job_count: 2,
                fair_share: 0.5,
                created_at: now - 86400 * 30,
            },
            Tenant {
                id: "tenant-production".to_string(),
                name: "Production Inference".to_string(),
                status: TenantStatus::Active,
                priority_weight: 2.0,
                max_gpus: 4,
                max_concurrent_jobs: 5,
                current_gpu_usage: 0,
                current_job_count: 1,
                fair_share: 0.35,
                created_at: now - 86400 * 60,
            },
            Tenant {
                id: "tenant-student".to_string(),
                name: "Student Projects".to_string(),
                status: TenantStatus::Active,
                priority_weight: 0.5,
                max_gpus: 2,
                max_concurrent_jobs: 3,
                current_gpu_usage: 0,
                current_job_count: 0,
                fair_share: 0.15,
                created_at: now - 86400 * 7,
            },
        ];
        for tenant in mock_tenants {
            tenants.insert(tenant.id.clone(), tenant);
        }

        // Mock jobs
        let mock_jobs = vec![
            SchedulerJob {
                id: "job-001".to_string(),
                name: "LLaMA-70B Fine-tuning".to_string(),
                tenant_id: "tenant-research".to_string(),
                status: JobStatus::Running,
                priority: JobPriority::High,
                gpus_requested: 2,
                assigned_gpus: vec![0, 2],
                created_at: now - 3600,
                started_at: Some(now - 3000),
                completed_at: None,
                estimated_duration_secs: Some(7200),
                progress: Some(0.42),
                error_message: None,
            },
            SchedulerJob {
                id: "job-002".to_string(),
                name: "Vision Transformer Training".to_string(),
                tenant_id: "tenant-research".to_string(),
                status: JobStatus::Queued,
                priority: JobPriority::Normal,
                gpus_requested: 4,
                assigned_gpus: vec![],
                created_at: now - 1800,
                started_at: None,
                completed_at: None,
                estimated_duration_secs: Some(10800),
                progress: None,
                error_message: None,
            },
            SchedulerJob {
                id: "job-003".to_string(),
                name: "Batch Inference Pipeline".to_string(),
                tenant_id: "tenant-production".to_string(),
                status: JobStatus::Queued,
                priority: JobPriority::Critical,
                gpus_requested: 1,
                assigned_gpus: vec![],
                created_at: now - 600,
                started_at: None,
                completed_at: None,
                estimated_duration_secs: Some(1800),
                progress: None,
                error_message: None,
            },
            SchedulerJob {
                id: "job-004".to_string(),
                name: "ResNet Hyperparameter Sweep".to_string(),
                tenant_id: "tenant-student".to_string(),
                status: JobStatus::Completed,
                priority: JobPriority::Low,
                gpus_requested: 1,
                assigned_gpus: vec![],
                created_at: now - 86400,
                started_at: Some(now - 82800),
                completed_at: Some(now - 79200),
                estimated_duration_secs: Some(3600),
                progress: Some(1.0),
                error_message: None,
            },
        ];
        for job in mock_jobs {
            jobs.insert(job.id.clone(), job);
        }
    }

    /// Initialize the SLAI scheduler (when using real implementation).
    #[cfg(feature = "embedded-slai")]
    pub async fn initialize(&self) -> Result<(), String> {
        // Initialize GPU detector
        let detector = GpuDetector::new();
        *self.detector.write().await = Some(detector);

        // Get available GPUs
        let gpu_indices: Vec<u32> = self.detect_gpus().await?
            .iter()
            .filter(|g| g.status == GpuStatus::Available)
            .map(|g| g.index)
            .collect();

        // Initialize scheduler with available GPUs
        let scheduler = FairShareScheduler::new()
            .with_available_gpus(gpu_indices);
        *self.scheduler.write().await = Some(scheduler);

        Ok(())
    }

    #[cfg(not(feature = "embedded-slai"))]
    pub async fn initialize(&self) -> Result<(), String> {
        // Mock initialization - already done in new()
        Ok(())
    }

    /// Detect available GPUs.
    #[cfg(feature = "embedded-slai")]
    pub async fn detect_gpus(&self) -> Result<Vec<GpuInfo>, String> {
        let detector = self.detector.read().await;
        let detector = detector.as_ref().ok_or("Detector not initialized")?;

        let slai_gpus = detector.detect_all()
            .map_err(|e| format!("GPU detection failed: {}", e))?;

        let gpus: Vec<GpuInfo> = slai_gpus.into_iter().map(|g| {
            let memory_used = g.memory_total.saturating_sub(g.memory_available);
            GpuInfo {
                index: g.index as u32,
                name: g.name,
                vendor: format!("{:?}", g.vendor),
                memory_total_mb: g.memory_total / (1024 * 1024),
                memory_used_mb: memory_used / (1024 * 1024),
                memory_utilization: if g.memory_total > 0 {
                    memory_used as f64 / g.memory_total as f64
                } else {
                    0.0
                },
                compute_utilization: 0.0, // Not available from GpuInfo
                temperature_c: g.temperature_celsius.map(|t| t as f64),
                power_watts: g.power_watts.map(|p| p as f64),
                status: GpuStatus::Available,
            }
        }).collect();

        *self.gpus.write().await = gpus.clone();
        Ok(gpus)
    }

    #[cfg(not(feature = "embedded-slai"))]
    pub async fn detect_gpus(&self) -> Result<Vec<GpuInfo>, String> {
        // Return cached mock GPUs
        Ok(self.gpus.read().await.clone())
    }

    /// Get all GPUs.
    pub async fn get_gpus(&self) -> Vec<GpuInfo> {
        self.gpus.read().await.clone()
    }

    /// Submit a job.
    #[cfg(feature = "embedded-slai")]
    pub async fn submit_job(&self, request: SubmitJobRequest) -> Result<SchedulerJob, String> {
        let priority = match request.priority.clone().unwrap_or_default() {
            JobPriority::Critical => SlaiJobPriority::Critical,
            JobPriority::High => SlaiJobPriority::High,
            JobPriority::Normal => SlaiJobPriority::Normal,
            JobPriority::Low => SlaiJobPriority::Low,
        };

        let slai_job = SlaiJob::new(&request.name)
            .with_tenant(&request.tenant_id)
            .with_gpus(request.gpus_requested)
            .with_priority(priority);

        let mut scheduler_guard = self.scheduler.write().await;
        let scheduler = scheduler_guard.as_mut().ok_or("Scheduler not initialized")?;
        let job_id = scheduler.submit(slai_job)
            .map_err(|e| format!("Job submission failed: {}", e))?;

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let job = SchedulerJob {
            id: job_id.clone(),
            name: request.name,
            tenant_id: request.tenant_id,
            status: JobStatus::Queued,
            priority: request.priority.unwrap_or_default(),
            gpus_requested: request.gpus_requested,
            assigned_gpus: vec![],
            created_at: now,
            started_at: None,
            completed_at: None,
            estimated_duration_secs: request.estimated_duration_secs,
            progress: None,
            error_message: None,
        };

        self.jobs.insert(job_id, job.clone());

        // Publish to hpc-channels
        if let Some(tx) = hpc_channels::sender::<hpc_channels::SchedulerMessage>(channels::SCHEDULER_JOBS) {
            let _ = tx.send(hpc_channels::SchedulerMessage::JobSubmitted {
                job_id: job.id.clone(),
                tenant_id: job.tenant_id.clone(),
            }).await;
        }

        Ok(job)
    }

    #[cfg(not(feature = "embedded-slai"))]
    pub async fn submit_job(&self, request: SubmitJobRequest) -> Result<SchedulerJob, String> {
        // Relaxed: independent job ID counter
        let job_num = self.job_counter.fetch_add(1, Ordering::Relaxed);
        let job_id = format!("job-{:03}", job_num);

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let job = SchedulerJob {
            id: job_id.clone(),
            name: request.name,
            tenant_id: request.tenant_id,
            status: JobStatus::Queued,
            priority: request.priority.unwrap_or_default(),
            gpus_requested: request.gpus_requested,
            assigned_gpus: vec![],
            created_at: now,
            started_at: None,
            completed_at: None,
            estimated_duration_secs: request.estimated_duration_secs,
            progress: None,
            error_message: None,
        };

        self.jobs.insert(job_id, job.clone());

        // Publish to hpc-channels
        if let Some(tx) = hpc_channels::sender::<hpc_channels::SchedulerMessage>(channels::SCHEDULER_JOBS) {
            let _ = tx.send(hpc_channels::SchedulerMessage::JobSubmitted {
                job_id: job.id.clone(),
                tenant_id: job.tenant_id.clone(),
            }).await;
        }

        Ok(job)
    }

    /// Get a job by ID.
    pub async fn get_job(&self, job_id: &str) -> Option<SchedulerJob> {
        self.jobs.get(job_id).map(|r| r.clone())
    }

    /// List all jobs.
    pub async fn list_jobs(&self) -> Vec<SchedulerJob> {
        self.jobs.iter().map(|r| r.value().clone()).collect()
    }

    /// List jobs for a tenant.
    pub async fn list_jobs_for_tenant(&self, tenant_id: &str) -> Vec<SchedulerJob> {
        self.jobs
            .iter()
            .filter(|r| r.value().tenant_id == tenant_id)
            .map(|r| r.value().clone())
            .collect()
    }

    /// Cancel a job.
    pub async fn cancel_job(&self, job_id: &str) -> Result<SchedulerJob, String> {
        let mut job_ref = self.jobs.get_mut(job_id)
            .ok_or(format!("Job not found: {}", job_id))?;

        if job_ref.status == JobStatus::Completed || job_ref.status == JobStatus::Failed {
            return Err("Cannot cancel a finished job".to_string());
        }

        job_ref.status = JobStatus::Cancelled;
        job_ref.assigned_gpus.clear();

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        job_ref.completed_at = Some(now);

        let result = job_ref.clone();
        drop(job_ref); // Release the lock before async operation

        // Publish to hpc-channels
        if let Some(tx) = hpc_channels::sender::<hpc_channels::SchedulerMessage>(channels::SCHEDULER_JOBS) {
            let _ = tx.send(hpc_channels::SchedulerMessage::JobCancelled {
                job_id: result.id.clone(),
            }).await;
        }

        Ok(result)
    }

    /// Create a tenant.
    pub async fn create_tenant(&self, request: CreateTenantRequest) -> Result<Tenant, String> {
        // Relaxed: independent tenant ID counter
        let tenant_num = self.tenant_counter.fetch_add(1, Ordering::Relaxed);
        let tenant_id = format!("tenant-{}", tenant_num);

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let tenant = Tenant {
            id: tenant_id.clone(),
            name: request.name,
            status: TenantStatus::Active,
            priority_weight: request.priority_weight.unwrap_or(1.0),
            max_gpus: request.max_gpus.unwrap_or(4),
            max_concurrent_jobs: request.max_concurrent_jobs.unwrap_or(10),
            current_gpu_usage: 0,
            current_job_count: 0,
            fair_share: 0.0,
            created_at: now,
        };

        self.tenants.insert(tenant_id, tenant.clone());

        // Publish to hpc-channels
        if let Some(tx) = hpc_channels::sender::<hpc_channels::SchedulerMessage>(channels::SCHEDULER_TENANTS) {
            let _ = tx.send(hpc_channels::SchedulerMessage::TenantUpdated {
                tenant_id: tenant.id.clone(),
                status: format!("{:?}", tenant.status),
            }).await;
        }

        Ok(tenant)
    }

    /// Get a tenant by ID.
    pub async fn get_tenant(&self, tenant_id: &str) -> Option<Tenant> {
        self.tenants.get(tenant_id).map(|r| r.clone())
    }

    /// List all tenants.
    pub async fn list_tenants(&self) -> Vec<Tenant> {
        self.tenants.iter().map(|r| r.value().clone()).collect()
    }

    /// Suspend a tenant.
    pub async fn suspend_tenant(&self, tenant_id: &str) -> Result<Tenant, String> {
        let mut tenant_ref = self.tenants.get_mut(tenant_id)
            .ok_or(format!("Tenant not found: {}", tenant_id))?;
        tenant_ref.status = TenantStatus::Suspended;
        let result = tenant_ref.clone();
        drop(tenant_ref); // Release the lock before async operation

        // Publish to hpc-channels
        if let Some(tx) = hpc_channels::sender::<hpc_channels::SchedulerMessage>(channels::SCHEDULER_TENANTS) {
            let _ = tx.send(hpc_channels::SchedulerMessage::TenantUpdated {
                tenant_id: result.id.clone(),
                status: "Suspended".to_string(),
            }).await;
        }

        Ok(result)
    }

    /// Resume a tenant.
    pub async fn resume_tenant(&self, tenant_id: &str) -> Result<Tenant, String> {
        let mut tenant_ref = self.tenants.get_mut(tenant_id)
            .ok_or(format!("Tenant not found: {}", tenant_id))?;
        tenant_ref.status = TenantStatus::Active;
        let result = tenant_ref.clone();
        drop(tenant_ref); // Release the lock before async operation

        // Publish to hpc-channels
        if let Some(tx) = hpc_channels::sender::<hpc_channels::SchedulerMessage>(channels::SCHEDULER_TENANTS) {
            let _ = tx.send(hpc_channels::SchedulerMessage::TenantUpdated {
                tenant_id: result.id.clone(),
                status: "Active".to_string(),
            }).await;
        }

        Ok(result)
    }

    /// Get scheduler summary.
    pub async fn get_summary(&self) -> SchedulerSummary {
        let gpus = self.gpus.read().await;

        let total_gpus = gpus.len() as u32;
        let available_gpus = gpus.iter().filter(|g| g.status == GpuStatus::Available).count() as u32;
        let busy_gpus = gpus.iter().filter(|g| g.status == GpuStatus::Busy).count() as u32;

        let running_jobs = self.jobs.iter().filter(|r| r.value().status == JobStatus::Running).count();
        let queued_jobs = self.jobs.iter().filter(|r| r.value().status == JobStatus::Queued || r.value().status == JobStatus::Pending).count();
        let completed_jobs = self.jobs.iter().filter(|r| r.value().status == JobStatus::Completed).count();
        let failed_jobs = self.jobs.iter().filter(|r| r.value().status == JobStatus::Failed || r.value().status == JobStatus::Cancelled).count();

        let active_tenants = self.tenants.iter().filter(|r| r.value().status == TenantStatus::Active).count();

        SchedulerSummary {
            total_gpus,
            available_gpus,
            busy_gpus,
            total_jobs: self.jobs.len(),
            running_jobs,
            queued_jobs,
            completed_jobs,
            failed_jobs,
            total_tenants: self.tenants.len(),
            active_tenants,
        }
    }

    /// Get fair share allocation.
    pub async fn get_fair_share(&self) -> Vec<FairShareAllocation> {
        self.tenants.iter().map(|tenant_ref| {
            let t = tenant_ref.value();
            let gpu_count = self.jobs.iter()
                .filter(|r| r.value().tenant_id == t.id && r.value().status == JobStatus::Running)
                .map(|r| r.value().assigned_gpus.len() as u32)
                .sum();

            FairShareAllocation {
                tenant_id: t.id.clone(),
                tenant_name: t.name.clone(),
                allocated_share: t.fair_share,
                current_usage: t.current_gpu_usage as f64 / t.max_gpus.max(1) as f64,
                gpu_count,
            }
        }).collect()
    }
}

impl Default for SlaiBridge {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_bridge_creation() {
        let bridge = SlaiBridge::new();
        // Wait for mock data initialization
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        let gpus = bridge.get_gpus().await;
        assert!(!gpus.is_empty());
    }

    #[tokio::test]
    async fn test_submit_job() {
        let bridge = SlaiBridge::new();
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        let request = SubmitJobRequest {
            name: "Test Job".to_string(),
            tenant_id: "tenant-research".to_string(),
            gpus_requested: 2,
            priority: Some(JobPriority::Normal),
            estimated_duration_secs: Some(3600),
        };

        let job = bridge.submit_job(request).await.unwrap();
        assert_eq!(job.status, JobStatus::Queued);
        assert_eq!(job.gpus_requested, 2);
    }

    #[tokio::test]
    async fn test_list_tenants() {
        let bridge = SlaiBridge::new();
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        let tenants = bridge.list_tenants().await;
        assert!(!tenants.is_empty());
    }
}
