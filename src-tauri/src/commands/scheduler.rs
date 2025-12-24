//! Scheduler Commands
//!
//! Tauri commands for SLAI GPU scheduling and multi-tenant job management.

use crate::slai_bridge::{
    CreateTenantRequest, FairShareAllocation, GpuInfo, JobPriority, SchedulerJob,
    SchedulerSummary, SubmitJobRequest, Tenant,
};
use crate::state::AppState;
use serde::{Deserialize, Serialize};
use tauri::State;

/// Job submission input from frontend.
#[derive(Debug, Serialize, Deserialize)]
pub struct SubmitJobInput {
    pub name: String,
    pub tenant_id: String,
    pub gpus_requested: u32,
    #[serde(default)]
    pub priority: Option<String>,
    pub estimated_duration_secs: Option<u64>,
}

/// Tenant creation input from frontend.
#[derive(Debug, Serialize, Deserialize)]
pub struct CreateTenantInput {
    pub name: String,
    pub priority_weight: Option<f64>,
    pub max_gpus: Option<u32>,
    pub max_concurrent_jobs: Option<u32>,
}

/// Detect available GPUs (scheduler-specific).
#[tauri::command]
pub async fn scheduler_detect_gpus(state: State<'_, AppState>) -> Result<Vec<GpuInfo>, String> {
    state.slai.detect_gpus().await
}

/// Get all GPUs (cached).
#[tauri::command]
pub async fn scheduler_list_gpus(state: State<'_, AppState>) -> Result<Vec<GpuInfo>, String> {
    Ok(state.slai.get_gpus().await)
}

/// Submit a new scheduler job.
#[tauri::command]
pub async fn submit_scheduler_job(
    input: SubmitJobInput,
    state: State<'_, AppState>,
) -> Result<SchedulerJob, String> {
    tracing::info!("Submitting job: {} for tenant {}", input.name, input.tenant_id);

    let priority = match input.priority.as_deref() {
        Some("critical") => Some(JobPriority::Critical),
        Some("high") => Some(JobPriority::High),
        Some("normal") => Some(JobPriority::Normal),
        Some("low") => Some(JobPriority::Low),
        _ => None,
    };

    let request = SubmitJobRequest {
        name: input.name,
        tenant_id: input.tenant_id,
        gpus_requested: input.gpus_requested,
        priority,
        estimated_duration_secs: input.estimated_duration_secs,
    };

    state.slai.submit_job(request).await
}

/// Get a scheduler job by ID.
#[tauri::command]
pub async fn get_scheduler_job(
    job_id: String,
    state: State<'_, AppState>,
) -> Result<SchedulerJob, String> {
    let job: Option<SchedulerJob> = state.slai.get_job(&job_id).await;
    job.ok_or_else(|| format!("Job not found: {}", job_id))
}

/// List all scheduler jobs.
#[tauri::command]
pub async fn list_scheduler_jobs(state: State<'_, AppState>) -> Result<Vec<SchedulerJob>, String> {
    Ok(state.slai.list_jobs().await)
}

/// List jobs for a specific tenant.
#[tauri::command]
pub async fn list_jobs_by_tenant(
    tenant_id: String,
    state: State<'_, AppState>,
) -> Result<Vec<SchedulerJob>, String> {
    Ok(state.slai.list_jobs_for_tenant(&tenant_id).await)
}

/// Cancel a scheduler job.
#[tauri::command]
pub async fn cancel_scheduler_job(
    job_id: String,
    state: State<'_, AppState>,
) -> Result<SchedulerJob, String> {
    tracing::info!("Cancelling job: {}", job_id);
    state.slai.cancel_job(&job_id).await
}

/// Create a new tenant.
#[tauri::command]
pub async fn create_tenant(
    input: CreateTenantInput,
    state: State<'_, AppState>,
) -> Result<Tenant, String> {
    tracing::info!("Creating tenant: {}", input.name);

    let request = CreateTenantRequest {
        name: input.name,
        priority_weight: input.priority_weight,
        max_gpus: input.max_gpus,
        max_concurrent_jobs: input.max_concurrent_jobs,
    };

    state.slai.create_tenant(request).await
}

/// Get a tenant by ID.
#[tauri::command]
pub async fn get_tenant(tenant_id: String, state: State<'_, AppState>) -> Result<Tenant, String> {
    let tenant: Option<Tenant> = state.slai.get_tenant(&tenant_id).await;
    tenant.ok_or_else(|| format!("Tenant not found: {}", tenant_id))
}

/// List all tenants.
#[tauri::command]
pub async fn list_tenants(state: State<'_, AppState>) -> Result<Vec<Tenant>, String> {
    Ok(state.slai.list_tenants().await)
}

/// Suspend a tenant.
#[tauri::command]
pub async fn suspend_tenant(
    tenant_id: String,
    state: State<'_, AppState>,
) -> Result<Tenant, String> {
    tracing::info!("Suspending tenant: {}", tenant_id);
    state.slai.suspend_tenant(&tenant_id).await
}

/// Resume a suspended tenant.
#[tauri::command]
pub async fn resume_tenant(
    tenant_id: String,
    state: State<'_, AppState>,
) -> Result<Tenant, String> {
    tracing::info!("Resuming tenant: {}", tenant_id);
    state.slai.resume_tenant(&tenant_id).await
}

/// Get scheduler summary.
#[tauri::command]
pub async fn get_scheduler_summary(
    state: State<'_, AppState>,
) -> Result<SchedulerSummary, String> {
    Ok(state.slai.get_summary().await)
}

/// Get fair share allocation for all tenants.
#[tauri::command]
pub async fn get_fair_share(
    state: State<'_, AppState>,
) -> Result<Vec<FairShareAllocation>, String> {
    Ok(state.slai.get_fair_share().await)
}
