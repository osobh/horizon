//! SLAI Scheduler Commands
//!
//! Tauri IPC commands for SLAI GPU scheduler integration.
//!
//! Provides:
//! - GPU detection and inventory
//! - Job submission, cancellation, listing
//! - Tenant management
//! - Scheduler statistics and fair-share allocation

use crate::slai_bridge::{
    FairShareInfo, GpuInfo, JobInfo, JobList, JobRequest, SchedulerStats, TenantInfo,
};
use crate::state::AppState;
use std::collections::HashMap;
use tauri::State;

/// Get scheduler statistics.
#[tauri::command]
pub async fn get_slai_stats(state: State<'_, AppState>) -> Result<SchedulerStats, String> {
    Ok(state.slai.get_scheduler_stats().await)
}

/// Detect and list all GPUs.
#[tauri::command]
pub async fn get_slai_gpus(state: State<'_, AppState>) -> Result<Vec<GpuInfo>, String> {
    Ok(state.slai.detect_gpus().await)
}

/// Get fair-share allocation per tenant.
#[tauri::command]
pub async fn get_slai_fair_share(
    state: State<'_, AppState>,
) -> Result<HashMap<String, FairShareInfo>, String> {
    Ok(state.slai.get_fair_share().await)
}

/// List all registered tenants.
#[tauri::command]
pub async fn list_slai_tenants(state: State<'_, AppState>) -> Result<Vec<TenantInfo>, String> {
    Ok(state.slai.list_tenants().await)
}

/// Create a new tenant.
#[tauri::command]
pub async fn create_slai_tenant(
    state: State<'_, AppState>,
    name: String,
    max_gpus: u32,
    max_concurrent_jobs: u32,
) -> Result<TenantInfo, String> {
    state.slai.create_tenant(name, max_gpus, max_concurrent_jobs).await
}

/// List all jobs (queued, running, completed).
#[tauri::command]
pub async fn list_slai_jobs(state: State<'_, AppState>) -> Result<JobList, String> {
    Ok(state.slai.list_jobs().await)
}

/// Submit a new job.
#[tauri::command]
pub async fn submit_slai_job(
    state: State<'_, AppState>,
    job: JobRequest,
) -> Result<String, String> {
    state.slai.submit_job(job).await
}

/// Cancel a job by ID.
#[tauri::command]
pub async fn cancel_slai_job(state: State<'_, AppState>, job_id: String) -> Result<(), String> {
    state.slai.cancel_job(&job_id).await
}

/// Schedule the next pending job (for demo/testing).
#[tauri::command]
pub async fn schedule_slai_next(state: State<'_, AppState>) -> Result<Option<JobInfo>, String> {
    Ok(state.slai.schedule_next().await)
}
