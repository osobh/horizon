//! Tauri command handlers for HPC-CI integration.
//!
//! These functions are exposed to the frontend via Tauri's IPC mechanism.

use crate::hpcci_bridge::{
    AgentSummary, ApprovalRequest, DashboardSummary, HpcCiStatus, LogsResponse, PipelineDetail,
    PipelineFilter, PipelineSummary, TriggerParams,
};
use crate::state::AppState;
use tauri::State;

/// Get HPC-CI server status.
#[tauri::command]
pub async fn get_hpcci_status(state: State<'_, AppState>) -> Result<HpcCiStatus, String> {
    Ok(state.hpcci.get_status().await)
}

/// Set HPC-CI server URL.
#[tauri::command]
pub async fn set_hpcci_server_url(state: State<'_, AppState>, url: String) -> Result<(), String> {
    state.hpcci.set_server_url(url).await;
    Ok(())
}

/// List pipelines with optional filters.
#[tauri::command]
pub async fn list_hpcci_pipelines(
    state: State<'_, AppState>,
    filter: Option<PipelineFilter>,
) -> Result<Vec<PipelineSummary>, String> {
    state
        .hpcci
        .list_pipelines(filter.unwrap_or_default())
        .await
}

/// Get pipeline details.
#[tauri::command]
pub async fn get_hpcci_pipeline(
    state: State<'_, AppState>,
    id: String,
) -> Result<PipelineDetail, String> {
    state.hpcci.get_pipeline(&id).await
}

/// Trigger a new pipeline.
#[tauri::command]
pub async fn trigger_hpcci_pipeline(
    state: State<'_, AppState>,
    params: TriggerParams,
) -> Result<String, String> {
    state.hpcci.trigger_pipeline(params).await
}

/// Cancel a running pipeline.
#[tauri::command]
pub async fn cancel_hpcci_pipeline(state: State<'_, AppState>, id: String) -> Result<(), String> {
    state.hpcci.cancel_pipeline(&id).await
}

/// Retry a failed pipeline.
#[tauri::command]
pub async fn retry_hpcci_pipeline(
    state: State<'_, AppState>,
    id: String,
) -> Result<String, String> {
    state.hpcci.retry_pipeline(&id).await
}

/// List all agents.
#[tauri::command]
pub async fn list_hpcci_agents(state: State<'_, AppState>) -> Result<Vec<AgentSummary>, String> {
    state.hpcci.list_agents().await
}

/// Drain an agent (stop accepting new jobs).
#[tauri::command]
pub async fn drain_hpcci_agent(state: State<'_, AppState>, id: String) -> Result<(), String> {
    state.hpcci.drain_agent(&id).await
}

/// Enable an agent (start accepting jobs).
#[tauri::command]
pub async fn enable_hpcci_agent(state: State<'_, AppState>, id: String) -> Result<(), String> {
    state.hpcci.enable_agent(&id).await
}

/// Get pending approvals.
#[tauri::command]
pub async fn get_hpcci_approvals(
    state: State<'_, AppState>,
) -> Result<Vec<ApprovalRequest>, String> {
    state.hpcci.get_approvals().await
}

/// Submit an approval decision.
#[tauri::command]
pub async fn submit_hpcci_approval(
    state: State<'_, AppState>,
    id: String,
    approved: bool,
    comment: Option<String>,
) -> Result<(), String> {
    state.hpcci.submit_approval(&id, approved, comment).await
}

/// Get dashboard summary statistics.
#[tauri::command]
pub async fn get_hpcci_dashboard_summary(
    state: State<'_, AppState>,
) -> Result<DashboardSummary, String> {
    state.hpcci.get_dashboard_summary().await
}

/// Get pipeline logs.
#[tauri::command]
pub async fn get_hpcci_pipeline_logs(
    state: State<'_, AppState>,
    id: String,
    offset: Option<u64>,
) -> Result<LogsResponse, String> {
    state.hpcci.get_pipeline_logs(&id, offset).await
}
