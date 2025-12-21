//! Data Pipeline Commands (Synergy 4)
//!
//! Tauri commands for GPU-accelerated data pipeline (WARP integration).

use crate::state::AppState;
use crate::data_pipeline_bridge::{
    DataPipelineStatus, PipelineStats, StageStats, TransferJob,
};
use tauri::State;

/// Get complete data pipeline status.
#[tauri::command]
pub async fn get_data_pipeline_status(state: State<'_, AppState>) -> Result<DataPipelineStatus, String> {
    Ok(state.data_pipeline.get_status().await)
}

/// Get pipeline statistics.
#[tauri::command]
pub async fn get_pipeline_stats(state: State<'_, AppState>) -> Result<PipelineStats, String> {
    Ok(state.data_pipeline.get_stats().await)
}

/// Get stage statistics.
#[tauri::command]
pub async fn get_pipeline_stages(state: State<'_, AppState>) -> Result<Vec<StageStats>, String> {
    Ok(state.data_pipeline.get_stages().await)
}

/// Get active transfer jobs.
#[tauri::command]
pub async fn get_pipeline_jobs(state: State<'_, AppState>) -> Result<Vec<TransferJob>, String> {
    Ok(state.data_pipeline.get_active_jobs().await)
}

/// Simulate pipeline activity (for demo purposes).
#[tauri::command]
pub async fn simulate_pipeline_activity(state: State<'_, AppState>) -> Result<(), String> {
    state.data_pipeline.simulate_activity().await;
    Ok(())
}
