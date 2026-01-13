//! Tauri command handlers for RustySpark integration.
//!
//! These functions are exposed to the frontend via Tauri's IPC mechanism.

use crate::rustyspark_bridge::{
    RustySparkStatus, SparkJob, SparkJobStatus, SparkStage, SparkSummary, SparkTask,
};
use crate::state::AppState;
use tauri::State;

/// Get RustySpark server status.
#[tauri::command]
pub async fn get_rustyspark_status(state: State<'_, AppState>) -> Result<RustySparkStatus, String> {
    Ok(state.rustyspark.get_status().await)
}

/// Set RustySpark server URL.
#[tauri::command]
pub async fn set_rustyspark_server_url(
    state: State<'_, AppState>,
    url: String,
) -> Result<(), String> {
    state.rustyspark.set_server_url(url).await;
    Ok(())
}

/// Get summary statistics.
#[tauri::command]
pub async fn get_rustyspark_summary(
    state: State<'_, AppState>,
) -> Result<SparkSummary, String> {
    state.rustyspark.get_summary().await
}

/// List Spark jobs.
#[tauri::command]
pub async fn list_spark_jobs(
    state: State<'_, AppState>,
    status: Option<SparkJobStatus>,
    limit: Option<i64>,
) -> Result<Vec<SparkJob>, String> {
    state.rustyspark.list_jobs(status, limit).await
}

/// Get Spark job by ID.
#[tauri::command]
pub async fn get_spark_job(
    state: State<'_, AppState>,
    id: String,
) -> Result<SparkJob, String> {
    state.rustyspark.get_job(&id).await
}

/// Get stages for a job.
#[tauri::command]
pub async fn get_spark_job_stages(
    state: State<'_, AppState>,
    job_id: String,
) -> Result<Vec<SparkStage>, String> {
    state.rustyspark.get_job_stages(&job_id).await
}

/// Get tasks for a stage.
#[tauri::command]
pub async fn get_spark_stage_tasks(
    state: State<'_, AppState>,
    stage_id: i64,
) -> Result<Vec<SparkTask>, String> {
    state.rustyspark.get_stage_tasks(stage_id).await
}

/// Cancel a Spark job.
#[tauri::command]
pub async fn cancel_spark_job(
    state: State<'_, AppState>,
    id: String,
) -> Result<(), String> {
    state.rustyspark.cancel_job(&id).await
}
