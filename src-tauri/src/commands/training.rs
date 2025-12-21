//! Training Job Commands
//!
//! Commands for managing ML training jobs via RustyTorch.

use crate::state::AppState;
use crate::training_bridge::{
    TrainingConfig as BridgeConfig, TrainingJob, TrainingSummary,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tauri::State;

/// Training configuration from frontend.
#[derive(Debug, Serialize, Deserialize)]
pub struct TrainingConfigInput {
    pub name: String,
    pub model: String,
    pub dataset: String,
    pub epochs: u32,
    pub batch_size: u32,
    pub learning_rate: f64,
    pub optimizer: Option<String>,
    pub distributed: Option<bool>,
    pub world_size: Option<u32>,
}

/// Start a new training job.
#[tauri::command]
pub async fn start_training(
    config: TrainingConfigInput,
    state: State<'_, AppState>,
) -> Result<TrainingJob, String> {
    tracing::info!("Starting training job: {}", config.name);

    let bridge_config = BridgeConfig {
        model: config.model,
        dataset: config.dataset,
        epochs: config.epochs,
        batch_size: config.batch_size,
        learning_rate: config.learning_rate,
        optimizer: config.optimizer.unwrap_or_else(|| "AdamW".to_string()),
        distributed: if config.distributed.unwrap_or(false) {
            Some(crate::training_bridge::DistributedConfig {
                world_size: config.world_size.unwrap_or(1),
                strategy: "data_parallel".to_string(),
            })
        } else {
            None
        },
        hyperparameters: HashMap::new(),
    };

    state.training.start_training(config.name, bridge_config).await
}

/// Get the status of a training job.
#[tauri::command]
pub async fn get_training_status(
    job_id: String,
    state: State<'_, AppState>,
) -> Result<TrainingJob, String> {
    state
        .training
        .get_job(&job_id)
        .await
        .ok_or_else(|| format!("Job not found: {}", job_id))
}

/// List all training jobs.
#[tauri::command]
pub async fn list_training_jobs(state: State<'_, AppState>) -> Result<Vec<TrainingJob>, String> {
    Ok(state.training.get_all_jobs().await)
}

/// Get training summary.
#[tauri::command]
pub async fn get_training_summary(state: State<'_, AppState>) -> Result<TrainingSummary, String> {
    Ok(state.training.get_summary().await)
}

/// Pause a training job.
#[tauri::command]
pub async fn pause_training(job_id: String, state: State<'_, AppState>) -> Result<(), String> {
    state.training.pause_job(&job_id).await
}

/// Resume a paused training job.
#[tauri::command]
pub async fn resume_training(job_id: String, state: State<'_, AppState>) -> Result<(), String> {
    state.training.resume_job(&job_id).await
}

/// Cancel a training job.
#[tauri::command]
pub async fn cancel_training(job_id: String, state: State<'_, AppState>) -> Result<(), String> {
    state.training.cancel_job(&job_id).await
}
