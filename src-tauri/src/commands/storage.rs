//! Storage Commands
//!
//! Commands for file transfer via WARP.

use crate::state::AppState;
use crate::storage_bridge::{
    FileInfo, StorageStats, Transfer, TransferConfig,
};
use serde::{Deserialize, Serialize};
use tauri::State;

/// Upload configuration from frontend.
#[derive(Debug, Serialize, Deserialize)]
pub struct UploadInput {
    pub source: String,
    pub destination: String,
    pub config: Option<TransferConfig>,
}

/// Download configuration from frontend.
#[derive(Debug, Serialize, Deserialize)]
pub struct DownloadInput {
    pub source: String,
    pub destination: String,
    pub config: Option<TransferConfig>,
}

/// Upload a file to the cluster via WARP.
#[tauri::command]
pub async fn upload_file(
    input: UploadInput,
    state: State<'_, AppState>,
) -> Result<Transfer, String> {
    tracing::info!("Uploading file: {} to {}", input.source, input.destination);
    state.storage.upload(input.source, input.destination, input.config).await
}

/// Download a file from the cluster via WARP.
#[tauri::command]
pub async fn download_file(
    input: DownloadInput,
    state: State<'_, AppState>,
) -> Result<Transfer, String> {
    tracing::info!("Downloading file: {} to {}", input.source, input.destination);
    state.storage.download(input.source, input.destination, input.config).await
}

/// Get a transfer by ID.
#[tauri::command]
pub async fn get_transfer(
    transfer_id: String,
    state: State<'_, AppState>,
) -> Result<Transfer, String> {
    state
        .storage
        .get_transfer(&transfer_id)
        .await
        .ok_or_else(|| format!("Transfer not found: {}", transfer_id))
}

/// List all transfers.
#[tauri::command]
pub async fn list_transfers(state: State<'_, AppState>) -> Result<Vec<Transfer>, String> {
    Ok(state.storage.get_all_transfers().await)
}

/// List active transfers.
#[tauri::command]
pub async fn list_active_transfers(state: State<'_, AppState>) -> Result<Vec<Transfer>, String> {
    Ok(state.storage.get_active_transfers().await)
}

/// Get storage statistics.
#[tauri::command]
pub async fn get_storage_stats(state: State<'_, AppState>) -> Result<StorageStats, String> {
    Ok(state.storage.get_stats().await)
}

/// Pause a transfer.
#[tauri::command]
pub async fn pause_transfer(
    transfer_id: String,
    state: State<'_, AppState>,
) -> Result<(), String> {
    state.storage.pause_transfer(&transfer_id).await
}

/// Resume a paused transfer.
#[tauri::command]
pub async fn resume_transfer(
    transfer_id: String,
    state: State<'_, AppState>,
) -> Result<(), String> {
    state.storage.resume_transfer(&transfer_id).await
}

/// Cancel a transfer.
#[tauri::command]
pub async fn cancel_transfer(
    transfer_id: String,
    state: State<'_, AppState>,
) -> Result<(), String> {
    state.storage.cancel_transfer(&transfer_id).await
}

/// List files in a directory (local or remote).
#[tauri::command]
pub async fn list_files(
    path: String,
    state: State<'_, AppState>,
) -> Result<Vec<FileInfo>, String> {
    state.storage.list_files(&path).await
}
