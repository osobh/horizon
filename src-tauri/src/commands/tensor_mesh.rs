//! Tensor Mesh Commands
//!
//! Tauri commands for distributed GPU-to-GPU tensor operations.

use crate::state::AppState;
use crate::tensor_mesh_bridge::{
    CollectiveStats, TensorMeshStatus, TensorNode, TensorTransfer,
};
use tauri::State;

/// Get tensor mesh status (nodes, connections, stats).
#[tauri::command]
pub async fn get_tensor_mesh_status(state: State<'_, AppState>) -> Result<TensorMeshStatus, String> {
    Ok(state.tensor_mesh.get_status().await)
}

/// Get collective operation statistics.
#[tauri::command]
pub async fn get_collective_stats(state: State<'_, AppState>) -> Result<CollectiveStats, String> {
    Ok(state.tensor_mesh.get_collective_stats().await)
}

/// Get active tensor transfers.
#[tauri::command]
pub async fn get_active_transfers(state: State<'_, AppState>) -> Result<Vec<TensorTransfer>, String> {
    Ok(state.tensor_mesh.get_active_transfers().await)
}

/// Get tensor mesh nodes.
#[tauri::command]
pub async fn get_tensor_nodes(state: State<'_, AppState>) -> Result<Vec<TensorNode>, String> {
    Ok(state.tensor_mesh.get_nodes().await)
}

/// Simulate tensor mesh activity (for demo purposes).
#[tauri::command]
pub async fn simulate_tensor_mesh_activity(state: State<'_, AppState>) -> Result<(), String> {
    state.tensor_mesh.simulate_activity().await;
    Ok(())
}
