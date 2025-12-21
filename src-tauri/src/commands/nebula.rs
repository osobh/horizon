//! Nebula Commands
//!
//! Tauri commands for RDMA transport, ZK proofs, and mesh topology.

use crate::nebula_bridge::{MeshTopology, NebulaStatus, RdmaStats, ZkStats};
use crate::state::AppState;
use tauri::State;

/// Get combined nebula status (RDMA + ZK + topology).
#[tauri::command]
pub async fn get_nebula_status(state: State<'_, AppState>) -> Result<NebulaStatus, String> {
    Ok(state.nebula.get_status().await)
}

/// Get RDMA transport statistics.
#[tauri::command]
pub async fn get_rdma_stats(state: State<'_, AppState>) -> Result<RdmaStats, String> {
    Ok(state.nebula.get_rdma_stats().await)
}

/// Get ZK proof generation statistics.
#[tauri::command]
pub async fn get_zk_stats(state: State<'_, AppState>) -> Result<ZkStats, String> {
    Ok(state.nebula.get_zk_stats().await)
}

/// Get mesh network topology.
#[tauri::command]
pub async fn get_mesh_topology(state: State<'_, AppState>) -> Result<MeshTopology, String> {
    Ok(state.nebula.get_topology().await)
}

/// Simulate network activity (for demo purposes).
#[tauri::command]
pub async fn simulate_nebula_activity(state: State<'_, AppState>) -> Result<(), String> {
    state.nebula.simulate_activity().await;
    Ok(())
}
