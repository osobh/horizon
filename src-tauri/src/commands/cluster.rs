//! Cluster Management Commands
//!
//! Commands for interacting with StratoSwarm clusters.

use crate::cluster_bridge::{ClusterNode, ClusterStats};
use crate::state::AppState;
use serde::{Deserialize, Serialize};
use tauri::State;

#[derive(Debug, Serialize, Deserialize)]
pub struct ClusterStatus {
    pub connected: bool,
    pub endpoint: Option<String>,
    pub node_count: usize,
    pub total_gpus: u32,
    pub total_memory_gb: f32,
    pub healthy_nodes: usize,
}

/// Get the current cluster connection status.
#[tauri::command]
pub async fn get_cluster_status(state: State<'_, AppState>) -> Result<ClusterStatus, String> {
    let stats = state.cluster.get_statistics().await;
    let connected = state.cluster.is_connected().await;
    let endpoint = state.cluster.endpoint().await;

    Ok(ClusterStatus {
        connected,
        endpoint,
        node_count: stats.total_nodes,
        total_gpus: stats.total_gpu_count,
        total_memory_gb: stats.total_memory_gb,
        healthy_nodes: stats.online_nodes,
    })
}

/// List all nodes in the cluster.
#[tauri::command]
pub async fn list_nodes(state: State<'_, AppState>) -> Result<Vec<ClusterNode>, String> {
    Ok(state.cluster.get_nodes().await)
}

/// Get detailed node information.
#[tauri::command]
pub async fn get_node(id: String, state: State<'_, AppState>) -> Result<Option<ClusterNode>, String> {
    Ok(state.cluster.get_node(&id).await)
}

/// Get cluster statistics.
#[tauri::command]
pub async fn get_cluster_stats(state: State<'_, AppState>) -> Result<ClusterStats, String> {
    Ok(state.cluster.get_statistics().await)
}

/// Connect to a StratoSwarm cluster.
#[tauri::command]
pub async fn connect_cluster(
    endpoint: String,
    state: State<'_, AppState>,
) -> Result<ClusterStatus, String> {
    tracing::info!("Connecting to cluster at {}", endpoint);

    // Connect via the cluster bridge
    state.cluster.connect(&endpoint).await?;

    // Return updated status
    get_cluster_status(state).await
}

/// Disconnect from the cluster.
#[tauri::command]
pub async fn disconnect_cluster(state: State<'_, AppState>) -> Result<(), String> {
    state.cluster.disconnect().await
}
