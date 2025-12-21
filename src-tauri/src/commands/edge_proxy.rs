//! Edge Proxy Commands (Synergy 5)
//!
//! Tauri commands for Vortex edge proxy + SLAI brain integration.

use crate::state::AppState;
use crate::edge_proxy_bridge::{
    EdgeProxyBrainStatus, EdgeProxyStatus, BrainStatus,
    FailurePrediction, BackendHealth, RoutingDecision,
};
use tauri::State;

/// Get complete edge proxy and brain status.
#[tauri::command]
pub async fn get_edge_proxy_status(state: State<'_, AppState>) -> Result<EdgeProxyBrainStatus, String> {
    Ok(state.edge_proxy.get_status().await)
}

/// Get edge proxy status only.
#[tauri::command]
pub async fn get_proxy_status(state: State<'_, AppState>) -> Result<EdgeProxyStatus, String> {
    Ok(state.edge_proxy.get_proxy_status().await)
}

/// Get SLAI brain status.
#[tauri::command]
pub async fn get_brain_status(state: State<'_, AppState>) -> Result<BrainStatus, String> {
    Ok(state.edge_proxy.get_brain_status().await)
}

/// Get failure predictions.
#[tauri::command]
pub async fn get_failure_predictions(state: State<'_, AppState>) -> Result<Vec<FailurePrediction>, String> {
    Ok(state.edge_proxy.get_predictions().await)
}

/// Get backend health.
#[tauri::command]
pub async fn get_backend_health(state: State<'_, AppState>) -> Result<Vec<BackendHealth>, String> {
    Ok(state.edge_proxy.get_backend_health().await)
}

/// Get recent routing decisions.
#[tauri::command]
pub async fn get_routing_decisions(state: State<'_, AppState>) -> Result<Vec<RoutingDecision>, String> {
    Ok(state.edge_proxy.get_routing_decisions().await)
}

/// Simulate edge proxy activity (for demo purposes).
#[tauri::command]
pub async fn simulate_edge_proxy_activity(state: State<'_, AppState>) -> Result<(), String> {
    state.edge_proxy.simulate_activity().await;
    Ok(())
}
