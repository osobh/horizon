//! Evolution Engine Commands
//!
//! Commands for interacting with the stratoswarm evolution engines:
//! - ADAS (Automated Design of Agentic Systems)
//! - DGM (Discovered Growth Mode)
//! - SwarmAgentic (Population-based optimization)

use crate::evolution_bridge::{
    AdasMetrics, DgmMetrics, EvolutionEvent, EvolutionStatus, SwarmMetrics,
};
use crate::state::AppState;
use tauri::State;

/// Get the combined status of all evolution engines.
#[tauri::command]
pub async fn get_evolution_status(state: State<'_, AppState>) -> Result<EvolutionStatus, String> {
    Ok(state.evolution.get_status().await)
}

/// Get ADAS (Automated Design of Agentic Systems) metrics.
#[tauri::command]
pub async fn get_adas_metrics(state: State<'_, AppState>) -> Result<AdasMetrics, String> {
    Ok(state.evolution.get_adas_metrics().await)
}

/// Get DGM (Discovered Growth Mode) metrics.
#[tauri::command]
pub async fn get_dgm_metrics(state: State<'_, AppState>) -> Result<DgmMetrics, String> {
    Ok(state.evolution.get_dgm_metrics().await)
}

/// Get SwarmAgentic metrics.
#[tauri::command]
pub async fn get_swarm_metrics(state: State<'_, AppState>) -> Result<SwarmMetrics, String> {
    Ok(state.evolution.get_swarm_metrics().await)
}

/// Get recent evolution events.
#[tauri::command]
pub async fn get_evolution_events(
    limit: Option<usize>,
    state: State<'_, AppState>,
) -> Result<Vec<EvolutionEvent>, String> {
    Ok(state.evolution.get_events(limit.unwrap_or(10)).await)
}

/// Simulate one evolution step (for demo/testing).
#[tauri::command]
pub async fn simulate_evolution_step(state: State<'_, AppState>) -> Result<(), String> {
    state.evolution.simulate_step().await;
    Ok(())
}
