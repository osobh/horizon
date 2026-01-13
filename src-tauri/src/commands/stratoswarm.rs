//! Tauri command handlers for StratoSwarm integration.
//!
//! These functions are exposed to the frontend via Tauri's IPC mechanism.

use crate::state::AppState;
use crate::stratoswarm_bridge::{
    AgentStatus, AgentTask, AgentTier, EvolutionEvent, StratoSwarmStatus, SwarmAgent, SwarmStats,
};
use tauri::State;

/// Get StratoSwarm connection status.
#[tauri::command]
pub async fn get_stratoswarm_status(state: State<'_, AppState>) -> Result<StratoSwarmStatus, String> {
    Ok(state.stratoswarm.get_status().await)
}

/// Set StratoSwarm cluster URL.
#[tauri::command]
pub async fn set_stratoswarm_cluster_url(
    state: State<'_, AppState>,
    url: String,
) -> Result<(), String> {
    state.stratoswarm.set_cluster_url(url).await;
    Ok(())
}

/// Get swarm statistics.
#[tauri::command]
pub async fn get_swarm_stats(state: State<'_, AppState>) -> Result<SwarmStats, String> {
    state.stratoswarm.get_swarm_stats().await
}

/// List swarm agents with optional filters.
#[tauri::command]
pub async fn list_swarm_agents(
    state: State<'_, AppState>,
    status: Option<AgentStatus>,
    tier: Option<AgentTier>,
) -> Result<Vec<SwarmAgent>, String> {
    state.stratoswarm.list_agents(status, tier).await
}

/// Get agent by ID.
#[tauri::command]
pub async fn get_swarm_agent(state: State<'_, AppState>, id: String) -> Result<SwarmAgent, String> {
    state.stratoswarm.get_agent(&id).await
}

/// Get recent swarm evolution events.
#[tauri::command]
pub async fn get_swarm_evolution_events(
    state: State<'_, AppState>,
    limit: Option<usize>,
) -> Result<Vec<EvolutionEvent>, String> {
    state.stratoswarm.get_evolution_events(limit).await
}

/// Get active agent tasks.
#[tauri::command]
pub async fn get_active_agent_tasks(state: State<'_, AppState>) -> Result<Vec<AgentTask>, String> {
    state.stratoswarm.get_active_tasks().await
}

/// Trigger evolution for an agent.
#[tauri::command]
pub async fn trigger_agent_evolution(
    state: State<'_, AppState>,
    agent_id: String,
) -> Result<EvolutionEvent, String> {
    state.stratoswarm.trigger_evolution(&agent_id).await
}

/// Simulate swarm activity for demo purposes.
#[tauri::command]
pub async fn simulate_swarm_activity(state: State<'_, AppState>) -> Result<(), String> {
    state.stratoswarm.simulate_activity().await
}
