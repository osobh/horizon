//! Argus Observability Commands
//!
//! Tauri IPC commands for Argus integration providing:
//! - PromQL metrics querying (instant and range)
//! - Alert management and listing
//! - Scrape target health monitoring
//! - Server status and configuration

use crate::argus_bridge::{Alert, ArgusStatus, QueryResult, Target};
use crate::state::AppState;
use tauri::State;

/// Get Argus server status.
#[tauri::command]
pub async fn get_argus_status(state: State<'_, AppState>) -> Result<ArgusStatus, String> {
    Ok(state.argus.get_status().await)
}

/// Set the Argus server URL.
#[tauri::command]
pub async fn set_argus_server_url(
    state: State<'_, AppState>,
    url: String,
) -> Result<(), String> {
    state.argus.set_server_url(url).await;
    Ok(())
}

/// Execute an instant PromQL query.
#[tauri::command]
pub async fn query_argus_metrics(
    state: State<'_, AppState>,
    query: String,
) -> Result<QueryResult, String> {
    state.argus.query_instant(&query).await
}

/// Execute a range PromQL query.
#[tauri::command]
pub async fn query_argus_metrics_range(
    state: State<'_, AppState>,
    query: String,
    start: i64,
    end: i64,
    step: u64,
) -> Result<QueryResult, String> {
    state.argus.query_range(&query, start, end, step).await
}

/// Get all alerts (firing, pending, resolved).
#[tauri::command]
pub async fn get_argus_alerts(state: State<'_, AppState>) -> Result<Vec<Alert>, String> {
    state.argus.get_alerts().await
}

/// Get all scrape targets.
#[tauri::command]
pub async fn get_argus_targets(state: State<'_, AppState>) -> Result<Vec<Target>, String> {
    state.argus.get_targets().await
}
