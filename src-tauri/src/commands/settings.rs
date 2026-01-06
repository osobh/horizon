//! Settings Commands
//!
//! Tauri commands for policy management, quota configuration, and app settings.

use crate::settings_bridge::{AppSettings, Policy, Quota, SettingsSummary};
use crate::state::AppState;
use tauri::State;

/// Get complete settings summary.
#[tauri::command]
pub async fn get_settings_summary(state: State<'_, AppState>) -> Result<SettingsSummary, String> {
    Ok(state.settings.get_summary().await)
}

/// Get all policies.
#[tauri::command]
pub async fn get_policies(state: State<'_, AppState>) -> Result<Vec<Policy>, String> {
    Ok(state.settings.get_policies().await)
}

/// Get all quotas.
#[tauri::command]
pub async fn get_quotas(state: State<'_, AppState>) -> Result<Vec<Quota>, String> {
    Ok(state.settings.get_quotas().await)
}

/// Get application settings.
#[tauri::command]
pub async fn get_app_settings(state: State<'_, AppState>) -> Result<AppSettings, String> {
    Ok(state.settings.get_app_settings().await)
}

/// Create a new policy.
#[tauri::command]
pub async fn create_policy(
    policy: Policy,
    state: State<'_, AppState>,
) -> Result<Policy, String> {
    state.settings.create_policy(policy).await.map_err(|e| e.into())
}

/// Update an existing policy.
#[tauri::command]
pub async fn update_policy(
    id: String,
    policy: serde_json::Value,
    state: State<'_, AppState>,
) -> Result<(), String> {
    state
        .settings
        .update_policy(id, policy)
        .await
        .map_err(|e| e.into())
}

/// Delete a policy.
#[tauri::command]
pub async fn delete_policy(
    id: String,
    state: State<'_, AppState>,
) -> Result<(), String> {
    state.settings.delete_policy(id).await.map_err(|e| e.into())
}

/// Toggle a policy's enabled status.
#[tauri::command]
pub async fn toggle_policy(
    id: String,
    enabled: bool,
    state: State<'_, AppState>,
) -> Result<(), String> {
    state
        .settings
        .toggle_policy(id, enabled)
        .await
        .map_err(|e| e.into())
}

/// Set a new quota.
#[tauri::command]
pub async fn set_quota(
    quota: Quota,
    state: State<'_, AppState>,
) -> Result<Quota, String> {
    state.settings.set_quota(quota).await.map_err(|e| e.into())
}

/// Update an existing quota.
#[tauri::command]
pub async fn update_quota(
    id: String,
    quota: serde_json::Value,
    state: State<'_, AppState>,
) -> Result<(), String> {
    state
        .settings
        .update_quota(id, quota)
        .await
        .map_err(|e| e.into())
}

/// Delete a quota.
#[tauri::command]
pub async fn delete_quota(
    id: String,
    state: State<'_, AppState>,
) -> Result<(), String> {
    state.settings.delete_quota(id).await.map_err(|e| e.into())
}

/// Update application settings.
#[tauri::command]
pub async fn update_app_settings(
    settings: serde_json::Value,
    state: State<'_, AppState>,
) -> Result<(), String> {
    state
        .settings
        .update_app_settings(settings)
        .await
        .map_err(|e| e.into())
}
