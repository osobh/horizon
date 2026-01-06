//! Intelligence Commands
//!
//! Tauri commands for executive intelligence, efficiency analysis, and vendor tracking.

use crate::intelligence_bridge::{
    ExecutiveKpi, IdleResource, IntelligenceAlert, IntelligenceSummary, ProfitMargin, VendorUtilization,
};
use crate::state::AppState;
use tauri::State;

/// Get complete intelligence summary.
#[tauri::command]
pub async fn get_intelligence_summary(state: State<'_, AppState>) -> Result<IntelligenceSummary, String> {
    Ok(state.intelligence.get_summary().await)
}

/// Get idle resources detected by efficiency intelligence.
#[tauri::command]
pub async fn get_idle_resources(state: State<'_, AppState>) -> Result<Vec<IdleResource>, String> {
    Ok(state.intelligence.get_idle_resources().await)
}

/// Get profit margins, optionally filtered by type.
#[tauri::command]
pub async fn get_profit_margins(
    margin_type: Option<String>,
    state: State<'_, AppState>,
) -> Result<Vec<ProfitMargin>, String> {
    Ok(state.intelligence.get_profit_margins(margin_type).await)
}

/// Get vendor utilizations.
#[tauri::command]
pub async fn get_vendor_utilizations(state: State<'_, AppState>) -> Result<Vec<VendorUtilization>, String> {
    Ok(state.intelligence.get_vendor_utilizations().await)
}

/// Get executive KPIs.
#[tauri::command]
pub async fn get_executive_kpis(state: State<'_, AppState>) -> Result<Vec<ExecutiveKpi>, String> {
    Ok(state.intelligence.get_kpis().await)
}

/// Get intelligence alerts.
#[tauri::command]
pub async fn get_intelligence_alerts(state: State<'_, AppState>) -> Result<Vec<IntelligenceAlert>, String> {
    Ok(state.intelligence.get_alerts().await)
}

/// Acknowledge an intelligence alert.
#[tauri::command]
pub async fn acknowledge_intelligence_alert(
    alert_id: String,
    state: State<'_, AppState>,
) -> Result<(), String> {
    state
        .intelligence
        .acknowledge_alert(alert_id)
        .await
        .map_err(|e| e.into())
}

/// Terminate an idle resource.
#[tauri::command]
pub async fn terminate_idle_resource(
    resource_id: String,
    state: State<'_, AppState>,
) -> Result<(), String> {
    state
        .intelligence
        .terminate_idle_resource(resource_id)
        .await
        .map_err(|e| e.into())
}
