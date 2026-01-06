//! Cost Intelligence Commands
//!
//! Tauri commands for cost attribution, forecasting, and budget alerts.

use crate::costs_bridge::{BudgetAlert, ChargebackReport, CostAttribution, CostForecast, CostSummary};
use crate::state::AppState;
use tauri::State;

/// Get complete cost summary.
#[tauri::command]
pub async fn get_cost_summary(state: State<'_, AppState>) -> Result<CostSummary, String> {
    Ok(state.costs.get_summary().await)
}

/// Get cost attributions, optionally filtered by type.
#[tauri::command]
pub async fn get_cost_attributions(
    attribution_type: Option<String>,
    state: State<'_, AppState>,
) -> Result<Vec<CostAttribution>, String> {
    Ok(state.costs.get_attributions(attribution_type).await)
}

/// Get cost forecasts for the specified number of weeks.
#[tauri::command]
pub async fn get_cost_forecasts(
    weeks: Option<u32>,
    state: State<'_, AppState>,
) -> Result<Vec<CostForecast>, String> {
    Ok(state.costs.get_forecasts(weeks.unwrap_or(13)).await)
}

/// Get budget alerts.
#[tauri::command]
pub async fn get_budget_alerts(state: State<'_, AppState>) -> Result<Vec<BudgetAlert>, String> {
    Ok(state.costs.get_alerts().await)
}

/// List all cost reports.
#[tauri::command]
pub async fn list_cost_reports(state: State<'_, AppState>) -> Result<Vec<ChargebackReport>, String> {
    Ok(state.costs.get_reports().await)
}

/// Generate a chargeback report.
#[tauri::command]
pub async fn generate_chargeback_report(
    period_start: String,
    period_end: String,
    format: String,
    state: State<'_, AppState>,
) -> Result<ChargebackReport, String> {
    state
        .costs
        .generate_chargeback_report(period_start, period_end, format)
        .await
        .map_err(|e| e.into())
}

/// Generate a showback report for a team.
#[tauri::command]
pub async fn generate_showback_report(
    team_id: String,
    period_start: String,
    period_end: String,
    state: State<'_, AppState>,
) -> Result<ChargebackReport, String> {
    state
        .costs
        .generate_showback_report(team_id, period_start, period_end)
        .await
        .map_err(|e| e.into())
}

/// Set a budget threshold and alert percentage.
#[tauri::command]
pub async fn set_budget_threshold(
    name: String,
    threshold_usd: f64,
    alert_at_pct: f64,
    state: State<'_, AppState>,
) -> Result<(), String> {
    state
        .costs
        .set_budget_threshold(name, threshold_usd, alert_at_pct)
        .await
        .map_err(|e| e.into())
}
