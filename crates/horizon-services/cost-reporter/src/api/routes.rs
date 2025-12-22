use axum::{
    routing::{get, post, put},
    Router,
};
use std::sync::Arc;
use tower_http::trace::TraceLayer;

use crate::api::{handlers, AppState};

pub fn create_router(state: Arc<AppState>) -> Router {
    Router::new()
        // Health endpoints
        .route("/health", get(handlers::health))
        .route("/ready", get(handlers::ready))
        // Finance - Cost Summary & Breakdown
        .route("/api/v1/finance/costs/summary", get(handlers::get_cost_summary))
        .route("/api/v1/finance/costs/breakdown", get(handlers::get_cost_breakdown))
        // Finance - Budgets
        .route("/api/v1/finance/budgets", get(handlers::get_team_budgets))
        .route("/api/v1/finance/budgets/:id", get(handlers::get_budget))
        .route("/api/v1/finance/budgets/:id", put(handlers::update_budget))
        // Finance - Chargeback
        .route("/api/v1/finance/chargeback/generate", post(handlers::generate_chargeback_report))
        .route("/api/v1/finance/chargeback/reports", get(handlers::get_chargeback_reports))
        .route("/api/v1/finance/chargeback/reports/:id", get(handlers::get_chargeback_report))
        // Finance - Cost Optimizations
        .route("/api/v1/finance/optimizations", get(handlers::get_cost_optimizations))
        .route("/api/v1/finance/optimizations/:id", get(handlers::get_cost_optimization))
        .route("/api/v1/finance/optimizations/:id/implement", post(handlers::implement_optimization))
        .route("/api/v1/finance/optimizations/:id/reject", post(handlers::reject_optimization))
        // Finance - Cost Alerts
        .route("/api/v1/finance/alerts", get(handlers::get_cost_alerts))
        .route("/api/v1/finance/alerts", post(handlers::create_cost_alert))
        .route("/api/v1/finance/alerts/:id/acknowledge", post(handlers::acknowledge_cost_alert))
        .route("/api/v1/finance/alerts/:id/resolve", post(handlers::resolve_cost_alert))
        .route("/api/v1/finance/alerts/configurations", get(handlers::get_alert_configurations))
        .route("/api/v1/finance/alerts/configurations/:id", put(handlers::update_alert_configuration))
        // Showback reports (existing)
        .route(
            "/api/v1/reports/showback/team/:team_id",
            get(handlers::get_team_showback),
        )
        .route(
            "/api/v1/reports/showback/user/:user_id",
            get(handlers::get_user_showback),
        )
        .route(
            "/api/v1/reports/showback/top-spenders",
            get(handlers::get_top_spenders),
        )
        // Chargeback reports (existing)
        .route(
            "/api/v1/reports/chargeback/customer/:customer_id",
            get(handlers::get_customer_chargeback),
        )
        // Trend analysis (existing)
        .route("/api/v1/reports/trends/daily", get(handlers::get_daily_trends))
        .route("/api/v1/reports/trends/monthly", get(handlers::get_monthly_trends))
        .route("/api/v1/reports/trends/forecast", get(handlers::get_forecast))
        // Export endpoints (existing)
        .route("/api/v1/export/csv", get(handlers::export_csv))
        .route("/api/v1/export/json", get(handlers::export_json))
        .route("/api/v1/export/markdown", get(handlers::export_markdown))
        // Admin endpoints (existing)
        .route("/api/v1/admin/refresh-views", post(handlers::refresh_views))
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}
