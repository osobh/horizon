use super::handlers::*;
use axum::{
    routing::{delete, get, patch, post},
    Router,
};
use std::sync::Arc;

pub fn create_router(state: Arc<AppState>) -> Router {
    Router::new()
        // Health
        .route("/health", get(health))
        // Executive Metrics
        .route("/api/v1/executive/metrics", get(get_executive_metrics))
        // Strategic KPIs
        .route("/api/v1/executive/kpis", get(get_strategic_kpis))
        .route("/api/v1/executive/kpis/:id", get(get_kpi))
        .route("/api/v1/executive/kpis/:id", patch(update_kpi_target))
        // Financial
        .route(
            "/api/v1/executive/financial/summary",
            get(get_financial_summary),
        )
        .route(
            "/api/v1/executive/financial/history",
            get(get_financial_history),
        )
        // Initiatives
        .route("/api/v1/executive/initiatives", get(get_initiatives))
        .route("/api/v1/executive/initiatives", post(create_initiative))
        .route("/api/v1/executive/initiatives/:id", get(get_initiative))
        .route("/api/v1/executive/initiatives/:id", patch(update_initiative))
        .route("/api/v1/executive/initiatives/:id", delete(delete_initiative))
        // Capacity Insights
        .route(
            "/api/v1/executive/capacity/insights",
            get(get_capacity_insights),
        )
        .route("/api/v1/executive/capacity/gaps", get(get_capacity_gaps))
        // Alerts
        .route("/api/v1/executive/alerts", get(get_alerts))
        .route("/api/v1/executive/alerts/:id/resolve", patch(resolve_alert))
        .route("/api/v1/executive/alerts/:id", delete(dismiss_alert))
        // Team Performance
        .route(
            "/api/v1/executive/teams/performance",
            get(get_team_performance),
        )
        .route(
            "/api/v1/executive/teams/:id/performance",
            get(get_team_performance_by_id),
        )
        // Investment Recommendations
        .route(
            "/api/v1/executive/recommendations",
            get(get_investment_recommendations),
        )
        .route(
            "/api/v1/executive/recommendations/:id/accept",
            post(accept_recommendation),
        )
        .route(
            "/api/v1/executive/recommendations/:id/reject",
            post(reject_recommendation),
        )
        // Reports
        .route("/api/v1/executive/reports", get(list_executive_reports))
        .route("/api/v1/executive/reports/:id", get(get_executive_report))
        .route(
            "/api/v1/executive/reports/generate",
            post(generate_executive_report),
        )
        // Dashboard Aggregation
        .route("/api/v1/executive/dashboard", get(get_dashboard_data))
        .with_state(state)
}
