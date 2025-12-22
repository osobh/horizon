use crate::db::MarginRepository;
use crate::error::Result;
use crate::models::*;
use crate::profiler::CustomerProfiler;
use crate::simulator::PricingSimulator;
use axum::{
    extract::{Path, Query, State},
    Json,
};
use rust_decimal::Decimal;
use serde::Deserialize;
use std::sync::Arc;
use uuid::Uuid;

#[derive(Clone)]
pub struct AppState {
    pub repository: MarginRepository,
    pub profiler: CustomerProfiler,
    pub simulator: PricingSimulator,
    pub at_risk_threshold: Decimal,
}

pub async fn health() -> &'static str {
    "OK"
}

pub async fn ready(State(state): State<Arc<AppState>>) -> Result<&'static str> {
    // Test database connection
    let _ = state.repository.get_margin_analysis().await?;
    Ok("READY")
}

// Customer profiles

#[derive(Deserialize)]
pub struct ListQuery {
    #[serde(default = "default_limit")]
    limit: i64,
    #[serde(default)]
    offset: i64,
}

fn default_limit() -> i64 {
    50
}

pub async fn list_customers(
    State(state): State<Arc<AppState>>,
    Query(query): Query<ListQuery>,
) -> Result<Json<Vec<CustomerProfile>>> {
    let profiles = state
        .repository
        .list_profiles(query.limit, query.offset)
        .await?;
    Ok(Json(profiles))
}

pub async fn get_customer_profile(
    State(state): State<Arc<AppState>>,
    Path(customer_id): Path<String>,
) -> Result<Json<CustomerProfile>> {
    let profile = state.repository.get_profile(&customer_id).await?;
    Ok(Json(profile))
}

pub async fn refresh_customer_profile(
    State(state): State<Arc<AppState>>,
    Path(customer_id): Path<String>,
    Json(_request): Json<RefreshProfileRequest>,
) -> Result<Json<CustomerProfile>> {
    // In a real implementation, we'd fetch costs from cost-attributor
    let profile = state.repository.get_profile(&customer_id).await?;

    // For now, recalculate with existing data
    let updated = state
        .repository
        .update_profile_metrics(&customer_id, profile.total_revenue, profile.total_cost)
        .await?;

    Ok(Json(updated))
}

// Analysis endpoints

pub async fn get_margin_analysis(
    State(state): State<Arc<AppState>>,
) -> Result<Json<MarginAnalysis>> {
    let analysis = state.repository.get_margin_analysis().await?;
    Ok(Json(analysis))
}

pub async fn get_segment_analysis(
    State(state): State<Arc<AppState>>,
) -> Result<Json<Vec<SegmentAnalysis>>> {
    let analysis = state.repository.get_segment_analysis().await?;
    Ok(Json(analysis))
}

pub async fn get_top_contributors(
    State(state): State<Arc<AppState>>,
    Query(query): Query<ListQuery>,
) -> Result<Json<Vec<TopContributor>>> {
    let contributors = state
        .profiler
        .identify_top_contributors(query.limit)
        .await?;
    Ok(Json(contributors))
}

pub async fn get_at_risk_customers(
    State(state): State<Arc<AppState>>,
) -> Result<Json<Vec<CustomerProfile>>> {
    let at_risk = state
        .profiler
        .identify_at_risk(state.at_risk_threshold)
        .await?;
    Ok(Json(at_risk))
}

// Simulations

pub async fn create_simulation(
    State(state): State<Arc<AppState>>,
    Json(request): Json<CreateSimulationRequest>,
) -> Result<Json<PricingSimulation>> {
    let simulation = state.simulator.create_simulation(request).await?;
    Ok(Json(simulation))
}

pub async fn get_simulation(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> Result<Json<PricingSimulation>> {
    let simulation = state.simulator.get_simulation(id).await?;
    Ok(Json(simulation))
}
