use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    Json,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{
    api::state::AppState,
    attribution::{CostAllocator, JobData, PricingRates, ResourceUsage},
    models::{CostAttribution, CostAttributionQuery, CostRollup, CreateCostAttribution, PricingModel},
};

#[derive(Debug, Serialize, Deserialize)]
pub struct CreateAttributionRequest {
    pub job_id: Option<Uuid>,
    pub user_id: String,
    pub team_id: Option<String>,
    pub customer_id: Option<String>,
    pub gpu_cost: rust_decimal::Decimal,
    pub cpu_cost: rust_decimal::Decimal,
    pub network_cost: rust_decimal::Decimal,
    pub storage_cost: rust_decimal::Decimal,
    pub total_cost: rust_decimal::Decimal,
    pub period_start: DateTime<Utc>,
    pub period_end: DateTime<Utc>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CalculateAttributionRequest {
    pub job_id: Uuid,
    pub user_id: String,
    pub team_id: Option<String>,
    pub customer_id: Option<String>,
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub gpu_count: usize,
    pub gpu_type: String,
    pub network_ingress_gb: rust_decimal::Decimal,
    pub network_egress_gb: rust_decimal::Decimal,
    pub storage_gb: rust_decimal::Decimal,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RollupQueryParams {
    pub start_date: DateTime<Utc>,
    pub end_date: DateTime<Utc>,
}

pub async fn create_attribution(
    State(state): State<AppState>,
    Json(req): Json<CreateAttributionRequest>,
) -> Result<(StatusCode, Json<CostAttribution>), (StatusCode, String)> {
    let attribution = CreateCostAttribution {
        job_id: req.job_id,
        user_id: req.user_id,
        team_id: req.team_id,
        customer_id: req.customer_id,
        gpu_cost: req.gpu_cost,
        cpu_cost: req.cpu_cost,
        network_cost: req.network_cost,
        storage_cost: req.storage_cost,
        total_cost: req.total_cost,
        period_start: req.period_start,
        period_end: req.period_end,
    };

    let result = state
        .repository
        .create_attribution(&attribution)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok((StatusCode::CREATED, Json(result)))
}

pub async fn get_attribution(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
) -> Result<Json<CostAttribution>, (StatusCode, String)> {
    let result = state
        .repository
        .get_attribution(id)
        .await
        .map_err(|e| (StatusCode::NOT_FOUND, e.to_string()))?;

    Ok(Json(result))
}

pub async fn query_attributions(
    State(state): State<AppState>,
    Query(query): Query<CostAttributionQuery>,
) -> Result<Json<Vec<CostAttribution>>, (StatusCode, String)> {
    let results = state
        .repository
        .query_attributions(&query)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok(Json(results))
}

pub async fn calculate_attribution(
    State(state): State<AppState>,
    Json(req): Json<CalculateAttributionRequest>,
) -> Result<(StatusCode, Json<CostAttribution>), (StatusCode, String)> {
    // Get pricing for GPU type
    let pricing = state
        .repository
        .get_current_pricing(&req.gpu_type, PricingModel::OnDemand, req.start_time)
        .await
        .map_err(|e| (StatusCode::NOT_FOUND, format!("GPU pricing not found: {}", e)))?;

    // Parse rates from config
    let network_rate = req.network_ingress_gb
        .checked_add(req.network_egress_gb)
        .map(|_| {
            state
                .config
                .attribution
                .network_rate_per_gb
                .parse::<rust_decimal::Decimal>()
                .unwrap_or_default()
        })
        .unwrap_or_default();

    let storage_rate = state
        .config
        .attribution
        .storage_rate_per_gb_hour
        .parse::<rust_decimal::Decimal>()
        .unwrap_or_default();

    let pricing_rates = PricingRates {
        gpu_hourly_rate: pricing.hourly_rate,
        cpu_hourly_rate: rust_decimal::Decimal::ZERO,
        network_rate_per_gb: network_rate,
        storage_rate_per_gb_hour: storage_rate,
    };

    let job_data = JobData {
        job_id: req.job_id,
        user_id: req.user_id,
        team_id: req.team_id,
        customer_id: req.customer_id,
        start_time: req.start_time,
        end_time: req.end_time,
        gpu_count: req.gpu_count,
        gpu_type: req.gpu_type,
    };

    let usage = ResourceUsage {
        network_ingress_gb: req.network_ingress_gb,
        network_egress_gb: req.network_egress_gb,
        storage_gb: req.storage_gb,
    };

    let allocator = CostAllocator::new(pricing_rates.clone());
    let attribution = allocator
        .allocate_job_cost(&job_data, &usage, Some(&pricing_rates))
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let result = state
        .repository
        .create_attribution(&attribution)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok((StatusCode::CREATED, Json(result)))
}

pub async fn rollup_by_user(
    State(state): State<AppState>,
    Path(user_id): Path<String>,
    Query(params): Query<RollupQueryParams>,
) -> Result<Json<CostRollup>, (StatusCode, String)> {
    let result = state
        .repository
        .rollup_by_user(&user_id, params.start_date, params.end_date)
        .await
        .map_err(|e| (StatusCode::NOT_FOUND, e.to_string()))?;

    Ok(Json(result))
}

pub async fn rollup_by_team(
    State(state): State<AppState>,
    Path(team_id): Path<String>,
    Query(params): Query<RollupQueryParams>,
) -> Result<Json<CostRollup>, (StatusCode, String)> {
    let result = state
        .repository
        .rollup_by_team(&team_id, params.start_date, params.end_date)
        .await
        .map_err(|e| (StatusCode::NOT_FOUND, e.to_string()))?;

    Ok(Json(result))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_attribution_request_serialization() {
        let now = Utc::now();
        let req = CreateAttributionRequest {
            job_id: Some(Uuid::new_v4()),
            user_id: "user123".to_string(),
            team_id: Some("team456".to_string()),
            customer_id: None,
            gpu_cost: rust_decimal::Decimal::new(1000, 2),
            cpu_cost: rust_decimal::Decimal::ZERO,
            network_cost: rust_decimal::Decimal::ZERO,
            storage_cost: rust_decimal::Decimal::ZERO,
            total_cost: rust_decimal::Decimal::new(1000, 2),
            period_start: now,
            period_end: now + chrono::Duration::hours(1),
        };

        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("user123"));
    }
}
