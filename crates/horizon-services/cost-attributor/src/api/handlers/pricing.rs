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
    models::{CreateGpuPricing, GpuPricing, GpuPricingQuery, PricingModel, UpdateGpuPricing},
};

#[derive(Debug, Serialize, Deserialize)]
pub struct CreatePricingRequest {
    pub gpu_type: String,
    pub region: Option<String>,
    pub pricing_model: PricingModel,
    pub hourly_rate: rust_decimal::Decimal,
    pub effective_start: DateTime<Utc>,
    pub effective_end: Option<DateTime<Utc>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct UpdatePricingRequest {
    pub hourly_rate: Option<rust_decimal::Decimal>,
    pub effective_end: Option<DateTime<Utc>>,
}

pub async fn create_pricing(
    State(state): State<AppState>,
    Json(req): Json<CreatePricingRequest>,
) -> Result<(StatusCode, Json<GpuPricing>), (StatusCode, String)> {
    let pricing = CreateGpuPricing {
        gpu_type: req.gpu_type,
        region: req.region,
        pricing_model: req.pricing_model,
        hourly_rate: req.hourly_rate,
        effective_start: req.effective_start,
        effective_end: req.effective_end,
    };

    let result = state
        .repository
        .create_pricing(&pricing)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok((StatusCode::CREATED, Json(result)))
}

pub async fn query_pricing(
    State(state): State<AppState>,
    Query(query): Query<GpuPricingQuery>,
) -> Result<Json<Vec<GpuPricing>>, (StatusCode, String)> {
    let results = state
        .repository
        .query_pricing(&query)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok(Json(results))
}

pub async fn get_pricing_by_type(
    State(state): State<AppState>,
    Path(gpu_type): Path<String>,
) -> Result<Json<GpuPricing>, (StatusCode, String)> {
    let result = state
        .repository
        .get_current_pricing(&gpu_type, PricingModel::OnDemand, Utc::now())
        .await
        .map_err(|e| (StatusCode::NOT_FOUND, e.to_string()))?;

    Ok(Json(result))
}

pub async fn update_pricing(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
    Json(req): Json<UpdatePricingRequest>,
) -> Result<Json<GpuPricing>, (StatusCode, String)> {
    let update = UpdateGpuPricing {
        hourly_rate: req.hourly_rate,
        effective_end: req.effective_end,
    };

    let result = state
        .repository
        .update_pricing(id, &update)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok(Json(result))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_create_pricing_request_serialization() {
        let now = Utc::now();
        let req = CreatePricingRequest {
            gpu_type: "A100".to_string(),
            region: Some("us-east-1".to_string()),
            pricing_model: PricingModel::OnDemand,
            hourly_rate: dec!(3.50),
            effective_start: now,
            effective_end: None,
        };

        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("A100"));
        assert!(json.contains("on_demand"));
    }

    #[test]
    fn test_update_pricing_request() {
        let req = UpdatePricingRequest {
            hourly_rate: Some(dec!(4.00)),
            effective_end: Some(Utc::now()),
        };

        assert!(req.hourly_rate.is_some());
        assert!(req.effective_end.is_some());
    }
}
