use axum::{
    extract::{Query, State},
    Json,
};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::api::state::AppState;

#[derive(Debug, Deserialize, ToSchema)]
pub struct GpuAvailabilityQuery {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub count: Option<usize>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct GpuAvailability {
    pub gpu_type: String,
    pub available_count: usize,
    pub total_count: usize,
    pub utilization_percent: f64,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct GpuAvailabilityResponse {
    pub gpus: Vec<GpuAvailability>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Get GPU availability across the cluster
#[utoipa::path(
    get,
    path = "/api/gpu/availability",
    params(
        ("gpu_type" = Option<String>, Query, description = "Filter by GPU type"),
        ("count" = Option<usize>, Query, description = "Minimum count required")
    ),
    responses(
        (status = 200, description = "GPU availability information", body = GpuAvailabilityResponse),
        (status = 503, description = "Inventory service unavailable")
    ),
    tag = "resources"
)]
pub async fn get_gpu_availability(
    State(_state): State<AppState>,
    Query(query): Query<GpuAvailabilityQuery>,
) -> Result<Json<GpuAvailabilityResponse>, crate::HpcError> {
    // TODO: Integrate with inventory service to get actual GPU availability
    // For now, return mock data based on common GPU types

    let gpu_types = if let Some(gpu_type) = query.gpu_type {
        vec![gpu_type]
    } else {
        vec![
            "H100".to_string(),
            "A100".to_string(),
            "V100".to_string(),
            "RTX4090".to_string(),
        ]
    };

    let gpus: Vec<GpuAvailability> = gpu_types
        .into_iter()
        .map(|gpu_type| {
            // Mock data - in production, query inventory service
            let (available, total) = match gpu_type.as_str() {
                "H100" => (12, 64),
                "A100" => (24, 128),
                "V100" => (48, 256),
                "RTX4090" => (8, 32),
                _ => (0, 0),
            };

            let utilization = if total > 0 {
                ((total - available) as f64 / total as f64) * 100.0
            } else {
                0.0
            };

            GpuAvailability {
                gpu_type,
                available_count: available,
                total_count: total,
                utilization_percent: utilization,
            }
        })
        .collect();

    Ok(Json(GpuAvailabilityResponse {
        gpus,
        timestamp: chrono::Utc::now(),
    }))
}

#[derive(Debug, Deserialize, ToSchema)]
pub struct CostEstimateRequest {
    pub gpu_count: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration_hours: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cpu_cores: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_gb: Option<u64>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct CostEstimateResponse {
    pub estimated_cost: f64,
    pub currency: String,
    pub breakdown: CostBreakdown,
    pub duration_hours: f64,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct CostBreakdown {
    pub gpu_cost: f64,
    pub cpu_cost: f64,
    pub memory_cost: f64,
    pub storage_cost: f64,
}

/// Get cost estimate for a job
#[utoipa::path(
    post,
    path = "/api/jobs/estimate",
    request_body = CostEstimateRequest,
    responses(
        (status = 200, description = "Cost estimate", body = CostEstimateResponse),
        (status = 400, description = "Invalid request")
    ),
    tag = "resources"
)]
pub async fn estimate_job_cost(
    State(state): State<AppState>,
    Json(request): Json<CostEstimateRequest>,
) -> Result<Json<CostEstimateResponse>, crate::HpcError> {
    // Use pricing configuration from app state
    // In production, this can be updated via config or fetched from cost-attributor service

    let duration_hours = request.duration_hours.unwrap_or(1.0);
    let gpu_type = request.gpu_type.as_deref().unwrap_or("A100");
    let pricing = &state.pricing;

    // Get GPU hourly rate from config, with fallback to default rate
    let gpu_hourly_rate = pricing
        .gpu_hourly_rates
        .get(gpu_type)
        .copied()
        .unwrap_or(2.0); // Default rate for unknown GPU types

    let gpu_cost = gpu_hourly_rate * request.gpu_count as f64 * duration_hours;

    let cpu_cores = request.cpu_cores.unwrap_or(0);
    let cpu_cost = (cpu_cores as f64 * pricing.cpu_per_core_hour) * duration_hours;

    let memory_gb = request.memory_gb.unwrap_or(0);
    let memory_cost = (memory_gb as f64 * pricing.memory_per_gb_hour) * duration_hours;

    // Storage cost could be calculated from job spec if storage is requested
    let storage_cost = 0.0;

    let total_cost = gpu_cost + cpu_cost + memory_cost + storage_cost;

    Ok(Json(CostEstimateResponse {
        estimated_cost: total_cost,
        currency: "USD".to_string(),
        breakdown: CostBreakdown {
            gpu_cost,
            cpu_cost,
            memory_cost,
            storage_cost,
        },
        duration_hours,
    }))
}
