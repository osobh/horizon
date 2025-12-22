use axum::{
    routing::{delete, get, post, put},
    Router,
};
use std::sync::Arc;
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

use crate::{
    api::handlers::{allocations, health, quotas},
    db::QuotaRepository,
    service::{AllocationService, QuotaService},
};

#[derive(Clone)]
pub struct AppState {
    pub quota_service: QuotaService,
    pub allocation_service: AllocationService,
}

pub fn create_router(repository: QuotaRepository) -> Router {
    let quota_service = QuotaService::new(repository.clone());
    let allocation_service = AllocationService::new(repository);

    let state = Arc::new(AppState {
        quota_service,
        allocation_service,
    });

    #[derive(OpenApi)]
    #[openapi(
        paths(
            health::health,
        ),
        components(schemas(
            crate::api::dto::HealthResponse,
            crate::models::Quota,
            crate::models::CreateQuotaRequest,
            crate::models::UpdateQuotaRequest,
            crate::models::Allocation,
            crate::models::AllocationCheckRequest,
            crate::models::AllocationCheckResponse,
            crate::models::QuotaUsageStats,
            crate::models::UsageHistory,
        )),
        tags(
            (name = "quota-manager", description = "Quota Management API")
        )
    )]
    struct ApiDoc;

    Router::new()
        .merge(SwaggerUi::new("/swagger-ui").url("/api-docs/openapi.json", ApiDoc::openapi()))
        .route("/health", get(health::health))
        .route("/api/v1/quotas", post(quotas::create_quota))
        .route("/api/v1/quotas", get(quotas::list_quotas))
        .route("/api/v1/quotas/:id", get(quotas::get_quota))
        .route("/api/v1/quotas/:id", put(quotas::update_quota))
        .route("/api/v1/quotas/:id", delete(quotas::delete_quota))
        .route("/api/v1/quotas/:id/usage", get(quotas::get_usage_stats))
        .route("/api/v1/quotas/:id/history", get(quotas::get_usage_history))
        .route("/api/v1/allocations/check", post(allocations::check_allocation))
        .route("/api/v1/allocations", post(allocations::create_allocation))
        .route("/api/v1/allocations/:id", get(allocations::get_allocation))
        .route("/api/v1/allocations/:id", delete(allocations::release_allocation))
        .route("/api/v1/quotas/:id/allocations", get(allocations::list_allocations))
        .with_state(state)
}
