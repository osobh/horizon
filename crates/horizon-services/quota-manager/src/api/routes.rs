use axum::{
    routing::{delete, get, post, put},
    Router,
};
use std::sync::Arc;
use tokio::sync::broadcast::Sender as BroadcastSender;
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

use crate::{
    api::handlers::{allocations, health, quotas},
    db::QuotaRepository,
    service::{AllocationService, QuotaService},
};
use hpc_channels::{broadcast, channels, QuotaMessage};

#[derive(Clone)]
pub struct AppState {
    pub quota_service: QuotaService,
    pub allocation_service: AllocationService,
    /// Channel for quota lifecycle events (create, update, delete)
    pub quota_events: BroadcastSender<QuotaMessage>,
    /// Channel for allocation events (granted, released)
    pub allocation_events: BroadcastSender<QuotaMessage>,
    /// Channel for quota alerts (warnings, exceeded)
    pub alert_events: BroadcastSender<QuotaMessage>,
}

impl AppState {
    /// Publish a quota lifecycle event (non-blocking, ignores if no subscribers)
    pub fn publish_quota_event(&self, event: QuotaMessage) {
        let _ = self.quota_events.send(event);
    }

    /// Publish an allocation event (non-blocking, ignores if no subscribers)
    pub fn publish_allocation_event(&self, event: QuotaMessage) {
        let _ = self.allocation_events.send(event);
    }

    /// Publish a quota alert event (non-blocking, ignores if no subscribers)
    pub fn publish_alert_event(&self, event: QuotaMessage) {
        let _ = self.alert_events.send(event);
    }
}

pub fn create_router(repository: QuotaRepository) -> Router {
    let quota_service = QuotaService::new(repository.clone());
    let allocation_service = AllocationService::new(repository);

    // Create broadcast channels for quota events
    let quota_events = broadcast::<QuotaMessage>(channels::QUOTA_LIFECYCLE, 256);
    let allocation_events = broadcast::<QuotaMessage>(channels::QUOTA_ALLOCATIONS, 256);
    let alert_events = broadcast::<QuotaMessage>(channels::QUOTA_ALERTS, 64);

    let state = Arc::new(AppState {
        quota_service,
        allocation_service,
        quota_events,
        allocation_events,
        alert_events,
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
        .route(
            "/api/v1/allocations/check",
            post(allocations::check_allocation),
        )
        .route("/api/v1/allocations", post(allocations::create_allocation))
        .route("/api/v1/allocations/:id", get(allocations::get_allocation))
        .route(
            "/api/v1/allocations/:id",
            delete(allocations::release_allocation),
        )
        .route(
            "/api/v1/quotas/:id/allocations",
            get(allocations::list_allocations),
        )
        .with_state(state)
}
