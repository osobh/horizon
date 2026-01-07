use crate::api::handlers::{
    create_policy, delete_policy, evaluate, get_policy, get_policy_versions, health_check,
    list_policies, update_policy,
};
use crate::db::PolicyRepository;
use axum::routing::{delete, get, post, put};
use axum::Router;
use hpc_channels::{broadcast, channels, GovernorMessage};
use std::sync::Arc;
use tokio::sync::broadcast::Sender as BroadcastSender;
use tower_http::trace::TraceLayer;
use tracing::warn;
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

/// Application state for the governor service.
#[derive(Clone)]
pub struct AppState {
    /// Policy repository for database operations.
    pub repo: PolicyRepository,
    /// Channel for policy lifecycle events (create, update, delete).
    pub policy_events: BroadcastSender<GovernorMessage>,
    /// Channel for evaluation events.
    pub evaluation_events: BroadcastSender<GovernorMessage>,
    /// Channel for access denied alerts.
    pub alert_events: BroadcastSender<GovernorMessage>,
}

impl AppState {
    /// Create a new AppState with broadcast channels.
    pub fn new(repo: PolicyRepository) -> Self {
        let policy_events = broadcast::<GovernorMessage>(channels::GOVERNOR_POLICIES, 256);
        let evaluation_events = broadcast::<GovernorMessage>(channels::GOVERNOR_EVALUATIONS, 1024);
        let alert_events = broadcast::<GovernorMessage>(channels::GOVERNOR_ALERTS, 256);

        Self {
            repo,
            policy_events,
            evaluation_events,
            alert_events,
        }
    }

    /// Publish a policy lifecycle event (non-blocking).
    pub fn publish_policy_event(&self, event: GovernorMessage) {
        if let Err(e) = self.policy_events.send(event) {
            warn!(error = ?e, "No subscribers for policy event");
        }
    }

    /// Publish an evaluation event (non-blocking).
    pub fn publish_evaluation_event(&self, event: GovernorMessage) {
        if let Err(e) = self.evaluation_events.send(event) {
            warn!(error = ?e, "No subscribers for evaluation event");
        }
    }

    /// Publish an alert event (non-blocking).
    pub fn publish_alert_event(&self, event: GovernorMessage) {
        if let Err(e) = self.alert_events.send(event) {
            warn!(error = ?e, "No subscribers for alert event");
        }
    }
}

#[derive(OpenApi)]
#[openapi(
    paths(
        crate::api::handlers::health::health_check,
        crate::api::handlers::policies::create_policy,
        crate::api::handlers::policies::list_policies,
        crate::api::handlers::policies::get_policy,
        crate::api::handlers::policies::update_policy,
        crate::api::handlers::policies::delete_policy,
        crate::api::handlers::policies::get_policy_versions,
        crate::api::handlers::evaluate::evaluate,
    ),
    components(
        schemas(
            crate::api::dto::CreatePolicyRequest,
            crate::api::dto::UpdatePolicyRequest,
            crate::api::dto::PolicyResponse,
            crate::api::dto::PolicyVersionResponse,
            crate::api::dto::EvaluateRequest,
            crate::api::dto::EvaluateResponse,
            crate::api::dto::Principal,
            crate::api::dto::Resource,
            crate::api::dto::HealthResponse,
        )
    ),
    tags(
        (name = "governor", description = "Policy management and evaluation API")
    )
)]
pub struct ApiDoc;

pub fn create_router(repo: PolicyRepository) -> Router {
    let state = Arc::new(AppState::new(repo));

    let api_routes = Router::new()
        .route("/policies", post(create_policy))
        .route("/policies", get(list_policies))
        .route("/policies/:name", get(get_policy))
        .route("/policies/:name", put(update_policy))
        .route("/policies/:name", delete(delete_policy))
        .route("/policies/:name/versions", get(get_policy_versions))
        .route("/evaluate", post(evaluate))
        .with_state(state);

    Router::new()
        .route("/health", get(health_check))
        .nest("/api/v1", api_routes)
        .merge(SwaggerUi::new("/swagger-ui").url("/api-docs/openapi.json", ApiDoc::openapi()))
        .layer(TraceLayer::new_for_http())
}
