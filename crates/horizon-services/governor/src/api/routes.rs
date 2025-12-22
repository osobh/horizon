use crate::api::handlers::{
    create_policy, delete_policy, evaluate, get_policy, get_policy_versions, health_check,
    list_policies, update_policy,
};
use crate::db::PolicyRepository;
use axum::routing::{delete, get, post, put};
use axum::Router;
use tower_http::trace::TraceLayer;
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

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
    let api_routes = Router::new()
        .route("/policies", post(create_policy))
        .route("/policies", get(list_policies))
        .route("/policies/:name", get(get_policy))
        .route("/policies/:name", put(update_policy))
        .route("/policies/:name", delete(delete_policy))
        .route("/policies/:name/versions", get(get_policy_versions))
        .route("/evaluate", post(evaluate))
        .with_state(repo);

    Router::new()
        .route("/health", get(health_check))
        .nest("/api/v1", api_routes)
        .merge(SwaggerUi::new("/swagger-ui").url("/api-docs/openapi.json", ApiDoc::openapi()))
        .layer(TraceLayer::new_for_http())
}
