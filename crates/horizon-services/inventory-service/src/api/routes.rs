use axum::{
    routing::{get, post},
    Router,
};
use sqlx::PgPool;
use tower_http::{compression::CompressionLayer, trace::TraceLayer};
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

use super::handlers;
use super::state::AppState;
use crate::api::models::{
    CreateAssetRequest, DiscoverAssetsRequest, HealthResponse, ListAssetsQuery,
    ListAssetsResponse, ListHistoryResponse, Pagination, UpdateAssetRequest,
};
use crate::models::{Asset, AssetHistory, AssetMetrics, AssetStatus, AssetType, ProviderType};

#[derive(OpenApi)]
#[openapi(
    paths(
        handlers::assets::list_assets,
        handlers::assets::create_asset,
        handlers::assets::get_asset,
        handlers::assets::update_asset,
        handlers::assets::decommission_asset,
        handlers::assets::discover_assets,
        handlers::history::list_asset_history,
        handlers::metrics::get_asset_metrics,
        handlers::health::health_check,
    ),
    components(schemas(
        Asset,
        AssetHistory,
        AssetMetrics,
        AssetType,
        AssetStatus,
        ProviderType,
        CreateAssetRequest,
        UpdateAssetRequest,
        ListAssetsQuery,
        ListAssetsResponse,
        ListHistoryResponse,
        Pagination,
        DiscoverAssetsRequest,
        HealthResponse,
    )),
    tags(
        (name = "assets", description = "Asset management endpoints"),
        (name = "metrics", description = "Asset metrics endpoints"),
        (name = "history", description = "Asset history endpoints"),
        (name = "health", description = "Health check endpoints"),
    ),
    info(
        title = "Inventory Service API",
        version = "0.1.0",
        description = "GPU asset inventory management service for Horizon platform",
    )
)]
struct ApiDoc;

pub fn create_routes(pool: PgPool) -> Router {
    let state = AppState::new(pool);

    let api_routes = Router::new()
        .route("/api/v1/assets", get(handlers::assets::list_assets).post(handlers::assets::create_asset))
        .route(
            "/api/v1/assets/:id",
            get(handlers::assets::get_asset)
                .put(handlers::assets::update_asset)
                .delete(handlers::assets::decommission_asset),
        )
        .route("/api/v1/assets/discover", post(handlers::assets::discover_assets))
        .route("/api/v1/assets/:id/history", get(handlers::history::list_asset_history))
        .route("/api/v1/assets/:id/metrics", get(handlers::metrics::get_asset_metrics));

    Router::new()
        .merge(SwaggerUi::new("/swagger-ui").url("/api-docs/openapi.json", ApiDoc::openapi()))
        .merge(api_routes)
        .route("/health", get(handlers::health::health_check))
        .layer(CompressionLayer::new())
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_create_routes() {
        let pool = PgPool::connect_lazy("postgres://localhost/test").unwrap();
        let router = create_routes(pool);
        assert!(std::mem::size_of_val(&router) > 0);
    }

    #[test]
    fn test_openapi_generation() {
        let openapi = ApiDoc::openapi();
        assert_eq!(openapi.info.title, "Inventory Service API");
        assert!(!openapi.paths.paths.is_empty());
    }
}
