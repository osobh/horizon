use axum::{
    routing::{get, post},
    Router,
};
use std::sync::Arc;
use tower_http::trace::TraceLayer;
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

use super::handlers::{self, AppState};

#[derive(OpenApi)]
#[openapi(
    paths(
        handlers::health_check,
        handlers::forecast_gpu_demand,
        handlers::backtest_model,
    ),
    components(schemas(
        crate::models::ForecastRequest,
        crate::models::ForecastResult,
        crate::models::ForecastPoint,
        crate::models::AccuracyMetrics,
        crate::models::BacktestRequest,
        crate::models::BacktestResult,
    )),
    tags(
        (name = "capacity-modeler", description = "GPU capacity forecasting API")
    ),
    info(
        title = "Capacity Modeler API",
        version = "0.1.0",
        description = "13-week GPU capacity forecasting service"
    )
)]
pub struct ApiDoc;

pub fn create_routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/health", get(handlers::health_check))
        .route(
            "/api/v1/forecast/gpu-demand",
            get(handlers::forecast_gpu_demand),
        )
        .route(
            "/api/v1/forecast/backtest",
            post(handlers::backtest_model),
        )
        .merge(SwaggerUi::new("/swagger-ui").url("/api-docs/openapi.json", ApiDoc::openapi()))
        .layer(TraceLayer::new_for_http())
}
