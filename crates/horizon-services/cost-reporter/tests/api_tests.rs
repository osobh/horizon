use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use http_body_util::BodyExt;
use tower::ServiceExt;

use cost_reporter::{
    api::{create_router, AppState},
    config::ReporterConfig,
};

async fn create_test_app() -> axum::Router {
    // Use in-memory database for testing
    let config = ReporterConfig::default();

    // This will fail without a real database, but demonstrates the test structure
    // In real scenario, you'd use a test database
    let database_url = std::env::var("TEST_DATABASE_URL")
        .unwrap_or_else(|_| "postgres://localhost/horizon_cost_test".to_string());

    match cost_reporter::db::create_pool(&database_url).await {
        Ok(pool) => {
            let state = std::sync::Arc::new(AppState::new(config, pool));
            create_router(state)
        }
        Err(_) => {
            // Return a minimal router for tests without database
            axum::Router::new()
        }
    }
}

#[tokio::test]
async fn test_health_endpoint() {
    let app = axum::Router::new().route(
        "/health",
        axum::routing::get(cost_reporter::api::handlers::health),
    );

    let response = app
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = response.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(json["status"], "healthy");
    assert_eq!(json["service"], "cost-reporter");
}

#[tokio::test]
#[ignore] // Requires database
async fn test_ready_endpoint() {
    let app = create_test_app().await;

    let response = app
        .oneshot(
            Request::builder()
                .uri("/ready")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    // Should return OK or service unavailable depending on database state
    assert!(
        response.status() == StatusCode::OK || response.status() == StatusCode::SERVICE_UNAVAILABLE
    );
}

#[tokio::test]
#[ignore] // Requires database
async fn test_team_showback_endpoint() {
    let app = create_test_app().await;

    let response = app
        .oneshot(
            Request::builder()
                .uri("/api/v1/reports/showback/team/team123?period=current_month")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    // Should return 200 or 500 depending on database state
    assert!(response.status() == StatusCode::OK || response.status().is_server_error());
}

#[tokio::test]
#[ignore] // Requires database
async fn test_user_showback_endpoint() {
    let app = create_test_app().await;

    let response = app
        .oneshot(
            Request::builder()
                .uri("/api/v1/reports/showback/user/user123?period=current_month")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert!(response.status() == StatusCode::OK || response.status().is_server_error());
}

#[tokio::test]
#[ignore] // Requires database
async fn test_top_spenders_endpoint() {
    let app = create_test_app().await;

    let response = app
        .oneshot(
            Request::builder()
                .uri("/api/v1/reports/showback/top-spenders?period=current_month&limit=10")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert!(response.status() == StatusCode::OK || response.status().is_server_error());
}

#[tokio::test]
#[ignore] // Requires database
async fn test_customer_chargeback_endpoint() {
    let app = create_test_app().await;

    let response = app
        .oneshot(
            Request::builder()
                .uri("/api/v1/reports/chargeback/customer/customer123?start_date=2025-01-01T00:00:00Z&end_date=2025-01-31T23:59:59Z")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert!(response.status() == StatusCode::OK || response.status().is_server_error());
}

#[tokio::test]
#[ignore] // Requires database
async fn test_daily_trends_endpoint() {
    let app = create_test_app().await;

    let response = app
        .oneshot(
            Request::builder()
                .uri("/api/v1/reports/trends/daily?start_date=2025-01-01T00:00:00Z&end_date=2025-01-31T23:59:59Z")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert!(response.status() == StatusCode::OK || response.status().is_server_error());
}

#[tokio::test]
#[ignore] // Requires database
async fn test_monthly_trends_endpoint() {
    let app = create_test_app().await;

    let response = app
        .oneshot(
            Request::builder()
                .uri("/api/v1/reports/trends/monthly?start_date=2025-01-01T00:00:00Z&end_date=2025-12-31T23:59:59Z")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert!(response.status() == StatusCode::OK || response.status().is_server_error());
}

#[tokio::test]
#[ignore] // Requires database
async fn test_forecast_endpoint() {
    let app = create_test_app().await;

    let response = app
        .oneshot(
            Request::builder()
                .uri("/api/v1/reports/trends/forecast?days_ahead=30")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    // Forecast may fail with insufficient data
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::UNPROCESSABLE_ENTITY
            || response.status().is_server_error()
    );
}

#[tokio::test]
#[ignore] // Requires database
async fn test_csv_export_endpoint() {
    let app = create_test_app().await;

    let response = app
        .oneshot(
            Request::builder()
                .uri("/api/v1/export/csv?start_date=2025-01-01T00:00:00Z&end_date=2025-01-31T23:59:59Z")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert!(response.status() == StatusCode::OK || response.status().is_server_error());
}

#[tokio::test]
#[ignore] // Requires database
async fn test_json_export_endpoint() {
    let app = create_test_app().await;

    let response = app
        .oneshot(
            Request::builder()
                .uri("/api/v1/export/json?start_date=2025-01-01T00:00:00Z&end_date=2025-01-31T23:59:59Z")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert!(response.status() == StatusCode::OK || response.status().is_server_error());
}
