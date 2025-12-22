use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use http_body_util::BodyExt;
use serde_json::Value;
use tower::ServiceExt;

mod common;
use common::test_app::TestApp;

// ==================== Health Check Tests ====================

#[tokio::test]
async fn test_health_check() {
    let test_app = TestApp::new().await;

    let app = scheduler::api::routes::create_router(test_app.scheduler.clone());

    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = response.into_body().collect().await.unwrap().to_bytes();
    let health: Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(health["status"], "healthy");

    test_app.cleanup().await;
}
