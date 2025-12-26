use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tower::ServiceExt;

mod common;
use common::test_app::TestApp;

// ==================== Validation & Error Tests ====================

#[tokio::test]
async fn test_submit_job_invalid_request() {
    let test_app = TestApp::new().await;

    let invalid_request = json!({
        "gpu_count": 2,
        "priority": "High"
    });

    let app = scheduler::api::routes::create_router(test_app.scheduler.clone());

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/v1/jobs")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&invalid_request).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert!(
        response.status() == StatusCode::BAD_REQUEST || response.status() == StatusCode::UNPROCESSABLE_ENTITY,
        "Expected 400 or 422, got {}",
        response.status()
    );

    test_app.cleanup().await;
}

#[tokio::test]
async fn test_submit_job_zero_gpus() {
    let test_app = TestApp::new().await;

    let invalid_request = json!({
        "user_id": "user1",
        "gpu_count": 0,
    });

    let app = scheduler::api::routes::create_router(test_app.scheduler.clone());

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/v1/jobs")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&invalid_request).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert!(
        response.status() == StatusCode::BAD_REQUEST || response.status() == StatusCode::UNPROCESSABLE_ENTITY,
        "Expected 400 or 422 for zero GPU count, got {}",
        response.status()
    );

    test_app.cleanup().await;
}

#[tokio::test]
async fn test_submit_job_large_gpu_count() {
    let test_app = TestApp::new().await;

    let request = json!({
        "user_id": "user1",
        "gpu_count": 1000,
        "priority": "High",
    });

    let app = scheduler::api::routes::create_router(test_app.scheduler.clone());

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/v1/jobs")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&request).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    // Should accept the request even if we don't have resources
    assert_eq!(response.status(), StatusCode::CREATED);

    test_app.cleanup().await;
}

#[tokio::test]
async fn test_get_job_invalid_uuid() {
    let test_app = TestApp::new().await;

    let app = scheduler::api::routes::create_router(test_app.scheduler.clone());

    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/api/v1/jobs/not-a-valid-uuid")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    // Axum should return 400 for invalid UUID format
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);

    test_app.cleanup().await;
}

#[tokio::test]
async fn test_cancel_already_cancelled_job() {
    let test_app = TestApp::new().await;

    let repository = scheduler::db::JobRepository::new(test_app.pool.clone());
    let job = scheduler::models::Job::builder()
        .user_id("user1")
        .gpu_count(2)
        .build()
        .unwrap();
    let created_job = repository.create(&job).await.unwrap();

    let app = scheduler::api::routes::create_router(test_app.scheduler.clone());

    // Cancel once
    let response1 = app
        .clone()
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri(&format!("/api/v1/jobs/{}", created_job.id))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response1.status(), StatusCode::OK);

    // Try to cancel again - should fail with state transition error
    let response2 = app
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri(&format!("/api/v1/jobs/{}", created_job.id))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    // Should return 400 BAD_REQUEST for invalid state transition
    assert_eq!(response2.status(), StatusCode::BAD_REQUEST);

    test_app.cleanup().await;
}

#[tokio::test]
async fn test_submit_job_malformed_json() {
    let test_app = TestApp::new().await;

    let app = scheduler::api::routes::create_router(test_app.scheduler.clone());

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/v1/jobs")
                .header("content-type", "application/json")
                .body(Body::from("{invalid json}"))
                .unwrap(),
        )
        .await
        .unwrap();

    assert!(
        response.status() == StatusCode::BAD_REQUEST || response.status() == StatusCode::UNPROCESSABLE_ENTITY,
        "Expected 400 or 422 for malformed JSON, got {}",
        response.status()
    );

    test_app.cleanup().await;
}
