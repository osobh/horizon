use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tower::ServiceExt;

mod common;
use common::test_app::TestApp;

// ==================== Queue & Priority Tests ====================

#[tokio::test]
async fn test_get_queue_status() {
    let test_app = TestApp::new().await;

    // Submit some jobs through the scheduler
    let job1 = scheduler::models::Job::builder()
        .user_id("user1")
        .gpu_count(2)
        .priority(scheduler::models::Priority::High)
        .build()
        .unwrap();
    test_app.scheduler.submit_job(job1).await.unwrap();

    let job2 = scheduler::models::Job::builder()
        .user_id("user2")
        .gpu_count(1)
        .priority(scheduler::models::Priority::Normal)
        .build()
        .unwrap();
    test_app.scheduler.submit_job(job2).await.unwrap();

    let app = scheduler::api::routes::create_router(test_app.scheduler.clone());

    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/api/v1/queue")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = response.into_body().collect().await.unwrap().to_bytes();
    let queue_stats: Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(queue_stats["total"], 2);
    assert_eq!(queue_stats["high_priority"], 1);
    assert_eq!(queue_stats["normal_priority"], 1);

    test_app.cleanup().await;
}

#[tokio::test]
async fn test_queue_status_empty() {
    let test_app = TestApp::new().await;

    let app = scheduler::api::routes::create_router(test_app.scheduler.clone());

    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/api/v1/queue")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = response.into_body().collect().await.unwrap().to_bytes();
    let queue_stats: Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(queue_stats["total"], 0);
    assert_eq!(queue_stats["high_priority"], 0);
    assert_eq!(queue_stats["normal_priority"], 0);
    assert_eq!(queue_stats["low_priority"], 0);

    test_app.cleanup().await;
}

#[tokio::test]
async fn test_list_jobs_by_priority() {
    let test_app = TestApp::new().await;

    let repository = scheduler::db::JobRepository::new(test_app.pool.clone());

    let job1 = scheduler::models::Job::builder()
        .user_id("user1")
        .gpu_count(2)
        .priority(scheduler::models::Priority::High)
        .build()
        .unwrap();
    repository.create(&job1).await.unwrap();

    let job2 = scheduler::models::Job::builder()
        .user_id("user2")
        .gpu_count(1)
        .priority(scheduler::models::Priority::High)
        .build()
        .unwrap();
    repository.create(&job2).await.unwrap();

    let job3 = scheduler::models::Job::builder()
        .user_id("user3")
        .gpu_count(4)
        .priority(scheduler::models::Priority::Normal)
        .build()
        .unwrap();
    repository.create(&job3).await.unwrap();

    let app = scheduler::api::routes::create_router(test_app.scheduler.clone());

    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/api/v1/jobs?priority=High")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = response.into_body().collect().await.unwrap().to_bytes();
    let list_response: Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(list_response["total"], 2);
    for job in list_response["jobs"].as_array().unwrap() {
        assert_eq!(job["priority"], "High");
    }

    test_app.cleanup().await;
}

#[tokio::test]
async fn test_list_jobs_multiple_filters() {
    let test_app = TestApp::new().await;

    let repository = scheduler::db::JobRepository::new(test_app.pool.clone());

    let job1 = scheduler::models::Job::builder()
        .user_id("alice")
        .gpu_count(2)
        .priority(scheduler::models::Priority::High)
        .build()
        .unwrap();
    repository.create(&job1).await.unwrap();

    let job2 = scheduler::models::Job::builder()
        .user_id("bob")
        .gpu_count(4)
        .priority(scheduler::models::Priority::Normal)
        .build()
        .unwrap();
    repository.create(&job2).await.unwrap();

    let job3 = scheduler::models::Job::builder()
        .user_id("alice")
        .gpu_count(1)
        .priority(scheduler::models::Priority::Low)
        .build()
        .unwrap();
    repository.create(&job3).await.unwrap();

    let app = scheduler::api::routes::create_router(test_app.scheduler.clone());

    // Filter by user_id
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/api/v1/jobs?user_id=alice")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = response.into_body().collect().await.unwrap().to_bytes();
    let list_response: Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(list_response["total"], 2);

    test_app.cleanup().await;
}
