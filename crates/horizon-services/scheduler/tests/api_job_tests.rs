use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tower::ServiceExt;

mod common;
use common::test_app::TestApp;

// ==================== Basic Job Operations ====================

#[tokio::test]
async fn test_submit_job() {
    let test_app = TestApp::new().await;

    let job_request = json!({
        "user_id": "user1",
        "gpu_count": 2,
        "priority": "High",
        "command": "python train.py"
    });

    let app = scheduler::api::routes::create_router(test_app.scheduler.clone());

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/v1/jobs")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&job_request).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    let status = response.status();
    let body = response.into_body().collect().await.unwrap().to_bytes();

    if status != StatusCode::CREATED {
        let error: Value = serde_json::from_slice(&body).unwrap_or(json!({"error": "Unknown error"}));
        panic!(
            "Expected 201 CREATED but got {}. Error: {}",
            status,
            error
        );
    }

    let job: Value = serde_json::from_slice(&body).unwrap();

    assert!(job["id"].is_string());
    assert_eq!(job["user_id"], "user1");
    assert_eq!(job["state"], "Queued");
    assert_eq!(job["priority"], "High");
    assert_eq!(job["resources"]["gpu_count"], 2);

    test_app.cleanup().await;
}

#[tokio::test]
async fn test_get_job_by_id() {
    let test_app = TestApp::new().await;

    let repository = scheduler::db::JobRepository::new(test_app.pool.clone());
    let job = scheduler::models::Job::builder()
        .user_id("user1")
        .job_name("test-job")
        .gpu_count(4)
        .priority(scheduler::models::Priority::Normal)
        .command("python script.py")
        .build()
        .unwrap();

    let created_job = repository.create(&job).await.unwrap();

    let app = scheduler::api::routes::create_router(test_app.scheduler.clone());

    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri(&format!("/api/v1/jobs/{}", created_job.id))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = response.into_body().collect().await.unwrap().to_bytes();
    let job_response: Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(job_response["id"], created_job.id.to_string());
    assert_eq!(job_response["user_id"], "user1");
    assert_eq!(job_response["job_name"], "test-job");

    test_app.cleanup().await;
}

#[tokio::test]
async fn test_get_job_not_found() {
    let test_app = TestApp::new().await;

    let app = scheduler::api::routes::create_router(test_app.scheduler.clone());

    let fake_id = uuid::Uuid::new_v4();
    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri(&format!("/api/v1/jobs/{}", fake_id))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::NOT_FOUND);

    let body = response.into_body().collect().await.unwrap().to_bytes();
    let error: Value = serde_json::from_slice(&body).unwrap();

    assert!(error["error"].as_str().unwrap().contains("not found"));

    test_app.cleanup().await;
}

#[tokio::test]
async fn test_list_all_jobs() {
    let test_app = TestApp::new().await;

    let repository = scheduler::db::JobRepository::new(test_app.pool.clone());

    for i in 1..=3 {
        let job = scheduler::models::Job::builder()
            .user_id(&format!("user{}", i))
            .gpu_count(i)
            .build()
            .unwrap();
        repository.create(&job).await.unwrap();
    }

    let app = scheduler::api::routes::create_router(test_app.scheduler.clone());

    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/api/v1/jobs")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    let status = response.status();
    let body = response.into_body().collect().await.unwrap().to_bytes();

    if status != StatusCode::OK {
        let error: Value = serde_json::from_slice(&body).unwrap_or(json!({"error": "Unknown"}));
        panic!("Expected 200 OK but got {}. Error: {}", status, error);
    }

    let list_response: Value = serde_json::from_slice(&body).unwrap();

    assert!(list_response["jobs"].is_array());
    assert_eq!(list_response["jobs"].as_array().unwrap().len(), 3);
    assert_eq!(list_response["total"], 3);

    test_app.cleanup().await;
}

#[tokio::test]
async fn test_list_jobs_filtered_by_state() {
    let test_app = TestApp::new().await;

    let repository = scheduler::db::JobRepository::new(test_app.pool.clone());

    let job1 = scheduler::models::Job::builder()
        .user_id("user1")
        .gpu_count(2)
        .build()
        .unwrap();
    repository.create(&job1).await.unwrap();

    let mut job2 = scheduler::models::Job::builder()
        .user_id("user2")
        .gpu_count(2)
        .build()
        .unwrap();
    job2 = repository.create(&job2).await.unwrap();
    job2.transition_to(scheduler::models::JobState::Scheduled).unwrap();
    repository.update(&job2).await.unwrap();

    let app = scheduler::api::routes::create_router(test_app.scheduler.clone());

    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/api/v1/jobs?state=Queued")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = response.into_body().collect().await.unwrap().to_bytes();
    let list_response: Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(list_response["jobs"].as_array().unwrap().len(), 1);
    assert_eq!(list_response["jobs"][0]["state"], "Queued");
    assert_eq!(list_response["total"], 1);

    test_app.cleanup().await;
}

#[tokio::test]
async fn test_cancel_job() {
    let test_app = TestApp::new().await;

    let repository = scheduler::db::JobRepository::new(test_app.pool.clone());
    let job = scheduler::models::Job::builder()
        .user_id("user1")
        .gpu_count(2)
        .build()
        .unwrap();
    let created_job = repository.create(&job).await.unwrap();

    let app = scheduler::api::routes::create_router(test_app.scheduler.clone());

    let response = app
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri(&format!("/api/v1/jobs/{}", created_job.id))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = response.into_body().collect().await.unwrap().to_bytes();
    let cancelled_job: Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(cancelled_job["state"], "Cancelled");

    test_app.cleanup().await;
}

#[tokio::test]
async fn test_cancel_job_not_found() {
    let test_app = TestApp::new().await;

    let app = scheduler::api::routes::create_router(test_app.scheduler.clone());

    let fake_id = uuid::Uuid::new_v4();
    let response = app
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri(&format!("/api/v1/jobs/{}", fake_id))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::NOT_FOUND);

    test_app.cleanup().await;
}

#[tokio::test]
async fn test_submit_job_all_fields() {
    let test_app = TestApp::new().await;

    let job_request = json!({
        "user_id": "user1",
        "job_name": "complex-training-job",
        "gpu_count": 8,
        "gpu_type": "H100",
        "cpu_cores": 64,
        "memory_gb": 512,
        "priority": "High",
        "command": "python train.py --epochs 100",
        "working_dir": "/home/user1/project",
        "environment": {
            "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
            "PYTHONPATH": "/usr/local/lib/python3.9",
        }
    });

    let app = scheduler::api::routes::create_router(test_app.scheduler.clone());

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/v1/jobs")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&job_request).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::CREATED);

    let body = response.into_body().collect().await.unwrap().to_bytes();
    let job_response: Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(job_response["job_name"], "complex-training-job");
    assert_eq!(job_response["resources"]["gpu_type"], "H100");
    assert_eq!(job_response["resources"]["cpu_cores"], 64);
    assert_eq!(job_response["resources"]["memory_gb"], 512);
    assert_eq!(job_response["command"], "python train.py --epochs 100");

    test_app.cleanup().await;
}
