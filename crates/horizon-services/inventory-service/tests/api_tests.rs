mod helpers;

use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tower::ServiceExt;

use inventory_service::{
    api::models::ListAssetsResponse,
    models::{Asset, AssetStatus, AssetType, ProviderType},
};

use helpers::{clean_database, create_test_app, setup_test_db};

#[tokio::test]
#[ignore] // Requires PostgreSQL instance
async fn test_health_check() {
    let pool = setup_test_db().await;
    let app = create_test_app(pool);

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
    assert_eq!(health["database"], "connected");
}

#[tokio::test]
#[ignore] // Requires PostgreSQL instance
async fn test_create_asset() {
    let pool = setup_test_db().await;
    let app = create_test_app(pool.clone());

    let payload = json!({
        "asset_type": "gpu",
        "provider": "baremetal",
        "hostname": "gpu-node-01",
        "location": "us-west-1a",
        "metadata": {
            "gpu_model": "H100",
            "gpu_memory_gb": 80
        }
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/v1/assets")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&payload).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    let status = response.status();
    let body = response.into_body().collect().await.unwrap().to_bytes();

    if status != StatusCode::CREATED {
        let error_text = String::from_utf8_lossy(&body);
        eprintln!("Error response: {}", error_text);
        panic!("Expected CREATED (201), got {}", status);
    }
    assert_eq!(status, StatusCode::CREATED);

    let asset: Asset = serde_json::from_slice(&body).unwrap();

    assert_eq!(asset.asset_type, AssetType::Gpu);
    assert_eq!(asset.provider, ProviderType::Baremetal);
    assert_eq!(asset.hostname, Some("gpu-node-01".to_string()));
    assert_eq!(asset.location, Some("us-west-1a".to_string()));

    clean_database(&pool).await;
}

#[tokio::test]
#[ignore] // Requires PostgreSQL instance
async fn test_get_asset() {
    let pool = setup_test_db().await;
    let app = create_test_app(pool.clone());

    let payload = json!({
        "asset_type": "gpu",
        "provider": "baremetal",
        "hostname": "gpu-node-02"
    });

    let create_response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/v1/assets")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&payload).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    let body = create_response.into_body().collect().await.unwrap().to_bytes();
    let created_asset: Asset = serde_json::from_slice(&body).unwrap();

    let get_response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri(format!("/api/v1/assets/{}", created_asset.id))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(get_response.status(), StatusCode::OK);

    let body = get_response.into_body().collect().await.unwrap().to_bytes();
    let retrieved_asset: Asset = serde_json::from_slice(&body).unwrap();

    assert_eq!(retrieved_asset.id, created_asset.id);
    assert_eq!(retrieved_asset.hostname, Some("gpu-node-02".to_string()));

    clean_database(&pool).await;
}

#[tokio::test]
#[ignore] // Requires PostgreSQL instance
async fn test_list_assets() {
    let pool = setup_test_db().await;
    let app = create_test_app(pool.clone());

    for i in 0..5 {
        let payload = json!({
            "asset_type": "gpu",
            "provider": "baremetal",
            "hostname": format!("gpu-node-{:02}", i),
            "status": "available"
        });

        app.clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/v1/assets")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_string(&payload).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
    }

    let list_response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/api/v1/assets?page=1&page_size=10")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    let status = list_response.status();
    let body = list_response.into_body().collect().await.unwrap().to_bytes();

    if status != StatusCode::OK {
        let error_text = String::from_utf8_lossy(&body);
        eprintln!("Error response: {}", error_text);
        panic!("Expected OK (200), got {}", status);
    }

    let list_response: ListAssetsResponse = serde_json::from_slice(&body).unwrap();

    assert_eq!(list_response.data.len(), 5);
    assert_eq!(list_response.pagination.total_items, 5);
    assert_eq!(list_response.pagination.page, 1);

    clean_database(&pool).await;
}

#[tokio::test]
#[ignore] // Requires PostgreSQL instance
async fn test_list_assets_with_filters() {
    let pool = setup_test_db().await;
    let app = create_test_app(pool.clone());

    let payload_gpu = json!({
        "asset_type": "gpu",
        "provider": "baremetal",
        "hostname": "gpu-node-01",
        "status": "available"
    });

    let payload_node = json!({
        "asset_type": "node",
        "provider": "aws",
        "hostname": "compute-node-01",
        "status": "available"
    });

    app.clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/v1/assets")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&payload_gpu).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    app.clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/v1/assets")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&payload_node).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    let list_response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/api/v1/assets?asset_type=gpu")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    let body = list_response.into_body().collect().await.unwrap().to_bytes();
    let list_response: ListAssetsResponse = serde_json::from_slice(&body).unwrap();

    assert_eq!(list_response.data.len(), 1);
    assert!(list_response.data.iter().all(|a| a.asset_type == AssetType::Gpu));

    clean_database(&pool).await;
}

#[tokio::test]
#[ignore] // Requires PostgreSQL instance
async fn test_update_asset() {
    let pool = setup_test_db().await;
    let app = create_test_app(pool.clone());

    let create_payload = json!({
        "asset_type": "gpu",
        "provider": "baremetal",
        "hostname": "gpu-node-update",
        "status": "available"
    });

    let create_response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/v1/assets")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&create_payload).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    let body = create_response.into_body().collect().await.unwrap().to_bytes();
    let created_asset: Asset = serde_json::from_slice(&body).unwrap();

    let update_payload = json!({
        "status": "maintenance",
        "metadata": {
            "reason": "scheduled maintenance"
        }
    });

    let update_response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("PUT")
                .uri(format!("/api/v1/assets/{}", created_asset.id))
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&update_payload).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    let status = update_response.status();
    let body = update_response.into_body().collect().await.unwrap().to_bytes();

    if status != StatusCode::OK {
        let error_text = String::from_utf8_lossy(&body);
        eprintln!("Error response: {}", error_text);
        panic!("Expected OK (200), got {}", status);
    }

    let updated_asset: Asset = serde_json::from_slice(&body).unwrap();

    assert_eq!(updated_asset.status, AssetStatus::Maintenance);

    clean_database(&pool).await;
}

#[tokio::test]
#[ignore] // Requires PostgreSQL instance
async fn test_decommission_asset() {
    let pool = setup_test_db().await;
    let app = create_test_app(pool.clone());

    let create_payload = json!({
        "asset_type": "gpu",
        "provider": "baremetal",
        "hostname": "gpu-node-decommission"
    });

    let create_response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/v1/assets")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&create_payload).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    let body = create_response.into_body().collect().await.unwrap().to_bytes();
    let created_asset: Asset = serde_json::from_slice(&body).unwrap();

    let delete_response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri(format!("/api/v1/assets/{}", created_asset.id))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    let status = delete_response.status();
    if status != StatusCode::NO_CONTENT {
        let body = delete_response.into_body().collect().await.unwrap().to_bytes();
        let error_text = String::from_utf8_lossy(&body);
        eprintln!("Error response: {}", error_text);
        panic!("Expected NO_CONTENT (204), got {}", status);
    }
    assert_eq!(status, StatusCode::NO_CONTENT);

    let get_response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri(format!("/api/v1/assets/{}", created_asset.id))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    let body = get_response.into_body().collect().await.unwrap().to_bytes();
    let asset: Asset = serde_json::from_slice(&body).unwrap();

    assert_eq!(asset.status, AssetStatus::Decommissioned);
    assert!(asset.decommissioned_at.is_some());

    clean_database(&pool).await;
}

#[tokio::test]
#[ignore] // Requires PostgreSQL instance
async fn test_discover_assets() {
    let pool = setup_test_db().await;
    let app = create_test_app(pool.clone());

    let payload = json!({
        "node": {
            "hostname": "gpu-node-discover",
            "provider": "baremetal",
            "location": "us-west-1a",
            "metadata": {
                "cpu_model": "AMD EPYC 9654",
                "cpu_cores": 96
            }
        },
        "gpus": [
            {
                "gpu_uuid": "GPU-12345678",
                "metadata": {
                    "gpu_model": "H100",
                    "gpu_memory_gb": 80
                }
            },
            {
                "gpu_uuid": "GPU-87654321",
                "metadata": {
                    "gpu_model": "H100",
                    "gpu_memory_gb": 80
                }
            }
        ]
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/v1/assets/discover")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&payload).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = response.into_body().collect().await.unwrap().to_bytes();
    let discover_response: Value = serde_json::from_slice(&body).unwrap();

    assert!(discover_response["node_id"].is_string());
    assert_eq!(discover_response["gpu_ids"].as_array().unwrap().len(), 2);
    assert_eq!(discover_response["created"], 3); // 1 node + 2 GPUs

    clean_database(&pool).await;
}

#[tokio::test]
#[ignore] // Requires PostgreSQL instance
async fn test_asset_not_found() {
    let pool = setup_test_db().await;
    let app = create_test_app(pool);

    let random_id = uuid::Uuid::new_v4();

    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri(format!("/api/v1/assets/{}", random_id))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
#[ignore] // Requires PostgreSQL instance
async fn test_pagination() {
    let pool = setup_test_db().await;
    let app = create_test_app(pool.clone());

    for i in 0..25 {
        let payload = json!({
            "asset_type": "gpu",
            "provider": "baremetal",
            "hostname": format!("gpu-{:03}", i)
        });

        app.clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/v1/assets")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_string(&payload).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
    }

    let page1_response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/api/v1/assets?page=1&page_size=10")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    let body = page1_response.into_body().collect().await.unwrap().to_bytes();
    let page1: ListAssetsResponse = serde_json::from_slice(&body).unwrap();

    assert_eq!(page1.data.len(), 10);
    assert_eq!(page1.pagination.page, 1);
    assert_eq!(page1.pagination.total_items, 25);
    assert_eq!(page1.pagination.total_pages, 3);

    let page2_response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/api/v1/assets?page=2&page_size=10")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    let body = page2_response.into_body().collect().await.unwrap().to_bytes();
    let page2: ListAssetsResponse = serde_json::from_slice(&body).unwrap();

    assert_eq!(page2.data.len(), 10);
    assert_eq!(page2.pagination.page, 2);

    clean_database(&pool).await;
}
