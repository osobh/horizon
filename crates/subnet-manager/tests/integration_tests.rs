//! Integration tests for subnet-manager API
//!
//! Tests full API workflows including subnet creation, node assignment,
//! policy evaluation, and migration flows.

use axum::body::Body;
use axum::http::{header, Method, Request, StatusCode};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use std::sync::Arc;
use subnet_manager::api::{create_router, AppState};
use tower::ServiceExt;

/// Helper to create a test app
fn create_test_app() -> axum::Router {
    let state = Arc::new(AppState::new());
    create_router(state)
}

/// Helper to make a JSON request
fn json_request(method: Method, uri: &str, body: Option<Value>) -> Request<Body> {
    let builder = Request::builder()
        .method(method)
        .uri(uri)
        .header(header::CONTENT_TYPE, "application/json");

    match body {
        Some(json) => builder.body(Body::from(json.to_string())).unwrap(),
        None => builder.body(Body::empty()).unwrap(),
    }
}

/// Helper to extract JSON from response
async fn response_json(response: axum::response::Response) -> Value {
    let body = response.into_body().collect().await.unwrap().to_bytes();
    serde_json::from_slice(&body).unwrap_or(Value::Null)
}

// ============================================================================
// Health Check Tests
// ============================================================================

#[tokio::test]
async fn test_health_endpoint() {
    let app = create_test_app();

    let response = app
        .oneshot(json_request(Method::GET, "/health", None))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_ready_endpoint() {
    let app = create_test_app();

    let response = app
        .oneshot(json_request(Method::GET, "/ready", None))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

// ============================================================================
// Subnet CRUD Tests
// ============================================================================

#[tokio::test]
async fn test_create_subnet() {
    let state = Arc::new(AppState::new());
    let app = create_router(state.clone());

    let create_body = json!({
        "name": "test-tenant-subnet",
        "description": "Test subnet for tenant isolation",
        "purpose": "tenant",
        "prefix_len": 24
    });

    let response = app
        .oneshot(json_request(Method::POST, "/api/v1/subnets", Some(create_body)))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::CREATED);

    let json = response_json(response).await;
    assert_eq!(json["name"], "test-tenant-subnet");
    assert_eq!(json["purpose"], "tenant");
    assert!(json["id"].as_str().is_some());
}

#[tokio::test]
async fn test_create_subnet_from_template() {
    let state = Arc::new(AppState::new());
    let app = create_router(state.clone());

    // First list templates to get a valid template ID
    let list_response = app
        .clone()
        .oneshot(json_request(Method::GET, "/api/v1/templates", None))
        .await
        .unwrap();

    let templates = response_json(list_response).await;
    let template_id = templates[0]["id"].as_str().unwrap();

    // Create subnet from template
    let create_body = json!({
        "name": "datacenter-nodes",
        "template_id": template_id
    });

    let response = app
        .oneshot(json_request(Method::POST, "/api/v1/subnets", Some(create_body)))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::CREATED);

    let json = response_json(response).await;
    assert_eq!(json["name"], "datacenter-nodes");
}

#[tokio::test]
async fn test_list_subnets_with_filter() {
    let state = Arc::new(AppState::new());
    let app = create_router(state.clone());

    // Create a tenant subnet
    let create_body = json!({
        "name": "tenant-1",
        "purpose": "tenant",
        "prefix_len": 24
    });

    app.clone()
        .oneshot(json_request(Method::POST, "/api/v1/subnets", Some(create_body)))
        .await
        .unwrap();

    // Create a node_type subnet
    let create_body2 = json!({
        "name": "datacenter-1",
        "purpose": "node_type",
        "node_type": "data_center",
        "prefix_len": 20
    });

    app.clone()
        .oneshot(json_request(Method::POST, "/api/v1/subnets", Some(create_body2)))
        .await
        .unwrap();

    // List all subnets
    let response = app
        .clone()
        .oneshot(json_request(Method::GET, "/api/v1/subnets", None))
        .await
        .unwrap();

    let json = response_json(response).await;
    assert!(json["total"].as_u64().unwrap() >= 2);

    // Filter by purpose
    let response = app
        .oneshot(json_request(
            Method::GET,
            "/api/v1/subnets?purpose=tenant",
            None,
        ))
        .await
        .unwrap();

    let json = response_json(response).await;
    for item in json["items"].as_array().unwrap() {
        assert_eq!(item["purpose"], "tenant");
    }
}

#[tokio::test]
async fn test_get_subnet_stats() {
    let state = Arc::new(AppState::new());
    let app = create_router(state.clone());

    // Create a subnet
    let create_body = json!({
        "name": "stats-test",
        "purpose": "tenant",
        "prefix_len": 24
    });

    let create_response = app
        .clone()
        .oneshot(json_request(Method::POST, "/api/v1/subnets", Some(create_body)))
        .await
        .unwrap();

    let created = response_json(create_response).await;
    let subnet_id = created["id"].as_str().unwrap();

    // Get stats
    let response = app
        .oneshot(json_request(
            Method::GET,
            &format!("/api/v1/subnets/{}/stats", subnet_id),
            None,
        ))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let json = response_json(response).await;
    assert_eq!(json["subnet_id"], subnet_id);
    assert!(json["total_capacity"].as_u64().unwrap() > 0);
    assert_eq!(json["allocated_ips"], 0);
}

#[tokio::test]
async fn test_delete_subnet() {
    let state = Arc::new(AppState::new());
    let app = create_router(state.clone());

    // Create a subnet
    let create_body = json!({
        "name": "to-delete",
        "purpose": "tenant",
        "prefix_len": 24
    });

    let create_response = app
        .clone()
        .oneshot(json_request(Method::POST, "/api/v1/subnets", Some(create_body)))
        .await
        .unwrap();

    let created = response_json(create_response).await;
    let subnet_id = created["id"].as_str().unwrap();

    // Delete subnet
    let response = app
        .clone()
        .oneshot(json_request(
            Method::DELETE,
            &format!("/api/v1/subnets/{}", subnet_id),
            None,
        ))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    // Verify it's gone
    let response = app
        .oneshot(json_request(
            Method::GET,
            &format!("/api/v1/subnets/{}", subnet_id),
            None,
        ))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

// ============================================================================
// Node Assignment Tests
// ============================================================================

#[tokio::test]
async fn test_assign_node_to_subnet() {
    let state = Arc::new(AppState::new());
    let app = create_router(state.clone());

    // Create a subnet
    let create_body = json!({
        "name": "node-assignment-test",
        "purpose": "tenant",
        "prefix_len": 24
    });

    let create_response = app
        .clone()
        .oneshot(json_request(Method::POST, "/api/v1/subnets", Some(create_body)))
        .await
        .unwrap();

    let created = response_json(create_response).await;
    let subnet_id = created["id"].as_str().unwrap();

    // Assign a node
    let node_id = uuid::Uuid::new_v4().to_string();
    let assign_body = json!({
        "node_id": node_id,
        "wg_public_key": "dGVzdC1wdWJsaWMta2V5LWJhc2U2NC1lbmNvZGVk"
    });

    let response = app
        .clone()
        .oneshot(json_request(
            Method::POST,
            &format!("/api/v1/subnets/{}/nodes", subnet_id),
            Some(assign_body),
        ))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::CREATED);

    let json = response_json(response).await;
    assert_eq!(json["node_id"], node_id);
    assert_eq!(json["subnet_id"], subnet_id);
    assert!(json["assigned_ip"].as_str().is_some());

    // List nodes in subnet
    let response = app
        .oneshot(json_request(
            Method::GET,
            &format!("/api/v1/subnets/{}/nodes", subnet_id),
            None,
        ))
        .await
        .unwrap();

    let json = response_json(response).await;
    assert_eq!(json.as_array().unwrap().len(), 1);
}

#[tokio::test]
async fn test_unassign_node() {
    let state = Arc::new(AppState::new());
    let app = create_router(state.clone());

    // Create subnet and assign node
    let create_body = json!({
        "name": "unassign-test",
        "purpose": "tenant",
        "prefix_len": 24
    });

    let create_response = app
        .clone()
        .oneshot(json_request(Method::POST, "/api/v1/subnets", Some(create_body)))
        .await
        .unwrap();

    let created = response_json(create_response).await;
    let subnet_id = created["id"].as_str().unwrap();

    let node_id = uuid::Uuid::new_v4().to_string();
    let assign_body = json!({
        "node_id": node_id,
        "wg_public_key": "dGVzdC1wdWJsaWMta2V5LWJhc2U2NC1lbmNvZGVk"
    });

    app.clone()
        .oneshot(json_request(
            Method::POST,
            &format!("/api/v1/subnets/{}/nodes", subnet_id),
            Some(assign_body),
        ))
        .await
        .unwrap();

    // Unassign node
    let response = app
        .clone()
        .oneshot(json_request(
            Method::DELETE,
            &format!("/api/v1/subnets/{}/nodes/{}", subnet_id, node_id),
            None,
        ))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    // Verify node is gone
    let response = app
        .oneshot(json_request(
            Method::GET,
            &format!("/api/v1/subnets/{}/nodes", subnet_id),
            None,
        ))
        .await
        .unwrap();

    let json = response_json(response).await;
    assert_eq!(json.as_array().unwrap().len(), 0);
}

// ============================================================================
// Policy Tests
// ============================================================================

#[tokio::test]
async fn test_list_policies() {
    let app = create_test_app();

    let response = app
        .oneshot(json_request(Method::GET, "/api/v1/policies", None))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_evaluate_policy() {
    let state = Arc::new(AppState::new());
    let app = create_router(state.clone());

    // Evaluate policy with node attributes
    let eval_body = json!({
        "node_type": "data_center",
        "gpu_count": 4,
        "cpu_cores": 64,
        "memory_gb": 256
    });

    let response = app
        .oneshot(json_request(
            Method::POST,
            "/api/v1/policies/evaluate",
            Some(eval_body),
        ))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let json = response_json(response).await;
    // Result depends on configured policies
    assert!(json.get("matched").is_some());
}

// ============================================================================
// Route Tests
// ============================================================================

#[tokio::test]
async fn test_create_cross_subnet_route() {
    let state = Arc::new(AppState::new());
    let app = create_router(state.clone());

    // Create two subnets
    let subnet1_body = json!({
        "name": "route-source",
        "purpose": "tenant",
        "prefix_len": 24
    });

    let response1 = app
        .clone()
        .oneshot(json_request(Method::POST, "/api/v1/subnets", Some(subnet1_body)))
        .await
        .unwrap();
    let subnet1 = response_json(response1).await;
    let source_id = subnet1["id"].as_str().unwrap();

    let subnet2_body = json!({
        "name": "route-dest",
        "purpose": "tenant",
        "prefix_len": 24
    });

    let response2 = app
        .clone()
        .oneshot(json_request(Method::POST, "/api/v1/subnets", Some(subnet2_body)))
        .await
        .unwrap();
    let subnet2 = response_json(response2).await;
    let dest_id = subnet2["id"].as_str().unwrap();

    // Create route
    let route_body = json!({
        "source_subnet_id": source_id,
        "destination_subnet_id": dest_id,
        "direction": "bidirectional",
        "description": "Test route"
    });

    let response = app
        .clone()
        .oneshot(json_request(Method::POST, "/api/v1/routes", Some(route_body)))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::CREATED);

    let json = response_json(response).await;
    assert_eq!(json["source_subnet_id"], source_id);
    assert_eq!(json["destination_subnet_id"], dest_id);
    assert_eq!(json["direction"], "bidirectional");

    // List routes
    let response = app
        .oneshot(json_request(Method::GET, "/api/v1/routes", None))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

// ============================================================================
// Template Tests
// ============================================================================

#[tokio::test]
async fn test_list_templates_has_system_templates() {
    let app = create_test_app();

    let response = app
        .oneshot(json_request(Method::GET, "/api/v1/templates", None))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let json = response_json(response).await;
    let templates = json.as_array().unwrap();

    // Should have system templates
    assert!(templates.len() >= 9);

    // Verify expected templates exist
    let names: Vec<&str> = templates
        .iter()
        .map(|t| t["name"].as_str().unwrap())
        .collect();

    assert!(names.contains(&"tenant-isolation-standard"));
    assert!(names.contains(&"nodetype-datacenter"));
    assert!(names.contains(&"nodetype-workstation"));
    assert!(names.contains(&"nodetype-laptop"));
    assert!(names.contains(&"nodetype-edge"));
    assert!(names.contains(&"geographic-region"));
}

// ============================================================================
// Stats Tests
// ============================================================================

#[tokio::test]
async fn test_manager_stats() {
    let state = Arc::new(AppState::new());
    let app = create_router(state.clone());

    // Create some subnets first
    let create_body = json!({
        "name": "stats-subnet-1",
        "purpose": "tenant",
        "prefix_len": 24
    });

    app.clone()
        .oneshot(json_request(Method::POST, "/api/v1/subnets", Some(create_body)))
        .await
        .unwrap();

    let response = app
        .oneshot(json_request(Method::GET, "/api/v1/stats", None))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let json = response_json(response).await;
    assert!(json["total_subnets"].as_u64().unwrap() >= 1);
    assert!(json.get("active_subnets").is_some());
    assert!(json.get("total_nodes").is_some());
    assert!(json.get("total_policies").is_some());
    assert!(json.get("subnets_by_purpose").is_some());
}

#[tokio::test]
async fn test_migration_stats() {
    let app = create_test_app();

    let response = app
        .oneshot(json_request(Method::GET, "/api/v1/migrations/stats", None))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let json = response_json(response).await;
    assert!(json.get("total").is_some());
    assert!(json.get("pending").is_some());
    assert!(json.get("in_progress").is_some());
    assert!(json.get("completed").is_some());
}

// ============================================================================
// Error Handling Tests
// ============================================================================

#[tokio::test]
async fn test_get_nonexistent_subnet() {
    let app = create_test_app();

    let response = app
        .oneshot(json_request(
            Method::GET,
            "/api/v1/subnets/00000000-0000-0000-0000-000000000000",
            None,
        ))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::NOT_FOUND);

    let json = response_json(response).await;
    assert_eq!(json["code"], "NOT_FOUND");
}

#[tokio::test]
async fn test_get_nonexistent_policy() {
    let app = create_test_app();

    let response = app
        .oneshot(json_request(
            Method::GET,
            "/api/v1/policies/00000000-0000-0000-0000-000000000000",
            None,
        ))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_cannot_delete_non_empty_subnet() {
    let state = Arc::new(AppState::new());
    let app = create_router(state.clone());

    // Create subnet
    let create_body = json!({
        "name": "non-empty-subnet",
        "purpose": "tenant",
        "prefix_len": 24
    });

    let create_response = app
        .clone()
        .oneshot(json_request(Method::POST, "/api/v1/subnets", Some(create_body)))
        .await
        .unwrap();

    let created = response_json(create_response).await;
    let subnet_id = created["id"].as_str().unwrap();

    // Assign a node
    let assign_body = json!({
        "node_id": uuid::Uuid::new_v4().to_string(),
        "wg_public_key": "dGVzdC1wdWJsaWMta2V5LWJhc2U2NC1lbmNvZGVk"
    });

    app.clone()
        .oneshot(json_request(
            Method::POST,
            &format!("/api/v1/subnets/{}/nodes", subnet_id),
            Some(assign_body),
        ))
        .await
        .unwrap();

    // Try to delete - should fail
    let response = app
        .oneshot(json_request(
            Method::DELETE,
            &format!("/api/v1/subnets/{}", subnet_id),
            None,
        ))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::CONFLICT);
}

// ============================================================================
// Workflow Tests
// ============================================================================

#[tokio::test]
async fn test_full_subnet_lifecycle() {
    let state = Arc::new(AppState::new());
    let app = create_router(state.clone());

    // 1. Create subnet
    let create_body = json!({
        "name": "lifecycle-test",
        "description": "Full lifecycle test",
        "purpose": "tenant",
        "prefix_len": 24
    });

    let response = app
        .clone()
        .oneshot(json_request(Method::POST, "/api/v1/subnets", Some(create_body)))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::CREATED);

    let subnet = response_json(response).await;
    let subnet_id = subnet["id"].as_str().unwrap();

    // 2. Get subnet details
    let response = app
        .clone()
        .oneshot(json_request(
            Method::GET,
            &format!("/api/v1/subnets/{}", subnet_id),
            None,
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    // 3. Assign multiple nodes
    let mut node_ids = Vec::new();
    for i in 0..3 {
        let node_id = uuid::Uuid::new_v4().to_string();
        node_ids.push(node_id.clone());

        let assign_body = json!({
            "node_id": node_id,
            "wg_public_key": format!("cHVibGljLWtleS17fQ=={}", i)
        });

        let response = app
            .clone()
            .oneshot(json_request(
                Method::POST,
                &format!("/api/v1/subnets/{}/nodes", subnet_id),
                Some(assign_body),
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::CREATED);
    }

    // 4. Check stats
    let response = app
        .clone()
        .oneshot(json_request(
            Method::GET,
            &format!("/api/v1/subnets/{}/stats", subnet_id),
            None,
        ))
        .await
        .unwrap();
    let stats = response_json(response).await;
    assert_eq!(stats["allocated_ips"], 3);

    // 5. List nodes
    let response = app
        .clone()
        .oneshot(json_request(
            Method::GET,
            &format!("/api/v1/subnets/{}/nodes", subnet_id),
            None,
        ))
        .await
        .unwrap();
    let nodes = response_json(response).await;
    assert_eq!(nodes.as_array().unwrap().len(), 3);

    // 6. Unassign all nodes
    for node_id in &node_ids {
        let response = app
            .clone()
            .oneshot(json_request(
                Method::DELETE,
                &format!("/api/v1/subnets/{}/nodes/{}", subnet_id, node_id),
                None,
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    // 7. Delete subnet
    let response = app
        .oneshot(json_request(
            Method::DELETE,
            &format!("/api/v1/subnets/{}", subnet_id),
            None,
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_multi_tenant_isolation() {
    let state = Arc::new(AppState::new());
    let app = create_router(state.clone());

    // Create subnets for different tenants
    let tenant1_id = uuid::Uuid::new_v4().to_string();
    let tenant2_id = uuid::Uuid::new_v4().to_string();

    let subnet1_body = json!({
        "name": "tenant-1-subnet",
        "purpose": "tenant",
        "tenant_id": tenant1_id,
        "prefix_len": 24
    });

    let response1 = app
        .clone()
        .oneshot(json_request(Method::POST, "/api/v1/subnets", Some(subnet1_body)))
        .await
        .unwrap();
    let subnet1 = response_json(response1).await;

    let subnet2_body = json!({
        "name": "tenant-2-subnet",
        "purpose": "tenant",
        "tenant_id": tenant2_id,
        "prefix_len": 24
    });

    let response2 = app
        .clone()
        .oneshot(json_request(Method::POST, "/api/v1/subnets", Some(subnet2_body)))
        .await
        .unwrap();
    let subnet2 = response_json(response2).await;

    // Verify different CIDRs
    assert_ne!(subnet1["cidr"], subnet2["cidr"]);

    // Filter by tenant
    let response = app
        .oneshot(json_request(
            Method::GET,
            &format!("/api/v1/subnets?tenant_id={}", tenant1_id),
            None,
        ))
        .await
        .unwrap();

    let json = response_json(response).await;
    let items = json["items"].as_array().unwrap();
    assert_eq!(items.len(), 1);
    assert_eq!(items[0]["tenant_id"], tenant1_id);
}
