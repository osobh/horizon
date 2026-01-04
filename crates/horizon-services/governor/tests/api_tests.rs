use axum::body::Body;
use axum::http::{Request, StatusCode};
use horizon_governor::{create_router, PolicyRepository};
use serde_json::json;
use sqlx::PgPool;
use tower::ServiceExt;

struct TestApp {
    pool: PgPool,
    repo: PolicyRepository,
}

impl TestApp {
    async fn new() -> Self {
        let database_url = std::env::var("DATABASE_URL").unwrap_or_else(|_| {
            "postgresql://postgres:postgres@localhost:5432/governor_test".to_string()
        });

        let pool = PgPool::connect(&database_url)
            .await
            .expect("Failed to connect to test database");

        sqlx::query("DROP TABLE IF EXISTS policy_versions, policies CASCADE")
            .execute(&pool)
            .await
            .expect("Failed to clean database");

        sqlx::migrate!("./migrations")
            .run(&pool)
            .await
            .expect("Failed to run migrations");

        let repo = PolicyRepository::new(pool.clone());

        Self { pool, repo }
    }

    async fn cleanup(&self) {
        sqlx::query("DROP TABLE IF EXISTS policy_versions, policies CASCADE")
            .execute(&self.pool)
            .await
            .expect("Failed to clean database");
    }

    fn router(&self) -> axum::Router {
        create_router(self.repo.clone())
    }
}

#[tokio::test]
async fn test_health_check() {
    let app = TestApp::new().await;

    let response = app
        .router()
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    app.cleanup().await;
}

#[tokio::test]
async fn test_create_policy() {
    let app = TestApp::new().await;

    let policy_yaml = r#"
apiVersion: policy.horizon.dev/v1
kind: Policy
metadata:
  name: test-policy
spec:
  principals:
    - type: role
      value: admin
  resources:
    - type: job
      pattern: "jobs/*"
  rules:
    - effect: allow
      actions: [submit]
"#;

    let request_body = json!({
        "name": "test-policy",
        "content": policy_yaml,
        "description": "Test policy",
        "created_by": "test-user"
    });

    let response = app
        .router()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/v1/policies")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&request_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::CREATED);

    app.cleanup().await;
}

#[tokio::test]
async fn test_create_policy_duplicate() {
    let app = TestApp::new().await;

    let policy_yaml = r#"
apiVersion: policy.horizon.dev/v1
kind: Policy
metadata:
  name: test-policy
spec:
  principals:
    - type: role
      value: admin
  resources:
    - type: job
      pattern: "jobs/*"
  rules:
    - effect: allow
      actions: [submit]
"#;

    let request_body = json!({
        "name": "duplicate-policy",
        "content": policy_yaml,
        "description": "Test policy",
        "created_by": "test-user"
    });

    let response1 = app
        .router()
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/v1/policies")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&request_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response1.status(), StatusCode::CREATED);

    let response2 = app
        .router()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/v1/policies")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&request_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response2.status(), StatusCode::CONFLICT);

    app.cleanup().await;
}

#[tokio::test]
async fn test_create_policy_invalid_content() {
    let app = TestApp::new().await;

    let request_body = json!({
        "name": "invalid-policy",
        "content": "not valid yaml: [[[",
        "description": "Test policy",
        "created_by": "test-user"
    });

    let response = app
        .router()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/v1/policies")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&request_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);

    app.cleanup().await;
}

#[tokio::test]
async fn test_get_policy() {
    let app = TestApp::new().await;

    let policy_yaml = r#"
apiVersion: policy.horizon.dev/v1
kind: Policy
metadata:
  name: get-test-policy
spec:
  principals:
    - type: role
      value: admin
  resources:
    - type: job
      pattern: "jobs/*"
  rules:
    - effect: allow
      actions: [submit]
"#;

    app.repo
        .create(
            "get-test-policy",
            policy_yaml,
            Some("Test policy"),
            "test-user",
        )
        .await
        .unwrap();

    let response = app
        .router()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/api/v1/policies/get-test-policy")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    app.cleanup().await;
}

#[tokio::test]
async fn test_get_policy_not_found() {
    let app = TestApp::new().await;

    let response = app
        .router()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/api/v1/policies/nonexistent")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::NOT_FOUND);

    app.cleanup().await;
}

#[tokio::test]
async fn test_list_policies() {
    let app = TestApp::new().await;

    let policy_yaml = r#"
apiVersion: policy.horizon.dev/v1
kind: Policy
metadata:
  name: list-test-policy
spec:
  principals:
    - type: role
      value: admin
  resources:
    - type: job
      pattern: "jobs/*"
  rules:
    - effect: allow
      actions: [submit]
"#;

    app.repo
        .create(
            "list-test-policy-1",
            policy_yaml,
            Some("Test policy 1"),
            "test-user",
        )
        .await
        .unwrap();

    app.repo
        .create(
            "list-test-policy-2",
            policy_yaml,
            Some("Test policy 2"),
            "test-user",
        )
        .await
        .unwrap();

    let response = app
        .router()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/api/v1/policies")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    app.cleanup().await;
}

#[tokio::test]
async fn test_update_policy() {
    let app = TestApp::new().await;

    let policy_yaml_v1 = r#"
apiVersion: policy.horizon.dev/v1
kind: Policy
metadata:
  name: update-test-policy
spec:
  principals:
    - type: role
      value: admin
  resources:
    - type: job
      pattern: "jobs/*"
  rules:
    - effect: allow
      actions: [submit]
"#;

    app.repo
        .create(
            "update-test-policy",
            policy_yaml_v1,
            Some("Test policy v1"),
            "test-user",
        )
        .await
        .unwrap();

    let policy_yaml_v2 = r#"
apiVersion: policy.horizon.dev/v1
kind: Policy
metadata:
  name: update-test-policy
spec:
  principals:
    - type: role
      value: user
  resources:
    - type: job
      pattern: "jobs/*"
  rules:
    - effect: allow
      actions: [submit, cancel]
"#;

    let request_body = json!({
        "content": policy_yaml_v2,
        "description": "Test policy v2",
        "created_by": "test-user"
    });

    let response = app
        .router()
        .oneshot(
            Request::builder()
                .method("PUT")
                .uri("/api/v1/policies/update-test-policy")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&request_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    app.cleanup().await;
}

#[tokio::test]
async fn test_update_policy_not_found() {
    let app = TestApp::new().await;

    let policy_yaml = r#"
apiVersion: policy.horizon.dev/v1
kind: Policy
metadata:
  name: nonexistent
spec:
  principals:
    - type: role
      value: admin
  resources:
    - type: job
      pattern: "jobs/*"
  rules:
    - effect: allow
      actions: [submit]
"#;

    let request_body = json!({
        "content": policy_yaml,
        "description": "Test policy",
        "created_by": "test-user"
    });

    let response = app
        .router()
        .oneshot(
            Request::builder()
                .method("PUT")
                .uri("/api/v1/policies/nonexistent")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&request_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::NOT_FOUND);

    app.cleanup().await;
}

#[tokio::test]
async fn test_delete_policy() {
    let app = TestApp::new().await;

    let policy_yaml = r#"
apiVersion: policy.horizon.dev/v1
kind: Policy
metadata:
  name: delete-test-policy
spec:
  principals:
    - type: role
      value: admin
  resources:
    - type: job
      pattern: "jobs/*"
  rules:
    - effect: allow
      actions: [submit]
"#;

    app.repo
        .create(
            "delete-test-policy",
            policy_yaml,
            Some("Test policy"),
            "test-user",
        )
        .await
        .unwrap();

    let response = app
        .router()
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri("/api/v1/policies/delete-test-policy")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::NO_CONTENT);

    app.cleanup().await;
}

#[tokio::test]
async fn test_delete_policy_not_found() {
    let app = TestApp::new().await;

    let response = app
        .router()
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri("/api/v1/policies/nonexistent")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::NOT_FOUND);

    app.cleanup().await;
}

#[tokio::test]
async fn test_get_policy_versions() {
    let app = TestApp::new().await;

    let policy_yaml_v1 = r#"
apiVersion: policy.horizon.dev/v1
kind: Policy
metadata:
  name: version-test-policy
spec:
  principals:
    - type: role
      value: admin
  resources:
    - type: job
      pattern: "jobs/*"
  rules:
    - effect: allow
      actions: [submit]
"#;

    app.repo
        .create(
            "version-test-policy",
            policy_yaml_v1,
            Some("Test policy v1"),
            "test-user",
        )
        .await
        .unwrap();

    let policy_yaml_v2 = r#"
apiVersion: policy.horizon.dev/v1
kind: Policy
metadata:
  name: version-test-policy
spec:
  principals:
    - type: role
      value: user
  resources:
    - type: job
      pattern: "jobs/*"
  rules:
    - effect: allow
      actions: [submit, cancel]
"#;

    app.repo
        .update(
            "version-test-policy",
            policy_yaml_v2,
            Some("Test policy v2"),
            "test-user",
        )
        .await
        .unwrap();

    let response = app
        .router()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/api/v1/policies/version-test-policy/versions")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    app.cleanup().await;
}

#[tokio::test]
async fn test_evaluate_allow() {
    let app = TestApp::new().await;

    let policy_yaml = r#"
apiVersion: policy.horizon.dev/v1
kind: Policy
metadata:
  name: gpu-access-policy
spec:
  principals:
    - type: role
      value: gpu-user
  resources:
    - type: job
      pattern: "jobs/*"
  rules:
    - effect: allow
      actions: [submit]
"#;

    app.repo
        .create(
            "gpu-access-policy",
            policy_yaml,
            Some("GPU access policy"),
            "test-user",
        )
        .await
        .unwrap();

    let request_body = json!({
        "principal": {
            "user_id": "alice",
            "roles": ["gpu-user"],
            "teams": ["ml-research"]
        },
        "resource": {
            "type": "job",
            "id": "jobs/123",
            "attributes": {
                "gpu_count": 4,
                "gpu_type": "H100"
            }
        },
        "action": "submit"
    });

    let response = app
        .router()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/v1/evaluate")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&request_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    app.cleanup().await;
}

#[tokio::test]
async fn test_evaluate_deny() {
    let app = TestApp::new().await;

    let policy_yaml = r#"
apiVersion: policy.horizon.dev/v1
kind: Policy
metadata:
  name: admin-only-policy
spec:
  principals:
    - type: role
      value: admin
  resources:
    - type: cluster
      pattern: "clusters/*"
  rules:
    - effect: allow
      actions: [delete]
"#;

    app.repo
        .create(
            "admin-only-policy",
            policy_yaml,
            Some("Admin only policy"),
            "test-user",
        )
        .await
        .unwrap();

    let request_body = json!({
        "principal": {
            "user_id": "bob",
            "roles": ["user"],
            "teams": []
        },
        "resource": {
            "type": "cluster",
            "id": "clusters/prod",
            "attributes": {}
        },
        "action": "delete"
    });

    let response = app
        .router()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/v1/evaluate")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&request_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    app.cleanup().await;
}

#[tokio::test]
async fn test_concurrent_policy_creation() {
    let app = TestApp::new().await;

    let policy_yaml = r#"
apiVersion: policy.horizon.dev/v1
kind: Policy
metadata:
  name: concurrent-test
spec:
  principals:
    - type: role
      value: admin
  resources:
    - type: job
      pattern: "jobs/*"
  rules:
    - effect: allow
      actions: [submit]
"#;

    let mut handles = vec![];

    for i in 0..10 {
        let repo = app.repo.clone();
        let yaml = policy_yaml.to_string();
        handles.push(tokio::spawn(async move {
            repo.create(
                &format!("concurrent-policy-{}", i),
                &yaml,
                Some(&format!("Concurrent test policy {}", i)),
                "test-user",
            )
            .await
        }));
    }

    for handle in handles {
        let result = handle.await.unwrap();
        assert!(result.is_ok());
    }

    let policies = app.repo.list(false).await.unwrap();
    assert_eq!(policies.len(), 10);

    app.cleanup().await;
}

#[tokio::test]
async fn test_policy_with_multiple_rules() {
    let app = TestApp::new().await;

    let policy_yaml = r#"
apiVersion: policy.horizon.dev/v1
kind: Policy
metadata:
  name: multi-rule-policy
spec:
  principals:
    - type: role
      value: admin
  resources:
    - type: job
      pattern: "jobs/*"
  rules:
    - effect: allow
      actions: [submit, cancel, delete]
"#;

    let request_body = json!({
        "name": "multi-rule-policy",
        "content": policy_yaml,
        "description": "Multi-rule test policy",
        "created_by": "test-user"
    });

    let response = app
        .router()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/v1/policies")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&request_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::CREATED);

    app.cleanup().await;
}

#[tokio::test]
async fn test_update_policy_increments_version() {
    let app = TestApp::new().await;

    let policy_yaml_v1 = r#"
apiVersion: policy.horizon.dev/v1
kind: Policy
metadata:
  name: version-increment-test
spec:
  principals:
    - type: role
      value: admin
  resources:
    - type: job
      pattern: "jobs/*"
  rules:
    - effect: allow
      actions: [submit]
"#;

    app.repo
        .create(
            "version-increment-test",
            policy_yaml_v1,
            Some("Version 1"),
            "user1",
        )
        .await
        .unwrap();

    let v1_policy = app
        .repo
        .get_by_name("version-increment-test")
        .await
        .unwrap();
    assert_eq!(v1_policy.version, 1);

    let policy_yaml_v2 = r#"
apiVersion: policy.horizon.dev/v1
kind: Policy
metadata:
  name: version-increment-test
spec:
  principals:
    - type: role
      value: user
  resources:
    - type: job
      pattern: "jobs/*"
  rules:
    - effect: allow
      actions: [submit, cancel]
"#;

    let request_body = json!({
        "content": policy_yaml_v2,
        "description": "Version 2",
        "created_by": "user2"
    });

    let response = app
        .router()
        .oneshot(
            Request::builder()
                .method("PUT")
                .uri("/api/v1/policies/version-increment-test")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&request_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let v2_policy = app
        .repo
        .get_by_name("version-increment-test")
        .await
        .unwrap();
    assert_eq!(v2_policy.version, 2);

    app.cleanup().await;
}

#[tokio::test]
async fn test_list_policies_empty_database() {
    let app = TestApp::new().await;

    let response = app
        .router()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/api/v1/policies")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    app.cleanup().await;
}

#[tokio::test]
async fn test_evaluate_with_multiple_policies() {
    let app = TestApp::new().await;

    let policy_yaml_1 = r#"
apiVersion: policy.horizon.dev/v1
kind: Policy
metadata:
  name: policy-1
spec:
  principals:
    - type: role
      value: user
  resources:
    - type: job
      pattern: "jobs/*"
  rules:
    - effect: allow
      actions: [view]
"#;

    let policy_yaml_2 = r#"
apiVersion: policy.horizon.dev/v1
kind: Policy
metadata:
  name: policy-2
spec:
  principals:
    - type: role
      value: admin
  resources:
    - type: job
      pattern: "jobs/*"
  rules:
    - effect: allow
      actions: [submit, delete]
"#;

    app.repo
        .create("policy-1", policy_yaml_1, Some("Policy 1"), "test-user")
        .await
        .unwrap();

    app.repo
        .create("policy-2", policy_yaml_2, Some("Policy 2"), "test-user")
        .await
        .unwrap();

    let request_body = json!({
        "principal": {
            "user_id": "admin-user",
            "roles": ["admin"],
            "teams": []
        },
        "resource": {
            "type": "job",
            "id": "jobs/456",
            "attributes": {}
        },
        "action": "submit"
    });

    let response = app
        .router()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/v1/evaluate")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&request_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    app.cleanup().await;
}

#[tokio::test]
async fn test_policy_with_conditions() {
    let app = TestApp::new().await;

    let policy_yaml = r#"
apiVersion: policy.horizon.dev/v1
kind: Policy
metadata:
  name: conditional-policy
spec:
  principals:
    - type: role
      value: gpu-user
  resources:
    - type: job
      pattern: "jobs/*"
  rules:
    - effect: allow
      actions: [submit]
      conditions:
        - field: resource.gpu_count
          operator: lte
          value: 8
"#;

    let request_body = json!({
        "name": "conditional-policy",
        "content": policy_yaml,
        "description": "Conditional test policy",
        "created_by": "test-user"
    });

    let response = app
        .router()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/v1/policies")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&request_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::CREATED);

    app.cleanup().await;
}

#[tokio::test]
async fn test_evaluate_with_attributes() {
    let app = TestApp::new().await;

    let policy_yaml = r#"
apiVersion: policy.horizon.dev/v1
kind: Policy
metadata:
  name: attribute-policy
spec:
  principals:
    - type: role
      value: developer
  resources:
    - type: deployment
      pattern: "deployments/*"
  rules:
    - effect: allow
      actions: [deploy]
      conditions:
        - field: resource.environment
          operator: eq
          value: dev
"#;

    app.repo
        .create(
            "attribute-policy",
            policy_yaml,
            Some("Attribute policy"),
            "test-user",
        )
        .await
        .unwrap();

    let request_body = json!({
        "principal": {
            "user_id": "dev-user",
            "roles": ["developer"],
            "teams": []
        },
        "resource": {
            "type": "deployment",
            "id": "deployments/app-v2",
            "attributes": {
                "environment": "dev",
                "region": "us-west-2"
            }
        },
        "action": "deploy"
    });

    let response = app
        .router()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/v1/evaluate")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&request_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    app.cleanup().await;
}

#[tokio::test]
async fn test_get_versions_after_updates() {
    let app = TestApp::new().await;

    let policy_yaml = r#"
apiVersion: policy.horizon.dev/v1
kind: Policy
metadata:
  name: multi-version-policy
spec:
  principals:
    - type: role
      value: admin
  resources:
    - type: job
      pattern: "jobs/*"
  rules:
    - effect: allow
      actions: [submit]
"#;

    app.repo
        .create(
            "multi-version-policy",
            policy_yaml,
            Some("Version 1"),
            "test-user",
        )
        .await
        .unwrap();

    let response = app
        .router()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/api/v1/policies/multi-version-policy/versions")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    app.cleanup().await;
}

#[tokio::test]
async fn test_policy_description_optional() {
    let app = TestApp::new().await;

    let policy_yaml = r#"
apiVersion: policy.horizon.dev/v1
kind: Policy
metadata:
  name: no-description-policy
spec:
  principals:
    - type: role
      value: admin
  resources:
    - type: job
      pattern: "jobs/*"
  rules:
    - effect: allow
      actions: [submit]
"#;

    let request_body = json!({
        "name": "no-description-policy",
        "content": policy_yaml,
        "created_by": "test-user"
    });

    let response = app
        .router()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/v1/policies")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&request_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::CREATED);

    app.cleanup().await;
}

#[tokio::test]
async fn test_evaluate_empty_policies() {
    let app = TestApp::new().await;

    let request_body = json!({
        "principal": {
            "user_id": "test-user",
            "roles": ["user"],
            "teams": []
        },
        "resource": {
            "type": "job",
            "id": "jobs/789",
            "attributes": {}
        },
        "action": "submit"
    });

    let response = app
        .router()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/v1/evaluate")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&request_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    app.cleanup().await;
}

#[tokio::test]
async fn test_policy_with_teams() {
    let app = TestApp::new().await;

    let policy_yaml = r#"
apiVersion: policy.horizon.dev/v1
kind: Policy
metadata:
  name: team-policy
spec:
  principals:
    - type: team
      value: engineering
  resources:
    - type: repository
      pattern: "repos/*"
  rules:
    - effect: allow
      actions: [read, write]
"#;

    let request_body = json!({
        "name": "team-policy",
        "content": policy_yaml,
        "description": "Team-based policy",
        "created_by": "test-user"
    });

    let response = app
        .router()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/v1/policies")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&request_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::CREATED);

    app.cleanup().await;
}
