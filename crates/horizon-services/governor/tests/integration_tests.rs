use horizon_governor::{PolicyRepository, PolicyService};
use sqlx::PgPool;

struct TestContext {
    pool: PgPool,
    service: PolicyService,
}

impl TestContext {
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
        let service = PolicyService::new(repo);

        Self { pool, service }
    }

    async fn cleanup(&self) {
        sqlx::query("DROP TABLE IF EXISTS policy_versions, policies CASCADE")
            .execute(&self.pool)
            .await
            .expect("Failed to clean database");
    }
}

#[tokio::test]
async fn test_policy_lifecycle() {
    let ctx = TestContext::new().await;

    let policy_yaml = r#"
apiVersion: policy.horizon.dev/v1
kind: Policy
metadata:
  name: lifecycle-test
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

    let created = ctx
        .service
        .create_policy(
            "lifecycle-test",
            policy_yaml,
            Some("Lifecycle test"),
            "test-user",
        )
        .await
        .unwrap();

    assert_eq!(created.name, "lifecycle-test");
    assert_eq!(created.version, 1);

    let fetched = ctx.service.get_policy("lifecycle-test").await.unwrap();
    assert_eq!(fetched.id, created.id);
    assert_eq!(fetched.name, "lifecycle-test");

    let updated_yaml = r#"
apiVersion: policy.horizon.dev/v1
kind: Policy
metadata:
  name: lifecycle-test
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

    let updated = ctx
        .service
        .update_policy(
            "lifecycle-test",
            updated_yaml,
            Some("Lifecycle test v2"),
            "test-user",
        )
        .await
        .unwrap();

    assert_eq!(updated.version, 2);

    ctx.service.delete_policy("lifecycle-test").await.unwrap();

    let result = ctx.service.get_policy("lifecycle-test").await;
    assert!(result.is_err());

    ctx.cleanup().await;
}

#[tokio::test]
async fn test_version_history() {
    let ctx = TestContext::new().await;

    let policy_yaml_v1 = r#"
apiVersion: policy.horizon.dev/v1
kind: Policy
metadata:
  name: version-history-test
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

    let repo = PolicyRepository::new(ctx.pool.clone());

    repo.create(
        "version-history-test",
        policy_yaml_v1,
        Some("Version 1"),
        "user1",
    )
    .await
    .unwrap();

    let policy_yaml_v2 = r#"
apiVersion: policy.horizon.dev/v1
kind: Policy
metadata:
  name: version-history-test
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

    repo.update(
        "version-history-test",
        policy_yaml_v2,
        Some("Version 2"),
        "user2",
    )
    .await
    .unwrap();

    let policy_yaml_v3 = r#"
apiVersion: policy.horizon.dev/v1
kind: Policy
metadata:
  name: version-history-test
spec:
  principals:
    - type: role
      value: guest
  resources:
    - type: job
      pattern: "jobs/*"
  rules:
    - effect: allow
      actions: [view]
"#;

    repo.update(
        "version-history-test",
        policy_yaml_v3,
        Some("Version 3"),
        "user3",
    )
    .await
    .unwrap();

    let versions = repo.get_versions("version-history-test").await.unwrap();

    assert_eq!(versions.len(), 3);
    assert_eq!(versions[0].version, 3);
    assert_eq!(versions[1].version, 2);
    assert_eq!(versions[2].version, 1);

    ctx.cleanup().await;
}

#[tokio::test]
async fn test_policy_validation() {
    let ctx = TestContext::new().await;

    let invalid_yaml = "not valid yaml: [[[";

    let result = ctx
        .service
        .create_policy("invalid-policy", invalid_yaml, Some("Invalid"), "test-user")
        .await;

    assert!(result.is_err());

    ctx.cleanup().await;
}

#[tokio::test]
async fn test_database_persistence() {
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

    let policy_yaml = r#"
apiVersion: policy.horizon.dev/v1
kind: Policy
metadata:
  name: persistence-test
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

    let created = repo
        .create(
            "persistence-test",
            policy_yaml,
            Some("Persistence test"),
            "test-user",
        )
        .await
        .unwrap();

    drop(repo);
    drop(pool);

    let pool2 = PgPool::connect(&database_url)
        .await
        .expect("Failed to reconnect to test database");

    let repo2 = PolicyRepository::new(pool2.clone());

    let fetched = repo2.get_by_name("persistence-test").await.unwrap();

    assert_eq!(fetched.id, created.id);
    assert_eq!(fetched.name, created.name);
    assert_eq!(fetched.content, created.content);

    sqlx::query("DROP TABLE IF EXISTS policy_versions, policies CASCADE")
        .execute(&pool2)
        .await
        .expect("Failed to clean database");
}

#[tokio::test]
async fn test_list_policies_filtering() {
    let ctx = TestContext::new().await;

    let policy_yaml = r#"
apiVersion: policy.horizon.dev/v1
kind: Policy
metadata:
  name: filter-test
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

    let repo = PolicyRepository::new(ctx.pool.clone());

    repo.create(
        "filter-test-1",
        policy_yaml,
        Some("Filter test 1"),
        "test-user",
    )
    .await
    .unwrap();

    repo.create(
        "filter-test-2",
        policy_yaml,
        Some("Filter test 2"),
        "test-user",
    )
    .await
    .unwrap();

    repo.create(
        "filter-test-3",
        policy_yaml,
        Some("Filter test 3"),
        "test-user",
    )
    .await
    .unwrap();

    let all_policies = repo.list(false).await.unwrap();
    assert_eq!(all_policies.len(), 3);

    let enabled_policies = repo.list(true).await.unwrap();
    assert_eq!(enabled_policies.len(), 3);

    ctx.cleanup().await;
}

#[tokio::test]
async fn test_policy_updates_create_versions() {
    let ctx = TestContext::new().await;

    let policy_yaml_v1 = r#"
apiVersion: policy.horizon.dev/v1
kind: Policy
metadata:
  name: update-version-test
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

    let repo = PolicyRepository::new(ctx.pool.clone());

    let created = repo
        .create(
            "update-version-test",
            policy_yaml_v1,
            Some("Version 1"),
            "user1",
        )
        .await
        .unwrap();

    assert_eq!(created.version, 1);

    let policy_yaml_v2 = r#"
apiVersion: policy.horizon.dev/v1
kind: Policy
metadata:
  name: update-version-test
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

    let updated = repo
        .update(
            "update-version-test",
            policy_yaml_v2,
            Some("Version 2"),
            "user2",
        )
        .await
        .unwrap();

    assert_eq!(updated.version, 2);

    let versions = repo.get_versions("update-version-test").await.unwrap();
    assert_eq!(versions.len(), 2);

    ctx.cleanup().await;
}

#[tokio::test]
async fn test_policy_deletion_cascades_versions() {
    let ctx = TestContext::new().await;

    let policy_yaml = r#"
apiVersion: policy.horizon.dev/v1
kind: Policy
metadata:
  name: cascade-test
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

    let repo = PolicyRepository::new(ctx.pool.clone());

    repo.create(
        "cascade-test",
        policy_yaml,
        Some("Cascade test"),
        "test-user",
    )
    .await
    .unwrap();

    repo.update(
        "cascade-test",
        policy_yaml,
        Some("Cascade test v2"),
        "test-user",
    )
    .await
    .unwrap();

    let versions_before = repo.get_versions("cascade-test").await.unwrap();
    assert_eq!(versions_before.len(), 2);

    repo.delete("cascade-test").await.unwrap();

    let policy_result = repo.get_by_name("cascade-test").await;
    assert!(policy_result.is_err());

    ctx.cleanup().await;
}

#[tokio::test]
async fn test_multiple_policies_same_time() {
    let ctx = TestContext::new().await;

    let policy_yaml_1 = r#"
apiVersion: policy.horizon.dev/v1
kind: Policy
metadata:
  name: multi-1
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

    let policy_yaml_2 = r#"
apiVersion: policy.horizon.dev/v1
kind: Policy
metadata:
  name: multi-2
spec:
  principals:
    - type: role
      value: user
  resources:
    - type: cluster
      pattern: "clusters/*"
  rules:
    - effect: allow
      actions: [view]
"#;

    let repo = PolicyRepository::new(ctx.pool.clone());

    let handle1 = {
        let repo = repo.clone();
        let yaml = policy_yaml_1.to_string();
        tokio::spawn(async move {
            repo.create("multi-1", &yaml, Some("Multi 1"), "test-user")
                .await
        })
    };

    let handle2 = {
        let repo = repo.clone();
        let yaml = policy_yaml_2.to_string();
        tokio::spawn(async move {
            repo.create("multi-2", &yaml, Some("Multi 2"), "test-user")
                .await
        })
    };

    let result1 = handle1.await.unwrap();
    let result2 = handle2.await.unwrap();

    assert!(result1.is_ok());
    assert!(result2.is_ok());

    let policies = repo.list(false).await.unwrap();
    assert_eq!(policies.len(), 2);

    ctx.cleanup().await;
}

#[tokio::test]
async fn test_policy_evaluation_integration() {
    let ctx = TestContext::new().await;

    let policy_yaml = r#"
apiVersion: policy.horizon.dev/v1
kind: Policy
metadata:
  name: eval-integration-test
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

    let repo = PolicyRepository::new(ctx.pool.clone());

    repo.create(
        "eval-integration-test",
        policy_yaml,
        Some("Eval integration test"),
        "test-user",
    )
    .await
    .unwrap();

    let policies = repo.get_all_enabled_policies().await.unwrap();
    assert_eq!(policies.len(), 1);

    let policy = &policies[0];
    let parsed_policy = hpc_policy::parse_policy(&policy.content).unwrap();

    assert_eq!(parsed_policy.metadata.name, "eval-integration-test");

    ctx.cleanup().await;
}

#[tokio::test]
async fn test_empty_database_queries() {
    let ctx = TestContext::new().await;

    let repo = PolicyRepository::new(ctx.pool.clone());

    let policies = repo.list(false).await.unwrap();
    assert_eq!(policies.len(), 0);

    let result = repo.get_by_name("nonexistent").await;
    assert!(result.is_err());

    ctx.cleanup().await;
}
