use horizon_quota_manager::{models::*, Config, DbPool, QuotaRepository};
use rust_decimal_macros::dec;
use sqlx::PgPool;
use uuid::Uuid;

async fn setup_test_db() -> PgPool {
    let config = Config::default();
    let db_pool = DbPool::new(&config.database)
        .await
        .expect("Failed to create database pool");

    // Run migrations
    sqlx::migrate!("./migrations")
        .run(db_pool.inner())
        .await
        .expect("Failed to run migrations");

    // Clean up existing test data
    sqlx::query("TRUNCATE TABLE usage_history, allocations, quotas CASCADE")
        .execute(db_pool.inner())
        .await
        .expect("Failed to truncate tables");

    db_pool.inner().clone()
}

#[tokio::test]
async fn test_create_quota() {
    let pool = setup_test_db().await;
    let repo = QuotaRepository::new(pool);

    let req = CreateQuotaRequest {
        entity_type: EntityType::Organization,
        entity_id: "acme-corp".to_string(),
        parent_id: None,
        resource_type: ResourceType::GpuHours,
        limit_value: dec!(1000.0),
        soft_limit: Some(dec!(800.0)),
        burst_limit: Some(dec!(1200.0)),
        overcommit_ratio: Some(dec!(1.5)),
    };

    let quota = repo
        .create_quota(req)
        .await
        .expect("Failed to create quota");

    assert_eq!(quota.entity_type, EntityType::Organization);
    assert_eq!(quota.entity_id, "acme-corp");
    assert_eq!(quota.resource_type, ResourceType::GpuHours);
    assert_eq!(quota.limit_value, dec!(1000.0));
    assert_eq!(quota.soft_limit, Some(dec!(800.0)));
    assert_eq!(quota.burst_limit, Some(dec!(1200.0)));
    assert_eq!(quota.overcommit_ratio, dec!(1.5));
}

#[tokio::test]
async fn test_create_quota_duplicate_error() {
    let pool = setup_test_db().await;
    let repo = QuotaRepository::new(pool);

    let req = CreateQuotaRequest {
        entity_type: EntityType::Team,
        entity_id: "ml-team".to_string(),
        parent_id: None,
        resource_type: ResourceType::ConcurrentGpus,
        limit_value: dec!(16.0),
        soft_limit: None,
        burst_limit: None,
        overcommit_ratio: None,
    };

    repo.create_quota(req.clone())
        .await
        .expect("Failed to create quota");
    let result = repo.create_quota(req).await;

    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        horizon_quota_manager::QuotaError::AlreadyExists(_)
    ));
}

#[tokio::test]
async fn test_get_quota() {
    let pool = setup_test_db().await;
    let repo = QuotaRepository::new(pool);

    let req = CreateQuotaRequest {
        entity_type: EntityType::User,
        entity_id: "alice".to_string(),
        parent_id: None,
        resource_type: ResourceType::StorageGb,
        limit_value: dec!(500.0),
        soft_limit: None,
        burst_limit: None,
        overcommit_ratio: None,
    };

    let created = repo
        .create_quota(req)
        .await
        .expect("Failed to create quota");
    let fetched = repo
        .get_quota(created.id)
        .await
        .expect("Failed to get quota");

    assert_eq!(created.id, fetched.id);
    assert_eq!(fetched.entity_id, "alice");
}

#[tokio::test]
async fn test_get_quota_not_found() {
    let pool = setup_test_db().await;
    let repo = QuotaRepository::new(pool);

    let result = repo.get_quota(Uuid::new_v4()).await;
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        horizon_quota_manager::QuotaError::NotFound(_)
    ));
}

#[tokio::test]
async fn test_get_quota_by_entity() {
    let pool = setup_test_db().await;
    let repo = QuotaRepository::new(pool);

    let req = CreateQuotaRequest {
        entity_type: EntityType::Team,
        entity_id: "engineering".to_string(),
        parent_id: None,
        resource_type: ResourceType::CpuHours,
        limit_value: dec!(5000.0),
        soft_limit: None,
        burst_limit: None,
        overcommit_ratio: None,
    };

    repo.create_quota(req)
        .await
        .expect("Failed to create quota");

    let fetched = repo
        .get_quota_by_entity(EntityType::Team, "engineering", ResourceType::CpuHours)
        .await
        .expect("Failed to get quota by entity");

    assert_eq!(fetched.entity_id, "engineering");
    assert_eq!(fetched.resource_type, ResourceType::CpuHours);
}

#[tokio::test]
async fn test_list_quotas() {
    let pool = setup_test_db().await;
    let repo = QuotaRepository::new(pool);

    // Create multiple quotas
    for i in 0..3 {
        let req = CreateQuotaRequest {
            entity_type: EntityType::User,
            entity_id: format!("user-{}", i),
            parent_id: None,
            resource_type: ResourceType::GpuHours,
            limit_value: dec!(100.0),
            soft_limit: None,
            burst_limit: None,
            overcommit_ratio: None,
        };
        repo.create_quota(req)
            .await
            .expect("Failed to create quota");
    }

    let quotas = repo
        .list_quotas(Some(EntityType::User))
        .await
        .expect("Failed to list quotas");
    assert_eq!(quotas.len(), 3);
}

#[tokio::test]
async fn test_update_quota() {
    let pool = setup_test_db().await;
    let repo = QuotaRepository::new(pool);

    let req = CreateQuotaRequest {
        entity_type: EntityType::Organization,
        entity_id: "test-org".to_string(),
        parent_id: None,
        resource_type: ResourceType::MemoryGb,
        limit_value: dec!(1000.0),
        soft_limit: None,
        burst_limit: None,
        overcommit_ratio: None,
    };

    let created = repo
        .create_quota(req)
        .await
        .expect("Failed to create quota");

    let update_req = UpdateQuotaRequest {
        limit_value: Some(dec!(2000.0)),
        soft_limit: Some(dec!(1500.0)),
        burst_limit: None,
        overcommit_ratio: None,
    };

    let updated = repo
        .update_quota(created.id, update_req)
        .await
        .expect("Failed to update quota");

    assert_eq!(updated.limit_value, dec!(2000.0));
    assert_eq!(updated.soft_limit, Some(dec!(1500.0)));
    assert!(updated.updated_at > created.updated_at);
}

#[tokio::test]
async fn test_delete_quota() {
    let pool = setup_test_db().await;
    let repo = QuotaRepository::new(pool);

    let req = CreateQuotaRequest {
        entity_type: EntityType::User,
        entity_id: "to-delete".to_string(),
        parent_id: None,
        resource_type: ResourceType::GpuHours,
        limit_value: dec!(50.0),
        soft_limit: None,
        burst_limit: None,
        overcommit_ratio: None,
    };

    let created = repo
        .create_quota(req)
        .await
        .expect("Failed to create quota");
    repo.delete_quota(created.id)
        .await
        .expect("Failed to delete quota");

    let result = repo.get_quota(created.id).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_create_allocation() {
    let pool = setup_test_db().await;
    let repo = QuotaRepository::new(pool);

    // First create a quota
    let quota_req = CreateQuotaRequest {
        entity_type: EntityType::User,
        entity_id: "bob".to_string(),
        parent_id: None,
        resource_type: ResourceType::GpuHours,
        limit_value: dec!(100.0),
        soft_limit: None,
        burst_limit: None,
        overcommit_ratio: None,
    };
    let quota = repo
        .create_quota(quota_req)
        .await
        .expect("Failed to create quota");

    // Create allocation
    let alloc_req = CreateAllocationRequest {
        quota_id: quota.id,
        job_id: Uuid::new_v4(),
        resource_type: ResourceType::GpuHours,
        allocated_value: dec!(10.0),
        metadata: None,
    };

    let allocation = repo
        .create_allocation(alloc_req)
        .await
        .expect("Failed to create allocation");

    assert_eq!(allocation.quota_id, quota.id);
    assert_eq!(allocation.allocated_value, dec!(10.0));
    assert!(allocation.is_active());
    assert_eq!(allocation.version, 0);
}

#[tokio::test]
async fn test_get_current_usage() {
    let pool = setup_test_db().await;
    let repo = QuotaRepository::new(pool);

    // Create quota
    let quota_req = CreateQuotaRequest {
        entity_type: EntityType::User,
        entity_id: "charlie".to_string(),
        parent_id: None,
        resource_type: ResourceType::ConcurrentGpus,
        limit_value: dec!(32.0),
        soft_limit: None,
        burst_limit: None,
        overcommit_ratio: None,
    };
    let quota = repo
        .create_quota(quota_req)
        .await
        .expect("Failed to create quota");

    // Create multiple allocations
    for _i in 0..3 {
        let alloc_req = CreateAllocationRequest {
            quota_id: quota.id,
            job_id: Uuid::new_v4(),
            resource_type: ResourceType::ConcurrentGpus,
            allocated_value: dec!(8.0),
            metadata: None,
        };
        repo.create_allocation(alloc_req)
            .await
            .expect("Failed to create allocation");
    }

    let usage = repo
        .get_current_usage(quota.id)
        .await
        .expect("Failed to get usage");
    assert_eq!(usage, dec!(24.0));
}

#[tokio::test]
async fn test_release_allocation() {
    let pool = setup_test_db().await;
    let repo = QuotaRepository::new(pool);

    // Create quota and allocation
    let quota_req = CreateQuotaRequest {
        entity_type: EntityType::User,
        entity_id: "diana".to_string(),
        parent_id: None,
        resource_type: ResourceType::GpuHours,
        limit_value: dec!(100.0),
        soft_limit: None,
        burst_limit: None,
        overcommit_ratio: None,
    };
    let quota = repo
        .create_quota(quota_req)
        .await
        .expect("Failed to create quota");

    let alloc_req = CreateAllocationRequest {
        quota_id: quota.id,
        job_id: Uuid::new_v4(),
        resource_type: ResourceType::GpuHours,
        allocated_value: dec!(15.0),
        metadata: None,
    };
    let allocation = repo
        .create_allocation(alloc_req)
        .await
        .expect("Failed to create allocation");

    // Release allocation (optimistic lock should increment version)
    let released = repo
        .release_allocation(allocation.id)
        .await
        .expect("Failed to release allocation");

    assert!(!released.is_active());
    assert!(released.released_at.is_some());
    assert_eq!(released.version, 1); // Version incremented

    // Usage should be 0 after release
    let usage = repo
        .get_current_usage(quota.id)
        .await
        .expect("Failed to get usage");
    assert_eq!(usage, dec!(0.0));
}

#[tokio::test]
async fn test_release_allocation_already_released() {
    let pool = setup_test_db().await;
    let repo = QuotaRepository::new(pool);

    let quota_req = CreateQuotaRequest {
        entity_type: EntityType::User,
        entity_id: "eve".to_string(),
        parent_id: None,
        resource_type: ResourceType::GpuHours,
        limit_value: dec!(100.0),
        soft_limit: None,
        burst_limit: None,
        overcommit_ratio: None,
    };
    let quota = repo
        .create_quota(quota_req)
        .await
        .expect("Failed to create quota");

    let alloc_req = CreateAllocationRequest {
        quota_id: quota.id,
        job_id: Uuid::new_v4(),
        resource_type: ResourceType::GpuHours,
        allocated_value: dec!(20.0),
        metadata: None,
    };
    let allocation = repo
        .create_allocation(alloc_req)
        .await
        .expect("Failed to create allocation");

    repo.release_allocation(allocation.id)
        .await
        .expect("Failed to release allocation");

    // Try to release again - should fail
    let result = repo.release_allocation(allocation.id).await;
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        horizon_quota_manager::QuotaError::AllocationNotFound(_)
    ));
}

#[tokio::test]
async fn test_list_active_allocations() {
    let pool = setup_test_db().await;
    let repo = QuotaRepository::new(pool);

    let quota_req = CreateQuotaRequest {
        entity_type: EntityType::Team,
        entity_id: "data-team".to_string(),
        parent_id: None,
        resource_type: ResourceType::GpuHours,
        limit_value: dec!(500.0),
        soft_limit: None,
        burst_limit: None,
        overcommit_ratio: None,
    };
    let quota = repo
        .create_quota(quota_req)
        .await
        .expect("Failed to create quota");

    // Create 5 allocations, release 2
    let mut allocation_ids = Vec::new();
    for _ in 0..5 {
        let alloc_req = CreateAllocationRequest {
            quota_id: quota.id,
            job_id: Uuid::new_v4(),
            resource_type: ResourceType::GpuHours,
            allocated_value: dec!(10.0),
            metadata: None,
        };
        let alloc = repo
            .create_allocation(alloc_req)
            .await
            .expect("Failed to create allocation");
        allocation_ids.push(alloc.id);
    }

    repo.release_allocation(allocation_ids[0])
        .await
        .expect("Failed to release");
    repo.release_allocation(allocation_ids[1])
        .await
        .expect("Failed to release");

    let active = repo
        .list_active_allocations(quota.id)
        .await
        .expect("Failed to list active allocations");
    assert_eq!(active.len(), 3);
}

#[tokio::test]
async fn test_record_and_get_usage_history() {
    let pool = setup_test_db().await;
    let repo = QuotaRepository::new(pool);

    let quota_req = CreateQuotaRequest {
        entity_type: EntityType::User,
        entity_id: "frank".to_string(),
        parent_id: None,
        resource_type: ResourceType::CpuHours,
        limit_value: dec!(1000.0),
        soft_limit: None,
        burst_limit: None,
        overcommit_ratio: None,
    };
    let quota = repo
        .create_quota(quota_req)
        .await
        .expect("Failed to create quota");

    let job_id = Uuid::new_v4();

    // Record allocation
    repo.record_usage(
        quota.id,
        EntityType::User,
        "frank",
        ResourceType::CpuHours,
        dec!(50.0),
        OperationType::Allocate,
        Some(job_id),
        None,
    )
    .await
    .expect("Failed to record usage");

    // Record release
    repo.record_usage(
        quota.id,
        EntityType::User,
        "frank",
        ResourceType::CpuHours,
        dec!(50.0),
        OperationType::Release,
        Some(job_id),
        None,
    )
    .await
    .expect("Failed to record usage");

    let history = repo
        .get_usage_history(quota.id, Some(10))
        .await
        .expect("Failed to get history");
    assert_eq!(history.len(), 2);
    assert_eq!(history[0].operation, OperationType::Release); // Most recent first
    assert_eq!(history[1].operation, OperationType::Allocate);
}
