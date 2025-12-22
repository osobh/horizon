// Service layer integration tests
use horizon_quota_manager::{
    models::*,
    service::{AllocationService, QuotaService},
    QuotaRepository, DbPool, Config,
};
use rust_decimal_macros::dec;
use sqlx::PgPool;
use uuid::Uuid;

async fn setup_test_db() -> PgPool {
    let config = Config::default();
    let db_pool = DbPool::new(&config.database)
        .await
        .expect("Failed to create database pool");

    sqlx::migrate!("./migrations")
        .run(db_pool.inner())
        .await
        .expect("Failed to run migrations");

    sqlx::query("TRUNCATE TABLE usage_history, allocations, quotas CASCADE")
        .execute(db_pool.inner())
        .await
        .expect("Failed to truncate tables");

    db_pool.inner().clone()
}

#[tokio::test]
async fn test_quota_service_hierarchical_validation() {
    let pool = setup_test_db().await;
    let repo = QuotaRepository::new(pool);
    let service = QuotaService::new(repo);

    // Create org quota
    let org_req = CreateQuotaRequest {
        entity_type: EntityType::Organization,
        entity_id: "acme".to_string(),
        parent_id: None,
        resource_type: ResourceType::GpuHours,
        limit_value: dec!(1000.0),
        soft_limit: None,
        burst_limit: None,
        overcommit_ratio: None,
    };
    let org_quota = service.create_quota(org_req).await.expect("Failed to create org quota");

    // Create team quota under org
    let team_req = CreateQuotaRequest {
        entity_type: EntityType::Team,
        entity_id: "ml-team".to_string(),
        parent_id: Some(org_quota.id),
        resource_type: ResourceType::GpuHours,
        limit_value: dec!(500.0),
        soft_limit: None,
        burst_limit: None,
        overcommit_ratio: None,
    };
    let team_quota = service.create_quota(team_req).await.expect("Failed to create team quota");

    // Create user quota under team
    let user_req = CreateQuotaRequest {
        entity_type: EntityType::User,
        entity_id: "alice".to_string(),
        parent_id: Some(team_quota.id),
        resource_type: ResourceType::GpuHours,
        limit_value: dec!(100.0),
        soft_limit: Some(dec!(80.0)),
        burst_limit: None,
        overcommit_ratio: None,
    };
    let user_quota = service.create_quota(user_req).await.expect("Failed to create user quota");

    assert_eq!(user_quota.parent_id, Some(team_quota.id));
    assert_eq!(user_quota.limit_value, dec!(100.0));
}

#[tokio::test]
async fn test_quota_service_invalid_hierarchy() {
    let pool = setup_test_db().await;
    let repo = QuotaRepository::new(pool);
    let service = QuotaService::new(repo);

    // Create user quota (no parent)
    let user_req = CreateQuotaRequest {
        entity_type: EntityType::User,
        entity_id: "bob".to_string(),
        parent_id: None,
        resource_type: ResourceType::CpuHours,
        limit_value: dec!(100.0),
        soft_limit: None,
        burst_limit: None,
        overcommit_ratio: None,
    };
    let user_quota = service.create_quota(user_req).await.expect("Failed to create user quota");

    // Try to create org quota under user (invalid hierarchy)
    let org_req = CreateQuotaRequest {
        entity_type: EntityType::Organization,
        entity_id: "invalid".to_string(),
        parent_id: Some(user_quota.id),
        resource_type: ResourceType::CpuHours,
        limit_value: dec!(50.0),
        soft_limit: None,
        burst_limit: None,
        overcommit_ratio: None,
    };

    let result = service.create_quota(org_req).await;
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), horizon_quota_manager::QuotaError::InvalidHierarchy(_)));
}

#[tokio::test]
async fn test_allocation_service_allocate_and_release() {
    let pool = setup_test_db().await;
    let repo = QuotaRepository::new(pool);
    let alloc_service = AllocationService::new(repo.clone());

    // Create quota
    let quota_req = CreateQuotaRequest {
        entity_type: EntityType::User,
        entity_id: "charlie".to_string(),
        parent_id: None,
        resource_type: ResourceType::ConcurrentGpus,
        limit_value: dec!(16.0),
        soft_limit: None,
        burst_limit: None,
        overcommit_ratio: None,
    };
    let quota = repo.create_quota(quota_req).await.expect("Failed to create quota");

    let job_id = Uuid::new_v4();

    // Allocate
    let allocation = alloc_service
        .allocate(
            EntityType::User,
            "charlie",
            job_id,
            ResourceType::ConcurrentGpus,
            dec!(8.0),
            None,
        )
        .await
        .expect("Failed to allocate");

    assert_eq!(allocation.allocated_value, dec!(8.0));
    assert!(allocation.is_active());

    // Check usage
    let usage = repo.get_current_usage(quota.id).await.expect("Failed to get usage");
    assert_eq!(usage, dec!(8.0));

    // Release
    let released = alloc_service
        .release(allocation.id)
        .await
        .expect("Failed to release");

    assert!(!released.is_active());
    assert_eq!(released.version, 1); // Version incremented by optimistic lock

    // Usage should be 0
    let usage_after = repo.get_current_usage(quota.id).await.expect("Failed to get usage");
    assert_eq!(usage_after, dec!(0.0));
}

#[tokio::test]
async fn test_allocation_service_quota_exceeded() {
    let pool = setup_test_db().await;
    let repo = QuotaRepository::new(pool);
    let alloc_service = AllocationService::new(repo.clone());

    // Create quota
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
    repo.create_quota(quota_req).await.expect("Failed to create quota");

    // Try to allocate more than limit
    let result = alloc_service
        .allocate(
            EntityType::User,
            "diana",
            Uuid::new_v4(),
            ResourceType::GpuHours,
            dec!(150.0),
            None,
        )
        .await;

    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), horizon_quota_manager::QuotaError::QuotaExceeded(_)));
}

#[tokio::test]
async fn test_allocation_service_hierarchical_enforcement() {
    let pool = setup_test_db().await;
    let repo = QuotaRepository::new(pool);
    let quota_service = QuotaService::new(repo.clone());
    let alloc_service = AllocationService::new(repo.clone());

    // Create org with limit 1000
    let org_req = CreateQuotaRequest {
        entity_type: EntityType::Organization,
        entity_id: "org1".to_string(),
        parent_id: None,
        resource_type: ResourceType::GpuHours,
        limit_value: dec!(1000.0),
        soft_limit: None,
        burst_limit: None,
        overcommit_ratio: None,
    };
    let org_quota = quota_service.create_quota(org_req).await.expect("Failed to create org");

    // Create team with limit 500
    let team_req = CreateQuotaRequest {
        entity_type: EntityType::Team,
        entity_id: "team1".to_string(),
        parent_id: Some(org_quota.id),
        resource_type: ResourceType::GpuHours,
        limit_value: dec!(500.0),
        soft_limit: None,
        burst_limit: None,
        overcommit_ratio: None,
    };
    let team_quota = quota_service.create_quota(team_req).await.expect("Failed to create team");

    // Create user with limit 200
    let user_req = CreateQuotaRequest {
        entity_type: EntityType::User,
        entity_id: "user1".to_string(),
        parent_id: Some(team_quota.id),
        resource_type: ResourceType::GpuHours,
        limit_value: dec!(200.0),
        soft_limit: None,
        burst_limit: None,
        overcommit_ratio: None,
    };
    quota_service.create_quota(user_req).await.expect("Failed to create user");

    // Allocate 150 to user (within user limit)
    let alloc1 = alloc_service
        .allocate(
            EntityType::User,
            "user1",
            Uuid::new_v4(),
            ResourceType::GpuHours,
            dec!(150.0),
            None,
        )
        .await
        .expect("Failed to allocate");

    assert_eq!(alloc1.allocated_value, dec!(150.0));

    // Try to allocate another 100 (would exceed user limit)
    let result = alloc_service
        .allocate(
            EntityType::User,
            "user1",
            Uuid::new_v4(),
            ResourceType::GpuHours,
            dec!(100.0),
            None,
        )
        .await;

    assert!(result.is_err());
}

#[tokio::test]
async fn test_quota_service_get_usage_stats() {
    let pool = setup_test_db().await;
    let repo = QuotaRepository::new(pool);
    let quota_service = QuotaService::new(repo.clone());
    let alloc_service = AllocationService::new(repo.clone());

    // Create quota
    let req = CreateQuotaRequest {
        entity_type: EntityType::User,
        entity_id: "eve".to_string(),
        parent_id: None,
        resource_type: ResourceType::StorageGb,
        limit_value: dec!(1000.0),
        soft_limit: Some(dec!(800.0)),
        burst_limit: None,
        overcommit_ratio: None,
    };
    let quota = repo.create_quota(req).await.expect("Failed to create quota");

    // Allocate 600GB
    alloc_service
        .allocate(
            EntityType::User,
            "eve",
            Uuid::new_v4(),
            ResourceType::StorageGb,
            dec!(600.0),
            None,
        )
        .await
        .expect("Failed to allocate");

    // Get stats
    let stats = quota_service.get_usage_stats(quota.id).await.expect("Failed to get stats");

    assert_eq!(stats.limit, dec!(1000.0));
    assert_eq!(stats.usage, dec!(600.0));
    assert_eq!(stats.available, dec!(400.0));
    assert_eq!(stats.utilization_percent, 60.0);
}
