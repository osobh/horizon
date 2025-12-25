//! Resource pool service for managing shared ephemeral resource pools.
//!
//! Provides:
//! - Pool lifecycle management (create, update, pause, resume, archive)
//! - Allocation requests with approval workflows
//! - Auto-approval based on email domains
//! - Usage tracking and statistics
//! - Integration with ephemeral quotas

use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::{
    db::QuotaRepository,
    error::{HpcError, QuotaErrorExt, Result},
    models::{
        AllocationRequestStatus, CreatePoolRequest, PoolAllocation, PoolAllocationRequest,
        PoolStats, PoolStatus, PoolType, ResourcePool, ResourceType, UpdatePoolRequest,
    },
};

/// Configuration for the pool service.
#[derive(Debug, Clone)]
pub struct PoolServiceConfig {
    /// Maximum pools per tenant (default: 50)
    pub max_pools_per_tenant: usize,
    /// Default allocation expiry duration in hours (default: 168 = 7 days)
    pub default_allocation_hours: i64,
    /// Enable automatic expiry processing
    pub auto_expiry_enabled: bool,
    /// Grace period before archiving expired pools (hours)
    pub archive_grace_period_hours: i64,
}

impl Default for PoolServiceConfig {
    fn default() -> Self {
        Self {
            max_pools_per_tenant: 50,
            default_allocation_hours: 168,
            auto_expiry_enabled: true,
            archive_grace_period_hours: 24,
        }
    }
}

/// Service for managing resource pools.
#[derive(Clone)]
pub struct ResourcePoolService {
    #[allow(dead_code)]
    repository: QuotaRepository,
    config: PoolServiceConfig,
    // In-memory cache for pools
    pools: Arc<RwLock<HashMap<Uuid, ResourcePool>>>,
    // Allocations by pool ID
    allocations: Arc<RwLock<HashMap<Uuid, Vec<PoolAllocation>>>>,
    // Allocations by user ID for quick lookup
    user_allocations: Arc<RwLock<HashMap<String, Vec<Uuid>>>>,
}

impl ResourcePoolService {
    /// Create a new resource pool service.
    pub fn new(repository: QuotaRepository) -> Self {
        Self::with_config(repository, PoolServiceConfig::default())
    }

    /// Create with custom configuration.
    pub fn with_config(repository: QuotaRepository, config: PoolServiceConfig) -> Self {
        Self {
            repository,
            config,
            pools: Arc::new(RwLock::new(HashMap::new())),
            allocations: Arc::new(RwLock::new(HashMap::new())),
            user_allocations: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create a new resource pool.
    pub async fn create_pool(&self, req: CreatePoolRequest) -> Result<ResourcePool> {
        // Check tenant limits
        let tenant_pools = self.get_pools_by_tenant(req.tenant_id).await?;
        if tenant_pools.len() >= self.config.max_pools_per_tenant {
            return Err(HpcError::quota_exceeded(format!(
                "Tenant has reached maximum pool limit of {}",
                self.config.max_pools_per_tenant
            )));
        }

        // Create the pool
        let mut pool = ResourcePool::new(
            &req.name,
            req.tenant_id,
            req.pool_type,
            req.resource_type,
            req.total_limit,
        );

        // Apply optional settings
        if let Some(desc) = req.description {
            pool = pool.with_description(desc);
        }

        if let Some(max) = req.max_allocation_per_user {
            pool = pool.with_max_per_user(max);
        }

        if let Some(min) = req.min_allocation_per_request {
            pool.min_allocation_per_request = min;
        }

        if let Some(requires) = req.requires_approval {
            pool.requires_approval = requires;
        }

        if let Some(domains) = req.auto_approve_domains {
            pool = pool.with_auto_approve_domains(domains);
        }

        if let Some(max_users) = req.max_concurrent_users {
            pool.max_concurrent_users = Some(max_users);
        }

        if let Some(starts) = req.starts_at {
            pool = pool.with_start(starts);
        }

        if let Some(expires) = req.expires_at {
            pool = pool.with_expiry(expires);
        }

        if let Some(sponsor) = req.sponsor_id {
            pool = pool.with_sponsor(sponsor);
        }

        if let Some(tw_id) = req.time_window_id {
            pool = pool.with_time_window(tw_id);
        }

        // Validate
        pool.validate()?;

        // Store
        {
            let mut pools = self.pools.write().await;
            pools.insert(pool.id, pool.clone());
        }
        {
            let mut allocations = self.allocations.write().await;
            allocations.insert(pool.id, Vec::new());
        }

        Ok(pool)
    }

    /// Get a pool by ID.
    pub async fn get_pool(&self, pool_id: Uuid) -> Result<ResourcePool> {
        let pools = self.pools.read().await;
        pools
            .get(&pool_id)
            .cloned()
            .ok_or_else(|| HpcError::not_found("resource_pool", pool_id.to_string()))
    }

    /// Get pools by tenant.
    pub async fn get_pools_by_tenant(&self, tenant_id: Uuid) -> Result<Vec<ResourcePool>> {
        let pools = self.pools.read().await;
        Ok(pools
            .values()
            .filter(|p| p.tenant_id == tenant_id)
            .cloned()
            .collect())
    }

    /// Get pools by type.
    pub async fn get_pools_by_type(&self, pool_type: PoolType) -> Result<Vec<ResourcePool>> {
        let pools = self.pools.read().await;
        Ok(pools
            .values()
            .filter(|p| p.pool_type == pool_type)
            .cloned()
            .collect())
    }

    /// Get active pools (can accept allocations).
    pub async fn get_active_pools(&self) -> Result<Vec<ResourcePool>> {
        let pools = self.pools.read().await;
        Ok(pools.values().filter(|p| p.can_allocate()).cloned().collect())
    }

    /// Update a pool.
    pub async fn update_pool(&self, pool_id: Uuid, req: UpdatePoolRequest) -> Result<ResourcePool> {
        let mut pool = self.get_pool(pool_id).await?;

        if let Some(name) = req.name {
            pool.name = name;
        }

        if let Some(desc) = req.description {
            pool.description = Some(desc);
        }

        if let Some(limit) = req.total_limit {
            if limit < pool.allocated {
                return Err(HpcError::invalid_input(
                    "total_limit",
                    format!(
                        "New limit {} is below current allocation {}",
                        limit, pool.allocated
                    ),
                ));
            }
            pool.total_limit = limit;
        }

        if let Some(max) = req.max_allocation_per_user {
            pool.max_allocation_per_user = max;
        }

        if let Some(min) = req.min_allocation_per_request {
            pool.min_allocation_per_request = min;
        }

        if let Some(requires) = req.requires_approval {
            pool.requires_approval = requires;
        }

        if let Some(domains) = req.auto_approve_domains {
            pool.auto_approve_domains = domains;
        }

        if let Some(max_users) = req.max_concurrent_users {
            pool.max_concurrent_users = Some(max_users);
        }

        if let Some(starts) = req.starts_at {
            pool.starts_at = Some(starts);
        }

        if let Some(expires) = req.expires_at {
            pool.expires_at = Some(expires);
        }

        if let Some(status) = req.status {
            pool.status = status;
        }

        pool.updated_at = Utc::now();
        pool.validate()?;

        {
            let mut pools = self.pools.write().await;
            pools.insert(pool.id, pool.clone());
        }

        Ok(pool)
    }

    /// Request an allocation from a pool.
    pub async fn request_allocation(
        &self,
        req: PoolAllocationRequest,
    ) -> Result<PoolAllocation> {
        let mut pool = self.get_pool(req.pool_id).await?;

        if !pool.can_allocate() {
            return Err(HpcError::quota_exceeded(format!(
                "Pool {} is not accepting allocations (status: {})",
                pool.name,
                pool.status.as_str()
            )));
        }

        // Validate amount
        if req.requested_amount < pool.min_allocation_per_request {
            return Err(HpcError::invalid_input(
                "requested_amount",
                format!(
                    "Amount {} is below minimum {}",
                    req.requested_amount, pool.min_allocation_per_request
                ),
            ));
        }

        if req.requested_amount > pool.max_allocation_per_user {
            return Err(HpcError::invalid_input(
                "requested_amount",
                format!(
                    "Amount {} exceeds max per user {}",
                    req.requested_amount, pool.max_allocation_per_user
                ),
            ));
        }

        // Check user's existing allocations
        let existing = self.get_user_allocations_for_pool(&req.user_id, req.pool_id).await?;
        let total_existing: Decimal = existing
            .iter()
            .filter(|a| a.is_active())
            .map(|a| a.allocated_amount)
            .sum();

        if total_existing + req.requested_amount > pool.max_allocation_per_user {
            return Err(HpcError::quota_exceeded(format!(
                "Total allocation would exceed max per user (existing: {}, requested: {}, max: {})",
                total_existing, req.requested_amount, pool.max_allocation_per_user
            )));
        }

        // Calculate expiry
        let expires_at = pool.expires_at.unwrap_or_else(|| {
            Utc::now() + chrono::Duration::hours(self.config.default_allocation_hours)
        });

        // Create allocation
        let mut allocation =
            PoolAllocation::new_pending(req.pool_id, &req.user_id, req.requested_amount, expires_at);

        if let Some(purpose) = req.purpose {
            allocation = allocation.with_purpose(purpose);
        }

        if let Some(identity_id) = req.ephemeral_identity_id {
            allocation = allocation.with_ephemeral_identity(identity_id);
        }

        // Check for auto-approval
        let should_auto_approve = if let Some(ref email) = req.user_email {
            pool.is_auto_approve_domain(email)
        } else {
            false
        };

        if should_auto_approve || !pool.requires_approval {
            // Auto-approve and allocate immediately
            if pool.available() < req.requested_amount {
                return Err(HpcError::quota_exceeded(format!(
                    "Pool has insufficient capacity ({} available, {} requested)",
                    pool.available(),
                    req.requested_amount
                )));
            }

            allocation.auto_approve();
            pool.allocated += req.requested_amount;
            pool.current_users += 1;
            pool.updated_at = Utc::now();

            if pool.available() == Decimal::ZERO {
                pool.status = PoolStatus::Exhausted;
            }
        } else {
            // Reserve capacity for pending request
            pool.reserve(req.requested_amount)?;
        }

        // Store allocation and update pool
        {
            let mut pools = self.pools.write().await;
            pools.insert(pool.id, pool);
        }
        {
            let mut allocations = self.allocations.write().await;
            allocations
                .entry(req.pool_id)
                .or_insert_with(Vec::new)
                .push(allocation.clone());
        }
        {
            let mut user_allocs = self.user_allocations.write().await;
            user_allocs
                .entry(req.user_id.clone())
                .or_insert_with(Vec::new)
                .push(allocation.id);
        }

        Ok(allocation)
    }

    /// Get an allocation by ID.
    pub async fn get_allocation(&self, allocation_id: Uuid) -> Result<PoolAllocation> {
        let allocations = self.allocations.read().await;
        for pool_allocs in allocations.values() {
            if let Some(alloc) = pool_allocs.iter().find(|a| a.id == allocation_id) {
                return Ok(alloc.clone());
            }
        }
        Err(HpcError::not_found("pool_allocation", allocation_id.to_string()))
    }

    /// Get allocations for a pool.
    pub async fn get_pool_allocations(&self, pool_id: Uuid) -> Result<Vec<PoolAllocation>> {
        let allocations = self.allocations.read().await;
        Ok(allocations.get(&pool_id).cloned().unwrap_or_default())
    }

    /// Get pending allocations for a pool (awaiting approval).
    pub async fn get_pending_allocations(&self, pool_id: Uuid) -> Result<Vec<PoolAllocation>> {
        let allocations = self.get_pool_allocations(pool_id).await?;
        Ok(allocations.into_iter().filter(|a| a.is_pending()).collect())
    }

    /// Get user's allocations.
    pub async fn get_user_allocations(&self, user_id: &str) -> Result<Vec<PoolAllocation>> {
        let user_allocs = self.user_allocations.read().await;
        let allocation_ids = user_allocs.get(user_id).cloned().unwrap_or_default();

        let allocations = self.allocations.read().await;
        let mut result = Vec::new();

        for pool_allocs in allocations.values() {
            for alloc in pool_allocs {
                if allocation_ids.contains(&alloc.id) {
                    result.push(alloc.clone());
                }
            }
        }

        Ok(result)
    }

    /// Get user's allocations for a specific pool.
    async fn get_user_allocations_for_pool(
        &self,
        user_id: &str,
        pool_id: Uuid,
    ) -> Result<Vec<PoolAllocation>> {
        let allocations = self.get_pool_allocations(pool_id).await?;
        Ok(allocations
            .into_iter()
            .filter(|a| a.user_id == user_id)
            .collect())
    }

    /// Approve a pending allocation.
    pub async fn approve_allocation(
        &self,
        allocation_id: Uuid,
        approved_by: Uuid,
    ) -> Result<PoolAllocation> {
        let mut allocation = self.get_allocation(allocation_id).await?;

        if !allocation.is_pending() {
            return Err(HpcError::invalid_input(
                "allocation",
                format!("Allocation is not pending (status: {:?})", allocation.status),
            ));
        }

        let mut pool = self.get_pool(allocation.pool_id).await?;

        // Check if pool still has capacity
        if pool.available() < allocation.allocated_amount {
            return Err(HpcError::quota_exceeded(format!(
                "Pool no longer has sufficient capacity ({} available, {} needed)",
                pool.available(),
                allocation.allocated_amount
            )));
        }

        // Convert reservation to allocation
        pool.cancel_reservation(allocation.allocated_amount);
        pool.allocated += allocation.allocated_amount;
        pool.current_users += 1;
        pool.updated_at = Utc::now();

        if pool.available() == Decimal::ZERO {
            pool.status = PoolStatus::Exhausted;
        }

        allocation.approve(approved_by);

        // Update storage
        self.update_allocation_in_store(allocation.clone()).await?;
        {
            let mut pools = self.pools.write().await;
            pools.insert(pool.id, pool);
        }

        Ok(allocation)
    }

    /// Reject a pending allocation.
    pub async fn reject_allocation(&self, allocation_id: Uuid) -> Result<PoolAllocation> {
        let mut allocation = self.get_allocation(allocation_id).await?;

        if !allocation.is_pending() {
            return Err(HpcError::invalid_input(
                "allocation",
                format!("Allocation is not pending (status: {:?})", allocation.status),
            ));
        }

        let mut pool = self.get_pool(allocation.pool_id).await?;

        // Release the reservation
        pool.cancel_reservation(allocation.allocated_amount);
        pool.updated_at = Utc::now();

        allocation.reject();

        // Update storage
        self.update_allocation_in_store(allocation.clone()).await?;
        {
            let mut pools = self.pools.write().await;
            pools.insert(pool.id, pool);
        }

        Ok(allocation)
    }

    /// Record usage against an allocation.
    pub async fn record_usage(
        &self,
        allocation_id: Uuid,
        amount: Decimal,
    ) -> Result<PoolAllocation> {
        let mut allocation = self.get_allocation(allocation_id).await?;

        if !allocation.is_active() {
            return Err(HpcError::invalid_input(
                "allocation",
                "Allocation is not active",
            ));
        }

        allocation.record_usage(amount)?;
        self.update_allocation_in_store(allocation.clone()).await?;

        Ok(allocation)
    }

    /// Release an allocation back to the pool.
    pub async fn release_allocation(&self, allocation_id: Uuid) -> Result<PoolAllocation> {
        let mut allocation = self.get_allocation(allocation_id).await?;

        if !allocation.is_active() {
            return Err(HpcError::invalid_input(
                "allocation",
                "Allocation is not active",
            ));
        }

        let mut pool = self.get_pool(allocation.pool_id).await?;

        // Return unused portion to pool
        let unused = allocation.remaining();
        pool.release(unused);

        allocation.release();

        // Update storage
        self.update_allocation_in_store(allocation.clone()).await?;
        {
            let mut pools = self.pools.write().await;
            pools.insert(pool.id, pool);
        }

        Ok(allocation)
    }

    /// Update an allocation in storage.
    async fn update_allocation_in_store(&self, allocation: PoolAllocation) -> Result<()> {
        let mut allocations = self.allocations.write().await;
        if let Some(pool_allocs) = allocations.get_mut(&allocation.pool_id) {
            if let Some(existing) = pool_allocs.iter_mut().find(|a| a.id == allocation.id) {
                *existing = allocation;
            }
        }
        Ok(())
    }

    /// Pause a pool.
    pub async fn pause_pool(&self, pool_id: Uuid) -> Result<ResourcePool> {
        let mut pool = self.get_pool(pool_id).await?;
        pool.pause();

        let mut pools = self.pools.write().await;
        pools.insert(pool.id, pool.clone());

        Ok(pool)
    }

    /// Resume a pool.
    pub async fn resume_pool(&self, pool_id: Uuid) -> Result<ResourcePool> {
        let mut pool = self.get_pool(pool_id).await?;
        pool.resume();

        let mut pools = self.pools.write().await;
        pools.insert(pool.id, pool.clone());

        Ok(pool)
    }

    /// Archive a pool.
    pub async fn archive_pool(&self, pool_id: Uuid) -> Result<ResourcePool> {
        let mut pool = self.get_pool(pool_id).await?;
        pool.archive();

        let mut pools = self.pools.write().await;
        pools.insert(pool.id, pool.clone());

        Ok(pool)
    }

    /// Get pool statistics.
    pub async fn get_pool_stats(&self, pool_id: Uuid) -> Result<PoolStats> {
        let pool = self.get_pool(pool_id).await?;
        let allocations = self.get_pool_allocations(pool_id).await?;

        let pending_requests = allocations.iter().filter(|a| a.is_pending()).count() as i32;
        let total_users = allocations.iter().map(|a| &a.user_id).collect::<std::collections::HashSet<_>>().len() as i32;

        let mut stats = pool.stats();
        stats.pending_requests = pending_requests;
        stats.total_users_served = total_users;

        Ok(stats)
    }

    /// Process expired pools.
    pub async fn process_pool_expirations(&self) -> Result<Vec<Uuid>> {
        let mut expired = Vec::new();

        let mut pools = self.pools.write().await;
        for pool in pools.values_mut() {
            if pool.is_expired() && pool.status != PoolStatus::Expired && pool.status != PoolStatus::Archived {
                pool.mark_expired();
                expired.push(pool.id);
            }
        }

        Ok(expired)
    }

    /// Process expired allocations.
    pub async fn process_allocation_expirations(&self) -> Result<Vec<Uuid>> {
        let mut expired = Vec::new();
        let now = Utc::now();

        let mut allocations = self.allocations.write().await;
        let mut pools = self.pools.write().await;

        for pool_allocs in allocations.values_mut() {
            for alloc in pool_allocs.iter_mut() {
                if alloc.expires_at <= now && alloc.is_active() {
                    // Get the pool
                    if let Some(pool) = pools.get_mut(&alloc.pool_id) {
                        let unused = alloc.remaining();
                        pool.release(unused);
                    }

                    alloc.status = AllocationRequestStatus::Expired;
                    alloc.updated_at = now;
                    expired.push(alloc.id);
                }
            }
        }

        Ok(expired)
    }

    /// Get service statistics.
    pub async fn get_service_stats(&self) -> PoolServiceStats {
        let pools = self.pools.read().await;
        let allocations = self.allocations.read().await;

        let total_pools = pools.len();
        let active_pools = pools.values().filter(|p| p.can_allocate()).count();

        let mut total_allocations = 0;
        let mut active_allocations = 0;
        let mut pending_allocations = 0;

        for pool_allocs in allocations.values() {
            total_allocations += pool_allocs.len();
            active_allocations += pool_allocs.iter().filter(|a| a.is_active()).count();
            pending_allocations += pool_allocs.iter().filter(|a| a.is_pending()).count();
        }

        let total_capacity: Decimal = pools.values().map(|p| p.total_limit).sum();
        let total_allocated: Decimal = pools.values().map(|p| p.allocated).sum();

        PoolServiceStats {
            total_pools,
            active_pools,
            total_allocations,
            active_allocations,
            pending_allocations,
            total_capacity,
            total_allocated,
        }
    }
}

/// Statistics for the pool service.
#[derive(Debug, Clone)]
pub struct PoolServiceStats {
    pub total_pools: usize,
    pub active_pools: usize,
    pub total_allocations: usize,
    pub active_allocations: usize,
    pub pending_allocations: usize,
    pub total_capacity: Decimal,
    pub total_allocated: Decimal,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::QuotaRepository;
    use rust_decimal_macros::dec;

    fn mock_repository() -> QuotaRepository {
        QuotaRepository::new_test()
    }

    async fn create_test_service() -> ResourcePoolService {
        ResourcePoolService::new(mock_repository())
    }

    fn tenant_id() -> Uuid {
        Uuid::new_v4()
    }

    fn create_pool_request() -> CreatePoolRequest {
        CreatePoolRequest {
            name: "Test Pool".to_string(),
            description: None,
            tenant_id: tenant_id(),
            pool_type: PoolType::Trial,
            resource_type: ResourceType::GpuHours,
            total_limit: dec!(1000),
            max_allocation_per_user: None,
            min_allocation_per_request: None,
            requires_approval: None,
            auto_approve_domains: None,
            max_concurrent_users: None,
            starts_at: None,
            expires_at: None,
            sponsor_id: None,
            time_window_id: None,
        }
    }

    #[tokio::test]
    async fn test_create_pool() {
        let service = create_test_service().await;
        let req = create_pool_request();

        let pool = service.create_pool(req.clone()).await.unwrap();

        assert_eq!(pool.name, req.name);
        assert_eq!(pool.pool_type, PoolType::Trial);
        assert_eq!(pool.total_limit, dec!(1000));
        assert_eq!(pool.status, PoolStatus::Active);
    }

    #[tokio::test]
    async fn test_create_pool_with_options() {
        let service = create_test_service().await;
        let mut req = create_pool_request();
        req.requires_approval = Some(true);
        req.auto_approve_domains = Some(vec!["university.edu".to_string()]);
        req.max_concurrent_users = Some(10);

        let pool = service.create_pool(req).await.unwrap();

        assert!(pool.requires_approval);
        assert_eq!(pool.auto_approve_domains, vec!["university.edu".to_string()]);
        assert_eq!(pool.max_concurrent_users, Some(10));
    }

    #[tokio::test]
    async fn test_request_allocation_auto_approve() {
        let service = create_test_service().await;
        let req = create_pool_request();
        let pool = service.create_pool(req).await.unwrap();

        let alloc_req = PoolAllocationRequest {
            pool_id: pool.id,
            user_id: "user1".to_string(),
            user_email: None,
            requested_amount: dec!(5),
            purpose: Some("Testing".to_string()),
            ephemeral_identity_id: None,
        };

        let allocation = service.request_allocation(alloc_req).await.unwrap();

        // Trial pools don't require approval by default
        assert_eq!(allocation.status, AllocationRequestStatus::AutoApproved);
        assert!(allocation.is_active());

        let updated_pool = service.get_pool(pool.id).await.unwrap();
        assert_eq!(updated_pool.allocated, dec!(5));
    }

    #[tokio::test]
    async fn test_request_allocation_requires_approval() {
        let service = create_test_service().await;
        let mut req = create_pool_request();
        req.pool_type = PoolType::Research; // Research requires approval
        let pool = service.create_pool(req).await.unwrap();

        let alloc_req = PoolAllocationRequest {
            pool_id: pool.id,
            user_id: "user1".to_string(),
            user_email: None,
            requested_amount: dec!(50),
            purpose: None,
            ephemeral_identity_id: None,
        };

        let allocation = service.request_allocation(alloc_req).await.unwrap();

        assert_eq!(allocation.status, AllocationRequestStatus::Pending);
        assert!(allocation.is_pending());

        let updated_pool = service.get_pool(pool.id).await.unwrap();
        assert_eq!(updated_pool.reserved, dec!(50));
        assert_eq!(updated_pool.allocated, dec!(0));
    }

    #[tokio::test]
    async fn test_request_allocation_domain_auto_approve() {
        let service = create_test_service().await;
        let mut req = create_pool_request();
        req.pool_type = PoolType::Research;
        req.auto_approve_domains = Some(vec!["university.edu".to_string()]);
        let pool = service.create_pool(req).await.unwrap();

        let alloc_req = PoolAllocationRequest {
            pool_id: pool.id,
            user_id: "user1".to_string(),
            user_email: Some("alice@university.edu".to_string()),
            requested_amount: dec!(50),
            purpose: None,
            ephemeral_identity_id: None,
        };

        let allocation = service.request_allocation(alloc_req).await.unwrap();

        assert_eq!(allocation.status, AllocationRequestStatus::AutoApproved);
    }

    #[tokio::test]
    async fn test_approve_allocation() {
        let service = create_test_service().await;
        let mut req = create_pool_request();
        req.pool_type = PoolType::Research;
        let pool = service.create_pool(req).await.unwrap();

        let alloc_req = PoolAllocationRequest {
            pool_id: pool.id,
            user_id: "user1".to_string(),
            user_email: None,
            requested_amount: dec!(50),
            purpose: None,
            ephemeral_identity_id: None,
        };

        let allocation = service.request_allocation(alloc_req).await.unwrap();
        let approved = service
            .approve_allocation(allocation.id, Uuid::new_v4())
            .await
            .unwrap();

        assert_eq!(approved.status, AllocationRequestStatus::Approved);

        let updated_pool = service.get_pool(pool.id).await.unwrap();
        assert_eq!(updated_pool.allocated, dec!(50));
        assert_eq!(updated_pool.reserved, dec!(0));
    }

    #[tokio::test]
    async fn test_reject_allocation() {
        let service = create_test_service().await;
        let mut req = create_pool_request();
        req.pool_type = PoolType::Research;
        let pool = service.create_pool(req).await.unwrap();

        let alloc_req = PoolAllocationRequest {
            pool_id: pool.id,
            user_id: "user1".to_string(),
            user_email: None,
            requested_amount: dec!(50),
            purpose: None,
            ephemeral_identity_id: None,
        };

        let allocation = service.request_allocation(alloc_req).await.unwrap();
        let rejected = service.reject_allocation(allocation.id).await.unwrap();

        assert_eq!(rejected.status, AllocationRequestStatus::Rejected);

        let updated_pool = service.get_pool(pool.id).await.unwrap();
        assert_eq!(updated_pool.reserved, dec!(0));
    }

    #[tokio::test]
    async fn test_record_usage() {
        let service = create_test_service().await;
        let req = create_pool_request();
        let pool = service.create_pool(req).await.unwrap();

        let alloc_req = PoolAllocationRequest {
            pool_id: pool.id,
            user_id: "user1".to_string(),
            user_email: None,
            requested_amount: dec!(10),
            purpose: None,
            ephemeral_identity_id: None,
        };

        let allocation = service.request_allocation(alloc_req).await.unwrap();
        let updated = service.record_usage(allocation.id, dec!(5)).await.unwrap();

        assert_eq!(updated.used_amount, dec!(5));
        assert_eq!(updated.remaining(), dec!(5));
    }

    #[tokio::test]
    async fn test_release_allocation() {
        let service = create_test_service().await;
        let req = create_pool_request();
        let pool = service.create_pool(req).await.unwrap();

        let alloc_req = PoolAllocationRequest {
            pool_id: pool.id,
            user_id: "user1".to_string(),
            user_email: None,
            requested_amount: dec!(10),
            purpose: None,
            ephemeral_identity_id: None,
        };

        let allocation = service.request_allocation(alloc_req).await.unwrap();
        service.record_usage(allocation.id, dec!(3)).await.unwrap();

        let released = service.release_allocation(allocation.id).await.unwrap();
        assert_eq!(released.status, AllocationRequestStatus::Released);

        // Pool should have 7 returned (10 - 3 used)
        let updated_pool = service.get_pool(pool.id).await.unwrap();
        assert_eq!(updated_pool.allocated, dec!(3)); // Only used amount remains accounted
    }

    #[tokio::test]
    async fn test_pause_resume_pool() {
        let service = create_test_service().await;
        let req = create_pool_request();
        let pool = service.create_pool(req).await.unwrap();

        let paused = service.pause_pool(pool.id).await.unwrap();
        assert_eq!(paused.status, PoolStatus::Paused);

        let resumed = service.resume_pool(pool.id).await.unwrap();
        assert_eq!(resumed.status, PoolStatus::Active);
    }

    #[tokio::test]
    async fn test_archive_pool() {
        let service = create_test_service().await;
        let req = create_pool_request();
        let pool = service.create_pool(req).await.unwrap();

        let archived = service.archive_pool(pool.id).await.unwrap();
        assert_eq!(archived.status, PoolStatus::Archived);
    }

    #[tokio::test]
    async fn test_get_pool_stats() {
        let service = create_test_service().await;
        let req = create_pool_request();
        let pool = service.create_pool(req).await.unwrap();

        // Create some allocations
        for i in 0..3 {
            let alloc_req = PoolAllocationRequest {
                pool_id: pool.id,
                user_id: format!("user{}", i),
                user_email: None,
                requested_amount: dec!(10),
                purpose: None,
                ephemeral_identity_id: None,
            };
            service.request_allocation(alloc_req).await.unwrap();
        }

        let stats = service.get_pool_stats(pool.id).await.unwrap();
        assert_eq!(stats.allocated, dec!(30));
        assert_eq!(stats.active_allocations, 3);
        assert_eq!(stats.total_users_served, 3);
    }

    #[tokio::test]
    async fn test_get_service_stats() {
        let service = create_test_service().await;

        // Create two pools
        let req1 = create_pool_request();
        let pool1 = service.create_pool(req1).await.unwrap();

        let mut req2 = create_pool_request();
        req2.name = "Pool 2".to_string();
        req2.total_limit = dec!(500);
        let _pool2 = service.create_pool(req2).await.unwrap();

        // Create allocation
        let alloc_req = PoolAllocationRequest {
            pool_id: pool1.id,
            user_id: "user1".to_string(),
            user_email: None,
            requested_amount: dec!(10),
            purpose: None,
            ephemeral_identity_id: None,
        };
        service.request_allocation(alloc_req).await.unwrap();

        let stats = service.get_service_stats().await;
        assert_eq!(stats.total_pools, 2);
        assert_eq!(stats.active_pools, 2);
        assert_eq!(stats.total_allocations, 1);
        assert_eq!(stats.total_capacity, dec!(1500));
        assert_eq!(stats.total_allocated, dec!(10));
    }

    #[tokio::test]
    async fn test_request_allocation_exceeds_max_per_user() {
        let service = create_test_service().await;
        let mut req = create_pool_request();
        req.max_allocation_per_user = Some(dec!(20));
        let pool = service.create_pool(req).await.unwrap();

        let alloc_req = PoolAllocationRequest {
            pool_id: pool.id,
            user_id: "user1".to_string(),
            user_email: None,
            requested_amount: dec!(30),
            purpose: None,
            ephemeral_identity_id: None,
        };

        let result = service.request_allocation(alloc_req).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_request_allocation_cumulative_max() {
        let service = create_test_service().await;
        let mut req = create_pool_request();
        req.max_allocation_per_user = Some(dec!(20));
        let pool = service.create_pool(req).await.unwrap();

        // First allocation
        let alloc_req1 = PoolAllocationRequest {
            pool_id: pool.id,
            user_id: "user1".to_string(),
            user_email: None,
            requested_amount: dec!(15),
            purpose: None,
            ephemeral_identity_id: None,
        };
        service.request_allocation(alloc_req1).await.unwrap();

        // Second allocation that would exceed max
        let alloc_req2 = PoolAllocationRequest {
            pool_id: pool.id,
            user_id: "user1".to_string(),
            user_email: None,
            requested_amount: dec!(10),
            purpose: None,
            ephemeral_identity_id: None,
        };
        let result = service.request_allocation(alloc_req2).await;
        assert!(result.is_err());
    }
}
