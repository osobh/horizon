//! Resource pools for shared ephemeral quota allocation.
//!
//! Provides pooled resources for bounties, hackathons, trial access,
//! and research grants with approval workflows and domain-based auto-approval.

use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use sqlx::FromRow;
use utoipa::ToSchema;
use uuid::Uuid;

use super::ResourceType;
use crate::error::{HpcError, QuotaErrorExt, Result};

/// Type of resource pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, sqlx::Type, ToSchema)]
#[sqlx(type_name = "text", rename_all = "snake_case")]
#[serde(rename_all = "snake_case")]
pub enum PoolType {
    /// Bounty pool for external contributor rewards
    Bounty,
    /// Trial/demo access for potential users
    Trial,
    /// Hackathon event pools
    Hackathon,
    /// Academic research grants
    Research,
    /// Training/workshop pools
    Training,
    /// Open source contributor access
    OpenSource,
}

impl PoolType {
    pub fn as_str(&self) -> &'static str {
        match self {
            PoolType::Bounty => "bounty",
            PoolType::Trial => "trial",
            PoolType::Hackathon => "hackathon",
            PoolType::Research => "research",
            PoolType::Training => "training",
            PoolType::OpenSource => "open_source",
        }
    }

    pub fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "bounty" => Ok(PoolType::Bounty),
            "trial" => Ok(PoolType::Trial),
            "hackathon" => Ok(PoolType::Hackathon),
            "research" => Ok(PoolType::Research),
            "training" => Ok(PoolType::Training),
            "open_source" => Ok(PoolType::OpenSource),
            _ => Err(HpcError::invalid_input(
                "pool_type",
                format!("Invalid pool type: {}", s),
            )),
        }
    }

    /// Get the default max allocation per user for this pool type.
    pub fn default_max_per_user(&self) -> Decimal {
        match self {
            PoolType::Bounty => Decimal::from(100),      // 100 GPU-hours per bounty
            PoolType::Trial => Decimal::from(10),        // 10 GPU-hours for trial
            PoolType::Hackathon => Decimal::from(50),    // 50 GPU-hours per participant
            PoolType::Research => Decimal::from(500),    // 500 GPU-hours for research
            PoolType::Training => Decimal::from(20),     // 20 GPU-hours for training
            PoolType::OpenSource => Decimal::from(50),   // 50 GPU-hours for OSS contributors
        }
    }

    /// Check if this pool type requires approval by default.
    pub fn requires_approval_by_default(&self) -> bool {
        matches!(self, PoolType::Research | PoolType::Bounty)
    }
}

/// Status of a resource pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, sqlx::Type, ToSchema)]
#[sqlx(type_name = "text", rename_all = "snake_case")]
#[serde(rename_all = "snake_case")]
pub enum PoolStatus {
    /// Pool is active and accepting allocations
    Active,
    /// Pool is paused - no new allocations
    Paused,
    /// Pool is exhausted - no remaining capacity
    Exhausted,
    /// Pool has expired
    Expired,
    /// Pool is archived - read-only
    Archived,
}

impl PoolStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            PoolStatus::Active => "active",
            PoolStatus::Paused => "paused",
            PoolStatus::Exhausted => "exhausted",
            PoolStatus::Expired => "expired",
            PoolStatus::Archived => "archived",
        }
    }

    pub fn can_allocate(&self) -> bool {
        matches!(self, PoolStatus::Active)
    }
}

/// A shared resource pool for ephemeral access.
#[derive(Debug, Clone, FromRow, Serialize, Deserialize, ToSchema)]
pub struct ResourcePool {
    /// Unique identifier for this pool
    pub id: Uuid,
    /// Human-readable name for the pool
    pub name: String,
    /// Optional description
    pub description: Option<String>,
    /// Organization that owns this pool
    pub tenant_id: Uuid,
    /// Type of pool (bounty, trial, hackathon, etc.)
    pub pool_type: PoolType,
    /// Resource type this pool provides
    pub resource_type: ResourceType,
    /// Total resource limit for the entire pool
    pub total_limit: Decimal,
    /// Currently allocated from this pool
    pub allocated: Decimal,
    /// Reserved but not yet used
    pub reserved: Decimal,
    /// Maximum allocation per user
    pub max_allocation_per_user: Decimal,
    /// Minimum allocation per request
    pub min_allocation_per_request: Decimal,
    /// Whether allocations require manual approval
    pub requires_approval: bool,
    /// Email domains that get auto-approved (e.g., ["university.edu"])
    #[sqlx(skip)]
    pub auto_approve_domains: Vec<String>,
    /// Maximum number of concurrent users
    pub max_concurrent_users: Option<i32>,
    /// Current number of active users
    pub current_users: i32,
    /// When the pool becomes active
    pub starts_at: Option<DateTime<Utc>>,
    /// When the pool expires
    pub expires_at: Option<DateTime<Utc>>,
    /// Current status of the pool
    pub status: PoolStatus,
    /// Sponsor who funded this pool
    pub sponsor_id: Option<Uuid>,
    /// Optional time window ID for scheduling
    pub time_window_id: Option<Uuid>,
    /// When this pool was created
    pub created_at: DateTime<Utc>,
    /// Last modification time
    pub updated_at: DateTime<Utc>,
}

/// Request to create a new resource pool.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct CreatePoolRequest {
    pub name: String,
    pub description: Option<String>,
    pub tenant_id: Uuid,
    pub pool_type: PoolType,
    pub resource_type: ResourceType,
    pub total_limit: Decimal,
    pub max_allocation_per_user: Option<Decimal>,
    pub min_allocation_per_request: Option<Decimal>,
    pub requires_approval: Option<bool>,
    pub auto_approve_domains: Option<Vec<String>>,
    pub max_concurrent_users: Option<i32>,
    pub starts_at: Option<DateTime<Utc>>,
    pub expires_at: Option<DateTime<Utc>>,
    pub sponsor_id: Option<Uuid>,
    pub time_window_id: Option<Uuid>,
}

/// Request to update an existing pool.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct UpdatePoolRequest {
    pub name: Option<String>,
    pub description: Option<String>,
    pub total_limit: Option<Decimal>,
    pub max_allocation_per_user: Option<Decimal>,
    pub min_allocation_per_request: Option<Decimal>,
    pub requires_approval: Option<bool>,
    pub auto_approve_domains: Option<Vec<String>>,
    pub max_concurrent_users: Option<i32>,
    pub starts_at: Option<DateTime<Utc>>,
    pub expires_at: Option<DateTime<Utc>>,
    pub status: Option<PoolStatus>,
}

/// A request to allocate from a resource pool.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct PoolAllocationRequest {
    pub pool_id: Uuid,
    pub user_id: String,
    pub user_email: Option<String>,
    pub requested_amount: Decimal,
    pub purpose: Option<String>,
    pub ephemeral_identity_id: Option<Uuid>,
}

/// Status of a pool allocation request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, sqlx::Type, ToSchema)]
#[sqlx(type_name = "text", rename_all = "snake_case")]
#[serde(rename_all = "snake_case")]
pub enum AllocationRequestStatus {
    /// Waiting for approval
    Pending,
    /// Approved and allocated
    Approved,
    /// Rejected by admin
    Rejected,
    /// Auto-approved based on domain
    AutoApproved,
    /// Allocation was released
    Released,
    /// Expired before approval
    Expired,
}

/// A tracked allocation from a resource pool.
#[derive(Debug, Clone, FromRow, Serialize, Deserialize, ToSchema)]
pub struct PoolAllocation {
    pub id: Uuid,
    pub pool_id: Uuid,
    pub user_id: String,
    pub ephemeral_identity_id: Option<Uuid>,
    pub allocated_amount: Decimal,
    pub used_amount: Decimal,
    pub status: AllocationRequestStatus,
    pub purpose: Option<String>,
    pub approved_by: Option<Uuid>,
    pub approved_at: Option<DateTime<Utc>>,
    pub released_at: Option<DateTime<Utc>>,
    pub expires_at: DateTime<Utc>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Summary statistics for a resource pool.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct PoolStats {
    pub pool_id: Uuid,
    pub total_limit: Decimal,
    pub allocated: Decimal,
    pub reserved: Decimal,
    pub available: Decimal,
    pub utilization_percent: Decimal,
    pub active_allocations: i32,
    pub pending_requests: i32,
    pub total_users_served: i32,
}

impl ResourcePool {
    /// Create a new resource pool with the given parameters.
    pub fn new(
        name: impl Into<String>,
        tenant_id: Uuid,
        pool_type: PoolType,
        resource_type: ResourceType,
        total_limit: Decimal,
    ) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            name: name.into(),
            description: None,
            tenant_id,
            pool_type,
            resource_type,
            total_limit,
            allocated: Decimal::ZERO,
            reserved: Decimal::ZERO,
            max_allocation_per_user: pool_type.default_max_per_user(),
            min_allocation_per_request: Decimal::ONE,
            requires_approval: pool_type.requires_approval_by_default(),
            auto_approve_domains: Vec::new(),
            max_concurrent_users: None,
            current_users: 0,
            starts_at: None,
            expires_at: None,
            status: PoolStatus::Active,
            sponsor_id: None,
            time_window_id: None,
            created_at: now,
            updated_at: now,
        }
    }

    /// Add a description to this pool.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Set the maximum allocation per user.
    pub fn with_max_per_user(mut self, max: Decimal) -> Self {
        self.max_allocation_per_user = max;
        self
    }

    /// Set domains that get auto-approved.
    pub fn with_auto_approve_domains(mut self, domains: Vec<String>) -> Self {
        self.auto_approve_domains = domains;
        self
    }

    /// Set an expiration time.
    pub fn with_expiry(mut self, expires_at: DateTime<Utc>) -> Self {
        self.expires_at = Some(expires_at);
        self
    }

    /// Set a start time.
    pub fn with_start(mut self, starts_at: DateTime<Utc>) -> Self {
        self.starts_at = Some(starts_at);
        self
    }

    /// Set the sponsor for this pool.
    pub fn with_sponsor(mut self, sponsor_id: Uuid) -> Self {
        self.sponsor_id = Some(sponsor_id);
        self
    }

    /// Set the time window for this pool.
    pub fn with_time_window(mut self, time_window_id: Uuid) -> Self {
        self.time_window_id = Some(time_window_id);
        self
    }

    /// Validate this pool configuration.
    pub fn validate(&self) -> Result<()> {
        if self.name.is_empty() {
            return Err(HpcError::invalid_configuration("Pool name cannot be empty"));
        }

        if self.total_limit <= Decimal::ZERO {
            return Err(HpcError::invalid_configuration(
                "Total limit must be positive",
            ));
        }

        if self.max_allocation_per_user <= Decimal::ZERO {
            return Err(HpcError::invalid_configuration(
                "Max allocation per user must be positive",
            ));
        }

        if self.max_allocation_per_user > self.total_limit {
            return Err(HpcError::invalid_configuration(
                "Max allocation per user cannot exceed total limit",
            ));
        }

        if self.min_allocation_per_request <= Decimal::ZERO {
            return Err(HpcError::invalid_configuration(
                "Min allocation per request must be positive",
            ));
        }

        if self.min_allocation_per_request > self.max_allocation_per_user {
            return Err(HpcError::invalid_configuration(
                "Min allocation per request cannot exceed max per user",
            ));
        }

        if let (Some(starts), Some(expires)) = (self.starts_at, self.expires_at) {
            if expires <= starts {
                return Err(HpcError::invalid_configuration(
                    "Expiry time must be after start time",
                ));
            }
        }

        Ok(())
    }

    /// Get the available capacity in this pool.
    pub fn available(&self) -> Decimal {
        let used = self.allocated + self.reserved;
        if used >= self.total_limit {
            Decimal::ZERO
        } else {
            self.total_limit - used
        }
    }

    /// Calculate utilization percentage (0-100).
    pub fn utilization_percent(&self) -> Decimal {
        if self.total_limit == Decimal::ZERO {
            Decimal::ZERO
        } else {
            (self.allocated / self.total_limit) * Decimal::from(100)
        }
    }

    /// Check if the pool is currently active and can accept allocations.
    pub fn can_allocate(&self) -> bool {
        if !self.status.can_allocate() {
            return false;
        }

        let now = Utc::now();

        // Check start time
        if let Some(starts_at) = self.starts_at {
            if now < starts_at {
                return false;
            }
        }

        // Check expiry
        if let Some(expires_at) = self.expires_at {
            if now >= expires_at {
                return false;
            }
        }

        // Check capacity
        self.available() > Decimal::ZERO
    }

    /// Check if the pool has expired.
    pub fn is_expired(&self) -> bool {
        if let Some(expires_at) = self.expires_at {
            Utc::now() >= expires_at
        } else {
            false
        }
    }

    /// Check if the pool has started.
    pub fn has_started(&self) -> bool {
        if let Some(starts_at) = self.starts_at {
            Utc::now() >= starts_at
        } else {
            true
        }
    }

    /// Check if an email domain qualifies for auto-approval.
    pub fn is_auto_approve_domain(&self, email: &str) -> bool {
        if self.auto_approve_domains.is_empty() {
            return false;
        }

        let domain = email.split('@').last().unwrap_or("");
        self.auto_approve_domains.iter().any(|d| {
            domain == d.as_str() || domain.ends_with(&format!(".{}", d))
        })
    }

    /// Try to allocate resources from this pool.
    pub fn try_allocate(&mut self, amount: Decimal, user_id: &str) -> Result<PoolAllocation> {
        if !self.can_allocate() {
            return Err(HpcError::quota_exceeded("Pool is not accepting allocations"));
        }

        if amount < self.min_allocation_per_request {
            return Err(HpcError::invalid_input(
                "amount",
                format!(
                    "Requested amount {} is below minimum {}",
                    amount, self.min_allocation_per_request
                ),
            ));
        }

        if amount > self.max_allocation_per_user {
            return Err(HpcError::invalid_input(
                "amount",
                format!(
                    "Requested amount {} exceeds max per user {}",
                    amount, self.max_allocation_per_user
                ),
            ));
        }

        if amount > self.available() {
            return Err(HpcError::quota_exceeded(format!(
                "Requested {} but only {} available",
                amount,
                self.available()
            )));
        }

        // Check concurrent user limit
        if let Some(max_users) = self.max_concurrent_users {
            if self.current_users >= max_users {
                return Err(HpcError::quota_exceeded(format!(
                    "Maximum concurrent users ({}) reached",
                    max_users
                )));
            }
        }

        // Perform allocation
        self.allocated += amount;
        self.current_users += 1;
        self.updated_at = Utc::now();

        // Update status if exhausted
        if self.available() == Decimal::ZERO {
            self.status = PoolStatus::Exhausted;
        }

        let now = Utc::now();
        Ok(PoolAllocation {
            id: Uuid::new_v4(),
            pool_id: self.id,
            user_id: user_id.to_string(),
            ephemeral_identity_id: None,
            allocated_amount: amount,
            used_amount: Decimal::ZERO,
            status: AllocationRequestStatus::Approved,
            purpose: None,
            approved_by: None,
            approved_at: Some(now),
            released_at: None,
            expires_at: self.expires_at.unwrap_or(now + chrono::Duration::days(30)),
            created_at: now,
            updated_at: now,
        })
    }

    /// Reserve resources (pending approval).
    pub fn reserve(&mut self, amount: Decimal) -> Result<()> {
        if amount > self.available() {
            return Err(HpcError::quota_exceeded(format!(
                "Cannot reserve {} - only {} available",
                amount,
                self.available()
            )));
        }

        self.reserved += amount;
        self.updated_at = Utc::now();
        Ok(())
    }

    /// Cancel a reservation.
    pub fn cancel_reservation(&mut self, amount: Decimal) {
        self.reserved = (self.reserved - amount).max(Decimal::ZERO);
        self.updated_at = Utc::now();
    }

    /// Release an allocation back to the pool.
    pub fn release(&mut self, amount: Decimal) {
        self.allocated = (self.allocated - amount).max(Decimal::ZERO);
        self.current_users = (self.current_users - 1).max(0);
        self.updated_at = Utc::now();

        // Reactivate if was exhausted
        if self.status == PoolStatus::Exhausted && self.available() > Decimal::ZERO {
            self.status = PoolStatus::Active;
        }
    }

    /// Pause the pool (no new allocations).
    pub fn pause(&mut self) {
        self.status = PoolStatus::Paused;
        self.updated_at = Utc::now();
    }

    /// Resume the pool.
    pub fn resume(&mut self) {
        if self.available() > Decimal::ZERO {
            self.status = PoolStatus::Active;
        } else {
            self.status = PoolStatus::Exhausted;
        }
        self.updated_at = Utc::now();
    }

    /// Archive the pool.
    pub fn archive(&mut self) {
        self.status = PoolStatus::Archived;
        self.updated_at = Utc::now();
    }

    /// Mark as expired.
    pub fn mark_expired(&mut self) {
        self.status = PoolStatus::Expired;
        self.updated_at = Utc::now();
    }

    /// Get pool statistics.
    pub fn stats(&self) -> PoolStats {
        PoolStats {
            pool_id: self.id,
            total_limit: self.total_limit,
            allocated: self.allocated,
            reserved: self.reserved,
            available: self.available(),
            utilization_percent: self.utilization_percent(),
            active_allocations: self.current_users,
            pending_requests: 0, // Would need to be populated from DB
            total_users_served: 0, // Would need to be populated from DB
        }
    }
}

impl PoolAllocation {
    /// Create a new pending allocation request.
    pub fn new_pending(
        pool_id: Uuid,
        user_id: impl Into<String>,
        requested_amount: Decimal,
        expires_at: DateTime<Utc>,
    ) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            pool_id,
            user_id: user_id.into(),
            ephemeral_identity_id: None,
            allocated_amount: requested_amount,
            used_amount: Decimal::ZERO,
            status: AllocationRequestStatus::Pending,
            purpose: None,
            approved_by: None,
            approved_at: None,
            released_at: None,
            expires_at,
            created_at: now,
            updated_at: now,
        }
    }

    /// Link this allocation to an ephemeral identity.
    pub fn with_ephemeral_identity(mut self, identity_id: Uuid) -> Self {
        self.ephemeral_identity_id = Some(identity_id);
        self
    }

    /// Set the purpose for this allocation.
    pub fn with_purpose(mut self, purpose: impl Into<String>) -> Self {
        self.purpose = Some(purpose.into());
        self
    }

    /// Check if allocation is active (approved and not released/expired).
    pub fn is_active(&self) -> bool {
        matches!(
            self.status,
            AllocationRequestStatus::Approved | AllocationRequestStatus::AutoApproved
        ) && self.released_at.is_none()
            && Utc::now() < self.expires_at
    }

    /// Check if allocation is pending approval.
    pub fn is_pending(&self) -> bool {
        self.status == AllocationRequestStatus::Pending
    }

    /// Get remaining allocation amount.
    pub fn remaining(&self) -> Decimal {
        if self.allocated_amount > self.used_amount {
            self.allocated_amount - self.used_amount
        } else {
            Decimal::ZERO
        }
    }

    /// Record usage against this allocation.
    pub fn record_usage(&mut self, amount: Decimal) -> Result<()> {
        if amount > self.remaining() {
            return Err(HpcError::quota_exceeded(format!(
                "Usage {} exceeds remaining allocation {}",
                amount,
                self.remaining()
            )));
        }

        self.used_amount += amount;
        self.updated_at = Utc::now();
        Ok(())
    }

    /// Approve this allocation.
    pub fn approve(&mut self, approved_by: Uuid) {
        self.status = AllocationRequestStatus::Approved;
        self.approved_by = Some(approved_by);
        self.approved_at = Some(Utc::now());
        self.updated_at = Utc::now();
    }

    /// Auto-approve based on domain matching.
    pub fn auto_approve(&mut self) {
        self.status = AllocationRequestStatus::AutoApproved;
        self.approved_at = Some(Utc::now());
        self.updated_at = Utc::now();
    }

    /// Reject this allocation request.
    pub fn reject(&mut self) {
        self.status = AllocationRequestStatus::Rejected;
        self.updated_at = Utc::now();
    }

    /// Release this allocation.
    pub fn release(&mut self) {
        self.status = AllocationRequestStatus::Released;
        self.released_at = Some(Utc::now());
        self.updated_at = Utc::now();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    fn tenant_id() -> Uuid {
        Uuid::new_v4()
    }

    #[test]
    fn test_pool_type_from_str() {
        assert_eq!(PoolType::from_str("bounty").unwrap(), PoolType::Bounty);
        assert_eq!(PoolType::from_str("trial").unwrap(), PoolType::Trial);
        assert_eq!(PoolType::from_str("hackathon").unwrap(), PoolType::Hackathon);
        assert_eq!(PoolType::from_str("research").unwrap(), PoolType::Research);
        assert_eq!(PoolType::from_str("training").unwrap(), PoolType::Training);
        assert_eq!(PoolType::from_str("open_source").unwrap(), PoolType::OpenSource);
        assert!(PoolType::from_str("invalid").is_err());
    }

    #[test]
    fn test_pool_type_default_max_per_user() {
        assert_eq!(PoolType::Bounty.default_max_per_user(), dec!(100));
        assert_eq!(PoolType::Trial.default_max_per_user(), dec!(10));
        assert_eq!(PoolType::Hackathon.default_max_per_user(), dec!(50));
        assert_eq!(PoolType::Research.default_max_per_user(), dec!(500));
    }

    #[test]
    fn test_pool_type_requires_approval() {
        assert!(PoolType::Research.requires_approval_by_default());
        assert!(PoolType::Bounty.requires_approval_by_default());
        assert!(!PoolType::Trial.requires_approval_by_default());
        assert!(!PoolType::Hackathon.requires_approval_by_default());
    }

    #[test]
    fn test_pool_status_can_allocate() {
        assert!(PoolStatus::Active.can_allocate());
        assert!(!PoolStatus::Paused.can_allocate());
        assert!(!PoolStatus::Exhausted.can_allocate());
        assert!(!PoolStatus::Expired.can_allocate());
        assert!(!PoolStatus::Archived.can_allocate());
    }

    #[test]
    fn test_resource_pool_new() {
        let pool = ResourcePool::new(
            "Test Pool",
            tenant_id(),
            PoolType::Trial,
            ResourceType::GpuHours,
            dec!(1000),
        );

        assert_eq!(pool.name, "Test Pool");
        assert_eq!(pool.pool_type, PoolType::Trial);
        assert_eq!(pool.total_limit, dec!(1000));
        assert_eq!(pool.allocated, dec!(0));
        assert_eq!(pool.status, PoolStatus::Active);
        assert!(!pool.requires_approval); // Trial doesn't require approval
    }

    #[test]
    fn test_resource_pool_validate_empty_name() {
        let pool = ResourcePool::new("", tenant_id(), PoolType::Trial, ResourceType::GpuHours, dec!(1000));
        assert!(pool.validate().is_err());
    }

    #[test]
    fn test_resource_pool_validate_zero_limit() {
        let pool = ResourcePool::new("Test", tenant_id(), PoolType::Trial, ResourceType::GpuHours, dec!(0));
        assert!(pool.validate().is_err());
    }

    #[test]
    fn test_resource_pool_validate_max_exceeds_total() {
        let mut pool =
            ResourcePool::new("Test", tenant_id(), PoolType::Trial, ResourceType::GpuHours, dec!(100));
        pool.max_allocation_per_user = dec!(200);
        assert!(pool.validate().is_err());
    }

    #[test]
    fn test_resource_pool_validate_min_exceeds_max() {
        let mut pool =
            ResourcePool::new("Test", tenant_id(), PoolType::Trial, ResourceType::GpuHours, dec!(100));
        pool.min_allocation_per_request = dec!(50);
        pool.max_allocation_per_user = dec!(10);
        assert!(pool.validate().is_err());
    }

    #[test]
    fn test_resource_pool_validate_expiry_before_start() {
        let mut pool =
            ResourcePool::new("Test", tenant_id(), PoolType::Trial, ResourceType::GpuHours, dec!(100));
        pool.starts_at = Some(Utc::now() + chrono::Duration::days(1));
        pool.expires_at = Some(Utc::now());
        assert!(pool.validate().is_err());
    }

    #[test]
    fn test_resource_pool_validate_valid() {
        let pool =
            ResourcePool::new("Test Pool", tenant_id(), PoolType::Trial, ResourceType::GpuHours, dec!(1000));
        assert!(pool.validate().is_ok());
    }

    #[test]
    fn test_resource_pool_available() {
        let mut pool =
            ResourcePool::new("Test", tenant_id(), PoolType::Trial, ResourceType::GpuHours, dec!(1000));

        assert_eq!(pool.available(), dec!(1000));

        pool.allocated = dec!(400);
        pool.reserved = dec!(100);
        assert_eq!(pool.available(), dec!(500));
    }

    #[test]
    fn test_resource_pool_utilization() {
        let mut pool =
            ResourcePool::new("Test", tenant_id(), PoolType::Trial, ResourceType::GpuHours, dec!(1000));

        assert_eq!(pool.utilization_percent(), dec!(0));

        pool.allocated = dec!(250);
        assert_eq!(pool.utilization_percent(), dec!(25));
    }

    #[test]
    fn test_resource_pool_can_allocate() {
        let mut pool =
            ResourcePool::new("Test", tenant_id(), PoolType::Trial, ResourceType::GpuHours, dec!(1000));

        assert!(pool.can_allocate());

        pool.status = PoolStatus::Paused;
        assert!(!pool.can_allocate());

        pool.status = PoolStatus::Active;
        pool.starts_at = Some(Utc::now() + chrono::Duration::hours(1));
        assert!(!pool.can_allocate());

        pool.starts_at = None;
        pool.expires_at = Some(Utc::now() - chrono::Duration::hours(1));
        assert!(!pool.can_allocate());
    }

    #[test]
    fn test_resource_pool_is_auto_approve_domain() {
        let pool = ResourcePool::new("Test", tenant_id(), PoolType::Trial, ResourceType::GpuHours, dec!(1000))
            .with_auto_approve_domains(vec!["university.edu".to_string(), "research.org".to_string()]);

        assert!(pool.is_auto_approve_domain("alice@university.edu"));
        assert!(pool.is_auto_approve_domain("bob@cs.university.edu"));
        assert!(pool.is_auto_approve_domain("carol@research.org"));
        assert!(!pool.is_auto_approve_domain("dan@gmail.com"));
    }

    #[test]
    fn test_resource_pool_try_allocate_success() {
        let mut pool =
            ResourcePool::new("Test", tenant_id(), PoolType::Trial, ResourceType::GpuHours, dec!(1000));

        let allocation = pool.try_allocate(dec!(5), "user1").unwrap();

        assert_eq!(allocation.allocated_amount, dec!(5));
        assert_eq!(allocation.user_id, "user1");
        assert_eq!(pool.allocated, dec!(5));
        assert_eq!(pool.current_users, 1);
    }

    #[test]
    fn test_resource_pool_try_allocate_below_minimum() {
        let mut pool =
            ResourcePool::new("Test", tenant_id(), PoolType::Trial, ResourceType::GpuHours, dec!(1000));
        pool.min_allocation_per_request = dec!(5);

        let result = pool.try_allocate(dec!(2), "user1");
        assert!(result.is_err());
    }

    #[test]
    fn test_resource_pool_try_allocate_exceeds_max_per_user() {
        let mut pool =
            ResourcePool::new("Test", tenant_id(), PoolType::Trial, ResourceType::GpuHours, dec!(1000));
        pool.max_allocation_per_user = dec!(10);

        let result = pool.try_allocate(dec!(20), "user1");
        assert!(result.is_err());
    }

    #[test]
    fn test_resource_pool_try_allocate_exceeds_available() {
        let mut pool =
            ResourcePool::new("Test", tenant_id(), PoolType::Trial, ResourceType::GpuHours, dec!(100));
        pool.allocated = dec!(95);

        let result = pool.try_allocate(dec!(10), "user1");
        assert!(result.is_err());
    }

    #[test]
    fn test_resource_pool_try_allocate_max_users() {
        let mut pool =
            ResourcePool::new("Test", tenant_id(), PoolType::Trial, ResourceType::GpuHours, dec!(1000));
        pool.max_concurrent_users = Some(2);
        pool.current_users = 2;

        let result = pool.try_allocate(dec!(5), "user3");
        assert!(result.is_err());
    }

    #[test]
    fn test_resource_pool_try_allocate_exhausts_pool() {
        let mut pool =
            ResourcePool::new("Test", tenant_id(), PoolType::Trial, ResourceType::GpuHours, dec!(10));

        pool.try_allocate(dec!(10), "user1").unwrap();

        assert_eq!(pool.status, PoolStatus::Exhausted);
        assert!(!pool.can_allocate());
    }

    #[test]
    fn test_resource_pool_reserve_cancel() {
        let mut pool =
            ResourcePool::new("Test", tenant_id(), PoolType::Trial, ResourceType::GpuHours, dec!(100));

        pool.reserve(dec!(20)).unwrap();
        assert_eq!(pool.reserved, dec!(20));
        assert_eq!(pool.available(), dec!(80));

        pool.cancel_reservation(dec!(20));
        assert_eq!(pool.reserved, dec!(0));
        assert_eq!(pool.available(), dec!(100));
    }

    #[test]
    fn test_resource_pool_release() {
        let mut pool =
            ResourcePool::new("Test", tenant_id(), PoolType::Trial, ResourceType::GpuHours, dec!(100));

        pool.try_allocate(dec!(10), "user1").unwrap();
        assert_eq!(pool.allocated, dec!(10));
        assert_eq!(pool.current_users, 1);

        pool.release(dec!(10));
        assert_eq!(pool.allocated, dec!(0));
        assert_eq!(pool.current_users, 0);
    }

    #[test]
    fn test_resource_pool_release_reactivates() {
        let mut pool =
            ResourcePool::new("Test", tenant_id(), PoolType::Trial, ResourceType::GpuHours, dec!(10));

        pool.try_allocate(dec!(10), "user1").unwrap();
        assert_eq!(pool.status, PoolStatus::Exhausted);

        pool.release(dec!(5));
        assert_eq!(pool.status, PoolStatus::Active);
    }

    #[test]
    fn test_resource_pool_pause_resume() {
        let mut pool =
            ResourcePool::new("Test", tenant_id(), PoolType::Trial, ResourceType::GpuHours, dec!(100));

        pool.pause();
        assert_eq!(pool.status, PoolStatus::Paused);
        assert!(!pool.can_allocate());

        pool.resume();
        assert_eq!(pool.status, PoolStatus::Active);
        assert!(pool.can_allocate());
    }

    #[test]
    fn test_pool_allocation_new_pending() {
        let pool_id = Uuid::new_v4();
        let expires = Utc::now() + chrono::Duration::days(7);

        let allocation = PoolAllocation::new_pending(pool_id, "user1", dec!(50), expires);

        assert_eq!(allocation.pool_id, pool_id);
        assert_eq!(allocation.user_id, "user1");
        assert_eq!(allocation.allocated_amount, dec!(50));
        assert_eq!(allocation.status, AllocationRequestStatus::Pending);
        assert!(allocation.is_pending());
    }

    #[test]
    fn test_pool_allocation_lifecycle() {
        let expires = Utc::now() + chrono::Duration::days(7);
        let mut allocation = PoolAllocation::new_pending(Uuid::new_v4(), "user1", dec!(50), expires);

        assert!(allocation.is_pending());
        assert!(!allocation.is_active());

        allocation.approve(Uuid::new_v4());
        assert!(allocation.is_active());
        assert_eq!(allocation.remaining(), dec!(50));

        allocation.record_usage(dec!(20)).unwrap();
        assert_eq!(allocation.remaining(), dec!(30));

        allocation.release();
        assert!(!allocation.is_active());
    }

    #[test]
    fn test_pool_allocation_auto_approve() {
        let expires = Utc::now() + chrono::Duration::days(7);
        let mut allocation = PoolAllocation::new_pending(Uuid::new_v4(), "user1", dec!(50), expires);

        allocation.auto_approve();
        assert_eq!(allocation.status, AllocationRequestStatus::AutoApproved);
        assert!(allocation.is_active());
    }

    #[test]
    fn test_pool_allocation_reject() {
        let expires = Utc::now() + chrono::Duration::days(7);
        let mut allocation = PoolAllocation::new_pending(Uuid::new_v4(), "user1", dec!(50), expires);

        allocation.reject();
        assert_eq!(allocation.status, AllocationRequestStatus::Rejected);
        assert!(!allocation.is_active());
    }

    #[test]
    fn test_pool_allocation_record_usage_exceeds() {
        let expires = Utc::now() + chrono::Duration::days(7);
        let mut allocation = PoolAllocation::new_pending(Uuid::new_v4(), "user1", dec!(50), expires);
        allocation.approve(Uuid::new_v4());

        let result = allocation.record_usage(dec!(60));
        assert!(result.is_err());
    }

    #[test]
    fn test_resource_pool_serialization() {
        let pool = ResourcePool::new("Test", tenant_id(), PoolType::Hackathon, ResourceType::GpuHours, dec!(500))
            .with_description("Hackathon compute pool")
            .with_auto_approve_domains(vec!["university.edu".to_string()]);

        let json = serde_json::to_string(&pool).unwrap();
        let deserialized: ResourcePool = serde_json::from_str(&json).unwrap();

        assert_eq!(pool.name, deserialized.name);
        assert_eq!(pool.pool_type, deserialized.pool_type);
        assert_eq!(pool.auto_approve_domains, deserialized.auto_approve_domains);
    }

    #[test]
    fn test_pool_stats() {
        let mut pool =
            ResourcePool::new("Test", tenant_id(), PoolType::Trial, ResourceType::GpuHours, dec!(1000));
        pool.allocated = dec!(250);
        pool.reserved = dec!(50);
        pool.current_users = 5;

        let stats = pool.stats();
        assert_eq!(stats.total_limit, dec!(1000));
        assert_eq!(stats.allocated, dec!(250));
        assert_eq!(stats.reserved, dec!(50));
        assert_eq!(stats.available, dec!(700));
        assert_eq!(stats.utilization_percent, dec!(25));
        assert_eq!(stats.active_allocations, 5);
    }
}
