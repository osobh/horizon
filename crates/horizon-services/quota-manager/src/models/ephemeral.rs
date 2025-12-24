//! Ephemeral quota types for time-limited external access.
//!
//! Provides time-bounded quotas that link sponsors to beneficiaries,
//! with support for time windows, burst allowances, and cost tracking.

use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use sqlx::FromRow;
use utoipa::ToSchema;
use uuid::Uuid;

use super::{ResourceType, TimeWindow};
use crate::error::{HpcError, QuotaErrorExt, Result};

/// Status of an ephemeral quota.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, sqlx::Type, ToSchema)]
#[sqlx(type_name = "text", rename_all = "snake_case")]
#[serde(rename_all = "snake_case")]
pub enum EphemeralQuotaStatus {
    /// Quota is pending activation
    Pending,
    /// Quota is active and can be used
    Active,
    /// Quota is suspended (temporarily disabled)
    Suspended,
    /// Quota has expired
    Expired,
    /// Quota was revoked
    Revoked,
    /// Quota was exhausted
    Exhausted,
}

impl EphemeralQuotaStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            EphemeralQuotaStatus::Pending => "pending",
            EphemeralQuotaStatus::Active => "active",
            EphemeralQuotaStatus::Suspended => "suspended",
            EphemeralQuotaStatus::Expired => "expired",
            EphemeralQuotaStatus::Revoked => "revoked",
            EphemeralQuotaStatus::Exhausted => "exhausted",
        }
    }

    pub fn is_usable(&self) -> bool {
        matches!(self, EphemeralQuotaStatus::Active)
    }

    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            EphemeralQuotaStatus::Expired
                | EphemeralQuotaStatus::Revoked
                | EphemeralQuotaStatus::Exhausted
        )
    }
}

/// An ephemeral quota providing time-limited resource access.
#[derive(Debug, Clone, FromRow, Serialize, Deserialize, ToSchema)]
pub struct EphemeralQuota {
    /// Unique identifier for this ephemeral quota
    pub id: Uuid,
    /// Links to the base quota definition (optional)
    pub quota_id: Option<Uuid>,
    /// The ephemeral identity this quota belongs to
    pub ephemeral_identity_id: Uuid,
    /// Organization/tenant context
    pub tenant_id: Uuid,
    /// Who is sponsoring/paying for this quota
    pub sponsor_id: String,
    /// Who is using this quota (ephemeral user)
    pub beneficiary_id: String,
    /// Type of resource being allocated
    pub resource_type: ResourceType,
    /// Total limit for this ephemeral period
    pub limit_value: Decimal,
    /// Amount currently used
    pub used_value: Decimal,
    /// Amount reserved for pending operations
    pub reserved_value: Decimal,
    /// When this quota becomes active
    pub starts_at: DateTime<Utc>,
    /// When this quota expires
    pub expires_at: DateTime<Utc>,
    /// Time window ID for scheduling constraints (optional)
    pub time_window_id: Option<Uuid>,
    /// Whether burst is enabled beyond the limit
    pub burst_enabled: bool,
    /// Multiplier for burst capacity (e.g., 1.5x)
    pub burst_multiplier: Decimal,
    /// Resource pool this quota draws from (optional)
    pub pool_id: Option<Uuid>,
    /// Total cost incurred by this quota
    pub actual_cost: Decimal,
    /// Cost rate per unit
    pub cost_rate: Decimal,
    /// Current status
    pub status: EphemeralQuotaStatus,
    /// Reason for current status (especially for revocation/suspension)
    pub status_reason: Option<String>,
    /// When this quota was created
    pub created_at: DateTime<Utc>,
    /// Last modification time
    pub updated_at: DateTime<Utc>,
}

/// Request to create an ephemeral quota.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct CreateEphemeralQuotaRequest {
    pub ephemeral_identity_id: Uuid,
    pub tenant_id: Uuid,
    pub sponsor_id: String,
    pub beneficiary_id: String,
    pub resource_type: ResourceType,
    pub limit_value: Decimal,
    pub starts_at: Option<DateTime<Utc>>,
    pub expires_at: DateTime<Utc>,
    pub time_window_id: Option<Uuid>,
    pub burst_enabled: Option<bool>,
    pub burst_multiplier: Option<Decimal>,
    pub pool_id: Option<Uuid>,
    pub cost_rate: Option<Decimal>,
}

/// Request to update an ephemeral quota.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct UpdateEphemeralQuotaRequest {
    pub limit_value: Option<Decimal>,
    pub expires_at: Option<DateTime<Utc>>,
    pub time_window_id: Option<Uuid>,
    pub burst_enabled: Option<bool>,
    pub burst_multiplier: Option<Decimal>,
    pub status: Option<EphemeralQuotaStatus>,
    pub status_reason: Option<String>,
}

/// Result of checking if an ephemeral quota allows an operation.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct EphemeralQuotaCheckResult {
    /// Whether the operation is allowed
    pub allowed: bool,
    /// Amount available for use
    pub available: Decimal,
    /// Amount requested
    pub requested: Decimal,
    /// Whether burst was used
    pub using_burst: bool,
    /// If not allowed, the reason
    pub denial_reason: Option<String>,
    /// If using time window, whether currently in window
    pub in_time_window: Option<bool>,
}

/// Usage event for an ephemeral quota.
#[derive(Debug, Clone, FromRow, Serialize, Deserialize, ToSchema)]
pub struct EphemeralQuotaUsage {
    pub id: Uuid,
    pub ephemeral_quota_id: Uuid,
    pub operation_type: EphemeralOperationType,
    pub amount: Decimal,
    pub cost: Decimal,
    pub job_id: Option<Uuid>,
    pub description: Option<String>,
    pub timestamp: DateTime<Utc>,
}

/// Type of usage operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, sqlx::Type, ToSchema)]
#[sqlx(type_name = "text", rename_all = "snake_case")]
#[serde(rename_all = "snake_case")]
pub enum EphemeralOperationType {
    /// Resource was allocated/used
    Use,
    /// Resource was released
    Release,
    /// Reservation was made
    Reserve,
    /// Reservation was cancelled
    CancelReserve,
    /// Burst capacity was used
    Burst,
    /// Manual adjustment
    Adjustment,
}

impl EphemeralQuota {
    /// Create a new ephemeral quota.
    pub fn new(
        ephemeral_identity_id: Uuid,
        tenant_id: Uuid,
        sponsor_id: impl Into<String>,
        beneficiary_id: impl Into<String>,
        resource_type: ResourceType,
        limit_value: Decimal,
        expires_at: DateTime<Utc>,
    ) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            quota_id: None,
            ephemeral_identity_id,
            tenant_id,
            sponsor_id: sponsor_id.into(),
            beneficiary_id: beneficiary_id.into(),
            resource_type,
            limit_value,
            used_value: Decimal::ZERO,
            reserved_value: Decimal::ZERO,
            starts_at: now,
            expires_at,
            time_window_id: None,
            burst_enabled: false,
            burst_multiplier: Decimal::ONE,
            pool_id: None,
            actual_cost: Decimal::ZERO,
            cost_rate: Decimal::ZERO,
            status: EphemeralQuotaStatus::Pending,
            status_reason: None,
            created_at: now,
            updated_at: now,
        }
    }

    /// Link to a base quota definition.
    pub fn with_quota_id(mut self, quota_id: Uuid) -> Self {
        self.quota_id = Some(quota_id);
        self
    }

    /// Set a delayed start time.
    pub fn with_start(mut self, starts_at: DateTime<Utc>) -> Self {
        self.starts_at = starts_at;
        self
    }

    /// Enable burst with a multiplier.
    pub fn with_burst(mut self, multiplier: Decimal) -> Self {
        self.burst_enabled = true;
        self.burst_multiplier = multiplier;
        self
    }

    /// Link to a resource pool.
    pub fn with_pool(mut self, pool_id: Uuid) -> Self {
        self.pool_id = Some(pool_id);
        self
    }

    /// Link to a time window.
    pub fn with_time_window(mut self, time_window_id: Uuid) -> Self {
        self.time_window_id = Some(time_window_id);
        self
    }

    /// Set cost rate.
    pub fn with_cost_rate(mut self, rate: Decimal) -> Self {
        self.cost_rate = rate;
        self
    }

    /// Validate this ephemeral quota.
    pub fn validate(&self) -> Result<()> {
        if self.limit_value <= Decimal::ZERO {
            return Err(HpcError::invalid_configuration("Limit must be positive"));
        }

        if self.expires_at <= self.starts_at {
            return Err(HpcError::invalid_configuration(
                "Expiry must be after start time",
            ));
        }

        if self.burst_enabled && self.burst_multiplier <= Decimal::ONE {
            return Err(HpcError::invalid_configuration(
                "Burst multiplier must be greater than 1.0",
            ));
        }

        if self.cost_rate < Decimal::ZERO {
            return Err(HpcError::invalid_configuration(
                "Cost rate cannot be negative",
            ));
        }

        Ok(())
    }

    /// Calculate the effective limit including burst.
    pub fn effective_limit(&self) -> Decimal {
        if self.burst_enabled {
            self.limit_value * self.burst_multiplier
        } else {
            self.limit_value
        }
    }

    /// Get the available amount (not used or reserved).
    pub fn available(&self) -> Decimal {
        let used_and_reserved = self.used_value + self.reserved_value;
        let limit = self.effective_limit();
        if used_and_reserved >= limit {
            Decimal::ZERO
        } else {
            limit - used_and_reserved
        }
    }

    /// Calculate utilization percentage (0-100).
    pub fn utilization_percent(&self) -> Decimal {
        if self.limit_value == Decimal::ZERO {
            Decimal::ZERO
        } else {
            (self.used_value / self.limit_value) * Decimal::from(100)
        }
    }

    /// Check if the quota is currently usable.
    pub fn is_usable(&self) -> bool {
        if !self.status.is_usable() {
            return false;
        }

        let now = Utc::now();
        now >= self.starts_at && now < self.expires_at
    }

    /// Check if quota has started.
    pub fn has_started(&self) -> bool {
        Utc::now() >= self.starts_at
    }

    /// Check if quota has expired.
    pub fn is_expired(&self) -> bool {
        Utc::now() >= self.expires_at
    }

    /// Check if usage is within normal limits (not using burst).
    pub fn is_within_normal_limit(&self) -> bool {
        self.used_value + self.reserved_value <= self.limit_value
    }

    /// Check if using burst capacity.
    pub fn is_using_burst(&self) -> bool {
        self.burst_enabled && !self.is_within_normal_limit()
    }

    /// Time remaining until expiry.
    pub fn time_remaining(&self) -> Option<chrono::Duration> {
        let now = Utc::now();
        if now >= self.expires_at {
            None
        } else {
            Some(self.expires_at.signed_duration_since(now))
        }
    }

    /// Calculate cost for an amount.
    fn calculate_cost(&self, amount: Decimal) -> Decimal {
        amount * self.cost_rate
    }

    /// Check if a usage operation is allowed.
    pub fn check_usage(
        &self,
        amount: Decimal,
        time_window: Option<&TimeWindow>,
    ) -> EphemeralQuotaCheckResult {
        // Check status
        if !self.is_usable() {
            return EphemeralQuotaCheckResult {
                allowed: false,
                available: Decimal::ZERO,
                requested: amount,
                using_burst: false,
                denial_reason: Some(format!("Quota is not usable (status: {})", self.status.as_str())),
                in_time_window: None,
            };
        }

        // Check time window if configured
        let in_time_window = if let Some(window) = time_window {
            let check = window.is_allowed(Utc::now());
            if !check.allowed {
                return EphemeralQuotaCheckResult {
                    allowed: false,
                    available: self.available(),
                    requested: amount,
                    using_burst: false,
                    denial_reason: Some(check.reason),
                    in_time_window: Some(false),
                };
            }
            Some(true)
        } else {
            None
        };

        // Check available capacity
        let available = self.available();
        if amount > available {
            return EphemeralQuotaCheckResult {
                allowed: false,
                available,
                requested: amount,
                using_burst: false,
                denial_reason: Some(format!(
                    "Requested {} exceeds available {}",
                    amount, available
                )),
                in_time_window,
            };
        }

        // Check if this would use burst
        let would_use_burst = self.used_value + self.reserved_value + amount > self.limit_value;

        EphemeralQuotaCheckResult {
            allowed: true,
            available,
            requested: amount,
            using_burst: would_use_burst && self.burst_enabled,
            denial_reason: None,
            in_time_window,
        }
    }

    /// Record usage against this quota.
    pub fn record_usage(&mut self, amount: Decimal) -> Result<EphemeralQuotaUsage> {
        if !self.is_usable() {
            return Err(HpcError::quota_exceeded(format!(
                "Quota is not usable (status: {})",
                self.status.as_str()
            )));
        }

        if amount > self.available() {
            return Err(HpcError::quota_exceeded(format!(
                "Requested {} exceeds available {}",
                amount,
                self.available()
            )));
        }

        let cost = self.calculate_cost(amount);
        self.used_value += amount;
        self.actual_cost += cost;
        self.updated_at = Utc::now();

        // Check if exhausted
        if self.available() == Decimal::ZERO {
            self.status = EphemeralQuotaStatus::Exhausted;
            self.status_reason = Some("Quota exhausted".to_string());
        }

        let operation_type = if self.is_using_burst() {
            EphemeralOperationType::Burst
        } else {
            EphemeralOperationType::Use
        };

        Ok(EphemeralQuotaUsage {
            id: Uuid::new_v4(),
            ephemeral_quota_id: self.id,
            operation_type,
            amount,
            cost,
            job_id: None,
            description: None,
            timestamp: Utc::now(),
        })
    }

    /// Release usage back to the quota.
    pub fn release_usage(&mut self, amount: Decimal) -> Result<EphemeralQuotaUsage> {
        if amount > self.used_value {
            return Err(HpcError::invalid_input(
                "amount",
                format!("Cannot release {} - only {} used", amount, self.used_value),
            ));
        }

        let cost = self.calculate_cost(amount);
        self.used_value -= amount;
        self.actual_cost = (self.actual_cost - cost).max(Decimal::ZERO);
        self.updated_at = Utc::now();

        // Reactivate if was exhausted
        if self.status == EphemeralQuotaStatus::Exhausted && self.available() > Decimal::ZERO {
            self.status = EphemeralQuotaStatus::Active;
            self.status_reason = None;
        }

        Ok(EphemeralQuotaUsage {
            id: Uuid::new_v4(),
            ephemeral_quota_id: self.id,
            operation_type: EphemeralOperationType::Release,
            amount,
            cost,
            job_id: None,
            description: None,
            timestamp: Utc::now(),
        })
    }

    /// Reserve an amount for a pending operation.
    pub fn reserve(&mut self, amount: Decimal) -> Result<()> {
        if amount > self.available() {
            return Err(HpcError::quota_exceeded(format!(
                "Cannot reserve {} - only {} available",
                amount,
                self.available()
            )));
        }

        self.reserved_value += amount;
        self.updated_at = Utc::now();
        Ok(())
    }

    /// Cancel a reservation.
    pub fn cancel_reservation(&mut self, amount: Decimal) {
        self.reserved_value = (self.reserved_value - amount).max(Decimal::ZERO);
        self.updated_at = Utc::now();
    }

    /// Commit a reservation (convert to usage).
    pub fn commit_reservation(&mut self, amount: Decimal) -> Result<EphemeralQuotaUsage> {
        if amount > self.reserved_value {
            return Err(HpcError::invalid_input(
                "amount",
                format!(
                    "Cannot commit {} - only {} reserved",
                    amount, self.reserved_value
                ),
            ));
        }

        self.reserved_value -= amount;

        let cost = self.calculate_cost(amount);
        self.used_value += amount;
        self.actual_cost += cost;
        self.updated_at = Utc::now();

        Ok(EphemeralQuotaUsage {
            id: Uuid::new_v4(),
            ephemeral_quota_id: self.id,
            operation_type: EphemeralOperationType::Use,
            amount,
            cost,
            job_id: None,
            description: None,
            timestamp: Utc::now(),
        })
    }

    /// Activate this quota.
    pub fn activate(&mut self) -> Result<()> {
        if self.status != EphemeralQuotaStatus::Pending {
            return Err(HpcError::invalid_input(
                "status",
                format!("Cannot activate quota in {} status", self.status.as_str()),
            ));
        }

        self.status = EphemeralQuotaStatus::Active;
        self.status_reason = None;
        self.updated_at = Utc::now();
        Ok(())
    }

    /// Suspend this quota.
    pub fn suspend(&mut self, reason: impl Into<String>) {
        self.status = EphemeralQuotaStatus::Suspended;
        self.status_reason = Some(reason.into());
        self.updated_at = Utc::now();
    }

    /// Revoke this quota.
    pub fn revoke(&mut self, reason: impl Into<String>) {
        self.status = EphemeralQuotaStatus::Revoked;
        self.status_reason = Some(reason.into());
        self.updated_at = Utc::now();
    }

    /// Mark as expired.
    pub fn mark_expired(&mut self) {
        self.status = EphemeralQuotaStatus::Expired;
        self.status_reason = Some("Quota has expired".to_string());
        self.updated_at = Utc::now();
    }

    /// Check expiry and update status if needed.
    pub fn check_expiry(&mut self) -> bool {
        if self.is_expired() && !self.status.is_terminal() {
            self.mark_expired();
            true
        } else {
            false
        }
    }

    /// Extend the expiry time.
    pub fn extend_expiry(&mut self, new_expiry: DateTime<Utc>) -> Result<()> {
        if new_expiry <= self.expires_at {
            return Err(HpcError::invalid_input(
                "expires_at",
                "New expiry must be after current expiry",
            ));
        }

        if self.status.is_terminal() {
            return Err(HpcError::invalid_input(
                "status",
                format!("Cannot extend quota in {} status", self.status.as_str()),
            ));
        }

        self.expires_at = new_expiry;

        // Reactivate if was expired
        if self.status == EphemeralQuotaStatus::Expired {
            self.status = EphemeralQuotaStatus::Active;
            self.status_reason = None;
        }

        self.updated_at = Utc::now();
        Ok(())
    }

    /// Increase the limit.
    pub fn increase_limit(&mut self, additional: Decimal) {
        self.limit_value += additional;
        self.updated_at = Utc::now();

        // Reactivate if was exhausted
        if self.status == EphemeralQuotaStatus::Exhausted && self.available() > Decimal::ZERO {
            self.status = EphemeralQuotaStatus::Active;
            self.status_reason = None;
        }
    }
}

/// Summary of ephemeral quota usage for a sponsor.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct SponsorUsageSummary {
    pub sponsor_id: String,
    pub total_quotas: i32,
    pub active_quotas: i32,
    pub total_allocated: Decimal,
    pub total_used: Decimal,
    pub total_cost: Decimal,
    pub by_resource_type: Vec<ResourceTypeSummary>,
}

/// Usage summary per resource type.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ResourceTypeSummary {
    pub resource_type: ResourceType,
    pub allocated: Decimal,
    pub used: Decimal,
    pub cost: Decimal,
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    fn test_quota() -> EphemeralQuota {
        EphemeralQuota::new(
            Uuid::new_v4(),
            Uuid::new_v4(),
            "sponsor1",
            "user1",
            ResourceType::GpuHours,
            dec!(100),
            Utc::now() + chrono::Duration::days(7),
        )
    }

    #[test]
    fn test_ephemeral_quota_status_usable() {
        assert!(EphemeralQuotaStatus::Active.is_usable());
        assert!(!EphemeralQuotaStatus::Pending.is_usable());
        assert!(!EphemeralQuotaStatus::Suspended.is_usable());
        assert!(!EphemeralQuotaStatus::Expired.is_usable());
        assert!(!EphemeralQuotaStatus::Revoked.is_usable());
        assert!(!EphemeralQuotaStatus::Exhausted.is_usable());
    }

    #[test]
    fn test_ephemeral_quota_status_terminal() {
        assert!(!EphemeralQuotaStatus::Active.is_terminal());
        assert!(!EphemeralQuotaStatus::Pending.is_terminal());
        assert!(!EphemeralQuotaStatus::Suspended.is_terminal());
        assert!(EphemeralQuotaStatus::Expired.is_terminal());
        assert!(EphemeralQuotaStatus::Revoked.is_terminal());
        assert!(EphemeralQuotaStatus::Exhausted.is_terminal());
    }

    #[test]
    fn test_ephemeral_quota_new() {
        let quota = test_quota();

        assert_eq!(quota.sponsor_id, "sponsor1");
        assert_eq!(quota.beneficiary_id, "user1");
        assert_eq!(quota.limit_value, dec!(100));
        assert_eq!(quota.used_value, dec!(0));
        assert_eq!(quota.status, EphemeralQuotaStatus::Pending);
    }

    #[test]
    fn test_ephemeral_quota_validate_zero_limit() {
        let mut quota = test_quota();
        quota.limit_value = dec!(0);
        assert!(quota.validate().is_err());
    }

    #[test]
    fn test_ephemeral_quota_validate_expiry_before_start() {
        let mut quota = test_quota();
        quota.expires_at = quota.starts_at - chrono::Duration::hours(1);
        assert!(quota.validate().is_err());
    }

    #[test]
    fn test_ephemeral_quota_validate_burst_multiplier() {
        let mut quota = test_quota();
        quota.burst_enabled = true;
        quota.burst_multiplier = dec!(0.5);
        assert!(quota.validate().is_err());
    }

    #[test]
    fn test_ephemeral_quota_validate_negative_cost_rate() {
        let mut quota = test_quota();
        quota.cost_rate = dec!(-1);
        assert!(quota.validate().is_err());
    }

    #[test]
    fn test_ephemeral_quota_validate_valid() {
        let quota = test_quota();
        assert!(quota.validate().is_ok());
    }

    #[test]
    fn test_ephemeral_quota_effective_limit_no_burst() {
        let quota = test_quota();
        assert_eq!(quota.effective_limit(), dec!(100));
    }

    #[test]
    fn test_ephemeral_quota_effective_limit_with_burst() {
        let quota = test_quota().with_burst(dec!(1.5));
        assert_eq!(quota.effective_limit(), dec!(150));
    }

    #[test]
    fn test_ephemeral_quota_available() {
        let mut quota = test_quota();
        assert_eq!(quota.available(), dec!(100));

        quota.used_value = dec!(40);
        quota.reserved_value = dec!(10);
        assert_eq!(quota.available(), dec!(50));
    }

    #[test]
    fn test_ephemeral_quota_utilization() {
        let mut quota = test_quota();
        assert_eq!(quota.utilization_percent(), dec!(0));

        quota.used_value = dec!(25);
        assert_eq!(quota.utilization_percent(), dec!(25));
    }

    #[test]
    fn test_ephemeral_quota_is_usable() {
        let mut quota = test_quota();
        quota.status = EphemeralQuotaStatus::Active;
        assert!(quota.is_usable());

        quota.status = EphemeralQuotaStatus::Pending;
        assert!(!quota.is_usable());
    }

    #[test]
    fn test_ephemeral_quota_check_usage_not_usable() {
        let quota = test_quota(); // Status is Pending

        let result = quota.check_usage(dec!(10), None);
        assert!(!result.allowed);
        assert!(result.denial_reason.is_some());
    }

    #[test]
    fn test_ephemeral_quota_check_usage_success() {
        let mut quota = test_quota();
        quota.status = EphemeralQuotaStatus::Active;

        let result = quota.check_usage(dec!(10), None);
        assert!(result.allowed);
        assert!(!result.using_burst);
        assert!(result.denial_reason.is_none());
    }

    #[test]
    fn test_ephemeral_quota_check_usage_exceeds_available() {
        let mut quota = test_quota();
        quota.status = EphemeralQuotaStatus::Active;
        quota.used_value = dec!(95);

        let result = quota.check_usage(dec!(10), None);
        assert!(!result.allowed);
        assert!(result.denial_reason.is_some());
    }

    #[test]
    fn test_ephemeral_quota_check_usage_with_burst() {
        let mut quota = test_quota().with_burst(dec!(1.5));
        quota.status = EphemeralQuotaStatus::Active;
        quota.used_value = dec!(90);

        let result = quota.check_usage(dec!(20), None);
        assert!(result.allowed);
        assert!(result.using_burst);
    }

    #[test]
    fn test_ephemeral_quota_record_usage() {
        let mut quota = test_quota().with_cost_rate(dec!(0.10));
        quota.status = EphemeralQuotaStatus::Active;

        let usage = quota.record_usage(dec!(25)).unwrap();

        assert_eq!(usage.amount, dec!(25));
        assert_eq!(usage.cost, dec!(2.5));
        assert_eq!(quota.used_value, dec!(25));
        assert_eq!(quota.actual_cost, dec!(2.5));
    }

    #[test]
    fn test_ephemeral_quota_record_usage_exhausts() {
        let mut quota = test_quota();
        quota.status = EphemeralQuotaStatus::Active;

        quota.record_usage(dec!(100)).unwrap();

        assert_eq!(quota.status, EphemeralQuotaStatus::Exhausted);
    }

    #[test]
    fn test_ephemeral_quota_release_usage() {
        let mut quota = test_quota().with_cost_rate(dec!(0.10));
        quota.status = EphemeralQuotaStatus::Active;

        quota.record_usage(dec!(50)).unwrap();
        let release = quota.release_usage(dec!(20)).unwrap();

        assert_eq!(release.amount, dec!(20));
        assert_eq!(quota.used_value, dec!(30));
        assert_eq!(quota.actual_cost, dec!(3));
    }

    #[test]
    fn test_ephemeral_quota_release_reactivates() {
        let mut quota = test_quota();
        quota.status = EphemeralQuotaStatus::Exhausted;
        quota.used_value = dec!(100);

        quota.release_usage(dec!(10)).unwrap();

        assert_eq!(quota.status, EphemeralQuotaStatus::Active);
    }

    #[test]
    fn test_ephemeral_quota_reserve_commit() {
        let mut quota = test_quota().with_cost_rate(dec!(0.10));
        quota.status = EphemeralQuotaStatus::Active;

        quota.reserve(dec!(30)).unwrap();
        assert_eq!(quota.reserved_value, dec!(30));
        assert_eq!(quota.available(), dec!(70));

        let usage = quota.commit_reservation(dec!(30)).unwrap();
        assert_eq!(usage.amount, dec!(30));
        assert_eq!(quota.reserved_value, dec!(0));
        assert_eq!(quota.used_value, dec!(30));
    }

    #[test]
    fn test_ephemeral_quota_cancel_reservation() {
        let mut quota = test_quota();
        quota.status = EphemeralQuotaStatus::Active;

        quota.reserve(dec!(30)).unwrap();
        quota.cancel_reservation(dec!(30));

        assert_eq!(quota.reserved_value, dec!(0));
        assert_eq!(quota.available(), dec!(100));
    }

    #[test]
    fn test_ephemeral_quota_activate() {
        let mut quota = test_quota();
        assert_eq!(quota.status, EphemeralQuotaStatus::Pending);

        quota.activate().unwrap();
        assert_eq!(quota.status, EphemeralQuotaStatus::Active);
    }

    #[test]
    fn test_ephemeral_quota_activate_wrong_status() {
        let mut quota = test_quota();
        quota.status = EphemeralQuotaStatus::Active;

        assert!(quota.activate().is_err());
    }

    #[test]
    fn test_ephemeral_quota_suspend_revoke() {
        let mut quota = test_quota();
        quota.status = EphemeralQuotaStatus::Active;

        quota.suspend("Policy violation");
        assert_eq!(quota.status, EphemeralQuotaStatus::Suspended);
        assert_eq!(quota.status_reason, Some("Policy violation".to_string()));

        quota.status = EphemeralQuotaStatus::Active;
        quota.revoke("Sponsor requested");
        assert_eq!(quota.status, EphemeralQuotaStatus::Revoked);
    }

    #[test]
    fn test_ephemeral_quota_check_expiry() {
        let mut quota = EphemeralQuota::new(
            Uuid::new_v4(),
            Uuid::new_v4(),
            "sponsor1",
            "user1",
            ResourceType::GpuHours,
            dec!(100),
            Utc::now() - chrono::Duration::hours(1), // Already expired
        );
        quota.status = EphemeralQuotaStatus::Active;

        let expired = quota.check_expiry();
        assert!(expired);
        assert_eq!(quota.status, EphemeralQuotaStatus::Expired);
    }

    #[test]
    fn test_ephemeral_quota_extend_expiry() {
        let mut quota = test_quota();
        quota.status = EphemeralQuotaStatus::Active;
        let original_expiry = quota.expires_at;

        let new_expiry = original_expiry + chrono::Duration::days(7);
        quota.extend_expiry(new_expiry).unwrap();

        assert_eq!(quota.expires_at, new_expiry);
    }

    #[test]
    fn test_ephemeral_quota_extend_expiry_invalid() {
        let mut quota = test_quota();
        quota.status = EphemeralQuotaStatus::Active;
        let original_expiry = quota.expires_at;

        // Try to set earlier expiry
        let earlier = original_expiry - chrono::Duration::days(1);
        assert!(quota.extend_expiry(earlier).is_err());
    }

    #[test]
    fn test_ephemeral_quota_increase_limit() {
        let mut quota = test_quota();
        quota.status = EphemeralQuotaStatus::Exhausted;
        quota.used_value = dec!(100);

        quota.increase_limit(dec!(50));

        assert_eq!(quota.limit_value, dec!(150));
        assert_eq!(quota.status, EphemeralQuotaStatus::Active);
    }

    #[test]
    fn test_ephemeral_quota_serialization() {
        let quota = test_quota()
            .with_burst(dec!(1.5))
            .with_cost_rate(dec!(0.10));

        let json = serde_json::to_string(&quota).unwrap();
        let deserialized: EphemeralQuota = serde_json::from_str(&json).unwrap();

        assert_eq!(quota.sponsor_id, deserialized.sponsor_id);
        assert_eq!(quota.limit_value, deserialized.limit_value);
        assert_eq!(quota.burst_enabled, deserialized.burst_enabled);
        assert_eq!(quota.burst_multiplier, deserialized.burst_multiplier);
    }

    #[test]
    fn test_ephemeral_quota_time_remaining() {
        let quota = test_quota();
        let remaining = quota.time_remaining();
        assert!(remaining.is_some());
        assert!(remaining.unwrap().num_days() >= 6);

        // Expired quota
        let expired_quota = EphemeralQuota::new(
            Uuid::new_v4(),
            Uuid::new_v4(),
            "sponsor1",
            "user1",
            ResourceType::GpuHours,
            dec!(100),
            Utc::now() - chrono::Duration::hours(1),
        );
        assert!(expired_quota.time_remaining().is_none());
    }
}
