//! Ephemeral quota service with burst logic and time window support.
//!
//! Provides:
//! - Time-bounded quota management for ephemeral users
//! - Burst capacity with multipliers
//! - Sponsor cost attribution
//! - Automatic expiry handling
//! - Integration with resource pools

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
        EphemeralOperationType, EphemeralQuota, EphemeralQuotaCheckResult,
        EphemeralQuotaStatus, EphemeralQuotaUsage, CreateEphemeralQuotaRequest,
        UpdateEphemeralQuotaRequest, ResourceType, TimeWindow, SponsorUsageSummary,
        ResourceTypeSummary,
    },
};

/// Configuration for the ephemeral quota service.
#[derive(Debug, Clone)]
pub struct EphemeralQuotaServiceConfig {
    /// Maximum quotas per sponsor (default: 100)
    pub max_quotas_per_sponsor: usize,
    /// Maximum burst multiplier allowed (default: 2.0)
    pub max_burst_multiplier: Decimal,
    /// Cleanup interval for expired quotas (seconds)
    pub cleanup_interval_secs: u64,
    /// Enable automatic expiry processing
    pub auto_expiry_enabled: bool,
    /// Grace period before fully expiring (seconds)
    pub expiry_grace_period_secs: i64,
}

impl Default for EphemeralQuotaServiceConfig {
    fn default() -> Self {
        Self {
            max_quotas_per_sponsor: 100,
            max_burst_multiplier: Decimal::from(2),
            cleanup_interval_secs: 60,
            auto_expiry_enabled: true,
            expiry_grace_period_secs: 300, // 5 minutes
        }
    }
}

/// Service for managing ephemeral quotas.
#[derive(Clone)]
pub struct EphemeralQuotaService {
    repository: QuotaRepository,
    config: EphemeralQuotaServiceConfig,
    // In-memory cache for active quotas (for fast lookup)
    cache: Arc<RwLock<HashMap<Uuid, EphemeralQuota>>>,
    // Time window cache
    time_windows: Arc<RwLock<HashMap<Uuid, TimeWindow>>>,
}

impl EphemeralQuotaService {
    /// Create a new ephemeral quota service.
    pub fn new(repository: QuotaRepository) -> Self {
        Self::with_config(repository, EphemeralQuotaServiceConfig::default())
    }

    /// Create with custom configuration.
    pub fn with_config(repository: QuotaRepository, config: EphemeralQuotaServiceConfig) -> Self {
        Self {
            repository,
            config,
            cache: Arc::new(RwLock::new(HashMap::new())),
            time_windows: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create a new ephemeral quota.
    pub async fn create_quota(&self, req: CreateEphemeralQuotaRequest) -> Result<EphemeralQuota> {
        // Validate request
        self.validate_create_request(&req).await?;

        // Create the quota
        let mut quota = EphemeralQuota::new(
            req.ephemeral_identity_id,
            req.tenant_id,
            &req.sponsor_id,
            &req.beneficiary_id,
            req.resource_type,
            req.limit_value,
            req.expires_at,
        );

        // Apply optional settings
        if let Some(starts_at) = req.starts_at {
            quota = quota.with_start(starts_at);
        }

        if let Some(burst_multiplier) = req.burst_multiplier {
            if req.burst_enabled.unwrap_or(false) {
                self.validate_burst_multiplier(burst_multiplier)?;
                quota = quota.with_burst(burst_multiplier);
            }
        }

        if let Some(pool_id) = req.pool_id {
            quota = quota.with_pool(pool_id);
        }

        if let Some(time_window_id) = req.time_window_id {
            quota = quota.with_time_window(time_window_id);
        }

        if let Some(cost_rate) = req.cost_rate {
            quota = quota.with_cost_rate(cost_rate);
        }

        // Validate the quota
        quota.validate()?;

        // Cache the quota
        {
            let mut cache = self.cache.write().await;
            cache.insert(quota.id, quota.clone());
        }

        Ok(quota)
    }

    /// Validate create request.
    async fn validate_create_request(&self, req: &CreateEphemeralQuotaRequest) -> Result<()> {
        // Check sponsor limits
        let sponsor_quotas = self.get_quotas_by_sponsor(&req.sponsor_id).await?;
        if sponsor_quotas.len() >= self.config.max_quotas_per_sponsor {
            return Err(HpcError::quota_exceeded(format!(
                "Sponsor {} has reached maximum quota limit of {}",
                req.sponsor_id, self.config.max_quotas_per_sponsor
            )));
        }

        // Validate expiry is in the future
        if req.expires_at <= Utc::now() {
            return Err(HpcError::invalid_input(
                "expires_at",
                "Expiry must be in the future",
            ));
        }

        // Validate limit is positive
        if req.limit_value <= Decimal::ZERO {
            return Err(HpcError::invalid_input(
                "limit_value",
                "Limit must be positive",
            ));
        }

        Ok(())
    }

    /// Validate burst multiplier.
    fn validate_burst_multiplier(&self, multiplier: Decimal) -> Result<()> {
        if multiplier <= Decimal::ONE {
            return Err(HpcError::invalid_input(
                "burst_multiplier",
                "Burst multiplier must be greater than 1.0",
            ));
        }

        if multiplier > self.config.max_burst_multiplier {
            return Err(HpcError::invalid_input(
                "burst_multiplier",
                format!(
                    "Burst multiplier {} exceeds maximum {}",
                    multiplier, self.config.max_burst_multiplier
                ),
            ));
        }

        Ok(())
    }

    /// Get a quota by ID.
    pub async fn get_quota(&self, id: Uuid) -> Result<EphemeralQuota> {
        // Check cache first
        {
            let cache = self.cache.read().await;
            if let Some(quota) = cache.get(&id) {
                return Ok(quota.clone());
            }
        }

        Err(HpcError::not_found("ephemeral_quota", id.to_string()))
    }

    /// Get quotas by ephemeral identity.
    pub async fn get_quotas_by_identity(&self, identity_id: Uuid) -> Result<Vec<EphemeralQuota>> {
        let cache = self.cache.read().await;
        Ok(cache
            .values()
            .filter(|q| q.ephemeral_identity_id == identity_id)
            .cloned()
            .collect())
    }

    /// Get quotas by sponsor.
    pub async fn get_quotas_by_sponsor(&self, sponsor_id: &str) -> Result<Vec<EphemeralQuota>> {
        let cache = self.cache.read().await;
        Ok(cache
            .values()
            .filter(|q| q.sponsor_id == sponsor_id)
            .cloned()
            .collect())
    }

    /// Get quotas by beneficiary.
    pub async fn get_quotas_by_beneficiary(&self, beneficiary_id: &str) -> Result<Vec<EphemeralQuota>> {
        let cache = self.cache.read().await;
        Ok(cache
            .values()
            .filter(|q| q.beneficiary_id == beneficiary_id)
            .cloned()
            .collect())
    }

    /// Get active quotas for a beneficiary.
    pub async fn get_active_quotas(&self, beneficiary_id: &str) -> Result<Vec<EphemeralQuota>> {
        let cache = self.cache.read().await;
        Ok(cache
            .values()
            .filter(|q| q.beneficiary_id == beneficiary_id && q.is_usable())
            .cloned()
            .collect())
    }

    /// Update a quota.
    pub async fn update_quota(
        &self,
        id: Uuid,
        req: UpdateEphemeralQuotaRequest,
    ) -> Result<EphemeralQuota> {
        let mut quota = self.get_quota(id).await?;

        // Apply updates
        if let Some(limit) = req.limit_value {
            if limit < quota.used_value {
                return Err(HpcError::invalid_input(
                    "limit_value",
                    format!(
                        "New limit {} is below current usage {}",
                        limit, quota.used_value
                    ),
                ));
            }
            quota.limit_value = limit;
        }

        if let Some(expires_at) = req.expires_at {
            if expires_at <= Utc::now() {
                return Err(HpcError::invalid_input(
                    "expires_at",
                    "New expiry must be in the future",
                ));
            }
            quota.expires_at = expires_at;
        }

        if let Some(time_window_id) = req.time_window_id {
            quota.time_window_id = Some(time_window_id);
        }

        if let Some(burst_enabled) = req.burst_enabled {
            quota.burst_enabled = burst_enabled;
        }

        if let Some(burst_multiplier) = req.burst_multiplier {
            self.validate_burst_multiplier(burst_multiplier)?;
            quota.burst_multiplier = burst_multiplier;
        }

        if let Some(status) = req.status {
            quota.status = status;
        }

        if let Some(reason) = req.status_reason {
            quota.status_reason = Some(reason);
        }

        quota.updated_at = Utc::now();

        // Update cache
        {
            let mut cache = self.cache.write().await;
            cache.insert(quota.id, quota.clone());
        }

        Ok(quota)
    }

    /// Activate a pending quota.
    pub async fn activate_quota(&self, id: Uuid) -> Result<EphemeralQuota> {
        let mut quota = self.get_quota(id).await?;
        quota.activate()?;

        {
            let mut cache = self.cache.write().await;
            cache.insert(quota.id, quota.clone());
        }

        Ok(quota)
    }

    /// Suspend a quota.
    pub async fn suspend_quota(&self, id: Uuid, reason: &str) -> Result<EphemeralQuota> {
        let mut quota = self.get_quota(id).await?;
        quota.suspend(reason);

        {
            let mut cache = self.cache.write().await;
            cache.insert(quota.id, quota.clone());
        }

        Ok(quota)
    }

    /// Revoke a quota.
    pub async fn revoke_quota(&self, id: Uuid, reason: &str) -> Result<EphemeralQuota> {
        let mut quota = self.get_quota(id).await?;
        quota.revoke(reason);

        {
            let mut cache = self.cache.write().await;
            cache.insert(quota.id, quota.clone());
        }

        Ok(quota)
    }

    /// Check if a usage operation is allowed.
    pub async fn check_usage(
        &self,
        quota_id: Uuid,
        amount: Decimal,
    ) -> Result<EphemeralQuotaCheckResult> {
        let quota = self.get_quota(quota_id).await?;

        // Get time window if configured
        let time_window = if let Some(tw_id) = quota.time_window_id {
            let windows = self.time_windows.read().await;
            windows.get(&tw_id).cloned()
        } else {
            None
        };

        Ok(quota.check_usage(amount, time_window.as_ref()))
    }

    /// Record usage against a quota.
    pub async fn record_usage(
        &self,
        quota_id: Uuid,
        amount: Decimal,
        job_id: Option<Uuid>,
        description: Option<String>,
    ) -> Result<EphemeralQuotaUsage> {
        let mut quota = self.get_quota(quota_id).await?;

        // Check if usage is allowed
        let check = self.check_usage(quota_id, amount).await?;
        if !check.allowed {
            return Err(HpcError::quota_exceeded(
                check.denial_reason.unwrap_or_else(|| "Usage not allowed".to_string()),
            ));
        }

        // Record the usage
        let mut usage = quota.record_usage(amount)?;
        usage.job_id = job_id;
        usage.description = description;

        // Update cache
        {
            let mut cache = self.cache.write().await;
            cache.insert(quota.id, quota.clone());
        }

        Ok(usage)
    }

    /// Record usage with burst (allows exceeding normal limit up to burst limit).
    pub async fn record_usage_with_burst(
        &self,
        quota_id: Uuid,
        amount: Decimal,
        job_id: Option<Uuid>,
        description: Option<String>,
    ) -> Result<(EphemeralQuotaUsage, bool)> {
        let quota = self.get_quota(quota_id).await?;

        if !quota.burst_enabled {
            let usage = self.record_usage(quota_id, amount, job_id, description).await?;
            return Ok((usage, false));
        }

        // Check if this will use burst
        let normal_available = quota.limit_value - quota.used_value - quota.reserved_value;
        let using_burst = amount > normal_available;

        let usage = self.record_usage(quota_id, amount, job_id, description).await?;
        Ok((usage, using_burst))
    }

    /// Release usage back to quota.
    pub async fn release_usage(
        &self,
        quota_id: Uuid,
        amount: Decimal,
    ) -> Result<EphemeralQuotaUsage> {
        let mut quota = self.get_quota(quota_id).await?;
        let usage = quota.release_usage(amount)?;

        {
            let mut cache = self.cache.write().await;
            cache.insert(quota.id, quota.clone());
        }

        Ok(usage)
    }

    /// Reserve an amount for pending operations.
    pub async fn reserve(&self, quota_id: Uuid, amount: Decimal) -> Result<()> {
        let mut quota = self.get_quota(quota_id).await?;
        quota.reserve(amount)?;

        {
            let mut cache = self.cache.write().await;
            cache.insert(quota.id, quota);
        }

        Ok(())
    }

    /// Commit a reservation.
    pub async fn commit_reservation(
        &self,
        quota_id: Uuid,
        amount: Decimal,
    ) -> Result<EphemeralQuotaUsage> {
        let mut quota = self.get_quota(quota_id).await?;
        let usage = quota.commit_reservation(amount)?;

        {
            let mut cache = self.cache.write().await;
            cache.insert(quota.id, quota);
        }

        Ok(usage)
    }

    /// Cancel a reservation.
    pub async fn cancel_reservation(&self, quota_id: Uuid, amount: Decimal) -> Result<()> {
        let mut quota = self.get_quota(quota_id).await?;
        quota.cancel_reservation(amount);

        {
            let mut cache = self.cache.write().await;
            cache.insert(quota.id, quota);
        }

        Ok(())
    }

    /// Extend quota expiry.
    pub async fn extend_expiry(
        &self,
        quota_id: Uuid,
        new_expiry: DateTime<Utc>,
    ) -> Result<EphemeralQuota> {
        let mut quota = self.get_quota(quota_id).await?;
        quota.extend_expiry(new_expiry)?;

        {
            let mut cache = self.cache.write().await;
            cache.insert(quota.id, quota.clone());
        }

        Ok(quota)
    }

    /// Increase quota limit.
    pub async fn increase_limit(&self, quota_id: Uuid, additional: Decimal) -> Result<EphemeralQuota> {
        let mut quota = self.get_quota(quota_id).await?;
        quota.increase_limit(additional);

        {
            let mut cache = self.cache.write().await;
            cache.insert(quota.id, quota.clone());
        }

        Ok(quota)
    }

    /// Register a time window.
    pub async fn register_time_window(&self, time_window: TimeWindow) -> Result<()> {
        let mut windows = self.time_windows.write().await;
        windows.insert(time_window.id, time_window);
        Ok(())
    }

    /// Get sponsor usage summary.
    pub async fn get_sponsor_summary(&self, sponsor_id: &str) -> Result<SponsorUsageSummary> {
        let quotas = self.get_quotas_by_sponsor(sponsor_id).await?;

        let active_quotas = quotas
            .iter()
            .filter(|q| q.status == EphemeralQuotaStatus::Active)
            .count() as i32;

        let total_allocated: Decimal = quotas.iter().map(|q| q.limit_value).sum();
        let total_used: Decimal = quotas.iter().map(|q| q.used_value).sum();
        let total_cost: Decimal = quotas.iter().map(|q| q.actual_cost).sum();

        // Group by resource type
        let mut by_resource: HashMap<ResourceType, (Decimal, Decimal, Decimal)> = HashMap::new();
        for quota in &quotas {
            let entry = by_resource.entry(quota.resource_type).or_insert((
                Decimal::ZERO,
                Decimal::ZERO,
                Decimal::ZERO,
            ));
            entry.0 += quota.limit_value;
            entry.1 += quota.used_value;
            entry.2 += quota.actual_cost;
        }

        let by_resource_type: Vec<ResourceTypeSummary> = by_resource
            .into_iter()
            .map(|(rt, (allocated, used, cost))| ResourceTypeSummary {
                resource_type: rt,
                allocated,
                used,
                cost,
            })
            .collect();

        Ok(SponsorUsageSummary {
            sponsor_id: sponsor_id.to_string(),
            total_quotas: quotas.len() as i32,
            active_quotas,
            total_allocated,
            total_used,
            total_cost,
            by_resource_type,
        })
    }

    /// Process expired quotas.
    pub async fn process_expirations(&self) -> Result<Vec<Uuid>> {
        let mut expired = Vec::new();
        let now = Utc::now();
        let grace = chrono::Duration::seconds(self.config.expiry_grace_period_secs);

        let mut cache = self.cache.write().await;

        for quota in cache.values_mut() {
            if quota.expires_at + grace < now && !quota.status.is_terminal() {
                quota.mark_expired();
                expired.push(quota.id);
            }
        }

        Ok(expired)
    }

    /// Clean up old expired quotas from cache.
    pub async fn cleanup_expired(&self, older_than_hours: i64) -> Result<usize> {
        let cutoff = Utc::now() - chrono::Duration::hours(older_than_hours);

        let mut cache = self.cache.write().await;
        let before = cache.len();

        cache.retain(|_, q| {
            !(q.status.is_terminal() && q.updated_at < cutoff)
        });

        Ok(before - cache.len())
    }

    /// Get service statistics.
    pub async fn get_stats(&self) -> EphemeralQuotaStats {
        let cache = self.cache.read().await;

        let total = cache.len();
        let active = cache.values().filter(|q| q.status == EphemeralQuotaStatus::Active).count();
        let pending = cache.values().filter(|q| q.status == EphemeralQuotaStatus::Pending).count();
        let expired = cache.values().filter(|q| q.status == EphemeralQuotaStatus::Expired).count();
        let exhausted = cache.values().filter(|q| q.status == EphemeralQuotaStatus::Exhausted).count();

        let total_allocated: Decimal = cache.values().map(|q| q.limit_value).sum();
        let total_used: Decimal = cache.values().map(|q| q.used_value).sum();
        let total_cost: Decimal = cache.values().map(|q| q.actual_cost).sum();

        let burst_enabled = cache.values().filter(|q| q.burst_enabled).count();
        let using_burst = cache.values().filter(|q| q.is_using_burst()).count();

        EphemeralQuotaStats {
            total_quotas: total,
            active_quotas: active,
            pending_quotas: pending,
            expired_quotas: expired,
            exhausted_quotas: exhausted,
            total_allocated,
            total_used,
            total_cost,
            burst_enabled_count: burst_enabled,
            using_burst_count: using_burst,
        }
    }
}

/// Statistics for the ephemeral quota service.
#[derive(Debug, Clone)]
pub struct EphemeralQuotaStats {
    pub total_quotas: usize,
    pub active_quotas: usize,
    pub pending_quotas: usize,
    pub expired_quotas: usize,
    pub exhausted_quotas: usize,
    pub total_allocated: Decimal,
    pub total_used: Decimal,
    pub total_cost: Decimal,
    pub burst_enabled_count: usize,
    pub using_burst_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    fn mock_repository() -> QuotaRepository {
        // For tests, we just use a placeholder that won't be called
        // since we're using in-memory cache
        QuotaRepository::new_test()
    }

    async fn create_test_service() -> EphemeralQuotaService {
        EphemeralQuotaService::new(mock_repository())
    }

    fn create_test_request() -> CreateEphemeralQuotaRequest {
        CreateEphemeralQuotaRequest {
            ephemeral_identity_id: Uuid::new_v4(),
            tenant_id: Uuid::new_v4(),
            sponsor_id: "sponsor1".to_string(),
            beneficiary_id: "user1".to_string(),
            resource_type: ResourceType::GpuHours,
            limit_value: dec!(100),
            starts_at: None,
            expires_at: Utc::now() + chrono::Duration::days(7),
            time_window_id: None,
            burst_enabled: None,
            burst_multiplier: None,
            pool_id: None,
            cost_rate: None,
        }
    }

    #[tokio::test]
    async fn test_create_quota() {
        let service = create_test_service().await;
        let req = create_test_request();

        let quota = service.create_quota(req.clone()).await.unwrap();

        assert_eq!(quota.sponsor_id, req.sponsor_id);
        assert_eq!(quota.beneficiary_id, req.beneficiary_id);
        assert_eq!(quota.limit_value, req.limit_value);
        assert_eq!(quota.status, EphemeralQuotaStatus::Pending);
    }

    #[tokio::test]
    async fn test_create_quota_with_burst() {
        let service = create_test_service().await;
        let mut req = create_test_request();
        req.burst_enabled = Some(true);
        req.burst_multiplier = Some(dec!(1.5));

        let quota = service.create_quota(req).await.unwrap();

        assert!(quota.burst_enabled);
        assert_eq!(quota.burst_multiplier, dec!(1.5));
        assert_eq!(quota.effective_limit(), dec!(150));
    }

    #[tokio::test]
    async fn test_create_quota_invalid_burst_multiplier() {
        let service = create_test_service().await;
        let mut req = create_test_request();
        req.burst_enabled = Some(true);
        req.burst_multiplier = Some(dec!(0.5)); // Invalid: < 1.0

        let result = service.create_quota(req).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_create_quota_burst_exceeds_max() {
        let service = create_test_service().await;
        let mut req = create_test_request();
        req.burst_enabled = Some(true);
        req.burst_multiplier = Some(dec!(5.0)); // Exceeds default max of 2.0

        let result = service.create_quota(req).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_activate_quota() {
        let service = create_test_service().await;
        let req = create_test_request();
        let quota = service.create_quota(req).await.unwrap();

        let activated = service.activate_quota(quota.id).await.unwrap();
        assert_eq!(activated.status, EphemeralQuotaStatus::Active);
    }

    #[tokio::test]
    async fn test_record_usage() {
        let service = create_test_service().await;
        let req = create_test_request();
        let quota = service.create_quota(req).await.unwrap();
        service.activate_quota(quota.id).await.unwrap();

        let usage = service
            .record_usage(quota.id, dec!(25), None, None)
            .await
            .unwrap();

        assert_eq!(usage.amount, dec!(25));

        let updated = service.get_quota(quota.id).await.unwrap();
        assert_eq!(updated.used_value, dec!(25));
    }

    #[tokio::test]
    async fn test_record_usage_exceeds_quota() {
        let service = create_test_service().await;
        let req = create_test_request();
        let quota = service.create_quota(req).await.unwrap();
        service.activate_quota(quota.id).await.unwrap();

        let result = service
            .record_usage(quota.id, dec!(150), None, None)
            .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_record_usage_with_burst() {
        let service = create_test_service().await;
        let mut req = create_test_request();
        req.burst_enabled = Some(true);
        req.burst_multiplier = Some(dec!(1.5));

        let quota = service.create_quota(req).await.unwrap();
        service.activate_quota(quota.id).await.unwrap();

        // Use 110, which exceeds normal limit of 100 but within burst limit of 150
        let (usage, using_burst) = service
            .record_usage_with_burst(quota.id, dec!(110), None, None)
            .await
            .unwrap();

        assert_eq!(usage.amount, dec!(110));
        assert!(using_burst);
    }

    #[tokio::test]
    async fn test_reserve_and_commit() {
        let service = create_test_service().await;
        let req = create_test_request();
        let quota = service.create_quota(req).await.unwrap();
        service.activate_quota(quota.id).await.unwrap();

        service.reserve(quota.id, dec!(30)).await.unwrap();
        let reserved = service.get_quota(quota.id).await.unwrap();
        assert_eq!(reserved.reserved_value, dec!(30));
        assert_eq!(reserved.available(), dec!(70));

        let usage = service.commit_reservation(quota.id, dec!(30)).await.unwrap();
        assert_eq!(usage.amount, dec!(30));

        let committed = service.get_quota(quota.id).await.unwrap();
        assert_eq!(committed.reserved_value, dec!(0));
        assert_eq!(committed.used_value, dec!(30));
    }

    #[tokio::test]
    async fn test_release_usage() {
        let service = create_test_service().await;
        let req = create_test_request();
        let quota = service.create_quota(req).await.unwrap();
        service.activate_quota(quota.id).await.unwrap();

        service.record_usage(quota.id, dec!(50), None, None).await.unwrap();
        service.release_usage(quota.id, dec!(20)).await.unwrap();

        let updated = service.get_quota(quota.id).await.unwrap();
        assert_eq!(updated.used_value, dec!(30));
    }

    #[tokio::test]
    async fn test_extend_expiry() {
        let service = create_test_service().await;
        let req = create_test_request();
        let quota = service.create_quota(req.clone()).await.unwrap();
        service.activate_quota(quota.id).await.unwrap();

        let new_expiry = req.expires_at + chrono::Duration::days(7);
        let extended = service.extend_expiry(quota.id, new_expiry).await.unwrap();

        assert_eq!(extended.expires_at, new_expiry);
    }

    #[tokio::test]
    async fn test_increase_limit() {
        let service = create_test_service().await;
        let req = create_test_request();
        let quota = service.create_quota(req).await.unwrap();
        service.activate_quota(quota.id).await.unwrap();

        let increased = service.increase_limit(quota.id, dec!(50)).await.unwrap();
        assert_eq!(increased.limit_value, dec!(150));
    }

    #[tokio::test]
    async fn test_suspend_and_revoke() {
        let service = create_test_service().await;
        let req = create_test_request();
        let quota = service.create_quota(req).await.unwrap();
        service.activate_quota(quota.id).await.unwrap();

        let suspended = service.suspend_quota(quota.id, "Policy violation").await.unwrap();
        assert_eq!(suspended.status, EphemeralQuotaStatus::Suspended);

        // Create another quota to test revoke
        let req2 = create_test_request();
        let quota2 = service.create_quota(req2).await.unwrap();
        service.activate_quota(quota2.id).await.unwrap();

        let revoked = service.revoke_quota(quota2.id, "Sponsor request").await.unwrap();
        assert_eq!(revoked.status, EphemeralQuotaStatus::Revoked);
    }

    #[tokio::test]
    async fn test_get_sponsor_summary() {
        let service = create_test_service().await;

        // Create multiple quotas for same sponsor
        let mut req1 = create_test_request();
        req1.limit_value = dec!(100);
        req1.cost_rate = Some(dec!(0.10));

        let mut req2 = create_test_request();
        req2.limit_value = dec!(50);
        req2.cost_rate = Some(dec!(0.10));
        req2.resource_type = ResourceType::StorageGb;

        let quota1 = service.create_quota(req1).await.unwrap();
        let quota2 = service.create_quota(req2).await.unwrap();

        service.activate_quota(quota1.id).await.unwrap();
        service.activate_quota(quota2.id).await.unwrap();

        service.record_usage(quota1.id, dec!(25), None, None).await.unwrap();

        let summary = service.get_sponsor_summary("sponsor1").await.unwrap();

        assert_eq!(summary.total_quotas, 2);
        assert_eq!(summary.active_quotas, 2);
        assert_eq!(summary.total_allocated, dec!(150));
        assert_eq!(summary.total_used, dec!(25));
    }

    #[tokio::test]
    async fn test_get_stats() {
        let service = create_test_service().await;

        let req1 = create_test_request();
        let mut req2 = create_test_request();
        req2.burst_enabled = Some(true);
        req2.burst_multiplier = Some(dec!(1.5));

        let quota1 = service.create_quota(req1).await.unwrap();
        let quota2 = service.create_quota(req2).await.unwrap();

        service.activate_quota(quota1.id).await.unwrap();
        service.activate_quota(quota2.id).await.unwrap();

        let stats = service.get_stats().await;

        assert_eq!(stats.total_quotas, 2);
        assert_eq!(stats.active_quotas, 2);
        assert_eq!(stats.burst_enabled_count, 1);
    }

    #[tokio::test]
    async fn test_check_usage() {
        let service = create_test_service().await;
        let req = create_test_request();
        let quota = service.create_quota(req).await.unwrap();
        service.activate_quota(quota.id).await.unwrap();

        let check = service.check_usage(quota.id, dec!(50)).await.unwrap();
        assert!(check.allowed);
        assert_eq!(check.available, dec!(100));
        assert!(!check.using_burst);

        // Check more than available
        service.record_usage(quota.id, dec!(80), None, None).await.unwrap();
        let check2 = service.check_usage(quota.id, dec!(30)).await.unwrap();
        assert!(!check2.allowed);
    }
}
