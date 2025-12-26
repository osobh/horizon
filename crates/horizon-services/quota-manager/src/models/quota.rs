use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use sqlx::FromRow;
use utoipa::ToSchema;
use uuid::Uuid;

use crate::error::{HpcError, Result, QuotaErrorExt};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, sqlx::Type, ToSchema)]
#[sqlx(type_name = "text", rename_all = "lowercase")]
#[serde(rename_all = "lowercase")]
pub enum EntityType {
    Organization,
    Team,
    User,
}

impl EntityType {
    pub fn as_str(&self) -> &'static str {
        match self {
            EntityType::Organization => "organization",
            EntityType::Team => "team",
            EntityType::User => "user",
        }
    }

    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "organization" => Ok(EntityType::Organization),
            "team" => Ok(EntityType::Team),
            "user" => Ok(EntityType::User),
            _ => Err(HpcError::invalid_input("entity_type", format!("Invalid entity type: {}", s))),
        }
    }

    pub fn parent_type(&self) -> Option<EntityType> {
        match self {
            EntityType::Organization => None,
            EntityType::Team => Some(EntityType::Organization),
            EntityType::User => Some(EntityType::Team),
        }
    }

    pub fn child_type(&self) -> Option<EntityType> {
        match self {
            EntityType::Organization => Some(EntityType::Team),
            EntityType::Team => Some(EntityType::User),
            EntityType::User => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, sqlx::Type, ToSchema)]
#[sqlx(type_name = "text", rename_all = "snake_case")]
#[serde(rename_all = "snake_case")]
pub enum ResourceType {
    // Compute resources
    GpuHours,
    ConcurrentGpus,
    CpuHours,
    ConcurrentCpus,
    TpuHours,

    // Memory resources
    MemoryGb,
    MemoryGbHours,

    // Storage resources
    StorageGb,
    StorageGbHours,

    // Network resources
    NetworkGbps,
    NetworkGbpsHours,

    // Custom resources
    CustomLicenses,
    CustomApiCredits,
}

impl ResourceType {
    pub fn as_str(&self) -> &'static str {
        match self {
            ResourceType::GpuHours => "gpu_hours",
            ResourceType::ConcurrentGpus => "concurrent_gpus",
            ResourceType::CpuHours => "cpu_hours",
            ResourceType::ConcurrentCpus => "concurrent_cpus",
            ResourceType::TpuHours => "tpu_hours",
            ResourceType::MemoryGb => "memory_gb",
            ResourceType::MemoryGbHours => "memory_gb_hours",
            ResourceType::StorageGb => "storage_gb",
            ResourceType::StorageGbHours => "storage_gb_hours",
            ResourceType::NetworkGbps => "network_gbps",
            ResourceType::NetworkGbpsHours => "network_gbps_hours",
            ResourceType::CustomLicenses => "custom_licenses",
            ResourceType::CustomApiCredits => "custom_api_credits",
        }
    }

    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "gpu_hours" => Ok(ResourceType::GpuHours),
            "concurrent_gpus" => Ok(ResourceType::ConcurrentGpus),
            "cpu_hours" => Ok(ResourceType::CpuHours),
            "concurrent_cpus" => Ok(ResourceType::ConcurrentCpus),
            "tpu_hours" => Ok(ResourceType::TpuHours),
            "memory_gb" => Ok(ResourceType::MemoryGb),
            "memory_gb_hours" => Ok(ResourceType::MemoryGbHours),
            "storage_gb" => Ok(ResourceType::StorageGb),
            "storage_gb_hours" => Ok(ResourceType::StorageGbHours),
            "network_gbps" => Ok(ResourceType::NetworkGbps),
            "network_gbps_hours" => Ok(ResourceType::NetworkGbpsHours),
            "custom_licenses" => Ok(ResourceType::CustomLicenses),
            "custom_api_credits" => Ok(ResourceType::CustomApiCredits),
            _ => Err(HpcError::invalid_input("resource_type", format!("Invalid resource type: {}", s))),
        }
    }

    /// Check if this resource type is time-based (hours)
    pub fn is_time_based(&self) -> bool {
        matches!(
            self,
            ResourceType::GpuHours
                | ResourceType::CpuHours
                | ResourceType::TpuHours
                | ResourceType::MemoryGbHours
                | ResourceType::StorageGbHours
                | ResourceType::NetworkGbpsHours
        )
    }

    /// Check if this resource type is concurrent/capacity-based
    pub fn is_concurrent(&self) -> bool {
        matches!(
            self,
            ResourceType::ConcurrentGpus
                | ResourceType::ConcurrentCpus
                | ResourceType::MemoryGb
                | ResourceType::StorageGb
                | ResourceType::NetworkGbps
        )
    }

    /// Get the resource category
    pub fn category(&self) -> ResourceCategory {
        match self {
            ResourceType::GpuHours | ResourceType::ConcurrentGpus => ResourceCategory::Gpu,
            ResourceType::CpuHours | ResourceType::ConcurrentCpus => ResourceCategory::Cpu,
            ResourceType::TpuHours => ResourceCategory::Tpu,
            ResourceType::MemoryGb | ResourceType::MemoryGbHours => ResourceCategory::Memory,
            ResourceType::StorageGb | ResourceType::StorageGbHours => ResourceCategory::Storage,
            ResourceType::NetworkGbps | ResourceType::NetworkGbpsHours => ResourceCategory::Network,
            ResourceType::CustomLicenses | ResourceType::CustomApiCredits => ResourceCategory::Custom,
        }
    }
}

/// High-level resource category for grouping
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum ResourceCategory {
    Gpu,
    Cpu,
    Tpu,
    Memory,
    Storage,
    Network,
    Custom,
}

#[derive(Debug, Clone, FromRow, Serialize, Deserialize, ToSchema)]
pub struct Quota {
    pub id: Uuid,
    pub entity_type: EntityType,
    pub entity_id: String,
    pub parent_id: Option<Uuid>,
    pub resource_type: ResourceType,
    pub limit_value: Decimal,
    pub soft_limit: Option<Decimal>,
    pub burst_limit: Option<Decimal>,
    pub overcommit_ratio: Decimal,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct CreateQuotaRequest {
    pub entity_type: EntityType,
    pub entity_id: String,
    pub parent_id: Option<Uuid>,
    pub resource_type: ResourceType,
    pub limit_value: Decimal,
    pub soft_limit: Option<Decimal>,
    pub burst_limit: Option<Decimal>,
    pub overcommit_ratio: Option<Decimal>,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct UpdateQuotaRequest {
    pub limit_value: Option<Decimal>,
    pub soft_limit: Option<Decimal>,
    pub burst_limit: Option<Decimal>,
    pub overcommit_ratio: Option<Decimal>,
}

impl Quota {
    pub fn validate(&self) -> Result<()> {
        // Validate limit values
        if self.limit_value < Decimal::ZERO {
            return Err(HpcError::invalid_configuration(
                "limit_value must be non-negative",
            ));
        }

        if let Some(soft_limit) = self.soft_limit {
            if soft_limit < Decimal::ZERO || soft_limit > self.limit_value {
                return Err(HpcError::invalid_configuration(
                    "soft_limit must be between 0 and limit_value",
                ));
            }
        }

        if let Some(burst_limit) = self.burst_limit {
            if burst_limit < self.limit_value {
                return Err(HpcError::invalid_configuration(
                    "burst_limit must be >= limit_value",
                ));
            }
        }

        if self.overcommit_ratio < Decimal::ONE {
            return Err(HpcError::invalid_configuration(
                "overcommit_ratio must be >= 1.0",
            ));
        }

        Ok(())
    }

    pub fn effective_limit(&self) -> Decimal {
        self.limit_value * self.overcommit_ratio
    }

    pub fn is_soft_limit_exceeded(&self, usage: Decimal) -> bool {
        if let Some(soft_limit) = self.soft_limit {
            usage > soft_limit
        } else {
            false
        }
    }

    pub fn is_hard_limit_exceeded(&self, usage: Decimal) -> bool {
        usage > self.effective_limit()
    }

    pub fn can_burst(&self) -> bool {
        self.burst_limit.is_some()
    }

    pub fn available_quota(&self, current_usage: Decimal) -> Decimal {
        let effective = self.effective_limit();
        if current_usage >= effective {
            Decimal::ZERO
        } else {
            effective - current_usage
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_entity_type_hierarchy() {
        assert_eq!(EntityType::Organization.parent_type(), None);
        assert_eq!(EntityType::Team.parent_type(), Some(EntityType::Organization));
        assert_eq!(EntityType::User.parent_type(), Some(EntityType::Team));

        assert_eq!(EntityType::Organization.child_type(), Some(EntityType::Team));
        assert_eq!(EntityType::Team.child_type(), Some(EntityType::User));
        assert_eq!(EntityType::User.child_type(), None);
    }

    #[test]
    fn test_entity_type_from_str() {
        assert_eq!(EntityType::from_str("organization").unwrap(), EntityType::Organization);
        assert_eq!(EntityType::from_str("team").unwrap(), EntityType::Team);
        assert_eq!(EntityType::from_str("user").unwrap(), EntityType::User);
        assert!(EntityType::from_str("invalid").is_err());
    }

    #[test]
    fn test_resource_type_from_str() {
        assert_eq!(ResourceType::from_str("gpu_hours").unwrap(), ResourceType::GpuHours);
        assert_eq!(ResourceType::from_str("concurrent_gpus").unwrap(), ResourceType::ConcurrentGpus);
        assert_eq!(ResourceType::from_str("cpu_hours").unwrap(), ResourceType::CpuHours);
        assert_eq!(ResourceType::from_str("tpu_hours").unwrap(), ResourceType::TpuHours);
        assert_eq!(ResourceType::from_str("memory_gb").unwrap(), ResourceType::MemoryGb);
        assert_eq!(ResourceType::from_str("storage_gb_hours").unwrap(), ResourceType::StorageGbHours);
        assert_eq!(ResourceType::from_str("network_gbps").unwrap(), ResourceType::NetworkGbps);
        assert!(ResourceType::from_str("invalid").is_err());
    }

    #[test]
    fn test_resource_type_is_time_based() {
        assert!(ResourceType::GpuHours.is_time_based());
        assert!(ResourceType::CpuHours.is_time_based());
        assert!(ResourceType::TpuHours.is_time_based());
        assert!(ResourceType::MemoryGbHours.is_time_based());
        assert!(ResourceType::StorageGbHours.is_time_based());
        assert!(ResourceType::NetworkGbpsHours.is_time_based());

        assert!(!ResourceType::ConcurrentGpus.is_time_based());
        assert!(!ResourceType::MemoryGb.is_time_based());
        assert!(!ResourceType::StorageGb.is_time_based());
    }

    #[test]
    fn test_resource_type_is_concurrent() {
        assert!(ResourceType::ConcurrentGpus.is_concurrent());
        assert!(ResourceType::ConcurrentCpus.is_concurrent());
        assert!(ResourceType::MemoryGb.is_concurrent());
        assert!(ResourceType::StorageGb.is_concurrent());
        assert!(ResourceType::NetworkGbps.is_concurrent());

        assert!(!ResourceType::GpuHours.is_concurrent());
        assert!(!ResourceType::CpuHours.is_concurrent());
        assert!(!ResourceType::TpuHours.is_concurrent());
    }

    #[test]
    fn test_resource_type_category() {
        assert_eq!(ResourceType::GpuHours.category(), ResourceCategory::Gpu);
        assert_eq!(ResourceType::ConcurrentGpus.category(), ResourceCategory::Gpu);
        assert_eq!(ResourceType::CpuHours.category(), ResourceCategory::Cpu);
        assert_eq!(ResourceType::TpuHours.category(), ResourceCategory::Tpu);
        assert_eq!(ResourceType::MemoryGb.category(), ResourceCategory::Memory);
        assert_eq!(ResourceType::StorageGbHours.category(), ResourceCategory::Storage);
        assert_eq!(ResourceType::NetworkGbps.category(), ResourceCategory::Network);
        assert_eq!(ResourceType::CustomLicenses.category(), ResourceCategory::Custom);
    }

    #[test]
    fn test_quota_validate_negative_limit() {
        let quota = create_test_quota(dec!(-10.0), None, None, dec!(1.0));
        assert!(quota.validate().is_err());
    }

    #[test]
    fn test_quota_validate_soft_limit_exceeds_hard_limit() {
        let quota = create_test_quota(dec!(100.0), Some(dec!(150.0)), None, dec!(1.0));
        assert!(quota.validate().is_err());
    }

    #[test]
    fn test_quota_validate_burst_limit_below_hard_limit() {
        let quota = create_test_quota(dec!(100.0), None, Some(dec!(50.0)), dec!(1.0));
        assert!(quota.validate().is_err());
    }

    #[test]
    fn test_quota_validate_overcommit_below_one() {
        let quota = create_test_quota(dec!(100.0), None, None, dec!(0.5));
        assert!(quota.validate().is_err());
    }

    #[test]
    fn test_quota_validate_valid() {
        let quota = create_test_quota(dec!(100.0), Some(dec!(80.0)), Some(dec!(120.0)), dec!(1.5));
        assert!(quota.validate().is_ok());
    }

    #[test]
    fn test_quota_effective_limit() {
        let quota = create_test_quota(dec!(100.0), None, None, dec!(1.5));
        assert_eq!(quota.effective_limit(), dec!(150.0));
    }

    #[test]
    fn test_quota_soft_limit_exceeded() {
        let quota = create_test_quota(dec!(100.0), Some(dec!(80.0)), None, dec!(1.0));
        assert!(!quota.is_soft_limit_exceeded(dec!(70.0)));
        assert!(quota.is_soft_limit_exceeded(dec!(90.0)));
    }

    #[test]
    fn test_quota_hard_limit_exceeded() {
        let quota = create_test_quota(dec!(100.0), None, None, dec!(1.5));
        assert!(!quota.is_hard_limit_exceeded(dec!(140.0)));
        assert!(quota.is_hard_limit_exceeded(dec!(160.0)));
    }

    #[test]
    fn test_quota_available_quota() {
        let quota = create_test_quota(dec!(100.0), None, None, dec!(1.5));
        assert_eq!(quota.available_quota(dec!(50.0)), dec!(100.0));
        assert_eq!(quota.available_quota(dec!(150.0)), dec!(0.0));
        assert_eq!(quota.available_quota(dec!(160.0)), dec!(0.0));
    }

    #[test]
    fn test_quota_can_burst() {
        let quota_no_burst = create_test_quota(dec!(100.0), None, None, dec!(1.0));
        assert!(!quota_no_burst.can_burst());

        let quota_with_burst = create_test_quota(dec!(100.0), None, Some(dec!(120.0)), dec!(1.0));
        assert!(quota_with_burst.can_burst());
    }

    fn create_test_quota(
        limit: Decimal,
        soft_limit: Option<Decimal>,
        burst_limit: Option<Decimal>,
        overcommit_ratio: Decimal,
    ) -> Quota {
        Quota {
            id: Uuid::new_v4(),
            entity_type: EntityType::User,
            entity_id: "test-user".to_string(),
            parent_id: None,
            resource_type: ResourceType::GpuHours,
            limit_value: limit,
            soft_limit,
            burst_limit,
            overcommit_ratio,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        }
    }
}
