use rust_decimal::Decimal;
use rust_decimal::prelude::ToPrimitive;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;
use uuid::Uuid;

use super::{EntityType, Quota, ResourceType};
use crate::error::{HpcError, Result, QuotaErrorExt};

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct QuotaHierarchy {
    pub quota: Quota,
    pub parent: Option<Box<QuotaHierarchy>>,
    pub children: Vec<QuotaHierarchy>,
    pub current_usage: Decimal,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct QuotaUsageStats {
    pub quota_id: Uuid,
    pub entity_type: EntityType,
    pub entity_id: String,
    pub resource_type: ResourceType,
    pub limit: Decimal,
    pub effective_limit: Decimal,
    pub usage: Decimal,
    pub available: Decimal,
    pub utilization_percent: f64,
}

impl QuotaHierarchy {
    pub fn new(quota: Quota) -> Self {
        Self {
            quota,
            parent: None,
            children: Vec::new(),
            current_usage: Decimal::ZERO,
        }
    }

    pub fn with_parent(mut self, parent: QuotaHierarchy) -> Self {
        self.parent = Some(Box::new(parent));
        self
    }

    pub fn add_child(&mut self, child: QuotaHierarchy) {
        self.children.push(child);
    }

    pub fn validate_hierarchy(&self) -> Result<()> {
        // Check if parent relationship is valid
        if let Some(parent) = &self.parent {
            if self.quota.entity_type.parent_type() != Some(parent.quota.entity_type) {
                return Err(HpcError::invalid_hierarchy(format!(
                    "Invalid parent type: {:?} cannot be parent of {:?}",
                    parent.quota.entity_type, self.quota.entity_type
                )));
            }

            // Check if resource types match
            if self.quota.resource_type != parent.quota.resource_type {
                return Err(HpcError::invalid_hierarchy(
                    "Resource type must match parent",
                ));
            }

            // Check if quota doesn't exceed parent
            if self.quota.limit_value > parent.quota.limit_value {
                return Err(HpcError::invalid_hierarchy(format!(
                    "Quota limit {} exceeds parent limit {}",
                    self.quota.limit_value, parent.quota.limit_value
                )));
            }
        }

        // Validate all children
        for child in &self.children {
            child.validate_hierarchy()?;
        }

        Ok(())
    }

    pub fn total_children_quota(&self) -> Decimal {
        self.children
            .iter()
            .map(|c| c.quota.limit_value)
            .sum()
    }

    pub fn has_capacity_for_child(&self, child_limit: Decimal) -> bool {
        let total_children = self.total_children_quota();
        total_children + child_limit <= self.quota.effective_limit()
    }

    pub fn available_for_allocation(&self) -> Decimal {
        let effective = self.quota.effective_limit();
        if self.current_usage >= effective {
            Decimal::ZERO
        } else {
            effective - self.current_usage
        }
    }

    pub fn can_allocate(&self, amount: Decimal) -> bool {
        self.available_for_allocation() >= amount
    }
}

impl QuotaUsageStats {
    pub fn from_quota(quota: &Quota, usage: Decimal) -> Self {
        let effective_limit = quota.effective_limit();
        let available = if usage >= effective_limit {
            Decimal::ZERO
        } else {
            effective_limit - usage
        };

        let utilization_percent = if effective_limit > Decimal::ZERO {
            (usage / effective_limit).to_f64().unwrap_or(0.0) * 100.0
        } else {
            0.0
        };

        Self {
            quota_id: quota.id,
            entity_type: quota.entity_type,
            entity_id: quota.entity_id.clone(),
            resource_type: quota.resource_type,
            limit: quota.limit_value,
            effective_limit,
            usage,
            available,
            utilization_percent,
        }
    }

    pub fn is_at_soft_limit(&self, soft_limit: Option<Decimal>) -> bool {
        if let Some(soft) = soft_limit {
            self.usage >= soft
        } else {
            false
        }
    }

    pub fn is_at_hard_limit(&self) -> bool {
        self.usage >= self.effective_limit
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use rust_decimal_macros::dec;

    #[test]
    fn test_hierarchy_validation_valid() {
        let parent = create_test_hierarchy(EntityType::Organization, dec!(100.0));
        let child = create_test_hierarchy(EntityType::Team, dec!(50.0))
            .with_parent(parent);

        assert!(child.validate_hierarchy().is_ok());
    }

    #[test]
    fn test_hierarchy_validation_invalid_parent_type() {
        let parent = create_test_hierarchy(EntityType::User, dec!(100.0));
        let child = create_test_hierarchy(EntityType::Team, dec!(50.0))
            .with_parent(parent);

        assert!(child.validate_hierarchy().is_err());
    }

    #[test]
    fn test_hierarchy_validation_exceeds_parent_limit() {
        let parent = create_test_hierarchy(EntityType::Organization, dec!(100.0));
        let child = create_test_hierarchy(EntityType::Team, dec!(150.0))
            .with_parent(parent);

        assert!(child.validate_hierarchy().is_err());
    }

    #[test]
    fn test_hierarchy_total_children_quota() {
        let mut parent = create_test_hierarchy(EntityType::Organization, dec!(100.0));
        parent.add_child(create_test_hierarchy(EntityType::Team, dec!(30.0)));
        parent.add_child(create_test_hierarchy(EntityType::Team, dec!(40.0)));

        assert_eq!(parent.total_children_quota(), dec!(70.0));
    }

    #[test]
    fn test_hierarchy_has_capacity_for_child() {
        let mut parent = create_test_hierarchy(EntityType::Organization, dec!(100.0));
        parent.add_child(create_test_hierarchy(EntityType::Team, dec!(60.0)));

        assert!(parent.has_capacity_for_child(dec!(30.0)));
        assert!(!parent.has_capacity_for_child(dec!(50.0)));
    }

    #[test]
    fn test_hierarchy_available_for_allocation() {
        let mut hierarchy = create_test_hierarchy(EntityType::User, dec!(100.0));
        hierarchy.current_usage = dec!(40.0);

        assert_eq!(hierarchy.available_for_allocation(), dec!(60.0));
    }

    #[test]
    fn test_hierarchy_can_allocate() {
        let mut hierarchy = create_test_hierarchy(EntityType::User, dec!(100.0));
        hierarchy.current_usage = dec!(70.0);

        assert!(hierarchy.can_allocate(dec!(20.0)));
        assert!(!hierarchy.can_allocate(dec!(40.0)));
    }

    #[test]
    fn test_usage_stats_from_quota() {
        let quota = create_test_quota(dec!(100.0));
        let stats = QuotaUsageStats::from_quota(&quota, dec!(60.0));

        assert_eq!(stats.limit, dec!(100.0));
        assert_eq!(stats.effective_limit, dec!(100.0));
        assert_eq!(stats.usage, dec!(60.0));
        assert_eq!(stats.available, dec!(40.0));
        assert_eq!(stats.utilization_percent, 60.0);
    }

    #[test]
    fn test_usage_stats_with_overcommit() {
        let mut quota = create_test_quota(dec!(100.0));
        quota.overcommit_ratio = dec!(1.5);
        let stats = QuotaUsageStats::from_quota(&quota, dec!(120.0));

        assert_eq!(stats.limit, dec!(100.0));
        assert_eq!(stats.effective_limit, dec!(150.0));
        assert_eq!(stats.usage, dec!(120.0));
        assert_eq!(stats.available, dec!(30.0));
        assert_eq!(stats.utilization_percent, 80.0);
    }

    #[test]
    fn test_usage_stats_is_at_soft_limit() {
        let quota = create_test_quota(dec!(100.0));
        let stats = QuotaUsageStats::from_quota(&quota, dec!(85.0));

        assert!(stats.is_at_soft_limit(Some(dec!(80.0))));
        assert!(!stats.is_at_soft_limit(Some(dec!(90.0))));
    }

    #[test]
    fn test_usage_stats_is_at_hard_limit() {
        let quota = create_test_quota(dec!(100.0));
        let stats_below = QuotaUsageStats::from_quota(&quota, dec!(90.0));
        let stats_at = QuotaUsageStats::from_quota(&quota, dec!(100.0));
        let stats_above = QuotaUsageStats::from_quota(&quota, dec!(110.0));

        assert!(!stats_below.is_at_hard_limit());
        assert!(stats_at.is_at_hard_limit());
        assert!(stats_above.is_at_hard_limit());
    }

    fn create_test_hierarchy(entity_type: EntityType, limit: Decimal) -> QuotaHierarchy {
        QuotaHierarchy::new(create_test_quota_with_type(entity_type, limit))
    }

    fn create_test_quota(limit: Decimal) -> Quota {
        create_test_quota_with_type(EntityType::User, limit)
    }

    fn create_test_quota_with_type(entity_type: EntityType, limit: Decimal) -> Quota {
        Quota {
            id: Uuid::new_v4(),
            entity_type,
            entity_id: "test-entity".to_string(),
            parent_id: None,
            resource_type: ResourceType::GpuHours,
            limit_value: limit,
            soft_limit: None,
            burst_limit: None,
            overcommit_ratio: dec!(1.0),
            created_at: Utc::now(),
            updated_at: Utc::now(),
        }
    }
}
