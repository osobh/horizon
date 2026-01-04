use rust_decimal::Decimal;
use uuid::Uuid;

use crate::{
    db::QuotaRepository,
    error::{HpcError, QuotaErrorExt, Result},
    models::*,
};

#[derive(Clone)]
pub struct QuotaService {
    repository: QuotaRepository,
}

impl QuotaService {
    pub fn new(repository: QuotaRepository) -> Self {
        Self { repository }
    }

    pub async fn create_quota(&self, req: CreateQuotaRequest) -> Result<Quota> {
        // Validate parent exists if specified
        if let Some(parent_id) = req.parent_id {
            let parent = self.repository.get_quota(parent_id).await?;

            // Validate entity type hierarchy
            if req.entity_type.parent_type() != Some(parent.entity_type) {
                return Err(HpcError::invalid_hierarchy(format!(
                    "Entity type {:?} cannot have parent {:?}",
                    req.entity_type, parent.entity_type
                )));
            }

            // Validate resource types match
            if req.resource_type != parent.resource_type {
                return Err(HpcError::invalid_hierarchy(
                    "Resource type must match parent",
                ));
            }

            // Validate quota doesn't exceed parent
            if req.limit_value > parent.limit_value {
                return Err(HpcError::invalid_configuration(format!(
                    "Quota limit {} exceeds parent limit {}",
                    req.limit_value, parent.limit_value
                )));
            }

            // Check if parent has capacity for this child
            let siblings = self.repository.list_quotas(Some(req.entity_type)).await?;
            let siblings_total: Decimal = siblings
                .iter()
                .filter(|q| q.parent_id == Some(parent_id))
                .map(|q| q.limit_value)
                .sum();

            if siblings_total + req.limit_value > parent.effective_limit() {
                return Err(HpcError::invalid_configuration(format!(
                    "Parent quota has insufficient capacity. Current: {}, Requested: {}, Parent limit: {}",
                    siblings_total, req.limit_value, parent.effective_limit()
                )));
            }
        }

        self.repository.create_quota(req).await
    }

    pub async fn get_quota(&self, id: Uuid) -> Result<Quota> {
        self.repository.get_quota(id).await
    }

    pub async fn get_quota_by_entity(
        &self,
        entity_type: EntityType,
        entity_id: &str,
        resource_type: ResourceType,
    ) -> Result<Quota> {
        self.repository
            .get_quota_by_entity(entity_type, entity_id, resource_type)
            .await
    }

    pub async fn list_quotas(&self, entity_type: Option<EntityType>) -> Result<Vec<Quota>> {
        self.repository.list_quotas(entity_type).await
    }

    pub async fn update_quota(&self, id: Uuid, req: UpdateQuotaRequest) -> Result<Quota> {
        let current = self.repository.get_quota(id).await?;

        // If updating limit, validate against parent
        if let Some(new_limit) = req.limit_value {
            if let Some(parent_id) = current.parent_id {
                let parent = self.repository.get_quota(parent_id).await?;
                if new_limit > parent.limit_value {
                    return Err(HpcError::invalid_configuration(
                        "New limit exceeds parent limit",
                    ));
                }
            }

            // Validate not below current usage
            let current_usage = self.repository.get_current_usage(id).await?;
            if new_limit < current_usage {
                return Err(HpcError::invalid_configuration(format!(
                    "Cannot set limit {} below current usage {}",
                    new_limit, current_usage
                )));
            }
        }

        self.repository.update_quota(id, req).await
    }

    pub async fn delete_quota(&self, id: Uuid) -> Result<()> {
        // Check if there are active allocations
        let allocations = self.repository.list_active_allocations(id).await?;
        if !allocations.is_empty() {
            return Err(HpcError::invalid_configuration(format!(
                "Cannot delete quota with {} active allocations",
                allocations.len()
            )));
        }

        // Check if there are child quotas
        let all_quotas = self.repository.list_quotas(None).await?;
        let has_children = all_quotas.iter().any(|q| q.parent_id == Some(id));
        if has_children {
            return Err(HpcError::invalid_configuration(
                "Cannot delete quota with child quotas",
            ));
        }

        self.repository.delete_quota(id).await
    }

    pub async fn get_usage_stats(&self, id: Uuid) -> Result<QuotaUsageStats> {
        let quota = self.repository.get_quota(id).await?;
        let usage = self.repository.get_current_usage(id).await?;
        Ok(QuotaUsageStats::from_quota(&quota, usage))
    }

    pub async fn get_usage_history(
        &self,
        id: Uuid,
        limit: Option<i64>,
    ) -> Result<Vec<UsageHistory>> {
        self.repository.get_usage_history(id, limit).await
    }

    pub fn build_hierarchy<'a>(
        &'a self,
        quota_id: Uuid,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<QuotaHierarchy>> + Send + 'a>>
    {
        Box::pin(async move {
            let quota = self.repository.get_quota(quota_id).await?;
            let current_usage = self.repository.get_current_usage(quota_id).await?;

            let mut hierarchy = QuotaHierarchy::new(quota.clone());
            hierarchy.current_usage = current_usage;

            // Load parent if exists
            if let Some(parent_id) = quota.parent_id {
                let parent_hierarchy = self.build_hierarchy(parent_id).await?;
                hierarchy = hierarchy.with_parent(parent_hierarchy);
            }

            // Load children
            let all_quotas = self.repository.list_quotas(Some(quota.entity_type)).await?;
            for child_quota in all_quotas {
                if child_quota.parent_id == Some(quota_id) {
                    let child_hierarchy = self.build_hierarchy(child_quota.id).await?;
                    hierarchy.add_child(child_hierarchy);
                }
            }

            Ok(hierarchy)
        })
    }
}

#[cfg(test)]
mod tests {
    // Service tests are done via integration tests with real database
    // This keeps unit tests fast and focused
}
