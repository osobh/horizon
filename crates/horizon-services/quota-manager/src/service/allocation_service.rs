use rust_decimal::Decimal;
use uuid::Uuid;

use crate::{
    db::QuotaRepository,
    error::{HpcError, QuotaErrorExt, Result},
    models::*,
};

#[derive(Clone)]
pub struct AllocationService {
    repository: QuotaRepository,
}

impl AllocationService {
    pub fn new(repository: QuotaRepository) -> Self {
        Self { repository }
    }

    /// Check if an allocation request can be satisfied
    pub async fn check_allocation(
        &self,
        entity_type: EntityType,
        entity_id: &str,
        resource_type: ResourceType,
        requested_value: Decimal,
    ) -> Result<AllocationCheckResponse> {
        // Get the quota for this entity
        let quota = self
            .repository
            .get_quota_by_entity(entity_type, entity_id, resource_type)
            .await?;

        // Get current usage
        let current_usage = self.repository.get_current_usage(quota.id).await?;

        // Calculate available
        let available = quota.available_quota(current_usage);

        // Check if request can be satisfied
        let allowed = requested_value <= available;

        let reason = if !allowed {
            Some(format!(
                "Insufficient quota: requested {}, available {}, current usage {}, limit {}",
                requested_value,
                available,
                current_usage,
                quota.effective_limit()
            ))
        } else if quota.is_soft_limit_exceeded(current_usage + requested_value) {
            Some(format!(
                "Warning: allocation would exceed soft limit of {}",
                quota.soft_limit.unwrap()
            ))
        } else {
            None
        };

        Ok(AllocationCheckResponse {
            allowed,
            available,
            requested: requested_value,
            reason,
        })
    }

    /// Allocate quota for a job (with hierarchical enforcement)
    pub async fn allocate(
        &self,
        entity_type: EntityType,
        entity_id: &str,
        job_id: Uuid,
        resource_type: ResourceType,
        requested_value: Decimal,
        metadata: Option<serde_json::Value>,
    ) -> Result<Allocation> {
        if requested_value <= Decimal::ZERO {
            return Err(HpcError::invalid_configuration(
                "Allocation value must be positive",
            ));
        }

        // Get the quota
        let quota = self
            .repository
            .get_quota_by_entity(entity_type, entity_id, resource_type)
            .await?;

        // Check current usage
        let current_usage = self.repository.get_current_usage(quota.id).await?;
        let available = quota.available_quota(current_usage);

        if requested_value > available {
            return Err(HpcError::quota_exceeded(format!(
                "Insufficient quota: requested {}, available {}, current usage {}, limit {}",
                requested_value,
                available,
                current_usage,
                quota.effective_limit()
            )));
        }

        // Check hierarchical quotas (parent quotas must also have capacity)
        if let Some(parent_id) = quota.parent_id {
            self.check_parent_quota_recursive(parent_id, requested_value)
                .await?;
        }

        // Create allocation
        let req = CreateAllocationRequest {
            quota_id: quota.id,
            job_id,
            resource_type,
            allocated_value: requested_value,
            metadata,
        };

        let allocation = self.repository.create_allocation(req).await?;

        // Record in usage history
        self.repository
            .record_usage(
                quota.id,
                entity_type,
                entity_id,
                resource_type,
                requested_value,
                OperationType::Allocate,
                Some(job_id),
                None,
            )
            .await?;

        Ok(allocation)
    }

    /// Release an allocation
    pub async fn release(&self, allocation_id: Uuid) -> Result<Allocation> {
        let allocation = self.repository.get_allocation(allocation_id).await?;

        if !allocation.is_active() {
            return Err(HpcError::invalid_configuration(
                "Allocation already released",
            ));
        }

        // Release with optimistic locking
        let released = self.repository.release_allocation(allocation_id).await?;

        // Get quota info for history
        let quota = self.repository.get_quota(allocation.quota_id).await?;

        // Record in usage history
        self.repository
            .record_usage(
                quota.id,
                quota.entity_type,
                &quota.entity_id,
                allocation.resource_type,
                allocation.allocated_value,
                OperationType::Release,
                Some(allocation.job_id),
                None,
            )
            .await?;

        Ok(released)
    }

    /// Get allocation by ID
    pub async fn get_allocation(&self, id: Uuid) -> Result<Allocation> {
        self.repository.get_allocation(id).await
    }

    /// List active allocations for a quota
    pub async fn list_active_allocations(&self, quota_id: Uuid) -> Result<Vec<Allocation>> {
        self.repository.list_active_allocations(quota_id).await
    }

    /// Check parent quota recursively
    fn check_parent_quota_recursive<'a>(
        &'a self,
        parent_id: Uuid,
        requested_value: Decimal,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            let parent = self.repository.get_quota(parent_id).await?;
            let parent_usage = self.repository.get_current_usage(parent_id).await?;
            let parent_available = parent.available_quota(parent_usage);

            if requested_value > parent_available {
                return Err(HpcError::quota_exceeded(format!(
                    "Parent quota {} has insufficient capacity: requested {}, available {}",
                    parent_id, requested_value, parent_available
                )));
            }

            // Check grandparent if exists
            if let Some(grandparent_id) = parent.parent_id {
                self.check_parent_quota_recursive(grandparent_id, requested_value)
                    .await?;
            }

            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    // Service tests are done via integration tests with real database
    // This keeps unit tests fast and focused
}
