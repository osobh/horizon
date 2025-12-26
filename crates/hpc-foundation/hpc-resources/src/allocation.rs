use crate::types::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Represents an actual allocation of specific resource instances
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResourceAllocation {
    pub id: Uuid,
    pub request_id: Uuid,
    pub assignments: HashMap<ResourceType, Vec<ResourceAssignment>>,
    pub allocated_at: chrono::DateTime<chrono::Utc>,
}

/// A specific resource instance assigned to a request
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResourceAssignment {
    pub asset_id: Uuid,
    pub amount: f64,
    pub unit: ResourceUnit,
    pub metadata: Option<serde_json::Value>,
}

impl ResourceAllocation {
    pub fn new(request_id: Uuid) -> Self {
        Self {
            id: Uuid::new_v4(),
            request_id,
            assignments: HashMap::new(),
            allocated_at: chrono::Utc::now(),
        }
    }

    pub fn add_assignment(
        mut self,
        resource_type: ResourceType,
        assignment: ResourceAssignment,
    ) -> Self {
        self.assignments
            .entry(resource_type)
            .or_insert_with(Vec::new)
            .push(assignment);
        self
    }

    pub fn get_assignments(&self, resource_type: &ResourceType) -> Option<&Vec<ResourceAssignment>> {
        self.assignments.get(resource_type)
    }

    pub fn total_amount(&self, resource_type: &ResourceType) -> f64 {
        self.assignments
            .get(resource_type)
            .map(|assignments| assignments.iter().map(|a| a.amount).sum())
            .unwrap_or(0.0)
    }

    pub fn asset_ids(&self) -> Vec<Uuid> {
        self.assignments
            .values()
            .flat_map(|assignments| assignments.iter().map(|a| a.asset_id))
            .collect()
    }
}

impl ResourceAssignment {
    pub fn new(asset_id: Uuid, amount: f64, unit: ResourceUnit) -> Self {
        Self {
            asset_id,
            amount,
            unit,
            metadata: None,
        }
    }

    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = Some(metadata);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resource_allocation_new() {
        let request_id = Uuid::new_v4();
        let allocation = ResourceAllocation::new(request_id);

        assert_eq!(allocation.request_id, request_id);
        assert!(allocation.assignments.is_empty());
    }

    #[test]
    fn test_add_gpu_assignment() {
        let request_id = Uuid::new_v4();
        let gpu_id_1 = Uuid::new_v4();
        let gpu_id_2 = Uuid::new_v4();

        let allocation = ResourceAllocation::new(request_id)
            .add_assignment(
                ResourceType::Compute(ComputeType::Gpu),
                ResourceAssignment::new(gpu_id_1, 1.0, ResourceUnit::Count),
            )
            .add_assignment(
                ResourceType::Compute(ComputeType::Gpu),
                ResourceAssignment::new(gpu_id_2, 1.0, ResourceUnit::Count),
            );

        let gpu_assignments = allocation
            .get_assignments(&ResourceType::Compute(ComputeType::Gpu))
            .unwrap();

        assert_eq!(gpu_assignments.len(), 2);
        assert_eq!(gpu_assignments[0].asset_id, gpu_id_1);
        assert_eq!(gpu_assignments[1].asset_id, gpu_id_2);
    }

    #[test]
    fn test_total_amount() {
        let request_id = Uuid::new_v4();
        let allocation = ResourceAllocation::new(request_id)
            .add_assignment(
                ResourceType::Memory,
                ResourceAssignment::new(Uuid::new_v4(), 128.0, ResourceUnit::Gigabytes),
            )
            .add_assignment(
                ResourceType::Memory,
                ResourceAssignment::new(Uuid::new_v4(), 128.0, ResourceUnit::Gigabytes),
            );

        let total = allocation.total_amount(&ResourceType::Memory);
        assert_eq!(total, 256.0);
    }

    #[test]
    fn test_asset_ids() {
        let request_id = Uuid::new_v4();
        let asset_1 = Uuid::new_v4();
        let asset_2 = Uuid::new_v4();
        let asset_3 = Uuid::new_v4();

        let allocation = ResourceAllocation::new(request_id)
            .add_assignment(
                ResourceType::Compute(ComputeType::Gpu),
                ResourceAssignment::new(asset_1, 1.0, ResourceUnit::Count),
            )
            .add_assignment(
                ResourceType::Compute(ComputeType::Cpu),
                ResourceAssignment::new(asset_2, 64.0, ResourceUnit::Cores),
            )
            .add_assignment(
                ResourceType::Memory,
                ResourceAssignment::new(asset_3, 256.0, ResourceUnit::Gigabytes),
            );

        let asset_ids = allocation.asset_ids();
        assert_eq!(asset_ids.len(), 3);
        assert!(asset_ids.contains(&asset_1));
        assert!(asset_ids.contains(&asset_2));
        assert!(asset_ids.contains(&asset_3));
    }

    #[test]
    fn test_assignment_with_metadata() {
        let metadata = serde_json::json!({
            "vendor": "nvidia",
            "model": "H100",
            "pcie_gen": 5
        });

        let assignment = ResourceAssignment::new(Uuid::new_v4(), 1.0, ResourceUnit::Count)
            .with_metadata(metadata.clone());

        assert!(assignment.metadata.is_some());
        assert_eq!(assignment.metadata.unwrap(), metadata);
    }

    #[test]
    fn test_serialization() {
        // Test individual components can be serialized
        let assignment = ResourceAssignment::new(Uuid::new_v4(), 4.0, ResourceUnit::Count);
        let json = serde_json::to_string(&assignment).unwrap();
        let deserialized: ResourceAssignment = serde_json::from_str(&json).unwrap();

        assert_eq!(assignment.amount, deserialized.amount);
        assert_eq!(assignment.unit, deserialized.unit);
    }
}
