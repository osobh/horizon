//! Resource management for disaster recovery

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Recovery resource requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryResource {
    /// Resource ID
    pub id: Uuid,
    /// Resource type
    pub resource_type: ResourceType,
    /// Required capacity
    pub required_capacity: ResourceCapacity,
    /// Priority for allocation
    pub priority: u32,
    /// Duration needed
    pub duration_minutes: u64,
}

/// Resource type categories
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ResourceType {
    /// Compute resources
    Compute,
    /// Memory resources
    Memory,
    /// Storage resources
    Storage,
    /// Network bandwidth
    Network,
    /// Database connections
    Database,
    /// External API quotas
    Api,
    /// Human resources
    Human,
}

/// Resource capacity specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceCapacity {
    /// CPU cores required
    pub cpu_cores: f64,
    /// Memory in GB
    pub memory_gb: f64,
    /// Storage in GB
    pub storage_gb: f64,
    /// Network bandwidth in Mbps
    pub network_mbps: f64,
    /// Database connection count
    pub database_connections: u32,
    /// API requests per minute
    pub api_requests_per_minute: u32,
    /// Human resource count
    pub human_resources: u32,
}

impl ResourceCapacity {
    /// Create new resource capacity
    pub fn new() -> Self {
        Self {
            cpu_cores: 0.0,
            memory_gb: 0.0,
            storage_gb: 0.0,
            network_mbps: 0.0,
            database_connections: 0,
            api_requests_per_minute: 0,
            human_resources: 0,
        }
    }

    /// Check if this capacity can satisfy the required capacity
    pub fn can_satisfy(&self, required: &ResourceCapacity) -> bool {
        self.cpu_cores >= required.cpu_cores
            && self.memory_gb >= required.memory_gb
            && self.storage_gb >= required.storage_gb
            && self.network_mbps >= required.network_mbps
            && self.database_connections >= required.database_connections
            && self.api_requests_per_minute >= required.api_requests_per_minute
            && self.human_resources >= required.human_resources
    }

    /// Subtract required capacity from available capacity
    pub fn allocate(&mut self, required: &ResourceCapacity) -> Result<(), String> {
        if !self.can_satisfy(required) {
            return Err("Insufficient resources available".to_string());
        }

        self.cpu_cores -= required.cpu_cores;
        self.memory_gb -= required.memory_gb;
        self.storage_gb -= required.storage_gb;
        self.network_mbps -= required.network_mbps;
        self.database_connections -= required.database_connections;
        self.api_requests_per_minute -= required.api_requests_per_minute;
        self.human_resources -= required.human_resources;

        Ok(())
    }

    /// Add capacity back to the pool
    pub fn deallocate(&mut self, capacity: &ResourceCapacity) {
        self.cpu_cores += capacity.cpu_cores;
        self.memory_gb += capacity.memory_gb;
        self.storage_gb += capacity.storage_gb;
        self.network_mbps += capacity.network_mbps;
        self.database_connections += capacity.database_connections;
        self.api_requests_per_minute += capacity.api_requests_per_minute;
        self.human_resources += capacity.human_resources;
    }

    /// Calculate utilization percentage (0.0 to 1.0)
    pub fn utilization_ratio(&self, total: &ResourceCapacity) -> f64 {
        if total.cpu_cores == 0.0 {
            return 0.0;
        }

        let cpu_util = (total.cpu_cores - self.cpu_cores) / total.cpu_cores;
        let mem_util = if total.memory_gb > 0.0 {
            (total.memory_gb - self.memory_gb) / total.memory_gb
        } else {
            0.0
        };

        // Return highest utilization as bottleneck indicator
        cpu_util.max(mem_util)
    }
}

impl Default for ResourceCapacity {
    fn default() -> Self {
        Self::new()
    }
}

/// Resource pool manager
pub struct ResourcePool {
    /// Available resources by type
    pub pools: HashMap<ResourceType, ResourceCapacity>,
    /// Allocated resources tracking
    pub allocations: HashMap<Uuid, (ResourceType, ResourceCapacity)>,
}

impl ResourcePool {
    /// Create new resource pool
    pub fn new() -> Self {
        Self {
            pools: HashMap::new(),
            allocations: HashMap::new(),
        }
    }

    /// Add resources to pool
    pub fn add_resources(&mut self, resource_type: ResourceType, capacity: ResourceCapacity) {
        self.pools
            .entry(resource_type)
            .and_modify(|existing| existing.deallocate(&capacity))
            .or_insert(capacity);
    }

    /// Try to allocate resources
    pub fn allocate(
        &mut self,
        allocation_id: Uuid,
        resource_type: ResourceType,
        required: ResourceCapacity,
    ) -> Result<(), String> {
        if let Some(pool) = self.pools.get_mut(&resource_type) {
            pool.allocate(&required)?;
            self.allocations
                .insert(allocation_id, (resource_type, required));
            Ok(())
        } else {
            Err(format!("No resource pool for type {:?}", resource_type))
        }
    }

    /// Deallocate resources
    pub fn deallocate(&mut self, allocation_id: Uuid) -> Result<(), String> {
        if let Some((resource_type, capacity)) = self.allocations.remove(&allocation_id) {
            if let Some(pool) = self.pools.get_mut(&resource_type) {
                pool.deallocate(&capacity);
                Ok(())
            } else {
                Err(format!(
                    "Resource pool not found for type {:?}",
                    resource_type
                ))
            }
        } else {
            Err("Allocation not found".to_string())
        }
    }

    /// Get available capacity for resource type
    pub fn available_capacity(&self, resource_type: &ResourceType) -> Option<&ResourceCapacity> {
        self.pools.get(resource_type)
    }
}

impl Default for ResourcePool {
    fn default() -> Self {
        Self::new()
    }
}
