// Re-export ResourceX types for use in scheduler
pub use hpc_resources::{
    allocation::{ResourceAllocation as ResourceXAllocation, ResourceAssignment},
    request::{RequestPriority, ResourceRequest as ResourceXRequest},
    types::*,
};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;
use uuid::Uuid;

/// Scheduler-specific wrapper for ResourceRequest with OpenAPI schema support
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema, PartialEq)]
pub struct ResourceRequest {
    #[serde(flatten)]
    pub inner: ResourceXRequest,
}

impl ResourceRequest {
    /// Create a new resource request
    pub fn new() -> Self {
        Self {
            inner: ResourceXRequest::new(),
        }
    }

    /// Helper: Create a GPU-only request for backward compatibility
    pub fn gpu_only(vendor: GpuVendor, model: impl Into<String>, count: f64) -> Self {
        Self {
            inner: ResourceXRequest::new().add_gpu(vendor, model, count),
        }
    }

    /// Helper: Add GPU with vendor and model
    pub fn add_gpu(mut self, vendor: GpuVendor, model: impl Into<String>, count: f64) -> Self {
        self.inner = self.inner.add_gpu(vendor, model, count);
        self
    }

    /// Helper: Add CPU cores
    pub fn add_cpu_cores(mut self, cores: f64) -> Self {
        self.inner = self.inner.add_cpu_cores(cores);
        self
    }

    /// Helper: Add memory in GB
    pub fn add_memory_gb(mut self, gb: f64) -> Self {
        self.inner = self.inner.add_memory_gb(gb);
        self
    }

    /// Helper: Add storage
    pub fn add_storage(mut self, storage_type: StorageType, gb: f64) -> Self {
        self.inner = self.inner.add_storage(storage_type, gb);
        self
    }

    /// Helper: Add TPU
    pub fn add_tpu(mut self, variant: TpuVariant, count: f64) -> Self {
        self.inner = self.inner.add_tpu(variant, count);
        self
    }

    /// Set priority
    pub fn with_priority(mut self, priority: RequestPriority) -> Self {
        self.inner = self.inner.with_priority(priority);
        self
    }

    /// Set requester ID
    pub fn with_requester(mut self, requester_id: impl Into<String>) -> Self {
        self.inner = self.inner.with_requester(requester_id);
        self
    }

    /// Validate the request
    pub fn validate(&self) -> Result<(), String> {
        self.inner.validate()
    }

    /// Check if request has GPU resources
    pub fn has_gpu(&self) -> bool {
        self.inner.has_gpu()
    }

    /// Check if request has CPU resources
    pub fn has_cpu(&self) -> bool {
        self.inner.has_cpu()
    }

    /// Get GPU specification if present
    pub fn get_gpu_spec(&self) -> Option<&ResourceSpec> {
        self.inner.get_gpu_spec()
    }
}

impl Default for ResourceRequest {
    fn default() -> Self {
        Self::new()
    }
}

/// Scheduler-specific resource allocation with node tracking
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema, PartialEq)]
pub struct ResourceAllocation {
    /// Core resource allocation
    #[serde(flatten)]
    pub inner: ResourceXAllocation,

    /// Node IDs where resources are allocated
    pub node_ids: Vec<Uuid>,
}

impl ResourceAllocation {
    pub fn new(request_id: Uuid) -> Self {
        Self {
            inner: ResourceXAllocation::new(request_id),
            node_ids: Vec::new(),
        }
    }

    pub fn add_assignment(
        mut self,
        resource_type: ResourceType,
        assignment: ResourceAssignment,
    ) -> Self {
        self.inner = self.inner.add_assignment(resource_type, assignment);
        self
    }

    pub fn add_node(mut self, node_id: Uuid) -> Self {
        if !self.node_ids.contains(&node_id) {
            self.node_ids.push(node_id);
        }
        self
    }

    pub fn get_assignments(&self, resource_type: &ResourceType) -> Option<&Vec<ResourceAssignment>> {
        self.inner.get_assignments(resource_type)
    }

    pub fn total_amount(&self, resource_type: &ResourceType) -> f64 {
        self.inner.total_amount(resource_type)
    }

    pub fn asset_ids(&self) -> Vec<Uuid> {
        self.inner.asset_ids()
    }
}

impl Default for ResourceAllocation {
    fn default() -> Self {
        Self::new(Uuid::new_v4())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resource_request_builder() {
        let req = ResourceRequest::new()
            .add_gpu(GpuVendor::Nvidia, "H100", 4.0)
            .add_cpu_cores(32.0)
            .add_memory_gb(256.0);

        assert!(req.has_gpu());
        assert!(req.has_cpu());
        assert!(req.validate().is_ok());
    }

    #[test]
    fn test_resource_request_gpu_only() {
        let req = ResourceRequest::gpu_only(GpuVendor::Nvidia, "H100", 2.0);
        assert!(req.has_gpu());
        assert!(!req.has_cpu());
    }

    #[test]
    fn test_resource_request_minimal() {
        let req = ResourceRequest::new().add_cpu_cores(16.0);
        assert!(!req.has_gpu());
        assert!(req.has_cpu());
        assert!(req.validate().is_ok());
    }

    #[test]
    fn test_resource_request_validation() {
        let req = ResourceRequest::new();
        assert!(req.validate().is_err()); // Empty request should fail
    }

    #[test]
    fn test_resource_allocation_new() {
        let request_id = Uuid::new_v4();
        let alloc = ResourceAllocation::new(request_id);
        assert_eq!(alloc.inner.request_id, request_id);
        assert!(alloc.node_ids.is_empty());
    }

    #[test]
    fn test_resource_allocation_with_nodes() {
        let node1 = Uuid::new_v4();
        let node2 = Uuid::new_v4();
        let gpu_id = Uuid::new_v4();

        let alloc = ResourceAllocation::new(Uuid::new_v4())
            .add_node(node1)
            .add_node(node2)
            .add_assignment(
                ResourceType::Compute(ComputeType::Gpu),
                ResourceAssignment::new(gpu_id, 1.0, ResourceUnit::Count),
            );

        assert_eq!(alloc.node_ids.len(), 2);
        assert!(alloc.node_ids.contains(&node1));
        assert!(alloc.node_ids.contains(&node2));
        assert_eq!(
            alloc.total_amount(&ResourceType::Compute(ComputeType::Gpu)),
            1.0
        );
    }

    #[test]
    fn test_multi_resource_request() {
        let req = ResourceRequest::new()
            .add_gpu(GpuVendor::Amd, "MI300X", 8.0)
            .add_cpu_cores(128.0)
            .add_memory_gb(1024.0)
            .add_storage(StorageType::Nvme, 2048.0)
            .with_priority(RequestPriority::High);

        assert!(req.has_gpu());
        assert!(req.has_cpu());
        assert!(req.validate().is_ok());
    }

    #[test]
    fn test_tpu_request() {
        let req = ResourceRequest::new()
            .add_tpu(TpuVariant::GoogleV5p, 8.0)
            .add_memory_gb(512.0);

        assert!(req.inner.has_tpu());
        assert!(!req.has_gpu());
        assert!(req.validate().is_ok());
    }
}
