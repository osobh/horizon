use crate::types::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Priority levels for resource requests
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RequestPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Urgent = 3,
}

/// A request for multiple resource types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResourceRequest {
    pub id: Uuid,
    pub resources: HashMap<ResourceType, ResourceSpec>,
    pub priority: RequestPriority,
    pub requester_id: Option<String>,
}

impl ResourceRequest {
    pub fn new() -> Self {
        Self {
            id: Uuid::new_v4(),
            resources: HashMap::new(),
            priority: RequestPriority::Normal,
            requester_id: None,
        }
    }

    pub fn with_id(mut self, id: Uuid) -> Self {
        self.id = id;
        self
    }

    pub fn with_priority(mut self, priority: RequestPriority) -> Self {
        self.priority = priority;
        self
    }

    pub fn with_requester(mut self, requester_id: impl Into<String>) -> Self {
        self.requester_id = Some(requester_id.into());
        self
    }

    /// Add a resource specification
    pub fn add_resource(mut self, resource_type: ResourceType, spec: ResourceSpec) -> Self {
        self.resources.insert(resource_type, spec);
        self
    }

    /// Helper: Add CPU cores
    pub fn add_cpu_cores(self, cores: f64) -> Self {
        self.add_resource(
            ResourceType::Compute(ComputeType::Cpu),
            ResourceSpec::new(cores, ResourceUnit::Cores),
        )
    }

    /// Helper: Add GPU with vendor and model
    pub fn add_gpu(self, vendor: GpuVendor, model: impl Into<String>, count: f64) -> Self {
        self.add_resource(
            ResourceType::Compute(ComputeType::Gpu),
            ResourceSpec::new(count, ResourceUnit::Count)
                .with_vendor(vendor)
                .with_model(model),
        )
    }

    /// Helper: Add GPU with full constraints
    pub fn add_gpu_with_constraints(self, count: f64, constraints: ResourceConstraints) -> Self {
        self.add_resource(
            ResourceType::Compute(ComputeType::Gpu),
            ResourceSpec::new(count, ResourceUnit::Count).with_constraints(constraints),
        )
    }

    /// Helper: Add TPU
    pub fn add_tpu(self, variant: TpuVariant, count: f64) -> Self {
        let mut constraints = ResourceConstraints::default();
        constraints.tpu_variant = Some(variant);

        self.add_resource(
            ResourceType::Compute(ComputeType::Tpu),
            ResourceSpec::new(count, ResourceUnit::Count).with_constraints(constraints),
        )
    }

    /// Helper: Add memory in GB
    pub fn add_memory_gb(self, gb: f64) -> Self {
        self.add_resource(
            ResourceType::Memory,
            ResourceSpec::new(gb, ResourceUnit::Gigabytes),
        )
    }

    /// Helper: Add storage
    pub fn add_storage(self, storage_type: StorageType, gb: f64) -> Self {
        self.add_resource(
            ResourceType::Storage(storage_type),
            ResourceSpec::new(gb, ResourceUnit::Gigabytes),
        )
    }

    /// Helper: Add network bandwidth
    pub fn add_network_bandwidth(self, mbps: f64) -> Self {
        self.add_resource(
            ResourceType::Network(NetworkType::Bandwidth),
            ResourceSpec::new(mbps, ResourceUnit::MegabitsPerSecond),
        )
    }

    /// Helper: Add custom resource
    pub fn add_custom(self, name: impl Into<String>, amount: f64) -> Self {
        self.add_resource(
            ResourceType::Custom(name.into()),
            ResourceSpec::new(amount, ResourceUnit::Count),
        )
    }

    /// Check if request has any GPU resources
    pub fn has_gpu(&self) -> bool {
        self.resources
            .keys()
            .any(|rt| matches!(rt, ResourceType::Compute(ComputeType::Gpu)))
    }

    /// Check if request has any TPU resources
    pub fn has_tpu(&self) -> bool {
        self.resources
            .keys()
            .any(|rt| matches!(rt, ResourceType::Compute(ComputeType::Tpu)))
    }

    /// Check if request has any CPU resources
    pub fn has_cpu(&self) -> bool {
        self.resources
            .keys()
            .any(|rt| matches!(rt, ResourceType::Compute(ComputeType::Cpu)))
    }

    /// Get GPU spec if present
    pub fn get_gpu_spec(&self) -> Option<&ResourceSpec> {
        self.resources.get(&ResourceType::Compute(ComputeType::Gpu))
    }

    /// Get TPU spec if present
    pub fn get_tpu_spec(&self) -> Option<&ResourceSpec> {
        self.resources.get(&ResourceType::Compute(ComputeType::Tpu))
    }

    /// Get CPU spec if present
    pub fn get_cpu_spec(&self) -> Option<&ResourceSpec> {
        self.resources.get(&ResourceType::Compute(ComputeType::Cpu))
    }

    /// Validate that request has at least one resource
    pub fn validate(&self) -> Result<(), String> {
        if self.resources.is_empty() {
            return Err(
                "ResourceRequest must have at least one resource specification".to_string(),
            );
        }
        Ok(())
    }
}

impl Default for ResourceRequest {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resource_request_new() {
        let request = ResourceRequest::new();
        assert!(request.resources.is_empty());
        assert_eq!(request.priority, RequestPriority::Normal);
        assert!(request.requester_id.is_none());
    }

    #[test]
    fn test_add_cpu_cores() {
        let request = ResourceRequest::new().add_cpu_cores(64.0);

        assert_eq!(request.resources.len(), 1);
        assert!(request.has_cpu());
        assert!(!request.has_gpu());

        let cpu_spec = request.get_cpu_spec().unwrap();
        assert_eq!(cpu_spec.amount, 64.0);
        assert_eq!(cpu_spec.unit, ResourceUnit::Cores);
    }

    #[test]
    fn test_add_gpu_nvidia_h100() {
        let request = ResourceRequest::new().add_gpu(GpuVendor::Nvidia, "H100", 4.0);

        assert!(request.has_gpu());
        let gpu_spec = request.get_gpu_spec().unwrap();
        assert_eq!(gpu_spec.amount, 4.0);

        let constraints = gpu_spec.constraints.as_ref().unwrap();
        assert_eq!(constraints.vendor, Some(GpuVendor::Nvidia));
        assert_eq!(constraints.model, Some("H100".to_string()));
    }

    #[test]
    fn test_add_gpu_amd_mi300x() {
        let request = ResourceRequest::new().add_gpu(GpuVendor::Amd, "MI300X", 8.0);

        let gpu_spec = request.get_gpu_spec().unwrap();
        assert_eq!(gpu_spec.amount, 8.0);

        let constraints = gpu_spec.constraints.as_ref().unwrap();
        assert_eq!(constraints.vendor, Some(GpuVendor::Amd));
        assert_eq!(constraints.model, Some("MI300X".to_string()));
    }

    #[test]
    fn test_add_tpu() {
        let request = ResourceRequest::new().add_tpu(TpuVariant::GoogleV5e, 8.0);

        assert!(request.has_tpu());
        let tpu_spec = request.get_tpu_spec().unwrap();
        assert_eq!(tpu_spec.amount, 8.0);

        let constraints = tpu_spec.constraints.as_ref().unwrap();
        assert_eq!(constraints.tpu_variant, Some(TpuVariant::GoogleV5e));
    }

    #[test]
    fn test_add_memory() {
        let request = ResourceRequest::new().add_memory_gb(256.0);

        assert_eq!(request.resources.len(), 1);
        let memory_spec = request.resources.get(&ResourceType::Memory).unwrap();
        assert_eq!(memory_spec.amount, 256.0);
        assert_eq!(memory_spec.unit, ResourceUnit::Gigabytes);
    }

    #[test]
    fn test_add_storage() {
        let request = ResourceRequest::new().add_storage(StorageType::Nvme, 1024.0);

        let storage_spec = request
            .resources
            .get(&ResourceType::Storage(StorageType::Nvme))
            .unwrap();
        assert_eq!(storage_spec.amount, 1024.0);
    }

    #[test]
    fn test_add_network_bandwidth() {
        let request = ResourceRequest::new().add_network_bandwidth(10000.0);

        let network_spec = request
            .resources
            .get(&ResourceType::Network(NetworkType::Bandwidth))
            .unwrap();
        assert_eq!(network_spec.amount, 10000.0);
        assert_eq!(network_spec.unit, ResourceUnit::MegabitsPerSecond);
    }

    #[test]
    fn test_add_custom_resource() {
        let request = ResourceRequest::new().add_custom("software_licenses", 5.0);

        let custom_spec = request
            .resources
            .get(&ResourceType::Custom("software_licenses".to_string()))
            .unwrap();
        assert_eq!(custom_spec.amount, 5.0);
    }

    #[test]
    fn test_multi_resource_request() {
        let request = ResourceRequest::new()
            .add_gpu(GpuVendor::Nvidia, "H100", 4.0)
            .add_cpu_cores(64.0)
            .add_memory_gb(512.0)
            .add_storage(StorageType::Nvme, 2048.0)
            .with_priority(RequestPriority::High)
            .with_requester("user-123");

        assert_eq!(request.resources.len(), 4);
        assert!(request.has_gpu());
        assert!(request.has_cpu());
        assert_eq!(request.priority, RequestPriority::High);
        assert_eq!(request.requester_id, Some("user-123".to_string()));
    }

    #[test]
    fn test_cpu_only_request() {
        let request = ResourceRequest::new()
            .add_cpu_cores(128.0)
            .add_memory_gb(1024.0);

        assert!(!request.has_gpu());
        assert!(request.has_cpu());
        assert_eq!(request.resources.len(), 2);
    }

    #[test]
    fn test_validate_empty_request() {
        let request = ResourceRequest::new();
        assert!(request.validate().is_err());
    }

    #[test]
    fn test_validate_valid_request() {
        let request = ResourceRequest::new().add_cpu_cores(16.0);
        assert!(request.validate().is_ok());
    }

    #[test]
    fn test_priority_ordering() {
        assert!(RequestPriority::Urgent > RequestPriority::High);
        assert!(RequestPriority::High > RequestPriority::Normal);
        assert!(RequestPriority::Normal > RequestPriority::Low);
    }

    #[test]
    fn test_serialization() {
        // Test individual components can be serialized
        let priority = RequestPriority::High;
        let json = serde_json::to_string(&priority).unwrap();
        let deserialized: RequestPriority = serde_json::from_str(&json).unwrap();

        assert_eq!(priority, deserialized);
    }

    #[test]
    fn test_intel_gpu_request() {
        let request = ResourceRequest::new().add_gpu(GpuVendor::Intel, "Max-1550", 2.0);

        let gpu_spec = request.get_gpu_spec().unwrap();
        let constraints = gpu_spec.constraints.as_ref().unwrap();
        assert_eq!(constraints.vendor, Some(GpuVendor::Intel));
        assert_eq!(constraints.model, Some("Max-1550".to_string()));
    }

    #[test]
    fn test_apple_gpu_request() {
        let request = ResourceRequest::new().add_gpu(GpuVendor::Apple, "M3-Max", 1.0);

        let gpu_spec = request.get_gpu_spec().unwrap();
        let constraints = gpu_spec.constraints.as_ref().unwrap();
        assert_eq!(constraints.vendor, Some(GpuVendor::Apple));
        assert_eq!(constraints.model, Some("M3-Max".to_string()));
    }
}
