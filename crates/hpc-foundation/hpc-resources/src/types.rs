use serde::{Deserialize, Serialize};
use std::fmt;

/// Main resource type classification
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ResourceType {
    Compute(ComputeType),
    Memory,
    Storage(StorageType),
    Network(NetworkType),
    Custom(String),
}

/// Compute resource types
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ComputeType {
    Cpu,
    Gpu,
    Tpu,
    Fpga,
    Asic,
}

/// GPU vendor classification
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GpuVendor {
    Nvidia,
    Amd,
    Intel,
    Apple,
    Custom(String),
}

/// TPU variant classification
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TpuVariant {
    GoogleV2,
    GoogleV3,
    GoogleV4,
    GoogleV5e,
    GoogleV5p,
    Custom(String),
}

/// Storage resource types
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StorageType {
    Ssd,
    Nvme,
    Hdd,
    ObjectStorage,
    BlockStorage,
}

/// Network resource types
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NetworkType {
    Bandwidth,
    Iops,
    Connections,
}

/// Units for measuring resources
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResourceUnit {
    Count, // Simple count (4 GPUs, 16 CPUs)
    Bytes, // Memory/storage in bytes
    Kilobytes,
    Megabytes,
    Gigabytes,
    Terabytes,
    Cores,             // CPU cores
    Percent,           // Utilization percentage
    MegabitsPerSecond, // Network bandwidth
    MegabytesPerSecond,
    Custom(String),
}

/// Resource specification with constraints
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResourceSpec {
    pub amount: f64,
    pub unit: ResourceUnit,
    pub constraints: Option<ResourceConstraints>,
}

/// Constraints for resource matching
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ResourceConstraints {
    /// GPU vendor (Nvidia, AMD, Intel, Apple)
    pub vendor: Option<GpuVendor>,

    /// Specific model (H100, MI300X, Max-1550, M3-Max)
    pub model: Option<String>,

    /// Minimum memory in GB
    pub min_memory_gb: Option<u64>,

    /// Minimum compute capability (Nvidia CUDA compute capability)
    pub min_compute_capability: Option<String>,

    /// Architecture name (Hopper, CDNA3, Xe, etc.)
    pub architecture: Option<String>,

    /// TPU variant
    pub tpu_variant: Option<TpuVariant>,

    /// Minimum performance score
    pub min_performance_score: Option<f64>,

    /// Location preference (us-west-2a, etc.)
    pub location: Option<String>,
}

// Display implementations for better debugging
impl fmt::Display for ResourceType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ResourceType::Compute(ct) => write!(f, "Compute::{:?}", ct),
            ResourceType::Memory => write!(f, "Memory"),
            ResourceType::Storage(st) => write!(f, "Storage::{:?}", st),
            ResourceType::Network(nt) => write!(f, "Network::{:?}", nt),
            ResourceType::Custom(name) => write!(f, "Custom({})", name),
        }
    }
}

impl fmt::Display for GpuVendor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuVendor::Nvidia => write!(f, "Nvidia"),
            GpuVendor::Amd => write!(f, "AMD"),
            GpuVendor::Intel => write!(f, "Intel"),
            GpuVendor::Apple => write!(f, "Apple"),
            GpuVendor::Custom(name) => write!(f, "{}", name),
        }
    }
}

impl GpuVendor {
    pub fn as_str(&self) -> &str {
        match self {
            GpuVendor::Nvidia => "nvidia",
            GpuVendor::Amd => "amd",
            GpuVendor::Intel => "intel",
            GpuVendor::Apple => "apple",
            GpuVendor::Custom(name) => name,
        }
    }
}

impl TpuVariant {
    pub fn as_str(&self) -> &str {
        match self {
            TpuVariant::GoogleV2 => "v2",
            TpuVariant::GoogleV3 => "v3",
            TpuVariant::GoogleV4 => "v4",
            TpuVariant::GoogleV5e => "v5e",
            TpuVariant::GoogleV5p => "v5p",
            TpuVariant::Custom(name) => name,
        }
    }
}

impl ResourceUnit {
    pub fn to_bytes(&self, amount: f64) -> Option<f64> {
        match self {
            ResourceUnit::Bytes => Some(amount),
            ResourceUnit::Kilobytes => Some(amount * 1024.0),
            ResourceUnit::Megabytes => Some(amount * 1024.0 * 1024.0),
            ResourceUnit::Gigabytes => Some(amount * 1024.0 * 1024.0 * 1024.0),
            ResourceUnit::Terabytes => Some(amount * 1024.0 * 1024.0 * 1024.0 * 1024.0),
            _ => None,
        }
    }
}

impl ResourceSpec {
    pub fn new(amount: f64, unit: ResourceUnit) -> Self {
        Self {
            amount,
            unit,
            constraints: None,
        }
    }

    pub fn with_constraints(mut self, constraints: ResourceConstraints) -> Self {
        self.constraints = Some(constraints);
        self
    }

    pub fn with_vendor(mut self, vendor: GpuVendor) -> Self {
        self.constraints.get_or_insert_with(Default::default).vendor = Some(vendor);
        self
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.constraints.get_or_insert_with(Default::default).model = Some(model.into());
        self
    }

    pub fn with_min_memory_gb(mut self, memory_gb: u64) -> Self {
        self.constraints
            .get_or_insert_with(Default::default)
            .min_memory_gb = Some(memory_gb);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resource_type_creation() {
        let cpu = ResourceType::Compute(ComputeType::Cpu);
        let gpu = ResourceType::Compute(ComputeType::Gpu);
        let tpu = ResourceType::Compute(ComputeType::Tpu);
        let memory = ResourceType::Memory;
        let storage = ResourceType::Storage(StorageType::Nvme);

        assert_eq!(format!("{}", cpu), "Compute::Cpu");
        assert_eq!(format!("{}", gpu), "Compute::Gpu");
        assert_eq!(format!("{}", tpu), "Compute::Tpu");
        assert_eq!(format!("{}", memory), "Memory");
        assert_eq!(format!("{}", storage), "Storage::Nvme");
    }

    #[test]
    fn test_gpu_vendor_display() {
        assert_eq!(GpuVendor::Nvidia.to_string(), "Nvidia");
        assert_eq!(GpuVendor::Amd.to_string(), "AMD");
        assert_eq!(GpuVendor::Intel.to_string(), "Intel");
        assert_eq!(GpuVendor::Apple.to_string(), "Apple");
    }

    #[test]
    fn test_gpu_vendor_as_str() {
        assert_eq!(GpuVendor::Nvidia.as_str(), "nvidia");
        assert_eq!(GpuVendor::Amd.as_str(), "amd");
        assert_eq!(GpuVendor::Intel.as_str(), "intel");
        assert_eq!(GpuVendor::Apple.as_str(), "apple");
    }

    #[test]
    fn test_tpu_variant_as_str() {
        assert_eq!(TpuVariant::GoogleV5e.as_str(), "v5e");
        assert_eq!(TpuVariant::GoogleV5p.as_str(), "v5p");
        assert_eq!(TpuVariant::GoogleV4.as_str(), "v4");
    }

    #[test]
    fn test_resource_unit_to_bytes() {
        assert_eq!(ResourceUnit::Bytes.to_bytes(1024.0), Some(1024.0));
        assert_eq!(ResourceUnit::Kilobytes.to_bytes(1.0), Some(1024.0));
        assert_eq!(ResourceUnit::Megabytes.to_bytes(1.0), Some(1024.0 * 1024.0));
        assert_eq!(
            ResourceUnit::Gigabytes.to_bytes(1.0),
            Some(1024.0 * 1024.0 * 1024.0)
        );
        assert_eq!(ResourceUnit::Count.to_bytes(4.0), None);
    }

    #[test]
    fn test_resource_spec_creation() {
        let spec = ResourceSpec::new(4.0, ResourceUnit::Count);
        assert_eq!(spec.amount, 4.0);
        assert_eq!(spec.unit, ResourceUnit::Count);
        assert!(spec.constraints.is_none());
    }

    #[test]
    fn test_resource_spec_with_vendor() {
        let spec = ResourceSpec::new(8.0, ResourceUnit::Count)
            .with_vendor(GpuVendor::Nvidia)
            .with_model("H100")
            .with_min_memory_gb(80);

        assert_eq!(spec.amount, 8.0);
        assert!(spec.constraints.is_some());

        let constraints = spec.constraints.unwrap();
        assert_eq!(constraints.vendor, Some(GpuVendor::Nvidia));
        assert_eq!(constraints.model, Some("H100".to_string()));
        assert_eq!(constraints.min_memory_gb, Some(80));
    }

    #[test]
    fn test_resource_type_serialization() {
        let gpu = ResourceType::Compute(ComputeType::Gpu);
        let json = serde_json::to_string(&gpu).unwrap();
        let deserialized: ResourceType = serde_json::from_str(&json).unwrap();
        assert_eq!(gpu, deserialized);
    }

    #[test]
    fn test_gpu_vendor_serialization() {
        let nvidia = GpuVendor::Nvidia;
        let json = serde_json::to_string(&nvidia).unwrap();
        let deserialized: GpuVendor = serde_json::from_str(&json).unwrap();
        assert_eq!(nvidia, deserialized);
    }

    #[test]
    fn test_custom_resource_type() {
        let custom = ResourceType::Custom("software_licenses".to_string());
        assert_eq!(format!("{}", custom), "Custom(software_licenses)");
    }

    #[test]
    fn test_resource_constraints_default() {
        let constraints = ResourceConstraints::default();
        assert!(constraints.vendor.is_none());
        assert!(constraints.model.is_none());
        assert!(constraints.min_memory_gb.is_none());
    }
}
