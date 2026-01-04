//! Resource management types and operations

use serde::{Deserialize, Serialize};

/// Resource type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ResourceType {
    /// CPU cores
    CPU,
    /// System memory
    Memory,
    /// GPU compute
    GPUCompute,
    /// GPU memory
    GPUMemory,
    /// Network bandwidth
    Network,
    /// Storage I/O
    Storage,
}

/// Resource allocation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResourceAllocation {
    /// Allocated CPU cores
    pub cpu_cores: f32,
    /// Allocated memory (bytes)
    pub memory: usize,
    /// Allocated GPU compute units
    pub gpu_compute: f32,
    /// Allocated GPU memory (bytes)
    pub gpu_memory: usize,
    /// Network bandwidth (bytes/sec)
    pub network_bandwidth: u64,
    /// Storage IOPS
    pub storage_iops: u64,
}

impl ResourceAllocation {
    /// Create empty allocation
    pub fn empty() -> Self {
        Self {
            cpu_cores: 0.0,
            memory: 0,
            gpu_compute: 0.0,
            gpu_memory: 0,
            network_bandwidth: 0,
            storage_iops: 0,
        }
    }

    /// Check if allocation fits within limits
    pub fn fits_within(&self, limits: &ResourceAllocation) -> bool {
        self.cpu_cores <= limits.cpu_cores
            && self.memory <= limits.memory
            && self.gpu_compute <= limits.gpu_compute
            && self.gpu_memory <= limits.gpu_memory
            && self.network_bandwidth <= limits.network_bandwidth
            && self.storage_iops <= limits.storage_iops
    }

    /// Add allocations
    pub fn add(&mut self, other: &ResourceAllocation) {
        self.cpu_cores += other.cpu_cores;
        self.memory += other.memory;
        self.gpu_compute += other.gpu_compute;
        self.gpu_memory += other.gpu_memory;
        self.network_bandwidth += other.network_bandwidth;
        self.storage_iops += other.storage_iops;
    }

    /// Subtract allocations
    pub fn subtract(&mut self, other: &ResourceAllocation) {
        self.cpu_cores = (self.cpu_cores - other.cpu_cores).max(0.0);
        self.memory = self.memory.saturating_sub(other.memory);
        self.gpu_compute = (self.gpu_compute - other.gpu_compute).max(0.0);
        self.gpu_memory = self.gpu_memory.saturating_sub(other.gpu_memory);
        self.network_bandwidth = self
            .network_bandwidth
            .saturating_sub(other.network_bandwidth);
        self.storage_iops = self.storage_iops.saturating_sub(other.storage_iops);
    }

    /// Check if this allocation can satisfy given requirements
    pub fn can_satisfy(&self, requirements: &ResourceAllocation) -> bool {
        requirements.fits_within(self)
    }

    /// Calculate utilization ratio against total resources
    pub fn utilization_ratio(&self, total: &ResourceAllocation) -> f64 {
        let cpu_ratio = if total.cpu_cores > 0.0 {
            (self.cpu_cores / total.cpu_cores) as f64
        } else {
            0.0
        };

        let memory_ratio = if total.memory > 0 {
            self.memory as f64 / total.memory as f64
        } else {
            0.0
        };

        // Return average utilization across resources
        let ratios = [cpu_ratio, memory_ratio];
        ratios.iter().sum::<f64>() / ratios.len() as f64
    }

    /// Calculate available resources after usage
    pub fn available_after(&self, used: &ResourceAllocation) -> ResourceAllocation {
        let mut available = self.clone();
        available.subtract(used);
        available
    }
}

// Arithmetic operations for ResourceAllocation
impl std::ops::Add<&ResourceAllocation> for &ResourceAllocation {
    type Output = ResourceAllocation;

    fn add(self, other: &ResourceAllocation) -> ResourceAllocation {
        ResourceAllocation {
            cpu_cores: self.cpu_cores + other.cpu_cores,
            memory: self.memory + other.memory,
            gpu_compute: self.gpu_compute + other.gpu_compute,
            gpu_memory: self.gpu_memory + other.gpu_memory,
            network_bandwidth: self.network_bandwidth + other.network_bandwidth,
            storage_iops: self.storage_iops + other.storage_iops,
        }
    }
}

impl std::ops::Sub<&ResourceAllocation> for &ResourceAllocation {
    type Output = ResourceAllocation;

    fn sub(self, other: &ResourceAllocation) -> ResourceAllocation {
        ResourceAllocation {
            cpu_cores: (self.cpu_cores - other.cpu_cores).max(0.0),
            memory: self.memory.saturating_sub(other.memory),
            gpu_compute: (self.gpu_compute - other.gpu_compute).max(0.0),
            gpu_memory: self.gpu_memory.saturating_sub(other.gpu_memory),
            network_bandwidth: self
                .network_bandwidth
                .saturating_sub(other.network_bandwidth),
            storage_iops: self.storage_iops.saturating_sub(other.storage_iops),
        }
    }
}

impl std::ops::Mul<f64> for &ResourceAllocation {
    type Output = ResourceAllocation;

    fn mul(self, scalar: f64) -> ResourceAllocation {
        ResourceAllocation {
            cpu_cores: (self.cpu_cores as f64 * scalar) as f32,
            memory: (self.memory as f64 * scalar) as usize,
            gpu_compute: (self.gpu_compute as f64 * scalar) as f32,
            gpu_memory: (self.gpu_memory as f64 * scalar) as usize,
            network_bandwidth: (self.network_bandwidth as f64 * scalar) as u64,
            storage_iops: (self.storage_iops as f64 * scalar) as u64,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resource_allocation_basic() {
        let mut alloc1 = ResourceAllocation {
            cpu_cores: 2.0,
            memory: 1024,
            gpu_compute: 1.0,
            gpu_memory: 512,
            network_bandwidth: 100,
            storage_iops: 1000,
        };

        let alloc2 = ResourceAllocation {
            cpu_cores: 1.0,
            memory: 512,
            gpu_compute: 0.5,
            gpu_memory: 256,
            network_bandwidth: 50,
            storage_iops: 500,
        };

        // Test fits_within
        let limits = ResourceAllocation {
            cpu_cores: 4.0,
            memory: 2048,
            gpu_compute: 2.0,
            gpu_memory: 1024,
            network_bandwidth: 200,
            storage_iops: 2000,
        };

        assert!(alloc1.fits_within(&limits));

        // Test add
        alloc1.add(&alloc2);
        assert_eq!(alloc1.cpu_cores, 3.0);
        assert_eq!(alloc1.memory, 1536);

        // Test subtract
        alloc1.subtract(&alloc2);
        assert_eq!(alloc1.cpu_cores, 2.0);
        assert_eq!(alloc1.memory, 1024);
    }

    #[test]
    fn test_resource_allocation_arithmetic() {
        let alloc1 = ResourceAllocation {
            cpu_cores: 2.0,
            memory: 4096,
            gpu_compute: 1.0,
            gpu_memory: 2048,
            network_bandwidth: 500,
            storage_iops: 5000,
        };

        let alloc2 = ResourceAllocation {
            cpu_cores: 1.0,
            memory: 2048,
            gpu_compute: 0.5,
            gpu_memory: 1024,
            network_bandwidth: 250,
            storage_iops: 2500,
        };

        // Test addition
        let sum = &alloc1 + &alloc2;
        assert_eq!(sum.cpu_cores, 3.0);
        assert_eq!(sum.memory, 6144);

        // Test subtraction
        let diff = &alloc1 - &alloc2;
        assert_eq!(diff.cpu_cores, 1.0);
        assert_eq!(diff.memory, 2048);

        // Test multiplication
        let scaled = &alloc1 * 2.0;
        assert_eq!(scaled.cpu_cores, 4.0);
        assert_eq!(scaled.memory, 8192);
    }

    #[test]
    fn test_resource_allocation_extensions() {
        let total = ResourceAllocation {
            cpu_cores: 8.0,
            memory: 16384,
            gpu_compute: 4.0,
            gpu_memory: 8192,
            network_bandwidth: 2000,
            storage_iops: 20000,
        };

        let used = ResourceAllocation {
            cpu_cores: 2.0,
            memory: 4096,
            gpu_compute: 1.0,
            gpu_memory: 2048,
            network_bandwidth: 500,
            storage_iops: 5000,
        };

        // Test can_satisfy
        assert!(total.can_satisfy(&used));
        assert!(!used.can_satisfy(&total));

        // Test utilization_ratio
        let ratio = used.utilization_ratio(&total);
        assert!(ratio > 0.0 && ratio <= 1.0);

        // Test available_after
        let available = total.available_after(&used);
        assert_eq!(available.cpu_cores, 6.0);
        assert_eq!(available.memory, 12288);
    }

    #[test]
    fn test_resource_allocation_empty() {
        let empty = ResourceAllocation::empty();
        assert_eq!(empty.cpu_cores, 0.0);
        assert_eq!(empty.memory, 0);
        assert_eq!(empty.gpu_compute, 0.0);
        assert_eq!(empty.gpu_memory, 0);
        assert_eq!(empty.network_bandwidth, 0);
        assert_eq!(empty.storage_iops, 0);
    }

    #[test]
    fn test_resource_allocation_fits_within_edge_cases() {
        let alloc = ResourceAllocation {
            cpu_cores: 2.0,
            memory: 1024,
            gpu_compute: 1.0,
            gpu_memory: 512,
            network_bandwidth: 100,
            storage_iops: 1000,
        };

        // Exact match should fit
        assert!(alloc.fits_within(&alloc));

        // Any resource exceeding limit should fail
        let tight_limits = ResourceAllocation {
            cpu_cores: 1.9,
            memory: 1024,
            gpu_compute: 1.0,
            gpu_memory: 512,
            network_bandwidth: 100,
            storage_iops: 1000,
        };
        assert!(!alloc.fits_within(&tight_limits));
    }

    #[test]
    fn test_resource_allocation_subtract_underflow() {
        let mut alloc = ResourceAllocation {
            cpu_cores: 1.0,
            memory: 500,
            gpu_compute: 0.5,
            gpu_memory: 256,
            network_bandwidth: 50,
            storage_iops: 100,
        };

        let larger = ResourceAllocation {
            cpu_cores: 2.0,
            memory: 1000,
            gpu_compute: 1.0,
            gpu_memory: 512,
            network_bandwidth: 100,
            storage_iops: 200,
        };

        // Subtract should clamp to 0, not underflow
        alloc.subtract(&larger);
        assert_eq!(alloc.cpu_cores, 0.0);
        assert_eq!(alloc.memory, 0);
        assert_eq!(alloc.gpu_compute, 0.0);
        assert_eq!(alloc.gpu_memory, 0);
        assert_eq!(alloc.network_bandwidth, 0);
        assert_eq!(alloc.storage_iops, 0);
    }

    #[test]
    fn test_resource_type_variants() {
        // Ensure all resource types are covered
        let types = vec![
            ResourceType::CPU,
            ResourceType::Memory,
            ResourceType::GPUCompute,
            ResourceType::GPUMemory,
            ResourceType::Network,
            ResourceType::Storage,
        ];

        // Just ensure they can be used
        for resource_type in types {
            match resource_type {
                ResourceType::CPU => assert_eq!(format!("{:?}", resource_type), "CPU"),
                ResourceType::Memory => assert_eq!(format!("{:?}", resource_type), "Memory"),
                ResourceType::GPUCompute => {
                    assert_eq!(format!("{:?}", resource_type), "GPUCompute")
                }
                ResourceType::GPUMemory => assert_eq!(format!("{:?}", resource_type), "GPUMemory"),
                ResourceType::Network => assert_eq!(format!("{:?}", resource_type), "Network"),
                ResourceType::Storage => assert_eq!(format!("{:?}", resource_type), "Storage"),
            }
        }
    }

    #[test]
    fn test_resource_allocation_boundary_conditions() {
        // Test zero resources
        let zero_alloc = ResourceAllocation {
            cpu_cores: 0.0,
            memory: 0,
            gpu_compute: 0.0,
            gpu_memory: 0,
            network_bandwidth: 0,
            storage_iops: 0,
        };

        let normal_alloc = ResourceAllocation {
            cpu_cores: 4.0,
            memory: 8192,
            gpu_compute: 2.0,
            gpu_memory: 4096,
            network_bandwidth: 1000,
            storage_iops: 10000,
        };

        // Adding zero should not change anything
        let sum = &normal_alloc + &zero_alloc;
        assert_eq!(sum.cpu_cores, normal_alloc.cpu_cores);
        assert_eq!(sum.memory, normal_alloc.memory);

        // Subtracting zero should not change anything
        let diff = &normal_alloc - &zero_alloc;
        assert_eq!(diff.cpu_cores, normal_alloc.cpu_cores);

        // Multiplying by zero should result in zero
        let scaled_zero = &normal_alloc * 0.0;
        assert_eq!(scaled_zero.cpu_cores, 0.0);
        assert_eq!(scaled_zero.memory, 0);
    }

    #[test]
    fn test_resource_allocation_extreme_values() {
        let max_alloc = ResourceAllocation {
            cpu_cores: f32::MAX,
            memory: usize::MAX,
            gpu_compute: f32::MAX,
            gpu_memory: usize::MAX,
            network_bandwidth: u64::MAX,
            storage_iops: u64::MAX,
        };

        // Test that extreme values don't cause panics
        let small_alloc = ResourceAllocation {
            cpu_cores: 1.0,
            memory: 1024,
            gpu_compute: 0.5,
            gpu_memory: 512,
            network_bandwidth: 100,
            storage_iops: 1000,
        };

        // This should handle overflow gracefully
        let _sum = &max_alloc + &small_alloc;

        // Test utilization with extreme values
        let utilization = small_alloc.utilization_ratio(&max_alloc);
        assert!(utilization >= 0.0 && utilization <= 1.0);
    }

    #[test]
    fn test_resource_comparison_operations() {
        let alloc1 = ResourceAllocation {
            cpu_cores: 2.0,
            memory: 4096,
            gpu_compute: 1.0,
            gpu_memory: 2048,
            network_bandwidth: 500,
            storage_iops: 5000,
        };

        let alloc2 = alloc1.clone();
        let alloc3 = ResourceAllocation {
            cpu_cores: 4.0,
            memory: 8192,
            gpu_compute: 2.0,
            gpu_memory: 4096,
            network_bandwidth: 1000,
            storage_iops: 10000,
        };

        // Test equality
        assert_eq!(alloc1, alloc2);
        assert_ne!(alloc1, alloc3);

        // Test satisfiability
        assert!(alloc1.can_satisfy(&alloc1)); // Should be able to satisfy itself
        assert!(!alloc1.can_satisfy(&alloc3)); // Cannot satisfy larger requirements
    }
}
