//! Node classification system
//!
//! This module classifies nodes into different categories based on their
//! hardware capabilities, location, and operational characteristics.

use crate::discovery::{GpuInfo, PowerProfile};
use crate::{HardwareProfile, NetworkCharacteristics, Result};
use serde::{Deserialize, Serialize};

/// Node classification based on capabilities and characteristics
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NodeClass {
    /// High-performance datacenter nodes
    DataCenter {
        gpus: Vec<GpuInfo>,
        bandwidth: Bandwidth,
    },

    /// Workstation nodes with optional GPU
    Workstation {
        gpu: Option<GpuInfo>,
        schedule: Schedule,
    },

    /// Laptop nodes with mobility concerns
    Laptop {
        battery: bool,
        mobility: MobilityPattern,
    },

    /// Edge devices with power constraints
    Edge {
        device_type: EdgeType,
        power_budget: Watts,
    },
}

/// Network bandwidth classification
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Bandwidth {
    Gigabit(f32),
    MultiGigabit(f32), // 2.5G, 5G, etc.
    TenGigabit(f32),
    FortyGigabit(f32),
    HundredGigabit(f32),
    Custom(f32),
}

/// Node availability schedule
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Schedule {
    AlwaysOn,
    BusinessHours,
    AfterHours,
    Weekend,
    Custom(String),
}

/// Mobility pattern for laptop nodes
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MobilityPattern {
    Stationary,
    Occasional,
    Frequent,
    Continuous,
}

/// Edge device types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EdgeType {
    RaspberryPi,
    JetsonNano,
    JetsonXavier,
    JetsonOrin,
    IntelNuc,
    IntelNUC, // Alias for compatibility
    Other(String),
}

/// Power budget in watts
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Watts(pub f32);

/// Node classifier
pub struct NodeClassifier {
    /// Minimum memory for datacenter classification (GB)
    datacenter_min_memory: f32,
    /// Minimum cores for datacenter classification
    datacenter_min_cores: u32,
    /// Minimum bandwidth for datacenter classification (Mbps)
    datacenter_min_bandwidth: f32,
}

impl NodeClassifier {
    /// Create a new node classifier
    pub fn new() -> Self {
        Self {
            datacenter_min_memory: 128.0,
            datacenter_min_cores: 32,
            datacenter_min_bandwidth: 10000.0, // 10 Gbps
        }
    }

    /// Classify a node based on its profile
    pub fn classify(
        &self,
        hardware: &HardwareProfile,
        network: &NetworkCharacteristics,
        power: &PowerProfile,
    ) -> Result<NodeClass> {
        // Check if it's an edge device first
        if let Some(edge_type) = self.detect_edge_device(hardware) {
            let power_budget = self.estimate_power_budget(hardware, power);
            return Ok(NodeClass::Edge {
                device_type: edge_type,
                power_budget,
            });
        }

        // Check if it's a laptop
        if power.on_battery || self.is_laptop_hardware(hardware) {
            return Ok(NodeClass::Laptop {
                battery: power.on_battery,
                mobility: self.detect_mobility_pattern(power),
            });
        }

        // Check if it qualifies as datacenter
        if self.is_datacenter_grade(hardware, network) {
            let gpus = self.extract_gpus(hardware);
            let bandwidth = self.classify_bandwidth(network.bandwidth_mbps);
            return Ok(NodeClass::DataCenter { gpus, bandwidth });
        }

        // Default to workstation
        let gpu = self.extract_gpus(hardware).into_iter().next();
        let schedule = self.detect_schedule();
        Ok(NodeClass::Workstation { gpu, schedule })
    }

    /// Check if hardware is datacenter grade
    fn is_datacenter_grade(
        &self,
        hardware: &HardwareProfile,
        network: &NetworkCharacteristics,
    ) -> bool {
        hardware.memory_gb >= self.datacenter_min_memory
            && hardware.cpu_cores >= self.datacenter_min_cores
            && network.bandwidth_mbps >= self.datacenter_min_bandwidth
            && network.nat_type == crate::discovery::NatType::None
    }

    /// Check if hardware appears to be a laptop
    fn is_laptop_hardware(&self, hardware: &HardwareProfile) -> bool {
        // Heuristics for laptop detection
        let cpu_lower = hardware.cpu_model.to_lowercase();

        // Check for laptop-specific CPU suffixes (but not EPYC)

        // Only consider it a laptop if it has laptop CPU markers
        !cpu_lower.contains("epyc")
            && (cpu_lower.ends_with("u")
                || cpu_lower.ends_with("h")
                || cpu_lower.ends_with("p")
                || cpu_lower.contains("mobile"))
    }

    /// Detect edge device type
    fn detect_edge_device(&self, hardware: &HardwareProfile) -> Option<EdgeType> {
        let cpu_lower = hardware.cpu_model.to_lowercase();

        if cpu_lower.contains("bcm2711") || cpu_lower.contains("bcm2835") {
            return Some(EdgeType::RaspberryPi);
        }

        if cpu_lower.contains("tegra") {
            if hardware.memory_gb <= 4.0 {
                return Some(EdgeType::JetsonNano);
            } else if hardware.memory_gb <= 16.0 {
                return Some(EdgeType::JetsonXavier);
            } else {
                return Some(EdgeType::JetsonOrin);
            }
        }

        if (cpu_lower.contains("celeron") || cpu_lower.contains("atom"))
            && hardware.cpu_cores <= 4
            && hardware.memory_gb <= 16.0
        {
            return Some(EdgeType::IntelNUC);
        }

        None
    }

    /// Extract GPU information from hardware profile
    fn extract_gpus(&self, hardware: &HardwareProfile) -> Vec<GpuInfo> {
        hardware
            .gpus
            .iter()
            .map(|gpu_info| GpuInfo {
                index: gpu_info.index,
                name: gpu_info.name.clone(),
                memory_mb: gpu_info.memory_mb,
                compute_capability: gpu_info.compute_capability,
                pci_bus_id: gpu_info.pci_bus_id.clone(),
            })
            .collect()
    }

    /// Classify network bandwidth
    fn classify_bandwidth(&self, bandwidth_mbps: f32) -> Bandwidth {
        if bandwidth_mbps >= 100000.0 {
            Bandwidth::HundredGigabit(bandwidth_mbps / 1000.0)
        } else if bandwidth_mbps >= 10000.0 {
            Bandwidth::TenGigabit(bandwidth_mbps / 1000.0)
        } else if bandwidth_mbps >= 2000.0 {
            Bandwidth::MultiGigabit(bandwidth_mbps / 1000.0)
        } else {
            Bandwidth::Gigabit(bandwidth_mbps / 1000.0)
        }
    }

    /// Detect mobility pattern based on power profile
    fn detect_mobility_pattern(&self, power: &PowerProfile) -> MobilityPattern {
        // This would be enhanced with historical data
        if !power.on_battery {
            MobilityPattern::Stationary
        } else if power.battery_percentage.unwrap_or(100) > 80 {
            MobilityPattern::Occasional
        } else {
            MobilityPattern::Frequent
        }
    }

    /// Detect node schedule (would be enhanced with ML)
    fn detect_schedule(&self) -> Schedule {
        // This would learn from usage patterns
        Schedule::BusinessHours
    }

    /// Estimate power budget
    fn estimate_power_budget(&self, hardware: &HardwareProfile, power: &PowerProfile) -> Watts {
        // Simple estimation based on hardware
        let cpu_watts = (hardware.cpu_cores as f32) * 5.0; // 5W per core estimate
        let gpu_watts = hardware.gpus.len() as f32 * 75.0; // 75W per GPU estimate
        let base_watts = 20.0; // Base system power

        let total = cpu_watts + gpu_watts + base_watts;

        // Apply power plan multiplier
        let multiplier = match power.power_plan {
            crate::discovery::PowerPlan::Performance => 1.2,
            crate::discovery::PowerPlan::Balanced => 1.0,
            crate::discovery::PowerPlan::PowerSaver => 0.7,
            crate::discovery::PowerPlan::Unknown => 1.0,
        };

        Watts(total * multiplier)
    }
}

impl Default for NodeClassifier {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discovery::{GpuInfo, NatType, PowerPlan};

    fn create_test_hardware(cores: u32, memory_gb: f32, model: &str) -> HardwareProfile {
        HardwareProfile {
            cpu_model: model.to_string(),
            cpu_cores: cores,
            memory_gb,
            storage_gb: 1000.0,
            gpus: vec![],
        }
    }

    fn create_test_network(bandwidth: f32, nat: NatType) -> NetworkCharacteristics {
        NetworkCharacteristics {
            bandwidth_mbps: bandwidth,
            latency_ms: 1.0,
            jitter_ms: 0.1,
            packet_loss: 0.0,
            nat_type: nat,
        }
    }

    fn create_test_power(on_battery: bool) -> PowerProfile {
        PowerProfile {
            on_battery,
            battery_percentage: if on_battery { Some(85) } else { None },
            power_plan: PowerPlan::Balanced,
            estimated_runtime_minutes: if on_battery { Some(120) } else { None },
        }
    }

    #[test]
    fn test_datacenter_classification() {
        let classifier = NodeClassifier::new();
        let hardware = create_test_hardware(64, 256.0, "AMD EPYC 7763");
        let network = create_test_network(10000.0, NatType::None);
        let power = create_test_power(false);

        let class = classifier.classify(&hardware, &network, &power).unwrap();
        match class {
            NodeClass::DataCenter { bandwidth, .. } => {
                assert!(matches!(bandwidth, Bandwidth::TenGigabit(_)));
            }
            _ => panic!("Expected DataCenter classification"),
        }
    }

    #[test]
    fn test_laptop_classification() {
        let classifier = NodeClassifier::new();
        let hardware = create_test_hardware(8, 16.0, "Intel Core i7-1165G7U");
        let network = create_test_network(100.0, NatType::Symmetric);
        let power = create_test_power(true);

        let class = classifier.classify(&hardware, &network, &power).unwrap();
        match class {
            NodeClass::Laptop { battery, .. } => {
                assert!(battery);
            }
            _ => panic!("Expected Laptop classification"),
        }
    }

    #[test]
    fn test_workstation_classification() {
        let classifier = NodeClassifier::new();
        let mut hardware = create_test_hardware(16, 32.0, "Intel Core i9-13900K");
        hardware.gpus.push(GpuInfo {
            index: 0,
            name: "NVIDIA RTX 4090".to_string(),
            memory_mb: 24576,
            compute_capability: (8, 9),
            pci_bus_id: "0000:01:00.0".to_string(),
        });
        let network = create_test_network(1000.0, NatType::None);
        let power = create_test_power(false);

        let class = classifier.classify(&hardware, &network, &power).unwrap();
        match class {
            NodeClass::Workstation { gpu, .. } => {
                assert!(gpu.is_some());
                let gpu = gpu.unwrap();
                assert_eq!(gpu.memory_gb, 24.0);
            }
            _ => panic!("Expected Workstation classification"),
        }
    }

    #[test]
    fn test_edge_device_classification() {
        let classifier = NodeClassifier::new();
        let hardware = create_test_hardware(4, 4.0, "BCM2711");
        let network = create_test_network(100.0, NatType::PortRestricted);
        let power = create_test_power(false);

        let class = classifier.classify(&hardware, &network, &power).unwrap();
        match class {
            NodeClass::Edge { device_type, .. } => {
                assert_eq!(device_type, EdgeType::RaspberryPi);
            }
            _ => panic!("Expected Edge classification"),
        }
    }

    #[test]
    fn test_bandwidth_classification() {
        let classifier = NodeClassifier::new();

        let bandwidth = classifier.classify_bandwidth(500.0);
        assert!(matches!(bandwidth, Bandwidth::Gigabit(_)));

        let bandwidth = classifier.classify_bandwidth(2500.0);
        assert!(matches!(bandwidth, Bandwidth::MultiGigabit(_)));

        let bandwidth = classifier.classify_bandwidth(25000.0);
        assert!(matches!(bandwidth, Bandwidth::TenGigabit(_)));

        let bandwidth = classifier.classify_bandwidth(100000.0);
        assert!(matches!(bandwidth, Bandwidth::HundredGigabit(_)));
    }

    #[test]
    fn test_mobility_patterns() {
        let classifier = NodeClassifier::new();

        let power_stationary = PowerProfile {
            on_battery: false,
            battery_percentage: None,
            power_plan: PowerPlan::Performance,
            estimated_runtime_minutes: None,
        };
        assert_eq!(
            classifier.detect_mobility_pattern(&power_stationary),
            MobilityPattern::Stationary
        );

        let power_mobile = PowerProfile {
            on_battery: true,
            battery_percentage: Some(90),
            power_plan: PowerPlan::Balanced,
            estimated_runtime_minutes: Some(180),
        };
        assert_eq!(
            classifier.detect_mobility_pattern(&power_mobile),
            MobilityPattern::Occasional
        );
    }

    #[test]
    fn test_power_budget_estimation() {
        let classifier = NodeClassifier::new();
        let hardware = create_test_hardware(8, 16.0, "Intel i7");
        let power = create_test_power(false);

        let budget = classifier.estimate_power_budget(&hardware, &power);
        assert!(budget.0 > 0.0);
        assert!(budget.0 < 1000.0); // Reasonable power budget
    }

    #[test]
    fn test_gpu_extraction() {
        let classifier = NodeClassifier::new();
        let mut hardware = create_test_hardware(16, 32.0, "Intel i9");

        hardware.gpus.push(GpuInfo {
            index: 0,
            name: "NVIDIA RTX 3080".to_string(),
            memory_mb: 10240,
            compute_capability: (8, 6),
            pci_bus_id: "0000:01:00.0".to_string(),
        });

        hardware.gpus.push(GpuInfo {
            index: 1,
            name: "NVIDIA RTX 3080".to_string(),
            memory_mb: 10240,
            compute_capability: (8, 6),
            pci_bus_id: "0000:02:00.0".to_string(),
        });

        let gpus = classifier.extract_gpus(&hardware);
        assert_eq!(gpus.len(), 2);
        assert_eq!(gpus[0].memory_gb, 10.0);
        assert_eq!(gpus[0].compute_capability, (8, 6));
    }
}
