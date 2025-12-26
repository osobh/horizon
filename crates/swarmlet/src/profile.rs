//! Hardware profiling for swarmlet nodes

use crate::{Result, SwarmletError};
use serde::{Deserialize, Serialize};
use sysinfo::System;
use uuid::Uuid;

/// Complete hardware profile of a node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareProfile {
    pub node_id: Uuid,
    pub hostname: String,
    pub architecture: String,
    pub cpu: CpuInfo,
    pub memory: MemoryInfo,
    pub storage: StorageInfo,
    pub network: NetworkInfo,
    pub gpu: Option<GpuInfo>,
    pub capabilities: NodeCapabilities,
    pub thermal: Option<ThermalInfo>,
    pub power: Option<PowerInfo>,
    pub device_type: DeviceType,
}

/// CPU information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuInfo {
    pub model: String,
    pub cores: u32,
    pub threads: u32,
    pub frequency_mhz: u64,
    pub architecture: String,
    pub features: Vec<String>,
}

/// Memory information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryInfo {
    pub total_gb: f64,
    pub available_gb: f64,
    pub swap_gb: f64,
}

/// Storage information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageInfo {
    pub total_gb: f64,
    pub available_gb: f64,
    pub disks: Vec<DiskInfo>,
}

/// Individual disk information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiskInfo {
    pub name: String,
    pub mount_point: String,
    pub total_gb: f64,
    pub available_gb: f64,
    pub disk_type: String, // SSD, HDD, NVMe, etc.
}

/// Network information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkInfo {
    pub interfaces: Vec<NetworkInterface>,
    pub estimated_bandwidth_mbps: f64,
    pub connectivity: ConnectivityType,
}

/// Network interface information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkInterface {
    pub name: String,
    pub ip_addresses: Vec<String>,
    pub mac_address: Option<String>,
    pub is_up: bool,
    pub speed_mbps: Option<u64>,
}

/// GPU information (if available)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInfo {
    pub count: u32,
    pub models: Vec<String>,
    pub total_memory_gb: f64,
    pub compute_capability: Option<String>,
}

/// Node capabilities based on hardware profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeCapabilities {
    pub max_workloads: u32,
    pub memory_limit_gb: f64,
    pub cpu_limit_cores: u32,
    pub storage_limit_gb: f64,
    pub gpu_capable: bool,
    pub container_runtime: bool,
    pub network_bandwidth_mbps: f64,
    pub suitability: Vec<WorkloadType>,
}

/// Thermal information (especially important for edge devices)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalInfo {
    pub current_temp_celsius: f32,
    pub max_temp_celsius: f32,
    pub thermal_throttling: bool,
    pub cooling_type: CoolingType,
}

/// Power information (for battery-powered devices)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerInfo {
    pub on_battery: bool,
    pub battery_percentage: Option<u8>,
    pub estimated_runtime_minutes: Option<u32>,
    pub power_profile: PowerProfile,
}

/// Device type classification
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum DeviceType {
    Server,
    Workstation,
    Laptop,
    RaspberryPi,
    EdgeDevice,
    IoTDevice,
    Unknown,
}

/// Connectivity type
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ConnectivityType {
    Ethernet,
    WiFi,
    Cellular,
    Mixed,
    Unknown,
}

/// Workload suitability types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum WorkloadType {
    Compute,
    Storage,
    Network,
    Sensor,
    Gateway,
    Cache,
    Monitoring,
    Development,
    Testing,
}

/// Cooling type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CoolingType {
    Active,
    Passive,
    None,
}

/// Power profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PowerProfile {
    Performance,
    Balanced,
    PowerSaver,
    Unknown,
}

/// Hardware profiler
pub struct HardwareProfiler {
    system: System,
}

impl HardwareProfiler {
    /// Create a new hardware profiler
    pub fn new() -> Self {
        let mut system = System::new_all();
        system.refresh_all();

        Self { system }
    }

    /// Profile the local hardware
    pub async fn profile(&mut self) -> Result<HardwareProfile> {
        self.system.refresh_all();

        let node_id = Uuid::new_v4();
        let hostname = self.get_hostname();
        let architecture = self.get_architecture();

        let cpu = self.profile_cpu().await?;
        let memory = self.profile_memory().await?;
        let storage = self.profile_storage().await?;
        let network = self.profile_network().await?;
        let gpu = self.profile_gpu().await?;
        let thermal = self.profile_thermal().await?;
        let power = self.profile_power().await?;

        let device_type = self.classify_device(&cpu, &memory, &gpu, &power);
        let capabilities =
            self.calculate_capabilities(&cpu, &memory, &storage, &network, &gpu, &device_type);

        Ok(HardwareProfile {
            node_id,
            hostname,
            architecture,
            cpu,
            memory,
            storage,
            network,
            gpu,
            capabilities,
            thermal,
            power,
            device_type,
        })
    }

    fn get_hostname(&self) -> String {
        hostname::get()
            .unwrap_or_else(|_| "unknown".into())
            .to_string_lossy()
            .to_string()
    }

    fn get_architecture(&self) -> String {
        std::env::consts::ARCH.to_string()
    }

    async fn profile_cpu(&self) -> Result<CpuInfo> {
        let cpus = self.system.cpus();

        if cpus.is_empty() {
            return Err(SwarmletError::HardwareProfiling(
                "No CPU information available".to_string(),
            ));
        }

        let cpu = &cpus[0];
        let model = cpu.brand().to_string();
        let cores = self.system.physical_core_count().unwrap_or(1) as u32;
        let threads = cpus.len() as u32;
        let frequency_mhz = cpu.frequency();
        let architecture = self.get_architecture();

        // CPU features would need platform-specific detection
        let features = self.detect_cpu_features();

        Ok(CpuInfo {
            model,
            cores,
            threads,
            frequency_mhz,
            architecture,
            features,
        })
    }

    async fn profile_memory(&self) -> Result<MemoryInfo> {
        let total_memory = self.system.total_memory();
        let available_memory = self.system.available_memory();
        let total_swap = self.system.total_swap();

        Ok(MemoryInfo {
            total_gb: total_memory as f64 / (1024.0 * 1024.0 * 1024.0),
            available_gb: available_memory as f64 / (1024.0 * 1024.0 * 1024.0),
            swap_gb: total_swap as f64 / (1024.0 * 1024.0 * 1024.0),
        })
    }

    async fn profile_storage(&self) -> Result<StorageInfo> {
        let disks = sysinfo::Disks::new_with_refreshed_list();
        let mut disk_info = Vec::new();
        let mut total_gb = 0.0;
        let mut available_gb = 0.0;

        for disk in disks.iter() {
            let disk_total = disk.total_space() as f64 / (1024.0 * 1024.0 * 1024.0);
            let disk_available = disk.available_space() as f64 / (1024.0 * 1024.0 * 1024.0);

            total_gb += disk_total;
            available_gb += disk_available;

            disk_info.push(DiskInfo {
                name: disk.name().to_string_lossy().to_string(),
                mount_point: disk.mount_point().to_string_lossy().to_string(),
                total_gb: disk_total,
                available_gb: disk_available,
                disk_type: self.detect_disk_type(disk.name().to_string_lossy().as_ref()),
            });
        }

        Ok(StorageInfo {
            total_gb,
            available_gb,
            disks: disk_info,
        })
    }

    async fn profile_network(&self) -> Result<NetworkInfo> {
        let networks = sysinfo::Networks::new_with_refreshed_list();
        let mut interfaces = Vec::new();

        for (interface_name, data) in networks.iter() {
            interfaces.push(NetworkInterface {
                name: interface_name.clone(),
                ip_addresses: Vec::new(), // Would need platform-specific detection
                mac_address: None,        // Would need platform-specific detection
                is_up: data.total_received() > 0 || data.total_transmitted() > 0,
                speed_mbps: None, // Would need platform-specific detection
            });
        }

        // Estimate bandwidth based on interface types
        let estimated_bandwidth_mbps = self.estimate_bandwidth(&interfaces);
        let connectivity = self.detect_connectivity_type(&interfaces);

        Ok(NetworkInfo {
            interfaces,
            estimated_bandwidth_mbps,
            connectivity,
        })
    }

    async fn profile_gpu(&self) -> Result<Option<GpuInfo>> {
        // GPU detection would need platform-specific code
        // For now, return None - this would be enhanced with NVML, OpenCL, etc.
        Ok(None)
    }

    async fn profile_thermal(&self) -> Result<Option<ThermalInfo>> {
        // Thermal monitoring would need platform-specific code
        // Important for Raspberry Pi and other edge devices
        Ok(None)
    }

    async fn profile_power(&self) -> Result<Option<PowerInfo>> {
        // Battery status would need platform-specific code
        Ok(None)
    }

    fn classify_device(
        &self,
        cpu: &CpuInfo,
        memory: &MemoryInfo,
        gpu: &Option<GpuInfo>,
        power: &Option<PowerInfo>,
    ) -> DeviceType {
        // Raspberry Pi detection
        if cpu.model.to_lowercase().contains("arm")
            && cpu.model.to_lowercase().contains("cortex")
            && memory.total_gb <= 8.0
        {
            return DeviceType::RaspberryPi;
        }

        // Server detection (high core count, lots of memory)
        if cpu.cores >= 16 && memory.total_gb >= 32.0 {
            return DeviceType::Server;
        }

        // Laptop detection (battery powered)
        if let Some(power_info) = power {
            if power_info.on_battery {
                return DeviceType::Laptop;
            }
        }

        // Workstation detection (GPU + decent specs)
        if gpu.is_some() && memory.total_gb >= 16.0 {
            return DeviceType::Workstation;
        }

        // IoT device detection (very limited resources)
        if cpu.cores <= 2 && memory.total_gb <= 1.0 {
            return DeviceType::IoTDevice;
        }

        // Edge device (ARM or limited resources but not IoT)
        if cpu.architecture.contains("arm") || (cpu.cores <= 4 && memory.total_gb <= 4.0) {
            return DeviceType::EdgeDevice;
        }

        DeviceType::Unknown
    }

    fn calculate_capabilities(
        &self,
        cpu: &CpuInfo,
        memory: &MemoryInfo,
        storage: &StorageInfo,
        network: &NetworkInfo,
        gpu: &Option<GpuInfo>,
        device_type: &DeviceType,
    ) -> NodeCapabilities {
        // Conservative resource allocation
        let memory_limit_gb = (memory.available_gb * 0.8).max(0.1);
        let cpu_limit_cores = (cpu.cores as f32 * 0.8).max(1.0) as u32;
        let storage_limit_gb = (storage.available_gb * 0.5).max(1.0);

        // Estimate max workloads based on resources
        let max_workloads = ((cpu.cores as f32 * 2.0).min(memory_limit_gb as f32 * 2.0)) as u32;

        let gpu_capable = gpu.is_some();
        let container_runtime = true; // Assume Docker is available

        let suitability = self.determine_suitability(device_type, gpu_capable, memory_limit_gb);

        NodeCapabilities {
            max_workloads,
            memory_limit_gb,
            cpu_limit_cores,
            storage_limit_gb,
            gpu_capable,
            container_runtime,
            network_bandwidth_mbps: network.estimated_bandwidth_mbps,
            suitability,
        }
    }

    fn determine_suitability(
        &self,
        device_type: &DeviceType,
        gpu_capable: bool,
        memory_gb: f64,
    ) -> Vec<WorkloadType> {
        let mut suitability = Vec::new();

        match device_type {
            DeviceType::Server => {
                suitability.extend_from_slice(&[
                    WorkloadType::Compute,
                    WorkloadType::Storage,
                    WorkloadType::Network,
                    WorkloadType::Cache,
                ]);
            }
            DeviceType::Workstation => {
                suitability.extend_from_slice(&[
                    WorkloadType::Compute,
                    WorkloadType::Development,
                    WorkloadType::Testing,
                ]);
            }
            DeviceType::Laptop => {
                suitability.extend_from_slice(&[
                    WorkloadType::Development,
                    WorkloadType::Testing,
                    WorkloadType::Monitoring,
                ]);
            }
            DeviceType::RaspberryPi | DeviceType::EdgeDevice => {
                suitability.extend_from_slice(&[
                    WorkloadType::Sensor,
                    WorkloadType::Gateway,
                    WorkloadType::Monitoring,
                ]);
            }
            DeviceType::IoTDevice => {
                suitability.extend_from_slice(&[WorkloadType::Sensor, WorkloadType::Monitoring]);
            }
            DeviceType::Unknown => {
                suitability.push(WorkloadType::Monitoring);
            }
        }

        if gpu_capable {
            suitability.push(WorkloadType::Compute);
        }

        if memory_gb >= 4.0 {
            suitability.push(WorkloadType::Cache);
        }

        suitability
    }

    fn detect_cpu_features(&self) -> Vec<String> {
        // This would detect CPU features like AVX, SSE, etc.
        // For now, return empty - would need platform-specific code
        Vec::new()
    }

    fn detect_disk_type(&self, name: &str) -> String {
        let name_lower = name.to_lowercase();

        if name_lower.contains("nvme") {
            "NVMe".to_string()
        } else if name_lower.contains("ssd") {
            "SSD".to_string()
        } else if name_lower.starts_with("sd") {
            "SD Card".to_string()
        } else {
            "HDD".to_string()
        }
    }

    fn estimate_bandwidth(&self, interfaces: &[NetworkInterface]) -> f64 {
        // Simple heuristic - would need actual network testing
        if interfaces.iter().any(|i| i.name.contains("eth")) {
            100.0 // Assume 100 Mbps Ethernet
        } else if interfaces
            .iter()
            .any(|i| i.name.contains("wlan") || i.name.contains("wifi"))
        {
            50.0 // Assume 50 Mbps WiFi
        } else {
            10.0 // Conservative default
        }
    }

    fn detect_connectivity_type(&self, interfaces: &[NetworkInterface]) -> ConnectivityType {
        let has_ethernet = interfaces.iter().any(|i| i.name.contains("eth"));
        let has_wifi = interfaces
            .iter()
            .any(|i| i.name.contains("wlan") || i.name.contains("wifi"));

        match (has_ethernet, has_wifi) {
            (true, true) => ConnectivityType::Mixed,
            (true, false) => ConnectivityType::Ethernet,
            (false, true) => ConnectivityType::WiFi,
            (false, false) => ConnectivityType::Unknown,
        }
    }
}

impl Default for HardwareProfiler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// TDD Phase tracking
    #[derive(Debug, Clone, PartialEq)]
    enum TddPhase {
        Red,      // Write failing tests
        Green,    // Make tests pass
        Refactor, // Optimize implementation
    }

    #[test]
    fn test_hardware_profiler_creation() {
        let _profiler = HardwareProfiler::new();
        // Should not panic
        assert!(true);
    }

    #[test]
    fn test_cpu_info_serialization() {
        let cpu_info = CpuInfo {
            model: "Intel Core i7-9700K".to_string(),
            cores: 8,
            threads: 8,
            frequency_mhz: 3600,
            architecture: "x86_64".to_string(),
            features: vec!["sse".to_string(), "avx".to_string()],
        };

        // Test serialization
        let json = serde_json::to_string(&cpu_info).expect("Should serialize");
        assert!(json.contains("Intel Core i7-9700K"));
        assert!(json.contains("3600"));

        // Test deserialization
        let deserialized: CpuInfo = serde_json::from_str(&json).expect("Should deserialize");
        assert_eq!(deserialized.model, cpu_info.model);
        assert_eq!(deserialized.cores, 8);
    }

    #[test]
    fn test_memory_info_serialization() {
        let memory_info = MemoryInfo {
            total_gb: 16.0,
            available_gb: 8.5,
            swap_gb: 4.0,
        };

        let json = serde_json::to_string(&memory_info).expect("Should serialize");
        let deserialized: MemoryInfo = serde_json::from_str(&json).expect("Should deserialize");

        assert_eq!(deserialized.total_gb, 16.0);
        assert_eq!(deserialized.available_gb, 8.5);
        assert_eq!(deserialized.swap_gb, 4.0);
    }

    #[test]
    fn test_device_type_classification() {
        let profiler = HardwareProfiler::new();

        // Test Raspberry Pi detection
        let rpi_cpu = CpuInfo {
            model: "ARM Cortex-A72".to_string(),
            cores: 4,
            threads: 4,
            frequency_mhz: 1500,
            architecture: "aarch64".to_string(),
            features: vec![],
        };
        let rpi_memory = MemoryInfo {
            total_gb: 4.0,
            available_gb: 3.0,
            swap_gb: 0.0,
        };
        let device_type = profiler.classify_device(&rpi_cpu, &rpi_memory, &None, &None);
        assert_eq!(device_type, DeviceType::RaspberryPi);

        // Test Server detection
        let server_cpu = CpuInfo {
            model: "Intel Xeon E5-2686".to_string(),
            cores: 32,
            threads: 64,
            frequency_mhz: 2700,
            architecture: "x86_64".to_string(),
            features: vec![],
        };
        let server_memory = MemoryInfo {
            total_gb: 128.0,
            available_gb: 100.0,
            swap_gb: 16.0,
        };
        let device_type = profiler.classify_device(&server_cpu, &server_memory, &None, &None);
        assert_eq!(device_type, DeviceType::Server);

        // Test Workstation detection with GPU
        let workstation_cpu = CpuInfo {
            model: "Intel Core i9-10900K".to_string(),
            cores: 10,
            threads: 20,
            frequency_mhz: 3700,
            architecture: "x86_64".to_string(),
            features: vec![],
        };
        let workstation_memory = MemoryInfo {
            total_gb: 32.0,
            available_gb: 24.0,
            swap_gb: 8.0,
        };
        let gpu_info = Some(GpuInfo {
            count: 1,
            models: vec!["NVIDIA RTX 3090".to_string()],
            total_memory_gb: 24.0,
            compute_capability: Some("8.6".to_string()),
        });
        let device_type =
            profiler.classify_device(&workstation_cpu, &workstation_memory, &gpu_info, &None);
        assert_eq!(device_type, DeviceType::Workstation);

        // Test IoT device detection
        let iot_cpu = CpuInfo {
            model: "ESP32".to_string(),
            cores: 1,
            threads: 1,
            frequency_mhz: 80,
            architecture: "arm".to_string(),
            features: vec![],
        };
        let iot_memory = MemoryInfo {
            total_gb: 0.5,
            available_gb: 0.3,
            swap_gb: 0.0,
        };
        let device_type = profiler.classify_device(&iot_cpu, &iot_memory, &None, &None);
        assert_eq!(device_type, DeviceType::IoTDevice);
    }

    #[test]
    fn test_disk_type_detection() {
        let profiler = HardwareProfiler::new();

        assert_eq!(profiler.detect_disk_type("nvme0n1"), "NVMe");
        assert_eq!(profiler.detect_disk_type("/dev/ssd1"), "SSD");
        assert_eq!(profiler.detect_disk_type("sda"), "SD Card");
        assert_eq!(profiler.detect_disk_type("hda"), "HDD");
        assert_eq!(profiler.detect_disk_type("vda"), "HDD");
    }

    #[test]
    fn test_connectivity_type_detection() {
        let profiler = HardwareProfiler::new();

        // Test Ethernet only
        let eth_interfaces = vec![NetworkInterface {
            name: "eth0".to_string(),
            ip_addresses: vec![],
            mac_address: None,
            is_up: true,
            speed_mbps: None,
        }];
        assert_eq!(
            profiler.detect_connectivity_type(&eth_interfaces),
            ConnectivityType::Ethernet
        );

        // Test WiFi only
        let wifi_interfaces = vec![NetworkInterface {
            name: "wlan0".to_string(),
            ip_addresses: vec![],
            mac_address: None,
            is_up: true,
            speed_mbps: None,
        }];
        assert_eq!(
            profiler.detect_connectivity_type(&wifi_interfaces),
            ConnectivityType::WiFi
        );

        // Test Mixed
        let mixed_interfaces = vec![
            NetworkInterface {
                name: "eth0".to_string(),
                ip_addresses: vec![],
                mac_address: None,
                is_up: true,
                speed_mbps: None,
            },
            NetworkInterface {
                name: "wlan0".to_string(),
                ip_addresses: vec![],
                mac_address: None,
                is_up: true,
                speed_mbps: None,
            },
        ];
        assert_eq!(
            profiler.detect_connectivity_type(&mixed_interfaces),
            ConnectivityType::Mixed
        );
    }

    #[test]
    fn test_bandwidth_estimation() {
        let profiler = HardwareProfiler::new();

        // Test Ethernet
        let eth_interfaces = vec![NetworkInterface {
            name: "eth0".to_string(),
            ip_addresses: vec![],
            mac_address: None,
            is_up: true,
            speed_mbps: None,
        }];
        assert_eq!(profiler.estimate_bandwidth(&eth_interfaces), 100.0);

        // Test WiFi
        let wifi_interfaces = vec![NetworkInterface {
            name: "wlan0".to_string(),
            ip_addresses: vec![],
            mac_address: None,
            is_up: true,
            speed_mbps: None,
        }];
        assert_eq!(profiler.estimate_bandwidth(&wifi_interfaces), 50.0);

        // Test default
        let other_interfaces = vec![NetworkInterface {
            name: "lo".to_string(),
            ip_addresses: vec![],
            mac_address: None,
            is_up: true,
            speed_mbps: None,
        }];
        assert_eq!(profiler.estimate_bandwidth(&other_interfaces), 10.0);
    }

    #[test]
    fn test_calculate_capabilities() {
        let profiler = HardwareProfiler::new();

        let cpu = CpuInfo {
            model: "Test CPU".to_string(),
            cores: 8,
            threads: 16,
            frequency_mhz: 3000,
            architecture: "x86_64".to_string(),
            features: vec![],
        };

        let memory = MemoryInfo {
            total_gb: 16.0,
            available_gb: 12.0,
            swap_gb: 4.0,
        };

        let storage = StorageInfo {
            total_gb: 500.0,
            available_gb: 400.0,
            disks: vec![],
        };

        let network = NetworkInfo {
            interfaces: vec![],
            estimated_bandwidth_mbps: 100.0,
            connectivity: ConnectivityType::Ethernet,
        };

        let capabilities = profiler.calculate_capabilities(
            &cpu,
            &memory,
            &storage,
            &network,
            &None,
            &DeviceType::Workstation,
        );

        // Check conservative resource allocation
        assert!(capabilities.memory_limit_gb <= memory.available_gb * 0.8);
        assert!(capabilities.cpu_limit_cores <= cpu.cores);
        assert!(capabilities.storage_limit_gb <= storage.available_gb * 0.5);
        assert_eq!(capabilities.network_bandwidth_mbps, 100.0);
        assert!(!capabilities.gpu_capable);
        assert!(capabilities.container_runtime);
    }

    #[test]
    fn test_workload_suitability() {
        let profiler = HardwareProfiler::new();

        // Test server suitability
        let server_suitability = profiler.determine_suitability(&DeviceType::Server, false, 64.0);
        assert!(server_suitability.contains(&WorkloadType::Compute));
        assert!(server_suitability.contains(&WorkloadType::Storage));
        assert!(server_suitability.contains(&WorkloadType::Cache));

        // Test Raspberry Pi suitability
        let rpi_suitability = profiler.determine_suitability(&DeviceType::RaspberryPi, false, 2.0);
        assert!(rpi_suitability.contains(&WorkloadType::Sensor));
        assert!(rpi_suitability.contains(&WorkloadType::Gateway));
        assert!(rpi_suitability.contains(&WorkloadType::Monitoring));

        // Test GPU-enabled suitability
        let gpu_suitability = profiler.determine_suitability(&DeviceType::Workstation, true, 16.0);
        assert!(gpu_suitability.contains(&WorkloadType::Compute));
        assert!(gpu_suitability.contains(&WorkloadType::Development));
        assert!(gpu_suitability.contains(&WorkloadType::Cache)); // Has enough memory
    }

    #[test]
    fn test_node_capabilities_serialization() {
        let capabilities = NodeCapabilities {
            max_workloads: 10,
            memory_limit_gb: 8.0,
            cpu_limit_cores: 4,
            storage_limit_gb: 100.0,
            gpu_capable: true,
            container_runtime: true,
            network_bandwidth_mbps: 1000.0,
            suitability: vec![WorkloadType::Compute, WorkloadType::Storage],
        };

        let json = serde_json::to_string(&capabilities).expect("Should serialize");
        let deserialized: NodeCapabilities =
            serde_json::from_str(&json).expect("Should deserialize");

        assert_eq!(deserialized.max_workloads, 10);
        assert_eq!(deserialized.gpu_capable, true);
        assert_eq!(deserialized.suitability.len(), 2);
    }

    #[tokio::test]
    async fn test_profile_generation() {
        let mut profiler = HardwareProfiler::new();

        // This test actually generates a profile for the current system
        let result = profiler.profile().await;

        // Should succeed on any system
        assert!(result.is_ok());

        let profile = result.unwrap();

        // Basic validations
        assert!(!profile.hostname.is_empty());
        assert!(profile.cpu.cores > 0);
        assert!(profile.memory.total_gb > 0.0);
        assert!(profile.storage.total_gb > 0.0);
        assert!(!profile.network.interfaces.is_empty());

        // Capabilities should be set
        assert!(profile.capabilities.max_workloads > 0);
        assert!(profile.capabilities.memory_limit_gb > 0.0);
        assert!(profile.capabilities.cpu_limit_cores > 0);
    }
}
