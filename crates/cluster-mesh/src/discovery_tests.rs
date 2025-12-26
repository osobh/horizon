//! Additional comprehensive tests for discovery module

#[cfg(test)]
mod discovery_coverage_tests {
    use super::super::*;
    use crate::discovery::*;

    #[test]
    fn test_hardware_profile_default() {
        let profile = HardwareProfile::default();
        assert!(profile.cpu_model.is_empty());
        assert_eq!(profile.cpu_cores, 0);
        assert_eq!(profile.memory_gb, 0.0);
        assert_eq!(profile.storage_gb, 0.0);
        assert!(profile.gpus.is_empty());
    }

    #[test]
    fn test_gpu_info_serialization() {
        let gpu = GpuInfo {
            index: 0,
            name: "NVIDIA RTX 4090".to_string(),
            memory_mb: 24576,
            compute_capability: (8, 9),
            pci_bus_id: "0000:01:00.0".to_string(),
        };

        let serialized = serde_json::to_string(&gpu).unwrap();
        let deserialized: GpuInfo = serde_json::from_str(&serialized).unwrap();

        assert_eq!(gpu.index, deserialized.index);
        assert_eq!(gpu.name, deserialized.name);
        assert_eq!(gpu.memory_mb, deserialized.memory_mb);
        assert_eq!(gpu.compute_capability, deserialized.compute_capability);
        assert_eq!(gpu.pci_bus_id, deserialized.pci_bus_id);
    }

    #[test]
    fn test_network_characteristics_validation() {
        let network = NetworkCharacteristics {
            bandwidth_mbps: 1000.0,
            latency_ms: 0.5,
            jitter_ms: 0.1,
            packet_loss: 0.01,
            nat_type: NatType::RestrictedCone,
        };

        // Test serialization/deserialization
        let json = serde_json::to_string(&network).unwrap();
        let restored: NetworkCharacteristics = serde_json::from_str(&json).unwrap();

        assert_eq!(network.bandwidth_mbps, restored.bandwidth_mbps);
        assert_eq!(network.latency_ms, restored.latency_ms);
        assert_eq!(network.jitter_ms, restored.jitter_ms);
        assert_eq!(network.packet_loss, restored.packet_loss);
        assert_eq!(network.nat_type, restored.nat_type);
    }

    #[test]
    fn test_power_profile_scenarios() {
        // Test battery-powered scenario
        let battery_profile = PowerProfile {
            on_battery: true,
            battery_percentage: Some(75),
            power_plan: PowerPlan::PowerSaver,
            estimated_runtime_minutes: Some(180),
        };

        assert!(battery_profile.on_battery);
        assert_eq!(battery_profile.battery_percentage, Some(75));
        assert_eq!(battery_profile.power_plan, PowerPlan::PowerSaver);
        assert_eq!(battery_profile.estimated_runtime_minutes, Some(180));

        // Test AC-powered scenario
        let ac_profile = PowerProfile {
            on_battery: false,
            battery_percentage: None,
            power_plan: PowerPlan::Performance,
            estimated_runtime_minutes: None,
        };

        assert!(!ac_profile.on_battery);
        assert!(ac_profile.battery_percentage.is_none());
        assert_eq!(ac_profile.power_plan, PowerPlan::Performance);
        assert!(ac_profile.estimated_runtime_minutes.is_none());
    }

    #[test]
    fn test_power_plan_variants() {
        let plans = vec![
            PowerPlan::Performance,
            PowerPlan::Balanced,
            PowerPlan::PowerSaver,
            PowerPlan::Unknown,
        ];

        for plan in plans {
            let json = serde_json::to_string(&plan).unwrap();
            let restored: PowerPlan = serde_json::from_str(&json).unwrap();
            assert_eq!(plan, restored);
        }
    }

    #[test]
    fn test_all_nat_types() {
        use std::collections::HashSet;

        let nat_types = vec![
            NatType::None,
            NatType::FullCone,
            NatType::RestrictedCone,
            NatType::PortRestricted,
            NatType::Symmetric,
        ];

        // Test uniqueness
        let unique: HashSet<_> = nat_types.iter().collect();
        assert_eq!(unique.len(), nat_types.len());

        // Test each type
        for nat in &nat_types {
            let json = serde_json::to_string(nat).unwrap();
            let restored: NatType = serde_json::from_str(&json).unwrap();
            assert_eq!(nat, &restored);
        }
    }

    #[tokio::test]
    async fn test_discovery_with_custom_interval() {
        let discovery = NodeDiscovery::new().await.unwrap();
        assert_eq!(discovery.discovery_interval.as_secs(), 30);
    }

    #[test]
    fn test_hardware_profile_with_multiple_gpus() {
        let profile = HardwareProfile {
            cpu_model: "AMD EPYC 7763".to_string(),
            cpu_cores: 64,
            memory_gb: 512.0,
            storage_gb: 8000.0,
            gpus: vec![
                GpuInfo {
                    index: 0,
                    name: "NVIDIA A100".to_string(),
                    memory_mb: 40960,
                    compute_capability: (8, 0),
                    pci_bus_id: "0000:01:00.0".to_string(),
                },
                GpuInfo {
                    index: 1,
                    name: "NVIDIA A100".to_string(),
                    memory_mb: 40960,
                    compute_capability: (8, 0),
                    pci_bus_id: "0000:02:00.0".to_string(),
                },
            ],
        };

        assert_eq!(profile.gpus.len(), 2);
        assert!(profile.gpus.iter().all(|g| g.memory_mb == 40960));
        assert!(profile.gpus.iter().all(|g| g.compute_capability == (8, 0)));
    }

    #[test]
    fn test_network_edge_cases() {
        // Test zero values
        let zero_network = NetworkCharacteristics {
            bandwidth_mbps: 0.0,
            latency_ms: 0.0,
            jitter_ms: 0.0,
            packet_loss: 0.0,
            nat_type: NatType::None,
        };

        assert_eq!(zero_network.bandwidth_mbps, 0.0);
        assert_eq!(zero_network.packet_loss, 0.0);

        // Test high values
        let high_network = NetworkCharacteristics {
            bandwidth_mbps: 100_000.0, // 100 Gbps
            latency_ms: 1000.0,        // 1 second
            jitter_ms: 100.0,          // High jitter
            packet_loss: 50.0,         // 50% loss
            nat_type: NatType::Symmetric,
        };

        assert_eq!(high_network.bandwidth_mbps, 100_000.0);
        assert_eq!(high_network.packet_loss, 50.0);
    }

    #[test]
    fn test_power_profile_edge_cases() {
        // Test critical battery
        let critical = PowerProfile {
            on_battery: true,
            battery_percentage: Some(1),
            power_plan: PowerPlan::PowerSaver,
            estimated_runtime_minutes: Some(5),
        };

        assert_eq!(critical.battery_percentage, Some(1));
        assert_eq!(critical.estimated_runtime_minutes, Some(5));

        // Test full battery
        let full = PowerProfile {
            on_battery: true,
            battery_percentage: Some(100),
            power_plan: PowerPlan::Balanced,
            estimated_runtime_minutes: Some(600), // 10 hours
        };

        assert_eq!(full.battery_percentage, Some(100));
        assert_eq!(full.estimated_runtime_minutes, Some(600));
    }

    #[tokio::test]
    async fn test_network_interface_placeholder() {
        let mut discovery = NodeDiscovery::new().await.unwrap();
        let result = discovery.update_network_interfaces().await;
        assert!(result.is_ok());

        // Verify placeholder interface was added
        assert_eq!(discovery.network_interfaces.len(), 1);
        assert!(discovery.network_interfaces.contains_key("eth0"));
    }

    #[tokio::test]
    async fn test_bandwidth_estimation() {
        let discovery = NodeDiscovery::new().await.unwrap();
        let bandwidth = discovery.estimate_bandwidth().await.unwrap();

        // Should return reasonable default
        assert_eq!(bandwidth, 100.0);
        assert!(bandwidth > 0.0);
    }

    #[tokio::test]
    async fn test_latency_measurement() {
        let discovery = NodeDiscovery::new().await.unwrap();
        let (latency, jitter) = discovery.measure_latency().await.unwrap();

        assert_eq!(latency, 10.0);
        assert_eq!(jitter, 1.0);
        assert!(latency >= 0.0);
        assert!(jitter >= 0.0);
    }

    #[tokio::test]
    async fn test_nat_detection() {
        let discovery = NodeDiscovery::new().await.unwrap();
        let nat_type = discovery.detect_nat_type().await.unwrap();

        // Should return default FullCone
        assert_eq!(nat_type, NatType::FullCone);
    }

    #[tokio::test]
    async fn test_power_detection_ac() {
        let discovery = NodeDiscovery::new().await.unwrap();

        // Test AC power detection
        let on_battery = discovery.is_on_battery().await.unwrap();
        assert!(!on_battery); // Default is false (AC power)

        let battery_pct = discovery.get_battery_percentage().await.unwrap();
        assert!(battery_pct.is_none());

        let runtime = discovery.estimate_battery_runtime().await.unwrap();
        assert!(runtime.is_none());
    }

    #[tokio::test]
    async fn test_complete_profiling_workflow() {
        let mut discovery = NodeDiscovery::new().await.unwrap();

        // Profile all aspects
        let hardware = discovery.profile_hardware().await;
        assert!(hardware.is_ok());

        let network = discovery.profile_network().await;
        assert!(network.is_ok());

        let power = discovery.detect_power_profile().await;
        assert!(power.is_ok());

        // Verify results are consistent
        let hw = hardware.unwrap();
        assert!(hw.cpu_cores > 0);

        let net = network.unwrap();
        assert!(net.bandwidth_mbps > 0.0);

        let pwr = power.unwrap();
        if pwr.on_battery {
            assert!(pwr.battery_percentage.is_some());
        }
    }
}
