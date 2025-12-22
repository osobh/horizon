//! Additional comprehensive tests for classification module

#[cfg(test)]
mod classification_coverage_tests {
    use super::super::*;
    use crate::classification::*;
    use crate::discovery::{
        GpuInfo, HardwareProfile, NatType, NetworkCharacteristics, PowerPlan, PowerProfile,
    };

    #[test]
    fn test_bandwidth_variants() {
        let bandwidths = vec![
            Bandwidth::Gigabit(1.0),
            Bandwidth::TenGigabit(10.0),
            Bandwidth::FortyGigabit(40.0),
            Bandwidth::HundredGigabit(100.0),
            Bandwidth::Custom(25.0),
        ];

        for bw in bandwidths {
            let json = serde_json::to_string(&bw).unwrap();
            let restored: Bandwidth = serde_json::from_str(&json).unwrap();
            assert_eq!(bw, restored);
        }
    }

    #[test]
    fn test_schedule_variants() {
        let schedules = vec![
            Schedule::AlwaysOn,
            Schedule::BusinessHours,
            Schedule::AfterHours,
            Schedule::Weekend,
            Schedule::Custom("MWF 9-5".to_string()),
        ];

        for schedule in schedules {
            let json = serde_json::to_string(&schedule).unwrap();
            let restored: Schedule = serde_json::from_str(&json).unwrap();
            assert_eq!(schedule, restored);
        }
    }

    #[test]
    fn test_mobility_patterns() {
        let patterns = vec![
            MobilityPattern::Stationary,
            MobilityPattern::Occasional,
            MobilityPattern::Frequent,
            MobilityPattern::Continuous,
        ];

        for pattern in patterns {
            let json = serde_json::to_string(&pattern).unwrap();
            let restored: MobilityPattern = serde_json::from_str(&json).unwrap();
            assert_eq!(pattern, restored);
        }
    }

    #[test]
    fn test_edge_device_types() {
        let devices = vec![
            EdgeType::RaspberryPi,
            EdgeType::JetsonNano,
            EdgeType::JetsonXavier,
            EdgeType::IntelNuc,
            EdgeType::Other("Custom Board".to_string()),
        ];

        for device in devices {
            let json = serde_json::to_string(&device).unwrap();
            let restored: EdgeType = serde_json::from_str(&json).unwrap();
            assert_eq!(device, restored);
        }
    }

    #[test]
    fn test_watts_wrapper() {
        let watts = Watts(15.5);
        assert_eq!(watts.0, 15.5);

        let json = serde_json::to_string(&watts).unwrap();
        let restored: Watts = serde_json::from_str(&json).unwrap();
        assert_eq!(watts.0, restored.0);
    }

    #[test]
    fn test_node_class_variants() {
        // DataCenter node
        let dc = NodeClass::DataCenter {
            gpus: vec![GpuInfo {
                index: 0,
                name: "A100".to_string(),
                memory_mb: 40960,
                compute_capability: (8, 0),
                pci_bus_id: "0:0:0.0".to_string(),
            }],
            bandwidth: Bandwidth::HundredGigabit(100.0),
        };

        // Workstation node
        let ws = NodeClass::Workstation {
            gpu: Some(GpuInfo {
                index: 0,
                name: "RTX 4090".to_string(),
                memory_mb: 24576,
                compute_capability: (8, 9),
                pci_bus_id: "1:0:0.0".to_string(),
            }),
            schedule: Schedule::BusinessHours,
        };

        // Laptop node
        let laptop = NodeClass::Laptop {
            battery: true,
            mobility: MobilityPattern::Frequent,
        };

        // Edge node
        let edge = NodeClass::Edge {
            device_type: EdgeType::RaspberryPi,
            power_budget: Watts(10.0),
        };

        // Test serialization for all
        for node_class in vec![dc, ws, laptop, edge] {
            let json = serde_json::to_string(&node_class).unwrap();
            let restored: NodeClass = serde_json::from_str(&json).unwrap();
            assert_eq!(node_class, restored);
        }
    }

    #[test]
    fn test_node_classifier_new() {
        let classifier = NodeClassifier::new();
        // Just verify it can be created
        assert_eq!(
            std::mem::size_of_val(&classifier),
            std::mem::size_of::<NodeClassifier>()
        );
    }

    fn create_test_hardware(cores: u32, memory: f32, gpus: Vec<GpuInfo>) -> HardwareProfile {
        HardwareProfile {
            cpu_model: "Test CPU".to_string(),
            cpu_cores: cores,
            memory_gb: memory,
            storage_gb: 1000.0,
            gpus,
        }
    }

    fn create_test_network(bandwidth: f32) -> NetworkCharacteristics {
        NetworkCharacteristics {
            bandwidth_mbps: bandwidth,
            latency_ms: 1.0,
            jitter_ms: 0.1,
            packet_loss: 0.0,
            nat_type: NatType::None,
        }
    }

    fn create_test_power(on_battery: bool, percentage: Option<u8>) -> PowerProfile {
        PowerProfile {
            on_battery,
            battery_percentage: percentage,
            power_plan: PowerPlan::Balanced,
            estimated_runtime_minutes: percentage.map(|p| (p as u32) * 5),
        }
    }

    #[test]
    fn test_classification_datacenter() {
        let classifier = NodeClassifier::new();

        let hardware = create_test_hardware(
            128,
            1024.0,
            vec![
                GpuInfo {
                    index: 0,
                    name: "NVIDIA A100".to_string(),
                    memory_mb: 40960,
                    compute_capability: (8, 0),
                    pci_bus_id: "0:1:0.0".to_string(),
                },
                GpuInfo {
                    index: 1,
                    name: "NVIDIA A100".to_string(),
                    memory_mb: 40960,
                    compute_capability: (8, 0),
                    pci_bus_id: "0:2:0.0".to_string(),
                },
            ],
        );
        let network = create_test_network(100_000.0); // 100 Gbps
        let power = create_test_power(false, None);

        let class = classifier.classify(&hardware, &network, &power);
        assert!(matches!(class, NodeClass::DataCenter { .. }));

        if let NodeClass::DataCenter { gpus, bandwidth } = class {
            assert_eq!(gpus.len(), 2);
            assert!(matches!(bandwidth, Bandwidth::HundredGigabit(_)));
        }
    }

    #[test]
    fn test_classification_workstation() {
        let classifier = NodeClassifier::new();

        let hardware = create_test_hardware(
            16,
            64.0,
            vec![GpuInfo {
                index: 0,
                name: "RTX 4090".to_string(),
                memory_mb: 24576,
                compute_capability: (8, 9),
                pci_bus_id: "0:1:0.0".to_string(),
            }],
        );
        let network = create_test_network(1000.0); // 1 Gbps
        let power = create_test_power(false, None);

        let class = classifier.classify(&hardware, &network, &power);
        assert!(matches!(class, NodeClass::Workstation { .. }));

        if let NodeClass::Workstation { gpu, schedule } = class {
            assert!(gpu.is_some());
            assert_eq!(schedule, Schedule::AlwaysOn);
        }
    }

    #[test]
    fn test_classification_laptop() {
        let classifier = NodeClassifier::new();

        let hardware = create_test_hardware(8, 16.0, vec![]);
        let network = create_test_network(100.0);
        let power = create_test_power(true, Some(85));

        let class = classifier.classify(&hardware, &network, &power);
        assert!(matches!(class, NodeClass::Laptop { .. }));

        if let NodeClass::Laptop { battery, mobility } = class {
            assert!(battery);
            // Default mobility pattern
            assert_eq!(mobility, MobilityPattern::Occasional);
        }
    }

    #[test]
    fn test_classification_edge_raspberry_pi() {
        let classifier = NodeClassifier::new();

        let hardware = HardwareProfile {
            cpu_model: "ARM Cortex-A72".to_string(),
            cpu_cores: 4,
            memory_gb: 4.0,
            storage_gb: 32.0,
            gpus: vec![],
        };
        let network = create_test_network(100.0);
        let power = create_test_power(false, None);

        let class = classifier.classify(&hardware, &network, &power);
        assert!(matches!(class, NodeClass::Edge { .. }));

        if let NodeClass::Edge {
            device_type,
            power_budget,
        } = class
        {
            assert!(matches!(device_type, EdgeType::RaspberryPi));
            assert_eq!(power_budget.0, 5.0); // Default Pi power
        }
    }

    #[test]
    fn test_classification_edge_jetson() {
        let classifier = NodeClassifier::new();

        let hardware = HardwareProfile {
            cpu_model: "NVIDIA Carmel".to_string(),
            cpu_cores: 8,
            memory_gb: 32.0,
            storage_gb: 128.0,
            gpus: vec![],
        };
        let network = create_test_network(1000.0);
        let power = create_test_power(false, None);

        let class = classifier.classify(&hardware, &network, &power);
        assert!(matches!(class, NodeClass::Edge { .. }));

        if let NodeClass::Edge {
            device_type,
            power_budget,
        } = class
        {
            assert!(matches!(device_type, EdgeType::JetsonXavier));
            assert_eq!(power_budget.0, 30.0); // Xavier power
        }
    }

    #[test]
    fn test_classification_fallback() {
        let classifier = NodeClassifier::new();

        // Minimal hardware that doesn't fit any category well
        let hardware = create_test_hardware(2, 2.0, vec![]);
        let network = create_test_network(10.0);
        let power = create_test_power(false, None);

        let class = classifier.classify(&hardware, &network, &power);
        assert!(matches!(class, NodeClass::Edge { .. }));

        if let NodeClass::Edge { device_type, .. } = class {
            assert!(matches!(device_type, EdgeType::Other(_)));
        }
    }

    #[test]
    fn test_bandwidth_classification() {
        let classifier = NodeClassifier::new();

        // Test different bandwidth levels
        let test_cases = vec![
            (500.0, Bandwidth::Custom(0.5)),               // < 1 Gbps
            (1000.0, Bandwidth::Gigabit(1.0)),             // 1 Gbps
            (10_000.0, Bandwidth::TenGigabit(10.0)),       // 10 Gbps
            (40_000.0, Bandwidth::FortyGigabit(40.0)),     // 40 Gbps
            (100_000.0, Bandwidth::HundredGigabit(100.0)), // 100 Gbps
            (200_000.0, Bandwidth::HundredGigabit(100.0)), // > 100 Gbps caps at 100
        ];

        for (mbps, expected) in test_cases {
            let bw = classifier.classify_bandwidth(mbps);
            assert_eq!(bw, expected);
        }
    }

    #[test]
    fn test_edge_device_detection() {
        let classifier = NodeClassifier::new();

        let test_cases = vec![
            ("ARM Cortex-A72", 4, 4.0, EdgeType::RaspberryPi),
            ("ARM Cortex-A57", 4, 4.0, EdgeType::JetsonNano),
            ("NVIDIA Carmel", 8, 32.0, EdgeType::JetsonXavier),
            ("Intel Celeron", 4, 8.0, EdgeType::IntelNuc),
            (
                "Unknown ARM",
                2,
                1.0,
                EdgeType::Other("Unknown ARM".to_string()),
            ),
        ];

        for (cpu, cores, memory, expected) in test_cases {
            let hardware = HardwareProfile {
                cpu_model: cpu.to_string(),
                cpu_cores: cores,
                memory_gb: memory,
                storage_gb: 32.0,
                gpus: vec![],
            };

            let device = classifier.classify_edge_device(&hardware);
            assert_eq!(device, expected);
        }
    }

    #[test]
    fn test_edge_power_budget() {
        let classifier = NodeClassifier::new();

        let test_cases = vec![
            (EdgeType::RaspberryPi, 5.0),
            (EdgeType::JetsonNano, 10.0),
            (EdgeType::JetsonXavier, 30.0),
            (EdgeType::IntelNuc, 65.0),
            (EdgeType::Other("Custom".to_string()), 15.0),
        ];

        for (device_type, expected_watts) in test_cases {
            let power = classifier.estimate_edge_power(&device_type);
            assert_eq!(power.0, expected_watts);
        }
    }

    #[test]
    fn test_workstation_no_gpu() {
        let classifier = NodeClassifier::new();

        let hardware = create_test_hardware(32, 128.0, vec![]);
        let network = create_test_network(1000.0);
        let power = create_test_power(false, None);

        let class = classifier.classify(&hardware, &network, &power);
        assert!(matches!(class, NodeClass::Workstation { .. }));

        if let NodeClass::Workstation { gpu, .. } = class {
            assert!(gpu.is_none());
        }
    }

    #[test]
    fn test_node_class_equality() {
        let gpu1 = GpuInfo {
            index: 0,
            name: "RTX 4090".to_string(),
            memory_mb: 24576,
            compute_capability: (8, 9),
            pci_bus_id: "0:1:0.0".to_string(),
        };

        let gpu2 = gpu1.clone();

        let class1 = NodeClass::Workstation {
            gpu: Some(gpu1),
            schedule: Schedule::AlwaysOn,
        };

        let class2 = NodeClass::Workstation {
            gpu: Some(gpu2),
            schedule: Schedule::AlwaysOn,
        };

        assert_eq!(class1, class2);
    }
}
