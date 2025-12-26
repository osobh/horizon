//! Additional tests for distribution module

#[cfg(test)]
mod distribution_coverage_tests {
    use super::super::*;
    use crate::discovery::{HardwareProfile, NatType, NetworkCharacteristics};
    use crate::{
        classification::{Bandwidth, MobilityPattern, NodeClass, Schedule},
        JobPriority, ThermalConstraints,
    };
    use chrono::Utc;
    use uuid::Uuid;

    fn create_test_hardware(cores: u32, memory: f32) -> HardwareProfile {
        HardwareProfile {
            cpu_model: "Test CPU".to_string(),
            cpu_cores: cores,
            memory_gb: memory,
            storage_gb: 1000.0,
            gpus: vec![],
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

    fn create_test_node_with_class(
        id: Uuid,
        class: NodeClass,
        caps: NodeCapabilities,
    ) -> ClusterNode {
        ClusterNode {
            id,
            hostname: format!("node-{}", id),
            class,
            hardware: HardwareProfile::default(),
            network: NetworkCharacteristics {
                bandwidth_mbps: 1000.0,
                latency_ms: 1.0,
                jitter_ms: 0.1,
                packet_loss: 0.0,
                nat_type: NatType::None,
            },
            status: NodeStatus::Online,
            capabilities: caps,
            last_heartbeat: Utc::now(),
        }
    }

    #[tokio::test]
    async fn test_node_affinity() {
        let distributor = WorkDistributor::new();
        let node_id = Uuid::new_v4();

        let node = create_test_node_with_class(
            node_id,
            NodeClass::DataCenter {
                gpus: vec![],
                bandwidth: Bandwidth::TenGigabit(10.0),
            },
            NodeCapabilities {
                cpu_cores: 32,
                memory_gb: 128.0,
                gpu_count: 0,
                gpu_memory_gb: None,
                storage_gb: 1000.0,
                network_bandwidth_mbps: 10000.0,
                supports_gpu_direct: false,
                battery_powered: false,
                thermal_constraints: None,
            },
        );

        // Test RequiredClass affinity
        let job = Job {
            id: Uuid::new_v4(),
            name: "DC Job".to_string(),
            requirements: JobRequirements {
                cpu_cores: Some(4),
                memory_gb: Some(8.0),
                gpu_count: None,
                gpu_memory_gb: None,
                storage_gb: None,
                network_bandwidth_mbps: None,
                requires_gpu_direct: false,
                node_affinity: Some(NodeAffinity::RequiredClass(NodeClass::DataCenter {
                    gpus: vec![],
                    bandwidth: Bandwidth::TenGigabit(10.0),
                })),
                anti_affinity: None,
                locality_preference: LocalityPreference::None,
                max_latency_ms: None,
                battery_safe: true,
            },
            priority: JobPriority::Normal,
            submitted_at: Utc::now(),
        };

        assert!(distributor
            .meets_requirements(&job.requirements, &node.capabilities, &node)
            .unwrap());

        // Test RequiredNodes affinity
        let job_with_node_affinity = Job {
            id: Uuid::new_v4(),
            name: "Node Affinity Job".to_string(),
            requirements: JobRequirements {
                cpu_cores: Some(4),
                memory_gb: Some(8.0),
                gpu_count: None,
                gpu_memory_gb: None,
                storage_gb: None,
                network_bandwidth_mbps: None,
                requires_gpu_direct: false,
                node_affinity: Some(NodeAffinity::RequiredNodes(vec![node_id])),
                anti_affinity: None,
                locality_preference: LocalityPreference::None,
                max_latency_ms: None,
                battery_safe: true,
            },
            priority: JobPriority::Normal,
            submitted_at: Utc::now(),
        };

        assert!(distributor
            .meets_requirements(
                &job_with_node_affinity.requirements,
                &node.capabilities,
                &node
            )
            .unwrap());

        // Test anti-affinity
        let job_with_anti_affinity = Job {
            id: Uuid::new_v4(),
            name: "Anti-Affinity Job".to_string(),
            requirements: JobRequirements {
                cpu_cores: Some(4),
                memory_gb: Some(8.0),
                gpu_count: None,
                gpu_memory_gb: None,
                storage_gb: None,
                network_bandwidth_mbps: None,
                requires_gpu_direct: false,
                node_affinity: None,
                anti_affinity: Some(vec![node_id]),
                locality_preference: LocalityPreference::None,
                max_latency_ms: None,
                battery_safe: true,
            },
            priority: JobPriority::Normal,
            submitted_at: Utc::now(),
        };

        assert!(!distributor
            .meets_requirements(
                &job_with_anti_affinity.requirements,
                &node.capabilities,
                &node
            )
            .unwrap());
    }

    #[tokio::test]
    async fn test_scheduling_policies_coverage() {
        let distributor = WorkDistributor::new();

        // Create diverse nodes
        let nodes = vec![
            create_test_node_with_class(
                Uuid::new_v4(),
                NodeClass::DataCenter {
                    gpus: vec![],
                    bandwidth: Bandwidth::TenGigabit(10.0),
                },
                NodeCapabilities {
                    cpu_cores: 64,
                    memory_gb: 256.0,
                    gpu_count: 0,
                    gpu_memory_gb: None,
                    storage_gb: 10000.0,
                    network_bandwidth_mbps: 10000.0,
                    supports_gpu_direct: false,
                    battery_powered: false,
                    thermal_constraints: None,
                },
            ),
            create_test_node_with_class(
                Uuid::new_v4(),
                NodeClass::Workstation {
                    gpu: None,
                    schedule: Schedule::AlwaysOn,
                },
                NodeCapabilities {
                    cpu_cores: 16,
                    memory_gb: 32.0,
                    gpu_count: 0,
                    gpu_memory_gb: None,
                    storage_gb: 1000.0,
                    network_bandwidth_mbps: 1000.0,
                    supports_gpu_direct: false,
                    battery_powered: false,
                    thermal_constraints: None,
                },
            ),
            create_test_node_with_class(
                Uuid::new_v4(),
                NodeClass::Laptop {
                    battery: true,
                    mobility: MobilityPattern::Stationary,
                },
                NodeCapabilities {
                    cpu_cores: 8,
                    memory_gb: 16.0,
                    gpu_count: 0,
                    gpu_memory_gb: None,
                    storage_gb: 500.0,
                    network_bandwidth_mbps: 100.0,
                    supports_gpu_direct: false,
                    battery_powered: true,
                    thermal_constraints: Some(ThermalConstraints {
                        max_temp_celsius: 100.0,
                        throttle_temp_celsius: 85.0,
                        current_temp_celsius: 45.0,
                        power_budget_watts: Some(45.0),
                    }),
                },
            ),
        ];

        let job = Job {
            id: Uuid::new_v4(),
            name: "Test Job".to_string(),
            requirements: JobRequirements {
                cpu_cores: Some(4),
                memory_gb: Some(8.0),
                gpu_count: None,
                gpu_memory_gb: None,
                storage_gb: None,
                network_bandwidth_mbps: None,
                requires_gpu_direct: false,
                node_affinity: None,
                anti_affinity: None,
                locality_preference: LocalityPreference::None,
                max_latency_ms: None,
                battery_safe: false, // Don't schedule on battery nodes
            },
            priority: JobPriority::Normal,
            submitted_at: Utc::now(),
        };

        // Test PowerAware policy
        *distributor.policy.write().await = SchedulingPolicy::PowerAware;
        let selected = distributor.select_node(&job, &nodes).await.unwrap();
        assert!(!selected.capabilities.battery_powered);

        // Test LatencyOptimized policy
        *distributor.policy.write().await = SchedulingPolicy::LatencyOptimized;
        let selected = distributor.select_node(&job, &nodes).await.unwrap();
        assert!(selected.network.latency_ms <= 1.0);

        // Test with max latency requirement
        let low_latency_job = Job {
            id: Uuid::new_v4(),
            name: "Low Latency Job".to_string(),
            requirements: JobRequirements {
                cpu_cores: Some(4),
                memory_gb: Some(8.0),
                gpu_count: None,
                gpu_memory_gb: None,
                storage_gb: None,
                network_bandwidth_mbps: None,
                requires_gpu_direct: false,
                node_affinity: None,
                anti_affinity: None,
                locality_preference: LocalityPreference::None,
                max_latency_ms: Some(5.0),
                battery_safe: true,
            },
            priority: JobPriority::Normal,
            submitted_at: Utc::now(),
        };

        let selected = distributor
            .select_node(&low_latency_job, &nodes)
            .await
            .unwrap();
        assert!(selected.network.latency_ms <= 5.0);
    }

    #[tokio::test]
    async fn test_best_fit_scoring() {
        let distributor = WorkDistributor::new();

        // Pre-populate node loads
        let node_id = Uuid::new_v4();
        distributor
            .update_node_load(node_id, 0.5, 0.6, 0.7)
            .await
            .unwrap();

        let node = create_test_node_with_class(
            node_id,
            NodeClass::Workstation {
                gpu: None,
                schedule: Schedule::AlwaysOn,
            },
            NodeCapabilities {
                cpu_cores: 16,
                memory_gb: 32.0,
                gpu_count: 0,
                gpu_memory_gb: None,
                storage_gb: 1000.0,
                network_bandwidth_mbps: 1000.0,
                supports_gpu_direct: false,
                battery_powered: false,
                thermal_constraints: None,
            },
        );

        // Test with preferred class affinity
        let job = Job {
            id: Uuid::new_v4(),
            name: "Preferred Class Job".to_string(),
            requirements: JobRequirements {
                cpu_cores: Some(4),
                memory_gb: Some(8.0),
                gpu_count: None,
                gpu_memory_gb: None,
                storage_gb: None,
                network_bandwidth_mbps: None,
                requires_gpu_direct: false,
                node_affinity: Some(NodeAffinity::PreferredClass(NodeClass::Workstation {
                    gpu: None,
                    schedule: Schedule::AlwaysOn,
                })),
                anti_affinity: None,
                locality_preference: LocalityPreference::DataLocal,
                max_latency_ms: None,
                battery_safe: true,
            },
            priority: JobPriority::Normal,
            submitted_at: Utc::now(),
        };

        let score = distributor.calculate_fit_score(&job, &node).await.unwrap();
        // Should have penalties for load but bonuses for affinity
        assert!(score > 0.0);
        assert!(score < 100.0);
    }

    #[tokio::test]
    async fn test_job_status_transitions() {
        let distributor = WorkDistributor::new();
        let node_id = Uuid::new_v4();

        let job = Job {
            id: Uuid::new_v4(),
            name: "Status Test Job".to_string(),
            requirements: JobRequirements::default(),
            priority: JobPriority::Normal,
            submitted_at: Utc::now(),
        };

        // Schedule job
        distributor
            .schedule_on_node(job.clone(), node_id)
            .await
            .unwrap();

        // Check scheduled jobs
        let scheduled = distributor.scheduled_jobs.read().await;
        let scheduled_job = scheduled.get(&job.id).unwrap();
        assert_eq!(scheduled_job.status, JobStatus::Scheduled);
        assert_eq!(scheduled_job.node_id, node_id);
        assert!(scheduled_job.scheduled_at <= Utc::now());
    }

    #[tokio::test]
    async fn test_load_statistics() {
        let distributor = WorkDistributor::new();

        // Add some jobs to queue
        let mut queue = distributor.job_queue.lock().await;
        for i in 0..5 {
            queue.push(PrioritizedJob {
                job: Job {
                    id: Uuid::new_v4(),
                    name: format!("Queued Job {}", i),
                    requirements: JobRequirements::default(),
                    priority: JobPriority::Normal,
                    submitted_at: Utc::now(),
                },
                score: i,
            });
        }
        drop(queue);

        // Add scheduled jobs with various statuses
        let mut scheduled = distributor.scheduled_jobs.write().await;

        let statuses = vec![
            JobStatus::Running,
            JobStatus::Running,
            JobStatus::Completed,
            JobStatus::Completed,
            JobStatus::Completed,
            JobStatus::Failed("Test failure".to_string()),
            JobStatus::Cancelled,
            JobStatus::Migrating,
        ];

        for (i, status) in statuses.into_iter().enumerate() {
            let job = Job {
                id: Uuid::new_v4(),
                name: format!("Job {}", i),
                requirements: JobRequirements::default(),
                priority: JobPriority::Normal,
                submitted_at: Utc::now(),
            };

            scheduled.insert(
                job.id,
                ScheduledJob {
                    job,
                    node_id: Uuid::new_v4(),
                    scheduled_at: Utc::now(),
                    estimated_completion: None,
                    actual_start: if matches!(status, JobStatus::Running | JobStatus::Completed) {
                        Some(Utc::now())
                    } else {
                        None
                    },
                    actual_completion: if matches!(status, JobStatus::Completed) {
                        Some(Utc::now())
                    } else {
                        None
                    },
                    status,
                },
            );
        }
        drop(scheduled);

        // Get statistics
        let stats = distributor.get_statistics().await;
        assert_eq!(stats.queued_jobs, 5);
        assert_eq!(stats.total_jobs, 8);
        assert_eq!(stats.running_jobs, 2);
        assert_eq!(stats.completed_jobs, 3);
        assert_eq!(stats.failed_jobs, 1);
    }
}
