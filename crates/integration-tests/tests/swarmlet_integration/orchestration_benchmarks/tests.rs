//! Test cases for orchestration benchmarks

use super::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_container_startup_benchmark_red_phase() {
        // RED phase - benchmark should fail initially
        let mut benchmarks = WorkloadOrchestrationBenchmarks::new(TddPhase::Red);

        // Set aggressive targets (expected to fail)
        benchmarks.set_targets(OrchestrationTargets {
            max_startup_time_ms: 100, // Very aggressive
            min_concurrent_capacity: 200,
            max_resource_overhead_percent: 5.0,
            min_throughput_workloads_per_sec: 100.0,
            max_scheduling_latency_ms: 5,
            min_resource_utilization_percent: 90.0,
        });

        // Add test workloads
        for i in 0..10 {
            benchmarks.add_workload(BenchmarkWorkload {
                workload_id: Uuid::new_v4(),
                workload_type: WorkloadType::CpuIntensive {
                    iterations: 1000,
                    complexity: 10,
                },
                container_spec: ContainerSpec::default(),
                performance_expectations: PerformanceExpectations::default(),
            });
        }

        let result = benchmarks.run_benchmark("container_startup_red", 10).await;

        // In RED phase, we expect failure
        assert_eq!(result.phase, TddPhase::Red);
        assert!(
            !result.success,
            "RED phase should fail with aggressive targets"
        );
        assert!(result.efficiency_score < 100.0);
        assert!(!result.bottleneck_analysis.is_empty());
    }

    #[tokio::test]
    async fn test_concurrent_execution_benchmark_green_phase() {
        // GREEN phase - minimal implementation meets basic targets
        let mut benchmarks = WorkloadOrchestrationBenchmarks::new(TddPhase::Green);

        // Set achievable targets
        benchmarks.set_targets(OrchestrationTargets::default());

        // Add workloads
        for _ in 0..50 {
            benchmarks.add_workload(BenchmarkWorkload {
                workload_id: Uuid::new_v4(),
                workload_type: WorkloadType::Mixed {
                    cpu_percent: 20.0,
                    memory_mb: 256,
                    io_ops: 100,
                },
                container_spec: ContainerSpec::default(),
                performance_expectations: PerformanceExpectations::default(),
            });
        }

        let result = benchmarks
            .run_benchmark("concurrent_execution_green", 50)
            .await;

        assert_eq!(result.phase, TddPhase::Green);
        // Green phase might succeed with default targets
        if result.success {
            assert!(result.efficiency_score >= 50.0);
        }
    }

    #[tokio::test]
    async fn test_resource_utilization_benchmark() {
        let benchmarks = WorkloadOrchestrationBenchmarks::new(TddPhase::Refactor);
        let monitor = benchmarks.resource_monitor.clone();

        // Simulate node resources
        let mut mon = monitor.lock().await;
        mon.update_node(NodeResourceState {
            node_id: "node-1".to_string(),
            total_cpu_cores: 16.0,
            allocatable_cpu_cores: 15.0,
            used_cpu_cores: 12.0,
            total_memory_mb: 32768,
            allocatable_memory_mb: 30000,
            used_memory_mb: 24000,
            gpu_count: 2,
            available_gpu_count: 1,
            running_workloads: vec![],
        });

        let (cpu_util, mem_util, gpu_util) = mon.calculate_utilization();

        assert!(cpu_util > 0.0);
        assert!(mem_util > 0.0);
        assert_eq!(gpu_util, 50.0); // 1 of 2 GPUs used
    }

    #[tokio::test]
    async fn test_bottleneck_analysis() {
        let benchmarks = WorkloadOrchestrationBenchmarks::new(TddPhase::Refactor);

        let metrics = OrchestrationActuals {
            avg_startup_time_ms: 600, // Exceeds default target
            peak_concurrent_workloads: 100,
            resource_overhead_percent: 8.0,
            achieved_throughput: 30.0, // Below default target
            avg_scheduling_latency_ms: 10,
            resource_utilization_percent: 70.0, // Below default target
            failed_workloads: 2,
            total_workloads: 100,
        };

        let bottlenecks = benchmarks.analyze_bottlenecks(&metrics).await;

        assert!(!bottlenecks.is_empty());

        // Should identify startup time bottleneck
        assert!(bottlenecks
            .iter()
            .any(|b| matches!(b.component, BottleneckComponent::ContainerRuntime)));

        // Should identify resource utilization bottleneck
        assert!(bottlenecks
            .iter()
            .any(|b| matches!(b.component, BottleneckComponent::CpuScheduler)));

        // Check optimization suggestions
        for bottleneck in &bottlenecks {
            assert!(!bottleneck.suggested_optimizations.is_empty());
        }
    }

    #[tokio::test]
    async fn test_efficiency_score_calculation() {
        let benchmarks = WorkloadOrchestrationBenchmarks::new(TddPhase::Refactor);

        // Perfect metrics
        let perfect_metrics = OrchestrationActuals {
            avg_startup_time_ms: 400,
            peak_concurrent_workloads: 100,
            resource_overhead_percent: 8.0,
            achieved_throughput: 60.0,
            avg_scheduling_latency_ms: 8,
            resource_utilization_percent: 85.0,
            failed_workloads: 0,
            total_workloads: 100,
        };

        let perfect_score = benchmarks.calculate_efficiency(&perfect_metrics);
        assert!(perfect_score >= 95.0);

        // Poor metrics
        let poor_metrics = OrchestrationActuals {
            avg_startup_time_ms: 1000,
            peak_concurrent_workloads: 50,
            resource_overhead_percent: 20.0,
            achieved_throughput: 20.0,
            avg_scheduling_latency_ms: 50,
            resource_utilization_percent: 40.0,
            failed_workloads: 10,
            total_workloads: 100,
        };

        let poor_score = benchmarks.calculate_efficiency(&poor_metrics);
        assert!(poor_score < 50.0);
    }
}
