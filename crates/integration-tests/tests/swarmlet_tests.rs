//! Swarmlet Integration Tests
//!
//! This file serves as the entry point for all swarmlet-related integration tests.
//! It includes comprehensive testing for cluster formation, workload management,
//! GPU bridge functionality, performance benchmarks, and orchestration capabilities.

mod swarmlet_integration;

#[cfg(test)]
mod tests {
    use super::swarmlet_integration::*;

    #[tokio::test]
    async fn test_swarmlet_cluster_formation() {
        let tests = SwarmletClusterFormationTests::new().await;
        let results = tests.run_comprehensive_tests().await;

        // Verify all phases completed
        assert!(results
            .iter()
            .any(|r| r.phase == cluster_formation_tests::TddPhase::Red));
        assert!(results
            .iter()
            .any(|r| r.phase == cluster_formation_tests::TddPhase::Green));
        assert!(results
            .iter()
            .any(|r| r.phase == cluster_formation_tests::TddPhase::Refactor));

        // Verify success in final phase
        let refactor_results: Vec<_> = results
            .iter()
            .filter(|r| r.phase == cluster_formation_tests::TddPhase::Refactor)
            .collect();
        assert!(refactor_results.iter().all(|r| r.success));
    }

    #[tokio::test]
    async fn test_swarmlet_workload_management() {
        let tests = SwarmletWorkloadManagementTests::new().await;
        let results = tests.run_comprehensive_tests().await;

        // Verify workload tests completed successfully
        assert!(!results.is_empty());

        // Check final phase results
        let refactor_results: Vec<_> = results
            .iter()
            .filter(|r| r.phase == workload_management_tests::TddPhase::Refactor)
            .collect();
        assert!(refactor_results.iter().all(|r| r.success));
    }

    #[tokio::test]
    async fn test_swarmlet_gpu_bridge() {
        let tests = SwarmletGpuBridgeTests::new().await;
        let results = tests.run_comprehensive_tests().await;

        // Verify GPU bridge tests completed
        assert!(results.len() >= 15); // 5 tests × 3 phases

        // Check GPU utilization improved across phases
        let red_gpu_tests: Vec<_> = results
            .iter()
            .filter(|r| r.phase == gpu_bridge_tests::TddPhase::Red)
            .collect();
        let refactor_gpu_tests: Vec<_> = results
            .iter()
            .filter(|r| r.phase == gpu_bridge_tests::TddPhase::Refactor)
            .collect();

        // Verify performance improvements
        for refactor_test in &refactor_gpu_tests {
            if let Some(red_test) = red_gpu_tests
                .iter()
                .find(|r| r.test_name == refactor_test.test_name)
            {
                assert!(refactor_test.throughput_ops_per_sec >= red_test.throughput_ops_per_sec);
            }
        }
    }

    #[tokio::test]
    async fn test_swarmlet_performance_benchmarks() {
        let benchmarks = SwarmletPerformanceBenchmarks::new().await;
        let results = benchmarks.run_comprehensive_benchmarks().await;

        // Verify performance improvements
        assert!(!results.is_empty());

        // Check specific benchmark improvements
        for result in &results {
            if result.phase == performance_benchmarks::TddPhase::Refactor {
                assert!(result.improvement_factor >= 1.0);
                assert!(result.success);
            }
        }
    }

    #[tokio::test]
    async fn test_workload_orchestration_benchmarks() {
        let benchmarks = WorkloadOrchestrationBenchmarks::new().await;
        let results = benchmarks.run_comprehensive_benchmarks().await;

        // Verify orchestration efficiency
        assert!(!results.is_empty());

        // Check efficiency scores improved
        let refactor_results: Vec<_> = results
            .iter()
            .filter(|r| r.phase == orchestration_benchmarks::TddPhase::Refactor)
            .collect();

        for result in &refactor_results {
            assert!(result.efficiency_score > 0.8);
            assert!(result.success);
        }
    }

    #[tokio::test]
    async fn test_enterprise_workload_simulation() {
        let simulation = EnterpriseWorkloadSimulation::new().await;
        let results = simulation.run_comprehensive_tests().await;

        // Verify enterprise workload tests completed
        assert!(!results.is_empty());

        // Check all workload types were tested
        let has_web_app = results.iter().any(|r| {
            matches!(
                r.workload_type,
                enterprise_workload_simulation::EnterpriseWorkloadType::WebApplication { .. }
            )
        });
        let has_database = results.iter().any(|r| {
            matches!(
                r.workload_type,
                enterprise_workload_simulation::EnterpriseWorkloadType::DatabaseCluster { .. }
            )
        });
        let has_ml = results.iter().any(|r| {
            matches!(
                r.workload_type,
                enterprise_workload_simulation::EnterpriseWorkloadType::MachineLearning { .. }
            )
        });

        assert!(has_web_app);
        assert!(has_database);
        assert!(has_ml);

        // Verify success in final phase
        let refactor_results: Vec<_> = results
            .iter()
            .filter(|r| r.phase == enterprise_workload_simulation::TddPhase::Refactor)
            .collect();

        for result in &refactor_results {
            assert!(result.success);
            assert!(result.availability_percent > 99.0);
            assert!(result.resource_efficiency_percent > 80.0);
        }
    }
}
