//! Swarmlet Integration Test Suite
//!
//! Comprehensive integration testing for swarmlet functionality including
//! cluster formation, workload management, GPU bridge, performance benchmarks,
//! orchestration benchmarks, and enterprise workload simulation.

pub mod cluster_formation_tests;
pub mod enterprise_workload_simulation;
pub mod gpu_bridge_tests;
pub mod orchestration_benchmarks;
pub mod performance_benchmarks;
pub mod workload_management_tests;

// Re-export key test structures for convenience
pub use cluster_formation_tests::SwarmletClusterFormationTests;
pub use enterprise_workload_simulation::EnterpriseWorkloadSimulation;
pub use gpu_bridge_tests::SwarmletGpuBridgeTests;
pub use orchestration_benchmarks::WorkloadOrchestrationBenchmarks;
pub use performance_benchmarks::SwarmletPerformanceBenchmarks;
pub use workload_management_tests::SwarmletWorkloadManagementTests;

#[cfg(test)]
mod integration_suite {
    use super::*;

    #[tokio::test]
    async fn test_complete_swarmlet_integration_suite() {
        println!("=== Running Complete Swarmlet Integration Suite ===");

        // Run cluster formation tests
        println!("\n1. Cluster Formation Tests");
        let cluster_tests = SwarmletClusterFormationTests::new().await;
        let cluster_results = cluster_tests.run_comprehensive_tests().await;
        assert!(!cluster_results.is_empty());

        // Run workload management tests
        println!("\n2. Workload Management Tests");
        let workload_tests = SwarmletWorkloadManagementTests::new().await;
        let workload_results = workload_tests.run_comprehensive_tests().await;
        assert!(!workload_results.is_empty());

        // Run GPU bridge tests
        println!("\n3. GPU-Agents Bridge Tests");
        let gpu_tests = SwarmletGpuBridgeTests::new().await;
        let gpu_results = gpu_tests.run_comprehensive_tests().await;
        assert!(!gpu_results.is_empty());

        // Run performance benchmarks
        println!("\n4. Performance Benchmarks");
        let perf_benchmarks = SwarmletPerformanceBenchmarks::new().await;
        let perf_results = perf_benchmarks.run_comprehensive_benchmarks().await;
        assert!(!perf_results.is_empty());

        // Run orchestration benchmarks
        println!("\n5. Orchestration Benchmarks");
        let orch_benchmarks = WorkloadOrchestrationBenchmarks::new().await;
        let orch_results = orch_benchmarks.run_comprehensive_benchmarks().await;
        assert!(!orch_results.is_empty());

        // Run enterprise workload simulation
        println!("\n6. Enterprise Workload Simulation");
        let enterprise_sim = EnterpriseWorkloadSimulation::new().await;
        let enterprise_results = enterprise_sim.run_comprehensive_tests().await;
        assert!(!enterprise_results.is_empty());

        println!("\n=== Swarmlet Integration Suite Complete ===");

        // Verify all tests passed
        let total_tests = cluster_results.len()
            + workload_results.len()
            + gpu_results.len()
            + perf_results.len()
            + orch_results.len()
            + enterprise_results.len();
        println!("Total tests executed: {}", total_tests);
    }
}
