use gpu_agents::benchmarks::ProgressWriter;
use gpu_agents::scenarios::{ScenarioConfig, ScenarioType, SimpleBehavior};
use std::time::Duration;
use tempfile::TempDir;

#[test]
fn test_scenario_benchmark_integration() {
    // Create a simple scenario config
    let scenario = ScenarioConfig {
        id: "benchmark-test".to_string(),
        name: "Benchmark Test Scenario".to_string(),
        description: "Test scenario for benchmark integration".to_string(),
        scenario_type: ScenarioType::Simple {
            behavior: SimpleBehavior::Flocking,
            interaction_radius: 10.0,
            update_frequency: 60.0,
        },
        agent_count: 1000,
        duration: Duration::from_secs(2),
        seed: Some(42),
        objectives: vec![],
    };

    // Create scenario benchmark runner
    let runner = gpu_agents::benchmarks::ScenarioBenchmarkRunner::new().unwrap();

    // Run scenario as benchmark
    let result = runner.run_scenario_benchmark(&scenario).unwrap();

    assert_eq!(result.scenario_id, "benchmark-test");
    assert!(result.metrics.throughput > 0.0);
    assert!(result.metrics.latency_ms > 0.0);
    assert!(result.metrics.gpu_utilization > 0.0);
}

#[test]
fn test_scenario_benchmark_with_progress() {
    let temp_dir = TempDir::new().unwrap();
    let progress_log = temp_dir.path().join("progress.log");

    let scenario = ScenarioConfig {
        id: "progress-test".to_string(),
        name: "Progress Test".to_string(),
        description: "Test progress tracking".to_string(),
        scenario_type: ScenarioType::Simple {
            behavior: SimpleBehavior::RandomWalk,
            interaction_radius: 5.0,
            update_frequency: 30.0,
        },
        agent_count: 500,
        duration: Duration::from_secs(1),
        seed: Some(123),
        objectives: vec![],
    };

    // Create progress writer
    let progress_writer = ProgressWriter::new(progress_log.to_str().unwrap()).unwrap();

    // Create runner with progress tracking
    let runner = gpu_agents::benchmarks::ScenarioBenchmarkRunner::new().unwrap();

    // Run with progress
    let result = runner
        .run_scenario_benchmark_with_progress(&scenario, &progress_writer)
        .unwrap();

    assert!(result.metrics.completed);
    assert!(result.metrics.total_steps > 0);
}

#[test]
fn test_scenario_suite_benchmark() {
    use gpu_agents::benchmarks::ScenarioSuite;

    // Create a suite of scenarios
    let suite = ScenarioSuite::new("stress-test-suite")
        .add_scenario(ScenarioConfig {
            id: "small-agents".to_string(),
            name: "Small Agent Test".to_string(),
            description: "Test with 100 agents".to_string(),
            scenario_type: ScenarioType::Simple {
                behavior: SimpleBehavior::Flocking,
                interaction_radius: 10.0,
                update_frequency: 60.0,
            },
            agent_count: 100,
            duration: Duration::from_secs(1),
            seed: Some(1),
            objectives: vec![],
        })
        .add_scenario(ScenarioConfig {
            id: "medium-agents".to_string(),
            name: "Medium Agent Test".to_string(),
            description: "Test with 10000 agents".to_string(),
            scenario_type: ScenarioType::Simple {
                behavior: SimpleBehavior::Flocking,
                interaction_radius: 10.0,
                update_frequency: 60.0,
            },
            agent_count: 10000,
            duration: Duration::from_secs(1),
            seed: Some(2),
            objectives: vec![],
        });

    // Run suite
    let runner = gpu_agents::benchmarks::ScenarioBenchmarkRunner::new().unwrap();
    let results = runner.run_scenario_suite(&suite).unwrap();

    assert_eq!(results.suite_name, "stress-test-suite");
    assert_eq!(results.scenario_results.len(), 2);
    assert!(results.summary.total_duration > Duration::from_secs(0));
}

#[test]
fn test_scenario_comparison_benchmark() {
    use gpu_agents::benchmarks::ComparisonMetric;

    let scenario1 = ScenarioConfig {
        id: "baseline".to_string(),
        name: "Baseline".to_string(),
        description: "Baseline scenario".to_string(),
        scenario_type: ScenarioType::Simple {
            behavior: SimpleBehavior::RandomWalk,
            interaction_radius: 5.0,
            update_frequency: 30.0,
        },
        agent_count: 1000,
        duration: Duration::from_secs(1),
        seed: Some(42),
        objectives: vec![],
    };

    let scenario2 = ScenarioConfig {
        id: "optimized".to_string(),
        name: "Optimized".to_string(),
        description: "Optimized scenario".to_string(),
        scenario_type: ScenarioType::Simple {
            behavior: SimpleBehavior::RandomWalk,
            interaction_radius: 5.0,
            update_frequency: 60.0,
        },
        agent_count: 1000,
        duration: Duration::from_secs(1),
        seed: Some(42),
        objectives: vec![],
    };

    let runner = gpu_agents::benchmarks::ScenarioBenchmarkRunner::new().unwrap();
    let comparison = runner.compare_scenarios(&scenario1, &scenario2).unwrap();

    assert!(comparison.has_metric(ComparisonMetric::Throughput));
    assert!(comparison.has_metric(ComparisonMetric::GpuUtilization));
    assert!(comparison.improvement_percentage() != 0.0);
}

#[test]
fn test_scenario_to_benchmark_config() {
    use gpu_agents::benchmarks::scenario_to_benchmark_config;

    let scenario = ScenarioConfig {
        id: "config-test".to_string(),
        name: "Config Test".to_string(),
        description: "Test config conversion".to_string(),
        scenario_type: ScenarioType::Simple {
            behavior: SimpleBehavior::Seeking,
            interaction_radius: 15.0,
            update_frequency: 45.0,
        },
        agent_count: 5000,
        duration: Duration::from_secs(10),
        seed: Some(999),
        objectives: vec![],
    };

    let bench_config = scenario_to_benchmark_config(&scenario);

    assert_eq!(bench_config.agent_count, 5000);
    assert_eq!(bench_config.duration, Duration::from_secs(10));
    assert_eq!(bench_config.benchmark_name, "config-test");
}

#[test]
fn test_scenario_benchmark_report_generation() {
    use gpu_agents::benchmarks::{ReportFormat, ScenarioBenchmarkReport};

    let temp_dir = TempDir::new().unwrap();
    let output_path = temp_dir.path().join("report.html");

    let scenario = ScenarioConfig {
        id: "report-test".to_string(),
        name: "Report Test".to_string(),
        description: "Test report generation".to_string(),
        scenario_type: ScenarioType::Simple {
            behavior: SimpleBehavior::Avoidance,
            interaction_radius: 8.0,
            update_frequency: 30.0,
        },
        agent_count: 2000,
        duration: Duration::from_secs(3),
        seed: Some(555),
        objectives: vec![],
    };

    let runner = gpu_agents::benchmarks::ScenarioBenchmarkRunner::new().unwrap();
    let result = runner.run_scenario_benchmark(&scenario).unwrap();

    // Generate report
    let report = ScenarioBenchmarkReport::from_result(&result);
    report.save(&output_path, ReportFormat::Html).unwrap();

    assert!(output_path.exists());
    assert!(std::fs::metadata(&output_path).unwrap().len() > 0);
}
