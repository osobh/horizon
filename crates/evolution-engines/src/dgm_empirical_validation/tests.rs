//! Tests for empirical validation framework

use super::benchmark::{BenchmarkSuite, TaskExecutor};
use super::evaluator::EmpiricalEvaluator;
use super::metrics::MetricsCalculator;
use super::types::*;
use std::collections::HashMap;
use std::time::Duration;

type TestResult = Result<(), Box<dyn std::error::Error>>;

// Helper function to create test benchmark tasks
fn create_test_tasks() -> Vec<BenchmarkTask> {
    vec![
        BenchmarkTask {
            id: "task_001".to_string(),
            repository: "https://github.com/test/repo1".to_string(),
            description: "Fix null pointer exception in main.py".to_string(),
            language: "python".to_string(),
            test_command: "python -m pytest tests/test_main.py".to_string(),
            difficulty: 0.3,
            category: TaskCategory::BugFix,
        },
        BenchmarkTask {
            id: "task_002".to_string(),
            repository: "https://github.com/test/repo2".to_string(),
            description: "Add sorting functionality to list view".to_string(),
            language: "python".to_string(),
            test_command: "python -m pytest tests/test_sort.py".to_string(),
            difficulty: 0.5,
            category: TaskCategory::Feature,
        },
        BenchmarkTask {
            id: "task_003".to_string(),
            repository: "https://github.com/test/repo3".to_string(),
            description: "Refactor database connection logic".to_string(),
            language: "python".to_string(),
            test_command: "python -m pytest tests/test_db.py".to_string(),
            difficulty: 0.7,
            category: TaskCategory::Refactoring,
        },
    ]
}

// Helper function to create test task results
fn create_test_results() -> Vec<TaskResult> {
    vec![
        TaskResult {
            task_id: "task_001".to_string(),
            success: true,
            execution_time: Duration::from_secs(45),
            error: None,
            changes: vec![CodeChange {
                file: "main.py".to_string(),
                change_type: ChangeType::Modified,
                lines_added: 5,
                lines_removed: 2,
            }],
            resource_usage: ResourceUsage {
                cpu_time: Duration::from_secs(40),
                peak_memory: 100_000_000,
                tool_invocations: HashMap::from([("edit".to_string(), 3), ("bash".to_string(), 5)]),
            },
        },
        TaskResult {
            task_id: "task_002".to_string(),
            success: false,
            execution_time: Duration::from_secs(120),
            error: Some("Test failed: sorting not implemented correctly".to_string()),
            changes: vec![CodeChange {
                file: "views.py".to_string(),
                change_type: ChangeType::Modified,
                lines_added: 20,
                lines_removed: 5,
            }],
            resource_usage: ResourceUsage {
                cpu_time: Duration::from_secs(110),
                peak_memory: 150_000_000,
                tool_invocations: HashMap::from([
                    ("edit".to_string(), 8),
                    ("bash".to_string(), 10),
                ]),
            },
        },
        TaskResult {
            task_id: "task_003".to_string(),
            success: true,
            execution_time: Duration::from_secs(90),
            error: None,
            changes: vec![
                CodeChange {
                    file: "db.py".to_string(),
                    change_type: ChangeType::Modified,
                    lines_added: 30,
                    lines_removed: 40,
                },
                CodeChange {
                    file: "db_utils.py".to_string(),
                    change_type: ChangeType::Created,
                    lines_added: 50,
                    lines_removed: 0,
                },
            ],
            resource_usage: ResourceUsage {
                cpu_time: Duration::from_secs(85),
                peak_memory: 200_000_000,
                tool_invocations: HashMap::from([
                    ("edit".to_string(), 12),
                    ("bash".to_string(), 15),
                ]),
            },
        },
    ]
}

#[test]
fn test_benchmark_suite_creation() -> TestResult {
    let suite = BenchmarkSuite::new("SWE-bench".to_string());
    let tasks = create_test_tasks();

    let loaded_suite = suite.load_tasks(tasks.clone()).unwrap();

    assert_eq!(loaded_suite.name(), "SWE-bench");
    assert_eq!(loaded_suite.task_count(), 3);
    assert_eq!(loaded_suite.get_task("task_001").unwrap().id, "task_001");
    Ok(())
}

#[test]
fn test_benchmark_suite_filter_by_category() {
    let suite = BenchmarkSuite::new("test".to_string());
    let tasks = create_test_tasks();
    let loaded_suite = suite.load_tasks(tasks).unwrap();

    let bug_tasks = loaded_suite.filter_by_category(TaskCategory::BugFix);

    assert_eq!(bug_tasks.len(), 1);
    assert_eq!(bug_tasks[0].category, TaskCategory::BugFix);
}

#[test]
fn test_benchmark_suite_filter_by_difficulty() {
    let suite = BenchmarkSuite::new("test".to_string());
    let tasks = create_test_tasks();
    let loaded_suite = suite.load_tasks(tasks).unwrap();

    let easy_tasks = loaded_suite.filter_by_difficulty(0.0, 0.5);

    assert_eq!(easy_tasks.len(), 2);
    assert!(easy_tasks.iter().all(|t| t.difficulty <= 0.5));
}

#[test]
fn test_task_executor_execute() -> TestResult {
    let executor = TaskExecutor::new();
    let task = create_test_tasks()[0].clone();
    let agent_code = "def solve(): pass";

    let result = executor
        .execute(&task, agent_code, Duration::from_secs(60))?;

    assert_eq!(result.task_id, task.id);
    assert!(result.execution_time <= Duration::from_secs(60));
    Ok(())
}

#[test]
fn test_metrics_calculator_calculate() {
    let calculator = MetricsCalculator::new();
    let results = create_test_results();

    let metrics = calculator.calculate(&results).unwrap();

    assert_eq!(metrics.avg_execution_time, Duration::from_secs(85)); // (45+120+90)/3
    assert!((metrics.code_change_stats.avg_files_changed - 4.0 / 3.0).abs() < 0.0001); // 4 files / 3 tasks
    assert!(metrics
        .success_by_category
        .contains_key(&TaskCategory::BugFix));
}

#[test]
fn test_metrics_calculator_success_rate() {
    let calculator = MetricsCalculator::new();
    let results = create_test_results();

    let success_rate = calculator.calculate_success_rate(&results);

    assert_eq!(success_rate, 2.0 / 3.0); // 2 successes out of 3
}

#[test]
fn test_empirical_evaluator_evaluate() {
    let config = ValidationConfig::default();
    let evaluator = EmpiricalEvaluator::new(config);

    let agent_id = "agent_123";
    let agent_code = "def solve(): pass";
    let tasks = create_test_tasks();

    let task_refs: Vec<&BenchmarkTask> = tasks.iter().collect();
    let result = evaluator
        .evaluate(agent_id, agent_code, &task_refs)
        .unwrap();

    assert_eq!(result.agent_id, agent_id);
    // Result length depends on simulation - just check it's not empty
    assert!(!result.task_results.is_empty());
    assert!(result.success_rate >= 0.0 && result.success_rate <= 1.0);
}

#[test]
fn test_empirical_evaluator_staged_evaluation() {
    let mut config = ValidationConfig::default();
    config.staged_evaluation = true;
    config.min_success_rate = 0.4;

    let evaluator = EmpiricalEvaluator::new(config);

    let should_continue = evaluator.should_continue_evaluation(0.3, 10);
    assert!(!should_continue);

    let should_continue = evaluator.should_continue_evaluation(0.5, 10);
    assert!(should_continue);
}

#[test]
fn test_statistical_significance() -> TestResult {
    let evaluator = EmpiricalEvaluator::new(ValidationConfig::default());

    let results = create_test_results();
    let baseline_success_rate = 0.5;

    let significance = evaluator
        .calculate_statistical_significance(&results, baseline_success_rate)?;

    assert!(significance.p_value >= 0.0 && significance.p_value <= 1.0);
    assert!(significance.confidence_interval.0 <= significance.confidence_interval.1);
    assert_eq!(significance.is_significant, significance.p_value < 0.05);
    Ok(())
}

#[test]
fn test_baseline_comparison() {
    let evaluator = EmpiricalEvaluator::new(ValidationConfig::default());

    let current_rate = 0.7;
    let baseline_rate = 0.5;

    let comparison = evaluator.compare_with_baseline(current_rate, baseline_rate);

    assert_eq!(comparison.baseline_success_rate, baseline_rate);
    assert!((comparison.improvement - 0.2).abs() < 0.0001);
    assert!((comparison.relative_improvement - 40.0).abs() < 0.0001); // 40% improvement
}

#[test]
fn test_validation_report_generation() -> TestResult {
    let config = ValidationConfig::default();
    let evaluator = EmpiricalEvaluator::new(config.clone());

    let results = ValidationResult {
        agent_id: "agent_123".to_string(),
        success_rate: 0.7,
        metrics: MetricsCalculator::new()
            .calculate(&create_test_results())?,
        task_results: create_test_results(),
        statistical_significance: StatisticalSignificance {
            p_value: 0.03,
            confidence_interval: (0.6, 0.8),
            effect_size: 0.5,
            is_significant: true,
        },
        baseline_comparison: Some(BaselineComparison {
            baseline_success_rate: 0.5,
            improvement: 0.2,
            relative_improvement: 40.0,
        }),
    };

    let report = evaluator.generate_report(&results).unwrap();

    assert_eq!(report.config.benchmark_suite, config.benchmark_suite);
    assert_eq!(report.results.agent_id, "agent_123");
    assert!(report.shows_improvement);
    assert!(!report.recommendations.is_empty());
    Ok(())
}

#[test]
fn test_end_to_end_validation() {
    // Test complete validation workflow
    let config = ValidationConfig {
        benchmark_suite: "test-suite".to_string(),
        num_tasks: 3,
        task_timeout: Duration::from_secs(300),
        min_success_rate: 0.4,
        staged_evaluation: false,
        confidence_threshold: 0.95,
    };

    let evaluator = EmpiricalEvaluator::new(config);
    let suite = BenchmarkSuite::new("test-suite".to_string());
    let tasks = create_test_tasks();
    let loaded_suite = suite.load_tasks(tasks).unwrap();

    // Evaluate agent
    let agent_id = "test_agent";
    let agent_code = "def solve(): pass";
    let selected_tasks = loaded_suite.filter_by_difficulty(0.0, 1.0);

    let result = evaluator
        .evaluate(agent_id, agent_code, &selected_tasks)
        .unwrap();

    // Generate report
    let report = evaluator.generate_report(&result).unwrap();

    // Validate report
    assert_eq!(report.results.agent_id, agent_id);
    assert_eq!(report.results.task_results.len(), 3);
    assert!(report.timestamp > 0);

    // Check recommendations
    if result.success_rate < 0.5 {
        assert!(report
            .recommendations
            .iter()
            .any(|r| r.contains("performance")));
    }
}

#[test]
fn test_resource_efficiency_calculation() {
    let calculator = MetricsCalculator::new();
    let results = create_test_results();

    let efficiency = calculator.calculate_resource_efficiency(&results);

    assert!(efficiency >= 0.0 && efficiency <= 1.0);
}

#[test]
fn test_error_pattern_analysis() {
    let calculator = MetricsCalculator::new();
    let mut results = create_test_results();

    // Add more failures with patterns
    results.push(TaskResult {
        task_id: "task_004".to_string(),
        success: false,
        execution_time: Duration::from_secs(60),
        error: Some("Test failed: sorting not implemented correctly".to_string()),
        changes: vec![],
        resource_usage: ResourceUsage {
            cpu_time: Duration::from_secs(55),
            peak_memory: 80_000_000,
            tool_invocations: HashMap::new(),
        },
    });

    let patterns = calculator.analyze_error_patterns(&results);

    assert_eq!(patterns.get("sorting not implemented correctly"), Some(&2));
}
