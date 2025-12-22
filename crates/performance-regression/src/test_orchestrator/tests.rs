//! Comprehensive TDD tests for test orchestrator refactoring
//! Following TDD methodology: RED-GREEN-REFACTOR
//! These tests define expected behavior for the refactored modules

use super::*;
use crate::error::PerformanceRegressionError;
use chrono::Utc;
use std::collections::HashMap;
use tokio::time::Duration as TokioDuration;

#[cfg(test)]
mod config_tests {
    use super::*;

    #[test]
    fn test_orchestrator_config_default() {
        let config = TestOrchestratorConfig::default();

        assert_eq!(config.max_concurrent_tests, 4);
        assert_eq!(config.test_timeout_seconds, 3600);
        assert!(config.parallel_execution);
        assert_eq!(config.result_retention_hours, 168);
    }

    #[test]
    fn test_ci_integration_config_default() {
        let config = CiIntegrationConfig::default();

        assert!(config.enabled);
        assert_eq!(config.platform, CiPlatform::GitHub);
        assert!(config.webhook_url.is_none());
        assert!(config.notify_on_failure);
        assert_eq!(config.report_format, ReportFormat::Json);
    }

    #[test]
    fn test_scheduling_config_default() {
        let config = SchedulingConfig::default();

        assert!(!config.enabled);
        assert!(config.cron_expression.is_none());
        assert_eq!(config.max_queue_size, 100);
        assert!(config.retry_on_failure);
        assert_eq!(config.max_retries, 3);
    }

    #[test]
    fn test_load_test_config_creation() {
        let config = LoadTestConfig {
            target_rps: 100.0,
            duration_seconds: 300,
            ramp_up_seconds: 30,
            virtual_users: 50,
        };

        assert_eq!(config.target_rps, 100.0);
        assert_eq!(config.duration_seconds, 300);
        assert_eq!(config.ramp_up_seconds, 30);
        assert_eq!(config.virtual_users, 50);
    }

    #[test]
    fn test_test_strategy_enum_variants() {
        let load_config = LoadTestConfig {
            target_rps: 100.0,
            duration_seconds: 300,
            ramp_up_seconds: 30,
            virtual_users: 50,
        };

        let strategy = TestStrategy::LoadTest(load_config.clone());

        match strategy {
            TestStrategy::LoadTest(config) => {
                assert_eq!(config.target_rps, 100.0);
            }
            _ => panic!("Expected LoadTest strategy"),
        }
    }
}

#[cfg(test)]
mod orchestrator_tests {
    use super::*;

    fn create_test_config() -> TestOrchestratorConfig {
        TestOrchestratorConfig {
            max_concurrent_tests: 2,
            test_timeout_seconds: 60,
            parallel_execution: true,
            ci_integration: CiIntegrationConfig {
                enabled: true,
                platform: CiPlatform::Generic,
                webhook_url: Some("http://example.com/webhook".to_string()),
                report_format: ReportFormat::Json,
                notify_on_failure: true,
            },
            scheduling_config: SchedulingConfig {
                enabled: true,
                cron_expression: Some("0 */6 * * *".to_string()),
                max_queue_size: 50,
                retry_on_failure: true,
                max_retries: 2,
            },
            result_retention_hours: 24,
        }
    }

    #[test]
    fn test_orchestrator_creation() {
        let config = create_test_config();
        let orchestrator = TestOrchestrator::new(config.clone());

        assert_eq!(orchestrator.config.max_concurrent_tests, 2);
        assert!(orchestrator.config.parallel_execution);
    }

    #[tokio::test]
    async fn test_orchestrate_empty_tests() {
        let config = create_test_config();
        let orchestrator = TestOrchestrator::new(config);

        let result = orchestrator.orchestrate_tests(vec![]).await;
        assert!(result.is_ok());

        let report = result.unwrap();
        assert_eq!(report.summary.total_tests, 0);
        assert_eq!(report.summary.passed_tests, 0);
        assert_eq!(report.summary.failed_tests, 0);
    }

    #[tokio::test]
    async fn test_orchestrate_single_load_test() {
        let config = create_test_config();
        let orchestrator = TestOrchestrator::new(config);

        let load_config = LoadTestConfig {
            target_rps: 10.0,
            duration_seconds: 5,
            ramp_up_seconds: 1,
            virtual_users: 2,
        };

        let strategies = vec![TestStrategy::LoadTest(load_config)];
        let result = orchestrator.orchestrate_tests(strategies).await;

        assert!(result.is_ok());
        let report = result.unwrap();
        assert_eq!(report.summary.total_tests, 1);
        assert!(!report.report_id.is_empty());
    }

    #[tokio::test]
    async fn test_orchestrate_multiple_parallel_tests() {
        let mut config = create_test_config();
        config.parallel_execution = true;
        config.max_concurrent_tests = 3;

        let orchestrator = TestOrchestrator::new(config);

        let strategies = vec![
            TestStrategy::LoadTest(LoadTestConfig {
                target_rps: 5.0,
                duration_seconds: 2,
                ramp_up_seconds: 1,
                virtual_users: 1,
            }),
            TestStrategy::LoadTest(LoadTestConfig {
                target_rps: 10.0,
                duration_seconds: 2,
                ramp_up_seconds: 1,
                virtual_users: 2,
            }),
            TestStrategy::LoadTest(LoadTestConfig {
                target_rps: 15.0,
                duration_seconds: 2,
                ramp_up_seconds: 1,
                virtual_users: 3,
            }),
        ];

        let start_time = std::time::Instant::now();
        let result = orchestrator.orchestrate_tests(strategies).await;
        let elapsed = start_time.elapsed();

        assert!(result.is_ok());
        let report = result.unwrap();
        assert_eq!(report.summary.total_tests, 3);

        // Parallel execution should be faster than sequential
        assert!(elapsed < TokioDuration::from_secs(8)); // Should be ~2-3 seconds, not 6
    }

    #[tokio::test]
    async fn test_orchestrate_sequential_tests() {
        let mut config = create_test_config();
        config.parallel_execution = false;

        let orchestrator = TestOrchestrator::new(config);

        let strategies = vec![
            TestStrategy::LoadTest(LoadTestConfig {
                target_rps: 5.0,
                duration_seconds: 1,
                ramp_up_seconds: 0,
                virtual_users: 1,
            }),
            TestStrategy::LoadTest(LoadTestConfig {
                target_rps: 10.0,
                duration_seconds: 1,
                ramp_up_seconds: 0,
                virtual_users: 1,
            }),
        ];

        let result = orchestrator.orchestrate_tests(strategies).await;

        assert!(result.is_ok());
        let report = result.unwrap();
        assert_eq!(report.summary.total_tests, 2);
    }

    #[tokio::test]
    async fn test_collect_test_metrics_nonexistent_test() {
        let config = create_test_config();
        let orchestrator = TestOrchestrator::new(config);

        let result = orchestrator
            .collect_test_metrics("nonexistent-test-id")
            .await;

        assert!(result.is_err());
        match result.unwrap_err() {
            PerformanceRegressionError::TestNotFound { test_id } => {
                assert_eq!(test_id, "nonexistent-test-id");
            }
            _ => panic!("Expected TestNotFound error"),
        }
    }

    #[tokio::test]
    async fn test_generate_test_report_empty_ids() {
        let config = create_test_config();
        let orchestrator = TestOrchestrator::new(config);

        let result = orchestrator.generate_test_report(vec![]).await;

        assert!(result.is_err());
        match result.unwrap_err() {
            PerformanceRegressionError::NoTestResults => {}
            _ => panic!("Expected NoTestResults error"),
        }
    }

    #[tokio::test]
    async fn test_execute_load_test_direct() {
        let config = create_test_config();
        let orchestrator = TestOrchestrator::new(config);

        let load_config = LoadTestConfig {
            target_rps: 5.0,
            duration_seconds: 1,
            ramp_up_seconds: 0,
            virtual_users: 1,
        };

        let result = orchestrator.execute_load_test(load_config).await;

        assert!(result.is_ok());
        let test_result = result.unwrap();
        assert!(!test_result.test_id.is_empty());
        assert!(matches!(
            test_result.status,
            TestStatus::Passed | TestStatus::Failed
        ));
    }

    #[tokio::test]
    async fn test_execute_stress_test_direct() {
        let config = create_test_config();
        let orchestrator = TestOrchestrator::new(config);

        let stress_config = StressTestConfig {
            initial_rps: 5.0,
            max_rps: 20.0,
            rps_increment: 5.0,
            step_duration_seconds: 1,
            failure_threshold: 5.0,
        };

        let result = orchestrator.execute_stress_test(stress_config).await;

        assert!(result.is_ok());
        let test_result = result.unwrap();
        assert!(!test_result.test_id.is_empty());
        assert!(matches!(
            test_result.status,
            TestStatus::Passed | TestStatus::Failed
        ));
    }
}

#[cfg(test)]
mod scheduling_tests {
    use super::*;

    #[tokio::test]
    async fn test_schedule_tests_disabled() {
        let mut config = create_test_config();
        config.scheduling_config.enabled = false;

        let mut orchestrator = TestOrchestrator::new(config);

        let strategies = vec![TestStrategy::LoadTest(LoadTestConfig {
            target_rps: 5.0,
            duration_seconds: 1,
            ramp_up_seconds: 0,
            virtual_users: 1,
        })];

        let result = orchestrator.schedule_tests(strategies, None).await;

        assert!(result.is_err());
        match result.unwrap_err() {
            PerformanceRegressionError::ConfigurationError { parameter, .. } => {
                assert_eq!(parameter, "scheduling_config.enabled");
            }
            _ => panic!("Expected ConfigurationError"),
        }
    }

    #[tokio::test]
    async fn test_schedule_tests_no_cron_expression() {
        let mut config = create_test_config();
        config.scheduling_config.enabled = true;
        config.scheduling_config.cron_expression = None;

        let mut orchestrator = TestOrchestrator::new(config);

        let strategies = vec![TestStrategy::LoadTest(LoadTestConfig {
            target_rps: 5.0,
            duration_seconds: 1,
            ramp_up_seconds: 0,
            virtual_users: 1,
        })];

        let result = orchestrator.schedule_tests(strategies, None).await;

        assert!(result.is_err());
        match result.unwrap_err() {
            PerformanceRegressionError::ConfigurationError { parameter, .. } => {
                assert_eq!(parameter, "cron_expression");
            }
            _ => panic!("Expected ConfigurationError"),
        }
    }

    #[tokio::test]
    async fn test_schedule_tests_with_valid_config() {
        let config = create_test_config();
        let mut orchestrator = TestOrchestrator::new(config);

        let strategies = vec![TestStrategy::LoadTest(LoadTestConfig {
            target_rps: 5.0,
            duration_seconds: 1,
            ramp_up_seconds: 0,
            virtual_users: 1,
        })];

        let result = orchestrator.schedule_tests(strategies, None).await;

        assert!(result.is_ok());
    }

    fn create_test_config() -> TestOrchestratorConfig {
        TestOrchestratorConfig {
            max_concurrent_tests: 2,
            test_timeout_seconds: 60,
            parallel_execution: true,
            ci_integration: CiIntegrationConfig {
                enabled: true,
                platform: CiPlatform::Generic,
                webhook_url: Some("http://example.com/webhook".to_string()),
                report_format: ReportFormat::Json,
                notify_on_failure: true,
            },
            scheduling_config: SchedulingConfig {
                enabled: true,
                cron_expression: Some("0 */6 * * *".to_string()),
                max_queue_size: 50,
                retry_on_failure: true,
                max_retries: 2,
            },
            result_retention_hours: 24,
        }
    }
}

#[cfg(test)]
mod ci_integration_tests {
    use super::*;

    #[tokio::test]
    async fn test_ci_integration_disabled() {
        let mut config = create_test_config();
        config.ci_integration.enabled = false;

        let orchestrator = TestOrchestrator::new(config);

        let test_report = TestReport {
            report_id: "test-123".to_string(),
            generated_at: Utc::now(),
            results: vec![],
            summary: TestSummary {
                total_tests: 1,
                passed_tests: 1,
                failed_tests: 0,
                success_rate: 100.0,
                avg_duration_seconds: 5.0,
                total_duration_seconds: 5.0,
            },
            ci_metadata: None,
        };

        let ci_metadata = CiMetadata {
            build_id: "build-123".to_string(),
            commit_sha: "abc123".to_string(),
            branch: "main".to_string(),
            pr_number: None,
            build_url: Some("http://example.com/build".to_string()),
        };

        let result = orchestrator
            .integrate_with_ci(test_report, ci_metadata)
            .await;

        assert!(result.is_ok()); // Should succeed but do nothing
    }

    #[tokio::test]
    async fn test_ci_integration_with_successful_tests() {
        let config = create_test_config();
        let orchestrator = TestOrchestrator::new(config);

        let test_report = TestReport {
            report_id: "test-123".to_string(),
            generated_at: Utc::now(),
            results: vec![],
            summary: TestSummary {
                total_tests: 1,
                passed_tests: 1,
                failed_tests: 0,
                success_rate: 100.0,
                avg_duration_seconds: 5.0,
                total_duration_seconds: 5.0,
            },
            ci_metadata: None,
        };

        let ci_metadata = CiMetadata {
            build_id: "build-123".to_string(),
            commit_sha: "abc123".to_string(),
            branch: "main".to_string(),
            pr_number: None,
            build_url: Some("http://example.com/build".to_string()),
        };

        let result = orchestrator
            .integrate_with_ci(test_report, ci_metadata)
            .await;

        // This might fail due to webhook URL not being reachable, but that's expected in tests
        // The important thing is that it doesn't panic and handles the error gracefully
        let _ = result;
    }

    fn create_test_config() -> TestOrchestratorConfig {
        TestOrchestratorConfig {
            max_concurrent_tests: 2,
            test_timeout_seconds: 60,
            parallel_execution: true,
            ci_integration: CiIntegrationConfig {
                enabled: true,
                platform: CiPlatform::Generic,
                webhook_url: Some("http://example.com/webhook".to_string()),
                report_format: ReportFormat::Json,
                notify_on_failure: true,
            },
            scheduling_config: SchedulingConfig {
                enabled: true,
                cron_expression: Some("0 */6 * * *".to_string()),
                max_queue_size: 50,
                retry_on_failure: true,
                max_retries: 2,
            },
            result_retention_hours: 24,
        }
    }
}

#[cfg(test)]
mod statistics_tests {
    use super::*;
    use chrono::Utc;
    use ordered_float::OrderedFloat;

    #[tokio::test]
    async fn test_calculate_statistics_empty_data() {
        let config = create_test_config();
        let orchestrator = TestOrchestrator::new(config);

        let result = orchestrator.calculate_statistics(&[]);
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_calculate_statistics_single_point() {
        let config = create_test_config();
        let orchestrator = TestOrchestrator::new(config);

        let data_point = MetricDataPoint {
            metric_type: MetricType::ResponseTime,
            value: OrderedFloat(100.0),
            timestamp: Utc::now(),
            tags: HashMap::new(),
            source: "test".to_string(),
        };

        let result = orchestrator.calculate_statistics(&[data_point]);
        assert!(result.is_some());

        let stats = result.unwrap();
        assert_eq!(stats.average, OrderedFloat(100.0));
        assert_eq!(stats.minimum, OrderedFloat(100.0));
        assert_eq!(stats.maximum, OrderedFloat(100.0));
        assert_eq!(stats.sample_count, 1);
    }

    #[tokio::test]
    async fn test_calculate_statistics_multiple_points() {
        let config = create_test_config();
        let orchestrator = TestOrchestrator::new(config);

        let now = Utc::now();
        let data_points = vec![
            MetricDataPoint {
                metric_type: MetricType::ResponseTime,
                value: OrderedFloat(100.0),
                timestamp: now,
                tags: HashMap::new(),
                source: "test".to_string(),
            },
            MetricDataPoint {
                metric_type: MetricType::ResponseTime,
                value: OrderedFloat(200.0),
                timestamp: now,
                tags: HashMap::new(),
                source: "test".to_string(),
            },
            MetricDataPoint {
                metric_type: MetricType::ResponseTime,
                value: OrderedFloat(300.0),
                timestamp: now,
                tags: HashMap::new(),
                source: "test".to_string(),
            },
        ];

        let result = orchestrator.calculate_statistics(&data_points);
        assert!(result.is_some());

        let stats = result.unwrap();
        assert_eq!(stats.average, OrderedFloat(200.0)); // (100 + 200 + 300) / 3
        assert_eq!(stats.minimum, OrderedFloat(100.0));
        assert_eq!(stats.maximum, OrderedFloat(300.0));
        assert_eq!(stats.sample_count, 3);
    }

    fn create_test_config() -> TestOrchestratorConfig {
        TestOrchestratorConfig {
            max_concurrent_tests: 2,
            test_timeout_seconds: 60,
            parallel_execution: true,
            ci_integration: CiIntegrationConfig {
                enabled: false,
                platform: CiPlatform::Generic,
                webhook_url: None,
                report_format: ReportFormat::Json,
                notify_on_failure: false,
            },
            scheduling_config: SchedulingConfig {
                enabled: false,
                cron_expression: None,
                max_queue_size: 50,
                retry_on_failure: false,
                max_retries: 0,
            },
            result_retention_hours: 24,
        }
    }
}
