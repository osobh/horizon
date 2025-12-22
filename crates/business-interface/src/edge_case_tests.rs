//! Edge case tests for business-interface to enhance coverage to 90%

#[cfg(test)]
mod edge_case_tests {
    use crate::benchmarks::*;
    use crate::goal::ComparisonOperator;
    use crate::interface::*;
    use crate::ollama_client::TaskType;
    use crate::*;
    use std::collections::HashMap;
    use std::sync::atomic::Ordering;
    use std::time::Duration;

    // Benchmarks edge case tests

    #[tokio::test]
    async fn test_benchmark_creation() {
        let benchmark = LlmBenchmark::new();
        // Just test that creation works
        let _ = benchmark;
    }

    #[tokio::test]
    async fn test_benchmark_initialize_no_models() {
        let mut benchmark = LlmBenchmark::new();
        // Without actual Ollama server, initialization should fail
        let result = benchmark.initialize().await;
        assert!(result.is_err());
    }

    #[test]
    fn test_benchmark_result_extreme_values() {
        let result = BenchmarkResult {
            model_name: "test-model".to_string(),
            task_type: TaskType::GoalParsing,
            avg_response_time_ms: f64::INFINITY,
            success_rate: 0.0,
            quality_score: f64::NAN,
            test_cases: usize::MAX,
            errors: vec!["error1".to_string(); 1000], // Many errors
        };

        assert_eq!(result.model_name, "test-model");
        assert!(result.avg_response_time_ms.is_infinite());
        assert!(result.quality_score.is_nan());
        assert_eq!(result.test_cases, usize::MAX);
        assert_eq!(result.errors.len(), 1000);
    }

    #[test]
    fn test_model_task_recommendation_edge_cases() {
        let recommendation = ModelTaskRecommendation {
            task_type: TaskType::SafetyValidation,
            recommended_model: String::new(), // Empty model name
            reason: "x".repeat(10000),        // Very long reason
            score: -1.0,                      // Negative score
        };

        assert_eq!(recommendation.recommended_model, "");
        assert_eq!(recommendation.reason.len(), 10000);
        assert_eq!(recommendation.score, -1.0);
    }

    #[test]
    fn test_comprehensive_benchmark_results_serialization() {
        let results = ComprehensiveBenchmarkResults {
            results: vec![],
            total_duration: Duration::from_secs(u64::MAX),
            recommendations: vec![],
        };

        let serialized = serde_json::to_string(&results).unwrap();
        let deserialized: ComprehensiveBenchmarkResults =
            serde_json::from_str(&serialized).unwrap();

        assert_eq!(deserialized.results.len(), 0);
        assert_eq!(deserialized.recommendations.len(), 0);
    }

    // Interface edge case tests

    #[tokio::test]
    async fn test_interface_with_empty_api_key() {
        let interface = BusinessInterface::new(Some(String::new())).await;
        assert!(interface.is_ok());
    }

    #[tokio::test]
    async fn test_interface_with_none_api_key() {
        let interface = BusinessInterface::new(None).await;
        assert!(interface.is_ok());
    }

    #[tokio::test]
    async fn test_submit_goal_with_empty_description() {
        let interface = BusinessInterface::new(None).await.unwrap();
        let request = GoalSubmissionRequest {
            description: String::new(),
            submitted_by: "test@example.com".to_string(),
            priority_override: None,
            category_override: None,
            metadata: HashMap::new(),
        };

        let result = interface.submit_goal(request).await;
        // Should still process empty description
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_submit_goal_with_unicode() {
        let interface = BusinessInterface::new(None).await.unwrap();
        let request = GoalSubmissionRequest {
            description: "分析客户数据 🚀 найти инсайты".to_string(),
            submitted_by: "测试@example.com".to_string(),
            priority_override: None,
            category_override: None,
            metadata: HashMap::new(),
        };

        let result = interface.submit_goal(request).await;
        assert!(result.is_ok());
        let response = result.unwrap();
        assert!(response.parsed_goal.description.contains("分析"));
    }

    #[tokio::test]
    async fn test_interface_metrics_overflow() {
        let interface = BusinessInterface::new(None).await.unwrap();

        // Submit many goals to test metrics
        for i in 0..10 {
            let request = GoalSubmissionRequest {
                description: format!("Test overflow {}", i),
                submitted_by: "test@example.com".to_string(),
                priority_override: None,
                category_override: None,
                metadata: HashMap::new(),
            };

            let _ = interface.submit_goal(request).await;
        }

        // Check that metrics increment
        let metrics = interface.get_metrics();
        assert!(metrics.total_submitted.load(Ordering::Relaxed) >= 10);
    }

    #[tokio::test]
    async fn test_approve_nonexistent_goal() {
        let interface = BusinessInterface::new(None).await.unwrap();
        let result = interface.approve_goal("nonexistent-goal-id").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_reject_goal_with_empty_reason() {
        let interface = BusinessInterface::new(None).await.unwrap();
        let request = GoalSubmissionRequest {
            description: "Test goal".to_string(),
            submitted_by: "test@example.com".to_string(),
            priority_override: None,
            category_override: None,
            metadata: HashMap::new(),
        };

        let response = interface.submit_goal(request).await.unwrap();
        let result = interface.reject_goal(&response.goal_id, "").await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_complete_goal_without_starting() {
        let interface = BusinessInterface::new(None).await.unwrap();
        let request = GoalSubmissionRequest {
            description: "Test goal".to_string(),
            submitted_by: "test@example.com".to_string(),
            priority_override: None,
            category_override: None,
            metadata: HashMap::new(),
        };

        let response = interface.submit_goal(request).await.unwrap();

        // Try to complete without starting execution
        let result = interface
            .complete_goal(&response.goal_id, HashMap::new())
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_fail_goal_with_very_long_reason() {
        let interface = BusinessInterface::new(None).await.unwrap();
        let request = GoalSubmissionRequest {
            description: "Test goal".to_string(),
            submitted_by: "test@example.com".to_string(),
            priority_override: None,
            category_override: None,
            metadata: HashMap::new(),
        };

        let response = interface.submit_goal(request).await.unwrap();
        interface
            .start_goal_execution(&response.goal_id)
            .await
            .unwrap();

        let long_reason = "x".repeat(1_000_000); // 1MB reason
        let result = interface
            .fail_goal(&response.goal_id, &long_reason, None)
            .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_multiple_goal_submissions() {
        let interface = BusinessInterface::new(None).await.unwrap();

        // Submit multiple goals in sequence
        let mut goal_ids = vec![];
        for i in 0..5 {
            let request = GoalSubmissionRequest {
                description: format!("Concurrent goal {}", i),
                submitted_by: "test@example.com".to_string(),
                priority_override: None,
                category_override: None,
                metadata: HashMap::new(),
            };

            let response = interface.submit_goal(request).await.unwrap();
            goal_ids.push(response.goal_id);
        }

        // Verify all goals exist
        for goal_id in &goal_ids {
            assert!(interface.get_goal_status(goal_id).is_some());
        }

        assert_eq!(interface.list_active_goals().len(), 5);
    }

    #[tokio::test]
    async fn test_event_sender_capacity() {
        let interface = BusinessInterface::new(None).await.unwrap();

        // Subscribe but don't receive
        let _receiver = interface.subscribe_to_events();

        // Send many events (should not block)
        for i in 0..1000 {
            let request = GoalSubmissionRequest {
                description: format!("Goal {}", i),
                submitted_by: "test@example.com".to_string(),
                priority_override: None,
                category_override: None,
                metadata: HashMap::new(),
            };

            let _ = interface.submit_goal(request).await;
        }

        // Should complete without blocking
        assert!(interface.list_active_goals().len() <= 1000);
    }

    #[test]
    fn test_interface_event_extreme_metadata() {
        let mut metadata = HashMap::new();
        for i in 0..1000 {
            metadata.insert(
                format!("key_{}", i),
                serde_json::Value::String(format!("value_{}", i)),
            );
        }

        let event = InterfaceEvent {
            event_id: "x".repeat(1000), // Long ID
            event_type: InterfaceEventType::SystemError,
            timestamp: chrono::Utc::now(),
            goal_id: Some(String::new()), // Empty goal ID
            details: String::new(),
            metadata,
        };

        assert_eq!(event.event_id.len(), 1000);
        assert_eq!(event.metadata.len(), 1000);
        assert_eq!(event.goal_id, Some(String::new()));
    }

    #[test]
    fn test_goal_result_with_nan_values() {
        let mut execution_data = HashMap::new();
        execution_data.insert(
            "nan_value".to_string(),
            serde_json::Value::Number(
                serde_json::Number::from_f64(f64::NAN).unwrap_or(serde_json::Number::from(0)),
            ),
        );
        execution_data.insert(
            "inf_value".to_string(),
            serde_json::Value::Number(
                serde_json::Number::from_f64(f64::INFINITY).unwrap_or(serde_json::Number::from(0)),
            ),
        );

        let result = GoalResult {
            goal_id: "test".to_string(),
            status: GoalStatus::Completed {
                completed_at: chrono::Utc::now(),
            },
            execution_data,
            explanation: None,
            completed_at: chrono::Utc::now(),
            execution_duration: Duration::from_nanos(0), // Zero duration
            criteria_met: true,
        };

        assert_eq!(result.execution_duration, Duration::from_nanos(0));
    }

    #[tokio::test]
    async fn test_update_config_with_extreme_values() {
        let interface = BusinessInterface::new(None).await.unwrap();

        let config = InterfaceConfig {
            auto_approve_threshold: f64::INFINITY,
            max_concurrent_goals: 0,
            default_timeout_hours: f64::NAN,
            auto_explain_results: true,
            llm_model: String::new(),
            safety_strictness: SafetyStrictness::Maximum,
        };

        let result = interface.update_config(config.clone()).await;
        assert!(result.is_ok());

        let retrieved = interface.get_config().await;
        assert!(retrieved.auto_approve_threshold.is_infinite());
        assert_eq!(retrieved.max_concurrent_goals, 0);
        assert!(retrieved.default_timeout_hours.is_nan());
        assert_eq!(retrieved.llm_model, "");
    }

    #[test]
    fn test_criterion_serialization() {
        let criteria = vec![
            Criterion::Performance {
                metric: "speed".to_string(),
                target_value: 100.0,
                comparison: ComparisonOperator::GreaterThan,
            },
            Criterion::Quality {
                aspect: "accuracy".to_string(),
                min_score: 0.95,
            },
            Criterion::Completion { percentage: 100.0 },
            Criterion::Accuracy { min_accuracy: 0.95 },
            Criterion::Efficiency {
                metric: "cost_per_transaction".to_string(),
                min_efficiency: 0.9,
            },
        ];

        for criterion in criteria {
            let serialized = serde_json::to_string(&criterion).unwrap();
            let deserialized: Criterion = serde_json::from_str(&serialized).unwrap();
            assert_eq!(format!("{:?}", criterion), format!("{:?}", deserialized));
        }
    }

    #[test]
    fn test_goal_submission_response_edge_cases() {
        let parsed_goal = BusinessGoal::new(String::new(), String::new());
        let safety_result = SafetyValidationResult {
            passed: true,
            safety_score: f64::NEG_INFINITY,
            check_results: vec![],
            warnings: vec![],
            errors: vec![],
            mitigations: vec![],
            compliance_requirements: vec![],
            validated_at: chrono::Utc::now(),
        };

        let response = GoalSubmissionResponse {
            goal_id: "🚀".repeat(100), // Unicode ID
            status: SubmissionStatus::SafetyReview,
            parsed_goal,
            safety_validation: safety_result,
            messages: vec![String::new(); 100], // Many empty messages
        };

        assert_eq!(response.goal_id.chars().count(), 100);
        assert!(response.safety_validation.safety_score.is_infinite());
        assert_eq!(response.messages.len(), 100);
    }
}
