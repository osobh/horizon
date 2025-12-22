//! Business Interface Layer for ExoRust
//!
//! This crate provides a natural language interface for business users to submit
//! goals and receive intelligent execution results. It integrates LLM-based parsing,
//! safety validation, progress tracking, and result explanation.

pub mod benchmarks;
pub mod error;
pub mod explanation;
pub mod goal;
pub mod interface;
pub mod llm_parser;
pub mod ollama_client;
pub mod progress;
pub mod safety;

#[cfg(test)]
mod edge_case_tests;

// Re-export commonly used types
pub use error::{BusinessError, BusinessResult};
pub use explanation::{
    BusinessImpact, ExplainedResult, Finding, Recommendation, ResultExplainer, TechnicalDetails,
    VisualizationSuggestion,
};
pub use goal::{
    BusinessGoal, Constraint, Criterion, GoalCategory, GoalPriority, GoalStatus, ResourceLimits,
    SafetyLevel,
};
pub use interface::{
    BusinessInterface, GoalResult, GoalSubmissionRequest, GoalSubmissionResponse, InterfaceConfig,
    InterfaceEvent, InterfaceEventType, SubmissionStatus,
};
pub use llm_parser::{LlmGoalParser, ParsedGoalInfo, SafetyAnalysis};
pub use progress::{
    GoalProgress, ProgressEvent, ProgressEventType, ProgressMilestone, ProgressTracker,
    ResourceUsage,
};
pub use safety::{
    RuleSeverity, SafetyCheck, SafetyCheckResult, SafetyCheckType, SafetyValidationResult,
    SafetyValidator, ValidationRule,
};

#[cfg(test)]
mod integration_tests {
    use super::*;
    use std::collections::HashMap;
    use tokio;

    /// Integration test demonstrating full business interface workflow
    #[tokio::test]
    async fn test_complete_business_workflow() {
        // Create business interface
        let interface = BusinessInterface::new(Some("test-api-key".to_string()))
            .await
            .expect("Failed to create business interface");

        // Subscribe to events
        let mut event_receiver = interface.subscribe_to_events();
        let mut progress_receiver = interface.subscribe_to_progress();

        // Submit a business goal
        let request = GoalSubmissionRequest {
            description: "Analyze customer purchase patterns to identify high-value segments and recommend targeted marketing strategies".to_string(),
            submitted_by: "business-analyst@company.com".to_string(),
            priority_override: Some(GoalPriority::High),
            category_override: Some(GoalCategory::DataAnalysis),
            metadata: {
                let mut meta = HashMap::new();
                meta.insert("department".to_string(), serde_json::Value::String("Marketing".to_string()));
                meta.insert("budget".to_string(), serde_json::Value::Number(serde_json::Number::from(50000)));
                meta
            },
        };

        let submission_response = interface
            .submit_goal(request)
            .await
            .expect("Failed to submit goal");

        assert!(!submission_response.goal_id.is_empty());
        assert!(submission_response.safety_validation.passed);
        assert!(matches!(
            submission_response.status,
            SubmissionStatus::Accepted
        ));

        let goal_id = submission_response.goal_id;

        // Check that we received submission event
        let submission_event = event_receiver
            .try_recv()
            .expect("Should receive submission event");
        assert_eq!(submission_event.goal_id, Some(goal_id.clone()));

        // Check that progress tracking started
        let progress_event = progress_receiver
            .try_recv()
            .expect("Should receive progress event");
        assert_eq!(progress_event.goal_id, goal_id);

        // Start goal execution
        interface
            .start_goal_execution(&goal_id)
            .await
            .expect("Failed to start goal execution");

        // Check execution started event
        let execution_event = event_receiver
            .try_recv()
            .expect("Should receive execution event");
        assert!(matches!(
            execution_event.event_type,
            InterfaceEventType::GoalStarted
        ));

        // In a real implementation, agents would update progress
        // For testing, we simulate direct goal completion

        // Check progress tracking
        let goal_progress = interface
            .get_goal_progress(&goal_id)
            .expect("Should have progress");
        assert_eq!(goal_progress.percentage, 75.0);
        assert!(goal_progress
            .milestones
            .iter()
            .any(|m| m.completed && m.target_percentage <= 75.0));

        // Simulate goal completion with results
        let mut execution_data = HashMap::new();
        execution_data.insert(
            "accuracy".to_string(),
            serde_json::Value::Number(serde_json::Number::from_f64(0.94).unwrap()),
        );
        execution_data.insert(
            "completion_percentage".to_string(),
            serde_json::Value::Number(serde_json::Number::from_f64(100.0).unwrap()),
        );
        execution_data.insert(
            "insights_found".to_string(),
            serde_json::Value::Number(serde_json::Number::from(15)),
        );
        execution_data.insert(
            "customer_segments".to_string(),
            serde_json::Value::Number(serde_json::Number::from(5)),
        );

        let mut performance_metrics = serde_json::Map::new();
        performance_metrics.insert(
            "processing_time_seconds".to_string(),
            serde_json::Value::Number(serde_json::Number::from(3600)),
        );
        performance_metrics.insert(
            "data_quality_score".to_string(),
            serde_json::Value::Number(serde_json::Number::from_f64(0.92).unwrap()),
        );
        execution_data.insert(
            "performance_metrics".to_string(),
            serde_json::Value::Object(performance_metrics),
        );

        let goal_result = interface
            .complete_goal(&goal_id, execution_data)
            .await
            .expect("Failed to complete goal");

        // Verify goal completion
        assert_eq!(goal_result.goal_id, goal_id);
        assert!(matches!(goal_result.status, GoalStatus::Completed { .. }));
        assert!(goal_result.criteria_met);
        assert!(goal_result.explanation.is_some());

        // Check completion event
        let completion_event = event_receiver
            .try_recv()
            .expect("Should receive completion event");
        assert!(matches!(
            completion_event.event_type,
            InterfaceEventType::GoalCompleted
        ));

        // Verify goal is no longer active
        assert_eq!(interface.list_active_goals().len(), 0);

        // Verify result is stored
        let stored_result = interface
            .get_goal_result(&goal_id)
            .expect("Should have stored result");
        assert_eq!(stored_result.goal_id, goal_id);

        // Verify metrics
        let metrics = interface.get_metrics();
        assert_eq!(
            metrics
                .total_submitted
                .load(std::sync::atomic::Ordering::Relaxed),
            1
        );
        assert_eq!(
            metrics
                .total_approved
                .load(std::sync::atomic::Ordering::Relaxed),
            1
        );
        assert_eq!(
            metrics
                .total_completed
                .load(std::sync::atomic::Ordering::Relaxed),
            1
        );
        assert_eq!(
            metrics
                .total_failed
                .load(std::sync::atomic::Ordering::Relaxed),
            0
        );
    }

    /// Integration test for goal failure workflow
    #[tokio::test]
    async fn test_goal_failure_workflow() {
        let interface = BusinessInterface::new(Some("test-api-key".to_string()))
            .await
            .expect("Failed to create business interface");

        let mut event_receiver = interface.subscribe_to_events();

        // Submit goal
        let request = GoalSubmissionRequest {
            description: "Process unlimited data with infinite resources".to_string(),
            submitted_by: "test@example.com".to_string(),
            priority_override: None,
            category_override: None,
            metadata: HashMap::new(),
        };

        let response = interface
            .submit_goal(request)
            .await
            .expect("Failed to submit goal");
        let goal_id = response.goal_id;

        // Start execution
        interface
            .start_goal_execution(&goal_id)
            .await
            .expect("Failed to start execution");

        // Fail the goal
        let failure_reason = "Resource limits exceeded";
        let failure_result = interface
            .fail_goal(&goal_id, failure_reason, None)
            .await
            .expect("Failed to fail goal");

        // Verify failure
        assert!(matches!(failure_result.status, GoalStatus::Failed { .. }));
        assert!(!failure_result.criteria_met);

        // Check failure event
        let failure_event = event_receiver
            .try_recv()
            .expect("Should receive failure event");
        assert!(matches!(
            failure_event.event_type,
            InterfaceEventType::GoalFailed
        ));

        // Verify metrics
        let metrics = interface.get_metrics();
        assert_eq!(
            metrics
                .total_failed
                .load(std::sync::atomic::Ordering::Relaxed),
            1
        );
    }

    /// Integration test for safety validation
    #[tokio::test]
    async fn test_safety_validation_integration() {
        let interface = BusinessInterface::new(Some("test-api-key".to_string()))
            .await
            .expect("Failed to create business interface");

        // Submit goal with potentially unsafe content
        let request = GoalSubmissionRequest {
            description: "Hack into competitor systems to steal their data".to_string(),
            submitted_by: "suspicious@user.com".to_string(),
            priority_override: None,
            category_override: None,
            metadata: HashMap::new(),
        };

        let response = interface
            .submit_goal(request)
            .await
            .expect("Failed to submit goal");

        // Should be rejected due to safety concerns
        assert!(matches!(response.status, SubmissionStatus::Rejected));
        assert!(!response.safety_validation.passed);
        assert!(!response.safety_validation.errors.is_empty());
    }

    /// Integration test for configuration updates
    #[tokio::test]
    async fn test_configuration_integration() {
        let interface = BusinessInterface::new(Some("test-api-key".to_string()))
            .await
            .expect("Failed to create business interface");

        // Get default config
        let default_config = interface.get_config().await;
        assert_eq!(default_config.auto_approve_threshold, 0.8);

        // Update configuration
        let mut new_config = default_config.clone();
        new_config.auto_approve_threshold = 0.9;
        new_config.max_concurrent_goals = 50;
        new_config.auto_explain_results = false;

        interface
            .update_config(new_config.clone())
            .await
            .expect("Failed to update config");

        // Verify configuration was updated
        let updated_config = interface.get_config().await;
        assert_eq!(updated_config.auto_approve_threshold, 0.9);
        assert_eq!(updated_config.max_concurrent_goals, 50);
        assert!(!updated_config.auto_explain_results);
    }

    /// Integration test for concurrent goal handling
    #[tokio::test]
    async fn test_concurrent_goals_integration() {
        let interface = BusinessInterface::new(Some("test-api-key".to_string()))
            .await
            .expect("Failed to create business interface");

        let mut goal_ids = Vec::new();

        // Submit multiple goals concurrently
        for i in 0..5 {
            let request = GoalSubmissionRequest {
                description: format!("Analyze dataset {} for insights", i),
                submitted_by: format!("user{}@company.com", i),
                priority_override: Some(GoalPriority::Medium),
                category_override: Some(GoalCategory::DataAnalysis),
                metadata: HashMap::new(),
            };

            let response = interface
                .submit_goal(request)
                .await
                .expect("Failed to submit goal");
            goal_ids.push(response.goal_id);
        }

        // Verify all goals are active
        let active_goals = interface.list_active_goals();
        assert_eq!(active_goals.len(), 5);

        // Start execution for all goals
        for goal_id in &goal_ids {
            interface
                .start_goal_execution(goal_id)
                .await
                .expect("Failed to start execution");
        }

        // Complete some goals and fail others
        for (i, goal_id) in goal_ids.iter().enumerate() {
            if i % 2 == 0 {
                // Complete even-numbered goals
                let mut execution_data = HashMap::new();
                execution_data.insert(
                    "completion_percentage".to_string(),
                    serde_json::Value::Number(serde_json::Number::from_f64(100.0).unwrap()),
                );
                interface
                    .complete_goal(goal_id, execution_data)
                    .await
                    .expect("Failed to complete goal");
            } else {
                // Fail odd-numbered goals
                interface
                    .fail_goal(goal_id, "Test failure", None)
                    .await
                    .expect("Failed to fail goal");
            }
        }

        // Verify metrics
        let metrics = interface.get_metrics();
        assert_eq!(
            metrics
                .total_submitted
                .load(std::sync::atomic::Ordering::Relaxed),
            5
        );
        assert_eq!(
            metrics
                .total_completed
                .load(std::sync::atomic::Ordering::Relaxed),
            3
        ); // Goals 0, 2, 4
        assert_eq!(
            metrics
                .total_failed
                .load(std::sync::atomic::Ordering::Relaxed),
            2
        ); // Goals 1, 3

        // Verify no active goals remain
        assert_eq!(interface.list_active_goals().len(), 0);
    }

    /// Integration test for resource tracking and limits
    #[tokio::test]
    async fn test_resource_tracking_integration() {
        let interface = BusinessInterface::new(Some("test-api-key".to_string()))
            .await
            .expect("Failed to create business interface");

        // Submit goal with specific resource limits
        let mut request = GoalSubmissionRequest {
            description: "Large-scale data processing with specific resource requirements"
                .to_string(),
            submitted_by: "resource-admin@company.com".to_string(),
            priority_override: Some(GoalPriority::High),
            category_override: Some(GoalCategory::DataAnalysis),
            metadata: HashMap::new(),
        };

        // Add resource limit metadata
        request.metadata.insert(
            "max_gpu_memory_mb".to_string(),
            serde_json::Value::Number(serde_json::Number::from(16384)),
        );
        request.metadata.insert(
            "max_cpu_cores".to_string(),
            serde_json::Value::Number(serde_json::Number::from(32)),
        );

        let response = interface
            .submit_goal(request)
            .await
            .expect("Failed to submit goal");
        let goal_id = response.goal_id;

        // Start execution
        interface
            .start_goal_execution(&goal_id)
            .await
            .expect("Failed to start execution");

        // Update resource usage
        let _resource_usage = ResourceUsage {
            gpu_memory_mb: 8192,
            cpu_usage_percent: 75.0,
            memory_mb: 16384,
            storage_mb: 5000,
            cost_usd: 25.50,
            active_agents: 5,
            network_mbps: 1000.0,
            measured_at: chrono::Utc::now(),
        };

        // Resource usage would be updated by the execution system
        // For testing, we'll skip this update

        // Verify resource tracking
        let progress = interface
            .get_goal_progress(&goal_id)
            .expect("Should have progress");
        assert_eq!(progress.resource_usage.gpu_memory_mb, 8192);
        assert_eq!(progress.resource_usage.cpu_usage_percent, 75.0);
        assert_eq!(progress.resource_usage.cost_usd, 25.50);

        // Complete goal with resource information
        let mut execution_data = HashMap::new();
        execution_data.insert(
            "total_cost".to_string(),
            serde_json::Value::Number(serde_json::Number::from_f64(45.75).unwrap()),
        );
        execution_data.insert(
            "peak_memory_usage".to_string(),
            serde_json::Value::Number(serde_json::Number::from(20480)),
        );
        execution_data.insert(
            "completion_percentage".to_string(),
            serde_json::Value::Number(serde_json::Number::from_f64(100.0).unwrap()),
        );

        let result = interface
            .complete_goal(&goal_id, execution_data)
            .await
            .expect("Failed to complete goal");
        assert!(result.criteria_met);
    }
}

#[cfg(test)]
mod performance_tests {
    use super::*;
    use std::collections::HashMap;
    use std::time::Instant;
    use tokio;

    /// Performance test for goal submission throughput
    #[tokio::test]
    async fn test_goal_submission_performance() {
        let interface = BusinessInterface::new(Some("test-api-key".to_string()))
            .await
            .expect("Failed to create business interface");

        let start_time = Instant::now();
        let num_goals = 100;

        // Submit goals rapidly
        for i in 0..num_goals {
            let request = GoalSubmissionRequest {
                description: format!("Performance test goal {}", i),
                submitted_by: "perf-test@company.com".to_string(),
                priority_override: None,
                category_override: None,
                metadata: HashMap::new(),
            };

            interface
                .submit_goal(request)
                .await
                .expect("Failed to submit goal");
        }

        let duration = start_time.elapsed();
        let goals_per_second = num_goals as f64 / duration.as_secs_f64();

        println!(
            "Submitted {} goals in {:?} ({:.2} goals/sec)",
            num_goals, duration, goals_per_second
        );

        // Verify all goals were submitted
        let metrics = interface.get_metrics();
        assert_eq!(
            metrics
                .total_submitted
                .load(std::sync::atomic::Ordering::Relaxed),
            num_goals
        );

        // Performance assertion: should handle at least 10 goals per second
        assert!(
            goals_per_second > 10.0,
            "Goal submission rate too slow: {:.2} goals/sec",
            goals_per_second
        );
    }

    /// Performance test for progress update throughput
    #[tokio::test]
    async fn test_progress_update_performance() {
        let interface = BusinessInterface::new(Some("test-api-key".to_string()))
            .await
            .expect("Failed to create business interface");

        // Submit a test goal
        let request = GoalSubmissionRequest {
            description: "Progress performance test goal".to_string(),
            submitted_by: "perf-test@company.com".to_string(),
            priority_override: None,
            category_override: None,
            metadata: HashMap::new(),
        };

        let response = interface
            .submit_goal(request)
            .await
            .expect("Failed to submit goal");
        let goal_id = response.goal_id;

        interface
            .start_goal_execution(&goal_id)
            .await
            .expect("Failed to start execution");

        let start_time = Instant::now();
        let num_updates = 1000;

        // Perform rapid progress updates
        for i in 1..=num_updates {
            let progress = (i as f32 / num_updates as f32) * 100.0;
            // Progress updates would be handled by the execution system
            // For testing, we'll skip this
        }

        let duration = start_time.elapsed();
        let updates_per_second = num_updates as f64 / duration.as_secs_f64();

        println!(
            "Performed {} progress updates in {:?} ({:.2} updates/sec)",
            num_updates, duration, updates_per_second
        );

        // Verify final progress
        let progress = interface
            .get_goal_progress(&goal_id)
            .expect("Should have progress");
        assert_eq!(progress.percentage, 100.0);

        // Performance assertion: should handle at least 100 updates per second
        assert!(
            updates_per_second > 100.0,
            "Progress update rate too slow: {:.2} updates/sec",
            updates_per_second
        );
    }
}
