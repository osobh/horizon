//! # ExoRust Operational Tooling
//!
//! Production operational tooling for ExoRust agent deployment, monitoring, and management.
//! This crate provides the core infrastructure for blue-green deployments, canary testing,
//! rollback mechanisms, performance monitoring, and resource usage tracking.

#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod canary;
pub mod deployment;
pub mod error;
pub mod monitoring;
pub mod rollback;

pub use error::{OperationalError, OperationalResult};

/// Re-export commonly used types for convenience
pub mod prelude {
    pub use crate::canary::{CanaryConfig, CanaryResult, CanaryTesting, TestMetrics};
    pub use crate::deployment::{
        BlueGreenDeployment, DeploymentConfig, DeploymentStatus, DeploymentStrategy,
    };
    pub use crate::error::{OperationalError, OperationalResult};
    pub use crate::monitoring::{
        AgentAnalytics, GpuUtilization, OperationalMetrics, ResourceMonitor,
    };
    pub use crate::rollback::{CheckpointRef, RollbackManager, RollbackPlan, RollbackStrategy};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_exports() {
        // Test that main modules are accessible
        use prelude::*;

        // This test ensures our prelude exports are working
        let _error: OperationalError = OperationalError::DeploymentFailed("test".to_string());
    }

    #[test]
    fn test_prelude_canary_types() {
        use prelude::*;

        // Verify all canary types are accessible
        let _config = CanaryConfig::default();
        let _metrics = TestMetrics {
            total_requests: 1000,
            successful_requests: 1000,
            failed_requests: 0,
            avg_response_time_ms: 50.0,
            p95_response_time_ms: 100.0,
            throughput_rps: 1000.0,
            error_rate: 0.0,
        };
    }

    #[test]
    fn test_prelude_deployment_types() {
        use prelude::*;

        // Verify all deployment types are accessible
        let _status = DeploymentStatus::Preparing;
        let _strategy = DeploymentStrategy::BlueGreen;
        let _config = crate::deployment::DeploymentConfig {
            deployment_id: "test".to_string(),
            strategy: DeploymentStrategy::BlueGreen,
            agent_config: crate::deployment::AgentConfig::default(),
            resource_requirements: crate::deployment::ResourceRequirements::default(),
            health_check: crate::deployment::HealthCheckConfig::default(),
            timeout: std::time::Duration::from_secs(60),
            auto_rollback: true,
        };
    }

    #[test]
    fn test_prelude_monitoring_types() {
        use prelude::*;

        // Verify all monitoring types are accessible
        let _metrics = OperationalMetrics {
            timestamp: std::time::SystemTime::now(),
            resource_usage: crate::monitoring::ResourceMetrics::default(),
            performance: crate::monitoring::PerformanceMetrics::default(),
            errors: crate::monitoring::ErrorMetrics::default(),
        };
    }

    #[test]
    fn test_prelude_rollback_types() {
        use prelude::*;

        // Verify all rollback types are accessible
        let _strategy = RollbackStrategy::Immediate;
        let _checkpoint = CheckpointRef {
            checkpoint_id: "checkpoint-1".to_string(),
            timestamp: chrono::Utc::now(),
            version: "1.0.0".to_string(),
            metadata: std::collections::HashMap::new(),
        };
    }

    #[test]
    fn test_error_conversions() {
        use std::io;

        // Test From trait implementations
        let io_error = io::Error::new(io::ErrorKind::NotFound, "file not found");
        let op_error: OperationalError = io_error.into();
        assert!(matches!(op_error, OperationalError::IoError(_)));
    }

    #[test]
    fn test_result_type_alias() {
        // Test that OperationalResult works correctly
        fn test_function() -> OperationalResult<String> {
            Ok("success".to_string())
        }

        let result = test_function();
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "success");

        fn test_error_function() -> OperationalResult<String> {
            Err(OperationalError::ConfigurationError(
                "test error".to_string(),
            ))
        }

        let error_result = test_error_function();
        assert!(error_result.is_err());
    }

    #[test]
    fn test_all_modules_present() {
        // Ensure all expected modules are compiled
        use crate::canary;
        use crate::deployment;
        use crate::error;
        use crate::monitoring;
        use crate::rollback;

        // This test will fail to compile if any module is missing
        let _ = error::OperationalError::ConfigurationError("test".to_string());
        let _ = canary::CanaryTesting::new();
        let _ = deployment::BlueGreenDeployment::new(std::path::PathBuf::from("/tmp"));
        let _ = monitoring::OperationalMetrics::default();
        let _ = rollback::RollbackManager::new();
    }

    #[test]
    fn test_prelude_completeness() {
        // Verify prelude includes all essential types
        use prelude::*;

        // Create test metrics
        let metrics = TestMetrics {
            total_requests: 1000,
            successful_requests: 999,
            failed_requests: 1,
            avg_response_time_ms: 10.0,
            p95_response_time_ms: 20.0,
            throughput_rps: 1000.0,
            error_rate: 0.1,
        };

        let deployment_plan = RollbackPlan {
            plan_id: "test-plan".to_string(),
            deployment_id: "test-deploy".to_string(),
            strategy: RollbackStrategy::Gradual {
                steps: 5,
                interval: std::time::Duration::from_secs(60),
            },
            previous_state: CheckpointRef {
                checkpoint_id: "test".to_string(),
                timestamp: chrono::Utc::now(),
                version: "1.0.0".to_string(),
                metadata: Default::default(),
            },
            timeout: std::time::Duration::from_secs(300),
            verify_success: true,
            created_at: chrono::Utc::now(),
        };

        // Ensure we can use the types
        assert!(metrics.error_rate < 1.0);
        assert_eq!(deployment_plan.timeout.as_secs(), 300);
    }

    #[test]
    fn test_operational_tooling_feature_flags() {
        // Test that feature-gated code compiles correctly
        #[cfg(feature = "metrics")]
        {
            let _metrics = crate::monitoring::OperationalMetrics::default();
        }

        // Test succeeds if compilation works
        assert!(true);
    }
}
