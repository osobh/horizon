//! Cost optimization error types

use thiserror::Error;

/// Cost optimization error types
#[derive(Debug, Error)]
pub enum CostOptimizationError {
    /// Resource allocation failed
    #[error("Resource allocation failed: {reason}")]
    AllocationFailed { reason: String },

    /// Budget exceeded
    #[error("Budget exceeded: spent ${spent:.2}, budget ${budget:.2}")]
    BudgetExceeded { spent: f64, budget: f64 },

    /// No available resources
    #[error("No available resources: {resource_type}")]
    NoAvailableResources { resource_type: String },

    /// Spot instance terminated
    #[error("Spot instance terminated: {instance_id} - {reason}")]
    SpotInstanceTerminated { instance_id: String, reason: String },

    /// Pricing data unavailable
    #[error("Pricing data unavailable for {provider} in {region}")]
    PricingUnavailable { provider: String, region: String },

    /// Workload scheduling failed
    #[error("Workload scheduling failed: {workload_id} - {reason}")]
    SchedulingFailed { workload_id: String, reason: String },

    /// Prediction model error
    #[error("Cost prediction failed: {model} - {reason}")]
    PredictionError { model: String, reason: String },

    /// Resource limit exceeded
    #[error("Resource limit exceeded: {resource} - used {used}, limit {limit}")]
    ResourceLimitExceeded {
        resource: String,
        used: f64,
        limit: f64,
    },

    /// Invalid configuration
    #[error("Invalid configuration: {message}")]
    ConfigurationError { message: String },

    /// Metrics collection failed
    #[error("Metrics collection failed: {message}")]
    MetricsError { message: String },

    /// GPU not available
    #[error("GPU {gpu_id} not available or in use")]
    GpuUnavailable { gpu_id: String },

    /// Cost calculation error
    #[error("Cost calculation error: {details}")]
    CalculationError { details: String },

    /// I/O error
    #[error("I/O error: {source}")]
    IoError {
        #[from]
        source: std::io::Error,
    },

    /// JSON serialization/deserialization error
    #[error("JSON error: {source}")]
    JsonError {
        #[from]
        source: serde_json::Error,
    },
}

/// Cost optimization result type
pub type CostOptimizationResult<T> = Result<T, CostOptimizationError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allocation_failed_error() {
        let error = CostOptimizationError::AllocationFailed {
            reason: "Insufficient GPU memory".to_string(),
        };
        assert_eq!(
            error.to_string(),
            "Resource allocation failed: Insufficient GPU memory"
        );
    }

    #[test]
    fn test_budget_exceeded_error() {
        let error = CostOptimizationError::BudgetExceeded {
            spent: 1500.75,
            budget: 1000.00,
        };
        assert_eq!(
            error.to_string(),
            "Budget exceeded: spent $1500.75, budget $1000.00"
        );
    }

    #[test]
    fn test_budget_exceeded_edge_cases() {
        // Test zero budget
        let error = CostOptimizationError::BudgetExceeded {
            spent: 100.0,
            budget: 0.0,
        };
        assert!(error.to_string().contains("budget $0.00"));

        // Test very large numbers
        let error = CostOptimizationError::BudgetExceeded {
            spent: 999999999.99,
            budget: 1000000000.00,
        };
        assert!(error.to_string().contains("$999999999.99"));
        assert!(error.to_string().contains("$1000000000.00"));
    }

    #[test]
    fn test_no_available_resources_error() {
        let error = CostOptimizationError::NoAvailableResources {
            resource_type: "GPU RTX 4090".to_string(),
        };
        assert_eq!(error.to_string(), "No available resources: GPU RTX 4090");
    }

    #[test]
    fn test_spot_instance_terminated_error() {
        let error = CostOptimizationError::SpotInstanceTerminated {
            instance_id: "i-1234567890abcdef0".to_string(),
            reason: "Price exceeded bid".to_string(),
        };
        assert_eq!(
            error.to_string(),
            "Spot instance terminated: i-1234567890abcdef0 - Price exceeded bid"
        );
    }

    #[test]
    fn test_pricing_unavailable_error() {
        let error = CostOptimizationError::PricingUnavailable {
            provider: "AWS".to_string(),
            region: "us-east-1".to_string(),
        };
        assert_eq!(
            error.to_string(),
            "Pricing data unavailable for AWS in us-east-1"
        );
    }

    #[test]
    fn test_scheduling_failed_error() {
        let error = CostOptimizationError::SchedulingFailed {
            workload_id: "wl-12345".to_string(),
            reason: "No nodes meet requirements".to_string(),
        };
        assert_eq!(
            error.to_string(),
            "Workload scheduling failed: wl-12345 - No nodes meet requirements"
        );
    }

    #[test]
    fn test_prediction_error() {
        let error = CostOptimizationError::PredictionError {
            model: "ARIMA".to_string(),
            reason: "Insufficient historical data".to_string(),
        };
        assert_eq!(
            error.to_string(),
            "Cost prediction failed: ARIMA - Insufficient historical data"
        );
    }

    #[test]
    fn test_resource_limit_exceeded_error() {
        let error = CostOptimizationError::ResourceLimitExceeded {
            resource: "vCPU".to_string(),
            used: 64.0,
            limit: 32.0,
        };
        assert_eq!(
            error.to_string(),
            "Resource limit exceeded: vCPU - used 64, limit 32"
        );
    }

    #[test]
    fn test_configuration_error() {
        let error = CostOptimizationError::ConfigurationError {
            message: "Invalid budget period: -1 days".to_string(),
        };
        assert_eq!(
            error.to_string(),
            "Invalid configuration: Invalid budget period: -1 days"
        );
    }

    #[test]
    fn test_metrics_error() {
        let error = CostOptimizationError::MetricsError {
            source: "Prometheus API timeout".to_string(),
        };
        assert_eq!(
            error.to_string(),
            "Metrics collection failed: Prometheus API timeout"
        );
    }

    #[test]
    fn test_gpu_unavailable_error() {
        let error = CostOptimizationError::GpuUnavailable {
            gpu_id: "GPU-0".to_string(),
        };
        assert_eq!(error.to_string(), "GPU GPU-0 not available or in use");
    }

    #[test]
    fn test_calculation_error() {
        let error = CostOptimizationError::CalculationError {
            details: "Division by zero in cost averaging".to_string(),
        };
        assert_eq!(
            error.to_string(),
            "Cost calculation error: Division by zero in cost averaging"
        );
    }

    #[test]
    fn test_io_error_conversion() {
        let io_error = std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            "Cannot access pricing file",
        );
        let error = CostOptimizationError::from(io_error);
        assert!(error.to_string().contains("I/O error"));
        assert!(error.to_string().contains("Cannot access pricing file"));
    }

    #[test]
    fn test_json_error_conversion() {
        let json_str = "invalid json";
        let json_error: serde_json::Error =
            serde_json::from_str::<serde_json::Value>(json_str).unwrap_err();
        let error = CostOptimizationError::from(json_error);
        assert!(error.to_string().contains("JSON error"));
    }

    #[test]
    fn test_error_debug_formatting() {
        let error = CostOptimizationError::AllocationFailed {
            reason: "Test".to_string(),
        };
        let debug_str = format!("{:?}", error);
        assert!(debug_str.contains("AllocationFailed"));
        assert!(debug_str.contains("Test"));
    }

    #[test]
    fn test_error_empty_strings() {
        let errors = vec![
            CostOptimizationError::AllocationFailed {
                reason: String::new(),
            },
            CostOptimizationError::NoAvailableResources {
                resource_type: String::new(),
            },
            CostOptimizationError::SpotInstanceTerminated {
                instance_id: String::new(),
                reason: String::new(),
            },
            CostOptimizationError::PricingUnavailable {
                provider: String::new(),
                region: String::new(),
            },
            CostOptimizationError::SchedulingFailed {
                workload_id: String::new(),
                reason: String::new(),
            },
            CostOptimizationError::PredictionError {
                model: String::new(),
                reason: String::new(),
            },
            CostOptimizationError::ResourceLimitExceeded {
                resource: String::new(),
                used: 0.0,
                limit: 0.0,
            },
            CostOptimizationError::ConfigurationError {
                message: String::new(),
            },
            CostOptimizationError::MetricsError {
                source: String::new(),
            },
            CostOptimizationError::GpuUnavailable {
                gpu_id: String::new(),
            },
            CostOptimizationError::CalculationError {
                details: String::new(),
            },
        ];

        // All should handle empty strings gracefully
        for error in errors {
            let _ = error.to_string();
        }
    }

    #[test]
    fn test_error_unicode_strings() {
        let error = CostOptimizationError::AllocationFailed {
            reason: "内存不足 🚫".to_string(),
        };
        assert!(error.to_string().contains("内存不足 🚫"));

        let error = CostOptimizationError::ConfigurationError {
            message: "構成エラー: 無効な値".to_string(),
        };
        assert!(error.to_string().contains("構成エラー: 無効な値"));
    }

    #[test]
    fn test_error_very_long_strings() {
        let long_string = "x".repeat(1000);
        let error = CostOptimizationError::AllocationFailed {
            reason: long_string.clone(),
        };
        assert!(error.to_string().contains(&long_string));
    }

    #[test]
    fn test_error_special_characters() {
        let special = "Error: $100.00 @ 50% <discount> \"quoted\" 'text'";
        let error = CostOptimizationError::CalculationError {
            details: special.to_string(),
        };
        assert!(error.to_string().contains(special));
    }

    #[test]
    fn test_result_type() {
        let success: CostOptimizationResult<i32> = Ok(42);
        assert!(success.is_ok());
        assert_eq!(success.unwrap(), 42);

        let failure: CostOptimizationResult<i32> = Err(CostOptimizationError::AllocationFailed {
            reason: "Test".to_string(),
        });
        assert!(failure.is_err());
    }

    #[test]
    fn test_error_send_sync() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}

        assert_send::<CostOptimizationError>();
        assert_sync::<CostOptimizationError>();
    }

    #[test]
    fn test_resource_limit_float_formatting() {
        let error = CostOptimizationError::ResourceLimitExceeded {
            resource: "Memory GB".to_string(),
            used: 15.5678,
            limit: 16.0,
        };
        // Should display with default float formatting
        assert!(error.to_string().contains("15.5678"));
        assert!(error.to_string().contains("16"));
    }

    #[test]
    fn test_error_chaining() {
        fn may_fail() -> CostOptimizationResult<()> {
            let io_result: Result<(), std::io::Error> = Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "File not found",
            ));
            io_result?;
            Ok(())
        }

        let result = may_fail();
        assert!(result.is_err());
        match result.unwrap_err() {
            CostOptimizationError::IoError { .. } => {}
            _ => panic!("Expected IoError variant"),
        }
    }

    #[test]
    fn test_negative_budget_values() {
        let error = CostOptimizationError::BudgetExceeded {
            spent: -100.50,
            budget: -50.00,
        };
        assert!(error.to_string().contains("$-100.50"));
        assert!(error.to_string().contains("$-50.00"));
    }

    #[test]
    fn test_extreme_float_values() {
        let error = CostOptimizationError::ResourceLimitExceeded {
            resource: "Test".to_string(),
            used: f64::MAX,
            limit: f64::MIN,
        };
        let error_str = error.to_string();
        assert!(error_str.contains(&f64::MAX.to_string()));
        assert!(error_str.contains(&f64::MIN.to_string()));

        let error = CostOptimizationError::BudgetExceeded {
            spent: f64::INFINITY,
            budget: f64::NEG_INFINITY,
        };
        let error_str = error.to_string();
        assert!(error_str.contains("inf"));
    }
}
