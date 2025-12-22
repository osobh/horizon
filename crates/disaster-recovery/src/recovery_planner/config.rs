//! Recovery planner configuration types and defaults

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Recovery planner configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryPlannerConfig {
    /// Enable automatic plan optimization
    pub auto_optimization: bool,
    /// Maximum concurrent recoveries
    pub max_concurrent_recoveries: usize,
    /// Resource allocation timeout
    pub resource_allocation_timeout: Duration,
    /// Plan validation timeout
    pub plan_validation_timeout: Duration,
    /// Enable dependency cycle detection
    pub cycle_detection_enabled: bool,
    /// Retry configuration
    pub retry_config: RetryConfig,
    /// Enable cross-region coordination
    pub cross_region_enabled: bool,
    /// Recovery plan cache size
    pub plan_cache_size: usize,
}

/// Retry configuration for recovery steps
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    /// Maximum retry attempts
    pub max_attempts: u32,
    /// Base delay between retries
    pub base_delay: Duration,
    /// Maximum delay between retries
    pub max_delay: Duration,
    /// Exponential backoff multiplier
    pub backoff_multiplier: f64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            base_delay: Duration::from_secs(30),
            max_delay: Duration::from_secs(300),
            backoff_multiplier: 2.0,
        }
    }
}

impl Default for RecoveryPlannerConfig {
    fn default() -> Self {
        Self {
            auto_optimization: true,
            max_concurrent_recoveries: 10,
            resource_allocation_timeout: Duration::from_secs(300),
            plan_validation_timeout: Duration::from_secs(60),
            cycle_detection_enabled: true,
            retry_config: RetryConfig::default(),
            cross_region_enabled: false,
            plan_cache_size: 1000,
        }
    }
}
