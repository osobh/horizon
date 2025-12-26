use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::base::AutonomyLevel;
use crate::error::{AgentError, Result};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    pub name: String,
    pub autonomy_level: AutonomyLevel,
    pub max_retries: u32,
    pub timeout_seconds: u64,
    pub safety_thresholds: SafetyThresholds,
    pub metadata: HashMap<String, String>,
}

impl AgentConfig {
    pub fn new(name: String) -> Self {
        Self {
            name,
            autonomy_level: AutonomyLevel::Low,
            max_retries: 3,
            timeout_seconds: 300,
            safety_thresholds: SafetyThresholds::default(),
            metadata: HashMap::new(),
        }
    }

    pub fn with_autonomy_level(mut self, level: AutonomyLevel) -> Self {
        self.autonomy_level = level;
        self
    }

    pub fn with_max_retries(mut self, retries: u32) -> Self {
        self.max_retries = retries;
        self
    }

    pub fn with_timeout(mut self, seconds: u64) -> Self {
        self.timeout_seconds = seconds;
        self
    }

    pub fn with_safety_thresholds(mut self, thresholds: SafetyThresholds) -> Self {
        self.safety_thresholds = thresholds;
        self
    }

    pub fn validate(&self) -> Result<()> {
        if self.name.is_empty() {
            return Err(AgentError::InvalidConfiguration(
                "Agent name cannot be empty".to_string(),
            ));
        }

        if self.max_retries == 0 {
            return Err(AgentError::InvalidConfiguration(
                "Max retries must be greater than 0".to_string(),
            ));
        }

        if self.timeout_seconds == 0 {
            return Err(AgentError::InvalidConfiguration(
                "Timeout must be greater than 0".to_string(),
            ));
        }

        self.safety_thresholds.validate()?;

        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyThresholds {
    pub max_cost_per_action: f64,
    pub max_daily_cost: f64,
    pub max_concurrent_actions: u32,
    pub require_approval_above_cost: f64,
}

impl Default for SafetyThresholds {
    fn default() -> Self {
        Self {
            max_cost_per_action: 100.0,
            max_daily_cost: 1000.0,
            max_concurrent_actions: 10,
            require_approval_above_cost: 50.0,
        }
    }
}

impl SafetyThresholds {
    pub fn validate(&self) -> Result<()> {
        if self.max_cost_per_action <= 0.0 {
            return Err(AgentError::InvalidConfiguration(
                "Max cost per action must be positive".to_string(),
            ));
        }

        if self.max_daily_cost <= 0.0 {
            return Err(AgentError::InvalidConfiguration(
                "Max daily cost must be positive".to_string(),
            ));
        }

        if self.max_concurrent_actions == 0 {
            return Err(AgentError::InvalidConfiguration(
                "Max concurrent actions must be greater than 0".to_string(),
            ));
        }

        if self.require_approval_above_cost < 0.0 {
            return Err(AgentError::InvalidConfiguration(
                "Approval threshold must be non-negative".to_string(),
            ));
        }

        Ok(())
    }

    pub fn exceeds_action_cost(&self, cost: f64) -> bool {
        cost > self.max_cost_per_action
    }

    pub fn exceeds_daily_cost(&self, cost: f64) -> bool {
        cost > self.max_daily_cost
    }

    pub fn requires_approval(&self, cost: f64) -> bool {
        cost > self.require_approval_above_cost
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_config_creation() {
        let config = AgentConfig::new("test-agent".to_string());
        assert_eq!(config.name, "test-agent");
        assert_eq!(config.autonomy_level, AutonomyLevel::Low);
        assert_eq!(config.max_retries, 3);
        assert_eq!(config.timeout_seconds, 300);
    }

    #[test]
    fn test_agent_config_builder() {
        let config = AgentConfig::new("test-agent".to_string())
            .with_autonomy_level(AutonomyLevel::High)
            .with_max_retries(5)
            .with_timeout(600);

        assert_eq!(config.autonomy_level, AutonomyLevel::High);
        assert_eq!(config.max_retries, 5);
        assert_eq!(config.timeout_seconds, 600);
    }

    #[test]
    fn test_agent_config_validation_empty_name() {
        let config = AgentConfig::new("".to_string());
        let result = config.validate();
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), AgentError::InvalidConfiguration(_)));
    }

    #[test]
    fn test_agent_config_validation_zero_retries() {
        let config = AgentConfig::new("test".to_string()).with_max_retries(0);
        let result = config.validate();
        assert!(result.is_err());
    }

    #[test]
    fn test_agent_config_validation_zero_timeout() {
        let config = AgentConfig::new("test".to_string()).with_timeout(0);
        let result = config.validate();
        assert!(result.is_err());
    }

    #[test]
    fn test_agent_config_validation_valid() {
        let config = AgentConfig::new("test".to_string());
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_safety_thresholds_default() {
        let thresholds = SafetyThresholds::default();
        assert_eq!(thresholds.max_cost_per_action, 100.0);
        assert_eq!(thresholds.max_daily_cost, 1000.0);
        assert_eq!(thresholds.max_concurrent_actions, 10);
        assert_eq!(thresholds.require_approval_above_cost, 50.0);
    }

    #[test]
    fn test_safety_thresholds_validation() {
        let mut thresholds = SafetyThresholds::default();
        assert!(thresholds.validate().is_ok());

        thresholds.max_cost_per_action = -1.0;
        assert!(thresholds.validate().is_err());

        thresholds = SafetyThresholds::default();
        thresholds.max_daily_cost = 0.0;
        assert!(thresholds.validate().is_err());

        thresholds = SafetyThresholds::default();
        thresholds.max_concurrent_actions = 0;
        assert!(thresholds.validate().is_err());
    }

    #[test]
    fn test_safety_thresholds_exceeds_action_cost() {
        let thresholds = SafetyThresholds::default();
        assert!(!thresholds.exceeds_action_cost(50.0));
        assert!(!thresholds.exceeds_action_cost(100.0));
        assert!(thresholds.exceeds_action_cost(101.0));
    }

    #[test]
    fn test_safety_thresholds_exceeds_daily_cost() {
        let thresholds = SafetyThresholds::default();
        assert!(!thresholds.exceeds_daily_cost(500.0));
        assert!(!thresholds.exceeds_daily_cost(1000.0));
        assert!(thresholds.exceeds_daily_cost(1001.0));
    }

    #[test]
    fn test_safety_thresholds_requires_approval() {
        let thresholds = SafetyThresholds::default();
        assert!(!thresholds.requires_approval(30.0));
        assert!(!thresholds.requires_approval(50.0));
        assert!(thresholds.requires_approval(51.0));
    }
}
