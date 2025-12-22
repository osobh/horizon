use serde::{Deserialize, Serialize};
use std::time::Duration;

use crate::error::{AgentError, Result};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    pub max_attempts: u32,
    pub initial_delay_ms: u64,
    pub max_delay_ms: u64,
    pub multiplier: f64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_delay_ms: 100,
            max_delay_ms: 5000,
            multiplier: 2.0,
        }
    }
}

impl RetryConfig {
    pub fn new(max_attempts: u32) -> Self {
        Self {
            max_attempts,
            ..Default::default()
        }
    }

    pub fn with_initial_delay(mut self, delay_ms: u64) -> Self {
        self.initial_delay_ms = delay_ms;
        self
    }

    pub fn with_max_delay(mut self, delay_ms: u64) -> Self {
        self.max_delay_ms = delay_ms;
        self
    }

    pub fn with_multiplier(mut self, multiplier: f64) -> Self {
        self.multiplier = multiplier;
        self
    }

    pub fn calculate_delay(&self, attempt: u32) -> Duration {
        let delay = self.initial_delay_ms as f64 * self.multiplier.powi(attempt as i32);
        let delay = delay.min(self.max_delay_ms as f64);
        Duration::from_millis(delay as u64)
    }

    pub fn validate(&self) -> Result<()> {
        if self.max_attempts == 0 {
            return Err(AgentError::InvalidConfiguration(
                "Max attempts must be greater than 0".to_string(),
            ));
        }

        if self.multiplier <= 0.0 {
            return Err(AgentError::InvalidConfiguration(
                "Multiplier must be positive".to_string(),
            ));
        }

        if self.max_delay_ms < self.initial_delay_ms {
            return Err(AgentError::InvalidConfiguration(
                "Max delay must be greater than or equal to initial delay".to_string(),
            ));
        }

        Ok(())
    }
}

pub struct RetryStrategy {
    config: RetryConfig,
    current_attempt: u32,
}

impl RetryStrategy {
    pub fn new(config: RetryConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            config,
            current_attempt: 0,
        })
    }

    pub fn should_retry(&self) -> bool {
        self.current_attempt < self.config.max_attempts
    }

    pub fn next_delay(&mut self) -> Result<Duration> {
        if !self.should_retry() {
            return Err(AgentError::RetryLimitExceeded(format!(
                "Max attempts ({}) exceeded",
                self.config.max_attempts
            )));
        }

        let delay = self.config.calculate_delay(self.current_attempt);
        self.current_attempt += 1;
        Ok(delay)
    }

    pub fn reset(&mut self) {
        self.current_attempt = 0;
    }

    pub fn current_attempt(&self) -> u32 {
        self.current_attempt
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_retry_config_default() {
        let config = RetryConfig::default();
        assert_eq!(config.max_attempts, 3);
        assert_eq!(config.initial_delay_ms, 100);
        assert_eq!(config.max_delay_ms, 5000);
        assert_eq!(config.multiplier, 2.0);
    }

    #[test]
    fn test_retry_config_new() {
        let config = RetryConfig::new(5);
        assert_eq!(config.max_attempts, 5);
    }

    #[test]
    fn test_retry_config_builder() {
        let config = RetryConfig::new(3)
            .with_initial_delay(200)
            .with_max_delay(10000)
            .with_multiplier(1.5);

        assert_eq!(config.initial_delay_ms, 200);
        assert_eq!(config.max_delay_ms, 10000);
        assert_eq!(config.multiplier, 1.5);
    }

    #[test]
    fn test_retry_config_calculate_delay() {
        let config = RetryConfig::new(5)
            .with_initial_delay(100)
            .with_multiplier(2.0);

        assert_eq!(config.calculate_delay(0).as_millis(), 100);
        assert_eq!(config.calculate_delay(1).as_millis(), 200);
        assert_eq!(config.calculate_delay(2).as_millis(), 400);
        assert_eq!(config.calculate_delay(3).as_millis(), 800);
    }

    #[test]
    fn test_retry_config_calculate_delay_with_max() {
        let config = RetryConfig::new(5)
            .with_initial_delay(100)
            .with_max_delay(500)
            .with_multiplier(2.0);

        assert_eq!(config.calculate_delay(0).as_millis(), 100);
        assert_eq!(config.calculate_delay(1).as_millis(), 200);
        assert_eq!(config.calculate_delay(2).as_millis(), 400);
        assert_eq!(config.calculate_delay(3).as_millis(), 500); // Capped
        assert_eq!(config.calculate_delay(4).as_millis(), 500); // Capped
    }

    #[test]
    fn test_retry_config_validation() {
        let config = RetryConfig::default();
        assert!(config.validate().is_ok());

        let invalid = RetryConfig::new(0);
        assert!(invalid.validate().is_err());

        let invalid = RetryConfig::new(3).with_multiplier(-1.0);
        assert!(invalid.validate().is_err());

        let invalid = RetryConfig::new(3)
            .with_initial_delay(1000)
            .with_max_delay(500);
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn test_retry_strategy_creation() {
        let config = RetryConfig::default();
        let strategy = RetryStrategy::new(config);
        assert!(strategy.is_ok());

        let invalid = RetryConfig::new(0);
        let strategy = RetryStrategy::new(invalid);
        assert!(strategy.is_err());
    }

    #[test]
    fn test_retry_strategy_should_retry() {
        let config = RetryConfig::new(3);
        let mut strategy = RetryStrategy::new(config).unwrap();

        assert!(strategy.should_retry());
        strategy.next_delay().unwrap();
        assert!(strategy.should_retry());
        strategy.next_delay().unwrap();
        assert!(strategy.should_retry());
        strategy.next_delay().unwrap();
        assert!(!strategy.should_retry());
    }

    #[test]
    fn test_retry_strategy_next_delay() {
        let config = RetryConfig::new(3)
            .with_initial_delay(100)
            .with_multiplier(2.0);
        let mut strategy = RetryStrategy::new(config).unwrap();

        let delay1 = strategy.next_delay().unwrap();
        assert_eq!(delay1.as_millis(), 100);

        let delay2 = strategy.next_delay().unwrap();
        assert_eq!(delay2.as_millis(), 200);

        let delay3 = strategy.next_delay().unwrap();
        assert_eq!(delay3.as_millis(), 400);

        let result = strategy.next_delay();
        assert!(result.is_err());
    }

    #[test]
    fn test_retry_strategy_reset() {
        let config = RetryConfig::new(3);
        let mut strategy = RetryStrategy::new(config).unwrap();

        strategy.next_delay().unwrap();
        strategy.next_delay().unwrap();
        assert_eq!(strategy.current_attempt(), 2);

        strategy.reset();
        assert_eq!(strategy.current_attempt(), 0);
        assert!(strategy.should_retry());
    }

    #[test]
    fn test_retry_strategy_current_attempt() {
        let config = RetryConfig::new(3);
        let mut strategy = RetryStrategy::new(config).unwrap();

        assert_eq!(strategy.current_attempt(), 0);
        strategy.next_delay().unwrap();
        assert_eq!(strategy.current_attempt(), 1);
        strategy.next_delay().unwrap();
        assert_eq!(strategy.current_attempt(), 2);
    }

    #[test]
    fn test_retry_strategy_exponential_backoff() {
        let config = RetryConfig::new(4)
            .with_initial_delay(50)
            .with_multiplier(3.0);
        let mut strategy = RetryStrategy::new(config).unwrap();

        assert_eq!(strategy.next_delay().unwrap().as_millis(), 50);
        assert_eq!(strategy.next_delay().unwrap().as_millis(), 150);
        assert_eq!(strategy.next_delay().unwrap().as_millis(), 450);
        assert_eq!(strategy.next_delay().unwrap().as_millis(), 1350);
    }
}
