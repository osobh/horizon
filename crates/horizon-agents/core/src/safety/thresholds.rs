use crate::agent::SafetyThresholds;
use crate::error::{AgentError, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionCost {
    pub action_id: String,
    pub estimated_cost: f64,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl ActionCost {
    pub fn new(action_id: String, estimated_cost: f64) -> Self {
        Self {
            action_id,
            estimated_cost,
            timestamp: chrono::Utc::now(),
        }
    }
}

pub struct ThresholdManager {
    thresholds: SafetyThresholds,
    daily_cost: f64,
    daily_reset: chrono::DateTime<chrono::Utc>,
    concurrent_actions: u32,
}

impl ThresholdManager {
    pub fn new(thresholds: SafetyThresholds) -> Result<Self> {
        thresholds.validate()?;
        Ok(Self {
            thresholds,
            daily_cost: 0.0,
            daily_reset: chrono::Utc::now(),
            concurrent_actions: 0,
        })
    }

    pub fn check_action_cost(&self, cost: f64) -> Result<()> {
        if self.thresholds.exceeds_action_cost(cost) {
            return Err(AgentError::SafetyThresholdExceeded(format!(
                "Action cost ${:.2} exceeds maximum ${:.2}",
                cost, self.thresholds.max_cost_per_action
            )));
        }
        Ok(())
    }

    pub fn check_daily_cost(&mut self, additional_cost: f64) -> Result<()> {
        self.reset_if_needed();

        let new_total = self.daily_cost + additional_cost;
        if self.thresholds.exceeds_daily_cost(new_total) {
            return Err(AgentError::SafetyThresholdExceeded(format!(
                "Daily cost would exceed limit: ${:.2} + ${:.2} > ${:.2}",
                self.daily_cost, additional_cost, self.thresholds.max_daily_cost
            )));
        }
        Ok(())
    }

    pub fn check_concurrent_actions(&self) -> Result<()> {
        if self.concurrent_actions >= self.thresholds.max_concurrent_actions {
            return Err(AgentError::SafetyThresholdExceeded(format!(
                "Max concurrent actions ({}) reached",
                self.thresholds.max_concurrent_actions
            )));
        }
        Ok(())
    }

    pub fn record_action_start(&mut self, cost: f64) -> Result<()> {
        self.check_action_cost(cost)?;
        self.check_daily_cost(cost)?;
        self.check_concurrent_actions()?;

        self.concurrent_actions += 1;
        self.daily_cost += cost;
        Ok(())
    }

    pub fn record_action_end(&mut self) {
        if self.concurrent_actions > 0 {
            self.concurrent_actions -= 1;
        }
    }

    pub fn requires_approval(&self, cost: f64) -> bool {
        self.thresholds.requires_approval(cost)
    }

    pub fn get_daily_cost(&mut self) -> f64 {
        self.reset_if_needed();
        self.daily_cost
    }

    pub fn get_concurrent_actions(&self) -> u32 {
        self.concurrent_actions
    }

    fn reset_if_needed(&mut self) {
        let now = chrono::Utc::now();
        let elapsed = now.signed_duration_since(self.daily_reset);

        if elapsed.num_hours() >= 24 {
            self.daily_cost = 0.0;
            self.daily_reset = now;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_action_cost_creation() {
        let cost = ActionCost::new("action1".to_string(), 50.0);
        assert_eq!(cost.action_id, "action1");
        assert_eq!(cost.estimated_cost, 50.0);
    }

    #[test]
    fn test_threshold_manager_creation() {
        let thresholds = SafetyThresholds::default();
        let manager = ThresholdManager::new(thresholds);
        assert!(manager.is_ok());
    }

    #[test]
    fn test_threshold_manager_invalid_thresholds() {
        let mut thresholds = SafetyThresholds::default();
        thresholds.max_cost_per_action = -1.0;

        let manager = ThresholdManager::new(thresholds);
        assert!(manager.is_err());
    }

    #[test]
    fn test_check_action_cost_within_limit() {
        let thresholds = SafetyThresholds::default();
        let manager = ThresholdManager::new(thresholds).unwrap();

        assert!(manager.check_action_cost(50.0).is_ok());
        assert!(manager.check_action_cost(100.0).is_ok());
    }

    #[test]
    fn test_check_action_cost_exceeds_limit() {
        let thresholds = SafetyThresholds::default();
        let manager = ThresholdManager::new(thresholds).unwrap();

        let result = manager.check_action_cost(101.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_check_daily_cost_within_limit() {
        let thresholds = SafetyThresholds::default();
        let mut manager = ThresholdManager::new(thresholds).unwrap();

        assert!(manager.check_daily_cost(500.0).is_ok());
        assert!(manager.check_daily_cost(500.0).is_ok());
    }

    #[test]
    fn test_check_daily_cost_exceeds_limit() {
        let thresholds = SafetyThresholds::default();
        let mut manager = ThresholdManager::new(thresholds).unwrap();

        manager.daily_cost = 900.0;
        let result = manager.check_daily_cost(101.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_check_concurrent_actions_within_limit() {
        let thresholds = SafetyThresholds::default();
        let mut manager = ThresholdManager::new(thresholds).unwrap();

        manager.concurrent_actions = 5;
        assert!(manager.check_concurrent_actions().is_ok());
    }

    #[test]
    fn test_check_concurrent_actions_exceeds_limit() {
        let thresholds = SafetyThresholds::default();
        let mut manager = ThresholdManager::new(thresholds).unwrap();

        manager.concurrent_actions = 10;
        let result = manager.check_concurrent_actions();
        assert!(result.is_err());
    }

    #[test]
    fn test_record_action_start_success() {
        let thresholds = SafetyThresholds::default();
        let mut manager = ThresholdManager::new(thresholds).unwrap();

        assert!(manager.record_action_start(50.0).is_ok());
        assert_eq!(manager.get_concurrent_actions(), 1);
        assert_eq!(manager.get_daily_cost(), 50.0);
    }

    #[test]
    fn test_record_action_start_exceeds_cost() {
        let thresholds = SafetyThresholds::default();
        let mut manager = ThresholdManager::new(thresholds).unwrap();

        let result = manager.record_action_start(150.0);
        assert!(result.is_err());
        assert_eq!(manager.get_concurrent_actions(), 0);
    }

    #[test]
    fn test_record_action_end() {
        let thresholds = SafetyThresholds::default();
        let mut manager = ThresholdManager::new(thresholds).unwrap();

        manager.record_action_start(50.0).unwrap();
        assert_eq!(manager.get_concurrent_actions(), 1);

        manager.record_action_end();
        assert_eq!(manager.get_concurrent_actions(), 0);
    }

    #[test]
    fn test_multiple_concurrent_actions() {
        let thresholds = SafetyThresholds::default();
        let mut manager = ThresholdManager::new(thresholds).unwrap();

        manager.record_action_start(20.0).unwrap();
        manager.record_action_start(30.0).unwrap();
        manager.record_action_start(40.0).unwrap();

        assert_eq!(manager.get_concurrent_actions(), 3);
        assert_eq!(manager.get_daily_cost(), 90.0);

        manager.record_action_end();
        assert_eq!(manager.get_concurrent_actions(), 2);
    }

    #[test]
    fn test_requires_approval() {
        let thresholds = SafetyThresholds::default();
        let manager = ThresholdManager::new(thresholds).unwrap();

        assert!(!manager.requires_approval(30.0));
        assert!(!manager.requires_approval(50.0));
        assert!(manager.requires_approval(51.0));
        assert!(manager.requires_approval(100.0));
    }

    #[test]
    fn test_daily_reset() {
        let thresholds = SafetyThresholds::default();
        let mut manager = ThresholdManager::new(thresholds).unwrap();

        manager.record_action_start(100.0).unwrap();
        assert_eq!(manager.daily_cost, 100.0);

        // Simulate 25 hours passing
        manager.daily_reset = chrono::Utc::now() - chrono::Duration::hours(25);

        let cost = manager.get_daily_cost();
        assert_eq!(cost, 0.0);
    }

    #[test]
    fn test_get_concurrent_actions() {
        let thresholds = SafetyThresholds::default();
        let mut manager = ThresholdManager::new(thresholds).unwrap();

        assert_eq!(manager.get_concurrent_actions(), 0);

        manager.record_action_start(20.0).unwrap();
        assert_eq!(manager.get_concurrent_actions(), 1);

        manager.record_action_start(30.0).unwrap();
        assert_eq!(manager.get_concurrent_actions(), 2);
    }
}
