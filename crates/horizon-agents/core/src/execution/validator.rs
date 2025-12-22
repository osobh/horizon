use crate::agent::AutonomyLevel;
use crate::error::{AgentError, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationRule {
    pub name: String,
    pub description: String,
    pub required_autonomy: AutonomyLevel,
}

impl ValidationRule {
    pub fn new(name: String, description: String, required_autonomy: AutonomyLevel) -> Self {
        Self {
            name,
            description,
            required_autonomy,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationContext {
    pub autonomy_level: AutonomyLevel,
    pub estimated_cost: f64,
    pub is_destructive: bool,
}

impl ValidationContext {
    pub fn new(autonomy_level: AutonomyLevel) -> Self {
        Self {
            autonomy_level,
            estimated_cost: 0.0,
            is_destructive: false,
        }
    }

    pub fn with_cost(mut self, cost: f64) -> Self {
        self.estimated_cost = cost;
        self
    }

    pub fn with_destructive(mut self, destructive: bool) -> Self {
        self.is_destructive = destructive;
        self
    }
}

pub struct Validator {
    rules: Vec<ValidationRule>,
}

impl Validator {
    pub fn new() -> Self {
        Self { rules: Vec::new() }
    }

    pub fn add_rule(&mut self, rule: ValidationRule) {
        self.rules.push(rule);
    }

    pub fn validate(&self, context: &ValidationContext) -> Result<()> {
        // Check if autonomy level allows execution
        if !context.autonomy_level.can_execute() {
            return Err(AgentError::OperationNotAllowed {
                level: format!("{:?}", context.autonomy_level),
            });
        }

        // Check if destructive operations are allowed
        if context.is_destructive && context.autonomy_level == AutonomyLevel::Medium {
            return Err(AgentError::ValidationFailed(
                "Destructive operations require High or Full autonomy".to_string(),
            ));
        }

        // Validate against rules
        for rule in &self.rules {
            self.validate_rule(rule, context)?;
        }

        Ok(())
    }

    fn validate_rule(&self, rule: &ValidationRule, context: &ValidationContext) -> Result<()> {
        // Check if current autonomy level meets rule requirement
        if !self.autonomy_sufficient(context.autonomy_level, rule.required_autonomy) {
            return Err(AgentError::ValidationFailed(format!(
                "Rule '{}' requires {:?} autonomy, current level is {:?}",
                rule.name, rule.required_autonomy, context.autonomy_level
            )));
        }

        Ok(())
    }

    fn autonomy_sufficient(&self, current: AutonomyLevel, required: AutonomyLevel) -> bool {
        use AutonomyLevel::*;
        match required {
            ReadOnly => true,
            Low => matches!(current, Low | Medium | High | Full),
            Medium => matches!(current, Medium | High | Full),
            High => matches!(current, High | Full),
            Full => matches!(current, Full),
        }
    }
}

impl Default for Validator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validation_rule_creation() {
        let rule = ValidationRule::new(
            "test-rule".to_string(),
            "A test rule".to_string(),
            AutonomyLevel::Medium,
        );
        assert_eq!(rule.name, "test-rule");
        assert_eq!(rule.required_autonomy, AutonomyLevel::Medium);
    }

    #[test]
    fn test_validation_context_creation() {
        let ctx = ValidationContext::new(AutonomyLevel::High);
        assert_eq!(ctx.autonomy_level, AutonomyLevel::High);
        assert_eq!(ctx.estimated_cost, 0.0);
        assert!(!ctx.is_destructive);
    }

    #[test]
    fn test_validation_context_builder() {
        let ctx = ValidationContext::new(AutonomyLevel::High)
            .with_cost(100.0)
            .with_destructive(true);

        assert_eq!(ctx.estimated_cost, 100.0);
        assert!(ctx.is_destructive);
    }

    #[test]
    fn test_validator_creation() {
        let validator = Validator::new();
        assert_eq!(validator.rules.len(), 0);
    }

    #[test]
    fn test_validator_add_rule() {
        let mut validator = Validator::new();
        let rule = ValidationRule::new(
            "test-rule".to_string(),
            "A test rule".to_string(),
            AutonomyLevel::Medium,
        );

        validator.add_rule(rule);
        assert_eq!(validator.rules.len(), 1);
    }

    #[test]
    fn test_validator_readonly_cannot_execute() {
        let validator = Validator::new();
        let ctx = ValidationContext::new(AutonomyLevel::ReadOnly);

        let result = validator.validate(&ctx);
        assert!(result.is_err());
    }

    #[test]
    fn test_validator_low_cannot_execute() {
        let validator = Validator::new();
        let ctx = ValidationContext::new(AutonomyLevel::Low);

        let result = validator.validate(&ctx);
        assert!(result.is_err());
    }

    #[test]
    fn test_validator_medium_can_execute() {
        let validator = Validator::new();
        let ctx = ValidationContext::new(AutonomyLevel::Medium);

        let result = validator.validate(&ctx);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validator_destructive_requires_high() {
        let validator = Validator::new();
        let ctx = ValidationContext::new(AutonomyLevel::Medium).with_destructive(true);

        let result = validator.validate(&ctx);
        assert!(result.is_err());

        let ctx_high = ValidationContext::new(AutonomyLevel::High).with_destructive(true);
        let result_high = validator.validate(&ctx_high);
        assert!(result_high.is_ok());
    }

    #[test]
    fn test_validator_rule_enforcement() {
        let mut validator = Validator::new();
        let rule = ValidationRule::new(
            "high-risk-rule".to_string(),
            "Requires high autonomy".to_string(),
            AutonomyLevel::High,
        );
        validator.add_rule(rule);

        let ctx_medium = ValidationContext::new(AutonomyLevel::Medium);
        let result_medium = validator.validate(&ctx_medium);
        assert!(result_medium.is_err());

        let ctx_high = ValidationContext::new(AutonomyLevel::High);
        let result_high = validator.validate(&ctx_high);
        assert!(result_high.is_ok());
    }

    #[test]
    fn test_validator_multiple_rules() {
        let mut validator = Validator::new();

        let rule1 = ValidationRule::new(
            "rule1".to_string(),
            "Requires medium".to_string(),
            AutonomyLevel::Medium,
        );
        let rule2 = ValidationRule::new(
            "rule2".to_string(),
            "Requires high".to_string(),
            AutonomyLevel::High,
        );

        validator.add_rule(rule1);
        validator.add_rule(rule2);

        let ctx_medium = ValidationContext::new(AutonomyLevel::Medium);
        let result = validator.validate(&ctx_medium);
        assert!(result.is_err());

        let ctx_high = ValidationContext::new(AutonomyLevel::High);
        let result = validator.validate(&ctx_high);
        assert!(result.is_ok());
    }

    #[test]
    fn test_autonomy_sufficient() {
        let validator = Validator::new();

        // ReadOnly is always sufficient
        assert!(validator.autonomy_sufficient(AutonomyLevel::ReadOnly, AutonomyLevel::ReadOnly));

        // Medium can satisfy Low and Medium requirements
        assert!(validator.autonomy_sufficient(AutonomyLevel::Medium, AutonomyLevel::Low));
        assert!(validator.autonomy_sufficient(AutonomyLevel::Medium, AutonomyLevel::Medium));
        assert!(!validator.autonomy_sufficient(AutonomyLevel::Medium, AutonomyLevel::High));

        // Full can satisfy all requirements
        assert!(validator.autonomy_sufficient(AutonomyLevel::Full, AutonomyLevel::ReadOnly));
        assert!(validator.autonomy_sufficient(AutonomyLevel::Full, AutonomyLevel::Low));
        assert!(validator.autonomy_sufficient(AutonomyLevel::Full, AutonomyLevel::Medium));
        assert!(validator.autonomy_sufficient(AutonomyLevel::Full, AutonomyLevel::High));
        assert!(validator.autonomy_sufficient(AutonomyLevel::Full, AutonomyLevel::Full));
    }
}
