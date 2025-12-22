//! Safety validation and checks for business goals

use crate::error::{BusinessError, BusinessResult};
use crate::goal::{BusinessGoal, Constraint, SafetyLevel};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::debug;

/// Safety validator for business goals
pub struct SafetyValidator {
    /// Maximum allowed resource limits
    max_limits: ResourceLimits,
    /// Forbidden keywords in goal descriptions
    forbidden_keywords: Vec<String>,
    /// Required safety checks by category
    required_checks: HashMap<String, Vec<SafetyCheck>>,
    /// Validation rules
    validation_rules: Vec<ValidationRule>,
}

/// Safety validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyValidationResult {
    /// Whether the goal passed all safety checks
    pub passed: bool,
    /// Overall safety score (0.0 to 1.0)
    pub safety_score: f64,
    /// Individual check results
    pub check_results: Vec<SafetyCheckResult>,
    /// Warnings that don't fail validation
    pub warnings: Vec<String>,
    /// Errors that cause validation failure
    pub errors: Vec<String>,
    /// Recommended mitigations
    pub mitigations: Vec<String>,
    /// Compliance requirements
    pub compliance_requirements: Vec<String>,
    /// Validation timestamp
    pub validated_at: DateTime<Utc>,
}

/// Individual safety check result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyCheckResult {
    /// Check name
    pub check_name: String,
    /// Check type
    pub check_type: SafetyCheckType,
    /// Whether the check passed
    pub passed: bool,
    /// Check score (0.0 to 1.0)
    pub score: f64,
    /// Check details/reason
    pub details: String,
    /// Recommended actions if failed
    pub recommendations: Vec<String>,
}

/// Types of safety checks
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SafetyCheckType {
    /// Resource usage validation
    ResourceUsage,
    /// Data privacy and security
    DataPrivacy,
    /// Compliance requirements
    Compliance,
    /// Content safety
    ContentSafety,
    /// Operational safety
    OperationalSafety,
    /// Financial safety
    FinancialSafety,
    /// Time constraints safety
    TimeConstraints,
    /// Agent behavior safety
    AgentBehavior,
}

/// Safety check configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyCheck {
    /// Check name
    pub name: String,
    /// Check type
    pub check_type: SafetyCheckType,
    /// Minimum required safety level
    pub min_safety_level: SafetyLevel,
    /// Whether this check is mandatory
    pub mandatory: bool,
    /// Check weight in overall score
    pub weight: f64,
}

/// Resource limits for safety validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    /// Maximum GPU memory in MB
    pub max_gpu_memory_mb: u32,
    /// Maximum CPU cores
    pub max_cpu_cores: u32,
    /// Maximum system memory in MB
    pub max_memory_mb: u32,
    /// Maximum storage in MB
    pub max_storage_mb: u32,
    /// Maximum execution time in hours
    pub max_execution_hours: f64,
    /// Maximum cost in USD
    pub max_cost_usd: f64,
    /// Maximum number of agents
    pub max_agents: u32,
}

/// Validation rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationRule {
    /// Rule name
    pub name: String,
    /// Rule description
    pub description: String,
    /// Rule severity
    pub severity: RuleSeverity,
    /// Rule pattern or condition
    pub pattern: String,
    /// Violation message
    pub violation_message: String,
}

/// Rule severity levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, PartialOrd)]
pub enum RuleSeverity {
    Info = 1,
    Warning = 2,
    Error = 3,
    Critical = 4,
}

impl SafetyValidator {
    /// Create a new safety validator with default configuration
    pub fn new() -> Self {
        let max_limits = ResourceLimits {
            max_gpu_memory_mb: 32768, // 32GB
            max_cpu_cores: 64,
            max_memory_mb: 131072,   // 128GB
            max_storage_mb: 1048576, // 1TB
            max_execution_hours: 24.0,
            max_cost_usd: 10000.0,
            max_agents: 1000,
        };

        let forbidden_keywords = vec![
            "hack".to_string(),
            "exploit".to_string(),
            "virus".to_string(),
            "malware".to_string(),
            "attack".to_string(),
            "illegal".to_string(),
            "breach".to_string(),
            "steal".to_string(),
            "poison".to_string(),
            "bomb".to_string(),
        ];

        let mut required_checks = HashMap::new();
        required_checks.insert(
            "DataAnalysis".to_string(),
            vec![
                SafetyCheck {
                    name: "Data Privacy".to_string(),
                    check_type: SafetyCheckType::DataPrivacy,
                    min_safety_level: SafetyLevel::Medium,
                    mandatory: true,
                    weight: 0.3,
                },
                SafetyCheck {
                    name: "Resource Usage".to_string(),
                    check_type: SafetyCheckType::ResourceUsage,
                    min_safety_level: SafetyLevel::Low,
                    mandatory: true,
                    weight: 0.2,
                },
            ],
        );

        let validation_rules = vec![
            ValidationRule {
                name: "No Personal Data Processing".to_string(),
                description: "Goals should not process personal data without explicit consent"
                    .to_string(),
                severity: RuleSeverity::Error,
                pattern: r"(?i)(personal data|pii|social security|credit card)".to_string(),
                violation_message:
                    "Goal appears to process personal data without proper safeguards".to_string(),
            },
            ValidationRule {
                name: "Reasonable Resource Limits".to_string(),
                description: "Resource requests should be reasonable for the task".to_string(),
                severity: RuleSeverity::Warning,
                pattern: r"(?i)(unlimited|infinite|maximum)".to_string(),
                violation_message: "Goal requests excessive or unlimited resources".to_string(),
            },
        ];

        Self {
            max_limits,
            forbidden_keywords,
            required_checks,
            validation_rules,
        }
    }

    /// Validate a business goal for safety
    pub fn validate_goal(&self, goal: &BusinessGoal) -> BusinessResult<SafetyValidationResult> {
        debug!("Validating safety for goal: {}", goal.goal_id);

        let mut check_results = Vec::new();
        let mut warnings = Vec::new();
        let mut errors = Vec::new();
        let mut mitigations = Vec::new();
        let mut compliance_requirements = Vec::new();

        // Perform individual safety checks
        check_results.extend(self.check_resource_usage(goal)?);
        check_results.extend(self.check_data_privacy(goal)?);
        check_results.extend(self.check_content_safety(goal)?);
        check_results.extend(self.check_operational_safety(goal)?);
        check_results.extend(self.check_financial_safety(goal)?);
        check_results.extend(self.check_time_constraints(goal)?);
        check_results.extend(self.check_compliance(goal)?);

        // Apply validation rules
        for rule in &self.validation_rules {
            if let Some(violation) = self.apply_validation_rule(goal, rule)? {
                match rule.severity {
                    RuleSeverity::Info => {}
                    RuleSeverity::Warning => warnings.push(violation),
                    RuleSeverity::Error | RuleSeverity::Critical => errors.push(violation),
                }
            }
        }

        // Calculate overall safety score
        let total_weight: f64 = check_results.iter().map(|r| r.score).sum();
        let weighted_score: f64 = check_results.iter().map(|r| r.score).sum();
        let safety_score = if total_weight > 0.0 {
            weighted_score / check_results.len() as f64
        } else {
            0.0
        };

        // Determine if validation passed
        let passed = errors.is_empty()
            && check_results
                .iter()
                .all(|r| r.passed || !self.is_mandatory_check(&r.check_name))
            && safety_score >= 0.7; // Minimum safety score threshold

        // Collect mitigations and compliance requirements
        for result in &check_results {
            if !result.passed {
                mitigations.extend(result.recommendations.clone());
            }
        }

        // Add standard compliance requirements based on goal content
        if goal.description.to_lowercase().contains("personal")
            || goal.description.to_lowercase().contains("customer")
        {
            compliance_requirements.push("GDPR".to_string());
            compliance_requirements.push("CCPA".to_string());
        }

        if goal.description.to_lowercase().contains("health")
            || goal.description.to_lowercase().contains("medical")
        {
            compliance_requirements.push("HIPAA".to_string());
        }

        Ok(SafetyValidationResult {
            passed,
            safety_score,
            check_results,
            warnings,
            errors,
            mitigations,
            compliance_requirements,
            validated_at: Utc::now(),
        })
    }

    /// Check resource usage safety
    fn check_resource_usage(&self, goal: &BusinessGoal) -> BusinessResult<Vec<SafetyCheckResult>> {
        let mut results = Vec::new();
        let limits = &goal.resource_limits;

        // Check GPU memory
        if let Some(gpu_memory) = limits.max_gpu_memory_mb {
            let passed = gpu_memory <= self.max_limits.max_gpu_memory_mb;
            let score = if passed { 1.0 } else { 0.0 };

            results.push(SafetyCheckResult {
                check_name: "GPU Memory Limit".to_string(),
                check_type: SafetyCheckType::ResourceUsage,
                passed,
                score,
                details: format!(
                    "Requested {} MB, limit {} MB",
                    gpu_memory, self.max_limits.max_gpu_memory_mb
                ),
                recommendations: if passed {
                    Vec::new()
                } else {
                    vec![format!(
                        "Reduce GPU memory request to under {} MB",
                        self.max_limits.max_gpu_memory_mb
                    )]
                },
            });
        }

        // Check CPU cores
        if let Some(cpu_cores) = limits.max_cpu_cores {
            let passed = cpu_cores <= self.max_limits.max_cpu_cores;
            let score = if passed { 1.0 } else { 0.0 };

            results.push(SafetyCheckResult {
                check_name: "CPU Cores Limit".to_string(),
                check_type: SafetyCheckType::ResourceUsage,
                passed,
                score,
                details: format!(
                    "Requested {} cores, limit {} cores",
                    cpu_cores, self.max_limits.max_cpu_cores
                ),
                recommendations: if passed {
                    Vec::new()
                } else {
                    vec![format!(
                        "Reduce CPU core request to under {}",
                        self.max_limits.max_cpu_cores
                    )]
                },
            });
        }

        // Check cost
        if let Some(cost) = limits.max_cost_usd {
            let passed = cost <= self.max_limits.max_cost_usd;
            let score = if passed { 1.0 } else { 0.0 };

            results.push(SafetyCheckResult {
                check_name: "Cost Limit".to_string(),
                check_type: SafetyCheckType::FinancialSafety,
                passed,
                score,
                details: format!(
                    "Requested ${:.2}, limit ${:.2}",
                    cost, self.max_limits.max_cost_usd
                ),
                recommendations: if passed {
                    Vec::new()
                } else {
                    vec![format!(
                        "Reduce cost estimate to under ${:.2}",
                        self.max_limits.max_cost_usd
                    )]
                },
            });
        }

        Ok(results)
    }

    /// Check data privacy and security
    fn check_data_privacy(&self, goal: &BusinessGoal) -> BusinessResult<Vec<SafetyCheckResult>> {
        let mut results = Vec::new();
        let description = goal.description.to_lowercase();

        // Check for sensitive data indicators
        let sensitive_keywords = [
            "personal",
            "private",
            "confidential",
            "secret",
            "ssn",
            "credit card",
        ];
        let contains_sensitive = sensitive_keywords
            .iter()
            .any(|&keyword| description.contains(keyword));

        let has_privacy_constraints = goal
            .constraints
            .iter()
            .any(|c| matches!(c, Constraint::PrivacyRequirement { .. }));

        let passed = !contains_sensitive || has_privacy_constraints;
        let score = if passed { 1.0 } else { 0.5 };

        results.push(SafetyCheckResult {
            check_name: "Data Privacy Check".to_string(),
            check_type: SafetyCheckType::DataPrivacy,
            passed,
            score,
            details: if contains_sensitive {
                "Goal mentions sensitive data".to_string()
            } else {
                "No sensitive data indicators found".to_string()
            },
            recommendations: if passed {
                Vec::new()
            } else {
                vec![
                    "Add privacy constraints to the goal".to_string(),
                    "Ensure proper data handling procedures".to_string(),
                    "Consider data anonymization".to_string(),
                ]
            },
        });

        Ok(results)
    }

    /// Check content safety
    fn check_content_safety(&self, goal: &BusinessGoal) -> BusinessResult<Vec<SafetyCheckResult>> {
        let mut results = Vec::new();
        let description = goal.description.to_lowercase();

        // Check for forbidden keywords
        let forbidden_found: Vec<&String> = self
            .forbidden_keywords
            .iter()
            .filter(|&keyword| description.contains(keyword))
            .collect();

        let passed = forbidden_found.is_empty();
        let score = if passed { 1.0 } else { 0.0 };

        results.push(SafetyCheckResult {
            check_name: "Content Safety Check".to_string(),
            check_type: SafetyCheckType::ContentSafety,
            passed,
            score,
            details: if passed {
                "No forbidden content detected".to_string()
            } else {
                format!("Forbidden keywords found: {:?}", forbidden_found)
            },
            recommendations: if passed {
                Vec::new()
            } else {
                vec![
                    "Remove or rephrase problematic content".to_string(),
                    "Ensure goal description follows content policy".to_string(),
                ]
            },
        });

        Ok(results)
    }

    /// Check operational safety
    fn check_operational_safety(
        &self,
        goal: &BusinessGoal,
    ) -> BusinessResult<Vec<SafetyCheckResult>> {
        let mut results = Vec::new();

        // Check execution time limits
        if let Some(execution_time) = &goal.resource_limits.max_execution_time {
            let hours = execution_time.as_secs_f64() / 3600.0;
            let passed = hours <= self.max_limits.max_execution_hours;
            let score = if passed { 1.0 } else { 0.0 };

            results.push(SafetyCheckResult {
                check_name: "Execution Time Safety".to_string(),
                check_type: SafetyCheckType::OperationalSafety,
                passed,
                score,
                details: format!(
                    "Requested {:.1} hours, limit {:.1} hours",
                    hours, self.max_limits.max_execution_hours
                ),
                recommendations: if passed {
                    Vec::new()
                } else {
                    vec![format!(
                        "Reduce execution time to under {:.1} hours",
                        self.max_limits.max_execution_hours
                    )]
                },
            });
        }

        // Check agent count
        if let Some(agent_count) = goal.resource_limits.max_agents {
            let passed = agent_count <= self.max_limits.max_agents;
            let score = if passed { 1.0 } else { 0.0 };

            results.push(SafetyCheckResult {
                check_name: "Agent Count Safety".to_string(),
                check_type: SafetyCheckType::OperationalSafety,
                passed,
                score,
                details: format!(
                    "Requested {} agents, limit {} agents",
                    agent_count, self.max_limits.max_agents
                ),
                recommendations: if passed {
                    Vec::new()
                } else {
                    vec![format!(
                        "Reduce agent count to under {}",
                        self.max_limits.max_agents
                    )]
                },
            });
        }

        Ok(results)
    }

    /// Check financial safety
    fn check_financial_safety(
        &self,
        goal: &BusinessGoal,
    ) -> BusinessResult<Vec<SafetyCheckResult>> {
        let mut results = Vec::new();

        // Check budget constraints
        let has_budget_constraint = goal
            .constraints
            .iter()
            .any(|c| matches!(c, Constraint::BudgetLimit { .. }));

        let has_cost_limit = goal.resource_limits.max_cost_usd.is_some();

        let passed = has_budget_constraint || has_cost_limit;
        let score = if passed { 1.0 } else { 0.7 };

        results.push(SafetyCheckResult {
            check_name: "Financial Safety Check".to_string(),
            check_type: SafetyCheckType::FinancialSafety,
            passed,
            score,
            details: if passed {
                "Budget constraints or cost limits defined".to_string()
            } else {
                "No budget constraints defined".to_string()
            },
            recommendations: if passed {
                Vec::new()
            } else {
                vec![
                    "Add budget constraints to prevent cost overruns".to_string(),
                    "Define maximum acceptable cost".to_string(),
                ]
            },
        });

        Ok(results)
    }

    /// Check time constraints
    fn check_time_constraints(
        &self,
        goal: &BusinessGoal,
    ) -> BusinessResult<Vec<SafetyCheckResult>> {
        let mut results = Vec::new();

        // Check for reasonable deadlines
        let time_constraints: Vec<&Constraint> = goal
            .constraints
            .iter()
            .filter(|c| matches!(c, Constraint::TimeLimit { .. }))
            .collect();

        for constraint in time_constraints {
            if let Constraint::TimeLimit { deadline } = constraint {
                let time_until_deadline = *deadline - Utc::now();
                let hours_until = time_until_deadline.num_hours() as f64;

                let passed = hours_until >= 1.0; // At least 1 hour
                let score = if passed { 1.0 } else { 0.0 };

                results.push(SafetyCheckResult {
                    check_name: "Time Constraint Safety".to_string(),
                    check_type: SafetyCheckType::TimeConstraints,
                    passed,
                    score,
                    details: format!("Deadline in {:.1} hours", hours_until),
                    recommendations: if passed {
                        Vec::new()
                    } else {
                        vec!["Extend deadline to allow sufficient execution time".to_string()]
                    },
                });
            }
        }

        Ok(results)
    }

    /// Check compliance requirements
    fn check_compliance(&self, goal: &BusinessGoal) -> BusinessResult<Vec<SafetyCheckResult>> {
        let mut results = Vec::new();
        let description = goal.description.to_lowercase();

        // Check for compliance-related keywords
        let compliance_indicators = [
            ("gdpr", "GDPR compliance required for EU data"),
            ("hipaa", "HIPAA compliance required for health data"),
            ("sox", "SOX compliance required for financial data"),
            ("pci", "PCI DSS compliance required for payment data"),
        ];

        for (keyword, requirement) in compliance_indicators.iter() {
            if description.contains(keyword) {
                let has_compliance_constraint = goal
                    .constraints
                    .iter()
                    .any(|c| matches!(c, Constraint::ComplianceRequirement { .. }));

                let passed = has_compliance_constraint;
                let score = if passed { 1.0 } else { 0.5 };

                results.push(SafetyCheckResult {
                    check_name: format!("{} Compliance Check", keyword.to_uppercase()),
                    check_type: SafetyCheckType::Compliance,
                    passed,
                    score,
                    details: requirement.to_string(),
                    recommendations: if passed {
                        Vec::new()
                    } else {
                        vec![format!(
                            "Add {} compliance constraints",
                            keyword.to_uppercase()
                        )]
                    },
                });
            }
        }

        Ok(results)
    }

    /// Apply a validation rule to a goal
    fn apply_validation_rule(
        &self,
        goal: &BusinessGoal,
        rule: &ValidationRule,
    ) -> BusinessResult<Option<String>> {
        let regex = regex::Regex::new(&rule.pattern).map_err(|e| {
            BusinessError::ConfigurationError(format!("Invalid regex pattern: {}", e))
        })?;

        if regex.is_match(&goal.description) {
            Ok(Some(rule.violation_message.clone()))
        } else {
            Ok(None)
        }
    }

    /// Check if a safety check is mandatory
    fn is_mandatory_check(&self, check_name: &str) -> bool {
        // For now, treat certain checks as mandatory
        matches!(check_name, "Content Safety Check" | "Data Privacy Check")
    }

    /// Set custom resource limits
    pub fn set_resource_limits(&mut self, limits: ResourceLimits) {
        self.max_limits = limits;
    }

    /// Add forbidden keyword
    pub fn add_forbidden_keyword(&mut self, keyword: String) {
        self.forbidden_keywords.push(keyword);
    }

    /// Add validation rule
    pub fn add_validation_rule(&mut self, rule: ValidationRule) {
        self.validation_rules.push(rule);
    }

    /// Get current configuration
    pub fn get_config(&self) -> SafetyValidatorConfig {
        SafetyValidatorConfig {
            max_limits: self.max_limits.clone(),
            forbidden_keywords: self.forbidden_keywords.clone(),
            validation_rules: self.validation_rules.clone(),
        }
    }
}

/// Safety validator configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyValidatorConfig {
    pub max_limits: ResourceLimits,
    pub forbidden_keywords: Vec<String>,
    pub validation_rules: Vec<ValidationRule>,
}

impl Default for SafetyValidator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::goal::{BusinessGoal, GoalCategory, GoalPriority};
    use chrono::Duration;

    fn create_test_goal() -> BusinessGoal {
        let mut goal = BusinessGoal::new(
            "Analyze customer data for insights".to_string(),
            "test@example.com".to_string(),
        );
        goal.category = GoalCategory::DataAnalysis;
        goal.priority = GoalPriority::Medium;
        goal
    }

    fn create_test_validator() -> SafetyValidator {
        SafetyValidator::new()
    }

    #[test]
    fn test_validator_creation() {
        let validator = create_test_validator();
        assert_eq!(validator.max_limits.max_gpu_memory_mb, 32768);
        assert!(!validator.forbidden_keywords.is_empty());
        assert!(!validator.validation_rules.is_empty());
    }

    #[test]
    fn test_safe_goal_validation() {
        let validator = create_test_validator();
        let goal = create_test_goal();

        let result = validator.validate_goal(&goal).unwrap();
        assert!(result.passed);
        assert!(result.safety_score > 0.7);
        assert!(result.errors.is_empty());
    }

    #[test]
    fn test_excessive_resource_validation() {
        let validator = create_test_validator();
        let mut goal = create_test_goal();

        // Set excessive GPU memory
        goal.resource_limits.max_gpu_memory_mb = Some(65536); // 64GB, exceeds limit

        let result = validator.validate_goal(&goal).unwrap();
        assert!(!result.passed);
        assert!(result
            .check_results
            .iter()
            .any(|r| r.check_name == "GPU Memory Limit" && !r.passed));
    }

    #[test]
    fn test_forbidden_content_validation() {
        let validator = create_test_validator();
        let mut goal = create_test_goal();
        goal.description = "Hack into the system to get data".to_string();

        let result = validator.validate_goal(&goal).unwrap();
        assert!(!result.passed);
        assert!(result
            .check_results
            .iter()
            .any(|r| r.check_name == "Content Safety Check" && !r.passed));
    }

    #[test]
    fn test_sensitive_data_without_privacy_constraints() {
        let validator = create_test_validator();
        let mut goal = create_test_goal();
        goal.description = "Analyze personal data from customers".to_string();

        let result = validator.validate_goal(&goal).unwrap();
        assert!(result
            .check_results
            .iter()
            .any(|r| r.check_name == "Data Privacy Check" && !r.passed));
    }

    #[test]
    fn test_sensitive_data_with_privacy_constraints() {
        let validator = create_test_validator();
        let mut goal = create_test_goal();
        goal.description = "Analyze personal data from customers".to_string();
        goal.add_constraint(Constraint::PrivacyRequirement {
            classification: "Sensitive".to_string(),
            restrictions: vec!["Anonymization required".to_string()],
        });

        let result = validator.validate_goal(&goal).unwrap();
        assert!(result
            .check_results
            .iter()
            .any(|r| r.check_name == "Data Privacy Check" && r.passed));
    }

    #[test]
    fn test_excessive_execution_time() {
        let validator = create_test_validator();
        let mut goal = create_test_goal();
        goal.resource_limits.max_execution_time = Some(std::time::Duration::from_secs(48 * 3600)); // Exceeds 24h limit

        let result = validator.validate_goal(&goal).unwrap();
        assert!(result
            .check_results
            .iter()
            .any(|r| r.check_name == "Execution Time Safety" && !r.passed));
    }

    #[test]
    fn test_too_many_agents() {
        let validator = create_test_validator();
        let mut goal = create_test_goal();
        goal.resource_limits.max_agents = Some(2000); // Exceeds 1000 limit

        let result = validator.validate_goal(&goal).unwrap();
        assert!(result
            .check_results
            .iter()
            .any(|r| r.check_name == "Agent Count Safety" && !r.passed));
    }

    #[test]
    fn test_financial_safety_with_budget() {
        let validator = create_test_validator();
        let mut goal = create_test_goal();
        goal.add_constraint(Constraint::BudgetLimit {
            currency: "USD".to_string(),
            max_amount: 500.0,
        });

        let result = validator.validate_goal(&goal).unwrap();
        assert!(result
            .check_results
            .iter()
            .any(|r| r.check_name == "Financial Safety Check" && r.passed));
    }

    #[test]
    fn test_financial_safety_without_budget() {
        let validator = create_test_validator();
        let goal = create_test_goal();

        let result = validator.validate_goal(&goal).unwrap();
        // Should still pass but with lower score
        assert!(result
            .check_results
            .iter()
            .any(|r| r.check_name == "Financial Safety Check" && r.score == 0.7));
    }

    #[test]
    fn test_time_constraint_safety() {
        let validator = create_test_validator();
        let mut goal = create_test_goal();
        goal.add_constraint(Constraint::TimeLimit {
            deadline: Utc::now() + Duration::hours(2), // 2 hours from now
        });

        let result = validator.validate_goal(&goal).unwrap();
        assert!(result
            .check_results
            .iter()
            .any(|r| r.check_name == "Time Constraint Safety" && r.passed));
    }

    #[test]
    fn test_insufficient_time_constraint() {
        let validator = create_test_validator();
        let mut goal = create_test_goal();
        goal.add_constraint(Constraint::TimeLimit {
            deadline: Utc::now() + Duration::minutes(30), // Only 30 minutes
        });

        let result = validator.validate_goal(&goal).unwrap();
        assert!(result
            .check_results
            .iter()
            .any(|r| r.check_name == "Time Constraint Safety" && !r.passed));
    }

    #[test]
    fn test_compliance_check_gdpr() {
        let validator = create_test_validator();
        let mut goal = create_test_goal();
        goal.description =
            "Process European customer data for GDPR compliance analysis".to_string();

        let result = validator.validate_goal(&goal).unwrap();
        assert!(result
            .check_results
            .iter()
            .any(|r| r.check_name == "GDPR Compliance Check"));
    }

    #[test]
    fn test_compliance_check_hipaa() {
        let validator = create_test_validator();
        let mut goal = create_test_goal();
        goal.description = "Analyze patient data for HIPAA compliance report".to_string();

        let result = validator.validate_goal(&goal).unwrap();
        assert!(result
            .check_results
            .iter()
            .any(|r| r.check_name == "HIPAA Compliance Check"));
    }

    #[test]
    fn test_set_custom_resource_limits() {
        let mut validator = create_test_validator();
        let new_limits = ResourceLimits {
            max_gpu_memory_mb: 16384,
            max_cpu_cores: 32,
            max_memory_mb: 65536,
            max_storage_mb: 524288,
            max_execution_hours: 12.0,
            max_cost_usd: 5000.0,
            max_agents: 500,
        };

        validator.set_resource_limits(new_limits.clone());
        assert_eq!(validator.max_limits.max_gpu_memory_mb, 16384);
        assert_eq!(validator.max_limits.max_cpu_cores, 32);
    }

    #[test]
    fn test_add_forbidden_keyword() {
        let mut validator = create_test_validator();
        let initial_count = validator.forbidden_keywords.len();

        validator.add_forbidden_keyword("dangerous".to_string());
        assert_eq!(validator.forbidden_keywords.len(), initial_count + 1);
        assert!(validator
            .forbidden_keywords
            .contains(&"dangerous".to_string()));
    }

    #[test]
    fn test_add_validation_rule() {
        let mut validator = create_test_validator();
        let initial_count = validator.validation_rules.len();

        let new_rule = ValidationRule {
            name: "Test Rule".to_string(),
            description: "Test validation rule".to_string(),
            severity: RuleSeverity::Warning,
            pattern: r"test_pattern".to_string(),
            violation_message: "Test violation".to_string(),
        };

        validator.add_validation_rule(new_rule.clone());
        assert_eq!(validator.validation_rules.len(), initial_count + 1);
        assert!(validator
            .validation_rules
            .iter()
            .any(|r| r.name == "Test Rule"));
    }

    #[test]
    fn test_get_config() {
        let validator = create_test_validator();
        let config = validator.get_config();

        assert_eq!(
            config.max_limits.max_gpu_memory_mb,
            validator.max_limits.max_gpu_memory_mb
        );
        assert_eq!(
            config.forbidden_keywords.len(),
            validator.forbidden_keywords.len()
        );
        assert_eq!(
            config.validation_rules.len(),
            validator.validation_rules.len()
        );
    }

    #[test]
    fn test_safety_check_result_serialization() {
        let result = SafetyCheckResult {
            check_name: "Test Check".to_string(),
            check_type: SafetyCheckType::ResourceUsage,
            passed: true,
            score: 0.95,
            details: "Test details".to_string(),
            recommendations: vec!["Test recommendation".to_string()],
        };

        let serialized = serde_json::to_string(&result).unwrap();
        let deserialized: SafetyCheckResult = serde_json::from_str(&serialized).unwrap();

        assert_eq!(result.check_name, deserialized.check_name);
        assert_eq!(result.check_type, deserialized.check_type);
        assert_eq!(result.passed, deserialized.passed);
        assert_eq!(result.score, deserialized.score);
    }

    #[test]
    fn test_safety_validation_result_serialization() {
        let result = SafetyValidationResult {
            passed: true,
            safety_score: 0.85,
            check_results: Vec::new(),
            warnings: vec!["Test warning".to_string()],
            errors: Vec::new(),
            mitigations: vec!["Test mitigation".to_string()],
            compliance_requirements: vec!["GDPR".to_string()],
            validated_at: Utc::now(),
        };

        let serialized = serde_json::to_string(&result).unwrap();
        let deserialized: SafetyValidationResult = serde_json::from_str(&serialized).unwrap();

        assert_eq!(result.passed, deserialized.passed);
        assert_eq!(result.safety_score, deserialized.safety_score);
        assert_eq!(result.warnings, deserialized.warnings);
    }

    #[test]
    fn test_rule_severity_ordering() {
        assert!(RuleSeverity::Critical > RuleSeverity::Error);
        assert!(RuleSeverity::Error > RuleSeverity::Warning);
        assert!(RuleSeverity::Warning > RuleSeverity::Info);
    }

    #[test]
    fn test_safety_check_types() {
        let check_types = vec![
            SafetyCheckType::ResourceUsage,
            SafetyCheckType::DataPrivacy,
            SafetyCheckType::Compliance,
            SafetyCheckType::ContentSafety,
            SafetyCheckType::OperationalSafety,
            SafetyCheckType::FinancialSafety,
            SafetyCheckType::TimeConstraints,
            SafetyCheckType::AgentBehavior,
        ];

        for check_type in check_types {
            let serialized = serde_json::to_string(&check_type).unwrap();
            let deserialized: SafetyCheckType = serde_json::from_str(&serialized).unwrap();
            assert_eq!(check_type, deserialized);
        }
    }

    #[test]
    fn test_default_validator() {
        let validator = SafetyValidator::default();
        assert_eq!(validator.max_limits.max_gpu_memory_mb, 32768);
        assert!(!validator.forbidden_keywords.is_empty());
    }
}
