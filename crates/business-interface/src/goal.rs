//! Business goal types and management

use crate::error::{BusinessError, BusinessResult};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;
use uuid::Uuid;

/// Business goal submitted via natural language
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BusinessGoal {
    /// Unique goal identifier
    pub goal_id: String,
    /// Natural language description
    pub description: String,
    /// Parsed constraints
    pub constraints: Vec<Constraint>,
    /// Success criteria
    pub success_criteria: Vec<Criterion>,
    /// Resource limits
    pub resource_limits: ResourceLimits,
    /// Goal priority
    pub priority: GoalPriority,
    /// Goal category
    pub category: GoalCategory,
    /// Estimated completion time
    pub estimated_duration: Option<Duration>,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// User/organization that submitted the goal
    pub submitted_by: String,
    /// Current status
    pub status: GoalStatus,
    /// Progress percentage (0-100)
    pub progress: f32,
    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Constraint types for goals
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Constraint {
    /// Time constraint
    TimeLimit { deadline: DateTime<Utc> },
    /// Resource constraint
    ResourceLimit { resource: String, max_value: f64 },
    /// Safety constraint
    SafetyRequirement {
        requirement: String,
        level: SafetyLevel,
    },
    /// Compliance constraint
    ComplianceRequirement {
        standard: String,
        certification: String,
    },
    /// Budget constraint
    BudgetLimit { currency: String, max_amount: f64 },
    /// Geographic constraint
    GeographicLimit { regions: Vec<String> },
    /// Data privacy constraint
    PrivacyRequirement {
        classification: String,
        restrictions: Vec<String>,
    },
}

/// Success criteria for goals
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Criterion {
    /// Performance metric
    Performance {
        metric: String,
        target_value: f64,
        comparison: ComparisonOperator,
    },
    /// Quality metric
    Quality { aspect: String, min_score: f64 },
    /// Completion metric
    Completion { percentage: f32 },
    /// Accuracy metric
    Accuracy { min_accuracy: f64 },
    /// Efficiency metric
    Efficiency { metric: String, min_efficiency: f64 },
    /// User satisfaction metric
    UserSatisfaction { min_rating: f64, sample_size: u32 },
    /// Business metric
    BusinessMetric {
        kpi: String,
        target: f64,
        timeframe: Duration,
    },
}

/// Resource limits for goal execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    /// Maximum GPU memory in MB
    pub max_gpu_memory_mb: Option<u32>,
    /// Maximum CPU cores
    pub max_cpu_cores: Option<u32>,
    /// Maximum system memory in MB
    pub max_memory_mb: Option<u32>,
    /// Maximum storage in MB
    pub max_storage_mb: Option<u32>,
    /// Maximum execution time
    pub max_execution_time: Option<Duration>,
    /// Maximum cost in USD
    pub max_cost_usd: Option<f64>,
    /// Maximum number of agents
    pub max_agents: Option<u32>,
    /// Maximum network bandwidth in Mbps
    pub max_network_mbps: Option<f32>,
}

/// Goal priority levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum GoalPriority {
    Low = 1,
    Medium = 2,
    High = 3,
    Critical = 4,
    Emergency = 5,
}

/// Goal categories
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum GoalCategory {
    /// Data analysis and processing
    DataAnalysis,
    /// Machine learning and AI
    MachineLearning,
    /// Research and development
    Research,
    /// Business intelligence
    BusinessIntelligence,
    /// Optimization tasks
    Optimization,
    /// Simulation and modeling
    Simulation,
    /// Testing and validation
    Testing,
    /// Content generation
    ContentGeneration,
    /// Decision support
    DecisionSupport,
    /// Process automation
    Automation,
    /// Custom category
    Custom(String),
}

/// Safety levels for constraints
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, PartialOrd)]
pub enum SafetyLevel {
    Low = 1,
    Medium = 2,
    High = 3,
    Critical = 4,
}

/// Comparison operators for criteria
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ComparisonOperator {
    GreaterThan,
    GreaterThanOrEqual,
    LessThan,
    LessThanOrEqual,
    Equal,
    NotEqual,
}

/// Goal execution status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum GoalStatus {
    /// Goal submitted but not yet started
    Pending,
    /// Goal is being validated
    Validating,
    /// Goal validation failed
    ValidationFailed { reason: String },
    /// Goal is approved and queued
    Approved,
    /// Goal is currently executing
    Executing { started_at: DateTime<Utc> },
    /// Goal execution paused
    Paused { reason: String },
    /// Goal completed successfully
    Completed { completed_at: DateTime<Utc> },
    /// Goal execution failed
    Failed {
        reason: String,
        failed_at: DateTime<Utc>,
    },
    /// Goal was cancelled
    Cancelled {
        reason: String,
        cancelled_at: DateTime<Utc>,
    },
}

impl BusinessGoal {
    /// Create a new business goal
    pub fn new(description: String, submitted_by: String) -> Self {
        Self {
            goal_id: Uuid::new_v4().to_string(),
            description,
            constraints: Vec::new(),
            success_criteria: Vec::new(),
            resource_limits: ResourceLimits::default(),
            priority: GoalPriority::Medium,
            category: GoalCategory::Custom("General".to_string()),
            estimated_duration: None,
            created_at: Utc::now(),
            submitted_by,
            status: GoalStatus::Pending,
            progress: 0.0,
            metadata: HashMap::new(),
        }
    }

    /// Add a constraint to the goal
    pub fn add_constraint(&mut self, constraint: Constraint) {
        self.constraints.push(constraint);
    }

    /// Add a success criterion to the goal
    pub fn add_criterion(&mut self, criterion: Criterion) {
        self.success_criteria.push(criterion);
    }

    /// Update goal status
    pub fn update_status(&mut self, status: GoalStatus) {
        self.status = status;
    }

    /// Update progress
    pub fn update_progress(&mut self, progress: f32) -> BusinessResult<()> {
        if !(0.0..=100.0).contains(&progress) {
            return Err(BusinessError::GoalValidationFailed {
                reason: format!("Progress must be between 0 and 100, got {}", progress),
            });
        }
        self.progress = progress;
        Ok(())
    }

    /// Validate the goal
    pub fn validate(&self) -> BusinessResult<()> {
        if self.description.is_empty() {
            return Err(BusinessError::GoalValidationFailed {
                reason: "Goal description cannot be empty".to_string(),
            });
        }

        if self.submitted_by.is_empty() {
            return Err(BusinessError::GoalValidationFailed {
                reason: "Submitted by field cannot be empty".to_string(),
            });
        }

        // Validate constraints
        for constraint in &self.constraints {
            match constraint {
                Constraint::TimeLimit { deadline } => {
                    if *deadline <= Utc::now() {
                        return Err(BusinessError::GoalValidationFailed {
                            reason: "Deadline must be in the future".to_string(),
                        });
                    }
                }
                Constraint::ResourceLimit { max_value, .. } => {
                    if *max_value <= 0.0 {
                        return Err(BusinessError::GoalValidationFailed {
                            reason: "Resource limits must be positive".to_string(),
                        });
                    }
                }
                Constraint::BudgetLimit { max_amount, .. } => {
                    if *max_amount <= 0.0 {
                        return Err(BusinessError::GoalValidationFailed {
                            reason: "Budget limits must be positive".to_string(),
                        });
                    }
                }
                _ => {} // Other constraints don't need validation here
            }
        }

        // Validate success criteria
        for criterion in &self.success_criteria {
            match criterion {
                Criterion::Completion { percentage } => {
                    if !(0.0..=100.0).contains(percentage) {
                        return Err(BusinessError::GoalValidationFailed {
                            reason: "Completion percentage must be between 0 and 100".to_string(),
                        });
                    }
                }
                Criterion::Accuracy { min_accuracy } => {
                    if !(0.0..=1.0).contains(min_accuracy) {
                        return Err(BusinessError::GoalValidationFailed {
                            reason: "Accuracy must be between 0.0 and 1.0".to_string(),
                        });
                    }
                }
                _ => {} // Other criteria validated elsewhere
            }
        }

        Ok(())
    }

    /// Check if goal is active (executing or paused)
    pub fn is_active(&self) -> bool {
        matches!(
            self.status,
            GoalStatus::Executing { .. } | GoalStatus::Paused { .. }
        )
    }

    /// Check if goal is completed
    pub fn is_completed(&self) -> bool {
        matches!(
            self.status,
            GoalStatus::Completed { .. } | GoalStatus::Failed { .. } | GoalStatus::Cancelled { .. }
        )
    }

    /// Get estimated resource usage
    pub fn get_estimated_resources(&self) -> &ResourceLimits {
        &self.resource_limits
    }
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_gpu_memory_mb: Some(8192), // 8GB default
            max_cpu_cores: Some(8),
            max_memory_mb: Some(16384),  // 16GB default
            max_storage_mb: Some(10240), // 10GB default
            max_execution_time: Some(Duration::from_secs(3600)),
            max_cost_usd: Some(100.0),
            max_agents: Some(10),
            max_network_mbps: Some(1000.0),
        }
    }
}

impl Default for GoalPriority {
    fn default() -> Self {
        GoalPriority::Medium
    }
}

impl Default for GoalCategory {
    fn default() -> Self {
        GoalCategory::Custom("General".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration as ChronoDuration;

    fn create_test_goal() -> BusinessGoal {
        BusinessGoal::new(
            "Analyze customer data and generate insights".to_string(),
            "user@example.com".to_string(),
        )
    }

    #[test]
    fn test_business_goal_creation() {
        let goal = create_test_goal();
        assert!(!goal.goal_id.is_empty());
        assert_eq!(
            goal.description,
            "Analyze customer data and generate insights"
        );
        assert_eq!(goal.submitted_by, "user@example.com");
        assert_eq!(goal.status, GoalStatus::Pending);
        assert_eq!(goal.progress, 0.0);
        assert!(goal.constraints.is_empty());
        assert!(goal.success_criteria.is_empty());
    }

    #[test]
    fn test_add_constraint() {
        let mut goal = create_test_goal();
        let constraint = Constraint::TimeLimit {
            deadline: Utc::now() + ChronoDuration::hours(24),
        };
        goal.add_constraint(constraint.clone());
        assert_eq!(goal.constraints.len(), 1);
        assert_eq!(goal.constraints[0], constraint);
    }

    #[test]
    fn test_add_criterion() {
        let mut goal = create_test_goal();
        let criterion = Criterion::Accuracy { min_accuracy: 0.95 };
        goal.add_criterion(criterion.clone());
        assert_eq!(goal.success_criteria.len(), 1);
        assert_eq!(goal.success_criteria[0], criterion);
    }

    #[test]
    fn test_update_status() {
        let mut goal = create_test_goal();
        let new_status = GoalStatus::Executing {
            started_at: Utc::now(),
        };
        goal.update_status(new_status.clone());
        assert_eq!(goal.status, new_status);
    }

    #[test]
    fn test_update_progress_valid() {
        let mut goal = create_test_goal();
        assert!(goal.update_progress(50.0).is_ok());
        assert_eq!(goal.progress, 50.0);
    }

    #[test]
    fn test_update_progress_invalid() {
        let mut goal = create_test_goal();
        assert!(goal.update_progress(-10.0).is_err());
        assert!(goal.update_progress(150.0).is_err());
    }

    #[test]
    fn test_goal_validation_success() {
        let goal = create_test_goal();
        assert!(goal.validate().is_ok());
    }

    #[test]
    fn test_goal_validation_empty_description() {
        let mut goal = create_test_goal();
        goal.description = String::new();
        assert!(goal.validate().is_err());
    }

    #[test]
    fn test_goal_validation_empty_submitter() {
        let mut goal = create_test_goal();
        goal.submitted_by = String::new();
        assert!(goal.validate().is_err());
    }

    #[test]
    fn test_goal_validation_past_deadline() {
        let mut goal = create_test_goal();
        goal.add_constraint(Constraint::TimeLimit {
            deadline: Utc::now() - ChronoDuration::hours(1),
        });
        assert!(goal.validate().is_err());
    }

    #[test]
    fn test_goal_validation_negative_resource_limit() {
        let mut goal = create_test_goal();
        goal.add_constraint(Constraint::ResourceLimit {
            resource: "CPU".to_string(),
            max_value: -1.0,
        });
        assert!(goal.validate().is_err());
    }

    #[test]
    fn test_goal_validation_invalid_completion_percentage() {
        let mut goal = create_test_goal();
        goal.add_criterion(Criterion::Completion { percentage: 150.0 });
        assert!(goal.validate().is_err());
    }

    #[test]
    fn test_goal_validation_invalid_accuracy() {
        let mut goal = create_test_goal();
        goal.add_criterion(Criterion::Accuracy { min_accuracy: 1.5 });
        assert!(goal.validate().is_err());
    }

    #[test]
    fn test_is_active() {
        let mut goal = create_test_goal();
        assert!(!goal.is_active());

        goal.update_status(GoalStatus::Executing {
            started_at: Utc::now(),
        });
        assert!(goal.is_active());

        goal.update_status(GoalStatus::Paused {
            reason: "User request".to_string(),
        });
        assert!(goal.is_active());

        goal.update_status(GoalStatus::Completed {
            completed_at: Utc::now(),
        });
        assert!(!goal.is_active());
    }

    #[test]
    fn test_is_completed() {
        let mut goal = create_test_goal();
        assert!(!goal.is_completed());

        goal.update_status(GoalStatus::Executing {
            started_at: Utc::now(),
        });
        assert!(!goal.is_completed());

        goal.update_status(GoalStatus::Completed {
            completed_at: Utc::now(),
        });
        assert!(goal.is_completed());

        goal.update_status(GoalStatus::Failed {
            reason: "Test failure".to_string(),
            failed_at: Utc::now(),
        });
        assert!(goal.is_completed());

        goal.update_status(GoalStatus::Cancelled {
            reason: "User cancelled".to_string(),
            cancelled_at: Utc::now(),
        });
        assert!(goal.is_completed());
    }

    #[test]
    fn test_constraint_serialization() {
        let constraints = vec![
            Constraint::TimeLimit {
                deadline: Utc::now() + ChronoDuration::hours(24),
            },
            Constraint::ResourceLimit {
                resource: "GPU".to_string(),
                max_value: 8192.0,
            },
            Constraint::SafetyRequirement {
                requirement: "Data encryption".to_string(),
                level: SafetyLevel::High,
            },
            Constraint::BudgetLimit {
                currency: "USD".to_string(),
                max_amount: 1000.0,
            },
        ];

        for constraint in constraints {
            let serialized = serde_json::to_string(&constraint).unwrap();
            let deserialized: Constraint = serde_json::from_str(&serialized).unwrap();
            assert_eq!(constraint, deserialized);
        }
    }

    #[test]
    fn test_criterion_serialization() {
        let criteria = vec![
            Criterion::Performance {
                metric: "Throughput".to_string(),
                target_value: 1000.0,
                comparison: ComparisonOperator::GreaterThanOrEqual,
            },
            Criterion::Quality {
                aspect: "Accuracy".to_string(),
                min_score: 0.95,
            },
            Criterion::Completion { percentage: 100.0 },
            Criterion::Accuracy { min_accuracy: 0.99 },
        ];

        for criterion in criteria {
            let serialized = serde_json::to_string(&criterion).unwrap();
            let deserialized: Criterion = serde_json::from_str(&serialized).unwrap();
            assert_eq!(criterion, deserialized);
        }
    }

    #[test]
    fn test_resource_limits_default() {
        let limits = ResourceLimits::default();
        assert_eq!(limits.max_gpu_memory_mb, Some(8192));
        assert_eq!(limits.max_cpu_cores, Some(8));
        assert_eq!(limits.max_memory_mb, Some(16384));
        assert_eq!(limits.max_cost_usd, Some(100.0));
    }

    #[test]
    fn test_goal_priority_ordering() {
        assert!(GoalPriority::Critical > GoalPriority::High);
        assert!(GoalPriority::High > GoalPriority::Medium);
        assert!(GoalPriority::Medium > GoalPriority::Low);
        assert!(GoalPriority::Emergency > GoalPriority::Critical);
    }

    #[test]
    fn test_safety_level_ordering() {
        assert!(SafetyLevel::Critical > SafetyLevel::High);
        assert!(SafetyLevel::High > SafetyLevel::Medium);
        assert!(SafetyLevel::Medium > SafetyLevel::Low);
    }

    #[test]
    fn test_goal_serialization() {
        let goal = create_test_goal();
        let serialized = serde_json::to_string(&goal).unwrap();
        let deserialized: BusinessGoal = serde_json::from_str(&serialized).unwrap();
        assert_eq!(goal.goal_id, deserialized.goal_id);
        assert_eq!(goal.description, deserialized.description);
        assert_eq!(goal.submitted_by, deserialized.submitted_by);
    }

    #[test]
    fn test_goal_categories() {
        let categories = vec![
            GoalCategory::DataAnalysis,
            GoalCategory::MachineLearning,
            GoalCategory::Research,
            GoalCategory::Custom("MyCategory".to_string()),
        ];

        for category in categories {
            let serialized = serde_json::to_string(&category).unwrap();
            let deserialized: GoalCategory = serde_json::from_str(&serialized).unwrap();
            assert_eq!(category, deserialized);
        }
    }

    #[test]
    fn test_comparison_operators() {
        let operators = vec![
            ComparisonOperator::GreaterThan,
            ComparisonOperator::GreaterThanOrEqual,
            ComparisonOperator::LessThan,
            ComparisonOperator::LessThanOrEqual,
            ComparisonOperator::Equal,
            ComparisonOperator::NotEqual,
        ];

        for operator in operators {
            let serialized = serde_json::to_string(&operator).unwrap();
            let deserialized: ComparisonOperator = serde_json::from_str(&serialized).unwrap();
            assert_eq!(operator, deserialized);
        }
    }

    #[test]
    fn test_goal_status_variants() {
        let statuses = vec![
            GoalStatus::Pending,
            GoalStatus::Validating,
            GoalStatus::ValidationFailed {
                reason: "Test".to_string(),
            },
            GoalStatus::Approved,
            GoalStatus::Executing {
                started_at: Utc::now(),
            },
            GoalStatus::Completed {
                completed_at: Utc::now(),
            },
        ];

        for status in statuses {
            let serialized = serde_json::to_string(&status).unwrap();
            let deserialized: GoalStatus = serde_json::from_str(&serialized).unwrap();
            assert_eq!(status, deserialized);
        }
    }
}
