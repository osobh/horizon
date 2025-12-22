//! Recovery plan optimization algorithms

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Optimization goals for recovery planning
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OptimizationGoals {
    /// Minimize recovery time
    MinimizeTime,
    /// Minimize resource usage
    MinimizeCost,
    /// Maximize reliability
    MaximizeReliability,
    /// Balance time and cost
    Balanced,
}

/// Plan optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanOptimization {
    /// Primary optimization goal
    pub primary_goal: OptimizationGoals,
    /// Secondary goals with weights
    pub secondary_goals: HashMap<OptimizationGoals, f64>,
    /// Constraints to respect
    pub constraints: OptimizationConstraints,
    /// Maximum iterations for optimization algorithms
    pub max_iterations: u32,
    /// Convergence threshold
    pub convergence_threshold: f64,
}

/// Optimization constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConstraints {
    /// Maximum total recovery time
    pub max_recovery_time_minutes: Option<u64>,
    /// Maximum resource cost
    pub max_resource_cost: Option<f64>,
    /// Minimum reliability score
    pub min_reliability_score: Option<f64>,
    /// Required service level agreements
    pub sla_requirements: Vec<SlaRequirement>,
}

/// SLA requirement definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlaRequirement {
    /// Service ID
    pub service_id: Uuid,
    /// Required uptime percentage
    pub uptime_percentage: f64,
    /// Maximum allowed downtime in minutes
    pub max_downtime_minutes: u64,
}

/// Optimization result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationResult {
    /// Optimized plan ID
    pub plan_id: Uuid,
    /// Optimization score (higher is better)
    pub score: f64,
    /// Estimated recovery time
    pub estimated_recovery_time_minutes: u64,
    /// Estimated resource cost
    pub estimated_cost: f64,
    /// Reliability score
    pub reliability_score: f64,
    /// Steps that were reordered or modified
    pub modifications: Vec<PlanModification>,
    /// Optimization iterations performed
    pub iterations: u32,
}

/// Plan modification record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanModification {
    /// Type of modification
    pub modification_type: ModificationType,
    /// Step ID that was modified
    pub step_id: Uuid,
    /// Description of change
    pub description: String,
    /// Impact on metrics
    pub impact: ModificationImpact,
}

/// Types of plan modifications
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModificationType {
    /// Step was reordered
    Reordered,
    /// Step was parallelized with others
    Parallelized,
    /// Step resources were adjusted
    ResourceAdjusted,
    /// Step timeout was modified
    TimeoutAdjusted,
    /// Step was marked optional
    MadeOptional,
    /// Step was marked critical
    MadeCritical,
}

/// Impact of a modification on plan metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModificationImpact {
    /// Change in recovery time (negative is improvement)
    pub time_change_minutes: i64,
    /// Change in cost (negative is savings)
    pub cost_change: f64,
    /// Change in reliability score (positive is improvement)
    pub reliability_change: f64,
}

impl PlanOptimization {
    /// Create new optimization configuration
    pub fn new(goal: OptimizationGoals) -> Self {
        Self {
            primary_goal: goal,
            secondary_goals: HashMap::new(),
            constraints: OptimizationConstraints {
                max_recovery_time_minutes: None,
                max_resource_cost: None,
                min_reliability_score: None,
                sla_requirements: Vec::new(),
            },
            max_iterations: 100,
            convergence_threshold: 0.01,
        }
    }

    /// Add secondary optimization goal
    pub fn add_secondary_goal(&mut self, goal: OptimizationGoals, weight: f64) {
        self.secondary_goals.insert(goal, weight);
    }

    /// Set time constraint
    pub fn set_max_time(&mut self, minutes: u64) {
        self.constraints.max_recovery_time_minutes = Some(minutes);
    }

    /// Set cost constraint
    pub fn set_max_cost(&mut self, cost: f64) {
        self.constraints.max_resource_cost = Some(cost);
    }

    /// Set reliability constraint
    pub fn set_min_reliability(&mut self, score: f64) {
        self.constraints.min_reliability_score = Some(score);
    }

    /// Add SLA requirement
    pub fn add_sla_requirement(&mut self, requirement: SlaRequirement) {
        self.constraints.sla_requirements.push(requirement);
    }

    /// Calculate optimization score for a plan
    pub fn calculate_score(
        &self,
        recovery_time: u64,
        cost: f64,
        reliability: f64,
    ) -> f64 {
        let mut score = 0.0;

        // Primary goal scoring
        match self.primary_goal {
            OptimizationGoals::MinimizeTime => {
                score += 1000.0 / (recovery_time as f64 + 1.0); // Higher score for lower time
            }
            OptimizationGoals::MinimizeCost => {
                score += 1000.0 / (cost + 1.0); // Higher score for lower cost
            }
            OptimizationGoals::MaximizeReliability => {
                score += reliability * 1000.0; // Higher score for higher reliability
            }
            OptimizationGoals::Balanced => {
                score += (1000.0 / (recovery_time as f64 + 1.0)) * 0.4;
                score += (1000.0 / (cost + 1.0)) * 0.3;
                score += reliability * 1000.0 * 0.3;
            }
        }

        // Secondary goals scoring
        for (goal, weight) in &self.secondary_goals {
            let secondary_score = match goal {
                OptimizationGoals::MinimizeTime => 1000.0 / (recovery_time as f64 + 1.0),
                OptimizationGoals::MinimizeCost => 1000.0 / (cost + 1.0),
                OptimizationGoals::MaximizeReliability => reliability * 1000.0,
                OptimizationGoals::Balanced => {
                    ((1000.0 / (recovery_time as f64 + 1.0)) + (1000.0 / (cost + 1.0)) + (reliability * 1000.0)) / 3.0
                }
            };
            score += secondary_score * weight;
        }

        score
    }

    /// Check if plan satisfies constraints
    pub fn satisfies_constraints(
        &self,
        recovery_time: u64,
        cost: f64,
        reliability: f64,
    ) -> bool {
        if let Some(max_time) = self.constraints.max_recovery_time_minutes {
            if recovery_time > max_time {
                return false;
            }
        }

        if let Some(max_cost) = self.constraints.max_resource_cost {
            if cost > max_cost {
                return false;
            }
        }

        if let Some(min_reliability) = self.constraints.min_reliability_score {
            if reliability < min_reliability {
                return false;
            }
        }

        // Check SLA requirements
        for sla in &self.constraints.sla_requirements {
            // Check if recovery time exceeds maximum allowed downtime
            if recovery_time > sla.max_downtime_minutes {
                return false;
            }
            
            // For uptime percentage, assume current downtime reduces overall uptime
            // If we have 99.9% uptime requirement and this recovery takes too long,
            // it might violate the monthly/yearly uptime SLA
            let monthly_minutes = 30 * 24 * 60; // 30 days in minutes
            let max_downtime_for_sla = monthly_minutes as f64 * (1.0 - sla.uptime_percentage / 100.0);
            
            if recovery_time as f64 > max_downtime_for_sla {
                return false;
            }
        }
        
        true
    }
}
