//! Learning engine for DNA evolution through experience

use crate::dna::Skill;
use crate::error::Result;
use std::time::Duration;

/// Learning engine for DNA evolution
pub struct LearningEngine {
    learning_rate: f64,
}

impl LearningEngine {
    pub fn new(learning_rate: f64) -> Self {
        Self { learning_rate }
    }

    /// Analyze execution and suggest improvements
    pub async fn analyze_execution(
        &self,
        skill_id: &str,
        success: bool,
        _execution_time: Duration,
    ) -> Result<LearningResult> {
        // Calculate performance delta based on success and speed
        let performance_delta = if success {
            self.learning_rate * 0.1
        } else {
            -self.learning_rate * 0.05
        };

        Ok(LearningResult {
            skill_id: skill_id.to_string(),
            performance_delta,
            suggested_improvement: if success && performance_delta > 0.05 {
                Some(LearningImprovement::SkillUpgrade {
                    new_proficiency: 1, // Will be added to current
                    performance_delta,
                })
            } else {
                None
            },
        })
    }

    /// Learn from reconciliation operations
    pub async fn learn_from_reconciliation(
        &self,
        _context: &str,
        corrections_made: usize,
    ) -> Result<Vec<LearningImprovement>> {
        let mut improvements = Vec::new();

        if corrections_made > 0 {
            improvements.push(LearningImprovement::BehaviorAdjustment {
                parameter: "accuracy".to_string(),
                delta: self.learning_rate * 0.01 * corrections_made as f64,
            });
        }

        Ok(improvements)
    }

    /// Learn from postprocessing operations
    pub async fn learn_from_postprocessing(
        &self,
        _operation_id: &str,
        effectiveness: f64,
    ) -> Result<Vec<LearningImprovement>> {
        let mut improvements = Vec::new();

        if effectiveness > 0.8 {
            // High effectiveness suggests a new pattern worth learning
            improvements.push(LearningImprovement::BehaviorAdjustment {
                parameter: "confidence_threshold".to_string(),
                delta: -self.learning_rate * 0.01, // Lower threshold for faster action
            });
        }

        Ok(improvements)
    }
}

/// Result of learning analysis
#[derive(Debug)]
pub struct LearningResult {
    pub skill_id: String,
    pub performance_delta: f64,
    pub suggested_improvement: Option<LearningImprovement>,
}

/// Types of improvements from learning
#[derive(Debug)]
pub enum LearningImprovement {
    SkillUpgrade {
        new_proficiency: u32,
        performance_delta: f64,
    },
    BehaviorAdjustment {
        parameter: String,
        delta: f64,
    },
    NewSkillDiscovered {
        skill: Skill,
    },
}
