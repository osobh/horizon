//! Skill system for AgentDNA

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// A learned skill with evidence of competence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Skill {
    /// Unique skill identifier
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Description of what this skill does
    pub description: String,
    /// Skill category for organization
    pub category: SkillCategory,
    /// Proficiency level (1-10)
    pub proficiency_level: u32,
    /// Evidence of skill competence
    pub evidence: SkillEvidence,
    /// Prerequisites (other skill IDs)
    pub prerequisites: Vec<String>,
    /// Execution strategy
    pub execution: SkillExecution,
    /// When this skill was learned
    pub learned_at: DateTime<Utc>,
    /// Last successful use
    pub last_used: Option<DateTime<Utc>>,
}

impl Skill {
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        category: SkillCategory,
        execution: SkillExecution,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            description: String::new(),
            category,
            proficiency_level: 1,
            evidence: SkillEvidence::default(),
            prerequisites: Vec::new(),
            execution,
            learned_at: Utc::now(),
            last_used: None,
        }
    }

    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    pub fn with_proficiency(mut self, level: u32) -> Self {
        self.proficiency_level = level.clamp(1, 10);
        self
    }

    pub fn with_prerequisites(mut self, prerequisites: Vec<String>) -> Self {
        self.prerequisites = prerequisites;
        self
    }

    /// Calculate success rate from evidence
    pub fn success_rate(&self) -> f64 {
        let total = self.evidence.success_count + self.evidence.failure_count;
        if total == 0 {
            0.0
        } else {
            self.evidence.success_count as f64 / total as f64
        }
    }

    /// Check if skill needs refreshing (not used recently)
    pub fn needs_refresh(&self, threshold: Duration) -> bool {
        match self.last_used {
            Some(last) => Utc::now().signed_duration_since(last).to_std().unwrap_or_default() > threshold,
            None => true,
        }
    }
}

/// Categories of skills
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SkillCategory {
    // Migrated from existing agents
    CapacityPlanning,
    SchedulerOptimization,
    EfficiencyHunting,
    CostOptimization,
    PolicyGovernance,
    IncidentResponse,
    TelemetryMonitoring,
    AuditLogging,
    // New categories
    Orchestration,
    DataAnalysis,
    Reconciliation,
    Postprocessing,
    Learning,
    Communication,
    Custom(String),
}

impl SkillCategory {
    pub fn as_str(&self) -> &str {
        match self {
            Self::CapacityPlanning => "capacity_planning",
            Self::SchedulerOptimization => "scheduler_optimization",
            Self::EfficiencyHunting => "efficiency_hunting",
            Self::CostOptimization => "cost_optimization",
            Self::PolicyGovernance => "policy_governance",
            Self::IncidentResponse => "incident_response",
            Self::TelemetryMonitoring => "telemetry_monitoring",
            Self::AuditLogging => "audit_logging",
            Self::Orchestration => "orchestration",
            Self::DataAnalysis => "data_analysis",
            Self::Reconciliation => "reconciliation",
            Self::Postprocessing => "postprocessing",
            Self::Learning => "learning",
            Self::Communication => "communication",
            Self::Custom(s) => s,
        }
    }
}

/// Evidence supporting skill competence
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SkillEvidence {
    /// Number of successful executions
    pub success_count: u64,
    /// Number of failed executions
    pub failure_count: u64,
    /// Average execution time
    #[serde(with = "duration_serde")]
    pub avg_execution_time: Duration,
    /// Performance scores over time (limited history)
    pub performance_history: Vec<PerformancePoint>,
    /// Validation results
    pub validations: Vec<ValidationRecord>,
}

impl SkillEvidence {
    pub fn record_success(&mut self, execution_time: Duration, score: f64) {
        self.success_count += 1;
        self.update_avg_time(execution_time);
        self.add_performance_point(score);
    }

    pub fn record_failure(&mut self, execution_time: Duration) {
        self.failure_count += 1;
        self.update_avg_time(execution_time);
    }

    fn update_avg_time(&mut self, new_time: Duration) {
        let total = self.success_count + self.failure_count;
        if total == 1 {
            self.avg_execution_time = new_time;
        } else {
            // Exponential moving average
            let alpha = 0.1;
            let old_nanos = self.avg_execution_time.as_nanos() as f64;
            let new_nanos = new_time.as_nanos() as f64;
            let avg_nanos = old_nanos * (1.0 - alpha) + new_nanos * alpha;
            self.avg_execution_time = Duration::from_nanos(avg_nanos as u64);
        }
    }

    fn add_performance_point(&mut self, score: f64) {
        self.performance_history.push(PerformancePoint {
            timestamp: Utc::now(),
            score,
            context: String::new(),
        });
        // Keep only last 100 points
        if self.performance_history.len() > 100 {
            self.performance_history.remove(0);
        }
    }

    /// Calculate recent performance trend
    pub fn recent_trend(&self) -> f64 {
        if self.performance_history.len() < 2 {
            return 0.0;
        }
        let recent: Vec<f64> = self.performance_history.iter().rev().take(10).map(|p| p.score).collect();
        if recent.len() < 2 {
            return 0.0;
        }
        let first_half: f64 = recent[recent.len() / 2..].iter().sum::<f64>() / (recent.len() / 2) as f64;
        let second_half: f64 = recent[..recent.len() / 2].iter().sum::<f64>() / (recent.len() / 2) as f64;
        second_half - first_half
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformancePoint {
    pub timestamp: DateTime<Utc>,
    pub score: f64,
    pub context: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationRecord {
    pub timestamp: DateTime<Utc>,
    pub validator: String,
    pub success: bool,
    pub confidence: f64,
    pub details: serde_json::Value,
}

/// How a skill is executed
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SkillExecution {
    /// Hardcoded logic (migrated from existing agents)
    Builtin { handler_id: String },
    /// Learned workflow (from DiscoveredWorkflow pattern)
    LearnedWorkflow {
        workflow_id: String,
        generation: usize,
        parameters: HashMap<String, serde_json::Value>,
    },
    /// Pattern-based execution (from GrowthPattern)
    PatternBased { pattern_id: String, template: String },
    /// LLM-powered execution with prompt
    LLMPowered {
        system_prompt: String,
        tool_usage: Vec<String>,
    },
    /// Composite of multiple sub-skills
    Composite {
        sub_skills: Vec<String>,
        coordination: CoordinationStrategy,
    },
}

/// Coordination strategy for composite skills
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CoordinationStrategy {
    Sequential,
    Parallel,
    Pipeline,
    Hierarchical,
    Consensus,
}

/// Skill repertoire with indexing for fast lookup
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SkillRepertoire {
    /// All skills by ID
    pub skills: HashMap<String, Skill>,
    /// Index by category for fast lookup
    #[serde(skip)]
    by_category: HashMap<SkillCategory, Vec<String>>,
    /// Index by capability for service discovery
    #[serde(skip)]
    by_capability: HashMap<String, Vec<String>>,
}

impl SkillRepertoire {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_skill(&mut self, skill: Skill) {
        let id = skill.id.clone();
        let category = skill.category.clone();

        self.by_category
            .entry(category)
            .or_default()
            .push(id.clone());

        self.skills.insert(id, skill);
    }

    pub fn get_skill(&self, id: &str) -> Option<&Skill> {
        self.skills.get(id)
    }

    pub fn get_skill_mut(&mut self, id: &str) -> Option<&mut Skill> {
        self.skills.get_mut(id)
    }

    pub fn remove_skill(&mut self, id: &str) -> Option<Skill> {
        if let Some(skill) = self.skills.remove(id) {
            if let Some(ids) = self.by_category.get_mut(&skill.category) {
                ids.retain(|s| s != id);
            }
            Some(skill)
        } else {
            None
        }
    }

    pub fn find_by_category(&self, category: &SkillCategory) -> Vec<&Skill> {
        self.by_category
            .get(category)
            .map(|ids| ids.iter().filter_map(|id| self.skills.get(id)).collect())
            .unwrap_or_default()
    }

    pub fn total_proficiency(&self) -> f64 {
        if self.skills.is_empty() {
            return 0.0;
        }
        let sum: u32 = self.skills.values().map(|s| s.proficiency_level).sum();
        sum as f64 / self.skills.len() as f64
    }

    pub fn skill_count(&self) -> usize {
        self.skills.len()
    }

    /// Rebuild indexes after deserialization
    pub fn rebuild_indexes(&mut self) {
        self.by_category.clear();
        self.by_capability.clear();

        for (id, skill) in &self.skills {
            self.by_category
                .entry(skill.category.clone())
                .or_default()
                .push(id.clone());
        }
    }
}

mod duration_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::time::Duration;

    pub fn serialize<S>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        duration.as_nanos().serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        let nanos = u128::deserialize(deserializer)?;
        Ok(Duration::from_nanos(nanos as u64))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_skill_creation() {
        let skill = Skill::new(
            "test_skill",
            "Test Skill",
            SkillCategory::EfficiencyHunting,
            SkillExecution::Builtin {
                handler_id: "test".to_string(),
            },
        )
        .with_proficiency(5)
        .with_description("A test skill");

        assert_eq!(skill.id, "test_skill");
        assert_eq!(skill.proficiency_level, 5);
        assert_eq!(skill.success_rate(), 0.0);
    }

    #[test]
    fn test_skill_evidence() {
        let mut evidence = SkillEvidence::default();
        evidence.record_success(Duration::from_millis(100), 0.9);
        evidence.record_success(Duration::from_millis(150), 0.95);
        evidence.record_failure(Duration::from_millis(200));

        assert_eq!(evidence.success_count, 2);
        assert_eq!(evidence.failure_count, 1);
    }

    #[test]
    fn test_skill_repertoire() {
        let mut repertoire = SkillRepertoire::new();

        let skill1 = Skill::new(
            "skill1",
            "Skill 1",
            SkillCategory::EfficiencyHunting,
            SkillExecution::Builtin {
                handler_id: "h1".to_string(),
            },
        );

        let skill2 = Skill::new(
            "skill2",
            "Skill 2",
            SkillCategory::EfficiencyHunting,
            SkillExecution::Builtin {
                handler_id: "h2".to_string(),
            },
        );

        repertoire.add_skill(skill1);
        repertoire.add_skill(skill2);

        assert_eq!(repertoire.skill_count(), 2);
        assert_eq!(
            repertoire
                .find_by_category(&SkillCategory::EfficiencyHunting)
                .len(),
            2
        );
    }
}
