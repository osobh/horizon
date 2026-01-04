//! AgentDNA - The versioned capability blueprint
//!
//! This module defines the core DNA structures that configure a UniversalAgent.
//! DNA is versioned and tracks its evolution lineage, enabling:
//!
//! - Version management with semantic versioning + generation tracking
//! - Lineage tracking for evolution history
//! - Skill repertoire with evidence of competence
//! - Behavior and architecture configuration
//! - Capability advertisement for service discovery
//! - Fitness profiling with benchmarks

mod capabilities;
mod genome;
mod skills;
mod version;

pub use capabilities::*;
pub use genome::*;
pub use skills::*;
pub use version::*;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// The core AgentDNA structure - a versioned blueprint for agent capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentDNA {
    /// Unique identifier
    pub id: DNAId,
    /// Version information
    pub version: DNAVersion,
    /// Lineage tracking
    pub lineage: DNALineage,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Last modified timestamp
    pub updated_at: DateTime<Utc>,

    // === Core Components ===
    /// Skill repertoire with evidence
    pub skills: SkillRepertoire,
    /// Behavioral parameters
    pub behavior: BehaviorGenome,
    /// Architectural configuration
    pub architecture: ArchitectureGenome,
    /// Role capabilities (discoverable by services)
    pub capabilities: Vec<Capability>,

    // === Performance & Evidence ===
    /// Performance benchmarks with historical data
    pub benchmarks: BenchmarkEvidence,
    /// Fitness scores across different dimensions
    pub fitness: FitnessProfile,

    // === Metadata ===
    /// Human-readable name
    pub name: String,
    /// Tags for categorization and discovery
    pub tags: Vec<String>,
    /// Custom metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

impl AgentDNA {
    /// Create a new AgentDNA with the given name and version
    pub fn new(name: impl Into<String>, version: impl Into<String>) -> Self {
        let id = DNAId::new();
        let version_str = version.into();
        let parts: Vec<&str> = version_str.split('.').collect();
        let version = if parts.len() >= 3 {
            DNAVersion::new(
                parts[0].parse().unwrap_or(1),
                parts[1].parse().unwrap_or(0),
                parts[2].parse().unwrap_or(0),
            )
        } else {
            DNAVersion::initial()
        };

        Self {
            id,
            version,
            lineage: DNALineage::new(id),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            skills: SkillRepertoire::new(),
            behavior: BehaviorGenome::default(),
            architecture: ArchitectureGenome::default(),
            capabilities: Vec::new(),
            benchmarks: BenchmarkEvidence::new(),
            fitness: FitnessProfile::new(),
            name: name.into(),
            tags: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Create a child DNA from this parent
    pub fn spawn_child(&self) -> Self {
        let child_id = DNAId::new();
        Self {
            id: child_id,
            version: self.version.increment_patch(),
            lineage: DNALineage::with_parent(self.id, self.lineage.root_ancestor_id),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            skills: self.skills.clone(),
            behavior: self.behavior.clone(),
            architecture: self.architecture.clone(),
            capabilities: self.capabilities.clone(),
            benchmarks: BenchmarkEvidence::new(), // Fresh benchmarks for child
            fitness: self.fitness.clone(),
            name: self.name.clone(),
            tags: self.tags.clone(),
            metadata: self.metadata.clone(),
        }
    }

    /// Add a skill to the DNA
    pub fn add_skill(&mut self, skill: Skill) {
        self.skills.add_skill(skill.clone());
        self.record_modification(Modification::new(
            ModificationType::SkillAdded(skill.id),
            format!("Added skill: {}", skill.name),
            ModificationSource::Manual {
                operator: "system".to_string(),
            },
        ));
        self.updated_at = Utc::now();
    }

    /// Upgrade a skill's proficiency level
    pub fn upgrade_skill(&mut self, skill_id: &str, new_level: u32) -> bool {
        // Get the skill info first, then release the borrow before modifying lineage
        let skill_info = if let Some(skill) = self.skills.get_skill_mut(skill_id) {
            let old_level = skill.proficiency_level;
            skill.proficiency_level = new_level.clamp(1, 10);
            Some((old_level, skill.proficiency_level))
        } else {
            None
        };

        if let Some((old_level, new_proficiency)) = skill_info {
            self.record_modification(Modification::new(
                ModificationType::SkillUpgraded {
                    skill_id: skill_id.to_string(),
                    from_level: old_level,
                    to_level: new_proficiency,
                },
                format!(
                    "Upgraded skill {} from level {} to {}",
                    skill_id, old_level, new_proficiency
                ),
                ModificationSource::Learning {
                    episode_id: uuid::Uuid::new_v4().to_string(),
                },
            ));
            self.version = self.version.increment_patch();
            self.updated_at = Utc::now();
            true
        } else {
            false
        }
    }

    /// Add a capability
    pub fn add_capability(&mut self, capability: Capability) {
        self.capabilities.push(capability);
        self.updated_at = Utc::now();
    }

    /// Add a tag
    pub fn add_tag(&mut self, tag: impl Into<String>) {
        let tag = tag.into();
        if !self.tags.contains(&tag) {
            self.tags.push(tag);
        }
    }

    /// Record a modification in the lineage
    pub fn record_modification(&mut self, modification: Modification) {
        self.lineage.modification_chain.push(modification);
    }

    /// Adjust behavior and record the modification
    pub fn adjust_behavior(&mut self, parameter: &str, delta: f64, source: ModificationSource) {
        if self.behavior.adjust(parameter, delta) {
            self.record_modification(
                Modification::new(
                    ModificationType::BehaviorAdjusted {
                        parameter: parameter.to_string(),
                        delta,
                    },
                    format!("Adjusted {} by {}", parameter, delta),
                    source,
                )
                .with_performance_delta(delta),
            );
            self.version = self.version.increment_patch();
            self.updated_at = Utc::now();
        }
    }

    /// Check if this DNA can provide a specific capability
    pub fn has_capability(&self, capability_id: &str) -> bool {
        self.capabilities.iter().any(|c| c.id == capability_id)
    }

    /// Get total skill proficiency
    pub fn total_proficiency(&self) -> f64 {
        self.skills.total_proficiency()
    }

    /// Validate the DNA structure
    pub fn validate(&self) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();

        if self.name.is_empty() {
            errors.push("DNA name cannot be empty".to_string());
        }

        if self.skills.skill_count() == 0 {
            errors.push("DNA must have at least one skill".to_string());
        }

        // Validate skill prerequisites
        for (skill_id, skill) in &self.skills.skills {
            for prereq in &skill.prerequisites {
                if !self.skills.skills.contains_key(prereq) {
                    errors.push(format!(
                        "Skill {} has missing prerequisite: {}",
                        skill_id, prereq
                    ));
                }
            }
        }

        // Validate capability requirements
        for cap in &self.capabilities {
            for skill_id in &cap.required_skills {
                if !self.skills.skills.contains_key(skill_id) {
                    errors.push(format!(
                        "Capability {} requires missing skill: {}",
                        cap.id, skill_id
                    ));
                }
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

// === Seed DNA Templates ===

impl AgentDNA {
    /// Create a seed DNA for efficiency hunting
    pub fn efficiency_hunter_template() -> Self {
        let mut dna = Self::new("efficiency-hunter", "1.0.0");
        dna.add_tag("efficiency");
        dna.add_tag("optimization");

        // Add core skill
        let skill = Skill::new(
            "detect_idle_resources",
            "Detect Idle Resources",
            SkillCategory::EfficiencyHunting,
            SkillExecution::Builtin {
                handler_id: "efficiency_hunter".to_string(),
            },
        )
        .with_description("Detect and report idle or underutilized resources")
        .with_proficiency(5);
        dna.skills.add_skill(skill);

        let skill2 = Skill::new(
            "calculate_savings",
            "Calculate Savings",
            SkillCategory::EfficiencyHunting,
            SkillExecution::Builtin {
                handler_id: "efficiency_hunter".to_string(),
            },
        )
        .with_description("Calculate potential savings from resource optimization")
        .with_proficiency(5);
        dna.skills.add_skill(skill2);

        // Add capability
        dna.add_capability(
            Capability::new("efficiency_analysis", "Efficiency Analysis")
                .with_description(
                    "Analyze system efficiency and identify optimization opportunities",
                )
                .with_required_skills(vec![
                    "detect_idle_resources".to_string(),
                    "calculate_savings".to_string(),
                ])
                .with_quality_score(0.8)
                .with_tags(vec!["efficiency".to_string(), "analysis".to_string()]),
        );

        dna
    }

    /// Create a seed DNA for cost optimization
    pub fn cost_optimizer_template() -> Self {
        let mut dna = Self::new("cost-optimizer", "1.0.0");
        dna.add_tag("cost");
        dna.add_tag("optimization");

        let skill = Skill::new(
            "analyze_spend",
            "Analyze Spend",
            SkillCategory::CostOptimization,
            SkillExecution::Builtin {
                handler_id: "cost_optimizer".to_string(),
            },
        )
        .with_description("Analyze spending patterns and identify cost savings opportunities")
        .with_proficiency(5);
        dna.skills.add_skill(skill);

        dna.add_capability(
            Capability::new("cost_analysis", "Cost Analysis")
                .with_description("Analyze costs and provide optimization recommendations")
                .with_required_skills(vec!["analyze_spend".to_string()])
                .with_quality_score(0.8),
        );

        dna
    }

    /// Create a seed DNA for capacity planning
    pub fn capacity_planner_template() -> Self {
        let mut dna = Self::new("capacity-planner", "1.0.0");
        dna.add_tag("capacity");
        dna.add_tag("planning");

        let skill = Skill::new(
            "forecast_demand",
            "Forecast Demand",
            SkillCategory::CapacityPlanning,
            SkillExecution::Builtin {
                handler_id: "capacity_planner".to_string(),
            },
        )
        .with_description("Forecast future capacity demand based on historical patterns")
        .with_proficiency(5);
        dna.skills.add_skill(skill);

        dna.add_capability(
            Capability::new("capacity_planning", "Capacity Planning")
                .with_description("Plan and forecast capacity needs")
                .with_required_skills(vec!["forecast_demand".to_string()])
                .with_quality_score(0.8),
        );

        dna
    }

    /// Create a seed DNA for scheduler optimization
    pub fn scheduler_optimizer_template() -> Self {
        let mut dna = Self::new("scheduler-optimizer", "1.0.0");
        dna.add_tag("scheduler");
        dna.add_tag("optimization");

        let skill = Skill::new(
            "optimize_placement",
            "Optimize Placement",
            SkillCategory::SchedulerOptimization,
            SkillExecution::Builtin {
                handler_id: "scheduler_optimizer".to_string(),
            },
        )
        .with_description("Optimize job placement for resource efficiency")
        .with_proficiency(5);
        dna.skills.add_skill(skill);

        dna.add_capability(
            Capability::new("scheduler_optimization", "Scheduler Optimization")
                .with_description("Optimize job scheduling and placement")
                .with_required_skills(vec!["optimize_placement".to_string()])
                .with_quality_score(0.8),
        );

        dna
    }

    /// Create a seed DNA for orchestration
    pub fn orchestrator_template() -> Self {
        let mut dna = Self::new("orchestrator", "1.0.0");
        dna.add_tag("orchestration");
        dna.add_tag("coordination");

        let skill = Skill::new(
            "classify_intent",
            "Classify Intent",
            SkillCategory::Orchestration,
            SkillExecution::Builtin {
                handler_id: "orchestrator".to_string(),
            },
        )
        .with_description("Classify user intent and route to appropriate agent")
        .with_proficiency(5);
        dna.skills.add_skill(skill);

        let skill2 = Skill::new(
            "coordinate_agents",
            "Coordinate Agents",
            SkillCategory::Orchestration,
            SkillExecution::Builtin {
                handler_id: "orchestrator".to_string(),
            },
        )
        .with_description("Coordinate multiple agents to achieve complex goals")
        .with_proficiency(5);
        dna.skills.add_skill(skill2);

        dna.add_capability(
            Capability::new("orchestration", "Orchestration")
                .with_description("Orchestrate agent activities and coordinate responses")
                .with_required_skills(vec![
                    "classify_intent".to_string(),
                    "coordinate_agents".to_string(),
                ])
                .with_quality_score(0.9),
        );

        dna
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dna_creation() {
        let dna = AgentDNA::new("test-agent", "1.0.0");
        assert_eq!(dna.name, "test-agent");
        assert_eq!(dna.version.major, 1);
        assert_eq!(dna.version.minor, 0);
        assert_eq!(dna.version.patch, 0);
    }

    #[test]
    fn test_spawn_child() {
        let parent = AgentDNA::new("parent", "1.0.0");
        let child = parent.spawn_child();

        assert_ne!(parent.id, child.id);
        assert_eq!(child.lineage.parent_id, Some(parent.id));
        assert_eq!(child.version.patch, 1);
    }

    #[test]
    fn test_efficiency_hunter_template() {
        let dna = AgentDNA::efficiency_hunter_template();
        assert_eq!(dna.name, "efficiency-hunter");
        assert!(dna.skills.skill_count() >= 2);
        assert!(dna.has_capability("efficiency_analysis"));
    }

    #[test]
    fn test_skill_upgrade() {
        let mut dna = AgentDNA::efficiency_hunter_template();
        let old_version = dna.version.clone();

        assert!(dna.upgrade_skill("detect_idle_resources", 7));
        assert_eq!(
            dna.skills
                .get_skill("detect_idle_resources")
                .unwrap()
                .proficiency_level,
            7
        );
        assert!(dna.version.is_newer_than(&old_version));
    }

    #[test]
    fn test_validation() {
        let dna = AgentDNA::efficiency_hunter_template();
        assert!(dna.validate().is_ok());

        // Empty DNA should fail validation
        let empty_dna = AgentDNA::new("empty", "1.0.0");
        assert!(empty_dna.validate().is_err());
    }
}
