//! DNA versioning and lineage tracking

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Unique identifier for AgentDNA
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DNAId(pub Uuid);

impl DNAId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for DNAId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for DNAId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Semantic versioning for AgentDNA with evolution generation tracking
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DNAVersion {
    /// Major version - breaking changes to capabilities
    pub major: u32,
    /// Minor version - new skills or capabilities added
    pub minor: u32,
    /// Patch version - skill upgrades or behavior adjustments
    pub patch: u32,
    /// Evolution generation counter (monotonically increasing)
    pub generation: u64,
}

impl DNAVersion {
    pub fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
            generation: 0,
        }
    }

    pub fn initial() -> Self {
        Self::new(1, 0, 0)
    }

    /// Increment patch version (skill upgrades, behavior adjustments)
    pub fn increment_patch(&self) -> Self {
        Self {
            major: self.major,
            minor: self.minor,
            patch: self.patch + 1,
            generation: self.generation + 1,
        }
    }

    /// Increment minor version (new skills or capabilities)
    pub fn increment_minor(&self) -> Self {
        Self {
            major: self.major,
            minor: self.minor + 1,
            patch: 0,
            generation: self.generation + 1,
        }
    }

    /// Increment major version (breaking changes)
    pub fn increment_major(&self) -> Self {
        Self {
            major: self.major + 1,
            minor: 0,
            patch: 0,
            generation: self.generation + 1,
        }
    }

    /// Check if this version is compatible with another (same major version)
    pub fn is_compatible_with(&self, other: &DNAVersion) -> bool {
        self.major == other.major
    }

    /// Check if this version is newer than another
    pub fn is_newer_than(&self, other: &DNAVersion) -> bool {
        self.generation > other.generation
    }
}

impl Default for DNAVersion {
    fn default() -> Self {
        Self::initial()
    }
}

impl std::fmt::Display for DNAVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}.{}.{}-gen{}",
            self.major, self.minor, self.patch, self.generation
        )
    }
}

/// Lineage information tracking DNA evolution history
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DNALineage {
    /// Parent DNA ID (None for original/seed DNA)
    pub parent_id: Option<DNAId>,
    /// Root ancestor DNA ID
    pub root_ancestor_id: DNAId,
    /// Chain of modifications from root to this DNA
    pub modification_chain: Vec<Modification>,
    /// Branching point versions (for tracking divergence)
    pub branch_points: Vec<(DNAId, DNAVersion)>,
}

impl DNALineage {
    pub fn new(root_id: DNAId) -> Self {
        Self {
            parent_id: None,
            root_ancestor_id: root_id,
            modification_chain: Vec::new(),
            branch_points: Vec::new(),
        }
    }

    pub fn with_parent(parent_id: DNAId, root_ancestor_id: DNAId) -> Self {
        Self {
            parent_id: Some(parent_id),
            root_ancestor_id,
            modification_chain: Vec::new(),
            branch_points: Vec::new(),
        }
    }

    /// Get the number of modifications in the lineage
    pub fn modification_count(&self) -> usize {
        self.modification_chain.len()
    }

    /// Get the most recent modification
    pub fn latest_modification(&self) -> Option<&Modification> {
        self.modification_chain.last()
    }
}

/// A modification record in the DNA lineage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Modification {
    /// Unique modification ID
    pub id: Uuid,
    /// When the modification occurred
    pub timestamp: DateTime<Utc>,
    /// Type of modification
    pub modification_type: ModificationType,
    /// Human-readable description
    pub description: String,
    /// Performance delta from this modification (if measurable)
    pub performance_delta: Option<f64>,
    /// Source of the modification
    pub source: ModificationSource,
}

impl Modification {
    pub fn new(
        modification_type: ModificationType,
        description: impl Into<String>,
        source: ModificationSource,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            timestamp: Utc::now(),
            modification_type,
            description: description.into(),
            performance_delta: None,
            source,
        }
    }

    pub fn with_performance_delta(mut self, delta: f64) -> Self {
        self.performance_delta = Some(delta);
        self
    }
}

/// Types of modifications that can be made to DNA
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModificationType {
    /// A new skill was added
    SkillAdded(String),
    /// An existing skill was upgraded
    SkillUpgraded {
        skill_id: String,
        from_level: u32,
        to_level: u32,
    },
    /// A skill was removed
    SkillRemoved(String),
    /// Behavior parameter was adjusted
    BehaviorAdjusted { parameter: String, delta: f64 },
    /// Architecture configuration changed
    ArchitectureChanged(String),
    /// A learned pattern was applied
    PatternApplied(String),
    /// Random mutation occurred (during evolution)
    Mutation { mutation_rate: f64 },
    /// Crossover with another DNA
    Crossover { other_parent_id: DNAId },
    /// Rollback to previous version
    Rollback { target_version: DNAVersion },
}

/// Source of a DNA modification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModificationSource {
    /// From reconciliation process
    Reconciliation { context: String },
    /// From postprocessing operation
    Postprocessing { operation_id: String },
    /// From evolution process
    Evolution { generation: u64 },
    /// Manual modification by operator
    Manual { operator: String },
    /// From learning episode
    Learning { episode_id: String },
    /// From external feedback
    ExternalFeedback { source: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_increment() {
        let v = DNAVersion::initial();
        assert_eq!(v.major, 1);
        assert_eq!(v.minor, 0);
        assert_eq!(v.patch, 0);
        assert_eq!(v.generation, 0);

        let v2 = v.increment_patch();
        assert_eq!(v2.patch, 1);
        assert_eq!(v2.generation, 1);

        let v3 = v2.increment_minor();
        assert_eq!(v3.minor, 1);
        assert_eq!(v3.patch, 0);
        assert_eq!(v3.generation, 2);

        let v4 = v3.increment_major();
        assert_eq!(v4.major, 2);
        assert_eq!(v4.minor, 0);
        assert_eq!(v4.patch, 0);
        assert_eq!(v4.generation, 3);
    }

    #[test]
    fn test_version_compatibility() {
        let v1 = DNAVersion::new(1, 2, 3);
        let v2 = DNAVersion::new(1, 5, 0);
        let v3 = DNAVersion::new(2, 0, 0);

        assert!(v1.is_compatible_with(&v2));
        assert!(!v1.is_compatible_with(&v3));
    }

    #[test]
    fn test_lineage_creation() {
        let root_id = DNAId::new();
        let lineage = DNALineage::new(root_id);

        assert!(lineage.parent_id.is_none());
        assert_eq!(lineage.root_ancestor_id, root_id);
        assert!(lineage.modification_chain.is_empty());
    }
}
