//! Cost optimization recommendations

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Cost optimization recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostRecommendation {
    /// Recommendation ID
    pub id: Uuid,
    /// Title
    pub title: String,
    /// Description
    pub description: String,
    /// Potential savings
    pub potential_savings: f64,
    /// Implementation complexity
    pub complexity: ComplexityLevel,
    /// Priority
    pub priority: Priority,
}

/// Complexity level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComplexityLevel {
    /// Low complexity
    Low,
    /// Medium complexity
    Medium,
    /// High complexity
    High,
}

/// Priority level
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Priority {
    /// Low priority
    Low = 0,
    /// Medium priority
    Medium = 1,
    /// High priority
    High = 2,
    /// Critical priority
    Critical = 3,
}
