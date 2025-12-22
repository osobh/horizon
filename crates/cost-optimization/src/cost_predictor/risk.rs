//! Risk assessment for cost prediction

use serde::{Deserialize, Serialize};

/// Risk level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RiskLevel {
    /// Low risk
    Low,
    /// Medium risk
    Medium,
    /// High risk
    High,
    /// Critical risk
    Critical,
}

/// Risk factor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskFactor {
    /// Factor name
    pub name: String,
    /// Impact score (0-100)
    pub impact: f64,
    /// Likelihood (0-1)
    pub likelihood: f64,
    /// Mitigation strategy
    pub mitigation: String,
}
