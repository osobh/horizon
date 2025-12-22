//! Core types for budget management

use serde::{Deserialize, Serialize};

/// Action types for automated responses
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActionType {
    /// Send notification
    Notify,
    /// Scale down resources
    ScaleDown,
    /// Pause non-critical workloads
    PauseWorkloads,
    /// Block new deployments
    BlockDeployments,
    /// Enforce spending limits
    EnforceLimit,
}

/// Spending trend direction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpendTrend {
    /// Increasing spend
    Increasing,
    /// Decreasing spend
    Decreasing,
    /// Stable spend
    Stable,
    /// Volatile spend
    Volatile,
}
