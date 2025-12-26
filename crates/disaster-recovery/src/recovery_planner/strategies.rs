//! Recovery strategy types and implementations

use serde::{Deserialize, Serialize};

/// Recovery strategy type
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RecoveryStrategy {
    /// Active-Active deployment
    ActiveActive,
    /// Active-Passive with hot standby
    ActivePassiveHot,
    /// Active-Passive with warm standby
    ActivePassiveWarm,
    /// Active-Passive with cold standby
    ActivePassiveCold,
    /// Backup and restore
    BackupRestore,
    /// Pilot light
    PilotLight,
    /// Multi-site active
    MultiSiteActive,
}

impl RecoveryStrategy {
    /// Get typical RTO for this strategy in minutes
    pub fn typical_rto_minutes(&self) -> u64 {
        match self {
            RecoveryStrategy::ActiveActive => 0,           // Near-zero downtime
            RecoveryStrategy::ActivePassiveHot => 5,       // 5 minutes
            RecoveryStrategy::ActivePassiveWarm => 15,     // 15 minutes
            RecoveryStrategy::ActivePassiveCold => 60,     // 1 hour
            RecoveryStrategy::BackupRestore => 480,        // 8 hours
            RecoveryStrategy::PilotLight => 120,           // 2 hours
            RecoveryStrategy::MultiSiteActive => 2,        // 2 minutes
        }
    }

    /// Get typical cost factor (1.0 = baseline)
    pub fn cost_factor(&self) -> f64 {
        match self {
            RecoveryStrategy::ActiveActive => 2.0,         // 2x cost
            RecoveryStrategy::ActivePassiveHot => 1.8,     // 1.8x cost
            RecoveryStrategy::ActivePassiveWarm => 1.5,    // 1.5x cost
            RecoveryStrategy::ActivePassiveCold => 1.2,    // 1.2x cost
            RecoveryStrategy::BackupRestore => 1.0,        // Baseline
            RecoveryStrategy::PilotLight => 1.1,           // 1.1x cost
            RecoveryStrategy::MultiSiteActive => 2.5,      // 2.5x cost
        }
    }

    /// Get typical complexity score (1-5, 5 being most complex)
    pub fn complexity_score(&self) -> u8 {
        match self {
            RecoveryStrategy::ActiveActive => 4,
            RecoveryStrategy::ActivePassiveHot => 3,
            RecoveryStrategy::ActivePassiveWarm => 2,
            RecoveryStrategy::ActivePassiveCold => 2,
            RecoveryStrategy::BackupRestore => 1,
            RecoveryStrategy::PilotLight => 2,
            RecoveryStrategy::MultiSiteActive => 5,
        }
    }

    /// Check if strategy supports the given RTO requirement
    pub fn supports_rto(&self, required_rto_minutes: u64) -> bool {
        self.typical_rto_minutes() <= required_rto_minutes
    }

    /// Recommend strategy based on RTO/RPO requirements
    pub fn recommend_for_objectives(
        rto_minutes: u64,
        rpo_minutes: u64,
        budget_factor: f64, // 1.0 = normal budget, higher = more budget available
    ) -> Vec<Self> {
        let mut recommendations = Vec::new();

        // Add strategies that can meet RTO requirements
        let all_strategies = [
            RecoveryStrategy::ActiveActive,
            RecoveryStrategy::ActivePassiveHot,
            RecoveryStrategy::ActivePassiveWarm,
            RecoveryStrategy::ActivePassiveCold,
            RecoveryStrategy::BackupRestore,
            RecoveryStrategy::PilotLight,
            RecoveryStrategy::MultiSiteActive,
        ];

        for strategy in &all_strategies {
            if strategy.supports_rto(rto_minutes) && strategy.cost_factor() <= budget_factor {
                recommendations.push(strategy.clone());
            }
        }

        // Sort by cost efficiency (lower cost factor first)
        recommendations.sort_by(|a, b| {
            a.cost_factor().partial_cmp(&b.cost_factor()).unwrap_or(std::cmp::Ordering::Equal)
        });

        recommendations
    }
}
