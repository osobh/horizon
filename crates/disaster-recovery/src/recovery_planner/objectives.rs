//! Recovery Time and Point Objectives (RTO/RPO)

use serde::{Deserialize, Serialize};

/// Recovery objective definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryObjective {
    /// Recovery Time Objective (minutes)
    pub rto_minutes: u64,
    /// Recovery Point Objective (minutes)
    pub rpo_minutes: u64,
    /// Maximum tolerable downtime
    pub mto_minutes: u64,
    /// Maximum data loss tolerance
    pub mdo_minutes: u64,
    /// Service level agreement percentage
    pub sla_percentage: f64,
}

impl RecoveryObjective {
    /// Create new recovery objective
    pub fn new(
        rto_minutes: u64,
        rpo_minutes: u64,
        mto_minutes: u64,
        mdo_minutes: u64,
        sla_percentage: f64,
    ) -> Self {
        Self {
            rto_minutes,
            rpo_minutes,
            mto_minutes,
            mdo_minutes,
            sla_percentage,
        }
    }

    /// Check if objective is achievable with current resources
    pub fn is_achievable(&self, available_resources: &super::resources::ResourceCapacity) -> bool {
        // Basic feasibility check based on resource availability
        available_resources.cpu_cores >= 1.0 
            && available_resources.memory_gb >= 1.0
            && available_resources.storage_gb >= 1.0
    }

    /// Calculate objective compliance score (0.0 to 1.0)
    pub fn compliance_score(&self, actual_rto: u64, actual_rpo: u64) -> f64 {
        let rto_compliance = if actual_rto <= self.rto_minutes {
            1.0
        } else {
            self.rto_minutes as f64 / actual_rto as f64
        };

        let rpo_compliance = if actual_rpo <= self.rpo_minutes {
            1.0
        } else {
            self.rpo_minutes as f64 / actual_rpo as f64
        };

        (rto_compliance + rpo_compliance) / 2.0
    }
}
