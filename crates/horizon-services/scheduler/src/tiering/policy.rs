use serde::{Deserialize, Serialize};

/// Node tier for heterogeneous infrastructure
/// Tier0 (servers) → Tier1 (desktops) → Tier2 (laptops) → Tier3 (Raspberry Pi/experimental)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum NodeTier {
    Tier0 = 0,
    Tier1 = 1,
    Tier2 = 2,
    Tier3 = 3,
}

impl NodeTier {
    /// Get reliability SLA for this tier (0.0-1.0)
    pub fn reliability_sla(&self) -> f64 {
        match self {
            NodeTier::Tier0 => 0.95, // 95% uptime guarantee
            NodeTier::Tier1 => 0.80, // 80% uptime
            NodeTier::Tier2 => 0.55, // 55% uptime
            NodeTier::Tier3 => 0.30, // 30% uptime
        }
    }

    /// Get placement priority (higher is better, 0-100)
    pub fn placement_priority(&self) -> u8 {
        match self {
            NodeTier::Tier0 => 100,
            NodeTier::Tier1 => 70,
            NodeTier::Tier2 => 40,
            NodeTier::Tier3 => 10,
        }
    }

    /// Get cost multiplier (Tier0 is baseline 1.0)
    pub fn cost_multiplier(&self) -> f64 {
        match self {
            NodeTier::Tier0 => 1.0,   // Full cost (servers)
            NodeTier::Tier1 => 0.7,   // 30% discount (desktops)
            NodeTier::Tier2 => 0.4,   // 60% discount (laptops)
            NodeTier::Tier3 => 0.1,   // 90% discount (Raspberry Pi/experimental)
        }
    }

    /// Get tier name
    pub fn name(&self) -> &'static str {
        match self {
            NodeTier::Tier0 => "Tier0-Server",
            NodeTier::Tier1 => "Tier1-Desktop",
            NodeTier::Tier2 => "Tier2-Laptop",
            NodeTier::Tier3 => "Tier3-Experimental",
        }
    }

    /// Assign tier based on reliability score (0.0-1.0)
    pub fn from_reliability_score(score: f64) -> Self {
        match score {
            s if s >= 0.90 => NodeTier::Tier0,
            s if s >= 0.70 => NodeTier::Tier1,
            s if s >= 0.40 => NodeTier::Tier2,
            _ => NodeTier::Tier3,
        }
    }
}

/// Job criticality level
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum JobCriticality {
    Production = 3,
    Staging = 2,
    Development = 1,
    Experimental = 0,
}

impl JobCriticality {
    /// Get minimum required tier for this criticality level
    pub fn minimum_tier(&self) -> NodeTier {
        match self {
            JobCriticality::Production => NodeTier::Tier0,
            JobCriticality::Staging => NodeTier::Tier1,
            JobCriticality::Development => NodeTier::Tier2,
            JobCriticality::Experimental => NodeTier::Tier3,
        }
    }

    /// Check if job can run on given tier
    pub fn can_run_on_tier(&self, tier: NodeTier) -> bool {
        tier <= self.minimum_tier()
    }
}

/// Tiering policy for job placement
#[derive(Debug, Clone)]
pub struct TieringPolicy {
    /// Allow lower-criticality jobs on higher tiers
    pub allow_upward_placement: bool,
    /// Penalize cost for using higher tier than necessary
    pub cost_penalty_factor: f64,
}

impl TieringPolicy {
    /// Create a new tiering policy
    pub fn new() -> Self {
        Self {
            allow_upward_placement: true,
            cost_penalty_factor: 1.2, // 20% cost penalty for over-provisioning
        }
    }

    /// Create a strict policy (no upward placement)
    pub fn strict() -> Self {
        Self {
            allow_upward_placement: false,
            cost_penalty_factor: 1.5,
        }
    }

    /// Check if job can be placed on node
    pub fn can_place_job(&self, job_criticality: JobCriticality, node_tier: NodeTier) -> bool {
        let min_tier = job_criticality.minimum_tier();

        if node_tier == min_tier {
            // Exact match always allowed
            true
        } else if node_tier < min_tier {
            // Higher tier than needed
            self.allow_upward_placement
        } else {
            // Lower tier than needed - never allowed
            false
        }
    }

    /// Calculate effective cost with penalties
    pub fn calculate_effective_cost(
        &self,
        job_criticality: JobCriticality,
        node_tier: NodeTier,
        base_cost: f64,
    ) -> f64 {
        let min_tier = job_criticality.minimum_tier();
        let tier_cost = base_cost * node_tier.cost_multiplier();

        if node_tier < min_tier {
            // Using higher tier than needed - apply penalty
            tier_cost * self.cost_penalty_factor
        } else {
            tier_cost
        }
    }

    /// Get placement score (higher is better, 0.0-1.0)
    pub fn placement_score(&self, job_criticality: JobCriticality, node_tier: NodeTier) -> f64 {
        if !self.can_place_job(job_criticality, node_tier) {
            return 0.0;
        }

        let min_tier = job_criticality.minimum_tier();

        if node_tier == min_tier {
            // Perfect match
            1.0
        } else if node_tier < min_tier {
            // Over-provisioned (using higher tier)
            0.7
        } else {
            // Should never happen (can_place_job would return false)
            0.0
        }
    }
}

impl Default for TieringPolicy {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_tier_reliability_sla() {
        assert_eq!(NodeTier::Tier0.reliability_sla(), 0.95);
        assert_eq!(NodeTier::Tier1.reliability_sla(), 0.80);
        assert_eq!(NodeTier::Tier2.reliability_sla(), 0.55);
        assert_eq!(NodeTier::Tier3.reliability_sla(), 0.30);
    }

    #[test]
    fn test_node_tier_placement_priority() {
        assert_eq!(NodeTier::Tier0.placement_priority(), 100);
        assert_eq!(NodeTier::Tier1.placement_priority(), 70);
        assert_eq!(NodeTier::Tier2.placement_priority(), 40);
        assert_eq!(NodeTier::Tier3.placement_priority(), 10);
    }

    #[test]
    fn test_node_tier_cost_multiplier() {
        assert_eq!(NodeTier::Tier0.cost_multiplier(), 1.0);
        assert_eq!(NodeTier::Tier1.cost_multiplier(), 0.7);
        assert_eq!(NodeTier::Tier2.cost_multiplier(), 0.4);
        assert_eq!(NodeTier::Tier3.cost_multiplier(), 0.1);
    }

    #[test]
    fn test_node_tier_name() {
        assert_eq!(NodeTier::Tier0.name(), "Tier0-Server");
        assert_eq!(NodeTier::Tier1.name(), "Tier1-Desktop");
        assert_eq!(NodeTier::Tier2.name(), "Tier2-Laptop");
        assert_eq!(NodeTier::Tier3.name(), "Tier3-Experimental");
    }

    #[test]
    fn test_from_reliability_score_tier0() {
        assert_eq!(NodeTier::from_reliability_score(0.95), NodeTier::Tier0);
        assert_eq!(NodeTier::from_reliability_score(1.0), NodeTier::Tier0);
    }

    #[test]
    fn test_from_reliability_score_tier1() {
        assert_eq!(NodeTier::from_reliability_score(0.80), NodeTier::Tier1);
        assert_eq!(NodeTier::from_reliability_score(0.75), NodeTier::Tier1);
    }

    #[test]
    fn test_from_reliability_score_tier2() {
        assert_eq!(NodeTier::from_reliability_score(0.50), NodeTier::Tier2);
        assert_eq!(NodeTier::from_reliability_score(0.45), NodeTier::Tier2);
    }

    #[test]
    fn test_from_reliability_score_tier3() {
        assert_eq!(NodeTier::from_reliability_score(0.30), NodeTier::Tier3);
        assert_eq!(NodeTier::from_reliability_score(0.10), NodeTier::Tier3);
    }

    #[test]
    fn test_job_criticality_minimum_tier() {
        assert_eq!(JobCriticality::Production.minimum_tier(), NodeTier::Tier0);
        assert_eq!(JobCriticality::Staging.minimum_tier(), NodeTier::Tier1);
        assert_eq!(JobCriticality::Development.minimum_tier(), NodeTier::Tier2);
        assert_eq!(JobCriticality::Experimental.minimum_tier(), NodeTier::Tier3);
    }

    #[test]
    fn test_job_can_run_on_tier() {
        // Production job
        assert!(JobCriticality::Production.can_run_on_tier(NodeTier::Tier0));
        assert!(!JobCriticality::Production.can_run_on_tier(NodeTier::Tier1));

        // Experimental job can run on any tier
        assert!(JobCriticality::Experimental.can_run_on_tier(NodeTier::Tier0));
        assert!(JobCriticality::Experimental.can_run_on_tier(NodeTier::Tier3));
    }

    #[test]
    fn test_tiering_policy_creation() {
        let policy = TieringPolicy::new();
        assert!(policy.allow_upward_placement);
        assert_eq!(policy.cost_penalty_factor, 1.2);
    }

    #[test]
    fn test_tiering_policy_strict() {
        let policy = TieringPolicy::strict();
        assert!(!policy.allow_upward_placement);
        assert_eq!(policy.cost_penalty_factor, 1.5);
    }

    #[test]
    fn test_can_place_job_exact_match() {
        let policy = TieringPolicy::new();

        // Production on Tier0 (exact match)
        assert!(policy.can_place_job(JobCriticality::Production, NodeTier::Tier0));
    }

    #[test]
    fn test_can_place_job_upward_placement() {
        let policy = TieringPolicy::new();

        // Development on Tier0 (higher tier than needed)
        assert!(policy.can_place_job(JobCriticality::Development, NodeTier::Tier0));
    }

    #[test]
    fn test_can_place_job_downward_placement_rejected() {
        let policy = TieringPolicy::new();

        // Production on Tier2 (lower tier than needed) - should fail
        assert!(!policy.can_place_job(JobCriticality::Production, NodeTier::Tier2));
    }

    #[test]
    fn test_strict_policy_no_upward_placement() {
        let policy = TieringPolicy::strict();

        // Development on Tier0 (higher tier) - should fail in strict mode
        assert!(!policy.can_place_job(JobCriticality::Development, NodeTier::Tier0));
    }

    #[test]
    fn test_calculate_effective_cost_exact_match() {
        let policy = TieringPolicy::new();

        let cost = policy.calculate_effective_cost(
            JobCriticality::Production,
            NodeTier::Tier0,
            100.0,
        );

        // Tier0 multiplier is 1.0, no penalty for exact match
        assert_eq!(cost, 100.0);
    }

    #[test]
    fn test_calculate_effective_cost_with_penalty() {
        let policy = TieringPolicy::new();

        let cost = policy.calculate_effective_cost(
            JobCriticality::Development, // Min tier is Tier2
            NodeTier::Tier0,             // Using Tier0 (over-provisioned)
            100.0,
        );

        // Tier0 multiplier (1.0) * penalty (1.2) = 120.0
        assert_eq!(cost, 120.0);
    }

    #[test]
    fn test_placement_score_perfect_match() {
        let policy = TieringPolicy::new();

        let score = policy.placement_score(JobCriticality::Production, NodeTier::Tier0);

        assert_eq!(score, 1.0);
    }

    #[test]
    fn test_placement_score_over_provisioned() {
        let policy = TieringPolicy::new();

        let score = policy.placement_score(JobCriticality::Development, NodeTier::Tier0);

        assert_eq!(score, 0.7);
    }

    #[test]
    fn test_placement_score_invalid() {
        let policy = TieringPolicy::new();

        let score = policy.placement_score(JobCriticality::Production, NodeTier::Tier2);

        assert_eq!(score, 0.0);
    }
}
