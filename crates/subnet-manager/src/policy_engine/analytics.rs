//! Policy analytics and diagnostics
//!
//! Provides metrics, statistics, and diagnostic information about policy evaluation.

use crate::models::AssignmentPolicy;
use chrono::{DateTime, Utc};
use std::collections::HashMap;
use uuid::Uuid;

/// Policy evaluation statistics
#[derive(Debug, Clone, Default)]
pub struct PolicyStats {
    /// Total number of evaluations
    pub total_evaluations: u64,
    /// Number of successful matches
    pub successful_matches: u64,
    /// Number of failed matches (no policy matched)
    pub failed_matches: u64,
    /// Evaluations per policy
    pub policy_evaluations: HashMap<Uuid, PolicyEvaluationStats>,
    /// Average evaluation time in microseconds
    pub avg_evaluation_time_us: f64,
    /// Maximum evaluation time in microseconds
    pub max_evaluation_time_us: u64,
    /// Last evaluation timestamp
    pub last_evaluation: Option<DateTime<Utc>>,
}

/// Statistics for a single policy
#[derive(Debug, Clone, Default)]
pub struct PolicyEvaluationStats {
    /// Policy ID
    pub policy_id: Uuid,
    /// Policy name
    pub policy_name: String,
    /// Number of times this policy was evaluated
    pub evaluation_count: u64,
    /// Number of times this policy matched
    pub match_count: u64,
    /// Match rate (matches / evaluations)
    pub match_rate: f64,
    /// Last match timestamp
    pub last_match: Option<DateTime<Utc>>,
    /// Number of nodes assigned via this policy
    pub nodes_assigned: u64,
}

/// Policy analyzer for generating insights
pub struct PolicyAnalyzer {
    /// Collected statistics
    stats: PolicyStats,
    /// Evaluation history (ring buffer)
    history: Vec<EvaluationRecord>,
    /// Maximum history size
    max_history: usize,
}

/// Record of a single evaluation
#[derive(Debug, Clone)]
pub struct EvaluationRecord {
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Node ID being evaluated
    pub node_id: Option<Uuid>,
    /// Matched policy ID (if any)
    pub matched_policy_id: Option<Uuid>,
    /// Evaluation duration in microseconds
    pub duration_us: u64,
    /// Number of policies evaluated
    pub policies_evaluated: usize,
    /// Was the evaluation successful
    pub success: bool,
}

impl PolicyAnalyzer {
    /// Create a new analyzer
    pub fn new() -> Self {
        Self::with_history_size(1000)
    }

    /// Create with custom history size
    pub fn with_history_size(max_history: usize) -> Self {
        Self {
            stats: PolicyStats::default(),
            history: Vec::with_capacity(max_history),
            max_history,
        }
    }

    /// Record an evaluation result
    pub fn record_evaluation(
        &mut self,
        node_id: Option<Uuid>,
        matched_policy_id: Option<Uuid>,
        duration_us: u64,
        policies_evaluated: usize,
    ) {
        let now = Utc::now();
        let success = matched_policy_id.is_some();

        // Update overall stats
        self.stats.total_evaluations += 1;
        if success {
            self.stats.successful_matches += 1;
        } else {
            self.stats.failed_matches += 1;
        }

        // Update average evaluation time
        let n = self.stats.total_evaluations as f64;
        self.stats.avg_evaluation_time_us =
            (self.stats.avg_evaluation_time_us * (n - 1.0) + duration_us as f64) / n;

        // Update max evaluation time
        if duration_us > self.stats.max_evaluation_time_us {
            self.stats.max_evaluation_time_us = duration_us;
        }

        self.stats.last_evaluation = Some(now);

        // Update per-policy stats if matched
        if let Some(policy_id) = matched_policy_id {
            let policy_stats = self
                .stats
                .policy_evaluations
                .entry(policy_id)
                .or_insert_with(|| PolicyEvaluationStats {
                    policy_id,
                    ..Default::default()
                });
            policy_stats.evaluation_count += 1;
            policy_stats.match_count += 1;
            policy_stats.match_rate =
                policy_stats.match_count as f64 / policy_stats.evaluation_count as f64;
            policy_stats.last_match = Some(now);
            policy_stats.nodes_assigned += 1;
        }

        // Add to history
        let record = EvaluationRecord {
            timestamp: now,
            node_id,
            matched_policy_id,
            duration_us,
            policies_evaluated,
            success,
        };

        if self.history.len() >= self.max_history {
            self.history.remove(0);
        }
        self.history.push(record);
    }

    /// Get current statistics
    pub fn stats(&self) -> &PolicyStats {
        &self.stats
    }

    /// Get evaluation history
    pub fn history(&self) -> &[EvaluationRecord] {
        &self.history
    }

    /// Get top policies by match count
    pub fn top_policies(&self, limit: usize) -> Vec<&PolicyEvaluationStats> {
        let mut policies: Vec<_> = self.stats.policy_evaluations.values().collect();
        policies.sort_by(|a, b| b.match_count.cmp(&a.match_count));
        policies.truncate(limit);
        policies
    }

    /// Get policies that haven't matched recently
    pub fn stale_policies(&self, threshold: chrono::Duration) -> Vec<Uuid> {
        let cutoff = Utc::now() - threshold;
        self.stats
            .policy_evaluations
            .iter()
            .filter(|(_, stats)| {
                stats.last_match.map(|t| t < cutoff).unwrap_or(true)
            })
            .map(|(&id, _)| id)
            .collect()
    }

    /// Get success rate
    pub fn success_rate(&self) -> f64 {
        if self.stats.total_evaluations == 0 {
            0.0
        } else {
            self.stats.successful_matches as f64 / self.stats.total_evaluations as f64
        }
    }

    /// Get recent evaluation trend (success rate over last N evaluations)
    pub fn recent_trend(&self, window: usize) -> f64 {
        if self.history.is_empty() {
            return 0.0;
        }

        let recent: Vec<_> = self.history.iter().rev().take(window).collect();
        if recent.is_empty() {
            return 0.0;
        }

        let successes = recent.iter().filter(|r| r.success).count();
        successes as f64 / recent.len() as f64
    }

    /// Reset statistics
    pub fn reset(&mut self) {
        self.stats = PolicyStats::default();
        self.history.clear();
    }

    /// Initialize policy stats from existing policies
    pub fn initialize_policies(&mut self, policies: &[AssignmentPolicy]) {
        for policy in policies {
            self.stats
                .policy_evaluations
                .entry(policy.id)
                .or_insert_with(|| PolicyEvaluationStats {
                    policy_id: policy.id,
                    policy_name: policy.name.clone(),
                    ..Default::default()
                });
        }
    }
}

impl Default for PolicyAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Policy health check results
#[derive(Debug, Clone)]
pub struct PolicyHealthCheck {
    /// Overall health status
    pub status: HealthStatus,
    /// Individual policy health
    pub policy_health: Vec<PolicyHealth>,
    /// Issues found
    pub issues: Vec<HealthIssue>,
    /// Recommendations
    pub recommendations: Vec<String>,
}

/// Overall health status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthStatus {
    /// All policies healthy
    Healthy,
    /// Some issues detected
    Warning,
    /// Critical issues detected
    Critical,
}

/// Health of a single policy
#[derive(Debug, Clone)]
pub struct PolicyHealth {
    /// Policy ID
    pub policy_id: Uuid,
    /// Policy name
    pub policy_name: String,
    /// Is policy active
    pub is_active: bool,
    /// Match rate
    pub match_rate: f64,
    /// Days since last match
    pub days_since_last_match: Option<i64>,
    /// Health status
    pub status: HealthStatus,
}

/// Health issue
#[derive(Debug, Clone)]
pub struct HealthIssue {
    /// Issue severity
    pub severity: HealthStatus,
    /// Affected policy ID (if applicable)
    pub policy_id: Option<Uuid>,
    /// Issue description
    pub description: String,
}

/// Generate health check for policies
pub fn health_check(
    policies: &[AssignmentPolicy],
    stats: &PolicyStats,
) -> PolicyHealthCheck {
    let mut issues = Vec::new();
    let mut policy_health = Vec::new();
    let mut has_warning = false;
    let mut has_critical = false;

    // Check each policy
    for policy in policies {
        let policy_stats = stats.policy_evaluations.get(&policy.id);

        let is_active = policy.is_active();
        let match_rate = policy_stats.map(|s| s.match_rate).unwrap_or(0.0);
        let days_since_last_match = policy_stats
            .and_then(|s| s.last_match)
            .map(|t| (Utc::now() - t).num_days());

        let mut status = HealthStatus::Healthy;

        // Check for inactive policy
        if !is_active {
            status = HealthStatus::Warning;
            has_warning = true;
            issues.push(HealthIssue {
                severity: HealthStatus::Warning,
                policy_id: Some(policy.id),
                description: format!("Policy '{}' is inactive", policy.name),
            });
        }

        // Check for stale policy (no matches in 30+ days)
        if let Some(days) = days_since_last_match {
            if days > 30 {
                status = HealthStatus::Warning;
                has_warning = true;
                issues.push(HealthIssue {
                    severity: HealthStatus::Warning,
                    policy_id: Some(policy.id),
                    description: format!(
                        "Policy '{}' hasn't matched in {} days",
                        policy.name, days
                    ),
                });
            }
        }

        // Check for never-matched policy
        if policy_stats.map(|s| s.match_count).unwrap_or(0) == 0 {
            issues.push(HealthIssue {
                severity: HealthStatus::Warning,
                policy_id: Some(policy.id),
                description: format!("Policy '{}' has never matched", policy.name),
            });
        }

        policy_health.push(PolicyHealth {
            policy_id: policy.id,
            policy_name: policy.name.clone(),
            is_active,
            match_rate,
            days_since_last_match,
            status,
        });
    }

    // Check overall metrics
    if stats.total_evaluations > 100 {
        let success_rate = stats.successful_matches as f64 / stats.total_evaluations as f64;
        if success_rate < 0.5 {
            has_warning = true;
            issues.push(HealthIssue {
                severity: HealthStatus::Warning,
                policy_id: None,
                description: format!(
                    "Low overall match rate: {:.1}%",
                    success_rate * 100.0
                ),
            });
        }
        if success_rate < 0.1 {
            has_critical = true;
            issues.push(HealthIssue {
                severity: HealthStatus::Critical,
                policy_id: None,
                description: format!(
                    "Critical: Very low match rate: {:.1}%",
                    success_rate * 100.0
                ),
            });
        }
    }

    // Generate recommendations
    let mut recommendations = Vec::new();
    if has_critical {
        recommendations.push("Review policy rules - most nodes are not matching any policy".to_string());
    }
    if issues.iter().any(|i| i.description.contains("never matched")) {
        recommendations.push("Consider removing or updating policies that have never matched".to_string());
    }
    if policies.iter().filter(|p| p.is_active()).count() == 0 {
        recommendations.push("No active policies - all node assignments will require manual intervention".to_string());
    }

    let status = if has_critical {
        HealthStatus::Critical
    } else if has_warning {
        HealthStatus::Warning
    } else {
        HealthStatus::Healthy
    };

    PolicyHealthCheck {
        status,
        policy_health,
        issues,
        recommendations,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{NodeType, PolicyRule};

    #[test]
    fn test_record_evaluation() {
        let mut analyzer = PolicyAnalyzer::new();
        let policy_id = Uuid::new_v4();
        let node_id = Uuid::new_v4();

        analyzer.record_evaluation(Some(node_id), Some(policy_id), 100, 5);

        assert_eq!(analyzer.stats().total_evaluations, 1);
        assert_eq!(analyzer.stats().successful_matches, 1);
        assert_eq!(analyzer.stats().failed_matches, 0);
        assert!(analyzer.stats().policy_evaluations.contains_key(&policy_id));
    }

    #[test]
    fn test_failed_evaluation() {
        let mut analyzer = PolicyAnalyzer::new();

        analyzer.record_evaluation(Some(Uuid::new_v4()), None, 50, 3);

        assert_eq!(analyzer.stats().total_evaluations, 1);
        assert_eq!(analyzer.stats().successful_matches, 0);
        assert_eq!(analyzer.stats().failed_matches, 1);
    }

    #[test]
    fn test_success_rate() {
        let mut analyzer = PolicyAnalyzer::new();
        let policy_id = Uuid::new_v4();

        // 3 successes, 2 failures
        analyzer.record_evaluation(None, Some(policy_id), 100, 1);
        analyzer.record_evaluation(None, Some(policy_id), 100, 1);
        analyzer.record_evaluation(None, Some(policy_id), 100, 1);
        analyzer.record_evaluation(None, None, 100, 1);
        analyzer.record_evaluation(None, None, 100, 1);

        assert!((analyzer.success_rate() - 0.6).abs() < 0.001);
    }

    #[test]
    fn test_top_policies() {
        let mut analyzer = PolicyAnalyzer::new();
        let policy_a = Uuid::new_v4();
        let policy_b = Uuid::new_v4();

        // Policy A matches 5 times
        for _ in 0..5 {
            analyzer.record_evaluation(None, Some(policy_a), 100, 1);
        }
        // Policy B matches 2 times
        for _ in 0..2 {
            analyzer.record_evaluation(None, Some(policy_b), 100, 1);
        }

        let top = analyzer.top_policies(2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].policy_id, policy_a);
        assert_eq!(top[1].policy_id, policy_b);
    }

    #[test]
    fn test_health_check() {
        let subnet_id = Uuid::new_v4();
        let policy = AssignmentPolicy::new("Test", subnet_id, 100)
            .with_rule(PolicyRule::node_type_equals(NodeType::DataCenter));

        let stats = PolicyStats::default();
        let health = health_check(&[policy], &stats);

        // New policy with no matches should generate a warning
        assert!(!health.issues.is_empty());
        assert!(health.issues.iter().any(|i| i.description.contains("never matched")));
    }

    #[test]
    fn test_recent_trend() {
        let mut analyzer = PolicyAnalyzer::new();
        let policy_id = Uuid::new_v4();

        // 8 successes, 2 failures
        for _ in 0..8 {
            analyzer.record_evaluation(None, Some(policy_id), 100, 1);
        }
        for _ in 0..2 {
            analyzer.record_evaluation(None, None, 100, 1);
        }

        // Trend over last 10 should be 0.8
        assert!((analyzer.recent_trend(10) - 0.8).abs() < 0.001);
    }
}
