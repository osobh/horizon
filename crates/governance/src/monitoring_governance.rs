//! Governance system monitoring module
//!
//! The GovernanceMonitor provides system health monitoring, policy violation detection,
//! and performance metrics for governance decisions.

use chrono::{DateTime, Duration, Utc};
use dashmap::DashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use crate::governance_engine::{AgentGovernanceState, AuditEntry, ViolationSeverity};
use crate::{GovernanceError, Result};
use stratoswarm_agent_core::agent::AgentId;

/// Governance monitor for system health and metrics
pub struct GovernanceMonitor {
    metrics_buffer: Arc<RwLock<MetricsBuffer>>,
    alert_rules: DashMap<Uuid, AlertRule>,
    active_alerts: DashMap<Uuid, Alert>,
    performance_tracker: Arc<RwLock<PerformanceTracker>>,
    anomaly_detector: Arc<AnomalyDetector>,
}

/// Metrics buffer for collecting governance metrics
#[derive(Debug, Default)]
struct MetricsBuffer {
    decision_metrics: Vec<DecisionMetric>,
    violation_metrics: Vec<ViolationMetric>,
    resource_metrics: Vec<ResourceMetric>,
    performance_metrics: Vec<PerformanceMetric>,
}

/// Decision metric
#[derive(Debug, Clone)]
struct DecisionMetric {
    timestamp: DateTime<Utc>,
    decision_type: String,
    agent_id: Option<AgentId>,
    result: DecisionResult,
    duration_ms: u64,
}

/// Decision result
#[derive(Debug, Clone, Copy)]
enum DecisionResult {
    Approved,
    Denied,
    Deferred,
}

/// Violation metric
#[derive(Debug, Clone)]
struct ViolationMetric {
    timestamp: DateTime<Utc>,
    agent_id: AgentId,
    violation_type: String,
    severity: ViolationSeverity,
}

/// Resource metric
#[derive(Debug, Clone)]
struct ResourceMetric {
    timestamp: DateTime<Utc>,
    agent_id: AgentId,
    resource_type: ResourceType,
    usage: f64,
    limit: f64,
}

/// Resource types
#[derive(Debug, Clone, Copy)]
enum ResourceType {
    Memory,
    CPU,
    GPU,
    Network,
    Storage,
}

/// Performance metric
#[derive(Debug, Clone)]
struct PerformanceMetric {
    timestamp: DateTime<Utc>,
    operation: String,
    latency_ms: u64,
    success: bool,
}

/// Alert rule for monitoring
#[derive(Debug, Clone)]
pub struct AlertRule {
    pub rule_id: Uuid,
    pub name: String,
    pub condition: AlertCondition,
    pub severity: AlertSeverity,
    pub actions: Vec<AlertAction>,
    pub cooldown_seconds: u64,
    pub enabled: bool,
}

/// Alert conditions
#[derive(Debug, Clone)]
pub enum AlertCondition {
    ViolationThreshold {
        count: u32,
        window_seconds: u64,
    },
    ResourceUtilization {
        resource: String,
        threshold_percent: f64,
    },
    DecisionLatency {
        threshold_ms: u64,
    },
    AgentAnomaly {
        deviation_factor: f64,
    },
    SystemHealth {
        metric: String,
        threshold: f64,
    },
}

/// Alert severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

/// Alert actions
#[derive(Debug, Clone)]
pub enum AlertAction {
    LogAlert,
    NotifyAdmin(String),
    TriggerRemediation(String),
    SuspendAgent(AgentId),
}

/// Active alert
#[derive(Debug, Clone)]
pub struct Alert {
    pub alert_id: Uuid,
    pub rule_id: Uuid,
    pub triggered_at: DateTime<Utc>,
    pub severity: AlertSeverity,
    pub message: String,
    pub context: serde_json::Value,
    pub acknowledged: bool,
}

/// Performance tracker
#[derive(Debug, Default)]
struct PerformanceTracker {
    decision_latencies: Vec<(DateTime<Utc>, u64)>,
    throughput_per_minute: Vec<(DateTime<Utc>, u32)>,
    error_rates: Vec<(DateTime<Utc>, f64)>,
}

/// Anomaly detector
struct AnomalyDetector {
    baseline_metrics: Arc<RwLock<BaselineMetrics>>,
    detection_threshold: f64,
}

/// Baseline metrics for anomaly detection
#[derive(Debug, Default)]
struct BaselineMetrics {
    avg_decision_latency_ms: f64,
    avg_violations_per_hour: f64,
    avg_resource_utilization: std::collections::HashMap<String, f64>,
}

/// Governance metrics summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GovernanceMetrics {
    pub total_agents: usize,
    pub active_agents: usize,
    pub suspended_agents: usize,
    pub total_violations: usize,
    pub critical_violations: usize,
    pub decisions_made: usize,
    pub approvals: usize,
    pub denials: usize,
    pub avg_decision_latency_ms: f64,
    pub system_health_score: f64,
    pub active_alerts: usize,
}

impl GovernanceMonitor {
    /// Create a new governance monitor
    pub fn new() -> Self {
        let mut monitor = Self {
            metrics_buffer: Arc::new(RwLock::new(MetricsBuffer::default())),
            alert_rules: DashMap::new(),
            active_alerts: DashMap::new(),
            performance_tracker: Arc::new(RwLock::new(PerformanceTracker::default())),
            anomaly_detector: Arc::new(AnomalyDetector {
                baseline_metrics: Arc::new(RwLock::new(BaselineMetrics::default())),
                detection_threshold: 2.0, // 2 standard deviations
            }),
        };

        // Initialize default alert rules
        monitor.initialize_default_rules();

        monitor
    }

    /// Initialize default alert rules
    fn initialize_default_rules(&mut self) {
        // High violation rate rule
        let violation_rule = AlertRule {
            rule_id: Uuid::new_v4(),
            name: "high_violation_rate".to_string(),
            condition: AlertCondition::ViolationThreshold {
                count: 10,
                window_seconds: 300, // 5 minutes
            },
            severity: AlertSeverity::Warning,
            actions: vec![
                AlertAction::LogAlert,
                AlertAction::NotifyAdmin("governance@example.com".to_string()),
            ],
            cooldown_seconds: 600, // 10 minutes
            enabled: true,
        };
        self.alert_rules
            .insert(violation_rule.rule_id, violation_rule);

        // Resource exhaustion rule
        let resource_rule = AlertRule {
            rule_id: Uuid::new_v4(),
            name: "resource_exhaustion".to_string(),
            condition: AlertCondition::ResourceUtilization {
                resource: "memory".to_string(),
                threshold_percent: 90.0,
            },
            severity: AlertSeverity::Critical,
            actions: vec![
                AlertAction::LogAlert,
                AlertAction::TriggerRemediation("scale_resources".to_string()),
            ],
            cooldown_seconds: 300,
            enabled: true,
        };
        self.alert_rules
            .insert(resource_rule.rule_id, resource_rule);

        // High latency rule
        let latency_rule = AlertRule {
            rule_id: Uuid::new_v4(),
            name: "high_decision_latency".to_string(),
            condition: AlertCondition::DecisionLatency { threshold_ms: 1000 },
            severity: AlertSeverity::Warning,
            actions: vec![AlertAction::LogAlert],
            cooldown_seconds: 300,
            enabled: true,
        };
        self.alert_rules.insert(latency_rule.rule_id, latency_rule);
    }

    /// Record a governance decision
    pub async fn record_decision(
        &self,
        decision_type: String,
        agent_id: Option<AgentId>,
        result: DecisionResult,
        duration_ms: u64,
    ) {
        let metric = DecisionMetric {
            timestamp: Utc::now(),
            decision_type,
            agent_id,
            result,
            duration_ms,
        };

        self.metrics_buffer.write().decision_metrics.push(metric);
        self.performance_tracker
            .write()
            .decision_latencies
            .push((Utc::now(), duration_ms));

        // Check for latency alerts
        self.check_latency_alerts(duration_ms).await;
    }

    /// Record a policy violation
    pub async fn record_violation(
        &self,
        agent_id: AgentId,
        violation_type: String,
        severity: ViolationSeverity,
    ) {
        let metric = ViolationMetric {
            timestamp: Utc::now(),
            agent_id,
            violation_type,
            severity,
        };

        self.metrics_buffer.write().violation_metrics.push(metric);

        // Check for violation threshold alerts
        self.check_violation_alerts().await;
    }

    /// Record resource usage
    pub async fn record_resource_usage(
        &self,
        agent_id: AgentId,
        resource_type: ResourceType,
        usage: f64,
        limit: f64,
    ) {
        let metric = ResourceMetric {
            timestamp: Utc::now(),
            agent_id,
            resource_type,
            usage,
            limit,
        };

        self.metrics_buffer.write().resource_metrics.push(metric);

        // Check for resource alerts
        let utilization_percent = (usage / limit) * 100.0;
        self.check_resource_alerts(resource_type, utilization_percent)
            .await;
    }

    /// Record performance metric
    pub async fn record_performance(&self, operation: String, latency_ms: u64, success: bool) {
        let metric = PerformanceMetric {
            timestamp: Utc::now(),
            operation,
            latency_ms,
            success,
        };

        self.metrics_buffer.write().performance_metrics.push(metric);
    }

    /// Check for latency alerts
    async fn check_latency_alerts(&self, latency_ms: u64) {
        for rule in self.alert_rules.iter() {
            if !rule.enabled {
                continue;
            }

            if let AlertCondition::DecisionLatency { threshold_ms } = &rule.condition {
                if latency_ms > *threshold_ms {
                    self.trigger_alert(
                        rule.rule_id,
                        rule.severity,
                        format!(
                            "Decision latency {}ms exceeds threshold {}ms",
                            latency_ms, threshold_ms
                        ),
                        serde_json::json!({
                            "latency_ms": latency_ms,
                            "threshold_ms": threshold_ms,
                        }),
                    )
                    .await;
                }
            }
        }
    }

    /// Check for violation alerts
    async fn check_violation_alerts(&self) {
        let now = Utc::now();

        for rule in self.alert_rules.iter() {
            if !rule.enabled {
                continue;
            }

            if let AlertCondition::ViolationThreshold {
                count,
                window_seconds,
            } = &rule.condition
            {
                let window_start = now - Duration::seconds(*window_seconds as i64);

                let recent_violations = self
                    .metrics_buffer
                    .read()
                    .violation_metrics
                    .iter()
                    .filter(|v| v.timestamp > window_start)
                    .count();

                if recent_violations >= *count as usize {
                    self.trigger_alert(
                        rule.rule_id,
                        rule.severity,
                        format!(
                            "{} violations in last {} seconds",
                            recent_violations, window_seconds
                        ),
                        serde_json::json!({
                            "violation_count": recent_violations,
                            "window_seconds": window_seconds,
                        }),
                    )
                    .await;
                }
            }
        }
    }

    /// Check for resource alerts
    async fn check_resource_alerts(&self, resource_type: ResourceType, utilization_percent: f64) {
        let resource_name = match resource_type {
            ResourceType::Memory => "memory",
            ResourceType::CPU => "cpu",
            ResourceType::GPU => "gpu",
            ResourceType::Network => "network",
            ResourceType::Storage => "storage",
        };

        for rule in self.alert_rules.iter() {
            if !rule.enabled {
                continue;
            }

            if let AlertCondition::ResourceUtilization {
                resource,
                threshold_percent,
            } = &rule.condition
            {
                if resource == resource_name && utilization_percent > *threshold_percent {
                    self.trigger_alert(
                        rule.rule_id,
                        rule.severity,
                        format!(
                            "{} utilization {:.1}% exceeds threshold {:.1}%",
                            resource, utilization_percent, threshold_percent
                        ),
                        serde_json::json!({
                            "resource": resource,
                            "utilization_percent": utilization_percent,
                            "threshold_percent": threshold_percent,
                        }),
                    )
                    .await;
                }
            }
        }
    }

    /// Trigger an alert
    async fn trigger_alert(
        &self,
        rule_id: Uuid,
        severity: AlertSeverity,
        message: String,
        context: serde_json::Value,
    ) {
        // Check cooldown
        if self.is_in_cooldown(&rule_id) {
            return;
        }

        let alert = Alert {
            alert_id: Uuid::new_v4(),
            rule_id,
            triggered_at: Utc::now(),
            severity,
            message: message.clone(),
            context,
            acknowledged: false,
        };

        // Execute alert actions
        if let Some(rule) = self.alert_rules.get(&rule_id) {
            for action in &rule.actions {
                self.execute_alert_action(action, &alert).await;
            }
        }

        self.active_alerts.insert(alert.alert_id, alert);

        match severity {
            AlertSeverity::Info => info!("Alert: {}", message),
            AlertSeverity::Warning => warn!("Alert: {}", message),
            AlertSeverity::Error => error!("Alert: {}", message),
            AlertSeverity::Critical => error!("CRITICAL Alert: {}", message),
        }
    }

    /// Check if a rule is in cooldown
    fn is_in_cooldown(&self, rule_id: &Uuid) -> bool {
        if let Some(rule) = self.alert_rules.get(rule_id) {
            let cooldown_end = Utc::now() - Duration::seconds(rule.cooldown_seconds as i64);

            self.active_alerts.iter().any(|entry| {
                entry.value().rule_id == *rule_id && entry.value().triggered_at > cooldown_end
            })
        } else {
            false
        }
    }

    /// Execute an alert action
    async fn execute_alert_action(&self, action: &AlertAction, alert: &Alert) {
        match action {
            AlertAction::LogAlert => {
                // Already logged in trigger_alert
            }
            AlertAction::NotifyAdmin(email) => {
                info!("Notifying admin {} about alert: {}", email, alert.message);
                // In real implementation, send email/notification
            }
            AlertAction::TriggerRemediation(remediation) => {
                info!("Triggering remediation: {}", remediation);
                // In real implementation, trigger automated remediation
            }
            AlertAction::SuspendAgent(agent_id) => {
                warn!("Suspending agent {:?} due to alert", agent_id);
                // In real implementation, suspend the agent
            }
        }
    }

    /// Collect governance metrics
    pub async fn collect_metrics(
        &self,
        active_agents: &DashMap<AgentId, AgentGovernanceState>,
        audit_log: &[AuditEntry],
    ) -> GovernanceMetrics {
        let buffer = self.metrics_buffer.read();

        // Count agent states
        let mut active_count = 0;
        let mut suspended_count = 0;
        let mut total_violations = 0;
        let mut critical_violations = 0;

        for agent in active_agents.iter() {
            match agent.value().lifecycle_phase {
                crate::governance_engine::LifecyclePhase::Active => active_count += 1,
                crate::governance_engine::LifecyclePhase::Suspended => suspended_count += 1,
                _ => {}
            }

            total_violations += agent.value().violations.len();
            critical_violations += agent
                .value()
                .violations
                .iter()
                .filter(|v| v.severity == ViolationSeverity::Critical)
                .count();
        }

        // Count decisions
        let approvals = buffer
            .decision_metrics
            .iter()
            .filter(|d| matches!(d.result, DecisionResult::Approved))
            .count();

        let denials = buffer
            .decision_metrics
            .iter()
            .filter(|d| matches!(d.result, DecisionResult::Denied))
            .count();

        // Calculate average latency
        let avg_decision_latency_ms = if buffer.decision_metrics.is_empty() {
            0.0
        } else {
            buffer
                .decision_metrics
                .iter()
                .map(|d| d.duration_ms as f64)
                .sum::<f64>()
                / buffer.decision_metrics.len() as f64
        };

        // Calculate system health score (0-100)
        let system_health_score = self.calculate_health_score(
            active_count,
            suspended_count,
            critical_violations,
            avg_decision_latency_ms,
        );

        GovernanceMetrics {
            total_agents: active_agents.len(),
            active_agents: active_count,
            suspended_agents: suspended_count,
            total_violations,
            critical_violations,
            decisions_made: buffer.decision_metrics.len(),
            approvals,
            denials,
            avg_decision_latency_ms,
            system_health_score,
            active_alerts: self.active_alerts.len(),
        }
    }

    /// Calculate system health score
    fn calculate_health_score(
        &self,
        active_agents: usize,
        suspended_agents: usize,
        critical_violations: usize,
        avg_latency_ms: f64,
    ) -> f64 {
        let mut score = 100.0;

        // Penalize for suspended agents
        if active_agents > 0 {
            let suspension_rate = suspended_agents as f64 / active_agents as f64;
            score -= suspension_rate * 20.0;
        }

        // Penalize for critical violations
        score -= (critical_violations as f64) * 5.0;

        // Penalize for high latency
        if avg_latency_ms > 1000.0 {
            score -= 10.0;
        } else if avg_latency_ms > 500.0 {
            score -= 5.0;
        }

        // Penalize for active alerts
        let critical_alerts = self
            .active_alerts
            .iter()
            .filter(|a| a.value().severity == AlertSeverity::Critical)
            .count();
        score -= (critical_alerts as f64) * 10.0;

        score.max(0.0).min(100.0)
    }

    /// Detect anomalies in governance behavior
    pub async fn detect_anomalies(&self) -> Vec<Anomaly> {
        let mut anomalies = Vec::new();

        let buffer = self.metrics_buffer.read();
        let baseline = self.anomaly_detector.baseline_metrics.read();

        // Check decision latency anomalies
        if !buffer.decision_metrics.is_empty() {
            let recent_avg_latency = buffer
                .decision_metrics
                .iter()
                .take(100) // Last 100 decisions
                .map(|d| d.duration_ms as f64)
                .sum::<f64>()
                / 100.0.min(buffer.decision_metrics.len() as f64);

            if recent_avg_latency
                > baseline.avg_decision_latency_ms * self.anomaly_detector.detection_threshold
            {
                anomalies.push(Anomaly {
                    anomaly_type: AnomalyType::LatencySpike,
                    severity: AlertSeverity::Warning,
                    description: format!(
                        "Decision latency {:.1}ms is {:.1}x baseline",
                        recent_avg_latency,
                        recent_avg_latency / baseline.avg_decision_latency_ms
                    ),
                    detected_at: Utc::now(),
                });
            }
        }

        // Check violation rate anomalies
        let recent_violations = buffer
            .violation_metrics
            .iter()
            .filter(|v| v.timestamp > Utc::now() - Duration::hours(1))
            .count() as f64;

        if recent_violations
            > baseline.avg_violations_per_hour * self.anomaly_detector.detection_threshold
        {
            anomalies.push(Anomaly {
                anomaly_type: AnomalyType::ViolationSurge,
                severity: AlertSeverity::Error,
                description: format!(
                    "Violation rate {} per hour is {:.1}x baseline",
                    recent_violations,
                    recent_violations / baseline.avg_violations_per_hour
                ),
                detected_at: Utc::now(),
            });
        }

        anomalies
    }

    /// Update baseline metrics
    pub async fn update_baselines(&self) {
        let buffer = self.metrics_buffer.read();
        let mut baseline = self.anomaly_detector.baseline_metrics.write();

        // Update decision latency baseline
        if !buffer.decision_metrics.is_empty() {
            baseline.avg_decision_latency_ms = buffer
                .decision_metrics
                .iter()
                .map(|d| d.duration_ms as f64)
                .sum::<f64>()
                / buffer.decision_metrics.len() as f64;
        }

        // Update violation rate baseline
        let hour_ago = Utc::now() - Duration::hours(1);
        baseline.avg_violations_per_hour = buffer
            .violation_metrics
            .iter()
            .filter(|v| v.timestamp > hour_ago)
            .count() as f64;

        // Update resource utilization baselines
        for metric in &buffer.resource_metrics {
            let resource_name = format!("{:?}", metric.resource_type);
            let utilization = metric.usage / metric.limit;

            baseline
                .avg_resource_utilization
                .entry(resource_name)
                .and_modify(|avg| *avg = (*avg + utilization) / 2.0)
                .or_insert(utilization);
        }
    }

    /// Clean up old metrics
    pub async fn cleanup_old_metrics(&self, retention_hours: u64) {
        let cutoff = Utc::now() - Duration::hours(retention_hours as i64);
        let mut buffer = self.metrics_buffer.write();

        buffer.decision_metrics.retain(|m| m.timestamp > cutoff);
        buffer.violation_metrics.retain(|m| m.timestamp > cutoff);
        buffer.resource_metrics.retain(|m| m.timestamp > cutoff);
        buffer.performance_metrics.retain(|m| m.timestamp > cutoff);

        // Clean up performance tracker
        let mut tracker = self.performance_tracker.write();
        tracker.decision_latencies.retain(|(t, _)| *t > cutoff);
        tracker.throughput_per_minute.retain(|(t, _)| *t > cutoff);
        tracker.error_rates.retain(|(t, _)| *t > cutoff);
    }

    /// Get active alerts
    pub async fn get_active_alerts(&self) -> Vec<Alert> {
        self.active_alerts
            .iter()
            .map(|entry| entry.value().clone())
            .collect()
    }

    /// Acknowledge an alert
    pub async fn acknowledge_alert(&self, alert_id: Uuid) -> Result<()> {
        if let Some(mut alert) = self.active_alerts.get_mut(&alert_id) {
            alert.acknowledged = true;
            Ok(())
        } else {
            Err(GovernanceError::InternalError(
                "Alert not found".to_string(),
            ))
        }
    }
}

/// Anomaly detected in governance behavior
#[derive(Debug, Clone)]
pub struct Anomaly {
    pub anomaly_type: AnomalyType,
    pub severity: AlertSeverity,
    pub description: String,
    pub detected_at: DateTime<Utc>,
}

/// Types of anomalies
#[derive(Debug, Clone, Copy)]
pub enum AnomalyType {
    LatencySpike,
    ViolationSurge,
    ResourceAnomaly,
    BehaviorDeviation,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::test;

    #[test]
    async fn test_governance_monitor_creation() {
        let monitor = GovernanceMonitor::new();
        assert!(monitor.alert_rules.len() > 0); // Has default rules
        assert_eq!(monitor.active_alerts.len(), 0);
    }

    #[test]
    async fn test_record_decision() {
        let monitor = GovernanceMonitor::new();

        monitor
            .record_decision(
                "resource_allocation".to_string(),
                Some(AgentId::new()),
                DecisionResult::Approved,
                50,
            )
            .await;

        let buffer = monitor.metrics_buffer.read();
        assert_eq!(buffer.decision_metrics.len(), 1);
        assert_eq!(buffer.decision_metrics[0].duration_ms, 50);
    }

    #[test]
    async fn test_record_violation() {
        let monitor = GovernanceMonitor::new();
        let agent_id = AgentId::new();

        monitor
            .record_violation(
                agent_id,
                "policy_violation".to_string(),
                ViolationSeverity::Medium,
            )
            .await;

        let buffer = monitor.metrics_buffer.read();
        assert_eq!(buffer.violation_metrics.len(), 1);
        assert_eq!(
            buffer.violation_metrics[0].violation_type,
            "policy_violation"
        );
    }

    #[test]
    async fn test_record_resource_usage() {
        let monitor = GovernanceMonitor::new();
        let agent_id = AgentId::new();

        monitor
            .record_resource_usage(agent_id, ResourceType::Memory, 512.0, 1024.0)
            .await;

        let buffer = monitor.metrics_buffer.read();
        assert_eq!(buffer.resource_metrics.len(), 1);
        assert_eq!(buffer.resource_metrics[0].usage, 512.0);
    }

    #[test]
    async fn test_latency_alert_trigger() {
        let monitor = GovernanceMonitor::new();

        // Record high latency decision
        monitor
            .record_decision(
                "test_decision".to_string(),
                None,
                DecisionResult::Approved,
                2000, // 2 seconds, above default threshold
            )
            .await;

        // Should have triggered an alert
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        assert!(monitor.active_alerts.len() > 0);
    }

    #[test]
    async fn test_violation_threshold_alert() {
        let monitor = GovernanceMonitor::new();
        let agent_id = AgentId::new();

        // Record multiple violations
        for _ in 0..11 {
            monitor
                .record_violation(
                    agent_id,
                    "test_violation".to_string(),
                    ViolationSeverity::High,
                )
                .await;
        }

        // Check for violation alerts
        monitor.check_violation_alerts().await;

        // Should have triggered an alert
        assert!(monitor.active_alerts.len() > 0);
    }

    #[test]
    async fn test_resource_utilization_alert() {
        let monitor = GovernanceMonitor::new();
        let agent_id = AgentId::new();

        // Record high resource usage
        monitor
            .record_resource_usage(
                agent_id,
                ResourceType::Memory,
                950.0,
                1000.0, // 95% utilization
            )
            .await;

        // Should have triggered an alert
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        assert!(monitor.active_alerts.len() > 0);
    }

    #[test]
    async fn test_alert_cooldown() {
        let monitor = GovernanceMonitor::new();

        // Trigger an alert
        monitor
            .record_decision("test".to_string(), None, DecisionResult::Approved, 2000)
            .await;

        let initial_alerts = monitor.active_alerts.len();

        // Try to trigger same alert again
        monitor
            .record_decision("test".to_string(), None, DecisionResult::Approved, 2000)
            .await;

        // Should not create another alert due to cooldown
        assert_eq!(monitor.active_alerts.len(), initial_alerts);
    }

    #[test]
    async fn test_metrics_collection() {
        let monitor = GovernanceMonitor::new();
        let active_agents = DashMap::new();

        // Add some test data
        for i in 0..5 {
            monitor
                .record_decision(
                    format!("decision_{}", i),
                    Some(AgentId::new()),
                    if i % 2 == 0 {
                        DecisionResult::Approved
                    } else {
                        DecisionResult::Denied
                    },
                    100 + i * 10,
                )
                .await;
        }

        let metrics = monitor.collect_metrics(&active_agents, &[]).await;

        assert_eq!(metrics.decisions_made, 5);
        assert_eq!(metrics.approvals, 3); // 0, 2, 4
        assert_eq!(metrics.denials, 2); // 1, 3
        assert!(metrics.avg_decision_latency_ms > 0.0);
    }

    #[test]
    async fn test_health_score_calculation() {
        let monitor = GovernanceMonitor::new();

        let score = monitor.calculate_health_score(100, 10, 2, 500.0);

        // Score should be reduced due to:
        // - 10% suspension rate: -2 points
        // - 2 critical violations: -10 points
        // - No latency penalty (500ms is acceptable)
        // - No alert penalty in this test
        assert!(score < 100.0);
        assert!(score > 80.0);
    }

    #[test]
    async fn test_anomaly_detection() {
        let monitor = GovernanceMonitor::new();

        // Update baseline with normal values
        monitor
            .anomaly_detector
            .baseline_metrics
            .write()
            .avg_decision_latency_ms = 100.0;
        monitor
            .anomaly_detector
            .baseline_metrics
            .write()
            .avg_violations_per_hour = 2.0;

        // Record anomalous data
        for _ in 0..10 {
            monitor
                .record_decision(
                    "test".to_string(),
                    None,
                    DecisionResult::Approved,
                    500, // 5x baseline
                )
                .await;
        }

        let anomalies = monitor.detect_anomalies().await;
        assert!(anomalies
            .iter()
            .any(|a| matches!(a.anomaly_type, AnomalyType::LatencySpike)));
    }

    #[test]
    async fn test_baseline_update() {
        let monitor = GovernanceMonitor::new();

        // Record some metrics
        for i in 0..10 {
            monitor
                .record_decision(
                    "test".to_string(),
                    None,
                    DecisionResult::Approved,
                    100 + i * 10,
                )
                .await;
        }

        monitor.update_baselines().await;

        let baseline = monitor.anomaly_detector.baseline_metrics.read();
        assert!(baseline.avg_decision_latency_ms > 0.0);
    }

    #[test]
    async fn test_metrics_cleanup() {
        let monitor = GovernanceMonitor::new();

        // Record old and new metrics
        {
            let mut buffer = monitor.metrics_buffer.write();

            // Old metric
            buffer.decision_metrics.push(DecisionMetric {
                timestamp: Utc::now() - Duration::hours(25),
                decision_type: "old".to_string(),
                agent_id: None,
                result: DecisionResult::Approved,
                duration_ms: 100,
            });

            // New metric
            buffer.decision_metrics.push(DecisionMetric {
                timestamp: Utc::now(),
                decision_type: "new".to_string(),
                agent_id: None,
                result: DecisionResult::Approved,
                duration_ms: 100,
            });
        }

        monitor.cleanup_old_metrics(24).await;

        let buffer = monitor.metrics_buffer.read();
        assert_eq!(buffer.decision_metrics.len(), 1);
        assert_eq!(buffer.decision_metrics[0].decision_type, "new");
    }

    #[test]
    async fn test_alert_acknowledgment() {
        let monitor = GovernanceMonitor::new();

        // Create an alert
        let alert = Alert {
            alert_id: Uuid::new_v4(),
            rule_id: Uuid::new_v4(),
            triggered_at: Utc::now(),
            severity: AlertSeverity::Warning,
            message: "Test alert".to_string(),
            context: serde_json::json!({}),
            acknowledged: false,
        };

        let alert_id = alert.alert_id;
        monitor.active_alerts.insert(alert_id, alert);

        // Acknowledge it
        monitor.acknowledge_alert(alert_id).await?;

        let alert = monitor.active_alerts.get(&alert_id)?;
        assert!(alert.acknowledged);
    }

    #[test]
    async fn test_alert_severity_levels() {
        let severities = vec![
            AlertSeverity::Info,
            AlertSeverity::Warning,
            AlertSeverity::Error,
            AlertSeverity::Critical,
        ];

        for severity in severities {
            match severity {
                AlertSeverity::Info => assert_eq!(format!("{:?}", severity), "Info"),
                AlertSeverity::Warning => assert_eq!(format!("{:?}", severity), "Warning"),
                AlertSeverity::Error => assert_eq!(format!("{:?}", severity), "Error"),
                AlertSeverity::Critical => assert_eq!(format!("{:?}", severity), "Critical"),
            }
        }
    }

    #[test]
    async fn test_performance_tracking() {
        let monitor = GovernanceMonitor::new();

        monitor
            .record_performance("test_operation".to_string(), 150, true)
            .await;

        let buffer = monitor.metrics_buffer.read();
        assert_eq!(buffer.performance_metrics.len(), 1);
        assert_eq!(buffer.performance_metrics[0].operation, "test_operation");
        assert!(buffer.performance_metrics[0].success);
    }
}
