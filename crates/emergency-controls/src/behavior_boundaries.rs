//! Behavior boundaries and safety violation detection
//!
//! Monitors and enforces:
//! - Agent behavior patterns
//! - Safety violations
//! - Suspicious activities
//! - Resource access patterns
//! - Communication boundaries

use crate::EmergencyResult;
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{broadcast, RwLock};
use tracing::{error, info, warn};

/// Types of behaviors to monitor
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BehaviorType {
    /// Memory access patterns
    MemoryAccess,
    /// Network communication
    NetworkCommunication,
    /// File system operations
    FileSystemOperations,
    /// Agent spawning
    AgentSpawning,
    /// Resource allocation
    ResourceAllocation,
    /// Code execution
    CodeExecution,
    /// Data access
    DataAccess,
    /// System calls
    SystemCalls,
}

/// Severity levels for violations
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ViolationSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Behavior boundary configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehaviorBoundary {
    pub behavior_type: BehaviorType,
    pub description: String,
    pub max_frequency: Option<u32>, // Max occurrences per time window
    pub time_window: Option<Duration>,
    pub forbidden_patterns: Vec<String>,
    pub allowed_patterns: Vec<String>,
    pub severity: ViolationSeverity,
}

/// Safety violation record
#[derive(Debug, Clone)]
pub struct SafetyViolation {
    pub agent_id: String,
    pub behavior_type: BehaviorType,
    pub violation_type: String,
    pub details: String,
    pub severity: ViolationSeverity,
    pub timestamp: Instant,
    pub action_taken: ViolationAction,
}

/// Actions to take on violations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ViolationAction {
    Log,
    Warn,
    Suspend,
    Terminate,
    Isolate,
}

/// Behavior monitoring statistics
#[derive(Debug, Clone)]
pub struct BehaviorStats {
    pub agent_id: String,
    pub behavior_type: BehaviorType,
    pub count: u32,
    pub first_seen: Instant,
    pub last_seen: Instant,
}

/// Behavior monitor system
pub struct BehaviorMonitor {
    boundaries: Arc<RwLock<Vec<BehaviorBoundary>>>,
    violations: Arc<RwLock<Vec<SafetyViolation>>>,
    agent_behaviors: Arc<DashMap<(String, BehaviorType), BehaviorStats>>,
    event_sender: broadcast::Sender<SafetyViolation>,
    max_violations_stored: usize,
}

impl BehaviorMonitor {
    /// Create new behavior monitor
    pub fn new(boundaries: Vec<BehaviorBoundary>) -> Self {
        let (event_sender, _) = broadcast::channel(1000);

        Self {
            boundaries: Arc::new(RwLock::new(boundaries)),
            violations: Arc::new(RwLock::new(Vec::new())),
            agent_behaviors: Arc::new(DashMap::new()),
            event_sender,
            max_violations_stored: 10000,
        }
    }

    /// Monitor agent behavior
    pub async fn monitor_behavior(
        &self,
        agent_id: &str,
        behavior_type: BehaviorType,
        action: &str,
    ) -> EmergencyResult<()> {
        let now = Instant::now();

        // Update behavior statistics
        let key = (agent_id.to_string(), behavior_type);
        self.agent_behaviors
            .entry(key.clone())
            .and_modify(|stats| {
                stats.count += 1;
                stats.last_seen = now;
            })
            .or_insert(BehaviorStats {
                agent_id: agent_id.to_string(),
                behavior_type,
                count: 1,
                first_seen: now,
                last_seen: now,
            });

        // Check boundaries
        let boundaries = self.boundaries.read().await;
        for boundary in boundaries
            .iter()
            .filter(|b| b.behavior_type == behavior_type)
        {
            self.check_boundary(agent_id, behavior_type, action, boundary, now)
                .await?;
        }

        Ok(())
    }

    /// Check if behavior violates boundary
    async fn check_boundary(
        &self,
        agent_id: &str,
        behavior_type: BehaviorType,
        action: &str,
        boundary: &BehaviorBoundary,
        now: Instant,
    ) -> EmergencyResult<()> {
        let mut violation_detected = false;
        let mut violation_type = String::new();
        let mut details = String::new();

        // Check forbidden patterns
        for pattern in &boundary.forbidden_patterns {
            if action.contains(pattern) {
                violation_detected = true;
                violation_type = "ForbiddenPattern".to_string();
                details = format!("Action '{action}' matches forbidden pattern '{pattern}'");
                break;
            }
        }

        // Check allowed patterns (if specified)
        if !violation_detected && !boundary.allowed_patterns.is_empty() {
            let matches_allowed = boundary.allowed_patterns.iter().any(|p| action.contains(p));
            if !matches_allowed {
                violation_detected = true;
                violation_type = "NotInAllowedPatterns".to_string();
                details = format!("Action '{action}' does not match any allowed patterns");
            }
        }

        // Check frequency limits
        if !violation_detected {
            if let (Some(max_freq), Some(window)) = (boundary.max_frequency, boundary.time_window) {
                let key = (agent_id.to_string(), behavior_type);
                if let Some(stats) = self.agent_behaviors.get(&key) {
                    let duration = now.duration_since(stats.first_seen);
                    if duration <= window && stats.count > max_freq {
                        violation_detected = true;
                        violation_type = "FrequencyLimitExceeded".to_string();
                        details = format!(
                            "Behavior occurred {} times in {:?} (limit: {})",
                            stats.count, duration, max_freq
                        );
                    }
                }
            }
        }

        if violation_detected {
            let action_taken = self.determine_action(boundary.severity);
            let violation = SafetyViolation {
                agent_id: agent_id.to_string(),
                behavior_type,
                violation_type,
                details,
                severity: boundary.severity,
                timestamp: now,
                action_taken,
            };

            self.record_violation(violation).await?;
        }

        Ok(())
    }

    /// Determine action based on severity
    fn determine_action(&self, severity: ViolationSeverity) -> ViolationAction {
        match severity {
            ViolationSeverity::Low => ViolationAction::Log,
            ViolationSeverity::Medium => ViolationAction::Warn,
            ViolationSeverity::High => ViolationAction::Suspend,
            ViolationSeverity::Critical => ViolationAction::Terminate,
        }
    }

    /// Record a safety violation
    async fn record_violation(&self, violation: SafetyViolation) -> EmergencyResult<()> {
        // Log the violation
        match violation.severity {
            ViolationSeverity::Low => {
                info!(
                    "Safety violation (LOW): Agent {} - {}",
                    violation.agent_id, violation.details
                );
            }
            ViolationSeverity::Medium => {
                warn!(
                    "Safety violation (MEDIUM): Agent {} - {}",
                    violation.agent_id, violation.details
                );
            }
            ViolationSeverity::High => {
                error!(
                    "Safety violation (HIGH): Agent {} - {}",
                    violation.agent_id, violation.details
                );
            }
            ViolationSeverity::Critical => {
                error!(
                    "CRITICAL SAFETY VIOLATION: Agent {} - {}",
                    violation.agent_id, violation.details
                );
            }
        }

        // Store violation
        let mut violations = self.violations.write().await;
        violations.push(violation.clone());

        // Trim if too many violations
        if violations.len() > self.max_violations_stored {
            let drain_count = violations.len() - self.max_violations_stored;
            violations.drain(0..drain_count);
        }

        // Broadcast event
        let _ = self.event_sender.send(violation);

        Ok(())
    }

    /// Get violations for a specific agent
    pub async fn get_agent_violations(&self, agent_id: &str) -> Vec<SafetyViolation> {
        self.violations
            .read()
            .await
            .iter()
            .filter(|v| v.agent_id == agent_id)
            .cloned()
            .collect()
    }

    /// Get all violations
    pub async fn get_all_violations(&self) -> Vec<SafetyViolation> {
        self.violations.read().await.clone()
    }

    /// Get violations by severity
    pub async fn get_violations_by_severity(
        &self,
        severity: ViolationSeverity,
    ) -> Vec<SafetyViolation> {
        self.violations
            .read()
            .await
            .iter()
            .filter(|v| v.severity == severity)
            .cloned()
            .collect()
    }

    /// Clear violations
    pub async fn clear_violations(&self) {
        self.violations.write().await.clear();
    }

    /// Add boundary
    pub async fn add_boundary(&self, boundary: BehaviorBoundary) {
        self.boundaries.write().await.push(boundary);
    }

    /// Remove boundaries for behavior type
    pub async fn remove_boundaries(&self, behavior_type: BehaviorType) {
        self.boundaries
            .write()
            .await
            .retain(|b| b.behavior_type != behavior_type);
    }

    /// Get behavior statistics for agent
    pub fn get_agent_stats(&self, agent_id: &str) -> Vec<BehaviorStats> {
        self.agent_behaviors
            .iter()
            .filter(|entry| entry.key().0 == agent_id)
            .map(|entry| entry.value().clone())
            .collect()
    }

    /// Reset statistics for agent
    pub fn reset_agent_stats(&self, agent_id: &str) {
        self.agent_behaviors.retain(|k, _| k.0 != agent_id);
    }

    /// Subscribe to violation events
    pub fn subscribe(&self) -> broadcast::Receiver<SafetyViolation> {
        self.event_sender.subscribe()
    }

    /// Get recent violations within a time window
    pub async fn get_recent_violations(&self, duration: Duration) -> Vec<SafetyViolation> {
        let cutoff = Instant::now() - duration;
        self.violations
            .read()
            .await
            .iter()
            .filter(|v| v.timestamp > cutoff)
            .cloned()
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_behavior_monitor_creation() {
        let boundaries = vec![BehaviorBoundary {
            behavior_type: BehaviorType::MemoryAccess,
            description: "Restrict memory access patterns".to_string(),
            max_frequency: Some(100),
            time_window: Some(Duration::from_secs(60)),
            forbidden_patterns: vec!["kernel".to_string(), "system".to_string()],
            allowed_patterns: vec![],
            severity: ViolationSeverity::High,
        }];

        let monitor = BehaviorMonitor::new(boundaries);
        assert!(monitor.get_all_violations().await.is_empty());
    }

    #[tokio::test]
    async fn test_monitor_normal_behavior() {
        let boundaries = vec![BehaviorBoundary {
            behavior_type: BehaviorType::FileSystemOperations,
            description: "Monitor file operations".to_string(),
            max_frequency: Some(10),
            time_window: Some(Duration::from_secs(60)),
            forbidden_patterns: vec!["/etc/".to_string(), "/sys/".to_string()],
            allowed_patterns: vec![],
            severity: ViolationSeverity::Medium,
        }];

        let monitor = BehaviorMonitor::new(boundaries);

        // Normal behavior - should not trigger violation
        monitor
            .monitor_behavior(
                "agent-1",
                BehaviorType::FileSystemOperations,
                "read /home/user/file.txt",
            )
            .await
            .unwrap();

        assert!(monitor.get_all_violations().await.is_empty());

        // Check stats were recorded
        let stats = monitor.get_agent_stats("agent-1");
        assert_eq!(stats.len(), 1);
        assert_eq!(stats[0].count, 1);
    }

    #[tokio::test]
    async fn test_forbidden_pattern_violation() {
        let boundaries = vec![BehaviorBoundary {
            behavior_type: BehaviorType::FileSystemOperations,
            description: "Restrict system file access".to_string(),
            max_frequency: None,
            time_window: None,
            forbidden_patterns: vec!["/etc/".to_string(), "/sys/".to_string()],
            allowed_patterns: vec![],
            severity: ViolationSeverity::High,
        }];

        let monitor = BehaviorMonitor::new(boundaries);
        let mut receiver = monitor.subscribe();

        // Trigger violation
        monitor
            .monitor_behavior(
                "agent-1",
                BehaviorType::FileSystemOperations,
                "write /etc/passwd",
            )
            .await
            .unwrap();

        // Check violation was recorded
        let violations = monitor.get_all_violations().await;
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].violation_type, "ForbiddenPattern");
        assert!(violations[0].details.contains("/etc/"));
        assert_eq!(violations[0].severity, ViolationSeverity::High);
        assert_eq!(violations[0].action_taken, ViolationAction::Suspend);

        // Check event was broadcast
        let event = receiver.recv().await.unwrap();
        assert_eq!(event.agent_id, "agent-1");
    }

    #[tokio::test]
    async fn test_allowed_pattern_violation() {
        let boundaries = vec![BehaviorBoundary {
            behavior_type: BehaviorType::DataAccess,
            description: "Restrict data access to specific paths".to_string(),
            max_frequency: None,
            time_window: None,
            forbidden_patterns: vec![],
            allowed_patterns: vec!["/data/public/".to_string(), "/data/shared/".to_string()],
            severity: ViolationSeverity::Medium,
        }];

        let monitor = BehaviorMonitor::new(boundaries);

        // Allowed access - no violation
        monitor
            .monitor_behavior(
                "agent-1",
                BehaviorType::DataAccess,
                "read /data/public/file.csv",
            )
            .await
            .unwrap();
        assert!(monitor.get_all_violations().await.is_empty());

        // Disallowed access - violation
        monitor
            .monitor_behavior(
                "agent-1",
                BehaviorType::DataAccess,
                "read /data/private/secrets.txt",
            )
            .await
            .unwrap();

        let violations = monitor.get_all_violations().await;
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].violation_type, "NotInAllowedPatterns");
    }

    #[tokio::test]
    async fn test_frequency_limit_violation() {
        let boundaries = vec![BehaviorBoundary {
            behavior_type: BehaviorType::NetworkCommunication,
            description: "Limit network requests".to_string(),
            max_frequency: Some(5),
            time_window: Some(Duration::from_secs(10)),
            forbidden_patterns: vec![],
            allowed_patterns: vec![],
            severity: ViolationSeverity::Low,
        }];

        let monitor = BehaviorMonitor::new(boundaries);

        // Make requests within limit
        for i in 0..5 {
            monitor
                .monitor_behavior(
                    "agent-1",
                    BehaviorType::NetworkCommunication,
                    &format!("GET /api/data/{}", i),
                )
                .await
                .unwrap();
        }
        assert!(monitor.get_all_violations().await.is_empty());

        // Exceed limit
        monitor
            .monitor_behavior(
                "agent-1",
                BehaviorType::NetworkCommunication,
                "GET /api/data/6",
            )
            .await
            .unwrap();

        let violations = monitor.get_all_violations().await;
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].violation_type, "FrequencyLimitExceeded");
        assert!(violations[0].details.contains("6 times"));
    }

    #[tokio::test]
    async fn test_multiple_agents() {
        let boundaries = vec![BehaviorBoundary {
            behavior_type: BehaviorType::AgentSpawning,
            description: "Limit agent spawning".to_string(),
            max_frequency: Some(3),
            time_window: Some(Duration::from_secs(60)),
            forbidden_patterns: vec![],
            allowed_patterns: vec![],
            severity: ViolationSeverity::Medium,
        }];

        let monitor = BehaviorMonitor::new(boundaries);

        // Agent 1 spawns within limit
        for i in 0..3 {
            monitor
                .monitor_behavior(
                    "agent-1",
                    BehaviorType::AgentSpawning,
                    &format!("spawn agent-1-{}", i),
                )
                .await
                .unwrap();
        }

        // Agent 2 spawns within limit
        for i in 0..3 {
            monitor
                .monitor_behavior(
                    "agent-2",
                    BehaviorType::AgentSpawning,
                    &format!("spawn agent-2-{}", i),
                )
                .await
                .unwrap();
        }

        // No violations yet
        assert!(monitor.get_all_violations().await.is_empty());

        // Agent 1 exceeds limit
        monitor
            .monitor_behavior("agent-1", BehaviorType::AgentSpawning, "spawn agent-1-3")
            .await
            .unwrap();

        let violations = monitor.get_agent_violations("agent-1").await;
        assert_eq!(violations.len(), 1);

        // Agent 2 still has no violations
        assert!(monitor.get_agent_violations("agent-2").await.is_empty());
    }

    #[tokio::test]
    async fn test_severity_actions() {
        let boundaries = vec![
            BehaviorBoundary {
                behavior_type: BehaviorType::SystemCalls,
                description: "Monitor system calls".to_string(),
                max_frequency: None,
                time_window: None,
                forbidden_patterns: vec!["exec".to_string()],
                allowed_patterns: vec![],
                severity: ViolationSeverity::Low,
            },
            BehaviorBoundary {
                behavior_type: BehaviorType::SystemCalls,
                description: "Critical system calls".to_string(),
                max_frequency: None,
                time_window: None,
                forbidden_patterns: vec!["shutdown".to_string()],
                allowed_patterns: vec![],
                severity: ViolationSeverity::Critical,
            },
        ];

        let monitor = BehaviorMonitor::new(boundaries);

        // Low severity violation
        monitor
            .monitor_behavior("agent-1", BehaviorType::SystemCalls, "exec ls")
            .await
            .unwrap();

        // Critical severity violation
        monitor
            .monitor_behavior("agent-2", BehaviorType::SystemCalls, "shutdown -h now")
            .await
            .unwrap();

        let violations = monitor.get_all_violations().await;
        assert_eq!(violations.len(), 2);

        // Check actions match severity
        let low_violation = violations
            .iter()
            .find(|v| v.severity == ViolationSeverity::Low)
            .unwrap();
        assert_eq!(low_violation.action_taken, ViolationAction::Log);

        let critical_violation = violations
            .iter()
            .find(|v| v.severity == ViolationSeverity::Critical)
            .unwrap();
        assert_eq!(critical_violation.action_taken, ViolationAction::Terminate);
    }

    #[tokio::test]
    async fn test_get_violations_by_severity() {
        let boundaries = vec![
            BehaviorBoundary {
                behavior_type: BehaviorType::MemoryAccess,
                description: "Memory violations".to_string(),
                max_frequency: None,
                time_window: None,
                forbidden_patterns: vec!["kernel".to_string()],
                allowed_patterns: vec![],
                severity: ViolationSeverity::High,
            },
            BehaviorBoundary {
                behavior_type: BehaviorType::NetworkCommunication,
                description: "Network violations".to_string(),
                max_frequency: None,
                time_window: None,
                forbidden_patterns: vec!["malicious.com".to_string()],
                allowed_patterns: vec![],
                severity: ViolationSeverity::Critical,
            },
        ];

        let monitor = BehaviorMonitor::new(boundaries);

        // Create violations of different severities
        monitor
            .monitor_behavior(
                "agent-1",
                BehaviorType::MemoryAccess,
                "access kernel memory",
            )
            .await
            .unwrap();
        monitor
            .monitor_behavior(
                "agent-2",
                BehaviorType::NetworkCommunication,
                "connect malicious.com",
            )
            .await
            .unwrap();

        // Get high severity violations
        let high_violations = monitor
            .get_violations_by_severity(ViolationSeverity::High)
            .await;
        assert_eq!(high_violations.len(), 1);
        assert_eq!(high_violations[0].behavior_type, BehaviorType::MemoryAccess);

        // Get critical severity violations
        let critical_violations = monitor
            .get_violations_by_severity(ViolationSeverity::Critical)
            .await;
        assert_eq!(critical_violations.len(), 1);
        assert_eq!(
            critical_violations[0].behavior_type,
            BehaviorType::NetworkCommunication
        );
    }

    #[tokio::test]
    async fn test_clear_violations() {
        let boundaries = vec![BehaviorBoundary {
            behavior_type: BehaviorType::CodeExecution,
            description: "Monitor code execution".to_string(),
            max_frequency: None,
            time_window: None,
            forbidden_patterns: vec!["eval".to_string()],
            allowed_patterns: vec![],
            severity: ViolationSeverity::Medium,
        }];

        let monitor = BehaviorMonitor::new(boundaries);

        // Create some violations
        monitor
            .monitor_behavior("agent-1", BehaviorType::CodeExecution, "eval('dangerous')")
            .await
            .unwrap();
        monitor
            .monitor_behavior(
                "agent-2",
                BehaviorType::CodeExecution,
                "eval('more danger')",
            )
            .await
            .unwrap();

        assert_eq!(monitor.get_all_violations().await.len(), 2);

        // Clear violations
        monitor.clear_violations().await;
        assert!(monitor.get_all_violations().await.is_empty());
    }

    #[tokio::test]
    async fn test_add_remove_boundaries() {
        let monitor = BehaviorMonitor::new(vec![]);

        // Add boundary
        let boundary = BehaviorBoundary {
            behavior_type: BehaviorType::ResourceAllocation,
            description: "Monitor resource allocation".to_string(),
            max_frequency: Some(10),
            time_window: Some(Duration::from_secs(60)),
            forbidden_patterns: vec![],
            allowed_patterns: vec![],
            severity: ViolationSeverity::Low,
        };
        monitor.add_boundary(boundary).await;

        // Test boundary works
        for _ in 0..11 {
            monitor
                .monitor_behavior(
                    "agent-1",
                    BehaviorType::ResourceAllocation,
                    "allocate memory",
                )
                .await
                .unwrap();
        }
        assert_eq!(monitor.get_all_violations().await.len(), 1);

        // Remove boundary
        monitor
            .remove_boundaries(BehaviorType::ResourceAllocation)
            .await;
        monitor.clear_violations().await;

        // Test boundary no longer applies
        for _ in 0..20 {
            monitor
                .monitor_behavior(
                    "agent-1",
                    BehaviorType::ResourceAllocation,
                    "allocate memory",
                )
                .await
                .unwrap();
        }
        assert!(monitor.get_all_violations().await.is_empty());
    }

    #[tokio::test]
    async fn test_reset_agent_stats() {
        let monitor = BehaviorMonitor::new(vec![]);

        // Create stats for multiple agents
        monitor
            .monitor_behavior("agent-1", BehaviorType::MemoryAccess, "read memory")
            .await
            .unwrap();
        monitor
            .monitor_behavior("agent-1", BehaviorType::NetworkCommunication, "send data")
            .await
            .unwrap();
        monitor
            .monitor_behavior("agent-2", BehaviorType::FileSystemOperations, "write file")
            .await
            .unwrap();

        // Verify stats exist
        assert_eq!(monitor.get_agent_stats("agent-1").len(), 2);
        assert_eq!(monitor.get_agent_stats("agent-2").len(), 1);

        // Reset agent-1 stats
        monitor.reset_agent_stats("agent-1");

        // Verify only agent-1 stats were reset
        assert!(monitor.get_agent_stats("agent-1").is_empty());
        assert_eq!(monitor.get_agent_stats("agent-2").len(), 1);
    }

    #[tokio::test]
    async fn test_max_violations_stored() {
        let boundaries = vec![BehaviorBoundary {
            behavior_type: BehaviorType::DataAccess,
            description: "Data access violations".to_string(),
            max_frequency: None,
            time_window: None,
            forbidden_patterns: vec!["secret".to_string()],
            allowed_patterns: vec![],
            severity: ViolationSeverity::Low,
        }];

        let mut monitor = BehaviorMonitor::new(boundaries);
        monitor.max_violations_stored = 5; // Set low limit for testing

        // Create more violations than the limit
        for i in 0..10 {
            monitor
                .monitor_behavior(
                    "agent-1",
                    BehaviorType::DataAccess,
                    &format!("access secret-{}", i),
                )
                .await
                .unwrap();
        }

        // Should only store the latest 5 violations
        let violations = monitor.get_all_violations().await;
        assert_eq!(violations.len(), 5);

        // Verify we have the latest violations (5-9)
        for (i, violation) in violations.iter().enumerate() {
            assert!(violation.details.contains(&format!("secret-{}", i + 5)));
        }
    }

    #[tokio::test]
    async fn test_behavior_type_serialization() {
        let behavior_type = BehaviorType::SystemCalls;
        let serialized = serde_json::to_string(&behavior_type).unwrap();
        let deserialized: BehaviorType = serde_json::from_str(&serialized).unwrap();
        assert_eq!(behavior_type, deserialized);
    }

    #[tokio::test]
    async fn test_violation_severity_ordering() {
        assert!(ViolationSeverity::Low < ViolationSeverity::Medium);
        assert!(ViolationSeverity::Medium < ViolationSeverity::High);
        assert!(ViolationSeverity::High < ViolationSeverity::Critical);
    }
}
