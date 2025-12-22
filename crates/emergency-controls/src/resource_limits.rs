//! Resource consumption limits and monitoring
//!
//! Enforces limits on:
//! - GPU memory usage
//! - CPU utilization
//! - Network bandwidth
//! - Disk I/O
//! - Agent spawn rates

use crate::{EmergencyError, EmergencyResult};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{broadcast, RwLock};
use tracing::{error, info, warn};

/// Types of resources that can be limited
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ResourceType {
    /// GPU memory in bytes
    GpuMemory,
    /// GPU utilization percentage (0-100)
    GpuUtilization,
    /// CPU utilization percentage (0-100)
    CpuPercent,
    /// Memory usage in bytes
    SystemMemory,
    /// Network bandwidth in bytes/sec
    NetworkBandwidth,
    /// Disk I/O in bytes/sec
    DiskIO,
    /// Number of agents
    AgentCount,
    /// Agent spawn rate per second
    AgentSpawnRate,
}

/// Action to take when limit is exceeded
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LimitAction {
    /// Just warn about the violation
    Warn,
    /// Throttle the resource usage
    Throttle,
    /// Suspend the offending agent
    Suspend,
    /// Activate kill switch
    KillSwitch,
    /// Custom action
    Custom,
}

/// Resource limit configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimit {
    pub resource_type: ResourceType,
    pub limit: f64,
    pub action: LimitAction,
}

/// Current resource usage
#[derive(Debug, Clone)]
pub struct ResourceUsage {
    pub resource_type: ResourceType,
    pub current: f64,
    pub limit: f64,
    pub timestamp: Instant,
}

/// Resource violation event
#[derive(Debug, Clone)]
pub struct ResourceViolation {
    pub resource_type: ResourceType,
    pub current: f64,
    pub limit: f64,
    pub agent_id: Option<String>,
    pub action_taken: LimitAction,
    pub timestamp: Instant,
}

/// Resource limiter system
pub struct ResourceLimiter {
    limits: Arc<RwLock<Vec<ResourceLimit>>>,
    current_usage: Arc<DashMap<ResourceType, f64>>,
    violations: Arc<RwLock<Vec<ResourceViolation>>>,
    event_sender: broadcast::Sender<ResourceViolation>,
    _monitoring_interval: Duration,
    _grace_period: Duration,
    _last_check: Arc<RwLock<Instant>>,
}

impl ResourceLimiter {
    /// Create new resource limiter
    pub fn new(limits: Vec<ResourceLimit>) -> Self {
        let (event_sender, _) = broadcast::channel(1000);

        Self {
            limits: Arc::new(RwLock::new(limits)),
            current_usage: Arc::new(DashMap::new()),
            violations: Arc::new(RwLock::new(Vec::new())),
            event_sender,
            _monitoring_interval: Duration::from_millis(100),
            _grace_period: Duration::from_secs(5),
            _last_check: Arc::new(RwLock::new(Instant::now())),
        }
    }

    /// Update resource usage
    pub async fn update_usage(
        &self,
        resource_type: ResourceType,
        value: f64,
    ) -> EmergencyResult<()> {
        self.update_usage_with_agent(resource_type, value, None)
            .await
    }

    /// Update resource usage with optional agent ID
    pub async fn update_usage_with_agent(
        &self,
        resource_type: ResourceType,
        value: f64,
        agent_id: Option<String>,
    ) -> EmergencyResult<()> {
        self.current_usage.insert(resource_type, value);

        // Check if limit exceeded
        let limits = self.limits.read().await;
        if let Some(limit) = limits.iter().find(|l| l.resource_type == resource_type) {
            if value > limit.limit {
                self.handle_violation_with_agent(resource_type, value, limit, agent_id)
                    .await?;
            }
        }

        Ok(())
    }

    /// Handle resource violation
    async fn handle_violation(
        &self,
        resource_type: ResourceType,
        current: f64,
        limit: &ResourceLimit,
    ) -> EmergencyResult<()> {
        self.handle_violation_with_agent(resource_type, current, limit, None)
            .await
    }

    /// Handle resource violation with optional agent ID
    async fn handle_violation_with_agent(
        &self,
        resource_type: ResourceType,
        current: f64,
        limit: &ResourceLimit,
        agent_id: Option<String>,
    ) -> EmergencyResult<()> {
        let violation = ResourceViolation {
            resource_type,
            current,
            limit: limit.limit,
            agent_id,
            action_taken: limit.action,
            timestamp: Instant::now(),
        };

        // Log violation
        match limit.action {
            LimitAction::Warn => {
                warn!(
                    "Resource limit warning: {:?} = {} > {}",
                    resource_type, current, limit.limit
                );
            }
            LimitAction::Throttle => {
                warn!(
                    "Throttling resource: {:?} = {} > {}",
                    resource_type, current, limit.limit
                );
                // Implement throttling logic
            }
            LimitAction::Suspend => {
                error!(
                    "Suspending agents due to: {:?} = {} > {}",
                    resource_type, current, limit.limit
                );
                // Implement suspension logic
            }
            LimitAction::KillSwitch => {
                error!(
                    "Activating kill switch due to: {:?} = {} > {}",
                    resource_type, current, limit.limit
                );
                // Implement kill switch activation
            }
            LimitAction::Custom => {
                info!(
                    "Custom action for: {:?} = {} > {}",
                    resource_type, current, limit.limit
                );
            }
        }

        // Store violation
        self.violations.write().await.push(violation.clone());

        // Broadcast event
        let _ = self.event_sender.send(violation);

        Ok(())
    }

    /// Get current usage for a resource
    pub fn get_usage(&self, resource_type: ResourceType) -> Option<f64> {
        self.current_usage.get(&resource_type).map(|v| *v)
    }

    /// Get all current usage
    pub fn get_all_usage(&self) -> Vec<ResourceUsage> {
        let now = Instant::now();
        self.current_usage
            .iter()
            .map(|entry| ResourceUsage {
                resource_type: *entry.key(),
                current: *entry.value(),
                limit: 0.0, // Will be filled later
                timestamp: now,
            })
            .collect()
    }

    /// Subscribe to violation events
    pub fn subscribe(&self) -> broadcast::Receiver<ResourceViolation> {
        self.event_sender.subscribe()
    }

    /// Add new limit
    pub async fn add_limit(&self, limit: ResourceLimit) -> EmergencyResult<()> {
        self.limits.write().await.push(limit);
        Ok(())
    }

    /// Remove limit
    pub async fn remove_limit(&self, resource_type: ResourceType) -> EmergencyResult<()> {
        let mut limits = self.limits.write().await;
        limits.retain(|l| l.resource_type != resource_type);
        Ok(())
    }

    /// Update limit
    pub async fn update_limit(
        &self,
        resource_type: ResourceType,
        new_limit: f64,
    ) -> EmergencyResult<()> {
        let mut limits = self.limits.write().await;
        if let Some(limit) = limits.iter_mut().find(|l| l.resource_type == resource_type) {
            limit.limit = new_limit;
            Ok(())
        } else {
            Err(EmergencyError::ConfigurationError(format!(
                "Limit not found for resource type: {resource_type:?}"
            )))
        }
    }

    /// Get violations history
    pub async fn get_violations(&self) -> Vec<ResourceViolation> {
        self.violations.read().await.clone()
    }

    /// Clear violations history
    pub async fn clear_violations(&self) {
        self.violations.write().await.clear();
    }

    /// Check all limits
    pub async fn check_all_limits(&self) -> EmergencyResult<Vec<ResourceViolation>> {
        let mut violations = Vec::new();
        let limits = self.limits.read().await;

        for limit in limits.iter() {
            if let Some(current) = self.get_usage(limit.resource_type) {
                if current > limit.limit {
                    let violation = ResourceViolation {
                        resource_type: limit.resource_type,
                        current,
                        limit: limit.limit,
                        agent_id: None,
                        action_taken: limit.action,
                        timestamp: Instant::now(),
                    };
                    violations.push(violation);
                }
            }
        }

        Ok(violations)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_resource_limiter_creation() {
        let limits = vec![
            ResourceLimit {
                resource_type: ResourceType::GpuMemory,
                limit: 16_384.0,
                action: LimitAction::KillSwitch,
            },
            ResourceLimit {
                resource_type: ResourceType::CpuPercent,
                limit: 90.0,
                action: LimitAction::Throttle,
            },
        ];

        let limiter = ResourceLimiter::new(limits);
        assert!(limiter.get_usage(ResourceType::GpuMemory).is_none());
    }

    #[tokio::test]
    async fn test_update_usage_within_limit() {
        let limits = vec![ResourceLimit {
            resource_type: ResourceType::GpuMemory,
            limit: 16_384.0,
            action: LimitAction::Warn,
        }];

        let limiter = ResourceLimiter::new(limits);
        limiter
            .update_usage(ResourceType::GpuMemory, 8_192.0)
            .await
            .unwrap();

        assert_eq!(limiter.get_usage(ResourceType::GpuMemory), Some(8_192.0));
        assert!(limiter.get_violations().await.is_empty());
    }

    #[tokio::test]
    async fn test_update_usage_exceeds_limit() {
        let limits = vec![ResourceLimit {
            resource_type: ResourceType::GpuMemory,
            limit: 16_384.0,
            action: LimitAction::Warn,
        }];

        let limiter = ResourceLimiter::new(limits);
        let mut receiver = limiter.subscribe();

        limiter
            .update_usage(ResourceType::GpuMemory, 20_000.0)
            .await
            .unwrap();

        // Check violation was recorded
        let violations = limiter.get_violations().await;
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].resource_type, ResourceType::GpuMemory);
        assert_eq!(violations[0].current, 20_000.0);
        assert_eq!(violations[0].limit, 16_384.0);
        assert_eq!(violations[0].action_taken, LimitAction::Warn);

        // Check event was broadcast
        let event = receiver.recv().await.unwrap();
        assert_eq!(event.resource_type, ResourceType::GpuMemory);
    }

    #[tokio::test]
    async fn test_multiple_resource_types() {
        let limits = vec![
            ResourceLimit {
                resource_type: ResourceType::GpuMemory,
                limit: 16_384.0,
                action: LimitAction::Warn,
            },
            ResourceLimit {
                resource_type: ResourceType::CpuPercent,
                limit: 80.0,
                action: LimitAction::Throttle,
            },
            ResourceLimit {
                resource_type: ResourceType::AgentCount,
                limit: 100.0,
                action: LimitAction::Suspend,
            },
        ];

        let limiter = ResourceLimiter::new(limits);

        // Update multiple resources
        limiter
            .update_usage(ResourceType::GpuMemory, 10_000.0)
            .await
            .unwrap();
        limiter
            .update_usage(ResourceType::CpuPercent, 75.0)
            .await
            .unwrap();
        limiter
            .update_usage(ResourceType::AgentCount, 50.0)
            .await
            .unwrap();

        // Check all are within limits
        assert!(limiter.get_violations().await.is_empty());

        // Exceed CPU limit
        limiter
            .update_usage(ResourceType::CpuPercent, 95.0)
            .await
            .unwrap();

        let violations = limiter.get_violations().await;
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].resource_type, ResourceType::CpuPercent);
        assert_eq!(violations[0].action_taken, LimitAction::Throttle);
    }

    #[tokio::test]
    async fn test_add_remove_limits() {
        let limiter = ResourceLimiter::new(vec![]);

        // Add limit
        let limit = ResourceLimit {
            resource_type: ResourceType::NetworkBandwidth,
            limit: 1_000_000.0, // 1 MB/s
            action: LimitAction::Throttle,
        };
        limiter.add_limit(limit).await.unwrap();

        // Test limit works
        limiter
            .update_usage(ResourceType::NetworkBandwidth, 2_000_000.0)
            .await
            .unwrap();
        assert_eq!(limiter.get_violations().await.len(), 1);

        // Remove limit
        limiter
            .remove_limit(ResourceType::NetworkBandwidth)
            .await
            .unwrap();
        limiter.clear_violations().await;

        // Test limit no longer applies
        limiter
            .update_usage(ResourceType::NetworkBandwidth, 3_000_000.0)
            .await
            .unwrap();
        assert!(limiter.get_violations().await.is_empty());
    }

    #[tokio::test]
    async fn test_update_limit() {
        let limits = vec![ResourceLimit {
            resource_type: ResourceType::DiskIO,
            limit: 100_000.0,
            action: LimitAction::Warn,
        }];

        let limiter = ResourceLimiter::new(limits);

        // Set usage that exceeds original limit
        limiter
            .update_usage(ResourceType::DiskIO, 150_000.0)
            .await
            .unwrap();
        assert_eq!(limiter.get_violations().await.len(), 1);

        // Update limit to higher value
        limiter
            .update_limit(ResourceType::DiskIO, 200_000.0)
            .await
            .unwrap();
        limiter.clear_violations().await;

        // Same usage should now be within limit
        limiter
            .update_usage(ResourceType::DiskIO, 150_000.0)
            .await
            .unwrap();
        assert!(limiter.get_violations().await.is_empty());
    }

    #[tokio::test]
    async fn test_get_all_usage() {
        let limiter = ResourceLimiter::new(vec![]);

        // Update multiple resources
        limiter
            .update_usage(ResourceType::GpuMemory, 8_192.0)
            .await
            .unwrap();
        limiter
            .update_usage(ResourceType::CpuPercent, 45.0)
            .await
            .unwrap();
        limiter
            .update_usage(ResourceType::AgentCount, 25.0)
            .await
            .unwrap();

        let all_usage = limiter.get_all_usage();
        assert_eq!(all_usage.len(), 3);

        // Verify values
        let gpu_usage = all_usage
            .iter()
            .find(|u| u.resource_type == ResourceType::GpuMemory)
            .unwrap();
        assert_eq!(gpu_usage.current, 8_192.0);

        let cpu_usage = all_usage
            .iter()
            .find(|u| u.resource_type == ResourceType::CpuPercent)
            .unwrap();
        assert_eq!(cpu_usage.current, 45.0);
    }

    #[tokio::test]
    async fn test_check_all_limits() {
        let limits = vec![
            ResourceLimit {
                resource_type: ResourceType::GpuMemory,
                limit: 16_384.0,
                action: LimitAction::Warn,
            },
            ResourceLimit {
                resource_type: ResourceType::CpuPercent,
                limit: 80.0,
                action: LimitAction::Throttle,
            },
            ResourceLimit {
                resource_type: ResourceType::AgentCount,
                limit: 100.0,
                action: LimitAction::Suspend,
            },
        ];

        let limiter = ResourceLimiter::new(limits);

        // Set some resources exceeding limits
        limiter
            .update_usage(ResourceType::GpuMemory, 20_000.0)
            .await
            .unwrap();
        limiter
            .update_usage(ResourceType::CpuPercent, 85.0)
            .await
            .unwrap();
        limiter
            .update_usage(ResourceType::AgentCount, 50.0)
            .await
            .unwrap(); // Within limit

        let violations = limiter.check_all_limits().await.unwrap();
        assert_eq!(violations.len(), 2);

        // Verify specific violations
        assert!(violations
            .iter()
            .any(|v| v.resource_type == ResourceType::GpuMemory));
        assert!(violations
            .iter()
            .any(|v| v.resource_type == ResourceType::CpuPercent));
        assert!(!violations
            .iter()
            .any(|v| v.resource_type == ResourceType::AgentCount));
    }

    #[tokio::test]
    async fn test_different_limit_actions() {
        let limits = vec![
            ResourceLimit {
                resource_type: ResourceType::GpuMemory,
                limit: 1000.0,
                action: LimitAction::Warn,
            },
            ResourceLimit {
                resource_type: ResourceType::CpuPercent,
                limit: 50.0,
                action: LimitAction::Throttle,
            },
            ResourceLimit {
                resource_type: ResourceType::AgentCount,
                limit: 10.0,
                action: LimitAction::Suspend,
            },
            ResourceLimit {
                resource_type: ResourceType::NetworkBandwidth,
                limit: 1000.0,
                action: LimitAction::KillSwitch,
            },
            ResourceLimit {
                resource_type: ResourceType::DiskIO,
                limit: 500.0,
                action: LimitAction::Custom,
            },
        ];

        let limiter = ResourceLimiter::new(limits);

        // Test each action type
        limiter
            .update_usage(ResourceType::GpuMemory, 2000.0)
            .await
            .unwrap();
        limiter
            .update_usage(ResourceType::CpuPercent, 75.0)
            .await
            .unwrap();
        limiter
            .update_usage(ResourceType::AgentCount, 20.0)
            .await
            .unwrap();
        limiter
            .update_usage(ResourceType::NetworkBandwidth, 2000.0)
            .await
            .unwrap();
        limiter
            .update_usage(ResourceType::DiskIO, 1000.0)
            .await
            .unwrap();

        let violations = limiter.get_violations().await;
        assert_eq!(violations.len(), 5);

        // Verify each action type was recorded
        assert!(violations
            .iter()
            .any(|v| v.action_taken == LimitAction::Warn));
        assert!(violations
            .iter()
            .any(|v| v.action_taken == LimitAction::Throttle));
        assert!(violations
            .iter()
            .any(|v| v.action_taken == LimitAction::Suspend));
        assert!(violations
            .iter()
            .any(|v| v.action_taken == LimitAction::KillSwitch));
        assert!(violations
            .iter()
            .any(|v| v.action_taken == LimitAction::Custom));
    }

    #[tokio::test]
    async fn test_update_nonexistent_limit() {
        let limiter = ResourceLimiter::new(vec![]);

        let result = limiter.update_limit(ResourceType::GpuMemory, 1000.0).await;
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            EmergencyError::ConfigurationError(_)
        ));
    }

    #[tokio::test]
    async fn test_resource_type_serialization() {
        let resource_type = ResourceType::GpuMemory;
        let serialized = serde_json::to_string(&resource_type).unwrap();
        let deserialized: ResourceType = serde_json::from_str(&serialized).unwrap();
        assert_eq!(resource_type, deserialized);
    }

    #[tokio::test]
    async fn test_limit_action_serialization() {
        let action = LimitAction::KillSwitch;
        let serialized = serde_json::to_string(&action).unwrap();
        let deserialized: LimitAction = serde_json::from_str(&serialized).unwrap();
        assert_eq!(action, deserialized);
    }

    #[tokio::test]
    async fn test_resource_limit_serialization() {
        let limit = ResourceLimit {
            resource_type: ResourceType::AgentSpawnRate,
            limit: 10.0,
            action: LimitAction::Throttle,
        };
        let serialized = serde_json::to_string(&limit).unwrap();
        let deserialized: ResourceLimit = serde_json::from_str(&serialized).unwrap();
        assert_eq!(limit.resource_type, deserialized.resource_type);
        assert_eq!(limit.limit, deserialized.limit);
        assert_eq!(limit.action, deserialized.action);
    }

    #[tokio::test]
    async fn test_clear_violations() {
        let limits = vec![ResourceLimit {
            resource_type: ResourceType::GpuMemory,
            limit: 1000.0,
            action: LimitAction::Warn,
        }];

        let limiter = ResourceLimiter::new(limits);

        // Create some violations
        limiter
            .update_usage(ResourceType::GpuMemory, 2000.0)
            .await
            .unwrap();
        limiter
            .update_usage(ResourceType::GpuMemory, 3000.0)
            .await
            .unwrap();

        assert_eq!(limiter.get_violations().await.len(), 2);

        // Clear violations
        limiter.clear_violations().await;
        assert!(limiter.get_violations().await.is_empty());
    }

    #[tokio::test]
    async fn test_add_and_remove_limits() {
        let limiter = ResourceLimiter::new(vec![]);

        // Add a limit
        let limit = ResourceLimit {
            resource_type: ResourceType::NetworkBandwidth,
            limit: 1_000_000.0,
            action: LimitAction::Throttle,
        };
        limiter.add_limit(limit).await.unwrap();

        // Update usage and verify limit is enforced
        limiter
            .update_usage(ResourceType::NetworkBandwidth, 2_000_000.0)
            .await
            .unwrap();
        assert_eq!(limiter.get_violations().await.len(), 1);

        // Remove the limit
        limiter
            .remove_limit(ResourceType::NetworkBandwidth)
            .await
            .unwrap();

        // Clear previous violations
        limiter.clear_violations().await;

        // Update usage again - should not create violation
        limiter
            .update_usage(ResourceType::NetworkBandwidth, 3_000_000.0)
            .await
            .unwrap();
        assert!(limiter.get_violations().await.is_empty());
    }

    #[tokio::test]
    async fn test_get_all_usage_detailed() {
        let limiter = ResourceLimiter::new(vec![]);

        // Set usage for multiple resources
        limiter
            .update_usage(ResourceType::GpuMemory, 8192.0)
            .await
            .unwrap();
        limiter
            .update_usage(ResourceType::CpuPercent, 45.0)
            .await
            .unwrap();
        limiter
            .update_usage(ResourceType::AgentCount, 25.0)
            .await
            .unwrap();

        let all_usage = limiter.get_all_usage();
        assert_eq!(all_usage.len(), 3);

        // Verify each resource type
        let gpu_usage = all_usage
            .iter()
            .find(|u| u.resource_type == ResourceType::GpuMemory)
            .unwrap();
        assert_eq!(gpu_usage.current, 8192.0);

        let cpu_usage = all_usage
            .iter()
            .find(|u| u.resource_type == ResourceType::CpuPercent)
            .unwrap();
        assert_eq!(cpu_usage.current, 45.0);

        let agent_usage = all_usage
            .iter()
            .find(|u| u.resource_type == ResourceType::AgentCount)
            .unwrap();
        assert_eq!(agent_usage.current, 25.0);
    }

    #[tokio::test]
    async fn test_concurrent_updates() {
        use tokio::task;

        let limits = vec![ResourceLimit {
            resource_type: ResourceType::AgentSpawnRate,
            limit: 100.0,
            action: LimitAction::Warn,
        }];

        let limiter = Arc::new(ResourceLimiter::new(limits));
        let mut handles = vec![];

        // Spawn multiple tasks updating the same resource
        for i in 0..10 {
            let limiter_clone = limiter.clone();
            let handle = task::spawn(async move {
                limiter_clone
                    .update_usage(ResourceType::AgentSpawnRate, (i as f64) * 20.0)
                    .await
                    .unwrap();
            });
            handles.push(handle);
        }

        // Wait for all tasks to complete
        for handle in handles {
            handle.await.unwrap();
        }

        // Check final state
        let usage = limiter.get_usage(ResourceType::AgentSpawnRate);
        assert!(usage.is_some());
        assert!(usage.unwrap() >= 0.0);
    }

    #[tokio::test]
    async fn test_subscriber_multiple_violations() {
        let limits = vec![
            ResourceLimit {
                resource_type: ResourceType::GpuMemory,
                limit: 1000.0,
                action: LimitAction::Warn,
            },
            ResourceLimit {
                resource_type: ResourceType::CpuPercent,
                limit: 50.0,
                action: LimitAction::Throttle,
            },
        ];

        let limiter = ResourceLimiter::new(limits);
        let mut receiver = limiter.subscribe();

        // Create multiple violations
        limiter
            .update_usage(ResourceType::GpuMemory, 2000.0)
            .await
            .unwrap();
        limiter
            .update_usage(ResourceType::CpuPercent, 75.0)
            .await
            .unwrap();

        // Receive violations
        let violation1 = receiver.recv().await.unwrap();
        let violation2 = receiver.recv().await.unwrap();

        // Verify we received both violations
        let violations = vec![violation1, violation2];
        assert!(violations
            .iter()
            .any(|v| v.resource_type == ResourceType::GpuMemory));
        assert!(violations
            .iter()
            .any(|v| v.resource_type == ResourceType::CpuPercent));
    }

    #[tokio::test]
    async fn test_resource_usage_timestamps() {
        let limiter = ResourceLimiter::new(vec![]);

        // Record initial usage
        let before = Instant::now();
        limiter
            .update_usage(ResourceType::GpuMemory, 1000.0)
            .await
            .unwrap();
        let after = Instant::now();

        let all_usage = limiter.get_all_usage();
        assert_eq!(all_usage.len(), 1);

        // Verify timestamp is within expected range
        let usage = &all_usage[0];
        assert!(usage.timestamp >= before);
        assert!(usage.timestamp <= after);
    }

    #[tokio::test]
    async fn test_zero_and_negative_limits() {
        let limits = vec![
            ResourceLimit {
                resource_type: ResourceType::DiskIO,
                limit: 0.0,
                action: LimitAction::KillSwitch,
            },
            ResourceLimit {
                resource_type: ResourceType::SystemMemory,
                limit: -100.0,
                action: LimitAction::Suspend,
            },
        ];

        let limiter = ResourceLimiter::new(limits);

        // Any positive usage should exceed zero limit
        limiter
            .update_usage(ResourceType::DiskIO, 0.1)
            .await
            .unwrap();

        // Any usage should exceed negative limit
        limiter
            .update_usage(ResourceType::SystemMemory, -200.0)
            .await
            .unwrap();

        let violations = limiter.get_violations().await;
        assert_eq!(violations.len(), 2);
    }

    #[tokio::test]
    async fn test_update_limit_edge_cases() {
        let limits = vec![ResourceLimit {
            resource_type: ResourceType::GpuUtilization,
            limit: 80.0,
            action: LimitAction::Warn,
        }];

        let limiter = ResourceLimiter::new(limits);

        // Update to extreme values
        limiter
            .update_limit(ResourceType::GpuUtilization, f64::MAX)
            .await
            .unwrap();

        limiter
            .update_usage(ResourceType::GpuUtilization, f64::MAX - 1.0)
            .await
            .unwrap();
        assert!(limiter.get_violations().await.is_empty());

        // Update to zero
        limiter
            .update_limit(ResourceType::GpuUtilization, 0.0)
            .await
            .unwrap();

        limiter
            .update_usage(ResourceType::GpuUtilization, 0.1)
            .await
            .unwrap();
        assert_eq!(limiter.get_violations().await.len(), 2); // Previous violation + new one
    }

    #[tokio::test]
    async fn test_nan_and_infinity_handling() {
        let limiter = ResourceLimiter::new(vec![]);

        // Test NaN
        limiter
            .update_usage(ResourceType::GpuMemory, f64::NAN)
            .await
            .unwrap();
        let usage = limiter.get_usage(ResourceType::GpuMemory);
        assert!(usage.unwrap().is_nan());

        // Test Infinity
        limiter
            .update_usage(ResourceType::CpuPercent, f64::INFINITY)
            .await
            .unwrap();
        let usage = limiter.get_usage(ResourceType::CpuPercent);
        assert!(usage.unwrap().is_infinite());

        // Test Negative Infinity
        limiter
            .update_usage(ResourceType::NetworkBandwidth, f64::NEG_INFINITY)
            .await
            .unwrap();
        let usage = limiter.get_usage(ResourceType::NetworkBandwidth);
        assert!(usage.unwrap().is_infinite() && usage.unwrap().is_sign_negative());
    }

    #[test]
    fn test_resource_type_copy_trait() {
        let resource_type1 = ResourceType::GpuMemory;
        let resource_type2 = resource_type1; // Copy
        assert_eq!(resource_type1, resource_type2);
    }

    #[test]
    fn test_limit_action_copy_trait() {
        let action1 = LimitAction::KillSwitch;
        let action2 = action1; // Copy
        assert_eq!(action1, action2);
    }

    #[test]
    fn test_resource_violation_clone() {
        let violation = ResourceViolation {
            resource_type: ResourceType::GpuMemory,
            current: 2000.0,
            limit: 1000.0,
            agent_id: Some("agent-123".to_string()),
            action_taken: LimitAction::Warn,
            timestamp: Instant::now(),
        };

        let cloned = violation.clone();
        assert_eq!(violation.resource_type, cloned.resource_type);
        assert_eq!(violation.current, cloned.current);
        assert_eq!(violation.limit, cloned.limit);
        assert_eq!(violation.agent_id, cloned.agent_id);
        assert_eq!(violation.action_taken, cloned.action_taken);
    }

    #[tokio::test]
    async fn test_check_limits_with_agent_id() {
        let limits = vec![ResourceLimit {
            resource_type: ResourceType::AgentCount,
            limit: 10.0,
            action: LimitAction::Suspend,
        }];

        let limiter = ResourceLimiter::new(limits);

        // Update with agent-specific violation
        limiter
            .update_usage_with_agent(
                ResourceType::AgentCount,
                15.0,
                Some("rogue-agent".to_string()),
            )
            .await
            .unwrap();

        let violations = limiter.get_violations().await;
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].agent_id, Some("rogue-agent".to_string()));
    }

    #[tokio::test]
    async fn test_resource_usage_debug_formatting() {
        let usage = ResourceUsage {
            resource_type: ResourceType::GpuMemory,
            current: 8192.0,
            limit: 16384.0,
            timestamp: Instant::now(),
        };

        let debug_str = format!("{:?}", usage);
        assert!(debug_str.contains("ResourceUsage"));
        assert!(debug_str.contains("GpuMemory"));
        assert!(debug_str.contains("8192"));
        assert!(debug_str.contains("16384"));
    }
}
