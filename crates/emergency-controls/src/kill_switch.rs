//! Kill switch mechanisms for emergency agent termination

use crate::error::{EmergencyError, EmergencyResult};
use crate::resource_limits::ResourceType;
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::{broadcast, RwLock};
use tracing::{error, info, warn};
use uuid::Uuid;

/// Kill switch system for emergency agent termination
pub struct KillSwitchSystem {
    /// Global kill switch state
    global_kill: Arc<AtomicBool>,
    /// Per-agent kill switches
    agent_kill_switches: DashMap<String, AgentKillSwitch>,
    /// Kill switch triggers
    triggers: Arc<RwLock<Vec<KillSwitchTrigger>>>,
    /// Event broadcaster
    event_sender: broadcast::Sender<KillSwitchEvent>,
    /// System configuration
    config: Arc<RwLock<KillSwitchConfig>>,
    /// Kill switch metrics
    metrics: Arc<KillSwitchMetrics>,
}

/// Individual agent kill switch
#[derive(Debug, Clone)]
pub struct AgentKillSwitch {
    /// Agent ID
    pub agent_id: String,
    /// Kill switch activated
    pub activated: Arc<AtomicBool>,
    /// Activation time
    pub activated_at: Option<DateTime<Utc>>,
    /// Activation reason
    pub reason: Option<String>,
    /// Automatic reset time
    pub auto_reset_at: Option<DateTime<Utc>>,
}

/// Kill switch trigger configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KillSwitchTrigger {
    /// Trigger ID
    pub trigger_id: String,
    /// Trigger name
    pub name: String,
    /// Trigger type
    pub trigger_type: TriggerType,
    /// Enabled state
    pub enabled: bool,
    /// Trigger condition
    pub condition: TriggerCondition,
    /// Action to take
    pub action: KillSwitchAction,
    /// Priority level
    pub priority: TriggerPriority,
}

/// Kill switch event for broadcasting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KillSwitchEvent {
    /// Event ID
    pub event_id: String,
    /// Event type
    pub event_type: KillSwitchEventType,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Target (agent ID or "global")
    pub target: String,
    /// Trigger that caused the event
    pub trigger: Option<String>,
    /// Event details
    pub details: String,
}

/// Kill switch configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KillSwitchConfig {
    /// Enable global kill switch
    pub global_kill_enabled: bool,
    /// Auto-reset duration for agent kills
    pub agent_auto_reset_duration: Option<std::time::Duration>,
    /// Maximum concurrent kills
    pub max_concurrent_kills: u32,
    /// Kill confirmation required
    pub confirmation_required: bool,
    /// Audit all kill events
    pub audit_enabled: bool,
    /// Emergency contact notification
    pub notify_on_global_kill: bool,
    /// Pattern-based kill triggers
    pub patterns: Vec<String>,
    /// Auto-reset delay
    pub auto_reset_delay: Option<std::time::Duration>,
    /// Trigger conditions
    pub triggers: Vec<KillSwitchTrigger>,
}

/// Kill switch metrics
#[derive(Debug, Default)]
pub struct KillSwitchMetrics {
    /// Total global kills activated
    pub global_kills: AtomicU64,
    /// Total agent kills activated
    pub agent_kills: AtomicU64,
    /// Currently active kills
    pub active_kills: AtomicU64,
    /// Failed kill attempts
    pub failed_kills: AtomicU64,
    /// Auto-resets performed
    pub auto_resets: AtomicU64,
}

// Enums
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TriggerType {
    ResourceLimit,
    SafetyViolation,
    Manual,
    Timeout,
    HealthCheck,
    ExternalSignal,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TriggerCondition {
    /// Resource usage exceeds threshold
    ResourceThreshold { resource: String, threshold: f64 },
    /// Safety rule violated
    SafetyRuleViolation { rule_id: String },
    /// Agent unresponsive for duration
    Unresponsive { duration: std::time::Duration },
    /// Manual trigger
    ManualActivation { authorized_by: String },
    /// Health check failure
    HealthCheckFailure { check_name: String },
    /// External signal received
    ExternalSignal { signal: String },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum KillSwitchAction {
    /// Kill single agent
    KillAgent { agent_id: String },
    /// Kill all agents
    GlobalKill,
    /// Kill agents matching pattern
    PatternKill { pattern: String },
    /// Suspend agent execution
    SuspendAgent { agent_id: String },
    /// Isolate agent network
    IsolateAgent { agent_id: String },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum KillSwitchEventType {
    GlobalKillActivated,
    GlobalKillDeactivated,
    AgentKilled,
    AgentSuspended,
    AgentIsolated,
    KillFailed,
    AutoReset,
    TriggerActivated,
    Reset,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, PartialOrd)]
pub enum TriggerPriority {
    Low = 1,
    Medium = 2,
    High = 3,
    Critical = 4,
    Emergency = 5,
}

impl KillSwitchSystem {
    /// Create a new kill switch system with config
    pub fn new(config: KillSwitchConfig) -> Self {
        Self::with_config(config)
    }

    /// Create with specific configuration
    pub fn with_config(config: KillSwitchConfig) -> Self {
        let (event_sender, _) = broadcast::channel(1000);

        Self {
            global_kill: Arc::new(AtomicBool::new(false)),
            agent_kill_switches: DashMap::new(),
            triggers: Arc::new(RwLock::new(Vec::new())),
            event_sender,
            config: Arc::new(RwLock::new(config)),
            metrics: Arc::new(KillSwitchMetrics::default()),
        }
    }

    /// Activate global kill switch
    pub async fn activate_global_kill(&self, reason: &str) -> EmergencyResult<()> {
        let config = self.config.read().await;

        if !config.global_kill_enabled {
            return Err(EmergencyError::KillSwitchFailed {
                reason: "Global kill switch is disabled".to_string(),
            });
        }

        if config.confirmation_required {
            warn!("Global kill switch activation requested but confirmation required");
            // In production, this would wait for confirmation
        }

        // Set global kill flag
        self.global_kill.store(true, Ordering::SeqCst);
        self.metrics.global_kills.fetch_add(1, Ordering::Relaxed);

        // Kill all agents
        let agent_count = self.agent_kill_switches.len();
        for entry in self.agent_kill_switches.iter() {
            entry.value().activated.store(true, Ordering::SeqCst);
        }

        self.metrics
            .active_kills
            .store(agent_count as u64, Ordering::Relaxed);

        // Broadcast event
        let event = KillSwitchEvent {
            event_id: Uuid::new_v4().to_string(),
            event_type: KillSwitchEventType::GlobalKillActivated,
            timestamp: Utc::now(),
            target: "global".to_string(),
            trigger: None,
            details: reason.to_string(),
        };

        self.broadcast_event(event).await;

        if config.notify_on_global_kill {
            error!("EMERGENCY: Global kill switch activated - {}", reason);
        }

        info!(
            "Global kill switch activated: {} agents terminated",
            agent_count
        );
        Ok(())
    }

    /// Deactivate global kill switch
    pub async fn deactivate_global_kill(&self) -> EmergencyResult<()> {
        self.global_kill.store(false, Ordering::SeqCst);

        // Optionally reset agent kills
        for entry in self.agent_kill_switches.iter() {
            entry.value().activated.store(false, Ordering::SeqCst);
        }

        self.metrics.active_kills.store(0, Ordering::Relaxed);

        let event = KillSwitchEvent {
            event_id: Uuid::new_v4().to_string(),
            event_type: KillSwitchEventType::GlobalKillDeactivated,
            timestamp: Utc::now(),
            target: "global".to_string(),
            trigger: None,
            details: "Global kill switch deactivated".to_string(),
        };

        self.broadcast_event(event).await;

        info!("Global kill switch deactivated");
        Ok(())
    }

    /// Kill a specific agent
    pub async fn kill_agent(&self, agent_id: &str, reason: &str) -> EmergencyResult<()> {
        let config = self.config.read().await;

        // Check concurrent kill limit
        let active_kills = self.metrics.active_kills.load(Ordering::Relaxed);
        if active_kills >= config.max_concurrent_kills as u64 {
            return Err(EmergencyError::KillSwitchFailed {
                reason: format!(
                    "Maximum concurrent kills ({}) reached",
                    config.max_concurrent_kills
                ),
            });
        }

        // Create or update agent kill switch
        let mut kill_switch = self
            .agent_kill_switches
            .entry(agent_id.to_string())
            .or_insert_with(|| AgentKillSwitch {
                agent_id: agent_id.to_string(),
                activated: Arc::new(AtomicBool::new(false)),
                activated_at: None,
                reason: None,
                auto_reset_at: None,
            });

        // Activate kill switch
        kill_switch.activated.store(true, Ordering::SeqCst);
        kill_switch.activated_at = Some(Utc::now());
        kill_switch.reason = Some(reason.to_string());

        // Set auto-reset if configured
        if let Some(duration) = config.agent_auto_reset_duration {
            kill_switch.auto_reset_at =
                Some(Utc::now() + chrono::Duration::from_std(duration).unwrap());
        }

        self.metrics.agent_kills.fetch_add(1, Ordering::Relaxed);
        self.metrics.active_kills.fetch_add(1, Ordering::Relaxed);

        // Broadcast event
        let event = KillSwitchEvent {
            event_id: Uuid::new_v4().to_string(),
            event_type: KillSwitchEventType::AgentKilled,
            timestamp: Utc::now(),
            target: agent_id.to_string(),
            trigger: None,
            details: reason.to_string(),
        };

        self.broadcast_event(event).await;

        info!("Agent {} kill switch activated: {}", agent_id, reason);
        Ok(())
    }

    /// Check if global kill is active
    pub fn is_global_kill_active(&self) -> bool {
        self.global_kill.load(Ordering::SeqCst)
    }

    /// Check if agent is killed
    pub fn is_agent_killed(&self, agent_id: &str) -> bool {
        if self.is_global_kill_active() {
            return true;
        }

        self.agent_kill_switches
            .get(agent_id)
            .map(|ks| ks.activated.load(Ordering::SeqCst))
            .unwrap_or(false)
    }

    /// Add a kill switch trigger
    pub async fn add_trigger(&self, trigger: KillSwitchTrigger) -> EmergencyResult<()> {
        let mut triggers = self.triggers.write().await;

        // Check for duplicate trigger
        if triggers.iter().any(|t| t.trigger_id == trigger.trigger_id) {
            return Err(EmergencyError::ConfigurationError(format!(
                "Trigger {} already exists",
                trigger.trigger_id
            )));
        }

        triggers.push(trigger.clone());

        info!(
            "Added kill switch trigger: {} ({})",
            trigger.name, trigger.trigger_id
        );
        Ok(())
    }

    /// Remove a kill switch trigger
    pub async fn remove_trigger(&self, trigger_id: &str) -> EmergencyResult<()> {
        let mut triggers = self.triggers.write().await;

        let initial_len = triggers.len();
        triggers.retain(|t| t.trigger_id != trigger_id);

        if triggers.len() == initial_len {
            return Err(EmergencyError::ConfigurationError(format!(
                "Trigger {trigger_id} not found"
            )));
        }

        info!("Removed kill switch trigger: {}", trigger_id);
        Ok(())
    }

    /// Evaluate triggers and execute actions
    pub async fn evaluate_triggers(
        &self,
        context: &TriggerContext,
    ) -> EmergencyResult<Vec<String>> {
        let triggers = self.triggers.read().await;
        let mut activated_triggers = Vec::new();

        for trigger in triggers.iter() {
            if !trigger.enabled {
                continue;
            }

            if self.evaluate_condition(&trigger.condition, context) {
                match &trigger.action {
                    KillSwitchAction::GlobalKill => {
                        self.activate_global_kill(&format!("Trigger: {}", trigger.name))
                            .await?;
                    }
                    KillSwitchAction::KillAgent { agent_id } => {
                        self.kill_agent(agent_id, &format!("Trigger: {}", trigger.name))
                            .await?;
                    }
                    KillSwitchAction::SuspendAgent { agent_id } => {
                        self.suspend_agent(agent_id, &format!("Trigger: {}", trigger.name))
                            .await?;
                    }
                    KillSwitchAction::IsolateAgent { agent_id } => {
                        self.isolate_agent(agent_id, &format!("Trigger: {}", trigger.name))
                            .await?;
                    }
                    KillSwitchAction::PatternKill { pattern } => {
                        self.pattern_kill(pattern, &format!("Trigger: {}", trigger.name))
                            .await?;
                    }
                }

                activated_triggers.push(trigger.trigger_id.clone());

                let event = KillSwitchEvent {
                    event_id: Uuid::new_v4().to_string(),
                    event_type: KillSwitchEventType::TriggerActivated,
                    timestamp: Utc::now(),
                    target: format!("trigger:{}", trigger.trigger_id),
                    trigger: Some(trigger.trigger_id.clone()),
                    details: format!("Trigger {} activated", trigger.name),
                };

                self.broadcast_event(event).await;
            }
        }

        Ok(activated_triggers)
    }

    /// Process auto-resets
    pub async fn process_auto_resets(&self) -> EmergencyResult<u32> {
        let now = Utc::now();
        let mut reset_count = 0;

        for mut entry in self.agent_kill_switches.iter_mut() {
            let kill_switch = entry.value_mut();

            if let Some(reset_time) = kill_switch.auto_reset_at {
                if now >= reset_time && kill_switch.activated.load(Ordering::SeqCst) {
                    kill_switch.activated.store(false, Ordering::SeqCst);
                    kill_switch.activated_at = None;
                    kill_switch.reason = None;
                    kill_switch.auto_reset_at = None;

                    reset_count += 1;
                    self.metrics.auto_resets.fetch_add(1, Ordering::Relaxed);
                    self.metrics.active_kills.fetch_sub(1, Ordering::Relaxed);

                    let event = KillSwitchEvent {
                        event_id: Uuid::new_v4().to_string(),
                        event_type: KillSwitchEventType::AutoReset,
                        timestamp: Utc::now(),
                        target: kill_switch.agent_id.clone(),
                        trigger: None,
                        details: "Auto-reset after timeout".to_string(),
                    };

                    self.broadcast_event(event).await;
                }
            }
        }

        if reset_count > 0 {
            info!("Auto-reset {} agent kill switches", reset_count);
        }

        Ok(reset_count)
    }

    /// Subscribe to kill switch events
    pub fn subscribe(&self) -> broadcast::Receiver<KillSwitchEvent> {
        self.event_sender.subscribe()
    }

    /// Get kill switch metrics
    pub fn get_metrics(&self) -> KillSwitchMetrics {
        KillSwitchMetrics {
            global_kills: AtomicU64::new(self.metrics.global_kills.load(Ordering::Relaxed)),
            agent_kills: AtomicU64::new(self.metrics.agent_kills.load(Ordering::Relaxed)),
            active_kills: AtomicU64::new(self.metrics.active_kills.load(Ordering::Relaxed)),
            failed_kills: AtomicU64::new(self.metrics.failed_kills.load(Ordering::Relaxed)),
            auto_resets: AtomicU64::new(self.metrics.auto_resets.load(Ordering::Relaxed)),
        }
    }

    /// Update configuration
    pub async fn update_config(&self, config: KillSwitchConfig) -> EmergencyResult<()> {
        let mut current_config = self.config.write().await;
        *current_config = config;

        info!("Kill switch configuration updated");
        Ok(())
    }

    /// Get active kill switches
    pub fn get_active_kills(&self) -> Vec<AgentKillSwitch> {
        self.agent_kill_switches
            .iter()
            .filter(|entry| entry.value().activated.load(Ordering::SeqCst))
            .map(|entry| entry.value().clone())
            .collect()
    }

    // Private helper methods

    async fn suspend_agent(&self, agent_id: &str, reason: &str) -> EmergencyResult<()> {
        // In a real implementation, this would suspend agent execution
        let event = KillSwitchEvent {
            event_id: Uuid::new_v4().to_string(),
            event_type: KillSwitchEventType::AgentSuspended,
            timestamp: Utc::now(),
            target: agent_id.to_string(),
            trigger: None,
            details: reason.to_string(),
        };

        self.broadcast_event(event).await;
        info!("Agent {} suspended: {}", agent_id, reason);
        Ok(())
    }

    async fn isolate_agent(&self, agent_id: &str, reason: &str) -> EmergencyResult<()> {
        // In a real implementation, this would isolate agent network
        let event = KillSwitchEvent {
            event_id: Uuid::new_v4().to_string(),
            event_type: KillSwitchEventType::AgentIsolated,
            timestamp: Utc::now(),
            target: agent_id.to_string(),
            trigger: None,
            details: reason.to_string(),
        };

        self.broadcast_event(event).await;
        info!("Agent {} isolated: {}", agent_id, reason);
        Ok(())
    }

    pub async fn pattern_kill(&self, pattern: &str, reason: &str) -> EmergencyResult<()> {
        // Collect matching agent IDs first to avoid holding iterator while modifying
        let matching_agents: Vec<String> = self
            .agent_kill_switches
            .iter()
            .filter(|entry| entry.key().contains(pattern))
            .map(|entry| entry.key().clone())
            .collect();

        let mut killed_count = 0;
        for agent_id in matching_agents {
            self.kill_agent(&agent_id, reason).await?;
            killed_count += 1;
        }

        info!(
            "Pattern kill '{}' terminated {} agents",
            pattern, killed_count
        );
        Ok(())
    }

    fn evaluate_condition(&self, condition: &TriggerCondition, context: &TriggerContext) -> bool {
        match condition {
            TriggerCondition::ResourceThreshold {
                resource,
                threshold,
            } => context
                .resource_usage
                .get(resource)
                .map(|&usage| usage > *threshold)
                .unwrap_or(false),
            TriggerCondition::SafetyRuleViolation { rule_id } => {
                context.safety_violations.contains(rule_id)
            }
            TriggerCondition::Unresponsive { duration } => context
                .agent_unresponsive_duration
                .map(|d| d >= *duration)
                .unwrap_or(false),
            TriggerCondition::ManualActivation { authorized_by } => {
                context.manual_trigger_by.as_ref() == Some(authorized_by)
            }
            TriggerCondition::HealthCheckFailure { check_name } => {
                context.failed_health_checks.contains(check_name)
            }
            TriggerCondition::ExternalSignal { signal } => {
                context.external_signals.contains(signal)
            }
        }
    }

    /// Activate kill switch with reason
    pub async fn activate(&mut self, reason: &str) -> EmergencyResult<()> {
        self.activate_global_kill(reason).await?;

        // Start auto-reset timer if configured
        if let Some(delay) = self.config.read().await.auto_reset_delay {
            let kill_switch = self.clone();
            tokio::spawn(async move {
                tokio::time::sleep(delay).await;
                if let Err(e) = kill_switch.reset().await {
                    error!("Auto-reset failed: {}", e);
                }
            });
        }

        Ok(())
    }

    /// Reset kill switch
    pub async fn reset(&self) -> EmergencyResult<()> {
        self.global_kill.store(false, Ordering::SeqCst);

        // Reset all agent kill switches
        for entry in self.agent_kill_switches.iter() {
            entry.value().activated.store(false, Ordering::SeqCst);
        }

        self.metrics.active_kills.store(0, Ordering::Relaxed);
        self.metrics.auto_resets.fetch_add(1, Ordering::Relaxed);

        let event = KillSwitchEvent {
            event_id: Uuid::new_v4().to_string(),
            event_type: KillSwitchEventType::Reset,
            timestamp: Utc::now(),
            target: "global".to_string(),
            trigger: None,
            details: "Kill switch reset".to_string(),
        };

        self.broadcast_event(event).await;
        info!("Kill switch reset");
        Ok(())
    }

    /// Check if kill switch is active
    pub fn is_active(&self) -> bool {
        self.global_kill.load(Ordering::SeqCst)
    }

    /// Check resource trigger
    pub async fn check_resource_trigger(
        &mut self,
        resource: ResourceType,
        value: f64,
    ) -> EmergencyResult<()> {
        let should_activate = {
            let config = self.config.read().await;

            config
                .triggers
                .iter()
                .find(|trigger| {
                    if trigger.enabled {
                        if let TriggerCondition::ResourceThreshold {
                            resource: r,
                            threshold,
                        } = &trigger.condition
                        {
                            r == &format!("{resource:?}") && value > *threshold
                        } else {
                            false
                        }
                    } else {
                        false
                    }
                })
                .map(|trigger| {
                    if let TriggerCondition::ResourceThreshold {
                        resource: r,
                        threshold,
                    } = &trigger.condition
                    {
                        format!("Resource {r} exceeded threshold: {value} > {threshold}")
                    } else {
                        String::new()
                    }
                })
        };

        if let Some(reason) = should_activate {
            self.activate(&reason).await?;
        }

        Ok(())
    }

    /// Check pattern trigger
    pub async fn check_pattern_trigger(&mut self, text: &str) -> EmergencyResult<()> {
        let matched_pattern = {
            let config = self.config.read().await;

            // Check simple patterns first
            config
                .patterns
                .iter()
                .find(|pattern| text.contains(pattern.as_str()))
                .cloned()
        };

        if let Some(pattern) = matched_pattern {
            self.activate(&format!("Pattern '{pattern}' matched in '{text}'"))
                .await?;
        }

        Ok(())
    }

    /// Clone for async operations
    fn clone(&self) -> Self {
        Self {
            global_kill: self.global_kill.clone(),
            agent_kill_switches: self.agent_kill_switches.clone(),
            triggers: self.triggers.clone(),
            event_sender: self.event_sender.clone(),
            config: self.config.clone(),
            metrics: self.metrics.clone(),
        }
    }

    async fn broadcast_event(&self, event: KillSwitchEvent) {
        if self.event_sender.send(event.clone()).is_err() {
            warn!("No subscribers for kill switch event: {}", event.event_id);
        }
    }
}

/// Context for trigger evaluation
#[derive(Debug, Clone, Default)]
pub struct TriggerContext {
    /// Current resource usage
    pub resource_usage: std::collections::HashMap<String, f64>,
    /// Active safety violations
    pub safety_violations: Vec<String>,
    /// Agent unresponsive duration
    pub agent_unresponsive_duration: Option<std::time::Duration>,
    /// Manual trigger authorized by
    pub manual_trigger_by: Option<String>,
    /// Failed health checks
    pub failed_health_checks: Vec<String>,
    /// External signals received
    pub external_signals: Vec<String>,
}

impl Default for KillSwitchConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl KillSwitchConfig {
    fn new() -> Self {
        Self {
            global_kill_enabled: true,
            agent_auto_reset_duration: Some(std::time::Duration::from_secs(300)), // 5 minutes
            max_concurrent_kills: 1000,
            confirmation_required: false,
            audit_enabled: true,
            notify_on_global_kill: true,
            patterns: vec![],
            auto_reset_delay: None,
            triggers: vec![],
        }
    }
}

impl Default for KillSwitchSystem {
    fn default() -> Self {
        Self::new(KillSwitchConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn create_test_system() -> KillSwitchSystem {
        KillSwitchSystem::new(KillSwitchConfig::default())
    }

    #[tokio::test]
    async fn test_kill_switch_creation() {
        let system = create_test_system();
        assert!(!system.is_global_kill_active());
        assert_eq!(system.agent_kill_switches.len(), 0);

        let metrics = system.get_metrics();
        assert_eq!(metrics.global_kills.load(Ordering::Relaxed), 0);
        assert_eq!(metrics.agent_kills.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn test_global_kill_activation() {
        let system = create_test_system();

        // Add some agents
        system.agent_kill_switches.insert(
            "agent-1".to_string(),
            AgentKillSwitch {
                agent_id: "agent-1".to_string(),
                activated: Arc::new(AtomicBool::new(false)),
                activated_at: None,
                reason: None,
                auto_reset_at: None,
            },
        );
        system.agent_kill_switches.insert(
            "agent-2".to_string(),
            AgentKillSwitch {
                agent_id: "agent-2".to_string(),
                activated: Arc::new(AtomicBool::new(false)),
                activated_at: None,
                reason: None,
                auto_reset_at: None,
            },
        );

        assert!(system
            .activate_global_kill("Emergency shutdown")
            .await
            .is_ok());
        assert!(system.is_global_kill_active());

        // Check all agents are killed
        assert!(system.is_agent_killed("agent-1"));
        assert!(system.is_agent_killed("agent-2"));
        assert!(system.is_agent_killed("any-agent")); // Global kill affects all

        let metrics = system.get_metrics();
        assert_eq!(metrics.global_kills.load(Ordering::Relaxed), 1);
        assert_eq!(metrics.active_kills.load(Ordering::Relaxed), 2);
    }

    #[tokio::test]
    async fn test_global_kill_deactivation() {
        let system = create_test_system();

        system.activate_global_kill("Test").await.unwrap();
        assert!(system.is_global_kill_active());

        assert!(system.deactivate_global_kill().await.is_ok());
        assert!(!system.is_global_kill_active());

        let metrics = system.get_metrics();
        assert_eq!(metrics.active_kills.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn test_global_kill_disabled() {
        let system = create_test_system();

        let mut config = KillSwitchConfig::default();
        config.global_kill_enabled = false;
        system.update_config(config).await.unwrap();

        let result = system.activate_global_kill("Test").await;
        assert!(result.is_err());
        assert!(!system.is_global_kill_active());
    }

    #[tokio::test]
    async fn test_individual_agent_kill() {
        let system = create_test_system();

        assert!(system
            .kill_agent("agent-123", "Resource limit exceeded")
            .await
            .is_ok());
        assert!(system.is_agent_killed("agent-123"));
        assert!(!system.is_agent_killed("agent-456")); // Other agents not affected

        let metrics = system.get_metrics();
        assert_eq!(metrics.agent_kills.load(Ordering::Relaxed), 1);
        assert_eq!(metrics.active_kills.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn test_concurrent_kill_limit() {
        let system = create_test_system();

        let mut config = KillSwitchConfig::default();
        config.max_concurrent_kills = 2;
        system.update_config(config).await.unwrap();

        // Kill 2 agents (at limit)
        system.kill_agent("agent-1", "Test").await.unwrap();
        system.kill_agent("agent-2", "Test").await.unwrap();

        // Third kill should fail
        let result = system.kill_agent("agent-3", "Test").await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Maximum concurrent kills"));
    }

    #[tokio::test]
    async fn test_auto_reset() {
        let system = create_test_system();

        let mut config = KillSwitchConfig::default();
        config.agent_auto_reset_duration = Some(Duration::from_millis(100));
        system.update_config(config).await.unwrap();

        system.kill_agent("agent-1", "Test").await.unwrap();
        assert!(system.is_agent_killed("agent-1"));

        // Wait for auto-reset
        tokio::time::sleep(Duration::from_millis(150)).await;

        let reset_count = system.process_auto_resets().await.unwrap();
        assert_eq!(reset_count, 1);
        assert!(!system.is_agent_killed("agent-1"));

        let metrics = system.get_metrics();
        assert_eq!(metrics.auto_resets.load(Ordering::Relaxed), 1);
        assert_eq!(metrics.active_kills.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn test_trigger_management() {
        let system = create_test_system();

        let trigger = KillSwitchTrigger {
            trigger_id: "trigger-1".to_string(),
            name: "High Memory Usage".to_string(),
            trigger_type: TriggerType::ResourceLimit,
            enabled: true,
            condition: TriggerCondition::ResourceThreshold {
                resource: "memory".to_string(),
                threshold: 90.0,
            },
            action: KillSwitchAction::GlobalKill,
            priority: TriggerPriority::High,
        };

        assert!(system.add_trigger(trigger.clone()).await.is_ok());

        // Duplicate trigger should fail
        assert!(system.add_trigger(trigger).await.is_err());

        // Remove trigger
        assert!(system.remove_trigger("trigger-1").await.is_ok());

        // Remove non-existent trigger should fail
        assert!(system.remove_trigger("trigger-1").await.is_err());
    }

    #[tokio::test]
    async fn test_trigger_evaluation_resource_threshold() {
        let system = create_test_system();

        let trigger = KillSwitchTrigger {
            trigger_id: "mem-trigger".to_string(),
            name: "Memory Limit".to_string(),
            trigger_type: TriggerType::ResourceLimit,
            enabled: true,
            condition: TriggerCondition::ResourceThreshold {
                resource: "memory".to_string(),
                threshold: 80.0,
            },
            action: KillSwitchAction::KillAgent {
                agent_id: "agent-1".to_string(),
            },
            priority: TriggerPriority::High,
        };

        system.add_trigger(trigger).await.unwrap();

        // Below threshold - no activation
        let mut context = TriggerContext::default();
        context.resource_usage.insert("memory".to_string(), 70.0);

        let activated = system.evaluate_triggers(&context).await.unwrap();
        assert_eq!(activated.len(), 0);
        assert!(!system.is_agent_killed("agent-1"));

        // Above threshold - activation
        context.resource_usage.insert("memory".to_string(), 85.0);

        let activated = system.evaluate_triggers(&context).await.unwrap();
        assert_eq!(activated.len(), 1);
        assert!(system.is_agent_killed("agent-1"));
    }

    #[tokio::test]
    async fn test_trigger_evaluation_safety_violation() {
        let system = create_test_system();

        let trigger = KillSwitchTrigger {
            trigger_id: "safety-trigger".to_string(),
            name: "Safety Violation".to_string(),
            trigger_type: TriggerType::SafetyViolation,
            enabled: true,
            condition: TriggerCondition::SafetyRuleViolation {
                rule_id: "memory-access-violation".to_string(),
            },
            action: KillSwitchAction::GlobalKill,
            priority: TriggerPriority::Critical,
        };

        system.add_trigger(trigger).await.unwrap();

        let mut context = TriggerContext::default();
        context
            .safety_violations
            .push("memory-access-violation".to_string());

        let activated = system.evaluate_triggers(&context).await.unwrap();
        assert_eq!(activated.len(), 1);
        assert!(system.is_global_kill_active());
    }

    #[tokio::test]
    async fn test_trigger_evaluation_unresponsive() {
        let system = create_test_system();

        let trigger = KillSwitchTrigger {
            trigger_id: "unresponsive-trigger".to_string(),
            name: "Agent Unresponsive".to_string(),
            trigger_type: TriggerType::Timeout,
            enabled: true,
            condition: TriggerCondition::Unresponsive {
                duration: Duration::from_secs(30),
            },
            action: KillSwitchAction::SuspendAgent {
                agent_id: "agent-1".to_string(),
            },
            priority: TriggerPriority::Medium,
        };

        system.add_trigger(trigger).await.unwrap();

        // Not unresponsive long enough
        let mut context = TriggerContext::default();
        context.agent_unresponsive_duration = Some(Duration::from_secs(20));

        let activated = system.evaluate_triggers(&context).await.unwrap();
        assert_eq!(activated.len(), 0);

        // Unresponsive long enough
        context.agent_unresponsive_duration = Some(Duration::from_secs(35));

        let activated = system.evaluate_triggers(&context).await.unwrap();
        assert_eq!(activated.len(), 1);
    }

    #[tokio::test]
    async fn test_pattern_kill() {
        let system = create_test_system();

        // Add agents with pattern
        system.kill_agent("test-agent-1", "Setup").await.unwrap();
        system.kill_agent("test-agent-2", "Setup").await.unwrap();
        system.kill_agent("prod-agent-1", "Setup").await.unwrap();

        // Reset all
        system.deactivate_global_kill().await.unwrap();

        // Pattern kill "test-"
        system
            .pattern_kill("test-", "Pattern kill test agents")
            .await
            .unwrap();

        assert!(system.is_agent_killed("test-agent-1"));
        assert!(system.is_agent_killed("test-agent-2"));
        assert!(!system.is_agent_killed("prod-agent-1"));
    }

    #[tokio::test]
    async fn test_event_subscription() {
        let system = create_test_system();
        let mut receiver = system.subscribe();

        system.kill_agent("agent-1", "Test event").await.unwrap();

        // Should receive kill event
        let event = receiver.try_recv().unwrap();
        assert_eq!(event.event_type, KillSwitchEventType::AgentKilled);
        assert_eq!(event.target, "agent-1");
    }

    #[tokio::test]
    async fn test_get_active_kills() {
        let system = create_test_system();

        system.kill_agent("agent-1", "Test").await.unwrap();
        system.kill_agent("agent-2", "Test").await.unwrap();
        system.kill_agent("agent-3", "Test").await.unwrap();

        // Reset one
        system
            .agent_kill_switches
            .get_mut("agent-2")
            .unwrap()
            .activated
            .store(false, Ordering::SeqCst);

        let active_kills = system.get_active_kills();
        assert_eq!(active_kills.len(), 2);
        assert!(active_kills.iter().any(|ks| ks.agent_id == "agent-1"));
        assert!(active_kills.iter().any(|ks| ks.agent_id == "agent-3"));
        assert!(!active_kills.iter().any(|ks| ks.agent_id == "agent-2"));
    }

    #[test]
    fn test_trigger_priority_ordering() {
        assert!(TriggerPriority::Emergency > TriggerPriority::Critical);
        assert!(TriggerPriority::Critical > TriggerPriority::High);
        assert!(TriggerPriority::High > TriggerPriority::Medium);
        assert!(TriggerPriority::Medium > TriggerPriority::Low);
    }

    #[test]
    fn test_kill_switch_config_default() {
        let config = KillSwitchConfig::default();
        assert!(config.global_kill_enabled);
        assert_eq!(config.max_concurrent_kills, 1000);
        assert!(!config.confirmation_required);
        assert!(config.audit_enabled);
        assert!(config.notify_on_global_kill);
    }

    #[test]
    fn test_trigger_serialization() {
        let trigger = KillSwitchTrigger {
            trigger_id: "test".to_string(),
            name: "Test Trigger".to_string(),
            trigger_type: TriggerType::ResourceLimit,
            enabled: true,
            condition: TriggerCondition::ResourceThreshold {
                resource: "cpu".to_string(),
                threshold: 95.0,
            },
            action: KillSwitchAction::GlobalKill,
            priority: TriggerPriority::High,
        };

        let serialized = serde_json::to_string(&trigger).unwrap();
        let deserialized: KillSwitchTrigger = serde_json::from_str(&serialized).unwrap();

        assert_eq!(trigger.trigger_id, deserialized.trigger_id);
        assert_eq!(trigger.name, deserialized.name);
        assert_eq!(trigger.trigger_type, deserialized.trigger_type);
    }

    #[test]
    fn test_event_serialization() {
        let event = KillSwitchEvent {
            event_id: "event-123".to_string(),
            event_type: KillSwitchEventType::GlobalKillActivated,
            timestamp: Utc::now(),
            target: "global".to_string(),
            trigger: Some("trigger-1".to_string()),
            details: "Test event".to_string(),
        };

        let serialized = serde_json::to_string(&event).unwrap();
        let deserialized: KillSwitchEvent = serde_json::from_str(&serialized).unwrap();

        assert_eq!(event.event_id, deserialized.event_id);
        assert_eq!(event.event_type, deserialized.event_type);
        assert_eq!(event.target, deserialized.target);
    }
}
