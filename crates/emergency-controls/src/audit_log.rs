//! Audit logging for emergency control system
//!
//! Provides comprehensive logging for:
//! - All emergency control actions
//! - Safety violations
//! - Resource limit breaches
//! - Kill switch activations
//! - Recovery procedures

use crate::{EmergencyError, EmergencyResult};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::fs::{self, OpenOptions};
use tokio::io::AsyncWriteExt;
use tokio::sync::{mpsc, RwLock};
use tracing::error;

/// Audit event severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AuditLevel {
    Debug,
    Info,
    Warning,
    Error,
    Critical,
}

/// Types of audit events
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AuditEventType {
    /// Kill switch activation
    KillSwitchActivated,
    /// Kill switch deactivation
    KillSwitchDeactivated,
    /// Resource limit exceeded
    ResourceLimitExceeded,
    /// Safety violation detected
    SafetyViolation,
    /// Agent suspended
    AgentSuspended,
    /// Agent terminated
    AgentTerminated,
    /// Recovery initiated
    RecoveryInitiated,
    /// Recovery completed
    RecoveryCompleted,
    /// Configuration changed
    ConfigurationChanged,
    /// System startup
    SystemStartup,
    /// System shutdown
    SystemShutdown,
    /// Manual override
    ManualOverride,
}

/// Audit event record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEvent {
    pub id: String,
    pub timestamp: DateTime<Utc>,
    pub event_type: AuditEventType,
    pub level: AuditLevel,
    pub agent_id: Option<String>,
    pub message: String,
    pub details: serde_json::Value,
    pub user: Option<String>,
    pub source: String,
}

impl AuditEvent {
    /// Create new audit event
    pub fn new(event_type: AuditEventType, level: AuditLevel, message: String) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
            event_type,
            level,
            agent_id: None,
            message,
            details: serde_json::json!({}),
            user: None,
            source: "emergency-controls".to_string(),
        }
    }

    /// Set agent ID
    pub fn with_agent(mut self, agent_id: String) -> Self {
        self.agent_id = Some(agent_id);
        self
    }

    /// Set details
    pub fn with_details(mut self, details: serde_json::Value) -> Self {
        self.details = details;
        self
    }

    /// Set user
    pub fn with_user(mut self, user: String) -> Self {
        self.user = Some(user);
        self
    }

    /// Set source
    pub fn with_source(mut self, source: String) -> Self {
        self.source = source;
        self
    }
}

/// Audit logger configuration
#[derive(Debug, Clone)]
pub struct AuditConfig {
    pub log_path: PathBuf,
    pub max_file_size: u64,
    pub max_files: usize,
    pub buffer_size: usize,
    pub min_level: AuditLevel,
}

impl Default for AuditConfig {
    fn default() -> Self {
        Self {
            log_path: PathBuf::from("/var/log/exorust/emergency-audit.log"),
            max_file_size: 100 * 1024 * 1024, // 100MB
            max_files: 10,
            buffer_size: 1000,
            min_level: AuditLevel::Info,
        }
    }
}

/// Audit logger system
pub struct AuditLogger {
    config: Arc<RwLock<AuditConfig>>,
    sender: mpsc::Sender<AuditEvent>,
    events_buffer: Arc<RwLock<Vec<AuditEvent>>>,
}

impl AuditLogger {
    /// Create new audit logger
    pub async fn new(config: AuditConfig) -> EmergencyResult<Self> {
        // Ensure log directory exists
        if let Some(parent) = config.log_path.parent() {
            fs::create_dir_all(parent)
                .await
                .map_err(|e| EmergencyError::AuditLogError {
                    operation: "create_dir".to_string(),
                    reason: e.to_string(),
                })?;
        }

        let (sender, mut receiver) = mpsc::channel(config.buffer_size);
        let events_buffer = Arc::new(RwLock::new(Vec::new()));

        let logger = Self {
            config: Arc::new(RwLock::new(config.clone())),
            sender,
            events_buffer: events_buffer.clone(),
        };

        // Spawn background writer task
        let config_clone = logger.config.clone();
        tokio::spawn(async move {
            while let Some(event) = receiver.recv().await {
                if let Err(e) = Self::write_event(&config_clone, &event).await {
                    error!("Failed to write audit event: {}", e);
                }

                // Also store in memory buffer
                let mut buffer = events_buffer.write().await;
                buffer.push(event);
                if buffer.len() > 10000 {
                    buffer.drain(0..5000); // Keep latest 5000 events
                }
            }
        });

        Ok(logger)
    }

    /// Log an audit event
    pub async fn log(&self, event: AuditEvent) -> EmergencyResult<()> {
        let config = self.config.read().await;

        // Check minimum level
        if event.level < config.min_level {
            return Ok(());
        }

        // Send to writer task
        self.sender
            .send(event)
            .await
            .map_err(|e| EmergencyError::AuditLogError {
                operation: "send_event".to_string(),
                reason: e.to_string(),
            })?;

        Ok(())
    }

    /// Write event to file
    async fn write_event(
        config: &Arc<RwLock<AuditConfig>>,
        event: &AuditEvent,
    ) -> EmergencyResult<()> {
        let config = config.read().await;

        // Check if rotation needed
        if let Ok(metadata) = fs::metadata(&config.log_path).await {
            if metadata.len() >= config.max_file_size {
                Self::rotate_logs(&config).await?;
            }
        }

        // Write event
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&config.log_path)
            .await
            .map_err(|e| EmergencyError::AuditLogError {
                operation: "open_file".to_string(),
                reason: e.to_string(),
            })?;

        let json = serde_json::to_string(&event).map_err(|e| EmergencyError::AuditLogError {
            operation: "serialize_event".to_string(),
            reason: e.to_string(),
        })?;

        file.write_all(format!("{json}\n").as_bytes())
            .await
            .map_err(|e| EmergencyError::AuditLogError {
                operation: "write_event".to_string(),
                reason: e.to_string(),
            })?;

        file.flush()
            .await
            .map_err(|e| EmergencyError::AuditLogError {
                operation: "flush_file".to_string(),
                reason: e.to_string(),
            })?;

        Ok(())
    }

    /// Rotate log files
    async fn rotate_logs(config: &AuditConfig) -> EmergencyResult<()> {
        // Rename existing files
        for i in (1..config.max_files).rev() {
            let old_path = if i == 1 {
                config.log_path.clone()
            } else {
                config.log_path.with_extension(format!("log.{}", i - 1))
            };

            let new_path = config.log_path.with_extension(format!("log.{i}"));

            if old_path.exists() {
                fs::rename(&old_path, &new_path).await.map_err(|e| {
                    EmergencyError::AuditLogError {
                        operation: "rotate_logs".to_string(),
                        reason: e.to_string(),
                    }
                })?;
            }
        }

        // Delete oldest file if exists
        let oldest = config
            .log_path
            .with_extension(format!("log.{}", config.max_files));
        if oldest.exists() {
            fs::remove_file(&oldest)
                .await
                .map_err(|e| EmergencyError::AuditLogError {
                    operation: "delete_old_log".to_string(),
                    reason: e.to_string(),
                })?;
        }

        Ok(())
    }

    /// Query events from memory buffer
    pub async fn query_events<F>(&self, filter: F) -> Vec<AuditEvent>
    where
        F: Fn(&AuditEvent) -> bool,
    {
        self.events_buffer
            .read()
            .await
            .iter()
            .filter(|e| filter(e))
            .cloned()
            .collect()
    }

    /// Get events by type
    pub async fn get_events_by_type(&self, event_type: AuditEventType) -> Vec<AuditEvent> {
        self.query_events(|e| e.event_type == event_type).await
    }

    /// Get events by agent
    pub async fn get_events_by_agent(&self, agent_id: &str) -> Vec<AuditEvent> {
        let agent_id = agent_id.to_string();
        self.query_events(|e| e.agent_id.as_ref() == Some(&agent_id))
            .await
    }

    /// Get events by level
    pub async fn get_events_by_level(&self, min_level: AuditLevel) -> Vec<AuditEvent> {
        self.query_events(|e| e.level >= min_level).await
    }

    /// Get recent events
    pub async fn get_recent_events(&self, count: usize) -> Vec<AuditEvent> {
        let buffer = self.events_buffer.read().await;
        let start = buffer.len().saturating_sub(count);
        buffer[start..].to_vec()
    }

    /// Clear memory buffer
    pub async fn clear_buffer(&self) {
        self.events_buffer.write().await.clear();
    }

    /// Update configuration
    pub async fn update_config(&self, config: AuditConfig) -> EmergencyResult<()> {
        *self.config.write().await = config;

        // Log configuration change
        let event = AuditEvent::new(
            AuditEventType::ConfigurationChanged,
            AuditLevel::Info,
            "Audit logger configuration updated".to_string(),
        );
        self.log(event).await?;

        Ok(())
    }

    /// Log kill switch activation
    pub async fn log_kill_switch_activated(
        &self,
        agent_id: Option<String>,
        reason: &str,
    ) -> EmergencyResult<()> {
        let mut event = AuditEvent::new(
            AuditEventType::KillSwitchActivated,
            AuditLevel::Critical,
            format!("Kill switch activated: {reason}"),
        );

        if let Some(id) = agent_id {
            event = event.with_agent(id);
        }

        event = event.with_details(serde_json::json!({
            "reason": reason,
            "timestamp": Utc::now().to_rfc3339(),
        }));

        self.log(event).await
    }

    /// Log resource limit exceeded
    pub async fn log_resource_limit_exceeded(
        &self,
        resource_type: &str,
        current: f64,
        limit: f64,
        agent_id: Option<String>,
    ) -> EmergencyResult<()> {
        let mut event = AuditEvent::new(
            AuditEventType::ResourceLimitExceeded,
            AuditLevel::Warning,
            format!("Resource limit exceeded: {resource_type} ({current} > {limit})"),
        );

        if let Some(id) = agent_id {
            event = event.with_agent(id);
        }

        event = event.with_details(serde_json::json!({
            "resource_type": resource_type,
            "current_value": current,
            "limit": limit,
            "exceeded_by": current - limit,
        }));

        self.log(event).await
    }

    /// Log safety violation
    pub async fn log_safety_violation(
        &self,
        agent_id: &str,
        violation_type: &str,
        details: &str,
        severity: &str,
    ) -> EmergencyResult<()> {
        let level = match severity {
            "critical" => AuditLevel::Critical,
            "high" => AuditLevel::Error,
            "medium" => AuditLevel::Warning,
            _ => AuditLevel::Info,
        };

        let event = AuditEvent::new(
            AuditEventType::SafetyViolation,
            level,
            format!("Safety violation detected: {violation_type}"),
        )
        .with_agent(agent_id.to_string())
        .with_details(serde_json::json!({
            "violation_type": violation_type,
            "details": details,
            "severity": severity,
        }));

        self.log(event).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    async fn create_test_logger() -> (AuditLogger, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let log_path = temp_dir.path().join("test-audit.log");

        let config = AuditConfig {
            log_path,
            max_file_size: 1024 * 1024, // 1MB
            max_files: 3,
            buffer_size: 100,
            min_level: AuditLevel::Debug,
        };

        let logger = AuditLogger::new(config).await.unwrap();

        // Give writer task time to start
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

        (logger, temp_dir)
    }

    #[tokio::test]
    async fn test_audit_logger_creation() {
        let (logger, _temp_dir) = create_test_logger().await;
        assert!(logger.get_recent_events(10).await.is_empty());
    }

    #[tokio::test]
    async fn test_log_basic_event() {
        let (logger, _temp_dir) = create_test_logger().await;

        let event = AuditEvent::new(
            AuditEventType::SystemStartup,
            AuditLevel::Info,
            "Emergency control system started".to_string(),
        );

        logger.log(event).await.unwrap();

        // Give writer time to process
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

        let events = logger.get_recent_events(10).await;
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, AuditEventType::SystemStartup);
        assert_eq!(events[0].level, AuditLevel::Info);
    }

    #[tokio::test]
    async fn test_log_with_details() {
        let (logger, _temp_dir) = create_test_logger().await;

        let event = AuditEvent::new(
            AuditEventType::AgentSuspended,
            AuditLevel::Warning,
            "Agent suspended due to violation".to_string(),
        )
        .with_agent("agent-123".to_string())
        .with_details(serde_json::json!({
            "reason": "Memory violation",
            "duration": "60s",
        }))
        .with_user("admin".to_string());

        logger.log(event).await.unwrap();

        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

        let events = logger.get_recent_events(1).await;
        assert_eq!(events[0].agent_id, Some("agent-123".to_string()));
        assert_eq!(events[0].user, Some("admin".to_string()));
        assert_eq!(events[0].details["reason"], "Memory violation");
    }

    #[tokio::test]
    async fn test_minimum_level_filtering() {
        let (logger, temp_dir) = create_test_logger().await;

        // Update config to only log Warning and above
        let log_path = temp_dir.path().join("test-audit.log");
        let config = AuditConfig {
            log_path,
            max_file_size: 1024 * 1024,
            max_files: 3,
            buffer_size: 100,
            min_level: AuditLevel::Warning,
        };
        logger.update_config(config).await.unwrap();

        // Log events of different levels
        logger
            .log(AuditEvent::new(
                AuditEventType::SystemStartup,
                AuditLevel::Debug,
                "Debug message".to_string(),
            ))
            .await
            .unwrap();

        logger
            .log(AuditEvent::new(
                AuditEventType::ConfigurationChanged,
                AuditLevel::Info,
                "Info message".to_string(),
            ))
            .await
            .unwrap();

        logger
            .log(AuditEvent::new(
                AuditEventType::ResourceLimitExceeded,
                AuditLevel::Warning,
                "Warning message".to_string(),
            ))
            .await
            .unwrap();

        logger
            .log(AuditEvent::new(
                AuditEventType::KillSwitchActivated,
                AuditLevel::Critical,
                "Critical message".to_string(),
            ))
            .await
            .unwrap();

        tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;

        // Should only have Warning and Critical events (plus the config change event)
        let events = logger.get_recent_events(10).await;
        // Due to filtering, we should have 2 events (warning + critical)
        // The config change event is Info level and comes from update_config itself
        assert!(events.len() >= 2); // At least 2 events (warning + critical)

        // Verify no Debug or Info events except the config change
        for event in &events {
            if event.event_type != AuditEventType::ConfigurationChanged {
                assert!(event.level >= AuditLevel::Warning);
            }
        }
    }

    #[tokio::test]
    async fn test_query_by_event_type() {
        let (logger, _temp_dir) = create_test_logger().await;

        // Log different event types
        logger
            .log(AuditEvent::new(
                AuditEventType::KillSwitchActivated,
                AuditLevel::Critical,
                "Kill switch 1".to_string(),
            ))
            .await
            .unwrap();

        logger
            .log(AuditEvent::new(
                AuditEventType::ResourceLimitExceeded,
                AuditLevel::Warning,
                "Resource limit".to_string(),
            ))
            .await
            .unwrap();

        logger
            .log(AuditEvent::new(
                AuditEventType::KillSwitchActivated,
                AuditLevel::Critical,
                "Kill switch 2".to_string(),
            ))
            .await
            .unwrap();

        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

        let kill_switch_events = logger
            .get_events_by_type(AuditEventType::KillSwitchActivated)
            .await;
        assert_eq!(kill_switch_events.len(), 2);

        let resource_events = logger
            .get_events_by_type(AuditEventType::ResourceLimitExceeded)
            .await;
        assert_eq!(resource_events.len(), 1);
    }

    #[tokio::test]
    async fn test_query_by_agent() {
        let (logger, _temp_dir) = create_test_logger().await;

        // Log events for different agents
        logger
            .log(
                AuditEvent::new(
                    AuditEventType::AgentSuspended,
                    AuditLevel::Warning,
                    "Agent 1 suspended".to_string(),
                )
                .with_agent("agent-1".to_string()),
            )
            .await
            .unwrap();

        logger
            .log(
                AuditEvent::new(
                    AuditEventType::SafetyViolation,
                    AuditLevel::Error,
                    "Agent 2 violation".to_string(),
                )
                .with_agent("agent-2".to_string()),
            )
            .await
            .unwrap();

        logger
            .log(
                AuditEvent::new(
                    AuditEventType::AgentTerminated,
                    AuditLevel::Error,
                    "Agent 1 terminated".to_string(),
                )
                .with_agent("agent-1".to_string()),
            )
            .await
            .unwrap();

        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

        let agent1_events = logger.get_events_by_agent("agent-1").await;
        assert_eq!(agent1_events.len(), 2);

        let agent2_events = logger.get_events_by_agent("agent-2").await;
        assert_eq!(agent2_events.len(), 1);
    }

    #[tokio::test]
    async fn test_query_by_level() {
        let (logger, _temp_dir) = create_test_logger().await;

        // Log events of different levels
        logger
            .log(AuditEvent::new(
                AuditEventType::SystemStartup,
                AuditLevel::Info,
                "Info event".to_string(),
            ))
            .await
            .unwrap();

        logger
            .log(AuditEvent::new(
                AuditEventType::ResourceLimitExceeded,
                AuditLevel::Warning,
                "Warning event".to_string(),
            ))
            .await
            .unwrap();

        logger
            .log(AuditEvent::new(
                AuditEventType::SafetyViolation,
                AuditLevel::Error,
                "Error event".to_string(),
            ))
            .await
            .unwrap();

        logger
            .log(AuditEvent::new(
                AuditEventType::KillSwitchActivated,
                AuditLevel::Critical,
                "Critical event".to_string(),
            ))
            .await
            .unwrap();

        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

        let error_and_above = logger.get_events_by_level(AuditLevel::Error).await;
        assert_eq!(error_and_above.len(), 2); // Error and Critical

        let all_events = logger.get_events_by_level(AuditLevel::Debug).await;
        assert_eq!(all_events.len(), 4); // All events
    }

    #[tokio::test]
    async fn test_specialized_log_methods() {
        let (logger, _temp_dir) = create_test_logger().await;

        // Test kill switch log
        logger
            .log_kill_switch_activated(
                Some("agent-123".to_string()),
                "Emergency shutdown requested",
            )
            .await
            .unwrap();

        // Test resource limit log
        logger
            .log_resource_limit_exceeded(
                "GPU Memory",
                32768.0,
                16384.0,
                Some("agent-456".to_string()),
            )
            .await
            .unwrap();

        // Test safety violation log
        logger
            .log_safety_violation(
                "agent-789",
                "MemoryAccessViolation",
                "Attempted to access kernel memory",
                "critical",
            )
            .await
            .unwrap();

        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        let events = logger.get_recent_events(10).await;
        assert!(events.len() >= 3);

        // Verify kill switch event
        let kill_switch = events
            .iter()
            .find(|e| e.event_type == AuditEventType::KillSwitchActivated)
            .unwrap();
        assert_eq!(kill_switch.agent_id, Some("agent-123".to_string()));
        assert!(kill_switch.message.contains("Emergency shutdown requested"));

        // Verify resource limit event
        let resource = events
            .iter()
            .find(|e| e.event_type == AuditEventType::ResourceLimitExceeded)
            .unwrap();
        assert_eq!(resource.agent_id, Some("agent-456".to_string()));
        assert_eq!(resource.details["resource_type"], "GPU Memory");
        assert_eq!(resource.details["current_value"], 32768.0);

        // Verify safety violation event
        let safety = events
            .iter()
            .find(|e| e.event_type == AuditEventType::SafetyViolation)
            .unwrap();
        assert_eq!(safety.agent_id, Some("agent-789".to_string()));
        assert_eq!(safety.level, AuditLevel::Critical);
    }

    #[tokio::test]
    async fn test_buffer_limit() {
        let (logger, _temp_dir) = create_test_logger().await;

        // Log many events to exceed buffer limit
        for i in 0..15000 {
            logger
                .log(AuditEvent::new(
                    AuditEventType::SystemStartup,
                    AuditLevel::Info,
                    format!("Event {}", i),
                ))
                .await
                .unwrap();
        }

        tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

        // Buffer should be trimmed to ~5000 events
        let events = logger.get_recent_events(20000).await;
        assert!(events.len() <= 10000);
        assert!(events.len() >= 5000);
    }

    #[tokio::test]
    async fn test_clear_buffer() {
        let (logger, _temp_dir) = create_test_logger().await;

        // Log some events
        for i in 0..10 {
            logger
                .log(AuditEvent::new(
                    AuditEventType::SystemStartup,
                    AuditLevel::Info,
                    format!("Event {}", i),
                ))
                .await
                .unwrap();
        }

        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

        assert!(!logger.get_recent_events(20).await.is_empty());

        // Clear buffer
        logger.clear_buffer().await;
        assert!(logger.get_recent_events(20).await.is_empty());
    }

    #[tokio::test]
    async fn test_file_writing() {
        let (logger, temp_dir) = create_test_logger().await;
        let log_path = temp_dir.path().join("test-audit.log");

        // Log an event
        logger
            .log(AuditEvent::new(
                AuditEventType::SystemStartup,
                AuditLevel::Info,
                "Test file writing".to_string(),
            ))
            .await
            .unwrap();

        // Give writer time to flush
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        // Verify file exists and contains the event
        assert!(log_path.exists());
        let contents = fs::read_to_string(&log_path).await.unwrap();
        assert!(contents.contains("Test file writing"));
        assert!(contents.contains("SystemStartup"));
    }

    #[tokio::test]
    async fn test_audit_level_ordering() {
        assert!(AuditLevel::Debug < AuditLevel::Info);
        assert!(AuditLevel::Info < AuditLevel::Warning);
        assert!(AuditLevel::Warning < AuditLevel::Error);
        assert!(AuditLevel::Error < AuditLevel::Critical);
    }

    #[tokio::test]
    async fn test_event_serialization() {
        let event = AuditEvent::new(
            AuditEventType::KillSwitchActivated,
            AuditLevel::Critical,
            "Test serialization".to_string(),
        )
        .with_agent("agent-123".to_string())
        .with_details(serde_json::json!({"key": "value"}));

        let serialized = serde_json::to_string(&event).unwrap();
        let deserialized: AuditEvent = serde_json::from_str(&serialized).unwrap();

        assert_eq!(event.id, deserialized.id);
        assert_eq!(event.event_type, deserialized.event_type);
        assert_eq!(event.level, deserialized.level);
        assert_eq!(event.message, deserialized.message);
        assert_eq!(event.agent_id, deserialized.agent_id);
    }
}
