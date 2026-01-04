//! Context management for intent orchestration

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::execution::ExecutionRecord;
use super::intents::Intent;

/// Intent context containing all relevant information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentContext {
    /// Current intent
    pub intent: Intent,
    /// Session context
    pub session: SessionContext,
    /// System state
    pub system_state: SystemState,
    /// Ambient context
    pub ambient: AmbientContext,
    /// Previous execution records
    pub history: Vec<ExecutionRecord>,
}

/// Session context for user interaction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionContext {
    /// Session ID
    pub session_id: String,
    /// User ID
    pub user_id: String,
    /// User roles
    pub roles: Vec<String>,
    /// Session start time
    pub started_at: DateTime<Utc>,
    /// Previous intents in session
    pub intent_history: Vec<Intent>,
    /// Session metadata
    pub metadata: HashMap<String, String>,
}

/// Current system state
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SystemState {
    /// Active services
    pub active_services: Vec<String>,
    /// Resource utilization
    pub resource_usage: HashMap<String, f64>,
    /// Current errors/warnings
    pub alerts: Vec<Alert>,
    /// System metrics
    pub metrics: HashMap<String, f64>,
    /// System configuration
    pub config: HashMap<String, String>,
}

/// System alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    /// Alert level
    pub level: AlertLevel,
    /// Alert message
    pub message: String,
    /// Alert source
    pub source: String,
    /// Alert timestamp
    pub timestamp: DateTime<Utc>,
}

/// Alert severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlertLevel {
    /// Informational
    Info,
    /// Warning
    Warning,
    /// Error
    Error,
    /// Critical
    Critical,
}

/// Ambient context for environmental factors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AmbientContext {
    /// Current time
    pub current_time: DateTime<Utc>,
    /// Environment (dev, staging, prod)
    pub environment: String,
    /// Geographic region
    pub region: String,
    /// Compliance requirements
    pub compliance_requirements: Vec<String>,
    /// Resource constraints
    pub constraints: ResourceConstraints,
    /// Feature flags
    pub feature_flags: HashMap<String, bool>,
}

/// Resource constraints
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ResourceConstraints {
    /// Maximum CPU cores
    pub max_cpu: Option<u32>,
    /// Maximum memory in GB
    pub max_memory_gb: Option<u32>,
    /// Maximum storage in GB
    pub max_storage_gb: Option<u32>,
    /// Maximum network bandwidth in Mbps
    pub max_bandwidth_mbps: Option<u32>,
    /// Budget limit
    pub budget_limit: Option<f64>,
}

impl IntentContext {
    /// Create new intent context
    pub fn new(intent: Intent, session: SessionContext) -> Self {
        Self {
            intent,
            session,
            system_state: SystemState::default(),
            ambient: AmbientContext::default(),
            history: Vec::new(),
        }
    }

    /// Add execution record to history
    pub fn add_execution_record(&mut self, record: ExecutionRecord) {
        self.history.push(record);

        // Keep only last 10 records
        if self.history.len() > 10 {
            self.history.remove(0);
        }
    }

    /// Check if user has required role
    pub fn has_role(&self, role: &str) -> bool {
        self.session.roles.contains(&role.to_string())
    }

    /// Get resource availability
    pub fn get_available_resources(&self) -> HashMap<String, f64> {
        let mut available = HashMap::new();

        if let Some(max_cpu) = self.ambient.constraints.max_cpu {
            let used = self.system_state.resource_usage.get("cpu").unwrap_or(&0.0);
            available.insert("cpu".to_string(), max_cpu as f64 - used);
        }

        if let Some(max_memory) = self.ambient.constraints.max_memory_gb {
            let used = self
                .system_state
                .resource_usage
                .get("memory_gb")
                .unwrap_or(&0.0);
            available.insert("memory_gb".to_string(), max_memory as f64 - used);
        }

        available
    }

    /// Check if action is allowed in current context
    pub fn is_action_allowed(&self, action: &str) -> bool {
        // Check environment restrictions
        if self.ambient.environment == "production" {
            match action {
                "delete" | "destroy" | "reset" => false,
                _ => true,
            }
        } else {
            true
        }
    }

    /// Get context priority based on alerts
    pub fn get_priority(&self) -> u8 {
        if self
            .system_state
            .alerts
            .iter()
            .any(|a| matches!(a.level, AlertLevel::Critical))
        {
            1 // Highest priority
        } else if self
            .system_state
            .alerts
            .iter()
            .any(|a| matches!(a.level, AlertLevel::Error))
        {
            2
        } else if self
            .system_state
            .alerts
            .iter()
            .any(|a| matches!(a.level, AlertLevel::Warning))
        {
            3
        } else {
            5 // Normal priority
        }
    }
}

impl Default for SessionContext {
    fn default() -> Self {
        Self {
            session_id: uuid::Uuid::new_v4().to_string(),
            user_id: "system".to_string(),
            roles: vec!["user".to_string()],
            started_at: Utc::now(),
            intent_history: Vec::new(),
            metadata: HashMap::new(),
        }
    }
}

impl Default for AmbientContext {
    fn default() -> Self {
        Self {
            current_time: Utc::now(),
            environment: "development".to_string(),
            region: "us-east-1".to_string(),
            compliance_requirements: Vec::new(),
            constraints: ResourceConstraints::default(),
            feature_flags: HashMap::new(),
        }
    }
}
