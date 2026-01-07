//! Application state management
//!
//! Provides shared state between CLI and TUI modes for tracking
//! deployments, metrics, and runtime information.

use crate::core::{
    config::AppConfig,
    profile::Environment,
    project::{get_all_projects, Project},
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Shared application state type
pub type SharedState = Arc<RwLock<AppState>>;

/// Main application state
#[derive(Debug, Clone)]
pub struct AppState {
    /// Application configuration
    pub config: AppConfig,

    /// Current active environment
    pub active_environment: Environment,

    /// Available projects
    pub projects: Vec<Project>,

    /// Current deployment status
    pub deployments: DeploymentStatus,

    /// Real-time metrics
    pub metrics: Metrics,

    /// Recent log entries
    pub logs: Vec<LogEntry>,
}

impl AppState {
    /// Create new application state
    pub fn new(config: AppConfig) -> Self {
        let active_environment = Environment::from_str(&config.default_environment)
            .unwrap_or(Environment::Dev);

        Self {
            config,
            active_environment,
            projects: get_all_projects(),
            deployments: DeploymentStatus::default(),
            metrics: Metrics::default(),
            logs: Vec::new(),
        }
    }

    /// Create shared state
    pub fn shared(config: AppConfig) -> SharedState {
        Arc::new(RwLock::new(Self::new(config)))
    }

    /// Add a log entry
    pub fn add_log(&mut self, level: LogLevel, message: String) {
        self.logs.push(LogEntry {
            timestamp: chrono::Utc::now(),
            level,
            message,
        });

        // Keep logs under max limit
        let max = self.config.tui.max_log_entries;
        while self.logs.len() > max {
            self.logs.remove(0);
        }
    }

    /// Get project by ID
    pub fn get_project(&self, id: &str) -> Option<&Project> {
        self.projects.iter().find(|p| p.id == id)
    }
}

/// Deployment status across environments
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DeploymentStatus {
    /// Local deployments
    pub local: EnvironmentStatus,
    /// Staging deployments
    pub staging: EnvironmentStatus,
    /// Production deployments
    pub production: EnvironmentStatus,
}

impl DeploymentStatus {
    /// Get status for an environment
    pub fn get(&self, env: Environment) -> &EnvironmentStatus {
        match env {
            Environment::Dev => &self.local,
            Environment::Staging => &self.staging,
            Environment::Prod => &self.production,
        }
    }

    /// Get mutable status for an environment
    pub fn get_mut(&mut self, env: Environment) -> &mut EnvironmentStatus {
        match env {
            Environment::Dev => &mut self.local,
            Environment::Staging => &mut self.staging,
            Environment::Prod => &mut self.production,
        }
    }
}

/// Status of deployments in a single environment
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EnvironmentStatus {
    /// Running services
    pub services: HashMap<String, ServiceStatus>,
    /// Overall health
    pub health: Health,
    /// Last update time
    pub last_updated: Option<chrono::DateTime<chrono::Utc>>,
}

/// Status of a single service
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceStatus {
    /// Project ID
    pub project_id: String,
    /// Running status
    pub status: ServiceState,
    /// Number of replicas running
    pub replicas_running: u32,
    /// Total replicas desired
    pub replicas_desired: u32,
    /// Service endpoints
    pub endpoints: Vec<String>,
    /// Health check status
    pub health: Health,
    /// Last health check time
    pub last_health_check: Option<chrono::DateTime<chrono::Utc>>,
}

/// Service running state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ServiceState {
    Pending,
    Starting,
    Running,
    Stopping,
    Stopped,
    Failed,
    Unknown,
}

impl ServiceState {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Starting => "starting",
            Self::Running => "running",
            Self::Stopping => "stopping",
            Self::Stopped => "stopped",
            Self::Failed => "failed",
            Self::Unknown => "unknown",
        }
    }

    pub fn is_healthy(&self) -> bool {
        matches!(self, Self::Running)
    }
}

/// Health status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum Health {
    Healthy,
    Degraded,
    Unhealthy,
    #[default]
    Unknown,
}

impl Health {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Healthy => "healthy",
            Self::Degraded => "degraded",
            Self::Unhealthy => "unhealthy",
            Self::Unknown => "unknown",
        }
    }
}

/// Real-time metrics
#[derive(Debug, Clone, Default)]
pub struct Metrics {
    /// GPU utilization metrics
    pub gpu_usage: Vec<GpuMetric>,
    /// Cluster health metrics
    pub cluster_health: ClusterHealth,
    /// Job status list
    pub job_status: Vec<JobStatus>,
}

/// GPU utilization metric
#[derive(Debug, Clone)]
pub struct GpuMetric {
    /// GPU index
    pub index: usize,
    /// GPU name
    pub name: String,
    /// Usage percentage (0-100)
    pub usage_percent: f32,
    /// Memory used in MB
    pub memory_used_mb: f32,
    /// Total memory in MB
    pub memory_total_mb: f32,
    /// Temperature in Celsius
    pub temperature_c: Option<f32>,
}

/// Cluster health metrics
#[derive(Debug, Clone, Default)]
pub struct ClusterHealth {
    /// Total nodes
    pub total_nodes: u32,
    /// Healthy nodes
    pub healthy_nodes: u32,
    /// Running pods/containers
    pub running_pods: u32,
    /// Pending pods/containers
    pub pending_pods: u32,
    /// Failed pods/containers
    pub failed_pods: u32,
}

/// Job status
#[derive(Debug, Clone)]
pub struct JobStatus {
    /// Job ID
    pub id: String,
    /// Job name
    pub name: String,
    /// Status string
    pub status: String,
    /// Progress (0.0 - 1.0)
    pub progress: f32,
    /// Start time
    pub started_at: Option<chrono::DateTime<chrono::Utc>>,
}

/// Log entry
#[derive(Debug, Clone)]
pub struct LogEntry {
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Log level
    pub level: LogLevel,
    /// Log message
    pub message: String,
}

/// Log level
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogLevel {
    Debug,
    Info,
    Warn,
    Error,
}

impl LogLevel {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Debug => "DEBUG",
            Self::Info => "INFO",
            Self::Warn => "WARN",
            Self::Error => "ERROR",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_app_state_creation() {
        let config = AppConfig::default();
        let state = AppState::new(config);
        assert!(!state.projects.is_empty());
    }

    #[test]
    fn test_log_limit() {
        let mut config = AppConfig::default();
        config.tui.max_log_entries = 5;
        let mut state = AppState::new(config);

        for i in 0..10 {
            state.add_log(LogLevel::Info, format!("Log {}", i));
        }

        assert_eq!(state.logs.len(), 5);
    }
}
