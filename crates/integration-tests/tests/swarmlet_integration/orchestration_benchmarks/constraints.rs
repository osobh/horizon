//! Scheduling constraints and lifecycle management

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Scheduling constraints for workloads
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulingConstraints {
    pub node_selector: HashMap<String, String>,
    pub affinity_rules: Vec<AffinityRule>,
    pub tolerations: Vec<Toleration>,
    pub priority_class: PriorityClass,
    pub deadline_constraints: Option<DeadlineConstraints>,
}

/// Affinity rule for scheduling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AffinityRule {
    pub affinity_type: AffinityType,
    pub label_selector: HashMap<String, String>,
    pub topology_key: String,
}

/// Type of affinity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AffinityType {
    NodeAffinity,
    PodAffinity,
    PodAntiAffinity,
}

/// Toleration for node taints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Toleration {
    pub key: String,
    pub operator: TolerationOperator,
    pub value: Option<String>,
    pub effect: Option<TaintEffect>,
}

/// Toleration operator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TolerationOperator {
    Exists,
    Equal,
}

/// Taint effect
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaintEffect {
    NoSchedule,
    PreferNoSchedule,
    NoExecute,
}

/// Priority class for scheduling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PriorityClass {
    Low,
    Default,
    High,
    Critical,
    System,
}

/// Deadline constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeadlineConstraints {
    pub max_execution_time: Duration,
    pub max_queue_time: Duration,
    pub failure_action: DeadlineFailureAction,
}

/// Action on deadline failure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeadlineFailureAction {
    Cancel,
    Retry,
    Escalate,
    Continue,
}

/// Lifecycle hooks for containers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LifecycleHooks {
    pub pre_start: Option<Hook>,
    pub post_start: Option<Hook>,
    pub pre_stop: Option<Hook>,
    pub liveness_probe: Option<Probe>,
    pub readiness_probe: Option<Probe>,
    pub startup_probe: Option<Probe>,
}

/// Hook definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hook {
    pub exec_command: Option<Vec<String>>,
    pub http_get: Option<HttpGet>,
}

/// HTTP GET request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HttpGet {
    pub path: String,
    pub port: u16,
    pub scheme: String,
}

/// Health probe definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Probe {
    pub probe_type: ProbeType,
    pub initial_delay_seconds: u32,
    pub period_seconds: u32,
    pub timeout_seconds: u32,
    pub success_threshold: u32,
    pub failure_threshold: u32,
}

/// Type of probe
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProbeType {
    HttpGet(HttpGet),
    TcpSocket { port: u16 },
    Exec { command: Vec<String> },
}

impl Default for SchedulingConstraints {
    fn default() -> Self {
        Self {
            node_selector: HashMap::new(),
            affinity_rules: Vec::new(),
            tolerations: Vec::new(),
            priority_class: PriorityClass::Default,
            deadline_constraints: None,
        }
    }
}

impl Default for LifecycleHooks {
    fn default() -> Self {
        Self {
            pre_start: None,
            post_start: None,
            pre_stop: None,
            liveness_probe: None,
            readiness_probe: None,
            startup_probe: None,
        }
    }
}
