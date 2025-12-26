//! Metrics collection for orchestration

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::intents::IntentType;
use super::execution::ExecutionStatus;

/// Orchestration metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrchestrationMetrics {
    /// Total intents processed
    pub total_intents: u64,
    /// Successful intent classifications
    pub successful_classifications: u64,
    /// Failed classifications
    pub failed_classifications: u64,
    /// Intent type distribution
    pub intent_distribution: HashMap<String, u64>,
    /// Total executions
    pub total_executions: u64,
    /// Successful executions
    pub successful_executions: u64,
    /// Failed executions
    pub failed_executions: u64,
    /// Average classification time
    pub avg_classification_time_ms: f64,
    /// Average execution time
    pub avg_execution_time_ms: f64,
    /// Resource utilization
    pub resource_utilization: ResourceMetrics,
    /// Error distribution
    pub error_distribution: HashMap<String, u64>,
    /// Metrics start time
    pub started_at: DateTime<Utc>,
    /// Last update time
    pub last_updated: DateTime<Utc>,
}

/// Resource utilization metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceMetrics {
    /// CPU usage percentage
    pub cpu_usage_percent: f64,
    /// Memory usage in MB
    pub memory_usage_mb: f64,
    /// GPU usage percentage
    pub gpu_usage_percent: Option<f64>,
    /// Network throughput in Mbps
    pub network_throughput_mbps: f64,
    /// Active threads
    pub active_threads: usize,
    /// Queue depth
    pub queue_depth: usize,
}

impl Default for OrchestrationMetrics {
    fn default() -> Self {
        Self {
            total_intents: 0,
            successful_classifications: 0,
            failed_classifications: 0,
            intent_distribution: HashMap::new(),
            total_executions: 0,
            successful_executions: 0,
            failed_executions: 0,
            avg_classification_time_ms: 0.0,
            avg_execution_time_ms: 0.0,
            resource_utilization: ResourceMetrics::default(),
            error_distribution: HashMap::new(),
            started_at: Utc::now(),
            last_updated: Utc::now(),
        }
    }
}

impl Default for ResourceMetrics {
    fn default() -> Self {
        Self {
            cpu_usage_percent: 0.0,
            memory_usage_mb: 0.0,
            gpu_usage_percent: None,
            network_throughput_mbps: 0.0,
            active_threads: 0,
            queue_depth: 0,
        }
    }
}

impl OrchestrationMetrics {
    /// Record intent classification
    pub fn record_classification(&mut self, intent_type: &IntentType, success: bool, duration_ms: f64) {
        self.total_intents += 1;
        
        if success {
            self.successful_classifications += 1;
        } else {
            self.failed_classifications += 1;
        }
        
        // Update intent distribution
        let intent_str = format!("{:?}", intent_type);
        *self.intent_distribution.entry(intent_str).or_insert(0) += 1;
        
        // Update average classification time
        let total_time = self.avg_classification_time_ms * (self.total_intents - 1) as f64;
        self.avg_classification_time_ms = (total_time + duration_ms) / self.total_intents as f64;
        
        self.last_updated = Utc::now();
    }

    /// Record execution
    pub fn record_execution(&mut self, status: ExecutionStatus, duration_ms: f64) {
        self.total_executions += 1;
        
        match status {
            ExecutionStatus::Completed => self.successful_executions += 1,
            ExecutionStatus::Failed | ExecutionStatus::TimedOut => self.failed_executions += 1,
            _ => {}
        }
        
        // Update average execution time
        let total_time = self.avg_execution_time_ms * (self.total_executions - 1) as f64;
        self.avg_execution_time_ms = (total_time + duration_ms) / self.total_executions as f64;
        
        self.last_updated = Utc::now();
    }

    /// Record error
    pub fn record_error(&mut self, error_type: String) {
        *self.error_distribution.entry(error_type).or_insert(0) += 1;
        self.last_updated = Utc::now();
    }

    /// Update resource metrics
    pub fn update_resources(&mut self, resources: ResourceMetrics) {
        self.resource_utilization = resources;
        self.last_updated = Utc::now();
    }

    /// Get classification success rate
    pub fn classification_success_rate(&self) -> f64 {
        if self.total_intents == 0 {
            0.0
        } else {
            self.successful_classifications as f64 / self.total_intents as f64
        }
    }

    /// Get execution success rate
    pub fn execution_success_rate(&self) -> f64 {
        if self.total_executions == 0 {
            0.0
        } else {
            self.successful_executions as f64 / self.total_executions as f64
        }
    }

    /// Get uptime duration
    pub fn uptime(&self) -> Duration {
        self.last_updated - self.started_at
    }

    /// Get throughput (intents per second)
    pub fn throughput(&self) -> f64 {
        let uptime_seconds = self.uptime().num_seconds() as f64;
        if uptime_seconds > 0.0 {
            self.total_intents as f64 / uptime_seconds
        } else {
            0.0
        }
    }

    /// Reset metrics
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Get summary
    pub fn summary(&self) -> MetricsSummary {
        MetricsSummary {
            total_intents: self.total_intents,
            classification_success_rate: self.classification_success_rate(),
            execution_success_rate: self.execution_success_rate(),
            avg_classification_time_ms: self.avg_classification_time_ms,
            avg_execution_time_ms: self.avg_execution_time_ms,
            throughput: self.throughput(),
            uptime_hours: self.uptime().num_hours() as f64,
        }
    }
}

/// Metrics summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsSummary {
    /// Total intents processed
    pub total_intents: u64,
    /// Classification success rate (0-1)
    pub classification_success_rate: f64,
    /// Execution success rate (0-1)
    pub execution_success_rate: f64,
    /// Average classification time in ms
    pub avg_classification_time_ms: f64,
    /// Average execution time in ms
    pub avg_execution_time_ms: f64,
    /// Throughput (intents/second)
    pub throughput: f64,
    /// Uptime in hours
    pub uptime_hours: f64,
}