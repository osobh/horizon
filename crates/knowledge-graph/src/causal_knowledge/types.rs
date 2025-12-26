//! Core types for causal relationships

use chrono::{DateTime, Duration as ChronoDuration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Causal relationship with confidence and temporal information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalRelationship {
    /// Unique identifier
    pub id: String,
    /// Source node in causal relationship
    pub cause_node_id: String,
    /// Target node in causal relationship
    pub effect_node_id: String,
    /// Causal strength (0.0 to 1.0)
    pub causal_strength: f64,
    /// Confidence in this causal relationship
    pub confidence: f64,
    /// Temporal delay between cause and effect
    pub temporal_delay: ChronoDuration,
    /// Type of causal relationship
    pub causal_type: CausalType,
    /// Evidence supporting this relationship
    pub evidence: super::evidence::CausalEvidence,
    /// Uncertainty bounds
    pub uncertainty_bounds: UncertaintyBounds,
}

/// Types of causal relationships
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CausalType {
    /// Direct causal relationship
    Direct,
    /// Indirect (mediated) causal relationship
    Indirect { mediators: Vec<String> },
    /// Bi-directional causal relationship
    Bidirectional,
    /// Common cause relationship
    CommonCause { common_cause_id: String },
    /// Spurious correlation (no true causation)
    Spurious,
}

/// Direction of observed effects
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EffectDirection {
    Positive,
    Negative,
    NonLinear,
    Threshold,
}

/// Uncertainty bounds for causal estimates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UncertaintyBounds {
    /// Lower bound of causal strength estimate
    pub lower_bound: f64,
    /// Upper bound of causal strength estimate
    pub upper_bound: f64,
    /// Epistemic uncertainty (model uncertainty)
    pub epistemic_uncertainty: f64,
    /// Aleatoric uncertainty (data uncertainty)
    pub aleatoric_uncertainty: f64,
}

/// GPU metrics for causal inference
#[derive(Debug, Clone)]
pub struct GpuMetrics {
    /// GPU utilization percentage
    pub utilization: f32,
    /// Memory usage percentage
    pub memory_usage: f32,
    /// Number of parallel batches processed
    pub parallel_batches: usize,
    /// Inference throughput (operations per second)
    pub throughput: f32,
}

/// Real-time pattern detection metrics
#[derive(Debug, Clone)]
pub struct RealTimeMetrics {
    /// Average processing latency per event
    pub average_processing_latency: Duration,
    /// Events processed per second
    pub throughput: f32,
    /// Buffer utilization
    pub buffer_utilization: f32,
}

/// Temporal event in knowledge graph
#[derive(Debug, Clone)]
pub struct TemporalEvent {
    /// Event identifier
    pub event_id: String,
    /// Node involved in event
    pub node_id: String,
    /// When the event occurred
    pub timestamp: DateTime<Utc>,
    /// Type of event
    pub event_type: EventType,
    /// Event properties
    pub properties: HashMap<String, serde_json::Value>,
}

/// Types of temporal events
#[derive(Debug, Clone)]
pub enum EventType {
    NodeCreated,
    NodeUpdated,
    NodeDeleted,
    EdgeCreated,
    EdgeUpdated,
    EdgeDeleted,
    PropertyChanged,
}
