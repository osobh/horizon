//! Type definitions for Darwin-Gödel integration system

use crate::dgm_empirical_validation::ValidationResult;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};

/// Configuration for Darwin-Gödel integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationConfig {
    /// Default validation mode
    pub default_mode: ValidationMode,
    /// Automatic mode switching enabled
    pub auto_switching: bool,
    /// Formal proof timeout threshold
    pub formal_proof_timeout: Duration,
    /// Empirical validation confidence threshold
    pub empirical_confidence_threshold: f64,
    /// Context analysis window size
    pub context_window_size: usize,
    /// Mode switching cooldown period
    pub switching_cooldown: Duration,
    /// Resource utilization thresholds
    pub resource_thresholds: ResourceThresholds,
}

impl Default for IntegrationConfig {
    fn default() -> Self {
        Self {
            default_mode: ValidationMode::Hybrid,
            auto_switching: true,
            formal_proof_timeout: Duration::from_secs(300),
            empirical_confidence_threshold: 0.95,
            context_window_size: 50,
            switching_cooldown: Duration::from_secs(60),
            resource_thresholds: ResourceThresholds::default(),
        }
    }
}

/// Validation modes available in the system
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ValidationMode {
    /// Traditional Gödel Machine with formal proofs
    FormalProof,
    /// Darwin-style empirical validation
    Empirical,
    /// Hybrid approach using both modes
    Hybrid,
    /// Adaptive mode switching based on context
    Adaptive,
}

/// Resource utilization thresholds for mode switching
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceThresholds {
    /// CPU usage threshold (percentage)
    pub cpu_threshold: f64,
    /// Memory usage threshold (bytes)
    pub memory_threshold: u64,
    /// Time budget threshold
    pub time_threshold: Duration,
    /// Proof complexity threshold
    pub complexity_threshold: u32,
}

impl Default for ResourceThresholds {
    fn default() -> Self {
        Self {
            cpu_threshold: 80.0,
            memory_threshold: 1_000_000_000, // 1GB
            time_threshold: Duration::from_secs(600),
            complexity_threshold: 1000,
        }
    }
}

/// Request for validation with contextual information
#[derive(Debug, Clone)]
pub struct ValidationRequest {
    /// Request ID
    pub id: String,
    /// Agent modification to validate
    pub modification: String,
    /// Current context metrics
    pub context: ContextMetrics,
    /// Preferred validation mode (optional)
    pub preferred_mode: Option<ValidationMode>,
    /// Time budget for validation
    pub time_budget: Option<Duration>,
    /// Criticality level of the modification
    pub criticality: CriticalityLevel,
}

/// Criticality levels for modifications
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CriticalityLevel {
    /// Low impact changes
    Low,
    /// Medium impact changes
    Medium,
    /// High impact changes requiring careful validation
    High,
    /// Critical changes affecting core functionality
    Critical,
}

/// Response from validation process
#[derive(Debug, Clone)]
pub struct ValidationResponse {
    /// Request ID
    pub request_id: String,
    /// Validation mode used
    pub mode_used: ValidationMode,
    /// Validation result
    pub result: ValidationResult,
    /// Confidence in the result
    pub confidence: f64,
    /// Time taken for validation
    pub validation_time: Duration,
    /// Resources consumed
    pub resource_usage: ResourceUsage,
    /// Recommendations for future validations
    pub recommendations: Vec<String>,
}

/// Resource usage during validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    /// CPU time consumed
    pub cpu_time: Duration,
    /// Peak memory usage
    pub peak_memory: u64,
    /// Number of proof steps (for formal validation)
    pub proof_steps: Option<u32>,
    /// Number of empirical tests (for empirical validation)
    pub test_count: Option<u32>,
}

/// Contextual metrics for mode decision making
#[derive(Debug, Clone)]
pub struct ContextMetrics {
    /// Current system load
    pub system_load: f64,
    /// Available computational resources
    pub available_resources: AvailableResources,
    /// Recent validation history
    pub validation_history: Vec<ValidationHistoryEntry>,
    /// Modification complexity estimate
    pub complexity_estimate: u32,
    /// Time pressure indicator
    pub time_pressure: f64,
    /// Success rates by mode
    pub mode_success_rates: HashMap<ValidationMode, f64>,
}

/// Available computational resources
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AvailableResources {
    /// Available CPU cores
    pub cpu_cores: u32,
    /// Available memory in bytes
    pub memory_bytes: u64,
    /// Available time budget
    pub time_budget: Duration,
    /// Parallel processing capability
    pub parallel_capacity: u32,
}

/// Historical validation entry
#[derive(Debug, Clone)]
pub struct ValidationHistoryEntry {
    /// Timestamp of validation
    pub timestamp: SystemTime,
    /// Mode used
    pub mode: ValidationMode,
    /// Success indicator
    pub success: bool,
    /// Time taken
    pub duration: Duration,
    /// Confidence achieved
    pub confidence: f64,
}

/// Mode switching decision with rationale
#[derive(Debug, Clone)]
pub struct ModeDecision {
    /// Recommended validation mode
    pub recommended_mode: ValidationMode,
    /// Confidence in the recommendation
    pub confidence: f64,
    /// Rationale for the decision
    pub rationale: String,
    /// Expected benefits
    pub expected_benefits: Vec<String>,
    /// Potential risks
    pub potential_risks: Vec<String>,
    /// Estimated resource requirements
    pub resource_estimate: ResourceUsage,
}

/// Mode switching trigger event
#[derive(Debug, Clone)]
pub struct ModeSwitch {
    /// Switch ID
    pub id: String,
    /// Timestamp of switch
    pub timestamp: SystemTime,
    /// Previous mode
    pub from_mode: ValidationMode,
    /// New mode
    pub to_mode: ValidationMode,
    /// Trigger reason
    pub reason: SwitchReason,
    /// Context at time of switch
    pub context: ContextMetrics,
}

/// Reasons for mode switching
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SwitchReason {
    /// Resource constraints
    ResourceConstraints,
    /// Time pressure
    TimePressure,
    /// Complexity threshold exceeded
    ComplexityThreshold,
    /// Poor performance in current mode
    PoorPerformance,
    /// User preference
    UserPreference,
    /// Context change
    ContextChange,
    /// Automatic optimization
    AutoOptimization,
}

/// Performance metrics for the integration system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationMetrics {
    /// Total validations performed
    pub total_validations: u64,
    /// Validations by mode
    pub validations_by_mode: HashMap<ValidationMode, u64>,
    /// Success rates by mode
    pub success_rates_by_mode: HashMap<ValidationMode, f64>,
    /// Average validation times by mode
    pub avg_times_by_mode: HashMap<ValidationMode, Duration>,
    /// Mode switches performed
    pub mode_switches: u64,
    /// Switch success rate
    pub switch_success_rate: f64,
    /// Resource efficiency by mode
    pub resource_efficiency_by_mode: HashMap<ValidationMode, f64>,
}
