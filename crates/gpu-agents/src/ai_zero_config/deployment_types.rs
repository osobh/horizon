//! Deployment-related types for AI Assistant Zero-Config Integration
//! Split from types.rs to keep files under 750 lines

use std::collections::HashMap;
use std::time::Duration;

/// Natural language deployment intent
#[derive(Debug, Clone)]
pub struct DeploymentIntent {
    pub natural_language: String,
    pub intent_type: IntentType,
    pub target_environment: TargetEnvironment,
    pub performance_requirements: PerformanceRequirements,
    pub security_requirements: SecurityRequirements,
    pub confidence_score: f64,
}

/// Types of deployment intents
#[derive(Debug, Clone, PartialEq)]
pub enum IntentType {
    WebService,
    MicroService,
    DatabaseService,
    MLService,
    BackgroundWorker,
    APIGateway,
    StaticSite,
    CronJob,
    EventProcessor,
    StreamProcessor,
}

/// Target deployment environment
#[derive(Debug, Clone, PartialEq)]
pub enum TargetEnvironment {
    Development,
    Staging,
    Production,
    Testing,
    Edge,
    Hybrid,
}

/// Performance requirements for deployment
#[derive(Debug, Clone)]
pub struct PerformanceRequirements {
    pub min_cpu_cores: Option<u32>,
    pub min_memory_mb: Option<u64>,
    pub min_storage_gb: Option<u64>,
    pub max_latency_ms: Option<u64>,
    pub min_throughput_rps: Option<u64>,
    pub gpu_required: bool,
    pub gpu_memory_mb: Option<u64>,
    pub high_availability: bool,
    pub auto_scaling: bool,
}

/// Security requirements for deployment
#[derive(Debug, Clone)]
pub struct SecurityRequirements {
    pub encryption_at_rest: bool,
    pub encryption_in_transit: bool,
    pub authentication_required: bool,
    pub authorization_levels: Vec<String>,
    pub compliance_frameworks: Vec<ComplianceFramework>,
    pub network_isolation: bool,
    pub secrets_management: bool,
    pub audit_logging: bool,
}

/// Compliance frameworks
#[derive(Debug, Clone, PartialEq)]
pub enum ComplianceFramework {
    GDPR,
    HIPAA,
    SOC2,
    PCI,
    FedRAMP,
    ISO27001,
}

/// Deployment configuration
#[derive(Debug, Clone)]
pub struct DeploymentConfiguration {
    pub container_image: String,
    pub environment_variables: HashMap<String, String>,
    pub command: Option<Vec<String>>,
    pub args: Option<Vec<String>>,
    pub working_directory: Option<String>,
    pub health_checks: HealthCheckConfiguration,
    pub resource_limits: ResourceLimits,
    pub restart_policy: RestartPolicy,
}

/// Health check configuration
#[derive(Debug, Clone)]
pub struct HealthCheckConfiguration {
    pub readiness_probe: Option<Probe>,
    pub liveness_probe: Option<Probe>,
    pub startup_probe: Option<Probe>,
}

/// Health check probe
#[derive(Debug, Clone)]
pub struct Probe {
    pub probe_type: ProbeType,
    pub path: Option<String>,
    pub port: Option<u16>,
    pub initial_delay_seconds: u32,
    pub period_seconds: u32,
    pub timeout_seconds: u32,
    pub failure_threshold: u32,
    pub success_threshold: u32,
}

/// Types of health check probes
#[derive(Debug, Clone, PartialEq)]
pub enum ProbeType {
    HTTP,
    TCP,
    Exec,
    GRPC,
}

/// Resource limits and requests
#[derive(Debug, Clone)]
pub struct ResourceLimits {
    pub cpu_limit: Option<String>,
    pub memory_limit: Option<String>,
    pub storage_limit: Option<String>,
    pub gpu_limit: Option<u32>,
    pub cpu_request: Option<String>,
    pub memory_request: Option<String>,
    pub storage_request: Option<String>,
}

/// Restart policy for containers
#[derive(Debug, Clone, PartialEq)]
pub enum RestartPolicy {
    Always,
    OnFailure,
    Never,
    UnlessStopped,
}

/// Deployment patterns detected
#[derive(Debug, Clone, PartialEq)]
pub enum DeploymentPattern {
    Monolith,
    Microservices,
    Serverless,
    Container,
    StaticFiles,
    Database,
    Queue,
    Cache,
    Proxy,
    LoadBalancer,
}

/// Deployment result
#[derive(Debug, Clone)]
pub struct DeploymentResult {
    pub success: bool,
    pub deployment_id: String,
    pub service_endpoints: Vec<String>,
    pub deployment_time: Duration,
    pub resource_usage: ResourceUsageReport,
    pub configuration_applied: super::GeneratedConfiguration,
    pub validation_results: Vec<ValidationResult>,
    pub cost_estimate: CostEstimate,
}

/// Resource usage report
#[derive(Debug, Clone)]
pub struct ResourceUsageReport {
    pub cpu_usage: f64,
    pub memory_usage: f64,
    pub storage_usage: f64,
    pub network_usage: f64,
    pub gpu_usage: Option<f64>,
    pub cost_per_hour: f64,
}

/// Validation results
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub validation_type: ValidationType,
    pub status: ValidationStatus,
    pub message: String,
    pub details: HashMap<String, String>,
}

/// Validation types
#[derive(Debug, Clone, PartialEq)]
pub enum ValidationType {
    Security,
    Performance,
    Compliance,
    CostOptimization,
    BestPractices,
}

/// Validation status
#[derive(Debug, Clone, PartialEq)]
pub enum ValidationStatus {
    Passed,
    Failed,
    Warning,
    Skipped,
}

/// Cost estimates
#[derive(Debug, Clone)]
pub struct CostEstimate {
    pub hourly_cost: f64,
    pub daily_cost: f64,
    pub monthly_cost: f64,
    pub cost_breakdown: HashMap<String, f64>,
    pub cost_factors: Vec<CostFactor>,
}

/// Cost factors
#[derive(Debug, Clone)]
pub struct CostFactor {
    pub factor_name: String,
    pub cost_contribution: f64,
    pub optimization_potential: f64,
}