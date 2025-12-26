//! Infrastructure-related types for AI Assistant Zero-Config Integration
//! Split from types.rs to keep files under 750 lines

use std::collections::HashMap;

/// Infrastructure configuration
#[derive(Debug, Clone)]
pub struct InfrastructureConfiguration {
    pub cluster_config: ClusterConfiguration,
    pub node_config: NodeConfiguration,
    pub load_balancer_config: LoadBalancerConfiguration,
    pub ingress_config: IngressConfiguration,
    pub service_mesh_config: Option<ServiceMeshConfiguration>,
}

/// Cluster configuration
#[derive(Debug, Clone)]
pub struct ClusterConfiguration {
    pub cluster_type: ClusterType,
    pub node_count: u32,
    pub availability_zones: Vec<String>,
    pub kubernetes_version: String,
    pub cluster_autoscaling: bool,
    pub cluster_monitoring: bool,
    pub cluster_logging: bool,
}

/// Types of clusters
#[derive(Debug, Clone, PartialEq)]
pub enum ClusterType {
    Managed,
    SelfManaged,
    Serverless,
    Edge,
    Hybrid,
}

/// Node configuration
#[derive(Debug, Clone)]
pub struct NodeConfiguration {
    pub instance_type: String,
    pub operating_system: OperatingSystem,
    pub disk_size_gb: u32,
    pub disk_type: DiskType,
    pub preemptible: bool,
    pub gpu_type: Option<GPUType>,
    pub node_labels: HashMap<String, String>,
    pub node_taints: Vec<NodeTaint>,
}

/// Operating systems
#[derive(Debug, Clone, PartialEq)]
pub enum OperatingSystem {
    Ubuntu,
    AmazonLinux,
    RHEL,
    CoreOS,
    Windows,
    ContainerOptimized,
}

/// Disk types
#[derive(Debug, Clone, PartialEq)]
pub enum DiskType {
    SSD,
    HDD,
    NVMe,
    Network,
}

/// GPU types
#[derive(Debug, Clone, PartialEq)]
pub enum GPUType {
    V100,
    A100,
    T4,
    K80,
    P100,
    RTX3080,
    RTX4090,
}

/// Node taints for scheduling
#[derive(Debug, Clone)]
pub struct NodeTaint {
    pub key: String,
    pub value: String,
    pub effect: TaintEffect,
}

/// Taint effects
#[derive(Debug, Clone, PartialEq)]
pub enum TaintEffect {
    NoSchedule,
    PreferNoSchedule,
    NoExecute,
}

/// Load balancer configuration
#[derive(Debug, Clone)]
pub struct LoadBalancerConfiguration {
    pub load_balancer_type: LoadBalancerType,
    pub algorithm: LoadBalancingAlgorithm,
    pub health_check_path: String,
    pub session_affinity: bool,
    pub ssl_termination: bool,
    pub rate_limiting: Option<RateLimitConfiguration>,
}

/// Load balancer types
#[derive(Debug, Clone, PartialEq)]
pub enum LoadBalancerType {
    Application,
    Network,
    Classic,
    Gateway,
}

/// Load balancing algorithms
#[derive(Debug, Clone, PartialEq)]
pub enum LoadBalancingAlgorithm {
    RoundRobin,
    LeastConnections,
    IPHash,
    WeightedRoundRobin,
    Random,
}

/// Rate limiting configuration
#[derive(Debug, Clone)]
pub struct RateLimitConfiguration {
    pub requests_per_second: u32,
    pub burst_size: u32,
    pub sliding_window: std::time::Duration,
}

/// Ingress configuration
#[derive(Debug, Clone)]
pub struct IngressConfiguration {
    pub ingress_class: String,
    pub tls_enabled: bool,
    pub certificate_management: CertificateManagement,
    pub annotations: HashMap<String, String>,
    pub custom_rules: Vec<IngressRule>,
}

/// Certificate management options
#[derive(Debug, Clone, PartialEq)]
pub enum CertificateManagement {
    Manual,
    LetsEncrypt,
    CertManager,
    CloudProvider,
}

/// Ingress routing rules
#[derive(Debug, Clone)]
pub struct IngressRule {
    pub host: String,
    pub path: String,
    pub path_type: PathType,
    pub backend_service: String,
    pub backend_port: u16,
}

/// Path matching types
#[derive(Debug, Clone, PartialEq)]
pub enum PathType {
    Exact,
    Prefix,
    ImplementationSpecific,
}

/// Service mesh configuration
#[derive(Debug, Clone)]
pub struct ServiceMeshConfiguration {
    pub mesh_type: ServiceMeshType,
    pub mutual_tls: bool,
    pub traffic_policies: Vec<TrafficPolicy>,
    pub observability_enabled: bool,
    pub security_policies: Vec<SecurityPolicy>,
}

/// Service mesh types
#[derive(Debug, Clone, PartialEq)]
pub enum ServiceMeshType {
    Istio,
    Linkerd,
    Consul,
    AppMesh,
    OpenServiceMesh,
}

/// Traffic policies
#[derive(Debug, Clone)]
pub struct TrafficPolicy {
    pub name: String,
    pub traffic_split: HashMap<String, u32>,
    pub timeout: Option<std::time::Duration>,
    pub retry_policy: Option<RetryPolicy>,
    pub circuit_breaker: Option<CircuitBreakerPolicy>,
}

/// Retry policy configuration
#[derive(Debug, Clone)]
pub struct RetryPolicy {
    pub max_attempts: u32,
    pub timeout: std::time::Duration,
    pub backoff_strategy: BackoffStrategy,
}

/// Backoff strategies
#[derive(Debug, Clone, PartialEq)]
pub enum BackoffStrategy {
    Fixed,
    Exponential,
    Linear,
}

/// Circuit breaker policy
#[derive(Debug, Clone)]
pub struct CircuitBreakerPolicy {
    pub failure_threshold: u32,
    pub success_threshold: u32,
    pub timeout: std::time::Duration,
}

/// Security policies
#[derive(Debug, Clone)]
pub struct SecurityPolicy {
    pub name: String,
    pub policy_type: SecurityPolicyType,
    pub rules: Vec<SecurityRule>,
}

/// Security policy types
#[derive(Debug, Clone, PartialEq)]
pub enum SecurityPolicyType {
    NetworkPolicy,
    PodSecurityPolicy,
    RBAC,
    AuthorizationPolicy,
}

/// Security rules
#[derive(Debug, Clone)]
pub struct SecurityRule {
    pub action: SecurityAction,
    pub source: SecurityPrincipal,
    pub destination: SecurityPrincipal,
    pub conditions: Vec<SecurityCondition>,
}

/// Security actions
#[derive(Debug, Clone, PartialEq)]
pub enum SecurityAction {
    Allow,
    Deny,
    Log,
    Audit,
}

/// Security principals
#[derive(Debug, Clone)]
pub struct SecurityPrincipal {
    pub principal_type: PrincipalType,
    pub identifiers: Vec<String>,
}

/// Principal types
#[derive(Debug, Clone, PartialEq)]
pub enum PrincipalType {
    User,
    ServiceAccount,
    Group,
    Namespace,
    IPAddress,
    CIDR,
}

/// Security conditions
#[derive(Debug, Clone)]
pub struct SecurityCondition {
    pub condition_type: ConditionType,
    pub values: Vec<String>,
}

/// Condition types
#[derive(Debug, Clone, PartialEq)]
pub enum ConditionType {
    Header,
    Method,
    Path,
    Query,
    SourceIP,
    DestinationPort,
}

/// Infrastructure recommendations
#[derive(Debug, Clone)]
pub struct InfrastructureRecommendation {
    pub recommendation_type: InfrastructureRecommendationType,
    pub confidence_score: f64,
    pub reasoning: String,
    pub implementation_steps: Vec<String>,
    pub estimated_cost_impact: f64,
    pub performance_impact: f64,
    pub security_impact: f64,
}

/// Infrastructure recommendation types
#[derive(Debug, Clone, PartialEq)]
pub enum InfrastructureRecommendationType {
    InstanceType,
    StorageType,
    NetworkConfiguration,
    SecurityConfiguration,
    MonitoringSetup,
    BackupStrategy,
    ScalingPolicy,
    CostOptimization,
}