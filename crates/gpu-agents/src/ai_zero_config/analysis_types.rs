//! Analysis-related types for AI Assistant Zero-Config Integration
//! Split from types.rs to keep files under 750 lines

use std::collections::HashMap;
use std::path::PathBuf;

/// Code analysis result
#[derive(Debug, Clone)]
pub struct AnalysisResult {
    pub primary_language: ProgrammingLanguage,
    pub detected_languages: Vec<ProgrammingLanguage>,
    pub frameworks: Vec<Framework>,
    pub dependencies: Vec<Dependency>,
    pub estimated_resources: ResourceEstimate,
    pub deployment_patterns: Vec<super::DeploymentPattern>,
    pub security_analysis: SecurityAnalysis,
    pub complexity_metrics: ComplexityMetrics,
    pub architecture_style: ArchitectureStyle,
}

/// Programming languages
#[derive(Debug, Clone, PartialEq)]
pub enum ProgrammingLanguage {
    Rust,
    Python,
    JavaScript,
    TypeScript,
    Go,
    Java,
    CSharp,
    CPlusPlus,
    C,
    Ruby,
    PHP,
    Swift,
    Kotlin,
    Scala,
    Unknown,
}

/// Software frameworks
#[derive(Debug, Clone, PartialEq)]
pub struct Framework {
    pub name: String,
    pub framework_type: FrameworkType,
    pub version: Option<String>,
}

/// Types of frameworks
#[derive(Debug, Clone, PartialEq)]
pub enum FrameworkType {
    WebFramework,
    DatabaseORM,
    MLFramework,
    TestingFramework,
    UIFramework,
    MessageQueue,
    Cache,
    Monitoring,
    Security,
    BuildTool,
}

/// Dependency information
#[derive(Debug, Clone)]
pub struct Dependency {
    pub name: String,
    pub version: Option<String>,
    pub dependency_type: DependencyType,
    pub security_vulnerabilities: Vec<SecurityVulnerability>,
    pub license: Option<String>,
}

/// Types of dependencies
#[derive(Debug, Clone, PartialEq)]
pub enum DependencyType {
    Runtime,
    Development,
    Testing,
    Optional,
    Peer,
    System,
}

/// Security vulnerability information
#[derive(Debug, Clone)]
pub struct SecurityVulnerability {
    pub id: String,
    pub severity: VulnerabilitySeverity,
    pub description: String,
    pub fixed_version: Option<String>,
}

/// Vulnerability severity levels
#[derive(Debug, Clone, PartialEq)]
pub enum VulnerabilitySeverity {
    Critical,
    High,
    Medium,
    Low,
    Info,
}

/// Resource estimation
#[derive(Debug, Clone)]
pub struct ResourceEstimate {
    pub cpu_cores: f64,
    pub memory_mb: u64,
    pub storage_gb: u64,
    pub network_bandwidth_mbps: u64,
    pub gpu_memory_mb: Option<u64>,
    pub estimated_cost_monthly: f64,
    pub scaling_characteristics: ScalingCharacteristics,
}

/// Scaling behavior characteristics
#[derive(Debug, Clone)]
pub struct ScalingCharacteristics {
    pub cpu_scaling: ScalingPattern,
    pub memory_scaling: ScalingPattern,
    pub storage_scaling: ScalingPattern,
    pub network_scaling: ScalingPattern,
    pub scale_to_zero_capable: bool,
}

/// Scaling patterns
#[derive(Debug, Clone, PartialEq)]
pub enum ScalingPattern {
    Linear,
    Logarithmic,
    Exponential,
    Constant,
    Seasonal,
    Burst,
}

/// Security analysis results
#[derive(Debug, Clone)]
pub struct SecurityAnalysis {
    pub authentication_methods: Vec<AuthenticationMethod>,
    pub data_sensitivity_level: DataSensitivityLevel,
    pub network_exposure: NetworkExposure,
    pub secret_usage: Vec<SecretUsage>,
    pub security_best_practices: SecurityPracticeCompliance,
}

/// Authentication methods
#[derive(Debug, Clone, PartialEq)]
pub enum AuthenticationMethod {
    JWT,
    OAuth2,
    BasicAuth,
    ApiKey,
    Certificate,
    SAML,
    LDAP,
    None,
}

/// Data sensitivity levels
#[derive(Debug, Clone, PartialEq)]
pub enum DataSensitivityLevel {
    Public,
    Internal,
    Confidential,
    Restricted,
    TopSecret,
}

/// Network exposure levels
#[derive(Debug, Clone, PartialEq)]
pub enum NetworkExposure {
    Public,
    Internal,
    Private,
    VPN,
    Isolated,
}

/// Secret usage patterns
#[derive(Debug, Clone)]
pub struct SecretUsage {
    pub secret_type: SecretType,
    pub usage_context: String,
    pub best_practice_compliance: bool,
}

/// Types of secrets
#[derive(Debug, Clone, PartialEq)]
pub enum SecretType {
    DatabasePassword,
    APIKey,
    CertificateKey,
    EncryptionKey,
    SessionSecret,
    TokenSigningKey,
}

/// Security best practice compliance
#[derive(Debug, Clone)]
pub struct SecurityPracticeCompliance {
    pub input_validation: bool,
    pub output_encoding: bool,
    pub sql_injection_protection: bool,
    pub xss_protection: bool,
    pub csrf_protection: bool,
    pub secure_headers: bool,
    pub rate_limiting: bool,
}

/// Code complexity metrics
#[derive(Debug, Clone)]
pub struct ComplexityMetrics {
    pub cyclomatic_complexity: u32,
    pub lines_of_code: u32,
    pub number_of_files: u32,
    pub dependency_count: u32,
    pub maintainability_index: f64,
    pub technical_debt_ratio: f64,
}

/// Architecture styles
#[derive(Debug, Clone, PartialEq)]
pub enum ArchitectureStyle {
    Layered,
    EventDriven,
    Microkernel,
    Pipeline,
    ClientServer,
    ComponentBased,
    ServiceOriented,
    EventSourcing,
    CQRS,
    Hexagonal,
}

/// Codebase context for analysis
#[derive(Debug, Clone)]
pub struct CodebaseContext {
    pub root_path: PathBuf,
    pub total_files: u32,
    pub total_lines: u32,
    pub languages: HashMap<ProgrammingLanguage, LanguageStats>,
    pub project_structure: ProjectStructure,
    pub build_systems: Vec<BuildSystem>,
    pub package_managers: Vec<PackageManager>,
}

/// Language statistics
#[derive(Debug, Clone)]
pub struct LanguageStats {
    pub file_count: u32,
    pub line_count: u32,
    pub percentage: f64,
}

/// Project structure information
#[derive(Debug, Clone)]
pub struct ProjectStructure {
    pub project_type: ProjectType,
    pub modules: Vec<Module>,
    pub entry_points: Vec<String>,
    pub test_directories: Vec<String>,
    pub documentation_files: Vec<String>,
}

/// Project types
#[derive(Debug, Clone, PartialEq)]
pub enum ProjectType {
    Library,
    Application,
    Service,
    Framework,
    Tool,
    Monorepo,
    Plugin,
}

/// Module information
#[derive(Debug, Clone)]
pub struct Module {
    pub name: String,
    pub path: String,
    pub module_type: ModuleType,
    pub dependencies: Vec<String>,
    pub exports: Vec<String>,
}

/// Module types
#[derive(Debug, Clone, PartialEq)]
pub enum ModuleType {
    Core,
    Feature,
    Utility,
    Test,
    Configuration,
    Documentation,
}

/// Build systems
#[derive(Debug, Clone, PartialEq)]
pub enum BuildSystem {
    Cargo,
    NPM,
    Yarn,
    Maven,
    Gradle,
    Make,
    CMake,
    Bazel,
    Buck,
}

/// Package managers
#[derive(Debug, Clone, PartialEq)]
pub enum PackageManager {
    Cargo,
    NPM,
    Yarn,
    Pip,
    Pipenv,
    Poetry,
    Conda,
    Maven,
    Gradle,
}