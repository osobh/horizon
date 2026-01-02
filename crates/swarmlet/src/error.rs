//! Error types for the swarmlet

use thiserror::Error;

/// Result type for swarmlet operations
pub type Result<T> = std::result::Result<T, SwarmletError>;

/// Swarmlet error types
#[derive(Error, Debug)]
pub enum SwarmletError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Network error: {0}")]
    Network(#[from] reqwest::Error),

    #[error("Join protocol error: {0}")]
    JoinProtocol(String),

    #[error("Hardware profiling error: {0}")]
    HardwareProfiling(String),

    #[error("Cluster discovery error: {0}")]
    Discovery(String),

    #[error("Authentication error: {0}")]
    Authentication(String),

    #[error("Configuration error: {0}")]
    Configuration(String),

    #[error("Workload execution error: {0}")]
    WorkloadExecution(String),

    #[error("Command execution error: {0}")]
    CommandExecution(String),

    #[error("Agent runtime error: {0}")]
    AgentRuntime(String),

    #[error("Invalid token: {0}")]
    InvalidToken(String),

    #[error("Connection timeout")]
    Timeout,

    #[error("No cluster found")]
    NoClusterFound,

    #[error("Cluster rejection: {0}")]
    ClusterRejection(String),

    #[error("Feature not implemented: {0}")]
    NotImplemented(String),

    #[error("Docker error: {0}")]
    Docker(String),

    #[cfg(feature = "docker")]
    #[error("Docker API error: {0}")]
    DockerApi(#[from] bollard::errors::Error),

    #[error("TOML parsing error: {0}")]
    TomlParsing(#[from] toml::de::Error),

    #[error("Crypto error: {0}")]
    Crypto(String),

    #[error("System error: {0}")]
    System(String),

    #[error("WireGuard error: {0}")]
    WireGuard(String),

    #[error("API error: {0}")]
    Api(String),

    #[error("Build error in phase '{phase}': {message}")]
    Build {
        phase: BuildPhase,
        message: String,
        #[source]
        source: Option<Box<SwarmletError>>,
    },
}

/// Build phases for error context
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuildPhase {
    PreparingEnvironment,
    FetchingSource,
    ProvisioningToolchain,
    Building,
    CollectingArtifacts,
    Cleanup,
}

impl std::fmt::Display for BuildPhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BuildPhase::PreparingEnvironment => write!(f, "preparing_environment"),
            BuildPhase::FetchingSource => write!(f, "fetching_source"),
            BuildPhase::ProvisioningToolchain => write!(f, "provisioning_toolchain"),
            BuildPhase::Building => write!(f, "building"),
            BuildPhase::CollectingArtifacts => write!(f, "collecting_artifacts"),
            BuildPhase::Cleanup => write!(f, "cleanup"),
        }
    }
}

impl SwarmletError {
    /// Create a build error with phase context
    pub fn build_error(phase: BuildPhase, message: impl Into<String>) -> Self {
        SwarmletError::Build {
            phase,
            message: message.into(),
            source: None,
        }
    }

    /// Create a build error with phase context and source error
    pub fn build_error_with_source(phase: BuildPhase, message: impl Into<String>, source: SwarmletError) -> Self {
        SwarmletError::Build {
            phase,
            message: message.into(),
            source: Some(Box::new(source)),
        }
    }
}

impl SwarmletError {
    /// Check if the error is recoverable (e.g., temporary network issues)
    pub fn is_recoverable(&self) -> bool {
        matches!(self, SwarmletError::Network(_) | SwarmletError::Timeout | SwarmletError::Discovery(_) | SwarmletError::CommandExecution(_) | SwarmletError::Io(_))
    }

    /// Get error severity level
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            SwarmletError::Authentication(_) => ErrorSeverity::Critical,
            SwarmletError::InvalidToken(_) => ErrorSeverity::Critical,
            SwarmletError::ClusterRejection(_) => ErrorSeverity::High,
            SwarmletError::Configuration(_) => ErrorSeverity::High,
            SwarmletError::AgentRuntime(_) => ErrorSeverity::High,
            SwarmletError::Network(_) => ErrorSeverity::Medium,
            SwarmletError::Timeout => ErrorSeverity::Medium,
            SwarmletError::Discovery(_) => ErrorSeverity::Medium,
            SwarmletError::CommandExecution(_) => ErrorSeverity::Medium,
            SwarmletError::HardwareProfiling(_) => ErrorSeverity::Low,
            SwarmletError::NotImplemented(_) => ErrorSeverity::Low,
            _ => ErrorSeverity::Medium,
        }
    }
}

/// Error severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorSeverity {
    Low,
    Medium,
    High,
    Critical,
}

impl std::fmt::Display for ErrorSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ErrorSeverity::Low => write!(f, "LOW"),
            ErrorSeverity::Medium => write!(f, "MEDIUM"),
            ErrorSeverity::High => write!(f, "HIGH"),
            ErrorSeverity::Critical => write!(f, "CRITICAL"),
        }
    }
}
