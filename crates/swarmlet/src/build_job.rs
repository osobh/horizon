//! Build job types for CI/CD build containers
//!
//! This module defines the core data structures for Rust/cargo build jobs,
//! including job specifications, toolchain configuration, and build status.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use uuid::Uuid;

/// Represents a Rust/cargo build job request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildJob {
    /// Unique job identifier
    pub id: Uuid,
    /// Cargo command to execute
    pub command: CargoCommand,
    /// Rust toolchain specification
    pub toolchain: RustToolchain,
    /// Source code location
    pub source: BuildSource,
    /// Target triple (e.g., "x86_64-unknown-linux-gnu")
    pub target: Option<String>,
    /// Build profile
    pub profile: BuildProfile,
    /// Cargo features to enable
    pub features: Vec<String>,
    /// Environment variables for the build
    pub environment: HashMap<String, String>,
    /// Resource limits for the build container
    pub resource_limits: BuildResourceLimits,
    /// Cache configuration
    pub cache_config: CacheConfig,
    /// When the job was created
    pub created_at: DateTime<Utc>,
    /// Optional deadline for the build
    pub deadline: Option<DateTime<Utc>>,
}

impl BuildJob {
    /// Create a new build job with default settings
    pub fn new(command: CargoCommand, source: BuildSource) -> Self {
        Self {
            id: Uuid::new_v4(),
            command,
            toolchain: RustToolchain::default(),
            source,
            target: None,
            profile: BuildProfile::Debug,
            features: Vec::new(),
            environment: HashMap::new(),
            resource_limits: BuildResourceLimits::default(),
            cache_config: CacheConfig::default(),
            created_at: Utc::now(),
            deadline: None,
        }
    }

    /// Set the toolchain for this build
    pub fn with_toolchain(mut self, toolchain: RustToolchain) -> Self {
        self.toolchain = toolchain;
        self
    }

    /// Set the build profile
    pub fn with_profile(mut self, profile: BuildProfile) -> Self {
        self.profile = profile;
        self
    }

    /// Add features to enable
    pub fn with_features(mut self, features: Vec<String>) -> Self {
        self.features = features;
        self
    }

    /// Set resource limits
    pub fn with_resource_limits(mut self, limits: BuildResourceLimits) -> Self {
        self.resource_limits = limits;
        self
    }

    /// Build cargo command line arguments
    pub fn cargo_args(&self) -> Vec<String> {
        let mut args = Vec::new();

        // Add the main command
        match &self.command {
            CargoCommand::Build => args.push("build".to_string()),
            CargoCommand::Test { filter } => {
                args.push("test".to_string());
                if let Some(f) = filter {
                    args.push(f.clone());
                }
            }
            CargoCommand::Check => args.push("check".to_string()),
            CargoCommand::Clippy { deny_warnings } => {
                args.push("clippy".to_string());
                if *deny_warnings {
                    args.push("--".to_string());
                    args.push("-D".to_string());
                    args.push("warnings".to_string());
                }
            }
            CargoCommand::Doc { open: _ } => args.push("doc".to_string()),
            CargoCommand::Run {
                bin,
                args: ref run_args,
            } => {
                args.push("run".to_string());
                if let Some(ref b) = bin {
                    args.push("--bin".to_string());
                    args.push(b.clone());
                }
                if !run_args.is_empty() {
                    args.push("--".to_string());
                    args.extend(run_args.clone());
                }
            }
            CargoCommand::Bench { filter } => {
                args.push("bench".to_string());
                if let Some(f) = filter {
                    args.push(f.clone());
                }
            }
        }

        // Add profile flag
        match &self.profile {
            BuildProfile::Release => args.push("--release".to_string()),
            BuildProfile::Custom(name) => {
                args.push("--profile".to_string());
                args.push(name.clone());
            }
            BuildProfile::Debug => {} // Default, no flag needed
        }

        // Add target if specified
        if let Some(target) = &self.target {
            args.push("--target".to_string());
            args.push(target.clone());
        }

        // Add features
        if !self.features.is_empty() {
            args.push("--features".to_string());
            args.push(self.features.join(","));
        }

        args
    }
}

/// Cargo commands supported for build jobs
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
#[derive(Default)]
pub enum CargoCommand {
    /// cargo build
    #[default]
    Build,
    /// cargo test with optional filter
    Test { filter: Option<String> },
    /// cargo check
    Check,
    /// cargo clippy with optional deny warnings
    Clippy { deny_warnings: bool },
    /// cargo doc
    Doc { open: bool },
    /// cargo run with optional binary name and arguments
    Run {
        bin: Option<String>,
        args: Vec<String>,
    },
    /// cargo bench with optional filter
    Bench { filter: Option<String> },
}

/// Rust toolchain specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RustToolchain {
    /// Channel: "stable", "beta", "nightly", or specific version like "1.75.0"
    pub channel: String,
    /// Optional date for nightly (e.g., "2024-01-15")
    pub date: Option<String>,
    /// Additional components (e.g., "rustfmt", "clippy", "rust-src")
    pub components: Vec<String>,
    /// Target triples to install
    pub targets: Vec<String>,
}

impl Default for RustToolchain {
    fn default() -> Self {
        Self {
            channel: "stable".to_string(),
            date: None,
            components: vec!["rustfmt".to_string(), "clippy".to_string()],
            targets: Vec::new(),
        }
    }
}

impl RustToolchain {
    /// Create a stable toolchain
    pub fn stable() -> Self {
        Self::default()
    }

    /// Create a nightly toolchain
    pub fn nightly() -> Self {
        Self {
            channel: "nightly".to_string(),
            ..Default::default()
        }
    }

    /// Create a specific version toolchain
    pub fn version(version: impl Into<String>) -> Self {
        Self {
            channel: version.into(),
            ..Default::default()
        }
    }

    /// Get the toolchain string for rustup
    pub fn toolchain_string(&self) -> String {
        if let Some(date) = &self.date {
            format!("{}-{}", self.channel, date)
        } else {
            self.channel.clone()
        }
    }
}

/// Source code location for a build job
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum BuildSource {
    /// Git repository with optional branch/tag/commit
    Git {
        url: String,
        reference: Option<GitReference>,
        depth: Option<u32>,
    },
    /// Pre-uploaded source archive (tar.gz)
    Archive { url: String, sha256: Option<String> },
    /// Reference to previously cached source
    Cached { hash: String },
    /// Local path (for testing)
    Local { path: PathBuf },
}

/// Git reference type
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum GitReference {
    Branch(String),
    Tag(String),
    Commit(String),
}

/// Build profile
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum BuildProfile {
    #[default]
    Debug,
    Release,
    Custom(String),
}

/// Resource limits for a build container
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildResourceLimits {
    /// CPU cores to allocate (fractional, e.g., 2.5 cores)
    pub cpu_cores: Option<f32>,
    /// Memory limit in bytes
    pub memory_bytes: Option<u64>,
    /// Disk space limit in bytes
    pub disk_bytes: Option<u64>,
    /// Build timeout in seconds
    pub timeout_seconds: Option<u64>,
}

impl Default for BuildResourceLimits {
    fn default() -> Self {
        Self {
            cpu_cores: Some(4.0),
            memory_bytes: Some(8 * 1024 * 1024 * 1024), // 8GB
            disk_bytes: Some(20 * 1024 * 1024 * 1024),  // 20GB
            timeout_seconds: Some(3600),                // 1 hour
        }
    }
}

impl BuildResourceLimits {
    /// Create minimal resource limits for quick builds
    pub fn minimal() -> Self {
        Self {
            cpu_cores: Some(1.0),
            memory_bytes: Some(2 * 1024 * 1024 * 1024), // 2GB
            disk_bytes: Some(5 * 1024 * 1024 * 1024),   // 5GB
            timeout_seconds: Some(600),                 // 10 minutes
        }
    }

    /// Create resource limits for large builds
    pub fn large() -> Self {
        Self {
            cpu_cores: Some(8.0),
            memory_bytes: Some(32 * 1024 * 1024 * 1024), // 32GB
            disk_bytes: Some(100 * 1024 * 1024 * 1024),  // 100GB
            timeout_seconds: Some(7200),                 // 2 hours
        }
    }
}

/// Cache configuration for build jobs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Use shared cargo registry
    pub use_cargo_registry: bool,
    /// Use sccache for compilation cache
    pub use_sccache: bool,
    /// Cache the target directory between builds
    pub cache_target: bool,
    /// Unique cache key for this project
    pub cache_key: Option<String>,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            use_cargo_registry: true,
            use_sccache: true,
            cache_target: true,
            cache_key: None,
        }
    }
}

impl CacheConfig {
    /// Configuration with no caching
    pub fn no_cache() -> Self {
        Self {
            use_cargo_registry: false,
            use_sccache: false,
            cache_target: false,
            cache_key: None,
        }
    }
}

/// An active build job running on this swarmlet
#[derive(Debug, Clone, Serialize)]
pub struct ActiveBuildJob {
    /// Job ID
    pub id: Uuid,
    /// Original job specification
    pub job: BuildJob,
    /// Current status
    pub status: BuildJobStatus,
    /// When the job started
    pub started_at: DateTime<Utc>,
    /// Process ID (for native Linux isolation)
    pub pid: Option<u32>,
    /// Container ID (for Docker backend)
    pub container_id: Option<String>,
    /// Build log entries
    pub output_log: Vec<BuildLogEntry>,
    /// Build artifacts produced
    pub artifacts: Vec<BuildArtifact>,
    /// Resource usage statistics
    pub resource_usage: BuildResourceUsage,
}

/// Current status of a build job
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum BuildJobStatus {
    /// Job is queued waiting for execution
    Queued,
    /// Preparing the build environment
    PreparingEnvironment,
    /// Fetching source code
    FetchingSource,
    /// Provisioning the Rust toolchain
    ProvisioningToolchain,
    /// Build is running
    Building,
    /// Tests are running
    Testing,
    /// Collecting build artifacts
    CollectingArtifacts,
    /// Build completed successfully
    Completed,
    /// Build failed with error
    Failed { error: String },
    /// Build was cancelled
    Cancelled,
    /// Build timed out
    TimedOut,
}

impl BuildJobStatus {
    /// Check if the status represents a terminal state
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            BuildJobStatus::Completed
                | BuildJobStatus::Failed { .. }
                | BuildJobStatus::Cancelled
                | BuildJobStatus::TimedOut
        )
    }

    /// Check if the status represents a successful completion
    pub fn is_success(&self) -> bool {
        matches!(self, BuildJobStatus::Completed)
    }
}

/// Build log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildLogEntry {
    /// When the log entry was recorded
    pub timestamp: DateTime<Utc>,
    /// Stream type (stdout, stderr, or system)
    pub stream: LogStream,
    /// Log message
    pub message: String,
}

/// Log stream type
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LogStream {
    Stdout,
    Stderr,
    System,
}

/// Build artifact
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildArtifact {
    /// Artifact name (e.g., binary name)
    pub name: String,
    /// Path to the artifact
    pub path: PathBuf,
    /// Size in bytes
    pub size_bytes: u64,
    /// Type of artifact
    pub artifact_type: ArtifactType,
    /// SHA256 hash of the artifact
    pub sha256: String,
}

/// Type of build artifact
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactType {
    Binary,
    Library,
    TestResults,
    Documentation,
    Coverage,
}

/// Resource usage statistics for a build
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BuildResourceUsage {
    /// Total CPU time consumed in seconds
    pub cpu_seconds: f64,
    /// Peak memory usage in megabytes
    pub peak_memory_mb: f32,
    /// Bytes read from disk
    pub disk_read_bytes: u64,
    /// Bytes written to disk
    pub disk_write_bytes: u64,
    /// Total compile time in seconds
    pub compile_time_seconds: f64,
    /// Number of crates compiled
    pub crates_compiled: u32,
    /// Cache hits (sccache)
    pub cache_hits: u32,
    /// Cache misses (sccache)
    pub cache_misses: u32,
}

/// Build result returned after execution
#[derive(Debug, Clone)]
pub struct BuildResult {
    /// Exit code from cargo
    pub exit_code: i32,
    /// Resource usage during build
    pub resource_usage: BuildResourceUsage,
    /// Total duration
    pub duration: std::time::Duration,
    /// Collected artifacts
    pub artifacts: Vec<BuildArtifact>,
}

impl BuildResult {
    /// Check if the build succeeded (exit code 0)
    pub fn is_success(&self) -> bool {
        self.exit_code == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_job_creation() {
        let source = BuildSource::Git {
            url: "https://github.com/example/repo.git".to_string(),
            reference: Some(GitReference::Branch("main".to_string())),
            depth: Some(1),
        };
        let job = BuildJob::new(CargoCommand::Build, source);

        assert!(!job.id.is_nil());
        assert_eq!(job.toolchain.channel, "stable");
    }

    #[test]
    fn test_cargo_args_build() {
        let source = BuildSource::Local {
            path: PathBuf::from("."),
        };
        let job = BuildJob::new(CargoCommand::Build, source)
            .with_profile(BuildProfile::Release)
            .with_features(vec!["foo".to_string(), "bar".to_string()]);

        let args = job.cargo_args();
        assert!(args.contains(&"build".to_string()));
        assert!(args.contains(&"--release".to_string()));
        assert!(args.contains(&"--features".to_string()));
        assert!(args.contains(&"foo,bar".to_string()));
    }

    #[test]
    fn test_cargo_args_test() {
        let source = BuildSource::Local {
            path: PathBuf::from("."),
        };
        let job = BuildJob::new(
            CargoCommand::Test {
                filter: Some("my_test".to_string()),
            },
            source,
        );

        let args = job.cargo_args();
        assert!(args.contains(&"test".to_string()));
        assert!(args.contains(&"my_test".to_string()));
    }

    #[test]
    fn test_toolchain_string() {
        let stable = RustToolchain::stable();
        assert_eq!(stable.toolchain_string(), "stable");

        let nightly = RustToolchain::nightly();
        assert_eq!(nightly.toolchain_string(), "nightly");

        let dated = RustToolchain {
            channel: "nightly".to_string(),
            date: Some("2024-01-15".to_string()),
            ..Default::default()
        };
        assert_eq!(dated.toolchain_string(), "nightly-2024-01-15");
    }

    #[test]
    fn test_build_job_status_terminal() {
        assert!(!BuildJobStatus::Queued.is_terminal());
        assert!(!BuildJobStatus::Building.is_terminal());
        assert!(BuildJobStatus::Completed.is_terminal());
        assert!(BuildJobStatus::Failed {
            error: "test".to_string()
        }
        .is_terminal());
        assert!(BuildJobStatus::Cancelled.is_terminal());
        assert!(BuildJobStatus::TimedOut.is_terminal());
    }

    #[test]
    fn test_resource_limits_defaults() {
        let limits = BuildResourceLimits::default();
        assert_eq!(limits.cpu_cores, Some(4.0));
        assert_eq!(limits.memory_bytes, Some(8 * 1024 * 1024 * 1024));
    }
}
