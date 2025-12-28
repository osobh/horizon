//! Install script generation for curl-based node bootstrapping.
//!
//! This module generates dynamic shell scripts that can be served via HTTP
//! to enable one-liner node onboarding:
//!
//! ```bash
//! curl -sSL "https://cluster.example.com/api/v1/install?token=TOKEN" | bash
//! ```

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Template script embedded at compile time.
const INSTALL_TEMPLATE: &str = include_str!("../scripts/install-template.sh");

/// Configuration for generating an install script.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstallScriptConfig {
    /// Cluster hostname or IP address
    pub cluster_host: String,
    /// Cluster port
    pub cluster_port: u16,
    /// Join token for authentication
    pub token: String,
    /// When the token expires
    pub token_expiry: DateTime<Utc>,
    /// Swarmlet version to install
    pub swarmlet_version: String,
    /// Docker image for container deployment
    pub docker_image: String,
    /// Base URL for binary releases
    pub releases_base_url: String,
    /// Binary checksums for verification
    pub checksums: BinaryChecksums,
}

/// SHA-256 checksums for platform-specific binaries.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BinaryChecksums {
    pub linux_amd64: String,
    pub linux_arm64: String,
    pub darwin_amd64: String,
    pub darwin_arm64: String,
}

impl InstallScriptConfig {
    /// Create a new config with default values.
    pub fn new(cluster_host: String, cluster_port: u16, token: String) -> Self {
        Self {
            cluster_host,
            cluster_port,
            token,
            token_expiry: Utc::now() + chrono::Duration::hours(24),
            swarmlet_version: env!("CARGO_PKG_VERSION").to_string(),
            docker_image: "stratoswarm/swarmlet".to_string(),
            releases_base_url: "https://releases.stratoswarm.io/swarmlet".to_string(),
            checksums: BinaryChecksums::default(),
        }
    }

    /// Set the token expiry time.
    #[must_use]
    pub fn with_expiry(mut self, expiry: DateTime<Utc>) -> Self {
        self.token_expiry = expiry;
        self
    }

    /// Set the swarmlet version.
    #[must_use]
    pub fn with_version(mut self, version: String) -> Self {
        self.swarmlet_version = version;
        self
    }

    /// Set the Docker image.
    #[must_use]
    pub fn with_docker_image(mut self, image: String) -> Self {
        self.docker_image = image;
        self
    }

    /// Set the releases base URL.
    #[must_use]
    pub fn with_releases_url(mut self, url: String) -> Self {
        self.releases_base_url = url;
        self
    }

    /// Set the binary checksums.
    #[must_use]
    pub fn with_checksums(mut self, checksums: BinaryChecksums) -> Self {
        self.checksums = checksums;
        self
    }
}

/// Generate an install script with the given configuration.
///
/// This function takes the template script and replaces placeholders
/// with the actual configuration values.
///
/// # Arguments
///
/// * `config` - The configuration for the install script
///
/// # Returns
///
/// A string containing the complete shell script ready for execution.
///
/// # Example
///
/// ```
/// use cluster_mesh::install_script::{generate_install_script, InstallScriptConfig};
///
/// let config = InstallScriptConfig::new(
///     "cluster.example.com".to_string(),
///     7946,
///     "my-secret-token".to_string(),
/// );
///
/// let script = generate_install_script(&config);
/// assert!(script.contains("cluster.example.com"));
/// ```
pub fn generate_install_script(config: &InstallScriptConfig) -> String {
    INSTALL_TEMPLATE
        .replace("{{GENERATED_AT}}", &Utc::now().to_rfc3339())
        .replace("{{TOKEN_EXPIRY}}", &config.token_expiry.to_rfc3339())
        .replace("{{CLUSTER_HOST}}", &config.cluster_host)
        .replace("{{CLUSTER_PORT}}", &config.cluster_port.to_string())
        .replace("{{CLUSTER_TOKEN}}", &config.token)
        .replace("{{SWARMLET_VERSION}}", &config.swarmlet_version)
        .replace("{{DOCKER_IMAGE}}", &config.docker_image)
        .replace("{{RELEASES_BASE_URL}}", &config.releases_base_url)
        .replace("{{CHECKSUM_LINUX_AMD64}}", &config.checksums.linux_amd64)
        .replace("{{CHECKSUM_LINUX_ARM64}}", &config.checksums.linux_arm64)
        .replace("{{CHECKSUM_DARWIN_AMD64}}", &config.checksums.darwin_amd64)
        .replace("{{CHECKSUM_DARWIN_ARM64}}", &config.checksums.darwin_arm64)
}

/// HTTP query parameters for the install endpoint.
#[derive(Debug, Clone, Deserialize)]
pub struct InstallParams {
    /// Join token (required)
    pub token: String,
    /// Optional node name override
    pub node_name: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_install_script() {
        let config = InstallScriptConfig::new(
            "test-cluster.local".to_string(),
            7946,
            "test-token-12345".to_string(),
        );

        let script = generate_install_script(&config);

        // Verify placeholders are replaced
        assert!(script.contains("test-cluster.local"));
        assert!(script.contains("7946"));
        assert!(script.contains("test-token-12345"));

        // Verify no template placeholders remain
        assert!(!script.contains("{{CLUSTER_HOST}}"));
        assert!(!script.contains("{{CLUSTER_TOKEN}}"));
    }

    #[test]
    fn test_config_builder() {
        let checksums = BinaryChecksums {
            linux_amd64: "abc123".to_string(),
            linux_arm64: "def456".to_string(),
            darwin_amd64: "ghi789".to_string(),
            darwin_arm64: "jkl012".to_string(),
        };

        let config = InstallScriptConfig::new("cluster.io".to_string(), 8080, "token".to_string())
            .with_version("2.0.0".to_string())
            .with_docker_image("custom/image".to_string())
            .with_checksums(checksums);

        assert_eq!(config.swarmlet_version, "2.0.0");
        assert_eq!(config.docker_image, "custom/image");
        assert_eq!(config.checksums.linux_amd64, "abc123");
    }

    #[test]
    fn test_script_has_shebang() {
        let config = InstallScriptConfig::new("cluster.io".to_string(), 7946, "token".to_string());

        let script = generate_install_script(&config);
        assert!(script.starts_with("#!/bin/bash"));
    }

    #[test]
    fn test_script_validates_placeholder_check() {
        // The script should detect if it still has placeholder values
        let config =
            InstallScriptConfig::new("cluster.io".to_string(), 7946, "real-token".to_string());

        let script = generate_install_script(&config);

        // The validation check in the script looks for {{CLUSTER_TOKEN}}
        // If the token is properly replaced, this check should pass
        assert!(!script.contains("{{CLUSTER_TOKEN}}"));
    }
}
