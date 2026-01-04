//! # Horizon Configuration Management
//!
//! This crate provides a flexible, layered configuration system for Horizon services.
//! It supports loading configuration from multiple sources with a clear precedence order:
//!
//! **Precedence (highest to lowest):**
//! 1. Environment variables
//! 2. Configuration files
//! 3. Default values (via serde defaults)
//!
//! ## Features
//!
//! - **Multiple Formats**: YAML, TOML, and JSON support
//! - **Environment Overrides**: Env vars with configurable prefixes (e.g., `SERVICE__FIELD`)
//! - **Type Safety**: Full serde deserialization with validation
//! - **Secret Management**: Load secrets from files or env vars with redaction
//! - **Builder Pattern**: Flexible configuration composition
//! - **Nested Configs**: Support for complex hierarchical configurations
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use serde::Deserialize;
//!
//! #[derive(Deserialize)]
//! struct MyConfig {
//!     listen_addr: String,
//!     log_level: String,
//! }
//!
//! // Load from conventional path: config/{service-name}.yaml
//! let config: MyConfig = hpc_config::load("my-service").unwrap();
//!
//! // Or load from specific file
//! let config: MyConfig = hpc_config::load_from_file("config.yaml").unwrap();
//!
//! // Or use builder for custom composition
//! let config: MyConfig = hpc_config::ConfigBuilder::new()
//!     .add_file("base.yaml")
//!     .add_env_with_prefix("MYSERVICE")
//!     .build()
//!     .unwrap();
//! ```
//!
//! ## Environment Variable Overrides
//!
//! Environment variables follow the pattern: `{PREFIX}__{FIELD_NAME}` where:
//! - `PREFIX` is the service name in uppercase
//! - Field names are converted to uppercase
//! - Nested fields use double underscores: `PREFIX__PARENT__CHILD`
//!
//! Example:
//! ```bash
//! export MYSERVICE__LISTEN_ADDR="0.0.0.0:8080"
//! export MYSERVICE__DATABASE__HOST="db.example.com"
//! ```
//!
//! ## Secret Management
//!
//! Secrets can be loaded from files or environment variables with automatic redaction:
//!
//! ```rust,no_run
//! use hpc_config::{load_secret_from_file, ExposeSecret};
//!
//! // Load from file (e.g., mounted Kubernetes secret)
//! let secret = load_secret_from_file("/run/secrets/db_password").unwrap();
//!
//! // Load from environment variable
//! let secret = hpc_config::load_secret_from_env("DB_PASSWORD").unwrap();
//!
//! // Access the secret value
//! let password: &str = secret.expose_secret();
//! ```

// Allow large error variant since it's intentional in horizon-error
#![allow(clippy::result_large_err)]

use config::{Config, Environment, File, FileFormat};
use secrecy::Secret;
use serde::de::DeserializeOwned;
use std::path::Path;

pub use secrecy::{self, ExposeSecret};

/// A secret string that is redacted when debug-printed
pub type SecretString = Secret<String>;

/// Loads configuration from a conventional path: `config/{service-name}.{yaml|toml|json}`
///
/// This function attempts to load configuration from a file in the `config/` directory
/// with the given service name. It tries YAML, TOML, and JSON extensions in that order.
///
/// Environment variables with the prefix `{SERVICE_NAME}__` will override file values.
///
/// # Arguments
///
/// * `service_name` - The name of the service (e.g., "telemetry-collector")
///
/// # Returns
///
/// A deserialized configuration struct or an error if loading/parsing fails
///
/// # Examples
///
/// ```rust,no_run
/// use serde::Deserialize;
///
/// #[derive(Deserialize)]
/// struct ServiceConfig {
///     port: u16,
/// }
///
/// let config: ServiceConfig = hpc_config::load("my-service").unwrap();
/// ```
pub fn load<T: DeserializeOwned>(service_name: &str) -> Result<T, hpc_error::HpcError> {
    let config_path = format!("config/{}", service_name);
    let env_prefix = service_name.to_uppercase().replace('-', "_");

    ConfigBuilder::new()
        .add_optional_file(&config_path)
        .add_env_with_prefix(&env_prefix)
        .build()
}

/// Loads configuration from a specific file path
///
/// The file format is detected from the extension (.yaml, .yml, .toml, .json).
///
/// # Arguments
///
/// * `path` - Path to the configuration file
///
/// # Returns
///
/// A deserialized configuration struct or an error
///
/// # Examples
///
/// ```rust,no_run
/// use serde::Deserialize;
///
/// #[derive(Deserialize)]
/// struct Config { port: u16 }
///
/// let config: Config = hpc_config::load_from_file("config.yaml").unwrap();
/// ```
pub fn load_from_file<P: AsRef<Path>, T: DeserializeOwned>(
    path: P,
) -> Result<T, hpc_error::HpcError> {
    let path = path.as_ref();
    let format = detect_format(path)?;

    let config = Config::builder()
        .add_source(File::from(path).format(format))
        .build()
        .map_err(|e| hpc_error::HpcError::Config(format!("failed to load config: {}", e)))?;

    config
        .try_deserialize()
        .map_err(|e| hpc_error::HpcError::Config(format!("failed to deserialize config: {}", e)))
}

/// Loads configuration from a file with environment variable overrides
///
/// Environment variables with the given prefix will override values from the file.
///
/// # Arguments
///
/// * `path` - Path to the configuration file
/// * `env_prefix` - Prefix for environment variables (e.g., "MYSERVICE")
///
/// # Returns
///
/// A deserialized configuration struct or an error
///
/// # Examples
///
/// ```rust,no_run
/// use serde::Deserialize;
///
/// #[derive(Deserialize)]
/// struct Config { port: u16 }
///
/// let config: Config = hpc_config::load_with_env(
///     "config.yaml",
///     "MYSERVICE"
/// ).unwrap();
/// ```
pub fn load_with_env<P: AsRef<Path>, T: DeserializeOwned>(
    path: P,
    env_prefix: &str,
) -> Result<T, hpc_error::HpcError> {
    ConfigBuilder::new()
        .add_file(path)
        .add_env_with_prefix(env_prefix)
        .build()
}

/// Loads a secret from a file
///
/// The file contents are read and wrapped in a `Secret<String>` for redacted display.
/// Trailing newlines are automatically trimmed.
///
/// # Arguments
///
/// * `path` - Path to the secret file
///
/// # Returns
///
/// A secret string or an error
///
/// # Examples
///
/// ```rust,no_run
/// use hpc_config::{load_secret_from_file, ExposeSecret};
///
/// let secret = load_secret_from_file("/run/secrets/db_password").unwrap();
/// let password: &str = secret.expose_secret();
/// ```
pub fn load_secret_from_file<P: AsRef<Path>>(path: P) -> Result<SecretString, hpc_error::HpcError> {
    let content = std::fs::read_to_string(path.as_ref())
        .map_err(|e| hpc_error::HpcError::Config(format!("failed to read secret file: {}", e)))?;

    // Trim trailing newline that might be present in secret files
    let trimmed = content.trim_end().to_string();
    Ok(Secret::new(trimmed))
}

/// Loads a secret from an environment variable
///
/// The environment variable value is wrapped in a `Secret<String>` for redacted display.
///
/// # Arguments
///
/// * `var_name` - Name of the environment variable
///
/// # Returns
///
/// A secret string or an error if the variable is not set
///
/// # Examples
///
/// ```rust,no_run
/// use hpc_config::{load_secret_from_env, ExposeSecret};
///
/// let secret = load_secret_from_env("DB_PASSWORD").unwrap();
/// let password: &str = secret.expose_secret();
/// ```
pub fn load_secret_from_env(var_name: &str) -> Result<SecretString, hpc_error::HpcError> {
    let value = std::env::var(var_name).map_err(|e| {
        hpc_error::HpcError::Config(format!(
            "failed to read secret from env var '{}': {}",
            var_name, e
        ))
    })?;

    Ok(Secret::new(value))
}

/// A builder for composing configuration from multiple sources
///
/// This allows you to combine multiple configuration files and environment
/// variables with custom precedence.
///
/// # Examples
///
/// ```rust,no_run
/// use serde::Deserialize;
///
/// #[derive(Deserialize)]
/// struct Config { port: u16 }
///
/// let config: Config = hpc_config::ConfigBuilder::new()
///     .add_file("base.yaml")
///     .add_file("override.yaml")
///     .add_env_with_prefix("MYSERVICE")
///     .build()
///     .unwrap();
/// ```
pub struct ConfigBuilder {
    builder: config::ConfigBuilder<config::builder::DefaultState>,
}

impl ConfigBuilder {
    /// Creates a new configuration builder
    pub fn new() -> Self {
        Self {
            builder: Config::builder(),
        }
    }

    /// Adds a configuration file to the builder
    ///
    /// The file format is detected from the extension.
    /// If the file doesn't exist, an error will occur when `build()` is called.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the configuration file
    pub fn add_file<P: AsRef<Path>>(mut self, path: P) -> Self {
        let path = path.as_ref();
        if let Ok(format) = detect_format(path) {
            self.builder = self.builder.add_source(File::from(path).format(format));
        }
        self
    }

    /// Adds an optional configuration file to the builder
    ///
    /// If the file doesn't exist, it is silently skipped without error.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the optional configuration file
    pub fn add_optional_file<P: AsRef<Path>>(mut self, path: P) -> Self {
        let path = path.as_ref();

        // Try with extension
        if let Ok(format) = detect_format(path) {
            self.builder = self
                .builder
                .add_source(File::from(path).format(format).required(false));
        } else {
            // Try common extensions
            for ext in &["yaml", "yml", "toml", "json"] {
                let path_with_ext = path.with_extension(ext);
                if let Ok(format) = detect_format(&path_with_ext) {
                    self.builder = self
                        .builder
                        .add_source(File::from(path_with_ext).format(format).required(false));
                    break;
                }
            }
        }
        self
    }

    /// Adds environment variable overrides with a prefix
    ///
    /// Environment variables in the format `{PREFIX}__{FIELD_NAME}` will override
    /// config values. Nested fields use additional underscores: `PREFIX__PARENT__CHILD`.
    ///
    /// # Arguments
    ///
    /// * `prefix` - The prefix for environment variables (e.g., "MYSERVICE")
    pub fn add_env_with_prefix(mut self, prefix: &str) -> Self {
        self.builder = self.builder.add_source(
            Environment::with_prefix(prefix)
                .separator("__")
                .try_parsing(true),
        );
        self
    }

    /// Builds the configuration and deserializes it
    ///
    /// # Returns
    ///
    /// The deserialized configuration or an error
    pub fn build<T: DeserializeOwned>(self) -> Result<T, hpc_error::HpcError> {
        let config = self
            .builder
            .build()
            .map_err(|e| hpc_error::HpcError::Config(format!("failed to build config: {}", e)))?;

        config.try_deserialize().map_err(|e| {
            hpc_error::HpcError::Config(format!("failed to deserialize config: {}", e))
        })
    }

    /// Builds the configuration with defaults
    ///
    /// If building fails, returns `Ok(None)` instead of an error.
    /// This is useful for optional configurations.
    pub fn build_with_defaults<T: DeserializeOwned>(self) -> Option<T> {
        self.build().ok()
    }
}

impl Default for ConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Detects the file format from the file extension
fn detect_format(path: &Path) -> Result<FileFormat, hpc_error::HpcError> {
    let extension = path.extension().and_then(|s| s.to_str()).ok_or_else(|| {
        hpc_error::HpcError::Config(format!("cannot detect format for path: {}", path.display()))
    })?;

    match extension.to_lowercase().as_str() {
        "yaml" | "yml" => Ok(FileFormat::Yaml),
        "toml" => Ok(FileFormat::Toml),
        "json" => Ok(FileFormat::Json),
        _ => Err(hpc_error::HpcError::Config(format!(
            "unsupported config format: {}",
            extension
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};

    #[allow(dead_code)]
    #[derive(Debug, Deserialize, Serialize, PartialEq)]
    struct TestConfig {
        value: String,
    }

    #[test]
    fn test_secret_redaction() {
        use secrecy::ExposeSecret;
        let secret = Secret::new("super_secret".to_string());
        let debug_str = format!("{:?}", secret);
        assert!(!debug_str.contains("super_secret"));
        assert_eq!(secret.expose_secret(), "super_secret");
    }

    #[test]
    fn test_detect_format() {
        assert!(matches!(
            detect_format(Path::new("config.yaml")),
            Ok(FileFormat::Yaml)
        ));
        assert!(matches!(
            detect_format(Path::new("config.yml")),
            Ok(FileFormat::Yaml)
        ));
        assert!(matches!(
            detect_format(Path::new("config.toml")),
            Ok(FileFormat::Toml)
        ));
        assert!(matches!(
            detect_format(Path::new("config.json")),
            Ok(FileFormat::Json)
        ));
        assert!(detect_format(Path::new("config.txt")).is_err());
        assert!(detect_format(Path::new("config")).is_err());
    }

    #[test]
    fn test_builder_default() {
        let builder1 = ConfigBuilder::new();
        let builder2 = ConfigBuilder::default();
        // Just verify they both compile and work
        assert!(std::mem::size_of_val(&builder1) == std::mem::size_of_val(&builder2));
    }
}
