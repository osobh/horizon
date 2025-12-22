//! # Service Configuration Example
//!
//! This example demonstrates how to use the horizon-configx crate to load
//! configuration for a Horizon service with multiple sources and secret management.
//!
//! ## Usage
//!
//! ```bash
//! # Load from default config file
//! cargo run --example service_config
//!
//! # Override with environment variables
//! export EXAMPLE__SERVER__PORT=9999
//! export EXAMPLE__DATABASE__PASSWORD="secret123"
//! cargo run --example service_config
//! ```

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// Server configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
struct ServerConfig {
    /// Host address to listen on
    #[serde(default = "default_host")]
    pub host: String,

    /// Port to listen on
    #[serde(default = "default_port")]
    pub port: u16,

    /// Enable TLS
    #[serde(default)]
    pub enable_tls: bool,

    /// Request timeout in milliseconds
    #[serde(default = "default_timeout")]
    pub timeout_ms: u64,
}

fn default_host() -> String {
    "0.0.0.0".to_string()
}

fn default_port() -> u16 {
    8080
}

fn default_timeout() -> u64 {
    5000
}

/// Database configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
struct DatabaseConfig {
    /// Database host
    pub host: String,

    /// Database port
    #[serde(default = "default_db_port")]
    pub port: u16,

    /// Database name
    pub database: String,

    /// Username for authentication
    pub username: String,

    /// Optional password (loaded from secret)
    #[serde(skip)]
    #[allow(dead_code)]
    pub password: Option<String>,
}

fn default_db_port() -> u16 {
    5432
}

/// Observability configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
struct ObservabilityConfig {
    /// Enable metrics export
    #[serde(default = "default_true")]
    pub metrics_enabled: bool,

    /// Prometheus metrics port
    #[serde(default = "default_metrics_port")]
    pub metrics_port: u16,

    /// Enable distributed tracing
    #[serde(default = "default_true")]
    pub tracing_enabled: bool,

    /// OTLP endpoint for traces
    pub otlp_endpoint: Option<String>,

    /// Log level
    #[serde(default = "default_log_level")]
    pub log_level: String,
}

fn default_true() -> bool {
    true
}

fn default_metrics_port() -> u16 {
    9090
}

fn default_log_level() -> String {
    "info".to_string()
}

/// Complete service configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
struct ServiceConfig {
    /// Service name
    pub name: String,

    /// Server configuration
    pub server: ServerConfig,

    /// Database configuration
    pub database: DatabaseConfig,

    /// Observability configuration
    pub observability: ObservabilityConfig,

    /// Optional feature flags
    #[serde(default)]
    pub features: Vec<String>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Horizon ConfigX Example ===\n");

    // Create example configuration file if it doesn't exist
    let config_path = "example-config.yaml";
    if !Path::new(config_path).exists() {
        create_example_config(config_path)?;
        println!("✓ Created example configuration file: {}\n", config_path);
    }

    // Example 1: Load configuration from file only
    println!("1. Loading configuration from file...");
    let config: ServiceConfig = hpc_config::load_from_file(config_path)?;
    println!("   Service: {}", config.name);
    println!("   Server: {}:{}", config.server.host, config.server.port);
    println!(
        "   Database: {}@{}/{}",
        config.database.username, config.database.host, config.database.database
    );
    println!();

    // Example 2: Load with environment variable overrides
    println!("2. Loading configuration with environment overrides...");
    println!("   Set EXAMPLE__SERVER__PORT=9999 to override port");

    let config_with_env: ServiceConfig = hpc_config::load_with_env(config_path, "EXAMPLE")?;
    println!("   Server port: {}", config_with_env.server.port);
    println!();

    // Example 3: Using ConfigBuilder for advanced scenarios
    println!("3. Using ConfigBuilder for flexible composition...");
    let config_builder: ServiceConfig = hpc_config::ConfigBuilder::new()
        .add_file(config_path)
        .add_env_with_prefix("EXAMPLE")
        .build()?;
    println!("   Log level: {}", config_builder.observability.log_level);
    println!("   Features: {:?}", config_builder.features);
    println!();

    // Example 4: Secret management
    println!("4. Secret management demonstration...");

    // Create a temporary secret file
    let secret_file = "db-password.secret";
    fs::write(secret_file, "super_secret_password_123")?;

    let db_password = hpc_config::load_secret_from_file(secret_file)?;
    println!("   ✓ Loaded database password from file");
    println!("   Secret (redacted): {:?}", db_password);

    // Access the actual secret value when needed
    use hpc_config::ExposeSecret;
    let password_value = db_password.expose_secret();
    println!("   Secret length: {} characters", password_value.len());

    // Cleanup
    fs::remove_file(secret_file)?;
    println!();

    // Example 5: Optional configuration files
    println!("5. Optional configuration files...");
    let optional_config: Option<ServiceConfig> = hpc_config::ConfigBuilder::new()
        .add_optional_file("non-existent.yaml")
        .add_file(config_path)
        .build_with_defaults();

    if let Some(cfg) = optional_config {
        println!("   ✓ Configuration loaded successfully");
        println!("   Metrics enabled: {}", cfg.observability.metrics_enabled);
    }
    println!();

    // Example 6: Multiple file sources with precedence
    println!("6. Layered configuration from multiple sources...");

    // Create override file
    let override_file = "override.yaml";
    fs::write(
        override_file,
        r#"
name: "example-service-override"
server:
  port: 7777
observability:
  log_level: "debug"
"#,
    )?;

    let layered_config: ServiceConfig = hpc_config::ConfigBuilder::new()
        .add_file(config_path) // Base config
        .add_file(override_file) // Override config
        .add_env_with_prefix("EXAMPLE") // Env overrides both
        .build()?;

    println!("   Service name: {} (overridden)", layered_config.name);
    println!(
        "   Server port: {} (overridden)",
        layered_config.server.port
    );
    println!(
        "   Log level: {} (overridden)",
        layered_config.observability.log_level
    );

    // Cleanup
    fs::remove_file(override_file)?;
    println!();

    println!("=== Example completed successfully! ===");
    println!("\nTry running with environment variables:");
    println!("  export EXAMPLE__SERVER__PORT=9999");
    println!("  export EXAMPLE__OBSERVABILITY__LOG_LEVEL=debug");
    println!("  cargo run --example service_config");

    Ok(())
}

fn create_example_config(path: &str) -> std::io::Result<()> {
    let config = r#"# Example Service Configuration
# This configuration demonstrates all features of horizon-configx

name: "example-service"

server:
  host: "0.0.0.0"
  port: 8080
  enable_tls: false
  timeout_ms: 5000

database:
  host: "localhost"
  port: 5432
  database: "horizon"
  username: "horizon_user"
  # Password loaded from secret file or env var

observability:
  metrics_enabled: true
  metrics_port: 9090
  tracing_enabled: true
  otlp_endpoint: "http://localhost:4317"
  log_level: "info"

features:
  - "rate_limiting"
  - "caching"
  - "health_checks"
"#;

    fs::write(path, config)
}
