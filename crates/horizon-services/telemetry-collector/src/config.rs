use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectorConfig {
    pub server: ServerConfig,
    pub security: SecurityConfig,
    pub influxdb: InfluxDbConfig,
    pub parquet: ParquetConfig,
    pub limits: LimitsConfig,
    pub observability: ObservabilityConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub listen_addr: String,
    pub max_connections: u32,
    pub connection_timeout_secs: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    pub tls_cert_path: String,
    pub tls_key_path: String,
    pub tls_ca_path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InfluxDbConfig {
    pub url: String,
    pub org: String,
    pub bucket: String,
    pub token: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParquetConfig {
    pub output_dir: String,
    pub rotation_interval_secs: u64,
    pub compression: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LimitsConfig {
    pub max_cardinality: u64,
    pub max_batch_size: usize,
    pub backpressure_threshold: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservabilityConfig {
    pub metrics_port: u16,
    pub log_level: String,
}

impl CollectorConfig {
    /// Load configuration from a YAML file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path.as_ref())
            .context("Failed to read configuration file")?;

        // Substitute environment variables
        let content = Self::substitute_env_vars(&content)?;

        let config: Self = serde_yaml::from_str(&content)
            .context("Failed to parse configuration YAML")?;

        config.validate()?;
        Ok(config)
    }

    /// Substitute environment variables in format ${VAR_NAME}
    fn substitute_env_vars(content: &str) -> Result<String> {
        let mut result = content.to_string();

        // Find all ${VAR_NAME} patterns
        let re = regex::Regex::new(r"\$\{([^}]+)\}").unwrap();

        for cap in re.captures_iter(content) {
            let full_match = &cap[0];
            let var_name = &cap[1];

            let var_value = std::env::var(var_name)
                .with_context(|| format!("Environment variable '{}' not found", var_name))?;

            result = result.replace(full_match, &var_value);
        }

        Ok(result)
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<()> {
        // Validate server configuration
        self.validate_listen_addr()?;

        if self.server.max_connections == 0 {
            return Err(anyhow!("max_connections must be greater than 0"));
        }

        if self.server.connection_timeout_secs == 0 {
            return Err(anyhow!("connection_timeout_secs must be greater than 0"));
        }

        // Validate security paths (just check they're not empty)
        // Skip TLS validation if paths are default placeholder values
        if !self.security.tls_cert_path.starts_with("/etc/horizon/certs/") {
            if self.security.tls_cert_path.is_empty() {
                return Err(anyhow!("tls_cert_path cannot be empty"));
            }
        }
        if !self.security.tls_key_path.starts_with("/etc/horizon/certs/") {
            if self.security.tls_key_path.is_empty() {
                return Err(anyhow!("tls_key_path cannot be empty"));
            }
        }
        if !self.security.tls_ca_path.starts_with("/etc/horizon/certs/") {
            if self.security.tls_ca_path.is_empty() {
                return Err(anyhow!("tls_ca_path cannot be empty"));
            }
        }

        // Validate InfluxDB configuration
        if self.influxdb.url.is_empty() {
            return Err(anyhow!("influxdb.url cannot be empty"));
        }
        if self.influxdb.org.is_empty() {
            return Err(anyhow!("influxdb.org cannot be empty"));
        }
        if self.influxdb.bucket.is_empty() {
            return Err(anyhow!("influxdb.bucket cannot be empty"));
        }
        // Token can be empty for development/testing
        // if self.influxdb.token.is_empty() {
        //     return Err(anyhow!("influxdb.token cannot be empty"));
        // }

        // Validate Parquet configuration
        if self.parquet.output_dir.is_empty() {
            return Err(anyhow!("parquet.output_dir cannot be empty"));
        }
        if self.parquet.rotation_interval_secs == 0 {
            return Err(anyhow!("parquet.rotation_interval_secs must be greater than 0"));
        }

        // Validate compression type
        match self.parquet.compression.as_str() {
            "snappy" | "gzip" | "lz4" | "zstd" | "none" => {}
            _ => return Err(anyhow!("Invalid compression type: {}", self.parquet.compression)),
        }

        // Validate limits
        if self.limits.max_cardinality == 0 {
            return Err(anyhow!("max_cardinality must be greater than 0"));
        }
        if self.limits.max_batch_size == 0 {
            return Err(anyhow!("max_batch_size must be greater than 0"));
        }
        if self.limits.backpressure_threshold == 0 {
            return Err(anyhow!("backpressure_threshold must be greater than 0"));
        }

        // Validate observability
        if self.observability.metrics_port == 0 {
            return Err(anyhow!("metrics_port must be greater than 0"));
        }

        Ok(())
    }

    fn validate_listen_addr(&self) -> Result<()> {
        // Parse the address to ensure it's valid
        let parts: Vec<&str> = self.server.listen_addr.split(':').collect();
        if parts.len() != 2 {
            return Err(anyhow!("Invalid listen_addr format, expected 'host:port'"));
        }

        let port: u16 = parts[1]
            .parse()
            .map_err(|_| anyhow!("Invalid port number in listen_addr"))?;

        if port == 0 {
            return Err(anyhow!("Port cannot be 0"));
        }

        Ok(())
    }
}

impl Default for CollectorConfig {
    fn default() -> Self {
        Self {
            server: ServerConfig {
                listen_addr: "0.0.0.0:5001".to_string(),
                max_connections: 1000,
                connection_timeout_secs: 300,
            },
            security: SecurityConfig {
                tls_cert_path: "/etc/horizon/certs/server.pem".to_string(),
                tls_key_path: "/etc/horizon/certs/server-key.pem".to_string(),
                tls_ca_path: "/etc/horizon/certs/ca.pem".to_string(),
            },
            influxdb: InfluxDbConfig {
                url: "http://localhost:8086".to_string(),
                org: "horizon".to_string(),
                bucket: "telemetry".to_string(),
                token: "".to_string(),
            },
            parquet: ParquetConfig {
                output_dir: "/var/lib/horizon/telemetry".to_string(),
                rotation_interval_secs: 3600,
                compression: "snappy".to_string(),
            },
            limits: LimitsConfig {
                max_cardinality: 100000,
                max_batch_size: 1000,
                backpressure_threshold: 5000,
            },
            observability: ObservabilityConfig {
                metrics_port: 9091,
                log_level: "info".to_string(),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_is_valid() {
        let mut config = CollectorConfig::default();
        // Set token to non-empty for validation
        config.influxdb.token = "test-token".to_string();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_validate_listen_addr() {
        let mut config = CollectorConfig::default();
        config.influxdb.token = "test-token".to_string();

        config.server.listen_addr = "0.0.0.0:5001".to_string();
        assert!(config.validate().is_ok());

        config.server.listen_addr = "127.0.0.1:8080".to_string();
        assert!(config.validate().is_ok());

        config.server.listen_addr = "invalid".to_string();
        assert!(config.validate().is_err());

        config.server.listen_addr = "0.0.0.0:99999".to_string();
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_compression_validation() {
        let mut config = CollectorConfig::default();
        config.influxdb.token = "test-token".to_string();

        for compression in &["snappy", "gzip", "lz4", "zstd", "none"] {
            config.parquet.compression = compression.to_string();
            assert!(config.validate().is_ok());
        }

        config.parquet.compression = "invalid".to_string();
        assert!(config.validate().is_err());
    }
}
