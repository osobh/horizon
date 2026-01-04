use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub server: ServerConfig,
    pub database: DatabaseConfig,
    pub scheduler: SchedulerConfig,
    pub attribution: AttributionConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    #[serde(default = "default_host")]
    pub host: String,
    #[serde(default = "default_port")]
    pub port: u16,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseConfig {
    pub url: String,
    #[serde(default = "default_max_connections")]
    pub max_connections: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerConfig {
    pub url: String,
    #[serde(default = "default_timeout_seconds")]
    pub timeout_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttributionConfig {
    #[serde(default = "default_accuracy_threshold")]
    pub accuracy_threshold_percent: f64,
    #[serde(default = "default_network_rate_per_gb")]
    pub network_rate_per_gb: String, // Decimal as string
    #[serde(default = "default_storage_rate_per_gb_hour")]
    pub storage_rate_per_gb_hour: String, // Decimal as string
}

fn default_host() -> String {
    "0.0.0.0".to_string()
}

fn default_port() -> u16 {
    8083
}

fn default_max_connections() -> u32 {
    10
}

fn default_timeout_seconds() -> u64 {
    30
}

fn default_accuracy_threshold() -> f64 {
    5.0
}

fn default_network_rate_per_gb() -> String {
    "0.09".to_string()
}

fn default_storage_rate_per_gb_hour() -> String {
    "0.00001368".to_string() // ~$0.01/GB/month
}

impl Default for Config {
    fn default() -> Self {
        Self {
            server: ServerConfig {
                host: default_host(),
                port: default_port(),
            },
            database: DatabaseConfig {
                url: "postgres://localhost/horizon".to_string(),
                max_connections: default_max_connections(),
            },
            scheduler: SchedulerConfig {
                url: "http://localhost:8080".to_string(),
                timeout_seconds: default_timeout_seconds(),
            },
            attribution: AttributionConfig {
                accuracy_threshold_percent: default_accuracy_threshold(),
                network_rate_per_gb: default_network_rate_per_gb(),
                storage_rate_per_gb_hour: default_storage_rate_per_gb_hour(),
            },
        }
    }
}

impl Config {
    pub fn from_env() -> Result<Self, hpc_error::HpcError> {
        hpc_config::ConfigBuilder::new()
            .add_optional_file("config/cost-attributor")
            .add_env_with_prefix("COST_ATTRIBUTOR")
            .build()
    }

    pub fn validate(&self) -> crate::error::Result<()> {
        if self.database.url.is_empty() {
            return Err(hpc_error::HpcError::config("Database URL cannot be empty"));
        }

        if self.scheduler.url.is_empty() {
            return Err(hpc_error::HpcError::config("Scheduler URL cannot be empty"));
        }

        if self.attribution.accuracy_threshold_percent <= 0.0
            || self.attribution.accuracy_threshold_percent > 100.0
        {
            return Err(hpc_error::HpcError::config(
                "Accuracy threshold must be between 0 and 100",
            ));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.server.host, "0.0.0.0");
        assert_eq!(config.server.port, 8083);
        assert_eq!(config.database.max_connections, 10);
        assert_eq!(config.attribution.accuracy_threshold_percent, 5.0);
    }

    #[test]
    fn test_config_validation_success() {
        let config = Config::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validation_empty_database_url() {
        let mut config = Config::default();
        config.database.url = String::new();
        let result = config.validate();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Database URL cannot be empty"));
    }

    #[test]
    fn test_config_validation_empty_scheduler_url() {
        let mut config = Config::default();
        config.scheduler.url = String::new();
        let result = config.validate();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Scheduler URL cannot be empty"));
    }

    #[test]
    fn test_config_validation_invalid_accuracy_threshold() {
        let mut config = Config::default();
        config.attribution.accuracy_threshold_percent = 0.0;
        let result = config.validate();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Accuracy threshold must be between 0 and 100"));
    }

    #[test]
    fn test_config_validation_accuracy_threshold_too_high() {
        let mut config = Config::default();
        config.attribution.accuracy_threshold_percent = 150.0;
        let result = config.validate();
        assert!(result.is_err());
    }

    #[test]
    fn test_config_serialization() {
        let config = Config::default();
        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("server"));
        assert!(json.contains("database"));
        assert!(json.contains("scheduler"));
    }

    #[test]
    fn test_config_deserialization() {
        let json = r#"{
            "server": {"host": "127.0.0.1", "port": 9000},
            "database": {"url": "postgres://test", "max_connections": 20},
            "scheduler": {"url": "http://scheduler", "timeout_seconds": 60},
            "attribution": {
                "accuracy_threshold_percent": 10.0,
                "network_rate_per_gb": "0.10",
                "storage_rate_per_gb_hour": "0.00002"
            }
        }"#;

        let config: Config = serde_json::from_str(json).unwrap();
        assert_eq!(config.server.host, "127.0.0.1");
        assert_eq!(config.server.port, 9000);
        assert_eq!(config.database.max_connections, 20);
        assert_eq!(config.attribution.accuracy_threshold_percent, 10.0);
    }
}
