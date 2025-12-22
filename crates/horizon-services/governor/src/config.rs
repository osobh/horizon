use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub server: ServerConfig,
    pub database: DatabaseConfig,
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

    #[serde(default = "default_min_connections")]
    pub min_connections: u32,
}

fn default_host() -> String {
    "0.0.0.0".to_string()
}

fn default_port() -> u16 {
    8080
}

fn default_max_connections() -> u32 {
    10
}

fn default_min_connections() -> u32 {
    2
}

impl Default for Config {
    fn default() -> Self {
        Self {
            server: ServerConfig {
                host: default_host(),
                port: default_port(),
            },
            database: DatabaseConfig {
                url: "postgresql://localhost/governor".to_string(),
                max_connections: default_max_connections(),
                min_connections: default_min_connections(),
            },
        }
    }
}

impl Config {
    pub fn from_env() -> Result<Self, hpc_error::HpcError> {
        hpc_config::ConfigBuilder::new()
            .add_optional_file("config/governor")
            .add_env_with_prefix("GOVERNOR")
            .build()
    }
}
