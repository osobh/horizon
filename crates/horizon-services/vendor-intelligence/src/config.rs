use hpc_config::ConfigBuilder;
use serde::Deserialize;
use std::net::SocketAddr;

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    #[serde(default = "default_http_addr")]
    pub http_addr: SocketAddr,
    pub database: DatabaseConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DatabaseConfig {
    pub url: String,
    #[serde(default = "default_max_connections")]
    pub max_connections: u32,
}

impl Config {
    pub fn from_env() -> Result<Self, hpc_error::HpcError> {
        ConfigBuilder::new()
            .add_optional_file("config/vendor-intelligence")
            .add_env_with_prefix("VENDOR_INTELLIGENCE")
            .build()
    }
}

fn default_http_addr() -> SocketAddr {
    std::env::var("LISTEN_ADDR")
        .unwrap_or_else(|_| "0.0.0.0:8091".to_string())
        .parse()
        .expect("Invalid LISTEN_ADDR format")
}

fn default_max_connections() -> u32 {
    10
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_http_addr() {
        let addr = default_http_addr();
        assert_eq!(addr.to_string(), "0.0.0.0:8093");
    }

    #[test]
    fn test_default_max_connections() {
        assert_eq!(default_max_connections(), 10);
    }
}
