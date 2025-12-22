use rust_decimal::Decimal;
use serde::Deserialize;
use std::net::SocketAddr;

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    #[serde(default = "default_http_addr")]
    pub http_addr: SocketAddr,

    pub database: DatabaseConfig,

    #[serde(default)]
    pub margin: MarginConfig,

    #[serde(default)]
    pub cost_attributor: CostAttributorConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DatabaseConfig {
    pub url: String,
    #[serde(default = "default_max_connections")]
    pub max_connections: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MarginConfig {
    #[serde(default = "default_at_risk_threshold")]
    pub at_risk_threshold: Decimal,

    #[serde(default = "default_high_confidence_threshold")]
    pub high_confidence_threshold: Decimal,

    #[serde(default = "default_ltv_months")]
    pub ltv_months: i32,
}

impl Default for MarginConfig {
    fn default() -> Self {
        Self {
            at_risk_threshold: default_at_risk_threshold(),
            high_confidence_threshold: default_high_confidence_threshold(),
            ltv_months: default_ltv_months(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct CostAttributorConfig {
    #[serde(default = "default_cost_attributor_url")]
    pub url: String,
    #[serde(default = "default_timeout_seconds")]
    pub timeout_seconds: u64,
}

impl Default for CostAttributorConfig {
    fn default() -> Self {
        Self {
            url: default_cost_attributor_url(),
            timeout_seconds: default_timeout_seconds(),
        }
    }
}

impl Config {
    pub fn from_env() -> Result<Self, hpc_error::HpcError> {
        hpc_config::ConfigBuilder::new()
            .add_optional_file("config/margin-intelligence")
            .add_env_with_prefix("MARGIN_INTELLIGENCE")
            .build()
    }
}

fn default_http_addr() -> SocketAddr {
    std::env::var("LISTEN_ADDR")
        .unwrap_or_else(|_| "0.0.0.0:8088".to_string())
        .parse()
        .expect("Invalid LISTEN_ADDR format")
}

fn default_max_connections() -> u32 {
    10
}

fn default_at_risk_threshold() -> Decimal {
    Decimal::from(10) // 10% margin threshold
}

fn default_high_confidence_threshold() -> Decimal {
    Decimal::from_str_exact("0.75").unwrap() // 75% confidence
}

fn default_ltv_months() -> i32 {
    12 // 12 months for LTV calculation
}

fn default_cost_attributor_url() -> String {
    "http://localhost:8082".to_string()
}

fn default_timeout_seconds() -> u64 {
    30
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_values() {
        let config = MarginConfig::default();
        assert_eq!(config.at_risk_threshold, Decimal::from(10));
        assert_eq!(config.high_confidence_threshold, Decimal::from_str_exact("0.75").unwrap());
        assert_eq!(config.ltv_months, 12);
    }

    #[test]
    fn test_default_cost_attributor_config() {
        let config = CostAttributorConfig::default();
        assert_eq!(config.url, "http://localhost:8082");
        assert_eq!(config.timeout_seconds, 30);
    }

    #[test]
    fn test_default_http_addr() {
        let addr = default_http_addr();
        assert_eq!(addr.to_string(), "0.0.0.0:8090");
    }

    #[test]
    fn test_default_max_connections() {
        assert_eq!(default_max_connections(), 10);
    }
}
