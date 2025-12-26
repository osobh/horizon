use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub listen_addr: String,
    pub database_url: String,
    pub log_level: String,
    pub otlp_endpoint: Option<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            listen_addr: std::env::var("LISTEN_ADDR")
                .unwrap_or_else(|_| "0.0.0.0:8071".to_string()),
            database_url: std::env::var("DATABASE_URL")
                .unwrap_or_else(|_| "postgres://postgres:postgres@localhost/inventory".to_string()),
            log_level: std::env::var("LOG_LEVEL").unwrap_or_else(|_| "info".to_string()),
            otlp_endpoint: std::env::var("OTLP_ENDPOINT").ok(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = Config::default();
        assert_eq!(config.listen_addr, "0.0.0.0:8080");
        assert!(!config.database_url.is_empty());
    }
}
