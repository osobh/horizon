use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReporterConfig {
    pub database_url: String,
    pub host: String,
    pub port: u16,
    pub attributor_url: String,
    pub default_forecast_days: usize,
    pub max_export_records: usize,
    pub view_refresh_interval_secs: u64,
}

impl Default for ReporterConfig {
    fn default() -> Self {
        Self {
            database_url: std::env::var("DATABASE_URL")
                .unwrap_or_else(|_| "postgres://localhost/horizon_cost".to_string()),
            host: std::env::var("REPORTER_HOST").unwrap_or_else(|_| "127.0.0.1".to_string()),
            port: std::env::var("REPORTER_PORT")
                .ok()
                .and_then(|p| p.parse().ok())
                .unwrap_or(8085),
            attributor_url: std::env::var("ATTRIBUTOR_URL")
                .unwrap_or_else(|_| "http://localhost:8084".to_string()),
            default_forecast_days: 30,
            max_export_records: 10000,
            view_refresh_interval_secs: 3600, // 1 hour
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ReporterConfig::default();
        assert_eq!(config.port, 8085);
        assert_eq!(config.default_forecast_days, 30);
        assert_eq!(config.max_export_records, 10000);
    }
}
