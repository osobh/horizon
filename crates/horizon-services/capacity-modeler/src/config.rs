use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub listen_addr: String,
    pub influxdb_url: String,
    pub influxdb_org: String,
    pub influxdb_bucket: String,
    pub influxdb_token: String,
    pub min_historical_days: usize,
    pub default_forecast_weeks: u8,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            listen_addr: std::env::var("LISTEN_ADDR")
                .unwrap_or_else(|_| "0.0.0.0:8081".to_string()),
            influxdb_url: std::env::var("INFLUXDB_URL")
                .unwrap_or_else(|_| "http://localhost:8086".to_string()),
            influxdb_org: std::env::var("INFLUXDB_ORG")
                .unwrap_or_else(|_| "horizon".to_string()),
            influxdb_bucket: std::env::var("INFLUXDB_BUCKET")
                .unwrap_or_else(|_| "metrics".to_string()),
            influxdb_token: std::env::var("INFLUXDB_TOKEN").unwrap_or_else(|_| "".to_string()),
            min_historical_days: std::env::var("MIN_HISTORICAL_DAYS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(180),
            default_forecast_weeks: std::env::var("DEFAULT_FORECAST_WEEKS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(13),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.listen_addr, "0.0.0.0:8081");
        assert_eq!(config.influxdb_url, "http://localhost:8086");
        assert_eq!(config.min_historical_days, 180);
        assert_eq!(config.default_forecast_weeks, 13);
    }
}
