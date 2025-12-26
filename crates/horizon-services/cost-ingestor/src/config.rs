use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub server: ServerConfig,
    pub database: DatabaseConfig,
    pub ingestion: IngestionConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseConfig {
    pub url: String,
    pub max_connections: u32,
    pub min_connections: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestionConfig {
    pub batch_size: usize,
    pub max_retries: u32,
}

impl Config {
    pub fn from_env() -> crate::error::Result<Self> {
        Ok(Self {
            server: ServerConfig {
                host: std::env::var("SERVER_HOST").unwrap_or_else(|_| "0.0.0.0".to_string()),
                port: std::env::var("SERVER_PORT")
                    .unwrap_or_else(|_| "8080".to_string())
                    .parse()
                    .map_err(|e| crate::error::HpcError::config(format!("Invalid port: {}", e)))?,
            },
            database: DatabaseConfig {
                url: std::env::var("DATABASE_URL")
                    .map_err(|_| crate::error::HpcError::config("DATABASE_URL not set".to_string()))?,
                max_connections: std::env::var("DB_MAX_CONNECTIONS")
                    .unwrap_or_else(|_| "10".to_string())
                    .parse()
                    .unwrap_or(10),
                min_connections: std::env::var("DB_MIN_CONNECTIONS")
                    .unwrap_or_else(|_| "2".to_string())
                    .parse()
                    .unwrap_or(2),
            },
            ingestion: IngestionConfig {
                batch_size: std::env::var("INGESTION_BATCH_SIZE")
                    .unwrap_or_else(|_| "100".to_string())
                    .parse()
                    .unwrap_or(100),
                max_retries: std::env::var("INGESTION_MAX_RETRIES")
                    .unwrap_or_else(|_| "3".to_string())
                    .parse()
                    .unwrap_or(3),
            },
        })
    }

    pub fn database_url(&self) -> &str {
        &self.database.url
    }

    pub fn server_address(&self) -> String {
        format!("{}:{}", self.server.host, self.server.port)
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            server: ServerConfig {
                host: "0.0.0.0".to_string(),
                port: 8080,
            },
            database: DatabaseConfig {
                url: "postgres://localhost/horizon_cost".to_string(),
                max_connections: 10,
                min_connections: 2,
            },
            ingestion: IngestionConfig {
                batch_size: 100,
                max_retries: 3,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = Config::default();
        assert_eq!(config.server.host, "0.0.0.0");
        assert_eq!(config.server.port, 8080);
        assert_eq!(config.database.max_connections, 10);
        assert_eq!(config.ingestion.batch_size, 100);
    }

    #[test]
    fn test_server_address() {
        let config = Config::default();
        assert_eq!(config.server_address(), "0.0.0.0:8080");
    }

    #[test]
    fn test_database_url() {
        let config = Config::default();
        assert_eq!(config.database_url(), "postgres://localhost/horizon_cost");
    }
}
