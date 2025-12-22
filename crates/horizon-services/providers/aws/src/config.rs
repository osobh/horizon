use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AwsConfig {
    pub region: String,
    pub access_key_id: Option<String>,
    pub secret_access_key: Option<String>,
    pub endpoint: Option<String>,
}

impl AwsConfig {
    pub fn new(region: String) -> Self {
        Self {
            region,
            access_key_id: None,
            secret_access_key: None,
            endpoint: None,
        }
    }

    pub fn with_credentials(
        region: String,
        access_key_id: String,
        secret_access_key: String,
    ) -> Self {
        Self {
            region,
            access_key_id: Some(access_key_id),
            secret_access_key: Some(secret_access_key),
            endpoint: None,
        }
    }

    pub fn with_endpoint(mut self, endpoint: String) -> Self {
        self.endpoint = Some(endpoint);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_creation() {
        let config = AwsConfig::new("us-east-1".to_string());
        assert_eq!(config.region, "us-east-1");
        assert!(config.access_key_id.is_none());
        assert!(config.secret_access_key.is_none());
    }

    #[test]
    fn test_config_with_credentials() {
        let config = AwsConfig::with_credentials(
            "us-west-2".to_string(),
            "AKIAIOSFODNN7EXAMPLE".to_string(),
            "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY".to_string(),
        );
        assert_eq!(config.region, "us-west-2");
        assert!(config.access_key_id.is_some());
        assert!(config.secret_access_key.is_some());
    }

    #[test]
    fn test_config_with_endpoint() {
        let config = AwsConfig::new("us-east-1".to_string())
            .with_endpoint("http://localhost:4566".to_string());
        assert_eq!(config.endpoint, Some("http://localhost:4566".to_string()));
    }
}
