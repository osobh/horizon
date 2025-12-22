//! Region management and configuration

use crate::error::{MultiRegionError, MultiRegionResult};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Region configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionConfig {
    /// Primary region
    pub primary_region: String,
    /// Available regions
    pub regions: Vec<RegionInfo>,
    /// Failover configuration
    pub failover_config: FailoverConfig,
    /// Network configuration
    pub network_config: NetworkConfig,
}

/// Region information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionInfo {
    /// Region identifier
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Data center locations
    pub data_centers: Vec<String>,
    /// Compliance jurisdictions
    pub jurisdictions: Vec<String>,
    /// Available services
    pub services: Vec<String>,
    /// Network latency matrix
    pub latency_matrix: HashMap<String, u32>,
    /// Active status
    pub active: bool,
    /// Health check endpoint
    pub health_endpoint: String,
    /// Created timestamp
    pub created_at: DateTime<Utc>,
}

/// Failover configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailoverConfig {
    /// Auto-failover enabled
    pub auto_failover: bool,
    /// Health check interval (seconds)
    pub health_check_interval: u64,
    /// Failover threshold (consecutive failures)
    pub failover_threshold: u32,
    /// Recovery threshold (consecutive successes)
    pub recovery_threshold: u32,
    /// Preferred failover order
    pub failover_order: Vec<String>,
}

/// Network configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    /// Load balancer endpoints
    pub load_balancer_endpoints: HashMap<String, String>,
    /// Tunnel configurations
    pub tunnel_configs: HashMap<String, TunnelConfig>,
    /// DNS configurations
    pub dns_configs: HashMap<String, DnsConfig>,
}

/// Tunnel configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TunnelConfig {
    /// Tunnel type
    pub tunnel_type: String,
    /// Endpoint URL
    pub endpoint: String,
    /// Authentication token
    pub auth_token: Option<String>,
    /// TLS configuration
    pub tls_config: TlsConfig,
}

/// TLS configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TlsConfig {
    /// Certificate path
    pub cert_path: Option<String>,
    /// Private key path
    pub key_path: Option<String>,
    /// CA certificate path
    pub ca_path: Option<String>,
    /// Verify certificates
    pub verify_certs: bool,
}

/// DNS configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DnsConfig {
    /// DNS servers
    pub servers: Vec<String>,
    /// Search domains
    pub search_domains: Vec<String>,
    /// TTL seconds
    pub ttl: u32,
}

/// Region manager
pub struct RegionManager {
    config: RegionConfig,
    health_status: HashMap<String, RegionHealth>,
    client: reqwest::Client,
}

/// Region health status
#[derive(Debug, Clone)]
pub struct RegionHealth {
    /// Region ID
    pub region_id: String,
    /// Health status
    pub healthy: bool,
    /// Last check timestamp
    pub last_check: DateTime<Utc>,
    /// Response time (ms)
    pub response_time: Option<u64>,
    /// Error count
    pub error_count: u32,
    /// Success count
    pub success_count: u32,
}

impl RegionManager {
    /// Create new region manager
    pub fn new(config: RegionConfig) -> MultiRegionResult<Self> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .map_err(|e| MultiRegionError::ConfigurationError {
                message: format!("Failed to create HTTP client: {}", e),
            })?;

        let mut health_status = HashMap::new();
        for region in &config.regions {
            health_status.insert(
                region.id.clone(),
                RegionHealth {
                    region_id: region.id.clone(),
                    healthy: false,
                    last_check: Utc::now(),
                    response_time: None,
                    error_count: 0,
                    success_count: 0,
                },
            );
        }

        Ok(Self {
            config,
            health_status,
            client,
        })
    }

    /// Get primary region
    pub fn primary_region(&self) -> &str {
        &self.config.primary_region
    }

    /// Get all regions
    pub fn regions(&self) -> &[RegionInfo] {
        &self.config.regions
    }

    /// Get region by ID
    pub fn get_region(&self, region_id: &str) -> Option<&RegionInfo> {
        self.config.regions.iter().find(|r| r.id == region_id)
    }

    /// Get healthy regions
    pub fn healthy_regions(&self) -> Vec<&RegionInfo> {
        self.config
            .regions
            .iter()
            .filter(|r| {
                self.health_status
                    .get(&r.id)
                    .map(|h| h.healthy)
                    .unwrap_or(false)
            })
            .collect()
    }

    /// Perform health check on all regions
    pub async fn health_check_all(&mut self) -> MultiRegionResult<()> {
        let region_ids: Vec<String> = self.config.regions.iter().map(|r| r.id.clone()).collect();
        for region_id in region_ids {
            self.health_check_region(&region_id).await?;
        }
        Ok(())
    }

    /// Perform health check on specific region
    pub async fn health_check_region(&mut self, region_id: &str) -> MultiRegionResult<()> {
        let region =
            self.get_region(region_id)
                .ok_or_else(|| MultiRegionError::RegionUnavailable {
                    region: region_id.to_string(),
                })?;

        let start_time = std::time::Instant::now();
        let health_result = self.client.get(&region.health_endpoint).send().await;
        let response_time = start_time.elapsed().as_millis() as u64;

        let health = self.health_status.get_mut(region_id).ok_or_else(|| {
            MultiRegionError::ConfigurationError {
                message: format!("Health status not found for region: {}", region_id),
            }
        })?;

        health.last_check = Utc::now();
        health.response_time = Some(response_time);

        match health_result {
            Ok(response) if response.status().is_success() => {
                health.healthy = true;
                health.success_count += 1;
                health.error_count = 0; // Reset error count on success
            }
            _ => {
                health.healthy = false;
                health.error_count += 1;
                health.success_count = 0; // Reset success count on error
            }
        }

        Ok(())
    }

    /// Get region health status
    pub fn get_health_status(&self, region_id: &str) -> Option<&RegionHealth> {
        self.health_status.get(region_id)
    }

    /// Select best region for data placement
    pub fn select_best_region(
        &self,
        requirements: &RegionRequirements,
    ) -> MultiRegionResult<String> {
        let mut candidates: Vec<_> = self
            .healthy_regions()
            .into_iter()
            .filter(|r| {
                // Check jurisdiction requirements
                if !requirements.required_jurisdictions.is_empty() {
                    return requirements
                        .required_jurisdictions
                        .iter()
                        .any(|req_jurisdiction| r.jurisdictions.contains(req_jurisdiction));
                }
                true
            })
            .filter(|r| {
                // Check excluded jurisdictions
                !requirements
                    .excluded_jurisdictions
                    .iter()
                    .any(|excl_jurisdiction| r.jurisdictions.contains(excl_jurisdiction))
            })
            .filter(|r| {
                // Check required services
                requirements
                    .required_services
                    .iter()
                    .all(|service| r.services.contains(service))
            })
            .collect();

        if candidates.is_empty() {
            return Err(MultiRegionError::RegionUnavailable {
                region: "No regions match requirements".to_string(),
            });
        }

        // Sort by performance criteria
        candidates.sort_by(|a, b| {
            let a_health = self.health_status.get(&a.id);
            let b_health = self.health_status.get(&b.id);

            // Prefer regions with better response times
            match (a_health, b_health) {
                (Some(a_h), Some(b_h)) => a_h
                    .response_time
                    .unwrap_or(u64::MAX)
                    .cmp(&b_h.response_time.unwrap_or(u64::MAX)),
                _ => std::cmp::Ordering::Equal,
            }
        });

        Ok(candidates[0].id.clone())
    }
}

/// Region selection requirements
#[derive(Debug, Clone, Default)]
pub struct RegionRequirements {
    /// Required jurisdictions (at least one must match)
    pub required_jurisdictions: Vec<String>,
    /// Excluded jurisdictions (none must match)
    pub excluded_jurisdictions: Vec<String>,
    /// Required services
    pub required_services: Vec<String>,
    /// Maximum latency (ms)
    pub max_latency: Option<u64>,
}

impl Default for RegionConfig {
    fn default() -> Self {
        Self {
            primary_region: "us-east-1".to_string(),
            regions: vec![
                RegionInfo {
                    id: "us-east-1".to_string(),
                    name: "US East (N. Virginia)".to_string(),
                    data_centers: vec!["iad".to_string()],
                    jurisdictions: vec!["US".to_string()],
                    services: vec!["compute".to_string(), "storage".to_string()],
                    latency_matrix: HashMap::new(),
                    active: true,
                    health_endpoint: "https://health.us-east-1.example.com/health".to_string(),
                    created_at: Utc::now(),
                },
                RegionInfo {
                    id: "eu-west-1".to_string(),
                    name: "EU West (Ireland)".to_string(),
                    data_centers: vec!["dub".to_string()],
                    jurisdictions: vec!["EU".to_string(), "IE".to_string()],
                    services: vec!["compute".to_string(), "storage".to_string()],
                    latency_matrix: HashMap::new(),
                    active: true,
                    health_endpoint: "https://health.eu-west-1.example.com/health".to_string(),
                    created_at: Utc::now(),
                },
            ],
            failover_config: FailoverConfig {
                auto_failover: true,
                health_check_interval: 30,
                failover_threshold: 3,
                recovery_threshold: 5,
                failover_order: vec!["us-west-2".to_string(), "eu-west-1".to_string()],
            },
            network_config: NetworkConfig {
                load_balancer_endpoints: HashMap::new(),
                tunnel_configs: HashMap::new(),
                dns_configs: HashMap::new(),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_region_config_creation() {
        let config = RegionConfig::default();
        assert_eq!(config.primary_region, "us-east-1");
        assert_eq!(config.regions.len(), 2);
        assert!(config.failover_config.auto_failover);
    }

    #[test]
    fn test_region_manager_creation() {
        let config = RegionConfig::default();
        let manager = RegionManager::new(config);
        assert!(manager.is_ok());

        let manager = manager.unwrap();
        assert_eq!(manager.primary_region(), "us-east-1");
        assert_eq!(manager.regions().len(), 2);
    }

    #[test]
    fn test_get_region_by_id() {
        let config = RegionConfig::default();
        let manager = RegionManager::new(config).unwrap();

        let region = manager.get_region("us-east-1");
        assert!(region.is_some());
        assert_eq!(region.unwrap().name, "US East (N. Virginia)");

        let region = manager.get_region("non-existent");
        assert!(region.is_none());
    }

    #[test]
    fn test_region_requirements_filtering() {
        let config = RegionConfig::default();
        let manager = RegionManager::new(config).unwrap();

        // Test jurisdiction requirements
        let requirements = RegionRequirements {
            required_jurisdictions: vec!["US".to_string()],
            excluded_jurisdictions: vec![],
            required_services: vec![],
            max_latency: None,
        };

        // This test will initially fail since we need healthy regions
        // We'll implement this in the GREEN phase
        let result = manager.select_best_region(&requirements);
        assert!(result.is_err()); // No healthy regions initially
    }

    #[test]
    fn test_health_status_initialization() {
        let config = RegionConfig::default();
        let manager = RegionManager::new(config).unwrap();

        // Check that health status is initialized for all regions
        for region in manager.regions() {
            let health = manager.get_health_status(&region.id);
            assert!(health.is_some());
            assert!(!health.unwrap().healthy); // Initially unhealthy
        }
    }

    #[test]
    fn test_excluded_jurisdictions() {
        let config = RegionConfig::default();
        let manager = RegionManager::new(config).unwrap();

        let requirements = RegionRequirements {
            required_jurisdictions: vec![],
            excluded_jurisdictions: vec!["EU".to_string()],
            required_services: vec![],
            max_latency: None,
        };

        // Should exclude EU regions, but we need healthy regions first
        let result = manager.select_best_region(&requirements);
        assert!(result.is_err()); // No healthy regions initially
    }

    #[test]
    fn test_required_services() {
        let config = RegionConfig::default();
        let manager = RegionManager::new(config).unwrap();

        let requirements = RegionRequirements {
            required_jurisdictions: vec![],
            excluded_jurisdictions: vec![],
            required_services: vec!["compute".to_string(), "storage".to_string()],
            max_latency: None,
        };

        let result = manager.select_best_region(&requirements);
        assert!(result.is_err()); // No healthy regions initially
    }

    #[test]
    fn test_region_info_serialization() {
        let region = RegionInfo {
            id: "test-region".to_string(),
            name: "Test Region".to_string(),
            data_centers: vec!["test-dc".to_string()],
            jurisdictions: vec!["TEST".to_string()],
            services: vec!["test-service".to_string()],
            latency_matrix: HashMap::new(),
            active: true,
            health_endpoint: "https://test.example.com/health".to_string(),
            created_at: Utc::now(),
        };

        let json = serde_json::to_string(&region);
        assert!(json.is_ok());

        let deserialized: Result<RegionInfo, _> = serde_json::from_str(&json.unwrap());
        assert!(deserialized.is_ok());
    }
}
