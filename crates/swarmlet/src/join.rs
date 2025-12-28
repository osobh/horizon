//! Join protocol for connecting to StratoSwarm clusters

use crate::{
    profile::HardwareProfile,
    security::JoinToken,
    wireguard::{WireGuardConfigRequest, WireGuardManager, WireGuardPeerConfig},
    Result, SwarmletError,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;
use tracing::{debug, info, warn};
use uuid::Uuid;

/// Result of successful cluster join
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JoinResult {
    pub node_id: Uuid,
    pub cluster_name: String,
    pub node_certificate: String,
    pub cluster_endpoints: Vec<String>,
    pub assigned_capabilities: Vec<String>,
    pub heartbeat_interval: Duration,
    pub api_endpoints: ClusterApiEndpoints,
    /// WireGuard configuration received from coordinator
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wireguard_config: Option<WireGuardJoinConfig>,
    /// Subnet assignment information
    #[serde(skip_serializing_if = "Option::is_none")]
    pub subnet_info: Option<SubnetAssignment>,
}

/// WireGuard configuration returned during join
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireGuardJoinConfig {
    /// Interface name to use
    pub interface_name: String,
    /// Listen port for WireGuard
    pub listen_port: u16,
    /// Node's assigned address with CIDR
    pub address: String,
    /// MTU to use
    pub mtu: u16,
    /// Initial peers to configure
    pub peers: Vec<WireGuardPeerInfo>,
}

/// Peer information from coordinator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireGuardPeerInfo {
    pub node_id: Uuid,
    pub hostname: String,
    pub public_key: String,
    pub allowed_ips: Vec<String>,
    pub endpoint: Option<String>,
    pub persistent_keepalive: u16,
}

/// Subnet assignment information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubnetAssignment {
    pub subnet_id: Uuid,
    pub subnet_name: String,
    pub assigned_ip: String,
}

/// Cluster API endpoints for the joined node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterApiEndpoints {
    pub workload_api: String,
    pub metrics_api: String,
    pub logs_api: String,
    pub health_check: String,
}

/// Join protocol handler
pub struct JoinProtocol {
    token: JoinToken,
    cluster_address: String,
    hardware_profile: HardwareProfile,
    node_name: Option<String>,
    client: reqwest::Client,
    /// WireGuard manager for key generation and config application
    wireguard_manager: Option<Arc<WireGuardManager>>,
    /// WireGuard public key (generated before join)
    wg_public_key: Option<String>,
}

impl JoinProtocol {
    /// Create a new join protocol instance
    pub fn new(token: String, cluster_address: String, hardware_profile: HardwareProfile) -> Self {
        let join_token = JoinToken::new(token);

        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            token: join_token,
            cluster_address,
            hardware_profile,
            node_name: None,
            client,
            wireguard_manager: None,
            wg_public_key: None,
        }
    }

    /// Set the node name for this swarmlet
    pub fn set_node_name(&mut self, name: String) {
        self.node_name = Some(name);
    }

    /// Set the WireGuard manager for mesh networking
    pub fn set_wireguard_manager(&mut self, manager: Arc<WireGuardManager>) {
        self.wireguard_manager = Some(manager);
    }

    /// Generate WireGuard keypair for this node
    ///
    /// This should be called before `join()` to include the public key
    /// in the join request.
    pub async fn generate_wireguard_keypair(&mut self) -> Result<String> {
        let manager = self.wireguard_manager.as_ref().ok_or_else(|| {
            SwarmletError::WireGuard("WireGuard manager not set".to_string())
        })?;

        let public_key = manager.generate_keypair().await?;
        self.wg_public_key = Some(public_key.clone());

        info!("Generated WireGuard keypair for cluster join");
        Ok(public_key)
    }

    /// Perform the join handshake with the cluster
    pub async fn join(&self) -> Result<JoinResult> {
        info!(
            "Starting join handshake with cluster at {}",
            self.cluster_address
        );

        // Step 1: Validate token with cluster
        self.validate_token().await?;

        // Step 2: Submit join request with hardware profile and WireGuard key
        let join_request = self.create_join_request();
        let join_response = self.submit_join_request(join_request).await?;

        // Step 3: Verify cluster response and certificate
        self.verify_join_response(&join_response).await?;

        // Step 4: Apply WireGuard configuration if provided
        if let Some(wg_config) = &join_response.wireguard_config {
            if let Some(manager) = &self.wireguard_manager {
                self.apply_wireguard_config(manager, wg_config).await?;
                info!(
                    "WireGuard mesh configured on interface {} with {} peers",
                    wg_config.interface_name,
                    wg_config.peers.len()
                );
            }
        }

        info!(
            "Successfully joined cluster! Node ID: {}",
            join_response.node_id
        );
        Ok(join_response)
    }

    /// Apply WireGuard configuration received from coordinator
    async fn apply_wireguard_config(
        &self,
        manager: &WireGuardManager,
        config: &WireGuardJoinConfig,
    ) -> Result<()> {
        debug!(
            "Applying WireGuard config: interface={}, address={}",
            config.interface_name, config.address
        );

        // Convert peers to WireGuardPeerConfig
        let peers: Vec<WireGuardPeerConfig> = config
            .peers
            .iter()
            .map(|p| WireGuardPeerConfig {
                public_key: p.public_key.clone(),
                preshared_key: None,
                allowed_ips: p.allowed_ips.clone(),
                endpoint: p.endpoint.clone(),
                persistent_keepalive: Some(p.persistent_keepalive),
            })
            .collect();

        let wg_config = WireGuardConfigRequest {
            interface_name: config.interface_name.clone(),
            private_key: None, // Manager already has the key
            listen_port: config.listen_port,
            address: config.address.clone(),
            mtu: Some(config.mtu),
            peers,
            config_version: "initial".to_string(),
            signature: None,
        };

        let response = manager.apply_config(wg_config).await?;

        if response.success {
            Ok(())
        } else {
            Err(SwarmletError::WireGuard(
                response.error.unwrap_or_else(|| "Unknown error".to_string()),
            ))
        }
    }

    /// Validate the join token with the cluster
    async fn validate_token(&self) -> Result<()> {
        debug!("Validating join token");

        let url = format!("http://{}/api/v1/join/validate", self.cluster_address);

        let request = TokenValidationRequest {
            token: self.token.raw_token().to_string(),
            swarmlet_version: crate::VERSION.to_string(),
            timestamp: chrono::Utc::now(),
        };

        let response = self.client.post(&url).json(&request).send().await?;

        if response.status().is_success() {
            let validation_response: TokenValidationResponse = response.json().await?;

            if validation_response.valid {
                debug!("Token validation successful");
                Ok(())
            } else {
                Err(SwarmletError::InvalidToken(
                    validation_response
                        .reason
                        .unwrap_or_else(|| "Unknown reason".to_string()),
                ))
            }
        } else {
            Err(SwarmletError::JoinProtocol(format!(
                "Token validation failed with status: {}",
                response.status()
            )))
        }
    }

    /// Create a join request with hardware profile
    fn create_join_request(&self) -> JoinRequest {
        let node_name = self
            .node_name
            .clone()
            .unwrap_or_else(|| self.hardware_profile.hostname.clone());

        JoinRequest {
            token: self.token.raw_token().to_string(),
            node_name,
            hardware_profile: self.hardware_profile.clone(),
            swarmlet_version: crate::VERSION.to_string(),
            requested_capabilities: self.determine_requested_capabilities(),
            timestamp: chrono::Utc::now(),
            wg_public_key: self.wg_public_key.clone(),
            wg_listen_port: Some(51820), // Default WireGuard port
        }
    }

    /// Submit join request to cluster
    async fn submit_join_request(&self, request: JoinRequest) -> Result<JoinResult> {
        debug!("Submitting join request");

        let url = format!("http://{}/api/v1/join/request", self.cluster_address);

        let response = self.client.post(&url).json(&request).send().await?;

        if response.status().is_success() {
            let join_response: JoinResponse = response.json().await?;

            if join_response.accepted {
                debug!("Join request accepted by cluster");

                Ok(JoinResult {
                    node_id: join_response.node_id,
                    cluster_name: join_response.cluster_name,
                    node_certificate: join_response.node_certificate,
                    cluster_endpoints: join_response.cluster_endpoints,
                    assigned_capabilities: join_response.assigned_capabilities,
                    heartbeat_interval: Duration::from_secs(join_response.heartbeat_interval_secs),
                    api_endpoints: join_response.api_endpoints,
                    wireguard_config: join_response.wireguard_config,
                    subnet_info: join_response.subnet,
                })
            } else {
                Err(SwarmletError::ClusterRejection(
                    join_response
                        .rejection_reason
                        .unwrap_or_else(|| "Unknown reason".to_string()),
                ))
            }
        } else if response.status() == 429 {
            Err(SwarmletError::ClusterRejection(
                "Cluster is at capacity - try again later".to_string(),
            ))
        } else {
            Err(SwarmletError::JoinProtocol(format!(
                "Join request failed with status: {}",
                response.status()
            )))
        }
    }

    /// Verify the join response and certificate
    async fn verify_join_response(&self, join_result: &JoinResult) -> Result<()> {
        debug!("Verifying join response and certificate");

        // Verify certificate format (basic validation)
        if join_result.node_certificate.is_empty() {
            return Err(SwarmletError::JoinProtocol(
                "Empty node certificate received".to_string(),
            ));
        }

        // Verify cluster endpoints are reachable
        for endpoint in &join_result.cluster_endpoints {
            match reqwest::get(&format!("http://{endpoint}/health")).await {
                Ok(response) if response.status().is_success() => {
                    debug!("Verified endpoint: {}", endpoint);
                }
                Ok(response) => {
                    warn!(
                        "Endpoint {} returned status: {}",
                        endpoint,
                        response.status()
                    );
                }
                Err(e) => {
                    warn!("Could not reach endpoint {}: {}", endpoint, e);
                }
            }
        }

        // Test API endpoints
        let health_check_url = &join_result.api_endpoints.health_check;
        match self.client.get(health_check_url).send().await {
            Ok(response) if response.status().is_success() => {
                debug!("Health check endpoint verified: {}", health_check_url);
            }
            Ok(response) => {
                warn!("Health check failed with status: {}", response.status());
            }
            Err(e) => {
                warn!("Could not reach health check endpoint: {}", e);
            }
        }

        Ok(())
    }

    /// Determine capabilities to request from cluster
    fn determine_requested_capabilities(&self) -> Vec<String> {
        let mut capabilities = Vec::new();

        // Basic capabilities all swarmlets should have
        capabilities.push("workload_execution".to_string());
        capabilities.push("health_reporting".to_string());
        capabilities.push("metrics_collection".to_string());

        // Hardware-specific capabilities
        if self.hardware_profile.capabilities.gpu_capable {
            capabilities.push("gpu_workloads".to_string());
        }

        if self.hardware_profile.capabilities.container_runtime {
            capabilities.push("container_execution".to_string());
        }

        // Device-type specific capabilities
        match self.hardware_profile.device_type {
            crate::profile::DeviceType::RaspberryPi | crate::profile::DeviceType::EdgeDevice => {
                capabilities.push("edge_computing".to_string());
                capabilities.push("sensor_data".to_string());
            }
            crate::profile::DeviceType::Server => {
                capabilities.push("high_performance".to_string());
                capabilities.push("storage_services".to_string());
            }
            crate::profile::DeviceType::Workstation => {
                capabilities.push("development_workloads".to_string());
                capabilities.push("testing_services".to_string());
            }
            _ => {}
        }

        capabilities
    }
}

/// Token validation request
#[derive(Debug, Serialize, Deserialize)]
struct TokenValidationRequest {
    token: String,
    swarmlet_version: String,
    timestamp: chrono::DateTime<chrono::Utc>,
}

/// Token validation response
#[derive(Debug, Serialize, Deserialize)]
struct TokenValidationResponse {
    valid: bool,
    reason: Option<String>,
    expires_at: Option<chrono::DateTime<chrono::Utc>>,
}

/// Join request sent to cluster
#[derive(Debug, Serialize, Deserialize)]
struct JoinRequest {
    token: String,
    node_name: String,
    hardware_profile: HardwareProfile,
    swarmlet_version: String,
    requested_capabilities: Vec<String>,
    timestamp: chrono::DateTime<chrono::Utc>,
    /// WireGuard public key for mesh networking
    #[serde(skip_serializing_if = "Option::is_none")]
    wg_public_key: Option<String>,
    /// Preferred WireGuard listen port
    #[serde(skip_serializing_if = "Option::is_none")]
    wg_listen_port: Option<u16>,
}

/// Join response from cluster
#[derive(Debug, Serialize, Deserialize)]
struct JoinResponse {
    accepted: bool,
    node_id: Uuid,
    cluster_name: String,
    node_certificate: String,
    cluster_endpoints: Vec<String>,
    assigned_capabilities: Vec<String>,
    heartbeat_interval_secs: u64,
    api_endpoints: ClusterApiEndpoints,
    rejection_reason: Option<String>,
    /// WireGuard configuration (if mesh networking is enabled)
    #[serde(skip_serializing_if = "Option::is_none")]
    wireguard_config: Option<WireGuardJoinConfig>,
    /// Subnet assignment information
    #[serde(skip_serializing_if = "Option::is_none")]
    subnet: Option<SubnetAssignment>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profile::{DeviceType, HardwareProfiler, NodeCapabilities, WorkloadType};

    fn create_test_hardware_profile() -> HardwareProfile {
        let mut _profiler = HardwareProfiler::new();
        // Create a mock profile for testing
        HardwareProfile {
            node_id: Uuid::new_v4(),
            hostname: "test-node".to_string(),
            architecture: "x86_64".to_string(),
            cpu: crate::profile::CpuInfo {
                model: "Test CPU".to_string(),
                cores: 4,
                threads: 8,
                frequency_mhz: 3000,
                architecture: "x86_64".to_string(),
                features: vec![],
            },
            memory: crate::profile::MemoryInfo {
                total_gb: 8.0,
                available_gb: 6.0,
                swap_gb: 2.0,
            },
            storage: crate::profile::StorageInfo {
                total_gb: 500.0,
                available_gb: 400.0,
                disks: vec![],
            },
            network: crate::profile::NetworkInfo {
                interfaces: vec![],
                estimated_bandwidth_mbps: 100.0,
                connectivity: crate::profile::ConnectivityType::Ethernet,
            },
            gpu: None,
            capabilities: NodeCapabilities {
                max_workloads: 8,
                memory_limit_gb: 4.0,
                cpu_limit_cores: 3,
                storage_limit_gb: 100.0,
                gpu_capable: false,
                container_runtime: true,
                network_bandwidth_mbps: 100.0,
                suitability: vec![WorkloadType::Development, WorkloadType::Testing],
            },
            thermal: None,
            power: None,
            device_type: DeviceType::Workstation,
        }
    }

    #[test]
    fn test_join_protocol_creation() {
        let profile = create_test_hardware_profile();
        let join_protocol = JoinProtocol::new(
            "test_token_123".to_string(),
            "192.168.1.100:7946".to_string(),
            profile,
        );

        assert_eq!(join_protocol.cluster_address, "192.168.1.100:7946");
        assert!(join_protocol.node_name.is_none());
    }

    #[test]
    fn test_set_node_name() {
        let profile = create_test_hardware_profile();
        let mut join_protocol = JoinProtocol::new(
            "test_token_123".to_string(),
            "192.168.1.100:7946".to_string(),
            profile,
        );

        join_protocol.set_node_name("my-swarmlet".to_string());
        assert_eq!(join_protocol.node_name, Some("my-swarmlet".to_string()));
    }

    #[test]
    fn test_determine_requested_capabilities() {
        let profile = create_test_hardware_profile();
        let join_protocol = JoinProtocol::new(
            "test_token_123".to_string(),
            "192.168.1.100:7946".to_string(),
            profile,
        );

        let capabilities = join_protocol.determine_requested_capabilities();

        // Should have basic capabilities
        assert!(capabilities.contains(&"workload_execution".to_string()));
        assert!(capabilities.contains(&"health_reporting".to_string()));
        assert!(capabilities.contains(&"metrics_collection".to_string()));

        // Should have container capability
        assert!(capabilities.contains(&"container_execution".to_string()));

        // Should have workstation-specific capabilities
        assert!(capabilities.contains(&"development_workloads".to_string()));
        assert!(capabilities.contains(&"testing_services".to_string()));
    }

    #[test]
    fn test_join_result_serialization() {
        let join_result = JoinResult {
            node_id: Uuid::new_v4(),
            cluster_name: "prod-cluster".to_string(),
            node_certificate: "-----BEGIN CERTIFICATE-----\ntest\n-----END CERTIFICATE-----"
                .to_string(),
            cluster_endpoints: vec!["http://api1.cluster.local".to_string()],
            assigned_capabilities: vec!["compute".to_string()],
            heartbeat_interval: Duration::from_secs(30),
            api_endpoints: ClusterApiEndpoints {
                workload_api: "http://workload.api".to_string(),
                metrics_api: "http://metrics.api".to_string(),
                logs_api: "http://logs.api".to_string(),
                health_check: "http://health.api".to_string(),
            },
            wireguard_config: None,
            subnet_info: None,
        };

        // Test serialization
        let json = serde_json::to_string(&join_result).expect("Should serialize");
        assert!(json.contains("prod-cluster"));
        assert!(json.contains("workload.api"));

        // Test deserialization
        let deserialized: JoinResult = serde_json::from_str(&json).expect("Should deserialize");
        assert_eq!(deserialized.node_id, join_result.node_id);
        assert_eq!(deserialized.cluster_name, "prod-cluster");
    }

    #[test]
    fn test_cluster_api_endpoints() {
        let endpoints = ClusterApiEndpoints {
            workload_api: "http://localhost:8081".to_string(),
            metrics_api: "http://localhost:9090".to_string(),
            logs_api: "http://localhost:8082".to_string(),
            health_check: "http://localhost:8080".to_string(),
        };

        // Test all fields are accessible
        assert_eq!(endpoints.workload_api, "http://localhost:8081");
        assert_eq!(endpoints.metrics_api, "http://localhost:9090");
        assert_eq!(endpoints.logs_api, "http://localhost:8082");
        assert_eq!(endpoints.health_check, "http://localhost:8080");
    }

    #[test]
    fn test_token_validation_request() {
        let request = TokenValidationRequest {
            token: "secret-token".to_string(),
            swarmlet_version: "1.0.0".to_string(),
            timestamp: chrono::Utc::now(),
        };

        // Test serialization
        let json = serde_json::to_string(&request).expect("Should serialize");
        assert!(json.contains("secret-token"));
        assert!(json.contains("1.0.0"));
    }

    #[test]
    fn test_join_request_creation() {
        let profile = create_test_hardware_profile();
        let join_protocol = JoinProtocol::new(
            "test_token".to_string(),
            "cluster.local".to_string(),
            profile.clone(),
        );

        let join_request = join_protocol.create_join_request();

        assert_eq!(join_request.token, "test_token");
        assert_eq!(join_request.swarmlet_version, crate::VERSION);
        assert_eq!(join_request.hardware_profile.hostname, profile.hostname);
        assert_eq!(join_request.node_name, "test-node"); // defaults to hostname
    }

    #[test]
    fn test_join_request_with_node_name() {
        let profile = create_test_hardware_profile();
        let mut join_protocol = JoinProtocol::new(
            "test_token".to_string(),
            "cluster.local".to_string(),
            profile,
        );

        join_protocol.set_node_name("custom-node".to_string());
        let join_request = join_protocol.create_join_request();

        assert_eq!(join_request.node_name, "custom-node");
    }

    #[test]
    fn test_determine_capabilities_with_gpu() {
        let mut profile = create_test_hardware_profile();

        // Add GPU to profile
        profile.gpu = Some(crate::profile::GpuInfo {
            count: 1,
            models: vec!["NVIDIA RTX 4090".to_string()],
            total_memory_gb: 24.0,
            compute_capability: Some("8.9".to_string()),
        });
        profile.capabilities.gpu_capable = true;

        let join_protocol = JoinProtocol::new(
            "test_token".to_string(),
            "cluster.local".to_string(),
            profile,
        );

        let capabilities = join_protocol.determine_requested_capabilities();

        // Should have GPU capabilities
        assert!(capabilities.contains(&"gpu_workloads".to_string()));
        // Should still have basic capabilities
        assert!(capabilities.contains(&"workload_execution".to_string()));
        assert!(capabilities.contains(&"container_execution".to_string()));
    }

    #[tokio::test]
    async fn test_join_protocol_timeout() {
        let profile = create_test_hardware_profile();
        let join_protocol = JoinProtocol::new(
            "test_token".to_string(),
            "invalid-cluster-address".to_string(),
            profile,
        );

        // This should fail with a network error
        let result = join_protocol.join().await;
        assert!(result.is_err());
    }
}
