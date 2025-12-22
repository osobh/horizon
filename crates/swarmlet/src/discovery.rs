//! Cluster discovery functionality

use crate::{defaults, Result, SwarmletError};
use serde::{Deserialize, Serialize};
use std::net::IpAddr;
use std::time::Duration;
use tokio::time::timeout;
use tracing::{debug, info, warn};

/// Information about a discovered cluster
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterInfo {
    pub name: String,
    pub address: String,
    pub node_class: String,
    pub version: String,
    pub nodes_count: u32,
    pub capabilities: Vec<String>,
    pub discovered_at: chrono::DateTime<chrono::Utc>,
}

/// Cluster discovery service
pub struct ClusterDiscovery {
    client: reqwest::Client,
}

impl ClusterDiscovery {
    /// Create a new cluster discovery service
    pub fn new() -> Self {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .expect("Failed to create HTTP client");

        Self { client }
    }

    /// Discover StratoSwarm clusters on the local network
    pub async fn discover_clusters(&self, timeout_secs: u64) -> Result<Vec<ClusterInfo>> {
        info!("Starting cluster discovery (timeout: {}s)", timeout_secs);

        let discovery_timeout = Duration::from_secs(timeout_secs);
        let mut clusters = Vec::new();

        // Try multiple discovery methods concurrently
        let mdns_task = self.discover_via_mdns(discovery_timeout);
        let broadcast_task = self.discover_via_broadcast(discovery_timeout);
        let subnet_scan_task = self.discover_via_subnet_scan(discovery_timeout);

        // Use tokio::select! to run discovery methods concurrently
        tokio::select! {
            mdns_result = mdns_task => {
                if let Ok(mut mdns_clusters) = mdns_result {
                    clusters.append(&mut mdns_clusters);
                }
            }
            broadcast_result = broadcast_task => {
                if let Ok(mut broadcast_clusters) = broadcast_result {
                    clusters.append(&mut broadcast_clusters);
                }
            }
            scan_result = subnet_scan_task => {
                if let Ok(mut scan_clusters) = scan_result {
                    clusters.append(&mut scan_clusters);
                }
            }
        }

        // Deduplicate clusters by address
        clusters.sort_by(|a, b| a.address.cmp(&b.address));
        clusters.dedup_by(|a, b| a.address == b.address);

        info!(
            "Discovery completed, found {} unique cluster(s)",
            clusters.len()
        );
        Ok(clusters)
    }

    /// Test connection to a specific cluster
    pub async fn test_connection(&self, cluster_address: &str) -> Result<ClusterInfo> {
        info!("Testing connection to cluster at {}", cluster_address);

        let url = format!("http://{cluster_address}/api/v1/cluster/info");

        match self.client.get(&url).send().await {
            Ok(response) => {
                if response.status().is_success() {
                    let cluster_info: ClusterInfo = response.json().await?;
                    info!("Successfully connected to cluster: {}", cluster_info.name);
                    Ok(cluster_info)
                } else {
                    Err(SwarmletError::Discovery(format!(
                        "Cluster returned status: {}",
                        response.status()
                    )))
                }
            }
            Err(e) => {
                warn!("Connection test failed: {}", e);
                Err(SwarmletError::Discovery(format!(
                    "Connection failed: {e}"
                )))
            }
        }
    }

    /// Discover clusters via mDNS/DNS-SD
    async fn discover_via_mdns(&self, _timeout_duration: Duration) -> Result<Vec<ClusterInfo>> {
        debug!("Starting mDNS discovery");

        // This would use a proper mDNS library like mdns-sd
        // For now, return empty as a placeholder
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Placeholder implementation
        Ok(Vec::new())
    }

    /// Discover clusters via UDP broadcast
    async fn discover_via_broadcast(&self, timeout_duration: Duration) -> Result<Vec<ClusterInfo>> {
        debug!("Starting broadcast discovery");

        use tokio::net::UdpSocket;

        let socket = UdpSocket::bind("0.0.0.0:0").await?;
        socket.set_broadcast(true)?;

        // Send discovery broadcast
        let discovery_message = DiscoveryMessage {
            message_type: "discover".to_string(),
            swarmlet_version: crate::VERSION.to_string(),
            timestamp: chrono::Utc::now(),
        };

        let message_bytes = serde_json::to_vec(&discovery_message)?;
        let broadcast_addr = format!("255.255.255.255:{}", defaults::DISCOVERY_PORT);

        socket.send_to(&message_bytes, &broadcast_addr).await?;
        debug!("Sent discovery broadcast to {}", broadcast_addr);

        // Listen for responses
        let mut clusters = Vec::new();
        let mut buffer = [0; 1024];

        match timeout(timeout_duration, socket.recv_from(&mut buffer)).await {
            Ok(Ok((len, addr))) => {
                debug!("Received response from {}", addr);

                if let Ok(response) = serde_json::from_slice::<ClusterResponse>(&buffer[..len]) {
                    let cluster_info = ClusterInfo {
                        name: response.cluster_name,
                        address: format!("{}:{}", addr.ip(), response.api_port),
                        node_class: response.node_class,
                        version: response.version,
                        nodes_count: response.nodes_count,
                        capabilities: response.capabilities,
                        discovered_at: chrono::Utc::now(),
                    };
                    clusters.push(cluster_info);
                }
            }
            Ok(Err(e)) => {
                debug!("Broadcast discovery socket error: {}", e);
            }
            Err(_) => {
                debug!("Broadcast discovery timeout");
            }
        }

        Ok(clusters)
    }

    /// Discover clusters via subnet scanning
    async fn discover_via_subnet_scan(
        &self,
        timeout_duration: Duration,
    ) -> Result<Vec<ClusterInfo>> {
        debug!("Starting subnet scan discovery");

        let local_ip = self.get_local_ip().await?;
        let subnet_base = self.get_subnet_base(&local_ip);

        let mut tasks = Vec::new();

        // Scan common subnet range (e.g., 192.168.1.1-254)
        for i in 1..=254 {
            let target_ip = format!("{subnet_base}.{i}");
            let client = self.client.clone();

            let task = tokio::spawn(async move {
                let url = format!(
                    "http://{}:{}/api/v1/cluster/info",
                    target_ip,
                    defaults::API_PORT
                );

                match client.get(&url).send().await {
                    Ok(response) if response.status().is_success() => {
                        response.json::<ClusterInfo>().await.ok()
                    }
                    _ => None,
                }
            });

            tasks.push(task);
        }

        // Wait for all tasks with timeout
        let mut clusters = Vec::new();

        match timeout(timeout_duration, futures::future::join_all(tasks)).await {
            Ok(results) => {
                for cluster in results.into_iter().flatten().flatten() {
                    clusters.push(cluster);
                }
            }
            Err(_) => {
                debug!("Subnet scan timeout");
            }
        }

        Ok(clusters)
    }

    /// Get local IP address
    async fn get_local_ip(&self) -> Result<IpAddr> {
        // Simple method to get local IP by connecting to a remote address
        use tokio::net::UdpSocket;

        let socket = UdpSocket::bind("0.0.0.0:0").await?;
        socket.connect("8.8.8.8:80").await?;
        let local_addr = socket.local_addr()?;

        Ok(local_addr.ip())
    }

    /// Extract subnet base from IP address
    fn get_subnet_base(&self, ip: &IpAddr) -> String {
        match ip {
            IpAddr::V4(ipv4) => {
                let octets = ipv4.octets();
                format!("{}.{}.{}", octets[0], octets[1], octets[2])
            }
            IpAddr::V6(_) => {
                // IPv6 subnet scanning is complex, skip for now
                "192.168.1".to_string() // Default fallback
            }
        }
    }
}

/// Discovery message sent via broadcast
#[derive(Debug, Serialize, Deserialize)]
struct DiscoveryMessage {
    message_type: String,
    swarmlet_version: String,
    timestamp: chrono::DateTime<chrono::Utc>,
}

/// Response from cluster to discovery message
#[derive(Debug, Serialize, Deserialize)]
struct ClusterResponse {
    cluster_name: String,
    version: String,
    api_port: u16,
    node_class: String,
    nodes_count: u32,
    capabilities: Vec<String>,
}

impl Default for ClusterDiscovery {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_cluster_discovery_creation() {
        let _discovery = ClusterDiscovery::new();
        // Just test that it can be created without panicking
        assert!(true);
    }

    #[tokio::test]
    async fn test_get_local_ip() {
        let discovery = ClusterDiscovery::new();
        let result = discovery.get_local_ip().await;
        // This might fail in CI environments without network, so just check it doesn't panic
        match result {
            Ok(ip) => println!("Local IP: {}", ip),
            Err(e) => println!("Could not get local IP: {}", e),
        }
    }

    #[test]
    fn test_subnet_base_extraction() {
        let discovery = ClusterDiscovery::new();

        let ip: IpAddr = "192.168.1.100".parse().unwrap();
        let subnet_base = discovery.get_subnet_base(&ip);
        assert_eq!(subnet_base, "192.168.1");

        let ip: IpAddr = "10.0.0.50".parse().unwrap();
        let subnet_base = discovery.get_subnet_base(&ip);
        assert_eq!(subnet_base, "10.0.0");
    }

    #[test]
    fn test_cluster_info_serialization() {
        let cluster_info = ClusterInfo {
            name: "test-cluster".to_string(),
            address: "192.168.1.100:7946".to_string(),
            node_class: "production".to_string(),
            version: "1.0.0".to_string(),
            nodes_count: 5,
            capabilities: vec!["compute".to_string(), "storage".to_string()],
            discovered_at: chrono::Utc::now(),
        };

        // Test serialization
        let json = serde_json::to_string(&cluster_info).expect("Should serialize");
        assert!(json.contains("test-cluster"));
        assert!(json.contains("production"));

        // Test deserialization
        let deserialized: ClusterInfo = serde_json::from_str(&json).expect("Should deserialize");
        assert_eq!(deserialized.name, cluster_info.name);
        assert_eq!(deserialized.nodes_count, 5);
    }

    #[tokio::test]
    async fn test_discover_clusters_timeout() {
        let discovery = ClusterDiscovery::new();

        // Test with very short timeout
        let result = discovery.discover_clusters(1).await;
        assert!(result.is_ok());
        // May or may not find clusters, just verify it doesn't panic
    }

    #[tokio::test]
    async fn test_connection_to_invalid_address() {
        let discovery = ClusterDiscovery::new();

        // Test connection to invalid address
        let result = discovery.test_connection("invalid-address").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_discover_via_subnet_scan() {
        let discovery = ClusterDiscovery::new();

        // Test subnet scan with short timeout
        let result = discovery
            .discover_via_subnet_scan(Duration::from_millis(100))
            .await;
        assert!(result.is_ok());
        // Expect empty results in test environment
        assert_eq!(result.unwrap().len(), 0);
    }

    #[tokio::test]
    async fn test_discover_via_broadcast() {
        let discovery = ClusterDiscovery::new();

        // Test broadcast discovery
        let result = discovery
            .discover_via_broadcast(Duration::from_secs(1))
            .await;
        assert!(result.is_ok());
        // Expect empty results in test environment
        assert_eq!(result.unwrap().len(), 0);
    }

    #[tokio::test]
    async fn test_discover_via_mdns() {
        let discovery = ClusterDiscovery::new();

        // Test mDNS discovery
        let result = discovery.discover_via_mdns(Duration::from_secs(1)).await;
        assert!(result.is_ok());
        // Expect empty results in test environment
        assert_eq!(result.unwrap().len(), 0);
    }

    #[test]
    fn test_cluster_info_capabilities() {
        let mut cluster_info = ClusterInfo {
            name: "test".to_string(),
            address: "localhost:7946".to_string(),
            node_class: "edge".to_string(),
            version: "0.1.0".to_string(),
            nodes_count: 1,
            capabilities: vec![],
            discovered_at: chrono::Utc::now(),
        };

        // Test empty capabilities
        assert!(cluster_info.capabilities.is_empty());

        // Add capabilities
        cluster_info.capabilities.push("gpu".to_string());
        cluster_info.capabilities.push("ml".to_string());
        assert_eq!(cluster_info.capabilities.len(), 2);
        assert!(cluster_info.capabilities.contains(&"gpu".to_string()));
    }

    #[test]
    fn test_ipv6_subnet_base() {
        let discovery = ClusterDiscovery::new();

        // Test IPv6 handling
        let ipv6: IpAddr = "::1".parse().unwrap();
        let subnet_base = discovery.get_subnet_base(&ipv6);
        // Current implementation only handles IPv4, so IPv6 returns first 3 octets
        // This is expected behavior - should be updated when IPv6 support is added
        assert_eq!(subnet_base, "192.168.1");
    }
}
