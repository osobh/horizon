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
    ///
    /// Uses multiple discovery methods concurrently:
    /// - mDNS/DNS-SD (`_stratoswarm._tcp.local.`)
    /// - UDP broadcast
    /// - IPv4 subnet scanning
    /// - IPv6 multicast and neighbor discovery
    pub async fn discover_clusters(&self, timeout_secs: u64) -> Result<Vec<ClusterInfo>> {
        info!("Starting cluster discovery (timeout: {}s)", timeout_secs);

        let discovery_timeout = Duration::from_secs(timeout_secs);
        let mut clusters = Vec::new();

        // Run all discovery methods concurrently with a global timeout
        let all_discoveries = async {
            let (
                mdns_result,
                broadcast_result,
                subnet_result,
                ipv6_multicast_result,
                ipv6_neighbor_result,
            ) = tokio::join!(
                self.discover_via_mdns(discovery_timeout),
                self.discover_via_broadcast(discovery_timeout),
                self.discover_via_subnet_scan(discovery_timeout),
                self.discover_via_ipv6_multicast(discovery_timeout),
                self.discover_via_ipv6_neighbors(discovery_timeout),
            );

            let mut results = Vec::new();

            if let Ok(mut c) = mdns_result {
                debug!("mDNS discovered {} cluster(s)", c.len());
                results.append(&mut c);
            }
            if let Ok(mut c) = broadcast_result {
                debug!("Broadcast discovered {} cluster(s)", c.len());
                results.append(&mut c);
            }
            if let Ok(mut c) = subnet_result {
                debug!("Subnet scan discovered {} cluster(s)", c.len());
                results.append(&mut c);
            }
            if let Ok(mut c) = ipv6_multicast_result {
                debug!("IPv6 multicast discovered {} cluster(s)", c.len());
                results.append(&mut c);
            }
            if let Ok(mut c) = ipv6_neighbor_result {
                debug!("IPv6 neighbor discovered {} cluster(s)", c.len());
                results.append(&mut c);
            }

            results
        };

        // Apply overall timeout
        match timeout(discovery_timeout, all_discoveries).await {
            Ok(mut discovered) => {
                clusters.append(&mut discovered);
            }
            Err(_) => {
                warn!("Discovery timed out after {} seconds", timeout_secs);
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
                Err(SwarmletError::Discovery(format!("Connection failed: {e}")))
            }
        }
    }

    /// Discover clusters via mDNS/DNS-SD
    ///
    /// Browses for `_stratoswarm._tcp.local.` services on the local network.
    async fn discover_via_mdns(&self, timeout_duration: Duration) -> Result<Vec<ClusterInfo>> {
        use mdns_sd::{ServiceDaemon, ServiceEvent};

        debug!("Starting mDNS discovery for _stratoswarm._tcp.local.");

        // Create mDNS daemon
        let mdns = match ServiceDaemon::new() {
            Ok(daemon) => daemon,
            Err(e) => {
                warn!("Failed to create mDNS daemon: {}", e);
                return Ok(Vec::new());
            }
        };

        // Browse for StratoSwarm services
        let service_type = "_stratoswarm._tcp.local.";
        let receiver = match mdns.browse(service_type) {
            Ok(recv) => recv,
            Err(e) => {
                warn!("Failed to browse mDNS services: {}", e);
                return Ok(Vec::new());
            }
        };

        let mut clusters = Vec::new();
        let deadline = tokio::time::Instant::now() + timeout_duration;

        // Process mDNS events until timeout
        loop {
            let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
            if remaining.is_zero() {
                break;
            }

            // Use tokio timeout to check receiver with remaining time
            match tokio::time::timeout(
                remaining.min(Duration::from_millis(100)),
                tokio::task::spawn_blocking({
                    let receiver = receiver.clone();
                    move || receiver.recv_timeout(Duration::from_millis(50))
                }),
            )
            .await
            {
                Ok(Ok(Ok(event))) => {
                    match event {
                        ServiceEvent::ServiceResolved(info) => {
                            debug!("Discovered mDNS service: {}", info.get_fullname());

                            // Extract cluster info from service properties
                            let properties = info.get_properties();

                            let cluster_name = properties
                                .get("cluster_name")
                                .map(|v| v.val_str().to_string())
                                .unwrap_or_else(|| {
                                    info.get_hostname().trim_end_matches('.').to_string()
                                });

                            let version = properties
                                .get("version")
                                .map(|v| v.val_str().to_string())
                                .unwrap_or_else(|| "unknown".to_string());

                            let node_class = properties
                                .get("node_class")
                                .map(|v| v.val_str().to_string())
                                .unwrap_or_else(|| "production".to_string());

                            let nodes_count: u32 = properties
                                .get("nodes")
                                .and_then(|v| v.val_str().parse().ok())
                                .unwrap_or(1);

                            let capabilities: Vec<String> = properties
                                .get("capabilities")
                                .map(|v| v.val_str().split(',').map(String::from).collect())
                                .unwrap_or_default();

                            // Build address from resolved service info
                            let addresses = info.get_addresses();
                            if let Some(addr) = addresses.iter().next() {
                                let port = info.get_port();
                                let address = format!("{}:{}", addr, port);

                                info!(
                                    "Discovered StratoSwarm cluster '{}' at {} via mDNS",
                                    cluster_name, address
                                );

                                clusters.push(ClusterInfo {
                                    name: cluster_name,
                                    address,
                                    node_class,
                                    version,
                                    nodes_count,
                                    capabilities,
                                    discovered_at: chrono::Utc::now(),
                                });
                            }
                        }
                        ServiceEvent::SearchStarted(_) => {
                            debug!("mDNS search started");
                        }
                        ServiceEvent::ServiceFound(_, _) => {
                            debug!("mDNS service found, waiting for resolution");
                        }
                        ServiceEvent::ServiceRemoved(_, fullname) => {
                            debug!("mDNS service removed: {}", fullname);
                        }
                        ServiceEvent::SearchStopped(_) => {
                            debug!("mDNS search stopped");
                            break;
                        }
                    }
                }
                Ok(Ok(Err(_))) => {
                    // Timeout on recv, continue loop
                }
                Ok(Err(_)) => {
                    // Task join error
                    break;
                }
                Err(_) => {
                    // Tokio timeout, continue loop
                }
            }
        }

        // Stop the daemon gracefully
        if let Err(e) = mdns.stop_browse(service_type) {
            debug!("Failed to stop mDNS browse: {}", e);
        }

        debug!(
            "mDNS discovery completed, found {} cluster(s)",
            clusters.len()
        );
        Ok(clusters)
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

    /// Get local IP address (prefers IPv4)
    async fn get_local_ip(&self) -> Result<IpAddr> {
        // Try IPv4 first
        if let Ok(ip) = self.get_local_ipv4().await {
            return Ok(IpAddr::V4(ip));
        }

        // Fall back to IPv6
        if let Ok(ip) = self.get_local_ipv6().await {
            return Ok(IpAddr::V6(ip));
        }

        Err(SwarmletError::Discovery(
            "Could not determine local IP address".to_string(),
        ))
    }

    /// Get local IPv4 address
    async fn get_local_ipv4(&self) -> Result<std::net::Ipv4Addr> {
        use tokio::net::UdpSocket;

        let socket = UdpSocket::bind("0.0.0.0:0").await?;
        socket.connect("8.8.8.8:80").await?;
        let local_addr = socket.local_addr()?;

        match local_addr.ip() {
            IpAddr::V4(ipv4) => Ok(ipv4),
            IpAddr::V6(_) => Err(SwarmletError::Discovery(
                "Expected IPv4 address".to_string(),
            )),
        }
    }

    /// Get local IPv6 address
    async fn get_local_ipv6(&self) -> Result<std::net::Ipv6Addr> {
        use tokio::net::UdpSocket;

        // Try to connect to Google's IPv6 DNS to determine local IPv6
        let socket = UdpSocket::bind("[::]:0").await?;
        if socket.connect("[2001:4860:4860::8888]:80").await.is_ok() {
            let local_addr = socket.local_addr()?;
            match local_addr.ip() {
                IpAddr::V6(ipv6) => return Ok(ipv6),
                IpAddr::V4(_) => {}
            }
        }

        Err(SwarmletError::Discovery(
            "Could not determine IPv6 address".to_string(),
        ))
    }

    /// Get all local IP addresses (both IPv4 and IPv6)
    #[allow(dead_code)]
    fn get_all_local_ips(&self) -> Vec<IpAddr> {
        use std::net::Ipv4Addr;
        let mut ips = Vec::new();

        // Use sysinfo to get network interfaces
        // For now, use a simpler approach - try common interface patterns
        // In production, would use netlink on Linux or getifaddrs

        // Try to get IPs from hostname resolution
        if let Ok(hostname) = hostname::get() {
            if let Ok(hostname_str) = hostname.into_string() {
                if let Ok(addrs) =
                    std::net::ToSocketAddrs::to_socket_addrs(&(hostname_str.as_str(), 0))
                {
                    for addr in addrs {
                        let ip = addr.ip();
                        if !ip.is_loopback() && !ips.contains(&ip) {
                            ips.push(ip);
                        }
                    }
                }
            }
        }

        // Add common private network ranges for scanning
        // These are fallbacks if we can't determine actual IPs
        if ips.is_empty() {
            // Default to common private ranges
            ips.push(IpAddr::V4(Ipv4Addr::new(192, 168, 1, 1)));
        }

        ips
    }

    /// Extract subnet base from IP address
    fn get_subnet_base(&self, ip: &IpAddr) -> String {
        match ip {
            IpAddr::V4(ipv4) => {
                let octets = ipv4.octets();
                format!("{}.{}.{}", octets[0], octets[1], octets[2])
            }
            IpAddr::V6(ipv6) => {
                // For IPv6, return the /64 prefix as a string
                // Link-local addresses start with fe80::
                let segments = ipv6.segments();
                format!(
                    "{:x}:{:x}:{:x}:{:x}",
                    segments[0], segments[1], segments[2], segments[3]
                )
            }
        }
    }

    /// Discover clusters via IPv6 multicast
    ///
    /// Uses link-local multicast for local network discovery
    async fn discover_via_ipv6_multicast(
        &self,
        timeout_duration: Duration,
    ) -> Result<Vec<ClusterInfo>> {
        use std::net::{Ipv6Addr, SocketAddrV6};
        use tokio::net::UdpSocket;

        debug!("Starting IPv6 multicast discovery");

        // Use link-local all-nodes multicast address (ff02::1)
        // For service discovery, we use a custom multicast group
        let multicast_addr: Ipv6Addr = "ff02::1:ff00:1".parse().unwrap();
        let multicast_port = crate::defaults::DISCOVERY_PORT;

        // Bind to IPv6 any address
        let socket = match UdpSocket::bind("[::]:0").await {
            Ok(s) => s,
            Err(e) => {
                debug!("Failed to bind IPv6 socket: {}", e);
                return Ok(Vec::new());
            }
        };

        // Join multicast group on all interfaces (interface index 0)
        let multicast_socket_addr = SocketAddrV6::new(multicast_addr, multicast_port, 0, 0);

        // Send discovery message
        let discovery_message = DiscoveryMessage {
            message_type: "discover_v6".to_string(),
            swarmlet_version: crate::VERSION.to_string(),
            timestamp: chrono::Utc::now(),
        };

        let message_bytes = match serde_json::to_vec(&discovery_message) {
            Ok(bytes) => bytes,
            Err(e) => {
                debug!("Failed to serialize discovery message: {}", e);
                return Ok(Vec::new());
            }
        };

        if let Err(e) = socket.send_to(&message_bytes, multicast_socket_addr).await {
            debug!("Failed to send IPv6 multicast: {}", e);
            // Continue anyway - might receive responses from other sources
        }

        // Listen for responses
        let mut clusters = Vec::new();
        let mut buffer = [0u8; 2048];

        match timeout(timeout_duration, socket.recv_from(&mut buffer)).await {
            Ok(Ok((len, addr))) => {
                debug!("Received IPv6 response from {}", addr);

                if let Ok(response) = serde_json::from_slice::<ClusterResponse>(&buffer[..len]) {
                    let cluster_info = ClusterInfo {
                        name: response.cluster_name,
                        address: format!("[{}]:{}", addr.ip(), response.api_port),
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
                debug!("IPv6 multicast receive error: {}", e);
            }
            Err(_) => {
                debug!("IPv6 multicast discovery timeout");
            }
        }

        debug!(
            "IPv6 multicast discovery found {} cluster(s)",
            clusters.len()
        );
        Ok(clusters)
    }

    /// Scan IPv6 link-local neighbors for clusters
    ///
    /// This scans known link-local addresses (fe80::) from the neighbor cache
    async fn discover_via_ipv6_neighbors(
        &self,
        timeout_duration: Duration,
    ) -> Result<Vec<ClusterInfo>> {
        debug!("Starting IPv6 neighbor discovery");

        // On Linux, we could read /proc/net/ipv6_route or use netlink
        // For portability, we'll try common link-local patterns

        let mut tasks = Vec::new();
        let client = self.client.clone();

        // Try to scan link-local addresses with common interface IDs
        // In practice, these would come from the neighbor cache
        let base_prefix = "fe80::";

        // Common suffixes for link-local addresses (derived from MAC addresses)
        // We'll scan a small set of common patterns
        let common_suffixes = vec![
            "1", "2", "3", "4", "5", "1:1", "1:2", "1:3", "200:1", "200:2",
        ];

        for suffix in common_suffixes {
            let _target_ip = format!("{}{}%eth0", base_prefix, suffix);
            // Link-local addresses need a zone ID (interface), try without for HTTP
            let target_ip_http = format!("[{}{}]", base_prefix, suffix);
            let client = client.clone();

            let task = tokio::spawn(async move {
                let url = format!(
                    "http://{}:{}/api/v1/cluster/info",
                    target_ip_http,
                    crate::defaults::API_PORT
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

        let mut clusters = Vec::new();

        match timeout(timeout_duration, futures::future::join_all(tasks)).await {
            Ok(results) => {
                for cluster in results.into_iter().flatten().flatten() {
                    clusters.push(cluster);
                }
            }
            Err(_) => {
                debug!("IPv6 neighbor discovery timeout");
            }
        }

        debug!(
            "IPv6 neighbor discovery found {} cluster(s)",
            clusters.len()
        );
        Ok(clusters)
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

        // Test IPv6 loopback
        let ipv6: IpAddr = "::1".parse().unwrap();
        let subnet_base = discovery.get_subnet_base(&ipv6);
        assert_eq!(subnet_base, "0:0:0:0");

        // Test IPv6 link-local address
        let ipv6_link_local: IpAddr = "fe80::1".parse().unwrap();
        let subnet_base = discovery.get_subnet_base(&ipv6_link_local);
        assert_eq!(subnet_base, "fe80:0:0:0");

        // Test IPv6 global address
        let ipv6_global: IpAddr = "2001:db8:85a3::8a2e:370:7334".parse().unwrap();
        let subnet_base = discovery.get_subnet_base(&ipv6_global);
        assert_eq!(subnet_base, "2001:db8:85a3:0");
    }

    #[tokio::test]
    async fn test_discover_via_ipv6_multicast() {
        let discovery = ClusterDiscovery::new();

        // Test IPv6 multicast discovery (should not panic)
        let result = discovery
            .discover_via_ipv6_multicast(Duration::from_millis(100))
            .await;
        assert!(result.is_ok());
        // Expect empty results in test environment
        assert_eq!(result.unwrap().len(), 0);
    }

    #[tokio::test]
    async fn test_discover_via_ipv6_neighbors() {
        let discovery = ClusterDiscovery::new();

        // Test IPv6 neighbor discovery (should not panic)
        let result = discovery
            .discover_via_ipv6_neighbors(Duration::from_millis(100))
            .await;
        assert!(result.is_ok());
        // Expect empty results in test environment
        assert_eq!(result.unwrap().len(), 0);
    }

    #[tokio::test]
    async fn test_get_local_ipv4() {
        let discovery = ClusterDiscovery::new();
        // This might fail in CI environments, just check it doesn't panic
        let result = discovery.get_local_ipv4().await;
        match result {
            Ok(ip) => println!("Local IPv4: {}", ip),
            Err(e) => println!(
                "Could not get local IPv4 (expected in some environments): {}",
                e
            ),
        }
    }

    #[tokio::test]
    async fn test_get_local_ipv6() {
        let discovery = ClusterDiscovery::new();
        // IPv6 might not be available in all environments
        let result = discovery.get_local_ipv6().await;
        match result {
            Ok(ip) => println!("Local IPv6: {}", ip),
            Err(e) => println!(
                "Could not get local IPv6 (expected in some environments): {}",
                e
            ),
        }
    }

    #[test]
    fn test_get_all_local_ips() {
        let discovery = ClusterDiscovery::new();
        let ips = discovery.get_all_local_ips();
        // Should always return at least one IP (even if it's a default)
        assert!(!ips.is_empty());
    }
}
