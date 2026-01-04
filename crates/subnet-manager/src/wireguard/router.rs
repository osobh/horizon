//! Quality-aware routing for WireGuard mesh
//!
//! Integrates connection quality metrics with routing decisions to select
//! the best path through the mesh based on RTT, jitter, and packet loss.

use super::quality::{ConnectionQuality, QualityMetrics};
use super::subnet_aware::{SubnetAwareWireGuard, SubnetPeer, SubnetWireGuardError};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::net::{Ipv4Addr, SocketAddr};
use std::sync::Arc;
use std::time::{Duration, Instant};
use thiserror::Error;
use tokio::sync::RwLock;
use tracing::{debug, info, instrument, warn};

/// Errors for quality-aware routing
#[derive(Debug, Error)]
pub enum RouterError {
    #[error("No route available to {0}")]
    NoRoute(Ipv4Addr),

    #[error("No healthy routes available to {0}")]
    NoHealthyRoute(Ipv4Addr),

    #[error("Peer not found: {0}")]
    PeerNotFound(String),

    #[error("Probe failed: {0}")]
    ProbeFailed(String),

    #[error("Subnet error: {0}")]
    SubnetError(#[from] SubnetWireGuardError),
}

/// Route selection strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RouteStrategy {
    /// Select lowest latency route
    #[default]
    LowestLatency,
    /// Select route with best overall quality score
    BestQuality,
    /// Select route with lowest packet loss
    LowestLoss,
    /// Round-robin among healthy routes
    RoundRobin,
    /// Weighted selection based on quality scores
    Weighted,
}

/// Quality thresholds for route selection
#[derive(Debug, Clone)]
pub struct QualityThresholds {
    /// Maximum acceptable RTT
    pub max_rtt: Duration,
    /// Maximum acceptable jitter
    pub max_jitter: Duration,
    /// Maximum acceptable packet loss percentage
    pub max_packet_loss: f32,
    /// Minimum quality score (0-100)
    pub min_quality_score: u8,
}

impl Default for QualityThresholds {
    fn default() -> Self {
        Self {
            max_rtt: Duration::from_millis(500),
            max_jitter: Duration::from_millis(100),
            max_packet_loss: 10.0,
            min_quality_score: 30,
        }
    }
}

/// A possible route to a destination
#[derive(Debug, Clone)]
pub struct PeerRoute {
    /// Peer to route through
    pub peer: SubnetPeer,
    /// Quality metrics for this route
    pub quality: Option<QualityMetrics>,
    /// Whether this route is healthy
    pub is_healthy: bool,
    /// Route selection score (higher is better)
    pub score: u8,
}

/// Quality-aware router
///
/// Wraps `SubnetAwareWireGuard` and adds quality-based route selection.
pub struct QualityAwareRouter {
    /// Underlying WireGuard layer
    subnet_wg: Arc<SubnetAwareWireGuard>,
    /// Quality tracker per peer (keyed by public_key)
    peer_quality: DashMap<String, ConnectionQuality>,
    /// Route selection strategy
    strategy: RwLock<RouteStrategy>,
    /// Quality thresholds
    thresholds: RwLock<QualityThresholds>,
    /// Round-robin counter
    rr_counter: RwLock<usize>,
}

impl QualityAwareRouter {
    /// Create a new quality-aware router
    pub fn new(subnet_wg: Arc<SubnetAwareWireGuard>) -> Self {
        Self {
            subnet_wg,
            peer_quality: DashMap::new(),
            strategy: RwLock::new(RouteStrategy::default()),
            thresholds: RwLock::new(QualityThresholds::default()),
            rr_counter: RwLock::new(0),
        }
    }

    /// Set the route selection strategy
    pub async fn set_strategy(&self, strategy: RouteStrategy) {
        *self.strategy.write().await = strategy;
    }

    /// Set quality thresholds
    pub async fn set_thresholds(&self, thresholds: QualityThresholds) {
        *self.thresholds.write().await = thresholds;
    }

    /// Record a packet send for quality tracking
    pub fn record_send(&self, peer_public_key: &str) -> u64 {
        self.peer_quality
            .entry(peer_public_key.to_string())
            .or_default()
            .record_send()
    }

    /// Record a packet acknowledgment with RTT
    pub fn record_ack(&self, peer_public_key: &str, seq: u64, rtt: Duration) {
        if let Some(mut quality) = self.peer_quality.get_mut(peer_public_key) {
            quality.record_ack(seq, rtt);
        }
    }

    /// Get quality metrics for a peer
    pub fn get_peer_quality(&self, peer_public_key: &str) -> Option<QualityMetrics> {
        self.peer_quality.get(peer_public_key).map(|q| q.metrics())
    }

    /// Get all peer quality metrics
    pub fn get_all_quality(&self) -> Vec<(String, QualityMetrics)> {
        self.peer_quality
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().metrics()))
            .collect()
    }

    /// Check if a peer route is healthy based on thresholds
    pub async fn is_route_healthy(&self, peer_public_key: &str) -> bool {
        let thresholds = self.thresholds.read().await;

        if let Some(quality) = self.peer_quality.get(peer_public_key) {
            let metrics = quality.metrics();
            metrics.rtt < thresholds.max_rtt
                && metrics.jitter < thresholds.max_jitter
                && metrics.packet_loss < thresholds.max_packet_loss
                && metrics.score() >= thresholds.min_quality_score
        } else {
            // No quality data yet - assume healthy
            true
        }
    }

    /// Get all routes to a destination with quality information
    #[instrument(skip(self))]
    pub async fn get_routes(&self, dest_ip: Ipv4Addr) -> Vec<PeerRoute> {
        let thresholds = self.thresholds.read().await;

        // Get the direct route from address map
        if let Some((subnet_id, public_key)) = self.subnet_wg.route_to(dest_ip) {
            let peers = self.subnet_wg.get_subnet_peers(subnet_id);

            return peers
                .into_iter()
                .filter(|p| p.public_key == public_key)
                .map(|peer| {
                    let quality = self.peer_quality.get(&peer.public_key).map(|q| q.metrics());

                    let (is_healthy, score) = if let Some(ref q) = quality {
                        let healthy = q.rtt < thresholds.max_rtt
                            && q.jitter < thresholds.max_jitter
                            && q.packet_loss < thresholds.max_packet_loss;
                        (healthy, q.score())
                    } else {
                        (true, 50) // Default score for unknown quality
                    };

                    PeerRoute {
                        peer,
                        quality,
                        is_healthy,
                        score,
                    }
                })
                .collect();
        }

        Vec::new()
    }

    /// Select the best route to a destination based on quality
    #[instrument(skip(self))]
    pub async fn select_best_route(&self, dest_ip: Ipv4Addr) -> Result<PeerRoute, RouterError> {
        let routes = self.get_routes(dest_ip).await;

        if routes.is_empty() {
            return Err(RouterError::NoRoute(dest_ip));
        }

        let strategy = *self.strategy.read().await;

        // Filter to healthy routes first
        let healthy_routes: Vec<_> = routes.iter().filter(|r| r.is_healthy).collect();

        // If no healthy routes, use all routes but warn
        let candidates = if healthy_routes.is_empty() {
            warn!("No healthy routes to {}, using degraded routes", dest_ip);
            routes.iter().collect()
        } else {
            healthy_routes
        };

        let selected = match strategy {
            RouteStrategy::LowestLatency => candidates.into_iter().min_by_key(|r| {
                r.quality
                    .as_ref()
                    .map(|q| q.rtt)
                    .unwrap_or(Duration::from_secs(999))
            }),
            RouteStrategy::BestQuality => candidates.into_iter().max_by_key(|r| r.score),
            RouteStrategy::LowestLoss => candidates.into_iter().min_by(|a, b| {
                let a_loss = a.quality.as_ref().map(|q| q.packet_loss).unwrap_or(100.0);
                let b_loss = b.quality.as_ref().map(|q| q.packet_loss).unwrap_or(100.0);
                a_loss
                    .partial_cmp(&b_loss)
                    .unwrap_or(std::cmp::Ordering::Equal)
            }),
            RouteStrategy::RoundRobin => {
                if candidates.is_empty() {
                    None
                } else {
                    let mut counter = self.rr_counter.write().await;
                    let idx = *counter % candidates.len();
                    *counter = counter.wrapping_add(1);
                    candidates.into_iter().nth(idx)
                }
            }
            RouteStrategy::Weighted => {
                // Weighted random selection based on quality scores
                let total_score: u32 = candidates.iter().map(|r| r.score as u32).sum();
                if total_score == 0 {
                    candidates.into_iter().next()
                } else {
                    use rand::Rng;
                    let mut rng = rand::thread_rng();
                    let target = rng.gen_range(0..total_score);
                    let mut cumulative = 0u32;

                    candidates.into_iter().find(|r| {
                        cumulative += r.score as u32;
                        cumulative > target
                    })
                }
            }
        };

        selected
            .cloned()
            .ok_or(RouterError::NoHealthyRoute(dest_ip))
    }

    /// Get routing statistics
    pub async fn get_stats(&self) -> RouterStats {
        let all_quality = self.get_all_quality();
        let thresholds = self.thresholds.read().await;

        let total_peers = all_quality.len();
        let healthy_peers = all_quality
            .iter()
            .filter(|(_, q)| {
                q.rtt < thresholds.max_rtt
                    && q.jitter < thresholds.max_jitter
                    && q.packet_loss < thresholds.max_packet_loss
            })
            .count();

        let avg_rtt = if all_quality.is_empty() {
            Duration::ZERO
        } else {
            let total: Duration = all_quality.iter().map(|(_, q)| q.rtt).sum();
            total / all_quality.len() as u32
        };

        let avg_score = if all_quality.is_empty() {
            0
        } else {
            all_quality
                .iter()
                .map(|(_, q)| q.score() as u32)
                .sum::<u32>()
                / all_quality.len() as u32
        };

        RouterStats {
            total_peers,
            healthy_peers,
            degraded_peers: total_peers - healthy_peers,
            average_rtt: avg_rtt,
            average_score: avg_score as u8,
            strategy: *self.strategy.read().await,
        }
    }
}

/// Router statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterStats {
    /// Total number of tracked peers
    pub total_peers: usize,
    /// Number of healthy peers
    pub healthy_peers: usize,
    /// Number of degraded peers
    pub degraded_peers: usize,
    /// Average RTT across all peers
    #[serde(with = "humantime_serde")]
    pub average_rtt: Duration,
    /// Average quality score
    pub average_score: u8,
    /// Current route selection strategy
    pub strategy: RouteStrategy,
}

impl Serialize for RouteStrategy {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let s = match self {
            RouteStrategy::LowestLatency => "lowest_latency",
            RouteStrategy::BestQuality => "best_quality",
            RouteStrategy::LowestLoss => "lowest_loss",
            RouteStrategy::RoundRobin => "round_robin",
            RouteStrategy::Weighted => "weighted",
        };
        serializer.serialize_str(s)
    }
}

impl<'de> Deserialize<'de> for RouteStrategy {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        match s.as_str() {
            "lowest_latency" => Ok(RouteStrategy::LowestLatency),
            "best_quality" => Ok(RouteStrategy::BestQuality),
            "lowest_loss" => Ok(RouteStrategy::LowestLoss),
            "round_robin" => Ok(RouteStrategy::RoundRobin),
            "weighted" => Ok(RouteStrategy::Weighted),
            _ => Err(serde::de::Error::unknown_variant(
                &s,
                &[
                    "lowest_latency",
                    "best_quality",
                    "lowest_loss",
                    "round_robin",
                    "weighted",
                ],
            )),
        }
    }
}

/// Active quality prober
///
/// Sends UDP probes through the WireGuard tunnel to measure RTT,
/// jitter, and packet loss.
pub struct QualityProber {
    /// Router to update with quality metrics
    router: Arc<QualityAwareRouter>,
    /// Probe interval
    probe_interval: Duration,
    /// Probe timeout
    probe_timeout: Duration,
    /// UDP port for probing
    probe_port: u16,
    /// Running flag
    running: RwLock<bool>,
}

/// Probe configuration
#[derive(Debug, Clone)]
pub struct ProbeConfig {
    /// Interval between probe rounds
    pub interval: Duration,
    /// Timeout for individual probes
    pub timeout: Duration,
    /// UDP port for probes
    pub port: u16,
    /// Number of probes per peer
    pub probes_per_peer: usize,
}

impl Default for ProbeConfig {
    fn default() -> Self {
        Self {
            interval: Duration::from_secs(30),
            timeout: Duration::from_secs(5),
            port: 51821, // Default probe port (WG port + 1)
            probes_per_peer: 3,
        }
    }
}

/// Result of a probe
#[derive(Debug, Clone)]
pub struct ProbeResult {
    /// Peer public key
    pub peer_public_key: String,
    /// Whether the probe succeeded
    pub success: bool,
    /// Round-trip time (if successful)
    pub rtt: Option<Duration>,
    /// Error message (if failed)
    pub error: Option<String>,
}

impl QualityProber {
    /// Create a new quality prober
    pub fn new(router: Arc<QualityAwareRouter>, config: ProbeConfig) -> Self {
        Self {
            router,
            probe_interval: config.interval,
            probe_timeout: config.timeout,
            probe_port: config.port,
            running: RwLock::new(false),
        }
    }

    /// Check if the prober is running
    pub async fn is_running(&self) -> bool {
        *self.running.read().await
    }

    /// Start the probing loop (background task)
    pub async fn start(&self) {
        let mut running = self.running.write().await;
        if *running {
            return;
        }
        *running = true;
        info!("Quality prober started");
    }

    /// Stop the probing loop
    pub async fn stop(&self) {
        let mut running = self.running.write().await;
        *running = false;
        info!("Quality prober stopped");
    }

    /// Probe a single peer
    #[instrument(skip(self))]
    pub async fn probe_peer(&self, peer: &SubnetPeer) -> ProbeResult {
        let endpoint = match &peer.endpoint {
            Some(ep) => ep,
            None => {
                return ProbeResult {
                    peer_public_key: peer.public_key.clone(),
                    success: false,
                    rtt: None,
                    error: Some("No endpoint".to_string()),
                };
            }
        };

        // Create probe address (same host, probe port)
        let probe_addr = SocketAddr::new(endpoint.ip(), self.probe_port);

        // Record send
        let seq = self.router.record_send(&peer.public_key);

        // Send UDP probe
        let start = Instant::now();

        match self.send_probe(probe_addr, seq).await {
            Ok(response_seq) if response_seq == seq => {
                let rtt = start.elapsed();
                self.router.record_ack(&peer.public_key, seq, rtt);

                debug!(
                    peer = %peer.public_key,
                    rtt_ms = rtt.as_millis(),
                    "Probe successful"
                );

                ProbeResult {
                    peer_public_key: peer.public_key.clone(),
                    success: true,
                    rtt: Some(rtt),
                    error: None,
                }
            }
            Ok(_) => ProbeResult {
                peer_public_key: peer.public_key.clone(),
                success: false,
                rtt: None,
                error: Some("Sequence mismatch".to_string()),
            },
            Err(e) => {
                debug!(
                    peer = %peer.public_key,
                    error = %e,
                    "Probe failed"
                );

                ProbeResult {
                    peer_public_key: peer.public_key.clone(),
                    success: false,
                    rtt: None,
                    error: Some(e),
                }
            }
        }
    }

    /// Probe all peers
    pub async fn probe_all(&self) -> Vec<ProbeResult> {
        let peers = self.router.subnet_wg.get_all_peers();
        let mut results = Vec::with_capacity(peers.len());

        for peer in peers {
            let result = self.probe_peer(&peer).await;
            results.push(result);
        }

        results
    }

    /// Send a UDP probe and wait for response
    async fn send_probe(&self, addr: SocketAddr, seq: u64) -> Result<u64, String> {
        // Create UDP socket
        let socket = tokio::net::UdpSocket::bind("0.0.0.0:0")
            .await
            .map_err(|e| e.to_string())?;

        // Create probe packet (8-byte sequence number)
        let packet = seq.to_be_bytes();

        // Send probe
        socket
            .send_to(&packet, addr)
            .await
            .map_err(|e| e.to_string())?;

        // Wait for response with timeout
        let mut buf = [0u8; 8];

        match tokio::time::timeout(self.probe_timeout, socket.recv_from(&mut buf)).await {
            Ok(Ok((8, _))) => {
                let response_seq = u64::from_be_bytes(buf);
                Ok(response_seq)
            }
            Ok(Ok((n, _))) => Err(format!("Invalid response size: {}", n)),
            Ok(Err(e)) => Err(e.to_string()),
            Err(_) => Err("Timeout".to_string()),
        }
    }

    /// Run probing loop (call from background task)
    pub async fn run_probe_loop(&self) {
        while *self.running.read().await {
            let results = self.probe_all().await;

            let successful = results.iter().filter(|r| r.success).count();
            let total = results.len();

            debug!(
                successful = successful,
                total = total,
                "Probe round complete"
            );

            tokio::time::sleep(self.probe_interval).await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::Ipv4Addr;
    use uuid::Uuid;

    fn create_test_router() -> QualityAwareRouter {
        let wg = Arc::new(SubnetAwareWireGuard::new());
        QualityAwareRouter::new(wg)
    }

    #[tokio::test]
    async fn test_router_creation() {
        let router = create_test_router();
        let stats = router.get_stats().await;

        assert_eq!(stats.total_peers, 0);
        assert_eq!(stats.healthy_peers, 0);
    }

    #[tokio::test]
    async fn test_quality_tracking() {
        let router = create_test_router();

        // Record some packets
        let seq1 = router.record_send("peer1");
        let seq2 = router.record_send("peer1");

        router.record_ack("peer1", seq1, Duration::from_millis(20));
        router.record_ack("peer1", seq2, Duration::from_millis(25));

        let quality = router.get_peer_quality("peer1");
        assert!(quality.is_some());

        let metrics = quality.unwrap();
        assert!(metrics.rtt > Duration::ZERO);
        assert_eq!(metrics.samples, 2);
    }

    #[tokio::test]
    async fn test_route_health_check() {
        let router = create_test_router();

        // Unknown peer should be considered healthy
        assert!(router.is_route_healthy("unknown").await);

        // Record good quality
        for _ in 0..5 {
            let seq = router.record_send("good_peer");
            router.record_ack("good_peer", seq, Duration::from_millis(20));
        }
        assert!(router.is_route_healthy("good_peer").await);

        // Record poor quality (high latency)
        for _ in 0..5 {
            let seq = router.record_send("slow_peer");
            router.record_ack("slow_peer", seq, Duration::from_millis(600));
        }
        assert!(!router.is_route_healthy("slow_peer").await);
    }

    #[tokio::test]
    async fn test_strategy_change() {
        let router = create_test_router();

        router.set_strategy(RouteStrategy::LowestLatency).await;
        let stats = router.get_stats().await;
        assert_eq!(stats.strategy, RouteStrategy::LowestLatency);

        router.set_strategy(RouteStrategy::BestQuality).await;
        let stats = router.get_stats().await;
        assert_eq!(stats.strategy, RouteStrategy::BestQuality);
    }

    #[tokio::test]
    async fn test_threshold_configuration() {
        let router = create_test_router();

        let custom_thresholds = QualityThresholds {
            max_rtt: Duration::from_millis(100),
            max_jitter: Duration::from_millis(20),
            max_packet_loss: 5.0,
            min_quality_score: 50,
        };

        router.set_thresholds(custom_thresholds.clone()).await;

        // Record metrics that exceed the stricter threshold
        for _ in 0..5 {
            let seq = router.record_send("test_peer");
            router.record_ack("test_peer", seq, Duration::from_millis(150));
        }

        // Should be unhealthy with stricter thresholds
        assert!(!router.is_route_healthy("test_peer").await);
    }

    #[test]
    fn test_probe_config_default() {
        let config = ProbeConfig::default();

        assert_eq!(config.interval, Duration::from_secs(30));
        assert_eq!(config.timeout, Duration::from_secs(5));
        assert_eq!(config.port, 51821);
        assert_eq!(config.probes_per_peer, 3);
    }

    #[tokio::test]
    async fn test_prober_lifecycle() {
        let router = Arc::new(create_test_router());
        let prober = QualityProber::new(router, ProbeConfig::default());

        assert!(!prober.is_running().await);

        prober.start().await;
        assert!(prober.is_running().await);

        prober.stop().await;
        assert!(!prober.is_running().await);
    }

    #[tokio::test]
    async fn test_probe_result_no_endpoint() {
        let router = Arc::new(create_test_router());
        let prober = QualityProber::new(router, ProbeConfig::default());

        // Peer without endpoint
        let peer = SubnetPeer::new(
            Uuid::new_v4(),
            "test_key",
            Ipv4Addr::new(10, 0, 0, 1),
            Uuid::new_v4(),
        );

        let result = prober.probe_peer(&peer).await;
        assert!(!result.success);
        assert!(result.error.is_some());
    }

    #[test]
    fn test_route_strategy_serialization() {
        let strategy = RouteStrategy::BestQuality;
        let json = serde_json::to_string(&strategy).unwrap();
        assert_eq!(json, "\"best_quality\"");

        let parsed: RouteStrategy = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, RouteStrategy::BestQuality);
    }
}
