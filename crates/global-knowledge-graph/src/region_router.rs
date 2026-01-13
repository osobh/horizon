//! Intelligent region routing

use crate::error::{GlobalKnowledgeGraphError, GlobalKnowledgeGraphResult};
use async_trait::async_trait;
use dashmap::DashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tokio::time::interval;

/// Region router configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionRouterConfig {
    /// Enable latency-based routing
    pub enable_latency_routing: bool,
    /// Enable compliance-aware routing
    pub enable_compliance_routing: bool,
    /// Health check interval
    pub health_check_interval: Duration,
    /// Latency measurement interval
    pub latency_measurement_interval: Duration,
    /// Maximum acceptable latency in milliseconds
    pub max_acceptable_latency_ms: u64,
    /// Failover threshold (number of failures)
    pub failover_threshold: u32,
    /// Load balancing strategy
    pub load_balancing_strategy: LoadBalancingStrategy,
}

impl Default for RegionRouterConfig {
    fn default() -> Self {
        Self {
            enable_latency_routing: true,
            enable_compliance_routing: true,
            health_check_interval: Duration::from_secs(30),
            latency_measurement_interval: Duration::from_secs(60),
            max_acceptable_latency_ms: 100,
            failover_threshold: 3,
            load_balancing_strategy: LoadBalancingStrategy::WeightedRoundRobin,
        }
    }
}

/// Load balancing strategy
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum LoadBalancingStrategy {
    /// Round robin
    RoundRobin,
    /// Weighted round robin based on capacity
    WeightedRoundRobin,
    /// Least connections
    LeastConnections,
    /// Latency-based
    LatencyBased,
    /// Random selection
    Random,
}

/// Region information
#[derive(Debug, Clone)]
pub struct RegionInfo {
    /// Region name
    pub name: String,
    /// Region endpoint
    pub endpoint: String,
    /// Geographic location
    pub location: GeoLocation,
    /// Supported compliance regulations
    pub compliance_regulations: HashSet<String>,
    /// Current capacity (0-100)
    pub capacity: u8,
    /// Is region healthy
    pub is_healthy: bool,
    /// Last health check
    pub last_health_check: Instant,
    /// Current latency in milliseconds
    pub current_latency_ms: u64,
    /// Average latency in milliseconds
    pub avg_latency_ms: f64,
    /// Active connections
    pub active_connections: u32,
    /// Consecutive failures
    pub consecutive_failures: u32,
}

/// Geographic location
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeoLocation {
    /// Latitude
    pub latitude: f64,
    /// Longitude
    pub longitude: f64,
    /// Region code (e.g., "US-EAST", "EU-WEST")
    pub region_code: String,
}

/// Routing decision
#[derive(Debug, Clone)]
pub struct RoutingDecision {
    /// Selected region
    pub region: String,
    /// Routing reason
    pub reason: RoutingReason,
    /// Estimated latency
    pub estimated_latency_ms: u64,
    /// Alternative regions
    pub alternatives: Vec<String>,
}

/// Routing reason
#[derive(Debug, Clone, PartialEq)]
pub enum RoutingReason {
    /// Lowest latency
    LowestLatency,
    /// Compliance requirement
    ComplianceRequired,
    /// Load balancing
    LoadBalancing,
    /// Failover
    Failover,
    /// Capacity-based
    CapacityBased,
}

/// Region health checker trait
#[async_trait]
pub trait RegionHealthChecker: Send + Sync {
    /// Check if region is healthy
    async fn check_health(&self, region: &str) -> GlobalKnowledgeGraphResult<bool>;

    /// Measure latency to region
    async fn measure_latency(&self, region: &str) -> GlobalKnowledgeGraphResult<u64>;

    /// Get current capacity
    async fn get_capacity(&self, region: &str) -> GlobalKnowledgeGraphResult<u8>;
}

/// Mock health checker for testing
#[cfg(test)]
pub struct MockHealthChecker {
    healthy_regions: Arc<RwLock<HashSet<String>>>,
    latencies: Arc<DashMap<String, u64>>,
    capacities: Arc<DashMap<String, u8>>,
}

#[cfg(test)]
impl MockHealthChecker {
    pub fn new() -> Self {
        Self {
            healthy_regions: Arc::new(RwLock::new(HashSet::new())),
            latencies: Arc::new(DashMap::new()),
            capacities: Arc::new(DashMap::new()),
        }
    }

    pub fn set_healthy(&self, region: &str, healthy: bool) {
        if healthy {
            self.healthy_regions.write().insert(region.to_string());
        } else {
            self.healthy_regions.write().remove(region);
        }
    }

    pub fn set_latency(&self, region: &str, latency: u64) {
        self.latencies.insert(region.to_string(), latency);
    }

    pub fn set_capacity(&self, region: &str, capacity: u8) {
        self.capacities.insert(region.to_string(), capacity);
    }
}

#[cfg(test)]
#[async_trait]
impl RegionHealthChecker for MockHealthChecker {
    async fn check_health(&self, region: &str) -> GlobalKnowledgeGraphResult<bool> {
        Ok(self.healthy_regions.read().contains(region))
    }

    async fn measure_latency(&self, region: &str) -> GlobalKnowledgeGraphResult<u64> {
        Ok(self.latencies.get(region).map(|v| *v).unwrap_or(50))
    }

    async fn get_capacity(&self, region: &str) -> GlobalKnowledgeGraphResult<u8> {
        Ok(self.capacities.get(region).map(|v| *v).unwrap_or(50))
    }
}

/// Intelligent region router
pub struct RegionRouter {
    config: Arc<RegionRouterConfig>,
    regions: Arc<DashMap<String, RegionInfo>>,
    health_checker: Arc<dyn RegionHealthChecker>,
    routing_metrics: Arc<DashMap<String, RoutingMetrics>>,
    round_robin_counter: Arc<RwLock<usize>>,
    compliance_cache: Arc<DashMap<String, HashSet<String>>>,
    shutdown_tx: mpsc::Sender<()>,
    shutdown_rx: Arc<RwLock<Option<mpsc::Receiver<()>>>>,
}

/// Routing metrics
#[derive(Debug, Clone, Default)]
pub struct RoutingMetrics {
    /// Total requests routed
    pub total_requests: u64,
    /// Successful routings
    pub successful_routings: u64,
    /// Failed routings
    pub failed_routings: u64,
    /// Average routing time
    pub avg_routing_time_ms: f64,
}

impl RegionRouter {
    /// Create new region router
    pub fn new(config: RegionRouterConfig, health_checker: Arc<dyn RegionHealthChecker>) -> Self {
        let (shutdown_tx, shutdown_rx) = mpsc::channel(1);

        Self {
            config: Arc::new(config),
            regions: Arc::new(DashMap::new()),
            health_checker,
            routing_metrics: Arc::new(DashMap::new()),
            round_robin_counter: Arc::new(RwLock::new(0)),
            compliance_cache: Arc::new(DashMap::new()),
            shutdown_tx,
            shutdown_rx: Arc::new(RwLock::new(Some(shutdown_rx))),
        }
    }

    /// Register region
    pub fn register_region(&self, region: RegionInfo) {
        self.regions.insert(region.name.clone(), region.clone());
        self.routing_metrics
            .insert(region.name.clone(), RoutingMetrics::default());
    }

    /// Start region router
    pub async fn start(&self) {
        let config = self.config.clone();
        let regions = self.regions.clone();
        let health_checker = self.health_checker.clone();

        // Take ownership of the receiver
        let mut shutdown_rx = self
            .shutdown_rx
            .write()
            .take()
            .expect("start called multiple times");

        tokio::spawn(async move {
            let mut health_interval = interval(config.health_check_interval);
            let mut latency_interval = interval(config.latency_measurement_interval);

            loop {
                tokio::select! {
                    _ = health_interval.tick() => {
                        Self::perform_health_checks(&regions, &health_checker).await;
                    }
                    _ = latency_interval.tick() => {
                        if config.enable_latency_routing {
                            Self::measure_latencies(&regions, &health_checker).await;
                        }
                    }
                    _ = shutdown_rx.recv() => {
                        break;
                    }
                }
            }
        });
    }

    /// Stop region router
    pub async fn stop(&self) {
        let _ = self.shutdown_tx.send(()).await;
    }

    /// Route request to optimal region
    pub async fn route_request(
        &self,
        source_location: Option<GeoLocation>,
        compliance_requirements: Option<HashSet<String>>,
    ) -> GlobalKnowledgeGraphResult<RoutingDecision> {
        let start = Instant::now();

        // Get available regions
        let available_regions = self
            .get_available_regions(compliance_requirements.as_ref())
            .await?;

        if available_regions.is_empty() {
            return Err(GlobalKnowledgeGraphError::RegionUnavailable {
                region: "any".to_string(),
                reason: "No available regions match requirements".to_string(),
            });
        }

        // Select region based on strategy
        let decision = match self.config.load_balancing_strategy {
            LoadBalancingStrategy::RoundRobin => self.round_robin_select(&available_regions),
            LoadBalancingStrategy::WeightedRoundRobin => {
                self.weighted_round_robin_select(&available_regions)
            }
            LoadBalancingStrategy::LeastConnections => {
                self.least_connections_select(&available_regions)
            }
            LoadBalancingStrategy::LatencyBased => {
                self.latency_based_select(&available_regions, source_location)
                    .await
            }
            LoadBalancingStrategy::Random => self.random_select(&available_regions),
        };

        // Update metrics
        let routing_time = start.elapsed().as_millis() as u64;
        self.update_routing_metrics(&decision.region, routing_time, true);

        Ok(decision)
    }

    /// Get available regions
    async fn get_available_regions(
        &self,
        compliance_requirements: Option<&HashSet<String>>,
    ) -> GlobalKnowledgeGraphResult<Vec<RegionInfo>> {
        let mut available = Vec::new();

        for region_ref in self.regions.iter() {
            let region = region_ref.value();

            // Check health
            if !region.is_healthy {
                continue;
            }

            // Check capacity
            if region.capacity < 10 {
                continue;
            }

            // Check compliance if required
            if self.config.enable_compliance_routing {
                if let Some(requirements) = compliance_requirements {
                    if !requirements.is_subset(&region.compliance_regulations) {
                        continue;
                    }
                }
            }

            available.push(region.clone());
        }

        Ok(available)
    }

    /// Round robin selection
    fn round_robin_select(&self, regions: &[RegionInfo]) -> RoutingDecision {
        let mut counter = self.round_robin_counter.write();
        let index = *counter % regions.len();
        *counter = (*counter + 1) % regions.len();

        let selected = &regions[index];

        RoutingDecision {
            region: selected.name.clone(),
            reason: RoutingReason::LoadBalancing,
            estimated_latency_ms: selected.avg_latency_ms as u64,
            alternatives: regions
                .iter()
                .filter(|r| r.name != selected.name)
                .map(|r| r.name.clone())
                .collect(),
        }
    }

    /// Weighted round robin selection
    fn weighted_round_robin_select(&self, regions: &[RegionInfo]) -> RoutingDecision {
        // Calculate total weight based on capacity
        let total_weight: u32 = regions.iter().map(|r| r.capacity as u32).sum();

        if total_weight == 0 {
            return self.round_robin_select(regions);
        }

        // Generate random number and select based on weight
        let random = rand::random::<u32>() % total_weight;
        let mut cumulative = 0;

        for region in regions {
            cumulative += region.capacity as u32;
            if random < cumulative {
                return RoutingDecision {
                    region: region.name.clone(),
                    reason: RoutingReason::CapacityBased,
                    estimated_latency_ms: region.avg_latency_ms as u64,
                    alternatives: regions
                        .iter()
                        .filter(|r| r.name != region.name)
                        .map(|r| r.name.clone())
                        .collect(),
                };
            }
        }

        // Fallback (shouldn't reach here)
        self.round_robin_select(regions)
    }

    /// Least connections selection
    fn least_connections_select(&self, regions: &[RegionInfo]) -> RoutingDecision {
        let Some(selected) = regions.iter().min_by_key(|r| r.active_connections) else {
            return self.round_robin_select(regions);
        };

        RoutingDecision {
            region: selected.name.clone(),
            reason: RoutingReason::LoadBalancing,
            estimated_latency_ms: selected.avg_latency_ms as u64,
            alternatives: regions
                .iter()
                .filter(|r| r.name != selected.name)
                .map(|r| r.name.clone())
                .collect(),
        }
    }

    /// Latency-based selection
    async fn latency_based_select(
        &self,
        regions: &[RegionInfo],
        source_location: Option<GeoLocation>,
    ) -> RoutingDecision {
        let mut best_region = &regions[0];
        let mut best_latency = u64::MAX;

        for region in regions {
            let mut estimated_latency = region.current_latency_ms;

            // Add geographic distance penalty if source location is provided
            if let Some(ref source) = source_location {
                let distance_penalty = self.calculate_distance_penalty(source, &region.location);
                estimated_latency += distance_penalty;
            }

            if estimated_latency < best_latency {
                best_latency = estimated_latency;
                best_region = region;
            }
        }

        RoutingDecision {
            region: best_region.name.clone(),
            reason: RoutingReason::LowestLatency,
            estimated_latency_ms: best_latency,
            alternatives: regions
                .iter()
                .filter(|r| r.name != best_region.name)
                .map(|r| r.name.clone())
                .collect(),
        }
    }

    /// Random selection
    fn random_select(&self, regions: &[RegionInfo]) -> RoutingDecision {
        let index = rand::random::<usize>() % regions.len();
        let selected = &regions[index];

        RoutingDecision {
            region: selected.name.clone(),
            reason: RoutingReason::LoadBalancing,
            estimated_latency_ms: selected.avg_latency_ms as u64,
            alternatives: regions
                .iter()
                .filter(|r| r.name != selected.name)
                .map(|r| r.name.clone())
                .collect(),
        }
    }

    /// Calculate distance penalty based on geographic distance
    fn calculate_distance_penalty(&self, source: &GeoLocation, target: &GeoLocation) -> u64 {
        // Simplified haversine distance calculation
        let lat1 = source.latitude.to_radians();
        let lat2 = target.latitude.to_radians();
        let lon1 = source.longitude.to_radians();
        let lon2 = target.longitude.to_radians();

        let dlat = lat2 - lat1;
        let dlon = lon2 - lon1;

        let a = (dlat / 2.0).sin().powi(2) + lat1.cos() * lat2.cos() * (dlon / 2.0).sin().powi(2);
        let c = 2.0 * a.sqrt().asin();
        let distance_km = 6371.0 * c; // Earth's radius in km

        // Approximate latency penalty: 1ms per 100km
        (distance_km / 100.0) as u64
    }

    /// Handle failover
    pub async fn handle_failover(
        &self,
        failed_region: &str,
    ) -> GlobalKnowledgeGraphResult<RoutingDecision> {
        // Mark region as unhealthy
        if let Some(mut region) = self.regions.get_mut(failed_region) {
            region.is_healthy = false;
            region.consecutive_failures += 1;

            if region.consecutive_failures >= self.config.failover_threshold {
                tracing::error!(
                    "Region {} marked as failed after {} consecutive failures",
                    failed_region,
                    region.consecutive_failures
                );
            }
        }

        // Route to alternative region
        let available_regions = self.get_available_regions(None).await?;

        if available_regions.is_empty() {
            return Err(GlobalKnowledgeGraphError::RegionUnavailable {
                region: "any".to_string(),
                reason: "No healthy regions available for failover".to_string(),
            });
        }

        let mut decision = self.latency_based_select(&available_regions, None).await;
        decision.reason = RoutingReason::Failover;

        Ok(decision)
    }

    /// Update region info
    pub fn update_region_info(&self, region_name: &str, update_fn: impl FnOnce(&mut RegionInfo)) {
        if let Some(mut region) = self.regions.get_mut(region_name) {
            update_fn(&mut region);
        }
    }

    /// Get region info
    pub fn get_region_info(&self, region_name: &str) -> Option<RegionInfo> {
        self.regions.get(region_name).map(|r| r.clone())
    }

    /// Get all regions
    pub fn get_all_regions(&self) -> Vec<RegionInfo> {
        self.regions
            .iter()
            .map(|entry| entry.value().clone())
            .collect()
    }

    /// Update routing metrics
    fn update_routing_metrics(&self, region: &str, routing_time_ms: u64, success: bool) {
        self.routing_metrics.alter(region, |_, mut metrics| {
            metrics.total_requests += 1;

            if success {
                metrics.successful_routings += 1;
            } else {
                metrics.failed_routings += 1;
            }

            // Update average routing time
            let new_avg = if metrics.total_requests == 1 {
                routing_time_ms as f64
            } else {
                (metrics.avg_routing_time_ms * (metrics.total_requests - 1) as f64
                    + routing_time_ms as f64)
                    / metrics.total_requests as f64
            };
            metrics.avg_routing_time_ms = new_avg;

            metrics
        });
    }

    /// Get routing metrics
    pub fn get_routing_metrics(&self, region: &str) -> Option<RoutingMetrics> {
        self.routing_metrics.get(region).map(|m| m.clone())
    }

    /// Perform health checks
    async fn perform_health_checks(
        regions: &Arc<DashMap<String, RegionInfo>>,
        health_checker: &Arc<dyn RegionHealthChecker>,
    ) {
        for mut region_ref in regions.iter_mut() {
            let region = region_ref.value_mut();
            let region_name = region.name.clone();

            match health_checker.check_health(&region_name).await {
                Ok(healthy) => {
                    region.is_healthy = healthy;
                    region.last_health_check = Instant::now();

                    if healthy {
                        region.consecutive_failures = 0;
                    }
                }
                Err(e) => {
                    tracing::error!("Health check failed for region {}: {}", region_name, e);
                    region.consecutive_failures += 1;
                }
            }

            // Get current capacity
            if let Ok(capacity) = health_checker.get_capacity(&region_name).await {
                region.capacity = capacity;
            }
        }
    }

    /// Measure latencies
    async fn measure_latencies(
        regions: &Arc<DashMap<String, RegionInfo>>,
        health_checker: &Arc<dyn RegionHealthChecker>,
    ) {
        for mut region_ref in regions.iter_mut() {
            let region = region_ref.value_mut();
            let region_name = region.name.clone();

            if let Ok(latency) = health_checker.measure_latency(&region_name).await {
                region.current_latency_ms = latency;

                // Update average latency
                if region.avg_latency_ms == 0.0 {
                    region.avg_latency_ms = latency as f64;
                } else {
                    region.avg_latency_ms = region.avg_latency_ms * 0.9 + latency as f64 * 0.1;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_region(name: &str, lat: f64, lon: f64) -> RegionInfo {
        let mut compliance = HashSet::new();
        compliance.insert("GDPR".to_string());

        RegionInfo {
            name: name.to_string(),
            endpoint: format!("https://{}.example.com", name),
            location: GeoLocation {
                latitude: lat,
                longitude: lon,
                region_code: name.to_uppercase(),
            },
            compliance_regulations: compliance,
            capacity: 80,
            is_healthy: true,
            last_health_check: Instant::now(),
            current_latency_ms: 50,
            avg_latency_ms: 50.0,
            active_connections: 10,
            consecutive_failures: 0,
        }
    }

    #[tokio::test]
    async fn test_region_router_creation() {
        let config = RegionRouterConfig::default();
        let health_checker = Arc::new(MockHealthChecker::new());
        let router = RegionRouter::new(config, health_checker);

        assert_eq!(router.regions.len(), 0);
    }

    #[tokio::test]
    async fn test_register_region() {
        let health_checker = Arc::new(MockHealthChecker::new());
        let router = RegionRouter::new(RegionRouterConfig::default(), health_checker);

        let region = create_test_region("us-east-1", 40.7128, -74.0060);
        router.register_region(region);

        assert_eq!(router.regions.len(), 1);
        assert_eq!(router.routing_metrics.len(), 1);
    }

    #[tokio::test]
    async fn test_route_request_single_region() {
        let health_checker = Arc::new(MockHealthChecker::new());
        health_checker.set_healthy("us-east-1", true);

        let router = RegionRouter::new(RegionRouterConfig::default(), health_checker);
        let region = create_test_region("us-east-1", 40.7128, -74.0060);
        router.register_region(region);

        let decision = router.route_request(None, None).await?;
        assert_eq!(decision.region, "us-east-1");
    }

    #[tokio::test]
    async fn test_route_request_no_available_regions() {
        let health_checker = Arc::new(MockHealthChecker::new());
        let router = RegionRouter::new(RegionRouterConfig::default(), health_checker);

        let result = router.route_request(None, None).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_round_robin_routing() {
        let config = RegionRouterConfig {
            load_balancing_strategy: LoadBalancingStrategy::RoundRobin,
            ..Default::default()
        };

        let health_checker = Arc::new(MockHealthChecker::new());
        health_checker.set_healthy("us-east-1", true);
        health_checker.set_healthy("eu-west-1", true);

        let router = RegionRouter::new(config, health_checker);

        router.register_region(create_test_region("us-east-1", 40.7128, -74.0060));
        router.register_region(create_test_region("eu-west-1", 51.5074, -0.1278));

        let decision1 = router.route_request(None, None).await.unwrap();
        let decision2 = router.route_request(None, None).await.unwrap();

        assert_ne!(decision1.region, decision2.region);
    }

    #[tokio::test]
    async fn test_latency_based_routing() {
        let config = RegionRouterConfig {
            load_balancing_strategy: LoadBalancingStrategy::LatencyBased,
            ..Default::default()
        };

        let health_checker = Arc::new(MockHealthChecker::new());
        health_checker.set_healthy("us-east-1", true);
        health_checker.set_healthy("eu-west-1", true);
        health_checker.set_latency("us-east-1", 20);
        health_checker.set_latency("eu-west-1", 100);

        let router = RegionRouter::new(config, health_checker);

        let mut region1 = create_test_region("us-east-1", 40.7128, -74.0060);
        region1.current_latency_ms = 20;
        region1.avg_latency_ms = 20.0;

        let mut region2 = create_test_region("eu-west-1", 51.5074, -0.1278);
        region2.current_latency_ms = 100;
        region2.avg_latency_ms = 100.0;

        router.register_region(region1);
        router.register_region(region2);

        let decision = router.route_request(None, None).await.unwrap();
        assert_eq!(decision.region, "us-east-1");
        assert_eq!(decision.reason, RoutingReason::LowestLatency);
    }

    #[tokio::test]
    async fn test_compliance_aware_routing() {
        let health_checker = Arc::new(MockHealthChecker::new());
        health_checker.set_healthy("us-east-1", true);
        health_checker.set_healthy("eu-west-1", true);

        let router = RegionRouter::new(RegionRouterConfig::default(), health_checker);

        let mut region1 = create_test_region("us-east-1", 40.7128, -74.0060);
        region1.compliance_regulations.clear();
        region1.compliance_regulations.insert("SOC2".to_string());

        let mut region2 = create_test_region("eu-west-1", 51.5074, -0.1278);
        region2.compliance_regulations.clear();
        region2.compliance_regulations.insert("GDPR".to_string());

        router.register_region(region1);
        router.register_region(region2);

        let mut requirements = HashSet::new();
        requirements.insert("GDPR".to_string());

        let decision = router
            .route_request(None, Some(requirements))
            .await
            .unwrap();
        assert_eq!(decision.region, "eu-west-1");
    }

    #[tokio::test]
    async fn test_capacity_based_routing() {
        let config = RegionRouterConfig {
            load_balancing_strategy: LoadBalancingStrategy::WeightedRoundRobin,
            ..Default::default()
        };

        let health_checker = Arc::new(MockHealthChecker::new());
        health_checker.set_healthy("us-east-1", true);
        health_checker.set_healthy("eu-west-1", true);
        health_checker.set_capacity("us-east-1", 90);
        health_checker.set_capacity("eu-west-1", 10);

        let router = RegionRouter::new(config, health_checker);

        let mut region1 = create_test_region("us-east-1", 40.7128, -74.0060);
        region1.capacity = 90;

        let mut region2 = create_test_region("eu-west-1", 51.5074, -0.1278);
        region2.capacity = 10;

        router.register_region(region1);
        router.register_region(region2);

        // With weighted round robin, us-east-1 should be selected more often
        let mut us_count = 0;
        for _ in 0..10 {
            let decision = router.route_request(None, None).await.unwrap();
            if decision.region == "us-east-1" {
                us_count += 1;
            }
        }

        assert!(us_count > 5); // Should be selected more than half the time
    }

    #[tokio::test]
    async fn test_least_connections_routing() {
        let config = RegionRouterConfig {
            load_balancing_strategy: LoadBalancingStrategy::LeastConnections,
            ..Default::default()
        };

        let health_checker = Arc::new(MockHealthChecker::new());
        health_checker.set_healthy("us-east-1", true);
        health_checker.set_healthy("eu-west-1", true);

        let router = RegionRouter::new(config, health_checker);

        let mut region1 = create_test_region("us-east-1", 40.7128, -74.0060);
        region1.active_connections = 50;

        let mut region2 = create_test_region("eu-west-1", 51.5074, -0.1278);
        region2.active_connections = 10;

        router.register_region(region1);
        router.register_region(region2);

        let decision = router.route_request(None, None).await.unwrap();
        assert_eq!(decision.region, "eu-west-1");
    }

    #[tokio::test]
    async fn test_failover() {
        let health_checker = Arc::new(MockHealthChecker::new());
        health_checker.set_healthy("us-east-1", true);
        health_checker.set_healthy("eu-west-1", true);

        let router = RegionRouter::new(RegionRouterConfig::default(), health_checker);

        router.register_region(create_test_region("us-east-1", 40.7128, -74.0060));
        router.register_region(create_test_region("eu-west-1", 51.5074, -0.1278));

        let decision = router.handle_failover("us-east-1").await?;
        assert_eq!(decision.region, "eu-west-1");
        assert_eq!(decision.reason, RoutingReason::Failover);

        // Check that us-east-1 is marked as unhealthy
        let region_info = router.get_region_info("us-east-1").unwrap();
        assert!(!region_info.is_healthy);
    }

    #[tokio::test]
    async fn test_update_region_info() {
        let health_checker = Arc::new(MockHealthChecker::new());
        let router = RegionRouter::new(RegionRouterConfig::default(), health_checker);

        router.register_region(create_test_region("us-east-1", 40.7128, -74.0060));

        router.update_region_info("us-east-1", |info| {
            info.capacity = 50;
            info.active_connections = 25;
        });

        let info = router.get_region_info("us-east-1").unwrap();
        assert_eq!(info.capacity, 50);
        assert_eq!(info.active_connections, 25);
    }

    #[tokio::test]
    async fn test_routing_metrics() {
        let health_checker = Arc::new(MockHealthChecker::new());
        health_checker.set_healthy("us-east-1", true);

        let router = RegionRouter::new(RegionRouterConfig::default(), health_checker);
        router.register_region(create_test_region("us-east-1", 40.7128, -74.0060));

        // Route multiple requests
        for _ in 0..5 {
            router.route_request(None, None).await?;
        }

        let metrics = router.get_routing_metrics("us-east-1").unwrap();
        assert_eq!(metrics.total_requests, 5);
        assert_eq!(metrics.successful_routings, 5);
        assert_eq!(metrics.failed_routings, 0);
    }

    #[tokio::test]
    async fn test_geographic_distance_calculation() {
        let health_checker = Arc::new(MockHealthChecker::new());
        let router = RegionRouter::new(RegionRouterConfig::default(), health_checker);

        let new_york = GeoLocation {
            latitude: 40.7128,
            longitude: -74.0060,
            region_code: "US-EAST".to_string(),
        };

        let london = GeoLocation {
            latitude: 51.5074,
            longitude: -0.1278,
            region_code: "EU-WEST".to_string(),
        };

        let penalty = router.calculate_distance_penalty(&new_york, &london);
        assert!(penalty > 0); // Should have some penalty for transatlantic distance
    }

    #[tokio::test]
    async fn test_start_stop_router() {
        let health_checker = Arc::new(MockHealthChecker::new());
        let router = RegionRouter::new(RegionRouterConfig::default(), health_checker);

        router.start().await;
        tokio::time::sleep(Duration::from_millis(100)).await;

        router.stop().await;
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    #[tokio::test]
    async fn test_unhealthy_region_exclusion() {
        let health_checker = Arc::new(MockHealthChecker::new());
        health_checker.set_healthy("us-east-1", false);
        health_checker.set_healthy("eu-west-1", true);

        let router = RegionRouter::new(RegionRouterConfig::default(), health_checker);

        let mut region1 = create_test_region("us-east-1", 40.7128, -74.0060);
        region1.is_healthy = false;

        let region2 = create_test_region("eu-west-1", 51.5074, -0.1278);

        router.register_region(region1);
        router.register_region(region2);

        let decision = router.route_request(None, None).await.unwrap();
        assert_eq!(decision.region, "eu-west-1");
    }

    #[tokio::test]
    async fn test_low_capacity_exclusion() {
        let health_checker = Arc::new(MockHealthChecker::new());
        health_checker.set_healthy("us-east-1", true);
        health_checker.set_healthy("eu-west-1", true);

        let router = RegionRouter::new(RegionRouterConfig::default(), health_checker);

        let mut region1 = create_test_region("us-east-1", 40.7128, -74.0060);
        region1.capacity = 5; // Below threshold

        let region2 = create_test_region("eu-west-1", 51.5074, -0.1278);

        router.register_region(region1);
        router.register_region(region2);

        let decision = router.route_request(None, None).await.unwrap();
        assert_eq!(decision.region, "eu-west-1");
    }

    #[tokio::test]
    async fn test_alternatives_in_decision() {
        let health_checker = Arc::new(MockHealthChecker::new());
        health_checker.set_healthy("us-east-1", true);
        health_checker.set_healthy("eu-west-1", true);
        health_checker.set_healthy("ap-southeast-1", true);

        let router = RegionRouter::new(RegionRouterConfig::default(), health_checker);

        router.register_region(create_test_region("us-east-1", 40.7128, -74.0060));
        router.register_region(create_test_region("eu-west-1", 51.5074, -0.1278));
        router.register_region(create_test_region("ap-southeast-1", 1.3521, 103.8198));

        let decision = router.route_request(None, None).await.unwrap();
        assert_eq!(decision.alternatives.len(), 2);
    }
}
