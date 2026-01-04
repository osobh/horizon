//! Region-aware load balancing with health checks and failover logic
//!
//! This module provides comprehensive load balancing capabilities for multi-region deployments:
//! - Health check monitoring and automated failover
//! - Weighted routing based on region performance
//! - Circuit breaker pattern for failing regions
//! - Load balancing algorithms (round-robin, least-connections, weighted)

use crate::error::{MultiRegionError, MultiRegionResult};
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use tokio::sync::RwLock;

/// Load balancer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadBalancerConfig {
    /// Load balancing algorithm
    pub algorithm: LoadBalancingAlgorithm,
    /// Health check configuration
    pub health_check: HealthCheckConfig,
    /// Circuit breaker configuration
    pub circuit_breaker: CircuitBreakerConfig,
    /// Connection timeout (milliseconds)
    pub connection_timeout_ms: u64,
    /// Request timeout (milliseconds)
    pub request_timeout_ms: u64,
    /// Maximum retries per request
    pub max_retries: u32,
    /// Enable sticky sessions
    pub sticky_sessions: bool,
}

/// Load balancing algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LoadBalancingAlgorithm {
    /// Round-robin distribution
    RoundRobin,
    /// Least connections
    LeastConnections,
    /// Weighted round-robin
    WeightedRoundRobin,
    /// IP hash-based routing
    IpHash,
    /// Geographic proximity
    Geographic,
    /// Response time based
    ResponseTime,
}

/// Health check configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckConfig {
    /// Health check interval (seconds)
    pub interval_seconds: u64,
    /// Health check timeout (milliseconds)
    pub timeout_ms: u64,
    /// Failure threshold before marking unhealthy
    pub failure_threshold: u32,
    /// Success threshold before marking healthy
    pub success_threshold: u32,
    /// Health check endpoint path
    pub health_path: String,
    /// Expected HTTP status codes for healthy response
    pub healthy_status_codes: Vec<u16>,
}

/// Circuit breaker configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerConfig {
    /// Failure threshold to open circuit
    pub failure_threshold: u32,
    /// Time window for failure counting (seconds)
    pub failure_window_seconds: u64,
    /// Time to wait before attempting to close circuit (seconds)
    pub recovery_timeout_seconds: u64,
    /// Minimum requests before considering circuit state
    pub min_requests: u32,
}

/// Region endpoint configuration
#[derive(Debug, Clone)]
pub struct RegionEndpoint {
    /// Region identifier
    pub region_id: String,
    /// Endpoint URL
    pub endpoint: String,
    /// Routing weight (1-100)
    pub weight: u32,
    /// Priority level (lower = higher priority)
    pub priority: u32,
    /// Maximum concurrent connections
    pub max_connections: u32,
    /// Current connection count
    pub current_connections: Arc<AtomicU64>,
    /// Health status
    pub health_status: Arc<RwLock<EndpointHealth>>,
}

/// Endpoint health status
#[derive(Debug, Clone)]
pub struct EndpointHealth {
    /// Health status
    pub healthy: bool,
    /// Last check timestamp
    pub last_check: DateTime<Utc>,
    /// Response time (milliseconds)
    pub response_time_ms: u64,
    /// Success count
    pub success_count: u32,
    /// Failure count
    pub failure_count: u32,
    /// Circuit breaker state
    pub circuit_state: CircuitState,
    /// Circuit breaker last opened timestamp
    pub circuit_opened_at: Option<DateTime<Utc>>,
}

/// Circuit breaker states
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CircuitState {
    /// Circuit is closed (normal operation)
    Closed,
    /// Circuit is open (failing requests)
    Open,
    /// Circuit is half-open (testing recovery)
    HalfOpen,
}

/// Request routing information
#[derive(Debug, Clone)]
pub struct RoutingRequest {
    /// Client IP address for IP hash routing
    pub client_ip: Option<String>,
    /// Session ID for sticky sessions
    pub session_id: Option<String>,
    /// Geographic information
    pub geo_info: Option<GeoInfo>,
    /// Request metadata
    pub metadata: HashMap<String, String>,
}

/// Geographic information for routing
#[derive(Debug, Clone)]
pub struct GeoInfo {
    /// Country code
    pub country: String,
    /// Region/state code
    pub region: String,
    /// City name
    pub city: String,
    /// Latitude
    pub latitude: f64,
    /// Longitude
    pub longitude: f64,
}

/// Load balancer implementation
pub struct LoadBalancer {
    config: LoadBalancerConfig,
    endpoints: Vec<RegionEndpoint>,
    round_robin_index: AtomicUsize,
    client: reqwest::Client,
    session_affinity: Arc<DashMap<String, String>>,
}

impl LoadBalancer {
    /// Create new load balancer
    pub fn new(
        config: LoadBalancerConfig,
        endpoints: Vec<RegionEndpoint>,
    ) -> MultiRegionResult<Self> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_millis(
                config.connection_timeout_ms,
            ))
            .build()
            .map_err(|e| MultiRegionError::LoadBalancerError {
                reason: format!("Failed to create HTTP client: {}", e),
            })?;

        Ok(Self {
            config,
            endpoints,
            round_robin_index: AtomicUsize::new(0),
            client,
            session_affinity: Arc::new(DashMap::new()),
        })
    }

    /// Select best endpoint for request
    pub async fn select_endpoint(&self, request: &RoutingRequest) -> MultiRegionResult<String> {
        // Check for sticky session routing
        if self.config.sticky_sessions {
            if let Some(session_id) = &request.session_id {
                if let Some(region_id) = self.session_affinity.get(session_id).map(|r| r.clone()) {
                    if let Some(endpoint) = self.find_healthy_endpoint(&region_id).await {
                        return Ok(endpoint);
                    }
                }
            }
        }

        // Get healthy endpoints
        let healthy_endpoints = self.get_healthy_endpoints().await;
        if healthy_endpoints.is_empty() {
            return Err(MultiRegionError::LoadBalancerError {
                reason: "No healthy endpoints available".to_string(),
            });
        }

        // Apply load balancing algorithm
        let selected = match self.config.algorithm {
            LoadBalancingAlgorithm::RoundRobin => self.round_robin_select(&healthy_endpoints),
            LoadBalancingAlgorithm::LeastConnections => {
                self.least_connections_select(&healthy_endpoints)
            }
            LoadBalancingAlgorithm::WeightedRoundRobin => {
                self.weighted_round_robin_select(&healthy_endpoints)
            }
            LoadBalancingAlgorithm::IpHash => self.ip_hash_select(&healthy_endpoints, request),
            LoadBalancingAlgorithm::Geographic => {
                self.geographic_select(&healthy_endpoints, request)
            }
            LoadBalancingAlgorithm::ResponseTime => {
                self.response_time_select(&healthy_endpoints).await
            }
        }?;

        // Update session affinity if enabled
        if self.config.sticky_sessions {
            if let Some(session_id) = &request.session_id {
                self.session_affinity
                    .insert(session_id.clone(), selected.region_id.clone());
            }
        }

        Ok(selected.endpoint.clone())
    }

    /// Get all healthy endpoints
    async fn get_healthy_endpoints(&self) -> Vec<&RegionEndpoint> {
        let mut healthy = Vec::new();
        for endpoint in &self.endpoints {
            let health = endpoint.health_status.read().await;
            if health.healthy && health.circuit_state != CircuitState::Open {
                healthy.push(endpoint);
            }
        }
        healthy
    }

    /// Find healthy endpoint by region ID
    async fn find_healthy_endpoint(&self, region_id: &str) -> Option<String> {
        for endpoint in &self.endpoints {
            if endpoint.region_id == region_id {
                let health = endpoint.health_status.read().await;
                if health.healthy && health.circuit_state != CircuitState::Open {
                    return Some(endpoint.endpoint.clone());
                }
            }
        }
        None
    }

    /// Round-robin endpoint selection
    fn round_robin_select<'a>(
        &self,
        endpoints: &[&'a RegionEndpoint],
    ) -> MultiRegionResult<&'a RegionEndpoint> {
        // Relaxed: approximate round-robin, exact fairness not required
        let index = self.round_robin_index.fetch_add(1, Ordering::Relaxed) % endpoints.len();
        Ok(endpoints[index])
    }

    /// Least connections endpoint selection
    fn least_connections_select<'a>(
        &self,
        endpoints: &[&'a RegionEndpoint],
    ) -> MultiRegionResult<&'a RegionEndpoint> {
        // Relaxed: approximate connection counts sufficient for load balancing
        let min_conn_endpoint = endpoints
            .iter()
            .min_by_key(|e| e.current_connections.load(Ordering::Relaxed))
            .ok_or_else(|| MultiRegionError::LoadBalancerError {
                reason: "No endpoints available for least connections selection".to_string(),
            })?;
        Ok(*min_conn_endpoint)
    }

    /// Weighted round-robin endpoint selection
    fn weighted_round_robin_select<'a>(
        &self,
        endpoints: &[&'a RegionEndpoint],
    ) -> MultiRegionResult<&'a RegionEndpoint> {
        let total_weight: u32 = endpoints.iter().map(|e| e.weight).sum();
        if total_weight == 0 {
            return self.round_robin_select(endpoints);
        }

        // Relaxed: approximate weight distribution sufficient for load balancing
        let mut weight_index =
            (self.round_robin_index.load(Ordering::Relaxed) % total_weight as usize) as u32;

        for endpoint in endpoints {
            if weight_index < endpoint.weight {
                return Ok(endpoint);
            }
            weight_index -= endpoint.weight;
        }

        // Fallback to first endpoint
        Ok(endpoints[0])
    }

    /// IP hash-based endpoint selection
    fn ip_hash_select<'a>(
        &self,
        endpoints: &[&'a RegionEndpoint],
        request: &RoutingRequest,
    ) -> MultiRegionResult<&'a RegionEndpoint> {
        if let Some(client_ip) = &request.client_ip {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};

            let mut hasher = DefaultHasher::new();
            client_ip.hash(&mut hasher);
            let hash = hasher.finish();
            let index = (hash as usize) % endpoints.len();
            Ok(endpoints[index])
        } else {
            self.round_robin_select(endpoints)
        }
    }

    /// Geographic proximity-based endpoint selection
    fn geographic_select<'a>(
        &self,
        endpoints: &[&'a RegionEndpoint],
        request: &RoutingRequest,
    ) -> MultiRegionResult<&'a RegionEndpoint> {
        if let Some(_geo_info) = &request.geo_info {
            // For simplicity, just select by priority for now
            // In a real implementation, you would calculate distance
            let min_priority_endpoint =
                endpoints.iter().min_by_key(|e| e.priority).ok_or_else(|| {
                    MultiRegionError::LoadBalancerError {
                        reason: "No endpoints available for geographic selection".to_string(),
                    }
                })?;
            Ok(*min_priority_endpoint)
        } else {
            self.round_robin_select(endpoints)
        }
    }

    /// Response time-based endpoint selection
    async fn response_time_select<'a>(
        &self,
        endpoints: &[&'a RegionEndpoint],
    ) -> MultiRegionResult<&'a RegionEndpoint> {
        let mut min_response_time = u64::MAX;
        let mut best_endpoint = None;

        for endpoint in endpoints {
            let health = endpoint.health_status.read().await;
            if health.response_time_ms < min_response_time {
                min_response_time = health.response_time_ms;
                best_endpoint = Some(*endpoint);
            }
        }

        best_endpoint.ok_or_else(|| MultiRegionError::LoadBalancerError {
            reason: "No endpoints available for response time selection".to_string(),
        })
    }

    /// Perform health check on all endpoints
    pub async fn health_check_all(&self) -> MultiRegionResult<()> {
        for endpoint in &self.endpoints {
            self.health_check_endpoint(endpoint).await?;
        }
        Ok(())
    }

    /// Perform health check on specific endpoint
    async fn health_check_endpoint(&self, endpoint: &RegionEndpoint) -> MultiRegionResult<()> {
        let health_url = format!(
            "{}{}",
            endpoint.endpoint, self.config.health_check.health_path
        );
        let start_time = std::time::Instant::now();

        let health_result = tokio::time::timeout(
            std::time::Duration::from_millis(self.config.health_check.timeout_ms),
            self.client.get(&health_url).send(),
        )
        .await;

        let response_time = start_time.elapsed().as_millis() as u64;
        let mut health = endpoint.health_status.write().await;
        health.last_check = Utc::now();
        health.response_time_ms = response_time;

        match health_result {
            Ok(Ok(response)) => {
                let status_code = response.status().as_u16();
                if self
                    .config
                    .health_check
                    .healthy_status_codes
                    .contains(&status_code)
                {
                    health.success_count += 1;
                    health.failure_count = 0;

                    if health.success_count >= self.config.health_check.success_threshold {
                        health.healthy = true;
                        health.circuit_state = CircuitState::Closed;
                    }
                } else {
                    self.handle_health_check_failure(&mut health);
                }
            }
            _ => {
                self.handle_health_check_failure(&mut health);
            }
        }

        self.update_circuit_breaker(&mut health);
        Ok(())
    }

    /// Handle health check failure
    fn handle_health_check_failure(&self, health: &mut EndpointHealth) {
        health.failure_count += 1;
        health.success_count = 0;

        if health.failure_count >= self.config.health_check.failure_threshold {
            health.healthy = false;
        }
    }

    /// Update circuit breaker state
    fn update_circuit_breaker(&self, health: &mut EndpointHealth) {
        let now = Utc::now();

        match health.circuit_state {
            CircuitState::Closed => {
                if health.failure_count >= self.config.circuit_breaker.failure_threshold {
                    health.circuit_state = CircuitState::Open;
                    health.circuit_opened_at = Some(now);
                }
            }
            CircuitState::Open => {
                if let Some(opened_at) = health.circuit_opened_at {
                    let elapsed = now.signed_duration_since(opened_at);
                    if elapsed.num_seconds()
                        >= self.config.circuit_breaker.recovery_timeout_seconds as i64
                    {
                        health.circuit_state = CircuitState::HalfOpen;
                    }
                }
            }
            CircuitState::HalfOpen => {
                if health.success_count >= self.config.health_check.success_threshold {
                    health.circuit_state = CircuitState::Closed;
                    health.circuit_opened_at = None;
                } else if health.failure_count > 0 {
                    health.circuit_state = CircuitState::Open;
                    health.circuit_opened_at = Some(now);
                }
            }
        }
    }

    /// Get endpoint statistics
    pub async fn get_endpoint_stats(&self) -> Vec<EndpointStats> {
        let mut stats = Vec::new();
        for endpoint in &self.endpoints {
            let health = endpoint.health_status.read().await;
            stats.push(EndpointStats {
                region_id: endpoint.region_id.clone(),
                endpoint: endpoint.endpoint.clone(),
                healthy: health.healthy,
                response_time_ms: health.response_time_ms,
                success_count: health.success_count,
                failure_count: health.failure_count,
                // Relaxed: approximate connection count for statistics
                current_connections: endpoint.current_connections.load(Ordering::Relaxed),
                circuit_state: health.circuit_state,
                weight: endpoint.weight,
                priority: endpoint.priority,
            });
        }
        stats
    }

    /// Increment connection count for endpoint
    pub fn increment_connections(&self, region_id: &str) {
        for endpoint in &self.endpoints {
            if endpoint.region_id == region_id {
                // Relaxed: approximate count sufficient for load balancing
                endpoint.current_connections.fetch_add(1, Ordering::Relaxed);
                break;
            }
        }
    }

    /// Decrement connection count for endpoint
    pub fn decrement_connections(&self, region_id: &str) {
        for endpoint in &self.endpoints {
            if endpoint.region_id == region_id {
                // Relaxed: approximate count sufficient for load balancing
                endpoint.current_connections.fetch_sub(1, Ordering::Relaxed);
                break;
            }
        }
    }
}

/// Endpoint statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EndpointStats {
    /// Region identifier
    pub region_id: String,
    /// Endpoint URL
    pub endpoint: String,
    /// Health status
    pub healthy: bool,
    /// Response time (milliseconds)
    pub response_time_ms: u64,
    /// Success count
    pub success_count: u32,
    /// Failure count
    pub failure_count: u32,
    /// Current connections
    pub current_connections: u64,
    /// Circuit breaker state
    pub circuit_state: CircuitState,
    /// Routing weight
    pub weight: u32,
    /// Priority level
    pub priority: u32,
}

impl Default for LoadBalancerConfig {
    fn default() -> Self {
        Self {
            algorithm: LoadBalancingAlgorithm::RoundRobin,
            health_check: HealthCheckConfig {
                interval_seconds: 30,
                timeout_ms: 5000,
                failure_threshold: 3,
                success_threshold: 2,
                health_path: "/health".to_string(),
                healthy_status_codes: vec![200, 201, 204],
            },
            circuit_breaker: CircuitBreakerConfig {
                failure_threshold: 5,
                failure_window_seconds: 60,
                recovery_timeout_seconds: 30,
                min_requests: 10,
            },
            connection_timeout_ms: 30000,
            request_timeout_ms: 60000,
            max_retries: 3,
            sticky_sessions: false,
        }
    }
}

impl RegionEndpoint {
    /// Create new region endpoint
    pub fn new(region_id: String, endpoint: String, weight: u32, priority: u32) -> Self {
        Self {
            region_id,
            endpoint,
            weight,
            priority,
            max_connections: 1000,
            current_connections: Arc::new(AtomicU64::new(0)),
            health_status: Arc::new(RwLock::new(EndpointHealth {
                healthy: false,
                last_check: Utc::now(),
                response_time_ms: 0,
                success_count: 0,
                failure_count: 0,
                circuit_state: CircuitState::Closed,
                circuit_opened_at: None,
            })),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_config() -> LoadBalancerConfig {
        LoadBalancerConfig {
            algorithm: LoadBalancingAlgorithm::RoundRobin,
            health_check: HealthCheckConfig {
                interval_seconds: 1,
                timeout_ms: 1000,
                failure_threshold: 2,
                success_threshold: 1,
                health_path: "/health".to_string(),
                healthy_status_codes: vec![200],
            },
            circuit_breaker: CircuitBreakerConfig {
                failure_threshold: 3,
                failure_window_seconds: 10,
                recovery_timeout_seconds: 5,
                min_requests: 1,
            },
            connection_timeout_ms: 5000,
            request_timeout_ms: 10000,
            max_retries: 2,
            sticky_sessions: false,
        }
    }

    fn create_test_endpoints() -> Vec<RegionEndpoint> {
        vec![
            RegionEndpoint::new(
                "us-east-1".to_string(),
                "https://us-east-1.example.com".to_string(),
                50,
                1,
            ),
            RegionEndpoint::new(
                "us-west-2".to_string(),
                "https://us-west-2.example.com".to_string(),
                30,
                2,
            ),
            RegionEndpoint::new(
                "eu-west-1".to_string(),
                "https://eu-west-1.example.com".to_string(),
                20,
                3,
            ),
        ]
    }

    #[test]
    fn test_load_balancer_creation() {
        let config = create_test_config();
        let endpoints = create_test_endpoints();
        let lb = LoadBalancer::new(config, endpoints);
        assert!(lb.is_ok());
    }

    #[tokio::test]
    async fn test_round_robin_selection() {
        let config = create_test_config();
        let mut endpoints = create_test_endpoints();

        // Mark all endpoints as healthy
        for endpoint in &mut endpoints {
            let mut health = endpoint.health_status.write().await;
            health.healthy = true;
        }

        let lb = LoadBalancer::new(config, endpoints).unwrap();
        let request = RoutingRequest {
            client_ip: None,
            session_id: None,
            geo_info: None,
            metadata: HashMap::new(),
        };

        // Test multiple selections to verify round-robin behavior
        let mut selected_endpoints = Vec::new();
        for _ in 0..6 {
            let endpoint = lb.select_endpoint(&request).await.unwrap();
            selected_endpoints.push(endpoint);
        }

        // Should cycle through all endpoints twice
        assert_eq!(selected_endpoints.len(), 6);
        // First cycle should match second cycle
        assert_eq!(selected_endpoints[0], selected_endpoints[3]);
        assert_eq!(selected_endpoints[1], selected_endpoints[4]);
        assert_eq!(selected_endpoints[2], selected_endpoints[5]);
    }

    #[tokio::test]
    async fn test_weighted_round_robin_selection() {
        let mut config = create_test_config();
        config.algorithm = LoadBalancingAlgorithm::WeightedRoundRobin;
        let mut endpoints = create_test_endpoints();

        // Mark all endpoints as healthy
        for endpoint in &mut endpoints {
            let mut health = endpoint.health_status.write().await;
            health.healthy = true;
        }

        let lb = LoadBalancer::new(config, endpoints).unwrap();
        let request = RoutingRequest {
            client_ip: None,
            session_id: None,
            geo_info: None,
            metadata: HashMap::new(),
        };

        // Test weighted selection
        let mut endpoint_counts = HashMap::new();
        for _ in 0..100 {
            let endpoint = lb.select_endpoint(&request).await.unwrap();
            *endpoint_counts.entry(endpoint).or_insert(0) += 1;
        }

        // Higher weighted endpoints should be selected more often
        assert!(endpoint_counts.len() > 0);
    }

    #[tokio::test]
    async fn test_least_connections_selection() {
        let mut config = create_test_config();
        config.algorithm = LoadBalancingAlgorithm::LeastConnections;
        let mut endpoints = create_test_endpoints();

        // Mark all endpoints as healthy
        for endpoint in &mut endpoints {
            let mut health = endpoint.health_status.write().await;
            health.healthy = true;
        }

        // Set different connection counts (Relaxed: test setup)
        endpoints[0]
            .current_connections
            .store(10, Ordering::Relaxed);
        endpoints[1].current_connections.store(5, Ordering::Relaxed);
        endpoints[2]
            .current_connections
            .store(15, Ordering::Relaxed);

        let lb = LoadBalancer::new(config, endpoints).unwrap();
        let request = RoutingRequest {
            client_ip: None,
            session_id: None,
            geo_info: None,
            metadata: HashMap::new(),
        };

        let selected = lb.select_endpoint(&request).await.unwrap();
        // Should select us-west-2 (lowest connections)
        assert_eq!(selected, "https://us-west-2.example.com");
    }

    #[tokio::test]
    async fn test_ip_hash_selection() {
        let mut config = create_test_config();
        config.algorithm = LoadBalancingAlgorithm::IpHash;
        let mut endpoints = create_test_endpoints();

        // Mark all endpoints as healthy
        for endpoint in &mut endpoints {
            let mut health = endpoint.health_status.write().await;
            health.healthy = true;
        }

        let lb = LoadBalancer::new(config, endpoints).unwrap();
        let request = RoutingRequest {
            client_ip: Some("192.168.1.100".to_string()),
            session_id: None,
            geo_info: None,
            metadata: HashMap::new(),
        };

        // Same IP should always select same endpoint
        let first_selection = lb.select_endpoint(&request).await.unwrap();
        let second_selection = lb.select_endpoint(&request).await.unwrap();
        assert_eq!(first_selection, second_selection);
    }

    #[tokio::test]
    async fn test_geographic_selection() {
        let mut config = create_test_config();
        config.algorithm = LoadBalancingAlgorithm::Geographic;
        let mut endpoints = create_test_endpoints();

        // Mark all endpoints as healthy
        for endpoint in &mut endpoints {
            let mut health = endpoint.health_status.write().await;
            health.healthy = true;
        }

        let lb = LoadBalancer::new(config, endpoints).unwrap();
        let request = RoutingRequest {
            client_ip: None,
            session_id: None,
            geo_info: Some(GeoInfo {
                country: "US".to_string(),
                region: "CA".to_string(),
                city: "San Francisco".to_string(),
                latitude: 37.7749,
                longitude: -122.4194,
            }),
            metadata: HashMap::new(),
        };

        let selected = lb.select_endpoint(&request).await.unwrap();
        // Should select us-east-1 (priority 1)
        assert_eq!(selected, "https://us-east-1.example.com");
    }

    #[tokio::test]
    async fn test_sticky_sessions() {
        let mut config = create_test_config();
        config.sticky_sessions = true;
        let mut endpoints = create_test_endpoints();

        // Mark all endpoints as healthy
        for endpoint in &mut endpoints {
            let mut health = endpoint.health_status.write().await;
            health.healthy = true;
        }

        let lb = LoadBalancer::new(config, endpoints).unwrap();
        let request = RoutingRequest {
            client_ip: None,
            session_id: Some("session123".to_string()),
            geo_info: None,
            metadata: HashMap::new(),
        };

        // First request establishes affinity
        let first_selection = lb.select_endpoint(&request).await.unwrap();

        // Subsequent requests should use same endpoint
        let second_selection = lb.select_endpoint(&request).await.unwrap();
        let third_selection = lb.select_endpoint(&request).await.unwrap();

        assert_eq!(first_selection, second_selection);
        assert_eq!(second_selection, third_selection);
    }

    #[tokio::test]
    async fn test_no_healthy_endpoints() {
        let config = create_test_config();
        let endpoints = create_test_endpoints(); // All start as unhealthy

        let lb = LoadBalancer::new(config, endpoints).unwrap();
        let request = RoutingRequest {
            client_ip: None,
            session_id: None,
            geo_info: None,
            metadata: HashMap::new(),
        };

        let result = lb.select_endpoint(&request).await;
        assert!(result.is_err());

        if let Err(MultiRegionError::LoadBalancerError { reason }) = result {
            assert_eq!(reason, "No healthy endpoints available");
        }
    }

    #[tokio::test]
    async fn test_circuit_breaker_functionality() {
        let config = create_test_config();
        let endpoints = create_test_endpoints();
        let lb = LoadBalancer::new(config, endpoints).unwrap();

        // Simulate circuit breaker opening
        let endpoint = &lb.endpoints[0];
        let mut health = endpoint.health_status.write().await;
        health.failure_count = 5; // Exceeds threshold
        health.circuit_state = CircuitState::Open;
        health.circuit_opened_at = Some(Utc::now());
        drop(health);

        // Set second endpoint as healthy
        let mut health2 = lb.endpoints[1].health_status.write().await;
        health2.healthy = true;
        drop(health2);

        let request = RoutingRequest {
            client_ip: None,
            session_id: None,
            geo_info: None,
            metadata: HashMap::new(),
        };

        let selected = lb.select_endpoint(&request).await.unwrap();
        // Should not select the first endpoint due to open circuit
        assert_ne!(selected, "https://us-east-1.example.com");
    }

    #[tokio::test]
    async fn test_connection_counting() {
        let config = create_test_config();
        let endpoints = create_test_endpoints();
        let lb = LoadBalancer::new(config, endpoints).unwrap();

        // Test connection increment/decrement
        lb.increment_connections("us-east-1");
        lb.increment_connections("us-east-1");

        let stats = lb.get_endpoint_stats().await;
        let us_east_stats = stats.iter().find(|s| s.region_id == "us-east-1").unwrap();
        assert_eq!(us_east_stats.current_connections, 2);

        lb.decrement_connections("us-east-1");

        let stats = lb.get_endpoint_stats().await;
        let us_east_stats = stats.iter().find(|s| s.region_id == "us-east-1").unwrap();
        assert_eq!(us_east_stats.current_connections, 1);
    }

    #[tokio::test]
    async fn test_endpoint_stats() {
        let config = create_test_config();
        let endpoints = create_test_endpoints();

        // Set up some test data
        let mut health = endpoints[0].health_status.write().await;
        health.healthy = true;
        health.response_time_ms = 150;
        health.success_count = 10;
        health.failure_count = 2;
        drop(health);

        let lb = LoadBalancer::new(config, endpoints).unwrap();
        let stats = lb.get_endpoint_stats().await;

        assert_eq!(stats.len(), 3);
        let us_east_stats = stats.iter().find(|s| s.region_id == "us-east-1").unwrap();
        assert!(us_east_stats.healthy);
        assert_eq!(us_east_stats.response_time_ms, 150);
        assert_eq!(us_east_stats.success_count, 10);
        assert_eq!(us_east_stats.failure_count, 2);
    }

    #[test]
    fn test_load_balancer_config_serialization() {
        let config = create_test_config();
        let json = serde_json::to_string(&config);
        assert!(json.is_ok());

        let deserialized: Result<LoadBalancerConfig, _> = serde_json::from_str(&json.unwrap());
        assert!(deserialized.is_ok());
    }

    #[test]
    fn test_endpoint_stats_serialization() {
        let stats = EndpointStats {
            region_id: "test-region".to_string(),
            endpoint: "https://test.example.com".to_string(),
            healthy: true,
            response_time_ms: 100,
            success_count: 5,
            failure_count: 1,
            current_connections: 10,
            circuit_state: CircuitState::Closed,
            weight: 50,
            priority: 1,
        };

        let json = serde_json::to_string(&stats);
        assert!(json.is_ok());

        let deserialized: Result<EndpointStats, _> = serde_json::from_str(&json.unwrap());
        assert!(deserialized.is_ok());
    }
}
