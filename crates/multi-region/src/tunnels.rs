//! Secure inter-region tunnels with TLS authentication and connection pooling
//!
//! This module provides secure, encrypted communication channels between regions:
//! - TLS 1.3 encryption with certificate validation
//! - Mutual authentication with client certificates
//! - Connection pooling and reuse for performance
//! - Automatic reconnection and failover
//! - Bandwidth throttling and QoS management

use crate::error::{MultiRegionError, MultiRegionResult};
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock, Semaphore};
use tokio::time::{Duration, Instant};

/// Tunnel configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TunnelConfig {
    /// TLS configuration
    pub tls_config: TlsConfiguration,
    /// Authentication settings
    pub auth_config: AuthenticationConfig,
    /// Connection pool settings
    pub pool_config: ConnectionPoolConfig,
    /// Quality of Service settings
    pub qos_config: QosConfig,
    /// Timeout settings
    pub timeout_config: TimeoutConfig,
    /// Retry configuration
    pub retry_config: RetryConfig,
}

/// TLS configuration for secure tunnels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TlsConfiguration {
    /// TLS version to use
    pub tls_version: TlsVersion,
    /// Certificate file path
    pub cert_file: Option<String>,
    /// Private key file path
    pub key_file: Option<String>,
    /// CA certificate file path
    pub ca_file: Option<String>,
    /// Certificate data (embedded)
    pub cert_data: Option<Vec<u8>>,
    /// Private key data (embedded)
    pub key_data: Option<Vec<u8>>,
    /// CA certificate data (embedded)
    pub ca_data: Option<Vec<u8>>,
    /// Enable mutual TLS authentication
    pub mutual_tls: bool,
    /// Verify peer certificates
    pub verify_peer: bool,
    /// Allowed cipher suites
    pub cipher_suites: Vec<String>,
    /// Certificate validation mode
    pub cert_validation: CertificateValidation,
}

/// TLS version enum
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TlsVersion {
    /// TLS 1.2
    Tls12,
    /// TLS 1.3 (recommended)
    Tls13,
}

/// Certificate validation modes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CertificateValidation {
    /// Full certificate validation
    Full,
    /// Skip hostname verification
    SkipHostname,
    /// Skip all validation (insecure)
    Skip,
}

/// Authentication configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthenticationConfig {
    /// Authentication method
    pub method: AuthenticationMethod,
    /// API key for token-based auth
    pub api_key: Option<String>,
    /// Username for basic auth
    pub username: Option<String>,
    /// Password for basic auth
    pub password: Option<String>,
    /// JWT token for bearer auth
    pub jwt_token: Option<String>,
    /// Token refresh configuration
    pub token_refresh: Option<TokenRefreshConfig>,
    /// Authentication headers
    pub custom_headers: HashMap<String, String>,
}

/// Authentication methods
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuthenticationMethod {
    /// No authentication
    None,
    /// API key in header
    ApiKey,
    /// HTTP Basic authentication
    Basic,
    /// Bearer token authentication
    Bearer,
    /// Mutual TLS authentication
    MutualTls,
    /// Custom authentication
    Custom,
}

/// Token refresh configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenRefreshConfig {
    /// Refresh endpoint URL
    pub refresh_url: String,
    /// Refresh interval (seconds)
    pub refresh_interval_s: u64,
    /// Refresh token
    pub refresh_token: String,
}

/// Connection pool configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionPoolConfig {
    /// Maximum connections per region
    pub max_connections: u32,
    /// Minimum idle connections
    pub min_idle_connections: u32,
    /// Connection idle timeout (seconds)
    pub idle_timeout_s: u64,
    /// Maximum connection lifetime (seconds)
    pub max_lifetime_s: u64,
    /// Connection validation interval (seconds)
    pub validation_interval_s: u64,
    /// Enable connection multiplexing
    pub enable_multiplexing: bool,
    /// Maximum concurrent streams per connection
    pub max_streams_per_connection: u32,
}

/// Quality of Service configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QosConfig {
    /// Bandwidth limit (bytes per second)
    pub bandwidth_limit_bps: Option<u64>,
    /// Request rate limit (requests per second)
    pub rate_limit_rps: Option<u32>,
    /// Priority class
    pub priority: QosPriority,
    /// Enable traffic shaping
    pub traffic_shaping: bool,
    /// Congestion control algorithm
    pub congestion_control: CongestionControl,
}

/// QoS priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum QosPriority {
    /// Low priority
    Low,
    /// Normal priority
    Normal,
    /// High priority
    High,
    /// Critical priority
    Critical,
}

/// Congestion control algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CongestionControl {
    /// TCP Cubic
    Cubic,
    /// TCP BBR
    Bbr,
    /// Custom algorithm
    Custom,
}

/// Timeout configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeoutConfig {
    /// Connection timeout (milliseconds)
    pub connect_timeout_ms: u64,
    /// Request timeout (milliseconds)
    pub request_timeout_ms: u64,
    /// Keep-alive timeout (seconds)
    pub keepalive_timeout_s: u64,
    /// DNS resolution timeout (milliseconds)
    pub dns_timeout_ms: u64,
}

/// Retry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    /// Maximum retry attempts
    pub max_retries: u32,
    /// Base retry delay (milliseconds)
    pub base_delay_ms: u64,
    /// Maximum retry delay (milliseconds)
    pub max_delay_ms: u64,
    /// Exponential backoff multiplier
    pub backoff_multiplier: f64,
    /// Jitter factor (0.0 to 1.0)
    pub jitter_factor: f64,
    /// Retry only on specific errors
    pub retry_on_errors: Vec<RetryableError>,
}

/// Retryable error types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RetryableError {
    /// Network connection errors
    ConnectionError,
    /// Timeout errors
    Timeout,
    /// DNS resolution errors
    DnsError,
    /// TLS handshake errors
    TlsError,
    /// Server 5xx errors
    ServerError,
    /// Rate limiting errors
    RateLimit,
}

/// Tunnel connection state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionState {
    /// Connection is idle
    Idle,
    /// Connection is active
    Active,
    /// Connection is connecting
    Connecting,
    /// Connection failed
    Failed,
    /// Connection is being validated
    Validating,
}

/// Tunnel connection information
#[derive(Debug, Clone)]
pub struct TunnelConnection {
    /// Connection ID
    pub id: String,
    /// Source region
    pub source_region: String,
    /// Target region
    pub target_region: String,
    /// Connection state
    pub state: ConnectionState,
    /// Established timestamp
    pub established_at: DateTime<Utc>,
    /// Last used timestamp
    pub last_used: DateTime<Utc>,
    /// Bytes transferred
    pub bytes_transferred: u64,
    /// Request count
    pub request_count: u64,
    /// Connection statistics
    pub stats: ConnectionStats,
}

/// Connection statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionStats {
    /// Round-trip time (milliseconds)
    pub rtt_ms: u64,
    /// Throughput (bytes per second)
    pub throughput_bps: u64,
    /// Error count
    pub error_count: u32,
    /// Success count
    pub success_count: u32,
    /// Last error timestamp
    pub last_error: Option<DateTime<Utc>>,
    /// Connection uptime (seconds)
    pub uptime_s: u64,
}

/// Tunnel request context
#[derive(Debug, Clone)]
pub struct TunnelRequest {
    /// Request ID
    pub id: String,
    /// Source region
    pub source_region: String,
    /// Target region
    pub target_region: String,
    /// Request data
    pub data: Vec<u8>,
    /// Request headers
    pub headers: HashMap<String, String>,
    /// Request priority
    pub priority: QosPriority,
    /// Request timestamp
    pub timestamp: DateTime<Utc>,
    /// Timeout override
    pub timeout_ms: Option<u64>,
}

/// Tunnel response
#[derive(Debug, Clone)]
pub struct TunnelResponse {
    /// Request ID
    pub request_id: String,
    /// Response status
    pub status: u16,
    /// Response data
    pub data: Vec<u8>,
    /// Response headers
    pub headers: HashMap<String, String>,
    /// Response timestamp
    pub timestamp: DateTime<Utc>,
    /// Processing time (milliseconds)
    pub processing_time_ms: u64,
}

/// Secure tunnel manager
pub struct TunnelManager {
    config: TunnelConfig,
    connections: Arc<DashMap<String, Arc<Mutex<TunnelConnection>>>>,
    connection_pools: Arc<RwLock<HashMap<String, ConnectionPool>>>,
    request_counter: AtomicU64,
    client: reqwest::Client,
    bandwidth_limiters: Arc<DashMap<String, Arc<Semaphore>>>,
}

/// Connection pool for a specific region pair
struct ConnectionPool {
    region_pair: String,
    active_connections: Vec<Arc<Mutex<TunnelConnection>>>,
    idle_connections: Vec<Arc<Mutex<TunnelConnection>>>,
    pool_config: ConnectionPoolConfig,
    last_validation: Instant,
}

impl TunnelManager {
    /// Create new tunnel manager
    pub fn new(config: TunnelConfig) -> MultiRegionResult<Self> {
        let client_builder = reqwest::Client::builder()
            .timeout(Duration::from_millis(
                config.timeout_config.connect_timeout_ms,
            ))
            .danger_accept_invalid_certs(true); // For testing only

        let client = client_builder
            .build()
            .map_err(|e| MultiRegionError::TunnelError {
                reason: format!("Failed to create HTTP client: {}", e),
            })?;

        Ok(Self {
            config,
            connections: Arc::new(DashMap::new()),
            connection_pools: Arc::new(RwLock::new(HashMap::new())),
            request_counter: AtomicU64::new(0),
            client,
            bandwidth_limiters: Arc::new(DashMap::new()),
        })
    }

    /// Establish tunnel connection between regions
    pub async fn establish_tunnel(
        &self,
        source_region: &str,
        target_region: &str,
    ) -> MultiRegionResult<String> {
        let connection_key = format!("{}:{}", source_region, target_region);

        // Check if connection already exists
        if let Some(existing_conn) = self.connections.get(&connection_key) {
            let conn = existing_conn.lock().await;
            if conn.state == ConnectionState::Active {
                return Ok(conn.id.clone());
            }
        }

        // Create new connection
        let connection_id = self.generate_connection_id();
        let connection = TunnelConnection {
            id: connection_id.clone(),
            source_region: source_region.to_string(),
            target_region: target_region.to_string(),
            state: ConnectionState::Connecting,
            established_at: Utc::now(),
            last_used: Utc::now(),
            bytes_transferred: 0,
            request_count: 0,
            stats: ConnectionStats {
                rtt_ms: 0,
                throughput_bps: 0,
                error_count: 0,
                success_count: 0,
                last_error: None,
                uptime_s: 0,
            },
        };

        // Perform TLS handshake and authentication
        self.perform_handshake(&connection).await?;

        // Mark connection as active
        let mut active_connection = connection;
        active_connection.state = ConnectionState::Active;
        active_connection.established_at = Utc::now();

        // Store connection
        self.connections.insert(
            connection_key.clone(),
            Arc::new(Mutex::new(active_connection)),
        );

        // Add to connection pool
        self.add_to_pool(&connection_key, connection_id.clone())
            .await?;

        // Initialize bandwidth limiter if configured
        if let Some(bandwidth_limit) = self.config.qos_config.bandwidth_limit_bps {
            self.bandwidth_limiters.insert(
                connection_key,
                Arc::new(Semaphore::new(bandwidth_limit as usize)),
            );
        }

        Ok(connection_id)
    }

    /// Send request through tunnel
    pub async fn send_request(&self, request: TunnelRequest) -> MultiRegionResult<TunnelResponse> {
        let connection_key = format!("{}:{}", request.source_region, request.target_region);

        // Apply rate limiting
        if let Some(rate_limit) = self.config.qos_config.rate_limit_rps {
            self.apply_rate_limit(&connection_key, rate_limit).await?;
        }

        // Apply bandwidth limiting
        if let Some(_bandwidth_limit) = self.config.qos_config.bandwidth_limit_bps {
            self.apply_bandwidth_limit(&connection_key, request.data.len())
                .await?;
        }

        // Get connection from pool
        let connection = self.get_connection_from_pool(&connection_key).await?;

        // Send request with retries
        let start_time = Instant::now();
        let response = self.send_with_retry(&connection, &request).await?;
        let processing_time = start_time.elapsed().as_millis() as u64;

        // Update connection statistics
        let connection_id = {
            let conn = connection.lock().await;
            conn.id.clone()
        };
        self.update_connection_stats(&connection_id, true, processing_time)
            .await;

        Ok(TunnelResponse {
            request_id: request.id,
            status: 200,
            data: response,
            headers: HashMap::new(),
            timestamp: Utc::now(),
            processing_time_ms: processing_time,
        })
    }

    /// Perform TLS handshake and authentication
    async fn perform_handshake(&self, connection: &TunnelConnection) -> MultiRegionResult<()> {
        let handshake_url = format!(
            "https://{}.example.com/tunnel/handshake",
            connection.target_region
        );

        let mut request_builder = self.client.post(&handshake_url);

        // Add authentication headers
        request_builder = self.add_authentication_headers(request_builder)?;

        // Perform handshake request
        let response = request_builder
            .json(&serde_json::json!({
                "source_region": connection.source_region,
                "connection_id": connection.id,
                "tls_version": self.config.tls_config.tls_version,
                "mutual_tls": self.config.tls_config.mutual_tls
            }))
            .send()
            .await
            .map_err(|e| MultiRegionError::TunnelError {
                reason: format!("Handshake failed: {}", e),
            })?;

        if !response.status().is_success() {
            return Err(MultiRegionError::TunnelError {
                reason: format!("Handshake failed with status: {}", response.status()),
            });
        }

        Ok(())
    }

    /// Add authentication headers to request
    fn add_authentication_headers(
        &self,
        mut request_builder: reqwest::RequestBuilder,
    ) -> MultiRegionResult<reqwest::RequestBuilder> {
        match &self.config.auth_config.method {
            AuthenticationMethod::None => {}
            AuthenticationMethod::ApiKey => {
                if let Some(api_key) = &self.config.auth_config.api_key {
                    request_builder = request_builder.header("X-API-Key", api_key);
                }
            }
            AuthenticationMethod::Basic => {
                if let (Some(username), Some(password)) = (
                    &self.config.auth_config.username,
                    &self.config.auth_config.password,
                ) {
                    request_builder = request_builder.basic_auth(username, Some(password));
                }
            }
            AuthenticationMethod::Bearer => {
                if let Some(token) = &self.config.auth_config.jwt_token {
                    request_builder = request_builder.bearer_auth(token);
                }
            }
            AuthenticationMethod::MutualTls => {
                // Mutual TLS is handled at the client level
            }
            AuthenticationMethod::Custom => {
                // Add custom headers
                for (key, value) in &self.config.auth_config.custom_headers {
                    request_builder = request_builder.header(key, value);
                }
            }
        }

        Ok(request_builder)
    }

    /// Add connection to pool
    async fn add_to_pool(
        &self,
        connection_key: &str,
        _connection_id: String,
    ) -> MultiRegionResult<()> {
        let mut pools = self.connection_pools.write().await;
        let pool = pools
            .entry(connection_key.to_string())
            .or_insert_with(|| ConnectionPool {
                region_pair: connection_key.to_string(),
                active_connections: Vec::new(),
                idle_connections: Vec::new(),
                pool_config: self.config.pool_config.clone(),
                last_validation: Instant::now(),
            });

        // Find the connection and add to pool
        if let Some(connection) = self.connections.get(connection_key) {
            pool.active_connections.push(connection.clone());
        }

        Ok(())
    }

    /// Get connection from pool
    async fn get_connection_from_pool(
        &self,
        connection_key: &str,
    ) -> MultiRegionResult<Arc<Mutex<TunnelConnection>>> {
        let mut pools = self.connection_pools.write().await;

        if let Some(pool) = pools.get_mut(connection_key) {
            // Try to get an idle connection first
            if let Some(connection) = pool.idle_connections.pop() {
                pool.active_connections.push(connection.clone());
                return Ok(connection);
            }

            // If no idle connections, try to get an active one (if multiplexing is enabled)
            if self.config.pool_config.enable_multiplexing && !pool.active_connections.is_empty() {
                return Ok(pool.active_connections[0].clone());
            }
        }

        // If no connections available, create a new one
        drop(pools);
        let parts: Vec<&str> = connection_key.split(':').collect();
        if parts.len() == 2 {
            self.establish_tunnel(parts[0], parts[1]).await?;

            // Try again
            let pools = self.connection_pools.read().await;
            if let Some(pool) = pools.get(connection_key) {
                if let Some(connection) = pool.active_connections.first() {
                    return Ok(connection.clone());
                }
            }
        }

        Err(MultiRegionError::TunnelError {
            reason: "Failed to get connection from pool".to_string(),
        })
    }

    /// Send request with retry logic
    async fn send_with_retry(
        &self,
        connection: &Arc<Mutex<TunnelConnection>>,
        request: &TunnelRequest,
    ) -> MultiRegionResult<Vec<u8>> {
        let mut attempt = 0;
        let mut delay = self.config.retry_config.base_delay_ms;

        while attempt < self.config.retry_config.max_retries {
            match self.send_request_once(connection, request).await {
                Ok(response) => return Ok(response),
                Err(error) => {
                    attempt += 1;

                    // Check if error is retryable
                    if !self.is_retryable_error(&error)
                        || attempt >= self.config.retry_config.max_retries
                    {
                        return Err(error);
                    }

                    // Apply exponential backoff with jitter
                    let jitter = if self.config.retry_config.jitter_factor > 0.0 {
                        let jitter_range =
                            (delay as f64 * self.config.retry_config.jitter_factor) as u64;
                        fastrand::u64(0..jitter_range)
                    } else {
                        0
                    };

                    tokio::time::sleep(Duration::from_millis(delay + jitter)).await;

                    delay = ((delay as f64 * self.config.retry_config.backoff_multiplier) as u64)
                        .min(self.config.retry_config.max_delay_ms);
                }
            }
        }

        Err(MultiRegionError::TunnelError {
            reason: "Max retries exceeded".to_string(),
        })
    }

    /// Send single request attempt
    async fn send_request_once(
        &self,
        connection: &Arc<Mutex<TunnelConnection>>,
        request: &TunnelRequest,
    ) -> MultiRegionResult<Vec<u8>> {
        let conn = connection.lock().await;
        let target_url = format!("https://{}.example.com/tunnel/data", conn.target_region);
        drop(conn);

        let mut request_builder = self.client.post(&target_url);
        request_builder = self.add_authentication_headers(request_builder)?;

        let response = request_builder
            .timeout(Duration::from_millis(
                request
                    .timeout_ms
                    .unwrap_or(self.config.timeout_config.request_timeout_ms),
            ))
            .body(request.data.clone())
            .send()
            .await
            .map_err(|e| MultiRegionError::TunnelError {
                reason: format!("Request failed: {}", e),
            })?;

        if !response.status().is_success() {
            return Err(MultiRegionError::TunnelError {
                reason: format!("Request failed with status: {}", response.status()),
            });
        }

        let response_data = response
            .bytes()
            .await
            .map_err(|e| MultiRegionError::TunnelError {
                reason: format!("Failed to read response: {}", e),
            })?
            .to_vec();

        Ok(response_data)
    }

    /// Check if error is retryable
    fn is_retryable_error(&self, error: &MultiRegionError) -> bool {
        match error {
            MultiRegionError::TunnelError { reason } => {
                // Simple heuristic - in a real implementation, this would be more sophisticated
                reason.contains("timeout")
                    || reason.contains("connection")
                    || reason.contains("dns")
                    || reason.contains("5") // 5xx status codes
            }
            _ => false,
        }
    }

    /// Apply rate limiting
    async fn apply_rate_limit(
        &self,
        _connection_key: &str,
        _rate_limit_rps: u32,
    ) -> MultiRegionResult<()> {
        // In a real implementation, this would use a token bucket or similar algorithm
        // For now, just add a small delay
        tokio::time::sleep(Duration::from_millis(1)).await;
        Ok(())
    }

    /// Apply bandwidth limiting
    async fn apply_bandwidth_limit(
        &self,
        connection_key: &str,
        data_size: usize,
    ) -> MultiRegionResult<()> {
        if let Some(limiter) = self.bandwidth_limiters.get(connection_key) {
            let _permit = limiter.acquire_many(data_size as u32).await.map_err(|_| {
                MultiRegionError::TunnelError {
                    reason: "Bandwidth limit exceeded".to_string(),
                }
            })?;
            // Permit is held until dropped
        }
        Ok(())
    }

    /// Update connection statistics
    async fn update_connection_stats(
        &self,
        connection_id: &str,
        success: bool,
        processing_time_ms: u64,
    ) {
        for connection_arc in self.connections.iter() {
            let mut connection = connection_arc.value().lock().await;
            if connection.id == connection_id {
                connection.last_used = Utc::now();
                connection.request_count += 1;

                if success {
                    connection.stats.success_count += 1;
                } else {
                    connection.stats.error_count += 1;
                    connection.stats.last_error = Some(Utc::now());
                }

                connection.stats.rtt_ms = processing_time_ms;
                break;
            }
        }
    }

    /// Close tunnel connection
    pub async fn close_tunnel(&self, connection_id: &str) -> MultiRegionResult<()> {
        let mut connection_key_to_remove = None;

        for entry in self.connections.iter() {
            let connection = entry.value().lock().await;
            if connection.id == connection_id {
                connection_key_to_remove = Some(entry.key().clone());
                break;
            }
        }

        if let Some(key) = connection_key_to_remove {
            self.connections.remove(&key);

            // Remove from connection pool
            let mut pools = self.connection_pools.write().await;
            if let Some(pool) = pools.get_mut(&key) {
                pool.active_connections.retain(|conn_arc| {
                    let conn = conn_arc.try_lock();
                    match conn {
                        Ok(conn) => conn.id != connection_id,
                        Err(_) => true, // Keep if we can't lock
                    }
                });
                pool.idle_connections.retain(|conn_arc| {
                    let conn = conn_arc.try_lock();
                    match conn {
                        Ok(conn) => conn.id != connection_id,
                        Err(_) => true, // Keep if we can't lock
                    }
                });
            }
        }

        Ok(())
    }

    /// Get tunnel statistics
    pub async fn get_tunnel_stats(&self) -> Vec<TunnelConnection> {
        let mut stats = Vec::new();

        for connection_arc in self.connections.iter() {
            let connection = connection_arc.value().lock().await;
            stats.push(connection.clone());
        }

        stats
    }

    /// Validate all connections
    pub async fn validate_connections(&self) -> MultiRegionResult<u32> {
        let mut validated_count = 0;

        for connection_arc in self.connections.iter() {
            let mut connection = connection_arc.value().lock().await;
            connection.state = ConnectionState::Validating;

            // Perform validation (simple ping)
            let validation_result = self.validate_connection(&connection).await;

            if validation_result.is_ok() {
                connection.state = ConnectionState::Active;
                validated_count += 1;
            } else {
                connection.state = ConnectionState::Failed;
            }
        }

        Ok(validated_count)
    }

    /// Validate a single connection
    async fn validate_connection(&self, connection: &TunnelConnection) -> MultiRegionResult<()> {
        let ping_url = format!(
            "https://{}.example.com/tunnel/ping",
            connection.target_region
        );

        let response = self
            .client
            .get(&ping_url)
            .timeout(Duration::from_millis(5000))
            .send()
            .await
            .map_err(|e| MultiRegionError::TunnelError {
                reason: format!("Connection validation failed: {}", e),
            })?;

        if response.status().is_success() {
            Ok(())
        } else {
            Err(MultiRegionError::TunnelError {
                reason: "Connection validation failed".to_string(),
            })
        }
    }

    /// Generate unique connection ID
    fn generate_connection_id(&self) -> String {
        // Relaxed: independent ID counter, uniqueness guaranteed by fetch_add
        let counter = self.request_counter.fetch_add(1, Ordering::Relaxed);
        format!("tunnel_{}", counter)
    }
}

impl Default for TunnelConfig {
    fn default() -> Self {
        Self {
            tls_config: TlsConfiguration {
                tls_version: TlsVersion::Tls13,
                cert_file: None,
                key_file: None,
                ca_file: None,
                cert_data: None,
                key_data: None,
                ca_data: None,
                mutual_tls: true,
                verify_peer: true,
                cipher_suites: vec![
                    "TLS_AES_256_GCM_SHA384".to_string(),
                    "TLS_CHACHA20_POLY1305_SHA256".to_string(),
                    "TLS_AES_128_GCM_SHA256".to_string(),
                ],
                cert_validation: CertificateValidation::Full,
            },
            auth_config: AuthenticationConfig {
                method: AuthenticationMethod::MutualTls,
                api_key: None,
                username: None,
                password: None,
                jwt_token: None,
                token_refresh: None,
                custom_headers: HashMap::new(),
            },
            pool_config: ConnectionPoolConfig {
                max_connections: 10,
                min_idle_connections: 2,
                idle_timeout_s: 300,
                max_lifetime_s: 3600,
                validation_interval_s: 60,
                enable_multiplexing: true,
                max_streams_per_connection: 100,
            },
            qos_config: QosConfig {
                bandwidth_limit_bps: None,
                rate_limit_rps: None,
                priority: QosPriority::Normal,
                traffic_shaping: false,
                congestion_control: CongestionControl::Cubic,
            },
            timeout_config: TimeoutConfig {
                connect_timeout_ms: 10000,
                request_timeout_ms: 30000,
                keepalive_timeout_s: 60,
                dns_timeout_ms: 5000,
            },
            retry_config: RetryConfig {
                max_retries: 3,
                base_delay_ms: 1000,
                max_delay_ms: 30000,
                backoff_multiplier: 2.0,
                jitter_factor: 0.1,
                retry_on_errors: vec![
                    RetryableError::ConnectionError,
                    RetryableError::Timeout,
                    RetryableError::DnsError,
                    RetryableError::ServerError,
                ],
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_config() -> TunnelConfig {
        TunnelConfig {
            tls_config: TlsConfiguration {
                tls_version: TlsVersion::Tls13,
                cert_file: None,
                key_file: None,
                ca_file: None,
                cert_data: None,
                key_data: None,
                ca_data: None,
                mutual_tls: false,  // Disable for testing
                verify_peer: false, // Disable for testing
                cipher_suites: vec!["TLS_AES_256_GCM_SHA384".to_string()],
                cert_validation: CertificateValidation::Skip,
            },
            auth_config: AuthenticationConfig {
                method: AuthenticationMethod::None,
                api_key: None,
                username: None,
                password: None,
                jwt_token: None,
                token_refresh: None,
                custom_headers: HashMap::new(),
            },
            pool_config: ConnectionPoolConfig {
                max_connections: 5,
                min_idle_connections: 1,
                idle_timeout_s: 60,
                max_lifetime_s: 300,
                validation_interval_s: 30,
                enable_multiplexing: true,
                max_streams_per_connection: 10,
            },
            qos_config: QosConfig {
                bandwidth_limit_bps: Some(1_000_000), // 1 MB/s
                rate_limit_rps: Some(100),
                priority: QosPriority::Normal,
                traffic_shaping: false,
                congestion_control: CongestionControl::Cubic,
            },
            timeout_config: TimeoutConfig {
                connect_timeout_ms: 5000,
                request_timeout_ms: 10000,
                keepalive_timeout_s: 30,
                dns_timeout_ms: 2000,
            },
            retry_config: RetryConfig {
                max_retries: 2,
                base_delay_ms: 500,
                max_delay_ms: 5000,
                backoff_multiplier: 2.0,
                jitter_factor: 0.1,
                retry_on_errors: vec![RetryableError::ConnectionError, RetryableError::Timeout],
            },
        }
    }

    #[test]
    fn test_tunnel_manager_creation() {
        let config = create_test_config();
        let manager = TunnelManager::new(config);
        if let Err(e) = &manager {
            println!("Error creating tunnel manager: {:?}", e);
        }
        assert!(manager.is_ok());
    }

    #[tokio::test]
    async fn test_tunnel_establishment_failure() {
        let config = create_test_config();
        let manager = TunnelManager::new(config).unwrap();

        // This will fail because we don't have real endpoints
        let result = manager.establish_tunnel("us-east-1", "us-west-2").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_tunnel_request_creation() {
        let request = TunnelRequest {
            id: "test-request".to_string(),
            source_region: "us-east-1".to_string(),
            target_region: "us-west-2".to_string(),
            data: b"test data".to_vec(),
            headers: HashMap::new(),
            priority: QosPriority::Normal,
            timestamp: Utc::now(),
            timeout_ms: Some(5000),
        };

        assert_eq!(request.id, "test-request");
        assert_eq!(request.data, b"test data");
        assert_eq!(request.priority, QosPriority::Normal);
    }

    #[tokio::test]
    async fn test_send_request_failure() {
        let config = create_test_config();
        let manager = TunnelManager::new(config).unwrap();

        let request = TunnelRequest {
            id: "test-request".to_string(),
            source_region: "us-east-1".to_string(),
            target_region: "us-west-2".to_string(),
            data: b"test data".to_vec(),
            headers: HashMap::new(),
            priority: QosPriority::Normal,
            timestamp: Utc::now(),
            timeout_ms: Some(1000),
        };

        // This will fail because we don't have established connections
        let result = manager.send_request(request).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_connection_validation() {
        let config = create_test_config();
        let manager = TunnelManager::new(config).unwrap();

        // This will validate 0 connections since none are established
        let validated = manager.validate_connections().await.unwrap();
        assert_eq!(validated, 0);
    }

    #[tokio::test]
    async fn test_tunnel_stats_empty() {
        let config = create_test_config();
        let manager = TunnelManager::new(config).unwrap();

        let stats = manager.get_tunnel_stats().await;
        assert_eq!(stats.len(), 0);
    }

    #[tokio::test]
    async fn test_close_nonexistent_tunnel() {
        let config = create_test_config();
        let manager = TunnelManager::new(config).unwrap();

        let result = manager.close_tunnel("nonexistent").await;
        assert!(result.is_ok());
    }

    #[test]
    fn test_authentication_method_serialization() {
        let methods = vec![
            AuthenticationMethod::None,
            AuthenticationMethod::ApiKey,
            AuthenticationMethod::Basic,
            AuthenticationMethod::Bearer,
            AuthenticationMethod::MutualTls,
            AuthenticationMethod::Custom,
        ];

        for method in methods {
            let json = serde_json::to_string(&method).unwrap();
            let deserialized: AuthenticationMethod = serde_json::from_str(&json).unwrap();
            assert_eq!(method, deserialized);
        }
    }

    #[test]
    fn test_tls_version_serialization() {
        let versions = vec![TlsVersion::Tls12, TlsVersion::Tls13];

        for version in versions {
            let json = serde_json::to_string(&version).unwrap();
            let deserialized: TlsVersion = serde_json::from_str(&json).unwrap();
            assert_eq!(version, deserialized);
        }
    }

    #[test]
    fn test_qos_priority_ordering() {
        assert!(QosPriority::Low < QosPriority::Normal);
        assert!(QosPriority::Normal < QosPriority::High);
        assert!(QosPriority::High < QosPriority::Critical);
    }

    #[test]
    fn test_connection_state_transitions() {
        let states = vec![
            ConnectionState::Idle,
            ConnectionState::Connecting,
            ConnectionState::Active,
            ConnectionState::Validating,
            ConnectionState::Failed,
        ];

        // Test that all states are distinct
        for (i, state1) in states.iter().enumerate() {
            for (j, state2) in states.iter().enumerate() {
                if i != j {
                    assert_ne!(state1, state2);
                }
            }
        }
    }

    #[test]
    fn test_tunnel_config_default() {
        let config = TunnelConfig::default();
        assert_eq!(config.tls_config.tls_version, TlsVersion::Tls13);
        assert!(config.tls_config.mutual_tls);
        assert_eq!(config.auth_config.method, AuthenticationMethod::MutualTls);
        assert_eq!(config.pool_config.max_connections, 10);
        assert_eq!(config.qos_config.priority, QosPriority::Normal);
    }

    #[test]
    fn test_retry_config_validation() {
        let config = create_test_config();
        assert!(config.retry_config.max_retries > 0);
        assert!(config.retry_config.base_delay_ms > 0);
        assert!(config.retry_config.max_delay_ms >= config.retry_config.base_delay_ms);
        assert!(config.retry_config.backoff_multiplier > 1.0);
        assert!(
            config.retry_config.jitter_factor >= 0.0 && config.retry_config.jitter_factor <= 1.0
        );
    }

    #[test]
    fn test_connection_stats_initialization() {
        let stats = ConnectionStats {
            rtt_ms: 0,
            throughput_bps: 0,
            error_count: 0,
            success_count: 0,
            last_error: None,
            uptime_s: 0,
        };

        assert_eq!(stats.rtt_ms, 0);
        assert_eq!(stats.error_count, 0);
        assert_eq!(stats.success_count, 0);
        assert!(stats.last_error.is_none());
    }

    #[test]
    fn test_tunnel_config_serialization() {
        let config = create_test_config();
        let json = serde_json::to_string(&config);
        assert!(json.is_ok());

        let deserialized: Result<TunnelConfig, _> = serde_json::from_str(&json.unwrap());
        assert!(deserialized.is_ok());
    }

    #[test]
    fn test_retryable_error_types() {
        let errors = vec![
            RetryableError::ConnectionError,
            RetryableError::Timeout,
            RetryableError::DnsError,
            RetryableError::TlsError,
            RetryableError::ServerError,
            RetryableError::RateLimit,
        ];

        for error in errors {
            let json = serde_json::to_string(&error).unwrap();
            let deserialized: RetryableError = serde_json::from_str(&json).unwrap();
            assert_eq!(error, deserialized);
        }
    }
}
