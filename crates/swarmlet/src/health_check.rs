//! Health check utilities for deployment verification
//!
//! Provides HTTP-based health checking for deployed services with retry logic,
//! configurable timeouts, and detailed status reporting.

use reqwest::Client;
use std::time::Duration;
use tracing::{debug, info, warn};

/// Health check result with detailed status information
#[derive(Debug, Clone)]
pub struct HealthCheckResult {
    /// Whether the health check passed
    pub passed: bool,
    /// HTTP status code if a response was received
    pub status_code: Option<u16>,
    /// Response latency in milliseconds
    pub latency_ms: u64,
    /// Error message if the check failed
    pub error: Option<String>,
    /// Number of attempts made
    pub attempts: u32,
}

impl HealthCheckResult {
    /// Create a successful health check result
    pub fn success(status_code: u16, latency_ms: u64, attempts: u32) -> Self {
        Self {
            passed: true,
            status_code: Some(status_code),
            latency_ms,
            error: None,
            attempts,
        }
    }

    /// Create a failed health check result
    pub fn failure(error: String, attempts: u32) -> Self {
        Self {
            passed: false,
            status_code: None,
            latency_ms: 0,
            error: Some(error),
            attempts,
        }
    }

    /// Create a failed health check result with status code
    pub fn http_failure(status_code: u16, latency_ms: u64, attempts: u32) -> Self {
        Self {
            passed: false,
            status_code: Some(status_code),
            latency_ms,
            error: Some(format!("HTTP {}", status_code)),
            attempts,
        }
    }
}

/// Configuration for health checks
#[derive(Debug, Clone)]
pub struct HealthCheckConfig {
    /// HTTP request timeout per attempt
    pub timeout: Duration,
    /// Maximum number of retry attempts
    pub max_retries: u32,
    /// Delay between retry attempts
    pub retry_interval: Duration,
    /// Initial delay before first check (for service startup)
    pub initial_delay: Duration,
    /// HTTP method to use (GET, HEAD)
    pub method: HealthCheckMethod,
    /// Expected status codes that count as healthy
    pub expected_status_codes: Vec<u16>,
}

impl Default for HealthCheckConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(5),
            max_retries: 3,
            retry_interval: Duration::from_secs(2),
            initial_delay: Duration::from_secs(0),
            method: HealthCheckMethod::Get,
            expected_status_codes: vec![200, 201, 204],
        }
    }
}

impl HealthCheckConfig {
    /// Create a quick health check config for fast probes
    pub fn quick() -> Self {
        Self {
            timeout: Duration::from_secs(2),
            max_retries: 1,
            retry_interval: Duration::from_millis(500),
            initial_delay: Duration::from_secs(0),
            method: HealthCheckMethod::Head,
            expected_status_codes: vec![200, 204],
        }
    }

    /// Create a thorough health check config for critical services
    pub fn thorough() -> Self {
        Self {
            timeout: Duration::from_secs(10),
            max_retries: 5,
            retry_interval: Duration::from_secs(3),
            initial_delay: Duration::from_secs(5),
            method: HealthCheckMethod::Get,
            expected_status_codes: vec![200, 201, 204],
        }
    }

    /// Set the timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Set the max retries
    pub fn with_max_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = max_retries;
        self
    }

    /// Set the retry interval
    pub fn with_retry_interval(mut self, interval: Duration) -> Self {
        self.retry_interval = interval;
        self
    }

    /// Set the initial delay
    pub fn with_initial_delay(mut self, delay: Duration) -> Self {
        self.initial_delay = delay;
        self
    }
}

/// HTTP method for health checks
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthCheckMethod {
    /// GET request (receives full response body)
    Get,
    /// HEAD request (headers only, faster)
    Head,
}

/// Health checker for verifying service availability
pub struct HealthChecker {
    /// HTTP client
    client: Client,
    /// Health check configuration
    config: HealthCheckConfig,
}

impl HealthChecker {
    /// Create a new health checker with default configuration
    pub fn new() -> Self {
        Self::with_config(HealthCheckConfig::default())
    }

    /// Create a new health checker with custom configuration
    pub fn with_config(config: HealthCheckConfig) -> Self {
        let client = Client::builder()
            .timeout(config.timeout)
            .build()
            .expect("Failed to build HTTP client");

        Self { client, config }
    }

    /// Perform a health check on the given endpoint
    ///
    /// # Arguments
    /// * `endpoint` - The URL to check (e.g., "http://localhost:8080/health")
    ///
    /// # Returns
    /// * `HealthCheckResult` - Detailed result of the health check
    pub async fn check(&self, endpoint: &str) -> HealthCheckResult {
        // Wait for initial delay if configured
        if !self.config.initial_delay.is_zero() {
            debug!(
                "Waiting {:?} before health check for {}",
                self.config.initial_delay, endpoint
            );
            tokio::time::sleep(self.config.initial_delay).await;
        }

        let mut attempts = 0;

        loop {
            attempts += 1;

            match self.do_check(endpoint).await {
                Ok(result) if result.passed => {
                    info!(
                        "Health check passed for {} (status: {}, latency: {}ms, attempts: {})",
                        endpoint,
                        result.status_code.unwrap_or(0),
                        result.latency_ms,
                        attempts
                    );
                    return HealthCheckResult {
                        attempts,
                        ..result
                    };
                }
                Ok(result) => {
                    // HTTP response received but status code indicates unhealthy
                    if attempts <= self.config.max_retries {
                        warn!(
                            "Health check failed for {} (status: {}, attempt {}/{}), retrying in {:?}",
                            endpoint,
                            result.status_code.unwrap_or(0),
                            attempts,
                            self.config.max_retries + 1,
                            self.config.retry_interval
                        );
                        tokio::time::sleep(self.config.retry_interval).await;
                    } else {
                        return HealthCheckResult {
                            attempts,
                            ..result
                        };
                    }
                }
                Err(e) => {
                    if attempts <= self.config.max_retries {
                        warn!(
                            "Health check error for {} (attempt {}/{}): {}, retrying in {:?}",
                            endpoint,
                            attempts,
                            self.config.max_retries + 1,
                            e,
                            self.config.retry_interval
                        );
                        tokio::time::sleep(self.config.retry_interval).await;
                    } else {
                        return HealthCheckResult::failure(e.to_string(), attempts);
                    }
                }
            }
        }
    }

    /// Perform a single health check without retries
    async fn do_check(&self, endpoint: &str) -> Result<HealthCheckResult, reqwest::Error> {
        let start = std::time::Instant::now();

        let response = match self.config.method {
            HealthCheckMethod::Get => self.client.get(endpoint).send().await?,
            HealthCheckMethod::Head => self.client.head(endpoint).send().await?,
        };

        let latency_ms = start.elapsed().as_millis() as u64;
        let status = response.status();
        let status_code = status.as_u16();

        let passed = self.config.expected_status_codes.contains(&status_code);

        if passed {
            Ok(HealthCheckResult::success(status_code, latency_ms, 1))
        } else {
            Ok(HealthCheckResult::http_failure(status_code, latency_ms, 1))
        }
    }

    /// Check if a TCP port is open (without HTTP request)
    ///
    /// Useful for checking if a service is listening before making HTTP requests.
    pub async fn check_port(host: &str, port: u16, timeout: Duration) -> bool {
        use tokio::net::TcpStream;

        match tokio::time::timeout(
            timeout,
            TcpStream::connect(format!("{}:{}", host, port)),
        )
        .await
        {
            Ok(Ok(_)) => {
                debug!("Port {}:{} is open", host, port);
                true
            }
            Ok(Err(e)) => {
                debug!("Port {}:{} connection failed: {}", host, port, e);
                false
            }
            Err(_) => {
                debug!("Port {}:{} connection timed out", host, port);
                false
            }
        }
    }

    /// Wait for a service to become available
    ///
    /// Continuously checks the endpoint until it becomes healthy or timeout is reached.
    ///
    /// # Arguments
    /// * `endpoint` - The URL to check
    /// * `timeout` - Maximum time to wait for the service
    /// * `poll_interval` - Time between checks
    ///
    /// # Returns
    /// * `Option<HealthCheckResult>` - Some(result) if service became healthy, None if timeout
    pub async fn wait_for_service(
        &self,
        endpoint: &str,
        timeout: Duration,
        poll_interval: Duration,
    ) -> Option<HealthCheckResult> {
        let start = std::time::Instant::now();

        info!(
            "Waiting for service at {} (timeout: {:?})",
            endpoint, timeout
        );

        loop {
            if start.elapsed() > timeout {
                warn!(
                    "Timeout waiting for service at {} after {:?}",
                    endpoint, timeout
                );
                return None;
            }

            // Use quick config for polling
            let quick_checker = HealthChecker::with_config(
                HealthCheckConfig::quick().with_max_retries(0),
            );

            let result = quick_checker.do_check(endpoint).await;

            if let Ok(result) = result {
                if result.passed {
                    info!(
                        "Service at {} is now healthy after {:?}",
                        endpoint,
                        start.elapsed()
                    );
                    return Some(result);
                }
            }

            tokio::time::sleep(poll_interval).await;
        }
    }
}

impl Default for HealthChecker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_check_result_success() {
        let result = HealthCheckResult::success(200, 50, 1);
        assert!(result.passed);
        assert_eq!(result.status_code, Some(200));
        assert_eq!(result.latency_ms, 50);
        assert!(result.error.is_none());
    }

    #[test]
    fn test_health_check_result_failure() {
        let result = HealthCheckResult::failure("Connection refused".to_string(), 3);
        assert!(!result.passed);
        assert!(result.status_code.is_none());
        assert_eq!(result.attempts, 3);
        assert!(result.error.is_some());
    }

    #[test]
    fn test_health_check_config_default() {
        let config = HealthCheckConfig::default();
        assert_eq!(config.timeout, Duration::from_secs(5));
        assert_eq!(config.max_retries, 3);
        assert!(config.expected_status_codes.contains(&200));
    }

    #[test]
    fn test_health_check_config_quick() {
        let config = HealthCheckConfig::quick();
        assert_eq!(config.timeout, Duration::from_secs(2));
        assert_eq!(config.max_retries, 1);
        assert_eq!(config.method, HealthCheckMethod::Head);
    }

    #[test]
    fn test_health_check_config_thorough() {
        let config = HealthCheckConfig::thorough();
        assert_eq!(config.timeout, Duration::from_secs(10));
        assert_eq!(config.max_retries, 5);
        assert_eq!(config.initial_delay, Duration::from_secs(5));
    }

    #[tokio::test]
    async fn test_check_port_closed() {
        // Port 49152 is unlikely to be in use
        let result = HealthChecker::check_port("127.0.0.1", 49152, Duration::from_millis(100)).await;
        assert!(!result);
    }
}
