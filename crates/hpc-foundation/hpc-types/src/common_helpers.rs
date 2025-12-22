//! Common message helpers
//!
//! This module provides helper functions for working with common messages
//! like health checks.

use crate::common::v1::{HealthCheckRequest, HealthCheckResponse};

/// Builder for HealthCheckResponse
///
/// # Examples
///
/// ```
/// use hpc_types::common_helpers::HealthCheckResponseBuilder;
///
/// let response = HealthCheckResponseBuilder::new()
///     .healthy(true)
///     .version("0.1.0")
///     .uptime_seconds(3600)
///     .build();
///
/// assert!(response.healthy);
/// assert_eq!(response.version, "0.1.0");
/// ```
pub struct HealthCheckResponseBuilder {
    response: HealthCheckResponse,
}

impl HealthCheckResponseBuilder {
    /// Creates a new health check response builder
    pub fn new() -> Self {
        Self {
            response: HealthCheckResponse {
                healthy: true,
                version: String::new(),
                uptime_seconds: 0,
            },
        }
    }

    /// Sets the healthy status
    pub fn healthy(mut self, healthy: bool) -> Self {
        self.response.healthy = healthy;
        self
    }

    /// Sets the version
    pub fn version(mut self, version: impl Into<String>) -> Self {
        self.response.version = version.into();
        self
    }

    /// Sets the uptime in seconds
    pub fn uptime_seconds(mut self, uptime: i64) -> Self {
        self.response.uptime_seconds = uptime;
        self
    }

    /// Builds the HealthCheckResponse
    pub fn build(self) -> HealthCheckResponse {
        self.response
    }
}

impl Default for HealthCheckResponseBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Creates a healthy response with version and uptime
pub fn healthy_response(version: impl Into<String>, uptime_seconds: i64) -> HealthCheckResponse {
    HealthCheckResponse {
        healthy: true,
        version: version.into(),
        uptime_seconds,
    }
}

/// Creates an unhealthy response
pub fn unhealthy_response(version: impl Into<String>) -> HealthCheckResponse {
    HealthCheckResponse {
        healthy: false,
        version: version.into(),
        uptime_seconds: 0,
    }
}

/// Creates a health check request
pub fn health_check_request() -> HealthCheckRequest {
    HealthCheckRequest {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_check_response_builder() {
        let response = HealthCheckResponseBuilder::new()
            .healthy(true)
            .version("0.1.0")
            .uptime_seconds(3600)
            .build();

        assert!(response.healthy);
        assert_eq!(response.version, "0.1.0");
        assert_eq!(response.uptime_seconds, 3600);
    }

    #[test]
    fn test_healthy_response_helper() {
        let response = healthy_response("0.1.0", 3600);
        assert!(response.healthy);
        assert_eq!(response.version, "0.1.0");
        assert_eq!(response.uptime_seconds, 3600);
    }

    #[test]
    fn test_unhealthy_response_helper() {
        let response = unhealthy_response("0.1.0");
        assert!(!response.healthy);
        assert_eq!(response.version, "0.1.0");
    }

    #[test]
    fn test_health_check_request_helper() {
        let request = health_check_request();
        // Just verify it can be created
        let _ = request;
    }
}
