// Service management utilities for E2E testing
//
// This module provides utilities for checking service health and availability
// in integration tests. Tests assume services are running on localhost.

use anyhow::{anyhow, Result};
use reqwest::Client;
use std::time::Duration;
use tokio::time::sleep;

/// Service endpoint configuration
#[derive(Debug, Clone)]
pub struct ServiceEndpoint {
    pub name: String,
    pub base_url: String,
    pub health_path: String,
}

impl ServiceEndpoint {
    pub fn scheduler() -> Self {
        let port = std::env::var("SCHEDULER_PORT").unwrap_or_else(|_| "8080".to_string());
        Self {
            name: "scheduler".to_string(),
            base_url: format!("http://localhost:{}", port),
            health_path: "/health".to_string(),
        }
    }

    pub fn governor() -> Self {
        let port = std::env::var("GOVERNOR_PORT").unwrap_or_else(|_| "8081".to_string());
        Self {
            name: "governor".to_string(),
            base_url: format!("http://localhost:{}", port),
            health_path: "/health".to_string(),
        }
    }

    pub fn quota_manager() -> Self {
        let port = std::env::var("QUOTA_MANAGER_PORT").unwrap_or_else(|_| "8082".to_string());
        Self {
            name: "quota-manager".to_string(),
            base_url: format!("http://localhost:{}", port),
            health_path: "/health".to_string(),
        }
    }

    pub fn api_gateway() -> Self {
        let port = std::env::var("API_GATEWAY_PORT").unwrap_or_else(|_| "8000".to_string());
        Self {
            name: "api-gateway".to_string(),
            base_url: format!("http://localhost:{}", port),
            health_path: "/health".to_string(),
        }
    }

    /// Check if the service is healthy
    pub async fn is_healthy(&self) -> bool {
        let client = Client::builder()
            .timeout(Duration::from_secs(2))
            .build()
            .unwrap();

        let url = format!("{}{}", self.base_url, self.health_path);

        match client.get(&url).send().await {
            Ok(resp) => resp.status().is_success(),
            Err(_) => false,
        }
    }

    /// Wait for service to become healthy (with timeout)
    pub async fn wait_for_health(&self, timeout: Duration, check_interval: Duration) -> Result<()> {
        let start = std::time::Instant::now();

        while start.elapsed() < timeout {
            if self.is_healthy().await {
                tracing::info!("Service {} is healthy", self.name);
                return Ok(());
            }

            sleep(check_interval).await;
        }

        Err(anyhow!(
            "Service {} did not become healthy within {:?}",
            self.name,
            timeout
        ))
    }

    /// Check if service is available (returns descriptive result)
    pub async fn check_available(&self) -> Result<()> {
        if self.is_healthy().await {
            Ok(())
        } else {
            Err(anyhow!(
                "Service {} is not available at {}. \
                Please ensure the service is running. \
                You can set {}_PORT environment variable to change the port.",
                self.name,
                self.base_url,
                self.name.to_uppercase().replace("-", "_")
            ))
        }
    }
}

/// Wait for multiple services to become healthy
pub async fn wait_for_services(
    services: &[ServiceEndpoint],
    timeout: Duration,
) -> Result<()> {
    let check_interval = Duration::from_millis(500);

    for service in services {
        service.wait_for_health(timeout, check_interval).await?;
    }

    Ok(())
}

/// Check if all required services are available for a test
/// Returns Ok if all are available, or skips the test with a descriptive message
pub async fn require_services(services: &[ServiceEndpoint]) -> Result<()> {
    for service in services {
        service.check_available().await?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_service_endpoint_creation() {
        let scheduler = ServiceEndpoint::scheduler();
        assert_eq!(scheduler.name, "scheduler");
        assert!(scheduler.base_url.contains("localhost"));
        assert_eq!(scheduler.health_path, "/health");

        let governor = ServiceEndpoint::governor();
        assert_eq!(governor.name, "governor");

        let quota_manager = ServiceEndpoint::quota_manager();
        assert_eq!(quota_manager.name, "quota-manager");

        let gateway = ServiceEndpoint::api_gateway();
        assert_eq!(gateway.name, "api-gateway");
    }

    #[tokio::test]
    async fn test_service_health_check_returns_result() {
        let scheduler = ServiceEndpoint::scheduler();
        // Just verify it returns a bool, don't require service to be running
        let _is_healthy = scheduler.is_healthy().await;
    }
}
