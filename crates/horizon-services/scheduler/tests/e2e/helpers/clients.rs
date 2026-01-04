// HTTP client helpers for service integration testing
//
// These clients provide typed interfaces to communicate with services
// in integration tests. They use reqwest for HTTP communication.

use anyhow::Result;
use reqwest::{Client, Response};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use uuid::Uuid;

/// Scheduler service client
#[derive(Clone)]
pub struct SchedulerClient {
    base_url: String,
    client: Client,
}

impl SchedulerClient {
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            client: Client::builder()
                .timeout(Duration::from_secs(10))
                .build()
                .unwrap(),
        }
    }

    pub fn from_env() -> Self {
        let port = std::env::var("SCHEDULER_PORT").unwrap_or_else(|_| "8080".to_string());
        Self::new(format!("http://localhost:{}", port))
    }

    pub async fn health(&self) -> Result<Response> {
        let url = format!("{}/health", self.base_url);
        Ok(self.client.get(&url).send().await?)
    }

    pub async fn submit_job(&self, request: &SubmitJobRequest) -> Result<Response> {
        let url = format!("{}/api/v1/jobs", self.base_url);
        Ok(self.client.post(&url).json(request).send().await?)
    }

    pub async fn get_job(&self, job_id: Uuid) -> Result<Response> {
        let url = format!("{}/api/v1/jobs/{}", self.base_url, job_id);
        Ok(self.client.get(&url).send().await?)
    }

    pub async fn list_jobs(&self) -> Result<Response> {
        let url = format!("{}/api/v1/jobs", self.base_url);
        Ok(self.client.get(&url).send().await?)
    }

    pub async fn cancel_job(&self, job_id: Uuid) -> Result<Response> {
        let url = format!("{}/api/v1/jobs/{}", self.base_url, job_id);
        Ok(self.client.delete(&url).send().await?)
    }

    pub async fn get_queue_status(&self) -> Result<Response> {
        let url = format!("{}/api/v1/queue", self.base_url);
        Ok(self.client.get(&url).send().await?)
    }
}

/// Governor service client
#[derive(Clone)]
pub struct GovernorClient {
    base_url: String,
    client: Client,
}

impl GovernorClient {
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            client: Client::builder()
                .timeout(Duration::from_secs(10))
                .build()
                .unwrap(),
        }
    }

    pub fn from_env() -> Self {
        let port = std::env::var("GOVERNOR_PORT").unwrap_or_else(|_| "8081".to_string());
        Self::new(format!("http://localhost:{}", port))
    }

    pub async fn health(&self) -> Result<Response> {
        let url = format!("{}/health", self.base_url);
        Ok(self.client.get(&url).send().await?)
    }

    pub async fn create_policy(&self, request: &CreatePolicyRequest) -> Result<Response> {
        let url = format!("{}/api/v1/policies", self.base_url);
        Ok(self.client.post(&url).json(request).send().await?)
    }

    pub async fn get_policy(&self, name: &str) -> Result<Response> {
        let url = format!("{}/api/v1/policies/{}", self.base_url, name);
        Ok(self.client.get(&url).send().await?)
    }

    pub async fn list_policies(&self) -> Result<Response> {
        let url = format!("{}/api/v1/policies", self.base_url);
        Ok(self.client.get(&url).send().await?)
    }

    pub async fn update_policy(
        &self,
        name: &str,
        request: &UpdatePolicyRequest,
    ) -> Result<Response> {
        let url = format!("{}/api/v1/policies/{}", self.base_url, name);
        Ok(self.client.put(&url).json(request).send().await?)
    }

    pub async fn delete_policy(&self, name: &str) -> Result<Response> {
        let url = format!("{}/api/v1/policies/{}", self.base_url, name);
        Ok(self.client.delete(&url).send().await?)
    }

    pub async fn evaluate(&self, request: &EvaluateRequest) -> Result<Response> {
        let url = format!("{}/api/v1/evaluate", self.base_url);
        Ok(self.client.post(&url).json(request).send().await?)
    }
}

/// Quota Manager service client
#[derive(Clone)]
pub struct QuotaManagerClient {
    base_url: String,
    client: Client,
}

impl QuotaManagerClient {
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            client: Client::builder()
                .timeout(Duration::from_secs(10))
                .build()
                .unwrap(),
        }
    }

    pub fn from_env() -> Self {
        let port = std::env::var("QUOTA_MANAGER_PORT").unwrap_or_else(|_| "8082".to_string());
        Self::new(format!("http://localhost:{}", port))
    }

    pub async fn health(&self) -> Result<Response> {
        let url = format!("{}/health", self.base_url);
        Ok(self.client.get(&url).send().await?)
    }

    pub async fn create_quota(&self, request: &CreateQuotaRequest) -> Result<Response> {
        let url = format!("{}/api/v1/quotas", self.base_url);
        Ok(self.client.post(&url).json(request).send().await?)
    }

    pub async fn get_quota(&self, id: Uuid) -> Result<Response> {
        let url = format!("{}/api/v1/quotas/{}", self.base_url, id);
        Ok(self.client.get(&url).send().await?)
    }

    pub async fn list_quotas(&self) -> Result<Response> {
        let url = format!("{}/api/v1/quotas", self.base_url);
        Ok(self.client.get(&url).send().await?)
    }

    pub async fn update_quota(&self, id: Uuid, request: &UpdateQuotaRequest) -> Result<Response> {
        let url = format!("{}/api/v1/quotas/{}", self.base_url, id);
        Ok(self.client.put(&url).json(request).send().await?)
    }

    pub async fn delete_quota(&self, id: Uuid) -> Result<Response> {
        let url = format!("{}/api/v1/quotas/{}", self.base_url, id);
        Ok(self.client.delete(&url).send().await?)
    }

    pub async fn check_allocation(&self, request: &AllocationCheckRequest) -> Result<Response> {
        let url = format!("{}/api/v1/allocations/check", self.base_url);
        Ok(self.client.post(&url).json(request).send().await?)
    }

    pub async fn create_allocation(&self, request: &CreateAllocationRequest) -> Result<Response> {
        let url = format!("{}/api/v1/allocations", self.base_url);
        Ok(self.client.post(&url).json(request).send().await?)
    }

    pub async fn get_allocation(&self, id: Uuid) -> Result<Response> {
        let url = format!("{}/api/v1/allocations/{}", self.base_url, id);
        Ok(self.client.get(&url).send().await?)
    }

    pub async fn release_allocation(&self, id: Uuid) -> Result<Response> {
        let url = format!("{}/api/v1/allocations/{}", self.base_url, id);
        Ok(self.client.delete(&url).send().await?)
    }

    pub async fn get_usage_stats(&self, quota_id: Uuid) -> Result<Response> {
        let url = format!("{}/api/v1/quotas/{}/usage", self.base_url, quota_id);
        Ok(self.client.get(&url).send().await?)
    }
}

/// API Gateway client
#[derive(Clone)]
pub struct ApiGatewayClient {
    base_url: String,
    client: Client,
}

impl ApiGatewayClient {
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            client: Client::builder()
                .timeout(Duration::from_secs(10))
                .build()
                .unwrap(),
        }
    }

    pub fn from_env() -> Self {
        let port = std::env::var("API_GATEWAY_PORT").unwrap_or_else(|_| "8000".to_string());
        Self::new(format!("http://localhost:{}", port))
    }

    /// Create client with authentication token
    pub fn with_token(base_url: impl Into<String>, token: impl Into<String>) -> Self {
        let mut headers = reqwest::header::HeaderMap::new();
        let token_value = format!("Bearer {}", token.into());
        headers.insert(reqwest::header::AUTHORIZATION, token_value.parse().unwrap());

        Self {
            base_url: base_url.into(),
            client: Client::builder()
                .timeout(Duration::from_secs(10))
                .default_headers(headers)
                .build()
                .unwrap(),
        }
    }

    pub async fn health(&self) -> Result<Response> {
        let url = format!("{}/health", self.base_url);
        Ok(self.client.get(&url).send().await?)
    }

    // Proxy methods for scheduler endpoints through gateway
    pub async fn submit_job(&self, request: &SubmitJobRequest) -> Result<Response> {
        let url = format!("{}/api/v1/jobs", self.base_url);
        Ok(self.client.post(&url).json(request).send().await?)
    }

    pub async fn get_job(&self, job_id: Uuid) -> Result<Response> {
        let url = format!("{}/api/v1/jobs/{}", self.base_url, job_id);
        Ok(self.client.get(&url).send().await?)
    }

    // Proxy methods for governor endpoints through gateway
    pub async fn evaluate_policy(&self, request: &EvaluateRequest) -> Result<Response> {
        let url = format!("{}/api/v1/evaluate", self.base_url);
        Ok(self.client.post(&url).json(request).send().await?)
    }

    // Proxy methods for quota manager endpoints through gateway
    pub async fn check_quota(&self, request: &AllocationCheckRequest) -> Result<Response> {
        let url = format!("{}/api/v1/allocations/check", self.base_url);
        Ok(self.client.post(&url).json(request).send().await?)
    }
}

// ============================================================================
// Request/Response DTOs
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubmitJobRequest {
    pub user_id: String,
    pub priority: i32,
    pub cpus: i32,
    pub memory_mb: i32,
    pub gpus: Option<i32>,
    pub gpu_type: Option<String>,
    pub estimated_duration_secs: Option<i32>,
    pub metadata: Option<serde_json::Value>,
}

impl Default for SubmitJobRequest {
    fn default() -> Self {
        Self {
            user_id: "test-user".to_string(),
            priority: 5,
            cpus: 2,
            memory_mb: 4096,
            gpus: None,
            gpu_type: None,
            estimated_duration_secs: Some(3600),
            metadata: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreatePolicyRequest {
    pub name: String,
    pub content: String,
    pub description: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdatePolicyRequest {
    pub content: String,
    pub description: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluateRequest {
    pub principal: Principal,
    pub action: String,
    pub resource: Resource,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Principal {
    pub id: String,
    #[serde(rename = "type")]
    pub principal_type: String,
    pub attributes: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Resource {
    pub id: String,
    #[serde(rename = "type")]
    pub resource_type: String,
    pub attributes: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateQuotaRequest {
    pub entity_type: String,
    pub entity_id: String,
    pub parent_id: Option<Uuid>,
    pub resource_type: String,
    pub limit_value: f64,
    pub period_days: Option<i32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateQuotaRequest {
    pub limit_value: Option<f64>,
    pub period_days: Option<i32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationCheckRequest {
    pub entity_type: String,
    pub entity_id: String,
    pub resource_type: String,
    pub amount: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateAllocationRequest {
    pub quota_id: Uuid,
    pub resource_id: String,
    pub resource_type: String,
    pub amount: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluateResponse {
    pub decision: String,
    pub policy_name: Option<String>,
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationCheckResponse {
    pub allowed: bool,
    pub available: f64,
    pub requested: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_creation() {
        let scheduler = SchedulerClient::new("http://localhost:8080");
        assert_eq!(scheduler.base_url, "http://localhost:8080");

        let governor = GovernorClient::new("http://localhost:8081");
        assert_eq!(governor.base_url, "http://localhost:8081");

        let quota_manager = QuotaManagerClient::new("http://localhost:8082");
        assert_eq!(quota_manager.base_url, "http://localhost:8082");

        let gateway = ApiGatewayClient::new("http://localhost:8000");
        assert_eq!(gateway.base_url, "http://localhost:8000");
    }

    #[test]
    fn test_client_from_env() {
        let _scheduler = SchedulerClient::from_env();
        let _governor = GovernorClient::from_env();
        let _quota_manager = QuotaManagerClient::from_env();
        let _gateway = ApiGatewayClient::from_env();
    }

    #[test]
    fn test_gateway_client_with_token() {
        let gateway = ApiGatewayClient::with_token("http://localhost:8000", "test-token");
        assert_eq!(gateway.base_url, "http://localhost:8000");
    }

    #[test]
    fn test_submit_job_request_default() {
        let request = SubmitJobRequest::default();
        assert_eq!(request.user_id, "test-user");
        assert_eq!(request.priority, 5);
        assert_eq!(request.cpus, 2);
        assert_eq!(request.memory_mb, 4096);
    }
}
