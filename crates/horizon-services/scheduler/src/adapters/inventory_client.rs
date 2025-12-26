use crate::config::InventoryConfig;
use crate::error::HpcError;
use crate::models::{Topology, Gpu};
use crate::Result;
use reqwest::Client;
use std::time::Duration;

/// Client for communicating with the inventory service
#[derive(Clone)]
pub struct InventoryClient {
    client: Client,
    base_url: String,
}

impl InventoryClient {
    pub fn new(config: &InventoryConfig) -> Result<Self> {
        let client = Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .build()?;

        Ok(Self {
            client,
            base_url: config.base_url.clone(),
        })
    }

    /// Get current cluster topology with available resources
    pub async fn get_topology(&self) -> Result<Topology> {
        let url = format!("{}/api/v1/topology", self.base_url);
        let response = self.client.get(&url).send().await?;

        if !response.status().is_success() {
            return Err(HpcError::internal(format!(
                "Inventory service returned status: {}",
                response.status()
            )));
        }

        let topology = response.json::<Topology>().await?;
        Ok(topology)
    }

    /// Get available GPUs matching criteria
    pub async fn get_available_gpus(&self, gpu_type: Option<&str>, count: usize) -> Result<Vec<Gpu>> {
        let mut url = format!("{}/api/v1/gpus?available=true&limit={}", self.base_url, count);

        if let Some(gpu_type) = gpu_type {
            url.push_str(&format!("&type={}", gpu_type));
        }

        let response = self.client.get(&url).send().await?;

        if !response.status().is_success() {
            return Err(HpcError::internal(format!(
                "Inventory service returned status: {}",
                response.status()
            )));
        }

        let gpus = response.json::<Vec<Gpu>>().await?;
        Ok(gpus)
    }

    /// Reserve GPUs for a job
    pub async fn reserve_gpus(&self, job_id: uuid::Uuid, gpu_ids: &[uuid::Uuid]) -> Result<()> {
        let url = format!("{}/api/v1/reservations", self.base_url);

        let payload = serde_json::json!({
            "job_id": job_id,
            "gpu_ids": gpu_ids,
        });

        let response = self.client.post(&url).json(&payload).send().await?;

        if !response.status().is_success() {
            return Err(HpcError::internal(format!(
                "Failed to reserve GPUs: {}",
                response.status()
            )));
        }

        Ok(())
    }

    /// Release GPU reservations
    pub async fn release_gpus(&self, job_id: uuid::Uuid) -> Result<()> {
        let url = format!("{}/api/v1/reservations/{}", self.base_url, job_id);

        let response = self.client.delete(&url).send().await?;

        if !response.status().is_success() {
            return Err(HpcError::internal(format!(
                "Failed to release GPUs: {}",
                response.status()
            )));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inventory_client_creation() {
        let config = InventoryConfig {
            base_url: "http://localhost:8081".to_string(),
            timeout_secs: 30,
            retry_attempts: 3,
        };

        let client = InventoryClient::new(&config);
        assert!(client.is_ok());
    }

    #[tokio::test]
    async fn test_reserve_gpus() {
        // This is a unit test - in integration tests we'd use a real service
        let config = InventoryConfig {
            base_url: "http://localhost:8081".to_string(),
            timeout_secs: 30,
            retry_attempts: 3,
        };

        let client = InventoryClient::new(&config).unwrap();
        let _job_id = uuid::Uuid::new_v4();
        let _gpu_ids = vec![uuid::Uuid::new_v4(), uuid::Uuid::new_v4()];

        // In a real test environment, this would connect to a test inventory service
        // For now, we just verify the client can be constructed
        assert_eq!(client.base_url, "http://localhost:8081");
    }
}
