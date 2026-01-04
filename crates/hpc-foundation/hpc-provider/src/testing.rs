/// Test utilities for provider implementations
use crate::{
    Availability, CapacityProvider, HealthStatus, Instance, InstanceState, ProviderError,
    ProviderResult, ProvisionResult, ProvisionSpec, Quote, QuoteRequest, ServiceQuotas, SpotPrices,
};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;

/// Mock provider for testing
#[derive(Clone)]
pub struct MockProvider {
    name: String,
    healthy: Arc<Mutex<bool>>,
    quotas: Arc<Mutex<ServiceQuotas>>,
    instances: Arc<Mutex<HashMap<String, Instance>>>,
}

impl MockProvider {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            healthy: Arc::new(Mutex::new(true)),
            quotas: Arc::new(Mutex::new(ServiceQuotas {
                max_instances: 100,
                current_instances: 0,
                max_vcpus: 400,
                current_vcpus: 0,
                max_gpus: 50,
                current_gpus: 0,
            })),
            instances: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn set_healthy(&self, healthy: bool) {
        *self.healthy.lock().unwrap() = healthy;
    }

    pub fn set_quotas(&self, quotas: ServiceQuotas) {
        *self.quotas.lock().unwrap() = quotas;
    }

    pub fn instance_count(&self) -> usize {
        self.instances.lock().unwrap().len()
    }
}

#[async_trait::async_trait]
impl CapacityProvider for MockProvider {
    fn name(&self) -> &str {
        &self.name
    }

    async fn get_quote(&self, request: &QuoteRequest) -> ProviderResult<Quote> {
        if request.count == 0 {
            return Err(ProviderError::InvalidRequest(
                "count must be greater than 0".to_string(),
            ));
        }

        Ok(Quote {
            provider: self.name.clone(),
            instance_type: request.instance_type.clone(),
            region: request.region.clone(),
            hourly_rate: dec!(1.50),
            spot_rate: if request.spot { Some(dec!(0.50)) } else { None },
            availability: Availability::Available,
            lead_time_hours: 0,
        })
    }

    async fn provision(&self, spec: &ProvisionSpec) -> ProviderResult<ProvisionResult> {
        if spec.count == 0 {
            return Err(ProviderError::InvalidRequest(
                "count must be greater than 0".to_string(),
            ));
        }

        let mut instances_map = self.instances.lock().unwrap();
        let mut quotas = self.quotas.lock().unwrap();

        if quotas.current_instances + spec.count > quotas.max_instances {
            return Err(ProviderError::QuotaExceeded(
                "instance quota exceeded".to_string(),
            ));
        }

        let instances: Vec<Instance> = (0..spec.count)
            .map(|i| {
                let id = format!("{}-instance-{}", self.name, instances_map.len() + i);
                let instance = Instance {
                    id: id.clone(),
                    instance_type: spec.instance_type.clone(),
                    region: spec.region.clone(),
                    public_ip: Some(format!("10.0.{}.{}", i / 256, i % 256)),
                    private_ip: format!("192.168.{}.{}", i / 256, i % 256),
                    state: InstanceState::Pending,
                };
                instances_map.insert(id, instance.clone());
                instance
            })
            .collect();

        quotas.current_instances += spec.count;

        let total_cost_estimate = dec!(1.50) * Decimal::from(spec.count);

        Ok(ProvisionResult {
            instances,
            total_cost_estimate,
        })
    }

    async fn deprovision(&self, instance_id: &str) -> ProviderResult<()> {
        if instance_id.is_empty() {
            return Err(ProviderError::InvalidRequest(
                "instance_id cannot be empty".to_string(),
            ));
        }

        let mut instances = self.instances.lock().unwrap();
        let mut quotas = self.quotas.lock().unwrap();

        if instances.remove(instance_id).is_some() {
            quotas.current_instances = quotas.current_instances.saturating_sub(1);
            Ok(())
        } else {
            Err(ProviderError::NotFound(format!(
                "instance {} not found",
                instance_id
            )))
        }
    }

    async fn check_spot_prices(
        &self,
        instance_type: &str,
        region: &str,
    ) -> ProviderResult<SpotPrices> {
        if instance_type.is_empty() {
            return Err(ProviderError::InvalidRequest(
                "instance_type cannot be empty".to_string(),
            ));
        }

        Ok(SpotPrices {
            instance_type: instance_type.to_string(),
            region: region.to_string(),
            current_price: dec!(0.50),
            average_price: dec!(0.55),
            min_price: dec!(0.40),
            max_price: dec!(0.80),
        })
    }

    async fn get_quotas(&self) -> ProviderResult<ServiceQuotas> {
        Ok(self.quotas.lock().unwrap().clone())
    }

    async fn health(&self) -> ProviderResult<HealthStatus> {
        let healthy = *self.healthy.lock().unwrap();
        Ok(HealthStatus {
            healthy,
            message: if healthy {
                "All systems operational".to_string()
            } else {
                "Service degraded".to_string()
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mock_provider_creation() {
        let provider = MockProvider::new("test-mock");
        assert_eq!(provider.name(), "test-mock");
        assert_eq!(provider.instance_count(), 0);
    }

    #[tokio::test]
    async fn test_mock_provider_set_healthy() {
        let provider = MockProvider::new("test-mock");
        provider.set_healthy(false);

        let health = provider.health().await.unwrap();
        assert!(!health.healthy);
    }

    #[tokio::test]
    async fn test_mock_provider_quota_enforcement() {
        let provider = MockProvider::new("test-mock");
        provider.set_quotas(ServiceQuotas {
            max_instances: 2,
            current_instances: 0,
            max_vcpus: 8,
            current_vcpus: 0,
            max_gpus: 4,
            current_gpus: 0,
        });

        let spec = ProvisionSpec {
            instance_type: "test.large".to_string(),
            region: "test-region".to_string(),
            count: 3,
            spot: false,
            tags: HashMap::new(),
        };

        let result = provider.provision(&spec).await;
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ProviderError::QuotaExceeded(_)
        ));
    }

    #[tokio::test]
    async fn test_mock_provider_provision_and_deprovision() {
        let provider = MockProvider::new("test-mock");

        let spec = ProvisionSpec {
            instance_type: "test.large".to_string(),
            region: "test-region".to_string(),
            count: 2,
            spot: false,
            tags: HashMap::new(),
        };

        let result = provider.provision(&spec).await.unwrap();
        assert_eq!(result.instances.len(), 2);
        assert_eq!(provider.instance_count(), 2);

        let instance_id = result.instances[0].id.clone();
        provider.deprovision(&instance_id).await.unwrap();
        assert_eq!(provider.instance_count(), 1);
    }

    #[tokio::test]
    async fn test_mock_provider_deprovision_nonexistent() {
        let provider = MockProvider::new("test-mock");

        let result = provider.deprovision("nonexistent").await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ProviderError::NotFound(_)));
    }
}
