use crate::config::AwsConfig;
use crate::pricing::PricingData;
use crate::quota::QuotaManager;
use crate::spot::SpotPriceManager;
use hpc_provider::{
    Availability, CapacityProvider, HealthStatus, Instance, InstanceState, ProviderError,
    ProviderResult, ProvisionResult, ProvisionSpec, Quote, QuoteRequest, ServiceQuotas,
    SpotPrices,
};
use rust_decimal::Decimal;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;

pub struct AwsProvider {
    #[allow(dead_code)]
    config: AwsConfig,
    pricing: PricingData,
    spot_manager: SpotPriceManager,
    quota_manager: QuotaManager,
    instances: Arc<Mutex<HashMap<String, Instance>>>,
    instance_counter: Arc<Mutex<u64>>,
}

impl AwsProvider {
    pub async fn new(config: AwsConfig) -> ProviderResult<Self> {
        Ok(Self {
            config,
            pricing: PricingData::new(),
            spot_manager: SpotPriceManager::new(),
            quota_manager: QuotaManager::new(),
            instances: Arc::new(Mutex::new(HashMap::new())),
            instance_counter: Arc::new(Mutex::new(0)),
        })
    }

    fn generate_instance_id(&self) -> String {
        let mut counter = self.instance_counter.lock().unwrap();
        *counter += 1;
        format!("i-{:016x}", *counter)
    }

    fn validate_instance_type(&self, instance_type: &str) -> ProviderResult<()> {
        if !self.pricing.is_supported(instance_type) {
            return Err(ProviderError::InvalidRequest(format!(
                "instance type {} not supported",
                instance_type
            )));
        }
        Ok(())
    }
}

#[async_trait::async_trait]
impl CapacityProvider for AwsProvider {
    fn name(&self) -> &str {
        "aws"
    }

    async fn get_quote(&self, request: &QuoteRequest) -> ProviderResult<Quote> {
        if request.count == 0 {
            return Err(ProviderError::InvalidRequest(
                "count must be greater than 0".to_string(),
            ));
        }

        self.validate_instance_type(&request.instance_type)?;

        let hourly_rate = self
            .pricing
            .get_on_demand_price(&request.instance_type)
            .ok_or_else(|| {
                ProviderError::InvalidRequest(format!(
                    "pricing not available for {}",
                    request.instance_type
                ))
            })?;

        let spot_rate = if request.spot {
            self.pricing.get_spot_base_price(&request.instance_type)
        } else {
            None
        };

        // Determine availability based on quota
        let availability = if self.quota_manager.has_capacity(request.count) {
            Availability::Available
        } else {
            Availability::Limited
        };

        Ok(Quote {
            provider: "aws".to_string(),
            instance_type: request.instance_type.clone(),
            region: request.region.clone(),
            hourly_rate,
            spot_rate,
            availability,
            lead_time_hours: 0,
        })
    }

    async fn provision(&self, spec: &ProvisionSpec) -> ProviderResult<ProvisionResult> {
        if spec.count == 0 {
            return Err(ProviderError::InvalidRequest(
                "count must be greater than 0".to_string(),
            ));
        }

        self.validate_instance_type(&spec.instance_type)?;

        if !self.quota_manager.has_capacity(spec.count) {
            return Err(ProviderError::QuotaExceeded(
                "insufficient capacity quota".to_string(),
            ));
        }

        let rate = if spec.spot {
            self.pricing
                .get_spot_base_price(&spec.instance_type)
                .unwrap_or_else(|| {
                    self.pricing
                        .get_on_demand_price(&spec.instance_type)
                        .unwrap()
                })
        } else {
            self.pricing
                .get_on_demand_price(&spec.instance_type)
                .unwrap()
        };

        let mut instances = Vec::new();
        let mut instances_map = self.instances.lock().unwrap();

        for i in 0..spec.count {
            let instance_id = self.generate_instance_id();
            let instance = Instance {
                id: instance_id.clone(),
                instance_type: spec.instance_type.clone(),
                region: spec.region.clone(),
                public_ip: Some(format!("54.{}.{}.{}", i / 65536, (i / 256) % 256, i % 256)),
                private_ip: format!("10.0.{}.{}", (i / 256) % 256, i % 256),
                state: InstanceState::Pending,
            };
            instances_map.insert(instance_id, instance.clone());
            instances.push(instance);
        }

        self.quota_manager.increment_instances(spec.count);

        let total_cost_estimate = rate * Decimal::from(spec.count);

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
        if instances.remove(instance_id).is_some() {
            self.quota_manager.decrement_instances(1);
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
        self.spot_manager.get_spot_prices(instance_type, region)
    }

    async fn get_quotas(&self) -> ProviderResult<ServiceQuotas> {
        self.quota_manager.get_quotas()
    }

    async fn health(&self) -> ProviderResult<HealthStatus> {
        // In a real implementation, this would check AWS service health
        Ok(HealthStatus {
            healthy: true,
            message: "AWS EC2 service operational".to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_aws_provider_creation() {
        let config = AwsConfig::new("us-east-1".to_string());
        let provider = AwsProvider::new(config).await.unwrap();
        assert_eq!(provider.name(), "aws");
    }

    #[tokio::test]
    async fn test_validate_instance_type() {
        let config = AwsConfig::new("us-east-1".to_string());
        let provider = AwsProvider::new(config).await.unwrap();
        assert!(provider.validate_instance_type("p4d.24xlarge").is_ok());
        assert!(provider.validate_instance_type("invalid.type").is_err());
    }

    #[tokio::test]
    async fn test_generate_instance_id() {
        let config = AwsConfig::new("us-east-1".to_string());
        let provider = AwsProvider::new(config).await.unwrap();
        let id1 = provider.generate_instance_id();
        let id2 = provider.generate_instance_id();
        assert!(id1.starts_with("i-"));
        assert!(id2.starts_with("i-"));
        assert_ne!(id1, id2);
    }

    #[tokio::test]
    async fn test_get_quote_invalid_count() {
        let config = AwsConfig::new("us-east-1".to_string());
        let provider = AwsProvider::new(config).await.unwrap();

        let request = QuoteRequest {
            instance_type: "p4d.24xlarge".to_string(),
            region: "us-east-1".to_string(),
            count: 0,
            duration_hours: None,
            spot: false,
        };

        let result = provider.get_quote(&request).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_provision_and_deprovision() {
        let config = AwsConfig::new("us-east-1".to_string());
        let provider = AwsProvider::new(config).await.unwrap();

        let spec = ProvisionSpec {
            instance_type: "p4d.24xlarge".to_string(),
            region: "us-east-1".to_string(),
            count: 2,
            spot: false,
            tags: HashMap::new(),
        };

        let result = provider.provision(&spec).await.unwrap();
        assert_eq!(result.instances.len(), 2);

        let instance_id = &result.instances[0].id;
        provider.deprovision(instance_id).await.unwrap();

        // Try to deprovision again should fail
        let result2 = provider.deprovision(instance_id).await;
        assert!(result2.is_err());
    }
}
