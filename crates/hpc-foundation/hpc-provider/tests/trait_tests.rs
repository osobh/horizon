use hpc_provider::{
    Availability, CapacityProvider, HealthStatus, Instance, InstanceState, ProviderError,
    ProviderResult, ProvisionResult, ProvisionSpec, Quote, QuoteRequest, ServiceQuotas,
    SpotPrices,
};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::collections::HashMap;

// Mock provider for testing trait implementation
struct MockProvider {
    name: String,
    healthy: bool,
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
            spot_rate: if request.spot {
                Some(dec!(0.50))
            } else {
                None
            },
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

        let instances: Vec<Instance> = (0..spec.count)
            .map(|i| Instance {
                id: format!("instance-{}", i),
                instance_type: spec.instance_type.clone(),
                region: spec.region.clone(),
                public_ip: Some(format!("10.0.{}.{}", i / 256, i % 256)),
                private_ip: format!("192.168.{}.{}", i / 256, i % 256),
                state: InstanceState::Pending,
            })
            .collect();

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
        Ok(())
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
        Ok(ServiceQuotas {
            max_instances: 100,
            current_instances: 10,
            max_vcpus: 400,
            current_vcpus: 40,
            max_gpus: 50,
            current_gpus: 5,
        })
    }

    async fn health(&self) -> ProviderResult<HealthStatus> {
        Ok(HealthStatus {
            healthy: self.healthy,
            message: if self.healthy {
                "All systems operational".to_string()
            } else {
                "Service degraded".to_string()
            },
        })
    }
}

#[tokio::test]
async fn test_provider_name() {
    let provider = MockProvider {
        name: "test-provider".to_string(),
        healthy: true,
    };
    assert_eq!(provider.name(), "test-provider");
}

#[tokio::test]
async fn test_get_quote_success() {
    let provider = MockProvider {
        name: "test-provider".to_string(),
        healthy: true,
    };

    let request = QuoteRequest {
        instance_type: "p4.xlarge".to_string(),
        region: "us-east-1".to_string(),
        count: 2,
        duration_hours: Some(24),
        spot: false,
    };

    let quote = provider.get_quote(&request).await.unwrap();
    assert_eq!(quote.provider, "test-provider");
    assert_eq!(quote.instance_type, "p4.xlarge");
    assert_eq!(quote.region, "us-east-1");
    assert_eq!(quote.hourly_rate, dec!(1.50));
    assert_eq!(quote.spot_rate, None);
    assert!(matches!(quote.availability, Availability::Available));
}

#[tokio::test]
async fn test_get_quote_with_spot() {
    let provider = MockProvider {
        name: "test-provider".to_string(),
        healthy: true,
    };

    let request = QuoteRequest {
        instance_type: "p4.xlarge".to_string(),
        region: "us-east-1".to_string(),
        count: 1,
        duration_hours: None,
        spot: true,
    };

    let quote = provider.get_quote(&request).await.unwrap();
    assert_eq!(quote.spot_rate, Some(dec!(0.50)));
}

#[tokio::test]
async fn test_get_quote_zero_count_fails() {
    let provider = MockProvider {
        name: "test-provider".to_string(),
        healthy: true,
    };

    let request = QuoteRequest {
        instance_type: "p4.xlarge".to_string(),
        region: "us-east-1".to_string(),
        count: 0,
        duration_hours: Some(24),
        spot: false,
    };

    let result = provider.get_quote(&request).await;
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), ProviderError::InvalidRequest(_)));
}

#[tokio::test]
async fn test_provision_success() {
    let provider = MockProvider {
        name: "test-provider".to_string(),
        healthy: true,
    };

    let mut tags = HashMap::new();
    tags.insert("environment".to_string(), "test".to_string());

    let spec = ProvisionSpec {
        instance_type: "p4.xlarge".to_string(),
        region: "us-east-1".to_string(),
        count: 3,
        spot: false,
        tags,
    };

    let result = provider.provision(&spec).await.unwrap();
    assert_eq!(result.instances.len(), 3);
    assert_eq!(result.total_cost_estimate, dec!(4.50)); // 1.50 * 3

    for (i, instance) in result.instances.iter().enumerate() {
        assert_eq!(instance.id, format!("instance-{}", i));
        assert_eq!(instance.instance_type, "p4.xlarge");
        assert_eq!(instance.region, "us-east-1");
        assert!(matches!(instance.state, InstanceState::Pending));
    }
}

#[tokio::test]
async fn test_provision_zero_count_fails() {
    let provider = MockProvider {
        name: "test-provider".to_string(),
        healthy: true,
    };

    let spec = ProvisionSpec {
        instance_type: "p4.xlarge".to_string(),
        region: "us-east-1".to_string(),
        count: 0,
        spot: false,
        tags: HashMap::new(),
    };

    let result = provider.provision(&spec).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_deprovision_success() {
    let provider = MockProvider {
        name: "test-provider".to_string(),
        healthy: true,
    };

    let result = provider.deprovision("instance-123").await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_deprovision_empty_id_fails() {
    let provider = MockProvider {
        name: "test-provider".to_string(),
        healthy: true,
    };

    let result = provider.deprovision("").await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_check_spot_prices() {
    let provider = MockProvider {
        name: "test-provider".to_string(),
        healthy: true,
    };

    let prices = provider
        .check_spot_prices("p4.xlarge", "us-east-1")
        .await
        .unwrap();

    assert_eq!(prices.instance_type, "p4.xlarge");
    assert_eq!(prices.region, "us-east-1");
    assert_eq!(prices.current_price, dec!(0.50));
    assert_eq!(prices.average_price, dec!(0.55));
    assert!(prices.min_price < prices.max_price);
}

#[tokio::test]
async fn test_check_spot_prices_empty_type_fails() {
    let provider = MockProvider {
        name: "test-provider".to_string(),
        healthy: true,
    };

    let result = provider.check_spot_prices("", "us-east-1").await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_get_quotas() {
    let provider = MockProvider {
        name: "test-provider".to_string(),
        healthy: true,
    };

    let quotas = provider.get_quotas().await.unwrap();
    assert_eq!(quotas.max_instances, 100);
    assert_eq!(quotas.current_instances, 10);
    assert!(quotas.current_instances < quotas.max_instances);
    assert!(quotas.current_vcpus < quotas.max_vcpus);
    assert!(quotas.current_gpus < quotas.max_gpus);
}

#[tokio::test]
async fn test_health_healthy() {
    let provider = MockProvider {
        name: "test-provider".to_string(),
        healthy: true,
    };

    let health = provider.health().await.unwrap();
    assert!(health.healthy);
    assert_eq!(health.message, "All systems operational");
}

#[tokio::test]
async fn test_health_unhealthy() {
    let provider = MockProvider {
        name: "test-provider".to_string(),
        healthy: false,
    };

    let health = provider.health().await.unwrap();
    assert!(!health.healthy);
    assert_eq!(health.message, "Service degraded");
}

#[test]
fn test_availability_variants() {
    let available = Availability::Available;
    let limited = Availability::Limited;
    let unavailable = Availability::Unavailable;

    assert!(matches!(available, Availability::Available));
    assert!(matches!(limited, Availability::Limited));
    assert!(matches!(unavailable, Availability::Unavailable));
}

#[test]
fn test_instance_state_variants() {
    assert!(matches!(InstanceState::Pending, InstanceState::Pending));
    assert!(matches!(InstanceState::Running, InstanceState::Running));
    assert!(matches!(InstanceState::Stopping, InstanceState::Stopping));
    assert!(matches!(InstanceState::Stopped, InstanceState::Stopped));
    assert!(matches!(
        InstanceState::Terminated,
        InstanceState::Terminated
    ));
}
