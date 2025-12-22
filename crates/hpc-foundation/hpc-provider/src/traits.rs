use crate::{
    HealthStatus, ProviderResult, ProvisionResult, ProvisionSpec, Quote, QuoteRequest,
    ServiceQuotas, SpotPrices,
};

/// Core trait that all capacity providers must implement
#[async_trait::async_trait]
pub trait CapacityProvider: Send + Sync {
    /// Get provider name/identifier
    fn name(&self) -> &str;

    /// Get pricing quote for requested capacity
    ///
    /// # Arguments
    /// * `request` - Quote request with instance type, region, count, etc.
    ///
    /// # Returns
    /// * `Ok(Quote)` - Pricing quote with hourly rates and availability
    /// * `Err(ProviderError)` - If request is invalid or quote cannot be obtained
    async fn get_quote(&self, request: &QuoteRequest) -> ProviderResult<Quote>;

    /// Provision capacity according to specification
    ///
    /// # Arguments
    /// * `spec` - Provisioning specification including instance type, count, tags
    ///
    /// # Returns
    /// * `Ok(ProvisionResult)` - List of provisioned instances and cost estimate
    /// * `Err(ProviderError)` - If provisioning fails
    async fn provision(&self, spec: &ProvisionSpec) -> ProviderResult<ProvisionResult>;

    /// Deprovision/terminate an instance
    ///
    /// # Arguments
    /// * `instance_id` - Unique identifier of the instance to deprovision
    ///
    /// # Returns
    /// * `Ok(())` - Instance successfully deprovisioned
    /// * `Err(ProviderError)` - If deprovisioning fails
    async fn deprovision(&self, instance_id: &str) -> ProviderResult<()>;

    /// Check current spot/preemptible instance prices
    ///
    /// # Arguments
    /// * `instance_type` - Instance type to check (e.g., "p4.xlarge")
    /// * `region` - Region to check prices in
    ///
    /// # Returns
    /// * `Ok(SpotPrices)` - Current, average, min, max spot prices
    /// * `Err(ProviderError)` - If prices cannot be retrieved
    async fn check_spot_prices(
        &self,
        instance_type: &str,
        region: &str,
    ) -> ProviderResult<SpotPrices>;

    /// Get service quotas/limits for this provider
    ///
    /// # Returns
    /// * `Ok(ServiceQuotas)` - Current and maximum resource quotas
    /// * `Err(ProviderError)` - If quotas cannot be retrieved
    async fn get_quotas(&self) -> ProviderResult<ServiceQuotas>;

    /// Health check for provider availability
    ///
    /// # Returns
    /// * `Ok(HealthStatus)` - Provider health status and message
    /// * `Err(ProviderError)` - If health check fails
    async fn health(&self) -> ProviderResult<HealthStatus>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Availability, Instance, InstanceState, ProviderError};
    use rust_decimal::Decimal;
    use rust_decimal_macros::dec;
    use std::collections::HashMap;

    struct TestProvider;

    #[async_trait::async_trait]
    impl CapacityProvider for TestProvider {
        fn name(&self) -> &str {
            "test"
        }

        async fn get_quote(&self, _request: &QuoteRequest) -> ProviderResult<Quote> {
            Ok(Quote {
                provider: "test".to_string(),
                instance_type: "test.large".to_string(),
                region: "test-region".to_string(),
                hourly_rate: dec!(1.00),
                spot_rate: None,
                availability: Availability::Available,
                lead_time_hours: 0,
            })
        }

        async fn provision(&self, spec: &ProvisionSpec) -> ProviderResult<ProvisionResult> {
            let instances = vec![Instance {
                id: "test-instance".to_string(),
                instance_type: spec.instance_type.clone(),
                region: spec.region.clone(),
                public_ip: None,
                private_ip: "127.0.0.1".to_string(),
                state: InstanceState::Pending,
            }];

            Ok(ProvisionResult {
                instances,
                total_cost_estimate: dec!(10.00),
            })
        }

        async fn deprovision(&self, _instance_id: &str) -> ProviderResult<()> {
            Ok(())
        }

        async fn check_spot_prices(
            &self,
            instance_type: &str,
            region: &str,
        ) -> ProviderResult<SpotPrices> {
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
                current_instances: 0,
                max_vcpus: 400,
                current_vcpus: 0,
                max_gpus: 50,
                current_gpus: 0,
            })
        }

        async fn health(&self) -> ProviderResult<HealthStatus> {
            Ok(HealthStatus {
                healthy: true,
                message: "OK".to_string(),
            })
        }
    }

    #[tokio::test]
    async fn test_trait_implementation() {
        let provider = TestProvider;
        assert_eq!(provider.name(), "test");
    }

    #[tokio::test]
    async fn test_trait_get_quote() {
        let provider = TestProvider;
        let request = QuoteRequest {
            instance_type: "test.large".to_string(),
            region: "test-region".to_string(),
            count: 1,
            duration_hours: None,
            spot: false,
        };

        let quote = provider.get_quote(&request).await.unwrap();
        assert_eq!(quote.provider, "test");
        assert_eq!(quote.hourly_rate, dec!(1.00));
    }

    #[tokio::test]
    async fn test_trait_provision() {
        let provider = TestProvider;
        let spec = ProvisionSpec {
            instance_type: "test.large".to_string(),
            region: "test-region".to_string(),
            count: 1,
            spot: false,
            tags: HashMap::new(),
        };

        let result = provider.provision(&spec).await.unwrap();
        assert_eq!(result.instances.len(), 1);
        assert_eq!(result.total_cost_estimate, dec!(10.00));
    }

    #[tokio::test]
    async fn test_trait_deprovision() {
        let provider = TestProvider;
        let result = provider.deprovision("test-instance").await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_trait_spot_prices() {
        let provider = TestProvider;
        let prices = provider
            .check_spot_prices("test.large", "test-region")
            .await
            .unwrap();
        assert_eq!(prices.current_price, dec!(0.50));
    }

    #[tokio::test]
    async fn test_trait_quotas() {
        let provider = TestProvider;
        let quotas = provider.get_quotas().await.unwrap();
        assert_eq!(quotas.max_instances, 100);
    }

    #[tokio::test]
    async fn test_trait_health() {
        let provider = TestProvider;
        let health = provider.health().await.unwrap();
        assert!(health.healthy);
        assert_eq!(health.message, "OK");
    }
}
