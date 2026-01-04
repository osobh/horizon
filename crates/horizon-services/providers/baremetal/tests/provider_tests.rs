use horizon_baremetal_provider::{
    calculate_hourly_rate, BareMetalConfig, BareMetalProvider, ServerStatus,
};
use hpc_provider::{CapacityProvider, InstanceState, ProvisionSpec, QuoteRequest};
use rust_decimal_macros::dec;
use std::collections::HashMap;

#[tokio::test]
async fn test_baremetal_provider_creation() {
    let config = BareMetalConfig::new("dc1".to_string());
    let provider = BareMetalProvider::new(config).await.unwrap();
    assert_eq!(provider.name(), "baremetal");
}

#[tokio::test]
async fn test_baremetal_get_quote() {
    let config = BareMetalConfig::new("dc1".to_string());
    let provider = BareMetalProvider::new(config).await.unwrap();

    let request = QuoteRequest {
        instance_type: "A100".to_string(),
        region: "dc1".to_string(),
        count: 1,
        duration_hours: None,
        spot: false,
    };

    let quote = provider.get_quote(&request).await.unwrap();
    assert_eq!(quote.provider, "baremetal");
    assert_eq!(quote.instance_type, "A100");
    assert!(quote.hourly_rate > dec!(0));
    assert_eq!(quote.spot_rate, None);
}

#[tokio::test]
async fn test_baremetal_provision_single_server() {
    let config = BareMetalConfig::new("dc1".to_string());
    let provider = BareMetalProvider::new(config).await.unwrap();

    let spec = ProvisionSpec {
        instance_type: "A100".to_string(),
        region: "dc1".to_string(),
        count: 1,
        spot: false,
        tags: HashMap::new(),
    };

    let result = provider.provision(&spec).await.unwrap();
    assert_eq!(result.instances.len(), 1);
    assert_eq!(result.instances[0].state, InstanceState::Running);
}

#[tokio::test]
async fn test_baremetal_provision_multiple_servers() {
    let config = BareMetalConfig::new("dc1".to_string());
    let provider = BareMetalProvider::new(config).await.unwrap();

    let spec = ProvisionSpec {
        instance_type: "V100".to_string(),
        region: "dc1".to_string(),
        count: 2,
        spot: false,
        tags: HashMap::new(),
    };

    let result = provider.provision(&spec).await.unwrap();
    assert_eq!(result.instances.len(), 2);
}

#[tokio::test]
async fn test_baremetal_deprovision() {
    let config = BareMetalConfig::new("dc1".to_string());
    let provider = BareMetalProvider::new(config).await.unwrap();

    let spec = ProvisionSpec {
        instance_type: "A100".to_string(),
        region: "dc1".to_string(),
        count: 1,
        spot: false,
        tags: HashMap::new(),
    };

    let result = provider.provision(&spec).await.unwrap();
    let instance_id = &result.instances[0].id;

    provider.deprovision(instance_id).await.unwrap();

    let result2 = provider.deprovision(instance_id).await;
    assert!(result2.is_err());
}

#[tokio::test]
async fn test_baremetal_check_pricing() {
    let config = BareMetalConfig::new("dc1".to_string());
    let provider = BareMetalProvider::new(config).await.unwrap();

    let prices = provider.check_spot_prices("H100", "dc1").await.unwrap();
    assert_eq!(prices.instance_type, "H100");
    // Bare-metal has fixed pricing
    assert_eq!(prices.current_price, prices.min_price);
    assert_eq!(prices.current_price, prices.max_price);
}

#[tokio::test]
async fn test_baremetal_get_quotas() {
    let config = BareMetalConfig::new("dc1".to_string());
    let provider = BareMetalProvider::new(config).await.unwrap();

    let quotas = provider.get_quotas().await.unwrap();
    assert_eq!(quotas.max_instances, 10);
}

#[tokio::test]
async fn test_baremetal_health() {
    let config = BareMetalConfig::new("dc1".to_string());
    let provider = BareMetalProvider::new(config).await.unwrap();

    let health = provider.health().await.unwrap();
    assert!(health.healthy);
}

#[test]
fn test_calculate_hourly_rate_h100() {
    let rate = calculate_hourly_rate("H100", 8);
    assert_eq!(rate, dec!(80.0));
}

#[test]
fn test_calculate_hourly_rate_a100() {
    let rate = calculate_hourly_rate("A100", 8);
    assert_eq!(rate, dec!(28.0));
}

#[test]
fn test_calculate_hourly_rate_v100() {
    let rate = calculate_hourly_rate("V100", 8);
    assert_eq!(rate, dec!(20.0));
}

#[test]
fn test_server_status() {
    assert_eq!(ServerStatus::Available, ServerStatus::Available);
    assert_ne!(ServerStatus::Available, ServerStatus::InUse);
}

#[tokio::test]
async fn test_insufficient_capacity() {
    let config = BareMetalConfig::new("dc1".to_string());
    let provider = BareMetalProvider::new(config).await.unwrap();

    // Try to provision more than available
    let spec = ProvisionSpec {
        instance_type: "H100".to_string(),
        region: "dc1".to_string(),
        count: 20,
        spot: false,
        tags: HashMap::new(),
    };

    let result = provider.provision(&spec).await;
    assert!(result.is_err());
}
