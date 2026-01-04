use horizon_azure_provider::{AzureConfig, AzureProvider};
use hpc_provider::{CapacityProvider, ProvisionSpec, QuoteRequest};
use rust_decimal_macros::dec;
use std::collections::HashMap;

#[tokio::test]
async fn test_azure_provider_creation() {
    let config = AzureConfig::new(
        "sub-123".to_string(),
        "rg-compute".to_string(),
        "eastus".to_string(),
    );
    let provider = AzureProvider::new(config).await.unwrap();
    assert_eq!(provider.name(), "azure");
}

#[tokio::test]
async fn test_azure_get_quote_ncv3() {
    let config = AzureConfig::new(
        "sub-123".to_string(),
        "rg-compute".to_string(),
        "eastus".to_string(),
    );
    let provider = AzureProvider::new(config).await.unwrap();

    let request = QuoteRequest {
        instance_type: "Standard_NC24s_v3".to_string(),
        region: "eastus".to_string(),
        count: 1,
        duration_hours: Some(24),
        spot: false,
    };

    let quote = provider.get_quote(&request).await.unwrap();
    assert_eq!(quote.provider, "azure");
    assert_eq!(quote.hourly_rate, dec!(12.24));
}

#[tokio::test]
async fn test_azure_get_quote_spot() {
    let config = AzureConfig::new(
        "sub-123".to_string(),
        "rg-compute".to_string(),
        "eastus".to_string(),
    );
    let provider = AzureProvider::new(config).await.unwrap();

    let request = QuoteRequest {
        instance_type: "Standard_NC6s_v3".to_string(),
        region: "eastus".to_string(),
        count: 1,
        duration_hours: None,
        spot: true,
    };

    let quote = provider.get_quote(&request).await.unwrap();
    assert!(quote.spot_rate.is_some());
    assert_eq!(quote.spot_rate.unwrap(), dec!(0.92));
}

#[tokio::test]
async fn test_azure_provision_single_vm() {
    let config = AzureConfig::new(
        "sub-123".to_string(),
        "rg-compute".to_string(),
        "eastus".to_string(),
    );
    let provider = AzureProvider::new(config).await.unwrap();

    let spec = ProvisionSpec {
        instance_type: "Standard_NC6s_v3".to_string(),
        region: "eastus".to_string(),
        count: 1,
        spot: false,
        tags: HashMap::new(),
    };

    let result = provider.provision(&spec).await.unwrap();
    assert_eq!(result.instances.len(), 1);
    assert!(result.instances[0].id.starts_with("azure-vm-"));
}

#[tokio::test]
async fn test_azure_provision_multiple_vms() {
    let config = AzureConfig::new(
        "sub-123".to_string(),
        "rg-compute".to_string(),
        "westus2".to_string(),
    );
    let provider = AzureProvider::new(config).await.unwrap();

    let spec = ProvisionSpec {
        instance_type: "Standard_NC12s_v3".to_string(),
        region: "westus2".to_string(),
        count: 2,
        spot: false,
        tags: HashMap::new(),
    };

    let result = provider.provision(&spec).await.unwrap();
    assert_eq!(result.instances.len(), 2);
}

#[tokio::test]
async fn test_azure_provision_spot_vm() {
    let config = AzureConfig::new(
        "sub-123".to_string(),
        "rg-compute".to_string(),
        "eastus".to_string(),
    );
    let provider = AzureProvider::new(config).await.unwrap();

    let spec = ProvisionSpec {
        instance_type: "Standard_NC6s_v3".to_string(),
        region: "eastus".to_string(),
        count: 1,
        spot: true,
        tags: HashMap::new(),
    };

    let result = provider.provision(&spec).await.unwrap();
    assert_eq!(result.instances.len(), 1);
    assert!(result.total_cost_estimate < dec!(3.06));
}

#[tokio::test]
async fn test_azure_deprovision() {
    let config = AzureConfig::new(
        "sub-123".to_string(),
        "rg-compute".to_string(),
        "eastus".to_string(),
    );
    let provider = AzureProvider::new(config).await.unwrap();

    let spec = ProvisionSpec {
        instance_type: "Standard_NC6s_v3".to_string(),
        region: "eastus".to_string(),
        count: 1,
        spot: false,
        tags: HashMap::new(),
    };

    let result = provider.provision(&spec).await.unwrap();
    let vm_id = &result.instances[0].id;

    provider.deprovision(vm_id).await.unwrap();

    let result2 = provider.deprovision(vm_id).await;
    assert!(result2.is_err());
}

#[tokio::test]
async fn test_azure_check_spot_prices() {
    let config = AzureConfig::new(
        "sub-123".to_string(),
        "rg-compute".to_string(),
        "eastus".to_string(),
    );
    let provider = AzureProvider::new(config).await.unwrap();

    let prices = provider
        .check_spot_prices("Standard_NC24s_v3", "eastus")
        .await
        .unwrap();

    assert_eq!(prices.instance_type, "Standard_NC24s_v3");
    assert!(prices.min_price <= prices.current_price);
    assert!(prices.max_price >= prices.current_price);
}

#[tokio::test]
async fn test_azure_get_quotas() {
    let config = AzureConfig::new(
        "sub-123".to_string(),
        "rg-compute".to_string(),
        "eastus".to_string(),
    );
    let provider = AzureProvider::new(config).await.unwrap();

    let quotas = provider.get_quotas().await.unwrap();
    assert_eq!(quotas.max_instances, 100);
}

#[tokio::test]
async fn test_azure_health() {
    let config = AzureConfig::new(
        "sub-123".to_string(),
        "rg-compute".to_string(),
        "eastus".to_string(),
    );
    let provider = AzureProvider::new(config).await.unwrap();

    let health = provider.health().await.unwrap();
    assert!(health.healthy);
}

#[test]
fn test_azure_config_creation() {
    let config = AzureConfig::new(
        "sub-456".to_string(),
        "rg-ml".to_string(),
        "westus".to_string(),
    );
    assert_eq!(config.subscription_id, "sub-456");
    assert_eq!(config.resource_group, "rg-ml");
    assert_eq!(config.location, "westus");
}

#[tokio::test]
async fn test_unsupported_vm_size() {
    let config = AzureConfig::new(
        "sub-123".to_string(),
        "rg-compute".to_string(),
        "eastus".to_string(),
    );
    let provider = AzureProvider::new(config).await.unwrap();

    let spec = ProvisionSpec {
        instance_type: "Invalid_VM_Size".to_string(),
        region: "eastus".to_string(),
        count: 1,
        spot: false,
        tags: HashMap::new(),
    };

    let result = provider.provision(&spec).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_zero_count_fails() {
    let config = AzureConfig::new(
        "sub-123".to_string(),
        "rg-compute".to_string(),
        "eastus".to_string(),
    );
    let provider = AzureProvider::new(config).await.unwrap();

    let spec = ProvisionSpec {
        instance_type: "Standard_NC6s_v3".to_string(),
        region: "eastus".to_string(),
        count: 0,
        spot: false,
        tags: HashMap::new(),
    };

    let result = provider.provision(&spec).await;
    assert!(result.is_err());
}
