use horizon_gcp_provider::{GcpConfig, GcpProvider};
use hpc_provider::{CapacityProvider, ProvisionSpec, QuoteRequest};
use rust_decimal_macros::dec;
use std::collections::HashMap;

#[tokio::test]
async fn test_gcp_provider_creation() {
    let config = GcpConfig::new("test-project".to_string(), "us-central1-a".to_string());
    let provider = GcpProvider::new(config).await.unwrap();
    assert_eq!(provider.name(), "gcp");
}

#[tokio::test]
async fn test_gcp_get_quote_a2_instance() {
    let config = GcpConfig::new("test-project".to_string(), "us-central1-a".to_string());
    let provider = GcpProvider::new(config).await.unwrap();

    let request = QuoteRequest {
        instance_type: "a2-highgpu-8g".to_string(),
        region: "us-central1".to_string(),
        count: 1,
        duration_hours: Some(24),
        spot: false,
    };

    let quote = provider.get_quote(&request).await.unwrap();
    assert_eq!(quote.provider, "gcp");
    assert_eq!(quote.instance_type, "a2-highgpu-8g");
    assert_eq!(quote.hourly_rate, dec!(29.39));
}

#[tokio::test]
async fn test_gcp_get_quote_preemptible() {
    let config = GcpConfig::new("test-project".to_string(), "us-central1-a".to_string());
    let provider = GcpProvider::new(config).await.unwrap();

    let request = QuoteRequest {
        instance_type: "a2-highgpu-4g".to_string(),
        region: "us-central1".to_string(),
        count: 1,
        duration_hours: None,
        spot: true,
    };

    let quote = provider.get_quote(&request).await.unwrap();
    assert!(quote.spot_rate.is_some());
    assert_eq!(quote.spot_rate.unwrap(), dec!(4.41));
}

#[tokio::test]
async fn test_gcp_provision_single_instance() {
    let config = GcpConfig::new("test-project".to_string(), "us-central1-a".to_string());
    let provider = GcpProvider::new(config).await.unwrap();

    let spec = ProvisionSpec {
        instance_type: "a2-highgpu-1g".to_string(),
        region: "us-central1".to_string(),
        count: 1,
        spot: false,
        tags: HashMap::new(),
    };

    let result = provider.provision(&spec).await.unwrap();
    assert_eq!(result.instances.len(), 1);
    assert!(result.instances[0].id.starts_with("gcp-instance-"));
}

#[tokio::test]
async fn test_gcp_provision_multiple_instances() {
    let config = GcpConfig::new("test-project".to_string(), "us-central1-a".to_string());
    let provider = GcpProvider::new(config).await.unwrap();

    let spec = ProvisionSpec {
        instance_type: "a2-highgpu-2g".to_string(),
        region: "us-central1".to_string(),
        count: 3,
        spot: false,
        tags: HashMap::new(),
    };

    let result = provider.provision(&spec).await.unwrap();
    assert_eq!(result.instances.len(), 3);
}

#[tokio::test]
async fn test_gcp_provision_preemptible() {
    let config = GcpConfig::new("test-project".to_string(), "us-central1-a".to_string());
    let provider = GcpProvider::new(config).await.unwrap();

    let spec = ProvisionSpec {
        instance_type: "a2-highgpu-1g".to_string(),
        region: "us-central1".to_string(),
        count: 1,
        spot: true,
        tags: HashMap::new(),
    };

    let result = provider.provision(&spec).await.unwrap();
    assert_eq!(result.instances.len(), 1);
    // Preemptible should have lower cost
    assert!(result.total_cost_estimate < dec!(3.67));
}

#[tokio::test]
async fn test_gcp_deprovision() {
    let config = GcpConfig::new("test-project".to_string(), "us-central1-a".to_string());
    let provider = GcpProvider::new(config).await.unwrap();

    let spec = ProvisionSpec {
        instance_type: "a2-highgpu-1g".to_string(),
        region: "us-central1".to_string(),
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
async fn test_gcp_check_spot_prices() {
    let config = GcpConfig::new("test-project".to_string(), "us-central1-a".to_string());
    let provider = GcpProvider::new(config).await.unwrap();

    let prices = provider
        .check_spot_prices("a2-highgpu-4g", "us-central1")
        .await
        .unwrap();

    assert_eq!(prices.instance_type, "a2-highgpu-4g");
    assert!(prices.min_price <= prices.current_price);
    assert!(prices.max_price >= prices.current_price);
}

#[tokio::test]
async fn test_gcp_get_quotas() {
    let config = GcpConfig::new("test-project".to_string(), "us-central1-a".to_string());
    let provider = GcpProvider::new(config).await.unwrap();

    let quotas = provider.get_quotas().await.unwrap();
    assert_eq!(quotas.max_instances, 100);
    assert_eq!(quotas.current_instances, 0);
}

#[tokio::test]
async fn test_gcp_health() {
    let config = GcpConfig::new("test-project".to_string(), "us-central1-a".to_string());
    let provider = GcpProvider::new(config).await.unwrap();

    let health = provider.health().await.unwrap();
    assert!(health.healthy);
}

#[test]
fn test_gcp_config_creation() {
    let config = GcpConfig::new("project-123".to_string(), "us-west1-b".to_string());
    assert_eq!(config.project_id, "project-123");
    assert_eq!(config.zone, "us-west1-b");
}

#[tokio::test]
async fn test_quota_enforcement() {
    let config = GcpConfig::new("test-project".to_string(), "us-central1-a".to_string());
    let provider = GcpProvider::new(config).await.unwrap();

    // Provision to exhaust quota by using unsupported type
    let spec = ProvisionSpec {
        instance_type: "unsupported-type".to_string(),
        region: "us-central1".to_string(),
        count: 1,
        spot: false,
        tags: HashMap::new(),
    };

    let result = provider.provision(&spec).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_zero_count_provision() {
    let config = GcpConfig::new("test-project".to_string(), "us-central1-a".to_string());
    let provider = GcpProvider::new(config).await.unwrap();

    let spec = ProvisionSpec {
        instance_type: "a2-highgpu-1g".to_string(),
        region: "us-central1".to_string(),
        count: 0,
        spot: false,
        tags: HashMap::new(),
    };

    let result = provider.provision(&spec).await;
    assert!(result.is_err());
}
