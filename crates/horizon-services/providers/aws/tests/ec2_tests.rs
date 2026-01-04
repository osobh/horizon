use horizon_aws_provider::{AwsConfig, AwsProvider};
use hpc_provider::{Availability, CapacityProvider, InstanceState, ProvisionSpec, QuoteRequest};
use rust_decimal_macros::dec;
use std::collections::HashMap;

#[tokio::test]
async fn test_aws_provider_creation() {
    let config = AwsConfig::new("us-east-1".to_string());
    let provider = AwsProvider::new(config).await.unwrap();
    assert_eq!(provider.name(), "aws");
}

#[tokio::test]
async fn test_aws_get_quote_p4_instance() {
    let config = AwsConfig::new("us-east-1".to_string());
    let provider = AwsProvider::new(config).await.unwrap();

    let request = QuoteRequest {
        instance_type: "p4d.24xlarge".to_string(),
        region: "us-east-1".to_string(),
        count: 1,
        duration_hours: Some(24),
        spot: false,
    };

    let quote = provider.get_quote(&request).await.unwrap();
    assert_eq!(quote.provider, "aws");
    assert_eq!(quote.instance_type, "p4d.24xlarge");
    assert_eq!(quote.region, "us-east-1");
    assert!(quote.hourly_rate > dec!(0));
    assert!(matches!(
        quote.availability,
        Availability::Available | Availability::Limited
    ));
}

#[tokio::test]
async fn test_aws_get_quote_p5_instance() {
    let config = AwsConfig::new("us-east-1".to_string());
    let provider = AwsProvider::new(config).await.unwrap();

    let request = QuoteRequest {
        instance_type: "p5.48xlarge".to_string(),
        region: "us-east-1".to_string(),
        count: 1,
        duration_hours: None,
        spot: false,
    };

    let quote = provider.get_quote(&request).await.unwrap();
    assert_eq!(quote.instance_type, "p5.48xlarge");
    assert!(quote.hourly_rate > dec!(0));
}

#[tokio::test]
async fn test_aws_get_quote_with_spot() {
    let config = AwsConfig::new("us-west-2".to_string());
    let provider = AwsProvider::new(config).await.unwrap();

    let request = QuoteRequest {
        instance_type: "p4d.24xlarge".to_string(),
        region: "us-west-2".to_string(),
        count: 2,
        duration_hours: Some(12),
        spot: true,
    };

    let quote = provider.get_quote(&request).await.unwrap();
    assert!(quote.spot_rate.is_some());
    let spot_rate = quote.spot_rate.unwrap();
    assert!(spot_rate < quote.hourly_rate);
    assert!(spot_rate > dec!(0));
}

#[tokio::test]
async fn test_aws_provision_single_instance() {
    let config = AwsConfig::new("us-east-1".to_string());
    let provider = AwsProvider::new(config).await.unwrap();

    let mut tags = HashMap::new();
    tags.insert("project".to_string(), "ml-training".to_string());

    let spec = ProvisionSpec {
        instance_type: "p4d.24xlarge".to_string(),
        region: "us-east-1".to_string(),
        count: 1,
        spot: false,
        tags,
    };

    let result = provider.provision(&spec).await.unwrap();
    assert_eq!(result.instances.len(), 1);
    assert!(result.total_cost_estimate > dec!(0));

    let instance = &result.instances[0];
    assert!(instance.id.starts_with("i-"));
    assert_eq!(instance.instance_type, "p4d.24xlarge");
    assert_eq!(instance.region, "us-east-1");
    assert!(matches!(
        instance.state,
        InstanceState::Pending | InstanceState::Running
    ));
}

#[tokio::test]
async fn test_aws_provision_multiple_instances() {
    let config = AwsConfig::new("us-west-2".to_string());
    let provider = AwsProvider::new(config).await.unwrap();

    let spec = ProvisionSpec {
        instance_type: "p4d.24xlarge".to_string(),
        region: "us-west-2".to_string(),
        count: 3,
        spot: false,
        tags: HashMap::new(),
    };

    let result = provider.provision(&spec).await.unwrap();
    assert_eq!(result.instances.len(), 3);

    for instance in &result.instances {
        assert!(instance.id.starts_with("i-"));
        assert_eq!(instance.instance_type, "p4d.24xlarge");
    }
}

#[tokio::test]
async fn test_aws_provision_spot_instance() {
    let config = AwsConfig::new("us-east-1".to_string());
    let provider = AwsProvider::new(config).await.unwrap();

    let spec = ProvisionSpec {
        instance_type: "p4d.24xlarge".to_string(),
        region: "us-east-1".to_string(),
        count: 1,
        spot: true,
        tags: HashMap::new(),
    };

    let result = provider.provision(&spec).await.unwrap();
    assert_eq!(result.instances.len(), 1);
    // Spot instances should have lower estimated cost
}

#[tokio::test]
async fn test_aws_deprovision_instance() {
    let config = AwsConfig::new("us-east-1".to_string());
    let provider = AwsProvider::new(config).await.unwrap();

    // First provision an instance
    let spec = ProvisionSpec {
        instance_type: "p4d.24xlarge".to_string(),
        region: "us-east-1".to_string(),
        count: 1,
        spot: false,
        tags: HashMap::new(),
    };

    let result = provider.provision(&spec).await.unwrap();
    let instance_id = &result.instances[0].id;

    // Then deprovision it
    provider.deprovision(instance_id).await.unwrap();
}

#[tokio::test]
async fn test_aws_health_check() {
    let config = AwsConfig::new("us-east-1".to_string());
    let provider = AwsProvider::new(config).await.unwrap();

    let health = provider.health().await.unwrap();
    assert!(health.healthy);
    assert!(!health.message.is_empty());
}
