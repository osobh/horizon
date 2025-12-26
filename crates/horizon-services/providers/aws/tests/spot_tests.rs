use horizon_aws_provider::{AwsConfig, AwsProvider};
use hpc_provider::{CapacityProvider, ProvisionSpec, QuoteRequest};
use rust_decimal_macros::dec;
use std::collections::HashMap;

#[tokio::test]
async fn test_spot_instance_pricing() {
    let config = AwsConfig::new("us-east-1".to_string());
    let provider = AwsProvider::new(config).await.unwrap();

    let request = QuoteRequest {
        instance_type: "p4d.24xlarge".to_string(),
        region: "us-east-1".to_string(),
        count: 1,
        duration_hours: None,
        spot: true,
    };

    let quote = provider.get_quote(&request).await.unwrap();
    assert!(quote.spot_rate.is_some());

    let spot = quote.spot_rate.unwrap();
    assert!(spot < quote.hourly_rate);
    assert!(spot > dec!(0));
}

#[tokio::test]
async fn test_spot_instance_provisioning() {
    let config = AwsConfig::new("us-east-1".to_string());
    let provider = AwsProvider::new(config).await.unwrap();

    let spec = ProvisionSpec {
        instance_type: "p4d.24xlarge".to_string(),
        region: "us-east-1".to_string(),
        count: 2,
        spot: true,
        tags: HashMap::new(),
    };

    let result = provider.provision(&spec).await.unwrap();
    assert_eq!(result.instances.len(), 2);
    assert!(result.total_cost_estimate > dec!(0));
}

#[tokio::test]
async fn test_spot_vs_ondemand_cost() {
    let config = AwsConfig::new("us-east-1".to_string());
    let provider = AwsProvider::new(config).await.unwrap();

    // Get on-demand quote
    let ondemand_request = QuoteRequest {
        instance_type: "p4d.24xlarge".to_string(),
        region: "us-east-1".to_string(),
        count: 1,
        duration_hours: None,
        spot: false,
    };

    let ondemand_quote = provider.get_quote(&ondemand_request).await.unwrap();

    // Get spot quote
    let spot_request = QuoteRequest {
        instance_type: "p4d.24xlarge".to_string(),
        region: "us-east-1".to_string(),
        count: 1,
        duration_hours: None,
        spot: true,
    };

    let spot_quote = provider.get_quote(&spot_request).await.unwrap();

    // Spot should be cheaper
    assert!(spot_quote.spot_rate.unwrap() < ondemand_quote.hourly_rate);
}

#[tokio::test]
async fn test_spot_price_variability() {
    let config = AwsConfig::new("us-east-1".to_string());
    let provider = AwsProvider::new(config).await.unwrap();

    let prices = provider
        .check_spot_prices("p4d.24xlarge", "us-east-1")
        .await
        .unwrap();

    // There should be some price variability
    assert!(prices.max_price > prices.min_price);
}
