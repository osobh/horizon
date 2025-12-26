use horizon_aws_provider::{AwsConfig, AwsProvider};
use hpc_provider::CapacityProvider;
use rust_decimal_macros::dec;

#[tokio::test]
async fn test_check_spot_prices_p4() {
    let config = AwsConfig::new("us-east-1".to_string());
    let provider = AwsProvider::new(config).await.unwrap();

    let prices = provider
        .check_spot_prices("p4d.24xlarge", "us-east-1")
        .await
        .unwrap();

    assert_eq!(prices.instance_type, "p4d.24xlarge");
    assert_eq!(prices.region, "us-east-1");
    assert!(prices.current_price > dec!(0));
    assert!(prices.average_price > dec!(0));
    assert!(prices.min_price <= prices.current_price);
    assert!(prices.max_price >= prices.current_price);
}

#[tokio::test]
async fn test_check_spot_prices_p5() {
    let config = AwsConfig::new("us-west-2".to_string());
    let provider = AwsProvider::new(config).await.unwrap();

    let prices = provider
        .check_spot_prices("p5.48xlarge", "us-west-2")
        .await
        .unwrap();

    assert_eq!(prices.instance_type, "p5.48xlarge");
    assert!(prices.current_price > dec!(0));
}

#[tokio::test]
async fn test_spot_price_history() {
    let config = AwsConfig::new("us-east-1".to_string());
    let provider = AwsProvider::new(config).await.unwrap();

    let prices = provider
        .check_spot_prices("p4d.24xlarge", "us-east-1")
        .await
        .unwrap();

    // Average should be between min and max
    assert!(prices.average_price >= prices.min_price);
    assert!(prices.average_price <= prices.max_price);
}

#[tokio::test]
async fn test_get_quotas() {
    let config = AwsConfig::new("us-east-1".to_string());
    let provider = AwsProvider::new(config).await.unwrap();

    let quotas = provider.get_quotas().await.unwrap();

    assert!(quotas.max_instances > 0);
    assert!(quotas.max_vcpus > 0);
    assert!(quotas.max_gpus > 0);
    assert!(quotas.current_instances <= quotas.max_instances);
    assert!(quotas.current_vcpus <= quotas.max_vcpus);
    assert!(quotas.current_gpus <= quotas.max_gpus);
}

#[tokio::test]
async fn test_pricing_consistency() {
    let config = AwsConfig::new("us-east-1".to_string());
    let provider = AwsProvider::new(config).await.unwrap();

    // Get spot prices
    let spot_prices = provider
        .check_spot_prices("p4d.24xlarge", "us-east-1")
        .await
        .unwrap();

    // Spot current should be less than historical max
    assert!(spot_prices.current_price <= spot_prices.max_price);
}
