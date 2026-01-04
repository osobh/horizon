use cost_ingestor::{
    ingest::{AwsCurNormalizer, AzureEaNormalizer, GcpBillingNormalizer, OnPremNormalizer},
    models::Provider,
    normalize::{BillingNormalizer, RawBillingData},
};
use rust_decimal_macros::dec;

const AWS_SAMPLE: &str = r#"lineItem/UsageAccountId,lineItem/ProductCode,lineItem/ResourceId,lineItem/UsageStartDate,lineItem/UsageEndDate,lineItem/UnblendedCost,lineItem/CurrencyCode,lineItem/LineItemType,product/region
123456789012,AmazonEC2,i-123,2024-01-01T00:00:00Z,2024-01-01T01:00:00Z,1.50,USD,Usage,us-east-1
"#;

#[test]
fn test_aws_cur_normalization() {
    let normalizer = AwsCurNormalizer::new();
    let raw = RawBillingData {
        provider: Provider::Aws,
        data: serde_json::Value::String(AWS_SAMPLE.to_string()),
    };

    let result = normalizer.normalize(&raw);
    assert!(result.is_ok());

    let records = result.unwrap();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].amount, dec!(1.50));
    assert_eq!(records[0].provider, Provider::Aws);
}

#[test]
fn test_gcp_billing_normalization() {
    let normalizer = GcpBillingNormalizer::new();
    let data = serde_json::json!([{
        "billing_account_id": "012345-6789AB-CDEF01",
        "service": {
            "id": "services/6F81-5844-456A",
            "description": "Compute Engine"
        },
        "sku": {
            "id": "0000-0000-0000",
            "description": "N1 Predefined Instance Core"
        },
        "usage_start_time": "2024-01-01T00:00:00Z",
        "usage_end_time": "2024-01-01T01:00:00Z",
        "project": {
            "id": "my-project-123",
            "name": "My Project"
        },
        "cost": 1.50,
        "currency": "USD",
        "usage": {
            "amount": 1.0,
            "unit": "hour"
        },
        "credits": null
    }]);

    let raw = RawBillingData {
        provider: Provider::Gcp,
        data,
    };

    let result = normalizer.normalize(&raw);
    assert!(result.is_ok());

    let records = result.unwrap();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].amount, dec!(1.50));
}

#[test]
fn test_azure_ea_normalization() {
    let normalizer = AzureEaNormalizer::new();
    let data = serde_json::Value::String(
        r#"[{
        "SubscriptionId": "12345678-1234-1234-1234-123456789abc",
        "ResourceGroup": "my-resource-group",
        "ResourceId": "/subscriptions/.../virtualMachines/vm1",
        "ServiceName": "Virtual Machines",
        "UsageDateTime": "2024-01-01T00:00:00Z",
        "Cost": 10.50,
        "Currency": "USD",
        "MeterName": "D4s v3",
        "Quantity": 1.0,
        "UnitOfMeasure": "Hour"
    }]"#
        .to_string(),
    );

    let raw = RawBillingData {
        provider: Provider::Azure,
        data,
    };

    let result = normalizer.normalize(&raw);
    assert!(result.is_ok());

    let records = result.unwrap();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].amount, dec!(10.50));
}

#[test]
fn test_onprem_normalization() {
    let normalizer = OnPremNormalizer::new();
    let data = serde_json::json!([{
        "server_id": "server-001",
        "location": "datacenter-1",
        "usage_start": "2024-01-01T00:00:00Z",
        "usage_end": "2024-01-01T01:00:00Z",
        "power_kwh": 5.0,
        "cooling_kwh": 3.0,
        "power_rate_per_kwh": 0.15,
        "cooling_rate_per_kwh": 0.10,
        "depreciation_cost": 1.50,
        "maintenance_cost": 0.50,
        "currency": "USD"
    }]);

    let raw = RawBillingData {
        provider: Provider::OnPrem,
        data,
    };

    let result = normalizer.normalize(&raw);
    assert!(result.is_ok());

    let records = result.unwrap();
    assert_eq!(records.len(), 1);
    assert!(records[0].amount > dec!(0));
}

#[test]
fn test_multiple_providers_in_sequence() {
    let aws_normalizer = AwsCurNormalizer::new();
    let gcp_normalizer = GcpBillingNormalizer::new();

    let aws_raw = RawBillingData {
        provider: Provider::Aws,
        data: serde_json::Value::String(AWS_SAMPLE.to_string()),
    };

    let aws_records = aws_normalizer.normalize(&aws_raw).unwrap();
    assert!(!aws_records.is_empty());

    let gcp_data = serde_json::json!([{
        "billing_account_id": "test",
        "service": {"id": "test", "description": "Test Service"},
        "sku": {"id": "test", "description": "Test SKU"},
        "usage_start_time": "2024-01-01T00:00:00Z",
        "usage_end_time": "2024-01-01T01:00:00Z",
        "project": {"id": "test-project", "name": "Test"},
        "cost": 5.0,
        "currency": "USD",
        "usage": null,
        "credits": null
    }]);

    let gcp_raw = RawBillingData {
        provider: Provider::Gcp,
        data: gcp_data,
    };

    let gcp_records = gcp_normalizer.normalize(&gcp_raw).unwrap();
    assert!(!gcp_records.is_empty());
}
