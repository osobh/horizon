use cost_ingestor::{
    models::{CreateBillingRecord, Provider},
    normalize::{GenericBillingRecord, NormalizedBillingSchema, parse_decimal, parse_iso_datetime},
};
use chrono::Utc;
use rust_decimal_macros::dec;
use std::collections::HashMap;

#[test]
fn test_parse_iso_datetime_valid() {
    let result = parse_iso_datetime("2024-01-01T00:00:00Z");
    assert!(result.is_ok());
}

#[test]
fn test_parse_iso_datetime_invalid() {
    let result = parse_iso_datetime("invalid-date");
    assert!(result.is_err());
}

#[test]
fn test_parse_decimal_valid() {
    let result = parse_decimal("123.45");
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), dec!(123.45));
}

#[test]
fn test_parse_decimal_invalid() {
    let result = parse_decimal("not-a-number");
    assert!(result.is_err());
}

#[test]
fn test_generic_billing_record_to_create_record() {
    let now = Utc::now();
    let later = now + chrono::Duration::hours(1);

    let generic = GenericBillingRecord {
        account_id: Some("test-account".to_string()),
        service: Some("TestService".to_string()),
        resource_id: Some("resource-123".to_string()),
        usage_start: now,
        usage_end: later,
        amount: dec!(100.0),
        currency: "USD".to_string(),
        metadata: HashMap::new(),
    };

    let result = generic.into_create_record(Provider::Aws);
    assert!(result.is_ok());

    let record = result.unwrap();
    assert_eq!(record.provider, Provider::Aws);
    assert_eq!(record.amount, dec!(100.0));
    assert_eq!(record.account_id, Some("test-account".to_string()));
    assert_eq!(record.service, Some("TestService".to_string()));
}

#[test]
fn test_generic_record_with_metadata() {
    let now = Utc::now();
    let later = now + chrono::Duration::hours(1);

    let mut metadata = HashMap::new();
    metadata.insert("key1".to_string(), "value1".to_string());
    metadata.insert("key2".to_string(), "value2".to_string());

    let generic = GenericBillingRecord {
        account_id: None,
        service: None,
        resource_id: None,
        usage_start: now,
        usage_end: later,
        amount: dec!(50.0),
        currency: "EUR".to_string(),
        metadata: metadata.clone(),
    };

    let result = generic.into_create_record(Provider::Gcp);
    assert!(result.is_ok());

    let record = result.unwrap();
    assert_eq!(record.currency, "EUR");

    let extracted_metadata: HashMap<String, String> =
        serde_json::from_value(record.raw_data).unwrap();
    assert_eq!(extracted_metadata.get("key1"), Some(&"value1".to_string()));
}

#[test]
fn test_normalized_schema_empty() {
    let _schema = NormalizedBillingSchema::new();
    // Schema created successfully
    assert!(true);
}

#[test]
fn test_create_billing_record_validation_negative_amount() {
    let now = Utc::now();
    let later = now + chrono::Duration::hours(1);

    let record = CreateBillingRecord::new(Provider::Aws, now, later, dec!(-10.0));
    let result = record.validate();
    assert!(result.is_err());
}

#[test]
fn test_create_billing_record_validation_invalid_time() {
    let now = Utc::now();
    let earlier = now - chrono::Duration::hours(1);

    let record = CreateBillingRecord::new(Provider::Aws, now, earlier, dec!(10.0));
    let result = record.validate();
    assert!(result.is_err());
}

#[test]
fn test_create_billing_record_validation_invalid_currency() {
    let now = Utc::now();
    let later = now + chrono::Duration::hours(1);

    let record = CreateBillingRecord::new(Provider::Aws, now, later, dec!(10.0))
        .with_currency("US".to_string());
    let result = record.validate();
    assert!(result.is_err());
}

#[test]
fn test_create_billing_record_builder_pattern() {
    let now = Utc::now();
    let later = now + chrono::Duration::hours(1);

    let record = CreateBillingRecord::new(Provider::Gcp, now, later, dec!(75.5))
        .with_account_id("account-123".to_string())
        .with_service("Storage".to_string())
        .with_resource_id("bucket-abc".to_string())
        .with_currency("EUR".to_string());

    assert_eq!(record.account_id, Some("account-123".to_string()));
    assert_eq!(record.service, Some("Storage".to_string()));
    assert_eq!(record.resource_id, Some("bucket-abc".to_string()));
    assert_eq!(record.currency, "EUR");
}
