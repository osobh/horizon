use crate::error::{HpcError, IngestorErrorExt};
use crate::models::{CreateBillingRecord, Provider};
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawBillingData {
    pub provider: Provider,
    pub data: serde_json::Value,
}

pub trait BillingNormalizer: Send + Sync {
    fn normalize(&self, raw: &RawBillingData) -> crate::error::Result<Vec<CreateBillingRecord>>;
    fn provider(&self) -> Provider;
}

pub struct NormalizedBillingSchema {
    normalizers: HashMap<Provider, Box<dyn BillingNormalizer>>,
}

impl NormalizedBillingSchema {
    pub fn new() -> Self {
        Self {
            normalizers: HashMap::new(),
        }
    }

    pub fn register_normalizer(
        &mut self,
        provider: Provider,
        normalizer: Box<dyn BillingNormalizer>,
    ) {
        self.normalizers.insert(provider, normalizer);
    }

    pub fn normalize(&self, raw: &RawBillingData) -> crate::error::Result<Vec<CreateBillingRecord>> {
        let normalizer = self
            .normalizers
            .get(&raw.provider)
            .ok_or_else(|| {
                HpcError::invalid_provider(format!(
                    "No normalizer registered for provider: {}",
                    raw.provider
                ))
            })?;

        normalizer.normalize(raw)
    }
}

impl Default for NormalizedBillingSchema {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenericBillingRecord {
    pub account_id: Option<String>,
    pub service: Option<String>,
    pub resource_id: Option<String>,
    pub usage_start: DateTime<Utc>,
    pub usage_end: DateTime<Utc>,
    pub amount: Decimal,
    pub currency: String,
    pub metadata: HashMap<String, String>,
}

impl GenericBillingRecord {
    pub fn into_create_record(
        self,
        provider: Provider,
    ) -> crate::error::Result<CreateBillingRecord> {
        let raw_data = serde_json::to_value(&self.metadata)?;

        let mut record = CreateBillingRecord::new(
            provider,
            self.usage_start,
            self.usage_end,
            self.amount,
        )
        .with_currency(self.currency)
        .with_raw_data(raw_data);

        if let Some(account_id) = self.account_id {
            record = record.with_account_id(account_id);
        }

        if let Some(service) = self.service {
            record = record.with_service(service);
        }

        if let Some(resource_id) = self.resource_id {
            record = record.with_resource_id(resource_id);
        }

        record.validate()?;
        Ok(record)
    }
}

pub fn parse_iso_datetime(s: &str) -> crate::error::Result<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(s)
        .map(|dt| dt.with_timezone(&Utc))
        .map_err(|e| HpcError::parse_error(format!("Invalid datetime: {}", e)))
}

pub fn parse_decimal(s: &str) -> crate::error::Result<Decimal> {
    s.parse::<Decimal>()
        .map_err(|e| HpcError::parse_error(format!("Invalid decimal: {}", e)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    struct MockNormalizer {
        provider: Provider,
    }

    impl BillingNormalizer for MockNormalizer {
        fn normalize(&self, raw: &RawBillingData) -> crate::error::Result<Vec<CreateBillingRecord>> {
            let amount = raw.data["amount"]
                .as_str()
                .ok_or_else(|| HpcError::parse_error("Missing amount"))?;
            let amount = parse_decimal(amount)?;

            let start = raw.data["start"]
                .as_str()
                .ok_or_else(|| HpcError::parse_error("Missing start"))?;
            let start = parse_iso_datetime(start)?;

            let end = raw.data["end"]
                .as_str()
                .ok_or_else(|| HpcError::parse_error("Missing end"))?;
            let end = parse_iso_datetime(end)?;

            let record = CreateBillingRecord::new(self.provider, start, end, amount);
            Ok(vec![record])
        }

        fn provider(&self) -> Provider {
            self.provider
        }
    }

    #[test]
    fn test_normalized_schema_register() {
        let mut schema = NormalizedBillingSchema::new();
        let normalizer = MockNormalizer {
            provider: Provider::Aws,
        };
        schema.register_normalizer(Provider::Aws, Box::new(normalizer));

        assert!(schema.normalizers.contains_key(&Provider::Aws));
    }

    #[test]
    fn test_normalized_schema_normalize_success() {
        let mut schema = NormalizedBillingSchema::new();
        let normalizer = MockNormalizer {
            provider: Provider::Aws,
        };
        schema.register_normalizer(Provider::Aws, Box::new(normalizer));

        let raw = RawBillingData {
            provider: Provider::Aws,
            data: serde_json::json!({
                "amount": "100.50",
                "start": "2024-01-01T00:00:00Z",
                "end": "2024-01-01T01:00:00Z",
            }),
        };

        let result = schema.normalize(&raw);
        assert!(result.is_ok());
        let records = result.unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].amount, dec!(100.50));
    }

    #[test]
    fn test_normalized_schema_no_normalizer() {
        let schema = NormalizedBillingSchema::new();
        let raw = RawBillingData {
            provider: Provider::Aws,
            data: serde_json::json!({}),
        };

        let result = schema.normalize(&raw);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("No normalizer registered"));
    }

    #[test]
    fn test_generic_billing_record_conversion() {
        let generic = GenericBillingRecord {
            account_id: Some("123456".to_string()),
            service: Some("EC2".to_string()),
            resource_id: Some("i-123".to_string()),
            usage_start: Utc::now(),
            usage_end: Utc::now() + chrono::Duration::hours(1),
            amount: dec!(50.0),
            currency: "USD".to_string(),
            metadata: HashMap::new(),
        };

        let record = generic.into_create_record(Provider::Aws);
        assert!(record.is_ok());
        let record = record.unwrap();
        assert_eq!(record.provider, Provider::Aws);
        assert_eq!(record.amount, dec!(50.0));
        assert_eq!(record.account_id, Some("123456".to_string()));
    }

    #[test]
    fn test_parse_iso_datetime() {
        let result = parse_iso_datetime("2024-01-01T00:00:00Z");
        assert!(result.is_ok());

        let result = parse_iso_datetime("invalid");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_decimal() {
        let result = parse_decimal("123.45");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), dec!(123.45));

        let result = parse_decimal("invalid");
        assert!(result.is_err());
    }
}
