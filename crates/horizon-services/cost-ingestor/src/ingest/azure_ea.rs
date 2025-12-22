use crate::error::{HpcError, IngestorErrorExt, Result};
use crate::models::{CreateBillingRecord, Provider};
use crate::normalize::{BillingNormalizer, GenericBillingRecord, RawBillingData};
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::Deserialize;
use std::collections::HashMap;

#[derive(Debug, Clone, Deserialize)]
pub struct AzureEaRecord {
    #[serde(rename = "SubscriptionId")]
    pub subscription_id: Option<String>,

    #[serde(rename = "ResourceGroup")]
    pub resource_group: Option<String>,

    #[serde(rename = "ResourceId")]
    pub resource_id: Option<String>,

    #[serde(rename = "ServiceName")]
    pub service_name: Option<String>,

    #[serde(rename = "UsageDateTime")]
    pub usage_date_time: String,

    #[serde(rename = "Cost")]
    pub cost: f64,

    #[serde(rename = "Currency")]
    pub currency: Option<String>,

    #[serde(rename = "MeterName")]
    pub meter_name: Option<String>,

    #[serde(rename = "Quantity")]
    pub quantity: Option<f64>,

    #[serde(rename = "UnitOfMeasure")]
    pub unit_of_measure: Option<String>,
}

pub struct AzureEaNormalizer;

impl AzureEaNormalizer {
    pub fn new() -> Self {
        Self
    }

    pub fn parse_json(json_data: &str) -> Result<Vec<AzureEaRecord>> {
        let records: Vec<AzureEaRecord> = serde_json::from_str(json_data)?;
        Ok(records)
    }

    fn normalize_record(&self, record: &AzureEaRecord) -> Result<CreateBillingRecord> {
        let usage_start = DateTime::parse_from_rfc3339(&record.usage_date_time)
            .map(|dt| dt.with_timezone(&Utc))
            .map_err(|e| HpcError::parse_error(format!("Invalid datetime: {}", e)))?;

        let usage_end = usage_start + chrono::Duration::hours(1);

        let amount = Decimal::try_from(record.cost)
            .map_err(|e| HpcError::parse_error(format!("Invalid cost: {}", e)))?;

        let currency = record
            .currency
            .clone()
            .unwrap_or_else(|| "USD".to_string());

        let mut metadata = HashMap::new();
        if let Some(ref resource_group) = record.resource_group {
            metadata.insert("resource_group".to_string(), resource_group.clone());
        }
        if let Some(ref meter_name) = record.meter_name {
            metadata.insert("meter_name".to_string(), meter_name.clone());
        }
        if let Some(quantity) = record.quantity {
            metadata.insert("quantity".to_string(), quantity.to_string());
        }
        if let Some(ref unit) = record.unit_of_measure {
            metadata.insert("unit_of_measure".to_string(), unit.clone());
        }

        let generic = GenericBillingRecord {
            account_id: record.subscription_id.clone(),
            service: record.service_name.clone(),
            resource_id: record.resource_id.clone(),
            usage_start,
            usage_end,
            amount,
            currency,
            metadata,
        };

        generic.into_create_record(Provider::Azure)
    }
}

impl Default for AzureEaNormalizer {
    fn default() -> Self {
        Self::new()
    }
}

impl BillingNormalizer for AzureEaNormalizer {
    fn normalize(&self, raw: &RawBillingData) -> Result<Vec<CreateBillingRecord>> {
        if raw.provider != Provider::Azure {
            return Err(HpcError::invalid_provider("Expected Azure provider"));
        }

        let json_data = raw.data
            .as_str()
            .ok_or_else(|| HpcError::parse_error("Expected JSON string data"))?;

        let azure_records = Self::parse_json(json_data)?;
        let mut normalized = Vec::new();

        for azure_record in &azure_records {
            let record = self.normalize_record(azure_record)?;
            normalized.push(record);
        }

        Ok(normalized)
    }

    fn provider(&self) -> Provider {
        Provider::Azure
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    const SAMPLE_AZURE_DATA: &str = r#"[
        {
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
        }
    ]"#;

    #[test]
    fn test_parse_azure_json() {
        let records = AzureEaNormalizer::parse_json(SAMPLE_AZURE_DATA).unwrap();
        assert_eq!(records.len(), 1);

        assert_eq!(
            records[0].subscription_id,
            Some("12345678-1234-1234-1234-123456789abc".to_string())
        );
        assert_eq!(records[0].cost, 10.50);
        assert_eq!(records[0].service_name, Some("Virtual Machines".to_string()));
    }

    #[test]
    fn test_normalize_azure_record() {
        let normalizer = AzureEaNormalizer::new();
        let azure_record = AzureEaRecord {
            subscription_id: Some("sub-123".to_string()),
            resource_group: Some("rg-prod".to_string()),
            resource_id: Some("/vm/vm1".to_string()),
            service_name: Some("Virtual Machines".to_string()),
            usage_date_time: "2024-01-01T00:00:00Z".to_string(),
            cost: 25.75,
            currency: Some("USD".to_string()),
            meter_name: Some("D4s v3".to_string()),
            quantity: Some(2.0),
            unit_of_measure: Some("Hour".to_string()),
        };

        let result = normalizer.normalize_record(&azure_record);
        assert!(result.is_ok());

        let record = result.unwrap();
        assert_eq!(record.provider, Provider::Azure);
        assert_eq!(record.amount, dec!(25.75));
        assert_eq!(record.currency, "USD");
        assert_eq!(record.account_id, Some("sub-123".to_string()));
    }

    #[test]
    fn test_azure_normalizer_full() {
        let normalizer = AzureEaNormalizer::new();
        let raw = RawBillingData {
            provider: Provider::Azure,
            data: serde_json::Value::String(SAMPLE_AZURE_DATA.to_string()),
        };

        let result = normalizer.normalize(&raw);
        assert!(result.is_ok());

        let records = result.unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].amount, dec!(10.50));
    }

    #[test]
    fn test_azure_normalizer_wrong_provider() {
        let normalizer = AzureEaNormalizer::new();
        let raw = RawBillingData {
            provider: Provider::Aws,
            data: serde_json::Value::String("[]".to_string()),
        };

        let result = normalizer.normalize(&raw);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Expected Azure provider"));
    }

    #[test]
    fn test_azure_normalizer_default_currency() {
        let normalizer = AzureEaNormalizer::new();
        let azure_record = AzureEaRecord {
            subscription_id: Some("sub-123".to_string()),
            resource_group: None,
            resource_id: None,
            service_name: None,
            usage_date_time: "2024-01-01T00:00:00Z".to_string(),
            cost: 5.0,
            currency: None,
            meter_name: None,
            quantity: None,
            unit_of_measure: None,
        };

        let result = normalizer.normalize_record(&azure_record);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().currency, "USD");
    }

    #[test]
    fn test_azure_record_usage_end_calculation() {
        let normalizer = AzureEaNormalizer::new();
        let azure_record = AzureEaRecord {
            subscription_id: None,
            resource_group: None,
            resource_id: None,
            service_name: None,
            usage_date_time: "2024-01-01T10:00:00Z".to_string(),
            cost: 1.0,
            currency: None,
            meter_name: None,
            quantity: None,
            unit_of_measure: None,
        };

        let result = normalizer.normalize_record(&azure_record);
        assert!(result.is_ok());

        let record = result.unwrap();
        let duration = record.usage_end - record.usage_start;
        assert_eq!(duration, chrono::Duration::hours(1));
    }
}
