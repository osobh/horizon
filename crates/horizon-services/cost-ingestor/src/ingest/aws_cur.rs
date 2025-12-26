use crate::error::{HpcError, IngestorErrorExt, Result};
use crate::models::{CreateBillingRecord, Provider};
use crate::normalize::{BillingNormalizer, GenericBillingRecord, RawBillingData, parse_decimal, parse_iso_datetime};
use serde::Deserialize;
use std::collections::HashMap;

#[derive(Debug, Clone, Deserialize)]
pub struct AwsCurRecord {
    #[serde(rename = "lineItem/UsageAccountId")]
    pub usage_account_id: Option<String>,

    #[serde(rename = "lineItem/ProductCode")]
    pub product_code: Option<String>,

    #[serde(rename = "lineItem/ResourceId")]
    pub resource_id: Option<String>,

    #[serde(rename = "lineItem/UsageStartDate")]
    pub usage_start_date: String,

    #[serde(rename = "lineItem/UsageEndDate")]
    pub usage_end_date: String,

    #[serde(rename = "lineItem/UnblendedCost")]
    pub unblended_cost: String,

    #[serde(rename = "lineItem/CurrencyCode")]
    pub currency_code: Option<String>,

    #[serde(rename = "lineItem/LineItemType")]
    pub line_item_type: Option<String>,

    #[serde(rename = "product/region")]
    pub region: Option<String>,
}

pub struct AwsCurNormalizer;

impl AwsCurNormalizer {
    pub fn new() -> Self {
        Self
    }

    pub fn parse_csv(csv_data: &str) -> Result<Vec<AwsCurRecord>> {
        let mut reader = csv::Reader::from_reader(csv_data.as_bytes());
        let mut records = Vec::new();

        for result in reader.deserialize() {
            let record: AwsCurRecord = result?;
            records.push(record);
        }

        Ok(records)
    }

    fn normalize_record(&self, record: &AwsCurRecord) -> Result<CreateBillingRecord> {
        let usage_start = parse_iso_datetime(&record.usage_start_date)?;
        let usage_end = parse_iso_datetime(&record.usage_end_date)?;
        let amount = parse_decimal(&record.unblended_cost)?;

        let currency = record
            .currency_code
            .clone()
            .unwrap_or_else(|| "USD".to_string());

        let mut metadata = HashMap::new();
        if let Some(ref line_item_type) = record.line_item_type {
            metadata.insert("line_item_type".to_string(), line_item_type.clone());
        }
        if let Some(ref region) = record.region {
            metadata.insert("region".to_string(), region.clone());
        }

        let generic = GenericBillingRecord {
            account_id: record.usage_account_id.clone(),
            service: record.product_code.clone(),
            resource_id: record.resource_id.clone(),
            usage_start,
            usage_end,
            amount,
            currency,
            metadata,
        };

        generic.into_create_record(Provider::Aws)
    }
}

impl Default for AwsCurNormalizer {
    fn default() -> Self {
        Self::new()
    }
}

impl BillingNormalizer for AwsCurNormalizer {
    fn normalize(&self, raw: &RawBillingData) -> Result<Vec<CreateBillingRecord>> {
        if raw.provider != Provider::Aws {
            return Err(HpcError::invalid_provider("Expected AWS provider"));
        }

        let csv_data = raw.data
            .as_str()
            .ok_or_else(|| HpcError::parse_error("Expected CSV string data"))?;

        let aws_records = Self::parse_csv(csv_data)?;
        let mut normalized = Vec::new();

        for aws_record in &aws_records {
            let record = self.normalize_record(aws_record)?;
            normalized.push(record);
        }

        Ok(normalized)
    }

    fn provider(&self) -> Provider {
        Provider::Aws
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    const SAMPLE_AWS_CUR: &str = r#"lineItem/UsageAccountId,lineItem/ProductCode,lineItem/ResourceId,lineItem/UsageStartDate,lineItem/UsageEndDate,lineItem/UnblendedCost,lineItem/CurrencyCode,lineItem/LineItemType,product/region
123456789012,AmazonEC2,i-1234567890abcdef0,2024-01-01T00:00:00Z,2024-01-01T01:00:00Z,1.50,USD,Usage,us-east-1
123456789012,AmazonS3,bucket-name,2024-01-01T00:00:00Z,2024-01-01T01:00:00Z,0.25,USD,Usage,us-west-2
"#;

    #[test]
    fn test_parse_aws_cur_csv() {
        let records = AwsCurNormalizer::parse_csv(SAMPLE_AWS_CUR).unwrap();
        assert_eq!(records.len(), 2);

        assert_eq!(records[0].usage_account_id, Some("123456789012".to_string()));
        assert_eq!(records[0].product_code, Some("AmazonEC2".to_string()));
        assert_eq!(records[0].resource_id, Some("i-1234567890abcdef0".to_string()));
        assert_eq!(records[0].unblended_cost, "1.50");

        assert_eq!(records[1].product_code, Some("AmazonS3".to_string()));
        assert_eq!(records[1].unblended_cost, "0.25");
    }

    #[test]
    fn test_normalize_aws_record() {
        let normalizer = AwsCurNormalizer::new();
        let aws_record = AwsCurRecord {
            usage_account_id: Some("123456789012".to_string()),
            product_code: Some("AmazonEC2".to_string()),
            resource_id: Some("i-123".to_string()),
            usage_start_date: "2024-01-01T00:00:00Z".to_string(),
            usage_end_date: "2024-01-01T01:00:00Z".to_string(),
            unblended_cost: "100.50".to_string(),
            currency_code: Some("USD".to_string()),
            line_item_type: Some("Usage".to_string()),
            region: Some("us-east-1".to_string()),
        };

        let result = normalizer.normalize_record(&aws_record);
        assert!(result.is_ok());

        let record = result.unwrap();
        assert_eq!(record.provider, Provider::Aws);
        assert_eq!(record.amount, dec!(100.50));
        assert_eq!(record.currency, "USD");
        assert_eq!(record.account_id, Some("123456789012".to_string()));
        assert_eq!(record.service, Some("AmazonEC2".to_string()));
    }

    #[test]
    fn test_aws_normalizer_full() {
        let normalizer = AwsCurNormalizer::new();
        let raw = RawBillingData {
            provider: Provider::Aws,
            data: serde_json::Value::String(SAMPLE_AWS_CUR.to_string()),
        };

        let result = normalizer.normalize(&raw);
        assert!(result.is_ok());

        let records = result.unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].amount, dec!(1.50));
        assert_eq!(records[1].amount, dec!(0.25));
    }

    #[test]
    fn test_aws_normalizer_wrong_provider() {
        let normalizer = AwsCurNormalizer::new();
        let raw = RawBillingData {
            provider: Provider::Gcp,
            data: serde_json::Value::String("".to_string()),
        };

        let result = normalizer.normalize(&raw);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Expected AWS provider"));
    }

    #[test]
    fn test_aws_normalizer_invalid_csv() {
        let normalizer = AwsCurNormalizer::new();
        let raw = RawBillingData {
            provider: Provider::Aws,
            data: serde_json::Value::String("invalid,csv,data\n1,2".to_string()),
        };

        let result = normalizer.normalize(&raw);
        assert!(result.is_err());
    }

    #[test]
    fn test_aws_normalizer_default_currency() {
        let normalizer = AwsCurNormalizer::new();
        let aws_record = AwsCurRecord {
            usage_account_id: Some("123".to_string()),
            product_code: None,
            resource_id: None,
            usage_start_date: "2024-01-01T00:00:00Z".to_string(),
            usage_end_date: "2024-01-01T01:00:00Z".to_string(),
            unblended_cost: "10.0".to_string(),
            currency_code: None,
            line_item_type: None,
            region: None,
        };

        let result = normalizer.normalize_record(&aws_record);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().currency, "USD");
    }
}
