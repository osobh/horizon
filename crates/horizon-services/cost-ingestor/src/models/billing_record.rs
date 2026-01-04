use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use sqlx::FromRow;
use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, sqlx::Type)]
#[sqlx(type_name = "text")]
pub enum Provider {
    #[serde(rename = "aws")]
    Aws,
    #[serde(rename = "gcp")]
    Gcp,
    #[serde(rename = "azure")]
    Azure,
    #[serde(rename = "onprem")]
    OnPrem,
}

impl std::fmt::Display for Provider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Provider::Aws => write!(f, "aws"),
            Provider::Gcp => write!(f, "gcp"),
            Provider::Azure => write!(f, "azure"),
            Provider::OnPrem => write!(f, "onprem"),
        }
    }
}

impl std::str::FromStr for Provider {
    type Err = crate::error::HpcError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        use crate::error::IngestorErrorExt;
        match s.to_lowercase().as_str() {
            "aws" => Ok(Provider::Aws),
            "gcp" => Ok(Provider::Gcp),
            "azure" => Ok(Provider::Azure),
            "onprem" => Ok(Provider::OnPrem),
            _ => Err(crate::error::HpcError::invalid_provider(s)),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct BillingRecord {
    pub id: Uuid,
    pub provider: String,
    pub account_id: Option<String>,
    pub service: Option<String>,
    pub resource_id: Option<String>,
    pub usage_start: DateTime<Utc>,
    pub usage_end: DateTime<Utc>,
    pub amount: Decimal,
    pub currency: String,
    pub raw_data: serde_json::Value,
    pub ingested_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateBillingRecord {
    pub provider: Provider,
    pub account_id: Option<String>,
    pub service: Option<String>,
    pub resource_id: Option<String>,
    pub usage_start: DateTime<Utc>,
    pub usage_end: DateTime<Utc>,
    pub amount: Decimal,
    pub currency: String,
    pub raw_data: serde_json::Value,
}

impl CreateBillingRecord {
    pub fn new(
        provider: Provider,
        usage_start: DateTime<Utc>,
        usage_end: DateTime<Utc>,
        amount: Decimal,
    ) -> Self {
        Self {
            provider,
            account_id: None,
            service: None,
            resource_id: None,
            usage_start,
            usage_end,
            amount,
            currency: "USD".to_string(),
            raw_data: serde_json::json!({}),
        }
    }

    pub fn with_account_id(mut self, account_id: String) -> Self {
        self.account_id = Some(account_id);
        self
    }

    pub fn with_service(mut self, service: String) -> Self {
        self.service = Some(service);
        self
    }

    pub fn with_resource_id(mut self, resource_id: String) -> Self {
        self.resource_id = Some(resource_id);
        self
    }

    pub fn with_currency(mut self, currency: String) -> Self {
        self.currency = currency;
        self
    }

    pub fn with_raw_data(mut self, raw_data: serde_json::Value) -> Self {
        self.raw_data = raw_data;
        self
    }

    pub fn validate(&self) -> crate::error::Result<()> {
        use crate::error::IngestorErrorExt;
        if self.amount < Decimal::ZERO {
            return Err(crate::error::HpcError::invalid_billing_data(
                "Amount cannot be negative",
            ));
        }

        if self.usage_end <= self.usage_start {
            return Err(crate::error::HpcError::invalid_billing_data(
                "Usage end must be after usage start",
            ));
        }

        if self.currency.len() != 3 {
            return Err(crate::error::HpcError::invalid_billing_data(
                "Currency must be 3-letter code",
            ));
        }

        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BillingRecordQuery {
    pub provider: Option<Provider>,
    pub account_id: Option<String>,
    pub service: Option<String>,
    pub resource_id: Option<String>,
    pub start_date: Option<DateTime<Utc>>,
    pub end_date: Option<DateTime<Utc>>,
    pub limit: Option<i64>,
    pub offset: Option<i64>,
}

impl Default for BillingRecordQuery {
    fn default() -> Self {
        Self {
            provider: None,
            account_id: None,
            service: None,
            resource_id: None,
            start_date: None,
            end_date: None,
            limit: Some(100),
            offset: Some(0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;
    use std::str::FromStr;

    #[test]
    fn test_provider_from_str() {
        assert_eq!(Provider::from_str("aws").unwrap(), Provider::Aws);
        assert_eq!(Provider::from_str("AWS").unwrap(), Provider::Aws);
        assert_eq!(Provider::from_str("gcp").unwrap(), Provider::Gcp);
        assert_eq!(Provider::from_str("azure").unwrap(), Provider::Azure);
        assert_eq!(Provider::from_str("onprem").unwrap(), Provider::OnPrem);
        assert!(Provider::from_str("invalid").is_err());
    }

    #[test]
    fn test_provider_display() {
        assert_eq!(Provider::Aws.to_string(), "aws");
        assert_eq!(Provider::Gcp.to_string(), "gcp");
        assert_eq!(Provider::Azure.to_string(), "azure");
        assert_eq!(Provider::OnPrem.to_string(), "onprem");
    }

    #[test]
    fn test_create_billing_record_builder() {
        let now = Utc::now();
        let later = now + chrono::Duration::hours(1);

        let record = CreateBillingRecord::new(Provider::Aws, now, later, dec!(100.50))
            .with_account_id("123456789".to_string())
            .with_service("EC2".to_string())
            .with_resource_id("i-1234567890abcdef0".to_string())
            .with_currency("USD".to_string());

        assert_eq!(record.provider, Provider::Aws);
        assert_eq!(record.account_id, Some("123456789".to_string()));
        assert_eq!(record.service, Some("EC2".to_string()));
        assert_eq!(record.amount, dec!(100.50));
        assert_eq!(record.currency, "USD");
    }

    #[test]
    fn test_validate_billing_record_success() {
        let now = Utc::now();
        let later = now + chrono::Duration::hours(1);

        let record = CreateBillingRecord::new(Provider::Aws, now, later, dec!(100.50));
        assert!(record.validate().is_ok());
    }

    #[test]
    fn test_validate_negative_amount() {
        let now = Utc::now();
        let later = now + chrono::Duration::hours(1);

        let record = CreateBillingRecord::new(Provider::Aws, now, later, dec!(-10.0));
        let result = record.validate();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Amount cannot be negative"));
    }

    #[test]
    fn test_validate_invalid_time_range() {
        let now = Utc::now();
        let earlier = now - chrono::Duration::hours(1);

        let record = CreateBillingRecord::new(Provider::Aws, now, earlier, dec!(100.0));
        let result = record.validate();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Usage end must be after usage start"));
    }

    #[test]
    fn test_validate_invalid_currency() {
        let now = Utc::now();
        let later = now + chrono::Duration::hours(1);

        let record = CreateBillingRecord::new(Provider::Aws, now, later, dec!(100.0))
            .with_currency("US".to_string());
        let result = record.validate();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Currency must be 3-letter code"));
    }

    #[test]
    fn test_billing_record_query_default() {
        let query = BillingRecordQuery::default();
        assert_eq!(query.limit, Some(100));
        assert_eq!(query.offset, Some(0));
        assert!(query.provider.is_none());
    }

    #[test]
    fn test_provider_serialization() {
        let provider = Provider::Aws;
        let json = serde_json::to_string(&provider).unwrap();
        assert_eq!(json, "\"aws\"");

        let deserialized: Provider = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, Provider::Aws);
    }
}
