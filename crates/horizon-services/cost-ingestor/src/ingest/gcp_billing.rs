use crate::error::{HpcError, IngestorErrorExt, Result};
use crate::models::{CreateBillingRecord, Provider};
use crate::normalize::{BillingNormalizer, GenericBillingRecord, RawBillingData};
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GcpBillingRecord {
    pub billing_account_id: Option<String>,
    pub service: Option<GcpService>,
    pub sku: Option<GcpSku>,
    pub usage_start_time: String,
    pub usage_end_time: String,
    pub project: Option<GcpProject>,
    pub cost: f64,
    pub currency: String,
    pub usage: Option<GcpUsage>,
    pub credits: Option<Vec<GcpCredit>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GcpService {
    pub id: String,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GcpSku {
    pub id: String,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GcpProject {
    pub id: String,
    pub name: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GcpUsage {
    pub amount: f64,
    pub unit: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GcpCredit {
    pub name: String,
    pub amount: f64,
}

pub struct GcpBillingNormalizer;

impl GcpBillingNormalizer {
    pub fn new() -> Self {
        Self
    }

    fn normalize_record(&self, record: &GcpBillingRecord) -> Result<CreateBillingRecord> {
        let usage_start = DateTime::parse_from_rfc3339(&record.usage_start_time)
            .map(|dt| dt.with_timezone(&Utc))
            .map_err(|e| HpcError::parse_error(format!("Invalid start time: {}", e)))?;

        let usage_end = DateTime::parse_from_rfc3339(&record.usage_end_time)
            .map(|dt| dt.with_timezone(&Utc))
            .map_err(|e| HpcError::parse_error(format!("Invalid end time: {}", e)))?;

        let net_cost = if let Some(credits) = &record.credits {
            let credit_total: f64 = credits.iter().map(|c| c.amount).sum();
            record.cost + credit_total
        } else {
            record.cost
        };

        let amount = Decimal::try_from(net_cost)
            .map_err(|e| HpcError::parse_error(format!("Invalid cost: {}", e)))?;

        let service_name = record
            .service
            .as_ref()
            .map(|s| s.description.clone());

        let resource_id = record
            .sku
            .as_ref()
            .map(|s| s.id.clone());

        let account_id = record
            .project
            .as_ref()
            .map(|p| p.id.clone())
            .or_else(|| record.billing_account_id.clone());

        let mut metadata = HashMap::new();
        if let Some(ref service) = record.service {
            metadata.insert("service_id".to_string(), service.id.clone());
        }
        if let Some(ref sku) = record.sku {
            metadata.insert("sku_description".to_string(), sku.description.clone());
        }
        if let Some(ref usage) = record.usage {
            metadata.insert("usage_amount".to_string(), usage.amount.to_string());
            metadata.insert("usage_unit".to_string(), usage.unit.clone());
        }

        let generic = GenericBillingRecord {
            account_id,
            service: service_name,
            resource_id,
            usage_start,
            usage_end,
            amount,
            currency: record.currency.clone(),
            metadata,
        };

        generic.into_create_record(Provider::Gcp)
    }
}

impl Default for GcpBillingNormalizer {
    fn default() -> Self {
        Self::new()
    }
}

impl BillingNormalizer for GcpBillingNormalizer {
    fn normalize(&self, raw: &RawBillingData) -> Result<Vec<CreateBillingRecord>> {
        if raw.provider != Provider::Gcp {
            return Err(HpcError::invalid_provider("Expected GCP provider"));
        }

        let records: Vec<GcpBillingRecord> = serde_json::from_value(raw.data.clone())?;
        let mut normalized = Vec::new();

        for gcp_record in &records {
            let record = self.normalize_record(gcp_record)?;
            normalized.push(record);
        }

        Ok(normalized)
    }

    fn provider(&self) -> Provider {
        Provider::Gcp
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    fn sample_gcp_record() -> GcpBillingRecord {
        GcpBillingRecord {
            billing_account_id: Some("012345-6789AB-CDEF01".to_string()),
            service: Some(GcpService {
                id: "services/6F81-5844-456A".to_string(),
                description: "Compute Engine".to_string(),
            }),
            sku: Some(GcpSku {
                id: "0000-0000-0000".to_string(),
                description: "N1 Predefined Instance Core".to_string(),
            }),
            usage_start_time: "2024-01-01T00:00:00Z".to_string(),
            usage_end_time: "2024-01-01T01:00:00Z".to_string(),
            project: Some(GcpProject {
                id: "my-project-123".to_string(),
                name: Some("My Project".to_string()),
            }),
            cost: 1.50,
            currency: "USD".to_string(),
            usage: Some(GcpUsage {
                amount: 1.0,
                unit: "hour".to_string(),
            }),
            credits: None,
        }
    }

    #[test]
    fn test_normalize_gcp_record() {
        let normalizer = GcpBillingNormalizer::new();
        let gcp_record = sample_gcp_record();

        let result = normalizer.normalize_record(&gcp_record);
        assert!(result.is_ok());

        let record = result.unwrap();
        assert_eq!(record.provider, Provider::Gcp);
        assert_eq!(record.amount, dec!(1.50));
        assert_eq!(record.currency, "USD");
        assert_eq!(record.account_id, Some("my-project-123".to_string()));
        assert_eq!(record.service, Some("Compute Engine".to_string()));
    }

    #[test]
    fn test_gcp_record_with_credits() {
        let normalizer = GcpBillingNormalizer::new();
        let mut gcp_record = sample_gcp_record();
        gcp_record.cost = 10.0;
        gcp_record.credits = Some(vec![
            GcpCredit {
                name: "Promotional credit".to_string(),
                amount: -2.0,
            },
            GcpCredit {
                name: "Free tier".to_string(),
                amount: -1.0,
            },
        ]);

        let result = normalizer.normalize_record(&gcp_record);
        assert!(result.is_ok());

        let record = result.unwrap();
        assert_eq!(record.amount, dec!(7.0));
    }

    #[test]
    fn test_gcp_normalizer_full() {
        let normalizer = GcpBillingNormalizer::new();
        let gcp_records = vec![
            sample_gcp_record(),
            {
                let mut r = sample_gcp_record();
                r.cost = 2.50;
                r
            },
        ];

        let raw = RawBillingData {
            provider: Provider::Gcp,
            data: serde_json::to_value(&gcp_records).unwrap(),
        };

        let result = normalizer.normalize(&raw);
        assert!(result.is_ok());

        let records = result.unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].amount, dec!(1.50));
        assert_eq!(records[1].amount, dec!(2.50));
    }

    #[test]
    fn test_gcp_normalizer_wrong_provider() {
        let normalizer = GcpBillingNormalizer::new();
        let raw = RawBillingData {
            provider: Provider::Aws,
            data: serde_json::json!([]),
        };

        let result = normalizer.normalize(&raw);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Expected GCP provider"));
    }

    #[test]
    fn test_gcp_normalizer_invalid_json() {
        let normalizer = GcpBillingNormalizer::new();
        let raw = RawBillingData {
            provider: Provider::Gcp,
            data: serde_json::json!("invalid"),
        };

        let result = normalizer.normalize(&raw);
        assert!(result.is_err());
    }

    #[test]
    fn test_gcp_record_no_project_uses_billing_account() {
        let normalizer = GcpBillingNormalizer::new();
        let mut gcp_record = sample_gcp_record();
        gcp_record.project = None;

        let result = normalizer.normalize_record(&gcp_record);
        assert!(result.is_ok());

        let record = result.unwrap();
        assert_eq!(record.account_id, Some("012345-6789AB-CDEF01".to_string()));
    }

    #[test]
    fn test_gcp_record_metadata() {
        let normalizer = GcpBillingNormalizer::new();
        let gcp_record = sample_gcp_record();

        let result = normalizer.normalize_record(&gcp_record);
        assert!(result.is_ok());

        let record = result.unwrap();
        let metadata: HashMap<String, String> = serde_json::from_value(record.raw_data).unwrap();
        assert!(metadata.contains_key("service_id"));
        assert!(metadata.contains_key("sku_description"));
        assert!(metadata.contains_key("usage_amount"));
    }
}
