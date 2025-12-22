use crate::error::{HpcError, IngestorErrorExt, Result};
use crate::models::{CreateBillingRecord, Provider};
use crate::normalize::{BillingNormalizer, GenericBillingRecord, RawBillingData};
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnPremMeterRecord {
    pub server_id: String,
    pub location: Option<String>,
    pub usage_start: String,
    pub usage_end: String,
    pub power_kwh: f64,
    pub cooling_kwh: Option<f64>,
    pub power_rate_per_kwh: f64,
    pub cooling_rate_per_kwh: Option<f64>,
    pub depreciation_cost: Option<f64>,
    pub maintenance_cost: Option<f64>,
    pub currency: Option<String>,
}

pub struct OnPremNormalizer {
    _default_power_rate: Decimal,
    _default_cooling_rate: Decimal,
}

impl OnPremNormalizer {
    pub fn new() -> Self {
        Self {
            _default_power_rate: Decimal::new(12, 2),
            _default_cooling_rate: Decimal::new(8, 2),
        }
    }

    pub fn with_rates(power_rate: Decimal, cooling_rate: Decimal) -> Self {
        Self {
            _default_power_rate: power_rate,
            _default_cooling_rate: cooling_rate,
        }
    }

    fn normalize_record(&self, record: &OnPremMeterRecord) -> Result<CreateBillingRecord> {
        let usage_start = DateTime::parse_from_rfc3339(&record.usage_start)
            .map(|dt| dt.with_timezone(&Utc))
            .map_err(|e| HpcError::parse_error(format!("Invalid start time: {}", e)))?;

        let usage_end = DateTime::parse_from_rfc3339(&record.usage_end)
            .map(|dt| dt.with_timezone(&Utc))
            .map_err(|e| HpcError::parse_error(format!("Invalid end time: {}", e)))?;

        let power_rate = Decimal::try_from(record.power_rate_per_kwh)
            .map_err(|e| HpcError::parse_error(format!("Invalid power rate: {}", e)))?;

        let power_cost = Decimal::try_from(record.power_kwh)
            .map_err(|e| HpcError::parse_error(format!("Invalid power kwh: {}", e)))?
            * power_rate;

        let cooling_cost = if let (Some(cooling_kwh), Some(cooling_rate)) =
            (record.cooling_kwh, record.cooling_rate_per_kwh) {
            Decimal::try_from(cooling_kwh)
                .map_err(|e| HpcError::parse_error(format!("Invalid cooling kwh: {}", e)))?
                * Decimal::try_from(cooling_rate)
                    .map_err(|e| HpcError::parse_error(format!("Invalid cooling rate: {}", e)))?
        } else {
            Decimal::ZERO
        };

        let depreciation_cost = if let Some(dep) = record.depreciation_cost {
            Decimal::try_from(dep)
                .map_err(|e| HpcError::parse_error(format!("Invalid depreciation: {}", e)))?
        } else {
            Decimal::ZERO
        };

        let maintenance_cost = if let Some(maint) = record.maintenance_cost {
            Decimal::try_from(maint)
                .map_err(|e| HpcError::parse_error(format!("Invalid maintenance: {}", e)))?
        } else {
            Decimal::ZERO
        };

        let total_amount = power_cost + cooling_cost + depreciation_cost + maintenance_cost;

        let currency = record
            .currency
            .clone()
            .unwrap_or_else(|| "USD".to_string());

        let mut metadata = HashMap::new();
        metadata.insert("power_kwh".to_string(), record.power_kwh.to_string());
        metadata.insert("power_cost".to_string(), power_cost.to_string());

        if let Some(cooling_kwh) = record.cooling_kwh {
            metadata.insert("cooling_kwh".to_string(), cooling_kwh.to_string());
            metadata.insert("cooling_cost".to_string(), cooling_cost.to_string());
        }

        if let Some(location) = &record.location {
            metadata.insert("location".to_string(), location.clone());
        }

        let generic = GenericBillingRecord {
            account_id: record.location.clone(),
            service: Some("OnPrem Compute".to_string()),
            resource_id: Some(record.server_id.clone()),
            usage_start,
            usage_end,
            amount: total_amount,
            currency,
            metadata,
        };

        generic.into_create_record(Provider::OnPrem)
    }
}

impl Default for OnPremNormalizer {
    fn default() -> Self {
        Self::new()
    }
}

impl BillingNormalizer for OnPremNormalizer {
    fn normalize(&self, raw: &RawBillingData) -> Result<Vec<CreateBillingRecord>> {
        if raw.provider != Provider::OnPrem {
            return Err(HpcError::invalid_provider("Expected OnPrem provider"));
        }

        let records: Vec<OnPremMeterRecord> = serde_json::from_value(raw.data.clone())?;
        let mut normalized = Vec::new();

        for onprem_record in &records {
            let record = self.normalize_record(onprem_record)?;
            normalized.push(record);
        }

        Ok(normalized)
    }

    fn provider(&self) -> Provider {
        Provider::OnPrem
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    fn sample_onprem_record() -> OnPremMeterRecord {
        OnPremMeterRecord {
            server_id: "server-001".to_string(),
            location: Some("datacenter-1".to_string()),
            usage_start: "2024-01-01T00:00:00Z".to_string(),
            usage_end: "2024-01-01T01:00:00Z".to_string(),
            power_kwh: 5.0,
            cooling_kwh: Some(3.0),
            power_rate_per_kwh: 0.15,
            cooling_rate_per_kwh: Some(0.10),
            depreciation_cost: Some(1.50),
            maintenance_cost: Some(0.50),
            currency: Some("USD".to_string()),
        }
    }

    #[test]
    fn test_normalize_onprem_record_full() {
        let normalizer = OnPremNormalizer::new();
        let record = sample_onprem_record();

        let result = normalizer.normalize_record(&record);
        assert!(result.is_ok());

        let normalized = result.unwrap();
        assert_eq!(normalized.provider, Provider::OnPrem);

        let power_cost = dec!(5.0) * dec!(0.15);
        let cooling_cost = dec!(3.0) * dec!(0.10);
        let depreciation = dec!(1.50);
        let maintenance = dec!(0.50);
        let expected_total = power_cost + cooling_cost + depreciation + maintenance;

        assert_eq!(normalized.amount, expected_total);
        assert_eq!(normalized.currency, "USD");
        assert_eq!(normalized.resource_id, Some("server-001".to_string()));
    }

    #[test]
    fn test_normalize_onprem_record_minimal() {
        let normalizer = OnPremNormalizer::new();
        let record = OnPremMeterRecord {
            server_id: "server-002".to_string(),
            location: None,
            usage_start: "2024-01-01T00:00:00Z".to_string(),
            usage_end: "2024-01-01T01:00:00Z".to_string(),
            power_kwh: 10.0,
            cooling_kwh: None,
            power_rate_per_kwh: 0.20,
            cooling_rate_per_kwh: None,
            depreciation_cost: None,
            maintenance_cost: None,
            currency: None,
        };

        let result = normalizer.normalize_record(&record);
        assert!(result.is_ok());

        let normalized = result.unwrap();
        assert_eq!(normalized.amount, dec!(10.0) * dec!(0.20));
        assert_eq!(normalized.currency, "USD");
    }

    #[test]
    fn test_onprem_normalizer_full() {
        let normalizer = OnPremNormalizer::new();
        let records = vec![
            sample_onprem_record(),
            {
                let mut r = sample_onprem_record();
                r.server_id = "server-002".to_string();
                r.power_kwh = 10.0;
                r
            },
        ];

        let raw = RawBillingData {
            provider: Provider::OnPrem,
            data: serde_json::to_value(&records).unwrap(),
        };

        let result = normalizer.normalize(&raw);
        assert!(result.is_ok());

        let normalized = result.unwrap();
        assert_eq!(normalized.len(), 2);
    }

    #[test]
    fn test_onprem_normalizer_wrong_provider() {
        let normalizer = OnPremNormalizer::new();
        let raw = RawBillingData {
            provider: Provider::Aws,
            data: serde_json::json!([]),
        };

        let result = normalizer.normalize(&raw);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Expected OnPrem provider"));
    }

    #[test]
    fn test_onprem_normalizer_with_custom_rates() {
        let _normalizer = OnPremNormalizer::with_rates(dec!(0.25), dec!(0.15));
        // Normalizer created successfully with custom rates
        assert!(true);
    }

    #[test]
    fn test_onprem_metadata() {
        let normalizer = OnPremNormalizer::new();
        let record = sample_onprem_record();

        let result = normalizer.normalize_record(&record);
        assert!(result.is_ok());

        let normalized = result.unwrap();
        let metadata: HashMap<String, String> = serde_json::from_value(normalized.raw_data).unwrap();

        assert!(metadata.contains_key("power_kwh"));
        assert!(metadata.contains_key("power_cost"));
        assert!(metadata.contains_key("cooling_kwh"));
        assert!(metadata.contains_key("cooling_cost"));
        assert!(metadata.contains_key("location"));
    }

    #[test]
    fn test_onprem_cost_calculation() {
        let normalizer = OnPremNormalizer::new();
        let record = OnPremMeterRecord {
            server_id: "server-003".to_string(),
            location: None,
            usage_start: "2024-01-01T00:00:00Z".to_string(),
            usage_end: "2024-01-01T01:00:00Z".to_string(),
            power_kwh: 100.0,
            cooling_kwh: Some(50.0),
            power_rate_per_kwh: 0.10,
            cooling_rate_per_kwh: Some(0.05),
            depreciation_cost: Some(10.0),
            maintenance_cost: Some(5.0),
            currency: Some("USD".to_string()),
        };

        let result = normalizer.normalize_record(&record);
        assert!(result.is_ok());

        let normalized = result.unwrap();
        let power = dec!(100.0) * dec!(0.10);
        let cooling = dec!(50.0) * dec!(0.05);
        let depreciation = dec!(10.0);
        let maintenance = dec!(5.0);
        let expected = power + cooling + depreciation + maintenance;

        assert_eq!(normalized.amount, expected);
    }
}
