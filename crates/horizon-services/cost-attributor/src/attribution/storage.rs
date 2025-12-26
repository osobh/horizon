use chrono::{DateTime, Utc};
use rust_decimal::Decimal;

/// Calculate storage cost from volume and duration
/// GB-hours = storage_gb * hours
pub fn calculate_storage_cost(
    storage_gb: Decimal,
    duration_start: DateTime<Utc>,
    duration_end: DateTime<Utc>,
    rate_per_gb_hour: Decimal,
) -> crate::error::Result<Decimal> {
    use crate::error::AttributorErrorExt;

    if storage_gb < Decimal::ZERO {
        return Err(crate::error::HpcError::calculation_error(
            "Storage volume cannot be negative",
        ));
    }

    if rate_per_gb_hour < Decimal::ZERO {
        return Err(crate::error::HpcError::calculation_error(
            "Rate per GB-hour cannot be negative",
        ));
    }

    if duration_end <= duration_start {
        return Err(crate::error::HpcError::invalid_time_range(
            duration_start.to_rfc3339(),
            duration_end.to_rfc3339(),
        ));
    }

    let duration_seconds = (duration_end - duration_start).num_seconds();
    let hours = Decimal::from(duration_seconds) / Decimal::from(3600);
    let gb_hours = storage_gb * hours;
    let cost = gb_hours * rate_per_gb_hour;

    Ok(cost)
}

/// Calculate storage cost for different tiers
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageTier {
    Standard,
    Infrequent,
    Archive,
    Ephemeral,
}

impl StorageTier {
    /// Get default rate per GB-hour for tier
    /// These are example rates
    pub fn default_rate(&self) -> Decimal {
        match self {
            StorageTier::Standard => Decimal::new(1368, 8),   // ~$0.01/GB/month
            StorageTier::Infrequent => Decimal::new(684, 8), // ~$0.005/GB/month
            StorageTier::Archive => Decimal::new(137, 8),    // ~$0.001/GB/month
            StorageTier::Ephemeral => Decimal::new(2736, 8),  // ~$0.02/GB/month (higher for fast access)
        }
    }
}

pub fn calculate_tiered_storage_cost(
    storage_gb: Decimal,
    duration_start: DateTime<Utc>,
    duration_end: DateTime<Utc>,
    tier: StorageTier,
) -> crate::error::Result<Decimal> {
    let rate = tier.default_rate();
    calculate_storage_cost(storage_gb, duration_start, duration_end, rate)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;
    use rust_decimal::Decimal;
    use std::str::FromStr;

    #[test]
    fn test_calculate_storage_cost_one_hour() {
        let storage = Decimal::from_str("100.0").unwrap();
        let start = Utc::now();
        let end = start + Duration::hours(1);
        let rate = Decimal::from_str("0.00001").unwrap();

        let result = calculate_storage_cost(storage, start, end, rate).unwrap();
        assert_eq!(result, Decimal::from_str("0.001").unwrap()); // 100 GB * 1 hour * 0.00001
    }

    #[test]
    fn test_calculate_storage_cost_one_day() {
        let storage = Decimal::from_str("100.0").unwrap();
        let start = Utc::now();
        let end = start + Duration::hours(24);
        let rate = Decimal::from_str("0.00001").unwrap();

        let result = calculate_storage_cost(storage, start, end, rate).unwrap();
        assert_eq!(result, Decimal::from_str("0.024").unwrap()); // 100 GB * 24 hours * 0.00001
    }

    #[test]
    fn test_calculate_storage_cost_one_month() {
        let storage = Decimal::from_str("100.0").unwrap();
        let start = Utc::now();
        let end = start + Duration::hours(730); // ~30 days
        let rate = Decimal::from_str("0.00001368").unwrap(); // $0.01/GB/month

        let result = calculate_storage_cost(storage, start, end, rate).unwrap();
        // 100 GB * 730 hours * 0.00001368 ≈ $0.999
        assert!(result > Decimal::from_str("0.99").unwrap() && result < Decimal::from_str("1.01").unwrap());
    }

    #[test]
    fn test_calculate_storage_cost_negative_volume() {
        let storage = Decimal::new(-100, 0);
        let start = Utc::now();
        let end = start + Duration::hours(1);
        let rate = Decimal::from_str("0.00001").unwrap();

        let result = calculate_storage_cost(storage, start, end, rate);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Storage volume cannot be negative"));
    }

    #[test]
    fn test_calculate_storage_cost_negative_rate() {
        let storage = Decimal::from_str("100.0").unwrap();
        let start = Utc::now();
        let end = start + Duration::hours(1);
        let rate = Decimal::new(-1, 5);

        let result = calculate_storage_cost(storage, start, end, rate);
        assert!(result.is_err());
    }

    #[test]
    fn test_calculate_storage_cost_invalid_time_range() {
        let storage = Decimal::from_str("100.0").unwrap();
        let start = Utc::now();
        let end = start - Duration::hours(1);
        let rate = Decimal::from_str("0.00001").unwrap();

        let result = calculate_storage_cost(storage, start, end, rate);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Invalid time range"));
    }

    #[test]
    fn test_calculate_storage_cost_zero_volume() {
        let storage = Decimal::ZERO;
        let start = Utc::now();
        let end = start + Duration::hours(1);
        let rate = Decimal::from_str("0.00001").unwrap();

        let result = calculate_storage_cost(storage, start, end, rate).unwrap();
        assert_eq!(result, Decimal::ZERO);
    }

    #[test]
    fn test_calculate_storage_cost_fractional_gb() {
        let storage = Decimal::from_str("0.5").unwrap();
        let start = Utc::now();
        let end = start + Duration::hours(2);
        let rate = Decimal::from_str("0.00001").unwrap();

        let result = calculate_storage_cost(storage, start, end, rate).unwrap();
        assert_eq!(result, Decimal::from_str("0.00001").unwrap()); // 0.5 GB * 2 hours * 0.00001
    }

    #[test]
    fn test_storage_tier_default_rates() {
        assert!(StorageTier::Standard.default_rate() > Decimal::ZERO);
        assert!(StorageTier::Infrequent.default_rate() < StorageTier::Standard.default_rate());
        assert!(StorageTier::Archive.default_rate() < StorageTier::Infrequent.default_rate());
        assert!(StorageTier::Ephemeral.default_rate() > StorageTier::Standard.default_rate());
    }

    #[test]
    fn test_calculate_tiered_storage_cost_standard() {
        let storage = Decimal::from_str("100.0").unwrap();
        let start = Utc::now();
        let end = start + Duration::hours(730);

        let result = calculate_tiered_storage_cost(storage, start, end, StorageTier::Standard).unwrap();
        // Should be approximately $1.00 for 100GB/month
        assert!(result > Decimal::from_str("0.99").unwrap() && result < Decimal::from_str("1.01").unwrap());
    }

    #[test]
    fn test_calculate_tiered_storage_cost_archive() {
        let storage = Decimal::from_str("100.0").unwrap();
        let start = Utc::now();
        let end = start + Duration::hours(730);

        let result = calculate_tiered_storage_cost(storage, start, end, StorageTier::Archive).unwrap();
        // Should be approximately $0.10 for 100GB/month in archive
        assert!(result > Decimal::from_str("0.09").unwrap() && result < Decimal::from_str("0.11").unwrap());
    }

    #[test]
    fn test_calculate_tiered_storage_cost_ephemeral() {
        let storage = Decimal::from_str("100.0").unwrap();
        let start = Utc::now();
        let end = start + Duration::hours(1);

        let result = calculate_tiered_storage_cost(storage, start, end, StorageTier::Ephemeral).unwrap();
        assert!(result > Decimal::ZERO);
    }

    #[test]
    fn test_large_volume_long_duration() {
        let storage = Decimal::from_str("10000.0").unwrap(); // 10 TB
        let start = Utc::now();
        let end = start + Duration::hours(730);
        let rate = Decimal::from_str("0.00001368").unwrap();

        let result = calculate_storage_cost(storage, start, end, rate).unwrap();
        // 10000 GB * 730 hours * 0.00001368 ≈ $99.864
        assert!(result > Decimal::from_str("99.0").unwrap() && result < Decimal::from_str("101.0").unwrap());
    }

    #[test]
    fn test_very_short_duration() {
        let storage = Decimal::from_str("100.0").unwrap();
        let start = Utc::now();
        let end = start + Duration::seconds(60);
        let rate = Decimal::from_str("0.00001").unwrap();

        let result = calculate_storage_cost(storage, start, end, rate).unwrap();
        // 100 GB * (60/3600) hours * 0.00001 ≈ 0.0000166667
        assert!(result > Decimal::ZERO);
        assert!(result < Decimal::from_str("0.0001").unwrap());
    }
}
