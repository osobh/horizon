use chrono::{DateTime, Utc};
use rust_decimal::Decimal;

/// Calculate GPU hours from job start/end times with GPU count
/// Precise to the second
pub fn calculate_gpu_hours(
    start: DateTime<Utc>,
    end: DateTime<Utc>,
    gpu_count: usize,
) -> crate::error::Result<Decimal> {
    use crate::error::AttributorErrorExt;

    if end <= start {
        return Err(crate::error::HpcError::invalid_time_range(
            start.to_rfc3339(),
            end.to_rfc3339(),
        ));
    }

    if gpu_count == 0 {
        return Err(crate::error::HpcError::calculation_error(
            "GPU count must be greater than 0",
        ));
    }

    let duration_seconds = (end - start).num_seconds();
    if duration_seconds < 0 {
        return Err(crate::error::HpcError::calculation_error(
            "Duration cannot be negative",
        ));
    }

    // Convert seconds to hours with high precision
    let hours = Decimal::from(duration_seconds) / Decimal::from(3600);
    let gpu_hours = hours * Decimal::from(gpu_count);

    Ok(gpu_hours)
}

/// Calculate GPU cost from GPU hours and hourly rate
pub fn calculate_gpu_cost(gpu_hours: Decimal, hourly_rate: Decimal) -> crate::error::Result<Decimal> {
    use crate::error::AttributorErrorExt;

    if gpu_hours < Decimal::ZERO {
        return Err(crate::error::HpcError::calculation_error(
            "GPU hours cannot be negative",
        ));
    }

    if hourly_rate < Decimal::ZERO {
        return Err(crate::error::HpcError::calculation_error(
            "Hourly rate cannot be negative",
        ));
    }

    Ok(gpu_hours * hourly_rate)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;
    use rust_decimal_macros::dec;

    #[test]
    fn test_calculate_gpu_hours_one_hour_one_gpu() {
        let start = Utc::now();
        let end = start + Duration::hours(1);

        let result = calculate_gpu_hours(start, end, 1).unwrap();
        assert_eq!(result, dec!(1.0));
    }

    #[test]
    fn test_calculate_gpu_hours_one_hour_four_gpus() {
        let start = Utc::now();
        let end = start + Duration::hours(1);

        let result = calculate_gpu_hours(start, end, 4).unwrap();
        assert_eq!(result, dec!(4.0));
    }

    #[test]
    fn test_calculate_gpu_hours_partial_hour() {
        let start = Utc::now();
        let end = start + Duration::minutes(30);

        let result = calculate_gpu_hours(start, end, 1).unwrap();
        assert_eq!(result, dec!(0.5));
    }

    #[test]
    fn test_calculate_gpu_hours_precise_seconds() {
        let start = Utc::now();
        let end = start + Duration::seconds(3600); // Exactly 1 hour

        let result = calculate_gpu_hours(start, end, 1).unwrap();
        assert_eq!(result, dec!(1.0));
    }

    #[test]
    fn test_calculate_gpu_hours_90_seconds() {
        let start = Utc::now();
        let end = start + Duration::seconds(90);

        let result = calculate_gpu_hours(start, end, 1).unwrap();
        assert_eq!(result, dec!(0.025)); // 90/3600 = 0.025
    }

    #[test]
    fn test_calculate_gpu_hours_multiple_gpus_partial_hour() {
        let start = Utc::now();
        let end = start + Duration::minutes(15);

        let result = calculate_gpu_hours(start, end, 8).unwrap();
        assert_eq!(result, dec!(2.0)); // 0.25 hours * 8 GPUs
    }

    #[test]
    fn test_calculate_gpu_hours_invalid_time_range() {
        let start = Utc::now();
        let end = start - Duration::hours(1);

        let result = calculate_gpu_hours(start, end, 1);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Invalid time range"));
    }

    #[test]
    fn test_calculate_gpu_hours_zero_gpus() {
        let start = Utc::now();
        let end = start + Duration::hours(1);

        let result = calculate_gpu_hours(start, end, 0);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("GPU count must be greater than 0"));
    }

    #[test]
    fn test_calculate_gpu_hours_equal_times() {
        let start = Utc::now();
        let end = start;

        let result = calculate_gpu_hours(start, end, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_calculate_gpu_cost_basic() {
        let gpu_hours = dec!(10.0);
        let hourly_rate = dec!(3.50);

        let result = calculate_gpu_cost(gpu_hours, hourly_rate).unwrap();
        assert_eq!(result, dec!(35.00));
    }

    #[test]
    fn test_calculate_gpu_cost_fractional_hours() {
        let gpu_hours = dec!(0.5);
        let hourly_rate = dec!(4.00);

        let result = calculate_gpu_cost(gpu_hours, hourly_rate).unwrap();
        assert_eq!(result, dec!(2.00));
    }

    #[test]
    fn test_calculate_gpu_cost_high_precision() {
        let gpu_hours = dec!(0.025); // 90 seconds
        let hourly_rate = dec!(3.50);

        let result = calculate_gpu_cost(gpu_hours, hourly_rate).unwrap();
        assert_eq!(result, dec!(0.0875));
    }

    #[test]
    fn test_calculate_gpu_cost_zero_hours() {
        let gpu_hours = Decimal::ZERO;
        let hourly_rate = dec!(3.50);

        let result = calculate_gpu_cost(gpu_hours, hourly_rate).unwrap();
        assert_eq!(result, Decimal::ZERO);
    }

    #[test]
    fn test_calculate_gpu_cost_negative_hours() {
        let gpu_hours = dec!(-1.0);
        let hourly_rate = dec!(3.50);

        let result = calculate_gpu_cost(gpu_hours, hourly_rate);
        assert!(result.is_err());
    }

    #[test]
    fn test_calculate_gpu_cost_negative_rate() {
        let gpu_hours = dec!(10.0);
        let hourly_rate = dec!(-3.50);

        let result = calculate_gpu_cost(gpu_hours, hourly_rate);
        assert!(result.is_err());
    }

    #[test]
    fn test_long_running_job() {
        let start = Utc::now();
        let end = start + Duration::hours(24 * 7); // 1 week

        let result = calculate_gpu_hours(start, end, 8).unwrap();
        assert_eq!(result, dec!(1344.0)); // 168 hours * 8 GPUs
    }

    #[test]
    fn test_very_short_job() {
        let start = Utc::now();
        let end = start + Duration::seconds(1);

        let result = calculate_gpu_hours(start, end, 1).unwrap();
        // 1/3600 ≈ 0.000277777...
        assert!(result > Decimal::ZERO);
        assert!(result < dec!(0.001));
    }
}
