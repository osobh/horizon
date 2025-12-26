use rust_decimal::Decimal;

pub struct MetricsMeasurement;

impl MetricsMeasurement {
    pub fn calculate_variance(baseline: Decimal, actual: Decimal) -> Decimal {
        actual - baseline
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_variance() {
        let var = MetricsMeasurement::calculate_variance(dec!(100), dec!(120));
        assert_eq!(var, dec!(20));
    }
}
