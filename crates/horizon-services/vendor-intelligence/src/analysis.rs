use rust_decimal::Decimal;

pub struct VendorAnalyzer;

impl VendorAnalyzer {
    pub fn calculate_utilization(committed: Decimal, used: Decimal) -> Decimal {
        if committed.is_zero() {
            Decimal::ZERO
        } else {
            (used / committed) * Decimal::from(100)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_utilization() {
        let util = VendorAnalyzer::calculate_utilization(dec!(100), dec!(75));
        assert_eq!(util, dec!(75));
    }
}
