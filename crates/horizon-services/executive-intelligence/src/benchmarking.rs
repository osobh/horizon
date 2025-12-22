use rust_decimal::Decimal;

pub struct IndustryBenchmark;

impl IndustryBenchmark {
    pub fn compare(our_value: Decimal, industry: Decimal) -> Decimal {
        our_value - industry
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_comparison() {
        let diff = IndustryBenchmark::compare(dec!(100), dec!(90));
        assert_eq!(diff, dec!(10));
    }
}
