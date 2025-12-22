use rust_decimal::Decimal;

pub struct CostAnalyzer;

impl CostAnalyzer {
    pub fn analyze_impact(cost: Decimal) -> Decimal {
        cost
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_analyze_impact() {
        let result = CostAnalyzer::analyze_impact(dec!(100));
        assert_eq!(result, dec!(100));
    }
}
