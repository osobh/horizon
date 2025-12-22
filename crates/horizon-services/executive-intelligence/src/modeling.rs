use rust_decimal::Decimal;

pub struct ScenarioModeler;

impl ScenarioModeler {
    pub fn run_simulation(base: Decimal) -> Decimal {
        base * Decimal::from_str_exact("1.1").unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_simulation() {
        let result = ScenarioModeler::run_simulation(dec!(100));
        assert_eq!(result, dec!(110));
    }
}
