use rust_decimal::Decimal;

pub struct MarginCalculator;

impl MarginCalculator {
    pub fn gross_margin(revenue: Decimal, cost: Decimal) -> Option<Decimal> {
        if revenue.is_zero() {
            return None;
        }
        Some((revenue - cost) / revenue * Decimal::from(100))
    }

    pub fn contribution_margin(revenue: Decimal, cost: Decimal) -> Decimal {
        revenue - cost
    }

    pub fn cogs(total_cost: Decimal, fixed_costs: Decimal) -> Decimal {
        total_cost - fixed_costs
    }

    pub fn operating_margin(
        revenue: Decimal,
        cost: Decimal,
        operating_expenses: Decimal,
    ) -> Option<Decimal> {
        if revenue.is_zero() {
            return None;
        }
        Some((revenue - cost - operating_expenses) / revenue * Decimal::from(100))
    }

    pub fn roi(gain: Decimal, cost: Decimal) -> Option<Decimal> {
        if cost.is_zero() {
            return None;
        }
        Some((gain - cost) / cost * Decimal::from(100))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_gross_margin() {
        assert_eq!(
            MarginCalculator::gross_margin(dec!(10000), dec!(7000)),
            Some(dec!(30.00))
        );
    }

    #[test]
    fn test_gross_margin_zero_revenue() {
        assert_eq!(MarginCalculator::gross_margin(dec!(0), dec!(100)), None);
    }

    #[test]
    fn test_contribution_margin() {
        assert_eq!(
            MarginCalculator::contribution_margin(dec!(10000), dec!(7000)),
            dec!(3000)
        );
    }

    #[test]
    fn test_cogs() {
        assert_eq!(MarginCalculator::cogs(dec!(10000), dec!(2000)), dec!(8000));
    }

    #[test]
    fn test_operating_margin() {
        assert_eq!(
            MarginCalculator::operating_margin(dec!(10000), dec!(6000), dec!(2000)),
            Some(dec!(20.00))
        );
    }

    #[test]
    fn test_roi() {
        assert_eq!(
            MarginCalculator::roi(dec!(15000), dec!(10000)),
            Some(dec!(50.00))
        );
    }
}
