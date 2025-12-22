use rust_decimal::Decimal;

/// Calculate network cost from ingress/egress data in GB
pub fn calculate_network_cost(
    ingress_gb: Decimal,
    egress_gb: Decimal,
    rate_per_gb: Decimal,
) -> crate::error::Result<Decimal> {
    use crate::error::AttributorErrorExt;

    if ingress_gb < Decimal::ZERO {
        return Err(crate::error::HpcError::calculation_error(
            "Ingress cannot be negative",
        ));
    }

    if egress_gb < Decimal::ZERO {
        return Err(crate::error::HpcError::calculation_error(
            "Egress cannot be negative",
        ));
    }

    if rate_per_gb < Decimal::ZERO {
        return Err(crate::error::HpcError::calculation_error(
            "Rate per GB cannot be negative",
        ));
    }

    // Typically, ingress is free and only egress is charged
    // But we support both for flexibility
    let total_gb = ingress_gb + egress_gb;
    let cost = total_gb * rate_per_gb;

    Ok(cost)
}

/// Calculate network cost for egress only (most common scenario)
pub fn calculate_egress_cost(egress_gb: Decimal, rate_per_gb: Decimal) -> crate::error::Result<Decimal> {
    calculate_network_cost(Decimal::ZERO, egress_gb, rate_per_gb)
}

/// Calculate cross-region transfer cost with different pricing
pub fn calculate_cross_region_cost(
    transfer_gb: Decimal,
    source_region: &str,
    dest_region: &str,
    rate_per_gb: Decimal,
) -> crate::error::Result<Decimal> {
    use crate::error::AttributorErrorExt;

    if transfer_gb < Decimal::ZERO {
        return Err(crate::error::HpcError::calculation_error(
            "Transfer amount cannot be negative",
        ));
    }

    if rate_per_gb < Decimal::ZERO {
        return Err(crate::error::HpcError::calculation_error(
            "Rate per GB cannot be negative",
        ));
    }

    // Same region = no cost
    if source_region == dest_region {
        return Ok(Decimal::ZERO);
    }

    let cost = transfer_gb * rate_per_gb;
    Ok(cost)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_calculate_network_cost_basic() {
        let ingress = dec!(10.0);
        let egress = dec!(50.0);
        let rate = dec!(0.09);

        let result = calculate_network_cost(ingress, egress, rate).unwrap();
        assert_eq!(result, dec!(5.40)); // (10 + 50) * 0.09
    }

    #[test]
    fn test_calculate_network_cost_zero_ingress() {
        let ingress = Decimal::ZERO;
        let egress = dec!(100.0);
        let rate = dec!(0.09);

        let result = calculate_network_cost(ingress, egress, rate).unwrap();
        assert_eq!(result, dec!(9.00));
    }

    #[test]
    fn test_calculate_network_cost_zero_egress() {
        let ingress = dec!(100.0);
        let egress = Decimal::ZERO;
        let rate = dec!(0.09);

        let result = calculate_network_cost(ingress, egress, rate).unwrap();
        assert_eq!(result, dec!(9.00));
    }

    #[test]
    fn test_calculate_network_cost_negative_ingress() {
        let ingress = dec!(-10.0);
        let egress = dec!(50.0);
        let rate = dec!(0.09);

        let result = calculate_network_cost(ingress, egress, rate);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Ingress cannot be negative"));
    }

    #[test]
    fn test_calculate_network_cost_negative_egress() {
        let ingress = dec!(10.0);
        let egress = dec!(-50.0);
        let rate = dec!(0.09);

        let result = calculate_network_cost(ingress, egress, rate);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Egress cannot be negative"));
    }

    #[test]
    fn test_calculate_network_cost_negative_rate() {
        let ingress = dec!(10.0);
        let egress = dec!(50.0);
        let rate = dec!(-0.09);

        let result = calculate_network_cost(ingress, egress, rate);
        assert!(result.is_err());
    }

    #[test]
    fn test_calculate_egress_cost_only() {
        let egress = dec!(100.0);
        let rate = dec!(0.09);

        let result = calculate_egress_cost(egress, rate).unwrap();
        assert_eq!(result, dec!(9.00));
    }

    #[test]
    fn test_calculate_egress_cost_fractional() {
        let egress = dec!(0.5);
        let rate = dec!(0.09);

        let result = calculate_egress_cost(egress, rate).unwrap();
        assert_eq!(result, dec!(0.045));
    }

    #[test]
    fn test_calculate_cross_region_cost_same_region() {
        let transfer = dec!(100.0);
        let rate = dec!(0.02);

        let result = calculate_cross_region_cost(transfer, "us-east-1", "us-east-1", rate).unwrap();
        assert_eq!(result, Decimal::ZERO);
    }

    #[test]
    fn test_calculate_cross_region_cost_different_region() {
        let transfer = dec!(100.0);
        let rate = dec!(0.02);

        let result = calculate_cross_region_cost(transfer, "us-east-1", "us-west-2", rate).unwrap();
        assert_eq!(result, dec!(2.00));
    }

    #[test]
    fn test_calculate_cross_region_cost_negative_transfer() {
        let transfer = dec!(-100.0);
        let rate = dec!(0.02);

        let result = calculate_cross_region_cost(transfer, "us-east-1", "us-west-2", rate);
        assert!(result.is_err());
    }

    #[test]
    fn test_calculate_cross_region_cost_negative_rate() {
        let transfer = dec!(100.0);
        let rate = dec!(-0.02);

        let result = calculate_cross_region_cost(transfer, "us-east-1", "us-west-2", rate);
        assert!(result.is_err());
    }

    #[test]
    fn test_large_transfer_cost() {
        let egress = dec!(10000.0); // 10 TB
        let rate = dec!(0.09);

        let result = calculate_egress_cost(egress, rate).unwrap();
        assert_eq!(result, dec!(900.00));
    }

    #[test]
    fn test_zero_cost_scenario() {
        let result = calculate_network_cost(Decimal::ZERO, Decimal::ZERO, dec!(0.09)).unwrap();
        assert_eq!(result, Decimal::ZERO);
    }
}
