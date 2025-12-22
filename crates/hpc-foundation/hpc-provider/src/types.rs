use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuoteRequest {
    pub instance_type: String,
    pub region: String,
    pub count: usize,
    pub duration_hours: Option<u32>,
    pub spot: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Quote {
    pub provider: String,
    pub instance_type: String,
    pub region: String,
    pub hourly_rate: Decimal,
    pub spot_rate: Option<Decimal>,
    pub availability: Availability,
    pub lead_time_hours: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvisionSpec {
    pub instance_type: String,
    pub region: String,
    pub count: usize,
    pub spot: bool,
    pub tags: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvisionResult {
    pub instances: Vec<Instance>,
    pub total_cost_estimate: Decimal,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Instance {
    pub id: String,
    pub instance_type: String,
    pub region: String,
    pub public_ip: Option<String>,
    pub private_ip: String,
    pub state: InstanceState,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum Availability {
    Available,
    Limited,
    Unavailable,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum InstanceState {
    Pending,
    Running,
    Stopping,
    Stopped,
    Terminated,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpotPrices {
    pub instance_type: String,
    pub region: String,
    pub current_price: Decimal,
    pub average_price: Decimal,
    pub min_price: Decimal,
    pub max_price: Decimal,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceQuotas {
    pub max_instances: usize,
    pub current_instances: usize,
    pub max_vcpus: usize,
    pub current_vcpus: usize,
    pub max_gpus: usize,
    pub current_gpus: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatus {
    pub healthy: bool,
    pub message: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;
    use std::collections::HashMap;

    #[test]
    fn test_quote_request_creation() {
        let request = QuoteRequest {
            instance_type: "p4.xlarge".to_string(),
            region: "us-east-1".to_string(),
            count: 5,
            duration_hours: Some(24),
            spot: true,
        };

        assert_eq!(request.instance_type, "p4.xlarge");
        assert_eq!(request.region, "us-east-1");
        assert_eq!(request.count, 5);
        assert_eq!(request.duration_hours, Some(24));
        assert!(request.spot);
    }

    #[test]
    fn test_quote_serialization() {
        let quote = Quote {
            provider: "aws".to_string(),
            instance_type: "p4.xlarge".to_string(),
            region: "us-east-1".to_string(),
            hourly_rate: dec!(2.50),
            spot_rate: Some(dec!(1.00)),
            availability: Availability::Available,
            lead_time_hours: 0,
        };

        let json = serde_json::to_string(&quote).unwrap();
        let deserialized: Quote = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.provider, "aws");
        assert_eq!(deserialized.hourly_rate, dec!(2.50));
        assert_eq!(deserialized.spot_rate, Some(dec!(1.00)));
    }

    #[test]
    fn test_provision_spec_with_tags() {
        let mut tags = HashMap::new();
        tags.insert("env".to_string(), "prod".to_string());
        tags.insert("team".to_string(), "ml".to_string());

        let spec = ProvisionSpec {
            instance_type: "p4.xlarge".to_string(),
            region: "us-west-2".to_string(),
            count: 10,
            spot: false,
            tags: tags.clone(),
        };

        assert_eq!(spec.tags.len(), 2);
        assert_eq!(spec.tags.get("env"), Some(&"prod".to_string()));
        assert_eq!(spec.tags.get("team"), Some(&"ml".to_string()));
    }

    #[test]
    fn test_instance_state_equality() {
        assert_eq!(InstanceState::Pending, InstanceState::Pending);
        assert_eq!(InstanceState::Running, InstanceState::Running);
        assert_ne!(InstanceState::Pending, InstanceState::Running);
    }

    #[test]
    fn test_availability_equality() {
        assert_eq!(Availability::Available, Availability::Available);
        assert_eq!(Availability::Limited, Availability::Limited);
        assert_ne!(Availability::Available, Availability::Unavailable);
    }

    #[test]
    fn test_spot_prices_structure() {
        let prices = SpotPrices {
            instance_type: "p4.xlarge".to_string(),
            region: "us-east-1".to_string(),
            current_price: dec!(0.75),
            average_price: dec!(0.80),
            min_price: dec!(0.50),
            max_price: dec!(1.20),
        };

        assert!(prices.current_price > prices.min_price);
        assert!(prices.current_price < prices.max_price);
        assert!(prices.average_price > prices.min_price);
        assert!(prices.average_price < prices.max_price);
    }

    #[test]
    fn test_service_quotas_structure() {
        let quotas = ServiceQuotas {
            max_instances: 100,
            current_instances: 50,
            max_vcpus: 400,
            current_vcpus: 200,
            max_gpus: 80,
            current_gpus: 40,
        };

        assert!(quotas.current_instances <= quotas.max_instances);
        assert!(quotas.current_vcpus <= quotas.max_vcpus);
        assert!(quotas.current_gpus <= quotas.max_gpus);
    }

    #[test]
    fn test_health_status() {
        let healthy = HealthStatus {
            healthy: true,
            message: "OK".to_string(),
        };

        assert!(healthy.healthy);
        assert_eq!(healthy.message, "OK");
    }

    #[test]
    fn test_provision_result_structure() {
        let instance = Instance {
            id: "i-123".to_string(),
            instance_type: "p4.xlarge".to_string(),
            region: "us-east-1".to_string(),
            public_ip: Some("1.2.3.4".to_string()),
            private_ip: "10.0.0.1".to_string(),
            state: InstanceState::Running,
        };

        let result = ProvisionResult {
            instances: vec![instance],
            total_cost_estimate: dec!(100.00),
        };

        assert_eq!(result.instances.len(), 1);
        assert_eq!(result.total_cost_estimate, dec!(100.00));
        assert_eq!(result.instances[0].id, "i-123");
    }
}
