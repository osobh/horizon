// API integration tests
// Note: These tests require a running database and are integration tests

use cost_attributor::api::handlers::health::HealthResponse;

#[tokio::test]
async fn test_health_check_response() {
    let response = cost_attributor::api::handlers::health::health_check().await;
    assert_eq!(response.0.status, "healthy");
    assert_eq!(response.0.service, "cost-attributor");
}

#[test]
fn test_attribution_request_serialization() {
    use chrono::Utc;
    use cost_attributor::api::handlers::attributions::CreateAttributionRequest;
    use rust_decimal_macros::dec;
    use uuid::Uuid;

    let now = Utc::now();
    let req = CreateAttributionRequest {
        job_id: Some(Uuid::new_v4()),
        user_id: "user123".to_string(),
        team_id: Some("team456".to_string()),
        customer_id: None,
        gpu_cost: dec!(100.00),
        cpu_cost: dec!(0.00),
        network_cost: dec!(10.00),
        storage_cost: dec!(1.00),
        total_cost: dec!(111.00),
        period_start: now,
        period_end: now + chrono::Duration::hours(1),
    };

    let json = serde_json::to_string(&req).unwrap();
    assert!(json.contains("user123"));

    let deserialized: CreateAttributionRequest = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.user_id, "user123");
}

#[test]
fn test_calculate_attribution_request_serialization() {
    use chrono::Utc;
    use cost_attributor::api::handlers::attributions::CalculateAttributionRequest;
    use rust_decimal_macros::dec;
    use uuid::Uuid;

    let now = Utc::now();
    let req = CalculateAttributionRequest {
        job_id: Uuid::new_v4(),
        user_id: "user123".to_string(),
        team_id: Some("team456".to_string()),
        customer_id: None,
        start_time: now,
        end_time: now + chrono::Duration::hours(2),
        gpu_count: 4,
        gpu_type: "A100".to_string(),
        network_ingress_gb: dec!(10.0),
        network_egress_gb: dec!(50.0),
        storage_gb: dec!(100.0),
    };

    let json = serde_json::to_string(&req).unwrap();
    assert!(json.contains("A100"));
    assert!(json.contains("user123"));
}

#[test]
fn test_pricing_request_serialization() {
    use chrono::Utc;
    use cost_attributor::{api::handlers::pricing::CreatePricingRequest, models::PricingModel};
    use rust_decimal_macros::dec;

    let now = Utc::now();
    let req = CreatePricingRequest {
        gpu_type: "A100".to_string(),
        region: Some("us-east-1".to_string()),
        pricing_model: PricingModel::OnDemand,
        hourly_rate: dec!(3.50),
        effective_start: now,
        effective_end: None,
    };

    let json = serde_json::to_string(&req).unwrap();
    assert!(json.contains("A100"));
    assert!(json.contains("on_demand"));

    let deserialized: CreatePricingRequest = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.gpu_type, "A100");
}

#[test]
fn test_rollup_query_params_serialization() {
    use chrono::Utc;
    use cost_attributor::api::handlers::attributions::RollupQueryParams;

    let now = Utc::now();
    let params = RollupQueryParams {
        start_date: now,
        end_date: now + chrono::Duration::hours(24),
    };

    let json = serde_json::to_string(&params).unwrap();
    let deserialized: RollupQueryParams = serde_json::from_str(&json).unwrap();

    assert_eq!(deserialized.start_date, params.start_date);
    assert_eq!(deserialized.end_date, params.end_date);
}

#[test]
fn test_health_response_format() {
    let response = HealthResponse {
        status: "healthy".to_string(),
        service: "cost-attributor".to_string(),
        version: "0.1.0".to_string(),
    };

    let json = serde_json::to_string(&response).unwrap();
    assert!(json.contains("healthy"));
    assert!(json.contains("cost-attributor"));
}

#[test]
fn test_cost_attribution_query_default() {
    use cost_attributor::models::CostAttributionQuery;

    let query = CostAttributionQuery::default();
    assert_eq!(query.limit, Some(100));
    assert_eq!(query.offset, Some(0));
}

#[test]
fn test_gpu_pricing_query_default() {
    use cost_attributor::models::GpuPricingQuery;

    let query = GpuPricingQuery::default();
    assert!(query.gpu_type.is_none());
    assert!(query.region.is_none());
}

#[test]
fn test_cost_rollup_structure() {
    use chrono::Utc;
    use cost_attributor::models::CostRollup;
    use rust_decimal_macros::dec;

    let now = Utc::now();
    let rollup = CostRollup {
        entity_id: "user123".to_string(),
        entity_type: "user".to_string(),
        total_gpu_cost: dec!(1000.00),
        total_cpu_cost: dec!(50.00),
        total_network_cost: dec!(100.00),
        total_storage_cost: dec!(10.00),
        total_cost: dec!(1160.00),
        job_count: 42,
        period_start: now,
        period_end: now + chrono::Duration::hours(24),
    };

    assert_eq!(rollup.entity_id, "user123");
    assert_eq!(rollup.job_count, 42);
}

#[test]
fn test_update_pricing_request() {
    use chrono::Utc;
    use cost_attributor::api::handlers::pricing::UpdatePricingRequest;
    use rust_decimal_macros::dec;

    let req = UpdatePricingRequest {
        hourly_rate: Some(dec!(4.00)),
        effective_end: Some(Utc::now()),
    };

    let json = serde_json::to_string(&req).unwrap();
    let deserialized: UpdatePricingRequest = serde_json::from_str(&json).unwrap();

    assert_eq!(deserialized.hourly_rate, Some(dec!(4.00)));
}
