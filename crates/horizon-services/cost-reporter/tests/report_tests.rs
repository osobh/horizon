use chrono::{Datelike, Utc};
use cost_reporter::models::{
    report::{ChargebackLineItem, ChargebackReport, Period, ShowbackReport},
    summary::CostBreakdown,
    trend::{TrendAnalysis, TrendDirection},
};
use rust_decimal_macros::dec;

#[test]
fn test_period_to_date_range_current_month() {
    let period = Period::CurrentMonth;
    let now = Utc::now();
    let (start, end) = period.to_date_range(now);

    assert_eq!(start.day(), 1);
    assert!(end >= start);
}

#[test]
fn test_period_to_date_range_year_to_date() {
    let period = Period::YearToDate;
    let now = Utc::now();
    let (start, end) = period.to_date_range(now);

    assert_eq!(start.month(), 1);
    assert_eq!(start.day(), 1);
    assert!(end >= start);
}

#[test]
fn test_showback_report_creation() {
    let now = Utc::now();
    let later = now + chrono::Duration::hours(24);

    let breakdown = CostBreakdown {
        gpu_cost: dec!(100.00),
        cpu_cost: dec!(20.00),
        network_cost: dec!(5.00),
        storage_cost: dec!(2.50),
        total_cost: dec!(127.50),
    };

    let report = ShowbackReport::new(
        "team123".to_string(),
        "team".to_string(),
        Period::CurrentMonth,
        now,
        later,
    )
    .with_breakdown(breakdown)
    .with_job_count(42);

    assert_eq!(report.entity_id, "team123");
    assert_eq!(report.entity_type, "team");
    assert_eq!(report.breakdown.total_cost, dec!(127.50));
    assert_eq!(report.job_count, 42);
}

#[test]
fn test_chargeback_report_creation() {
    let now = Utc::now();
    let later = now + chrono::Duration::hours(24);

    let mut report = ChargebackReport::new("customer123".to_string(), now, later);

    let item1 = ChargebackLineItem::new(
        "GPU Usage".to_string(),
        "gpu".to_string(),
        dec!(10.0),
        "hours".to_string(),
        dec!(5.00),
    );

    let item2 = ChargebackLineItem::new(
        "Storage".to_string(),
        "storage".to_string(),
        dec!(100.0),
        "GB".to_string(),
        dec!(0.10),
    );

    report.add_line_item(item1);
    report.add_line_item(item2);

    assert_eq!(report.customer_id, "customer123");
    assert_eq!(report.line_items.len(), 2);
    assert_eq!(report.total_amount, dec!(60.00)); // 50 + 10
}

#[test]
fn test_cost_breakdown_aggregation() {
    use cost_reporter::models::summary::{DailyCostSummary, HasCostBreakdown};

    let summaries = vec![
        DailyCostSummary {
            day: Utc::now(),
            team_id: Some("team1".to_string()),
            user_id: Some("user1".to_string()),
            total_cost: dec!(100.00),
            gpu_cost: dec!(80.00),
            cpu_cost: dec!(10.00),
            network_cost: dec!(5.00),
            storage_cost: dec!(5.00),
            job_count: 10,
        },
        DailyCostSummary {
            day: Utc::now(),
            team_id: Some("team1".to_string()),
            user_id: Some("user2".to_string()),
            total_cost: dec!(50.00),
            gpu_cost: dec!(40.00),
            cpu_cost: dec!(5.00),
            network_cost: dec!(3.00),
            storage_cost: dec!(2.00),
            job_count: 5,
        },
    ];

    let breakdown = CostBreakdown::from_summaries(&summaries);
    assert_eq!(breakdown.gpu_cost, dec!(120.00));
    assert_eq!(breakdown.cpu_cost, dec!(15.00));
    assert_eq!(breakdown.total_cost, dec!(150.00));
}

#[test]
fn test_trend_analysis_creation() {
    let trend = TrendAnalysis::new(
        TrendDirection::Increasing,
        dec!(15.5),
        dec!(100.00),
        0.5,
    )
    .with_confidence(0.85);

    assert_eq!(trend.direction, TrendDirection::Increasing);
    assert_eq!(trend.growth_rate, dec!(15.5));
    assert_eq!(trend.confidence, 0.85);
}

#[test]
fn test_trend_confidence_clamping() {
    let trend = TrendAnalysis::new(
        TrendDirection::Stable,
        dec!(0.0),
        dec!(50.00),
        0.0,
    )
    .with_confidence(1.5);

    assert_eq!(trend.confidence, 1.0); // Clamped to 1.0

    let trend2 = TrendAnalysis::new(
        TrendDirection::Stable,
        dec!(0.0),
        dec!(50.00),
        0.0,
    )
    .with_confidence(-0.5);

    assert_eq!(trend2.confidence, 0.0); // Clamped to 0.0
}

#[test]
fn test_forecast_point_confidence_clamping() {
    use cost_reporter::models::trend::ForecastPoint;

    let point = ForecastPoint::new(Utc::now(), dec!(100.00), 1.5);
    assert_eq!(point.confidence, 1.0);

    let point2 = ForecastPoint::new(Utc::now(), dec!(100.00), -0.5);
    assert_eq!(point2.confidence, 0.0);
}

#[test]
fn test_cost_forecast_avg_confidence() {
    use cost_reporter::models::trend::{CostForecast, ForecastPoint};

    let now = Utc::now();
    let points = vec![
        ForecastPoint::new(now, dec!(100.00), 0.9),
        ForecastPoint::new(now, dec!(110.00), 0.8),
        ForecastPoint::new(now, dec!(120.00), 0.7),
    ];

    let forecast = CostForecast::new(
        now,
        now,
        now,
        now,
        "linear_regression".to_string(),
        points,
    );

    assert!((forecast.avg_confidence - 0.8).abs() < 0.001);
}

#[test]
fn test_chargeback_line_item_calculation() {
    let item = ChargebackLineItem::new(
        "GPU Usage".to_string(),
        "gpu".to_string(),
        dec!(10.5),
        "hours".to_string(),
        dec!(5.00),
    );

    assert_eq!(item.amount, dec!(52.50));
    assert_eq!(item.quantity, dec!(10.5));
    assert_eq!(item.unit_price, dec!(5.00));
    assert_eq!(item.resource_type, "gpu");
}

#[test]
fn test_period_serialization() {
    let period = Period::CurrentMonth;
    let json = serde_json::to_string(&period).unwrap();
    assert_eq!(json, r#""current_month""#);

    let period2: Period = serde_json::from_str(&json).unwrap();
    assert_eq!(period2, Period::CurrentMonth);
}

#[test]
fn test_showback_report_serialization() {
    let now = Utc::now();
    let later = now + chrono::Duration::hours(24);

    let report = ShowbackReport::new(
        "team123".to_string(),
        "team".to_string(),
        Period::CurrentMonth,
        now,
        later,
    );

    let json = serde_json::to_string(&report).unwrap();
    assert!(json.contains("team123"));
    assert!(json.contains("team"));

    let _deserialized: ShowbackReport = serde_json::from_str(&json).unwrap();
}

#[test]
fn test_chargeback_report_serialization() {
    let now = Utc::now();
    let later = now + chrono::Duration::hours(24);

    let report = ChargebackReport::new("customer123".to_string(), now, later);

    let json = serde_json::to_string(&report).unwrap();
    assert!(json.contains("customer123"));

    let _deserialized: ChargebackReport = serde_json::from_str(&json).unwrap();
}
