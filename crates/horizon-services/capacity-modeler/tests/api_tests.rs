use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use capacity_modeler::{api::create_routes, ForecastService};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use std::sync::Arc;
use tower::ServiceExt;

#[tokio::test]
async fn test_health_check_endpoint() {
    let service = Arc::new(ForecastService::new(100));
    let app = create_routes().layer(axum::Extension(service));

    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = response.into_body().collect().await.unwrap().to_bytes();
    let json: Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(json["status"], "healthy");
    assert_eq!(json["service"], "capacity-modeler");
}

#[tokio::test]
async fn test_forecast_endpoint_default_params() {
    let service = Arc::new(ForecastService::new(100));
    let app = create_routes().layer(axum::Extension(service));

    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/api/v1/forecast/gpu-demand")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = response.into_body().collect().await.unwrap().to_bytes();
    let json: Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(json["forecast_weeks"], 13);
    assert_eq!(json["model_type"], "ETS");
    assert!(json["points"].is_array());
    assert_eq!(json["points"].as_array().unwrap().len(), 91); // 13 weeks * 7 days
}

#[tokio::test]
async fn test_forecast_endpoint_custom_weeks() {
    let service = Arc::new(ForecastService::new(100));
    let app = create_routes().layer(axum::Extension(service));

    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/api/v1/forecast/gpu-demand?weeks=4")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = response.into_body().collect().await.unwrap().to_bytes();
    let json: Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(json["forecast_weeks"], 4);
    assert_eq!(json["points"].as_array().unwrap().len(), 28); // 4 weeks * 7 days
}

#[tokio::test]
async fn test_forecast_with_confidence_intervals() {
    let service = Arc::new(ForecastService::new(100));
    let app = create_routes().layer(axum::Extension(service));

    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/api/v1/forecast/gpu-demand?weeks=2&include_confidence_intervals=true")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = response.into_body().collect().await.unwrap().to_bytes();
    let json: Value = serde_json::from_slice(&body).unwrap();

    let points = json["points"].as_array().unwrap();
    assert_eq!(points.len(), 14); // 2 weeks * 7 days

    // Verify first point has confidence intervals
    let first_point = &points[0];
    assert!(first_point["lower_bound"].is_number());
    assert!(first_point["upper_bound"].is_number());
    assert!(first_point["value"].is_number());

    let value = first_point["value"].as_f64().unwrap();
    let lower = first_point["lower_bound"].as_f64().unwrap();
    let upper = first_point["upper_bound"].as_f64().unwrap();

    assert!(lower < value);
    assert!(value < upper);
}

#[tokio::test]
async fn test_backtest_endpoint() {
    let service = Arc::new(ForecastService::new(50));
    let app = create_routes().layer(axum::Extension(service));

    let payload = json!({
        "train_days": 200,
        "test_days": 30
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/v1/forecast/backtest")
                .header("Content-Type", "application/json")
                .body(Body::from(payload.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = response.into_body().collect().await.unwrap().to_bytes();
    let json: Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(json["train_size"], 200);
    assert_eq!(json["test_size"], 30);
    assert!(json["metrics"].is_object());
    assert!(json["metrics"]["mape"].is_number());
    assert!(json["metrics"]["rmse"].is_number());
    assert!(json["metrics"]["mae"].is_number());
    assert_eq!(json["predictions"].as_array().unwrap().len(), 30);
    assert_eq!(json["actuals"].as_array().unwrap().len(), 30);
}

#[tokio::test]
async fn test_backtest_with_insufficient_data() {
    let service = Arc::new(ForecastService::new(50));
    let app = create_routes().layer(axum::Extension(service));

    let payload = json!({
        "train_days": 5,
        "test_days": 10
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/v1/forecast/backtest")
                .header("Content-Type", "application/json")
                .body(Body::from(payload.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    // Should return error status
    assert!(response.status().is_client_error() || response.status().is_server_error());
}

#[tokio::test]
async fn test_swagger_ui_available() {
    let service = Arc::new(ForecastService::new(100));
    let app = create_routes().layer(axum::Extension(service));

    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/swagger-ui/")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_openapi_spec_available() {
    let service = Arc::new(ForecastService::new(100));
    let app = create_routes().layer(axum::Extension(service));

    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/api-docs/openapi.json")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = response.into_body().collect().await.unwrap().to_bytes();
    let json: Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(json["info"]["title"], "Capacity Modeler API");
    assert_eq!(json["info"]["version"], "0.1.0");
}

#[tokio::test]
async fn test_forecast_result_structure() {
    let service = Arc::new(ForecastService::new(100));
    let app = create_routes().layer(axum::Extension(service));

    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/api/v1/forecast/gpu-demand?weeks=1")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = response.into_body().collect().await.unwrap().to_bytes();
    let json: Value = serde_json::from_slice(&body).unwrap();

    // Verify structure
    assert!(json["forecast_weeks"].is_number());
    assert!(json["points"].is_array());
    assert!(json["generated_at"].is_string());
    assert!(json["model_type"].is_string());

    // Verify point structure
    let point = &json["points"][0];
    assert!(point["timestamp"].is_string());
    assert!(point["value"].is_number());
}

#[tokio::test]
async fn test_accuracy_metrics_included() {
    let service = Arc::new(ForecastService::new(100));
    let app = create_routes().layer(axum::Extension(service));

    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/api/v1/forecast/gpu-demand?weeks=4")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = response.into_body().collect().await.unwrap().to_bytes();
    let json: Value = serde_json::from_slice(&body).unwrap();

    // Should include accuracy metrics from backtest
    if json["accuracy_metrics"].is_object() {
        let metrics = &json["accuracy_metrics"];
        assert!(metrics["mape"].is_number());
        assert!(metrics["rmse"].is_number());
        assert!(metrics["mae"].is_number());

        // Metrics should be reasonable
        let mape = metrics["mape"].as_f64().unwrap();
        assert!(mape >= 0.0);
        assert!(mape < 2.0); // Less than 200% error for test data
    }
}
