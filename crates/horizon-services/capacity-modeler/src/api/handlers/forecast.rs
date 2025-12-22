use axum::{extract::Query, http::StatusCode, Extension, Json};
use std::sync::Arc;

use crate::error::Result;
use crate::models::{BacktestRequest, BacktestResult, ForecastRequest, ForecastResult};
use crate::service::ForecastService;

/// Generate GPU demand forecast
#[utoipa::path(
    get,
    path = "/api/v1/forecast/gpu-demand",
    params(
        ("weeks" = Option<u8>, Query, description = "Number of weeks to forecast (default: 13)"),
        ("include_confidence_intervals" = Option<bool>, Query, description = "Include confidence intervals")
    ),
    responses(
        (status = 200, description = "Forecast generated successfully", body = ForecastResult),
        (status = 400, description = "Insufficient data or invalid parameters"),
        (status = 500, description = "Internal server error")
    )
)]
pub async fn forecast_gpu_demand(
    Extension(service): Extension<Arc<ForecastService>>,
    Query(request): Query<ForecastRequest>,
) -> Result<(StatusCode, Json<ForecastResult>)> {
    // For demo purposes, generate synthetic historical data
    // In production, this would fetch from InfluxDB
    let historical_data = generate_demo_data(200);

    let forecast = service.forecast_gpu_demand(
        &historical_data,
        request.weeks,
        request.include_confidence_intervals,
    )?;

    Ok((StatusCode::OK, Json(forecast)))
}

/// Backtest forecasting model
#[utoipa::path(
    post,
    path = "/api/v1/forecast/backtest",
    request_body = BacktestRequest,
    responses(
        (status = 200, description = "Backtest completed", body = BacktestResult),
        (status = 400, description = "Invalid parameters"),
        (status = 500, description = "Internal server error")
    )
)]
pub async fn backtest_model(
    Extension(service): Extension<Arc<ForecastService>>,
    Json(request): Json<BacktestRequest>,
) -> Result<(StatusCode, Json<BacktestResult>)> {
    // For demo purposes, generate synthetic historical data
    let historical_data = generate_demo_data(300);

    let backtest_result = service.backtest(&historical_data, request.train_days, request.test_days)?;

    Ok((StatusCode::OK, Json(backtest_result)))
}

/// Generate demo data for testing (will be replaced with InfluxDB queries)
fn generate_demo_data(count: usize) -> Vec<f64> {
    (0..count)
        .map(|i| {
            let base = 50.0;
            let trend = i as f64 * 0.05;
            let seasonal = ((i as f64 * 2.0 * std::f64::consts::PI) / 7.0).sin() * 8.0;
            let noise = (i as f64 * 37.0).sin() * 2.0; // Pseudo-random noise
            base + trend + seasonal + noise
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_demo_data() {
        let data = generate_demo_data(100);
        assert_eq!(data.len(), 100);

        // All values should be positive and reasonable
        for value in &data {
            assert!(*value > 0.0);
            assert!(*value < 200.0);
        }
    }

    #[tokio::test]
    async fn test_forecast_gpu_demand_handler() {
        let service = Arc::new(ForecastService::new(100));
        let request = ForecastRequest {
            weeks: 4,
            include_confidence_intervals: false,
        };

        let result = forecast_gpu_demand(Extension(service), Query(request)).await;

        assert!(result.is_ok());
        let (status, response) = result.unwrap();
        assert_eq!(status, StatusCode::OK);
        assert_eq!(response.0.forecast_weeks, 4);
        assert_eq!(response.0.points.len(), 28); // 4 weeks * 7 days
    }

    #[tokio::test]
    async fn test_backtest_handler() {
        let service = Arc::new(ForecastService::new(50));
        let request = BacktestRequest {
            train_days: 200,
            test_days: 30,
        };

        let result = backtest_model(Extension(service), Json(request)).await;

        assert!(result.is_ok());
        let (status, response) = result.unwrap();
        assert_eq!(status, StatusCode::OK);
        assert_eq!(response.0.train_size, 200);
        assert_eq!(response.0.test_size, 30);
    }
}
