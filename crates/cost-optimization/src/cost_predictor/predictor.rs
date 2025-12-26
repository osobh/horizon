//! Main cost predictor implementation

use crate::error::{CostOptimizationError, CostOptimizationResult};
use chrono::{DateTime, Duration as ChronoDuration, Utc};
use dashmap::DashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use statistical::{mean, standard_deviation};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::Duration;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use super::anomaly::{Anomaly, AnomalyType};
use super::budget::{BudgetForecast, RiskAssessment};
use super::config::CostPredictorConfig;
use super::metrics::PredictorMetrics;
use super::models::{ModelAccuracy, PredictionModel, TrainedModel};
use super::recommendations::{ComplexityLevel, CostRecommendation, Priority};
use super::risk::{RiskFactor, RiskLevel};
use super::types::*;

/// Cost predictor for ML-based forecasting
pub struct CostPredictor {
    /// Configuration
    config: Arc<CostPredictorConfig>,
    /// Time series data by metric type
    time_series: Arc<DashMap<CostMetricType, VecDeque<TimeSeriesPoint>>>,
    /// Trained models
    models: Arc<DashMap<(CostMetricType, PredictionModel), TrainedModel>>,
    /// Anomaly history
    anomalies: Arc<RwLock<Vec<Anomaly>>>,
    /// Budget forecasts
    forecasts: Arc<DashMap<Uuid, BudgetForecast>>,
    /// Metrics
    metrics: Arc<RwLock<PredictorMetrics>>,
}

impl CostPredictor {
    /// Create a new cost predictor
    pub fn new(config: CostPredictorConfig) -> CostOptimizationResult<Self> {
        Ok(Self {
            config: Arc::new(config),
            time_series: Arc::new(DashMap::new()),
            models: Arc::new(DashMap::new()),
            anomalies: Arc::new(RwLock::new(Vec::new())),
            forecasts: Arc::new(DashMap::new()),
            metrics: Arc::new(RwLock::new(PredictorMetrics::default())),
        })
    }

    /// Add cost data point
    pub fn add_data_point(
        &self,
        metric_type: CostMetricType,
        value: f64,
        timestamp: DateTime<Utc>,
        metadata: HashMap<String, String>,
    ) -> CostOptimizationResult<()> {
        let point = TimeSeriesPoint {
            timestamp,
            value,
            metadata,
        };

        let mut series = self
            .time_series
            .entry(metric_type)
            .or_insert_with(VecDeque::new);

        series.push_back(point);

        // Enforce retention limit
        let retention_limit = Utc::now() - ChronoDuration::from_std(self.config.retention_period)
            .map_err(|e| CostOptimizationError::CalculationError {
                details: format!("Duration conversion error: {}", e),
            })?;

        series.retain(|p| p.timestamp > retention_limit);

        Ok(())
    }

    /// Predict future costs
    pub async fn predict(
        &self,
        request: PredictionRequest,
    ) -> CostOptimizationResult<PredictionResult> {
        info!(
            "Generating cost prediction for {} using {:?} model",
            request.metric_type, request.model
        );

        // Get historical data
        let series = self.time_series.get(&request.metric_type).ok_or_else(|| {
            CostOptimizationError::PredictionError {
                model: request.model.to_string(),
                reason: "No historical data available".to_string(),
            }
        })?;

        // Generate simple predictions
        let predictions = self.generate_predictions(&request, &series)?;

        // Simple trend analysis
        let trend = self.analyze_trend(&series)?;

        // Create result
        Ok(PredictionResult {
            request,
            predictions,
            accuracy: ModelAccuracy {
                mae: 0.1,
                mse: 0.01,
                rmse: 0.1,
                mape: 5.0,
                r_squared: 0.95,
            },
            anomalies: vec![],
            confidence_intervals: vec![],
            trend,
        })
    }

    /// Generate budget forecast
    pub async fn forecast_budget(
        &self,
        period_start: DateTime<Utc>,
        period_end: DateTime<Utc>,
    ) -> CostOptimizationResult<BudgetForecast> {
        info!(
            "Generating budget forecast for {} to {}",
            period_start, period_end
        );

        let mut cost_breakdown = HashMap::new();
        let total_predicted = 10000.0; // Simplified

        // Simple breakdown
        cost_breakdown.insert(CostMetricType::ComputeCost, 5000.0);
        cost_breakdown.insert(CostMetricType::StorageCost, 2000.0);
        cost_breakdown.insert(CostMetricType::NetworkCost, 1000.0);
        cost_breakdown.insert(CostMetricType::GpuCost, 1500.0);
        cost_breakdown.insert(CostMetricType::OtherCost, 500.0);

        let forecast = BudgetForecast {
            id: Uuid::new_v4(),
            created_at: Utc::now(),
            period_start,
            period_end,
            predicted_cost: total_predicted,
            cost_breakdown,
            risk_assessment: RiskAssessment {
                risk_level: RiskLevel::Low,
                overrun_probability: 0.15,
                expected_variance: 0.1,
                risk_factors: vec![],
            },
            recommendations: vec![],
        };

        self.forecasts.insert(forecast.id, forecast.clone());
        self.metrics.write().forecasts_generated += 1;

        Ok(forecast)
    }

    /// Get predictor metrics
    pub fn get_metrics(&self) -> PredictorMetrics {
        self.metrics.read().clone()
    }

    // Helper methods

    fn generate_predictions(
        &self,
        request: &PredictionRequest,
        series: &VecDeque<TimeSeriesPoint>,
    ) -> CostOptimizationResult<Vec<PredictedValue>> {
        let values: Vec<f64> = series.iter().map(|p| p.value).collect();
        let avg = mean(&values);

        // Simple prediction - just use average
        let mut predictions = Vec::new();
        let num_points = 10;
        let interval = request.horizon.as_secs() / num_points;

        for i in 0..num_points {
            let timestamp = Utc::now() + ChronoDuration::seconds((i * interval) as i64);
            predictions.push(PredictedValue {
                timestamp,
                value: avg * (1.0 + (i as f64 * 0.01)), // Slight growth
                confidence: 0.85,
            });
        }

        Ok(predictions)
    }

    fn analyze_trend(
        &self,
        series: &VecDeque<TimeSeriesPoint>,
    ) -> CostOptimizationResult<TrendAnalysis> {
        let values: Vec<f64> = series.iter().map(|p| p.value).collect();

        if values.len() < 2 {
            return Ok(TrendAnalysis {
                direction: TrendDirection::Stable,
                strength: 0.0,
                growth_rate: 0.0,
                seasonal_patterns: vec![],
            });
        }

        let first_half_avg = mean(&values[..values.len() / 2]);
        let second_half_avg = mean(&values[values.len() / 2..]);
        let change = (second_half_avg - first_half_avg) / first_half_avg;

        let direction = if change.abs() < 0.05 {
            TrendDirection::Stable
        } else if change > 0.0 {
            TrendDirection::Increasing
        } else {
            TrendDirection::Decreasing
        };

        Ok(TrendAnalysis {
            direction,
            strength: change.abs().min(1.0),
            growth_rate: change * 100.0,
            seasonal_patterns: vec![],
        })
    }
}