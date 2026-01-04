use crate::error::{CapacityErrorExt, HpcError, Result};
use augurs_core::{Fit, Predict};
use augurs_ets::{AutoETS, FittedAutoETS};

/// Exponential Smoothing (ETS) forecaster for time-series data
pub struct EtsForecaster {
    model: Option<FittedAutoETS>,
}

/// Forecast result with confidence intervals
#[derive(Debug, Clone, PartialEq)]
pub struct ForecastWithIntervals {
    pub point_forecast: Vec<f64>,
    pub lower_bound: Vec<f64>,
    pub upper_bound: Vec<f64>,
}

impl EtsForecaster {
    /// Create a new untrained ETS forecaster
    pub fn new() -> Self {
        Self { model: None }
    }

    /// Check if the model has been trained
    pub fn is_trained(&self) -> bool {
        self.model.is_some()
    }

    /// Train the ETS model on historical data
    ///
    /// # Arguments
    /// * `data` - Historical time series data (minimum 10 points required)
    ///
    /// # Returns
    /// * `Ok(())` if training succeeds
    /// * `Err(Error)` if data is insufficient or training fails
    pub fn train(&mut self, data: &[f64]) -> Result<()> {
        if data.is_empty() {
            return Err(HpcError::insufficient_data("Cannot train on empty data"));
        }

        if data.len() < 10 {
            return Err(HpcError::insufficient_data(format!(
                "Need at least 10 data points, got {}",
                data.len()
            )));
        }

        // Use AutoETS to automatically select best model
        // Period of 7 for weekly seasonality, "ZZZ" means automatically determine error, trend, seasonal
        let auto_ets = AutoETS::new(7, "ZZZ")
            .map_err(|e| HpcError::training_failed(format!("Failed to create AutoETS: {}", e)))?;

        let fitted_model = auto_ets
            .fit(data)
            .map_err(|e| HpcError::training_failed(format!("ETS fitting failed: {}", e)))?;

        self.model = Some(fitted_model);
        Ok(())
    }

    /// Generate point forecasts for the specified horizon
    ///
    /// # Arguments
    /// * `horizon` - Number of periods to forecast
    ///
    /// # Returns
    /// * `Ok(Vec<f64>)` - Forecasted values
    /// * `Err(Error)` - If model not trained or forecast fails
    pub fn forecast(&self, horizon: usize) -> Result<Vec<f64>> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(HpcError::model_not_trained)?;

        let forecast = model
            .predict(horizon, None)
            .map_err(|e| HpcError::forecast_failed(format!("Forecast generation failed: {}", e)))?;

        Ok(forecast.point.into_iter().collect())
    }

    /// Generate forecasts with prediction intervals
    ///
    /// # Arguments
    /// * `horizon` - Number of periods to forecast
    /// * `level` - Confidence level (e.g., 0.95 for 95% confidence interval)
    ///
    /// # Returns
    /// * `Ok(ForecastWithIntervals)` - Point forecasts with upper/lower bounds
    /// * `Err(Error)` - If model not trained or forecast fails
    pub fn forecast_with_intervals(
        &self,
        horizon: usize,
        level: f64,
    ) -> Result<ForecastWithIntervals> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(HpcError::model_not_trained)?;

        // Generate forecast with prediction intervals
        let forecast = model
            .predict(horizon, Some(level))
            .map_err(|e| HpcError::forecast_failed(format!("Forecast generation failed: {}", e)))?;

        let point_forecast: Vec<f64> = forecast.point.into_iter().collect();

        // Extract prediction intervals
        let lower_bound: Vec<f64> = forecast
            .intervals
            .as_ref()
            .ok_or_else(|| HpcError::forecast_failed("No intervals in forecast"))?
            .lower
            .to_vec();

        let upper_bound: Vec<f64> = forecast
            .intervals
            .as_ref()
            .ok_or_else(|| HpcError::forecast_failed("No intervals in forecast"))?
            .upper
            .to_vec();

        Ok(ForecastWithIntervals {
            point_forecast,
            lower_bound,
            upper_bound,
        })
    }
}

impl Default for EtsForecaster {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_forecaster_not_trained() {
        let forecaster = EtsForecaster::new();
        assert!(!forecaster.is_trained());
    }

    #[test]
    fn test_train_with_sufficient_data() {
        let data: Vec<f64> = (1..=50).map(|x| x as f64).collect();
        let mut forecaster = EtsForecaster::new();

        assert!(forecaster.train(&data).is_ok());
        assert!(forecaster.is_trained());
    }

    #[test]
    fn test_forecast_untrained_model() {
        let forecaster = EtsForecaster::new();
        assert!(forecaster.forecast(10).is_err());
    }
}
