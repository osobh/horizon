use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ForecastRequest {
    #[serde(default = "default_weeks")]
    pub weeks: u8,

    #[serde(default)]
    pub include_confidence_intervals: bool,
}

fn default_weeks() -> u8 {
    13
}

impl Default for ForecastRequest {
    fn default() -> Self {
        Self {
            weeks: 13,
            include_confidence_intervals: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct BacktestRequest {
    pub train_days: usize,
    pub test_days: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forecast_request_default() {
        let req = ForecastRequest::default();
        assert_eq!(req.weeks, 13);
        assert!(req.include_confidence_intervals);
    }

    #[test]
    fn test_backtest_request() {
        let req = BacktestRequest {
            train_days: 150,
            test_days: 30,
        };
        assert_eq!(req.train_days, 150);
        assert_eq!(req.test_days, 30);
    }
}
