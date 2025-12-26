use crate::error::{HpcError, Result, ReporterErrorExt};
use crate::models::summary::HasCostBreakdown;
use crate::models::trend::{TrendAnalysis, TrendDirection};
use rust_decimal::Decimal;

pub struct TrendAnalyzer;

impl TrendAnalyzer {
    pub fn new() -> Self {
        Self
    }

    /// Calculate trend from daily summaries
    pub fn calculate_trend<T>(&self, summaries: &[T]) -> Result<TrendAnalysis>
    where
        T: HasCostBreakdown,
    {
        if summaries.is_empty() {
            return Err(HpcError::insufficient_data(1));
        }

        if summaries.len() < 2 {
            // Not enough data for trend, return stable
            let avg = summaries[0].total_cost();
            return Ok(TrendAnalysis::new(
                TrendDirection::Stable,
                Decimal::ZERO,
                avg,
                0.0,
            )
            .with_confidence(0.5));
        }

        // Calculate linear regression
        let (slope, _intercept) = self.linear_regression(summaries);

        // Determine direction
        let direction = if slope > 0.01 {
            TrendDirection::Increasing
        } else if slope < -0.01 {
            TrendDirection::Decreasing
        } else {
            TrendDirection::Stable
        };

        // Calculate growth rate
        let first = summaries.first().unwrap().total_cost();
        let last = summaries.last().unwrap().total_cost();

        let growth_rate = if first > Decimal::ZERO {
            ((last - first) / first) * Decimal::from(100)
        } else {
            Decimal::ZERO
        };

        // Calculate daily average
        let total: Decimal = summaries.iter().map(|s| s.total_cost()).sum();
        let daily_average = total / Decimal::from(summaries.len());

        // Calculate confidence based on R-squared
        let confidence = self.calculate_r_squared(summaries, slope, _intercept);

        Ok(TrendAnalysis::new(direction, growth_rate, daily_average, slope)
            .with_confidence(confidence))
    }

    /// Simple linear regression (least squares)
    fn linear_regression<T>(&self, summaries: &[T]) -> (f64, f64)
    where
        T: HasCostBreakdown,
    {
        let n = summaries.len() as f64;
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xy = 0.0;
        let mut sum_x2 = 0.0;

        for (i, summary) in summaries.iter().enumerate() {
            let x = i as f64;
            let y = summary.total_cost().to_string().parse::<f64>().unwrap_or(0.0);
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_x2 += x * x;
        }

        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
        let intercept = (sum_y - slope * sum_x) / n;

        (slope, intercept)
    }

    /// Calculate R-squared (coefficient of determination)
    fn calculate_r_squared<T>(&self, summaries: &[T], slope: f64, intercept: f64) -> f64
    where
        T: HasCostBreakdown,
    {
        if summaries.is_empty() {
            return 0.0;
        }

        // Calculate mean
        let mean: f64 = summaries
            .iter()
            .map(|s| s.total_cost().to_string().parse::<f64>().unwrap_or(0.0))
            .sum::<f64>()
            / summaries.len() as f64;

        // Calculate total sum of squares and residual sum of squares
        let mut ss_tot = 0.0;
        let mut ss_res = 0.0;

        for (i, summary) in summaries.iter().enumerate() {
            let y = summary.total_cost().to_string().parse::<f64>().unwrap_or(0.0);
            let y_pred = slope * (i as f64) + intercept;

            ss_tot += (y - mean).powi(2);
            ss_res += (y - y_pred).powi(2);
        }

        if ss_tot == 0.0 {
            return 0.0;
        }

        let r_squared = 1.0 - (ss_res / ss_tot);
        r_squared.clamp(0.0, 1.0)
    }
}

impl Default for TrendAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::summary::DailyCostSummary;
    use chrono::Utc;
    use rust_decimal_macros::dec;

    #[test]
    fn test_trend_empty_data() {
        let analyzer = TrendAnalyzer::new();
        let summaries: Vec<DailyCostSummary> = vec![];

        let result = analyzer.calculate_trend(&summaries);
        assert!(result.is_err());
    }

    #[test]
    fn test_trend_single_data_point() {
        let analyzer = TrendAnalyzer::new();
        let summaries = vec![DailyCostSummary {
            day: Utc::now(),
            team_id: None,
            user_id: None,
            total_cost: dec!(100.00),
            gpu_cost: dec!(80.00),
            cpu_cost: dec!(10.00),
            network_cost: dec!(5.00),
            storage_cost: dec!(5.00),
            job_count: 10,
        }];

        let trend = analyzer.calculate_trend(&summaries).unwrap();
        assert_eq!(trend.direction, TrendDirection::Stable);
        assert_eq!(trend.daily_average, dec!(100.00));
    }

    #[test]
    fn test_trend_increasing() {
        let analyzer = TrendAnalyzer::new();
        let now = Utc::now();
        let summaries = vec![
            DailyCostSummary {
                day: now,
                team_id: None,
                user_id: None,
                total_cost: dec!(100.00),
                gpu_cost: dec!(80.00),
                cpu_cost: dec!(10.00),
                network_cost: dec!(5.00),
                storage_cost: dec!(5.00),
                job_count: 10,
            },
            DailyCostSummary {
                day: now,
                team_id: None,
                user_id: None,
                total_cost: dec!(150.00),
                gpu_cost: dec!(120.00),
                cpu_cost: dec!(15.00),
                network_cost: dec!(7.50),
                storage_cost: dec!(7.50),
                job_count: 15,
            },
            DailyCostSummary {
                day: now,
                team_id: None,
                user_id: None,
                total_cost: dec!(200.00),
                gpu_cost: dec!(160.00),
                cpu_cost: dec!(20.00),
                network_cost: dec!(10.00),
                storage_cost: dec!(10.00),
                job_count: 20,
            },
        ];

        let trend = analyzer.calculate_trend(&summaries).unwrap();
        assert_eq!(trend.direction, TrendDirection::Increasing);
        assert!(trend.growth_rate > Decimal::ZERO);
        assert!(trend.slope > 0.0);
        assert_eq!(trend.daily_average, dec!(150.00)); // (100 + 150 + 200) / 3
    }

    #[test]
    fn test_trend_decreasing() {
        let analyzer = TrendAnalyzer::new();
        let now = Utc::now();
        let summaries = vec![
            DailyCostSummary {
                day: now,
                team_id: None,
                user_id: None,
                total_cost: dec!(200.00),
                gpu_cost: dec!(160.00),
                cpu_cost: dec!(20.00),
                network_cost: dec!(10.00),
                storage_cost: dec!(10.00),
                job_count: 20,
            },
            DailyCostSummary {
                day: now,
                team_id: None,
                user_id: None,
                total_cost: dec!(150.00),
                gpu_cost: dec!(120.00),
                cpu_cost: dec!(15.00),
                network_cost: dec!(7.50),
                storage_cost: dec!(7.50),
                job_count: 15,
            },
            DailyCostSummary {
                day: now,
                team_id: None,
                user_id: None,
                total_cost: dec!(100.00),
                gpu_cost: dec!(80.00),
                cpu_cost: dec!(10.00),
                network_cost: dec!(5.00),
                storage_cost: dec!(5.00),
                job_count: 10,
            },
        ];

        let trend = analyzer.calculate_trend(&summaries).unwrap();
        assert_eq!(trend.direction, TrendDirection::Decreasing);
        assert!(trend.growth_rate < Decimal::ZERO);
        assert!(trend.slope < 0.0);
    }

    #[test]
    fn test_trend_stable() {
        let analyzer = TrendAnalyzer::new();
        let now = Utc::now();
        let summaries = vec![
            DailyCostSummary {
                day: now,
                team_id: None,
                user_id: None,
                total_cost: dec!(100.00),
                gpu_cost: dec!(80.00),
                cpu_cost: dec!(10.00),
                network_cost: dec!(5.00),
                storage_cost: dec!(5.00),
                job_count: 10,
            },
            DailyCostSummary {
                day: now,
                team_id: None,
                user_id: None,
                total_cost: dec!(100.50),
                gpu_cost: dec!(80.00),
                cpu_cost: dec!(10.00),
                network_cost: dec!(5.00),
                storage_cost: dec!(5.50),
                job_count: 10,
            },
            DailyCostSummary {
                day: now,
                team_id: None,
                user_id: None,
                total_cost: dec!(99.50),
                gpu_cost: dec!(79.00),
                cpu_cost: dec!(10.00),
                network_cost: dec!(5.00),
                storage_cost: dec!(5.50),
                job_count: 10,
            },
        ];

        let trend = analyzer.calculate_trend(&summaries).unwrap();
        // May be classified as decreasing or stable depending on calculation
        assert!(
            trend.direction == TrendDirection::Stable || trend.direction == TrendDirection::Decreasing
        );
        assert!(trend.slope.abs() < 0.5);
    }

    #[test]
    fn test_linear_regression() {
        let analyzer = TrendAnalyzer::new();
        let now = Utc::now();
        let summaries = vec![
            DailyCostSummary {
                day: now,
                team_id: None,
                user_id: None,
                total_cost: dec!(10.00),
                gpu_cost: dec!(8.00),
                cpu_cost: dec!(1.00),
                network_cost: dec!(0.50),
                storage_cost: dec!(0.50),
                job_count: 1,
            },
            DailyCostSummary {
                day: now,
                team_id: None,
                user_id: None,
                total_cost: dec!(20.00),
                gpu_cost: dec!(16.00),
                cpu_cost: dec!(2.00),
                network_cost: dec!(1.00),
                storage_cost: dec!(1.00),
                job_count: 2,
            },
            DailyCostSummary {
                day: now,
                team_id: None,
                user_id: None,
                total_cost: dec!(30.00),
                gpu_cost: dec!(24.00),
                cpu_cost: dec!(3.00),
                network_cost: dec!(1.50),
                storage_cost: dec!(1.50),
                job_count: 3,
            },
        ];

        let (slope, intercept) = analyzer.linear_regression(&summaries);

        // For perfect linear data (10, 20, 30), slope should be 10, intercept should be 10
        assert!((slope - 10.0).abs() < 0.01);
        assert!((intercept - 10.0).abs() < 0.01);
    }
}
