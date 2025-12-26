use chrono::{DateTime, Utc};

use crate::db::Repository;
use crate::error::Result;
use crate::models::report::{Period, ShowbackReport};
use crate::models::summary::CostBreakdown;
use crate::models::trend::{TrendAnalysis, TrendDirection};
use crate::reports::TrendAnalyzer;

pub struct ShowbackGenerator {
    repository: Repository,
    trend_analyzer: TrendAnalyzer,
}

impl ShowbackGenerator {
    pub fn new(repository: Repository) -> Self {
        Self {
            repository,
            trend_analyzer: TrendAnalyzer::new(),
        }
    }

    /// Generate team showback report
    pub async fn generate_team_report(
        &self,
        team_id: &str,
        period: Period,
        period_start: Option<DateTime<Utc>>,
        period_end: Option<DateTime<Utc>>,
    ) -> Result<ShowbackReport> {
        let (start, end) = if period == Period::Custom {
            (period_start.unwrap(), period_end.unwrap())
        } else {
            period.to_date_range(Utc::now())
        };

        // Get daily summaries for this team
        let summaries = self
            .repository
            .get_daily_summaries(start, end, Some(team_id), None)
            .await?;

        // Calculate breakdown
        let breakdown = CostBreakdown::from_summaries(&summaries);

        // Calculate trends if we have enough data
        let trends = if summaries.len() >= 2 {
            Some(self.trend_analyzer.calculate_trend(&summaries)?)
        } else {
            None
        };

        // Get top users in this team
        let top_users = self
            .repository
            .get_top_spenders(start, end, "user", 10)
            .await?;

        // Get job count
        let job_count: i64 = summaries.iter().map(|s| s.job_count).sum();

        Ok(ShowbackReport::new(
            team_id.to_string(),
            "team".to_string(),
            period,
            start,
            end,
        )
        .with_breakdown(breakdown)
        .with_trends(trends.unwrap_or_else(|| {
            TrendAnalysis::new(
                TrendDirection::Stable,
                rust_decimal::Decimal::ZERO,
                rust_decimal::Decimal::ZERO,
                0.0,
            )
        }))
        .with_top_users(top_users)
        .with_job_count(job_count))
    }

    /// Generate user showback report
    pub async fn generate_user_report(
        &self,
        user_id: &str,
        period: Period,
        period_start: Option<DateTime<Utc>>,
        period_end: Option<DateTime<Utc>>,
    ) -> Result<ShowbackReport> {
        let (start, end) = if period == Period::Custom {
            (period_start.unwrap(), period_end.unwrap())
        } else {
            period.to_date_range(Utc::now())
        };

        // Get daily summaries for this user
        let summaries = self
            .repository
            .get_daily_summaries(start, end, None, Some(user_id))
            .await?;

        // Calculate breakdown
        let breakdown = CostBreakdown::from_summaries(&summaries);

        // Calculate trends if we have enough data
        let trends = if summaries.len() >= 2 {
            Some(self.trend_analyzer.calculate_trend(&summaries)?)
        } else {
            None
        };

        // Get job count
        let job_count: i64 = summaries.iter().map(|s| s.job_count).sum();

        Ok(ShowbackReport::new(
            user_id.to_string(),
            "user".to_string(),
            period,
            start,
            end,
        )
        .with_breakdown(breakdown)
        .with_trends(trends.unwrap_or_else(|| {
            TrendAnalysis::new(
                TrendDirection::Stable,
                rust_decimal::Decimal::ZERO,
                rust_decimal::Decimal::ZERO,
                0.0,
            )
        }))
        .with_job_count(job_count))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore] // Requires database
    async fn test_showback_generator_creation() {
        let database_url = std::env::var("DATABASE_URL")
            .unwrap_or_else(|_| "postgres://localhost/horizon_cost_test".to_string());

        let pool = crate::db::create_pool(&database_url).await.unwrap();
        let repo = Repository::new(pool);
        let generator = ShowbackGenerator::new(repo);

        // Just test construction
        let _ = generator;
    }
}
