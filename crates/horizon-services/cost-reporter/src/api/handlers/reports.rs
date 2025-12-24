use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    Json,
};
use chrono::{DateTime, Utc};
use hpc_channels::CostMessage;
use serde::Deserialize;
use std::sync::Arc;
use std::time::Instant;

use crate::api::AppState;
// HpcError and ReporterErrorExt are used in error handling but only indirectly
#[allow(unused_imports)]
use crate::error::{HpcError, ReporterErrorExt};
use crate::models::report::{ChargebackReport, Period, ShowbackReport};
use crate::models::summary::TopSpender;
use crate::models::trend::{CostForecast, DailyTrend, MonthlyTrend};

#[derive(Debug, Deserialize)]
pub struct ShowbackQuery {
    pub period: Option<String>,
    pub start_date: Option<DateTime<Utc>>,
    pub end_date: Option<DateTime<Utc>>,
}

impl ShowbackQuery {
    fn parse_period(&self) -> Period {
        match self.period.as_deref() {
            Some("current_month") => Period::CurrentMonth,
            Some("last_3_months") => Period::Last3Months,
            Some("ytd") | Some("year_to_date") => Period::YearToDate,
            Some("custom") => Period::Custom,
            _ => Period::CurrentMonth,
        }
    }
}

pub async fn get_team_showback(
    State(state): State<Arc<AppState>>,
    Path(team_id): Path<String>,
    Query(query): Query<ShowbackQuery>,
) -> Result<Json<ShowbackReport>, (StatusCode, String)> {
    let period = query.parse_period();

    let report = state
        .showback_generator
        .generate_team_report(&team_id, period, query.start_date, query.end_date)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    // Publish report generated event
    state.publish_report_event(CostMessage::ReportGenerated {
        report_type: "showback".to_string(),
        entity_type: "team".to_string(),
        entity_id: team_id,
        period_start: report.period_start.to_rfc3339(),
        period_end: report.period_end.to_rfc3339(),
    });

    Ok(Json(report))
}

pub async fn get_user_showback(
    State(state): State<Arc<AppState>>,
    Path(user_id): Path<String>,
    Query(query): Query<ShowbackQuery>,
) -> Result<Json<ShowbackReport>, (StatusCode, String)> {
    let period = query.parse_period();

    let report = state
        .showback_generator
        .generate_user_report(&user_id, period, query.start_date, query.end_date)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    // Publish report generated event
    state.publish_report_event(CostMessage::ReportGenerated {
        report_type: "showback".to_string(),
        entity_type: "user".to_string(),
        entity_id: user_id,
        period_start: report.period_start.to_rfc3339(),
        period_end: report.period_end.to_rfc3339(),
    });

    Ok(Json(report))
}

#[derive(Debug, Deserialize)]
pub struct TopSpendersQuery {
    pub period: Option<String>,
    pub limit: Option<i64>,
}

pub async fn get_top_spenders(
    State(state): State<Arc<AppState>>,
    Query(query): Query<TopSpendersQuery>,
) -> Result<Json<Vec<TopSpender>>, (StatusCode, String)> {
    let period = match query.period.as_deref() {
        Some("current_month") => Period::CurrentMonth,
        Some("last_3_months") => Period::Last3Months,
        Some("ytd") => Period::YearToDate,
        _ => Period::CurrentMonth,
    };

    let (start, end) = period.to_date_range(Utc::now());
    let limit = query.limit.unwrap_or(10);

    let spenders = state
        .repository
        .get_top_spenders(start, end, "user", limit)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok(Json(spenders))
}

#[derive(Debug, Deserialize)]
pub struct CustomerChargebackQuery {
    pub start_date: DateTime<Utc>,
    pub end_date: DateTime<Utc>,
    pub detailed: Option<bool>,
}

pub async fn get_customer_chargeback(
    State(state): State<Arc<AppState>>,
    Path(customer_id): Path<String>,
    Query(query): Query<CustomerChargebackQuery>,
) -> Result<Json<ChargebackReport>, (StatusCode, String)> {
    let report = if query.detailed.unwrap_or(false) {
        state
            .chargeback_generator
            .generate_detailed_report(&customer_id, query.start_date, query.end_date)
            .await
    } else {
        state
            .chargeback_generator
            .generate_customer_report(&customer_id, query.start_date, query.end_date)
            .await
    }
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    // Publish report generated event
    let report_type = if query.detailed.unwrap_or(false) {
        "chargeback_detailed"
    } else {
        "chargeback"
    };
    state.publish_report_event(CostMessage::ReportGenerated {
        report_type: report_type.to_string(),
        entity_type: "customer".to_string(),
        entity_id: customer_id,
        period_start: report.period_start.to_rfc3339(),
        period_end: report.period_end.to_rfc3339(),
    });

    Ok(Json(report))
}

#[derive(Debug, Deserialize)]
pub struct TrendQuery {
    pub start_date: DateTime<Utc>,
    pub end_date: DateTime<Utc>,
    pub team_id: Option<String>,
    pub user_id: Option<String>,
}

pub async fn get_daily_trends(
    State(state): State<Arc<AppState>>,
    Query(query): Query<TrendQuery>,
) -> Result<Json<Vec<DailyTrend>>, (StatusCode, String)> {
    let summaries = state
        .repository
        .get_daily_summaries(
            query.start_date,
            query.end_date,
            query.team_id.as_deref(),
            query.user_id.as_deref(),
        )
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let trends: Vec<DailyTrend> = summaries
        .into_iter()
        .map(|s| DailyTrend {
            date: s.day,
            total_cost: s.total_cost,
            gpu_cost: s.gpu_cost,
            cpu_cost: s.cpu_cost,
            network_cost: s.network_cost,
            storage_cost: s.storage_cost,
            job_count: s.job_count,
        })
        .collect();

    Ok(Json(trends))
}

pub async fn get_monthly_trends(
    State(state): State<Arc<AppState>>,
    Query(query): Query<TrendQuery>,
) -> Result<Json<Vec<MonthlyTrend>>, (StatusCode, String)> {
    let summaries = state
        .repository
        .get_monthly_summaries(
            query.start_date,
            query.end_date,
            query.team_id.as_deref(),
            query.user_id.as_deref(),
            None,
        )
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let trends: Vec<MonthlyTrend> = summaries
        .into_iter()
        .map(|s| MonthlyTrend {
            month: s.month,
            total_cost: s.total_cost,
            gpu_cost: s.gpu_cost,
            cpu_cost: s.cpu_cost,
            network_cost: s.network_cost,
            storage_cost: s.storage_cost,
            job_count: s.job_count,
        })
        .collect();

    Ok(Json(trends))
}

#[derive(Debug, Deserialize)]
pub struct ForecastQuery {
    pub team_id: Option<String>,
    pub user_id: Option<String>,
    pub days_ahead: Option<usize>,
}

pub async fn get_forecast(
    State(state): State<Arc<AppState>>,
    Query(query): Query<ForecastQuery>,
) -> Result<Json<CostForecast>, (StatusCode, String)> {
    let days_ahead = query.days_ahead.unwrap_or(state.config.default_forecast_days);

    // Get last 30 days of historical data
    let end = Utc::now();
    let start = end - chrono::Duration::days(30);

    let summaries = state
        .repository
        .get_daily_summaries(
            start,
            end,
            query.team_id.as_deref(),
            query.user_id.as_deref(),
        )
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let forecast = state
        .forecaster
        .forecast(&summaries, days_ahead)
        .map_err(|e| {
            // Check if it's an insufficient data error by inspecting the error message
            if e.to_string().contains("Insufficient data") {
                (StatusCode::UNPROCESSABLE_ENTITY, e.to_string())
            } else {
                (StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
            }
        })?;

    Ok(Json(forecast))
}

pub async fn refresh_views(
    State(state): State<Arc<AppState>>,
) -> Result<StatusCode, (StatusCode, String)> {
    let start = Instant::now();

    state
        .view_manager
        .refresh_views()
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let duration_ms = start.elapsed().as_millis() as u64;

    // Publish views refreshed event
    state.publish_report_event(CostMessage::ViewsRefreshed {
        timestamp_ms: Utc::now().timestamp_millis() as u64,
        duration_ms,
    });

    Ok(StatusCode::NO_CONTENT)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_showback_query_parse_period() {
        let query = ShowbackQuery {
            period: Some("current_month".to_string()),
            start_date: None,
            end_date: None,
        };
        assert_eq!(query.parse_period(), Period::CurrentMonth);

        let query = ShowbackQuery {
            period: Some("ytd".to_string()),
            start_date: None,
            end_date: None,
        };
        assert_eq!(query.parse_period(), Period::YearToDate);
    }
}
