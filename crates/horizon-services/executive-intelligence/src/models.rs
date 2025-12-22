use chrono::{DateTime, NaiveDate, Utc};
use serde::{Deserialize, Serialize};
use sqlx::FromRow;
use utoipa::ToSchema;
use uuid::Uuid;

// ==================== Executive Metrics ====================

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ExecutiveMetrics {
    pub total_revenue: f64,
    pub total_cost: f64,
    pub gross_margin_percent: f64,
    pub gpu_utilization: f64,
    pub active_teams: usize,
    pub active_jobs: usize,
    pub completed_jobs_month: usize,
    pub success_rate: f64,
    pub total_gpu_hours: f64,
    pub cost_per_gpu_hour: f64,
}

// ==================== Strategic KPIs ====================

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct StrategicKPI {
    pub id: Uuid,
    pub name: String,
    pub category: String,
    pub current_value: f64,
    pub target_value: f64,
    pub unit: String,
    pub status: String,
    pub trend: String,
    pub last_updated: DateTime<Utc>,
}

// Database model for strategic_kpis table
#[derive(Debug, Clone, FromRow)]
pub struct StrategicKPIDb {
    pub id: Uuid,
    pub name: String,
    pub category: String,
    pub current_value: sqlx::types::Decimal,
    pub target_value: sqlx::types::Decimal,
    pub unit: String,
    pub status: String,
    pub trend: String,
    pub last_updated: DateTime<Utc>,
}

impl From<StrategicKPIDb> for StrategicKPI {
    fn from(db: StrategicKPIDb) -> Self {
        use std::str::FromStr;
        let to_f64 = |bd: sqlx::types::Decimal| -> f64 {
            f64::from_str(&bd.to_string()).unwrap_or(0.0)
        };

        Self {
            id: db.id,
            name: db.name,
            category: db.category,
            current_value: to_f64(db.current_value),
            target_value: to_f64(db.target_value),
            unit: db.unit,
            status: db.status,
            trend: db.trend,
            last_updated: db.last_updated,
        }
    }
}

// ==================== Financial ====================

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct FinancialSummary {
    pub period: String,
    pub total_revenue: f64,
    pub total_cost: f64,
    pub gross_margin: f64,
    pub operating_expenses: f64,
    pub net_income: f64,
    pub revenue_by_customer: Vec<CustomerRevenue>,
    pub cost_by_category: Vec<CostCategory>,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct CustomerRevenue {
    pub customer_id: String,
    pub customer_name: String,
    pub revenue: f64,
    pub margin_percent: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct CostCategory {
    pub category: String,
    pub amount: f64,
    pub percentage: f64,
}

// ==================== Initiatives ====================

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct Initiative {
    pub id: Uuid,
    pub title: String,
    pub description: String,
    pub status: String,
    pub priority: String,
    pub owner: String,
    pub progress: f64,
    pub start_date: DateTime<Utc>,
    pub target_date: DateTime<Utc>,
    pub budget: f64,
    pub spent: f64,
    pub expected_roi: f64,
    pub last_updated: DateTime<Utc>,
}

// Database model for initiatives table
#[derive(Debug, Clone, FromRow)]
pub struct InitiativeDb {
    pub id: Uuid,
    pub title: String,
    pub description: String,
    pub status: String,
    pub priority: String,
    pub owner: String,
    pub progress: sqlx::types::Decimal,
    pub start_date: DateTime<Utc>,
    pub target_date: DateTime<Utc>,
    pub budget: sqlx::types::Decimal,
    pub spent: sqlx::types::Decimal,
    pub expected_roi: sqlx::types::Decimal,
    pub last_updated: DateTime<Utc>,
}

impl From<InitiativeDb> for Initiative {
    fn from(db: InitiativeDb) -> Self {
        use std::str::FromStr;
        let to_f64 = |bd: sqlx::types::Decimal| -> f64 {
            f64::from_str(&bd.to_string()).unwrap_or(0.0)
        };

        Self {
            id: db.id,
            title: db.title,
            description: db.description,
            status: db.status,
            priority: db.priority,
            owner: db.owner,
            progress: to_f64(db.progress),
            start_date: db.start_date,
            target_date: db.target_date,
            budget: to_f64(db.budget),
            spent: to_f64(db.spent),
            expected_roi: to_f64(db.expected_roi),
            last_updated: db.last_updated,
        }
    }
}

// ==================== Capacity Insights ====================

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct CapacityInsight {
    pub current_utilization: f64,
    pub forecasted_demand: Vec<DemandForecast>,
    pub capacity_gaps: Vec<CapacityGap>,
    pub recommendations: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct DemandForecast {
    pub date: String,
    pub predicted_utilization: f64,
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct CapacityGap {
    pub resource_type: String,
    pub timeframe: String,
    pub gap_percent: f64,
    pub severity: String,
    pub recommendation: String,
}

// ==================== Alerts ====================

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema, FromRow)]
pub struct StrategicAlert {
    pub id: Uuid,
    pub title: String,
    pub description: String,
    pub severity: String,
    pub category: String,
    pub impact: String,
    pub created_at: DateTime<Utc>,
    pub resolved: bool,
    pub resolved_at: Option<DateTime<Utc>>,
}

// ==================== Team Performance ====================

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct TeamPerformance {
    pub team_id: String,
    pub team_name: String,
    pub efficiency_score: f64,
    pub utilization_percent: f64,
    pub job_success_rate: f64,
    pub cost_per_job: f64,
    pub gpu_hours_used: f64,
    pub active_members: usize,
}

// ==================== Investment Recommendations ====================

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct InvestmentRecommendation {
    pub id: Uuid,
    pub title: String,
    pub description: String,
    pub category: String,
    pub priority: String,
    pub estimated_cost: f64,
    pub expected_roi: f64,
    pub payback_period_months: usize,
    pub rationale: String,
    pub created_at: DateTime<Utc>,
}

// Database model for investment_recommendations table
#[derive(Debug, Clone, FromRow)]
pub struct InvestmentRecommendationDb {
    pub id: Uuid,
    pub title: String,
    pub description: String,
    pub category: String,
    pub priority: String,
    pub estimated_cost: sqlx::types::Decimal,
    pub expected_roi: sqlx::types::Decimal,
    pub payback_period_months: i32,
    pub rationale: String,
    pub created_at: DateTime<Utc>,
}

impl From<InvestmentRecommendationDb> for InvestmentRecommendation {
    fn from(db: InvestmentRecommendationDb) -> Self {
        use std::str::FromStr;
        let to_f64 = |bd: sqlx::types::Decimal| -> f64 {
            f64::from_str(&bd.to_string()).unwrap_or(0.0)
        };

        Self {
            id: db.id,
            title: db.title,
            description: db.description,
            category: db.category,
            priority: db.priority,
            estimated_cost: to_f64(db.estimated_cost),
            expected_roi: to_f64(db.expected_roi),
            payback_period_months: db.payback_period_months as usize,
            rationale: db.rationale,
            created_at: db.created_at,
        }
    }
}

// ==================== Reports ====================

#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct ExecutiveReport {
    pub id: Uuid,
    pub report_type: String,
    pub report_period: NaiveDate,
    pub generated_at: DateTime<Utc>,
    pub content: serde_json::Value,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ReportListItem {
    pub id: String,
    pub period: String,
    pub generated_at: DateTime<Utc>,
}

// ==================== Dashboard ====================

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ExecutiveDashboardData {
    pub metrics: ExecutiveMetrics,
    pub kpis: Vec<StrategicKPI>,
    pub financial_summary: FinancialSummary,
    pub top_alerts: Vec<StrategicAlert>,
    pub top_initiatives: Vec<Initiative>,
    pub capacity_insight: CapacityInsight,
}

// ==================== Legacy ====================

#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct Scenario {
    pub id: Uuid,
    pub name: String,
    pub description: Option<String>,
    pub base_assumptions: Option<serde_json::Value>,
    pub simulated_assumptions: Option<serde_json::Value>,
    pub impact_analysis: Option<serde_json::Value>,
    pub created_by: Option<String>,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportSummary {
    pub total_reports: i64,
    pub latest_period: Option<NaiveDate>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_report_creation() {
        let report = ExecutiveReport {
            id: Uuid::new_v4(),
            report_type: "daily_digest".to_string(),
            report_period: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            generated_at: Utc::now(),
            content: serde_json::json!({}),
            created_at: Utc::now(),
        };
        assert_eq!(report.report_type, "daily_digest");
    }
}
