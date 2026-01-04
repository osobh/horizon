use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::FromRow;
use utoipa::ToSchema;
use uuid::Uuid;

// ==================== Cost Summary & Breakdown ====================

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct CostSummary {
    pub period: String,
    pub total_cost: f64,
    pub compute_cost: f64,
    pub storage_cost: f64,
    pub network_cost: f64,
    pub by_team: Vec<TeamCost>,
    pub trend: CostTrend,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct TeamCost {
    pub team_id: String,
    pub team_name: String,
    pub cost: f64,
    pub percentage: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct CostTrend {
    pub current: f64,
    pub previous: f64,
    pub change_percent: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct CostBreakdown {
    pub start_date: String,
    pub end_date: String,
    pub total_cost: f64,
    pub by_team: Vec<TeamCostDetail>,
    pub by_resource_type: Vec<ResourceTypeCost>,
    pub by_provider: Vec<ProviderCost>,
    pub by_region: Vec<RegionCost>,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct TeamCostDetail {
    pub team_id: String,
    pub team_name: String,
    pub total_cost: f64,
    pub compute_cost: f64,
    pub storage_cost: f64,
    pub network_cost: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ResourceTypeCost {
    pub resource_type: String,
    pub cost: f64,
    pub unit_count: f64,
    pub percentage: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ProviderCost {
    pub provider: String,
    pub cost: f64,
    pub percentage: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct RegionCost {
    pub region: String,
    pub cost: f64,
    pub percentage: f64,
}

// ==================== Budgets ====================

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct Budget {
    pub id: Uuid,
    pub team_id: String,
    pub team_name: String,
    pub amount: f64,
    pub spent: f64,
    pub remaining: f64,
    pub period: String,
    pub alert_threshold: f64,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

// Database model for budgets table (remaining is generated column)
#[derive(Debug, Clone, FromRow)]
pub struct BudgetDb {
    pub id: Uuid,
    pub team_id: String,
    pub team_name: String,
    pub amount: sqlx::types::Decimal,
    pub spent: sqlx::types::Decimal,
    pub remaining: sqlx::types::Decimal, // Generated column
    pub period: String,
    pub alert_threshold: sqlx::types::Decimal,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl From<BudgetDb> for Budget {
    fn from(db: BudgetDb) -> Self {
        use std::str::FromStr;
        let to_f64 =
            |bd: sqlx::types::Decimal| -> f64 { f64::from_str(&bd.to_string()).unwrap_or(0.0) };

        Self {
            id: db.id,
            team_id: db.team_id,
            team_name: db.team_name,
            amount: to_f64(db.amount),
            spent: to_f64(db.spent),
            remaining: to_f64(db.remaining),
            period: db.period,
            alert_threshold: to_f64(db.alert_threshold),
            created_at: db.created_at,
            updated_at: db.updated_at,
        }
    }
}

#[derive(Debug, Deserialize, ToSchema)]
pub struct UpdateBudgetRequest {
    pub amount: f64,
}

// ==================== Chargeback ====================

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ChargebackReport {
    pub id: Uuid,
    pub team_id: String,
    pub team_name: String,
    pub start_date: String,
    pub end_date: String,
    pub total_amount: f64,
    pub line_items: Vec<ChargebackLineItem>,
    pub status: String,
    pub generated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ChargebackLineItem {
    pub description: String,
    pub resource_type: String,
    pub quantity: f64,
    pub unit_price: f64,
    pub total: f64,
}

// Database model for chargeback_reports table
#[derive(Debug, Clone, FromRow)]
pub struct ChargebackReportDb {
    pub id: Uuid,
    pub team_id: String,
    pub team_name: String,
    pub start_date: String,
    pub end_date: String,
    pub total_amount: sqlx::types::Decimal,
    pub line_items: sqlx::types::JsonValue, // JSONB array
    pub status: String,
    pub generated_at: DateTime<Utc>,
}

impl From<ChargebackReportDb> for ChargebackReport {
    fn from(db: ChargebackReportDb) -> Self {
        use std::str::FromStr;

        // Extract line items from JSONB
        let line_items: Vec<ChargebackLineItem> = if let Some(arr) = db.line_items.as_array() {
            arr.iter()
                .filter_map(|v| serde_json::from_value(v.clone()).ok())
                .collect()
        } else {
            vec![]
        };

        Self {
            id: db.id,
            team_id: db.team_id,
            team_name: db.team_name,
            start_date: db.start_date,
            end_date: db.end_date,
            total_amount: f64::from_str(&db.total_amount.to_string()).unwrap_or(0.0),
            line_items,
            status: db.status,
            generated_at: db.generated_at,
        }
    }
}

#[derive(Debug, Deserialize, ToSchema)]
pub struct GenerateChargebackRequest {
    #[serde(rename = "teamId")]
    pub team_id: String,
    #[serde(rename = "startDate")]
    pub start_date: String,
    #[serde(rename = "endDate")]
    pub end_date: String,
}

#[derive(Debug, Deserialize, ToSchema)]
pub struct ChargebackFilters {
    #[serde(rename = "teamId", skip_serializing_if = "Option::is_none")]
    pub team_id: Option<String>,
    #[serde(rename = "startDate", skip_serializing_if = "Option::is_none")]
    pub start_date: Option<String>,
    #[serde(rename = "endDate", skip_serializing_if = "Option::is_none")]
    pub end_date: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<Vec<String>>,
    #[serde(rename = "minAmount", skip_serializing_if = "Option::is_none")]
    pub min_amount: Option<f64>,
    #[serde(rename = "maxAmount", skip_serializing_if = "Option::is_none")]
    pub max_amount: Option<f64>,
}

// ==================== Cost Optimizations ====================

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct CostOptimization {
    pub id: Uuid,
    pub title: String,
    pub description: String,
    pub category: String,
    pub potential_savings: f64,
    pub effort: String,
    pub priority: String,
    pub status: String,
    pub implementation_steps: Vec<String>,
    pub created_at: DateTime<Utc>,
}

// Database model for cost_optimizations table
#[derive(Debug, Clone, FromRow)]
pub struct CostOptimizationDb {
    pub id: Uuid,
    pub title: String,
    pub description: String,
    pub category: String,
    pub potential_savings: sqlx::types::Decimal,
    pub effort: String,
    pub priority: String,
    pub status: String,
    pub implementation_steps: sqlx::types::JsonValue, // JSONB array
    pub created_at: DateTime<Utc>,
}

impl From<CostOptimizationDb> for CostOptimization {
    fn from(db: CostOptimizationDb) -> Self {
        use std::str::FromStr;

        // Extract implementation steps from JSONB
        let implementation_steps = if let Some(arr) = db.implementation_steps.as_array() {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        } else {
            vec![]
        };

        Self {
            id: db.id,
            title: db.title,
            description: db.description,
            category: db.category,
            potential_savings: f64::from_str(&db.potential_savings.to_string()).unwrap_or(0.0),
            effort: db.effort,
            priority: db.priority,
            status: db.status,
            implementation_steps,
            created_at: db.created_at,
        }
    }
}

#[derive(Debug, Deserialize, ToSchema)]
pub struct ImplementOptimizationRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub notes: Option<String>,
}

#[derive(Debug, Deserialize, ToSchema)]
pub struct RejectOptimizationRequest {
    pub reason: String,
}

// ==================== Cost Alerts ====================

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct CostAlert {
    pub id: Uuid,
    pub title: String,
    pub description: String,
    pub severity: String,
    pub team_id: Option<String>,
    pub team_name: Option<String>,
    pub threshold_amount: f64,
    pub current_amount: f64,
    pub active: bool,
    pub acknowledged: bool,
    pub acknowledged_by: Option<String>,
    pub acknowledged_at: Option<DateTime<Utc>>,
    pub resolved: bool,
    pub resolved_at: Option<DateTime<Utc>>,
    pub resolution: Option<String>,
    pub created_at: DateTime<Utc>,
}

// Database model for cost_alerts table
#[derive(Debug, Clone, FromRow)]
pub struct CostAlertDb {
    pub id: Uuid,
    pub title: String,
    pub description: String,
    pub severity: String,
    pub team_id: Option<String>,
    pub team_name: Option<String>,
    pub threshold_amount: sqlx::types::Decimal,
    pub current_amount: sqlx::types::Decimal,
    pub active: bool,
    pub acknowledged: bool,
    pub acknowledged_by: Option<String>,
    pub acknowledged_at: Option<DateTime<Utc>>,
    pub resolved: bool,
    pub resolved_at: Option<DateTime<Utc>>,
    pub resolution: Option<String>,
    pub created_at: DateTime<Utc>,
}

impl From<CostAlertDb> for CostAlert {
    fn from(db: CostAlertDb) -> Self {
        use std::str::FromStr;
        let to_f64 =
            |bd: sqlx::types::Decimal| -> f64 { f64::from_str(&bd.to_string()).unwrap_or(0.0) };

        Self {
            id: db.id,
            title: db.title,
            description: db.description,
            severity: db.severity,
            team_id: db.team_id,
            team_name: db.team_name,
            threshold_amount: to_f64(db.threshold_amount),
            current_amount: to_f64(db.current_amount),
            active: db.active,
            acknowledged: db.acknowledged,
            acknowledged_by: db.acknowledged_by,
            acknowledged_at: db.acknowledged_at,
            resolved: db.resolved,
            resolved_at: db.resolved_at,
            resolution: db.resolution,
            created_at: db.created_at,
        }
    }
}

#[derive(Debug, Deserialize, ToSchema)]
pub struct CreateCostAlertRequest {
    pub title: String,
    pub description: String,
    pub severity: String,
    pub team_id: Option<String>,
    pub threshold_amount: f64,
}

#[derive(Debug, Deserialize, ToSchema)]
pub struct AcknowledgeAlertRequest {
    #[serde(rename = "userId")]
    pub user_id: String,
}

#[derive(Debug, Deserialize, ToSchema)]
pub struct ResolveAlertRequest {
    pub resolution: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct AlertConfiguration {
    pub id: Uuid,
    pub name: String,
    pub description: String,
    pub condition_type: String,
    pub threshold: f64,
    pub enabled: bool,
    pub notification_channels: Vec<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

// Database model for alert_configurations table
#[derive(Debug, Clone, FromRow)]
pub struct AlertConfigurationDb {
    pub id: Uuid,
    pub name: String,
    pub description: String,
    pub condition_type: String,
    pub threshold: sqlx::types::Decimal,
    pub enabled: bool,
    pub notification_channels: sqlx::types::JsonValue, // JSONB array
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl From<AlertConfigurationDb> for AlertConfiguration {
    fn from(db: AlertConfigurationDb) -> Self {
        use std::str::FromStr;

        // Extract notification channels from JSONB
        let notification_channels = if let Some(arr) = db.notification_channels.as_array() {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        } else {
            vec![]
        };

        Self {
            id: db.id,
            name: db.name,
            description: db.description,
            condition_type: db.condition_type,
            threshold: f64::from_str(&db.threshold.to_string()).unwrap_or(0.0),
            enabled: db.enabled,
            notification_channels,
            created_at: db.created_at,
            updated_at: db.updated_at,
        }
    }
}

#[derive(Debug, Deserialize, ToSchema)]
pub struct UpdateAlertConfigRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub threshold: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub notification_channels: Option<Vec<String>>,
}
