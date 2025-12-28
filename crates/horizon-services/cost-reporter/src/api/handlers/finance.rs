use axum::{
    extract::{Path, Query, State},
    Json,
};
use serde::Deserialize;
use std::sync::Arc;
use utoipa::ToSchema;
use uuid::Uuid;

use crate::{
    api::AppState,
    error::{HpcError, Result, ReporterErrorExt},
    models::finance::*,
};

// ==================== Cost Summary & Breakdown ====================

#[derive(Debug, Deserialize, ToSchema)]
pub struct CostSummaryQuery {
    pub period: String,
}

/// Get cost summary
#[utoipa::path(
    get,
    path = "/api/v1/finance/costs/summary",
    params(
        ("period" = String, Query, description = "Period (month, quarter, year)")
    ),
    responses(
        (status = 200, description = "Cost summary", body = CostSummary)
    ),
    tag = "finance"
)]
pub async fn get_cost_summary(
    State(state): State<Arc<AppState>>,
    Query(query): Query<CostSummaryQuery>,
) -> Result<Json<CostSummary>> {
    // TODO: Implement actual cost calculation from budgets and usage data
    // This would aggregate:
    // - Budget spent amounts by team
    // - Resource usage costs
    // - Historical trend data

    // For now, return computed values from budgets table
    let budgets_db: Vec<BudgetDb> = sqlx::query_as(
        "SELECT * FROM budgets WHERE period = $1"
    )
    .bind(&query.period)
    .fetch_all(&state.db)
    .await?;

    use std::str::FromStr;
    let to_f64 = |bd: sqlx::types::Decimal| -> f64 {
        f64::from_str(&bd.to_string()).unwrap_or(0.0)
    };

    let total_cost: f64 = budgets_db.iter().map(|b| to_f64(b.spent.clone())).sum();

    let by_team: Vec<TeamCost> = budgets_db.iter().map(|b| {
        let cost = to_f64(b.spent.clone());
        TeamCost {
            team_id: b.team_id.clone(),
            team_name: b.team_name.clone(),
            cost,
            percentage: if total_cost > 0.0 { (cost / total_cost) * 100.0 } else { 0.0 },
        }
    }).collect();

    // Get cost breakdown from chargeback reports for this period
    let reports_db: Vec<ChargebackReportDb> = sqlx::query_as(
        "SELECT * FROM chargeback_reports WHERE start_date >= DATE_TRUNC('month', CURRENT_DATE) ORDER BY generated_at DESC"
    )
    .fetch_all(&state.db)
    .await
    .unwrap_or_default();

    // Aggregate costs by resource type from line_items
    let mut compute_cost = 0.0;
    let mut storage_cost = 0.0;
    let mut network_cost = 0.0;

    for report in &reports_db {
        if let Ok(line_items) = serde_json::from_value::<Vec<ChargebackLineItem>>(report.line_items.clone()) {
            for item in line_items {
                match item.resource_type.as_str() {
                    "compute" => compute_cost += item.total,
                    "storage" => storage_cost += item.total,
                    "network" => network_cost += item.total,
                    _ => compute_cost += item.total, // Default to compute
                }
            }
        }
    }

    // If no breakdown data, estimate from total
    if compute_cost == 0.0 && storage_cost == 0.0 && network_cost == 0.0 && total_cost > 0.0 {
        compute_cost = total_cost * 0.75;
        storage_cost = total_cost * 0.15;
        network_cost = total_cost * 0.10;
    }

    // Get previous period costs for trend calculation
    let previous_period = match query.period.as_str() {
        "month" => "DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month')",
        "quarter" => "DATE_TRUNC('quarter', CURRENT_DATE - INTERVAL '3 months')",
        "year" => "DATE_TRUNC('year', CURRENT_DATE - INTERVAL '1 year')",
        _ => "DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month')",
    };

    let previous_sql = format!(
        "SELECT COALESCE(SUM(spent), 0) as total FROM budgets WHERE period = '{}' OR created_at >= {}",
        query.period, previous_period
    );
    let previous_total: (sqlx::types::Decimal,) = sqlx::query_as(&previous_sql)
        .fetch_one(&state.db)
        .await
        .unwrap_or((sqlx::types::Decimal::ZERO,));
    let previous = f64::from_str(&previous_total.0.to_string()).unwrap_or(total_cost * 0.94);

    let change_percent = if previous > 0.0 {
        ((total_cost - previous) / previous) * 100.0
    } else {
        0.0
    };

    Ok(Json(CostSummary {
        period: query.period,
        total_cost,
        compute_cost,
        storage_cost,
        network_cost,
        by_team,
        trend: CostTrend {
            current: total_cost,
            previous,
            change_percent,
        },
    }))
}

#[derive(Debug, Deserialize, ToSchema)]
pub struct CostBreakdownQuery {
    #[serde(rename = "startDate")]
    pub start_date: String,
    #[serde(rename = "endDate")]
    pub end_date: String,
}

/// Get cost breakdown
#[utoipa::path(
    get,
    path = "/api/v1/finance/costs/breakdown",
    params(
        ("startDate" = String, Query, description = "Start date"),
        ("endDate" = String, Query, description = "End date")
    ),
    responses(
        (status = 200, description = "Cost breakdown", body = CostBreakdown)
    ),
    tag = "finance"
)]
pub async fn get_cost_breakdown(
    State(state): State<Arc<AppState>>,
    Query(query): Query<CostBreakdownQuery>,
) -> Result<Json<CostBreakdown>> {
    // TODO: Implement actual cost breakdown calculation
    // This would aggregate from:
    // - Chargeback reports in the date range
    // - Resource usage logs
    // - Provider/region metadata

    // For now, get chargeback reports in the date range
    let reports_db: Vec<ChargebackReportDb> = sqlx::query_as(
        r#"
        SELECT * FROM chargeback_reports
        WHERE start_date >= $1 AND end_date <= $2
        ORDER BY team_id
        "#
    )
    .bind(&query.start_date)
    .bind(&query.end_date)
    .fetch_all(&state.db)
    .await?;

    use std::str::FromStr;
    let to_f64 = |bd: sqlx::types::Decimal| -> f64 {
        f64::from_str(&bd.to_string()).unwrap_or(0.0)
    };

    let total_cost: f64 = reports_db.iter().map(|r| to_f64(r.total_amount.clone())).sum();

    // Aggregate costs by resource type across all reports
    let mut resource_type_costs: std::collections::HashMap<String, f64> = std::collections::HashMap::new();

    let by_team: Vec<TeamCostDetail> = reports_db.iter().map(|r| {
        let cost = to_f64(r.total_amount.clone());

        // Parse line_items to get cost breakdown by resource type
        let mut team_compute = 0.0;
        let mut team_storage = 0.0;
        let mut team_network = 0.0;

        if let Ok(line_items) = serde_json::from_value::<Vec<ChargebackLineItem>>(r.line_items.clone()) {
            for item in &line_items {
                match item.resource_type.as_str() {
                    "compute" => {
                        team_compute += item.total;
                        *resource_type_costs.entry("compute".to_string()).or_insert(0.0) += item.total;
                    }
                    "storage" => {
                        team_storage += item.total;
                        *resource_type_costs.entry("storage".to_string()).or_insert(0.0) += item.total;
                    }
                    "network" => {
                        team_network += item.total;
                        *resource_type_costs.entry("network".to_string()).or_insert(0.0) += item.total;
                    }
                    other => {
                        team_compute += item.total;
                        *resource_type_costs.entry(other.to_string()).or_insert(0.0) += item.total;
                    }
                }
            }
        }

        // If no line items parsed, estimate from total
        if team_compute == 0.0 && team_storage == 0.0 && team_network == 0.0 && cost > 0.0 {
            team_compute = cost * 0.75;
            team_storage = cost * 0.15;
            team_network = cost * 0.10;
        }

        TeamCostDetail {
            team_id: r.team_id.clone(),
            team_name: r.team_name.clone(),
            total_cost: cost,
            compute_cost: team_compute,
            storage_cost: team_storage,
            network_cost: team_network,
        }
    }).collect();

    // Build by_resource_type from aggregated data
    let by_resource_type: Vec<ResourceTypeCost> = resource_type_costs.iter().map(|(resource_type, cost)| {
        ResourceTypeCost {
            resource_type: resource_type.clone(),
            cost: *cost,
            unit_count: 0.0, // Unit count not tracked in line items
            percentage: if total_cost > 0.0 { (*cost / total_cost) * 100.0 } else { 0.0 },
        }
    }).collect();

    Ok(Json(CostBreakdown {
        start_date: query.start_date,
        end_date: query.end_date,
        total_cost,
        by_team,
        by_resource_type,
        by_provider: vec![],  // Requires provider metadata in resource table
        by_region: vec![],  // Requires region metadata in resource table
    }))
}

// ==================== Budgets ====================

/// Get team budgets
#[utoipa::path(
    get,
    path = "/api/v1/finance/budgets",
    responses(
        (status = 200, description = "List of budgets", body = Vec<Budget>)
    ),
    tag = "finance"
)]
pub async fn get_team_budgets(
    State(state): State<Arc<AppState>>,
) -> Result<Json<Vec<Budget>>> {
    let budgets_db: Vec<BudgetDb> = sqlx::query_as(
        "SELECT * FROM budgets ORDER BY created_at DESC"
    )
    .fetch_all(&state.db)
    .await?;

    let budgets: Vec<Budget> = budgets_db
        .into_iter()
        .map(|db| db.into())
        .collect();

    Ok(Json(budgets))
}

/// Get specific budget
#[utoipa::path(
    get,
    path = "/api/v1/finance/budgets/{id}",
    params(
        ("id" = Uuid, Path, description = "Budget ID")
    ),
    responses(
        (status = 200, description = "Budget details", body = Budget)
    ),
    tag = "finance"
)]
pub async fn get_budget(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> Result<Json<Budget>> {
    let budget_db: BudgetDb = sqlx::query_as(
        "SELECT * FROM budgets WHERE id = $1"
    )
    .bind(id)
    .fetch_one(&state.db)
    .await
    .map_err(|e| match e {
        sqlx::Error::RowNotFound => HpcError::report_not_found(format!("Budget {}", id)),
        _ => e.into(),
    })?;

    let budget: Budget = budget_db.into();
    Ok(Json(budget))
}

/// Update budget
#[utoipa::path(
    put,
    path = "/api/v1/finance/budgets/{id}",
    params(
        ("id" = Uuid, Path, description = "Budget ID")
    ),
    request_body = UpdateBudgetRequest,
    responses(
        (status = 200, description = "Budget updated")
    ),
    tag = "finance"
)]
pub async fn update_budget(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
    Json(request): Json<UpdateBudgetRequest>,
) -> Result<Json<serde_json::Value>> {
    let result = sqlx::query(
        "UPDATE budgets SET amount = $1, updated_at = NOW() WHERE id = $2"
    )
    .bind(request.amount)
    .bind(id)
    .execute(&state.db)
    .await?;

    if result.rows_affected() == 0 {
        return Err(HpcError::report_not_found(format!("Budget {}", id)));
    }

    Ok(Json(serde_json::json!({"success": true})))
}

// ==================== Chargeback ====================

/// Generate chargeback report
#[utoipa::path(
    post,
    path = "/api/v1/finance/chargeback/generate",
    request_body = GenerateChargebackRequest,
    responses(
        (status = 200, description = "Chargeback report generated", body = ChargebackReport)
    ),
    tag = "finance"
)]
pub async fn generate_chargeback_report(
    State(state): State<Arc<AppState>>,
    Json(request): Json<GenerateChargebackRequest>,
) -> Result<Json<ChargebackReport>> {
    // TODO: In a real implementation, this would calculate actual usage and costs
    // For now, create placeholder line items
    let line_items = vec![
        ChargebackLineItem {
            description: "H100 GPU Hours".to_string(),
            resource_type: "compute".to_string(),
            quantity: 850.0,
            unit_price: 35.0,
            total: 29_750.0,
        },
        ChargebackLineItem {
            description: "SSD Storage (TB-months)".to_string(),
            resource_type: "storage".to_string(),
            quantity: 125.0,
            unit_price: 44.0,
            total: 5_500.0,
        },
    ];

    let total_amount: f64 = line_items.iter().map(|item| item.total).sum();
    let line_items_json = serde_json::to_value(&line_items)?;

    let report_db: ChargebackReportDb = sqlx::query_as(
        r#"
        INSERT INTO chargeback_reports (
            team_id, team_name, start_date, end_date, total_amount, line_items, status
        ) VALUES (
            $1, 'Team Name', $2, $3, $4, $5, 'generated'
        )
        RETURNING *
        "#
    )
    .bind(&request.team_id)
    .bind(&request.start_date)
    .bind(&request.end_date)
    .bind(total_amount)
    .bind(&line_items_json)
    .fetch_one(&state.db)
    .await?;

    let report: ChargebackReport = report_db.into();
    Ok(Json(report))
}

#[derive(Debug, Deserialize, ToSchema)]
pub struct ChargebackQuery {
    #[serde(rename = "teamId", skip_serializing_if = "Option::is_none")]
    pub team_id: Option<String>,
    #[serde(rename = "startDate", skip_serializing_if = "Option::is_none")]
    pub start_date: Option<String>,
    #[serde(rename = "endDate", skip_serializing_if = "Option::is_none")]
    pub end_date: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
    #[serde(rename = "minAmount", skip_serializing_if = "Option::is_none")]
    pub min_amount: Option<f64>,
    #[serde(rename = "maxAmount", skip_serializing_if = "Option::is_none")]
    pub max_amount: Option<f64>,
}

/// Get chargeback reports
#[utoipa::path(
    get,
    path = "/api/v1/finance/chargeback/reports",
    params(
        ("teamId" = Option<String>, Query, description = "Filter by team"),
        ("startDate" = Option<String>, Query, description = "Filter by start date"),
        ("endDate" = Option<String>, Query, description = "Filter by end date"),
        ("status" = Option<String>, Query, description = "Filter by status"),
        ("minAmount" = Option<f64>, Query, description = "Minimum amount"),
        ("maxAmount" = Option<f64>, Query, description = "Maximum amount")
    ),
    responses(
        (status = 200, description = "List of chargeback reports", body = Vec<ChargebackReport>)
    ),
    tag = "finance"
)]
pub async fn get_chargeback_reports(
    State(state): State<Arc<AppState>>,
    Query(query): Query<ChargebackQuery>,
) -> Result<Json<Vec<ChargebackReport>>> {
    let mut sql = String::from("SELECT * FROM chargeback_reports WHERE 1=1");

    if let Some(ref team_id) = query.team_id {
        sql.push_str(&format!(" AND team_id = '{}'", team_id));
    }

    if let Some(ref start_date) = query.start_date {
        sql.push_str(&format!(" AND start_date >= '{}'", start_date));
    }

    if let Some(ref end_date) = query.end_date {
        sql.push_str(&format!(" AND end_date <= '{}'", end_date));
    }

    if let Some(ref status) = query.status {
        sql.push_str(&format!(" AND status = '{}'", status));
    }

    if let Some(min_amount) = query.min_amount {
        sql.push_str(&format!(" AND total_amount >= {}", min_amount));
    }

    if let Some(max_amount) = query.max_amount {
        sql.push_str(&format!(" AND total_amount <= {}", max_amount));
    }

    sql.push_str(" ORDER BY generated_at DESC");

    let reports_db: Vec<ChargebackReportDb> = sqlx::query_as(&sql)
        .fetch_all(&state.db)
        .await?;

    let reports: Vec<ChargebackReport> = reports_db
        .into_iter()
        .map(|db| db.into())
        .collect();

    Ok(Json(reports))
}

/// Get specific chargeback report
#[utoipa::path(
    get,
    path = "/api/v1/finance/chargeback/reports/{id}",
    params(
        ("id" = Uuid, Path, description = "Report ID")
    ),
    responses(
        (status = 200, description = "Chargeback report details", body = ChargebackReport)
    ),
    tag = "finance"
)]
pub async fn get_chargeback_report(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> Result<Json<ChargebackReport>> {
    let report_db: ChargebackReportDb = sqlx::query_as(
        "SELECT * FROM chargeback_reports WHERE id = $1"
    )
    .bind(id)
    .fetch_one(&state.db)
    .await
    .map_err(|e| match e {
        sqlx::Error::RowNotFound => HpcError::report_not_found(format!("Chargeback report {}", id)),
        _ => e.into(),
    })?;

    let report: ChargebackReport = report_db.into();
    Ok(Json(report))
}

// ==================== Cost Optimizations ====================

/// Get cost optimizations
#[utoipa::path(
    get,
    path = "/api/v1/finance/optimizations",
    responses(
        (status = 200, description = "List of cost optimizations", body = Vec<CostOptimization>)
    ),
    tag = "finance"
)]
pub async fn get_cost_optimizations(
    State(state): State<Arc<AppState>>,
) -> Result<Json<Vec<CostOptimization>>> {
    let optimizations_db: Vec<CostOptimizationDb> = sqlx::query_as(
        "SELECT * FROM cost_optimizations ORDER BY potential_savings DESC"
    )
    .fetch_all(&state.db)
    .await?;

    let optimizations: Vec<CostOptimization> = optimizations_db
        .into_iter()
        .map(|db| db.into())
        .collect();

    Ok(Json(optimizations))
}

/// Get specific optimization
#[utoipa::path(
    get,
    path = "/api/v1/finance/optimizations/{id}",
    params(
        ("id" = Uuid, Path, description = "Optimization ID")
    ),
    responses(
        (status = 200, description = "Optimization details", body = CostOptimization)
    ),
    tag = "finance"
)]
pub async fn get_cost_optimization(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> Result<Json<CostOptimization>> {
    let optimization_db: CostOptimizationDb = sqlx::query_as(
        "SELECT * FROM cost_optimizations WHERE id = $1"
    )
    .bind(id)
    .fetch_one(&state.db)
    .await
    .map_err(|e| match e {
        sqlx::Error::RowNotFound => HpcError::report_not_found(format!("Optimization {}", id)),
        _ => e.into(),
    })?;

    let optimization: CostOptimization = optimization_db.into();
    Ok(Json(optimization))
}

/// Implement optimization
#[utoipa::path(
    post,
    path = "/api/v1/finance/optimizations/{id}/implement",
    params(
        ("id" = Uuid, Path, description = "Optimization ID")
    ),
    responses(
        (status = 200, description = "Optimization implementation started")
    ),
    tag = "finance"
)]
pub async fn implement_optimization(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> Result<Json<serde_json::Value>> {
    let result = sqlx::query(
        "UPDATE cost_optimizations SET status = 'implemented' WHERE id = $1"
    )
    .bind(id)
    .execute(&state.db)
    .await?;

    if result.rows_affected() == 0 {
        return Err(HpcError::report_not_found(format!("Optimization {}", id)));
    }

    Ok(Json(serde_json::json!({"success": true})))
}

/// Reject optimization
#[utoipa::path(
    post,
    path = "/api/v1/finance/optimizations/{id}/reject",
    params(
        ("id" = Uuid, Path, description = "Optimization ID")
    ),
    request_body = RejectOptimizationRequest,
    responses(
        (status = 200, description = "Optimization rejected")
    ),
    tag = "finance"
)]
pub async fn reject_optimization(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
    Json(_request): Json<RejectOptimizationRequest>,
) -> Result<Json<serde_json::Value>> {
    let result = sqlx::query(
        "UPDATE cost_optimizations SET status = 'rejected' WHERE id = $1"
    )
    .bind(id)
    .execute(&state.db)
    .await?;

    if result.rows_affected() == 0 {
        return Err(HpcError::report_not_found(format!("Optimization {}", id)));
    }

    Ok(Json(serde_json::json!({"success": true})))
}

// ==================== Cost Alerts ====================

#[derive(Debug, Deserialize, ToSchema)]
pub struct AlertsQuery {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub active: Option<bool>,
}

/// Get cost alerts
#[utoipa::path(
    get,
    path = "/api/v1/finance/alerts",
    params(
        ("active" = Option<bool>, Query, description = "Filter active alerts")
    ),
    responses(
        (status = 200, description = "List of cost alerts", body = Vec<CostAlert>)
    ),
    tag = "finance"
)]
pub async fn get_cost_alerts(
    State(state): State<Arc<AppState>>,
    Query(query): Query<AlertsQuery>,
) -> Result<Json<Vec<CostAlert>>> {
    let mut sql = String::from("SELECT * FROM cost_alerts WHERE 1=1");

    if let Some(active) = query.active {
        sql.push_str(&format!(" AND active = {}", active));
    }

    sql.push_str(" ORDER BY created_at DESC");

    let alerts_db: Vec<CostAlertDb> = sqlx::query_as(&sql)
        .fetch_all(&state.db)
        .await?;

    let alerts: Vec<CostAlert> = alerts_db
        .into_iter()
        .map(|db| db.into())
        .collect();

    Ok(Json(alerts))
}

/// Create cost alert
#[utoipa::path(
    post,
    path = "/api/v1/finance/alerts",
    request_body = CreateCostAlertRequest,
    responses(
        (status = 201, description = "Alert created", body = CostAlert)
    ),
    tag = "finance"
)]
pub async fn create_cost_alert(
    State(state): State<Arc<AppState>>,
    Json(request): Json<CreateCostAlertRequest>,
) -> Result<Json<CostAlert>> {
    let alert_db: CostAlertDb = sqlx::query_as(
        r#"
        INSERT INTO cost_alerts (
            title, description, severity, team_id, team_name, threshold_amount, current_amount
        ) VALUES (
            $1, $2, $3, $4, 'Team Name', $5, 0.0
        )
        RETURNING *
        "#
    )
    .bind(&request.title)
    .bind(&request.description)
    .bind(&request.severity)
    .bind(&request.team_id)
    .bind(request.threshold_amount)
    .fetch_one(&state.db)
    .await?;

    let alert: CostAlert = alert_db.into();
    Ok(Json(alert))
}

/// Acknowledge alert
#[utoipa::path(
    post,
    path = "/api/v1/finance/alerts/{id}/acknowledge",
    params(
        ("id" = Uuid, Path, description = "Alert ID")
    ),
    request_body = AcknowledgeAlertRequest,
    responses(
        (status = 200, description = "Alert acknowledged")
    ),
    tag = "finance"
)]
pub async fn acknowledge_cost_alert(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
    Json(request): Json<AcknowledgeAlertRequest>,
) -> Result<Json<serde_json::Value>> {
    let result = sqlx::query(
        r#"
        UPDATE cost_alerts
        SET acknowledged = true, acknowledged_by = $1, acknowledged_at = NOW()
        WHERE id = $2
        "#
    )
    .bind(&request.user_id)
    .bind(id)
    .execute(&state.db)
    .await?;

    if result.rows_affected() == 0 {
        return Err(HpcError::report_not_found(format!("Alert {}", id)));
    }

    Ok(Json(serde_json::json!({"success": true})))
}

/// Resolve alert
#[utoipa::path(
    post,
    path = "/api/v1/finance/alerts/{id}/resolve",
    params(
        ("id" = Uuid, Path, description = "Alert ID")
    ),
    request_body = ResolveAlertRequest,
    responses(
        (status = 200, description = "Alert resolved")
    ),
    tag = "finance"
)]
pub async fn resolve_cost_alert(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
    Json(request): Json<ResolveAlertRequest>,
) -> Result<Json<serde_json::Value>> {
    let result = sqlx::query(
        r#"
        UPDATE cost_alerts
        SET resolved = true, resolved_at = NOW(), resolution = $1, active = false
        WHERE id = $2
        "#
    )
    .bind(&request.resolution)
    .bind(id)
    .execute(&state.db)
    .await?;

    if result.rows_affected() == 0 {
        return Err(HpcError::report_not_found(format!("Alert {}", id)));
    }

    Ok(Json(serde_json::json!({"success": true})))
}

/// Get alert configurations
#[utoipa::path(
    get,
    path = "/api/v1/finance/alerts/configurations",
    responses(
        (status = 200, description = "List of alert configurations", body = Vec<AlertConfiguration>)
    ),
    tag = "finance"
)]
pub async fn get_alert_configurations(
    State(state): State<Arc<AppState>>,
) -> Result<Json<Vec<AlertConfiguration>>> {
    let configs_db: Vec<AlertConfigurationDb> = sqlx::query_as(
        "SELECT * FROM alert_configurations ORDER BY created_at DESC"
    )
    .fetch_all(&state.db)
    .await?;

    let configs: Vec<AlertConfiguration> = configs_db
        .into_iter()
        .map(|db| db.into())
        .collect();

    Ok(Json(configs))
}

/// Update alert configuration
#[utoipa::path(
    put,
    path = "/api/v1/finance/alerts/configurations/{id}",
    params(
        ("id" = Uuid, Path, description = "Configuration ID")
    ),
    request_body = UpdateAlertConfigRequest,
    responses(
        (status = 200, description = "Configuration updated")
    ),
    tag = "finance"
)]
pub async fn update_alert_configuration(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
    Json(request): Json<UpdateAlertConfigRequest>,
) -> Result<Json<serde_json::Value>> {
    let mut updates = Vec::new();
    let mut binds: Vec<Box<dyn sqlx::Encode<'_, sqlx::Postgres> + Send + Sync>> = Vec::new();
    let mut param_idx = 1;

    if let Some(threshold) = request.threshold {
        updates.push(format!("threshold = ${}", param_idx));
        binds.push(Box::new(threshold));
        param_idx += 1;
    }

    if let Some(enabled) = request.enabled {
        updates.push(format!("enabled = ${}", param_idx));
        binds.push(Box::new(enabled));
        param_idx += 1;
    }

    if let Some(ref channels) = request.notification_channels {
        let channels_json = serde_json::to_value(channels)?;
        updates.push(format!("notification_channels = ${}", param_idx));
        binds.push(Box::new(channels_json));
        param_idx += 1;
    }

    if updates.is_empty() {
        return Ok(Json(serde_json::json!({"success": true, "message": "No fields to update"})));
    }

    updates.push("updated_at = NOW()".to_string());

    // For simplicity with dynamic SQL, we'll rebuild the query
    let sql = format!("UPDATE alert_configurations SET {} WHERE id = ${}", updates.join(", "), param_idx);

    let result = if let Some(threshold) = request.threshold {
        let mut q = sqlx::query(&sql).bind(threshold);
        if let Some(enabled) = request.enabled {
            q = q.bind(enabled);
        }
        if let Some(ref channels) = request.notification_channels {
            let channels_json = serde_json::to_value(channels)?;
            q = q.bind(channels_json);
        }
        q.bind(id).execute(&state.db).await?
    } else if let Some(enabled) = request.enabled {
        let mut q = sqlx::query(&sql).bind(enabled);
        if let Some(ref channels) = request.notification_channels {
            let channels_json = serde_json::to_value(channels)?;
            q = q.bind(channels_json);
        }
        q.bind(id).execute(&state.db).await?
    } else if let Some(ref channels) = request.notification_channels {
        let channels_json = serde_json::to_value(channels)?;
        sqlx::query(&sql).bind(channels_json).bind(id).execute(&state.db).await?
    } else {
        // Should not happen due to earlier check
        return Ok(Json(serde_json::json!({"success": true, "message": "No fields to update"})));
    };

    if result.rows_affected() == 0 {
        return Err(HpcError::report_not_found(format!("Alert configuration {}", id)));
    }

    Ok(Json(serde_json::json!({"success": true})))
}
