//! Costs Bridge
//!
//! Integrates horizon-services cost intelligence with Horizon.
//! Provides access to cost attribution, forecasting, and budget alerts.
//!
//! Currently uses mock data until cost-attributor and cost-reporter are fully integrated.

use crate::error::HorizonError;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

/// Cost attribution entry.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CostAttribution {
    pub id: String,
    pub name: String,
    #[serde(rename = "type")]
    pub attribution_type: AttributionType,
    pub cost_usd: f64,
    pub cost_trend_pct: f64,
    pub gpu_hours: f64,
    pub storage_gb: f64,
    pub network_gb: f64,
    pub period: String,
}

/// Attribution type.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AttributionType {
    Team,
    Project,
    Resource,
    User,
}

/// Cost forecast entry.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CostForecast {
    pub week: u32,
    pub date: String,
    pub predicted_cost_usd: f64,
    pub confidence_low: f64,
    pub confidence_high: f64,
    pub trend: ForecastTrend,
}

/// Forecast trend.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ForecastTrend {
    Up,
    Down,
    Stable,
}

/// Budget alert.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BudgetAlert {
    pub id: String,
    pub name: String,
    pub threshold_usd: f64,
    pub current_usd: f64,
    pub percentage_used: f64,
    pub status: AlertStatus,
    pub alert_at_pct: f64,
}

/// Alert status.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AlertStatus {
    Ok,
    Warning,
    Critical,
}

/// Cost summary combining all cost data.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CostSummary {
    pub total_cost_usd: f64,
    pub total_cost_trend_pct: f64,
    pub gpu_cost_usd: f64,
    pub storage_cost_usd: f64,
    pub network_cost_usd: f64,
    pub compute_cost_usd: f64,
    pub period: String,
    pub attributions: Vec<CostAttribution>,
    pub forecasts: Vec<CostForecast>,
    pub alerts: Vec<BudgetAlert>,
}

/// Chargeback/showback report.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ChargebackReport {
    pub id: String,
    pub name: String,
    pub generated_at: String,
    pub period_start: String,
    pub period_end: String,
    pub total_cost_usd: f64,
    pub format: ReportFormat,
    pub download_url: String,
}

/// Report format.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ReportFormat {
    Csv,
    Json,
    Pdf,
}

/// Bridge to cost intelligence services.
pub struct CostsBridge {
    state: Arc<RwLock<MockCostsState>>,
}

struct MockCostsState {
    attributions: Vec<CostAttribution>,
    forecasts: Vec<CostForecast>,
    alerts: Vec<BudgetAlert>,
    reports: Vec<ChargebackReport>,
}

impl MockCostsState {
    fn new() -> Self {
        let attributions = vec![
            CostAttribution {
                id: "team-ml".to_string(),
                name: "ML Research Team".to_string(),
                attribution_type: AttributionType::Team,
                cost_usd: 15_420.50,
                cost_trend_pct: 8.5,
                gpu_hours: 2_456.0,
                storage_gb: 1_024.0,
                network_gb: 512.0,
                period: "2024-12".to_string(),
            },
            CostAttribution {
                id: "team-infra".to_string(),
                name: "Infrastructure Team".to_string(),
                attribution_type: AttributionType::Team,
                cost_usd: 8_230.25,
                cost_trend_pct: -2.3,
                gpu_hours: 1_124.0,
                storage_gb: 2_048.0,
                network_gb: 256.0,
                period: "2024-12".to_string(),
            },
            CostAttribution {
                id: "proj-llm".to_string(),
                name: "LLM Training Project".to_string(),
                attribution_type: AttributionType::Project,
                cost_usd: 12_500.00,
                cost_trend_pct: 15.2,
                gpu_hours: 1_890.0,
                storage_gb: 512.0,
                network_gb: 128.0,
                period: "2024-12".to_string(),
            },
            CostAttribution {
                id: "proj-vision".to_string(),
                name: "Computer Vision Project".to_string(),
                attribution_type: AttributionType::Project,
                cost_usd: 6_800.75,
                cost_trend_pct: 3.1,
                gpu_hours: 980.0,
                storage_gb: 768.0,
                network_gb: 192.0,
                period: "2024-12".to_string(),
            },
        ];

        let forecasts = (1..=13)
            .map(|week| {
                let base = 45_000.0 + (week as f64 * 500.0);
                CostForecast {
                    week,
                    date: format!("2025-W{:02}", week),
                    predicted_cost_usd: base,
                    confidence_low: base * 0.9,
                    confidence_high: base * 1.1,
                    trend: if week < 5 {
                        ForecastTrend::Up
                    } else if week < 10 {
                        ForecastTrend::Stable
                    } else {
                        ForecastTrend::Down
                    },
                }
            })
            .collect();

        let alerts = vec![
            BudgetAlert {
                id: "budget-ml".to_string(),
                name: "ML Team Monthly Budget".to_string(),
                threshold_usd: 20_000.0,
                current_usd: 15_420.50,
                percentage_used: 77.1,
                status: AlertStatus::Warning,
                alert_at_pct: 75.0,
            },
            BudgetAlert {
                id: "budget-infra".to_string(),
                name: "Infrastructure Monthly Budget".to_string(),
                threshold_usd: 15_000.0,
                current_usd: 8_230.25,
                percentage_used: 54.9,
                status: AlertStatus::Ok,
                alert_at_pct: 75.0,
            },
            BudgetAlert {
                id: "budget-gpu".to_string(),
                name: "GPU Compute Budget".to_string(),
                threshold_usd: 25_000.0,
                current_usd: 23_750.00,
                percentage_used: 95.0,
                status: AlertStatus::Critical,
                alert_at_pct: 90.0,
            },
        ];

        Self {
            attributions,
            forecasts,
            alerts,
            reports: Vec::new(),
        }
    }
}

impl CostsBridge {
    pub fn new() -> Self {
        Self {
            state: Arc::new(RwLock::new(MockCostsState::new())),
        }
    }

    pub async fn get_summary(&self) -> CostSummary {
        let state = self.state.read().await;
        let total: f64 = state.attributions.iter().map(|a| a.cost_usd).sum();

        CostSummary {
            total_cost_usd: total,
            total_cost_trend_pct: 5.2,
            gpu_cost_usd: total * 0.65,
            storage_cost_usd: total * 0.20,
            network_cost_usd: total * 0.05,
            compute_cost_usd: total * 0.10,
            period: "2024-12".to_string(),
            attributions: state.attributions.clone(),
            forecasts: state.forecasts.clone(),
            alerts: state.alerts.clone(),
        }
    }

    pub async fn get_attributions(&self, attribution_type: Option<String>) -> Vec<CostAttribution> {
        let state = self.state.read().await;
        match attribution_type {
            Some(t) => state
                .attributions
                .iter()
                .filter(|a| format!("{:?}", a.attribution_type).to_lowercase() == t.to_lowercase())
                .cloned()
                .collect(),
            None => state.attributions.clone(),
        }
    }

    pub async fn get_forecasts(&self, _weeks: u32) -> Vec<CostForecast> {
        let state = self.state.read().await;
        state.forecasts.clone()
    }

    pub async fn get_alerts(&self) -> Vec<BudgetAlert> {
        let state = self.state.read().await;
        state.alerts.clone()
    }

    pub async fn get_reports(&self) -> Vec<ChargebackReport> {
        let state = self.state.read().await;
        state.reports.clone()
    }

    pub async fn generate_chargeback_report(
        &self,
        period_start: String,
        period_end: String,
        format: String,
    ) -> Result<ChargebackReport, HorizonError> {
        // Use UUID instead of timestamp to prevent ID collisions
        let report = ChargebackReport {
            id: format!("report-{}", Uuid::new_v4()),
            name: format!("Chargeback Report {} - {}", period_start, period_end),
            generated_at: chrono::Utc::now().to_rfc3339(),
            period_start,
            period_end,
            total_cost_usd: 45_951.50,
            format: match format.as_str() {
                "json" => ReportFormat::Json,
                "pdf" => ReportFormat::Pdf,
                _ => ReportFormat::Csv,
            },
            download_url: "/api/reports/download/latest".to_string(),
        };

        let mut state = self.state.write().await;
        state.reports.push(report.clone());
        Ok(report)
    }

    pub async fn generate_showback_report(
        &self,
        team_id: String,
        period_start: String,
        period_end: String,
    ) -> Result<ChargebackReport, HorizonError> {
        // Validate team_id is not empty
        if team_id.trim().is_empty() {
            return Err(HorizonError::InvalidConfig(
                "Team ID cannot be empty".to_string(),
            ));
        }

        // Use UUID instead of timestamp to prevent ID collisions
        let report = ChargebackReport {
            id: format!("report-{}", Uuid::new_v4()),
            name: format!("Showback Report for {} ({} - {})", team_id, period_start, period_end),
            generated_at: chrono::Utc::now().to_rfc3339(),
            period_start,
            period_end,
            total_cost_usd: 15_420.50,
            format: ReportFormat::Csv,
            download_url: format!("/api/reports/download/{}", team_id),
        };

        let mut state = self.state.write().await;
        state.reports.push(report.clone());
        Ok(report)
    }

    pub async fn set_budget_threshold(
        &self,
        name: String,
        threshold_usd: f64,
        alert_at_pct: f64,
    ) -> Result<(), HorizonError> {
        // Validate inputs
        if name.trim().is_empty() {
            return Err(HorizonError::InvalidConfig(
                "Budget name cannot be empty".to_string(),
            ));
        }
        if threshold_usd <= 0.0 {
            return Err(HorizonError::InvalidConfig(
                "Threshold must be greater than 0".to_string(),
            ));
        }
        if !(0.0..=100.0).contains(&alert_at_pct) {
            return Err(HorizonError::InvalidConfig(
                "Alert percentage must be between 0 and 100".to_string(),
            ));
        }

        let mut state = self.state.write().await;
        let alert = state
            .alerts
            .iter_mut()
            .find(|a| a.name == name)
            .ok_or_else(|| HorizonError::NotFound(format!("Budget '{}' not found", name)))?;

        alert.threshold_usd = threshold_usd;
        alert.alert_at_pct = alert_at_pct;
        alert.percentage_used = (alert.current_usd / threshold_usd) * 100.0;
        alert.status = if alert.percentage_used >= 90.0 {
            AlertStatus::Critical
        } else if alert.percentage_used >= alert_at_pct {
            AlertStatus::Warning
        } else {
            AlertStatus::Ok
        };

        Ok(())
    }
}

impl Default for CostsBridge {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_get_summary_calculates_totals_correctly() {
        let bridge = CostsBridge::new();
        let summary = bridge.get_summary().await;

        // Verify total cost equals sum of attributions
        let expected_total: f64 = summary.attributions.iter().map(|a| a.cost_usd).sum();

        assert_eq!(summary.total_cost_usd, expected_total);
        assert!(!summary.attributions.is_empty());
        assert!(!summary.forecasts.is_empty());
        assert!(!summary.alerts.is_empty());
    }

    #[tokio::test]
    async fn test_get_attributions_all() {
        let bridge = CostsBridge::new();
        let attributions = bridge.get_attributions(None).await;

        assert!(attributions.len() >= 4, "Should have at least 4 attributions");
    }

    #[tokio::test]
    async fn test_get_attributions_filtered_by_team() {
        let bridge = CostsBridge::new();
        let team_attributions = bridge.get_attributions(Some("team".to_string())).await;

        assert!(team_attributions.len() >= 2, "Should have at least 2 team attributions");
        for attr in &team_attributions {
            assert!(
                format!("{:?}", attr.attribution_type)
                    .to_lowercase()
                    .contains("team")
            );
        }
    }

    #[tokio::test]
    async fn test_get_attributions_filtered_by_project() {
        let bridge = CostsBridge::new();
        let project_attributions = bridge.get_attributions(Some("project".to_string())).await;

        assert!(
            project_attributions.len() >= 2,
            "Should have at least 2 project attributions"
        );
        for attr in &project_attributions {
            assert!(
                format!("{:?}", attr.attribution_type)
                    .to_lowercase()
                    .contains("project")
            );
        }
    }

    #[tokio::test]
    async fn test_get_forecasts_returns_13_weeks() {
        let bridge = CostsBridge::new();
        let forecasts = bridge.get_forecasts(13).await;

        assert_eq!(forecasts.len(), 13, "Should have 13 weeks of forecasts");
        for forecast in &forecasts {
            assert!(forecast.confidence_low < forecast.predicted_cost_usd);
            assert!(forecast.predicted_cost_usd < forecast.confidence_high);
        }
    }

    #[tokio::test]
    async fn test_get_alerts() {
        let bridge = CostsBridge::new();
        let alerts = bridge.get_alerts().await;

        assert!(alerts.len() >= 3, "Should have at least 3 budget alerts");
        for alert in &alerts {
            assert!(alert.percentage_used >= 0.0 && alert.percentage_used <= 100.0);
        }
    }

    #[tokio::test]
    async fn test_set_budget_threshold_updates_alert_status() {
        let bridge = CostsBridge::new();

        // Initial state - ML Team budget should be in Warning status
        let alerts = bridge.get_alerts().await;
        let initial_alert = alerts
            .iter()
            .find(|a| a.name == "ML Team Monthly Budget")
            .expect("Test alert should exist");

        assert!(matches!(initial_alert.status, AlertStatus::Warning));

        // Update threshold to make it OK (increase threshold)
        let result = bridge
            .set_budget_threshold("ML Team Monthly Budget".to_string(), 25_000.0, 75.0)
            .await;
        assert!(result.is_ok());

        // Verify status changed to OK
        let updated_alerts = bridge.get_alerts().await;
        let updated_alert = updated_alerts
            .iter()
            .find(|a| a.name == "ML Team Monthly Budget")
            .expect("Alert should still exist");

        assert!(matches!(updated_alert.status, AlertStatus::Ok));
        assert_eq!(updated_alert.threshold_usd, 25_000.0);
    }

    #[tokio::test]
    async fn test_set_budget_threshold_validation_empty_name() {
        let bridge = CostsBridge::new();

        let result = bridge.set_budget_threshold("".to_string(), 10_000.0, 75.0).await;

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), HorizonError::InvalidConfig(_)));
    }

    #[tokio::test]
    async fn test_set_budget_threshold_validation_zero_threshold() {
        let bridge = CostsBridge::new();

        let result = bridge
            .set_budget_threshold("Test Budget".to_string(), 0.0, 75.0)
            .await;

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), HorizonError::InvalidConfig(_)));
    }

    #[tokio::test]
    async fn test_set_budget_threshold_validation_invalid_percentage() {
        let bridge = CostsBridge::new();

        // Test percentage > 100
        let result = bridge
            .set_budget_threshold("Test Budget".to_string(), 10_000.0, 150.0)
            .await;
        assert!(result.is_err());

        // Test percentage < 0
        let result = bridge
            .set_budget_threshold("Test Budget".to_string(), 10_000.0, -10.0)
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_set_budget_threshold_not_found() {
        let bridge = CostsBridge::new();

        let result = bridge
            .set_budget_threshold("Nonexistent Budget".to_string(), 10_000.0, 75.0)
            .await;

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), HorizonError::NotFound(_)));
    }

    #[tokio::test]
    async fn test_generate_chargeback_report() {
        let bridge = CostsBridge::new();

        let result = bridge
            .generate_chargeback_report(
                "2024-01-01".to_string(),
                "2024-01-31".to_string(),
                "csv".to_string(),
            )
            .await;

        assert!(result.is_ok());
        let report = result.unwrap();
        assert!(report.id.starts_with("report-"));
        assert!(matches!(report.format, ReportFormat::Csv));
        assert_eq!(report.period_start, "2024-01-01");
        assert_eq!(report.period_end, "2024-01-31");

        // Verify report was stored
        let reports = bridge.get_reports().await;
        assert_eq!(reports.len(), 1);
    }

    #[tokio::test]
    async fn test_generate_chargeback_report_different_formats() {
        let bridge = CostsBridge::new();

        // Test JSON format
        let json_report = bridge
            .generate_chargeback_report(
                "2024-01-01".to_string(),
                "2024-01-31".to_string(),
                "json".to_string(),
            )
            .await
            .unwrap();
        assert!(matches!(json_report.format, ReportFormat::Json));

        // Test PDF format
        let pdf_report = bridge
            .generate_chargeback_report(
                "2024-02-01".to_string(),
                "2024-02-28".to_string(),
                "pdf".to_string(),
            )
            .await
            .unwrap();
        assert!(matches!(pdf_report.format, ReportFormat::Pdf));

        // Test default (CSV) format
        let default_report = bridge
            .generate_chargeback_report(
                "2024-03-01".to_string(),
                "2024-03-31".to_string(),
                "unknown".to_string(),
            )
            .await
            .unwrap();
        assert!(matches!(default_report.format, ReportFormat::Csv));
    }

    #[tokio::test]
    async fn test_generate_showback_report() {
        let bridge = CostsBridge::new();

        let result = bridge
            .generate_showback_report(
                "ml-research".to_string(),
                "2024-01-01".to_string(),
                "2024-01-31".to_string(),
            )
            .await;

        assert!(result.is_ok());
        let report = result.unwrap();
        assert!(report.id.starts_with("report-"));
        assert!(report.name.contains("ml-research"));
        assert!(report.download_url.contains("ml-research"));
    }

    #[tokio::test]
    async fn test_generate_showback_report_validation_empty_team_id() {
        let bridge = CostsBridge::new();

        let result = bridge
            .generate_showback_report(
                "".to_string(),
                "2024-01-01".to_string(),
                "2024-01-31".to_string(),
            )
            .await;

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), HorizonError::InvalidConfig(_)));
    }

    #[tokio::test]
    async fn test_report_id_uniqueness() {
        let bridge = CostsBridge::new();

        // Generate multiple reports rapidly
        let report1 = bridge
            .generate_chargeback_report(
                "2024-01-01".to_string(),
                "2024-01-31".to_string(),
                "csv".to_string(),
            )
            .await
            .unwrap();

        let report2 = bridge
            .generate_chargeback_report(
                "2024-02-01".to_string(),
                "2024-02-28".to_string(),
                "json".to_string(),
            )
            .await
            .unwrap();

        // IDs should be unique (using UUID instead of timestamp)
        assert_ne!(report1.id, report2.id);
    }
}
