//! Intelligence Bridge
//!
//! Integrates horizon-services intelligence modules with Horizon.
//! Provides access to efficiency intelligence, margin analysis, vendor tracking, and executive KPIs.
//!
//! Currently uses mock data until intelligence services are fully integrated.

use crate::error::HorizonError;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Resource status.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ResourceStatus {
    Idle,
    Underutilized,
    Optimal,
    Overutilized,
}

/// Alert severity.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Severity {
    Info,
    Warning,
    Critical,
}

/// Idle resource detected by efficiency intelligence.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct IdleResource {
    pub id: String,
    pub name: String,
    #[serde(rename = "type")]
    pub resource_type: ResourceType,
    pub node_id: String,
    pub hostname: String,
    pub idle_since: String,
    pub idle_hours: f64,
    pub potential_savings_usd: f64,
    pub status: ResourceStatus,
    pub recommended_action: String,
}

/// Resource type.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ResourceType {
    Gpu,
    Cpu,
    Storage,
    Network,
}

/// Profit margin analysis.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ProfitMargin {
    pub id: String,
    pub name: String,
    #[serde(rename = "type")]
    pub margin_type: MarginType,
    pub revenue_usd: f64,
    pub cost_usd: f64,
    pub margin_usd: f64,
    pub margin_pct: f64,
    pub trend_pct: f64,
    pub period: String,
}

/// Margin type.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MarginType {
    Service,
    Tenant,
    Project,
}

/// Vendor utilization tracking.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct VendorUtilization {
    pub vendor_id: String,
    pub vendor_name: String,
    pub contract_value_usd: f64,
    pub used_value_usd: f64,
    pub utilization_pct: f64,
    pub contract_end: String,
    pub days_remaining: u32,
    pub status: VendorStatus,
    pub recommendations: Vec<String>,
}

/// Vendor status.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum VendorStatus {
    Underutilized,
    Optimal,
    Overutilized,
    Expiring,
}

/// Executive KPI.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ExecutiveKpi {
    pub name: String,
    pub value: f64,
    pub unit: String,
    pub trend_pct: f64,
    pub trend_direction: TrendDirection,
    pub target: Option<f64>,
    pub status: KpiStatus,
}

/// Trend direction.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TrendDirection {
    Up,
    Down,
    Stable,
}

/// KPI status.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KpiStatus {
    OnTrack,
    AtRisk,
    OffTrack,
}

/// Intelligence alert.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct IntelligenceAlert {
    pub id: String,
    pub title: String,
    pub description: String,
    pub severity: Severity,
    pub source: AlertSource,
    pub created_at: String,
    pub acknowledged: bool,
}

/// Alert source.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AlertSource {
    Efficiency,
    Margin,
    Vendor,
    Executive,
}

/// Intelligence summary combining all data.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct IntelligenceSummary {
    pub idle_resources: Vec<IdleResource>,
    pub total_potential_savings_usd: f64,
    pub profit_margins: Vec<ProfitMargin>,
    pub overall_margin_pct: f64,
    pub vendor_utilizations: Vec<VendorUtilization>,
    pub kpis: Vec<ExecutiveKpi>,
    pub alerts: Vec<IntelligenceAlert>,
}

/// Bridge to intelligence services.
pub struct IntelligenceBridge {
    state: Arc<RwLock<MockIntelligenceState>>,
}

struct MockIntelligenceState {
    idle_resources: Vec<IdleResource>,
    profit_margins: Vec<ProfitMargin>,
    vendor_utilizations: Vec<VendorUtilization>,
    kpis: Vec<ExecutiveKpi>,
    alerts: Vec<IntelligenceAlert>,
}

impl MockIntelligenceState {
    fn new() -> Self {
        let idle_resources = vec![
            IdleResource {
                id: "gpu-idle-1".to_string(),
                name: "NVIDIA A100 #3".to_string(),
                resource_type: ResourceType::Gpu,
                node_id: "gpu-server-1".to_string(),
                hostname: "gpu1.cluster.local".to_string(),
                idle_since: "2024-12-28T14:30:00Z".to_string(),
                idle_hours: 48.5,
                potential_savings_usd: 245.00,
                status: ResourceStatus::Idle,
                recommended_action: "Consider scheduling batch jobs or hibernating".to_string(),
            },
            IdleResource {
                id: "storage-idle-1".to_string(),
                name: "NVMe Pool B".to_string(),
                resource_type: ResourceType::Storage,
                node_id: "storage-1".to_string(),
                hostname: "storage1.cluster.local".to_string(),
                idle_since: "2024-12-25T08:00:00Z".to_string(),
                idle_hours: 120.0,
                potential_savings_usd: 85.00,
                status: ResourceStatus::Underutilized,
                recommended_action: "Consolidate with primary storage pool".to_string(),
            },
        ];

        let profit_margins = vec![
            ProfitMargin {
                id: "service-inference".to_string(),
                name: "Inference API".to_string(),
                margin_type: MarginType::Service,
                revenue_usd: 45_000.00,
                cost_usd: 28_500.00,
                margin_usd: 16_500.00,
                margin_pct: 36.7,
                trend_pct: 5.2,
                period: "2024-12".to_string(),
            },
            ProfitMargin {
                id: "service-training".to_string(),
                name: "Training Platform".to_string(),
                margin_type: MarginType::Service,
                revenue_usd: 32_000.00,
                cost_usd: 24_000.00,
                margin_usd: 8_000.00,
                margin_pct: 25.0,
                trend_pct: -2.3,
                period: "2024-12".to_string(),
            },
            ProfitMargin {
                id: "tenant-acme".to_string(),
                name: "ACME Corp".to_string(),
                margin_type: MarginType::Tenant,
                revenue_usd: 18_500.00,
                cost_usd: 12_000.00,
                margin_usd: 6_500.00,
                margin_pct: 35.1,
                trend_pct: 8.7,
                period: "2024-12".to_string(),
            },
        ];

        let vendor_utilizations = vec![
            VendorUtilization {
                vendor_id: "aws-reserved".to_string(),
                vendor_name: "AWS Reserved Instances".to_string(),
                contract_value_usd: 120_000.00,
                used_value_usd: 78_000.00,
                utilization_pct: 65.0,
                contract_end: "2025-06-30".to_string(),
                days_remaining: 180,
                status: VendorStatus::Underutilized,
                recommendations: vec![
                    "Consider selling unused reserved capacity".to_string(),
                    "Migrate more workloads to reserved instances".to_string(),
                ],
            },
            VendorUtilization {
                vendor_id: "nvidia-eula".to_string(),
                vendor_name: "NVIDIA Enterprise License".to_string(),
                contract_value_usd: 50_000.00,
                used_value_usd: 48_500.00,
                utilization_pct: 97.0,
                contract_end: "2025-03-31".to_string(),
                days_remaining: 90,
                status: VendorStatus::Expiring,
                recommendations: vec![
                    "Renew contract before expiration".to_string(),
                    "Negotiate volume discount for renewal".to_string(),
                ],
            },
        ];

        let kpis = vec![
            ExecutiveKpi {
                name: "GPU Utilization".to_string(),
                value: 78.5,
                unit: "%".to_string(),
                trend_pct: 3.2,
                trend_direction: TrendDirection::Up,
                target: Some(85.0),
                status: KpiStatus::AtRisk,
            },
            ExecutiveKpi {
                name: "Cost per GPU Hour".to_string(),
                value: 2.45,
                unit: "USD".to_string(),
                trend_pct: -5.1,
                trend_direction: TrendDirection::Down,
                target: Some(2.50),
                status: KpiStatus::OnTrack,
            },
            ExecutiveKpi {
                name: "Job Success Rate".to_string(),
                value: 98.7,
                unit: "%".to_string(),
                trend_pct: 0.5,
                trend_direction: TrendDirection::Stable,
                target: Some(99.0),
                status: KpiStatus::AtRisk,
            },
            ExecutiveKpi {
                name: "Average Queue Time".to_string(),
                value: 12.5,
                unit: "min".to_string(),
                trend_pct: -8.3,
                trend_direction: TrendDirection::Down,
                target: Some(15.0),
                status: KpiStatus::OnTrack,
            },
        ];

        let alerts = vec![
            IntelligenceAlert {
                id: "alert-1".to_string(),
                title: "High GPU Idle Time Detected".to_string(),
                description: "GPU Server 1 has 2 GPUs idle for over 24 hours".to_string(),
                severity: Severity::Warning,
                source: AlertSource::Efficiency,
                created_at: "2024-12-30T10:15:00Z".to_string(),
                acknowledged: false,
            },
            IntelligenceAlert {
                id: "alert-2".to_string(),
                title: "Vendor Contract Expiring Soon".to_string(),
                description: "NVIDIA Enterprise License expires in 90 days".to_string(),
                severity: Severity::Info,
                source: AlertSource::Vendor,
                created_at: "2024-12-30T08:00:00Z".to_string(),
                acknowledged: false,
            },
        ];

        Self {
            idle_resources,
            profit_margins,
            vendor_utilizations,
            kpis,
            alerts,
        }
    }
}

impl IntelligenceBridge {
    pub fn new() -> Self {
        Self {
            state: Arc::new(RwLock::new(MockIntelligenceState::new())),
        }
    }

    pub async fn get_summary(&self) -> IntelligenceSummary {
        let state = self.state.read().await;
        let total_savings: f64 = state.idle_resources.iter().map(|r| r.potential_savings_usd).sum();
        let total_margin: f64 = state.profit_margins.iter().map(|m| m.margin_usd).sum();
        let total_revenue: f64 = state.profit_margins.iter().map(|m| m.revenue_usd).sum();

        IntelligenceSummary {
            idle_resources: state.idle_resources.clone(),
            total_potential_savings_usd: total_savings,
            profit_margins: state.profit_margins.clone(),
            overall_margin_pct: if total_revenue > 0.0 { (total_margin / total_revenue) * 100.0 } else { 0.0 },
            vendor_utilizations: state.vendor_utilizations.clone(),
            kpis: state.kpis.clone(),
            alerts: state.alerts.clone(),
        }
    }

    pub async fn get_idle_resources(&self) -> Vec<IdleResource> {
        let state = self.state.read().await;
        state.idle_resources.clone()
    }

    pub async fn get_profit_margins(&self, margin_type: Option<String>) -> Vec<ProfitMargin> {
        let state = self.state.read().await;
        match margin_type {
            Some(t) => state
                .profit_margins
                .iter()
                .filter(|m| format!("{:?}", m.margin_type).to_lowercase() == t.to_lowercase())
                .cloned()
                .collect(),
            None => state.profit_margins.clone(),
        }
    }

    pub async fn get_vendor_utilizations(&self) -> Vec<VendorUtilization> {
        let state = self.state.read().await;
        state.vendor_utilizations.clone()
    }

    pub async fn get_kpis(&self) -> Vec<ExecutiveKpi> {
        let state = self.state.read().await;
        state.kpis.clone()
    }

    pub async fn get_alerts(&self) -> Vec<IntelligenceAlert> {
        let state = self.state.read().await;
        state.alerts.clone()
    }

    pub async fn acknowledge_alert(&self, alert_id: String) -> Result<(), HorizonError> {
        if alert_id.trim().is_empty() {
            return Err(HorizonError::InvalidConfig(
                "Alert ID cannot be empty".to_string(),
            ));
        }

        let mut state = self.state.write().await;
        let alert = state
            .alerts
            .iter_mut()
            .find(|a| a.id == alert_id)
            .ok_or_else(|| HorizonError::NotFound(format!("Alert '{}' not found", alert_id)))?;

        alert.acknowledged = true;
        Ok(())
    }

    pub async fn terminate_idle_resource(&self, resource_id: String) -> Result<(), HorizonError> {
        if resource_id.trim().is_empty() {
            return Err(HorizonError::InvalidConfig(
                "Resource ID cannot be empty".to_string(),
            ));
        }

        let mut state = self.state.write().await;
        let initial_len = state.idle_resources.len();
        state.idle_resources.retain(|r| r.id != resource_id);

        if state.idle_resources.len() == initial_len {
            return Err(HorizonError::NotFound(format!(
                "Idle resource '{}' not found",
                resource_id
            )));
        }

        Ok(())
    }
}

impl Default for IntelligenceBridge {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_get_summary_calculates_savings_correctly() {
        let bridge = IntelligenceBridge::new();
        let summary = bridge.get_summary().await;

        // Verify total savings equals sum of idle resource savings
        let expected_savings: f64 = summary
            .idle_resources
            .iter()
            .map(|r| r.potential_savings_usd)
            .sum();

        assert_eq!(summary.total_potential_savings_usd, expected_savings);
        assert!(!summary.idle_resources.is_empty());
        assert!(!summary.profit_margins.is_empty());
        assert!(!summary.vendor_utilizations.is_empty());
        assert!(!summary.kpis.is_empty());
        assert!(!summary.alerts.is_empty());
    }

    #[tokio::test]
    async fn test_get_summary_calculates_margin_correctly() {
        let bridge = IntelligenceBridge::new();
        let summary = bridge.get_summary().await;

        let total_margin: f64 = summary.profit_margins.iter().map(|m| m.margin_usd).sum();
        let total_revenue: f64 = summary.profit_margins.iter().map(|m| m.revenue_usd).sum();

        let expected_margin_pct = (total_margin / total_revenue) * 100.0;

        assert!((summary.overall_margin_pct - expected_margin_pct).abs() < 0.01);
    }

    #[tokio::test]
    async fn test_get_idle_resources() {
        let bridge = IntelligenceBridge::new();
        let resources = bridge.get_idle_resources().await;

        assert!(resources.len() >= 2, "Should have at least 2 idle resources");
        for resource in &resources {
            assert!(resource.idle_hours > 0.0);
            assert!(resource.potential_savings_usd > 0.0);
        }
    }

    #[tokio::test]
    async fn test_get_profit_margins_all() {
        let bridge = IntelligenceBridge::new();
        let margins = bridge.get_profit_margins(None).await;

        assert!(margins.len() >= 3, "Should have at least 3 profit margins");
    }

    #[tokio::test]
    async fn test_get_profit_margins_filtered_by_service() {
        let bridge = IntelligenceBridge::new();
        let service_margins = bridge.get_profit_margins(Some("service".to_string())).await;

        assert!(
            service_margins.len() >= 2,
            "Should have at least 2 service margins"
        );
        for margin in &service_margins {
            assert!(
                format!("{:?}", margin.margin_type)
                    .to_lowercase()
                    .contains("service")
            );
        }
    }

    #[tokio::test]
    async fn test_get_profit_margins_filtered_by_tenant() {
        let bridge = IntelligenceBridge::new();
        let tenant_margins = bridge.get_profit_margins(Some("tenant".to_string())).await;

        assert!(
            tenant_margins.len() >= 1,
            "Should have at least 1 tenant margin"
        );
        for margin in &tenant_margins {
            assert!(
                format!("{:?}", margin.margin_type)
                    .to_lowercase()
                    .contains("tenant")
            );
        }
    }

    #[tokio::test]
    async fn test_get_vendor_utilizations() {
        let bridge = IntelligenceBridge::new();
        let vendors = bridge.get_vendor_utilizations().await;

        assert!(vendors.len() >= 2, "Should have at least 2 vendor utilizations");
        for vendor in &vendors {
            assert!(vendor.utilization_pct >= 0.0 && vendor.utilization_pct <= 100.0);
            assert!(vendor.used_value_usd <= vendor.contract_value_usd);
        }
    }

    #[tokio::test]
    async fn test_get_kpis() {
        let bridge = IntelligenceBridge::new();
        let kpis = bridge.get_kpis().await;

        assert!(kpis.len() >= 4, "Should have at least 4 KPIs");
        for kpi in &kpis {
            assert!(!kpi.name.is_empty());
            assert!(!kpi.unit.is_empty());
        }
    }

    #[tokio::test]
    async fn test_get_alerts() {
        let bridge = IntelligenceBridge::new();
        let alerts = bridge.get_alerts().await;

        assert!(alerts.len() >= 2, "Should have at least 2 intelligence alerts");
        for alert in &alerts {
            assert!(!alert.id.is_empty());
            assert!(!alert.title.is_empty());
        }
    }

    #[tokio::test]
    async fn test_acknowledge_alert_success() {
        let bridge = IntelligenceBridge::new();

        // Get initial alerts
        let alerts = bridge.get_alerts().await;
        let test_alert = alerts.first().expect("Should have at least one alert");
        assert!(!test_alert.acknowledged);

        // Acknowledge the alert
        let result = bridge.acknowledge_alert(test_alert.id.clone()).await;
        assert!(result.is_ok());

        // Verify it was acknowledged
        let updated_alerts = bridge.get_alerts().await;
        let updated_alert = updated_alerts
            .iter()
            .find(|a| a.id == test_alert.id)
            .expect("Alert should still exist");
        assert!(updated_alert.acknowledged);
    }

    #[tokio::test]
    async fn test_acknowledge_alert_validation_empty_id() {
        let bridge = IntelligenceBridge::new();

        let result = bridge.acknowledge_alert("".to_string()).await;

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), HorizonError::InvalidConfig(_)));
    }

    #[tokio::test]
    async fn test_acknowledge_alert_not_found() {
        let bridge = IntelligenceBridge::new();

        let result = bridge
            .acknowledge_alert("nonexistent-alert-id".to_string())
            .await;

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), HorizonError::NotFound(_)));
    }

    #[tokio::test]
    async fn test_terminate_idle_resource_success() {
        let bridge = IntelligenceBridge::new();

        // Get initial resources
        let resources = bridge.get_idle_resources().await;
        let initial_count = resources.len();
        let test_resource = resources.first().expect("Should have at least one resource");

        // Terminate the resource
        let result = bridge.terminate_idle_resource(test_resource.id.clone()).await;
        assert!(result.is_ok());

        // Verify it was removed
        let updated_resources = bridge.get_idle_resources().await;
        assert_eq!(updated_resources.len(), initial_count - 1);
        assert!(!updated_resources.iter().any(|r| r.id == test_resource.id));
    }

    #[tokio::test]
    async fn test_terminate_idle_resource_validation_empty_id() {
        let bridge = IntelligenceBridge::new();

        let result = bridge.terminate_idle_resource("".to_string()).await;

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), HorizonError::InvalidConfig(_)));
    }

    #[tokio::test]
    async fn test_terminate_idle_resource_not_found() {
        let bridge = IntelligenceBridge::new();

        let result = bridge
            .terminate_idle_resource("nonexistent-resource-id".to_string())
            .await;

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), HorizonError::NotFound(_)));
    }

    #[tokio::test]
    async fn test_profit_margin_calculations() {
        let bridge = IntelligenceBridge::new();
        let margins = bridge.get_profit_margins(None).await;

        for margin in &margins {
            // Verify margin calculation
            let expected_margin = margin.revenue_usd - margin.cost_usd;
            assert!((margin.margin_usd - expected_margin).abs() < 0.01);

            // Verify margin percentage
            let expected_pct = (margin.margin_usd / margin.revenue_usd) * 100.0;
            assert!((margin.margin_pct - expected_pct).abs() < 0.1);
        }
    }

    #[tokio::test]
    async fn test_vendor_utilization_calculations() {
        let bridge = IntelligenceBridge::new();
        let vendors = bridge.get_vendor_utilizations().await;

        for vendor in &vendors {
            // Verify utilization percentage
            let expected_pct = (vendor.used_value_usd / vendor.contract_value_usd) * 100.0;
            assert!((vendor.utilization_pct - expected_pct).abs() < 0.1);
        }
    }
}
