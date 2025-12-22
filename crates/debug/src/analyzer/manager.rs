//! Analysis manager for coordinating analysis operations

use super::types::*;
use crate::DebugError;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

/// Analysis manager for coordinating analysis operations
pub struct AnalysisManager {
    engine: Arc<dyn AnalysisEngine + Send + Sync>,
    reports: Arc<RwLock<HashMap<Uuid, AnalysisReport>>>,
    config: AnalysisManagerConfig,
}

/// Configuration for analysis manager
#[derive(Debug, Clone)]
pub struct AnalysisManagerConfig {
    pub max_stored_reports: usize,
    pub auto_analyze: bool,
    pub report_retention_hours: u64,
}

impl Default for AnalysisManagerConfig {
    fn default() -> Self {
        Self {
            max_stored_reports: 1000,
            auto_analyze: true,
            report_retention_hours: 168, // 1 week
        }
    }
}

impl AnalysisManager {
    /// Create new analysis manager
    pub fn new(
        engine: Arc<dyn AnalysisEngine + Send + Sync>,
        config: AnalysisManagerConfig,
    ) -> Self {
        Self {
            engine,
            reports: Arc::new(RwLock::new(HashMap::new())),
            config,
        }
    }

    /// Run performance analysis and store report
    pub async fn run_performance_analysis(
        &self,
        target: AnalysisTarget,
    ) -> Result<Uuid, DebugError> {
        let report = self.engine.analyze_performance(target).await?;
        let report_id = report.report_id;

        {
            let mut reports = self.reports.write().await;
            reports.insert(report_id, report);

            // Cleanup old reports if needed
            if reports.len() > self.config.max_stored_reports {
                let oldest_id = reports
                    .iter()
                    .min_by_key(|(_, r)| r.timestamp)
                    .map(|(id, _)| *id);

                if let Some(id) = oldest_id {
                    reports.remove(&id);
                }
            }
        }

        Ok(report_id)
    }

    /// Run anomaly detection
    pub async fn run_anomaly_detection(
        &self,
        target: AnalysisTarget,
        baseline: Option<BaselineData>,
    ) -> Result<Vec<AnomalyResult>, DebugError> {
        self.engine.detect_anomalies(target, baseline).await
    }

    /// Run comparison analysis
    pub async fn run_comparison_analysis(
        &self,
        baseline: AnalysisTarget,
        comparison: AnalysisTarget,
    ) -> Result<Uuid, DebugError> {
        let report = self.engine.compare_analysis(baseline, comparison).await?;
        let report_id = report.report_id;

        {
            let mut reports = self.reports.write().await;
            reports.insert(report_id, report);
        }

        Ok(report_id)
    }

    /// Run trend analysis
    pub async fn run_trend_analysis(
        &self,
        metrics: Vec<String>,
        time_range: (u64, u64),
    ) -> Result<Vec<TrendAnalysis>, DebugError> {
        self.engine.analyze_trends(metrics, time_range).await
    }

    /// Get analysis report by ID
    pub async fn get_report(&self, report_id: Uuid) -> Option<AnalysisReport> {
        let reports = self.reports.read().await;
        reports.get(&report_id).cloned()
    }

    /// List all reports
    pub async fn list_reports(&self) -> Vec<AnalysisReport> {
        self.reports.read().await.values().cloned().collect()
    }

    /// Get reports by analysis type
    pub async fn get_reports_by_type(&self, analysis_type: AnalysisType) -> Vec<AnalysisReport> {
        self.reports
            .read()
            .await
            .values()
            .filter(|r| r.analysis_type == analysis_type)
            .cloned()
            .collect()
    }

    /// Get high-severity reports
    pub async fn get_critical_reports(&self) -> Vec<AnalysisReport> {
        self.reports
            .read()
            .await
            .values()
            .filter(|r| matches!(r.severity, Severity::Critical | Severity::High))
            .cloned()
            .collect()
    }

    /// Clear old reports based on retention policy
    pub async fn cleanup_old_reports(&self) {
        let cutoff_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
            - (self.config.report_retention_hours * 3600);

        let mut reports = self.reports.write().await;
        reports.retain(|_, report| report.timestamp > cutoff_time);
    }

    /// Export reports to JSON
    pub async fn export_reports(&self) -> Result<String, DebugError> {
        let reports = self.list_reports().await;
        serde_json::to_string_pretty(&reports).map_err(|e| DebugError::AnalysisFailed {
            reason: format!("Failed to serialize reports: {}", e),
        })
    }

    /// Import reports from JSON
    pub async fn import_reports(&self, json: &str) -> Result<usize, DebugError> {
        let imported_reports: Vec<AnalysisReport> =
            serde_json::from_str(json).map_err(|e| DebugError::AnalysisFailed {
                reason: format!("Failed to deserialize reports: {}", e),
            })?;

        let count = imported_reports.len();
        let mut reports = self.reports.write().await;

        for report in imported_reports {
            reports.insert(report.report_id, report);
        }

        Ok(count)
    }

    /// Get summary statistics for all reports
    pub async fn get_report_statistics(&self) -> HashMap<String, serde_json::Value> {
        let reports = self.reports.read().await;
        let mut stats = HashMap::new();

        stats.insert(
            "total_reports".to_string(),
            serde_json::Value::Number(serde_json::Number::from(reports.len())),
        );

        let mut severity_counts = HashMap::new();
        let mut type_counts = HashMap::new();

        for report in reports.values() {
            *severity_counts
                .entry(format!("{:?}", report.severity))
                .or_insert(0) += 1;
            *type_counts
                .entry(format!("{:?}", report.analysis_type))
                .or_insert(0) += 1;
        }

        stats.insert(
            "severity_distribution".to_string(),
            serde_json::to_value(severity_counts).unwrap(),
        );
        stats.insert(
            "type_distribution".to_string(),
            serde_json::to_value(type_counts).unwrap(),
        );

        stats
    }
}
