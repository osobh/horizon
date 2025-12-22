//! Waste analysis logic for detecting and quantifying resource waste
//!
//! This module analyzes resource usage to identify different types of waste
//! and provides recovery actions to reduce costs.

use crate::error::CostOptimizationResult;
use crate::usage_analyzer::types::*;
use std::collections::HashMap;

/// Waste analyzer for detecting resource waste and inefficiencies
pub struct WasteAnalyzer {
    /// Configuration
    config: UsageAnalyzerConfig,
}

impl WasteAnalyzer {
    /// Create a new waste analyzer
    pub fn new(config: UsageAnalyzerConfig) -> Self {
        Self { config }
    }

    /// Analyze waste based on usage patterns and costs
    pub fn analyze_waste(
        &self,
        request: &AnalysisRequest,
        statistics: &UsageStatistics,
        temporal: &TemporalAnalysis,
        cost_per_hour: f64,
    ) -> CostOptimizationResult<WasteAnalysis> {
        let mut waste_by_type = HashMap::new();
        let mut total_waste_cost = 0.0;

        // Analyze overprovisioning waste
        if let Some(waste) =
            self.analyze_overprovisioning_waste(request, statistics, cost_per_hour)?
        {
            total_waste_cost += waste.amount;
            waste_by_type.insert(WasteType::Overprovisioned, waste);
        }

        // Analyze idle resource waste
        if let Some(waste) = self.analyze_idle_waste(request, statistics, cost_per_hour)? {
            total_waste_cost += waste.amount;
            waste_by_type.insert(WasteType::Idle, waste);
        }

        // Analyze inefficient configuration waste
        if let Some(waste) =
            self.analyze_inefficient_waste(request, statistics, temporal, cost_per_hour)?
        {
            total_waste_cost += waste.amount;
            waste_by_type.insert(WasteType::Inefficient, waste);
        }

        // Analyze orphaned resource waste
        if let Some(waste) = self.analyze_orphaned_waste(request, statistics, cost_per_hour)? {
            total_waste_cost += waste.amount;
            waste_by_type.insert(WasteType::Orphaned, waste);
        }

        // Determine waste severity
        let severity = self.calculate_waste_severity(total_waste_cost);

        // Generate recovery actions
        let recovery_actions = self.generate_recovery_actions(&waste_by_type, request);

        Ok(WasteAnalysis {
            total_waste_cost,
            waste_by_type,
            severity,
            recovery_actions,
        })
    }

    /// Analyze overprovisioning waste
    pub fn analyze_overprovisioning_waste(
        &self,
        request: &AnalysisRequest,
        statistics: &UsageStatistics,
        cost_per_hour: f64,
    ) -> CostOptimizationResult<Option<WasteDetail>> {
        if statistics.p95_utilization >= self.config.overprovision_threshold {
            return Ok(None); // Not overprovisioned
        }

        let waste_percent = (100.0 - statistics.p95_utilization) / 100.0;
        let waste_amount = cost_per_hour * 24.0 * 30.0 * waste_percent; // Monthly waste

        Ok(Some(WasteDetail {
            waste_type: WasteType::Overprovisioned,
            amount: waste_amount,
            percentage: waste_percent * 100.0,
            affected_resources: vec![request.resource_id.clone()],
            duration: request.period,
        }))
    }

    /// Analyze idle resource waste
    pub fn analyze_idle_waste(
        &self,
        request: &AnalysisRequest,
        statistics: &UsageStatistics,
        cost_per_hour: f64,
    ) -> CostOptimizationResult<Option<WasteDetail>> {
        if statistics.idle_percentage < 20.0 {
            return Ok(None); // Not significantly idle
        }

        let waste_percent = statistics.idle_percentage / 100.0;
        let waste_amount = cost_per_hour * 24.0 * 30.0 * waste_percent; // Monthly waste

        Ok(Some(WasteDetail {
            waste_type: WasteType::Idle,
            amount: waste_amount,
            percentage: statistics.idle_percentage,
            affected_resources: vec![request.resource_id.clone()],
            duration: request.period,
        }))
    }

    /// Analyze inefficient configuration waste
    pub fn analyze_inefficient_waste(
        &self,
        request: &AnalysisRequest,
        statistics: &UsageStatistics,
        temporal: &TemporalAnalysis,
        cost_per_hour: f64,
    ) -> CostOptimizationResult<Option<WasteDetail>> {
        // Check for inefficient patterns
        let has_inefficiency = self.detect_configuration_inefficiency(statistics, temporal);

        if !has_inefficiency {
            return Ok(None);
        }

        // Estimate waste from inefficient configuration (e.g., no autoscaling, wrong instance type)
        let inefficiency_factor = self.calculate_inefficiency_factor(statistics, temporal);
        let waste_amount = cost_per_hour * 24.0 * 30.0 * inefficiency_factor;

        Ok(Some(WasteDetail {
            waste_type: WasteType::Inefficient,
            amount: waste_amount,
            percentage: inefficiency_factor * 100.0,
            affected_resources: vec![request.resource_id.clone()],
            duration: request.period,
        }))
    }

    /// Analyze orphaned resource waste
    pub fn analyze_orphaned_waste(
        &self,
        request: &AnalysisRequest,
        statistics: &UsageStatistics,
        cost_per_hour: f64,
    ) -> CostOptimizationResult<Option<WasteDetail>> {
        // Detect potential orphaned resources (extremely low usage with no clear purpose)
        let is_orphaned = statistics.avg_utilization < 1.0
            && statistics.peak_utilization < 5.0
            && statistics.idle_percentage > 95.0;

        if !is_orphaned {
            return Ok(None);
        }

        let waste_amount = cost_per_hour * 24.0 * 30.0; // Full cost as waste

        Ok(Some(WasteDetail {
            waste_type: WasteType::Orphaned,
            amount: waste_amount,
            percentage: 100.0,
            affected_resources: vec![request.resource_id.clone()],
            duration: request.period,
        }))
    }

    /// Detect configuration inefficiencies
    fn detect_configuration_inefficiency(
        &self,
        statistics: &UsageStatistics,
        temporal: &TemporalAnalysis,
    ) -> bool {
        // Check for high variance without autoscaling
        let high_variance = statistics.std_deviation > 20.0;

        // Check for clear on/off patterns without scheduling
        let has_clear_schedule =
            !temporal.low_periods.is_empty() && temporal.low_periods.iter().any(|p| p.can_shutdown);

        // Check for spiky usage without proper instance type
        let spiky_usage = statistics.p99_utilization > statistics.avg_utilization * 2.5;

        high_variance || has_clear_schedule || spiky_usage
    }

    /// Calculate inefficiency factor
    fn calculate_inefficiency_factor(
        &self,
        statistics: &UsageStatistics,
        temporal: &TemporalAnalysis,
    ) -> f64 {
        let mut factor = 0.0;

        // Variance inefficiency
        if statistics.std_deviation > 20.0 {
            factor += 0.15; // 15% waste from not having autoscaling
        }

        // Scheduling inefficiency
        let schedulable_hours: f64 = temporal
            .low_periods
            .iter()
            .filter(|p| p.can_shutdown)
            .map(|p| p.end.signed_duration_since(p.start).num_hours() as f64)
            .sum();

        if schedulable_hours > 8.0 {
            factor += (schedulable_hours / 24.0) * 0.8; // 80% of schedulable time is waste
        }

        // Instance type inefficiency (rough estimate)
        if statistics.p99_utilization > statistics.avg_utilization * 2.5 {
            factor += 0.1; // 10% waste from wrong instance type
        }

        factor.min(0.5) // Cap at 50% inefficiency
    }

    /// Calculate waste severity based on total cost
    pub fn calculate_waste_severity(&self, total_waste_cost: f64) -> WasteSeverity {
        if total_waste_cost < 100.0 {
            WasteSeverity::Minimal
        } else if total_waste_cost < 500.0 {
            WasteSeverity::Low
        } else if total_waste_cost < 1000.0 {
            WasteSeverity::Medium
        } else if total_waste_cost < 5000.0 {
            WasteSeverity::High
        } else {
            WasteSeverity::Critical
        }
    }

    /// Generate recovery actions for detected waste
    pub fn generate_recovery_actions(
        &self,
        waste_by_type: &HashMap<WasteType, WasteDetail>,
        request: &AnalysisRequest,
    ) -> Vec<RecoveryAction> {
        let mut actions = Vec::new();

        // Recovery action for overprovisioned resources
        if let Some(waste) = waste_by_type.get(&WasteType::Overprovisioned) {
            actions.push(RecoveryAction {
                action: "Rightsize resource to match actual usage patterns".to_string(),
                resources: waste.affected_resources.clone(),
                recovery_amount: waste.amount * 0.8, // 80% recovery potential
                priority: self.determine_action_priority(waste.amount),
            });
        }

        // Recovery action for idle resources
        if let Some(waste) = waste_by_type.get(&WasteType::Idle) {
            if waste.percentage > 80.0 {
                actions.push(RecoveryAction {
                    action: "Terminate idle resource after confirming it's not needed".to_string(),
                    resources: waste.affected_resources.clone(),
                    recovery_amount: waste.amount * 0.95, // 95% recovery potential
                    priority: ActionPriority::High,
                });
            } else {
                actions.push(RecoveryAction {
                    action: "Implement auto-shutdown during idle periods".to_string(),
                    resources: waste.affected_resources.clone(),
                    recovery_amount: waste.amount * 0.7, // 70% recovery potential
                    priority: ActionPriority::Medium,
                });
            }
        }

        // Recovery action for inefficient configuration
        if let Some(waste) = waste_by_type.get(&WasteType::Inefficient) {
            actions.push(RecoveryAction {
                action: "Optimize resource configuration (autoscaling, scheduling, instance type)"
                    .to_string(),
                resources: waste.affected_resources.clone(),
                recovery_amount: waste.amount * 0.6, // 60% recovery potential
                priority: ActionPriority::Medium,
            });
        }

        // Recovery action for orphaned resources
        if let Some(waste) = waste_by_type.get(&WasteType::Orphaned) {
            actions.push(RecoveryAction {
                action: "Investigate and terminate orphaned resource if confirmed unused"
                    .to_string(),
                resources: waste.affected_resources.clone(),
                recovery_amount: waste.amount * 0.9, // 90% recovery potential
                priority: ActionPriority::Urgent,
            });
        }

        // Sort actions by priority and potential recovery
        actions.sort_by(|a, b| {
            b.priority
                .cmp(&a.priority)
                .then_with(|| b.recovery_amount.partial_cmp(&a.recovery_amount).unwrap())
        });

        actions
    }

    /// Determine action priority based on waste amount
    fn determine_action_priority(&self, waste_amount: f64) -> ActionPriority {
        if waste_amount > 2000.0 {
            ActionPriority::Urgent
        } else if waste_amount > 1000.0 {
            ActionPriority::High
        } else if waste_amount > 200.0 {
            ActionPriority::Medium
        } else {
            ActionPriority::Low
        }
    }

    /// Calculate waste reduction potential
    pub fn calculate_reduction_potential(&self, waste_analysis: &WasteAnalysis) -> f64 {
        waste_analysis
            .recovery_actions
            .iter()
            .map(|action| action.recovery_amount)
            .sum()
    }

    /// Generate waste summary report
    pub fn generate_waste_summary(&self, waste_analysis: &WasteAnalysis) -> String {
        let mut summary = String::new();

        summary.push_str(&format!(
            "Total waste detected: ${:.2}/month (Severity: {:?})\n",
            waste_analysis.total_waste_cost, waste_analysis.severity
        ));

        if !waste_analysis.waste_by_type.is_empty() {
            summary.push_str("\nWaste breakdown:\n");
            for (waste_type, detail) in &waste_analysis.waste_by_type {
                summary.push_str(&format!(
                    "- {:?}: ${:.2} ({:.1}%)\n",
                    waste_type, detail.amount, detail.percentage
                ));
            }
        }

        if !waste_analysis.recovery_actions.is_empty() {
            summary.push_str("\nRecommended actions:\n");
            for (i, action) in waste_analysis.recovery_actions.iter().enumerate() {
                summary.push_str(&format!(
                    "{}. {} (Potential recovery: ${:.2})\n",
                    i + 1,
                    action.action,
                    action.recovery_amount
                ));
            }
        }

        summary
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resource_tracker::ResourceType;
    use std::time::Duration;

    fn create_test_request() -> AnalysisRequest {
        AnalysisRequest {
            resource_id: "test-resource".to_string(),
            resource_type: ResourceType::Cpu,
            period: Duration::from_secs(86400 * 30), // 30 days
            include_recommendations: true,
            confidence_threshold: 0.8,
            cost_per_hour: Some(10.0),
        }
    }

    fn create_test_temporal() -> TemporalAnalysis {
        TemporalAnalysis {
            hourly_pattern: vec![],
            daily_pattern: vec![],
            peak_times: vec![],
            low_periods: vec![LowPeriod {
                start: chrono::Utc::now() - chrono::Duration::hours(10),
                end: chrono::Utc::now() - chrono::Duration::hours(2),
                avg_utilization: 2.0,
                can_shutdown: true,
            }],
            trend: UsageTrend {
                direction: TrendDirection::Stable,
                rate: 0.0,
                confidence: 0.8,
            },
        }
    }

    #[test]
    fn test_overprovisioning_waste_detection() {
        let config = UsageAnalyzerConfig::default();
        let analyzer = WasteAnalyzer::new(config);

        let request = create_test_request();
        let statistics = UsageStatistics {
            avg_utilization: 20.0,
            peak_utilization: 30.0,
            min_utilization: 10.0,
            std_deviation: 5.0,
            p95_utilization: 25.0, // Below 30% threshold
            p99_utilization: 28.0,
            idle_percentage: 10.0,
            sample_count: 1000,
        };

        let waste = analyzer
            .analyze_overprovisioning_waste(&request, &statistics, 10.0)
            .unwrap()
            .unwrap();

        assert_eq!(waste.waste_type, WasteType::Overprovisioned);
        assert!(waste.amount > 0.0);
        assert_eq!(waste.percentage, 75.0); // (100 - 25) / 100 * 100
    }

    #[test]
    fn test_idle_waste_detection() {
        let config = UsageAnalyzerConfig::default();
        let analyzer = WasteAnalyzer::new(config);

        let request = create_test_request();
        let statistics = UsageStatistics {
            avg_utilization: 5.0,
            peak_utilization: 10.0,
            min_utilization: 0.0,
            std_deviation: 3.0,
            p95_utilization: 8.0,
            p99_utilization: 10.0,
            idle_percentage: 80.0, // High idle percentage
            sample_count: 1000,
        };

        let waste = analyzer
            .analyze_idle_waste(&request, &statistics, 10.0)
            .unwrap()
            .unwrap();

        assert_eq!(waste.waste_type, WasteType::Idle);
        assert!(waste.amount > 0.0);
        assert_eq!(waste.percentage, 80.0);
    }

    #[test]
    fn test_orphaned_waste_detection() {
        let config = UsageAnalyzerConfig::default();
        let analyzer = WasteAnalyzer::new(config);

        let request = create_test_request();
        let statistics = UsageStatistics {
            avg_utilization: 0.5, // Extremely low
            peak_utilization: 2.0,
            min_utilization: 0.0,
            std_deviation: 0.5,
            p95_utilization: 1.0,
            p99_utilization: 2.0,
            idle_percentage: 98.0, // Nearly always idle
            sample_count: 1000,
        };

        let waste = analyzer
            .analyze_orphaned_waste(&request, &statistics, 10.0)
            .unwrap()
            .unwrap();

        assert_eq!(waste.waste_type, WasteType::Orphaned);
        assert_eq!(waste.percentage, 100.0);
        assert_eq!(waste.amount, 10.0 * 24.0 * 30.0); // Full monthly cost
    }

    #[test]
    fn test_inefficient_waste_detection() {
        let config = UsageAnalyzerConfig::default();
        let analyzer = WasteAnalyzer::new(config);

        let request = create_test_request();
        let statistics = UsageStatistics {
            avg_utilization: 50.0,
            peak_utilization: 90.0,
            min_utilization: 10.0,
            std_deviation: 25.0, // High variance
            p95_utilization: 85.0,
            p99_utilization: 90.0,
            idle_percentage: 20.0,
            sample_count: 1000,
        };
        let temporal = create_test_temporal();

        let waste = analyzer
            .analyze_inefficient_waste(&request, &statistics, &temporal, 10.0)
            .unwrap()
            .unwrap();

        assert_eq!(waste.waste_type, WasteType::Inefficient);
        assert!(waste.amount > 0.0);
        assert!(waste.percentage > 0.0);
    }

    #[test]
    fn test_waste_severity_calculation() {
        let config = UsageAnalyzerConfig::default();
        let analyzer = WasteAnalyzer::new(config);

        assert_eq!(
            analyzer.calculate_waste_severity(50.0),
            WasteSeverity::Minimal
        );
        assert_eq!(analyzer.calculate_waste_severity(300.0), WasteSeverity::Low);
        assert_eq!(
            analyzer.calculate_waste_severity(750.0),
            WasteSeverity::Medium
        );
        assert_eq!(
            analyzer.calculate_waste_severity(2500.0),
            WasteSeverity::High
        );
        assert_eq!(
            analyzer.calculate_waste_severity(10000.0),
            WasteSeverity::Critical
        );
    }

    #[test]
    fn test_recovery_actions_generation() {
        let config = UsageAnalyzerConfig::default();
        let analyzer = WasteAnalyzer::new(config);

        let request = create_test_request();
        let mut waste_by_type = HashMap::new();

        // Add overprovisioning waste
        waste_by_type.insert(
            WasteType::Overprovisioned,
            WasteDetail {
                waste_type: WasteType::Overprovisioned,
                amount: 1000.0,
                percentage: 50.0,
                affected_resources: vec!["resource-1".to_string()],
                duration: Duration::from_secs(86400 * 30),
            },
        );

        // Add idle waste
        waste_by_type.insert(
            WasteType::Idle,
            WasteDetail {
                waste_type: WasteType::Idle,
                amount: 500.0,
                percentage: 60.0,
                affected_resources: vec!["resource-1".to_string()],
                duration: Duration::from_secs(86400 * 30),
            },
        );

        let actions = analyzer.generate_recovery_actions(&waste_by_type, &request);

        assert!(!actions.is_empty());
        assert_eq!(actions.len(), 2);

        // Check that actions are sorted by priority and recovery amount
        assert!(actions[0].priority >= actions[1].priority);

        // Verify recovery amounts
        for action in &actions {
            assert!(action.recovery_amount > 0.0);
        }
    }

    #[test]
    fn test_comprehensive_waste_analysis() {
        let config = UsageAnalyzerConfig::default();
        let analyzer = WasteAnalyzer::new(config);

        let request = create_test_request();
        let statistics = UsageStatistics {
            avg_utilization: 15.0,
            peak_utilization: 25.0,
            min_utilization: 5.0,
            std_deviation: 8.0,
            p95_utilization: 22.0, // Overprovisioned
            p99_utilization: 25.0,
            idle_percentage: 30.0, // Significantly idle
            sample_count: 1000,
        };
        let temporal = create_test_temporal();

        let analysis = analyzer
            .analyze_waste(&request, &statistics, &temporal, 10.0)
            .unwrap();

        // Should detect multiple types of waste
        assert!(analysis.total_waste_cost > 0.0);
        assert!(analysis.waste_by_type.len() > 1);
        assert!(!analysis.recovery_actions.is_empty());
        assert!(matches!(
            analysis.severity,
            WasteSeverity::Low | WasteSeverity::Medium | WasteSeverity::High
        ));
    }

    #[test]
    fn test_waste_summary_generation() {
        let config = UsageAnalyzerConfig::default();
        let analyzer = WasteAnalyzer::new(config);

        let mut waste_by_type = HashMap::new();
        waste_by_type.insert(
            WasteType::Idle,
            WasteDetail {
                waste_type: WasteType::Idle,
                amount: 500.0,
                percentage: 80.0,
                affected_resources: vec!["resource-1".to_string()],
                duration: Duration::from_secs(86400 * 30),
            },
        );

        let analysis = WasteAnalysis {
            total_waste_cost: 500.0,
            waste_by_type,
            severity: WasteSeverity::Medium,
            recovery_actions: vec![RecoveryAction {
                action: "Test action".to_string(),
                resources: vec!["resource-1".to_string()],
                recovery_amount: 400.0,
                priority: ActionPriority::High,
            }],
        };

        let summary = analyzer.generate_waste_summary(&analysis);

        assert!(summary.contains("$500.00"));
        assert!(summary.contains("Medium"));
        assert!(summary.contains("Idle"));
        assert!(summary.contains("Test action"));
    }

    #[test]
    fn test_action_priority_determination() {
        let config = UsageAnalyzerConfig::default();
        let analyzer = WasteAnalyzer::new(config);

        assert_eq!(
            analyzer.determine_action_priority(100.0),
            ActionPriority::Low
        );
        assert_eq!(
            analyzer.determine_action_priority(500.0),
            ActionPriority::Medium
        );
        assert_eq!(
            analyzer.determine_action_priority(1500.0),
            ActionPriority::High
        );
        assert_eq!(
            analyzer.determine_action_priority(3000.0),
            ActionPriority::Urgent
        );
    }
}
