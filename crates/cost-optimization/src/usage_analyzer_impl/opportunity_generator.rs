//! Optimization opportunity generation logic
//!
//! This module analyzes usage patterns and generates specific optimization
//! opportunities with cost savings estimates and implementation details.

use crate::error::CostOptimizationResult;
use crate::usage_analyzer::types::*;
use std::collections::HashMap;
use uuid::Uuid;

/// Opportunity generator for creating optimization recommendations
pub struct OpportunityGenerator {
    /// Configuration
    config: UsageAnalyzerConfig,
}

impl OpportunityGenerator {
    /// Create a new opportunity generator
    pub fn new(config: UsageAnalyzerConfig) -> Self {
        Self { config }
    }

    /// Identify optimization opportunities based on analysis
    pub fn identify_opportunities(
        &self,
        request: &AnalysisRequest,
        pattern: &UsagePattern,
        statistics: &UsageStatistics,
        temporal: &TemporalAnalysis,
    ) -> CostOptimizationResult<Vec<OptimizationOpportunity>> {
        let mut opportunities = Vec::new();

        // Rightsizing opportunity
        if statistics.p95_utilization < self.config.overprovision_threshold {
            if let Some(opportunity) = self.create_rightsizing_opportunity(request, statistics)? {
                opportunities.push(opportunity);
            }
        }

        // Idle termination opportunity
        if statistics.idle_percentage > 50.0 {
            if let Some(opportunity) =
                self.create_idle_termination_opportunity(request, statistics)?
            {
                opportunities.push(opportunity);
            }
        }

        // Spot instance opportunity
        if matches!(pattern, UsagePattern::Spiky | UsagePattern::Periodic) {
            if let Some(opportunity) = self.create_spot_opportunity(request, pattern)? {
                opportunities.push(opportunity);
            }
        }

        // Autoscaling opportunity
        if matches!(pattern, UsagePattern::Periodic | UsagePattern::Spiky) {
            if let Some(opportunity) = self.create_autoscaling_opportunity(request, temporal)? {
                opportunities.push(opportunity);
            }
        }

        // Schedule on/off opportunity
        if !temporal.low_periods.is_empty() {
            if let Some(opportunity) = self.create_schedule_opportunity(request, temporal)? {
                opportunities.push(opportunity);
            }
        }

        // Consolidation opportunity
        if statistics.avg_utilization < 20.0 && statistics.peak_utilization < 40.0 {
            if let Some(opportunity) = self.create_consolidation_opportunity(request, statistics)? {
                opportunities.push(opportunity);
            }
        }

        // Reserved instance opportunity
        if matches!(
            pattern,
            UsagePattern::ConstantHigh | UsagePattern::ConstantLow
        ) {
            if let Some(opportunity) =
                self.create_reserved_instance_opportunity(request, pattern)?
            {
                opportunities.push(opportunity);
            }
        }

        Ok(opportunities)
    }

    /// Create rightsizing opportunity
    pub fn create_rightsizing_opportunity(
        &self,
        request: &AnalysisRequest,
        statistics: &UsageStatistics,
    ) -> CostOptimizationResult<Option<OptimizationOpportunity>> {
        let current_cost = request.cost_per_hour.unwrap_or(1.0);
        let utilization_ratio = statistics.p95_utilization / 100.0;
        let recommended_size = utilization_ratio * 1.2; // 20% buffer

        if recommended_size >= 0.9 {
            return Ok(None); // Already properly sized
        }

        let new_cost = current_cost * recommended_size;
        let savings = current_cost - new_cost;
        let savings_percent = (savings / current_cost) * 100.0;

        Ok(Some(OptimizationOpportunity {
            id: Uuid::new_v4(),
            optimization_type: OptimizationType::Rightsize,
            title: "Rightsize overprovisioned resource".to_string(),
            description: format!(
                "Resource is only using {:.1}% capacity at 95th percentile. Consider downsizing.",
                statistics.p95_utilization
            ),
            estimated_savings: savings * 24.0 * 30.0, // Monthly savings
            savings_percent,
            effort: ImplementationEffort::Low,
            risk: RiskLevel::Low,
            recommendation: RecommendationDetails {
                current_config: HashMap::from([
                    ("size".to_string(), "current".to_string()),
                    (
                        "utilization".to_string(),
                        format!("{:.1}%", statistics.avg_utilization),
                    ),
                ]),
                recommended_config: HashMap::from([
                    ("size".to_string(), "smaller".to_string()),
                    ("target_utilization".to_string(), "60-80%".to_string()),
                ]),
                steps: vec![
                    "1. Create snapshot/backup of current resource".to_string(),
                    "2. Provision new smaller resource".to_string(),
                    "3. Migrate workload to new resource".to_string(),
                    "4. Monitor performance for 24 hours".to_string(),
                    "5. Terminate old resource after validation".to_string(),
                ],
                prerequisites: vec![
                    "Maintenance window scheduled".to_string(),
                    "Backup completed".to_string(),
                ],
                rollback_plan: "Revert to original resource if performance degrades".to_string(),
            },
        }))
    }

    /// Create idle termination opportunity
    pub fn create_idle_termination_opportunity(
        &self,
        request: &AnalysisRequest,
        statistics: &UsageStatistics,
    ) -> CostOptimizationResult<Option<OptimizationOpportunity>> {
        let current_cost = request.cost_per_hour.unwrap_or(1.0);
        let monthly_cost = current_cost * 24.0 * 30.0;

        Ok(Some(OptimizationOpportunity {
            id: Uuid::new_v4(),
            optimization_type: OptimizationType::TerminateIdle,
            title: "Terminate idle resource".to_string(),
            description: format!(
                "Resource has been idle {:.1}% of the time. Consider termination.",
                statistics.idle_percentage
            ),
            estimated_savings: monthly_cost,
            savings_percent: 100.0,
            effort: ImplementationEffort::Minimal,
            risk: RiskLevel::Low,
            recommendation: RecommendationDetails {
                current_config: HashMap::from([
                    ("status".to_string(), "running".to_string()),
                    (
                        "idle_percentage".to_string(),
                        format!("{:.1}%", statistics.idle_percentage),
                    ),
                ]),
                recommended_config: HashMap::from([(
                    "status".to_string(),
                    "terminated".to_string(),
                )]),
                steps: vec![
                    "1. Verify resource is not needed".to_string(),
                    "2. Create final backup if needed".to_string(),
                    "3. Terminate resource".to_string(),
                ],
                prerequisites: vec!["Confirm with resource owner".to_string()],
                rollback_plan: "Restore from backup if needed later".to_string(),
            },
        }))
    }

    /// Create spot instance opportunity
    pub fn create_spot_opportunity(
        &self,
        request: &AnalysisRequest,
        pattern: &UsagePattern,
    ) -> CostOptimizationResult<Option<OptimizationOpportunity>> {
        let current_cost = request.cost_per_hour.unwrap_or(1.0);
        let spot_discount = 0.7; // Assume 70% discount
        let new_cost = current_cost * (1.0 - spot_discount);
        let savings = current_cost - new_cost;
        let monthly_savings = savings * 24.0 * 30.0;

        Ok(Some(OptimizationOpportunity {
            id: Uuid::new_v4(),
            optimization_type: OptimizationType::ConvertToSpot,
            title: "Convert to spot instances".to_string(),
            description: format!(
                "Usage pattern ({}) is suitable for spot instances with potential 70% savings.",
                pattern
            ),
            estimated_savings: monthly_savings,
            savings_percent: spot_discount * 100.0,
            effort: ImplementationEffort::Medium,
            risk: RiskLevel::Medium,
            recommendation: RecommendationDetails {
                current_config: HashMap::from([
                    ("instance_type".to_string(), "on-demand".to_string()),
                    ("cost_per_hour".to_string(), format!("${:.2}", current_cost)),
                ]),
                recommended_config: HashMap::from([
                    ("instance_type".to_string(), "spot".to_string()),
                    ("cost_per_hour".to_string(), format!("${:.2}", new_cost)),
                ]),
                steps: vec![
                    "1. Implement checkpointing for workload".to_string(),
                    "2. Create spot instance request".to_string(),
                    "3. Configure interruption handling".to_string(),
                    "4. Test failover mechanisms".to_string(),
                ],
                prerequisites: vec![
                    "Workload is interruption-tolerant".to_string(),
                    "Checkpointing implemented".to_string(),
                ],
                rollback_plan: "Fallback to on-demand if interruptions exceed threshold"
                    .to_string(),
            },
        }))
    }

    /// Create autoscaling opportunity
    pub fn create_autoscaling_opportunity(
        &self,
        request: &AnalysisRequest,
        temporal: &TemporalAnalysis,
    ) -> CostOptimizationResult<Option<OptimizationOpportunity>> {
        let current_cost = request.cost_per_hour.unwrap_or(1.0);
        let low_usage_hours = temporal.low_periods.len() as f64;
        let savings_percent = (low_usage_hours / 24.0) * 0.5 * 100.0; // Assume 50% savings during low periods
        let monthly_savings = current_cost * 24.0 * 30.0 * (savings_percent / 100.0);

        Ok(Some(OptimizationOpportunity {
            id: Uuid::new_v4(),
            optimization_type: OptimizationType::EnableAutoscaling,
            title: "Enable autoscaling".to_string(),
            description: "Variable usage pattern detected. Autoscaling can optimize costs."
                .to_string(),
            estimated_savings: monthly_savings,
            savings_percent,
            effort: ImplementationEffort::Medium,
            risk: RiskLevel::Low,
            recommendation: RecommendationDetails {
                current_config: HashMap::from([("scaling".to_string(), "fixed".to_string())]),
                recommended_config: HashMap::from([
                    ("scaling".to_string(), "auto".to_string()),
                    ("min_capacity".to_string(), "20%".to_string()),
                    ("max_capacity".to_string(), "100%".to_string()),
                ]),
                steps: vec![
                    "1. Define scaling metrics and thresholds".to_string(),
                    "2. Configure autoscaling policies".to_string(),
                    "3. Test scaling behavior".to_string(),
                    "4. Monitor and adjust thresholds".to_string(),
                ],
                prerequisites: vec!["Application supports horizontal scaling".to_string()],
                rollback_plan: "Disable autoscaling and return to fixed capacity".to_string(),
            },
        }))
    }

    /// Create schedule on/off opportunity
    pub fn create_schedule_opportunity(
        &self,
        request: &AnalysisRequest,
        temporal: &TemporalAnalysis,
    ) -> CostOptimizationResult<Option<OptimizationOpportunity>> {
        let current_cost = request.cost_per_hour.unwrap_or(1.0);
        let low_hours: f64 = temporal
            .low_periods
            .iter()
            .filter(|p| p.can_shutdown)
            .map(|p| p.end.signed_duration_since(p.start).num_hours() as f64)
            .sum();

        let daily_low_hours = low_hours / 7.0; // Average per day
        let savings_percent = (daily_low_hours / 24.0) * 100.0;
        let monthly_savings = current_cost * daily_low_hours * 30.0;

        if monthly_savings < 10.0 {
            return Ok(None); // Too small to be worthwhile
        }

        Ok(Some(OptimizationOpportunity {
            id: Uuid::new_v4(),
            optimization_type: OptimizationType::ScheduleOnOff,
            title: "Schedule resource on/off times".to_string(),
            description: format!(
                "Resource is idle for {:.1} hours per day on average. Schedule shutdown during off-hours.",
                daily_low_hours
            ),
            estimated_savings: monthly_savings,
            savings_percent,
            effort: ImplementationEffort::Low,
            risk: RiskLevel::Low,
            recommendation: RecommendationDetails {
                current_config: HashMap::from([
                    ("schedule".to_string(), "always-on".to_string()),
                ]),
                recommended_config: HashMap::from([
                    ("schedule".to_string(), "business-hours".to_string()),
                    ("on_time".to_string(), "8:00 AM".to_string()),
                    ("off_time".to_string(), "6:00 PM".to_string()),
                ]),
                steps: vec![
                    "1. Implement start/stop automation".to_string(),
                    "2. Configure schedule based on usage patterns".to_string(),
                    "3. Set up notifications for schedule changes".to_string(),
                    "4. Create override mechanism for exceptions".to_string(),
                ],
                prerequisites: vec![
                    "Resource supports start/stop without data loss".to_string(),
                ],
                rollback_plan: "Disable scheduling and keep resource always-on".to_string(),
            },
        }))
    }

    /// Create consolidation opportunity
    pub fn create_consolidation_opportunity(
        &self,
        request: &AnalysisRequest,
        statistics: &UsageStatistics,
    ) -> CostOptimizationResult<Option<OptimizationOpportunity>> {
        let current_cost = request.cost_per_hour.unwrap_or(1.0);
        let consolidation_ratio = 0.5; // Assume can consolidate to 50% of current resources
        let savings = current_cost * consolidation_ratio;
        let monthly_savings = savings * 24.0 * 30.0;

        Ok(Some(OptimizationOpportunity {
            id: Uuid::new_v4(),
            optimization_type: OptimizationType::Consolidate,
            title: "Consolidate underutilized resources".to_string(),
            description: format!(
                "Resource utilization is low ({:.1}% average). Consider consolidating with other resources.",
                statistics.avg_utilization
            ),
            estimated_savings: monthly_savings,
            savings_percent: consolidation_ratio * 100.0,
            effort: ImplementationEffort::High,
            risk: RiskLevel::Medium,
            recommendation: RecommendationDetails {
                current_config: HashMap::from([
                    ("deployment".to_string(), "dedicated".to_string()),
                    ("utilization".to_string(), format!("{:.1}%", statistics.avg_utilization)),
                ]),
                recommended_config: HashMap::from([
                    ("deployment".to_string(), "shared".to_string()),
                    ("target_utilization".to_string(), "60-80%".to_string()),
                ]),
                steps: vec![
                    "1. Identify other underutilized resources".to_string(),
                    "2. Plan consolidation architecture".to_string(),
                    "3. Implement resource sharing/containerization".to_string(),
                    "4. Migrate workloads gradually".to_string(),
                    "5. Monitor performance and adjust".to_string(),
                ],
                prerequisites: vec![
                    "Compatible workloads identified".to_string(),
                    "Container orchestration available".to_string(),
                ],
                rollback_plan: "Separate workloads back to dedicated resources".to_string(),
            },
        }))
    }

    /// Create reserved instance opportunity
    pub fn create_reserved_instance_opportunity(
        &self,
        request: &AnalysisRequest,
        pattern: &UsagePattern,
    ) -> CostOptimizationResult<Option<OptimizationOpportunity>> {
        let current_cost = request.cost_per_hour.unwrap_or(1.0);
        let reserved_discount = match pattern {
            UsagePattern::ConstantHigh => 0.4, // 40% discount for consistent high usage
            UsagePattern::ConstantLow => 0.3,  // 30% discount for consistent low usage
            _ => return Ok(None),              // Not suitable for other patterns
        };

        let new_cost = current_cost * (1.0 - reserved_discount);
        let savings = current_cost - new_cost;
        let monthly_savings = savings * 24.0 * 30.0;

        Ok(Some(OptimizationOpportunity {
            id: Uuid::new_v4(),
            optimization_type: OptimizationType::UseReserved,
            title: "Convert to reserved instances".to_string(),
            description: format!(
                "Consistent usage pattern ({}) makes this suitable for reserved instances with {:.0}% savings.",
                pattern, reserved_discount * 100.0
            ),
            estimated_savings: monthly_savings * 12.0, // Annual savings
            savings_percent: reserved_discount * 100.0,
            effort: ImplementationEffort::Low,
            risk: RiskLevel::None,
            recommendation: RecommendationDetails {
                current_config: HashMap::from([
                    ("pricing".to_string(), "on-demand".to_string()),
                    ("commitment".to_string(), "none".to_string()),
                ]),
                recommended_config: HashMap::from([
                    ("pricing".to_string(), "reserved".to_string()),
                    ("commitment".to_string(), "1-year".to_string()),
                    ("payment".to_string(), "partial-upfront".to_string()),
                ]),
                steps: vec![
                    "1. Analyze long-term usage commitment".to_string(),
                    "2. Purchase reserved instances".to_string(),
                    "3. Apply reservations to existing resources".to_string(),
                    "4. Monitor utilization of reservations".to_string(),
                ],
                prerequisites: vec![
                    "Long-term usage predictability confirmed".to_string(),
                    "Budget approved for upfront costs".to_string(),
                ],
                rollback_plan: "Sell unused reservations on marketplace".to_string(),
            },
        }))
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
            period: Duration::from_secs(86400 * 7),
            include_recommendations: true,
            confidence_threshold: 0.8,
            cost_per_hour: Some(10.0),
        }
    }

    fn create_test_statistics(avg_util: f64, p95_util: f64, idle_pct: f64) -> UsageStatistics {
        UsageStatistics {
            avg_utilization: avg_util,
            peak_utilization: p95_util + 10.0,
            min_utilization: 0.0,
            std_deviation: 15.0,
            p95_utilization: p95_util,
            p99_utilization: p95_util + 5.0,
            idle_percentage: idle_pct,
            sample_count: 1000,
        }
    }

    fn create_test_temporal() -> TemporalAnalysis {
        TemporalAnalysis {
            hourly_pattern: vec![], // Simplified for testing
            daily_pattern: vec![],
            peak_times: vec![],
            low_periods: vec![LowPeriod {
                start: chrono::Utc::now() - chrono::Duration::hours(8),
                end: chrono::Utc::now(),
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
    fn test_rightsizing_opportunity() {
        let config = UsageAnalyzerConfig::default();
        let generator = OpportunityGenerator::new(config);

        let request = create_test_request();
        let statistics = create_test_statistics(20.0, 25.0, 10.0); // Low utilization

        let opportunity = generator.create_rightsizing_opportunity(&request, &statistics).unwrap().unwrap();

        assert_eq!(opportunity.optimization_type, OptimizationType::Rightsize);
        assert!(opportunity.estimated_savings > 0.0);
        assert!(opportunity.savings_percent > 0.0);
        assert_eq!(opportunity.effort, ImplementationEffort::Low);
        assert_eq!(opportunity.risk, RiskLevel::Low);
    }

    #[test]
    fn test_idle_termination_opportunity() {
        let config = UsageAnalyzerConfig::default();
        let generator = OpportunityGenerator::new(config);

        let request = create_test_request();
        let statistics = create_test_statistics(5.0, 10.0, 90.0); // Mostly idle

        let opportunity =
            generator.create_idle_termination_opportunity(&request, &statistics).unwrap().unwrap();

        assert_eq!(
            opportunity.optimization_type,
            OptimizationType::TerminateIdle
        );
        assert_eq!(opportunity.savings_percent, 100.0);
        assert_eq!(opportunity.effort, ImplementationEffort::Minimal);
    }

    #[test]
    fn test_spot_opportunity_for_spiky_pattern() {
        let config = UsageAnalyzerConfig::default();
        let generator = OpportunityGenerator::new(config);

        let request = create_test_request();
        let pattern = UsagePattern::Spiky;

        let opportunity = generator.create_spot_opportunity(&request, &pattern).unwrap().unwrap();

        assert_eq!(
            opportunity.optimization_type,
            OptimizationType::ConvertToSpot
        );
        assert_eq!(opportunity.savings_percent, 70.0);
        assert_eq!(opportunity.effort, ImplementationEffort::Medium);
        assert_eq!(opportunity.risk, RiskLevel::Medium);
    }

    #[test]
    fn test_autoscaling_opportunity() {
        let config = UsageAnalyzerConfig::default();
        let generator = OpportunityGenerator::new(config);

        let request = create_test_request();
        let temporal = create_test_temporal();

        let opportunity = generator.create_autoscaling_opportunity(&request, &temporal).unwrap().unwrap();

        assert_eq!(
            opportunity.optimization_type,
            OptimizationType::EnableAutoscaling
        );
        assert!(opportunity.estimated_savings > 0.0);
        assert_eq!(opportunity.effort, ImplementationEffort::Medium);
        assert_eq!(opportunity.risk, RiskLevel::Low);
    }

    #[test]
    fn test_schedule_opportunity() {
        let config = UsageAnalyzerConfig::default();
        let generator = OpportunityGenerator::new(config);

        let request = create_test_request();
        let temporal = create_test_temporal();

        let opportunity = generator.create_schedule_opportunity(&request, &temporal).unwrap().unwrap();

        assert_eq!(
            opportunity.optimization_type,
            OptimizationType::ScheduleOnOff
        );
        assert!(opportunity.estimated_savings > 0.0);
        assert_eq!(opportunity.effort, ImplementationEffort::Low);
        assert_eq!(opportunity.risk, RiskLevel::Low);
    }

    #[test]
    fn test_consolidation_opportunity() {
        let config = UsageAnalyzerConfig::default();
        let generator = OpportunityGenerator::new(config);

        let request = create_test_request();
        let statistics = create_test_statistics(15.0, 35.0, 20.0); // Low utilization, low peaks

        let opportunity = generator.create_consolidation_opportunity(&request, &statistics).unwrap().unwrap();

        assert_eq!(opportunity.optimization_type, OptimizationType::Consolidate);
        assert!(opportunity.estimated_savings > 0.0);
        assert_eq!(opportunity.effort, ImplementationEffort::High);
        assert_eq!(opportunity.risk, RiskLevel::Medium);
    }

    #[test]
    fn test_reserved_instance_opportunity() {
        let config = UsageAnalyzerConfig::default();
        let generator = OpportunityGenerator::new(config);

        let request = create_test_request();
        let pattern = UsagePattern::ConstantHigh;

        let opportunity = generator.create_reserved_instance_opportunity(&request, &pattern).unwrap().unwrap();

        assert_eq!(opportunity.optimization_type, OptimizationType::UseReserved);
        assert_eq!(opportunity.savings_percent, 40.0);
        assert_eq!(opportunity.effort, ImplementationEffort::Low);
        assert_eq!(opportunity.risk, RiskLevel::None);
    }

    #[test]
    fn test_no_reserved_instance_for_inappropriate_pattern() {
        let config = UsageAnalyzerConfig::default();
        let generator = OpportunityGenerator::new(config);

        let request = create_test_request();
        let pattern = UsagePattern::Spiky; // Not suitable for reserved instances

        let opportunity = generator.create_reserved_instance_opportunity(&request, &pattern).unwrap();

        assert!(opportunity.is_none());
    }

    #[test]
    fn test_identify_multiple_opportunities() {
        let config = UsageAnalyzerConfig::default();
        let generator = OpportunityGenerator::new(config);

        let request = create_test_request();
        let pattern = UsagePattern::Periodic;
        let statistics = create_test_statistics(25.0, 28.0, 30.0); // Low utilization, some idle
        let temporal = create_test_temporal();

        let opportunities = generator
            .identify_opportunities(&request, &pattern, &statistics, &temporal)
            .unwrap();

        // Should identify multiple opportunities
        assert!(opportunities.len() > 1);

        // Check for specific opportunity types
        let opt_types: Vec<OptimizationType> =
            opportunities.iter().map(|o| o.optimization_type).collect();

        assert!(opt_types.contains(&OptimizationType::Rightsize));
        assert!(opt_types.contains(&OptimizationType::ConvertToSpot));
        assert!(opt_types.contains(&OptimizationType::EnableAutoscaling));
    }
}
