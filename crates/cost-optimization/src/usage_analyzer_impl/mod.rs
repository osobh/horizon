//! Usage analysis module for resource patterns and optimization
//!
//! This module has been refactored using TDD methodology to split the original
//! 1671-line file into logical, maintainable modules under 750 lines each.
//!
//! ## Module Structure
//!
//! - `types`: Core data structures and enums
//! - `statistics`: Statistical analysis utilities
//! - `pattern_detector`: Pattern detection algorithms
//! - `temporal_analyzer`: Time-based usage analysis
//! - `opportunity_generator`: Optimization opportunity generation
//! - `waste_analyzer`: Waste detection and analysis
//!
//! ## Usage
//!
//! ```rust
//! use crate::usage_analyzer::{UsageAnalyzer, AnalysisRequest, UsageAnalyzerConfig};
//! use std::time::Duration;
//!
//! let config = UsageAnalyzerConfig::default();
//! let analyzer = UsageAnalyzer::new(config)?;
//!
//! let request = AnalysisRequest {
//!     resource_id: "my-resource".to_string(),
//!     resource_type: ResourceType::Cpu,
//!     period: Duration::from_secs(86400 * 7), // 7 days
//!     include_recommendations: true,
//!     confidence_threshold: 0.8,
//!     cost_per_hour: Some(1.0),
//! };
//!
//! let result = analyzer.analyze_usage(request, snapshots).await?;
//! ```

pub mod opportunity_generator;
pub mod pattern_detector;
pub mod statistics;
pub mod temporal_analyzer;
pub mod types;
pub mod waste_analyzer;

// Re-export all public types and functions for backward compatibility
pub use opportunity_generator::OpportunityGenerator;
pub use pattern_detector::PatternDetector;
pub use statistics::{percentile, OutlierStats, StatisticsCalculator, TemporalVariance};
pub use temporal_analyzer::TemporalAnalyzer;
pub use types::*;
pub use waste_analyzer::WasteAnalyzer;

use crate::error::CostOptimizationResult;
use crate::resource_tracker::ResourceSnapshot;
use chrono::Utc;
use dashmap::DashMap;
use parking_lot::RwLock;
use std::sync::Arc;
use tracing::{info, warn};

/// Main usage analyzer that coordinates all analysis components
pub struct UsageAnalyzer {
    /// Configuration
    config: Arc<UsageAnalyzerConfig>,
    /// Analysis result cache
    analysis_cache: Arc<DashMap<String, AnalysisResult>>,
    /// Pattern detection models cache
    pattern_models: Arc<DashMap<String, PatternModel>>,
    /// Performance and usage metrics
    metrics: Arc<RwLock<AnalyzerMetrics>>,
    /// Statistics calculator
    statistics_calculator: StatisticsCalculator,
    /// Pattern detector
    pattern_detector: PatternDetector,
    /// Temporal analyzer
    temporal_analyzer: TemporalAnalyzer,
    /// Opportunity generator
    opportunity_generator: OpportunityGenerator,
    /// Waste analyzer
    waste_analyzer: WasteAnalyzer,
}

impl UsageAnalyzer {
    /// Create a new usage analyzer with the given configuration
    pub fn new(config: UsageAnalyzerConfig) -> CostOptimizationResult<Self> {
        let config_arc = Arc::new(config.clone());

        Ok(Self {
            config: config_arc,
            analysis_cache: Arc::new(DashMap::new()),
            pattern_models: Arc::new(DashMap::new()),
            metrics: Arc::new(RwLock::new(AnalyzerMetrics::default())),
            statistics_calculator: StatisticsCalculator::new(config.clone()),
            pattern_detector: PatternDetector::new(config.clone()),
            temporal_analyzer: TemporalAnalyzer::new(config.clone()),
            opportunity_generator: OpportunityGenerator::new(config.clone()),
            waste_analyzer: WasteAnalyzer::new(config),
        })
    }

    /// Analyze resource usage patterns and generate recommendations
    pub async fn analyze_usage(
        &self,
        request: AnalysisRequest,
        snapshots: Vec<ResourceSnapshot>,
    ) -> CostOptimizationResult<AnalysisResult> {
        info!("Analyzing usage for resource: {}", request.resource_id);

        // Validate minimum data requirements
        if snapshots.len() < self.config.min_data_points {
            return Err(crate::error::CostOptimizationError::CalculationError {
                details: format!(
                    "Insufficient data points: {} < {}",
                    snapshots.len(),
                    self.config.min_data_points
                ),
            });
        }

        // Calculate usage statistics
        let statistics = self
            .statistics_calculator
            .calculate_statistics(&snapshots)?;

        // Detect usage pattern
        let (pattern, confidence) = self
            .pattern_detector
            .detect_pattern(&snapshots, &statistics)?;

        // Perform temporal analysis
        let temporal_analysis = self
            .temporal_analyzer
            .analyze_temporal_patterns(&snapshots)?;

        // Generate optimization opportunities if requested
        let opportunities = if request.include_recommendations {
            self.opportunity_generator.identify_opportunities(
                &request,
                &pattern,
                &statistics,
                &temporal_analysis,
            )?
        } else {
            Vec::new()
        };

        // Analyze waste patterns
        let waste_analysis = self.waste_analyzer.analyze_waste(
            &request,
            &statistics,
            &temporal_analysis,
            request.cost_per_hour.unwrap_or(0.0),
        )?;

        // Update performance metrics
        self.update_metrics(&pattern, &opportunities, &waste_analysis);

        // Store pattern model for future reference
        let pattern_model =
            self.pattern_detector
                .get_pattern_model(&request.resource_id, pattern, &snapshots);
        self.pattern_models
            .insert(request.resource_id.clone(), pattern_model);

        // Create final analysis result
        let result = AnalysisResult {
            request,
            pattern,
            confidence,
            statistics,
            temporal_analysis,
            opportunities,
            waste_analysis,
            generated_at: Utc::now(),
        };

        // Cache the result
        self.analysis_cache
            .insert(result.request.resource_id.clone(), result.clone());

        Ok(result)
    }

    /// Perform batch analysis on multiple resources
    pub async fn batch_analyze(
        &self,
        resources: Vec<(AnalysisRequest, Vec<ResourceSnapshot>)>,
    ) -> CostOptimizationResult<Vec<AnalysisResult>> {
        let mut results = Vec::new();

        for (request, snapshots) in resources {
            match self.analyze_usage(request, snapshots).await {
                Ok(result) => results.push(result),
                Err(e) => {
                    warn!("Failed to analyze resource: {}", e);
                    // Continue with other resources instead of failing the entire batch
                }
            }
        }

        Ok(results)
    }

    /// Get current analyzer performance metrics
    pub fn get_metrics(&self) -> AnalyzerMetrics {
        self.metrics.read().clone()
    }

    /// Clear analysis cache to free memory
    pub fn clear_cache(&self) {
        self.analysis_cache.clear();
        self.pattern_models.clear();
    }

    /// Get cached analysis result if available
    pub fn get_cached_result(&self, resource_id: &str) -> Option<AnalysisResult> {
        self.analysis_cache.get(resource_id).map(|r| r.clone())
    }

    /// Get stored pattern model for a resource
    pub fn get_pattern_model(&self, resource_id: &str) -> Option<PatternModel> {
        self.pattern_models.get(resource_id).map(|m| m.clone())
    }

    /// Update internal performance metrics
    fn update_metrics(
        &self,
        pattern: &UsagePattern,
        opportunities: &[OptimizationOpportunity],
        waste_analysis: &WasteAnalysis,
    ) {
        let mut metrics = self.metrics.write();

        metrics.total_analyses += 1;

        *metrics.patterns_detected.entry(*pattern).or_insert(0) += 1;

        metrics.opportunities_identified += opportunities.len() as u64;

        metrics.total_potential_savings += opportunities
            .iter()
            .map(|o| o.estimated_savings)
            .sum::<f64>();

        metrics.total_waste_detected += waste_analysis.total_waste_cost;
    }

    /// Generate a comprehensive analysis summary
    pub fn generate_summary(&self, result: &AnalysisResult) -> String {
        let mut summary = String::new();

        summary.push_str(&format!(
            "Usage Analysis Summary for {}\n",
            result.request.resource_id
        ));
        summary.push_str(&format!(
            "Pattern: {} (confidence: {:.1}%)\n",
            result.pattern,
            result.confidence * 100.0
        ));
        summary.push_str(&format!(
            "Average utilization: {:.1}%\n",
            result.statistics.avg_utilization
        ));
        summary.push_str(&format!(
            "Peak utilization: {:.1}%\n",
            result.statistics.peak_utilization
        ));
        summary.push_str(&format!(
            "Idle percentage: {:.1}%\n",
            result.statistics.idle_percentage
        ));

        if !result.opportunities.is_empty() {
            summary.push_str(&format!(
                "\nOptimization opportunities ({}):\n",
                result.opportunities.len()
            ));
            for (i, opp) in result.opportunities.iter().enumerate() {
                summary.push_str(&format!(
                    "{}. {} - ${:.2} savings\n",
                    i + 1,
                    opp.title,
                    opp.estimated_savings
                ));
            }
        }

        if result.waste_analysis.total_waste_cost > 0.0 {
            summary.push_str(&format!(
                "\nWaste detected: ${:.2} ({:?})\n",
                result.waste_analysis.total_waste_cost, result.waste_analysis.severity
            ));
        }

        summary
    }

    /// Validate configuration settings
    pub fn validate_config(&self) -> CostOptimizationResult<()> {
        if self.config.min_data_points == 0 {
            return Err(crate::error::CostOptimizationError::CalculationError {
                details: "min_data_points must be greater than 0".to_string(),
            });
        }

        if self.config.idle_threshold < 0.0 || self.config.idle_threshold > 100.0 {
            return Err(crate::error::CostOptimizationError::CalculationError {
                details: "idle_threshold must be between 0 and 100".to_string(),
            });
        }

        if self.config.overprovision_threshold < 0.0 || self.config.overprovision_threshold > 100.0
        {
            return Err(crate::error::CostOptimizationError::CalculationError {
                details: "overprovision_threshold must be between 0 and 100".to_string(),
            });
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resource_tracker::ResourceType;
    use std::collections::HashMap;
    use std::time::Duration;

    fn create_test_snapshots(count: usize, pattern: &str) -> Vec<ResourceSnapshot> {
        let mut snapshots = Vec::new();
        let base_time = Utc::now() - chrono::Duration::hours(count as i64);

        for i in 0..count {
            let utilization = match pattern {
                "constant_high" => 85.0 + (i as f64).sin() * 2.0,
                "constant_low" => 15.0 + (i as f64).sin() * 2.0,
                "spiky" => {
                    if i % 10 == 0 {
                        95.0
                    } else {
                        30.0
                    }
                }
                "idle" => {
                    if i % 50 == 0 {
                        5.0
                    } else {
                        1.0
                    }
                }
                _ => 50.0,
            };

            snapshots.push(ResourceSnapshot {
                timestamp: base_time + chrono::Duration::hours(i as i64),
                resource_type: ResourceType::Cpu,
                resource_id: "test-resource".to_string(),
                utilization: utilization.min(100.0).max(0.0),
                available: 100.0 - utilization.min(100.0).max(0.0),
                total: 100.0,
                metadata: HashMap::new(),
            });
        }

        snapshots
    }

    #[tokio::test]
    async fn test_usage_analyzer_creation() {
        let config = UsageAnalyzerConfig::default();
        let analyzer = UsageAnalyzer::new(config);
        assert!(analyzer.is_ok());
    }

    #[tokio::test]
    async fn test_full_analysis_workflow() {
        let config = UsageAnalyzerConfig::default();
        let analyzer = UsageAnalyzer::new(config).unwrap();

        let snapshots = create_test_snapshots(200, "constant_high");
        let request = AnalysisRequest {
            resource_id: "test-resource".to_string(),
            resource_type: ResourceType::Cpu,
            period: Duration::from_secs(86400 * 7),
            include_recommendations: true,
            confidence_threshold: 0.8,
            cost_per_hour: Some(1.0),
        };

        let result = analyzer.analyze_usage(request, snapshots).await.unwrap();

        // Verify all components worked
        assert_eq!(result.pattern, UsagePattern::ConstantHigh);
        assert!(result.confidence > 0.8);
        assert!(result.statistics.avg_utilization > 80.0);
        assert_eq!(result.temporal_analysis.hourly_pattern.len(), 24);
        assert_eq!(result.temporal_analysis.daily_pattern.len(), 7);
        // May or may not have opportunities depending on the specific pattern
        assert!(result.waste_analysis.total_waste_cost >= 0.0);
    }

    #[tokio::test]
    async fn test_batch_analysis() {
        let config = UsageAnalyzerConfig::default();
        let analyzer = UsageAnalyzer::new(config).unwrap();

        let batch = vec![
            (
                AnalysisRequest {
                    resource_id: "resource-1".to_string(),
                    resource_type: ResourceType::Cpu,
                    period: Duration::from_secs(86400),
                    include_recommendations: false,
                    confidence_threshold: 0.8,
                    cost_per_hour: Some(1.0),
                },
                create_test_snapshots(150, "constant_high"),
            ),
            (
                AnalysisRequest {
                    resource_id: "resource-2".to_string(),
                    resource_type: ResourceType::Gpu,
                    period: Duration::from_secs(86400),
                    include_recommendations: false,
                    confidence_threshold: 0.8,
                    cost_per_hour: Some(5.0),
                },
                create_test_snapshots(150, "spiky"),
            ),
        ];

        let results = analyzer.batch_analyze(batch).await.unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].pattern, UsagePattern::ConstantHigh);
        assert_eq!(results[1].pattern, UsagePattern::Spiky);
    }

    #[tokio::test]
    async fn test_caching_behavior() {
        let config = UsageAnalyzerConfig::default();
        let analyzer = UsageAnalyzer::new(config).unwrap();

        let snapshots = create_test_snapshots(200, "constant_high");
        let request = AnalysisRequest {
            resource_id: "cached-resource".to_string(),
            resource_type: ResourceType::Cpu,
            period: Duration::from_secs(86400 * 7),
            include_recommendations: false,
            confidence_threshold: 0.8,
            cost_per_hour: Some(1.0),
        };

        // First analysis
        let _result1 = analyzer.analyze_usage(request, snapshots).await.unwrap();

        // Check that result is cached
        let cached = analyzer.get_cached_result("cached-resource");
        assert!(cached.is_some());

        // Check that pattern model is stored
        let pattern_model = analyzer.get_pattern_model("cached-resource");
        assert!(pattern_model.is_some());
    }

    #[tokio::test]
    async fn test_metrics_tracking() {
        let config = UsageAnalyzerConfig::default();
        let analyzer = UsageAnalyzer::new(config).unwrap();

        // Initial metrics should be empty
        let initial_metrics = analyzer.get_metrics();
        assert_eq!(initial_metrics.total_analyses, 0);

        // Perform analysis
        let snapshots = create_test_snapshots(200, "constant_low");
        let request = AnalysisRequest {
            resource_id: "test-resource".to_string(),
            resource_type: ResourceType::Cpu,
            period: Duration::from_secs(86400 * 7),
            include_recommendations: true,
            confidence_threshold: 0.8,
            cost_per_hour: Some(10.0),
        };

        let _result = analyzer.analyze_usage(request, snapshots).await.unwrap();

        // Metrics should be updated
        let updated_metrics = analyzer.get_metrics();
        assert_eq!(updated_metrics.total_analyses, 1);
        assert!(!updated_metrics.patterns_detected.is_empty());
    }

    #[tokio::test]
    async fn test_insufficient_data_error() {
        let config = UsageAnalyzerConfig::default();
        let analyzer = UsageAnalyzer::new(config).unwrap();

        let few_snapshots = create_test_snapshots(50, "constant_high"); // Less than min required
        let request = AnalysisRequest {
            resource_id: "test-resource".to_string(),
            resource_type: ResourceType::Cpu,
            period: Duration::from_secs(86400 * 7),
            include_recommendations: true,
            confidence_threshold: 0.8,
            cost_per_hour: Some(1.0),
        };

        let result = analyzer.analyze_usage(request, few_snapshots).await;
        assert!(result.is_err());
    }

    #[test]
    fn test_config_validation() {
        let mut config = UsageAnalyzerConfig::default();
        let analyzer = UsageAnalyzer::new(config.clone()).unwrap();

        // Valid config should pass
        assert!(analyzer.validate_config().is_ok());

        // Invalid idle threshold
        config.idle_threshold = -1.0;
        let analyzer = UsageAnalyzer::new(config.clone()).unwrap();
        assert!(analyzer.validate_config().is_err());

        // Invalid overprovision threshold
        config.idle_threshold = 5.0;
        config.overprovision_threshold = 150.0;
        let analyzer = UsageAnalyzer::new(config.clone()).unwrap();
        assert!(analyzer.validate_config().is_err());

        // Invalid min data points
        config.overprovision_threshold = 30.0;
        config.min_data_points = 0;
        let analyzer = UsageAnalyzer::new(config).unwrap();
        assert!(analyzer.validate_config().is_err());
    }

    #[tokio::test]
    async fn test_summary_generation() {
        let config = UsageAnalyzerConfig::default();
        let analyzer = UsageAnalyzer::new(config).unwrap();

        let snapshots = create_test_snapshots(200, "idle");
        let request = AnalysisRequest {
            resource_id: "test-resource".to_string(),
            resource_type: ResourceType::Cpu,
            period: Duration::from_secs(86400 * 7),
            include_recommendations: true,
            confidence_threshold: 0.8,
            cost_per_hour: Some(5.0),
        };

        let result = analyzer.analyze_usage(request, snapshots).await.unwrap();
        let summary = analyzer.generate_summary(&result);

        assert!(summary.contains("test-resource"));
        assert!(summary.contains("Pattern:"));
        assert!(summary.contains("utilization:"));
    }

    #[test]
    fn test_cache_clearing() {
        let config = UsageAnalyzerConfig::default();
        let analyzer = UsageAnalyzer::new(config).unwrap();

        // Add some dummy data to cache
        analyzer.analysis_cache.insert(
            "test".to_string(),
            AnalysisResult {
                request: AnalysisRequest {
                    resource_id: "test".to_string(),
                    resource_type: ResourceType::Cpu,
                    period: Duration::from_secs(3600),
                    include_recommendations: false,
                    confidence_threshold: 0.8,
                    cost_per_hour: None,
                },
                pattern: UsagePattern::Idle,
                confidence: 0.9,
                statistics: UsageStatistics {
                    avg_utilization: 1.0,
                    peak_utilization: 2.0,
                    min_utilization: 0.0,
                    std_deviation: 0.5,
                    p95_utilization: 2.0,
                    p99_utilization: 2.0,
                    idle_percentage: 95.0,
                    sample_count: 100,
                },
                temporal_analysis: TemporalAnalysis {
                    hourly_pattern: vec![],
                    daily_pattern: vec![],
                    peak_times: vec![],
                    low_periods: vec![],
                    trend: UsageTrend {
                        direction: TrendDirection::Stable,
                        rate: 0.0,
                        confidence: 0.8,
                    },
                },
                opportunities: vec![],
                waste_analysis: WasteAnalysis {
                    total_waste_cost: 0.0,
                    waste_by_type: HashMap::new(),
                    severity: WasteSeverity::Minimal,
                    recovery_actions: vec![],
                },
                generated_at: Utc::now(),
            },
        );

        assert_eq!(analyzer.analysis_cache.len(), 1);

        analyzer.clear_cache();

        assert_eq!(analyzer.analysis_cache.len(), 0);
        assert_eq!(analyzer.pattern_models.len(), 0);
    }
}
