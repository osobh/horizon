//! Budget forecasting and risk assessment functionality
//!
//! This module provides comprehensive budget forecasting capabilities including
//! risk assessment, cost optimization recommendations, and scenario analysis.

use crate::error::{CostOptimizationError, CostOptimizationResult};
use super::types::*;
use chrono::{DateTime, Duration as ChronoDuration, Utc};
use std::collections::HashMap;
use std::time::Duration;
use tracing::{info, warn};
use uuid::Uuid;

/// Budget forecaster and risk assessor
pub struct BudgetForecaster {
    /// Configuration
    config: Arc<CostPredictorConfig>,
}

impl BudgetForecaster {
    /// Create a new budget forecaster
    pub fn new(config: Arc<CostPredictorConfig>) -> Self {
        Self { config }
    }

    /// Generate budget forecast for a time period
    pub async fn forecast_budget<F>(
        &self,
        period_start: DateTime<Utc>,
        period_end: DateTime<Utc>,
        prediction_fn: F,
    ) -> CostOptimizationResult<BudgetForecast>
    where
        F: Fn(PredictionRequest) -> std::pin::Pin<Box<dyn std::future::Future<Output = CostOptimizationResult<PredictionResult>> + Send>>,
    {
        info!(
            "Generating budget forecast for {} to {}",
            period_start, period_end
        );

        let mut cost_breakdown = HashMap::new();
        let mut total_predicted = 0.0;

        // Predict costs for each metric type
        for metric_type in &[
            CostMetricType::ComputeCost,
            CostMetricType::StorageCost,
            CostMetricType::NetworkCost,
            CostMetricType::GpuCost,
            CostMetricType::OtherCost,
        ] {
            let duration = (period_end - period_start).to_std().map_err(|e| {
                CostOptimizationError::CalculationError {
                    details: format!("Duration calculation error: {}", e),
                }
            })?;

            let request = PredictionRequest {
                metric_type: *metric_type,
                horizon: duration,
                model: self.config.default_model,
                confidence_level: 0.95,
                seasonality: Seasonality::Monthly,
                filters: HashMap::new(),
            };

            match prediction_fn(request).await {
                Ok(result) => {
                    let predicted_sum: f64 = result.predictions.iter().map(|p| p.value).sum();
                    cost_breakdown.insert(*metric_type, predicted_sum);
                    total_predicted += predicted_sum;
                }
                Err(e) => {
                    warn!("Failed to predict {} costs: {}", metric_type, e);
                    cost_breakdown.insert(*metric_type, 0.0);
                }
            }
        }

        // Assess risk
        let risk_assessment = self.assess_budget_risk(total_predicted, &cost_breakdown)?;

        // Generate recommendations
        let recommendations = self.generate_recommendations(total_predicted, &cost_breakdown)?;

        let forecast = BudgetForecast {
            id: Uuid::new_v4(),
            created_at: Utc::now(),
            period_start,
            period_end,
            predicted_cost: total_predicted,
            cost_breakdown,
            risk_assessment,
            recommendations,
        };

        Ok(forecast)
    }

    /// Assess budget risk factors
    pub fn assess_budget_risk(
        &self,
        total_predicted: f64,
        breakdown: &HashMap<CostMetricType, f64>,
    ) -> CostOptimizationResult<RiskAssessment> {
        let mut risk_factors = Vec::new();

        // Check for high GPU costs
        if let Some(gpu_cost) = breakdown.get(&CostMetricType::GpuCost) {
            if gpu_cost / total_predicted > 0.5 {
                risk_factors.push(RiskFactor {
                    name: "High GPU dependency".to_string(),
                    impact: 80.0,
                    likelihood: 0.7,
                    mitigation: "Consider spot instances or reserved capacity".to_string(),
                });
            }
        }

        // Check for rapid growth
        if total_predicted > 10000.0 {
            risk_factors.push(RiskFactor {
                name: "High monthly spend".to_string(),
                impact: 90.0,
                likelihood: 0.9,
                mitigation: "Implement cost controls and alerts".to_string(),
            });
        }

        // Check for compute cost concentration
        if let Some(compute_cost) = breakdown.get(&CostMetricType::ComputeCost) {
            if compute_cost / total_predicted > 0.6 {
                risk_factors.push(RiskFactor {
                    name: "High compute cost concentration".to_string(),
                    impact: 70.0,
                    likelihood: 0.6,
                    mitigation: "Diversify workloads and optimize instance types".to_string(),
                });
            }
        }

        // Check for storage cost growth
        if let Some(storage_cost) = breakdown.get(&CostMetricType::StorageCost) {
            if *storage_cost > 5000.0 {
                risk_factors.push(RiskFactor {
                    name: "High storage costs".to_string(),
                    impact: 60.0,
                    likelihood: 0.8,
                    mitigation: "Implement data lifecycle policies and archiving".to_string(),
                });
            }
        }

        // Check for network cost spikes
        if let Some(network_cost) = breakdown.get(&CostMetricType::NetworkCost) {
            if network_cost / total_predicted > 0.3 {
                risk_factors.push(RiskFactor {
                    name: "High network transfer costs".to_string(),
                    impact: 65.0,
                    likelihood: 0.5,
                    mitigation: "Optimize data transfer patterns and use CDN".to_string(),
                });
            }
        }

        let risk_level = match risk_factors.len() {
            0 => RiskLevel::Low,
            1 => RiskLevel::Medium,
            2..=3 => RiskLevel::High,
            _ => RiskLevel::Critical,
        };

        Ok(RiskAssessment {
            risk_level,
            overrun_probability: (risk_factors.len() as f64 * 0.15).min(0.9),
            expected_variance: 0.15 + (risk_factors.len() as f64 * 0.05), // Base 15% + 5% per risk factor
            risk_factors,
        })
    }

    /// Generate cost optimization recommendations
    pub fn generate_recommendations(
        &self,
        total_predicted: f64,
        breakdown: &HashMap<CostMetricType, f64>,
    ) -> CostOptimizationResult<Vec<CostRecommendation>> {
        let mut recommendations = Vec::new();

        // Check compute costs
        if let Some(compute_cost) = breakdown.get(&CostMetricType::ComputeCost) {
            if compute_cost / total_predicted > 0.3 {
                recommendations.push(CostRecommendation {
                    id: Uuid::new_v4(),
                    title: "Optimize compute resources".to_string(),
                    description: "Consider using spot instances or rightsizing instances"
                        .to_string(),
                    potential_savings: compute_cost * 0.3,
                    complexity: ComplexityLevel::Medium,
                    priority: Priority::High,
                });
            }

            if *compute_cost > 2000.0 {
                recommendations.push(CostRecommendation {
                    id: Uuid::new_v4(),
                    title: "Implement auto-scaling".to_string(),
                    description: "Set up automatic scaling based on demand to avoid overprovisioning"
                        .to_string(),
                    potential_savings: compute_cost * 0.25,
                    complexity: ComplexityLevel::Medium,
                    priority: Priority::High,
                });
            }
        }

        // Check storage costs
        if let Some(storage_cost) = breakdown.get(&CostMetricType::StorageCost) {
            if *storage_cost > 1000.0 {
                recommendations.push(CostRecommendation {
                    id: Uuid::new_v4(),
                    title: "Implement storage lifecycle policies".to_string(),
                    description: "Move infrequently accessed data to cheaper storage tiers"
                        .to_string(),
                    potential_savings: storage_cost * 0.4,
                    complexity: ComplexityLevel::Low,
                    priority: Priority::Medium,
                });
            }

            if *storage_cost > 3000.0 {
                recommendations.push(CostRecommendation {
                    id: Uuid::new_v4(),
                    title: "Enable data compression and deduplication".to_string(),
                    description: "Reduce storage footprint through compression and deduplication"
                        .to_string(),
                    potential_savings: storage_cost * 0.2,
                    complexity: ComplexityLevel::Low,
                    priority: Priority::Medium,
                });
            }
        }

        // Check GPU costs
        if let Some(gpu_cost) = breakdown.get(&CostMetricType::GpuCost) {
            if *gpu_cost > 5000.0 {
                recommendations.push(CostRecommendation {
                    id: Uuid::new_v4(),
                    title: "Optimize GPU utilization".to_string(),
                    description: "Use GPU scheduling and sharing to maximize utilization"
                        .to_string(),
                    potential_savings: gpu_cost * 0.35,
                    complexity: ComplexityLevel::High,
                    priority: Priority::Critical,
                });
            }

            if gpu_cost / total_predicted > 0.4 {
                recommendations.push(CostRecommendation {
                    id: Uuid::new_v4(),
                    title: "Consider GPU spot instances".to_string(),
                    description: "Use spot instances for fault-tolerant GPU workloads"
                        .to_string(),
                    potential_savings: gpu_cost * 0.6,
                    complexity: ComplexityLevel::High,
                    priority: Priority::High,
                });
            }
        }

        // Check network costs
        if let Some(network_cost) = breakdown.get(&CostMetricType::NetworkCost) {
            if *network_cost > 1500.0 {
                recommendations.push(CostRecommendation {
                    id: Uuid::new_v4(),
                    title: "Optimize data transfer patterns".to_string(),
                    description: "Minimize cross-region transfers and use content delivery networks"
                        .to_string(),
                    potential_savings: network_cost * 0.3,
                    complexity: ComplexityLevel::Medium,
                    priority: Priority::Medium,
                });
            }
        }

        // General high-cost recommendations
        if total_predicted > 15000.0 {
            recommendations.push(CostRecommendation {
                id: Uuid::new_v4(),
                title: "Implement comprehensive cost monitoring".to_string(),
                description: "Set up detailed cost monitoring and alerting across all services"
                    .to_string(),
                potential_savings: total_predicted * 0.1,
                complexity: ComplexityLevel::Low,
                priority: Priority::Critical,
            });

            recommendations.push(CostRecommendation {
                id: Uuid::new_v4(),
                title: "Consider reserved capacity".to_string(),
                description: "Purchase reserved instances for predictable workloads"
                    .to_string(),
                potential_savings: total_predicted * 0.2,
                complexity: ComplexityLevel::Medium,
                priority: Priority::High,
            });
        }

        // Sort by priority and potential savings
        recommendations.sort_by(|a, b| {
            b.priority
                .cmp(&a.priority)
                .then_with(|| b.potential_savings.partial_cmp(&a.potential_savings).unwrap())
        });

        Ok(recommendations)
    }

    /// Generate scenario analysis
    pub fn generate_scenarios(
        &self,
        base_forecast: &BudgetForecast,
    ) -> CostOptimizationResult<Vec<BudgetScenario>> {
        let mut scenarios = Vec::new();

        // Optimistic scenario (20% reduction)
        let optimistic_cost = base_forecast.predicted_cost * 0.8;
        let optimistic_breakdown: HashMap<CostMetricType, f64> = base_forecast
            .cost_breakdown
            .iter()
            .map(|(k, v)| (*k, v * 0.8))
            .collect();

        scenarios.push(BudgetScenario {
            id: Uuid::new_v4(),
            name: "Optimistic".to_string(),
            description: "Aggressive cost optimization implemented".to_string(),
            probability: 0.2,
            predicted_cost: optimistic_cost,
            cost_breakdown: optimistic_breakdown,
            assumptions: vec![
                "All cost optimization recommendations implemented".to_string(),
                "Significant efficiency gains achieved".to_string(),
                "No unexpected cost spikes".to_string(),
            ],
        });

        // Pessimistic scenario (30% increase)
        let pessimistic_cost = base_forecast.predicted_cost * 1.3;
        let pessimistic_breakdown: HashMap<CostMetricType, f64> = base_forecast
            .cost_breakdown
            .iter()
            .map(|(k, v)| (*k, v * 1.3))
            .collect();

        scenarios.push(BudgetScenario {
            id: Uuid::new_v4(),
            name: "Pessimistic".to_string(),
            description: "Higher than expected demand and cost spikes".to_string(),
            probability: 0.3,
            predicted_cost: pessimistic_cost,
            cost_breakdown: pessimistic_breakdown,
            assumptions: vec![
                "Demand growth exceeds expectations".to_string(),
                "Price increases for cloud services".to_string(),
                "Inefficient resource utilization".to_string(),
            ],
        });

        // Most likely scenario (base forecast)
        scenarios.push(BudgetScenario {
            id: Uuid::new_v4(),
            name: "Most Likely".to_string(),
            description: "Current trends continue with moderate optimization".to_string(),
            probability: 0.5,
            predicted_cost: base_forecast.predicted_cost,
            cost_breakdown: base_forecast.cost_breakdown.clone(),
            assumptions: vec![
                "Current usage patterns continue".to_string(),
                "Moderate optimization efforts".to_string(),
                "No major infrastructure changes".to_string(),
            ],
        });

        Ok(scenarios)
    }

    /// Calculate cost variance analysis
    pub fn analyze_cost_variance(
        &self,
        actual_costs: &HashMap<CostMetricType, f64>,
        predicted_costs: &HashMap<CostMetricType, f64>,
    ) -> CostVarianceAnalysis {
        let mut variance_by_type = HashMap::new();
        let mut total_actual = 0.0;
        let mut total_predicted = 0.0;

        for (cost_type, actual) in actual_costs {
            total_actual += actual;
            
            if let Some(predicted) = predicted_costs.get(cost_type) {
                total_predicted += predicted;
                
                let variance = (actual - predicted) / predicted * 100.0;
                variance_by_type.insert(*cost_type, CostVariance {
                    actual: *actual,
                    predicted: *predicted,
                    variance_percent: variance,
                    variance_amount: actual - predicted,
                });
            }
        }

        let overall_variance = if total_predicted > 0.0 {
            (total_actual - total_predicted) / total_predicted * 100.0
        } else {
            0.0
        };

        CostVarianceAnalysis {
            total_actual,
            total_predicted,
            overall_variance_percent: overall_variance,
            overall_variance_amount: total_actual - total_predicted,
            variance_by_type,
            accuracy_score: (100.0 - overall_variance.abs()).max(0.0),
        }
    }
}

/// Budget scenario for analysis
#[derive(Debug, Clone)]
pub struct BudgetScenario {
    /// Scenario ID
    pub id: Uuid,
    /// Scenario name
    pub name: String,
    /// Description
    pub description: String,
    /// Probability of occurrence
    pub probability: f64,
    /// Predicted cost
    pub predicted_cost: f64,
    /// Cost breakdown
    pub cost_breakdown: HashMap<CostMetricType, f64>,
    /// Key assumptions
    pub assumptions: Vec<String>,
}

/// Cost variance analysis
#[derive(Debug, Clone)]
pub struct CostVarianceAnalysis {
    /// Total actual cost
    pub total_actual: f64,
    /// Total predicted cost
    pub total_predicted: f64,
    /// Overall variance percentage
    pub overall_variance_percent: f64,
    /// Overall variance amount
    pub overall_variance_amount: f64,
    /// Variance by cost type
    pub variance_by_type: HashMap<CostMetricType, CostVariance>,
    /// Prediction accuracy score (0-100)
    pub accuracy_score: f64,
}

/// Individual cost variance
#[derive(Debug, Clone)]
pub struct CostVariance {
    /// Actual cost
    pub actual: f64,
    /// Predicted cost
    pub predicted: f64,
    /// Variance percentage
    pub variance_percent: f64,
    /// Variance amount
    pub variance_amount: f64,
}

// Import std for better organization
use std::sync::Arc;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_budget_forecaster_creation() {
        let config = Arc::new(CostPredictorConfig::default());
        let forecaster = BudgetForecaster::new(config);
        // Test passes if no panic occurs
    }

    #[tokio::test]
    async fn test_budget_risk_assessment() {
        let config = Arc::new(CostPredictorConfig::default());
        let forecaster = BudgetForecaster::new(config);

        let mut breakdown = HashMap::new();
        breakdown.insert(CostMetricType::ComputeCost, 1000.0);
        breakdown.insert(CostMetricType::StorageCost, 500.0);
        breakdown.insert(CostMetricType::GpuCost, 2000.0);
        breakdown.insert(CostMetricType::NetworkCost, 300.0);

        let total_cost = 3800.0;
        let risk_assessment = forecaster.assess_budget_risk(total_cost, &breakdown).unwrap();

        // Should detect high GPU dependency
        assert!(!risk_assessment.risk_factors.is_empty());
        let gpu_risk = risk_assessment.risk_factors.iter().find(|f| f.name.contains("GPU"));
        assert!(gpu_risk.is_some());
        
        assert!(risk_assessment.overrun_probability > 0.0);
        assert!(risk_assessment.expected_variance > 0.0);
    }

    #[test]
    fn test_cost_optimization_recommendations() {
        let config = Arc::new(CostPredictorConfig::default());
        let forecaster = BudgetForecaster::new(config);

        let mut breakdown = HashMap::new();
        breakdown.insert(CostMetricType::ComputeCost, 5000.0);
        breakdown.insert(CostMetricType::StorageCost, 2000.0);
        breakdown.insert(CostMetricType::GpuCost, 8000.0);
        breakdown.insert(CostMetricType::NetworkCost, 1000.0);

        let total_cost = 16000.0;
        let recommendations = forecaster.generate_recommendations(total_cost, &breakdown).unwrap();

        assert!(!recommendations.is_empty());
        
        // Should have recommendations for compute, storage, and GPU
        let compute_rec = recommendations.iter().find(|r| r.title.contains("compute"));
        let storage_rec = recommendations.iter().find(|r| r.title.contains("storage"));
        let gpu_rec = recommendations.iter().find(|r| r.title.contains("GPU"));
        
        assert!(compute_rec.is_some());
        assert!(storage_rec.is_some());
        assert!(gpu_rec.is_some());

        // Recommendations should be sorted by priority
        for i in 1..recommendations.len() {
            assert!(recommendations[i-1].priority >= recommendations[i].priority);
        }
    }

    #[test]
    fn test_scenario_generation() {
        let config = Arc::new(CostPredictorConfig::default());
        let forecaster = BudgetForecaster::new(config);

        let mut breakdown = HashMap::new();
        breakdown.insert(CostMetricType::ComputeCost, 1000.0);
        breakdown.insert(CostMetricType::StorageCost, 500.0);

        let base_forecast = BudgetForecast {
            id: Uuid::new_v4(),
            created_at: Utc::now(),
            period_start: Utc::now(),
            period_end: Utc::now() + ChronoDuration::days(30),
            predicted_cost: 1500.0,
            cost_breakdown: breakdown,
            risk_assessment: RiskAssessment {
                risk_level: RiskLevel::Low,
                overrun_probability: 0.1,
                expected_variance: 0.15,
                risk_factors: Vec::new(),
            },
            recommendations: Vec::new(),
        };

        let scenarios = forecaster.generate_scenarios(&base_forecast).unwrap();
        assert_eq!(scenarios.len(), 3);

        let optimistic = scenarios.iter().find(|s| s.name == "Optimistic").unwrap();
        let pessimistic = scenarios.iter().find(|s| s.name == "Pessimistic").unwrap();
        let likely = scenarios.iter().find(|s| s.name == "Most Likely").unwrap();

        assert!(optimistic.predicted_cost < base_forecast.predicted_cost);
        assert!(pessimistic.predicted_cost > base_forecast.predicted_cost);
        assert_eq!(likely.predicted_cost, base_forecast.predicted_cost);

        // Probabilities should sum to 1.0
        let total_probability: f64 = scenarios.iter().map(|s| s.probability).sum();
        assert!((total_probability - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_cost_variance_analysis() {
        let config = Arc::new(CostPredictorConfig::default());
        let forecaster = BudgetForecaster::new(config);

        let mut actual_costs = HashMap::new();
        actual_costs.insert(CostMetricType::ComputeCost, 1100.0);
        actual_costs.insert(CostMetricType::StorageCost, 450.0);

        let mut predicted_costs = HashMap::new();
        predicted_costs.insert(CostMetricType::ComputeCost, 1000.0);
        predicted_costs.insert(CostMetricType::StorageCost, 500.0);

        let variance_analysis = forecaster.analyze_cost_variance(&actual_costs, &predicted_costs);

        assert_eq!(variance_analysis.total_actual, 1550.0);
        assert_eq!(variance_analysis.total_predicted, 1500.0);
        assert!(variance_analysis.overall_variance_percent > 0.0); // Over budget
        assert!(variance_analysis.accuracy_score < 100.0);

        // Check individual variances
        let compute_variance = variance_analysis.variance_by_type.get(&CostMetricType::ComputeCost).unwrap();
        assert_eq!(compute_variance.actual, 1100.0);
        assert_eq!(compute_variance.predicted, 1000.0);
        assert_eq!(compute_variance.variance_percent, 10.0);

        let storage_variance = variance_analysis.variance_by_type.get(&CostMetricType::StorageCost).unwrap();
        assert_eq!(storage_variance.variance_percent, -10.0); // Under budget
    }

    #[test]
    fn test_risk_level_classification() {
        let config = Arc::new(CostPredictorConfig::default());
        let forecaster = BudgetForecaster::new(config);

        // Test low risk (no risk factors)
        let mut breakdown = HashMap::new();
        breakdown.insert(CostMetricType::ComputeCost, 500.0);
        let low_risk = forecaster.assess_budget_risk(1000.0, &breakdown)?;
        assert_eq!(low_risk.risk_level, RiskLevel::Low);

        // Test high risk (multiple factors)
        breakdown.insert(CostMetricType::GpuCost, 8000.0); // High GPU dependency
        breakdown.insert(CostMetricType::NetworkCost, 3000.0); // High network costs
        let high_risk = forecaster.assess_budget_risk(12000.0, &breakdown).unwrap();
        assert!(matches!(high_risk.risk_level, RiskLevel::High | RiskLevel::Critical));
    }

    #[test]
    fn test_recommendation_prioritization() {
        let config = Arc::new(CostPredictorConfig::default());
        let forecaster = BudgetForecaster::new(config);

        let mut breakdown = HashMap::new();
        breakdown.insert(CostMetricType::ComputeCost, 5000.0);
        breakdown.insert(CostMetricType::GpuCost, 10000.0);
        breakdown.insert(CostMetricType::StorageCost, 3000.0);

        let recommendations = forecaster.generate_recommendations(18000.0, &breakdown)?;
        
        // GPU recommendations should be highest priority due to cost and impact
        let first_rec = &recommendations[0];
        assert!(first_rec.title.contains("GPU") || first_rec.priority == Priority::Critical);
        
        // Check that all recommendations have valid savings
        for rec in &recommendations {
            assert!(rec.potential_savings > 0.0);
            assert!(!rec.title.is_empty());
            assert!(!rec.description.is_empty());
        }
    }
}