//! Cost Optimizer
//!
//! Intelligent cost optimization strategies for cloud resource provisioning.

use super::core::*;
use anyhow::Result;
use std::collections::HashMap;

/// Cost optimization strategies
#[derive(Debug, Clone)]
pub enum CostStrategy {
    /// Always use spot/preemptible instances when available
    SpotFirst {
        fallback_to_on_demand: bool,
        max_interruption_rate: f64,
    },
    /// Balance between cost and reliability
    Balanced {
        spot_percentage: f64,
        on_demand_percentage: f64,
    },
    /// Minimize cost at all costs
    AggressiveSavings,
    /// Prioritize reliability over cost
    ReliabilityFirst,
    /// Custom strategy with provider preferences
    Custom(HashMap<String, ProviderStrategy>),
}

/// Provider-specific strategy
#[derive(Debug, Clone)]
pub struct ProviderStrategy {
    pub use_spot: bool,
    pub preferred_regions: Vec<String>,
    pub max_price_per_hour: Option<f64>,
    pub instance_preferences: Vec<String>,
}

/// Cost optimizer
pub struct CostOptimizer {
    historical_prices: HashMap<String, Vec<PricePoint>>,
    interruption_history: HashMap<String, Vec<InterruptionEvent>>,
}

/// Historical price point
#[derive(Debug, Clone)]
struct PricePoint {
    timestamp: std::time::Instant,
    provider: String,
    region: String,
    instance_type: String,
    spot_price: f64,
    on_demand_price: f64,
}

/// Interruption event tracking
#[derive(Debug, Clone)]
struct InterruptionEvent {
    timestamp: std::time::Instant,
    provider: String,
    region: String,
    instance_type: String,
    interruption_rate: f64,
}

/// Optimized instance selection
#[derive(Debug)]
pub struct OptimizedInstance {
    pub provider: String,
    pub region: String,
    pub instance_type: String,
    pub use_spot: bool,
    pub estimated_cost: f64,
    pub estimated_savings: f64,
    pub reliability_score: f64,
}

impl CostOptimizer {
    /// Create new cost optimizer
    pub fn new() -> Self {
        Self {
            historical_prices: HashMap::new(),
            interruption_history: HashMap::new(),
        }
    }

    /// Optimize instance selection based on strategy
    pub async fn optimize_instance_selection(
        &self,
        provider: &dyn CloudProvider,
        requirements: &InstanceType,
        strategy: &CostStrategy,
    ) -> Result<OptimizedInstance> {
        match strategy {
            CostStrategy::SpotFirst {
                fallback_to_on_demand,
                max_interruption_rate,
            } => {
                self.optimize_spot_first(
                    provider,
                    requirements,
                    *fallback_to_on_demand,
                    *max_interruption_rate,
                )
                .await
            }
            CostStrategy::Balanced {
                spot_percentage,
                on_demand_percentage,
            } => {
                self.optimize_balanced(
                    provider,
                    requirements,
                    *spot_percentage,
                    *on_demand_percentage,
                )
                .await
            }
            CostStrategy::AggressiveSavings => {
                self.optimize_aggressive_savings(provider, requirements)
                    .await
            }
            CostStrategy::ReliabilityFirst => {
                self.optimize_reliability_first(provider, requirements)
                    .await
            }
            CostStrategy::Custom(provider_strategies) => {
                self.optimize_custom(provider, requirements, provider_strategies)
                    .await
            }
        }
    }

    /// Spot-first optimization
    async fn optimize_spot_first(
        &self,
        provider: &dyn CloudProvider,
        requirements: &InstanceType,
        fallback_to_on_demand: bool,
        max_interruption_rate: f64,
    ) -> Result<OptimizedInstance> {
        let regions = provider.list_regions().await?;
        let mut best_option = None;
        let mut best_savings = 0.0;

        for region in &regions {
            if !region.gpu_available {
                continue;
            }

            let availability = provider
                .check_spot_availability(&region.id, &requirements.id)
                .await?;

            if availability.available
                && availability.interruption_rate <= max_interruption_rate
                && availability.savings_percentage > best_savings
            {
                best_savings = availability.savings_percentage;
                best_option = Some(OptimizedInstance {
                    provider: provider.name().to_string(),
                    region: region.id.clone(),
                    instance_type: requirements.id.clone(),
                    use_spot: true,
                    estimated_cost: availability.current_price,
                    estimated_savings: availability.savings_percentage / 100.0,
                    reliability_score: 1.0 - availability.interruption_rate,
                });
            }
        }

        // Fallback to on-demand if needed
        if best_option.is_none() && fallback_to_on_demand {
            let region = regions
                .iter()
                .filter(|r| r.gpu_available)
                .min_by(|a, b| {
                    a.latency_ms
                        .partial_cmp(&b.latency_ms)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap();

            best_option = Some(OptimizedInstance {
                provider: provider.name().to_string(),
                region: region.id.clone(),
                instance_type: requirements.id.clone(),
                use_spot: false,
                estimated_cost: requirements.price_per_hour,
                estimated_savings: 0.0,
                reliability_score: 0.99, // On-demand is highly reliable
            });
        }

        best_option.ok_or_else(|| anyhow::anyhow!("No suitable instance found"))
    }

    /// Balanced optimization
    async fn optimize_balanced(
        &self,
        provider: &dyn CloudProvider,
        requirements: &InstanceType,
        _spot_percentage: f64,
        _on_demand_percentage: f64,
    ) -> Result<OptimizedInstance> {
        // For balanced strategy, check spot availability but with stricter requirements
        let regions = provider.list_regions().await?;

        let mut candidates = Vec::new();

        for region in regions {
            if !region.gpu_available {
                continue;
            }

            let availability = provider
                .check_spot_availability(&region.id, &requirements.id)
                .await?;

            // Score based on cost savings and reliability
            let score = availability.savings_percentage * (1.0 - availability.interruption_rate);

            candidates.push((region, availability, score));
        }

        // Pick the best scoring option
        let best = candidates.into_iter().max_by(|a, b| {
            a.2.partial_cmp(&b.2)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        if let Some((region, availability, _)) = best {
            Ok(OptimizedInstance {
                provider: provider.name().to_string(),
                region: region.id,
                instance_type: requirements.id.clone(),
                use_spot: availability.savings_percentage > 30.0, // Use spot if >30% savings
                estimated_cost: if availability.savings_percentage > 30.0 {
                    availability.current_price
                } else {
                    requirements.price_per_hour
                },
                estimated_savings: if availability.savings_percentage > 30.0 {
                    availability.savings_percentage / 100.0
                } else {
                    0.0
                },
                reliability_score: 1.0 - availability.interruption_rate * 0.5, // Weighted reliability
            })
        } else {
            Err(anyhow::anyhow!("No suitable balanced option found"))
        }
    }

    /// Aggressive savings optimization
    async fn optimize_aggressive_savings(
        &self,
        provider: &dyn CloudProvider,
        requirements: &InstanceType,
    ) -> Result<OptimizedInstance> {
        // Always choose cheapest option regardless of reliability
        let regions = provider.list_regions().await?;
        let mut cheapest_option = None;
        let mut lowest_price = f64::MAX;

        for region in regions {
            if !region.gpu_available {
                continue;
            }

            let availability = provider
                .check_spot_availability(&region.id, &requirements.id)
                .await?;

            if availability.available && availability.current_price < lowest_price {
                lowest_price = availability.current_price;
                cheapest_option = Some(OptimizedInstance {
                    provider: provider.name().to_string(),
                    region: region.id,
                    instance_type: requirements.id.clone(),
                    use_spot: true,
                    estimated_cost: availability.current_price,
                    estimated_savings: availability.savings_percentage / 100.0,
                    reliability_score: 1.0 - availability.interruption_rate,
                });
            }
        }

        cheapest_option.ok_or_else(|| anyhow::anyhow!("No cheap option found"))
    }

    /// Reliability-first optimization
    async fn optimize_reliability_first(
        &self,
        provider: &dyn CloudProvider,
        requirements: &InstanceType,
    ) -> Result<OptimizedInstance> {
        // Always use on-demand in the most reliable region
        let regions = provider.list_regions().await?;

        // Pick lowest latency region (usually most reliable)
        let best_region = regions
            .into_iter()
            .filter(|r| r.gpu_available)
            .min_by(|a, b| {
                a.latency_ms
                    .partial_cmp(&b.latency_ms)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .ok_or_else(|| anyhow::anyhow!("No GPU regions available"))?;

        Ok(OptimizedInstance {
            provider: provider.name().to_string(),
            region: best_region.id,
            instance_type: requirements.id.clone(),
            use_spot: false, // Never use spot for reliability
            estimated_cost: requirements.price_per_hour,
            estimated_savings: 0.0,
            reliability_score: 0.99,
        })
    }

    /// Custom optimization
    async fn optimize_custom(
        &self,
        provider: &dyn CloudProvider,
        requirements: &InstanceType,
        provider_strategies: &HashMap<String, ProviderStrategy>,
    ) -> Result<OptimizedInstance> {
        let strategy = provider_strategies
            .get(provider.name())
            .ok_or_else(|| anyhow::anyhow!("No strategy for provider {}", provider.name()))?;

        // Apply custom strategy
        let regions = if strategy.preferred_regions.is_empty() {
            provider.list_regions().await?
        } else {
            let all_regions = provider.list_regions().await?;
            all_regions
                .into_iter()
                .filter(|r| strategy.preferred_regions.contains(&r.id))
                .collect()
        };

        let mut best_option = None;
        let mut best_score = 0.0;

        for region in regions {
            if !region.gpu_available {
                continue;
            }

            if strategy.use_spot {
                let availability = provider
                    .check_spot_availability(&region.id, &requirements.id)
                    .await?;

                if let Some(max_price) = strategy.max_price_per_hour {
                    if availability.current_price > max_price {
                        continue;
                    }
                }

                let score = availability.savings_percentage;
                if score > best_score {
                    best_score = score;
                    best_option = Some(OptimizedInstance {
                        provider: provider.name().to_string(),
                        region: region.id,
                        instance_type: requirements.id.clone(),
                        use_spot: true,
                        estimated_cost: availability.current_price,
                        estimated_savings: availability.savings_percentage / 100.0,
                        reliability_score: 1.0 - availability.interruption_rate,
                    });
                }
            } else {
                // On-demand option
                if let Some(max_price) = strategy.max_price_per_hour {
                    if requirements.price_per_hour > max_price {
                        continue;
                    }
                }

                best_option = Some(OptimizedInstance {
                    provider: provider.name().to_string(),
                    region: region.id,
                    instance_type: requirements.id.clone(),
                    use_spot: false,
                    estimated_cost: requirements.price_per_hour,
                    estimated_savings: 0.0,
                    reliability_score: 0.99,
                });
                break; // First valid on-demand option
            }
        }

        best_option.ok_or_else(|| anyhow::anyhow!("No option matching custom strategy"))
    }

    /// Analyze cost trends
    pub fn analyze_cost_trends(&self, provider: &str, instance_type: &str) -> CostTrend {
        let history = self
            .historical_prices
            .get(&format!("{}-{}", provider, instance_type));

        if let Some(prices) = history {
            let recent_prices: Vec<f64> =
                prices.iter().rev().take(10).map(|p| p.spot_price).collect();

            if recent_prices.len() >= 2 {
                let avg_recent = recent_prices.iter().sum::<f64>() / recent_prices.len() as f64;
                let first = match recent_prices.last() {
                    Some(v) => v,
                    None => {
                        return CostTrend {
                            direction: TrendDirection::Unknown,
                            percentage_change: 0.0,
                            recommendation: "Insufficient data for trend analysis".to_string(),
                        }
                    }
                };
                let trend = (avg_recent - first) / first;

                return CostTrend {
                    direction: if trend > 0.05 {
                        TrendDirection::Increasing
                    } else if trend < -0.05 {
                        TrendDirection::Decreasing
                    } else {
                        TrendDirection::Stable
                    },
                    percentage_change: trend * 100.0,
                    recommendation: if trend > 0.1 {
                        "Consider locking in current prices or switching providers".to_string()
                    } else if trend < -0.1 {
                        "Good time to provision more resources".to_string()
                    } else {
                        "Prices are stable".to_string()
                    },
                };
            }
        }

        CostTrend {
            direction: TrendDirection::Unknown,
            percentage_change: 0.0,
            recommendation: "Insufficient data for trend analysis".to_string(),
        }
    }

    /// Get cost optimization recommendations
    pub fn get_recommendations(&self, current_spend: f64) -> Vec<CostRecommendation> {
        let mut recommendations = Vec::new();

        // Basic recommendations based on spend
        if current_spend > 10000.0 {
            recommendations.push(CostRecommendation {
                priority: Priority::High,
                category: "Reserved Instances".to_string(),
                description: "Consider reserved instances for 1-3 year commitments".to_string(),
                estimated_savings: current_spend * 0.3, // 30% typical savings
            });
        }

        if current_spend > 5000.0 {
            recommendations.push(CostRecommendation {
                priority: Priority::Medium,
                category: "Spot Instances".to_string(),
                description: "Increase spot instance usage for fault-tolerant workloads"
                    .to_string(),
                estimated_savings: current_spend * 0.5, // 50% typical savings
            });
        }

        recommendations.push(CostRecommendation {
            priority: Priority::Low,
            category: "Right-sizing".to_string(),
            description: "Review instance sizes and downsize underutilized resources".to_string(),
            estimated_savings: current_spend * 0.15, // 15% typical savings
        });

        recommendations
    }
}

/// Cost trend analysis
#[derive(Debug)]
pub struct CostTrend {
    pub direction: TrendDirection,
    pub percentage_change: f64,
    pub recommendation: String,
}

/// Trend direction
#[derive(Debug, PartialEq)]
pub enum TrendDirection {
    Increasing,
    Decreasing,
    Stable,
    Unknown,
}

/// Cost optimization recommendation
#[derive(Debug)]
pub struct CostRecommendation {
    pub priority: Priority,
    pub category: String,
    pub description: String,
    pub estimated_savings: f64,
}

/// Recommendation priority
#[derive(Debug, PartialEq, Ord, PartialOrd, Eq)]
pub enum Priority {
    High,
    Medium,
    Low,
}

impl Default for CostOptimizer {
    fn default() -> Self {
        Self::new()
    }
}
