//! Cost-benefit analysis for prefetching decisions
//!
//! Implements sophisticated cost modeling and benefit estimation
//! to make optimal prefetching decisions.

use super::*;
use std::collections::HashMap;
use std::time::Duration;

/// Advanced cost-benefit analyzer
pub struct AdvancedCostBenefitAnalyzer {
    /// Tier transfer costs
    transfer_costs: TransferCostModel,
    /// Access pattern analyzer
    pattern_analyzer: AccessPatternAnalyzer,
    /// Resource usage tracker
    resource_tracker: ResourceUsageTracker,
    /// Historical performance data
    performance_history: PerformanceHistory,
    /// Configuration
    config: CostBenefitConfig,
}

/// Cost-benefit configuration
#[derive(Debug, Clone)]
pub struct CostBenefitConfig {
    /// Minimum benefit ratio for approval
    pub min_benefit_ratio: f64,
    /// Energy cost weight
    pub energy_weight: f64,
    /// Latency cost weight
    pub latency_weight: f64,
    /// Bandwidth cost weight
    pub bandwidth_weight: f64,
    /// Include opportunity cost
    pub include_opportunity_cost: bool,
    /// Prediction confidence threshold
    pub confidence_threshold: f64,
}

impl Default for CostBenefitConfig {
    fn default() -> Self {
        Self {
            min_benefit_ratio: 1.2,
            energy_weight: 0.3,
            latency_weight: 0.5,
            bandwidth_weight: 0.2,
            include_opportunity_cost: true,
            confidence_threshold: 0.6,
        }
    }
}

impl AdvancedCostBenefitAnalyzer {
    /// Create new analyzer
    pub fn new(config: CostBenefitConfig) -> Self {
        Self {
            transfer_costs: TransferCostModel::new(),
            pattern_analyzer: AccessPatternAnalyzer::new(),
            resource_tracker: ResourceUsageTracker::new(),
            performance_history: PerformanceHistory::new(),
            config,
        }
    }

    /// Analyze prefetch decision
    pub fn analyze_prefetch(&self, request: &PrefetchAnalysisRequest) -> PrefetchAnalysisResult {
        // Calculate costs
        let transfer_cost = self.calculate_transfer_cost(request);
        let resource_cost = self.calculate_resource_cost(request);
        let opportunity_cost = if self.config.include_opportunity_cost {
            self.calculate_opportunity_cost(request)
        } else {
            Cost::zero()
        };

        let total_cost = Cost::combine(
            vec![&transfer_cost, &resource_cost, &opportunity_cost],
            &self.config,
        );

        // Calculate benefits
        let access_benefit = self.calculate_access_benefit(request);
        let pattern_benefit = self.calculate_pattern_benefit(request);
        let tier_benefit = self.calculate_tier_benefit(request);

        let total_benefit =
            Benefit::combine(vec![&access_benefit, &pattern_benefit, &tier_benefit]);

        // Make decision
        let net_benefit = total_benefit.value - total_cost.value;
        let benefit_ratio = if total_cost.value > 0.0 {
            total_benefit.value / total_cost.value
        } else {
            f64::INFINITY
        };

        let approved = benefit_ratio >= self.config.min_benefit_ratio
            && total_benefit.confidence >= self.config.confidence_threshold;

        PrefetchAnalysisResult {
            approved,
            total_cost,
            total_benefit,
            net_benefit,
            benefit_ratio,
            recommendation: self.generate_recommendation(approved, benefit_ratio, &request),
        }
    }

    /// Calculate transfer cost
    fn calculate_transfer_cost(&self, request: &PrefetchAnalysisRequest) -> Cost {
        let transfer_metrics =
            self.transfer_costs
                .calculate(request.from_tier, request.to_tier, request.size_bytes);

        Cost {
            value: transfer_metrics.total_cost,
            latency: transfer_metrics.latency,
            energy: transfer_metrics.energy_joules,
            bandwidth: transfer_metrics.bandwidth_usage,
            breakdown: CostBreakdown {
                transfer: transfer_metrics.total_cost * 0.7,
                queueing: transfer_metrics.total_cost * 0.2,
                overhead: transfer_metrics.total_cost * 0.1,
            },
        }
    }

    /// Calculate resource cost
    fn calculate_resource_cost(&self, request: &PrefetchAnalysisRequest) -> Cost {
        let resource_usage = self.resource_tracker.estimate_usage(
            request.to_tier,
            request.size_bytes,
            request.duration_hint,
        );

        Cost {
            value: resource_usage.cost,
            latency: Duration::ZERO,
            energy: resource_usage.energy,
            bandwidth: resource_usage.bandwidth,
            breakdown: CostBreakdown {
                transfer: 0.0,
                queueing: resource_usage.cost * 0.3,
                overhead: resource_usage.cost * 0.7,
            },
        }
    }

    /// Calculate opportunity cost
    fn calculate_opportunity_cost(&self, request: &PrefetchAnalysisRequest) -> Cost {
        // Cost of displacing other data
        let displacement_cost =
            self.estimate_displacement_cost(request.to_tier, request.size_bytes);

        // Cost of delayed other prefetches
        let delay_cost = self.estimate_delay_cost(request.priority, request.size_bytes);

        Cost {
            value: displacement_cost + delay_cost,
            latency: Duration::from_micros((delay_cost * 1000.0) as u64),
            energy: 0.0,
            bandwidth: 0.0,
            breakdown: CostBreakdown {
                transfer: 0.0,
                queueing: delay_cost,
                overhead: displacement_cost,
            },
        }
    }

    /// Calculate access benefit
    fn calculate_access_benefit(&self, request: &PrefetchAnalysisRequest) -> Benefit {
        let hit_probability = self
            .pattern_analyzer
            .predict_hit_probability(&request.access_history, request.prediction_window);

        let speedup = self.calculate_speedup(request.from_tier, request.to_tier);

        let saved_time =
            Duration::from_micros((request.avg_access_latency.as_micros() as f64 * speedup) as u64);

        Benefit {
            value: hit_probability * speedup * request.access_history.access_count as f64,
            confidence: hit_probability,
            expected_hits: (hit_probability * request.predicted_accesses as f64) as u32,
            saved_latency: saved_time,
            breakdown: BenefitBreakdown {
                latency_reduction: saved_time.as_secs_f64() * 1000.0,
                throughput_improvement: speedup,
                energy_savings: self.calculate_energy_savings(request),
            },
        }
    }

    /// Calculate pattern-based benefit
    fn calculate_pattern_benefit(&self, request: &PrefetchAnalysisRequest) -> Benefit {
        let pattern_score = self
            .pattern_analyzer
            .analyze_pattern(&request.access_history, &request.pattern_hint);

        let pattern_multiplier = match request.pattern_hint {
            Some(AccessPattern::Sequential) => 1.5,
            Some(AccessPattern::Strided(_)) => 1.3,
            Some(AccessPattern::Temporal) => 1.2,
            _ => 1.0,
        };

        Benefit {
            value: pattern_score * pattern_multiplier * 10.0,
            confidence: pattern_score,
            expected_hits: (pattern_score * 10.0) as u32,
            saved_latency: Duration::from_millis((pattern_score * 100.0) as u64),
            breakdown: BenefitBreakdown {
                latency_reduction: pattern_score * 50.0,
                throughput_improvement: pattern_multiplier,
                energy_savings: 0.0,
            },
        }
    }

    /// Calculate tier optimization benefit
    fn calculate_tier_benefit(&self, request: &PrefetchAnalysisRequest) -> Benefit {
        let tier_efficiency =
            self.calculate_tier_efficiency(request.to_tier, &request.access_history);

        let memory_pressure_relief = self.calculate_memory_pressure_relief(
            request.from_tier,
            request.to_tier,
            request.size_bytes,
        );

        Benefit {
            value: tier_efficiency * 5.0 + memory_pressure_relief * 3.0,
            confidence: 0.8,
            expected_hits: 0,
            saved_latency: Duration::ZERO,
            breakdown: BenefitBreakdown {
                latency_reduction: 0.0,
                throughput_improvement: tier_efficiency,
                energy_savings: memory_pressure_relief * 2.0,
            },
        }
    }

    /// Calculate speedup from tier migration
    fn calculate_speedup(&self, from: MemoryTier, to: MemoryTier) -> f64 {
        let latencies = [
            10.0,    // GPU - 10us
            50.0,    // CPU - 50us
            100.0,   // NVMe - 100us
            1000.0,  // SSD - 1ms
            10000.0, // HDD - 10ms
        ];

        let from_latency = latencies[from as usize];
        let to_latency = latencies[to as usize];

        from_latency / to_latency
    }

    /// Estimate displacement cost
    fn estimate_displacement_cost(&self, tier: MemoryTier, size: usize) -> f64 {
        // Simple model based on tier value
        let tier_values = [10.0, 5.0, 2.0, 1.0, 0.5];
        let tier_value = tier_values[tier as usize];

        (size as f64 / 1_048_576.0) * tier_value * 0.1
    }

    /// Estimate delay cost
    fn estimate_delay_cost(&self, priority: PrefetchPriority, size: usize) -> f64 {
        let priority_factor = match priority {
            PrefetchPriority::Critical => 0.1,
            PrefetchPriority::High => 0.3,
            PrefetchPriority::Normal => 0.5,
            PrefetchPriority::Low => 1.0,
        };

        (size as f64 / 1_048_576.0) * priority_factor
    }

    /// Calculate energy savings
    fn calculate_energy_savings(&self, request: &PrefetchAnalysisRequest) -> f64 {
        let energy_per_access = [
            0.1,  // GPU - 0.1 mJ
            0.5,  // CPU - 0.5 mJ
            1.0,  // NVMe - 1 mJ
            5.0,  // SSD - 5 mJ
            50.0, // HDD - 50 mJ
        ];

        let from_energy = energy_per_access[request.from_tier as usize];
        let to_energy = energy_per_access[request.to_tier as usize];

        let saved_energy = (from_energy - to_energy) * request.predicted_accesses as f64;
        saved_energy.max(0.0)
    }

    /// Calculate tier efficiency
    fn calculate_tier_efficiency(&self, tier: MemoryTier, history: &AccessHistory) -> f64 {
        let access_rate =
            history.access_count as f64 / history.last_access.elapsed().as_secs_f64().max(1.0);

        // Higher tiers more efficient for high access rates
        match tier {
            MemoryTier::GPU => access_rate.min(1.0),
            MemoryTier::CPU => (access_rate / 10.0).min(1.0),
            MemoryTier::NVMe => (access_rate / 100.0).min(1.0),
            MemoryTier::SSD => (access_rate / 1000.0).min(1.0),
            MemoryTier::HDD => (access_rate / 10000.0).min(1.0),
        }
    }

    /// Calculate memory pressure relief
    fn calculate_memory_pressure_relief(
        &self,
        from: MemoryTier,
        to: MemoryTier,
        size: usize,
    ) -> f64 {
        // Relief when moving to lower tier
        if (to as u8) > (from as u8) {
            (size as f64 / 1_048_576.0) * 0.1 * (to as u8 - from as u8) as f64
        } else {
            0.0
        }
    }

    /// Generate recommendation
    fn generate_recommendation(
        &self,
        approved: bool,
        benefit_ratio: f64,
        request: &PrefetchAnalysisRequest,
    ) -> PrefetchRecommendation {
        if approved {
            if benefit_ratio > 3.0 {
                PrefetchRecommendation::StronglyRecommended {
                    reason: "Very high benefit-to-cost ratio".to_string(),
                    urgency: PrefetchUrgency::Immediate,
                }
            } else if benefit_ratio > 2.0 {
                PrefetchRecommendation::Recommended {
                    reason: "Good benefit-to-cost ratio".to_string(),
                    urgency: PrefetchUrgency::Normal,
                }
            } else {
                PrefetchRecommendation::Marginal {
                    reason: "Modest benefit expected".to_string(),
                    alternative: self.suggest_alternative(request),
                }
            }
        } else {
            PrefetchRecommendation::NotRecommended {
                reason: format!("Benefit ratio {:.2} below threshold", benefit_ratio),
                alternative: self.suggest_alternative(request),
            }
        }
    }

    /// Suggest alternative strategy
    fn suggest_alternative(&self, request: &PrefetchAnalysisRequest) -> Option<String> {
        if request.size_bytes > 10_485_760 {
            Some("Consider partial prefetching or compression".to_string())
        } else if request.predicted_accesses < 5 {
            Some("Wait for more access evidence".to_string())
        } else {
            None
        }
    }
}

/// Prefetch analysis request
#[derive(Debug, Clone)]
pub struct PrefetchAnalysisRequest {
    pub page_id: u64,
    pub from_tier: MemoryTier,
    pub to_tier: MemoryTier,
    pub size_bytes: usize,
    pub access_history: AccessHistory,
    pub pattern_hint: Option<AccessPattern>,
    pub priority: PrefetchPriority,
    pub predicted_accesses: u32,
    pub prediction_window: Duration,
    pub avg_access_latency: Duration,
    pub duration_hint: Option<Duration>,
}

/// Prefetch analysis result
#[derive(Debug, Clone)]
pub struct PrefetchAnalysisResult {
    pub approved: bool,
    pub total_cost: Cost,
    pub total_benefit: Benefit,
    pub net_benefit: f64,
    pub benefit_ratio: f64,
    pub recommendation: PrefetchRecommendation,
}

/// Cost structure
#[derive(Debug, Clone)]
pub struct Cost {
    pub value: f64,
    pub latency: Duration,
    pub energy: f64,
    pub bandwidth: f64,
    pub breakdown: CostBreakdown,
}

impl Cost {
    fn zero() -> Self {
        Self {
            value: 0.0,
            latency: Duration::ZERO,
            energy: 0.0,
            bandwidth: 0.0,
            breakdown: CostBreakdown {
                transfer: 0.0,
                queueing: 0.0,
                overhead: 0.0,
            },
        }
    }

    fn combine(costs: Vec<&Cost>, config: &CostBenefitConfig) -> Self {
        let mut combined = Self::zero();

        for cost in costs {
            combined.value += cost.value;
            combined.latency += cost.latency;
            combined.energy += cost.energy;
            combined.bandwidth += cost.bandwidth;
            combined.breakdown.transfer += cost.breakdown.transfer;
            combined.breakdown.queueing += cost.breakdown.queueing;
            combined.breakdown.overhead += cost.breakdown.overhead;
        }

        // Apply weights
        combined.value = combined.latency.as_secs_f64() * config.latency_weight
            + combined.energy * config.energy_weight
            + combined.bandwidth * config.bandwidth_weight;

        combined
    }
}

/// Cost breakdown
#[derive(Debug, Clone)]
pub struct CostBreakdown {
    pub transfer: f64,
    pub queueing: f64,
    pub overhead: f64,
}

/// Benefit structure
#[derive(Debug, Clone)]
pub struct Benefit {
    pub value: f64,
    pub confidence: f64,
    pub expected_hits: u32,
    pub saved_latency: Duration,
    pub breakdown: BenefitBreakdown,
}

impl Benefit {
    fn combine(benefits: Vec<&Benefit>) -> Self {
        let mut combined = Self {
            value: 0.0,
            confidence: 1.0,
            expected_hits: 0,
            saved_latency: Duration::ZERO,
            breakdown: BenefitBreakdown {
                latency_reduction: 0.0,
                throughput_improvement: 0.0,
                energy_savings: 0.0,
            },
        };

        for benefit in benefits {
            combined.value += benefit.value;
            combined.confidence *= benefit.confidence;
            combined.expected_hits += benefit.expected_hits;
            combined.saved_latency += benefit.saved_latency;
            combined.breakdown.latency_reduction += benefit.breakdown.latency_reduction;
            combined.breakdown.throughput_improvement += benefit.breakdown.throughput_improvement;
            combined.breakdown.energy_savings += benefit.breakdown.energy_savings;
        }

        combined
    }
}

/// Benefit breakdown
#[derive(Debug, Clone)]
pub struct BenefitBreakdown {
    pub latency_reduction: f64,
    pub throughput_improvement: f64,
    pub energy_savings: f64,
}

/// Prefetch recommendation
#[derive(Debug, Clone)]
pub enum PrefetchRecommendation {
    StronglyRecommended {
        reason: String,
        urgency: PrefetchUrgency,
    },
    Recommended {
        reason: String,
        urgency: PrefetchUrgency,
    },
    Marginal {
        reason: String,
        alternative: Option<String>,
    },
    NotRecommended {
        reason: String,
        alternative: Option<String>,
    },
}

/// Prefetch urgency
#[derive(Debug, Clone)]
pub enum PrefetchUrgency {
    Immediate,
    Normal,
    Deferred,
}

/// Transfer cost model
struct TransferCostModel {
    tier_bandwidths: [f64; 5],
    tier_latencies: [Duration; 5],
    tier_energy: [f64; 5],
}

impl TransferCostModel {
    fn new() -> Self {
        Self {
            tier_bandwidths: [
                100.0, // GPU - 100 GB/s
                50.0,  // CPU - 50 GB/s
                7.0,   // NVMe - 7 GB/s
                0.5,   // SSD - 500 MB/s
                0.1,   // HDD - 100 MB/s
            ],
            tier_latencies: [
                Duration::from_micros(10),
                Duration::from_micros(50),
                Duration::from_micros(100),
                Duration::from_micros(1000),
                Duration::from_millis(10),
            ],
            tier_energy: [
                1.0,   // GPU - 1 mJ/GB
                5.0,   // CPU - 5 mJ/GB
                10.0,  // NVMe - 10 mJ/GB
                50.0,  // SSD - 50 mJ/GB
                100.0, // HDD - 100 mJ/GB
            ],
        }
    }

    fn calculate(&self, from: MemoryTier, to: MemoryTier, size: usize) -> TransferMetrics {
        let size_gb = size as f64 / 1_073_741_824.0;

        // Use minimum bandwidth between tiers
        let bandwidth = self.tier_bandwidths[from as usize].min(self.tier_bandwidths[to as usize]);

        let transfer_time = Duration::from_secs_f64(size_gb / bandwidth);
        let latency =
            self.tier_latencies[from as usize] + self.tier_latencies[to as usize] + transfer_time;

        let energy = size_gb * (self.tier_energy[from as usize] + self.tier_energy[to as usize]);

        TransferMetrics {
            latency,
            bandwidth_usage: size_gb,
            energy_joules: energy / 1000.0, // Convert mJ to J
            total_cost: latency.as_secs_f64() * 1000.0 + energy * 0.1,
        }
    }
}

/// Transfer metrics
#[derive(Debug, Clone)]
struct TransferMetrics {
    latency: Duration,
    bandwidth_usage: f64,
    energy_joules: f64,
    total_cost: f64,
}

/// Access pattern analyzer
struct AccessPatternAnalyzer {
    pattern_models: HashMap<AccessPattern, PatternModel>,
}

impl AccessPatternAnalyzer {
    fn new() -> Self {
        Self {
            pattern_models: HashMap::new(),
        }
    }

    fn predict_hit_probability(&self, history: &AccessHistory, window: Duration) -> f64 {
        // Simple model based on access frequency
        let access_rate =
            history.access_count as f64 / history.last_access.elapsed().as_secs_f64().max(1.0);

        let window_accesses = access_rate * window.as_secs_f64();

        // Sigmoid function for probability
        1.0 / (1.0 + (-window_accesses).exp())
    }

    fn analyze_pattern(&self, history: &AccessHistory, hint: &Option<AccessPattern>) -> f64 {
        if hint.is_some() {
            0.8 // High confidence with hint
        } else if history.access_intervals.len() >= 3 {
            // Analyze interval regularity
            let avg_interval = history
                .access_intervals
                .iter()
                .map(|d| d.as_secs_f64())
                .sum::<f64>()
                / history.access_intervals.len() as f64;

            let variance = history
                .access_intervals
                .iter()
                .map(|d| {
                    let diff = d.as_secs_f64() - avg_interval;
                    diff * diff
                })
                .sum::<f64>()
                / history.access_intervals.len() as f64;

            // Low variance means regular pattern
            1.0 / (1.0 + variance)
        } else {
            0.3 // Low confidence without data
        }
    }
}

/// Pattern model
struct PatternModel {
    pattern_type: AccessPattern,
    confidence: f64,
    parameters: Vec<f64>,
}

/// Resource usage tracker
struct ResourceUsageTracker {
    tier_usage: [f64; 5],
    tier_capacity: [usize; 5],
}

impl ResourceUsageTracker {
    fn new() -> Self {
        Self {
            tier_usage: [0.7, 0.5, 0.3, 0.2, 0.1], // Current usage ratios
            tier_capacity: [
                32 * 1024 * 1024 * 1024,   // GPU - 32GB
                96 * 1024 * 1024 * 1024,   // CPU - 96GB
                3200 * 1024 * 1024 * 1024, // NVMe - 3.2TB
                4500 * 1024 * 1024 * 1024, // SSD - 4.5TB
                3700 * 1024 * 1024 * 1024, // HDD - 3.7TB
            ],
        }
    }

    fn estimate_usage(
        &self,
        tier: MemoryTier,
        size: usize,
        duration: Option<Duration>,
    ) -> ResourceUsage {
        let tier_idx = tier as usize;
        let current_usage = self.tier_usage[tier_idx];
        let capacity = self.tier_capacity[tier_idx];

        let size_ratio = size as f64 / capacity as f64;
        let new_usage = current_usage + size_ratio;

        // Cost increases exponentially as tier fills up
        let usage_cost = if new_usage > 0.9 {
            100.0 * (new_usage - 0.9)
        } else if new_usage > 0.7 {
            10.0 * (new_usage - 0.7)
        } else {
            new_usage
        };

        ResourceUsage {
            cost: usage_cost,
            energy: size as f64 * 0.001, // 1 mJ per MB
            bandwidth: size as f64 / duration.unwrap_or(Duration::from_secs(1)).as_secs_f64(),
        }
    }
}

/// Resource usage
struct ResourceUsage {
    cost: f64,
    energy: f64,
    bandwidth: f64,
}

/// Performance history tracker
struct PerformanceHistory {
    prefetch_outcomes: Vec<PrefetchOutcome>,
    success_rate: f64,
}

impl PerformanceHistory {
    fn new() -> Self {
        Self {
            prefetch_outcomes: Vec::new(),
            success_rate: 0.0,
        }
    }

    fn record_outcome(&mut self, outcome: PrefetchOutcome) {
        self.prefetch_outcomes.push(outcome);

        // Keep last 1000 outcomes
        if self.prefetch_outcomes.len() > 1000 {
            self.prefetch_outcomes.remove(0);
        }

        // Update success rate
        let successes = self.prefetch_outcomes.iter().filter(|o| o.was_used).count();

        self.success_rate = successes as f64 / self.prefetch_outcomes.len() as f64;
    }
}

/// Prefetch outcome
#[derive(Debug, Clone)]
struct PrefetchOutcome {
    page_id: u64,
    was_used: bool,
    time_to_use: Option<Duration>,
    benefit_realized: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cost_benefit_analysis() {
        let analyzer = AdvancedCostBenefitAnalyzer::new(CostBenefitConfig::default());

        let request = PrefetchAnalysisRequest {
            page_id: 1000,
            from_tier: MemoryTier::SSD,
            to_tier: MemoryTier::GPU,
            size_bytes: 4096,
            access_history: AccessHistory {
                page_id: 1000,
                access_count: 100,
                last_access: Instant::now(),
                access_intervals: vec![Duration::from_millis(10); 5],
            },
            pattern_hint: Some(AccessPattern::Sequential),
            priority: PrefetchPriority::High,
            predicted_accesses: 50,
            prediction_window: Duration::from_secs(1),
            avg_access_latency: Duration::from_micros(1000),
            duration_hint: None,
        };

        let result = analyzer.analyze_prefetch(&request);
        assert!(result.benefit_ratio > 0.0);
    }

    #[test]
    fn test_transfer_cost_model() {
        let model = TransferCostModel::new();

        let metrics = model.calculate(
            MemoryTier::HDD,
            MemoryTier::GPU,
            10 * 1024 * 1024, // 10MB
        );

        assert!(metrics.latency > Duration::ZERO);
        assert!(metrics.energy_joules > 0.0);
        assert!(metrics.bandwidth_usage > 0.0);
    }
}
