//! Tier prediction for optimal data placement
//!
//! Predicts the best memory tier for data based on access patterns,
//! system state, and historical performance.

use super::*;
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Advanced tier predictor
pub struct AdvancedTierPredictor {
    /// Access frequency analyzer
    frequency_analyzer: FrequencyAnalyzer,
    /// Recency tracker
    recency_tracker: RecencyTracker,
    /// Working set estimator
    working_set_estimator: WorkingSetEstimator,
    /// Tier capacity monitor
    capacity_monitor: TierCapacityMonitor,
    /// Performance model
    performance_model: TierPerformanceModel,
    /// Configuration
    config: TierPredictorConfig,
}

/// Tier predictor configuration
#[derive(Debug, Clone)]
pub struct TierPredictorConfig {
    /// Hot data threshold (accesses per second)
    pub hot_threshold: f64,
    /// Warm data threshold
    pub warm_threshold: f64,
    /// Cold data threshold
    pub cold_threshold: f64,
    /// Recency weight
    pub recency_weight: f64,
    /// Frequency weight
    pub frequency_weight: f64,
    /// Working set weight
    pub working_set_weight: f64,
    /// Enable adaptive thresholds
    pub adaptive_thresholds: bool,
}

impl Default for TierPredictorConfig {
    fn default() -> Self {
        Self {
            hot_threshold: 100.0, // 100 accesses/sec
            warm_threshold: 10.0, // 10 accesses/sec
            cold_threshold: 1.0,  // 1 access/sec
            recency_weight: 0.4,
            frequency_weight: 0.4,
            working_set_weight: 0.2,
            adaptive_thresholds: true,
        }
    }
}

impl AdvancedTierPredictor {
    /// Create new tier predictor
    pub fn new(config: TierPredictorConfig) -> Self {
        Self {
            frequency_analyzer: FrequencyAnalyzer::new(),
            recency_tracker: RecencyTracker::new(10000),
            working_set_estimator: WorkingSetEstimator::new(),
            capacity_monitor: TierCapacityMonitor::new(),
            performance_model: TierPerformanceModel::new(),
            config,
        }
    }

    /// Predict optimal tier for page
    pub fn predict_optimal_tier(
        &self,
        page_info: &PageInfo,
        system_state: &SystemState,
    ) -> TierPrediction {
        // Calculate various scores
        let frequency_score = self.frequency_analyzer.analyze(&page_info.access_history);
        let recency_score = self.recency_tracker.get_recency_score(page_info.page_id);
        let working_set_score = self
            .working_set_estimator
            .is_in_working_set(page_info.page_id, &page_info.access_history);

        // Calculate weighted score
        let overall_score = frequency_score * self.config.frequency_weight
            + recency_score * self.config.recency_weight
            + working_set_score * self.config.working_set_weight;

        // Determine tier based on score and thresholds
        let recommended_tier = self.score_to_tier(overall_score, system_state);

        // Check capacity constraints
        let feasible_tier = self.capacity_monitor.find_feasible_tier(
            recommended_tier,
            page_info.size_bytes,
            system_state,
        );

        // Calculate confidence
        let confidence = self.calculate_confidence(
            frequency_score,
            recency_score,
            working_set_score,
            &page_info.access_history,
        );

        // Get migration benefit
        let migration_benefit = self.performance_model.estimate_migration_benefit(
            page_info.current_tier,
            feasible_tier,
            &page_info.access_history,
        );

        TierPrediction {
            recommended_tier: feasible_tier,
            confidence,
            scores: TierScores {
                frequency: frequency_score,
                recency: recency_score,
                working_set: working_set_score,
                overall: overall_score,
            },
            migration_benefit,
            alternative_tiers: self.get_alternative_tiers(
                feasible_tier,
                overall_score,
                system_state,
            ),
        }
    }

    /// Convert score to tier recommendation
    fn score_to_tier(&self, score: f64, system_state: &SystemState) -> MemoryTier {
        let thresholds = if self.config.adaptive_thresholds {
            self.get_adaptive_thresholds(system_state)
        } else {
            (
                self.config.hot_threshold,
                self.config.warm_threshold,
                self.config.cold_threshold,
            )
        };

        if score >= thresholds.0 {
            MemoryTier::GPU
        } else if score >= thresholds.1 {
            MemoryTier::CPU
        } else if score >= thresholds.2 {
            MemoryTier::NVMe
        } else if score >= 0.1 {
            MemoryTier::SSD
        } else {
            MemoryTier::HDD
        }
    }

    /// Get adaptive thresholds based on system state
    fn get_adaptive_thresholds(&self, system_state: &SystemState) -> (f64, f64, f64) {
        // Adjust thresholds based on memory pressure
        let gpu_pressure = system_state.tier_usage[MemoryTier::GPU as usize];
        let cpu_pressure = system_state.tier_usage[MemoryTier::CPU as usize];

        let hot_multiplier = if gpu_pressure > 0.8 { 1.5 } else { 1.0 };
        let warm_multiplier = if cpu_pressure > 0.8 { 1.3 } else { 1.0 };

        (
            self.config.hot_threshold * hot_multiplier,
            self.config.warm_threshold * warm_multiplier,
            self.config.cold_threshold,
        )
    }

    /// Calculate prediction confidence
    fn calculate_confidence(
        &self,
        frequency_score: f64,
        recency_score: f64,
        working_set_score: f64,
        history: &AccessHistory,
    ) -> f64 {
        // Base confidence on score consistency and data quality
        let score_variance =
            self.calculate_score_variance(frequency_score, recency_score, working_set_score);

        let data_quality = if history.access_count > 10 {
            1.0
        } else {
            history.access_count as f64 / 10.0
        };

        let base_confidence = 1.0 - score_variance.min(1.0);

        (base_confidence * data_quality).max(0.1).min(1.0)
    }

    /// Calculate variance in scores
    fn calculate_score_variance(&self, freq: f64, rec: f64, ws: f64) -> f64 {
        let mean = (freq + rec + ws) / 3.0;
        let variance = ((freq - mean).powi(2) + (rec - mean).powi(2) + (ws - mean).powi(2)) / 3.0;
        variance.sqrt()
    }

    /// Get alternative tier suggestions
    fn get_alternative_tiers(
        &self,
        primary: MemoryTier,
        score: f64,
        system_state: &SystemState,
    ) -> Vec<AlternativeTier> {
        let mut alternatives = Vec::new();

        // Consider adjacent tiers using safe tier navigation methods
        if let Some(higher_tier) = primary.higher_tier() {
            if self
                .capacity_monitor
                .has_capacity(higher_tier, system_state)
            {
                alternatives.push(AlternativeTier {
                    tier: higher_tier,
                    condition: TierCondition::IfAccessRateIncreases(score * 1.5),
                    benefit: 2.0,
                });
            }
        }

        if let Some(lower_tier) = primary.lower_tier() {
            alternatives.push(AlternativeTier {
                tier: lower_tier,
                condition: TierCondition::IfAccessRateDecreases(score * 0.5),
                benefit: 0.5,
            });
        }

        alternatives
    }

    /// Update predictor with access feedback
    pub fn update_access(&mut self, page_id: u64, tier: MemoryTier, timestamp: Instant) {
        self.recency_tracker.update(page_id, timestamp);
        self.working_set_estimator.update(page_id, timestamp);
        self.frequency_analyzer.update(page_id, timestamp);
    }

    /// Update tier capacity information
    pub fn update_capacity(&mut self, tier: MemoryTier, used: usize, total: usize) {
        self.capacity_monitor.update(tier, used, total);
    }
}

/// Page information for prediction
#[derive(Debug, Clone)]
pub struct PageInfo {
    pub page_id: u64,
    pub size_bytes: usize,
    pub current_tier: MemoryTier,
    pub access_history: AccessHistory,
    pub last_modified: Instant,
}

/// System state for prediction
#[derive(Debug, Clone)]
pub struct SystemState {
    pub tier_usage: [f64; 5],
    pub tier_bandwidth_available: [f64; 5],
    pub global_memory_pressure: f64,
    pub io_congestion: f64,
}

/// Tier prediction result
#[derive(Debug, Clone)]
pub struct TierPrediction {
    pub recommended_tier: MemoryTier,
    pub confidence: f64,
    pub scores: TierScores,
    pub migration_benefit: f64,
    pub alternative_tiers: Vec<AlternativeTier>,
}

/// Tier scores breakdown
#[derive(Debug, Clone)]
pub struct TierScores {
    pub frequency: f64,
    pub recency: f64,
    pub working_set: f64,
    pub overall: f64,
}

/// Alternative tier suggestion
#[derive(Debug, Clone)]
pub struct AlternativeTier {
    pub tier: MemoryTier,
    pub condition: TierCondition,
    pub benefit: f64,
}

/// Conditions for tier change
#[derive(Debug, Clone)]
pub enum TierCondition {
    IfAccessRateIncreases(f64),
    IfAccessRateDecreases(f64),
    IfMemoryPressureChanges(f64),
    Always,
}

/// Frequency analyzer
struct FrequencyAnalyzer {
    access_counts: HashMap<u64, AccessFrequency>,
}

impl FrequencyAnalyzer {
    fn new() -> Self {
        Self {
            access_counts: HashMap::new(),
        }
    }

    fn analyze(&self, history: &AccessHistory) -> f64 {
        if history.access_count == 0 {
            return 0.0;
        }

        let elapsed = history.last_access.elapsed().as_secs_f64().max(1.0);
        let access_rate = history.access_count as f64 / elapsed;

        // Apply logarithmic scaling
        (access_rate + 1.0).ln() * 10.0
    }

    fn update(&mut self, page_id: u64, timestamp: Instant) {
        let freq = self
            .access_counts
            .entry(page_id)
            .or_insert_with(|| AccessFrequency::new());

        freq.record_access(timestamp);
    }
}

/// Access frequency tracking
struct AccessFrequency {
    recent_accesses: VecDeque<Instant>,
    total_accesses: u64,
}

impl AccessFrequency {
    fn new() -> Self {
        Self {
            recent_accesses: VecDeque::with_capacity(100),
            total_accesses: 0,
        }
    }

    fn record_access(&mut self, timestamp: Instant) {
        self.recent_accesses.push_back(timestamp);
        self.total_accesses += 1;

        // Keep only recent accesses
        while self.recent_accesses.len() > 100 {
            self.recent_accesses.pop_front();
        }

        // Remove old accesses
        let cutoff = timestamp - Duration::from_secs(300); // 5 minutes
        while let Some(&front) = self.recent_accesses.front() {
            if front < cutoff {
                self.recent_accesses.pop_front();
            } else {
                break;
            }
        }
    }

    fn get_recent_rate(&self) -> f64 {
        if self.recent_accesses.len() < 2 {
            return 0.0;
        }

        let duration = self
            .recent_accesses
            .back()?
            .duration_since(*self.recent_accesses.front()?)
            .as_secs_f64();

        if duration > 0.0 {
            self.recent_accesses.len() as f64 / duration
        } else {
            0.0
        }
    }
}

/// Recency tracker
struct RecencyTracker {
    last_access: HashMap<u64, Instant>,
    capacity: usize,
}

impl RecencyTracker {
    fn new(capacity: usize) -> Self {
        Self {
            last_access: HashMap::with_capacity(capacity),
            capacity,
        }
    }

    fn update(&mut self, page_id: u64, timestamp: Instant) {
        if self.last_access.len() >= self.capacity && !self.last_access.contains_key(&page_id) {
            // Evict oldest
            if let Some((&oldest_page, _)) = self.last_access.iter().min_by_key(|(_, &time)| time) {
                self.last_access.remove(&oldest_page);
            }
        }

        self.last_access.insert(page_id, timestamp);
    }

    fn get_recency_score(&self, page_id: u64) -> f64 {
        if let Some(&last_time) = self.last_access.get(&page_id) {
            let age = last_time.elapsed().as_secs_f64();

            // Exponential decay
            (-age / 300.0).exp() // Half-life of 5 minutes
        } else {
            0.0
        }
    }
}

/// Working set estimator
struct WorkingSetEstimator {
    working_sets: Vec<WorkingSetWindow>,
    current_window: usize,
}

impl WorkingSetEstimator {
    fn new() -> Self {
        Self {
            working_sets: vec![WorkingSetWindow::new(); 4],
            current_window: 0,
        }
    }

    fn update(&mut self, page_id: u64, timestamp: Instant) {
        // Update current window
        self.working_sets[self.current_window].add_page(page_id);

        // Rotate windows periodically
        if self.working_sets[self.current_window].should_rotate(timestamp) {
            self.current_window = (self.current_window + 1) % self.working_sets.len();
            self.working_sets[self.current_window].clear();
        }
    }

    fn is_in_working_set(&self, page_id: u64, history: &AccessHistory) -> f64 {
        let mut presence_count = 0;

        for window in &self.working_sets {
            if window.contains(page_id) {
                presence_count += 1;
            }
        }

        let presence_ratio = presence_count as f64 / self.working_sets.len() as f64;

        // Boost score for frequently accessed pages
        if history.access_count > 50 {
            presence_ratio * 1.5
        } else {
            presence_ratio
        }
    }
}

/// Working set window
#[derive(Debug, Clone)]
struct WorkingSetWindow {
    pages: HashSet<u64>,
    start_time: Instant,
    window_duration: Duration,
}

impl WorkingSetWindow {
    fn new() -> Self {
        Self {
            pages: HashSet::new(),
            start_time: Instant::now(),
            window_duration: Duration::from_secs(60), // 1 minute windows
        }
    }

    fn add_page(&mut self, page_id: u64) {
        self.pages.insert(page_id);
    }

    fn contains(&self, page_id: u64) -> bool {
        self.pages.contains(&page_id)
    }

    fn should_rotate(&self, now: Instant) -> bool {
        now.duration_since(self.start_time) > self.window_duration
    }

    fn clear(&mut self) {
        self.pages.clear();
        self.start_time = Instant::now();
    }
}

/// Tier capacity monitor
struct TierCapacityMonitor {
    capacities: [TierCapacity; 5],
}

impl TierCapacityMonitor {
    fn new() -> Self {
        Self {
            capacities: [
                TierCapacity::new(32 * 1024 * 1024 * 1024),   // GPU - 32GB
                TierCapacity::new(96 * 1024 * 1024 * 1024),   // CPU - 96GB
                TierCapacity::new(3200 * 1024 * 1024 * 1024), // NVMe - 3.2TB
                TierCapacity::new(4500 * 1024 * 1024 * 1024), // SSD - 4.5TB
                TierCapacity::new(3700 * 1024 * 1024 * 1024), // HDD - 3.7TB
            ],
        }
    }

    fn update(&mut self, tier: MemoryTier, used: usize, total: usize) {
        self.capacities[tier as usize].update(used, total);
    }

    fn has_capacity(&self, tier: MemoryTier, system_state: &SystemState) -> bool {
        system_state.tier_usage[tier as usize] < 0.9
    }

    fn find_feasible_tier(
        &self,
        preferred: MemoryTier,
        size: usize,
        system_state: &SystemState,
    ) -> MemoryTier {
        // Try preferred tier first
        if self.can_accommodate(preferred, size, system_state) {
            return preferred;
        }

        // Try higher performance tiers (using safe iterator)
        for tier in preferred.higher_tiers() {
            if self.can_accommodate(tier, size, system_state) {
                return tier;
            }
        }

        // Try lower performance tiers (using safe iterator)
        for tier in preferred.lower_tiers() {
            if self.can_accommodate(tier, size, system_state) {
                return tier;
            }
        }

        // Default to lowest tier
        MemoryTier::HDD
    }

    fn can_accommodate(&self, tier: MemoryTier, size: usize, system_state: &SystemState) -> bool {
        let tier_idx = tier as usize;
        let capacity = &self.capacities[tier_idx];
        let usage = system_state.tier_usage[tier_idx];

        // Check if adding this size would exceed threshold
        let new_usage = usage + (size as f64 / capacity.total as f64);
        new_usage < 0.9
    }
}

/// Tier capacity tracking
#[derive(Debug, Clone)]
struct TierCapacity {
    total: usize,
    used: usize,
    reserved: usize,
}

impl TierCapacity {
    fn new(total: usize) -> Self {
        Self {
            total,
            used: 0,
            reserved: 0,
        }
    }

    fn update(&mut self, used: usize, total: usize) {
        self.used = used;
        self.total = total;
    }

    fn available(&self) -> usize {
        self.total.saturating_sub(self.used + self.reserved)
    }
}

/// Tier performance model
struct TierPerformanceModel {
    access_latencies: [Duration; 5],
    bandwidth_limits: [f64; 5],
}

impl TierPerformanceModel {
    fn new() -> Self {
        Self {
            access_latencies: [
                Duration::from_nanos(10000),  // GPU - 10us
                Duration::from_nanos(50000),  // CPU - 50us
                Duration::from_nanos(100000), // NVMe - 100us
                Duration::from_micros(1000),  // SSD - 1ms
                Duration::from_millis(10),    // HDD - 10ms
            ],
            bandwidth_limits: [
                100.0, // GPU - 100 GB/s
                50.0,  // CPU - 50 GB/s
                7.0,   // NVMe - 7 GB/s
                0.5,   // SSD - 500 MB/s
                0.1,   // HDD - 100 MB/s
            ],
        }
    }

    fn estimate_migration_benefit(
        &self,
        from: MemoryTier,
        to: MemoryTier,
        history: &AccessHistory,
    ) -> f64 {
        let from_latency = self.access_latencies[from as usize].as_secs_f64();
        let to_latency = self.access_latencies[to as usize].as_secs_f64();

        let latency_improvement = (from_latency - to_latency) / from_latency;

        // Factor in access frequency
        let access_rate =
            history.access_count as f64 / history.last_access.elapsed().as_secs_f64().max(1.0);

        latency_improvement * access_rate * 100.0
    }
}

use std::collections::HashSet;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tier_prediction() {
        let predictor = AdvancedTierPredictor::new(TierPredictorConfig::default());

        let page_info = PageInfo {
            page_id: 1000,
            size_bytes: 4096,
            current_tier: MemoryTier::SSD,
            access_history: AccessHistory {
                page_id: 1000,
                access_count: 1000,
                last_access: Instant::now(),
                access_intervals: vec![Duration::from_millis(10); 5],
            },
            last_modified: Instant::now(),
        };

        let system_state = SystemState {
            tier_usage: [0.5, 0.4, 0.3, 0.2, 0.1],
            tier_bandwidth_available: [50.0, 25.0, 3.5, 0.25, 0.05],
            global_memory_pressure: 0.6,
            io_congestion: 0.3,
        };

        let prediction = predictor.predict_optimal_tier(&page_info, &system_state);

        // Hot page should go to GPU or CPU
        assert!(matches!(
            prediction.recommended_tier,
            MemoryTier::GPU | MemoryTier::CPU
        ));
        assert!(prediction.confidence > 0.5);
    }

    #[test]
    fn test_working_set_estimation() {
        let mut estimator = WorkingSetEstimator::new();
        let now = Instant::now();

        // Add pages to working set
        for i in 0..100 {
            estimator.update(i, now);
        }

        // Check presence
        let history = AccessHistory {
            page_id: 50,
            access_count: 10,
            last_access: now,
            access_intervals: vec![],
        };

        let score = estimator.is_in_working_set(50, &history);
        assert!(score > 0.0);
    }
}
