//! Pattern recognition for prefetching
//!
//! Implements advanced pattern detection algorithms including
//! sequential, strided, temporal, and spatial patterns.

use super::*;
use anyhow::Result;
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

/// Advanced pattern recognizer
pub struct AdvancedPatternRecognizer {
    /// History window size
    window_size: usize,
    /// Access history
    access_window: VecDeque<PageAccess>,
    /// Detected patterns with confidence
    patterns: HashMap<AccessPattern, f64>,
    /// Pattern change detector
    change_detector: PatternChangeDetector,
    /// Correlation analyzer
    correlation_analyzer: CorrelationAnalyzer,
}

/// Page access record
#[derive(Debug, Clone)]
struct PageAccess {
    page_id: u64,
    timestamp: Instant,
    tier: MemoryTier,
    access_type: AccessType,
}

/// Access type
#[derive(Debug, Clone, PartialEq)]
enum AccessType {
    Read,
    Write,
    ReadWrite,
}

impl AdvancedPatternRecognizer {
    /// Create new pattern recognizer
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            access_window: VecDeque::with_capacity(window_size),
            patterns: HashMap::new(),
            change_detector: PatternChangeDetector::new(),
            correlation_analyzer: CorrelationAnalyzer::new(window_size),
        }
    }

    /// Record page access
    pub fn record_access(
        &mut self,
        page_id: u64,
        timestamp: Instant,
        tier: MemoryTier,
    ) -> Result<()> {
        let access = PageAccess {
            page_id,
            timestamp,
            tier,
            access_type: AccessType::Read, // Default for now
        };

        self.access_window.push_back(access);

        if self.access_window.len() > self.window_size {
            self.access_window.pop_front();
        }

        // Update pattern detection
        self.update_patterns()?;

        Ok(())
    }

    /// Update pattern detection
    fn update_patterns(&mut self) -> Result<()> {
        if self.access_window.len() < 3 {
            return Ok(());
        }

        // Clear old patterns
        self.patterns.clear();

        // Detect various patterns
        self.detect_sequential_pattern();
        self.detect_stride_patterns();
        self.detect_temporal_patterns();
        self.detect_spatial_patterns();
        self.detect_graph_patterns();

        // Check for pattern changes
        self.change_detector.check_pattern_change(&self.patterns);

        Ok(())
    }

    /// Detect sequential access pattern
    fn detect_sequential_pattern(&mut self) {
        let pages: Vec<u64> = self.access_window.iter().map(|a| a.page_id).collect();

        let mut sequential_count = 0;
        for i in 1..pages.len() {
            if pages[i] == pages[i - 1] + 1 {
                sequential_count += 1;
            }
        }

        let confidence = sequential_count as f64 / (pages.len() - 1).max(1) as f64;
        if confidence > 0.7 {
            self.patterns.insert(AccessPattern::Sequential, confidence);
        }
    }

    /// Detect stride patterns
    fn detect_stride_patterns(&mut self) {
        let pages: Vec<u64> = self.access_window.iter().map(|a| a.page_id).collect();

        if pages.len() < 3 {
            return;
        }

        // Try different stride values
        for stride in 1..=16 {
            let mut matches = 0;

            for i in 1..pages.len() {
                let expected = pages[i - 1].saturating_add(stride as u64);
                if pages[i] == expected {
                    matches += 1;
                }
            }

            let confidence = matches as f64 / (pages.len() - 1) as f64;
            if confidence > 0.7 && stride > 1 {
                self.patterns
                    .insert(AccessPattern::Strided(stride), confidence);
                break; // Use first matching stride
            }
        }
    }

    /// Detect temporal patterns
    fn detect_temporal_patterns(&mut self) {
        // Analyze time intervals between accesses
        let mut intervals = Vec::new();

        for i in 1..self.access_window.len() {
            let interval = self.access_window[i]
                .timestamp
                .duration_since(self.access_window[i - 1].timestamp);
            intervals.push(interval);
        }

        if intervals.is_empty() {
            return;
        }

        // Check for regular intervals
        let avg_interval = intervals.iter().sum::<Duration>() / intervals.len() as u32;
        let mut regular_count = 0;

        for interval in &intervals {
            let diff = if *interval > avg_interval {
                *interval - avg_interval
            } else {
                avg_interval - *interval
            };

            if diff < avg_interval / 10 {
                // Within 10% of average
                regular_count += 1;
            }
        }

        let confidence = regular_count as f64 / intervals.len() as f64;
        if confidence > 0.7 {
            self.patterns.insert(AccessPattern::Temporal, confidence);
        }
    }

    /// Detect spatial patterns
    fn detect_spatial_patterns(&mut self) {
        // Group accesses by spatial locality
        let pages: Vec<u64> = self.access_window.iter().map(|a| a.page_id).collect();

        if pages.len() < 4 {
            return;
        }

        // Check for locality within blocks
        let block_size = 64; // Pages per block
        let mut locality_count = 0;

        for i in 1..pages.len() {
            let block1 = pages[i - 1] / block_size;
            let block2 = pages[i] / block_size;

            if block1 == block2 || block1.abs_diff(block2) <= 1 {
                locality_count += 1;
            }
        }

        let confidence = locality_count as f64 / (pages.len() - 1) as f64;
        if confidence > 0.6 {
            self.patterns.insert(AccessPattern::Spatial, confidence);
        }
    }

    /// Detect graph-based patterns
    fn detect_graph_patterns(&mut self) {
        // Build access graph
        let mut transitions: HashMap<u64, HashMap<u64, u32>> = HashMap::new();

        for i in 1..self.access_window.len() {
            let from = self.access_window[i - 1].page_id;
            let to = self.access_window[i].page_id;

            *transitions
                .entry(from)
                .or_insert_with(HashMap::new)
                .entry(to)
                .or_insert(0) += 1;
        }

        // Find strong patterns
        let mut pattern_strength = 0.0;
        let mut total_transitions = 0;

        for (_, targets) in &transitions {
            for &count in targets.values() {
                if count > 1 {
                    pattern_strength += count as f64;
                }
                total_transitions += count;
            }
        }

        if total_transitions > 0 {
            let confidence = pattern_strength / total_transitions as f64;
            if confidence > 0.5 {
                self.patterns.insert(AccessPattern::Mixed, confidence);
            }
        }
    }

    /// Get dominant pattern
    pub fn get_dominant_pattern(&self) -> Option<(AccessPattern, f64)> {
        self.patterns
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(p, c)| (p.clone(), *c))
    }

    /// Get all detected patterns
    pub fn get_patterns(&self) -> &HashMap<AccessPattern, f64> {
        &self.patterns
    }

    /// Predict next pages based on patterns
    pub fn predict_next(&self, current_page: u64, count: usize) -> Vec<(u64, f64)> {
        let mut predictions = Vec::new();

        if let Some((pattern, confidence)) = self.get_dominant_pattern() {
            match pattern {
                AccessPattern::Sequential => {
                    for i in 1..=count {
                        predictions.push((current_page + i as u64, confidence));
                    }
                }
                AccessPattern::Strided(stride) => {
                    for i in 1..=count {
                        let next_page = current_page + (i * stride) as u64;
                        predictions.push((next_page, confidence));
                    }
                }
                AccessPattern::Spatial => {
                    // Predict nearby pages
                    let block_size = 64;
                    let current_block = current_page / block_size;

                    for offset in 1..=count {
                        let next_page = (current_block * block_size) + offset as u64;
                        predictions.push((next_page, confidence * 0.8));
                    }
                }
                _ => {
                    // Use correlation analysis for complex patterns
                    predictions = self
                        .correlation_analyzer
                        .predict_correlated_pages(current_page, count);
                }
            }
        }

        predictions
    }
}

/// Pattern change detector
struct PatternChangeDetector {
    previous_patterns: HashMap<AccessPattern, f64>,
    change_count: u32,
    stability_window: usize,
}

impl PatternChangeDetector {
    fn new() -> Self {
        Self {
            previous_patterns: HashMap::new(),
            change_count: 0,
            stability_window: 10,
        }
    }

    fn check_pattern_change(&mut self, current_patterns: &HashMap<AccessPattern, f64>) {
        if self.previous_patterns.is_empty() {
            self.previous_patterns = current_patterns.clone();
            return;
        }

        // Check for significant changes
        let mut changed = false;

        // Check for new patterns
        for (pattern, confidence) in current_patterns {
            if let Some(prev_confidence) = self.previous_patterns.get(pattern) {
                if (confidence - prev_confidence).abs() > 0.3 {
                    changed = true;
                }
            } else if *confidence > 0.5 {
                changed = true;
            }
        }

        // Check for disappeared patterns
        for (pattern, prev_confidence) in &self.previous_patterns {
            if !current_patterns.contains_key(pattern) && *prev_confidence > 0.5 {
                changed = true;
            }
        }

        if changed {
            self.change_count += 1;
            self.previous_patterns = current_patterns.clone();
        }
    }

    fn get_stability_score(&self) -> f64 {
        1.0 - (self.change_count as f64 / self.stability_window as f64).min(1.0)
    }
}

/// Correlation analyzer for complex patterns
struct CorrelationAnalyzer {
    correlation_matrix: HashMap<u64, HashMap<u64, f64>>,
    access_frequency: HashMap<u64, u32>,
    window_size: usize,
}

impl CorrelationAnalyzer {
    fn new(window_size: usize) -> Self {
        Self {
            correlation_matrix: HashMap::new(),
            access_frequency: HashMap::new(),
            window_size,
        }
    }

    fn update_correlations(&mut self, accesses: &VecDeque<PageAccess>) {
        // Update access frequency
        self.access_frequency.clear();
        for access in accesses {
            *self.access_frequency.entry(access.page_id).or_insert(0) += 1;
        }

        // Update correlation matrix
        self.correlation_matrix.clear();

        for i in 0..accesses.len() {
            for j in i + 1..accesses.len().min(i + 10) {
                let page1 = accesses[i].page_id;
                let page2 = accesses[j].page_id;

                if page1 != page2 {
                    let time_diff = accesses[j]
                        .timestamp
                        .duration_since(accesses[i].timestamp)
                        .as_secs_f64();

                    let correlation = 1.0 / (1.0 + time_diff);

                    *self
                        .correlation_matrix
                        .entry(page1)
                        .or_insert_with(HashMap::new)
                        .entry(page2)
                        .or_insert(0.0) += correlation;
                }
            }
        }
    }

    fn predict_correlated_pages(&self, page_id: u64, count: usize) -> Vec<(u64, f64)> {
        let mut predictions = Vec::new();

        if let Some(correlations) = self.correlation_matrix.get(&page_id) {
            // Sort by correlation strength
            let mut sorted_correlations: Vec<_> = correlations
                .iter()
                .map(|(&page, &corr)| (page, corr))
                .collect();

            sorted_correlations.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // Take top predictions
            for (page, correlation) in sorted_correlations.into_iter().take(count) {
                let frequency = self.access_frequency.get(&page).copied().unwrap_or(1) as f64;

                let confidence = (correlation * frequency) / self.window_size as f64;
                predictions.push((page, confidence.min(1.0)));
            }
        }

        predictions
    }
}

/// Markov chain predictor for sequence patterns
pub struct MarkovPredictor {
    order: usize,
    transition_matrix: HashMap<Vec<u64>, HashMap<u64, f64>>,
    state_history: VecDeque<u64>,
}

impl MarkovPredictor {
    pub fn new(order: usize) -> Self {
        Self {
            order,
            transition_matrix: HashMap::new(),
            state_history: VecDeque::with_capacity(order),
        }
    }

    pub fn update(&mut self, page_id: u64) {
        if self.state_history.len() >= self.order {
            // Create state from history
            let state: Vec<u64> = self.state_history.iter().cloned().collect();

            // Update transition count
            *self
                .transition_matrix
                .entry(state)
                .or_insert_with(HashMap::new)
                .entry(page_id)
                .or_insert(0.0) += 1.0;
        }

        // Update history
        self.state_history.push_back(page_id);
        if self.state_history.len() > self.order {
            self.state_history.pop_front();
        }
    }

    pub fn predict_next(&self, confidence_threshold: f64) -> Vec<(u64, f64)> {
        if self.state_history.len() < self.order {
            return Vec::new();
        }

        let state: Vec<u64> = self.state_history.iter().cloned().collect();

        if let Some(transitions) = self.transition_matrix.get(&state) {
            let total: f64 = transitions.values().sum();

            let mut predictions: Vec<_> = transitions
                .iter()
                .map(|(&page, &count)| (page, count / total))
                .filter(|(_, prob)| *prob >= confidence_threshold)
                .collect();

            predictions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            predictions
        } else {
            Vec::new()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sequential_pattern_detection() {
        let mut recognizer = AdvancedPatternRecognizer::new(10);
        let start_time = Instant::now();

        // Sequential accesses
        for i in 0..10 {
            recognizer
                .record_access(
                    i,
                    start_time + Duration::from_millis(i * 10),
                    MemoryTier::CPU,
                )
                .unwrap();
        }

        let patterns = recognizer.get_patterns();
        assert!(patterns.contains_key(&AccessPattern::Sequential));
        assert!(patterns[&AccessPattern::Sequential] > 0.9);
    }

    #[test]
    fn test_markov_predictor() {
        let mut predictor = MarkovPredictor::new(2);

        // Train with pattern: 1->2->3->1->2->3
        let sequence = vec![1, 2, 3, 1, 2, 3, 1, 2];
        for page in sequence {
            predictor.update(page);
        }

        // Should predict 3 after seeing [1, 2]
        let predictions = predictor.predict_next(0.5);
        assert!(!predictions.is_empty());
        assert_eq!(predictions[0].0, 3);
        assert!(predictions[0].1 > 0.9);
    }
}
