//! Real-time pattern detection and analysis

use chrono::Duration as ChronoDuration;
use std::collections::VecDeque;
use std::time::Duration;

use crate::KnowledgeGraphResult;
use super::types::{TemporalEvent, RealTimeMetrics};
use super::config::RealTimeDetectionConfig;

/// Detected causal pattern
#[derive(Debug, Clone)]
pub struct TemporalCausalPattern {
    /// Pattern identifier
    pub pattern_id: String,
    /// Nodes involved in pattern
    pub nodes: Vec<String>,
    /// Temporal signature
    pub temporal_signature: Vec<ChronoDuration>,
    /// Frequency of occurrence
    pub frequency: usize,
    /// Confidence in pattern
    pub confidence: f64,
}

/// Real-time pattern detector
pub struct RealTimePatternDetector {
    pub config: RealTimeDetectionConfig,
    pub event_buffer: VecDeque<TemporalEvent>,
    pub detected_patterns: Vec<TemporalCausalPattern>,
    pub processing_metrics: RealTimeMetrics,
}

impl RealTimePatternDetector {
    pub fn new(config: RealTimeDetectionConfig) -> Self {
        Self {
            config,
            event_buffer: VecDeque::new(),
            detected_patterns: Vec::new(),
            processing_metrics: RealTimeMetrics {
                average_processing_latency: Duration::from_micros(100),
                throughput: 0.0,
                buffer_utilization: 0.0,
            },
        }
    }

    pub async fn process_event(&mut self, event: TemporalEvent) -> KnowledgeGraphResult<()> {
        let start_time = std::time::Instant::now();
        
        // Add event to buffer
        self.event_buffer.push_back(event);
        
        // Maintain buffer size
        while self.event_buffer.len() > self.config.buffer_size {
            self.event_buffer.pop_front();
        }
        
        // Update buffer utilization
        self.processing_metrics.buffer_utilization = 
            self.event_buffer.len() as f32 / self.config.buffer_size as f32;
        
        // Detect patterns in current window
        self.detect_patterns_in_window();
        
        // Update processing metrics
        let processing_time = start_time.elapsed();
        self.processing_metrics.average_processing_latency = 
            (self.processing_metrics.average_processing_latency + processing_time) / 2;
        
        Ok(())
    }

    pub async fn start_detection(&mut self) -> KnowledgeGraphResult<()> {
        // Initialize detection parameters
        self.detected_patterns.clear();
        Ok(())
    }

    fn detect_patterns_in_window(&mut self) {
        if self.event_buffer.len() < 3 {
            return; // Need minimum events for pattern detection
        }

        // Simple pattern detection: look for sequences of related events
        let mut pattern_candidates: Vec<Vec<String>> = Vec::new();
        
        for window_start in 0..(self.event_buffer.len() - 2) {
            let window_events: Vec<_> = self.event_buffer
                .iter()
                .skip(window_start)
                .take(3)
                .collect();
            
            if self.events_form_pattern(&window_events) {
                let nodes: Vec<String> = window_events
                    .iter()
                    .map(|e| e.node_id.clone())
                    .collect();
                pattern_candidates.push(nodes);
            }
        }
        
        // Convert candidates to patterns
        for (idx, nodes) in pattern_candidates.into_iter().enumerate() {
            let pattern = TemporalCausalPattern {
                pattern_id: format!("pattern_{}", idx),
                temporal_signature: vec![
                    ChronoDuration::seconds(1),
                    ChronoDuration::seconds(2),
                    ChronoDuration::seconds(1),
                ],
                nodes,
                frequency: 1,
                confidence: 0.7,
            };
            
            // Check if pattern meets confidence threshold
            if pattern.confidence >= self.config.confidence_threshold {
                self.detected_patterns.push(pattern);
            }
        }
        
        // Keep only recent patterns
        self.detected_patterns.retain(|p| p.confidence > 0.5);
    }

    fn events_form_pattern(&self, events: &[&TemporalEvent]) -> bool {
        if events.len() < 2 {
            return false;
        }
        
        // Check temporal ordering
        for i in 1..events.len() {
            if events[i].timestamp <= events[i-1].timestamp {
                return false;
            }
        }
        
        // Check if events are related (simple heuristic)
        let time_diff = events.last().unwrap().timestamp - events.first().unwrap().timestamp;
        let max_pattern_duration = chrono::Duration::from_std(self.config.pattern_window.to_std().unwrap_or(std::time::Duration::from_secs(3600)));
        
        time_diff <= max_pattern_duration.unwrap_or(ChronoDuration::hours(1))
    }
}
