//! Connection quality metrics
//!
//! Tracks RTT, jitter, and packet loss for connection quality estimation.
//! Based on nebula-traverse quality tracking patterns.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Maximum samples to keep for rolling statistics
const MAX_SAMPLES: usize = 100;

/// Connection quality tracker
#[derive(Debug)]
pub struct ConnectionQuality {
    /// RTT samples (in microseconds)
    rtt_samples: VecDeque<u64>,

    /// Packet send timestamps (for loss detection)
    sent_packets: VecDeque<(u64, Instant)>,

    /// Next sequence number
    next_seq: u64,

    /// Acknowledged sequence numbers
    acked_seqs: VecDeque<u64>,

    /// Last update time
    last_update: Instant,
}

impl ConnectionQuality {
    /// Create a new quality tracker
    pub fn new() -> Self {
        Self {
            rtt_samples: VecDeque::with_capacity(MAX_SAMPLES),
            sent_packets: VecDeque::with_capacity(MAX_SAMPLES),
            next_seq: 0,
            acked_seqs: VecDeque::with_capacity(MAX_SAMPLES),
            last_update: Instant::now(),
        }
    }

    /// Record a packet being sent
    pub fn record_send(&mut self) -> u64 {
        let seq = self.next_seq;
        self.next_seq += 1;

        self.sent_packets.push_back((seq, Instant::now()));

        // Trim old entries
        while self.sent_packets.len() > MAX_SAMPLES {
            self.sent_packets.pop_front();
        }

        seq
    }

    /// Record a packet acknowledgment with RTT
    pub fn record_ack(&mut self, seq: u64, rtt: Duration) {
        // Record RTT
        let rtt_us = rtt.as_micros() as u64;
        self.rtt_samples.push_back(rtt_us);

        while self.rtt_samples.len() > MAX_SAMPLES {
            self.rtt_samples.pop_front();
        }

        // Record ack
        self.acked_seqs.push_back(seq);
        while self.acked_seqs.len() > MAX_SAMPLES {
            self.acked_seqs.pop_front();
        }

        self.last_update = Instant::now();
    }

    /// Get current quality metrics
    pub fn metrics(&self) -> QualityMetrics {
        let rtt = self.calculate_rtt();
        let jitter = self.calculate_jitter();
        let packet_loss = self.calculate_packet_loss();

        QualityMetrics {
            rtt,
            jitter,
            packet_loss,
            samples: self.rtt_samples.len(),
            last_update: self.last_update,
        }
    }

    /// Calculate average RTT
    fn calculate_rtt(&self) -> Duration {
        if self.rtt_samples.is_empty() {
            return Duration::ZERO;
        }

        let sum: u64 = self.rtt_samples.iter().sum();
        let avg = sum / self.rtt_samples.len() as u64;
        Duration::from_micros(avg)
    }

    /// Calculate jitter (variation in RTT)
    fn calculate_jitter(&self) -> Duration {
        if self.rtt_samples.len() < 2 {
            return Duration::ZERO;
        }

        let avg = self.rtt_samples.iter().sum::<u64>() / self.rtt_samples.len() as u64;

        let variance: u64 = self
            .rtt_samples
            .iter()
            .map(|&x| {
                let diff = if x > avg { x - avg } else { avg - x };
                diff * diff
            })
            .sum::<u64>()
            / self.rtt_samples.len() as u64;

        // Standard deviation
        let std_dev = (variance as f64).sqrt() as u64;
        Duration::from_micros(std_dev)
    }

    /// Calculate packet loss percentage
    fn calculate_packet_loss(&self) -> f32 {
        if self.sent_packets.is_empty() {
            return 0.0;
        }

        let total_sent = self.sent_packets.len();
        let acked: usize = self
            .sent_packets
            .iter()
            .filter(|(seq, _)| self.acked_seqs.contains(seq))
            .count();

        let lost = total_sent - acked;
        (lost as f32 / total_sent as f32) * 100.0
    }

    /// Check if connection quality is good
    /// RTT < 100ms, jitter < 20ms, loss < 5%
    pub fn is_good(&self) -> bool {
        let metrics = self.metrics();
        metrics.rtt < Duration::from_millis(100)
            && metrics.jitter < Duration::from_millis(20)
            && metrics.packet_loss < 5.0
    }

    /// Check if connection quality is acceptable
    /// RTT < 300ms, jitter < 50ms, loss < 10%
    pub fn is_acceptable(&self) -> bool {
        let metrics = self.metrics();
        metrics.rtt < Duration::from_millis(300)
            && metrics.jitter < Duration::from_millis(50)
            && metrics.packet_loss < 10.0
    }
}

impl Default for ConnectionQuality {
    fn default() -> Self {
        Self::new()
    }
}

/// Quality metrics snapshot
#[derive(Debug, Clone)]
pub struct QualityMetrics {
    /// Round-trip time
    pub rtt: Duration,

    /// Jitter (RTT variation)
    pub jitter: Duration,

    /// Packet loss percentage (0-100)
    pub packet_loss: f32,

    /// Number of samples used
    pub samples: usize,

    /// Last update time
    pub last_update: Instant,
}

impl QualityMetrics {
    /// Get quality score (0-100, higher is better)
    pub fn score(&self) -> u8 {
        let mut score = 100u8;

        // Penalize high RTT
        let rtt_penalty = (self.rtt.as_millis() / 10).min(30) as u8;
        score = score.saturating_sub(rtt_penalty);

        // Penalize high jitter
        let jitter_penalty = (self.jitter.as_millis() / 5).min(20) as u8;
        score = score.saturating_sub(jitter_penalty);

        // Penalize packet loss
        let loss_penalty = (self.packet_loss * 2.0).min(50.0) as u8;
        score = score.saturating_sub(loss_penalty);

        score
    }

    /// Get quality rating
    pub fn rating(&self) -> QualityRating {
        match self.score() {
            80..=100 => QualityRating::Excellent,
            60..=79 => QualityRating::Good,
            40..=59 => QualityRating::Fair,
            20..=39 => QualityRating::Poor,
            _ => QualityRating::Bad,
        }
    }
}

/// Quality rating
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QualityRating {
    Excellent,
    Good,
    Fair,
    Poor,
    Bad,
}

impl std::fmt::Display for QualityRating {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Excellent => write!(f, "Excellent"),
            Self::Good => write!(f, "Good"),
            Self::Fair => write!(f, "Fair"),
            Self::Poor => write!(f, "Poor"),
            Self::Bad => write!(f, "Bad"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quality_tracking() {
        let mut quality = ConnectionQuality::new();

        // Simulate some packets
        for _ in 0..10 {
            let seq = quality.record_send();
            quality.record_ack(seq, Duration::from_millis(20));
        }

        let metrics = quality.metrics();
        assert_eq!(metrics.samples, 10);
        assert!(metrics.rtt > Duration::ZERO);
        assert_eq!(metrics.packet_loss, 0.0);
    }

    #[test]
    fn test_packet_loss() {
        let mut quality = ConnectionQuality::new();

        // Send 10 packets
        for _ in 0..10 {
            quality.record_send();
        }

        // Only ack 8 of them
        for seq in 0..8 {
            quality.record_ack(seq, Duration::from_millis(20));
        }

        let metrics = quality.metrics();
        assert!(metrics.packet_loss > 0.0);
    }

    #[test]
    fn test_jitter_calculation() {
        let mut quality = ConnectionQuality::new();

        // Variable RTTs
        let rtts = [10, 30, 15, 45, 20, 35, 25, 40, 10, 50];

        for &rtt in rtts.iter() {
            let seq = quality.record_send();
            quality.record_ack(seq, Duration::from_millis(rtt));
        }

        let metrics = quality.metrics();
        assert!(metrics.jitter > Duration::ZERO);
    }

    #[test]
    fn test_quality_score() {
        // Perfect connection
        let excellent = QualityMetrics {
            rtt: Duration::from_millis(10),
            jitter: Duration::from_millis(2),
            packet_loss: 0.0,
            samples: 100,
            last_update: Instant::now(),
        };
        assert!(excellent.score() >= 80);
        assert_eq!(excellent.rating(), QualityRating::Excellent);

        // Poor connection
        let poor = QualityMetrics {
            rtt: Duration::from_millis(200),
            jitter: Duration::from_millis(50),
            packet_loss: 10.0,
            samples: 100,
            last_update: Instant::now(),
        };
        assert!(poor.score() < 60);
    }

    #[test]
    fn test_is_good() {
        let mut quality = ConnectionQuality::new();

        // Good connection with low RTT
        for _ in 0..10 {
            let seq = quality.record_send();
            quality.record_ack(seq, Duration::from_millis(20));
        }

        assert!(quality.is_good());
        assert!(quality.is_acceptable());
    }

    #[test]
    fn test_is_acceptable() {
        let mut quality = ConnectionQuality::new();

        // Acceptable connection with medium RTT
        for _ in 0..10 {
            let seq = quality.record_send();
            quality.record_ack(seq, Duration::from_millis(150));
        }

        assert!(!quality.is_good()); // Not good (RTT > 100ms)
        assert!(quality.is_acceptable()); // But acceptable (RTT < 300ms)
    }
}
