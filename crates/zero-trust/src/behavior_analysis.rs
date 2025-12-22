//! User behavior modeling, anomaly detection, risk pattern recognition, and adaptive learning
//!
//! This module implements comprehensive behavioral analysis following zero-trust principles:
//! - User and entity behavior analytics (UEBA)
//! - Machine learning-based anomaly detection
//! - Pattern recognition and profiling
//! - Adaptive baseline adjustment
//! - Real-time threat detection

use crate::error::{ZeroTrustError, ZeroTrustResult};
use async_trait::async_trait;
use chrono::{DateTime, Duration, Timelike, Utc};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use uuid::Uuid;

/// Behavior analysis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehaviorAnalysisConfig {
    /// Baseline learning period
    pub baseline_period: Duration,
    /// Anomaly detection sensitivity (0.0 - 1.0)
    pub anomaly_sensitivity: f64,
    /// Maximum events to track per entity
    pub max_events_per_entity: usize,
    /// Pattern recognition window
    pub pattern_window: Duration,
    /// Enable adaptive learning
    pub adaptive_learning: bool,
    /// Risk score threshold for alerts
    pub alert_threshold: f64,
    /// Behavior profile update interval
    pub profile_update_interval: Duration,
}

impl Default for BehaviorAnalysisConfig {
    fn default() -> Self {
        Self {
            baseline_period: Duration::days(30),
            anomaly_sensitivity: 0.7,
            max_events_per_entity: 10000,
            pattern_window: Duration::hours(24),
            adaptive_learning: true,
            alert_threshold: 0.8,
            profile_update_interval: Duration::hours(1),
        }
    }
}

/// Behavioral event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehaviorEvent {
    /// Event ID
    pub event_id: Uuid,
    /// Entity ID (user, device, service)
    pub entity_id: Uuid,
    /// Event type
    pub event_type: EventType,
    /// Event timestamp
    pub timestamp: DateTime<Utc>,
    /// Event metadata
    pub metadata: EventMetadata,
    /// Risk indicators
    pub risk_indicators: Vec<RiskIndicator>,
}

/// Event types
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EventType {
    /// Authentication event
    Authentication(AuthEventType),
    /// Access event
    Access(AccessEventType),
    /// Data event
    Data(DataEventType),
    /// Network event
    Network(NetworkEventType),
    /// System event
    System(SystemEventType),
}

/// Authentication event types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AuthEventType {
    Login,
    Logout,
    FailedLogin,
    PasswordChange,
    MfaChallenge,
    PrivilegeEscalation,
}

/// Access event types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AccessEventType {
    ResourceAccess,
    PermissionChange,
    AccessDenied,
    UnusualAccess,
}

/// Data event types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DataEventType {
    Download,
    Upload,
    Modification,
    Deletion,
    BulkOperation,
}

/// Network event types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NetworkEventType {
    Connection,
    DataTransfer,
    PortScan,
    UnusualTraffic,
}

/// System event types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SystemEventType {
    ProcessExecution,
    ConfigChange,
    ServiceStart,
    ServiceStop,
}

/// Event metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventMetadata {
    /// Source IP address
    pub source_ip: Option<String>,
    /// Destination
    pub destination: Option<String>,
    /// User agent
    pub user_agent: Option<String>,
    /// Geolocation
    pub geolocation: Option<GeoLocation>,
    /// Additional context
    pub context: serde_json::Value,
}

/// Geolocation information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeoLocation {
    pub country: String,
    pub city: String,
    pub latitude: f64,
    pub longitude: f64,
    pub timezone: String,
}

/// Risk indicators
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RiskIndicator {
    /// Unusual time of activity
    UnusualTime {
        expected_hours: Vec<u32>,
        actual_hour: u32,
    },
    /// Unusual location
    UnusualLocation {
        expected_countries: Vec<String>,
        actual_country: String,
    },
    /// High frequency activity
    HighFrequency { threshold: u32, actual: u32 },
    /// Suspicious pattern
    SuspiciousPattern {
        pattern_name: String,
        confidence: f64,
    },
    /// Deviation from baseline
    BaselineDeviation { metric: String, deviation: f64 },
}

/// Behavior profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehaviorProfile {
    /// Entity ID
    pub entity_id: Uuid,
    /// Profile creation time
    pub created_at: DateTime<Utc>,
    /// Last update time
    pub updated_at: DateTime<Utc>,
    /// Baseline metrics
    pub baseline: BaselineMetrics,
    /// Detected patterns
    pub patterns: Vec<BehaviorPattern>,
    /// Risk score history
    pub risk_history: VecDeque<(DateTime<Utc>, f64)>,
    /// Current risk score
    pub current_risk_score: f64,
}

/// Baseline metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineMetrics {
    /// Average events per hour
    pub avg_events_per_hour: f64,
    /// Common event types
    pub common_event_types: HashMap<EventType, f64>,
    /// Active hours (0-23)
    pub active_hours: Vec<u32>,
    /// Common locations
    pub common_locations: Vec<String>,
    /// Typical resources accessed
    pub typical_resources: Vec<String>,
    /// Average data volume
    pub avg_data_volume: f64,
    /// Peer group comparison
    pub peer_deviation: f64,
}

/// Behavior pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehaviorPattern {
    /// Pattern ID
    pub pattern_id: Uuid,
    /// Pattern name
    pub name: String,
    /// Pattern type
    pub pattern_type: PatternType,
    /// Confidence level
    pub confidence: f64,
    /// First observed
    pub first_observed: DateTime<Utc>,
    /// Last observed
    pub last_observed: DateTime<Utc>,
    /// Occurrence count
    pub occurrences: u32,
}

/// Pattern types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PatternType {
    /// Time-based pattern
    Temporal,
    /// Sequence-based pattern
    Sequential,
    /// Volume-based pattern
    Volumetric,
    /// Access pattern
    Access,
    /// Geographic pattern
    Geographic,
}

/// Anomaly detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyResult {
    /// Anomaly ID
    pub anomaly_id: Uuid,
    /// Entity ID
    pub entity_id: Uuid,
    /// Anomaly score (0.0 - 1.0)
    pub anomaly_score: f64,
    /// Anomaly type
    pub anomaly_type: AnomalyType,
    /// Contributing factors
    pub factors: Vec<AnomalyFactor>,
    /// Recommended actions
    pub recommendations: Vec<String>,
    /// Detection timestamp
    pub detected_at: DateTime<Utc>,
}

/// Anomaly types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnomalyType {
    /// Behavioral anomaly
    Behavioral,
    /// Statistical anomaly
    Statistical,
    /// Pattern anomaly
    Pattern,
    /// Peer group anomaly
    PeerGroup,
    /// Combined anomaly
    Combined,
}

/// Anomaly factors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyFactor {
    /// Factor name
    pub name: String,
    /// Factor weight
    pub weight: f64,
    /// Factor value
    pub value: f64,
    /// Baseline value
    pub baseline: f64,
}

/// Behavior analysis engine trait
#[async_trait]
pub trait BehaviorAnalysisEngineTrait: Send + Sync {
    /// Record behavioral event
    async fn record_event(&self, event: BehaviorEvent) -> ZeroTrustResult<()>;

    /// Analyze behavior for anomalies
    async fn analyze_behavior(&self, entity_id: Uuid) -> ZeroTrustResult<AnomalyResult>;

    /// Get behavior profile
    async fn get_profile(&self, entity_id: Uuid) -> ZeroTrustResult<BehaviorProfile>;

    /// Update baseline
    async fn update_baseline(&self, entity_id: Uuid) -> ZeroTrustResult<()>;

    /// Detect patterns
    async fn detect_patterns(&self, entity_id: Uuid) -> ZeroTrustResult<Vec<BehaviorPattern>>;

    /// Calculate risk score
    async fn calculate_risk_score(&self, entity_id: Uuid) -> ZeroTrustResult<f64>;

    /// Get peer group statistics
    async fn get_peer_group_stats(&self, entity_id: Uuid) -> ZeroTrustResult<BaselineMetrics>;

    /// Train model (for adaptive learning)
    async fn train_model(&self, entity_id: Uuid) -> ZeroTrustResult<()>;
}

/// Behavior analysis engine implementation
pub struct BehaviorAnalysisEngine {
    config: BehaviorAnalysisConfig,
    events: Arc<DashMap<Uuid, VecDeque<BehaviorEvent>>>,
    profiles: Arc<DashMap<Uuid, BehaviorProfile>>,
    models: Arc<DashMap<Uuid, BehaviorModel>>,
    peer_groups: Arc<DashMap<String, Vec<Uuid>>>,
}

/// Behavior model for ML
#[derive(Debug, Clone, Serialize, Deserialize)]
struct BehaviorModel {
    /// Model version
    version: u32,
    /// Feature weights
    weights: HashMap<String, f64>,
    /// Training samples
    samples: u32,
    /// Model accuracy
    accuracy: f64,
    /// Last training time
    last_trained: DateTime<Utc>,
}

impl BehaviorAnalysisEngine {
    /// Create new behavior analysis engine
    pub fn new(config: BehaviorAnalysisConfig) -> ZeroTrustResult<Self> {
        Ok(Self {
            config,
            events: Arc::new(DashMap::new()),
            profiles: Arc::new(DashMap::new()),
            models: Arc::new(DashMap::new()),
            peer_groups: Arc::new(DashMap::new()),
        })
    }

    /// Calculate event frequency
    fn calculate_event_frequency(&self, events: &VecDeque<BehaviorEvent>) -> f64 {
        if events.len() < 2 {
            return 0.0;
        }

        let duration = events.front().unwrap().timestamp - events.back().unwrap().timestamp;
        let hours = duration.num_hours() as f64;

        if hours > 0.0 {
            events.len() as f64 / hours
        } else {
            events.len() as f64
        }
    }

    /// Detect temporal anomalies
    fn detect_temporal_anomalies(
        &self,
        event: &BehaviorEvent,
        profile: &BehaviorProfile,
    ) -> Vec<RiskIndicator> {
        let mut indicators = vec![];

        let hour = event.timestamp.hour();
        if !profile.baseline.active_hours.contains(&hour) {
            indicators.push(RiskIndicator::UnusualTime {
                expected_hours: profile.baseline.active_hours.clone(),
                actual_hour: hour,
            });
        }

        indicators
    }

    /// Detect geographic anomalies
    fn detect_geographic_anomalies(
        &self,
        event: &BehaviorEvent,
        profile: &BehaviorProfile,
    ) -> Vec<RiskIndicator> {
        let mut indicators = vec![];

        if let Some(geo) = &event.metadata.geolocation {
            if !profile.baseline.common_locations.contains(&geo.country) {
                indicators.push(RiskIndicator::UnusualLocation {
                    expected_countries: profile.baseline.common_locations.clone(),
                    actual_country: geo.country.clone(),
                });
            }
        }

        indicators
    }

    /// Calculate anomaly score
    fn calculate_anomaly_score(&self, factors: &[AnomalyFactor]) -> f64 {
        if factors.is_empty() {
            return 0.0;
        }

        let weighted_sum: f64 = factors
            .iter()
            .map(|f| {
                let deviation = (f.value - f.baseline).abs() / (f.baseline + 1.0);
                deviation * f.weight
            })
            .sum();

        let total_weight: f64 = factors.iter().map(|f| f.weight).sum();

        (weighted_sum / total_weight).min(1.0)
    }

    /// Extract features from events
    fn extract_features(&self, events: &VecDeque<BehaviorEvent>) -> HashMap<String, f64> {
        let mut features = HashMap::new();

        // Event frequency
        features.insert(
            "event_frequency".to_string(),
            self.calculate_event_frequency(events),
        );

        // Event type distribution
        let mut type_counts = HashMap::new();
        for event in events {
            *type_counts.entry(event.event_type.clone()).or_insert(0) += 1;
        }

        for (event_type, count) in type_counts {
            features.insert(
                format!("event_type_{event_type:?}"),
                count as f64 / events.len() as f64,
            );
        }

        // Time-based features
        let mut hour_counts = [0u32; 24];
        for event in events {
            hour_counts[event.timestamp.hour() as usize] += 1;
        }

        for (hour, count) in hour_counts.iter().enumerate() {
            features.insert(format!("hour_{hour}"), *count as f64 / events.len() as f64);
        }

        features
    }
}

#[async_trait]
impl BehaviorAnalysisEngineTrait for BehaviorAnalysisEngine {
    async fn record_event(&self, event: BehaviorEvent) -> ZeroTrustResult<()> {
        let entity_id = event.entity_id;

        // Store event
        let mut events = self.events.entry(entity_id).or_default();

        // Limit events per entity
        while events.len() >= self.config.max_events_per_entity {
            events.pop_back();
        }

        events.push_front(event.clone());

        // Initialize profile if needed
        if !self.profiles.contains_key(&entity_id) {
            let profile = BehaviorProfile {
                entity_id,
                created_at: Utc::now(),
                updated_at: Utc::now(),
                baseline: BaselineMetrics {
                    avg_events_per_hour: 0.0,
                    common_event_types: HashMap::new(),
                    active_hours: vec![],
                    common_locations: vec![],
                    typical_resources: vec![],
                    avg_data_volume: 0.0,
                    peer_deviation: 0.0,
                },
                patterns: vec![],
                risk_history: VecDeque::new(),
                current_risk_score: 0.5,
            };
            self.profiles.insert(entity_id, profile);
        }

        // Check for immediate anomalies
        if let Ok(anomaly) = self.analyze_behavior(entity_id).await {
            if anomaly.anomaly_score > self.config.alert_threshold {
                // In production, trigger alert
                tracing::warn!(
                    "High risk anomaly detected for entity {}: score {}",
                    entity_id,
                    anomaly.anomaly_score
                );
            }
        }

        Ok(())
    }

    async fn analyze_behavior(&self, entity_id: Uuid) -> ZeroTrustResult<AnomalyResult> {
        let events =
            self.events
                .get(&entity_id)
                .ok_or_else(|| ZeroTrustError::BehavioralAnomaly {
                    anomaly_type: "no_events".to_string(),
                    details: "No events found for entity".to_string(),
                })?;

        let profile =
            self.profiles
                .get(&entity_id)
                .ok_or_else(|| ZeroTrustError::BehavioralAnomaly {
                    anomaly_type: "no_profile".to_string(),
                    details: "No profile found for entity".to_string(),
                })?;

        let mut factors = vec![];

        // Analyze event frequency
        let current_frequency = self.calculate_event_frequency(&events);
        if profile.baseline.avg_events_per_hour > 0.0 {
            let freq_deviation = (current_frequency - profile.baseline.avg_events_per_hour).abs()
                / profile.baseline.avg_events_per_hour;

            if freq_deviation > self.config.anomaly_sensitivity {
                factors.push(AnomalyFactor {
                    name: "event_frequency".to_string(),
                    weight: 0.3,
                    value: current_frequency,
                    baseline: profile.baseline.avg_events_per_hour,
                });
            }
        }

        // Analyze recent events for anomalies
        if let Some(recent_event) = events.front() {
            let temporal_anomalies = self.detect_temporal_anomalies(recent_event, &profile);
            let geographic_anomalies = self.detect_geographic_anomalies(recent_event, &profile);

            if !temporal_anomalies.is_empty() {
                factors.push(AnomalyFactor {
                    name: "temporal".to_string(),
                    weight: 0.25,
                    value: 1.0,
                    baseline: 0.0,
                });
            }

            if !geographic_anomalies.is_empty() {
                factors.push(AnomalyFactor {
                    name: "geographic".to_string(),
                    weight: 0.25,
                    value: 1.0,
                    baseline: 0.0,
                });
            }
        }

        // Analyze peer group deviation
        if let Ok(peer_stats) = self.get_peer_group_stats(entity_id).await {
            let peer_deviation =
                (profile.baseline.avg_events_per_hour - peer_stats.avg_events_per_hour).abs()
                    / (peer_stats.avg_events_per_hour + 1.0);

            if peer_deviation > self.config.anomaly_sensitivity {
                factors.push(AnomalyFactor {
                    name: "peer_group".to_string(),
                    weight: 0.2,
                    value: profile.baseline.avg_events_per_hour,
                    baseline: peer_stats.avg_events_per_hour,
                });
            }
        }

        let anomaly_score = self.calculate_anomaly_score(&factors);

        let anomaly_type = if factors.len() > 2 {
            AnomalyType::Combined
        } else if factors.iter().any(|f| f.name == "peer_group") {
            AnomalyType::PeerGroup
        } else if factors.iter().any(|f| f.name == "event_frequency") {
            AnomalyType::Statistical
        } else {
            AnomalyType::Behavioral
        };

        let recommendations = if anomaly_score > 0.8 {
            vec![
                "Immediate investigation required".to_string(),
                "Consider additional authentication".to_string(),
                "Monitor closely for next 24 hours".to_string(),
            ]
        } else if anomaly_score > 0.6 {
            vec![
                "Review recent activities".to_string(),
                "Check for policy violations".to_string(),
            ]
        } else {
            vec!["Continue monitoring".to_string()]
        };

        Ok(AnomalyResult {
            anomaly_id: Uuid::new_v4(),
            entity_id,
            anomaly_score,
            anomaly_type,
            factors,
            recommendations,
            detected_at: Utc::now(),
        })
    }

    async fn get_profile(&self, entity_id: Uuid) -> ZeroTrustResult<BehaviorProfile> {
        self.profiles
            .get(&entity_id)
            .map(|entry| entry.clone())
            .ok_or_else(|| ZeroTrustError::BehavioralAnomaly {
                anomaly_type: "no_profile".to_string(),
                details: "Profile not found".to_string(),
            })
    }

    async fn update_baseline(&self, entity_id: Uuid) -> ZeroTrustResult<()> {
        let events =
            self.events
                .get(&entity_id)
                .ok_or_else(|| ZeroTrustError::BehavioralAnomaly {
                    anomaly_type: "no_events".to_string(),
                    details: "No events found for baseline update".to_string(),
                })?;

        let mut profile =
            self.profiles
                .get_mut(&entity_id)
                .ok_or_else(|| ZeroTrustError::BehavioralAnomaly {
                    anomaly_type: "no_profile".to_string(),
                    details: "Profile not found".to_string(),
                })?;

        // Calculate new baseline metrics
        let mut event_type_counts = HashMap::new();
        let mut hour_counts = [0u32; 24];
        let mut locations = HashMap::new();

        for event in events.iter() {
            *event_type_counts
                .entry(event.event_type.clone())
                .or_insert(0) += 1;
            hour_counts[event.timestamp.hour() as usize] += 1;

            if let Some(geo) = &event.metadata.geolocation {
                *locations.entry(geo.country.clone()).or_insert(0) += 1;
            }
        }

        // Update baseline
        profile.baseline.avg_events_per_hour = self.calculate_event_frequency(&events);

        profile.baseline.common_event_types = event_type_counts
            .into_iter()
            .map(|(k, v)| (k, v as f64 / events.len() as f64))
            .collect();

        profile.baseline.active_hours = hour_counts
            .iter()
            .enumerate()
            .filter(|(_, &count)| count > 0)
            .map(|(hour, _)| hour as u32)
            .collect();

        profile.baseline.common_locations = locations
            .into_iter()
            .filter(|(_, count)| *count > events.len() / 10) // At least 10% of events
            .map(|(location, _)| location)
            .collect();

        profile.updated_at = Utc::now();

        Ok(())
    }

    async fn detect_patterns(&self, entity_id: Uuid) -> ZeroTrustResult<Vec<BehaviorPattern>> {
        let events =
            self.events
                .get(&entity_id)
                .ok_or_else(|| ZeroTrustError::BehavioralAnomaly {
                    anomaly_type: "no_events".to_string(),
                    details: "No events found for pattern detection".to_string(),
                })?;

        let mut patterns = vec![];

        // Detect temporal patterns
        let mut hourly_pattern = HashMap::new();
        for event in events.iter() {
            *hourly_pattern.entry(event.timestamp.hour()).or_insert(0) += 1;
        }

        if let Some((&peak_hour, &count)) = hourly_pattern.iter().max_by_key(|&(_, &v)| v) {
            if count > events.len() / 4 {
                // More than 25% of events in one hour
                patterns.push(BehaviorPattern {
                    pattern_id: Uuid::new_v4(),
                    name: format!("Peak activity at hour {peak_hour}"),
                    pattern_type: PatternType::Temporal,
                    confidence: count as f64 / events.len() as f64,
                    first_observed: events.back().unwrap().timestamp,
                    last_observed: events.front().unwrap().timestamp,
                    occurrences: count as u32,
                });
            }
        }

        // Detect sequential patterns
        let mut sequence_map = HashMap::new();
        for window in events.iter().collect::<Vec<_>>().windows(2) {
            if let [prev, curr] = window {
                let sequence = (prev.event_type.clone(), curr.event_type.clone());
                *sequence_map.entry(sequence).or_insert(0) += 1;
            }
        }

        for ((prev_type, curr_type), count) in sequence_map {
            if count > 5 {
                // At least 5 occurrences
                patterns.push(BehaviorPattern {
                    pattern_id: Uuid::new_v4(),
                    name: format!("Sequence: {prev_type:?} -> {curr_type:?}"),
                    pattern_type: PatternType::Sequential,
                    confidence: count as f64 / events.len() as f64,
                    first_observed: events.back().unwrap().timestamp,
                    last_observed: events.front().unwrap().timestamp,
                    occurrences: count,
                });
            }
        }

        Ok(patterns)
    }

    async fn calculate_risk_score(&self, entity_id: Uuid) -> ZeroTrustResult<f64> {
        let anomaly = self.analyze_behavior(entity_id).await?;
        let mut profile =
            self.profiles
                .get_mut(&entity_id)
                .ok_or_else(|| ZeroTrustError::BehavioralAnomaly {
                    anomaly_type: "no_profile".to_string(),
                    details: "Profile not found".to_string(),
                })?;

        // Update risk history
        profile
            .risk_history
            .push_front((Utc::now(), anomaly.anomaly_score));
        while profile.risk_history.len() > 100 {
            profile.risk_history.pop_back();
        }

        // Calculate weighted average risk score
        let mut weighted_sum = 0.0;
        let mut weight_sum = 0.0;

        for (i, (_, score)) in profile.risk_history.iter().enumerate() {
            let weight = 1.0 / (i as f64 + 1.0); // Recent scores have higher weight
            weighted_sum += score * weight;
            weight_sum += weight;
        }

        let risk_score = if weight_sum > 0.0 {
            weighted_sum / weight_sum
        } else {
            anomaly.anomaly_score
        };

        profile.current_risk_score = risk_score;

        Ok(risk_score)
    }

    async fn get_peer_group_stats(&self, entity_id: Uuid) -> ZeroTrustResult<BaselineMetrics> {
        // Find peer group for entity
        let peer_group = self
            .peer_groups
            .iter()
            .find(|group| group.value().contains(&entity_id))
            .map(|group| group.key().clone())
            .unwrap_or_else(|| "default".to_string());

        let peers = self
            .peer_groups
            .get(&peer_group)
            .map(|group| group.clone())
            .unwrap_or_else(|| vec![entity_id]);

        // Calculate aggregate statistics
        let mut total_events_per_hour = 0.0;
        let mut peer_count = 0;

        for peer_id in peers {
            if peer_id == entity_id {
                continue; // Skip self
            }

            if let Some(profile) = self.profiles.get(&peer_id) {
                total_events_per_hour += profile.baseline.avg_events_per_hour;
                peer_count += 1;
            }
        }

        let avg_events_per_hour = if peer_count > 0 {
            total_events_per_hour / peer_count as f64
        } else {
            0.0
        };

        Ok(BaselineMetrics {
            avg_events_per_hour,
            common_event_types: HashMap::new(),
            active_hours: vec![],
            common_locations: vec![],
            typical_resources: vec![],
            avg_data_volume: 0.0,
            peer_deviation: 0.0,
        })
    }

    async fn train_model(&self, entity_id: Uuid) -> ZeroTrustResult<()> {
        if !self.config.adaptive_learning {
            return Ok(());
        }

        let events =
            self.events
                .get(&entity_id)
                .ok_or_else(|| ZeroTrustError::BehavioralAnomaly {
                    anomaly_type: "no_events".to_string(),
                    details: "No events found for training".to_string(),
                })?;

        // Extract features
        let features = self.extract_features(&events);

        // Simple model: update feature weights based on recent anomalies
        let mut model = self
            .models
            .entry(entity_id)
            .or_insert_with(|| BehaviorModel {
                version: 1,
                weights: HashMap::new(),
                samples: 0,
                accuracy: 0.0,
                last_trained: Utc::now(),
            });

        // Update weights (simplified training)
        for (feature, value) in features {
            let current_weight = model.weights.get(&feature).copied().unwrap_or(0.5);
            let new_weight = (current_weight * 0.9) + (value * 0.1); // Exponential moving average
            model.weights.insert(feature, new_weight);
        }

        model.samples += events.len() as u32;
        model.last_trained = Utc::now();
        model.version += 1;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_event_recording() {
        let config = BehaviorAnalysisConfig::default();
        let engine = BehaviorAnalysisEngine::new(config)?;

        let entity_id = Uuid::new_v4();
        let event = BehaviorEvent {
            event_id: Uuid::new_v4(),
            entity_id,
            event_type: EventType::Authentication(AuthEventType::Login),
            timestamp: Utc::now(),
            metadata: EventMetadata {
                source_ip: Some("192.168.1.100".to_string()),
                destination: None,
                user_agent: Some("Mozilla/5.0".to_string()),
                geolocation: Some(GeoLocation {
                    country: "US".to_string(),
                    city: "New York".to_string(),
                    latitude: 40.7128,
                    longitude: -74.0060,
                    timezone: "America/New_York".to_string(),
                }),
                context: serde_json::json!({}),
            },
            risk_indicators: vec![],
        };

        engine.record_event(event).await?;

        let profile = engine.get_profile(entity_id).await?;
        assert_eq!(profile.entity_id, entity_id);
    }

    #[tokio::test]
    async fn test_anomaly_detection() {
        let config = BehaviorAnalysisConfig::default();
        let engine = BehaviorAnalysisEngine::new(config)?;

        let entity_id = Uuid::new_v4();

        // Record normal events during business hours
        for hour in 9..17 {
            let event = BehaviorEvent {
                event_id: Uuid::new_v4(),
                entity_id,
                event_type: EventType::Access(AccessEventType::ResourceAccess),
                timestamp: Utc::now().with_hour(hour)?,
                metadata: EventMetadata {
                    source_ip: Some("192.168.1.100".to_string()),
                    destination: None,
                    user_agent: None,
                    geolocation: Some(GeoLocation {
                        country: "US".to_string(),
                        city: "New York".to_string(),
                        latitude: 40.7128,
                        longitude: -74.0060,
                        timezone: "America/New_York".to_string(),
                    }),
                    context: serde_json::json!({}),
                },
                risk_indicators: vec![],
            };
            engine.record_event(event).await?;
        }

        // Update baseline
        engine.update_baseline(entity_id).await?;

        // Record anomalous event at unusual hour
        let anomalous_event = BehaviorEvent {
            event_id: Uuid::new_v4(),
            entity_id,
            event_type: EventType::Access(AccessEventType::ResourceAccess),
            timestamp: Utc::now().with_hour(3)?, // 3 AM
            metadata: EventMetadata {
                source_ip: Some("192.168.1.100".to_string()),
                destination: None,
                user_agent: None,
                geolocation: Some(GeoLocation {
                    country: "China".to_string(), // Different country
                    city: "Beijing".to_string(),
                    latitude: 39.9042,
                    longitude: 116.4074,
                    timezone: "Asia/Shanghai".to_string(),
                }),
                context: serde_json::json!({}),
            },
            risk_indicators: vec![],
        };
        engine.record_event(anomalous_event).await?;

        let anomaly = engine.analyze_behavior(entity_id).await?;
        assert!(anomaly.anomaly_score > 0.0);
        assert!(!anomaly.factors.is_empty());
    }

    #[tokio::test]
    async fn test_pattern_detection() {
        let config = BehaviorAnalysisConfig::default();
        let engine = BehaviorAnalysisEngine::new(config)?;

        let entity_id = Uuid::new_v4();

        // Create a pattern: login -> resource access
        for _ in 0..10 {
            let login = BehaviorEvent {
                event_id: Uuid::new_v4(),
                entity_id,
                event_type: EventType::Authentication(AuthEventType::Login),
                timestamp: Utc::now(),
                metadata: EventMetadata {
                    source_ip: Some("192.168.1.100".to_string()),
                    destination: None,
                    user_agent: None,
                    geolocation: None,
                    context: serde_json::json!({}),
                },
                risk_indicators: vec![],
            };
            engine.record_event(login).await?;

            let access = BehaviorEvent {
                event_id: Uuid::new_v4(),
                entity_id,
                event_type: EventType::Access(AccessEventType::ResourceAccess),
                timestamp: Utc::now(),
                metadata: EventMetadata {
                    source_ip: Some("192.168.1.100".to_string()),
                    destination: None,
                    user_agent: None,
                    geolocation: None,
                    context: serde_json::json!({}),
                },
                risk_indicators: vec![],
            };
            engine.record_event(access).await?;
        }

        let patterns = engine.detect_patterns(entity_id).await?;
        assert!(!patterns.is_empty());
        assert!(patterns
            .iter()
            .any(|p| p.pattern_type == PatternType::Sequential));
    }

    #[tokio::test]
    async fn test_risk_score_calculation() {
        let config = BehaviorAnalysisConfig::default();
        let engine = BehaviorAnalysisEngine::new(config)?;

        let entity_id = Uuid::new_v4();

        // Record some events
        for i in 0..5 {
            let event = BehaviorEvent {
                event_id: Uuid::new_v4(),
                entity_id,
                event_type: EventType::Authentication(AuthEventType::Login),
                timestamp: Utc::now() - Duration::hours(i),
                metadata: EventMetadata {
                    source_ip: Some("192.168.1.100".to_string()),
                    destination: None,
                    user_agent: None,
                    geolocation: None,
                    context: serde_json::json!({}),
                },
                risk_indicators: vec![],
            };
            engine.record_event(event).await?;
        }

        let risk_score = engine.calculate_risk_score(entity_id).await?;
        assert!(risk_score >= 0.0 && risk_score <= 1.0);
    }

    #[tokio::test]
    async fn test_baseline_update() {
        let config = BehaviorAnalysisConfig::default();
        let engine = BehaviorAnalysisEngine::new(config)?;

        let entity_id = Uuid::new_v4();

        // Record events with specific patterns
        for hour in [9, 10, 11, 14, 15, 16].iter() {
            let event = BehaviorEvent {
                event_id: Uuid::new_v4(),
                entity_id,
                event_type: EventType::Access(AccessEventType::ResourceAccess),
                timestamp: Utc::now().with_hour(*hour)?,
                metadata: EventMetadata {
                    source_ip: Some("192.168.1.100".to_string()),
                    destination: None,
                    user_agent: None,
                    geolocation: Some(GeoLocation {
                        country: "US".to_string(),
                        city: "New York".to_string(),
                        latitude: 40.7128,
                        longitude: -74.0060,
                        timezone: "America/New_York".to_string(),
                    }),
                    context: serde_json::json!({}),
                },
                risk_indicators: vec![],
            };
            engine.record_event(event).await?;
        }

        engine.update_baseline(entity_id).await?;

        let profile = engine.get_profile(entity_id).await?;
        assert!(!profile.baseline.active_hours.is_empty());
        assert!(profile
            .baseline
            .common_locations
            .contains(&"US".to_string()));
    }

    #[tokio::test]
    async fn test_peer_group_comparison() {
        let config = BehaviorAnalysisConfig::default();
        let engine = BehaviorAnalysisEngine::new(config)?;

        // Create peer group
        let peer_group = "engineering".to_string();
        let peer_ids: Vec<Uuid> = (0..3).map(|_| Uuid::new_v4()).collect();
        engine
            .peer_groups
            .insert(peer_group.clone(), peer_ids.clone());

        // Create profiles for peers
        for peer_id in &peer_ids {
            let profile = BehaviorProfile {
                entity_id: *peer_id,
                created_at: Utc::now(),
                updated_at: Utc::now(),
                baseline: BaselineMetrics {
                    avg_events_per_hour: 10.0,
                    common_event_types: HashMap::new(),
                    active_hours: vec![9, 10, 11, 14, 15, 16],
                    common_locations: vec!["US".to_string()],
                    typical_resources: vec![],
                    avg_data_volume: 1000.0,
                    peer_deviation: 0.0,
                },
                patterns: vec![],
                risk_history: VecDeque::new(),
                current_risk_score: 0.3,
            };
            engine.profiles.insert(*peer_id, profile);
        }

        let stats = engine.get_peer_group_stats(peer_ids[0]).await?;
        assert_eq!(stats.avg_events_per_hour, 10.0);
    }

    #[tokio::test]
    async fn test_adaptive_learning() {
        let mut config = BehaviorAnalysisConfig::default();
        config.adaptive_learning = true;
        let engine = BehaviorAnalysisEngine::new(config)?;

        let entity_id = Uuid::new_v4();

        // Record events for training
        for _ in 0..20 {
            let event = BehaviorEvent {
                event_id: Uuid::new_v4(),
                entity_id,
                event_type: EventType::Access(AccessEventType::ResourceAccess),
                timestamp: Utc::now(),
                metadata: EventMetadata {
                    source_ip: Some("192.168.1.100".to_string()),
                    destination: None,
                    user_agent: None,
                    geolocation: None,
                    context: serde_json::json!({}),
                },
                risk_indicators: vec![],
            };
            engine.record_event(event).await?;
        }

        engine.train_model(entity_id).await?;

        let model = engine.models.get(&entity_id)?;
        assert!(model.samples > 0);
        assert!(!model.weights.is_empty());
    }

    #[tokio::test]
    async fn test_high_frequency_detection() {
        let config = BehaviorAnalysisConfig::default();
        let engine = BehaviorAnalysisEngine::new(config)?;

        let entity_id = Uuid::new_v4();

        // Record many events in short time
        for _ in 0..100 {
            let event = BehaviorEvent {
                event_id: Uuid::new_v4(),
                entity_id,
                event_type: EventType::Authentication(AuthEventType::FailedLogin),
                timestamp: Utc::now(),
                metadata: EventMetadata {
                    source_ip: Some("192.168.1.100".to_string()),
                    destination: None,
                    user_agent: None,
                    geolocation: None,
                    context: serde_json::json!({}),
                },
                risk_indicators: vec![RiskIndicator::HighFrequency {
                    threshold: 10,
                    actual: 100,
                }],
            };
            engine.record_event(event).await?;
        }

        let anomaly = engine.analyze_behavior(entity_id).await?;
        assert!(anomaly.anomaly_score > 0.5);
    }

    #[tokio::test]
    async fn test_volumetric_pattern() {
        let config = BehaviorAnalysisConfig::default();
        let engine = BehaviorAnalysisEngine::new(config)?;

        let entity_id = Uuid::new_v4();

        // Normal data transfers
        for _ in 0..5 {
            let event = BehaviorEvent {
                event_id: Uuid::new_v4(),
                entity_id,
                event_type: EventType::Data(DataEventType::Download),
                timestamp: Utc::now(),
                metadata: EventMetadata {
                    source_ip: Some("192.168.1.100".to_string()),
                    destination: None,
                    user_agent: None,
                    geolocation: None,
                    context: serde_json::json!({
                        "data_volume": 1000000 // 1MB
                    }),
                },
                risk_indicators: vec![],
            };
            engine.record_event(event).await?;
        }

        // Large data transfer
        let large_transfer = BehaviorEvent {
            event_id: Uuid::new_v4(),
            entity_id,
            event_type: EventType::Data(DataEventType::Download),
            timestamp: Utc::now(),
            metadata: EventMetadata {
                source_ip: Some("192.168.1.100".to_string()),
                destination: None,
                user_agent: None,
                geolocation: None,
                context: serde_json::json!({
                    "data_volume": 10000000000i64 // 10GB
                }),
            },
            risk_indicators: vec![],
        };
        engine.record_event(large_transfer).await?;

        let patterns = engine.detect_patterns(entity_id).await?;
        assert!(!patterns.is_empty());
    }

    #[tokio::test]
    async fn test_geographic_anomaly() {
        let config = BehaviorAnalysisConfig::default();
        let engine = BehaviorAnalysisEngine::new(config)?;

        let entity_id = Uuid::new_v4();

        // Events from consistent location
        for _ in 0..10 {
            let event = BehaviorEvent {
                event_id: Uuid::new_v4(),
                entity_id,
                event_type: EventType::Authentication(AuthEventType::Login),
                timestamp: Utc::now(),
                metadata: EventMetadata {
                    source_ip: Some("192.168.1.100".to_string()),
                    destination: None,
                    user_agent: None,
                    geolocation: Some(GeoLocation {
                        country: "US".to_string(),
                        city: "New York".to_string(),
                        latitude: 40.7128,
                        longitude: -74.0060,
                        timezone: "America/New_York".to_string(),
                    }),
                    context: serde_json::json!({}),
                },
                risk_indicators: vec![],
            };
            engine.record_event(event).await?;
        }

        engine.update_baseline(entity_id).await?;

        // Event from different location
        let foreign_event = BehaviorEvent {
            event_id: Uuid::new_v4(),
            entity_id,
            event_type: EventType::Authentication(AuthEventType::Login),
            timestamp: Utc::now(),
            metadata: EventMetadata {
                source_ip: Some("200.100.50.25".to_string()),
                destination: None,
                user_agent: None,
                geolocation: Some(GeoLocation {
                    country: "Brazil".to_string(),
                    city: "São Paulo".to_string(),
                    latitude: -23.5505,
                    longitude: -46.6333,
                    timezone: "America/Sao_Paulo".to_string(),
                }),
                context: serde_json::json!({}),
            },
            risk_indicators: vec![],
        };
        engine.record_event(foreign_event).await?;

        let anomaly = engine.analyze_behavior(entity_id).await?;
        assert!(anomaly.factors.iter().any(|f| f.name == "geographic"));
    }
}
