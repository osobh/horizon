//! Risk scoring algorithms, threat assessment, contextual risk evaluation, and decision making
//!
//! This module implements comprehensive risk assessment following zero-trust principles:
//! - Multi-factor risk scoring
//! - Real-time threat assessment
//! - Contextual risk evaluation
//! - Adaptive risk-based access control
//! - Decision automation and recommendations

use crate::error::{ZeroTrustError, ZeroTrustResult};
use async_trait::async_trait;
use chrono::{DateTime, Duration, Timelike, Utc};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

/// Risk engine configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskEngineConfig {
    /// Base risk threshold for access denial
    pub deny_threshold: f64,
    /// Base risk threshold for additional verification
    pub challenge_threshold: f64,
    /// Risk score decay rate (per hour)
    pub risk_decay_rate: f64,
    /// Maximum risk score
    pub max_risk_score: f64,
    /// Enable contextual evaluation
    pub contextual_evaluation: bool,
    /// Risk history retention
    pub history_retention: Duration,
    /// Enable adaptive thresholds
    pub adaptive_thresholds: bool,
    /// Risk aggregation method
    pub aggregation_method: AggregationMethod,
}

impl Default for RiskEngineConfig {
    fn default() -> Self {
        Self {
            deny_threshold: 0.8,
            challenge_threshold: 0.6,
            risk_decay_rate: 0.05,
            max_risk_score: 1.0,
            contextual_evaluation: true,
            history_retention: Duration::days(30),
            adaptive_thresholds: true,
            aggregation_method: AggregationMethod::WeightedAverage,
        }
    }
}

/// Risk aggregation methods
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AggregationMethod {
    /// Maximum risk score
    Maximum,
    /// Average risk score
    Average,
    /// Weighted average
    WeightedAverage,
    /// Multiplicative
    Multiplicative,
}

/// Risk assessment request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskAssessmentRequest {
    /// Subject being assessed
    pub subject: RiskSubject,
    /// Context of the request
    pub context: RiskContext,
    /// Risk factors to evaluate
    pub factors: Vec<RiskFactor>,
    /// Historical data
    pub history: Option<RiskHistory>,
}

/// Risk subjects
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskSubject {
    /// User identity
    User { user_id: Uuid },
    /// Device
    Device { device_id: Uuid },
    /// Service or application
    Service { service_id: Uuid },
    /// Network connection
    Connection { connection_id: Uuid },
    /// Combined entity
    Combined { entity_ids: Vec<Uuid> },
}

/// Risk context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskContext {
    /// Action being attempted
    pub action: String,
    /// Resource being accessed
    pub resource: String,
    /// Request timestamp
    pub timestamp: DateTime<Utc>,
    /// Source location
    pub source_location: Option<String>,
    /// Request metadata
    pub metadata: serde_json::Value,
}

/// Risk factors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskFactor {
    /// Identity-related risk
    Identity {
        verification_level: f64,
        authentication_strength: f64,
        account_age_days: u32,
        failed_attempts: u32,
    },
    /// Device-related risk
    Device {
        trust_score: f64,
        compliance_status: bool,
        patch_level: f64,
        jailbreak_detected: bool,
    },
    /// Behavioral risk
    Behavioral {
        anomaly_score: f64,
        deviation_from_baseline: f64,
        suspicious_patterns: u32,
    },
    /// Network risk
    Network {
        connection_security: f64,
        location_trust: f64,
        vpn_usage: bool,
        threat_intelligence_score: f64,
    },
    /// Temporal risk
    Temporal {
        unusual_time: bool,
        frequency_anomaly: f64,
        session_duration: f64,
    },
    /// Data sensitivity
    DataSensitivity {
        classification_level: DataClassification,
        volume: f64,
        regulatory_requirements: Vec<String>,
    },
    /// Environmental risk
    Environmental {
        threat_level: ThreatLevel,
        ongoing_incidents: u32,
        geographic_risk: f64,
    },
}

/// Data classification levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum DataClassification {
    Public = 0,
    Internal = 1,
    Confidential = 2,
    Secret = 3,
    TopSecret = 4,
}

/// Threat levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ThreatLevel {
    Low = 0,
    Medium = 1,
    High = 2,
    Critical = 3,
}

/// Risk history
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskHistory {
    /// Historical risk scores
    pub scores: VecDeque<(DateTime<Utc>, f64)>,
    /// Past incidents
    pub incidents: Vec<SecurityIncident>,
    /// Average risk over time
    pub average_risk: f64,
    /// Peak risk score
    pub peak_risk: f64,
}

/// Security incident
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityIncident {
    /// Incident ID
    pub incident_id: Uuid,
    /// Incident type
    pub incident_type: String,
    /// Severity
    pub severity: ThreatLevel,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Resolution status
    pub resolved: bool,
}

/// Risk assessment result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskAssessment {
    /// Assessment ID
    pub assessment_id: Uuid,
    /// Overall risk score
    pub risk_score: f64,
    /// Risk level
    pub risk_level: RiskLevel,
    /// Contributing factors
    pub contributing_factors: Vec<RiskContribution>,
    /// Recommended action
    pub recommendation: RiskDecision,
    /// Mitigation suggestions
    pub mitigations: Vec<String>,
    /// Assessment timestamp
    pub timestamp: DateTime<Utc>,
    /// Confidence level
    pub confidence: f64,
}

/// Risk levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RiskLevel {
    Low = 0,
    Medium = 1,
    High = 2,
    Critical = 3,
}

/// Risk contribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskContribution {
    /// Factor name
    pub factor: String,
    /// Contribution to overall risk
    pub contribution: f64,
    /// Weight applied
    pub weight: f64,
    /// Details
    pub details: String,
}

/// Risk-based decisions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RiskDecision {
    /// Allow access
    Allow,
    /// Allow with monitoring
    AllowWithMonitoring,
    /// Require additional authentication
    ChallengeRequired,
    /// Require step-up authentication
    StepUpRequired,
    /// Deny access
    Deny,
    /// Quarantine for review
    Quarantine,
}

/// Risk policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskPolicy {
    /// Policy ID
    pub policy_id: Uuid,
    /// Policy name
    pub name: String,
    /// Conditions
    pub conditions: Vec<PolicyCondition>,
    /// Actions
    pub actions: Vec<PolicyAction>,
    /// Priority
    pub priority: u32,
    /// Enabled status
    pub enabled: bool,
}

/// Policy conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PolicyCondition {
    /// Risk score threshold
    RiskScoreAbove(f64),
    /// Specific risk level
    RiskLevel(RiskLevel),
    /// Factor-specific condition
    FactorCondition { factor_type: String, threshold: f64 },
    /// Time-based condition
    TimeCondition { start_hour: u32, end_hour: u32 },
}

/// Policy actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PolicyAction {
    /// Override decision
    OverrideDecision(RiskDecision),
    /// Adjust threshold
    AdjustThreshold {
        threshold_type: String,
        adjustment: f64,
    },
    /// Trigger alert
    TriggerAlert {
        severity: ThreatLevel,
        message: String,
    },
    /// Require approval
    RequireApproval { approver_role: String },
}

/// Risk engine trait
#[async_trait]
pub trait RiskEngineTrait: Send + Sync {
    /// Assess risk
    async fn assess_risk(&self, request: RiskAssessmentRequest) -> ZeroTrustResult<RiskAssessment>;

    /// Update risk history
    async fn update_history(&self, subject: RiskSubject, score: f64) -> ZeroTrustResult<()>;

    /// Get risk history
    async fn get_history(&self, subject: RiskSubject) -> ZeroTrustResult<RiskHistory>;

    /// Create risk policy
    async fn create_policy(&self, policy: RiskPolicy) -> ZeroTrustResult<()>;

    /// Evaluate policies
    async fn evaluate_policies(
        &self,
        assessment: &RiskAssessment,
    ) -> ZeroTrustResult<Vec<PolicyAction>>;

    /// Get risk trends
    async fn get_trends(
        &self,
        subject: RiskSubject,
        duration: Duration,
    ) -> ZeroTrustResult<Vec<(DateTime<Utc>, f64)>>;

    /// Calculate contextual risk
    async fn calculate_contextual_risk(
        &self,
        base_risk: f64,
        context: &RiskContext,
    ) -> ZeroTrustResult<f64>;

    /// Get risk recommendations
    async fn get_recommendations(
        &self,
        assessment: &RiskAssessment,
    ) -> ZeroTrustResult<Vec<String>>;
}

/// Risk engine implementation
pub struct RiskEngine {
    config: RiskEngineConfig,
    histories: Arc<DashMap<String, RiskHistory>>,
    policies: Arc<DashMap<Uuid, RiskPolicy>>,
    factor_weights: Arc<RwLock<HashMap<String, f64>>>,
    adaptive_thresholds: Arc<DashMap<String, f64>>,
}

impl RiskEngine {
    /// Create new risk engine
    pub fn new(config: RiskEngineConfig) -> ZeroTrustResult<Self> {
        let mut factor_weights = HashMap::new();

        // Default factor weights
        factor_weights.insert("identity".to_string(), 0.25);
        factor_weights.insert("device".to_string(), 0.20);
        factor_weights.insert("behavioral".to_string(), 0.20);
        factor_weights.insert("network".to_string(), 0.15);
        factor_weights.insert("temporal".to_string(), 0.10);
        factor_weights.insert("data_sensitivity".to_string(), 0.05);
        factor_weights.insert("environmental".to_string(), 0.05);

        Ok(Self {
            config,
            histories: Arc::new(DashMap::new()),
            policies: Arc::new(DashMap::new()),
            factor_weights: Arc::new(RwLock::new(factor_weights)),
            adaptive_thresholds: Arc::new(DashMap::new()),
        })
    }

    /// Get subject key
    fn get_subject_key(&self, subject: &RiskSubject) -> String {
        match subject {
            RiskSubject::User { user_id } => format!("user:{user_id}"),
            RiskSubject::Device { device_id } => format!("device:{device_id}"),
            RiskSubject::Service { service_id } => format!("service:{service_id}"),
            RiskSubject::Connection { connection_id } => format!("connection:{connection_id}"),
            RiskSubject::Combined { entity_ids } => {
                format!(
                    "combined:{}",
                    entity_ids
                        .iter()
                        .map(|id| id.to_string())
                        .collect::<Vec<_>>()
                        .join(",")
                )
            }
        }
    }

    /// Calculate factor risk score
    async fn calculate_factor_risk(&self, factor: &RiskFactor) -> (String, f64, String) {
        match factor {
            RiskFactor::Identity {
                verification_level,
                authentication_strength,
                account_age_days,
                failed_attempts,
            } => {
                let risk = 1.0 - (verification_level * 0.3 + authentication_strength * 0.3);
                let age_factor = (*account_age_days as f64 / 365.0).min(1.0);
                let risk = risk * (1.0 - age_factor * 0.2);
                let risk = risk + (*failed_attempts as f64 * 0.1).min(0.4);

                (
                    "identity".to_string(),
                    risk.min(1.0),
                    format!(
                        "Verification: {verification_level:.2}, Auth: {authentication_strength:.2}, Age: {account_age_days} days, Failed: {failed_attempts}"
                    ),
                )
            }

            RiskFactor::Device {
                trust_score,
                compliance_status,
                patch_level,
                jailbreak_detected,
            } => {
                let mut risk = 1.0 - trust_score;
                if !compliance_status {
                    risk += 0.3;
                }
                risk += (1.0 - patch_level) * 0.2;
                if *jailbreak_detected {
                    risk += 0.4;
                }

                (
                    "device".to_string(),
                    risk.min(1.0),
                    format!(
                        "Trust: {trust_score:.2}, Compliant: {compliance_status}, Patches: {patch_level:.2}, Jailbreak: {jailbreak_detected}"
                    ),
                )
            }

            RiskFactor::Behavioral {
                anomaly_score,
                deviation_from_baseline,
                suspicious_patterns,
            } => {
                let risk = anomaly_score * 0.5
                    + deviation_from_baseline * 0.3
                    + (*suspicious_patterns as f64 * 0.1).min(0.2);

                (
                    "behavioral".to_string(),
                    risk.min(1.0),
                    format!(
                        "Anomaly: {anomaly_score:.2}, Deviation: {deviation_from_baseline:.2}, Patterns: {suspicious_patterns}"
                    ),
                )
            }

            RiskFactor::Network {
                connection_security,
                location_trust,
                vpn_usage,
                threat_intelligence_score,
            } => {
                let mut risk = (1.0 - connection_security) * 0.3
                    + (1.0 - location_trust) * 0.3
                    + threat_intelligence_score * 0.4;
                if *vpn_usage {
                    risk *= 0.8; // VPN reduces risk
                }

                (
                    "network".to_string(),
                    risk.min(1.0),
                    format!(
                        "Security: {connection_security:.2}, Location: {location_trust:.2}, VPN: {vpn_usage}, Threat: {threat_intelligence_score:.2}"
                    ),
                )
            }

            RiskFactor::Temporal {
                unusual_time,
                frequency_anomaly,
                session_duration,
            } => {
                let mut risk = frequency_anomaly * 0.5;
                if *unusual_time {
                    risk += 0.3;
                }
                let duration_risk = if *session_duration > 8.0 { 0.2 } else { 0.0 };
                risk += duration_risk;

                (
                    "temporal".to_string(),
                    risk.min(1.0),
                    format!(
                        "Unusual: {unusual_time}, Frequency: {frequency_anomaly:.2}, Duration: {session_duration:.1}h"
                    ),
                )
            }

            RiskFactor::DataSensitivity {
                classification_level,
                volume,
                regulatory_requirements,
            } => {
                let classification_risk = *classification_level as u8 as f64 / 4.0;
                let volume_risk = (volume.log10() / 10.0).min(1.0);
                let regulatory_risk = (regulatory_requirements.len() as f64 * 0.1).min(0.3);
                let risk = classification_risk * 0.5 + volume_risk * 0.3 + regulatory_risk * 0.2;

                (
                    "data_sensitivity".to_string(),
                    risk.min(1.0),
                    format!(
                        "Class: {:?}, Volume: {:.0}, Regs: {}",
                        classification_level,
                        volume,
                        regulatory_requirements.len()
                    ),
                )
            }

            RiskFactor::Environmental {
                threat_level,
                ongoing_incidents,
                geographic_risk,
            } => {
                let threat_risk = *threat_level as u8 as f64 / 3.0;
                let incident_risk = (*ongoing_incidents as f64 * 0.1).min(0.3);
                let risk = threat_risk * 0.4 + incident_risk * 0.3 + geographic_risk * 0.3;

                (
                    "environmental".to_string(),
                    risk.min(1.0),
                    format!(
                        "Threat: {threat_level:?}, Incidents: {ongoing_incidents}, Geo: {geographic_risk:.2}"
                    ),
                )
            }
        }
    }

    /// Aggregate risk scores
    async fn aggregate_scores(&self, factor_scores: &[(String, f64, f64)]) -> f64 {
        match self.config.aggregation_method {
            AggregationMethod::Maximum => factor_scores
                .iter()
                .map(|(_, score, weight)| score * weight)
                .fold(0.0, f64::max),
            AggregationMethod::Average => {
                let sum: f64 = factor_scores
                    .iter()
                    .map(|(_, score, weight)| score * weight)
                    .sum();
                sum / factor_scores.len() as f64
            }
            AggregationMethod::WeightedAverage => {
                let weighted_sum: f64 = factor_scores
                    .iter()
                    .map(|(_, score, weight)| score * weight)
                    .sum();
                let weight_sum: f64 = factor_scores.iter().map(|(_, _, weight)| weight).sum();
                weighted_sum / weight_sum
            }
            AggregationMethod::Multiplicative => factor_scores
                .iter()
                .map(|(_, score, weight)| 1.0 - (1.0 - score) * weight)
                .product(),
        }
    }

    /// Determine risk level
    fn determine_risk_level(&self, risk_score: f64) -> RiskLevel {
        if risk_score >= 0.8 {
            RiskLevel::Critical
        } else if risk_score >= 0.6 {
            RiskLevel::High
        } else if risk_score >= 0.4 {
            RiskLevel::Medium
        } else {
            RiskLevel::Low
        }
    }

    /// Make risk decision
    fn make_decision(&self, risk_score: f64, context: &RiskContext) -> RiskDecision {
        if risk_score >= self.config.deny_threshold {
            RiskDecision::Deny
        } else if risk_score >= self.config.challenge_threshold {
            // Context-aware decision
            if context.action.contains("delete") || context.action.contains("admin") {
                RiskDecision::StepUpRequired
            } else {
                RiskDecision::ChallengeRequired
            }
        } else if risk_score >= 0.4 {
            RiskDecision::AllowWithMonitoring
        } else {
            RiskDecision::Allow
        }
    }
}

#[async_trait]
impl RiskEngineTrait for RiskEngine {
    async fn assess_risk(&self, request: RiskAssessmentRequest) -> ZeroTrustResult<RiskAssessment> {
        let assessment_id = Uuid::new_v4();
        let weights = self.factor_weights.read().await;

        // Calculate individual factor risks
        let mut factor_scores = vec![];
        let mut contributing_factors = vec![];

        for factor in &request.factors {
            let (factor_type, score, details) = self.calculate_factor_risk(factor).await;
            let weight = weights.get(&factor_type).copied().unwrap_or(0.1);

            factor_scores.push((factor_type.clone(), score, weight));
            contributing_factors.push(RiskContribution {
                factor: factor_type,
                contribution: score * weight,
                weight,
                details,
            });
        }

        // Aggregate scores
        let mut risk_score = self.aggregate_scores(&factor_scores).await;

        // Apply contextual adjustments
        if self.config.contextual_evaluation {
            risk_score = self
                .calculate_contextual_risk(risk_score, &request.context)
                .await?;
        }

        // Apply historical adjustments
        if let Some(history) = &request.history {
            if history.average_risk > 0.7 {
                risk_score *= 1.1; // Increase risk for historically risky subjects
            }
            if !history.incidents.is_empty() {
                risk_score *= 1.0 + (history.incidents.len() as f64 * 0.05).min(0.3);
            }
        }

        // Cap at maximum
        risk_score = risk_score.min(self.config.max_risk_score);

        // Determine risk level and decision
        let risk_level = self.determine_risk_level(risk_score);
        let recommendation = self.make_decision(risk_score, &request.context);

        // Generate mitigations
        let mitigations = self
            .get_recommendations(&RiskAssessment {
                assessment_id,
                risk_score,
                risk_level,
                contributing_factors: contributing_factors.clone(),
                recommendation,
                mitigations: vec![],
                timestamp: Utc::now(),
                confidence: 0.9,
            })
            .await?;

        // Update history
        self.update_history(request.subject.clone(), risk_score)
            .await?;

        Ok(RiskAssessment {
            assessment_id,
            risk_score,
            risk_level,
            contributing_factors,
            recommendation,
            mitigations,
            timestamp: Utc::now(),
            confidence: 0.9, // Could be calculated based on data quality
        })
    }

    async fn update_history(&self, subject: RiskSubject, score: f64) -> ZeroTrustResult<()> {
        let key = self.get_subject_key(&subject);
        let now = Utc::now();

        self.histories
            .entry(key)
            .and_modify(|history| {
                history.scores.push_front((now, score));

                // Maintain history size
                let cutoff = now - self.config.history_retention;
                history.scores.retain(|(timestamp, _)| *timestamp > cutoff);

                // Update statistics
                if !history.scores.is_empty() {
                    let sum: f64 = history.scores.iter().map(|(_, s)| s).sum();
                    history.average_risk = sum / history.scores.len() as f64;
                    history.peak_risk = history.scores.iter().map(|(_, s)| *s).fold(0.0, f64::max);
                }
            })
            .or_insert_with(|| RiskHistory {
                scores: VecDeque::from(vec![(now, score)]),
                incidents: vec![],
                average_risk: score,
                peak_risk: score,
            });

        Ok(())
    }

    async fn get_history(&self, subject: RiskSubject) -> ZeroTrustResult<RiskHistory> {
        let key = self.get_subject_key(&subject);

        self.histories
            .get(&key)
            .map(|entry| entry.clone())
            .ok_or_else(|| ZeroTrustError::RiskScoreTooHigh {
                score: 0.0,
                threshold: 0.0,
            })
    }

    async fn create_policy(&self, policy: RiskPolicy) -> ZeroTrustResult<()> {
        self.policies.insert(policy.policy_id, policy);
        Ok(())
    }

    async fn evaluate_policies(
        &self,
        assessment: &RiskAssessment,
    ) -> ZeroTrustResult<Vec<PolicyAction>> {
        let mut actions = vec![];

        let mut policies: Vec<_> = self.policies.iter().filter(|p| p.enabled).collect();

        policies.sort_by_key(|p| p.priority);

        for policy_ref in policies {
            let policy = policy_ref.value();
            let mut conditions_met = true;

            for condition in &policy.conditions {
                match condition {
                    PolicyCondition::RiskScoreAbove(threshold) => {
                        if assessment.risk_score <= *threshold {
                            conditions_met = false;
                            break;
                        }
                    }
                    PolicyCondition::RiskLevel(level) => {
                        if assessment.risk_level != *level {
                            conditions_met = false;
                            break;
                        }
                    }
                    PolicyCondition::FactorCondition {
                        factor_type,
                        threshold,
                    } => {
                        let factor_score = assessment
                            .contributing_factors
                            .iter()
                            .find(|f| f.factor == *factor_type)
                            .map(|f| f.contribution)
                            .unwrap_or(0.0);
                        if factor_score <= *threshold {
                            conditions_met = false;
                            break;
                        }
                    }
                    PolicyCondition::TimeCondition {
                        start_hour,
                        end_hour,
                    } => {
                        let current_hour = assessment.timestamp.hour();
                        if current_hour < *start_hour || current_hour > *end_hour {
                            conditions_met = false;
                            break;
                        }
                    }
                }
            }

            if conditions_met {
                actions.extend(policy.actions.clone());
            }
        }

        Ok(actions)
    }

    async fn get_trends(
        &self,
        subject: RiskSubject,
        duration: Duration,
    ) -> ZeroTrustResult<Vec<(DateTime<Utc>, f64)>> {
        let history = self.get_history(subject).await?;
        let cutoff = Utc::now() - duration;

        Ok(history
            .scores
            .into_iter()
            .filter(|(timestamp, _)| *timestamp > cutoff)
            .collect())
    }

    async fn calculate_contextual_risk(
        &self,
        base_risk: f64,
        context: &RiskContext,
    ) -> ZeroTrustResult<f64> {
        let mut adjusted_risk = base_risk;

        // Time-based adjustments
        let hour = context.timestamp.hour();
        if !(6..=22).contains(&hour) {
            adjusted_risk *= 1.2; // Higher risk during off-hours
        }

        // Action-based adjustments
        if context.action.contains("delete") || context.action.contains("modify") {
            adjusted_risk *= 1.15;
        } else if context.action.contains("read") {
            adjusted_risk *= 0.95;
        }

        // Resource-based adjustments
        if context.resource.contains("admin") || context.resource.contains("system") {
            adjusted_risk *= 1.25;
        }

        // Location-based adjustments
        if let Some(location) = &context.source_location {
            if location.contains("external") || location.contains("public") {
                adjusted_risk *= 1.3;
            }
        }

        Ok(adjusted_risk.min(self.config.max_risk_score))
    }

    async fn get_recommendations(
        &self,
        assessment: &RiskAssessment,
    ) -> ZeroTrustResult<Vec<String>> {
        let mut recommendations = vec![];

        // Analyze contributing factors
        for factor in &assessment.contributing_factors {
            match factor.factor.as_str() {
                "identity" if factor.contribution > 0.2 => {
                    recommendations.push(
                        "Strengthen identity verification with additional factors".to_string(),
                    );
                    recommendations.push("Consider requiring biometric authentication".to_string());
                }
                "device" if factor.contribution > 0.2 => {
                    recommendations.push("Ensure device compliance before access".to_string());
                    recommendations.push("Update device security patches".to_string());
                }
                "behavioral" if factor.contribution > 0.2 => {
                    recommendations.push("Review recent user activities for anomalies".to_string());
                    recommendations.push("Consider user security awareness training".to_string());
                }
                "network" if factor.contribution > 0.2 => {
                    recommendations
                        .push("Require VPN connection for sensitive resources".to_string());
                    recommendations.push("Implement network segmentation".to_string());
                }
                _ => {}
            }
        }

        // Decision-based recommendations
        match assessment.recommendation {
            RiskDecision::ChallengeRequired => {
                recommendations.push("Implement step-up authentication".to_string());
            }
            RiskDecision::AllowWithMonitoring => {
                recommendations.push("Enable detailed activity logging".to_string());
                recommendations
                    .push("Set up real-time alerts for suspicious activities".to_string());
            }
            RiskDecision::Deny => {
                recommendations.push("Review and update access policies".to_string());
                recommendations.push("Consider implementing conditional access".to_string());
            }
            _ => {}
        }

        Ok(recommendations)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_risk_assessment_basic() {
        let config = RiskEngineConfig::default();
        let engine = RiskEngine::new(config)?;

        let request = RiskAssessmentRequest {
            subject: RiskSubject::User {
                user_id: Uuid::new_v4(),
            },
            context: RiskContext {
                action: "read".to_string(),
                resource: "documents".to_string(),
                timestamp: Utc::now(),
                source_location: Some("internal".to_string()),
                metadata: serde_json::json!({}),
            },
            factors: vec![
                RiskFactor::Identity {
                    verification_level: 0.9,
                    authentication_strength: 0.8,
                    account_age_days: 365,
                    failed_attempts: 0,
                },
                RiskFactor::Device {
                    trust_score: 0.85,
                    compliance_status: true,
                    patch_level: 0.95,
                    jailbreak_detected: false,
                },
            ],
            history: None,
        };

        let assessment = engine.assess_risk(request).await?;
        assert!(assessment.risk_score < 0.5);
        assert_eq!(assessment.risk_level, RiskLevel::Low);
        assert_eq!(assessment.recommendation, RiskDecision::Allow);
    }

    #[tokio::test]
    async fn test_high_risk_assessment() {
        let config = RiskEngineConfig::default();
        let engine = RiskEngine::new(config)?;

        let request = RiskAssessmentRequest {
            subject: RiskSubject::User {
                user_id: Uuid::new_v4(),
            },
            context: RiskContext {
                action: "delete".to_string(),
                resource: "admin/users".to_string(),
                timestamp: Utc::now().with_hour(3)?, // 3 AM
                source_location: Some("external".to_string()),
                metadata: serde_json::json!({}),
            },
            factors: vec![
                RiskFactor::Identity {
                    verification_level: 0.3,
                    authentication_strength: 0.4,
                    account_age_days: 5,
                    failed_attempts: 3,
                },
                RiskFactor::Device {
                    trust_score: 0.2,
                    compliance_status: false,
                    patch_level: 0.3,
                    jailbreak_detected: true,
                },
                RiskFactor::Behavioral {
                    anomaly_score: 0.9,
                    deviation_from_baseline: 0.8,
                    suspicious_patterns: 5,
                },
            ],
            history: None,
        };

        let assessment = engine.assess_risk(request).await?;
        assert!(assessment.risk_score > 0.8);
        assert_eq!(assessment.risk_level, RiskLevel::Critical);
        assert_eq!(assessment.recommendation, RiskDecision::Deny);
    }

    #[tokio::test]
    async fn test_contextual_risk_adjustment() {
        let config = RiskEngineConfig::default();
        let engine = RiskEngine::new(config)?;

        let base_risk = 0.5;

        // High-risk context
        let high_risk_context = RiskContext {
            action: "delete".to_string(),
            resource: "admin/system".to_string(),
            timestamp: Utc::now().with_hour(2)?,
            source_location: Some("external".to_string()),
            metadata: serde_json::json!({}),
        };

        let adjusted_risk = engine
            .calculate_contextual_risk(base_risk, &high_risk_context)
            .await
            .unwrap();
        assert!(adjusted_risk > base_risk);

        // Low-risk context
        let low_risk_context = RiskContext {
            action: "read".to_string(),
            resource: "public/docs".to_string(),
            timestamp: Utc::now().with_hour(14)?,
            source_location: Some("internal".to_string()),
            metadata: serde_json::json!({}),
        };

        let adjusted_risk = engine
            .calculate_contextual_risk(base_risk, &low_risk_context)
            .await
            .unwrap();
        assert!(adjusted_risk < base_risk);
    }

    #[tokio::test]
    async fn test_risk_history() {
        let config = RiskEngineConfig::default();
        let engine = RiskEngine::new(config)?;

        let subject = RiskSubject::User {
            user_id: Uuid::new_v4(),
        };

        // Update history multiple times
        for i in 0..5 {
            let score = 0.3 + (i as f64 * 0.1);
            engine.update_history(subject.clone(), score).await?;
        }

        let history = engine.get_history(subject).await?;
        assert_eq!(history.scores.len(), 5);
        assert!(history.average_risk > 0.0);
        assert_eq!(history.peak_risk, 0.7);
    }

    #[tokio::test]
    async fn test_policy_evaluation() {
        let config = RiskEngineConfig::default();
        let engine = RiskEngine::new(config)?;

        // Create a policy
        let policy = RiskPolicy {
            policy_id: Uuid::new_v4(),
            name: "High Risk Alert".to_string(),
            conditions: vec![PolicyCondition::RiskScoreAbove(0.7)],
            actions: vec![
                PolicyAction::TriggerAlert {
                    severity: ThreatLevel::High,
                    message: "High risk activity detected".to_string(),
                },
                PolicyAction::OverrideDecision(RiskDecision::StepUpRequired),
            ],
            priority: 1,
            enabled: true,
        };

        engine.create_policy(policy).await?;

        // Create assessment
        let assessment = RiskAssessment {
            assessment_id: Uuid::new_v4(),
            risk_score: 0.8,
            risk_level: RiskLevel::High,
            contributing_factors: vec![],
            recommendation: RiskDecision::ChallengeRequired,
            mitigations: vec![],
            timestamp: Utc::now(),
            confidence: 0.9,
        };

        let actions = engine.evaluate_policies(&assessment).await?;
        assert_eq!(actions.len(), 2);
        assert!(actions
            .iter()
            .any(|a| matches!(a, PolicyAction::TriggerAlert { .. })));
    }

    #[tokio::test]
    async fn test_aggregation_methods() {
        let config_max = RiskEngineConfig {
            aggregation_method: AggregationMethod::Maximum,
            ..Default::default()
        };
        let engine_max = RiskEngine::new(config_max)?;

        let factor_scores = vec![
            ("factor1".to_string(), 0.3, 1.0),
            ("factor2".to_string(), 0.7, 1.0),
            ("factor3".to_string(), 0.5, 1.0),
        ];

        let max_score = engine_max.aggregate_scores(&factor_scores).await;
        assert_eq!(max_score, 0.7);

        let config_avg = RiskEngineConfig {
            aggregation_method: AggregationMethod::Average,
            ..Default::default()
        };
        let engine_avg = RiskEngine::new(config_avg)?;

        let avg_score = engine_avg.aggregate_scores(&factor_scores).await;
        assert!((avg_score - 0.5).abs() < 0.01);
    }

    #[tokio::test]
    async fn test_data_sensitivity_factor() {
        let config = RiskEngineConfig::default();
        let engine = RiskEngine::new(config)?;

        let request = RiskAssessmentRequest {
            subject: RiskSubject::User {
                user_id: Uuid::new_v4(),
            },
            context: RiskContext {
                action: "download".to_string(),
                resource: "classified/documents".to_string(),
                timestamp: Utc::now(),
                source_location: None,
                metadata: serde_json::json!({}),
            },
            factors: vec![RiskFactor::DataSensitivity {
                classification_level: DataClassification::TopSecret,
                volume: 1000000.0, // 1GB
                regulatory_requirements: vec!["GDPR".to_string(), "HIPAA".to_string()],
            }],
            history: None,
        };

        let assessment = engine.assess_risk(request).await?;
        assert!(assessment.risk_score > 0.5);
    }

    #[tokio::test]
    async fn test_environmental_factor() {
        let config = RiskEngineConfig::default();
        let engine = RiskEngine::new(config)?;

        let request = RiskAssessmentRequest {
            subject: RiskSubject::Device {
                device_id: Uuid::new_v4(),
            },
            context: RiskContext {
                action: "connect".to_string(),
                resource: "network".to_string(),
                timestamp: Utc::now(),
                source_location: Some("high-risk-country".to_string()),
                metadata: serde_json::json!({}),
            },
            factors: vec![RiskFactor::Environmental {
                threat_level: ThreatLevel::Critical,
                ongoing_incidents: 5,
                geographic_risk: 0.9,
            }],
            history: None,
        };

        let assessment = engine.assess_risk(request).await?;
        assert!(assessment.risk_score > 0.7);
        assert_eq!(assessment.risk_level, RiskLevel::High);
    }

    #[tokio::test]
    async fn test_risk_trends() {
        let config = RiskEngineConfig::default();
        let engine = RiskEngine::new(config)?;

        let subject = RiskSubject::User {
            user_id: Uuid::new_v4(),
        };

        // Create historical data
        for i in 0..10 {
            let score = 0.5 + (i as f64 * 0.05);
            engine.update_history(subject.clone(), score).await?;
            tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        }

        let trends = engine
            .get_trends(subject, Duration::hours(1))
            .await
            .unwrap();
        assert_eq!(trends.len(), 10);

        // Verify ascending trend
        let scores: Vec<f64> = trends.iter().map(|(_, score)| *score).collect();
        for i in 1..scores.len() {
            assert!(scores[i] >= scores[i - 1]);
        }
    }

    #[tokio::test]
    async fn test_recommendations() {
        let config = RiskEngineConfig::default();
        let engine = RiskEngine::new(config)?;

        let assessment = RiskAssessment {
            assessment_id: Uuid::new_v4(),
            risk_score: 0.7,
            risk_level: RiskLevel::High,
            contributing_factors: vec![
                RiskContribution {
                    factor: "identity".to_string(),
                    contribution: 0.3,
                    weight: 0.25,
                    details: "Weak authentication".to_string(),
                },
                RiskContribution {
                    factor: "network".to_string(),
                    contribution: 0.25,
                    weight: 0.15,
                    details: "Untrusted network".to_string(),
                },
            ],
            recommendation: RiskDecision::ChallengeRequired,
            mitigations: vec![],
            timestamp: Utc::now(),
            confidence: 0.9,
        };

        let recommendations = engine.get_recommendations(&assessment).await?;
        assert!(!recommendations.is_empty());
        assert!(recommendations.iter().any(|r| r.contains("identity")));
        assert!(recommendations
            .iter()
            .any(|r| r.contains("VPN") || r.contains("network")));
    }

    #[tokio::test]
    async fn test_combined_subject_risk() {
        let config = RiskEngineConfig::default();
        let engine = RiskEngine::new(config)?;

        let user_id = Uuid::new_v4();
        let device_id = Uuid::new_v4();

        let request = RiskAssessmentRequest {
            subject: RiskSubject::Combined {
                entity_ids: vec![user_id, device_id],
            },
            context: RiskContext {
                action: "access".to_string(),
                resource: "sensitive-data".to_string(),
                timestamp: Utc::now(),
                source_location: None,
                metadata: serde_json::json!({}),
            },
            factors: vec![
                RiskFactor::Identity {
                    verification_level: 0.7,
                    authentication_strength: 0.6,
                    account_age_days: 30,
                    failed_attempts: 1,
                },
                RiskFactor::Device {
                    trust_score: 0.6,
                    compliance_status: true,
                    patch_level: 0.8,
                    jailbreak_detected: false,
                },
            ],
            history: None,
        };

        let assessment = engine.assess_risk(request).await?;
        assert!(assessment.risk_score > 0.3);
        assert!(assessment.risk_score < 0.6);
    }
}
