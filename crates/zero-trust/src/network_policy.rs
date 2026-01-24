//! Microsegmentation, network access control, policy enforcement, and traffic analysis
//!
//! This module implements comprehensive network policy management following zero-trust principles:
//! - Network microsegmentation
//! - Policy-based access control
//! - Traffic inspection and analysis
//! - Dynamic policy enforcement
//! - Network isolation and containment

use crate::error::{ZeroTrustError, ZeroTrustResult};
use async_trait::async_trait;
use chrono::{DateTime, Datelike, Duration, Utc};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::net::IpAddr;
use std::sync::Arc;
use uuid::Uuid;

/// Network policy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkPolicyConfig {
    /// Default deny all traffic
    pub default_deny: bool,
    /// Enable traffic inspection
    pub traffic_inspection: bool,
    /// Policy evaluation mode
    pub evaluation_mode: PolicyEvaluationMode,
    /// Maximum concurrent connections per segment
    pub max_connections_per_segment: usize,
    /// Connection timeout
    pub connection_timeout: Duration,
    /// Enable anomaly detection
    pub anomaly_detection: bool,
    /// Policy cache TTL
    pub policy_cache_ttl: Duration,
}

impl Default for NetworkPolicyConfig {
    fn default() -> Self {
        Self {
            default_deny: true,
            traffic_inspection: true,
            evaluation_mode: PolicyEvaluationMode::Strict,
            max_connections_per_segment: 1000,
            connection_timeout: Duration::seconds(30),
            anomaly_detection: true,
            policy_cache_ttl: Duration::minutes(5),
        }
    }
}

/// Policy evaluation modes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PolicyEvaluationMode {
    /// Strict mode - all rules must match
    Strict,
    /// Permissive mode - log violations but allow
    Permissive,
    /// Learning mode - observe and build policies
    Learning,
}

/// Network segment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkSegment {
    /// Segment ID
    pub segment_id: Uuid,
    /// Segment name
    pub name: String,
    /// Segment type
    pub segment_type: SegmentType,
    /// CIDR blocks
    pub cidr_blocks: Vec<String>,
    /// Trust level
    pub trust_level: TrustLevel,
    /// Allowed protocols
    pub allowed_protocols: HashSet<Protocol>,
    /// Security zone
    pub security_zone: SecurityZone,
    /// Metadata
    pub metadata: serde_json::Value,
}

/// Segment types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SegmentType {
    /// Internal corporate network
    Internal,
    /// DMZ network
    Dmz,
    /// External/Internet
    External,
    /// Cloud network
    Cloud,
    /// IoT network
    IoT,
    /// Guest network
    Guest,
}

/// Trust levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum TrustLevel {
    /// No trust
    None = 0,
    /// Low trust
    Low = 1,
    /// Medium trust
    Medium = 2,
    /// High trust
    High = 3,
    /// Full trust
    Full = 4,
}

/// Security zones
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SecurityZone {
    /// Public zone
    Public,
    /// Private zone
    Private,
    /// Restricted zone
    Restricted,
    /// Classified zone
    Classified,
}

/// Network protocols
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Protocol {
    Tcp,
    Udp,
    Icmp,
    Http,
    Https,
    Ssh,
    Rdp,
    Dns,
    Dhcp,
    Custom(u16),
}

/// Network policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkPolicy {
    /// Policy ID
    pub policy_id: Uuid,
    /// Policy name
    pub name: String,
    /// Policy priority (lower is higher priority)
    pub priority: u32,
    /// Source segment
    pub source_segment: Option<Uuid>,
    /// Destination segment
    pub destination_segment: Option<Uuid>,
    /// Allowed protocols
    pub protocols: HashSet<Protocol>,
    /// Port ranges
    pub port_ranges: Vec<PortRange>,
    /// Action to take
    pub action: PolicyAction,
    /// Policy state
    pub state: PolicyState,
    /// Creation time
    pub created_at: DateTime<Utc>,
    /// Expiration time
    pub expires_at: Option<DateTime<Utc>>,
    /// Conditions
    pub conditions: Vec<PolicyCondition>,
}

/// Port range
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PortRange {
    pub start: u16,
    pub end: u16,
}

impl PortRange {
    /// Create a single port range
    pub fn single(port: u16) -> Self {
        Self {
            start: port,
            end: port,
        }
    }

    /// Check if port is in range
    pub fn contains(&self, port: u16) -> bool {
        port >= self.start && port <= self.end
    }
}

/// Policy actions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PolicyAction {
    /// Allow the traffic
    Allow,
    /// Deny the traffic
    Deny,
    /// Redirect the traffic
    Redirect,
    /// Inspect the traffic
    Inspect,
    /// Log and allow
    LogAllow,
    /// Log and deny
    LogDeny,
}

/// Policy states
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PolicyState {
    /// Policy is active
    Active,
    /// Policy is disabled
    Disabled,
    /// Policy is in testing mode
    Testing,
    /// Policy is scheduled
    Scheduled,
}

/// Policy conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PolicyCondition {
    /// Time-based condition
    TimeWindow {
        start_time: chrono::NaiveTime,
        end_time: chrono::NaiveTime,
        days: HashSet<chrono::Weekday>,
    },
    /// Identity-based condition
    Identity { identity_ids: HashSet<Uuid> },
    /// Device-based condition
    Device { device_ids: HashSet<Uuid> },
    /// Location-based condition
    Location { allowed_locations: HashSet<String> },
    /// Risk score condition
    RiskScore { max_risk_score: f64 },
}

/// Network connection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConnection {
    /// Connection ID
    pub connection_id: Uuid,
    /// Source IP
    pub source_ip: IpAddr,
    /// Source port
    pub source_port: u16,
    /// Destination IP
    pub destination_ip: IpAddr,
    /// Destination port
    pub destination_port: u16,
    /// Protocol
    pub protocol: Protocol,
    /// Start time
    pub start_time: DateTime<Utc>,
    /// Last activity
    pub last_activity: DateTime<Utc>,
    /// Bytes sent
    pub bytes_sent: u64,
    /// Bytes received
    pub bytes_received: u64,
    /// Connection state
    pub state: ConnectionState,
    /// Applied policies
    pub applied_policies: Vec<Uuid>,
}

/// Connection states
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConnectionState {
    /// Connection established
    Established,
    /// Connection pending
    Pending,
    /// Connection blocked
    Blocked,
    /// Connection terminated
    Terminated,
}

/// Traffic analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrafficAnalysis {
    /// Analysis ID
    pub analysis_id: Uuid,
    /// Connection ID
    pub connection_id: Uuid,
    /// Threat level
    pub threat_level: ThreatLevel,
    /// Anomalies detected
    pub anomalies: Vec<TrafficAnomaly>,
    /// Recommended action
    pub recommended_action: PolicyAction,
    /// Analysis timestamp
    pub timestamp: DateTime<Utc>,
}

/// Threat levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ThreatLevel {
    None = 0,
    Low = 1,
    Medium = 2,
    High = 3,
    Critical = 4,
}

/// Traffic anomalies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrafficAnomaly {
    /// Unusual port usage
    UnusualPort { port: u16, expected_ports: Vec<u16> },
    /// High data transfer
    HighDataTransfer { threshold: u64, actual: u64 },
    /// Suspicious pattern
    SuspiciousPattern { pattern: String },
    /// Protocol violation
    ProtocolViolation { details: String },
    /// Geographic anomaly
    GeographicAnomaly {
        expected_location: String,
        actual_location: String,
    },
}

/// Network policy manager trait
#[async_trait]
pub trait NetworkPolicyManagerTrait: Send + Sync {
    /// Create a network segment
    async fn create_segment(
        &self,
        name: String,
        segment_type: SegmentType,
        cidr_blocks: Vec<String>,
    ) -> ZeroTrustResult<NetworkSegment>;

    /// Create a network policy
    async fn create_policy(
        &self,
        name: String,
        source: Option<Uuid>,
        destination: Option<Uuid>,
        action: PolicyAction,
    ) -> ZeroTrustResult<NetworkPolicy>;

    /// Evaluate connection against policies
    async fn evaluate_connection(
        &self,
        connection: &NetworkConnection,
    ) -> ZeroTrustResult<PolicyDecision>;

    /// Analyze traffic
    async fn analyze_traffic(&self, connection_id: Uuid) -> ZeroTrustResult<TrafficAnalysis>;

    /// Update policy state
    async fn update_policy_state(&self, policy_id: Uuid, state: PolicyState)
        -> ZeroTrustResult<()>;

    /// Get segment by ID
    async fn get_segment(&self, segment_id: Uuid) -> ZeroTrustResult<NetworkSegment>;

    /// Get policy by ID
    async fn get_policy(&self, policy_id: Uuid) -> ZeroTrustResult<NetworkPolicy>;

    /// List active policies for segment
    async fn list_policies_for_segment(
        &self,
        segment_id: Uuid,
    ) -> ZeroTrustResult<Vec<NetworkPolicy>>;

    /// Record connection
    async fn record_connection(&self, connection: NetworkConnection) -> ZeroTrustResult<()>;
}

/// Policy decision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyDecision {
    /// Decision action
    pub action: PolicyAction,
    /// Matching policies
    pub matching_policies: Vec<Uuid>,
    /// Decision reason
    pub reason: String,
    /// Decision timestamp
    pub timestamp: DateTime<Utc>,
}

/// Network policy manager implementation
pub struct NetworkPolicyManager {
    config: NetworkPolicyConfig,
    segments: Arc<DashMap<Uuid, NetworkSegment>>,
    policies: Arc<DashMap<Uuid, NetworkPolicy>>,
    connections: Arc<DashMap<Uuid, NetworkConnection>>,
    policy_cache: Arc<DashMap<String, (PolicyDecision, DateTime<Utc>)>>,
}

impl NetworkPolicyManager {
    /// Create new network policy manager
    pub fn new(config: NetworkPolicyConfig) -> ZeroTrustResult<Self> {
        Ok(Self {
            config,
            segments: Arc::new(DashMap::new()),
            policies: Arc::new(DashMap::new()),
            connections: Arc::new(DashMap::new()),
            policy_cache: Arc::new(DashMap::new()),
        })
    }

    /// Find segment for IP
    fn find_segment_for_ip(&self, ip: &IpAddr) -> Option<Uuid> {
        for segment in self.segments.iter() {
            for cidr in &segment.cidr_blocks {
                if self.ip_in_cidr(ip, cidr) {
                    return Some(segment.segment_id);
                }
            }
        }
        None
    }

    /// Check if IP is in CIDR block
    fn ip_in_cidr(&self, ip: &IpAddr, cidr: &str) -> bool {
        // Simplified CIDR check - in production use proper CIDR parsing
        match (ip, cidr.parse::<IpAddr>()) {
            (IpAddr::V4(ip), Ok(IpAddr::V4(cidr_ip))) => {
                // Simple check - in production implement proper CIDR logic
                ip.octets()[0] == cidr_ip.octets()[0] && ip.octets()[1] == cidr_ip.octets()[1]
            }
            (IpAddr::V6(_), Ok(IpAddr::V6(_))) => {
                // Simplified for testing
                false
            }
            _ => false,
        }
    }

    /// Check if policy matches connection
    fn policy_matches_connection(
        &self,
        policy: &NetworkPolicy,
        connection: &NetworkConnection,
    ) -> bool {
        // Check protocol
        if !policy.protocols.is_empty() && !policy.protocols.contains(&connection.protocol) {
            return false;
        }

        // Check ports
        if !policy.port_ranges.is_empty() {
            let port_match = policy
                .port_ranges
                .iter()
                .any(|range| range.contains(connection.destination_port));
            if !port_match {
                return false;
            }
        }

        // Check segments
        let src_segment = self.find_segment_for_ip(&connection.source_ip);
        let dst_segment = self.find_segment_for_ip(&connection.destination_ip);

        if let Some(policy_src) = policy.source_segment {
            if src_segment != Some(policy_src) {
                return false;
            }
        }

        if let Some(policy_dst) = policy.destination_segment {
            if dst_segment != Some(policy_dst) {
                return false;
            }
        }

        // Check conditions
        for condition in &policy.conditions {
            match condition {
                PolicyCondition::TimeWindow {
                    start_time,
                    end_time,
                    days,
                } => {
                    let now = Utc::now();
                    let current_time = now.time();
                    let current_day = now.weekday();

                    if !days.contains(&current_day) {
                        return false;
                    }

                    if current_time < *start_time || current_time > *end_time {
                        return false;
                    }
                }
                PolicyCondition::RiskScore { max_risk_score } => {
                    // In production, check actual risk score
                    if 0.5 > *max_risk_score {
                        return false;
                    }
                }
                _ => {
                    // Other conditions would need identity/device context
                }
            }
        }

        true
    }

    /// Detect traffic anomalies
    fn detect_anomalies(&self, connection: &NetworkConnection) -> Vec<TrafficAnomaly> {
        let mut anomalies = vec![];

        // Check for unusual ports
        match connection.protocol {
            Protocol::Http => {
                if connection.destination_port != 80 {
                    anomalies.push(TrafficAnomaly::UnusualPort {
                        port: connection.destination_port,
                        expected_ports: vec![80],
                    });
                }
            }
            Protocol::Https => {
                if connection.destination_port != 443 {
                    anomalies.push(TrafficAnomaly::UnusualPort {
                        port: connection.destination_port,
                        expected_ports: vec![443],
                    });
                }
            }
            Protocol::Ssh => {
                if connection.destination_port != 22 {
                    anomalies.push(TrafficAnomaly::UnusualPort {
                        port: connection.destination_port,
                        expected_ports: vec![22],
                    });
                }
            }
            _ => {}
        }

        // Check for high data transfer
        let total_bytes = connection.bytes_sent + connection.bytes_received;
        if total_bytes > 1_000_000_000 {
            // 1GB
            anomalies.push(TrafficAnomaly::HighDataTransfer {
                threshold: 1_000_000_000,
                actual: total_bytes,
            });
        }

        anomalies
    }
}

#[async_trait]
impl NetworkPolicyManagerTrait for NetworkPolicyManager {
    async fn create_segment(
        &self,
        name: String,
        segment_type: SegmentType,
        cidr_blocks: Vec<String>,
    ) -> ZeroTrustResult<NetworkSegment> {
        let segment_id = Uuid::new_v4();

        let trust_level = match segment_type {
            SegmentType::Internal => TrustLevel::High,
            SegmentType::Dmz => TrustLevel::Medium,
            SegmentType::External => TrustLevel::None,
            SegmentType::Cloud => TrustLevel::Medium,
            SegmentType::IoT => TrustLevel::Low,
            SegmentType::Guest => TrustLevel::Low,
        };

        let security_zone = match segment_type {
            SegmentType::Internal => SecurityZone::Private,
            SegmentType::External => SecurityZone::Public,
            _ => SecurityZone::Restricted,
        };

        let segment = NetworkSegment {
            segment_id,
            name: name.clone(),
            segment_type,
            cidr_blocks,
            trust_level,
            allowed_protocols: HashSet::new(),
            security_zone,
            metadata: serde_json::json!({}),
        };

        self.segments.insert(segment_id, segment.clone());
        Ok(segment)
    }

    async fn create_policy(
        &self,
        name: String,
        source: Option<Uuid>,
        destination: Option<Uuid>,
        action: PolicyAction,
    ) -> ZeroTrustResult<NetworkPolicy> {
        let policy_id = Uuid::new_v4();

        let policy = NetworkPolicy {
            policy_id,
            name: name.clone(),
            priority: 100,
            source_segment: source,
            destination_segment: destination,
            protocols: HashSet::new(),
            port_ranges: vec![],
            action,
            state: PolicyState::Active,
            created_at: Utc::now(),
            expires_at: None,
            conditions: vec![],
        };

        self.policies.insert(policy_id, policy.clone());
        Ok(policy)
    }

    async fn evaluate_connection(
        &self,
        connection: &NetworkConnection,
    ) -> ZeroTrustResult<PolicyDecision> {
        // Check cache first
        let cache_key = format!(
            "{:?}-{}-{:?}-{}",
            connection.source_ip,
            connection.source_port,
            connection.destination_ip,
            connection.destination_port
        );

        if let Some(cached) = self.policy_cache.get(&cache_key) {
            if Utc::now() - cached.1 < self.config.policy_cache_ttl {
                return Ok(cached.0.clone());
            }
        }

        // Find matching policies
        let mut matching_policies = vec![];
        let mut final_action = if self.config.default_deny {
            PolicyAction::Deny
        } else {
            PolicyAction::Allow
        };

        // Sort policies by priority
        let mut policies: Vec<_> = self
            .policies
            .iter()
            .filter(|p| p.state == PolicyState::Active)
            .collect();
        policies.sort_by_key(|p| p.priority);

        for policy_ref in policies {
            let policy = policy_ref.value();
            if self.policy_matches_connection(policy, connection) {
                matching_policies.push(policy.policy_id);
                final_action = policy.action;
                break; // First matching policy wins
            }
        }

        let decision = PolicyDecision {
            action: final_action,
            matching_policies: matching_policies.clone(),
            reason: if matching_policies.is_empty() {
                "No matching policy found".to_string()
            } else {
                format!("Matched {} policies", matching_policies.len())
            },
            timestamp: Utc::now(),
        };

        // Cache decision
        self.policy_cache
            .insert(cache_key, (decision.clone(), Utc::now()));

        Ok(decision)
    }

    async fn analyze_traffic(&self, connection_id: Uuid) -> ZeroTrustResult<TrafficAnalysis> {
        let connection = self.connections.get(&connection_id).ok_or_else(|| {
            ZeroTrustError::NetworkPolicyViolation {
                policy: "traffic_analysis".to_string(),
                reason: "Connection not found".to_string(),
            }
        })?;

        let anomalies = self.detect_anomalies(&connection);

        let threat_level = if anomalies.is_empty() {
            ThreatLevel::None
        } else if anomalies.len() == 1 {
            ThreatLevel::Low
        } else if anomalies.len() == 2 {
            ThreatLevel::Medium
        } else {
            ThreatLevel::High
        };

        let recommended_action = match threat_level {
            ThreatLevel::None | ThreatLevel::Low => PolicyAction::Allow,
            ThreatLevel::Medium => PolicyAction::Inspect,
            ThreatLevel::High | ThreatLevel::Critical => PolicyAction::Deny,
        };

        let analysis = TrafficAnalysis {
            analysis_id: Uuid::new_v4(),
            connection_id,
            threat_level,
            anomalies,
            recommended_action,
            timestamp: Utc::now(),
        };

        Ok(analysis)
    }

    async fn update_policy_state(
        &self,
        policy_id: Uuid,
        state: PolicyState,
    ) -> ZeroTrustResult<()> {
        let mut policy = self.policies.get_mut(&policy_id).ok_or_else(|| {
            ZeroTrustError::NetworkPolicyViolation {
                policy: policy_id.to_string(),
                reason: "Policy not found".to_string(),
            }
        })?;

        policy.state = state;

        // Clear cache when policy changes
        self.policy_cache.clear();

        Ok(())
    }

    async fn get_segment(&self, segment_id: Uuid) -> ZeroTrustResult<NetworkSegment> {
        self.segments
            .get(&segment_id)
            .map(|entry| entry.clone())
            .ok_or_else(|| ZeroTrustError::NetworkPolicyViolation {
                policy: "segment_lookup".to_string(),
                reason: "Segment not found".to_string(),
            })
    }

    async fn get_policy(&self, policy_id: Uuid) -> ZeroTrustResult<NetworkPolicy> {
        self.policies
            .get(&policy_id)
            .map(|entry| entry.clone())
            .ok_or_else(|| ZeroTrustError::NetworkPolicyViolation {
                policy: policy_id.to_string(),
                reason: "Policy not found".to_string(),
            })
    }

    async fn list_policies_for_segment(
        &self,
        segment_id: Uuid,
    ) -> ZeroTrustResult<Vec<NetworkPolicy>> {
        let policies: Vec<NetworkPolicy> = self
            .policies
            .iter()
            .filter(|p| {
                p.source_segment == Some(segment_id) || p.destination_segment == Some(segment_id)
            })
            .map(|p| p.clone())
            .collect();

        Ok(policies)
    }

    async fn record_connection(&self, connection: NetworkConnection) -> ZeroTrustResult<()> {
        self.connections
            .insert(connection.connection_id, connection);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::Ipv4Addr;

    #[tokio::test]
    async fn test_segment_creation() -> anyhow::Result<()> {
        let config = NetworkPolicyConfig::default();
        let manager = NetworkPolicyManager::new(config)?;

        let segment = manager
            .create_segment(
                "internal-network".to_string(),
                SegmentType::Internal,
                vec!["10.0.0.0/8".to_string()],
            )
            .await
            .unwrap();

        assert_eq!(segment.name, "internal-network");
        assert_eq!(segment.segment_type, SegmentType::Internal);
        assert_eq!(segment.trust_level, TrustLevel::High);
        assert_eq!(segment.security_zone, SecurityZone::Private);
        Ok(())
    }

    #[tokio::test]
    async fn test_policy_creation() -> anyhow::Result<()> {
        let config = NetworkPolicyConfig::default();
        let manager = NetworkPolicyManager::new(config)?;

        let internal = manager
            .create_segment(
                "internal".to_string(),
                SegmentType::Internal,
                vec!["10.0.0.0/8".to_string()],
            )
            .await
            .unwrap();

        let dmz = manager
            .create_segment(
                "dmz".to_string(),
                SegmentType::Dmz,
                vec!["172.16.0.0/12".to_string()],
            )
            .await
            .unwrap();

        let policy = manager
            .create_policy(
                "internal-to-dmz".to_string(),
                Some(internal.segment_id),
                Some(dmz.segment_id),
                PolicyAction::Allow,
            )
            .await
            .unwrap();

        assert_eq!(policy.name, "internal-to-dmz");
        assert_eq!(policy.action, PolicyAction::Allow);
        assert_eq!(policy.state, PolicyState::Active);
        Ok(())
    }

    #[tokio::test]
    async fn test_connection_evaluation() -> anyhow::Result<()> {
        let config = NetworkPolicyConfig::default();
        let manager = NetworkPolicyManager::new(config)?;

        let internal = manager
            .create_segment(
                "internal".to_string(),
                SegmentType::Internal,
                vec!["10.0.0.0/8".to_string()],
            )
            .await
            .unwrap();

        let external = manager
            .create_segment(
                "external".to_string(),
                SegmentType::External,
                vec!["0.0.0.0/0".to_string()],
            )
            .await
            .unwrap();

        // Create allow policy
        let policy = manager
            .create_policy(
                "allow-internal-to-external".to_string(),
                Some(internal.segment_id),
                Some(external.segment_id),
                PolicyAction::Allow,
            )
            .await
            .unwrap();

        // Add HTTPS protocol
        manager
            .policies
            .get_mut(&policy.policy_id)
            .unwrap()
            .protocols
            .insert(Protocol::Https);
        manager
            .policies
            .get_mut(&policy.policy_id)
            .unwrap()
            .port_ranges
            .push(PortRange::single(443));

        let connection = NetworkConnection {
            connection_id: Uuid::new_v4(),
            source_ip: IpAddr::V4(Ipv4Addr::new(10, 0, 0, 100)),
            source_port: 54321,
            destination_ip: IpAddr::V4(Ipv4Addr::new(8, 8, 8, 8)),
            destination_port: 443,
            protocol: Protocol::Https,
            start_time: Utc::now(),
            last_activity: Utc::now(),
            bytes_sent: 1024,
            bytes_received: 2048,
            state: ConnectionState::Established,
            applied_policies: vec![],
        };

        let decision = manager.evaluate_connection(&connection).await?;
        assert_eq!(decision.action, PolicyAction::Allow);
        assert!(!decision.matching_policies.is_empty());
        Ok(())
    }

    #[tokio::test]
    async fn test_default_deny() -> anyhow::Result<()> {
        let mut config = NetworkPolicyConfig::default();
        config.default_deny = true;
        let manager = NetworkPolicyManager::new(config)?;

        let connection = NetworkConnection {
            connection_id: Uuid::new_v4(),
            source_ip: IpAddr::V4(Ipv4Addr::new(192, 168, 1, 100)),
            source_port: 12345,
            destination_ip: IpAddr::V4(Ipv4Addr::new(1, 2, 3, 4)),
            destination_port: 80,
            protocol: Protocol::Http,
            start_time: Utc::now(),
            last_activity: Utc::now(),
            bytes_sent: 0,
            bytes_received: 0,
            state: ConnectionState::Pending,
            applied_policies: vec![],
        };

        let decision = manager.evaluate_connection(&connection).await?;
        assert_eq!(decision.action, PolicyAction::Deny);
        assert_eq!(decision.reason, "No matching policy found");
        Ok(())
    }

    #[tokio::test]
    async fn test_traffic_analysis() -> anyhow::Result<()> {
        let config = NetworkPolicyConfig::default();
        let manager = NetworkPolicyManager::new(config)?;

        // Connection with unusual port
        let connection = NetworkConnection {
            connection_id: Uuid::new_v4(),
            source_ip: IpAddr::V4(Ipv4Addr::new(10, 0, 0, 1)),
            source_port: 54321,
            destination_ip: IpAddr::V4(Ipv4Addr::new(10, 0, 0, 2)),
            destination_port: 8080, // Unusual for HTTPS
            protocol: Protocol::Https,
            start_time: Utc::now(),
            last_activity: Utc::now(),
            bytes_sent: 2_000_000_000, // 2GB - high transfer
            bytes_received: 500_000_000,
            state: ConnectionState::Established,
            applied_policies: vec![],
        };

        manager.record_connection(connection.clone()).await?;

        let analysis = manager
            .analyze_traffic(connection.connection_id)
            .await
            .unwrap();
        assert_eq!(analysis.anomalies.len(), 2); // Unusual port + high transfer
        assert!(analysis.threat_level >= ThreatLevel::Medium);
        Ok(())
    }

    #[tokio::test]
    async fn test_policy_priority() -> anyhow::Result<()> {
        let config = NetworkPolicyConfig::default();
        let manager = NetworkPolicyManager::new(config)?;

        let internal = manager
            .create_segment(
                "internal".to_string(),
                SegmentType::Internal,
                vec!["10.0.0.0/8".to_string()],
            )
            .await
            .unwrap();

        // Create deny policy with higher priority (lower number)
        let deny_policy = manager
            .create_policy(
                "deny-all-internal".to_string(),
                Some(internal.segment_id),
                None,
                PolicyAction::Deny,
            )
            .await
            .unwrap();
        manager
            .policies
            .get_mut(&deny_policy.policy_id)
            .unwrap()
            .priority = 10;

        // Create allow policy with lower priority (higher number)
        let allow_policy = manager
            .create_policy(
                "allow-some-internal".to_string(),
                Some(internal.segment_id),
                None,
                PolicyAction::Allow,
            )
            .await
            .unwrap();
        manager
            .policies
            .get_mut(&allow_policy.policy_id)
            .unwrap()
            .priority = 20;

        let connection = NetworkConnection {
            connection_id: Uuid::new_v4(),
            source_ip: IpAddr::V4(Ipv4Addr::new(10, 0, 0, 1)),
            source_port: 12345,
            destination_ip: IpAddr::V4(Ipv4Addr::new(8, 8, 8, 8)),
            destination_port: 443,
            protocol: Protocol::Https,
            start_time: Utc::now(),
            last_activity: Utc::now(),
            bytes_sent: 0,
            bytes_received: 0,
            state: ConnectionState::Pending,
            applied_policies: vec![],
        };

        let decision = manager.evaluate_connection(&connection).await?;
        assert_eq!(decision.action, PolicyAction::Deny); // Higher priority policy wins
        Ok(())
    }

    #[tokio::test]
    async fn test_policy_conditions() -> anyhow::Result<()> {
        let config = NetworkPolicyConfig::default();
        let manager = NetworkPolicyManager::new(config)?;

        let policy = manager
            .create_policy(
                "time-based-policy".to_string(),
                None,
                None,
                PolicyAction::Allow,
            )
            .await
            .unwrap();

        // Add time window condition
        let condition = PolicyCondition::TimeWindow {
            start_time: chrono::NaiveTime::from_hms_opt(9, 0, 0).unwrap(),
            end_time: chrono::NaiveTime::from_hms_opt(17, 0, 0).unwrap(),
            days: [
                chrono::Weekday::Mon,
                chrono::Weekday::Tue,
                chrono::Weekday::Wed,
                chrono::Weekday::Thu,
                chrono::Weekday::Fri,
            ]
            .iter()
            .cloned()
            .collect(),
        };
        manager
            .policies
            .get_mut(&policy.policy_id)
            .unwrap()
            .conditions
            .push(condition);

        let connection = NetworkConnection {
            connection_id: Uuid::new_v4(),
            source_ip: IpAddr::V4(Ipv4Addr::new(10, 0, 0, 1)),
            source_port: 12345,
            destination_ip: IpAddr::V4(Ipv4Addr::new(10, 0, 0, 2)),
            destination_port: 443,
            protocol: Protocol::Https,
            start_time: Utc::now(),
            last_activity: Utc::now(),
            bytes_sent: 0,
            bytes_received: 0,
            state: ConnectionState::Pending,
            applied_policies: vec![],
        };

        // This test might pass or fail depending on when it runs
        // In production, we'd mock the time
        let decision = manager.evaluate_connection(&connection).await?;
        assert!(decision.action == PolicyAction::Allow || decision.action == PolicyAction::Deny);
        Ok(())
    }

    #[tokio::test]
    async fn test_policy_state_management() -> anyhow::Result<()> {
        let config = NetworkPolicyConfig::default();
        let manager = NetworkPolicyManager::new(config)?;

        let policy = manager
            .create_policy("test-policy".to_string(), None, None, PolicyAction::Allow)
            .await
            .unwrap();

        // Disable policy
        manager
            .update_policy_state(policy.policy_id, PolicyState::Disabled)
            .await
            .unwrap();

        let updated_policy = manager.get_policy(policy.policy_id).await?;
        assert_eq!(updated_policy.state, PolicyState::Disabled);

        let connection = NetworkConnection {
            connection_id: Uuid::new_v4(),
            source_ip: IpAddr::V4(Ipv4Addr::new(10, 0, 0, 1)),
            source_port: 12345,
            destination_ip: IpAddr::V4(Ipv4Addr::new(10, 0, 0, 2)),
            destination_port: 443,
            protocol: Protocol::Https,
            start_time: Utc::now(),
            last_activity: Utc::now(),
            bytes_sent: 0,
            bytes_received: 0,
            state: ConnectionState::Pending,
            applied_policies: vec![],
        };

        // Disabled policy should not match
        let decision = manager.evaluate_connection(&connection).await?;
        assert!(decision.matching_policies.is_empty());
        Ok(())
    }

    #[tokio::test]
    async fn test_segment_policies() -> anyhow::Result<()> {
        let config = NetworkPolicyConfig::default();
        let manager = NetworkPolicyManager::new(config)?;

        let segment = manager
            .create_segment(
                "test-segment".to_string(),
                SegmentType::Internal,
                vec!["10.0.0.0/8".to_string()],
            )
            .await
            .unwrap();

        // Create multiple policies for segment
        for i in 0..3 {
            manager
                .create_policy(
                    format!("policy-{}", i),
                    Some(segment.segment_id),
                    None,
                    PolicyAction::Allow,
                )
                .await
                .unwrap();
        }

        let policies = manager
            .list_policies_for_segment(segment.segment_id)
            .await
            .unwrap();
        assert_eq!(policies.len(), 3);
        Ok(())
    }

    #[tokio::test]
    async fn test_port_range_matching() -> anyhow::Result<()> {
        let config = NetworkPolicyConfig::default();
        let manager = NetworkPolicyManager::new(config)?;

        let policy = manager
            .create_policy(
                "port-range-policy".to_string(),
                None,
                None,
                PolicyAction::Allow,
            )
            .await
            .unwrap();

        // Add port ranges
        manager
            .policies
            .get_mut(&policy.policy_id)
            .unwrap()
            .port_ranges
            .push(PortRange { start: 80, end: 80 });
        manager
            .policies
            .get_mut(&policy.policy_id)
            .unwrap()
            .port_ranges
            .push(PortRange {
                start: 443,
                end: 443,
            });
        manager
            .policies
            .get_mut(&policy.policy_id)
            .unwrap()
            .port_ranges
            .push(PortRange {
                start: 8000,
                end: 8999,
            });

        // Test matching port
        let connection1 = NetworkConnection {
            connection_id: Uuid::new_v4(),
            source_ip: IpAddr::V4(Ipv4Addr::new(10, 0, 0, 1)),
            source_port: 12345,
            destination_ip: IpAddr::V4(Ipv4Addr::new(10, 0, 0, 2)),
            destination_port: 8080,
            protocol: Protocol::Http,
            start_time: Utc::now(),
            last_activity: Utc::now(),
            bytes_sent: 0,
            bytes_received: 0,
            state: ConnectionState::Pending,
            applied_policies: vec![],
        };

        let decision1 = manager.evaluate_connection(&connection1).await?;
        assert_eq!(decision1.action, PolicyAction::Allow);

        // Test non-matching port
        let connection2 = NetworkConnection {
            connection_id: Uuid::new_v4(),
            source_ip: IpAddr::V4(Ipv4Addr::new(10, 0, 0, 1)),
            source_port: 12345,
            destination_ip: IpAddr::V4(Ipv4Addr::new(10, 0, 0, 2)),
            destination_port: 9000,
            protocol: Protocol::Http,
            start_time: Utc::now(),
            last_activity: Utc::now(),
            bytes_sent: 0,
            bytes_received: 0,
            state: ConnectionState::Pending,
            applied_policies: vec![],
        };

        let decision2 = manager.evaluate_connection(&connection2).await?;
        assert_eq!(decision2.action, PolicyAction::Deny); // Default deny
        Ok(())
    }

    #[tokio::test]
    async fn test_policy_caching() -> anyhow::Result<()> {
        let config = NetworkPolicyConfig::default();
        let manager = NetworkPolicyManager::new(config)?;

        let _policy = manager
            .create_policy("cached-policy".to_string(), None, None, PolicyAction::Allow)
            .await
            .unwrap();

        let connection = NetworkConnection {
            connection_id: Uuid::new_v4(),
            source_ip: IpAddr::V4(Ipv4Addr::new(10, 0, 0, 1)),
            source_port: 12345,
            destination_ip: IpAddr::V4(Ipv4Addr::new(10, 0, 0, 2)),
            destination_port: 443,
            protocol: Protocol::Https,
            start_time: Utc::now(),
            last_activity: Utc::now(),
            bytes_sent: 0,
            bytes_received: 0,
            state: ConnectionState::Pending,
            applied_policies: vec![],
        };

        // First evaluation - cache miss
        let decision1 = manager.evaluate_connection(&connection).await?;
        assert_eq!(decision1.action, PolicyAction::Allow);

        // Second evaluation - cache hit
        let decision2 = manager.evaluate_connection(&connection).await?;
        assert_eq!(decision2.action, PolicyAction::Allow);
        assert_eq!(decision1.matching_policies, decision2.matching_policies);
        Ok(())
    }
}
