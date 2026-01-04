//! Session adapter for integrating ephemeral identities with zero-trust session management.
//!
//! This module bridges the ephemeral identity system with the zero-trust SessionManager,
//! providing:
//! - Ephemeral session creation from redeemed invitations
//! - Shorter session timeouts appropriate for ephemeral access
//! - Risk scoring integration based on ephemeral identity state
//! - Capability-based permission checking
//! - Automatic cleanup of expired ephemeral sessions

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use uuid::Uuid;

use crate::{CapabilitySet, DeviceBinding, EphemeralIdentity, EphemeralIdentityState};

/// Configuration for ephemeral sessions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EphemeralSessionConfig {
    /// Session timeout (default: 1 hour for ephemeral)
    pub session_timeout: Duration,
    /// Idle timeout (default: 10 minutes for ephemeral)
    pub idle_timeout: Duration,
    /// Maximum concurrent sessions per ephemeral identity (default: 1)
    pub max_concurrent_sessions: usize,
    /// Token rotation interval (default: 15 minutes)
    pub token_rotation_interval: Duration,
    /// Whether to enable strict device binding
    pub strict_device_binding: bool,
    /// Session verification interval (default: 2 minutes)
    pub verification_interval: Duration,
    /// Enable detailed activity logging
    pub activity_logging: bool,
    /// Risk threshold for automatic suspension (0.0-1.0)
    pub risk_suspension_threshold: f64,
    /// Risk threshold for automatic revocation (0.0-1.0)
    pub risk_revocation_threshold: f64,
}

impl Default for EphemeralSessionConfig {
    fn default() -> Self {
        Self {
            session_timeout: Duration::hours(1),
            idle_timeout: Duration::minutes(10),
            max_concurrent_sessions: 1,
            token_rotation_interval: Duration::minutes(15),
            strict_device_binding: true,
            verification_interval: Duration::minutes(2),
            activity_logging: true,
            risk_suspension_threshold: 0.6,
            risk_revocation_threshold: 0.85,
        }
    }
}

impl EphemeralSessionConfig {
    /// Create a strict configuration for high-security ephemeral access.
    pub fn strict() -> Self {
        Self {
            session_timeout: Duration::minutes(30),
            idle_timeout: Duration::minutes(5),
            max_concurrent_sessions: 1,
            token_rotation_interval: Duration::minutes(5),
            strict_device_binding: true,
            verification_interval: Duration::minutes(1),
            activity_logging: true,
            risk_suspension_threshold: 0.5,
            risk_revocation_threshold: 0.75,
        }
    }

    /// Create a relaxed configuration for demo/trial access.
    pub fn relaxed() -> Self {
        Self {
            session_timeout: Duration::hours(4),
            idle_timeout: Duration::minutes(30),
            max_concurrent_sessions: 2,
            token_rotation_interval: Duration::minutes(30),
            strict_device_binding: false,
            verification_interval: Duration::minutes(5),
            activity_logging: true,
            risk_suspension_threshold: 0.7,
            risk_revocation_threshold: 0.9,
        }
    }
}

/// Session binding information for ephemeral sessions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EphemeralSessionBinding {
    /// Device fingerprint
    pub device_fingerprint: Option<String>,
    /// IP address
    pub ip_address: String,
    /// User agent string
    pub user_agent: String,
    /// Browser/client fingerprint
    pub client_fingerprint: Option<String>,
    /// Geographic location (if available)
    pub location: Option<String>,
    /// Trusted device flag
    pub trusted_device: bool,
}

impl EphemeralSessionBinding {
    /// Create a new session binding from device info.
    pub fn new(ip_address: impl Into<String>, user_agent: impl Into<String>) -> Self {
        Self {
            device_fingerprint: None,
            ip_address: ip_address.into(),
            user_agent: user_agent.into(),
            client_fingerprint: None,
            location: None,
            trusted_device: false,
        }
    }

    /// Set device fingerprint.
    pub fn with_device_fingerprint(mut self, fingerprint: impl Into<String>) -> Self {
        self.device_fingerprint = Some(fingerprint.into());
        self
    }

    /// Set client fingerprint.
    pub fn with_client_fingerprint(mut self, fingerprint: impl Into<String>) -> Self {
        self.client_fingerprint = Some(fingerprint.into());
        self
    }

    /// Set location.
    pub fn with_location(mut self, location: impl Into<String>) -> Self {
        self.location = Some(location.into());
        self
    }

    /// Mark as trusted device.
    pub fn trusted(mut self) -> Self {
        self.trusted_device = true;
        self
    }

    /// Validate binding against an ephemeral identity's device binding.
    pub fn validate_against(&self, device_binding: &DeviceBinding) -> BindingValidation {
        let mut issues = Vec::new();
        let mut score: f64 = 1.0;

        // Check device fingerprint
        if let Some(ref expected) = self.device_fingerprint {
            if expected != &device_binding.fingerprint {
                issues.push("Device fingerprint mismatch".to_string());
                score -= 0.4;
            }
        }

        // Check IP address - IP changes are allowed for mobile users but logged
        if self.ip_address != device_binding.ip_address {
            issues.push("IP address changed (allowed for mobile)".to_string());
            score -= 0.1;
        }

        // Check user agent
        if self.user_agent != device_binding.user_agent {
            issues.push("User agent mismatch".to_string());
            score -= 0.2;
        }

        BindingValidation {
            valid: issues.is_empty() || score > 0.5,
            trust_score: score.max(0.0),
            issues,
        }
    }
}

/// Result of validating session binding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BindingValidation {
    /// Whether binding is considered valid
    pub valid: bool,
    /// Trust score (0.0-1.0)
    pub trust_score: f64,
    /// Any issues found
    pub issues: Vec<String>,
}

/// Activity record for ephemeral sessions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EphemeralSessionActivity {
    /// Activity timestamp
    pub timestamp: DateTime<Utc>,
    /// Type of activity
    pub activity_type: EphemeralActivityType,
    /// Resource being accessed
    pub resource: Option<String>,
    /// Action being performed
    pub action: Option<String>,
    /// Result of the activity
    pub result: ActivityResult,
    /// Risk score at time of activity
    pub risk_score: f64,
    /// Additional context
    pub context: Option<serde_json::Value>,
}

/// Types of ephemeral session activities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EphemeralActivityType {
    /// Session created
    SessionCreated,
    /// Resource accessed
    ResourceAccess,
    /// Capability checked
    CapabilityCheck,
    /// Token rotated
    TokenRotation,
    /// Session verified
    Verification,
    /// Risk assessed
    RiskAssessment,
    /// Binding validated
    BindingValidation,
    /// Rate limit checked
    RateLimitCheck,
    /// Session suspended
    SessionSuspended,
    /// Session revoked
    SessionRevoked,
    /// Session expired
    SessionExpired,
}

/// Result of an activity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActivityResult {
    /// Activity succeeded
    Success,
    /// Activity was denied
    Denied,
    /// Activity required additional verification
    Challenge,
    /// Activity failed due to error
    Error,
    /// Activity was rate limited
    RateLimited,
}

/// Verification result for ephemeral sessions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EphemeralVerificationResult {
    /// Whether verification passed
    pub verified: bool,
    /// Current risk score
    pub risk_score: f64,
    /// Risk level
    pub risk_level: RiskLevel,
    /// Required actions based on verification
    pub required_actions: Vec<RequiredAction>,
    /// Time until session expires
    pub time_remaining: Option<Duration>,
    /// Verification timestamp
    pub timestamp: DateTime<Utc>,
    /// Detailed verification report
    pub report: VerificationReport,
}

/// Risk levels for ephemeral sessions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RiskLevel {
    /// Low risk - normal operation
    Low,
    /// Medium risk - enhanced monitoring
    Medium,
    /// High risk - restricted operations
    High,
    /// Critical risk - immediate action required
    Critical,
}

impl RiskLevel {
    /// Determine risk level from score.
    pub fn from_score(score: f64) -> Self {
        if score < 0.25 {
            RiskLevel::Low
        } else if score < 0.5 {
            RiskLevel::Medium
        } else if score < 0.75 {
            RiskLevel::High
        } else {
            RiskLevel::Critical
        }
    }
}

/// Required actions from verification.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RequiredAction {
    /// Re-verify identity
    ReVerify,
    /// Rotate session token
    RotateToken,
    /// Update device binding
    UpdateBinding,
    /// Reduce capabilities/permissions
    ReduceCapabilities,
    /// Suspend session
    Suspend,
    /// Terminate session immediately
    Terminate,
    /// Notify sponsor
    NotifySponsor,
}

/// Detailed verification report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationReport {
    /// Identity state check
    pub identity_valid: bool,
    /// Device binding check
    pub binding_valid: bool,
    /// Time restriction check
    pub time_allowed: bool,
    /// Rate limit check
    pub within_rate_limits: bool,
    /// Capability check (for last operation)
    pub capabilities_valid: bool,
    /// Behavioral check
    pub behavior_normal: bool,
    /// Individual check details
    pub check_details: Vec<CheckDetail>,
}

/// Detail of a single verification check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckDetail {
    /// Check name
    pub check: String,
    /// Whether it passed
    pub passed: bool,
    /// Risk contribution (0.0-1.0)
    pub risk_contribution: f64,
    /// Description
    pub description: String,
}

/// Adapter for creating and managing ephemeral sessions.
pub struct EphemeralSessionAdapter {
    config: EphemeralSessionConfig,
}

impl EphemeralSessionAdapter {
    /// Create a new session adapter with default configuration.
    pub fn new() -> Self {
        Self::with_config(EphemeralSessionConfig::default())
    }

    /// Create with custom configuration.
    pub fn with_config(config: EphemeralSessionConfig) -> Self {
        Self { config }
    }

    /// Get the configuration.
    pub fn config(&self) -> &EphemeralSessionConfig {
        &self.config
    }

    /// Calculate the effective session timeout based on identity expiry.
    pub fn calculate_session_timeout(&self, identity: &EphemeralIdentity) -> Duration {
        let identity_remaining = identity.time_remaining();

        if identity_remaining <= Duration::zero() {
            Duration::zero() // Identity already expired
        } else if identity_remaining < self.config.session_timeout {
            // Use the shorter of session timeout and identity remaining time
            identity_remaining
        } else {
            self.config.session_timeout
        }
    }

    /// Build session metadata from an ephemeral identity.
    pub fn build_session_metadata(&self, identity: &EphemeralIdentity) -> EphemeralSessionMetadata {
        let permissions: HashSet<String> = identity
            .capabilities
            .capabilities
            .iter()
            .map(|c| c.to_string())
            .collect();

        EphemeralSessionMetadata {
            ephemeral_identity_id: identity.id,
            sponsor_id: identity.sponsor_id,
            tenant_id: identity.tenant_id,
            auth_method: EphemeralAuthMethod::InvitationRedemption,
            permissions,
            capabilities: identity.capabilities.clone(),
            risk_score: identity.risk_score,
            created_at: Utc::now(),
            identity_expires_at: identity.expires_at,
        }
    }

    /// Verify an ephemeral session.
    pub fn verify_session(
        &self,
        identity: &EphemeralIdentity,
        binding: &EphemeralSessionBinding,
        last_activity: DateTime<Utc>,
    ) -> EphemeralVerificationResult {
        let now = Utc::now();
        let mut risk_score = identity.risk_score;
        let mut required_actions = Vec::new();
        let mut check_details = Vec::new();

        // Check identity state
        let identity_valid = matches!(identity.state, EphemeralIdentityState::Active);
        if !identity_valid {
            risk_score += 0.5;
            required_actions.push(RequiredAction::Terminate);
            check_details.push(CheckDetail {
                check: "identity_state".to_string(),
                passed: false,
                risk_contribution: 0.5,
                description: format!("Identity is in {:?} state", identity.state),
            });
        } else {
            check_details.push(CheckDetail {
                check: "identity_state".to_string(),
                passed: true,
                risk_contribution: 0.0,
                description: "Identity is active".to_string(),
            });
        }

        // Check expiry
        let time_remaining = identity.time_remaining();
        let is_expired = identity.is_expired();
        if is_expired {
            risk_score = 1.0;
            required_actions.push(RequiredAction::Terminate);
            check_details.push(CheckDetail {
                check: "expiry".to_string(),
                passed: false,
                risk_contribution: 1.0,
                description: "Identity has expired".to_string(),
            });
        } else {
            // Increase risk as expiry approaches
            let expiry_risk = self.calculate_expiry_risk(time_remaining);
            if expiry_risk > 0.0 {
                risk_score += expiry_risk;
                check_details.push(CheckDetail {
                    check: "expiry".to_string(),
                    passed: true,
                    risk_contribution: expiry_risk,
                    description: format!(
                        "Identity expires in {} minutes",
                        time_remaining.num_minutes()
                    ),
                });
            } else {
                check_details.push(CheckDetail {
                    check: "expiry".to_string(),
                    passed: true,
                    risk_contribution: 0.0,
                    description: "Identity has sufficient time remaining".to_string(),
                });
            }
        }

        // Check device binding
        let binding_valid = if let Some(ref device_binding) = identity.device_binding {
            let validation = binding.validate_against(device_binding);
            if !validation.valid {
                risk_score += 0.3;
                required_actions.push(RequiredAction::UpdateBinding);
                check_details.push(CheckDetail {
                    check: "device_binding".to_string(),
                    passed: false,
                    risk_contribution: 0.3,
                    description: format!("Binding issues: {:?}", validation.issues),
                });
                false
            } else {
                check_details.push(CheckDetail {
                    check: "device_binding".to_string(),
                    passed: true,
                    risk_contribution: 0.0,
                    description: "Device binding verified".to_string(),
                });
                true
            }
        } else {
            // No device binding required
            check_details.push(CheckDetail {
                check: "device_binding".to_string(),
                passed: true,
                risk_contribution: 0.0,
                description: "No device binding configured".to_string(),
            });
            true
        };

        // Check idle timeout
        let idle_duration = now.signed_duration_since(last_activity);
        let idle_valid = idle_duration <= self.config.idle_timeout;
        if !idle_valid {
            risk_score += 0.2;
            required_actions.push(RequiredAction::ReVerify);
            check_details.push(CheckDetail {
                check: "idle_timeout".to_string(),
                passed: false,
                risk_contribution: 0.2,
                description: format!("Idle for {} minutes", idle_duration.num_minutes()),
            });
        } else {
            check_details.push(CheckDetail {
                check: "idle_timeout".to_string(),
                passed: true,
                risk_contribution: 0.0,
                description: "Within idle timeout".to_string(),
            });
        }

        // Check time restrictions
        let time_allowed = identity.capabilities.time_restrictions.is_allowed_at(now);
        if !time_allowed {
            risk_score += 0.15;
            check_details.push(CheckDetail {
                check: "time_restrictions".to_string(),
                passed: false,
                risk_contribution: 0.15,
                description: "Outside allowed time window".to_string(),
            });
        } else {
            check_details.push(CheckDetail {
                check: "time_restrictions".to_string(),
                passed: true,
                risk_contribution: 0.0,
                description: "Within allowed time window".to_string(),
            });
        }

        // Determine required actions based on risk score
        risk_score = risk_score.min(1.0);
        let risk_level = RiskLevel::from_score(risk_score);

        if risk_score >= self.config.risk_revocation_threshold {
            if !required_actions.contains(&RequiredAction::Terminate) {
                required_actions.push(RequiredAction::Terminate);
            }
            required_actions.push(RequiredAction::NotifySponsor);
        } else if risk_score >= self.config.risk_suspension_threshold {
            if !required_actions.contains(&RequiredAction::Suspend) {
                required_actions.push(RequiredAction::Suspend);
            }
        } else if risk_score >= 0.4 {
            required_actions.push(RequiredAction::RotateToken);
        }

        EphemeralVerificationResult {
            verified: identity_valid
                && !is_expired
                && risk_score < self.config.risk_suspension_threshold,
            risk_score,
            risk_level,
            required_actions,
            time_remaining: if is_expired {
                None
            } else {
                Some(time_remaining)
            },
            timestamp: now,
            report: VerificationReport {
                identity_valid,
                binding_valid,
                time_allowed,
                within_rate_limits: true, // Would need actual rate limit checking
                capabilities_valid: true, // Would need last operation context
                behavior_normal: risk_score < 0.5,
                check_details,
            },
        }
    }

    /// Check if an operation is allowed for the identity.
    pub fn check_operation(
        &self,
        identity: &EphemeralIdentity,
        action: &str,
        resource: &str,
    ) -> OperationCheck {
        // First check identity state
        if identity.state != EphemeralIdentityState::Active {
            return OperationCheck {
                allowed: false,
                reason: Some(format!("Identity is {:?}", identity.state)),
                risk_contribution: 0.5,
            };
        }

        // Check expiry
        if identity.is_expired() {
            return OperationCheck {
                allowed: false,
                reason: Some("Identity has expired".to_string()),
                risk_contribution: 1.0,
            };
        }

        // Check time restrictions
        if !identity.capabilities.is_time_allowed() {
            return OperationCheck {
                allowed: false,
                reason: Some("Outside allowed time window".to_string()),
                risk_contribution: 0.2,
            };
        }

        // Check capability
        match identity.capabilities.validate(action, resource) {
            Ok(()) => OperationCheck {
                allowed: true,
                reason: None,
                risk_contribution: 0.0,
            },
            Err(e) => OperationCheck {
                allowed: false,
                reason: Some(e.to_string()),
                risk_contribution: 0.1,
            },
        }
    }

    /// Calculate risk contribution from expiry proximity.
    fn calculate_expiry_risk(&self, remaining: Duration) -> f64 {
        let total_session = self.config.session_timeout.num_seconds() as f64;
        let remaining_secs = remaining.num_seconds() as f64;

        if remaining_secs <= 0.0 {
            return 1.0;
        }

        // Risk increases as we approach expiry
        let ratio = remaining_secs / total_session;
        if ratio > 0.5 {
            0.0
        } else if ratio > 0.25 {
            0.1
        } else if ratio > 0.1 {
            0.2
        } else {
            0.3
        }
    }

    /// Record an activity for audit purposes.
    pub fn record_activity(
        &self,
        activity_type: EphemeralActivityType,
        resource: Option<String>,
        action: Option<String>,
        result: ActivityResult,
        risk_score: f64,
    ) -> EphemeralSessionActivity {
        EphemeralSessionActivity {
            timestamp: Utc::now(),
            activity_type,
            resource,
            action,
            result,
            risk_score,
            context: None,
        }
    }
}

impl Default for EphemeralSessionAdapter {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of checking an operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationCheck {
    /// Whether operation is allowed
    pub allowed: bool,
    /// Reason if denied
    pub reason: Option<String>,
    /// Risk contribution of this operation
    pub risk_contribution: f64,
}

/// Session metadata specific to ephemeral sessions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EphemeralSessionMetadata {
    /// The ephemeral identity ID
    pub ephemeral_identity_id: Uuid,
    /// Sponsor who created the invitation
    pub sponsor_id: Uuid,
    /// Tenant context
    pub tenant_id: Uuid,
    /// Authentication method used
    pub auth_method: EphemeralAuthMethod,
    /// Permissions as strings (for compatibility)
    pub permissions: HashSet<String>,
    /// Full capability set
    pub capabilities: CapabilitySet,
    /// Current risk score
    pub risk_score: f64,
    /// Session creation time
    pub created_at: DateTime<Utc>,
    /// When the identity expires
    pub identity_expires_at: DateTime<Utc>,
}

/// Authentication methods for ephemeral access.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EphemeralAuthMethod {
    /// Redeemed an invitation link
    InvitationRedemption,
    /// Joined via QR code scan
    QrCodeScan,
    /// API token access
    ApiToken,
    /// Federated identity provider
    FederatedIdentity,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Capability;

    fn test_identity() -> EphemeralIdentity {
        let mut identity = EphemeralIdentity::new(
            Uuid::new_v4(),
            Uuid::new_v4(),
            CapabilitySet::new(),
            Duration::hours(24),
        );
        identity.state = EphemeralIdentityState::Active;
        identity
    }

    #[test]
    fn test_config_defaults() {
        let config = EphemeralSessionConfig::default();
        assert_eq!(config.session_timeout, Duration::hours(1));
        assert_eq!(config.idle_timeout, Duration::minutes(10));
        assert_eq!(config.max_concurrent_sessions, 1);
    }

    #[test]
    fn test_config_strict() {
        let config = EphemeralSessionConfig::strict();
        assert_eq!(config.session_timeout, Duration::minutes(30));
        assert_eq!(config.idle_timeout, Duration::minutes(5));
        assert!(config.strict_device_binding);
    }

    #[test]
    fn test_config_relaxed() {
        let config = EphemeralSessionConfig::relaxed();
        assert_eq!(config.session_timeout, Duration::hours(4));
        assert!(!config.strict_device_binding);
    }

    #[test]
    fn test_session_binding_new() {
        let binding = EphemeralSessionBinding::new("192.168.1.1", "Mozilla/5.0");
        assert_eq!(binding.ip_address, "192.168.1.1");
        assert_eq!(binding.user_agent, "Mozilla/5.0");
        assert!(!binding.trusted_device);
    }

    #[test]
    fn test_session_binding_builder() {
        let binding = EphemeralSessionBinding::new("192.168.1.1", "Mozilla/5.0")
            .with_device_fingerprint("abc123")
            .with_location("New York, US")
            .trusted();

        assert_eq!(binding.device_fingerprint, Some("abc123".to_string()));
        assert_eq!(binding.location, Some("New York, US".to_string()));
        assert!(binding.trusted_device);
    }

    #[test]
    fn test_risk_level_from_score() {
        assert_eq!(RiskLevel::from_score(0.1), RiskLevel::Low);
        assert_eq!(RiskLevel::from_score(0.3), RiskLevel::Medium);
        assert_eq!(RiskLevel::from_score(0.6), RiskLevel::High);
        assert_eq!(RiskLevel::from_score(0.9), RiskLevel::Critical);
    }

    #[test]
    fn test_adapter_calculate_session_timeout() {
        let adapter = EphemeralSessionAdapter::new();
        let identity = test_identity();

        let timeout = adapter.calculate_session_timeout(&identity);
        // Should be capped at config timeout (1 hour)
        assert_eq!(timeout, Duration::hours(1));
    }

    #[test]
    fn test_adapter_calculate_session_timeout_short_identity() {
        let adapter = EphemeralSessionAdapter::new();
        let mut identity = EphemeralIdentity::new(
            Uuid::new_v4(),
            Uuid::new_v4(),
            CapabilitySet::new(),
            Duration::minutes(30), // Shorter than session timeout
        );
        identity.state = EphemeralIdentityState::Active;

        let timeout = adapter.calculate_session_timeout(&identity);
        // Should use identity's remaining time
        assert!(timeout <= Duration::minutes(30));
    }

    #[test]
    fn test_adapter_build_session_metadata() {
        let adapter = EphemeralSessionAdapter::new();
        let mut caps = CapabilitySet::new();
        caps = caps.with_capability(Capability::new("read", "notebooks"));

        let mut identity =
            EphemeralIdentity::new(Uuid::new_v4(), Uuid::new_v4(), caps, Duration::hours(24));
        identity.state = EphemeralIdentityState::Active;

        let metadata = adapter.build_session_metadata(&identity);
        assert_eq!(metadata.ephemeral_identity_id, identity.id);
        assert_eq!(metadata.sponsor_id, identity.sponsor_id);
        assert!(metadata.permissions.contains("read:notebooks"));
    }

    #[test]
    fn test_adapter_verify_session_active() {
        let adapter = EphemeralSessionAdapter::new();
        let identity = test_identity();
        let binding = EphemeralSessionBinding::new("192.168.1.1", "Mozilla/5.0");
        let last_activity = Utc::now();

        let result = adapter.verify_session(&identity, &binding, last_activity);
        assert!(result.verified);
        // Ephemeral identities have base risk of 0.3, which is Medium
        assert_eq!(result.risk_level, RiskLevel::Medium);
    }

    #[test]
    fn test_adapter_verify_session_expired() {
        let adapter = EphemeralSessionAdapter::new();
        let mut identity = EphemeralIdentity::new(
            Uuid::new_v4(),
            Uuid::new_v4(),
            CapabilitySet::new(),
            Duration::hours(-1), // Already expired
        );
        identity.state = EphemeralIdentityState::Active;

        let binding = EphemeralSessionBinding::new("192.168.1.1", "Mozilla/5.0");
        let last_activity = Utc::now();

        let result = adapter.verify_session(&identity, &binding, last_activity);
        assert!(!result.verified);
        assert!(result.required_actions.contains(&RequiredAction::Terminate));
    }

    #[test]
    fn test_adapter_verify_session_idle() {
        let adapter = EphemeralSessionAdapter::new();
        let identity = test_identity();
        let binding = EphemeralSessionBinding::new("192.168.1.1", "Mozilla/5.0");
        let last_activity = Utc::now() - Duration::minutes(15); // Beyond idle timeout

        let result = adapter.verify_session(&identity, &binding, last_activity);
        // Should still verify but require re-verification
        assert!(result.required_actions.contains(&RequiredAction::ReVerify));
    }

    #[test]
    fn test_adapter_check_operation_allowed() {
        let adapter = EphemeralSessionAdapter::new();
        let mut caps = CapabilitySet::new();
        caps = caps.with_capability(Capability::new("read", "notebooks"));

        let mut identity =
            EphemeralIdentity::new(Uuid::new_v4(), Uuid::new_v4(), caps, Duration::hours(24));
        identity.state = EphemeralIdentityState::Active;

        let check = adapter.check_operation(&identity, "read", "notebooks");
        assert!(check.allowed);
        assert!(check.reason.is_none());
    }

    #[test]
    fn test_adapter_check_operation_denied() {
        let adapter = EphemeralSessionAdapter::new();
        let mut caps = CapabilitySet::new();
        caps = caps.with_capability(Capability::new("read", "notebooks"));

        let mut identity =
            EphemeralIdentity::new(Uuid::new_v4(), Uuid::new_v4(), caps, Duration::hours(24));
        identity.state = EphemeralIdentityState::Active;

        let check = adapter.check_operation(&identity, "write", "notebooks");
        assert!(!check.allowed);
        assert!(check.reason.is_some());
    }

    #[test]
    fn test_adapter_check_operation_inactive_identity() {
        let adapter = EphemeralSessionAdapter::new();
        let mut identity = EphemeralIdentity::new(
            Uuid::new_v4(),
            Uuid::new_v4(),
            CapabilitySet::new(),
            Duration::hours(24),
        );
        identity.state = EphemeralIdentityState::Suspended;

        let check = adapter.check_operation(&identity, "read", "notebooks");
        assert!(!check.allowed);
    }

    #[test]
    fn test_adapter_record_activity() {
        let adapter = EphemeralSessionAdapter::new();

        let activity = adapter.record_activity(
            EphemeralActivityType::ResourceAccess,
            Some("notebooks/123".to_string()),
            Some("read".to_string()),
            ActivityResult::Success,
            0.1,
        );

        assert_eq!(
            activity.activity_type,
            EphemeralActivityType::ResourceAccess
        );
        assert_eq!(activity.result, ActivityResult::Success);
        assert_eq!(activity.risk_score, 0.1);
    }

    #[test]
    fn test_binding_validation_match() {
        let device_binding = DeviceBinding::new(
            "fingerprint123".to_string(),
            "192.168.1.1".to_string(),
            "Mozilla/5.0".to_string(),
        );

        let session_binding = EphemeralSessionBinding::new("192.168.1.1", "Mozilla/5.0")
            .with_device_fingerprint("fingerprint123");

        let validation = session_binding.validate_against(&device_binding);
        assert!(validation.valid);
        assert_eq!(validation.trust_score, 1.0);
    }

    #[test]
    fn test_binding_validation_mismatch() {
        let device_binding = DeviceBinding::new(
            "fingerprint123".to_string(),
            "192.168.1.1".to_string(),
            "Mozilla/5.0".to_string(),
        );

        let session_binding = EphemeralSessionBinding::new("192.168.1.2", "Chrome/100")
            .with_device_fingerprint("different");

        let validation = session_binding.validate_against(&device_binding);
        assert!(!validation.valid);
        assert!(validation.trust_score < 0.5);
        assert!(!validation.issues.is_empty());
    }

    #[test]
    fn test_verification_report() {
        let adapter = EphemeralSessionAdapter::new();
        let identity = test_identity();
        let binding = EphemeralSessionBinding::new("192.168.1.1", "Mozilla/5.0");
        let last_activity = Utc::now();

        let result = adapter.verify_session(&identity, &binding, last_activity);
        assert!(result.report.identity_valid);
        assert!(!result.report.check_details.is_empty());
    }
}
