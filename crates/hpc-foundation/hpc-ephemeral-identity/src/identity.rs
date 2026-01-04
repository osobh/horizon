//! Ephemeral identity types and state management.
//!
//! An ephemeral identity represents a time-limited external user with scoped
//! permissions. These identities are created through invitation links and
//! automatically expire after their time-to-live.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::capabilities::CapabilitySet;
use crate::error::{EphemeralError, Result};

/// State of an ephemeral identity throughout its lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EphemeralIdentityState {
    /// Identity created but invitation not yet redeemed.
    Pending,
    /// Identity is active and can be used.
    Active,
    /// Identity has reached its expiration time.
    Expired,
    /// Identity was manually revoked by sponsor or admin.
    Revoked,
    /// Identity is suspended pending review (e.g., suspicious activity).
    Suspended,
}

impl EphemeralIdentityState {
    /// Returns true if the identity can perform operations.
    #[must_use]
    pub fn is_usable(&self) -> bool {
        matches!(self, EphemeralIdentityState::Active)
    }

    /// Returns true if the identity can be activated.
    #[must_use]
    pub fn can_activate(&self) -> bool {
        matches!(self, EphemeralIdentityState::Pending)
    }

    /// Returns true if the identity is in a terminal state.
    #[must_use]
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            EphemeralIdentityState::Expired | EphemeralIdentityState::Revoked
        )
    }
}

/// Device binding information for zero-trust verification.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceBinding {
    /// Unique fingerprint of the device.
    pub fingerprint: String,
    /// IP address at binding time.
    pub ip_address: String,
    /// User agent string from browser/client.
    pub user_agent: String,
    /// Geographic location if available.
    pub location: Option<String>,
    /// When the device was bound.
    pub bound_at: DateTime<Utc>,
}

impl DeviceBinding {
    /// Creates a new device binding with current timestamp.
    #[must_use]
    pub fn new(fingerprint: String, ip_address: String, user_agent: String) -> Self {
        Self {
            fingerprint,
            ip_address,
            user_agent,
            location: None,
            bound_at: Utc::now(),
        }
    }

    /// Creates a device binding with location information.
    #[must_use]
    pub fn with_location(mut self, location: String) -> Self {
        self.location = Some(location);
        self
    }

    /// Validates that a request matches this device binding.
    pub fn validate(&self, fingerprint: &str, ip_address: &str) -> Result<()> {
        if self.fingerprint != fingerprint {
            return Err(EphemeralError::DeviceBindingMismatch {
                expected: self.fingerprint.clone(),
                found: fingerprint.to_string(),
            });
        }
        // IP address changes are logged but allowed (mobile users)
        if self.ip_address != ip_address {
            // In production, this would emit a warning event
            // but not fail the validation
        }
        Ok(())
    }
}

/// Metadata associated with an ephemeral identity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdentityMetadata {
    /// Display name for the external user.
    pub display_name: Option<String>,
    /// Email address (may be temporary).
    pub email: Option<String>,
    /// Purpose or reason for ephemeral access.
    pub purpose: Option<String>,
    /// Custom attributes for extensibility.
    #[serde(default)]
    pub attributes: serde_json::Value,
}

impl Default for IdentityMetadata {
    fn default() -> Self {
        Self {
            display_name: None,
            email: None,
            purpose: None,
            attributes: serde_json::Value::Object(serde_json::Map::new()),
        }
    }
}

impl IdentityMetadata {
    /// Creates metadata with a display name.
    #[must_use]
    pub fn with_display_name(name: impl Into<String>) -> Self {
        Self {
            display_name: Some(name.into()),
            ..Default::default()
        }
    }

    /// Sets the email address.
    #[must_use]
    pub fn with_email(mut self, email: impl Into<String>) -> Self {
        self.email = Some(email.into());
        self
    }

    /// Sets the purpose.
    #[must_use]
    pub fn with_purpose(mut self, purpose: impl Into<String>) -> Self {
        self.purpose = Some(purpose.into());
        self
    }

    /// Adds a custom attribute.
    #[must_use]
    pub fn with_attribute(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        if let serde_json::Value::Object(ref mut map) = self.attributes {
            map.insert(key.into(), value);
        }
        self
    }
}

/// An ephemeral identity for time-limited external access.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EphemeralIdentity {
    /// Unique identifier for this ephemeral identity.
    pub id: Uuid,
    /// The sponsor who created this identity (internal user).
    pub sponsor_id: Uuid,
    /// Tenant this identity belongs to.
    pub tenant_id: Uuid,
    /// Current state of the identity.
    pub state: EphemeralIdentityState,
    /// Scoped capabilities for this identity.
    pub capabilities: CapabilitySet,
    /// When this identity expires (hard deadline).
    pub expires_at: DateTime<Utc>,
    /// When this identity was created.
    pub created_at: DateTime<Utc>,
    /// When this identity was last activated.
    pub activated_at: Option<DateTime<Utc>>,
    /// When this identity was revoked (if applicable).
    pub revoked_at: Option<DateTime<Utc>>,
    /// Who revoked this identity (if applicable).
    pub revoked_by: Option<Uuid>,
    /// Reason for revocation.
    pub revocation_reason: Option<String>,
    /// Device binding for zero-trust verification.
    pub device_binding: Option<DeviceBinding>,
    /// Associated metadata.
    pub metadata: IdentityMetadata,
    /// Current risk score (0.0 to 1.0).
    pub risk_score: f64,
    /// Number of operations performed.
    pub operation_count: u64,
    /// Last activity timestamp.
    pub last_activity: Option<DateTime<Utc>>,
}

impl EphemeralIdentity {
    /// Creates a new ephemeral identity in Pending state.
    #[must_use]
    pub fn new(
        sponsor_id: Uuid,
        tenant_id: Uuid,
        capabilities: CapabilitySet,
        ttl: chrono::Duration,
    ) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            sponsor_id,
            tenant_id,
            state: EphemeralIdentityState::Pending,
            capabilities,
            expires_at: now + ttl,
            created_at: now,
            activated_at: None,
            revoked_at: None,
            revoked_by: None,
            revocation_reason: None,
            device_binding: None,
            metadata: IdentityMetadata::default(),
            risk_score: 0.3, // Base risk for ephemeral identities
            operation_count: 0,
            last_activity: None,
        }
    }

    /// Sets the metadata for this identity.
    #[must_use]
    pub fn with_metadata(mut self, metadata: IdentityMetadata) -> Self {
        self.metadata = metadata;
        self
    }

    /// Sets the device binding.
    #[must_use]
    pub fn with_device_binding(mut self, binding: DeviceBinding) -> Self {
        self.device_binding = Some(binding);
        self
    }

    /// Activates the identity after successful invitation redemption.
    ///
    /// # Errors
    ///
    /// Returns an error if the identity cannot be activated from its current state.
    pub fn activate(&mut self, device_binding: DeviceBinding) -> Result<()> {
        if !self.state.can_activate() {
            return Err(EphemeralError::InvalidState {
                expected: "Pending".to_string(),
                found: format!("{:?}", self.state),
            });
        }

        if self.is_expired() {
            self.state = EphemeralIdentityState::Expired;
            return Err(EphemeralError::IdentityExpired(self.id));
        }

        self.state = EphemeralIdentityState::Active;
        self.activated_at = Some(Utc::now());
        self.device_binding = Some(device_binding);
        self.last_activity = Some(Utc::now());

        Ok(())
    }

    /// Revokes the identity.
    ///
    /// # Errors
    ///
    /// Returns an error if the identity is already in a terminal state.
    pub fn revoke(&mut self, revoked_by: Uuid, reason: impl Into<String>) -> Result<()> {
        if self.state.is_terminal() {
            return Err(EphemeralError::InvalidState {
                expected: "Active or Pending".to_string(),
                found: format!("{:?}", self.state),
            });
        }

        self.state = EphemeralIdentityState::Revoked;
        self.revoked_at = Some(Utc::now());
        self.revoked_by = Some(revoked_by);
        self.revocation_reason = Some(reason.into());

        Ok(())
    }

    /// Suspends the identity for review.
    ///
    /// # Errors
    ///
    /// Returns an error if the identity is not active.
    pub fn suspend(&mut self, reason: impl Into<String>) -> Result<()> {
        if self.state != EphemeralIdentityState::Active {
            return Err(EphemeralError::InvalidState {
                expected: "Active".to_string(),
                found: format!("{:?}", self.state),
            });
        }

        self.state = EphemeralIdentityState::Suspended;
        self.revocation_reason = Some(reason.into());

        Ok(())
    }

    /// Reactivates a suspended identity.
    ///
    /// # Errors
    ///
    /// Returns an error if the identity is not suspended or has expired.
    pub fn reactivate(&mut self) -> Result<()> {
        if self.state != EphemeralIdentityState::Suspended {
            return Err(EphemeralError::InvalidState {
                expected: "Suspended".to_string(),
                found: format!("{:?}", self.state),
            });
        }

        if self.is_expired() {
            self.state = EphemeralIdentityState::Expired;
            return Err(EphemeralError::IdentityExpired(self.id));
        }

        self.state = EphemeralIdentityState::Active;
        self.revocation_reason = None;

        Ok(())
    }

    /// Checks if the identity has passed its expiration time.
    #[must_use]
    pub fn is_expired(&self) -> bool {
        Utc::now() >= self.expires_at
    }

    /// Updates the state to Expired if past expiration time.
    /// Returns true if the state was changed.
    pub fn check_expiry(&mut self) -> bool {
        if self.is_expired() && self.state != EphemeralIdentityState::Expired {
            self.state = EphemeralIdentityState::Expired;
            return true;
        }
        false
    }

    /// Returns the time remaining until expiration.
    #[must_use]
    pub fn time_remaining(&self) -> chrono::Duration {
        let remaining = self.expires_at - Utc::now();
        if remaining < chrono::Duration::zero() {
            chrono::Duration::zero()
        } else {
            remaining
        }
    }

    /// Records an operation and updates risk score.
    pub fn record_operation(&mut self) {
        self.operation_count += 1;
        self.last_activity = Some(Utc::now());
        self.update_risk_score();
    }

    /// Validates the identity can perform an operation.
    ///
    /// # Errors
    ///
    /// Returns an error if the identity is not usable.
    pub fn validate_for_operation(&mut self) -> Result<()> {
        // Check expiry first
        self.check_expiry();

        match self.state {
            EphemeralIdentityState::Active => Ok(()),
            EphemeralIdentityState::Expired => Err(EphemeralError::IdentityExpired(self.id)),
            EphemeralIdentityState::Revoked => Err(EphemeralError::IdentityRevoked(self.id)),
            _ => Err(EphemeralError::InvalidState {
                expected: "Active".to_string(),
                found: format!("{:?}", self.state),
            }),
        }
    }

    /// Validates device binding matches.
    ///
    /// # Errors
    ///
    /// Returns an error if device binding doesn't match.
    pub fn validate_device(&self, fingerprint: &str, ip_address: &str) -> Result<()> {
        match &self.device_binding {
            Some(binding) => binding.validate(fingerprint, ip_address),
            None => Ok(()), // No binding required
        }
    }

    /// Updates the risk score based on current state.
    fn update_risk_score(&mut self) {
        let mut score = 0.3; // Base ephemeral risk

        // Time-based risk: increases as expiry approaches
        let total_ttl = (self.expires_at - self.created_at).num_seconds() as f64;
        let remaining = self.time_remaining().num_seconds() as f64;
        if total_ttl > 0.0 {
            let time_factor = 1.0 - (remaining / total_ttl);
            score += time_factor * 0.2; // Up to +0.2 as expiry approaches
        }

        // Operation count risk: high activity increases risk
        if self.operation_count > 100 {
            score += 0.1;
        } else if self.operation_count > 50 {
            score += 0.05;
        }

        // Inactivity risk: long idle periods are suspicious
        if let Some(last) = self.last_activity {
            let idle_minutes = (Utc::now() - last).num_minutes();
            if idle_minutes > 30 {
                score += 0.05;
            }
        }

        self.risk_score = score.min(1.0);
    }

    /// Returns the identity's current risk level.
    #[must_use]
    pub fn risk_level(&self) -> RiskLevel {
        if self.risk_score >= 0.8 {
            RiskLevel::Critical
        } else if self.risk_score >= 0.6 {
            RiskLevel::High
        } else if self.risk_score >= 0.4 {
            RiskLevel::Medium
        } else {
            RiskLevel::Low
        }
    }
}

/// Risk level categories for ephemeral identities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RiskLevel {
    /// Low risk - normal operations.
    Low,
    /// Medium risk - increased monitoring.
    Medium,
    /// High risk - step-up auth may be required.
    High,
    /// Critical risk - operations should be blocked.
    Critical,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capabilities::Capability;

    fn create_test_identity() -> EphemeralIdentity {
        let sponsor_id = Uuid::new_v4();
        let tenant_id = Uuid::new_v4();
        let capabilities = CapabilitySet::new()
            .with_capability(Capability::new("read", "notebooks"))
            .with_capability(Capability::new("write", "notebooks"));

        EphemeralIdentity::new(
            sponsor_id,
            tenant_id,
            capabilities,
            chrono::Duration::hours(1),
        )
    }

    fn create_test_device_binding() -> DeviceBinding {
        DeviceBinding::new(
            "fp-123456".to_string(),
            "192.168.1.100".to_string(),
            "TestAgent/1.0".to_string(),
        )
    }

    // === State Tests ===

    #[test]
    fn test_state_is_usable() {
        assert!(!EphemeralIdentityState::Pending.is_usable());
        assert!(EphemeralIdentityState::Active.is_usable());
        assert!(!EphemeralIdentityState::Expired.is_usable());
        assert!(!EphemeralIdentityState::Revoked.is_usable());
        assert!(!EphemeralIdentityState::Suspended.is_usable());
    }

    #[test]
    fn test_state_can_activate() {
        assert!(EphemeralIdentityState::Pending.can_activate());
        assert!(!EphemeralIdentityState::Active.can_activate());
        assert!(!EphemeralIdentityState::Expired.can_activate());
        assert!(!EphemeralIdentityState::Revoked.can_activate());
        assert!(!EphemeralIdentityState::Suspended.can_activate());
    }

    #[test]
    fn test_state_is_terminal() {
        assert!(!EphemeralIdentityState::Pending.is_terminal());
        assert!(!EphemeralIdentityState::Active.is_terminal());
        assert!(EphemeralIdentityState::Expired.is_terminal());
        assert!(EphemeralIdentityState::Revoked.is_terminal());
        assert!(!EphemeralIdentityState::Suspended.is_terminal());
    }

    // === DeviceBinding Tests ===

    #[test]
    fn test_device_binding_new() {
        let binding = DeviceBinding::new(
            "fingerprint".to_string(),
            "10.0.0.1".to_string(),
            "Chrome/100".to_string(),
        );

        assert_eq!(binding.fingerprint, "fingerprint");
        assert_eq!(binding.ip_address, "10.0.0.1");
        assert_eq!(binding.user_agent, "Chrome/100");
        assert!(binding.location.is_none());
    }

    #[test]
    fn test_device_binding_with_location() {
        let binding = DeviceBinding::new(
            "fp".to_string(),
            "10.0.0.1".to_string(),
            "Chrome".to_string(),
        )
        .with_location("New York, US".to_string());

        assert_eq!(binding.location, Some("New York, US".to_string()));
    }

    #[test]
    fn test_device_binding_validate_success() {
        let binding = create_test_device_binding();
        let result = binding.validate("fp-123456", "192.168.1.100");
        assert!(result.is_ok());
    }

    #[test]
    fn test_device_binding_validate_fingerprint_mismatch() {
        let binding = create_test_device_binding();
        let result = binding.validate("wrong-fingerprint", "192.168.1.100");
        assert!(matches!(
            result,
            Err(EphemeralError::DeviceBindingMismatch { .. })
        ));
    }

    #[test]
    fn test_device_binding_allows_ip_change() {
        let binding = create_test_device_binding();
        // Different IP should still pass (for mobile users)
        let result = binding.validate("fp-123456", "10.0.0.1");
        assert!(result.is_ok());
    }

    // === IdentityMetadata Tests ===

    #[test]
    fn test_metadata_default() {
        let metadata = IdentityMetadata::default();
        assert!(metadata.display_name.is_none());
        assert!(metadata.email.is_none());
        assert!(metadata.purpose.is_none());
    }

    #[test]
    fn test_metadata_builder() {
        let metadata = IdentityMetadata::with_display_name("Alice")
            .with_email("alice@example.com")
            .with_purpose("Code review")
            .with_attribute("team", serde_json::json!("platform"));

        assert_eq!(metadata.display_name, Some("Alice".to_string()));
        assert_eq!(metadata.email, Some("alice@example.com".to_string()));
        assert_eq!(metadata.purpose, Some("Code review".to_string()));
        assert_eq!(metadata.attributes["team"], "platform");
    }

    // === EphemeralIdentity Tests ===

    #[test]
    fn test_identity_new() {
        let identity = create_test_identity();

        assert_eq!(identity.state, EphemeralIdentityState::Pending);
        assert!(identity.activated_at.is_none());
        assert!(identity.device_binding.is_none());
        assert!(!identity.is_expired());
        assert_eq!(identity.operation_count, 0);
        assert!(identity.risk_score >= 0.3); // Base ephemeral risk
    }

    #[test]
    fn test_identity_with_metadata() {
        let metadata = IdentityMetadata::with_display_name("Bob");
        let identity = create_test_identity().with_metadata(metadata);

        assert_eq!(identity.metadata.display_name, Some("Bob".to_string()));
    }

    #[test]
    fn test_identity_activate_success() {
        let mut identity = create_test_identity();
        let binding = create_test_device_binding();

        let result = identity.activate(binding);

        assert!(result.is_ok());
        assert_eq!(identity.state, EphemeralIdentityState::Active);
        assert!(identity.activated_at.is_some());
        assert!(identity.device_binding.is_some());
    }

    #[test]
    fn test_identity_activate_already_active() {
        let mut identity = create_test_identity();
        let binding = create_test_device_binding();

        identity.activate(binding.clone()).unwrap();
        let result = identity.activate(binding);

        assert!(matches!(result, Err(EphemeralError::InvalidState { .. })));
    }

    #[test]
    fn test_identity_revoke_success() {
        let mut identity = create_test_identity();
        let binding = create_test_device_binding();
        identity.activate(binding).unwrap();

        let admin_id = Uuid::new_v4();
        let result = identity.revoke(admin_id, "Security concern");

        assert!(result.is_ok());
        assert_eq!(identity.state, EphemeralIdentityState::Revoked);
        assert!(identity.revoked_at.is_some());
        assert_eq!(identity.revoked_by, Some(admin_id));
        assert_eq!(
            identity.revocation_reason,
            Some("Security concern".to_string())
        );
    }

    #[test]
    fn test_identity_revoke_already_revoked() {
        let mut identity = create_test_identity();
        let admin_id = Uuid::new_v4();

        identity.revoke(admin_id, "First revoke").unwrap();
        let result = identity.revoke(admin_id, "Second revoke");

        assert!(matches!(result, Err(EphemeralError::InvalidState { .. })));
    }

    #[test]
    fn test_identity_suspend_success() {
        let mut identity = create_test_identity();
        let binding = create_test_device_binding();
        identity.activate(binding).unwrap();

        let result = identity.suspend("Suspicious activity");

        assert!(result.is_ok());
        assert_eq!(identity.state, EphemeralIdentityState::Suspended);
    }

    #[test]
    fn test_identity_suspend_not_active() {
        let mut identity = create_test_identity();
        let result = identity.suspend("Test");

        assert!(matches!(result, Err(EphemeralError::InvalidState { .. })));
    }

    #[test]
    fn test_identity_reactivate_success() {
        let mut identity = create_test_identity();
        let binding = create_test_device_binding();
        identity.activate(binding).unwrap();
        identity.suspend("Review").unwrap();

        let result = identity.reactivate();

        assert!(result.is_ok());
        assert_eq!(identity.state, EphemeralIdentityState::Active);
    }

    #[test]
    fn test_identity_time_remaining() {
        let identity = create_test_identity();
        let remaining = identity.time_remaining();

        // Should be close to 1 hour (created with 1 hour TTL)
        assert!(remaining.num_minutes() >= 59);
    }

    #[test]
    fn test_identity_record_operation() {
        let mut identity = create_test_identity();
        let binding = create_test_device_binding();
        identity.activate(binding).unwrap();

        assert_eq!(identity.operation_count, 0);

        identity.record_operation();
        assert_eq!(identity.operation_count, 1);
        assert!(identity.last_activity.is_some());

        identity.record_operation();
        assert_eq!(identity.operation_count, 2);
    }

    #[test]
    fn test_identity_validate_for_operation_active() {
        let mut identity = create_test_identity();
        let binding = create_test_device_binding();
        identity.activate(binding).unwrap();

        let result = identity.validate_for_operation();
        assert!(result.is_ok());
    }

    #[test]
    fn test_identity_validate_for_operation_pending() {
        let mut identity = create_test_identity();
        let result = identity.validate_for_operation();

        assert!(matches!(result, Err(EphemeralError::InvalidState { .. })));
    }

    #[test]
    fn test_identity_validate_for_operation_revoked() {
        let mut identity = create_test_identity();
        identity.revoke(Uuid::new_v4(), "Test").unwrap();

        let result = identity.validate_for_operation();
        assert!(matches!(result, Err(EphemeralError::IdentityRevoked(_))));
    }

    #[test]
    fn test_identity_validate_device() {
        let mut identity = create_test_identity();
        let binding = create_test_device_binding();
        identity.activate(binding).unwrap();

        let result = identity.validate_device("fp-123456", "192.168.1.100");
        assert!(result.is_ok());
    }

    #[test]
    fn test_identity_risk_level() {
        let mut identity = create_test_identity();

        // Default should be Low or Medium
        let level = identity.risk_level();
        assert!(matches!(level, RiskLevel::Low | RiskLevel::Medium));

        // Manually increase risk score
        identity.risk_score = 0.85;
        assert_eq!(identity.risk_level(), RiskLevel::Critical);

        identity.risk_score = 0.65;
        assert_eq!(identity.risk_level(), RiskLevel::High);

        identity.risk_score = 0.45;
        assert_eq!(identity.risk_level(), RiskLevel::Medium);

        identity.risk_score = 0.2;
        assert_eq!(identity.risk_level(), RiskLevel::Low);
    }

    #[test]
    fn test_identity_check_expiry() {
        let mut identity = EphemeralIdentity::new(
            Uuid::new_v4(),
            Uuid::new_v4(),
            CapabilitySet::new(),
            chrono::Duration::milliseconds(-1), // Already expired
        );

        let changed = identity.check_expiry();
        assert!(changed);
        assert_eq!(identity.state, EphemeralIdentityState::Expired);
    }

    #[test]
    fn test_identity_serialization() {
        let identity = create_test_identity();
        let json = serde_json::to_string(&identity).unwrap();
        let restored: EphemeralIdentity = serde_json::from_str(&json).unwrap();

        assert_eq!(identity.id, restored.id);
        assert_eq!(identity.sponsor_id, restored.sponsor_id);
        assert_eq!(identity.state, restored.state);
    }

    // === RiskLevel Tests ===

    #[test]
    fn test_risk_level_serialization() {
        let level = RiskLevel::High;
        let json = serde_json::to_string(&level).unwrap();
        assert_eq!(json, "\"high\"");

        let restored: RiskLevel = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, RiskLevel::High);
    }
}
