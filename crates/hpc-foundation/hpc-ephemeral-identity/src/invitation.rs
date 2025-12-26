//! Invitation links for creating ephemeral identities.
//!
//! Invitations use a two-factor approach:
//! 1. A magic link URL containing an encrypted token
//! 2. A redemption code (like 2FA) that must be entered separately

use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use chrono::{DateTime, Utc};
use hpc_crypto::hash;
use hpc_vault::{MasterKey, VaultEncryption};
use rand::Rng;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::capabilities::CapabilitySet;
use crate::error::{EphemeralError, Result};

/// Status of an invitation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InvitationStatus {
    /// Invitation is active and can be redeemed.
    Pending,
    /// Invitation has been successfully redeemed.
    Redeemed,
    /// Invitation has expired.
    Expired,
    /// Invitation was revoked before redemption.
    Revoked,
}

impl InvitationStatus {
    /// Returns true if the invitation can be redeemed.
    #[must_use]
    pub fn is_redeemable(&self) -> bool {
        matches!(self, InvitationStatus::Pending)
    }
}

/// Payload encrypted within the invitation token.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InvitationPayload {
    /// Invitation ID.
    pub invitation_id: Uuid,
    /// Sponsor who created this invitation.
    pub sponsor_id: Uuid,
    /// Tenant ID.
    pub tenant_id: Uuid,
    /// Capabilities to grant.
    pub capabilities: CapabilitySet,
    /// Time-to-live for the created identity.
    pub identity_ttl_seconds: i64,
    /// When this invitation expires.
    pub expires_at: DateTime<Utc>,
    /// Hash of the redemption code.
    pub redemption_code_hash: String,
    /// Maximum allowed redemption attempts.
    pub max_attempts: u32,
    /// Optional metadata.
    pub metadata: serde_json::Value,
}

/// An invitation link for creating an ephemeral identity.
#[derive(Debug, Clone)]
pub struct InvitationLink {
    /// Unique invitation ID.
    pub invitation_id: Uuid,
    /// The shareable magic link URL.
    pub link: String,
    /// The encrypted token (extracted from link).
    pub token: String,
    /// The redemption code (to be shared separately).
    pub redemption_code: String,
    /// When this invitation expires.
    pub expires_at: DateTime<Utc>,
    /// Sponsor who created this invitation.
    pub sponsor_id: Uuid,
    /// Tenant ID.
    pub tenant_id: Uuid,
    /// Capabilities to grant.
    pub capabilities: CapabilitySet,
    /// TTL for the created identity.
    pub identity_ttl: chrono::Duration,
    /// Current status.
    pub status: InvitationStatus,
    /// Number of redemption attempts made.
    pub attempts: u32,
    /// Maximum allowed attempts.
    pub max_attempts: u32,
    /// When the invitation was created.
    pub created_at: DateTime<Utc>,
    /// When the invitation was redeemed (if applicable).
    pub redeemed_at: Option<DateTime<Utc>>,
    /// ID of the identity created (if redeemed).
    pub identity_id: Option<Uuid>,
}

/// Builder for creating invitation links.
#[derive(Debug)]
pub struct InvitationBuilder {
    sponsor_id: Uuid,
    tenant_id: Uuid,
    capabilities: CapabilitySet,
    identity_ttl: chrono::Duration,
    invitation_ttl: chrono::Duration,
    max_attempts: u32,
    base_url: String,
    metadata: serde_json::Value,
}

impl InvitationBuilder {
    /// Creates a new invitation builder.
    #[must_use]
    pub fn new(sponsor_id: Uuid, tenant_id: Uuid) -> Self {
        Self {
            sponsor_id,
            tenant_id,
            capabilities: CapabilitySet::new(),
            identity_ttl: chrono::Duration::hours(24),
            invitation_ttl: chrono::Duration::hours(48),
            max_attempts: 3,
            base_url: "https://horizon.app/join".to_string(),
            metadata: serde_json::Value::Null,
        }
    }

    /// Sets the capabilities for the invitation.
    #[must_use]
    pub fn with_capabilities(mut self, capabilities: CapabilitySet) -> Self {
        self.capabilities = capabilities;
        self
    }

    /// Sets the TTL for the created identity.
    #[must_use]
    pub fn with_identity_ttl(mut self, ttl: chrono::Duration) -> Self {
        self.identity_ttl = ttl;
        self
    }

    /// Sets the TTL for the invitation itself.
    #[must_use]
    pub fn with_invitation_ttl(mut self, ttl: chrono::Duration) -> Self {
        self.invitation_ttl = ttl;
        self
    }

    /// Sets the maximum redemption attempts.
    #[must_use]
    pub fn with_max_attempts(mut self, max: u32) -> Self {
        self.max_attempts = max;
        self
    }

    /// Sets the base URL for the invitation link.
    #[must_use]
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Sets custom metadata.
    #[must_use]
    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = metadata;
        self
    }

    /// Builds the invitation link.
    ///
    /// # Errors
    ///
    /// Returns an error if encryption fails.
    pub fn build(self, encryption_key: &MasterKey) -> Result<InvitationLink> {
        let invitation_id = Uuid::new_v4();
        let now = Utc::now();
        let expires_at = now + self.invitation_ttl;

        // Generate a secure redemption code (6 alphanumeric characters)
        let redemption_code = generate_redemption_code();
        let redemption_code_hash = hash_redemption_code(&redemption_code);

        // Create the payload
        let payload = InvitationPayload {
            invitation_id,
            sponsor_id: self.sponsor_id,
            tenant_id: self.tenant_id,
            capabilities: self.capabilities.clone(),
            identity_ttl_seconds: self.identity_ttl.num_seconds(),
            expires_at,
            redemption_code_hash,
            max_attempts: self.max_attempts,
            metadata: self.metadata,
        };

        // Encrypt the payload - we need to create a new MasterKey from the provided one
        // since VaultEncryption takes ownership
        let payload_json = serde_json::to_vec(&payload)?;
        let key_for_encryption = MasterKey::from_base64(&encryption_key.to_base64())
            .map_err(|e| EphemeralError::CryptoError(e.to_string()))?;
        let vault = VaultEncryption::new(key_for_encryption);
        let encrypted = vault
            .encrypt(&payload_json)
            .map_err(|e| EphemeralError::CryptoError(e.to_string()))?;

        // Encode the encrypted payload
        let encrypted_json = serde_json::to_vec(&encrypted)?;
        let token = URL_SAFE_NO_PAD.encode(&encrypted_json);

        // Build the URL
        let link = format!("{}/{token}", self.base_url);

        Ok(InvitationLink {
            invitation_id,
            link,
            token,
            redemption_code,
            expires_at,
            sponsor_id: self.sponsor_id,
            tenant_id: self.tenant_id,
            capabilities: self.capabilities,
            identity_ttl: self.identity_ttl,
            status: InvitationStatus::Pending,
            attempts: 0,
            max_attempts: self.max_attempts,
            created_at: now,
            redeemed_at: None,
            identity_id: None,
        })
    }
}

impl InvitationLink {
    /// Creates a new invitation builder.
    #[must_use]
    pub fn builder(sponsor_id: Uuid, tenant_id: Uuid) -> InvitationBuilder {
        InvitationBuilder::new(sponsor_id, tenant_id)
    }

    /// Checks if the invitation has expired.
    #[must_use]
    pub fn is_expired(&self) -> bool {
        Utc::now() >= self.expires_at
    }

    /// Checks if the invitation can be redeemed.
    #[must_use]
    pub fn can_redeem(&self) -> bool {
        self.status.is_redeemable() && !self.is_expired() && self.attempts < self.max_attempts
    }

    /// Returns time remaining until expiration.
    #[must_use]
    pub fn time_remaining(&self) -> chrono::Duration {
        let remaining = self.expires_at - Utc::now();
        if remaining < chrono::Duration::zero() {
            chrono::Duration::zero()
        } else {
            remaining
        }
    }

    /// Revokes the invitation.
    ///
    /// # Errors
    ///
    /// Returns an error if the invitation is not in pending state.
    pub fn revoke(&mut self) -> Result<()> {
        if self.status != InvitationStatus::Pending {
            return Err(EphemeralError::InvalidState {
                expected: "Pending".to_string(),
                found: format!("{:?}", self.status),
            });
        }
        self.status = InvitationStatus::Revoked;
        Ok(())
    }

    /// Marks the invitation as expired.
    pub fn mark_expired(&mut self) {
        if self.status == InvitationStatus::Pending {
            self.status = InvitationStatus::Expired;
        }
    }
}

/// Result of a successful invitation redemption.
#[derive(Debug, Clone)]
pub struct RedemptionResult {
    /// The created ephemeral identity ID.
    pub identity_id: Uuid,
    /// Sponsor ID.
    pub sponsor_id: Uuid,
    /// Tenant ID.
    pub tenant_id: Uuid,
    /// Capabilities granted.
    pub capabilities: CapabilitySet,
    /// TTL for the identity.
    pub identity_ttl: chrono::Duration,
    /// Custom metadata from the invitation.
    pub metadata: serde_json::Value,
}

/// Redeemer for processing invitation tokens.
#[derive(Debug)]
pub struct InvitationRedeemer {
    /// Encryption key stored as base64 (MasterKey doesn't implement Clone).
    encryption_key_b64: String,
}

impl InvitationRedeemer {
    /// Creates a new invitation redeemer.
    #[must_use]
    pub fn new(encryption_key: MasterKey) -> Self {
        Self {
            encryption_key_b64: encryption_key.to_base64(),
        }
    }

    /// Gets the encryption key (recreated from stored base64).
    fn encryption_key(&self) -> MasterKey {
        MasterKey::from_base64(&self.encryption_key_b64)
            .expect("Stored encryption key should always be valid")
    }

    /// Validates and decrypts an invitation token.
    ///
    /// # Errors
    ///
    /// Returns an error if the token is invalid or expired.
    pub fn validate_token(&self, token: &str) -> Result<InvitationPayload> {
        // Decode the token
        let encrypted_bytes = URL_SAFE_NO_PAD
            .decode(token)
            .map_err(|e| EphemeralError::TokenValidationFailed(format!("Invalid encoding: {e}")))?;

        let encrypted: hpc_vault::EncryptedCredential =
            serde_json::from_slice(&encrypted_bytes)?;

        // Decrypt
        let vault = VaultEncryption::new(self.encryption_key());
        let decrypted = vault
            .decrypt(&encrypted)
            .map_err(|e| EphemeralError::DecryptionFailed(e.to_string()))?;

        // Parse payload
        let payload: InvitationPayload = serde_json::from_slice(decrypted.expose_secret())?;

        // Check expiration
        if Utc::now() >= payload.expires_at {
            return Err(EphemeralError::InvitationExpired(payload.invitation_id));
        }

        Ok(payload)
    }

    /// Redeems an invitation with the given token and redemption code.
    ///
    /// # Errors
    ///
    /// Returns an error if the token is invalid, code doesn't match, or max attempts exceeded.
    pub fn redeem(
        &self,
        token: &str,
        redemption_code: &str,
        attempts_so_far: u32,
    ) -> Result<RedemptionResult> {
        let payload = self.validate_token(token)?;

        // Check attempts
        if attempts_so_far >= payload.max_attempts {
            return Err(EphemeralError::MaxRedemptionAttemptsExceeded(
                payload.invitation_id,
            ));
        }

        // Verify redemption code
        let provided_hash = hash_redemption_code(redemption_code);
        if provided_hash != payload.redemption_code_hash {
            return Err(EphemeralError::InvalidRedemptionCode);
        }

        // Create redemption result
        Ok(RedemptionResult {
            identity_id: Uuid::new_v4(),
            sponsor_id: payload.sponsor_id,
            tenant_id: payload.tenant_id,
            capabilities: payload.capabilities,
            identity_ttl: chrono::Duration::seconds(payload.identity_ttl_seconds),
            metadata: payload.metadata,
        })
    }
}

/// Generates a secure 6-character alphanumeric redemption code.
fn generate_redemption_code() -> String {
    const CHARSET: &[u8] = b"ABCDEFGHJKLMNPQRSTUVWXYZ23456789"; // Excludes confusing chars
    let mut rng = rand::thread_rng();
    (0..6)
        .map(|_| {
            let idx = rng.gen_range(0..CHARSET.len());
            CHARSET[idx] as char
        })
        .collect()
}

/// Hashes a redemption code for secure storage.
fn hash_redemption_code(code: &str) -> String {
    let normalized = code.to_uppercase().trim().to_string();
    hash(normalized.as_bytes()).to_hex()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capabilities::Capability;

    fn create_test_encryption_key() -> MasterKey {
        MasterKey::generate()
    }

    fn create_test_capabilities() -> CapabilitySet {
        CapabilitySet::new()
            .with_capability(Capability::new("read", "notebooks"))
            .with_capability(Capability::new("write", "notebooks"))
    }

    // === InvitationStatus Tests ===

    #[test]
    fn test_invitation_status_is_redeemable() {
        assert!(InvitationStatus::Pending.is_redeemable());
        assert!(!InvitationStatus::Redeemed.is_redeemable());
        assert!(!InvitationStatus::Expired.is_redeemable());
        assert!(!InvitationStatus::Revoked.is_redeemable());
    }

    // === InvitationBuilder Tests ===

    #[test]
    fn test_invitation_builder_defaults() {
        let sponsor_id = Uuid::new_v4();
        let tenant_id = Uuid::new_v4();
        let key = create_test_encryption_key();

        let invitation = InvitationBuilder::new(sponsor_id, tenant_id)
            .build(&key)
            .unwrap();

        assert_eq!(invitation.sponsor_id, sponsor_id);
        assert_eq!(invitation.tenant_id, tenant_id);
        assert_eq!(invitation.status, InvitationStatus::Pending);
        assert_eq!(invitation.attempts, 0);
        assert_eq!(invitation.max_attempts, 3);
        assert!(invitation.link.starts_with("https://horizon.app/join/"));
    }

    #[test]
    fn test_invitation_builder_with_capabilities() {
        let sponsor_id = Uuid::new_v4();
        let tenant_id = Uuid::new_v4();
        let key = create_test_encryption_key();
        let capabilities = create_test_capabilities();

        let invitation = InvitationBuilder::new(sponsor_id, tenant_id)
            .with_capabilities(capabilities.clone())
            .build(&key)
            .unwrap();

        assert!(invitation.capabilities.allows("read", "notebooks"));
        assert!(invitation.capabilities.allows("write", "notebooks"));
    }

    #[test]
    fn test_invitation_builder_with_custom_ttl() {
        let sponsor_id = Uuid::new_v4();
        let tenant_id = Uuid::new_v4();
        let key = create_test_encryption_key();

        let invitation = InvitationBuilder::new(sponsor_id, tenant_id)
            .with_identity_ttl(chrono::Duration::hours(2))
            .with_invitation_ttl(chrono::Duration::hours(4))
            .build(&key)
            .unwrap();

        assert_eq!(invitation.identity_ttl, chrono::Duration::hours(2));
        // Invitation should expire in ~4 hours
        let remaining = invitation.time_remaining();
        assert!(remaining.num_hours() >= 3);
    }

    #[test]
    fn test_invitation_builder_with_max_attempts() {
        let sponsor_id = Uuid::new_v4();
        let tenant_id = Uuid::new_v4();
        let key = create_test_encryption_key();

        let invitation = InvitationBuilder::new(sponsor_id, tenant_id)
            .with_max_attempts(5)
            .build(&key)
            .unwrap();

        assert_eq!(invitation.max_attempts, 5);
    }

    #[test]
    fn test_invitation_builder_with_base_url() {
        let sponsor_id = Uuid::new_v4();
        let tenant_id = Uuid::new_v4();
        let key = create_test_encryption_key();

        let invitation = InvitationBuilder::new(sponsor_id, tenant_id)
            .with_base_url("https://custom.example.com/invite")
            .build(&key)
            .unwrap();

        assert!(invitation.link.starts_with("https://custom.example.com/invite/"));
    }

    // === InvitationLink Tests ===

    #[test]
    fn test_invitation_link_is_expired() {
        let sponsor_id = Uuid::new_v4();
        let tenant_id = Uuid::new_v4();
        let key = create_test_encryption_key();

        let mut invitation = InvitationBuilder::new(sponsor_id, tenant_id)
            .build(&key)
            .unwrap();

        assert!(!invitation.is_expired());

        // Manually expire
        invitation.expires_at = Utc::now() - chrono::Duration::hours(1);
        assert!(invitation.is_expired());
    }

    #[test]
    fn test_invitation_link_can_redeem() {
        let sponsor_id = Uuid::new_v4();
        let tenant_id = Uuid::new_v4();
        let key = create_test_encryption_key();

        let mut invitation = InvitationBuilder::new(sponsor_id, tenant_id)
            .with_max_attempts(3)
            .build(&key)
            .unwrap();

        assert!(invitation.can_redeem());

        // After max attempts
        invitation.attempts = 3;
        assert!(!invitation.can_redeem());

        // Reset attempts, but revoke
        invitation.attempts = 0;
        invitation.status = InvitationStatus::Revoked;
        assert!(!invitation.can_redeem());
    }

    #[test]
    fn test_invitation_link_revoke() {
        let sponsor_id = Uuid::new_v4();
        let tenant_id = Uuid::new_v4();
        let key = create_test_encryption_key();

        let mut invitation = InvitationBuilder::new(sponsor_id, tenant_id)
            .build(&key)
            .unwrap();

        let result = invitation.revoke();
        assert!(result.is_ok());
        assert_eq!(invitation.status, InvitationStatus::Revoked);
    }

    #[test]
    fn test_invitation_link_revoke_already_revoked() {
        let sponsor_id = Uuid::new_v4();
        let tenant_id = Uuid::new_v4();
        let key = create_test_encryption_key();

        let mut invitation = InvitationBuilder::new(sponsor_id, tenant_id)
            .build(&key)
            .unwrap();

        invitation.revoke().unwrap();
        let result = invitation.revoke();

        assert!(matches!(result, Err(EphemeralError::InvalidState { .. })));
    }

    #[test]
    fn test_invitation_link_mark_expired() {
        let sponsor_id = Uuid::new_v4();
        let tenant_id = Uuid::new_v4();
        let key = create_test_encryption_key();

        let mut invitation = InvitationBuilder::new(sponsor_id, tenant_id)
            .build(&key)
            .unwrap();

        invitation.mark_expired();
        assert_eq!(invitation.status, InvitationStatus::Expired);
    }

    #[test]
    fn test_invitation_link_redemption_code_format() {
        let sponsor_id = Uuid::new_v4();
        let tenant_id = Uuid::new_v4();
        let key = create_test_encryption_key();

        let invitation = InvitationBuilder::new(sponsor_id, tenant_id)
            .build(&key)
            .unwrap();

        // Code should be 6 characters
        assert_eq!(invitation.redemption_code.len(), 6);
        // Code should be uppercase alphanumeric
        assert!(invitation.redemption_code.chars().all(|c| c.is_ascii_alphanumeric()));
    }

    // === InvitationRedeemer Tests ===

    #[test]
    fn test_redeemer_validate_token() {
        let sponsor_id = Uuid::new_v4();
        let tenant_id = Uuid::new_v4();
        let key = create_test_encryption_key();

        let invitation = InvitationBuilder::new(sponsor_id, tenant_id)
            .with_capabilities(create_test_capabilities())
            .build(&key)
            .unwrap();

        let redeemer = InvitationRedeemer::new(key);
        let result = redeemer.validate_token(&invitation.token);

        assert!(result.is_ok());
    }

    #[test]
    fn test_redeemer_validate_token_wrong_key() {
        let sponsor_id = Uuid::new_v4();
        let tenant_id = Uuid::new_v4();
        let key = create_test_encryption_key();
        let wrong_key = create_test_encryption_key();

        let invitation = InvitationBuilder::new(sponsor_id, tenant_id)
            .build(&key)
            .unwrap();

        let redeemer = InvitationRedeemer::new(wrong_key);
        let result = redeemer.validate_token(&invitation.token);

        assert!(matches!(result, Err(EphemeralError::DecryptionFailed(_))));
    }

    #[test]
    fn test_redeemer_redeem_success() {
        let sponsor_id = Uuid::new_v4();
        let tenant_id = Uuid::new_v4();
        let key = create_test_encryption_key();

        let invitation = InvitationBuilder::new(sponsor_id, tenant_id)
            .with_capabilities(create_test_capabilities())
            .build(&key)
            .unwrap();

        let redeemer = InvitationRedeemer::new(key);
        let result = redeemer.redeem(&invitation.token, &invitation.redemption_code, 0);

        assert!(result.is_ok());
        let redemption = result.unwrap();
        assert_eq!(redemption.sponsor_id, sponsor_id);
        assert_eq!(redemption.tenant_id, tenant_id);
    }

    #[test]
    fn test_redeemer_redeem_wrong_code() {
        let sponsor_id = Uuid::new_v4();
        let tenant_id = Uuid::new_v4();
        let key = create_test_encryption_key();

        let invitation = InvitationBuilder::new(sponsor_id, tenant_id)
            .build(&key)
            .unwrap();

        let redeemer = InvitationRedeemer::new(key);
        let result = redeemer.redeem(&invitation.token, "WRONG1", 0);

        assert!(matches!(result, Err(EphemeralError::InvalidRedemptionCode)));
    }

    #[test]
    fn test_redeemer_redeem_case_insensitive_code() {
        let sponsor_id = Uuid::new_v4();
        let tenant_id = Uuid::new_v4();
        let key = create_test_encryption_key();

        let invitation = InvitationBuilder::new(sponsor_id, tenant_id)
            .build(&key)
            .unwrap();

        let redeemer = InvitationRedeemer::new(key);
        // Use lowercase version of code
        let lowercase_code = invitation.redemption_code.to_lowercase();
        let result = redeemer.redeem(&invitation.token, &lowercase_code, 0);

        assert!(result.is_ok());
    }

    #[test]
    fn test_redeemer_redeem_max_attempts_exceeded() {
        let sponsor_id = Uuid::new_v4();
        let tenant_id = Uuid::new_v4();
        let key = create_test_encryption_key();

        let invitation = InvitationBuilder::new(sponsor_id, tenant_id)
            .with_max_attempts(3)
            .build(&key)
            .unwrap();

        let redeemer = InvitationRedeemer::new(key);
        let result = redeemer.redeem(&invitation.token, &invitation.redemption_code, 3);

        assert!(matches!(
            result,
            Err(EphemeralError::MaxRedemptionAttemptsExceeded(_))
        ));
    }

    #[test]
    fn test_redeemer_validate_expired_token() {
        let sponsor_id = Uuid::new_v4();
        let tenant_id = Uuid::new_v4();
        let key = create_test_encryption_key();

        // Create invitation with very short TTL
        let invitation = InvitationBuilder::new(sponsor_id, tenant_id)
            .with_invitation_ttl(chrono::Duration::milliseconds(-1)) // Already expired
            .build(&key)
            .unwrap();

        let redeemer = InvitationRedeemer::new(key);
        let result = redeemer.validate_token(&invitation.token);

        assert!(matches!(result, Err(EphemeralError::InvitationExpired(_))));
    }

    // === Utility Function Tests ===

    #[test]
    fn test_generate_redemption_code() {
        let code1 = generate_redemption_code();
        let code2 = generate_redemption_code();

        assert_eq!(code1.len(), 6);
        assert_eq!(code2.len(), 6);
        // Very unlikely to be equal
        assert_ne!(code1, code2);
    }

    #[test]
    fn test_hash_redemption_code_consistent() {
        let code = "ABC123";
        let hash1 = hash_redemption_code(code);
        let hash2 = hash_redemption_code(code);

        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_hash_redemption_code_case_insensitive() {
        let hash_upper = hash_redemption_code("ABC123");
        let hash_lower = hash_redemption_code("abc123");

        assert_eq!(hash_upper, hash_lower);
    }

    // === Serialization Tests ===

    #[test]
    fn test_invitation_status_serialization() {
        let status = InvitationStatus::Pending;
        let json = serde_json::to_string(&status).unwrap();
        assert_eq!(json, "\"pending\"");

        let restored: InvitationStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, InvitationStatus::Pending);
    }
}
