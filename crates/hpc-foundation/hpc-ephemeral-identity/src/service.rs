//! Ephemeral identity service for managing the full lifecycle.
//!
//! This service provides the main API for:
//! - Creating invitations
//! - Redeeming invitations to create identities
//! - Managing identity lifecycle (activate, suspend, revoke)
//! - Token generation and validation
//! - Background cleanup of expired entities

use chrono::{Duration, Utc};
use dashmap::DashMap;
use hpc_crypto::KeyPair;
use hpc_vault::MasterKey;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::capabilities::CapabilitySet;
use crate::error::{EphemeralError, Result};
use crate::identity::{DeviceBinding, EphemeralIdentity, EphemeralIdentityState, IdentityMetadata};
use crate::invitation::{InvitationBuilder, InvitationLink, InvitationRedeemer, InvitationStatus};
use crate::token::{EphemeralToken, TokenClaims, TokenValidation, TokenValidator};

/// Configuration for the ephemeral identity service.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceConfig {
    /// Default TTL for ephemeral identities.
    pub default_identity_ttl: Duration,
    /// Default TTL for invitations.
    pub default_invitation_ttl: Duration,
    /// Maximum ephemeral identities per sponsor.
    pub max_identities_per_sponsor: usize,
    /// Maximum active invitations per sponsor.
    pub max_invitations_per_sponsor: usize,
    /// Token issuer identifier.
    pub token_issuer: String,
    /// Base URL for invitation links.
    pub invitation_base_url: String,
    /// How often to run cleanup (in seconds).
    pub cleanup_interval_secs: u64,
    /// Whether to encrypt tokens.
    pub encrypt_tokens: bool,
}

impl Default for ServiceConfig {
    fn default() -> Self {
        Self {
            default_identity_ttl: Duration::hours(24),
            default_invitation_ttl: Duration::hours(48),
            max_identities_per_sponsor: 10,
            max_invitations_per_sponsor: 20,
            token_issuer: "hpc-ephemeral".to_string(),
            invitation_base_url: "https://horizon.app/join".to_string(),
            cleanup_interval_secs: 300, // 5 minutes
            encrypt_tokens: true,
        }
    }
}

/// Statistics about the service state.
#[derive(Debug, Clone, Default)]
pub struct ServiceStats {
    /// Total identities ever created.
    pub total_identities_created: u64,
    /// Currently active identities.
    pub active_identities: usize,
    /// Pending invitations.
    pub pending_invitations: usize,
    /// Identities expired through cleanup.
    pub identities_expired: u64,
    /// Invitations expired through cleanup.
    pub invitations_expired: u64,
    /// Total tokens issued.
    pub tokens_issued: u64,
    /// Total tokens validated.
    pub tokens_validated: u64,
}

/// The main ephemeral identity service.
#[derive(Debug)]
pub struct EphemeralIdentityService {
    config: ServiceConfig,
    /// Signing keypair for tokens.
    keypair: KeyPair,
    /// Encryption key stored as base64 (MasterKey doesn't implement Clone).
    encryption_key_b64: String,
    /// Active identities by ID.
    identities: DashMap<Uuid, EphemeralIdentity>,
    /// Identities by sponsor ID.
    identities_by_sponsor: DashMap<Uuid, Vec<Uuid>>,
    /// Active invitations by ID.
    invitations: DashMap<Uuid, InvitationLink>,
    /// Invitations by sponsor ID.
    invitations_by_sponsor: DashMap<Uuid, Vec<Uuid>>,
    /// Token to identity mapping.
    token_index: DashMap<String, Uuid>,
    /// Service statistics.
    stats: RwLock<ServiceStats>,
}

impl EphemeralIdentityService {
    /// Creates a new ephemeral identity service.
    #[must_use]
    pub fn new(config: ServiceConfig) -> Self {
        let encryption_key = MasterKey::generate();
        Self {
            config,
            keypair: KeyPair::generate(),
            encryption_key_b64: encryption_key.to_base64(),
            identities: DashMap::new(),
            identities_by_sponsor: DashMap::new(),
            invitations: DashMap::new(),
            invitations_by_sponsor: DashMap::new(),
            token_index: DashMap::new(),
            stats: RwLock::new(ServiceStats::default()),
        }
    }

    /// Creates a service with provided keys.
    #[must_use]
    pub fn with_keys(config: ServiceConfig, keypair: KeyPair, encryption_key: MasterKey) -> Self {
        Self {
            config,
            keypair,
            encryption_key_b64: encryption_key.to_base64(),
            identities: DashMap::new(),
            identities_by_sponsor: DashMap::new(),
            invitations: DashMap::new(),
            invitations_by_sponsor: DashMap::new(),
            token_index: DashMap::new(),
            stats: RwLock::new(ServiceStats::default()),
        }
    }

    /// Gets the encryption key (recreated from stored base64).
    fn encryption_key(&self) -> MasterKey {
        MasterKey::from_base64(&self.encryption_key_b64)
            .expect("Stored encryption key should always be valid")
    }

    /// Returns the service configuration.
    #[must_use]
    pub fn config(&self) -> &ServiceConfig {
        &self.config
    }

    /// Returns the public key for token verification.
    #[must_use]
    pub fn public_key(&self) -> hpc_crypto::PublicKey {
        self.keypair.public_key()
    }

    /// Returns current service statistics.
    pub async fn stats(&self) -> ServiceStats {
        self.stats.read().await.clone()
    }

    // === Invitation Management ===

    /// Creates a new invitation.
    ///
    /// # Errors
    ///
    /// Returns an error if the sponsor has too many active invitations.
    pub async fn create_invitation(
        &self,
        sponsor_id: Uuid,
        tenant_id: Uuid,
        capabilities: CapabilitySet,
        identity_ttl: Option<Duration>,
        invitation_ttl: Option<Duration>,
    ) -> Result<InvitationLink> {
        // Check sponsor limits
        let sponsor_invitations = self
            .invitations_by_sponsor
            .get(&sponsor_id)
            .map(|v| v.len())
            .unwrap_or(0);

        if sponsor_invitations >= self.config.max_invitations_per_sponsor {
            return Err(EphemeralError::MaxIdentitiesExceeded(
                self.config.max_invitations_per_sponsor,
            ));
        }

        let invitation = InvitationBuilder::new(sponsor_id, tenant_id)
            .with_capabilities(capabilities)
            .with_identity_ttl(identity_ttl.unwrap_or(self.config.default_identity_ttl))
            .with_invitation_ttl(invitation_ttl.unwrap_or(self.config.default_invitation_ttl))
            .with_base_url(&self.config.invitation_base_url)
            .build(&self.encryption_key())?;

        let invitation_id = invitation.invitation_id;

        // Store the invitation
        self.invitations.insert(invitation_id, invitation.clone());
        self.invitations_by_sponsor
            .entry(sponsor_id)
            .or_default()
            .push(invitation_id);

        Ok(invitation)
    }

    /// Gets an invitation by ID.
    #[must_use]
    pub fn get_invitation(&self, invitation_id: Uuid) -> Option<InvitationLink> {
        self.invitations.get(&invitation_id).map(|r| r.clone())
    }

    /// Revokes an invitation.
    ///
    /// # Errors
    ///
    /// Returns an error if the invitation doesn't exist or is not pending.
    pub fn revoke_invitation(&self, invitation_id: Uuid) -> Result<()> {
        let mut invitation = self
            .invitations
            .get_mut(&invitation_id)
            .ok_or(EphemeralError::InvitationNotFound(invitation_id))?;

        invitation.revoke()?;
        Ok(())
    }

    /// Redeems an invitation to create an ephemeral identity.
    ///
    /// # Errors
    ///
    /// Returns an error if redemption fails.
    pub async fn redeem_invitation(
        &self,
        token: &str,
        redemption_code: &str,
        device_binding: DeviceBinding,
        metadata: Option<IdentityMetadata>,
    ) -> Result<(EphemeralIdentity, EphemeralToken)> {
        // Validate and redeem
        let redeemer = InvitationRedeemer::new(self.encryption_key());

        // Find the invitation to get attempt count
        let payload = redeemer.validate_token(token)?;
        let invitation_id = payload.invitation_id;

        let attempts = {
            let invitation = self
                .invitations
                .get(&invitation_id)
                .ok_or(EphemeralError::InvitationNotFound(invitation_id))?;

            if !invitation.can_redeem() {
                if invitation.is_expired() {
                    return Err(EphemeralError::InvitationExpired(invitation_id));
                }
                if invitation.status == InvitationStatus::Redeemed {
                    return Err(EphemeralError::InvitationAlreadyRedeemed(invitation_id));
                }
                return Err(EphemeralError::InvalidState {
                    expected: "Pending".to_string(),
                    found: format!("{:?}", invitation.status),
                });
            }

            invitation.attempts
        };

        // Increment attempt counter
        {
            if let Some(mut invitation) = self.invitations.get_mut(&invitation_id) {
                invitation.attempts += 1;
            }
        }

        // Attempt redemption
        let result = redeemer.redeem(token, redemption_code, attempts)?;

        // Check sponsor identity limits
        let sponsor_identities = self
            .identities_by_sponsor
            .get(&result.sponsor_id)
            .map(|v| v.len())
            .unwrap_or(0);

        if sponsor_identities >= self.config.max_identities_per_sponsor {
            return Err(EphemeralError::MaxIdentitiesExceeded(
                self.config.max_identities_per_sponsor,
            ));
        }

        // Create the identity
        let mut identity = EphemeralIdentity::new(
            result.sponsor_id,
            result.tenant_id,
            result.capabilities.clone(),
            result.identity_ttl,
        );

        if let Some(meta) = metadata {
            identity.metadata = meta;
        }

        // Activate the identity
        identity.activate(device_binding)?;

        // Create access token
        let token_claims = TokenClaims::new(
            identity.id,
            identity.sponsor_id,
            identity.tenant_id,
            result.capabilities,
            &self.config.token_issuer,
            result.identity_ttl,
        );

        let access_token = if self.config.encrypt_tokens {
            EphemeralToken::create_encrypted(token_claims, &self.keypair, self.encryption_key())?
        } else {
            EphemeralToken::create(token_claims, &self.keypair)?
        };

        // Store the identity
        let identity_id = identity.id;
        self.identities.insert(identity_id, identity.clone());
        self.identities_by_sponsor
            .entry(result.sponsor_id)
            .or_default()
            .push(identity_id);

        // Index the token
        let token_hash = crate::token::token_hash(access_token.as_str());
        self.token_index.insert(token_hash, identity_id);

        // Mark invitation as redeemed
        if let Some(mut invitation) = self.invitations.get_mut(&invitation_id) {
            invitation.status = InvitationStatus::Redeemed;
            invitation.redeemed_at = Some(Utc::now());
            invitation.identity_id = Some(identity_id);
        }

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.total_identities_created += 1;
            stats.active_identities = self.identities.len();
            stats.tokens_issued += 1;
        }

        Ok((identity, access_token))
    }

    // === Identity Management ===

    /// Gets an identity by ID.
    #[must_use]
    pub fn get_identity(&self, identity_id: Uuid) -> Option<EphemeralIdentity> {
        self.identities.get(&identity_id).map(|r| r.clone())
    }

    /// Gets all identities for a sponsor.
    #[must_use]
    pub fn get_sponsor_identities(&self, sponsor_id: Uuid) -> Vec<EphemeralIdentity> {
        self.identities_by_sponsor
            .get(&sponsor_id)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| self.identities.get(id).map(|r| r.clone()))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Suspends an identity.
    ///
    /// # Errors
    ///
    /// Returns an error if the identity doesn't exist or cannot be suspended.
    pub fn suspend_identity(&self, identity_id: Uuid, reason: impl Into<String>) -> Result<()> {
        let mut identity = self
            .identities
            .get_mut(&identity_id)
            .ok_or(EphemeralError::IdentityNotFound(identity_id))?;

        identity.suspend(reason)?;
        Ok(())
    }

    /// Reactivates a suspended identity.
    ///
    /// # Errors
    ///
    /// Returns an error if the identity doesn't exist or cannot be reactivated.
    pub fn reactivate_identity(&self, identity_id: Uuid) -> Result<()> {
        let mut identity = self
            .identities
            .get_mut(&identity_id)
            .ok_or(EphemeralError::IdentityNotFound(identity_id))?;

        identity.reactivate()?;
        Ok(())
    }

    /// Revokes an identity.
    ///
    /// # Errors
    ///
    /// Returns an error if the identity doesn't exist or cannot be revoked.
    pub fn revoke_identity(
        &self,
        identity_id: Uuid,
        revoked_by: Uuid,
        reason: impl Into<String>,
    ) -> Result<()> {
        let mut identity = self
            .identities
            .get_mut(&identity_id)
            .ok_or(EphemeralError::IdentityNotFound(identity_id))?;

        identity.revoke(revoked_by, reason)?;
        Ok(())
    }

    // === Token Operations ===

    /// Creates a new token for an existing identity.
    ///
    /// # Errors
    ///
    /// Returns an error if the identity doesn't exist or is not active.
    pub async fn create_token(
        &self,
        identity_id: Uuid,
        ttl: Option<Duration>,
    ) -> Result<EphemeralToken> {
        let identity = self
            .identities
            .get(&identity_id)
            .ok_or(EphemeralError::IdentityNotFound(identity_id))?;

        if identity.state != EphemeralIdentityState::Active {
            return Err(EphemeralError::InvalidState {
                expected: "Active".to_string(),
                found: format!("{:?}", identity.state),
            });
        }

        // Token TTL cannot exceed identity TTL
        let max_ttl = identity.time_remaining();
        let token_ttl = ttl.map(|t| t.min(max_ttl)).unwrap_or(max_ttl);

        let claims = TokenClaims::new(
            identity.id,
            identity.sponsor_id,
            identity.tenant_id,
            identity.capabilities.clone(),
            &self.config.token_issuer,
            token_ttl,
        );

        let token = if self.config.encrypt_tokens {
            EphemeralToken::create_encrypted(claims, &self.keypair, self.encryption_key())?
        } else {
            EphemeralToken::create(claims, &self.keypair)?
        };

        // Index the token
        let token_hash = crate::token::token_hash(token.as_str());
        self.token_index.insert(token_hash, identity_id);

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.tokens_issued += 1;
        }

        Ok(token)
    }

    /// Validates a token.
    ///
    /// # Errors
    ///
    /// Returns an error if the token is invalid.
    pub async fn validate_token(&self, token: &str) -> Result<TokenValidation> {
        let mut validator = TokenValidator::new(self.keypair.public_key())
            .with_expected_issuer(&self.config.token_issuer);

        if self.config.encrypt_tokens {
            validator = validator.with_encryption_key(self.encryption_key());
        }

        let validation = validator.validate(token)?;

        // Verify identity still exists and is active
        let identity = self
            .identities
            .get(&validation.claims.sub)
            .ok_or(EphemeralError::IdentityNotFound(validation.claims.sub))?;

        if identity.state != EphemeralIdentityState::Active {
            return Err(EphemeralError::InvalidState {
                expected: "Active".to_string(),
                found: format!("{:?}", identity.state),
            });
        }

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.tokens_validated += 1;
        }

        Ok(validation)
    }

    // === Cleanup ===

    /// Runs cleanup to expire old identities and invitations.
    pub async fn cleanup(&self) -> (usize, usize) {
        let mut expired_identities = 0;
        let mut expired_invitations = 0;

        // Expire identities
        for mut entry in self.identities.iter_mut() {
            if entry.check_expiry() {
                expired_identities += 1;
            }
        }

        // Expire invitations
        for mut entry in self.invitations.iter_mut() {
            if entry.is_expired() && entry.status == InvitationStatus::Pending {
                entry.mark_expired();
                expired_invitations += 1;
            }
        }

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.identities_expired += expired_identities as u64;
            stats.invitations_expired += expired_invitations as u64;
            stats.active_identities = self
                .identities
                .iter()
                .filter(|i| i.state == EphemeralIdentityState::Active)
                .count();
            stats.pending_invitations = self
                .invitations
                .iter()
                .filter(|i| i.status == InvitationStatus::Pending)
                .count();
        }

        (expired_identities, expired_invitations)
    }

    /// Removes expired entities from storage.
    pub fn purge_expired(&self) {
        // Remove expired identities
        self.identities.retain(|_, identity| {
            !matches!(
                identity.state,
                EphemeralIdentityState::Expired | EphemeralIdentityState::Revoked
            )
        });

        // Remove expired/redeemed invitations
        self.invitations
            .retain(|_, invitation| matches!(invitation.status, InvitationStatus::Pending));

        // Clean up sponsor mappings
        for mut entry in self.identities_by_sponsor.iter_mut() {
            entry.retain(|id| self.identities.contains_key(id));
        }

        for mut entry in self.invitations_by_sponsor.iter_mut() {
            entry.retain(|id| self.invitations.contains_key(id));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capabilities::Capability;

    fn create_test_service() -> EphemeralIdentityService {
        EphemeralIdentityService::new(ServiceConfig::default())
    }

    fn create_test_capabilities() -> CapabilitySet {
        CapabilitySet::new()
            .with_capability(Capability::new("read", "notebooks"))
            .with_capability(Capability::new("write", "notebooks"))
    }

    fn create_test_device_binding() -> DeviceBinding {
        DeviceBinding::new(
            "fp-test-123".to_string(),
            "192.168.1.100".to_string(),
            "TestAgent/1.0".to_string(),
        )
    }

    // === Service Creation Tests ===

    #[test]
    fn test_service_new() {
        let service = create_test_service();
        assert_eq!(service.config().token_issuer, "hpc-ephemeral");
    }

    #[tokio::test]
    async fn test_service_stats_initial() {
        let service = create_test_service();
        let stats = service.stats().await;

        assert_eq!(stats.total_identities_created, 0);
        assert_eq!(stats.active_identities, 0);
        assert_eq!(stats.pending_invitations, 0);
    }

    // === Invitation Tests ===

    #[tokio::test]
    async fn test_create_invitation() {
        let service = create_test_service();
        let sponsor_id = Uuid::new_v4();
        let tenant_id = Uuid::new_v4();

        let invitation = service
            .create_invitation(
                sponsor_id,
                tenant_id,
                create_test_capabilities(),
                None,
                None,
            )
            .await
            .unwrap();

        assert_eq!(invitation.sponsor_id, sponsor_id);
        assert_eq!(invitation.tenant_id, tenant_id);
        assert_eq!(invitation.status, InvitationStatus::Pending);
    }

    #[tokio::test]
    async fn test_create_invitation_max_limit() {
        let mut config = ServiceConfig::default();
        config.max_invitations_per_sponsor = 2;
        let service = EphemeralIdentityService::new(config);

        let sponsor_id = Uuid::new_v4();
        let tenant_id = Uuid::new_v4();

        // Create max invitations
        for _ in 0..2 {
            service
                .create_invitation(
                    sponsor_id,
                    tenant_id,
                    create_test_capabilities(),
                    None,
                    None,
                )
                .await
                .unwrap();
        }

        // Third should fail
        let result = service
            .create_invitation(
                sponsor_id,
                tenant_id,
                create_test_capabilities(),
                None,
                None,
            )
            .await;

        assert!(matches!(
            result,
            Err(EphemeralError::MaxIdentitiesExceeded(_))
        ));
    }

    #[tokio::test]
    async fn test_get_invitation() {
        let service = create_test_service();
        let sponsor_id = Uuid::new_v4();
        let tenant_id = Uuid::new_v4();

        let invitation = service
            .create_invitation(
                sponsor_id,
                tenant_id,
                create_test_capabilities(),
                None,
                None,
            )
            .await
            .unwrap();

        let retrieved = service.get_invitation(invitation.invitation_id);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().invitation_id, invitation.invitation_id);
    }

    #[tokio::test]
    async fn test_revoke_invitation() {
        let service = create_test_service();
        let sponsor_id = Uuid::new_v4();
        let tenant_id = Uuid::new_v4();

        let invitation = service
            .create_invitation(
                sponsor_id,
                tenant_id,
                create_test_capabilities(),
                None,
                None,
            )
            .await
            .unwrap();

        let result = service.revoke_invitation(invitation.invitation_id);
        assert!(result.is_ok());

        let retrieved = service.get_invitation(invitation.invitation_id).unwrap();
        assert_eq!(retrieved.status, InvitationStatus::Revoked);
    }

    // === Redemption Tests ===

    #[tokio::test]
    async fn test_redeem_invitation() {
        let service = create_test_service();
        let sponsor_id = Uuid::new_v4();
        let tenant_id = Uuid::new_v4();

        let invitation = service
            .create_invitation(
                sponsor_id,
                tenant_id,
                create_test_capabilities(),
                None,
                None,
            )
            .await
            .unwrap();

        let (identity, token) = service
            .redeem_invitation(
                &invitation.token,
                &invitation.redemption_code,
                create_test_device_binding(),
                None,
            )
            .await
            .unwrap();

        assert_eq!(identity.sponsor_id, sponsor_id);
        assert_eq!(identity.state, EphemeralIdentityState::Active);
        assert!(!token.as_str().is_empty());
    }

    #[tokio::test]
    async fn test_redeem_invitation_wrong_code() {
        let service = create_test_service();
        let sponsor_id = Uuid::new_v4();
        let tenant_id = Uuid::new_v4();

        let invitation = service
            .create_invitation(
                sponsor_id,
                tenant_id,
                create_test_capabilities(),
                None,
                None,
            )
            .await
            .unwrap();

        let result = service
            .redeem_invitation(
                &invitation.token,
                "WRONG1",
                create_test_device_binding(),
                None,
            )
            .await;

        assert!(matches!(result, Err(EphemeralError::InvalidRedemptionCode)));
    }

    #[tokio::test]
    async fn test_redeem_invitation_already_redeemed() {
        let service = create_test_service();
        let sponsor_id = Uuid::new_v4();
        let tenant_id = Uuid::new_v4();

        let invitation = service
            .create_invitation(
                sponsor_id,
                tenant_id,
                create_test_capabilities(),
                None,
                None,
            )
            .await
            .unwrap();

        // First redemption
        service
            .redeem_invitation(
                &invitation.token,
                &invitation.redemption_code,
                create_test_device_binding(),
                None,
            )
            .await
            .unwrap();

        // Second redemption should fail
        let result = service
            .redeem_invitation(
                &invitation.token,
                &invitation.redemption_code,
                create_test_device_binding(),
                None,
            )
            .await;

        assert!(matches!(
            result,
            Err(EphemeralError::InvitationAlreadyRedeemed(_))
        ));
    }

    // === Identity Management Tests ===

    #[tokio::test]
    async fn test_get_identity() {
        let service = create_test_service();
        let sponsor_id = Uuid::new_v4();
        let tenant_id = Uuid::new_v4();

        let invitation = service
            .create_invitation(
                sponsor_id,
                tenant_id,
                create_test_capabilities(),
                None,
                None,
            )
            .await
            .unwrap();

        let (identity, _) = service
            .redeem_invitation(
                &invitation.token,
                &invitation.redemption_code,
                create_test_device_binding(),
                None,
            )
            .await
            .unwrap();

        let retrieved = service.get_identity(identity.id);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().id, identity.id);
    }

    #[tokio::test]
    async fn test_suspend_identity() {
        let service = create_test_service();
        let sponsor_id = Uuid::new_v4();
        let tenant_id = Uuid::new_v4();

        let invitation = service
            .create_invitation(
                sponsor_id,
                tenant_id,
                create_test_capabilities(),
                None,
                None,
            )
            .await
            .unwrap();

        let (identity, _) = service
            .redeem_invitation(
                &invitation.token,
                &invitation.redemption_code,
                create_test_device_binding(),
                None,
            )
            .await
            .unwrap();

        service
            .suspend_identity(identity.id, "Suspicious activity")
            .unwrap();

        let retrieved = service.get_identity(identity.id).unwrap();
        assert_eq!(retrieved.state, EphemeralIdentityState::Suspended);
    }

    #[tokio::test]
    async fn test_reactivate_identity() {
        let service = create_test_service();
        let sponsor_id = Uuid::new_v4();
        let tenant_id = Uuid::new_v4();

        let invitation = service
            .create_invitation(
                sponsor_id,
                tenant_id,
                create_test_capabilities(),
                None,
                None,
            )
            .await
            .unwrap();

        let (identity, _) = service
            .redeem_invitation(
                &invitation.token,
                &invitation.redemption_code,
                create_test_device_binding(),
                None,
            )
            .await
            .unwrap();

        service.suspend_identity(identity.id, "Review").unwrap();
        service.reactivate_identity(identity.id).unwrap();

        let retrieved = service.get_identity(identity.id).unwrap();
        assert_eq!(retrieved.state, EphemeralIdentityState::Active);
    }

    #[tokio::test]
    async fn test_revoke_identity() {
        let service = create_test_service();
        let sponsor_id = Uuid::new_v4();
        let tenant_id = Uuid::new_v4();
        let admin_id = Uuid::new_v4();

        let invitation = service
            .create_invitation(
                sponsor_id,
                tenant_id,
                create_test_capabilities(),
                None,
                None,
            )
            .await
            .unwrap();

        let (identity, _) = service
            .redeem_invitation(
                &invitation.token,
                &invitation.redemption_code,
                create_test_device_binding(),
                None,
            )
            .await
            .unwrap();

        service
            .revoke_identity(identity.id, admin_id, "Security concern")
            .unwrap();

        let retrieved = service.get_identity(identity.id).unwrap();
        assert_eq!(retrieved.state, EphemeralIdentityState::Revoked);
    }

    // === Token Tests ===

    #[tokio::test]
    async fn test_create_token() {
        let service = create_test_service();
        let sponsor_id = Uuid::new_v4();
        let tenant_id = Uuid::new_v4();

        let invitation = service
            .create_invitation(
                sponsor_id,
                tenant_id,
                create_test_capabilities(),
                None,
                None,
            )
            .await
            .unwrap();

        let (identity, _) = service
            .redeem_invitation(
                &invitation.token,
                &invitation.redemption_code,
                create_test_device_binding(),
                None,
            )
            .await
            .unwrap();

        let token = service.create_token(identity.id, None).await.unwrap();
        assert!(!token.as_str().is_empty());
    }

    #[tokio::test]
    async fn test_validate_token() {
        let service = create_test_service();
        let sponsor_id = Uuid::new_v4();
        let tenant_id = Uuid::new_v4();

        let invitation = service
            .create_invitation(
                sponsor_id,
                tenant_id,
                create_test_capabilities(),
                None,
                None,
            )
            .await
            .unwrap();

        let (identity, token) = service
            .redeem_invitation(
                &invitation.token,
                &invitation.redemption_code,
                create_test_device_binding(),
                None,
            )
            .await
            .unwrap();

        let validation = service.validate_token(token.as_str()).await.unwrap();
        assert!(validation.is_valid);
        assert_eq!(validation.claims.sub, identity.id);
    }

    // === Cleanup Tests ===

    #[tokio::test]
    async fn test_cleanup() {
        let mut config = ServiceConfig::default();
        config.default_invitation_ttl = Duration::milliseconds(-1); // Already expired
        let service = EphemeralIdentityService::new(config);

        let sponsor_id = Uuid::new_v4();
        let tenant_id = Uuid::new_v4();

        // Create an already-expired invitation
        service
            .create_invitation(
                sponsor_id,
                tenant_id,
                create_test_capabilities(),
                None,
                None,
            )
            .await
            .unwrap();

        let (_, expired_invitations) = service.cleanup().await;
        assert_eq!(expired_invitations, 1);
    }

    #[tokio::test]
    async fn test_purge_expired() {
        let service = create_test_service();
        let sponsor_id = Uuid::new_v4();
        let tenant_id = Uuid::new_v4();

        let invitation = service
            .create_invitation(
                sponsor_id,
                tenant_id,
                create_test_capabilities(),
                None,
                None,
            )
            .await
            .unwrap();

        // Revoke the invitation
        service.revoke_invitation(invitation.invitation_id).unwrap();

        // Purge should remove revoked invitations
        service.purge_expired();

        let retrieved = service.get_invitation(invitation.invitation_id);
        assert!(retrieved.is_none());
    }

    // === Stats Tests ===

    #[tokio::test]
    async fn test_stats_after_operations() {
        let service = create_test_service();
        let sponsor_id = Uuid::new_v4();
        let tenant_id = Uuid::new_v4();

        let invitation = service
            .create_invitation(
                sponsor_id,
                tenant_id,
                create_test_capabilities(),
                None,
                None,
            )
            .await
            .unwrap();

        let (_identity, token) = service
            .redeem_invitation(
                &invitation.token,
                &invitation.redemption_code,
                create_test_device_binding(),
                None,
            )
            .await
            .unwrap();

        service.validate_token(token.as_str()).await.unwrap();

        let stats = service.stats().await;
        assert_eq!(stats.total_identities_created, 1);
        assert_eq!(stats.tokens_issued, 1);
        assert_eq!(stats.tokens_validated, 1);
    }
}
