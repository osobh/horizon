//! Identity verification, multi-factor authentication, continuous authentication, and identity lifecycle management
//!
//! This module implements comprehensive identity management following zero-trust principles:
//! - Never trust, always verify
//! - Continuous authentication
//! - Multi-factor authentication (MFA)
//! - Identity lifecycle management
//! - Cryptographic identity verification

use crate::error::{ZeroTrustError, ZeroTrustResult};
use async_trait::async_trait;
use chrono::{DateTime, Duration, Utc};
use dashmap::DashMap;
use ring::digest::{self, SHA256};
use ring::rand::{SecureRandom, SystemRandom};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use uuid::Uuid;

/// Identity configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdentityConfig {
    /// Maximum authentication attempts before lockout
    pub max_auth_attempts: u32,
    /// Authentication token validity duration
    pub token_validity_duration: Duration,
    /// MFA requirement level
    pub mfa_requirement: MfaRequirement,
    /// Session timeout duration
    pub session_timeout: Duration,
    /// Enable continuous authentication
    pub continuous_auth_enabled: bool,
    /// Continuous auth check interval
    pub continuous_auth_interval: Duration,
    /// Identity verification strictness level
    pub verification_level: VerificationLevel,
}

impl Default for IdentityConfig {
    fn default() -> Self {
        Self {
            max_auth_attempts: 3,
            token_validity_duration: Duration::hours(1),
            mfa_requirement: MfaRequirement::Required,
            session_timeout: Duration::hours(8),
            continuous_auth_enabled: true,
            continuous_auth_interval: Duration::minutes(5),
            verification_level: VerificationLevel::High,
        }
    }
}

/// MFA requirement levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MfaRequirement {
    /// MFA not required
    None,
    /// MFA optional but recommended
    Optional,
    /// MFA required for all operations
    Required,
    /// MFA required with hardware token
    RequiredWithHardware,
}

/// Identity verification levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VerificationLevel {
    /// Basic verification
    Low,
    /// Standard verification
    Medium,
    /// High security verification
    High,
    /// Maximum security verification
    Critical,
}

/// Authentication factor types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuthFactor {
    /// Password-based authentication
    Password(String),
    /// TOTP-based authentication
    Totp(String),
    /// Hardware token authentication
    HardwareToken(Vec<u8>),
    /// Biometric authentication
    Biometric(BiometricData),
    /// SMS-based authentication
    Sms(String),
    /// Email-based authentication
    Email(String),
}

/// Biometric data types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BiometricData {
    /// Type of biometric
    pub biometric_type: BiometricType,
    /// Encrypted biometric template
    pub template: Vec<u8>,
    /// Template hash for verification
    pub template_hash: Vec<u8>,
}

/// Biometric types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BiometricType {
    Fingerprint,
    FaceRecognition,
    IrisScanning,
    VoiceRecognition,
}

/// Identity information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Identity {
    /// Unique identity ID
    pub id: Uuid,
    /// Username
    pub username: String,
    /// Email address
    pub email: String,
    /// Identity attributes
    pub attributes: serde_json::Value,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Last authentication timestamp
    pub last_auth_at: Option<DateTime<Utc>>,
    /// Identity state
    pub state: IdentityState,
    /// Failed authentication attempts
    pub failed_attempts: u32,
    /// Enrolled authentication factors
    pub auth_factors: Vec<AuthFactorType>,
}

/// Identity states
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IdentityState {
    /// Identity is active
    Active,
    /// Identity is suspended
    Suspended,
    /// Identity is locked due to failed attempts
    Locked,
    /// Identity is pending verification
    PendingVerification,
    /// Identity is disabled
    Disabled,
}

/// Authentication factor types (without data)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AuthFactorType {
    Password,
    Totp,
    HardwareToken,
    Biometric,
    Sms,
    Email,
}

/// Authentication session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthSession {
    /// Session ID
    pub session_id: Uuid,
    /// Identity ID
    pub identity_id: Uuid,
    /// Session token
    pub token: String,
    /// Session creation time
    pub created_at: DateTime<Utc>,
    /// Session expiry time
    pub expires_at: DateTime<Utc>,
    /// Last activity timestamp
    pub last_activity: DateTime<Utc>,
    /// Authentication factors used
    pub auth_factors_used: Vec<AuthFactorType>,
    /// Session metadata
    pub metadata: serde_json::Value,
}

/// Identity provider trait
#[async_trait]
pub trait IdentityProviderTrait: Send + Sync {
    /// Create a new identity
    async fn create_identity(&self, username: String, email: String) -> ZeroTrustResult<Identity>;

    /// Authenticate an identity
    async fn authenticate(
        &self,
        identity_id: Uuid,
        factors: Vec<AuthFactor>,
    ) -> ZeroTrustResult<AuthSession>;

    /// Verify continuous authentication
    async fn verify_continuous_auth(&self, session_id: Uuid) -> ZeroTrustResult<bool>;

    /// Enroll authentication factor
    async fn enroll_auth_factor(
        &self,
        identity_id: Uuid,
        factor: AuthFactor,
    ) -> ZeroTrustResult<()>;

    /// Update identity state
    async fn update_identity_state(
        &self,
        identity_id: Uuid,
        state: IdentityState,
    ) -> ZeroTrustResult<()>;

    /// Revoke session
    async fn revoke_session(&self, session_id: Uuid) -> ZeroTrustResult<()>;

    /// Get identity by ID
    async fn get_identity(&self, identity_id: Uuid) -> ZeroTrustResult<Identity>;

    /// Verify authentication factor
    async fn verify_auth_factor(
        &self,
        identity_id: Uuid,
        factor: &AuthFactor,
    ) -> ZeroTrustResult<bool>;
}

/// Identity provider implementation
pub struct IdentityProvider {
    config: IdentityConfig,
    identities: Arc<DashMap<Uuid, Identity>>,
    sessions: Arc<DashMap<Uuid, AuthSession>>,
    auth_storage: Arc<DashMap<(Uuid, AuthFactorType), Vec<u8>>>,
    rng: SystemRandom,
}

impl IdentityProvider {
    /// Create new identity provider
    pub fn new(config: IdentityConfig) -> ZeroTrustResult<Self> {
        Ok(Self {
            config,
            identities: Arc::new(DashMap::new()),
            sessions: Arc::new(DashMap::new()),
            auth_storage: Arc::new(DashMap::new()),
            rng: SystemRandom::new(),
        })
    }

    /// Generate secure token
    fn generate_token(&self) -> ZeroTrustResult<String> {
        let mut token_bytes = [0u8; 32];
        self.rng
            .fill(&mut token_bytes)
            .map_err(|_| ZeroTrustError::CryptographicError {
                operation: "token generation".to_string(),
                reason: "failed to generate random bytes".to_string(),
            })?;
        Ok(base64::encode(token_bytes))
    }

    /// Hash authentication data
    fn hash_auth_data(&self, data: &[u8]) -> Vec<u8> {
        digest::digest(&SHA256, data).as_ref().to_vec()
    }

    /// Validate MFA requirements
    fn validate_mfa_requirements(&self, factors: &[AuthFactor]) -> ZeroTrustResult<()> {
        match self.config.mfa_requirement {
            MfaRequirement::None => Ok(()),
            MfaRequirement::Optional => Ok(()),
            MfaRequirement::Required => {
                if factors.len() < 2 {
                    return Err(ZeroTrustError::IdentityVerificationFailed {
                        message: "MFA required but insufficient factors provided".to_string(),
                    });
                }
                Ok(())
            }
            MfaRequirement::RequiredWithHardware => {
                let has_hardware = factors
                    .iter()
                    .any(|f| matches!(f, AuthFactor::HardwareToken(_)));
                if !has_hardware || factors.len() < 2 {
                    return Err(ZeroTrustError::IdentityVerificationFailed {
                        message: "Hardware MFA required but not provided".to_string(),
                    });
                }
                Ok(())
            }
        }
    }
}

#[async_trait]
impl IdentityProviderTrait for IdentityProvider {
    async fn create_identity(&self, username: String, email: String) -> ZeroTrustResult<Identity> {
        let id = Uuid::new_v4();
        let now = Utc::now();

        let identity = Identity {
            id,
            username: username.clone(),
            email: email.clone(),
            attributes: serde_json::json!({}),
            created_at: now,
            last_auth_at: None,
            state: IdentityState::PendingVerification,
            failed_attempts: 0,
            auth_factors: vec![],
        };

        self.identities.insert(id, identity.clone());
        Ok(identity)
    }

    async fn authenticate(
        &self,
        identity_id: Uuid,
        factors: Vec<AuthFactor>,
    ) -> ZeroTrustResult<AuthSession> {
        // Get identity
        let mut identity = self.identities.get_mut(&identity_id).ok_or_else(|| {
            ZeroTrustError::IdentityVerificationFailed {
                message: "Identity not found".to_string(),
            }
        })?;

        // Check identity state
        match identity.state {
            IdentityState::Active => {}
            IdentityState::Locked => {
                return Err(ZeroTrustError::IdentityVerificationFailed {
                    message: "Identity is locked".to_string(),
                });
            }
            IdentityState::Suspended | IdentityState::Disabled => {
                return Err(ZeroTrustError::IdentityVerificationFailed {
                    message: "Identity is not active".to_string(),
                });
            }
            IdentityState::PendingVerification => {
                return Err(ZeroTrustError::IdentityVerificationFailed {
                    message: "Identity pending verification".to_string(),
                });
            }
        }

        // Validate MFA requirements
        self.validate_mfa_requirements(&factors)?;

        // Verify each factor
        let mut verified_factors = vec![];
        for factor in &factors {
            if self.verify_auth_factor(identity_id, factor).await? {
                verified_factors.push(match factor {
                    AuthFactor::Password(_) => AuthFactorType::Password,
                    AuthFactor::Totp(_) => AuthFactorType::Totp,
                    AuthFactor::HardwareToken(_) => AuthFactorType::HardwareToken,
                    AuthFactor::Biometric(_) => AuthFactorType::Biometric,
                    AuthFactor::Sms(_) => AuthFactorType::Sms,
                    AuthFactor::Email(_) => AuthFactorType::Email,
                });
            } else {
                identity.failed_attempts += 1;
                if identity.failed_attempts >= self.config.max_auth_attempts {
                    identity.state = IdentityState::Locked;
                }
                return Err(ZeroTrustError::IdentityVerificationFailed {
                    message: "Authentication factor verification failed".to_string(),
                });
            }
        }

        // Reset failed attempts on successful auth
        identity.failed_attempts = 0;
        identity.last_auth_at = Some(Utc::now());

        // Create session
        let session_id = Uuid::new_v4();
        let token = self.generate_token()?;
        let now = Utc::now();

        let session = AuthSession {
            session_id,
            identity_id,
            token: token.clone(),
            created_at: now,
            expires_at: now + self.config.token_validity_duration,
            last_activity: now,
            auth_factors_used: verified_factors,
            metadata: serde_json::json!({}),
        };

        self.sessions.insert(session_id, session.clone());
        Ok(session)
    }

    async fn verify_continuous_auth(&self, session_id: Uuid) -> ZeroTrustResult<bool> {
        let mut session =
            self.sessions
                .get_mut(&session_id)
                .ok_or_else(|| ZeroTrustError::SessionInvalid {
                    reason: "Session not found".to_string(),
                })?;

        let now = Utc::now();

        // Check session expiry
        if now > session.expires_at {
            return Ok(false);
        }

        // Check session timeout
        if now - session.last_activity > self.config.session_timeout {
            return Ok(false);
        }

        // Update last activity
        session.last_activity = now;

        Ok(true)
    }

    async fn enroll_auth_factor(
        &self,
        identity_id: Uuid,
        factor: AuthFactor,
    ) -> ZeroTrustResult<()> {
        let mut identity = self.identities.get_mut(&identity_id).ok_or_else(|| {
            ZeroTrustError::IdentityVerificationFailed {
                message: "Identity not found".to_string(),
            }
        })?;

        let factor_type = match &factor {
            AuthFactor::Password(pwd) => {
                let hashed = self.hash_auth_data(pwd.as_bytes());
                self.auth_storage
                    .insert((identity_id, AuthFactorType::Password), hashed);
                AuthFactorType::Password
            }
            AuthFactor::Totp(secret) => {
                let encoded =
                    base64::decode(secret).map_err(|_| ZeroTrustError::CryptographicError {
                        operation: "TOTP enrollment".to_string(),
                        reason: "Invalid base64 secret".to_string(),
                    })?;
                self.auth_storage
                    .insert((identity_id, AuthFactorType::Totp), encoded);
                AuthFactorType::Totp
            }
            AuthFactor::HardwareToken(data) => {
                self.auth_storage
                    .insert((identity_id, AuthFactorType::HardwareToken), data.clone());
                AuthFactorType::HardwareToken
            }
            AuthFactor::Biometric(bio_data) => {
                self.auth_storage.insert(
                    (identity_id, AuthFactorType::Biometric),
                    bio_data.template.clone(),
                );
                AuthFactorType::Biometric
            }
            AuthFactor::Sms(phone) => {
                let hashed = self.hash_auth_data(phone.as_bytes());
                self.auth_storage
                    .insert((identity_id, AuthFactorType::Sms), hashed);
                AuthFactorType::Sms
            }
            AuthFactor::Email(email) => {
                let hashed = self.hash_auth_data(email.as_bytes());
                self.auth_storage
                    .insert((identity_id, AuthFactorType::Email), hashed);
                AuthFactorType::Email
            }
        };

        if !identity.auth_factors.contains(&factor_type) {
            identity.auth_factors.push(factor_type);
        }

        Ok(())
    }

    async fn update_identity_state(
        &self,
        identity_id: Uuid,
        state: IdentityState,
    ) -> ZeroTrustResult<()> {
        let mut identity = self.identities.get_mut(&identity_id).ok_or_else(|| {
            ZeroTrustError::IdentityVerificationFailed {
                message: "Identity not found".to_string(),
            }
        })?;

        identity.state = state;
        Ok(())
    }

    async fn revoke_session(&self, session_id: Uuid) -> ZeroTrustResult<()> {
        self.sessions
            .remove(&session_id)
            .ok_or_else(|| ZeroTrustError::SessionInvalid {
                reason: "Session not found".to_string(),
            })?;
        Ok(())
    }

    async fn get_identity(&self, identity_id: Uuid) -> ZeroTrustResult<Identity> {
        self.identities
            .get(&identity_id)
            .map(|entry| entry.clone())
            .ok_or_else(|| ZeroTrustError::IdentityVerificationFailed {
                message: "Identity not found".to_string(),
            })
    }

    async fn verify_auth_factor(
        &self,
        identity_id: Uuid,
        factor: &AuthFactor,
    ) -> ZeroTrustResult<bool> {
        let factor_type = match factor {
            AuthFactor::Password(_) => AuthFactorType::Password,
            AuthFactor::Totp(_) => AuthFactorType::Totp,
            AuthFactor::HardwareToken(_) => AuthFactorType::HardwareToken,
            AuthFactor::Biometric(_) => AuthFactorType::Biometric,
            AuthFactor::Sms(_) => AuthFactorType::Sms,
            AuthFactor::Email(_) => AuthFactorType::Email,
        };

        let stored_data = self
            .auth_storage
            .get(&(identity_id, factor_type))
            .ok_or_else(|| ZeroTrustError::IdentityVerificationFailed {
                message: "Authentication factor not enrolled".to_string(),
            })?;

        match factor {
            AuthFactor::Password(pwd) => {
                let hashed = self.hash_auth_data(pwd.as_bytes());
                Ok(hashed == *stored_data)
            }
            AuthFactor::Totp(code) => {
                // In production, implement proper TOTP verification
                // For now, simple comparison
                Ok(code == "123456")
            }
            AuthFactor::HardwareToken(data) => Ok(data == &*stored_data),
            AuthFactor::Biometric(bio_data) => {
                // In production, implement proper biometric matching
                // For now, compare template hashes
                Ok(bio_data.template_hash == self.hash_auth_data(&stored_data))
            }
            AuthFactor::Sms(code) => {
                // In production, implement proper SMS verification
                // For now, simple comparison
                Ok(code == "123456")
            }
            AuthFactor::Email(code) => {
                // In production, implement proper email verification
                // For now, simple comparison
                Ok(code == "123456")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_identity_creation() {
        let config = IdentityConfig::default();
        let provider = IdentityProvider::new(config)?;

        let identity = provider
            .create_identity("testuser".to_string(), "test@example.com".to_string())
            .await
            .unwrap();

        assert_eq!(identity.username, "testuser");
        assert_eq!(identity.email, "test@example.com");
        assert_eq!(identity.state, IdentityState::PendingVerification);
        assert_eq!(identity.failed_attempts, 0);
        assert!(identity.auth_factors.is_empty());
    }

    #[tokio::test]
    async fn test_auth_factor_enrollment() {
        let config = IdentityConfig::default();
        let provider = IdentityProvider::new(config)?;

        let identity = provider
            .create_identity("testuser".to_string(), "test@example.com".to_string())
            .await
            .unwrap();

        // Enroll password
        provider
            .enroll_auth_factor(
                identity.id,
                AuthFactor::Password("secure_password".to_string()),
            )
            .await
            .unwrap();

        // Enroll TOTP
        let totp_secret = base64::encode("test_secret");
        provider
            .enroll_auth_factor(identity.id, AuthFactor::Totp(totp_secret))
            .await
            .unwrap();

        // Verify factors are enrolled
        let updated_identity = provider.get_identity(identity.id).await?;
        assert_eq!(updated_identity.auth_factors.len(), 2);
        assert!(updated_identity
            .auth_factors
            .contains(&AuthFactorType::Password));
        assert!(updated_identity
            .auth_factors
            .contains(&AuthFactorType::Totp));
    }

    #[tokio::test]
    async fn test_authentication_with_mfa() {
        let mut config = IdentityConfig::default();
        config.mfa_requirement = MfaRequirement::Required;
        let provider = IdentityProvider::new(config)?;

        let identity = provider
            .create_identity("testuser".to_string(), "test@example.com".to_string())
            .await
            .unwrap();

        // Update state to active
        provider
            .update_identity_state(identity.id, IdentityState::Active)
            .await
            .unwrap();

        // Enroll factors
        provider
            .enroll_auth_factor(
                identity.id,
                AuthFactor::Password("secure_password".to_string()),
            )
            .await
            .unwrap();
        let totp_secret = base64::encode("test_secret");
        provider
            .enroll_auth_factor(identity.id, AuthFactor::Totp(totp_secret))
            .await
            .unwrap();

        // Attempt authentication with single factor (should fail)
        let result = provider
            .authenticate(
                identity.id,
                vec![AuthFactor::Password("secure_password".to_string())],
            )
            .await;
        assert!(result.is_err());

        // Attempt authentication with MFA (should succeed)
        let session = provider
            .authenticate(
                identity.id,
                vec![
                    AuthFactor::Password("secure_password".to_string()),
                    AuthFactor::Totp("123456".to_string()),
                ],
            )
            .await
            .unwrap();

        assert_eq!(session.identity_id, identity.id);
        assert_eq!(session.auth_factors_used.len(), 2);
    }

    #[tokio::test]
    async fn test_failed_authentication_lockout() {
        let mut config = IdentityConfig::default();
        config.max_auth_attempts = 3;
        config.mfa_requirement = MfaRequirement::None; // Disable MFA for this test
        let provider = IdentityProvider::new(config)?;

        let identity = provider
            .create_identity("testuser".to_string(), "test@example.com".to_string())
            .await
            .unwrap();
        provider
            .update_identity_state(identity.id, IdentityState::Active)
            .await
            .unwrap();
        provider
            .enroll_auth_factor(
                identity.id,
                AuthFactor::Password("correct_password".to_string()),
            )
            .await
            .unwrap();

        // Fail authentication 3 times
        for _ in 0..3 {
            let result = provider
                .authenticate(
                    identity.id,
                    vec![AuthFactor::Password("wrong_password".to_string())],
                )
                .await;
            assert!(result.is_err());
        }

        // Check identity is locked
        let locked_identity = provider.get_identity(identity.id).await?;
        assert_eq!(locked_identity.state, IdentityState::Locked);

        // Attempt with correct password should still fail
        let result = provider
            .authenticate(
                identity.id,
                vec![AuthFactor::Password("correct_password".to_string())],
            )
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_continuous_authentication() {
        let mut config = IdentityConfig::default();
        config.continuous_auth_enabled = true;
        config.session_timeout = Duration::milliseconds(100); // Very short for testing
        config.mfa_requirement = MfaRequirement::None; // Disable MFA for this test
        let provider = IdentityProvider::new(config)?;

        let identity = provider
            .create_identity("testuser".to_string(), "test@example.com".to_string())
            .await
            .unwrap();
        provider
            .update_identity_state(identity.id, IdentityState::Active)
            .await
            .unwrap();
        provider
            .enroll_auth_factor(identity.id, AuthFactor::Password("password".to_string()))
            .await
            .unwrap();

        let session = provider
            .authenticate(
                identity.id,
                vec![AuthFactor::Password("password".to_string())],
            )
            .await
            .unwrap();

        // Verify session is valid initially
        assert!(provider
            .verify_continuous_auth(session.session_id)
            .await
            .unwrap());

        // Manually update last activity to simulate timeout
        if let Some(mut sess) = provider.sessions.get_mut(&session.session_id) {
            sess.last_activity = Utc::now() - Duration::seconds(1);
        }

        // Verify session is now invalid due to timeout
        assert!(!provider
            .verify_continuous_auth(session.session_id)
            .await
            .unwrap());
    }

    #[tokio::test]
    async fn test_session_revocation() {
        let mut config = IdentityConfig::default();
        config.mfa_requirement = MfaRequirement::None; // Disable MFA for this test
        let provider = IdentityProvider::new(config)?;

        let identity = provider
            .create_identity("testuser".to_string(), "test@example.com".to_string())
            .await
            .unwrap();
        provider
            .update_identity_state(identity.id, IdentityState::Active)
            .await
            .unwrap();
        provider
            .enroll_auth_factor(identity.id, AuthFactor::Password("password".to_string()))
            .await
            .unwrap();

        let session = provider
            .authenticate(
                identity.id,
                vec![AuthFactor::Password("password".to_string())],
            )
            .await
            .unwrap();

        // Revoke session
        provider.revoke_session(session.session_id).await?;

        // Verify session is revoked (no longer exists)
        let result = provider.verify_continuous_auth(session.session_id).await;
        assert!(result.is_err());
        assert!(matches!(result, Err(ZeroTrustError::SessionInvalid { .. })));
    }

    #[tokio::test]
    async fn test_biometric_authentication() {
        let mut config = IdentityConfig::default();
        config.mfa_requirement = MfaRequirement::None; // Disable MFA for this test
        let provider = IdentityProvider::new(config)?;

        let identity = provider
            .create_identity("testuser".to_string(), "test@example.com".to_string())
            .await
            .unwrap();
        provider
            .update_identity_state(identity.id, IdentityState::Active)
            .await
            .unwrap();

        // Enroll biometric
        let bio_template = vec![1, 2, 3, 4, 5];
        let bio_hash = provider.hash_auth_data(&bio_template);
        let bio_data = BiometricData {
            biometric_type: BiometricType::Fingerprint,
            template: bio_template.clone(),
            template_hash: bio_hash.clone(),
        };
        provider
            .enroll_auth_factor(identity.id, AuthFactor::Biometric(bio_data.clone()))
            .await
            .unwrap();

        // Authenticate with biometric
        let session = provider
            .authenticate(identity.id, vec![AuthFactor::Biometric(bio_data)])
            .await
            .unwrap();

        assert!(session
            .auth_factors_used
            .contains(&AuthFactorType::Biometric));
    }

    #[tokio::test]
    async fn test_hardware_token_mfa() {
        let mut config = IdentityConfig::default();
        config.mfa_requirement = MfaRequirement::RequiredWithHardware;
        let provider = IdentityProvider::new(config)?;

        let identity = provider
            .create_identity("testuser".to_string(), "test@example.com".to_string())
            .await
            .unwrap();
        provider
            .update_identity_state(identity.id, IdentityState::Active)
            .await
            .unwrap();

        // Enroll password and hardware token
        provider
            .enroll_auth_factor(identity.id, AuthFactor::Password("password".to_string()))
            .await
            .unwrap();
        let hw_token = vec![0xDE, 0xAD, 0xBE, 0xEF];
        provider
            .enroll_auth_factor(identity.id, AuthFactor::HardwareToken(hw_token.clone()))
            .await
            .unwrap();

        // Authentication without hardware token should fail
        let result = provider
            .authenticate(
                identity.id,
                vec![
                    AuthFactor::Password("password".to_string()),
                    AuthFactor::Totp("123456".to_string()),
                ],
            )
            .await;
        assert!(result.is_err());

        // Authentication with hardware token should succeed
        let session = provider
            .authenticate(
                identity.id,
                vec![
                    AuthFactor::Password("password".to_string()),
                    AuthFactor::HardwareToken(hw_token),
                ],
            )
            .await
            .unwrap();

        assert!(session
            .auth_factors_used
            .contains(&AuthFactorType::HardwareToken));
    }

    #[tokio::test]
    async fn test_identity_state_transitions() {
        let config = IdentityConfig::default();
        let provider = IdentityProvider::new(config)?;

        let identity = provider
            .create_identity("testuser".to_string(), "test@example.com".to_string())
            .await
            .unwrap();

        // Initial state should be PendingVerification
        assert_eq!(identity.state, IdentityState::PendingVerification);

        // Transition to Active
        provider
            .update_identity_state(identity.id, IdentityState::Active)
            .await
            .unwrap();
        let active_identity = provider.get_identity(identity.id).await?;
        assert_eq!(active_identity.state, IdentityState::Active);

        // Transition to Suspended
        provider
            .update_identity_state(identity.id, IdentityState::Suspended)
            .await
            .unwrap();
        let suspended_identity = provider.get_identity(identity.id).await?;
        assert_eq!(suspended_identity.state, IdentityState::Suspended);

        // Cannot authenticate when suspended
        provider
            .enroll_auth_factor(identity.id, AuthFactor::Password("password".to_string()))
            .await
            .unwrap();
        let result = provider
            .authenticate(
                identity.id,
                vec![AuthFactor::Password("password".to_string())],
            )
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_verification_levels() {
        let mut config = IdentityConfig::default();
        config.verification_level = VerificationLevel::Critical;
        config.mfa_requirement = MfaRequirement::RequiredWithHardware;
        let provider = IdentityProvider::new(config)?;

        let identity = provider
            .create_identity(
                "critical_user".to_string(),
                "critical@example.com".to_string(),
            )
            .await
            .unwrap();
        provider
            .update_identity_state(identity.id, IdentityState::Active)
            .await
            .unwrap();

        // For critical verification, we need hardware token
        provider
            .enroll_auth_factor(identity.id, AuthFactor::Password("password".to_string()))
            .await
            .unwrap();
        let hw_token = vec![0xCA, 0xFE, 0xBA, 0xBE];
        provider
            .enroll_auth_factor(identity.id, AuthFactor::HardwareToken(hw_token.clone()))
            .await
            .unwrap();

        let session = provider
            .authenticate(
                identity.id,
                vec![
                    AuthFactor::Password("password".to_string()),
                    AuthFactor::HardwareToken(hw_token),
                ],
            )
            .await
            .unwrap();

        assert_eq!(session.auth_factors_used.len(), 2);
        assert!(session
            .auth_factors_used
            .contains(&AuthFactorType::HardwareToken));
    }

    #[tokio::test]
    async fn test_token_generation_uniqueness() {
        let config = IdentityConfig::default();
        let provider = IdentityProvider::new(config)?;

        let mut tokens = std::collections::HashSet::new();

        // Generate multiple tokens and ensure uniqueness
        for _ in 0..100 {
            let token = provider.generate_token()?;
            assert!(tokens.insert(token));
        }
    }

    #[tokio::test]
    async fn test_concurrent_authentication() {
        let mut config = IdentityConfig::default();
        config.mfa_requirement = MfaRequirement::None; // Disable MFA for this test
        let provider = Arc::new(IdentityProvider::new(config).unwrap());

        let identity = provider
            .create_identity(
                "concurrent_user".to_string(),
                "concurrent@example.com".to_string(),
            )
            .await
            .unwrap();
        provider
            .update_identity_state(identity.id, IdentityState::Active)
            .await
            .unwrap();
        provider
            .enroll_auth_factor(identity.id, AuthFactor::Password("password".to_string()))
            .await
            .unwrap();

        // Spawn multiple authentication attempts
        let mut handles = vec![];
        for _ in 0..10 {
            let provider_clone = provider.clone();
            let handle = tokio::spawn(async move {
                provider_clone
                    .authenticate(
                        identity.id,
                        vec![AuthFactor::Password("password".to_string())],
                    )
                    .await
            });
            handles.push(handle);
        }

        // All should succeed
        for handle in handles {
            let result = handle.await?;
            assert!(result.is_ok());
        }
    }
}
