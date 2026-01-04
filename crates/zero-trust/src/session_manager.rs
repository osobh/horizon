//! Session lifecycle, token validation, session security, and continuous session verification
//!
//! This module implements comprehensive session management following zero-trust principles:
//! - Secure session creation and lifecycle management
//! - Token-based authentication with rotation
//! - Continuous session verification
//! - Session risk assessment
//! - Concurrent session management

use crate::error::{ZeroTrustError, ZeroTrustResult};
use async_trait::async_trait;
use chrono::{DateTime, Duration, Utc};
use dashmap::DashMap;
use ring::hmac;
use ring::rand::{SecureRandom, SystemRandom};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::sync::Arc;
use uuid::Uuid;

/// Session manager configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionConfig {
    /// Session timeout duration
    pub session_timeout: Duration,
    /// Idle timeout duration
    pub idle_timeout: Duration,
    /// Maximum concurrent sessions per user
    pub max_concurrent_sessions: usize,
    /// Token rotation interval
    pub token_rotation_interval: Duration,
    /// Enable session binding
    pub session_binding: bool,
    /// Session verification interval
    pub verification_interval: Duration,
    /// Enable session recording
    pub session_recording: bool,
    /// Session token length
    pub token_length: usize,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            session_timeout: Duration::hours(8),
            idle_timeout: Duration::minutes(30),
            max_concurrent_sessions: 3,
            token_rotation_interval: Duration::hours(1),
            session_binding: true,
            verification_interval: Duration::minutes(5),
            session_recording: true,
            token_length: 32,
        }
    }
}

/// Session information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    /// Session ID
    pub session_id: Uuid,
    /// User ID
    pub user_id: Uuid,
    /// Session token
    pub token: String,
    /// Refresh token
    pub refresh_token: String,
    /// Session state
    pub state: SessionState,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Expiry timestamp
    pub expires_at: DateTime<Utc>,
    /// Last activity timestamp
    pub last_activity: DateTime<Utc>,
    /// Last token rotation
    pub last_rotation: DateTime<Utc>,
    /// Session binding data
    pub binding: SessionBinding,
    /// Session metadata
    pub metadata: SessionMetadata,
    /// Risk score
    pub risk_score: f64,
}

/// Session states
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SessionState {
    /// Active session
    Active,
    /// Session requires re-authentication
    RequiresAuth,
    /// Session is suspended
    Suspended,
    /// Session is expired
    Expired,
    /// Session is revoked
    Revoked,
}

/// Session binding information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionBinding {
    /// Device ID
    pub device_id: Option<Uuid>,
    /// IP address
    pub ip_address: String,
    /// User agent
    pub user_agent: String,
    /// Browser fingerprint
    pub fingerprint: Option<String>,
    /// Location
    pub location: Option<String>,
}

/// Session metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMetadata {
    /// Authentication method used
    pub auth_method: AuthMethod,
    /// Authentication factors used
    pub auth_factors: Vec<String>,
    /// Session permissions
    pub permissions: HashSet<String>,
    /// Session attributes
    pub attributes: serde_json::Value,
    /// Activity log
    pub activity_log: Vec<SessionActivity>,
}

/// Authentication methods
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuthMethod {
    /// Password-based
    Password,
    /// Multi-factor authentication
    Mfa,
    /// Single sign-on
    Sso,
    /// Certificate-based
    Certificate,
    /// Biometric
    Biometric,
}

/// Session activity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionActivity {
    /// Activity timestamp
    pub timestamp: DateTime<Utc>,
    /// Activity type
    pub activity_type: ActivityType,
    /// Resource accessed
    pub resource: Option<String>,
    /// Result
    pub result: ActivityResult,
    /// Risk score at time of activity
    pub risk_score: f64,
}

/// Activity types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActivityType {
    /// Resource access
    ResourceAccess,
    /// Permission check
    PermissionCheck,
    /// Token rotation
    TokenRotation,
    /// Risk assessment
    RiskAssessment,
    /// Session verification
    Verification,
}

/// Activity results
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActivityResult {
    Success,
    Denied,
    Challenge,
    Error,
}

/// Token validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenValidation {
    /// Is token valid
    pub valid: bool,
    /// Session ID if valid
    pub session_id: Option<Uuid>,
    /// Validation errors
    pub errors: Vec<String>,
    /// Token metadata
    pub metadata: Option<TokenMetadata>,
}

/// Token metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenMetadata {
    /// Token type
    pub token_type: TokenType,
    /// Issued at
    pub issued_at: DateTime<Utc>,
    /// Expires at
    pub expires_at: DateTime<Utc>,
    /// Token scope
    pub scope: Vec<String>,
}

/// Token types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TokenType {
    Access,
    Refresh,
    Temporary,
}

/// Session verification result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationResult {
    /// Verification status
    pub verified: bool,
    /// Risk score
    pub risk_score: f64,
    /// Required actions
    pub required_actions: Vec<RequiredAction>,
    /// Verification timestamp
    pub timestamp: DateTime<Utc>,
}

/// Required actions
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RequiredAction {
    /// Re-authenticate
    ReAuthenticate,
    /// Rotate token
    RotateToken,
    /// Update binding
    UpdateBinding,
    /// Reduce permissions
    ReducePermissions,
    /// Terminate session
    Terminate,
}

/// Session manager trait
#[async_trait]
pub trait SessionManagerTrait: Send + Sync {
    /// Create new session
    async fn create_session(
        &self,
        user_id: Uuid,
        binding: SessionBinding,
        auth_method: AuthMethod,
    ) -> ZeroTrustResult<Session>;

    /// Validate token
    async fn validate_token(&self, token: &str) -> ZeroTrustResult<TokenValidation>;

    /// Refresh session
    async fn refresh_session(&self, refresh_token: &str) -> ZeroTrustResult<Session>;

    /// Verify session
    async fn verify_session(&self, session_id: Uuid) -> ZeroTrustResult<VerificationResult>;

    /// Update session activity
    async fn update_activity(
        &self,
        session_id: Uuid,
        activity: SessionActivity,
    ) -> ZeroTrustResult<()>;

    /// Rotate token
    async fn rotate_token(&self, session_id: Uuid) -> ZeroTrustResult<String>;

    /// Revoke session
    async fn revoke_session(&self, session_id: Uuid) -> ZeroTrustResult<()>;

    /// Get active sessions for user
    async fn get_user_sessions(&self, user_id: Uuid) -> ZeroTrustResult<Vec<Session>>;

    /// Check session permissions
    async fn check_permission(&self, session_id: Uuid, permission: &str) -> ZeroTrustResult<bool>;
}

/// Session manager implementation
pub struct SessionManager {
    config: SessionConfig,
    sessions: Arc<DashMap<Uuid, Session>>,
    token_index: Arc<DashMap<String, Uuid>>,
    user_sessions: Arc<DashMap<Uuid, HashSet<Uuid>>>,
    hmac_key: hmac::Key,
    rng: SystemRandom,
}

impl SessionManager {
    /// Create new session manager
    pub fn new(config: SessionConfig) -> ZeroTrustResult<Self> {
        let rng = SystemRandom::new();

        // Generate HMAC key for token signing
        let mut key_bytes = [0u8; 32];
        rng.fill(&mut key_bytes)
            .map_err(|_| ZeroTrustError::CryptographicError {
                operation: "HMAC key generation".to_string(),
                reason: "Failed to generate random bytes".to_string(),
            })?;

        let hmac_key = hmac::Key::new(hmac::HMAC_SHA256, &key_bytes);

        Ok(Self {
            config,
            sessions: Arc::new(DashMap::new()),
            token_index: Arc::new(DashMap::new()),
            user_sessions: Arc::new(DashMap::new()),
            hmac_key,
            rng,
        })
    }

    /// Generate secure token
    fn generate_token(&self) -> ZeroTrustResult<String> {
        let mut token_bytes = vec![0u8; self.config.token_length];
        self.rng
            .fill(&mut token_bytes)
            .map_err(|_| ZeroTrustError::CryptographicError {
                operation: "token generation".to_string(),
                reason: "Failed to generate random bytes".to_string(),
            })?;
        Ok(base64::encode(token_bytes))
    }

    /// Sign token
    fn sign_token(&self, token: &str) -> String {
        let signature = hmac::sign(&self.hmac_key, token.as_bytes());
        base64::encode(signature.as_ref())
    }

    /// Verify token signature
    fn verify_token_signature(&self, token: &str, signature: &str) -> bool {
        if let Ok(signature_bytes) = base64::decode(signature) {
            hmac::verify(&self.hmac_key, token.as_bytes(), &signature_bytes).is_ok()
        } else {
            false
        }
    }

    /// Check session binding
    fn check_binding(&self, session: &Session, current_binding: &SessionBinding) -> bool {
        if !self.config.session_binding {
            return true;
        }

        // Check IP address
        if session.binding.ip_address != current_binding.ip_address {
            return false;
        }

        // Check user agent
        if session.binding.user_agent != current_binding.user_agent {
            return false;
        }

        // Check device ID if present
        if let (Some(session_device), Some(current_device)) =
            (&session.binding.device_id, &current_binding.device_id)
        {
            if session_device != current_device {
                return false;
            }
        }

        true
    }

    /// Calculate session risk
    fn calculate_session_risk(&self, session: &Session) -> f64 {
        let mut risk: f64 = 0.0;

        // Time-based risk
        let session_age = Utc::now() - session.created_at;
        if session_age > Duration::hours(4) {
            risk += 0.1;
        }

        // Inactivity risk
        let idle_time = Utc::now() - session.last_activity;
        if idle_time > self.config.idle_timeout / 2 {
            risk += 0.15;
        }

        // Token rotation risk
        let rotation_age = Utc::now() - session.last_rotation;
        if rotation_age > self.config.token_rotation_interval {
            risk += 0.2;
        }

        // Authentication method risk
        match session.metadata.auth_method {
            AuthMethod::Password => risk += 0.2,
            AuthMethod::Mfa => risk += 0.0,
            AuthMethod::Sso => risk += 0.1,
            AuthMethod::Certificate => risk += 0.05,
            AuthMethod::Biometric => risk += 0.0,
        }

        // Activity-based risk
        let recent_activities = session
            .metadata
            .activity_log
            .iter()
            .filter(|a| Utc::now() - a.timestamp < Duration::minutes(30))
            .collect::<Vec<_>>();

        let failed_activities = recent_activities
            .iter()
            .filter(|a| matches!(a.result, ActivityResult::Denied | ActivityResult::Error))
            .count();

        if failed_activities > 3 {
            risk += 0.25;
        }

        risk.min(1.0f64)
    }

    /// Cleanup expired sessions
    async fn cleanup_expired_sessions(&self) {
        let now = Utc::now();
        let expired_sessions: Vec<Uuid> = self
            .sessions
            .iter()
            .filter(|entry| {
                let session = entry.value();
                session.expires_at < now
                    || matches!(session.state, SessionState::Expired | SessionState::Revoked)
            })
            .map(|entry| *entry.key())
            .collect();

        for session_id in expired_sessions {
            if let Some((_, session)) = self.sessions.remove(&session_id) {
                self.token_index.remove(&session.token);
                self.token_index.remove(&session.refresh_token);

                if let Some(mut user_sessions) = self.user_sessions.get_mut(&session.user_id) {
                    user_sessions.remove(&session_id);
                }
            }
        }
    }
}

#[async_trait]
impl SessionManagerTrait for SessionManager {
    async fn create_session(
        &self,
        user_id: Uuid,
        binding: SessionBinding,
        auth_method: AuthMethod,
    ) -> ZeroTrustResult<Session> {
        // Check concurrent session limit
        let user_session_count = self
            .user_sessions
            .get(&user_id)
            .map(|sessions| sessions.len())
            .unwrap_or(0);

        if user_session_count >= self.config.max_concurrent_sessions {
            return Err(ZeroTrustError::SessionInvalid {
                reason: "Maximum concurrent sessions exceeded".to_string(),
            });
        }

        let session_id = Uuid::new_v4();
        let token = self.generate_token()?;
        let refresh_token = self.generate_token()?;
        let now = Utc::now();

        let session = Session {
            session_id,
            user_id,
            token: token.clone(),
            refresh_token: refresh_token.clone(),
            state: SessionState::Active,
            created_at: now,
            expires_at: now + self.config.session_timeout,
            last_activity: now,
            last_rotation: now,
            binding,
            metadata: SessionMetadata {
                auth_method,
                auth_factors: vec![],
                permissions: HashSet::new(),
                attributes: serde_json::json!({}),
                activity_log: vec![],
            },
            risk_score: 0.0,
        };

        // Store session
        self.sessions.insert(session_id, session.clone());
        self.token_index.insert(token, session_id);
        self.token_index.insert(refresh_token, session_id);

        // Update user sessions
        self.user_sessions
            .entry(user_id)
            .or_default()
            .insert(session_id);

        // Cleanup expired sessions
        self.cleanup_expired_sessions().await;

        Ok(session)
    }

    async fn validate_token(&self, token: &str) -> ZeroTrustResult<TokenValidation> {
        // Check if token exists
        let session_id = match self.token_index.get(token) {
            Some(id) => *id,
            None => {
                return Ok(TokenValidation {
                    valid: false,
                    session_id: None,
                    errors: vec!["Token not found".to_string()],
                    metadata: None,
                });
            }
        };

        // Get session
        let session = match self.sessions.get(&session_id) {
            Some(s) => s,
            None => {
                return Ok(TokenValidation {
                    valid: false,
                    session_id: None,
                    errors: vec!["Session not found".to_string()],
                    metadata: None,
                });
            }
        };

        let mut errors = vec![];
        let now = Utc::now();

        // Check session state
        if session.state != SessionState::Active {
            errors.push(format!("Session state: {:?}", session.state));
        }

        // Check expiry
        if now > session.expires_at {
            errors.push("Session expired".to_string());
        }

        // Check idle timeout
        if now - session.last_activity > self.config.idle_timeout {
            errors.push("Session idle timeout".to_string());
        }

        let valid = errors.is_empty();

        let metadata = if valid {
            Some(TokenMetadata {
                token_type: if token == session.token {
                    TokenType::Access
                } else {
                    TokenType::Refresh
                },
                issued_at: session.last_rotation,
                expires_at: session.expires_at,
                scope: session.metadata.permissions.iter().cloned().collect(),
            })
        } else {
            None
        };

        Ok(TokenValidation {
            valid,
            session_id: Some(session_id),
            errors,
            metadata,
        })
    }

    async fn refresh_session(&self, refresh_token: &str) -> ZeroTrustResult<Session> {
        // Validate refresh token
        let validation = self.validate_token(refresh_token).await?;
        if !validation.valid {
            return Err(ZeroTrustError::TokenInvalid {
                reason: validation.errors.join(", "),
            });
        }

        let session_id = validation.session_id.ok_or_else(|| ZeroTrustError::TokenInvalid {
            reason: "Missing session ID in token".to_string(),
        })?;
        let mut session =
            self.sessions
                .get_mut(&session_id)
                .ok_or_else(|| ZeroTrustError::SessionInvalid {
                    reason: "Session not found".to_string(),
                })?;

        // Check if refresh token matches
        if session.refresh_token != refresh_token {
            return Err(ZeroTrustError::TokenInvalid {
                reason: "Invalid refresh token".to_string(),
            });
        }

        // Generate new tokens
        let new_token = self.generate_token()?;
        let new_refresh_token = self.generate_token()?;

        // Remove old tokens from index
        self.token_index.remove(&session.token);
        self.token_index.remove(&session.refresh_token);

        // Update session
        session.token = new_token.clone();
        session.refresh_token = new_refresh_token.clone();
        session.last_rotation = Utc::now();
        session.expires_at = Utc::now() + self.config.session_timeout;

        // Add new tokens to index
        self.token_index.insert(new_token, session_id);
        self.token_index.insert(new_refresh_token, session_id);

        // Record activity
        let activity = SessionActivity {
            timestamp: Utc::now(),
            activity_type: ActivityType::TokenRotation,
            resource: None,
            result: ActivityResult::Success,
            risk_score: session.risk_score,
        };
        session.metadata.activity_log.push(activity);

        Ok(session.clone())
    }

    async fn verify_session(&self, session_id: Uuid) -> ZeroTrustResult<VerificationResult> {
        let mut session =
            self.sessions
                .get_mut(&session_id)
                .ok_or_else(|| ZeroTrustError::SessionInvalid {
                    reason: "Session not found".to_string(),
                })?;

        let mut required_actions = vec![];
        let now = Utc::now();

        // Check session expiry
        if now > session.expires_at {
            session.state = SessionState::Expired;
            required_actions.push(RequiredAction::Terminate);
        }

        // Check idle timeout
        if now - session.last_activity > self.config.idle_timeout {
            required_actions.push(RequiredAction::ReAuthenticate);
        }

        // Check token rotation
        if now - session.last_rotation > self.config.token_rotation_interval {
            required_actions.push(RequiredAction::RotateToken);
        }

        // Calculate risk score
        let risk_score = self.calculate_session_risk(&session);
        session.risk_score = risk_score;

        // Risk-based actions
        if risk_score > 0.8 {
            required_actions.push(RequiredAction::Terminate);
        } else if risk_score > 0.6 {
            required_actions.push(RequiredAction::ReAuthenticate);
            required_actions.push(RequiredAction::ReducePermissions);
        } else if risk_score > 0.4 {
            required_actions.push(RequiredAction::RotateToken);
        }

        // Record verification
        let activity = SessionActivity {
            timestamp: now,
            activity_type: ActivityType::Verification,
            resource: None,
            result: if required_actions.is_empty() {
                ActivityResult::Success
            } else {
                ActivityResult::Challenge
            },
            risk_score,
        };
        session.metadata.activity_log.push(activity);

        Ok(VerificationResult {
            verified: required_actions.is_empty(),
            risk_score,
            required_actions,
            timestamp: now,
        })
    }

    async fn update_activity(
        &self,
        session_id: Uuid,
        activity: SessionActivity,
    ) -> ZeroTrustResult<()> {
        let mut session =
            self.sessions
                .get_mut(&session_id)
                .ok_or_else(|| ZeroTrustError::SessionInvalid {
                    reason: "Session not found".to_string(),
                })?;

        session.last_activity = activity.timestamp;
        session.metadata.activity_log.push(activity);

        // Limit activity log size
        if session.metadata.activity_log.len() > 1000 {
            session.metadata.activity_log.remove(0);
        }

        Ok(())
    }

    async fn rotate_token(&self, session_id: Uuid) -> ZeroTrustResult<String> {
        let mut session =
            self.sessions
                .get_mut(&session_id)
                .ok_or_else(|| ZeroTrustError::SessionInvalid {
                    reason: "Session not found".to_string(),
                })?;

        // Generate new token
        let new_token = self.generate_token()?;

        // Remove old token from index
        self.token_index.remove(&session.token);

        // Update session
        session.token = new_token.clone();
        session.last_rotation = Utc::now();

        // Add new token to index
        self.token_index.insert(new_token.clone(), session_id);

        // Record activity
        let activity = SessionActivity {
            timestamp: Utc::now(),
            activity_type: ActivityType::TokenRotation,
            resource: None,
            result: ActivityResult::Success,
            risk_score: session.risk_score,
        };
        session.metadata.activity_log.push(activity);

        Ok(new_token)
    }

    async fn revoke_session(&self, session_id: Uuid) -> ZeroTrustResult<()> {
        let mut session =
            self.sessions
                .get_mut(&session_id)
                .ok_or_else(|| ZeroTrustError::SessionInvalid {
                    reason: "Session not found".to_string(),
                })?;

        session.state = SessionState::Revoked;

        // Remove tokens from index
        self.token_index.remove(&session.token);
        self.token_index.remove(&session.refresh_token);

        // Remove from user sessions
        if let Some(mut user_sessions) = self.user_sessions.get_mut(&session.user_id) {
            user_sessions.remove(&session_id);
        }

        Ok(())
    }

    async fn get_user_sessions(&self, user_id: Uuid) -> ZeroTrustResult<Vec<Session>> {
        let session_ids = self
            .user_sessions
            .get(&user_id)
            .map(|ids| ids.clone())
            .unwrap_or_default();

        let mut sessions = vec![];
        for session_id in session_ids {
            if let Some(session) = self.sessions.get(&session_id) {
                if matches!(
                    session.state,
                    SessionState::Active | SessionState::RequiresAuth
                ) {
                    sessions.push(session.clone());
                }
            }
        }

        Ok(sessions)
    }

    async fn check_permission(&self, session_id: Uuid, permission: &str) -> ZeroTrustResult<bool> {
        let mut session =
            self.sessions
                .get_mut(&session_id)
                .ok_or_else(|| ZeroTrustError::SessionInvalid {
                    reason: "Session not found".to_string(),
                })?;

        let has_permission = session.metadata.permissions.contains(permission);

        // Record activity
        let activity = SessionActivity {
            timestamp: Utc::now(),
            activity_type: ActivityType::PermissionCheck,
            resource: Some(permission.to_string()),
            result: if has_permission {
                ActivityResult::Success
            } else {
                ActivityResult::Denied
            },
            risk_score: session.risk_score,
        };
        session.metadata.activity_log.push(activity);

        Ok(has_permission)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_session_creation() {
        let config = SessionConfig::default();
        let manager = SessionManager::new(config)?;

        let user_id = Uuid::new_v4();
        let binding = SessionBinding {
            device_id: Some(Uuid::new_v4()),
            ip_address: "192.168.1.100".to_string(),
            user_agent: "Mozilla/5.0".to_string(),
            fingerprint: Some("abc123".to_string()),
            location: Some("New York".to_string()),
        };

        let session = manager
            .create_session(user_id, binding, AuthMethod::Mfa)
            .await
            .unwrap();

        assert_eq!(session.user_id, user_id);
        assert_eq!(session.state, SessionState::Active);
        assert!(!session.token.is_empty());
        assert!(!session.refresh_token.is_empty());
    }

    #[tokio::test]
    async fn test_token_validation() {
        let config = SessionConfig::default();
        let manager = SessionManager::new(config)?;

        let user_id = Uuid::new_v4();
        let binding = SessionBinding {
            device_id: None,
            ip_address: "192.168.1.100".to_string(),
            user_agent: "Mozilla/5.0".to_string(),
            fingerprint: None,
            location: None,
        };

        let session = manager
            .create_session(user_id, binding, AuthMethod::Password)
            .await
            .unwrap();

        // Validate valid token
        let validation = manager.validate_token(&session.token).await?;
        assert!(validation.valid);
        assert_eq!(validation.session_id, Some(session.session_id));

        // Validate invalid token
        let invalid_validation = manager.validate_token("invalid_token").await?;
        assert!(!invalid_validation.valid);
        assert!(!invalid_validation.errors.is_empty());
    }

    #[tokio::test]
    async fn test_session_refresh() {
        let config = SessionConfig::default();
        let manager = SessionManager::new(config)?;

        let user_id = Uuid::new_v4();
        let binding = SessionBinding {
            device_id: None,
            ip_address: "192.168.1.100".to_string(),
            user_agent: "Mozilla/5.0".to_string(),
            fingerprint: None,
            location: None,
        };

        let session = manager
            .create_session(user_id, binding, AuthMethod::Sso)
            .await
            .unwrap();
        let old_token = session.token.clone();

        // Refresh session
        let refreshed = manager
            .refresh_session(&session.refresh_token)
            .await
            .unwrap();

        assert_ne!(refreshed.token, old_token);
        assert_eq!(refreshed.session_id, session.session_id);
        assert_eq!(refreshed.state, SessionState::Active);
    }

    #[tokio::test]
    async fn test_concurrent_session_limit() {
        let mut config = SessionConfig::default();
        config.max_concurrent_sessions = 2;
        let manager = SessionManager::new(config)?;

        let user_id = Uuid::new_v4();
        let binding = SessionBinding {
            device_id: None,
            ip_address: "192.168.1.100".to_string(),
            user_agent: "Mozilla/5.0".to_string(),
            fingerprint: None,
            location: None,
        };

        // Create max sessions
        for i in 0..2 {
            let mut b = binding.clone();
            b.ip_address = format!("192.168.1.{}", 100 + i);
            manager
                .create_session(user_id, b, AuthMethod::Password)
                .await
                .unwrap();
        }

        // Try to create one more
        let result = manager
            .create_session(user_id, binding, AuthMethod::Password)
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_session_verification() {
        let config = SessionConfig::default();
        let manager = SessionManager::new(config)?;

        let user_id = Uuid::new_v4();
        let binding = SessionBinding {
            device_id: None,
            ip_address: "192.168.1.100".to_string(),
            user_agent: "Mozilla/5.0".to_string(),
            fingerprint: None,
            location: None,
        };

        let session = manager
            .create_session(user_id, binding, AuthMethod::Mfa)
            .await
            .unwrap();

        let verification = manager.verify_session(session.session_id).await?;
        assert!(verification.verified);
        assert!(verification.required_actions.is_empty());
        assert!(verification.risk_score < 0.5);
    }

    #[tokio::test]
    async fn test_token_rotation() {
        let config = SessionConfig::default();
        let manager = SessionManager::new(config)?;

        let user_id = Uuid::new_v4();
        let binding = SessionBinding {
            device_id: None,
            ip_address: "192.168.1.100".to_string(),
            user_agent: "Mozilla/5.0".to_string(),
            fingerprint: None,
            location: None,
        };

        let session = manager
            .create_session(user_id, binding, AuthMethod::Certificate)
            .await
            .unwrap();
        let old_token = session.token.clone();

        let new_token = manager.rotate_token(session.session_id).await?;

        assert_ne!(new_token, old_token);

        // Old token should be invalid
        let old_validation = manager.validate_token(&old_token).await?;
        assert!(!old_validation.valid);

        // New token should be valid
        let new_validation = manager.validate_token(&new_token).await?;
        assert!(new_validation.valid);
    }

    #[tokio::test]
    async fn test_session_revocation() {
        let config = SessionConfig::default();
        let manager = SessionManager::new(config)?;

        let user_id = Uuid::new_v4();
        let binding = SessionBinding {
            device_id: None,
            ip_address: "192.168.1.100".to_string(),
            user_agent: "Mozilla/5.0".to_string(),
            fingerprint: None,
            location: None,
        };

        let session = manager
            .create_session(user_id, binding, AuthMethod::Biometric)
            .await
            .unwrap();

        // Revoke session
        manager.revoke_session(session.session_id).await?;

        // Token should be invalid
        let validation = manager.validate_token(&session.token).await?;
        assert!(!validation.valid);

        // Session state should be revoked
        let revoked_session = manager.sessions.get(&session.session_id)?;
        assert_eq!(revoked_session.state, SessionState::Revoked);
    }

    #[tokio::test]
    async fn test_activity_tracking() {
        let config = SessionConfig::default();
        let manager = SessionManager::new(config)?;

        let user_id = Uuid::new_v4();
        let binding = SessionBinding {
            device_id: None,
            ip_address: "192.168.1.100".to_string(),
            user_agent: "Mozilla/5.0".to_string(),
            fingerprint: None,
            location: None,
        };

        let session = manager
            .create_session(user_id, binding, AuthMethod::Password)
            .await
            .unwrap();

        // Add some activities
        for i in 0..5 {
            let activity = SessionActivity {
                timestamp: Utc::now(),
                activity_type: ActivityType::ResourceAccess,
                resource: Some(format!("resource_{}", i)),
                result: if i % 2 == 0 {
                    ActivityResult::Success
                } else {
                    ActivityResult::Denied
                },
                risk_score: 0.3,
            };
            manager
                .update_activity(session.session_id, activity)
                .await
                .unwrap();
        }

        let updated_session = manager.sessions.get(&session.session_id)?;
        assert_eq!(updated_session.metadata.activity_log.len(), 5);
    }

    #[tokio::test]
    async fn test_permission_checking() {
        let config = SessionConfig::default();
        let manager = SessionManager::new(config)?;

        let user_id = Uuid::new_v4();
        let binding = SessionBinding {
            device_id: None,
            ip_address: "192.168.1.100".to_string(),
            user_agent: "Mozilla/5.0".to_string(),
            fingerprint: None,
            location: None,
        };

        let session = manager
            .create_session(user_id, binding, AuthMethod::Mfa)
            .await
            .unwrap();

        // Add permissions
        manager
            .sessions
            .get_mut(&session.session_id)
            .unwrap()
            .metadata
            .permissions
            .insert("read".to_string());
        manager
            .sessions
            .get_mut(&session.session_id)
            .unwrap()
            .metadata
            .permissions
            .insert("write".to_string());

        // Check permissions
        assert!(manager
            .check_permission(session.session_id, "read")
            .await
            .unwrap());
        assert!(manager
            .check_permission(session.session_id, "write")
            .await
            .unwrap());
        assert!(!manager
            .check_permission(session.session_id, "delete")
            .await
            .unwrap());
    }

    #[tokio::test]
    async fn test_user_sessions_retrieval() {
        let config = SessionConfig::default();
        let manager = SessionManager::new(config)?;

        let user_id = Uuid::new_v4();

        // Create multiple sessions
        for i in 0..3 {
            let binding = SessionBinding {
                device_id: Some(Uuid::new_v4()),
                ip_address: format!("192.168.1.{}", 100 + i),
                user_agent: "Mozilla/5.0".to_string(),
                fingerprint: None,
                location: None,
            };
            manager
                .create_session(user_id, binding, AuthMethod::Password)
                .await
                .unwrap();
        }

        let user_sessions = manager.get_user_sessions(user_id).await?;
        assert_eq!(user_sessions.len(), 3);
        assert!(user_sessions.iter().all(|s| s.user_id == user_id));
    }

    #[tokio::test]
    async fn test_session_risk_calculation() {
        let mut config = SessionConfig::default();
        config.idle_timeout = Duration::minutes(5);
        config.token_rotation_interval = Duration::minutes(10);
        let manager = SessionManager::new(config)?;

        let user_id = Uuid::new_v4();
        let binding = SessionBinding {
            device_id: None,
            ip_address: "192.168.1.100".to_string(),
            user_agent: "Mozilla/5.0".to_string(),
            fingerprint: None,
            location: None,
        };

        let session = manager
            .create_session(user_id, binding, AuthMethod::Password)
            .await
            .unwrap();

        // Simulate aging session
        let mut session_mut = manager.sessions.get_mut(&session.session_id)?;
        session_mut.created_at = Utc::now() - Duration::hours(5);
        session_mut.last_activity = Utc::now() - Duration::minutes(3);
        session_mut.last_rotation = Utc::now() - Duration::minutes(15);

        // Add failed activities
        for _ in 0..4 {
            session_mut.metadata.activity_log.push(SessionActivity {
                timestamp: Utc::now() - Duration::minutes(10),
                activity_type: ActivityType::ResourceAccess,
                resource: Some("sensitive".to_string()),
                result: ActivityResult::Denied,
                risk_score: 0.5,
            });
        }
        drop(session_mut);

        // Verify session - should have high risk
        let verification = manager.verify_session(session.session_id).await?;
        assert!(!verification.verified);
        assert!(verification.risk_score > 0.5);
        assert!(!verification.required_actions.is_empty());
    }

    #[tokio::test]
    async fn test_session_binding_verification() {
        let mut config = SessionConfig::default();
        config.session_binding = true;
        let manager = SessionManager::new(config)?;

        let user_id = Uuid::new_v4();
        let device_id = Uuid::new_v4();
        let binding = SessionBinding {
            device_id: Some(device_id),
            ip_address: "192.168.1.100".to_string(),
            user_agent: "Mozilla/5.0".to_string(),
            fingerprint: Some("fingerprint123".to_string()),
            location: Some("New York".to_string()),
        };

        let session = manager
            .create_session(user_id, binding.clone(), AuthMethod::Mfa)
            .await
            .unwrap();

        // Check with same binding
        assert!(manager.check_binding(&session, &binding));

        // Check with different IP
        let mut different_binding = binding.clone();
        different_binding.ip_address = "192.168.1.200".to_string();
        assert!(!manager.check_binding(&session, &different_binding));

        // Check with different device
        let mut different_device = binding.clone();
        different_device.device_id = Some(Uuid::new_v4());
        assert!(!manager.check_binding(&session, &different_device));
    }
}
