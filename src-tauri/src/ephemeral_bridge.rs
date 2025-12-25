//! Ephemeral Bridge
//!
//! Provides ephemeral identity and session management for Horizon.
//! Enables time-bounded, capability-restricted access for external collaborators
//! without requiring permanent account creation.
//!
//! Features:
//! - Create ephemeral collaboration sessions with specific permissions
//! - Generate shareable invite links with two-factor redemption
//! - Track real-time presence of ephemeral participants
//! - Manage session lifecycle (create, join, leave, expire)

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

/// Ephemeral session types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SessionType {
    /// Code review session - read-only access to code
    CodeReview,
    /// Pair programming - edit access to specific files
    PairProgramming,
    /// Training observation - view training metrics
    TrainingObserver,
    /// Model inference - run inference on shared models
    ModelInference,
    /// Data exploration - view and query datasets
    DataExploration,
    /// Custom session type
    Custom(String),
}

/// Permissions granted to ephemeral participants
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EphemeralPermissions {
    /// Can view files and code
    pub can_view: bool,
    /// Can edit files (if pair programming)
    pub can_edit: bool,
    /// Can execute code cells
    pub can_execute: bool,
    /// Can view training metrics
    pub can_view_training: bool,
    /// Can run inference
    pub can_inference: bool,
    /// Can access storage
    pub can_access_storage: bool,
    /// Allowed file patterns (glob patterns)
    pub allowed_paths: Vec<String>,
    /// Rate limits (requests per minute)
    pub rate_limit_rpm: u32,
    /// Maximum data transfer (bytes)
    pub max_data_transfer: u64,
}

impl EphemeralPermissions {
    /// Create read-only permissions
    pub fn read_only() -> Self {
        Self {
            can_view: true,
            rate_limit_rpm: 60,
            max_data_transfer: 100 * 1024 * 1024, // 100MB
            ..Default::default()
        }
    }

    /// Create pair programming permissions
    pub fn pair_programming() -> Self {
        Self {
            can_view: true,
            can_edit: true,
            can_execute: true,
            rate_limit_rpm: 120,
            max_data_transfer: 500 * 1024 * 1024, // 500MB
            ..Default::default()
        }
    }

    /// Create training observer permissions
    pub fn training_observer() -> Self {
        Self {
            can_view: true,
            can_view_training: true,
            rate_limit_rpm: 60,
            max_data_transfer: 50 * 1024 * 1024, // 50MB
            ..Default::default()
        }
    }
}

/// An active ephemeral session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EphemeralSession {
    /// Unique session identifier
    pub id: String,
    /// Session type
    pub session_type: SessionType,
    /// Session name/description
    pub name: String,
    /// Host user ID (who created the session)
    pub host_id: String,
    /// Permissions for this session
    pub permissions: EphemeralPermissions,
    /// When the session was created
    pub created_at: u64,
    /// When the session expires
    pub expires_at: u64,
    /// Current status
    pub status: SessionStatus,
    /// Number of active participants (excluding host)
    pub participant_count: u32,
    /// Maximum allowed participants
    pub max_participants: u32,
}

/// Session status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SessionStatus {
    /// Session is active and accepting participants
    Active,
    /// Session is full
    Full,
    /// Session has expired
    Expired,
    /// Session was ended by host
    Ended,
}

/// An ephemeral participant in a session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EphemeralParticipant {
    /// Participant's ephemeral ID
    pub id: String,
    /// Display name (provided during join)
    pub display_name: String,
    /// When they joined
    pub joined_at: u64,
    /// Connection quality
    pub connection_quality: ConnectionQuality,
    /// Whether they're currently active
    pub is_active: bool,
    /// Last activity timestamp
    pub last_activity: u64,
    /// Current activity (e.g., "viewing file.rs", "editing main.py")
    pub current_activity: Option<String>,
}

/// Connection quality indicator
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ConnectionQuality {
    /// Direct P2P connection
    Excellent,
    /// Relayed connection with low latency
    Good,
    /// Relayed connection with moderate latency
    Fair,
    /// High latency or unstable connection
    Poor,
    /// Disconnected
    Disconnected,
}

/// Invite link for joining an ephemeral session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InviteLink {
    /// The invite token (URL-safe string)
    pub token: String,
    /// Full URL to share
    pub url: String,
    /// When the invite expires
    pub expires_at: u64,
    /// Whether a redemption code is required
    pub requires_code: bool,
    /// The redemption code (if applicable, shown only to creator)
    pub redemption_code: Option<String>,
    /// Maximum number of uses
    pub max_uses: u32,
    /// Current number of uses
    pub current_uses: u32,
}

/// Response when creating a session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateSessionResponse {
    /// The created session
    pub session: EphemeralSession,
    /// The invite link to share
    pub invite: InviteLink,
}

/// Response when joining a session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JoinSessionResponse {
    /// The session joined
    pub session: EphemeralSession,
    /// The participant's ephemeral identity
    pub participant: EphemeralParticipant,
    /// Access token for this session
    pub access_token: String,
}

/// Device information for session join
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceInfo {
    /// Device type (desktop, laptop, mobile)
    pub device_type: String,
    /// Operating system
    pub os: String,
    /// Browser or client name
    pub client: String,
    /// Device fingerprint (for binding)
    pub fingerprint: Option<String>,
}

/// Bridge for managing ephemeral sessions
pub struct EphemeralBridge {
    /// Active sessions
    sessions: Arc<RwLock<HashMap<String, EphemeralSession>>>,
    /// Active participants by session ID
    participants: Arc<RwLock<HashMap<String, Vec<EphemeralParticipant>>>>,
    /// Pending invites by token
    invites: Arc<RwLock<HashMap<String, InviteLink>>>,
    /// Session ID -> Invite token mapping
    session_invites: Arc<RwLock<HashMap<String, String>>>,
}

impl EphemeralBridge {
    /// Create a new ephemeral bridge
    pub fn new() -> Self {
        Self {
            sessions: Arc::new(RwLock::new(HashMap::new())),
            participants: Arc::new(RwLock::new(HashMap::new())),
            invites: Arc::new(RwLock::new(HashMap::new())),
            session_invites: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create a new ephemeral session
    pub async fn create_session(
        &self,
        session_type: SessionType,
        name: String,
        permissions: EphemeralPermissions,
        ttl_minutes: i64,
        max_participants: u32,
    ) -> Result<CreateSessionResponse, String> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| e.to_string())?
            .as_secs();

        let session_id = Uuid::new_v4().to_string();
        let invite_token = generate_url_safe_token(32);
        let redemption_code = generate_redemption_code();

        let session = EphemeralSession {
            id: session_id.clone(),
            session_type,
            name,
            host_id: "local_user".to_string(), // TODO: Get from auth context
            permissions,
            created_at: now,
            expires_at: now + (ttl_minutes as u64 * 60),
            status: SessionStatus::Active,
            participant_count: 0,
            max_participants,
        };

        let invite = InviteLink {
            token: invite_token.clone(),
            url: format!("horizon://join/{}", invite_token),
            expires_at: session.expires_at,
            requires_code: true,
            redemption_code: Some(redemption_code),
            max_uses: max_participants,
            current_uses: 0,
        };

        // Store session and invite
        {
            let mut sessions = self.sessions.write().await;
            sessions.insert(session_id.clone(), session.clone());
        }
        {
            let mut invites = self.invites.write().await;
            invites.insert(invite_token.clone(), invite.clone());
        }
        {
            let mut session_invites = self.session_invites.write().await;
            session_invites.insert(session_id.clone(), invite_token);
        }
        {
            let mut participants = self.participants.write().await;
            participants.insert(session_id, Vec::new());
        }

        Ok(CreateSessionResponse { session, invite })
    }

    /// Get invite URL for a session
    pub async fn get_invite_url(&self, session_id: &str) -> Result<InviteLink, String> {
        let session_invites = self.session_invites.read().await;
        let token = session_invites
            .get(session_id)
            .ok_or_else(|| "Session not found".to_string())?;

        let invites = self.invites.read().await;
        invites
            .get(token)
            .cloned()
            .ok_or_else(|| "Invite not found".to_string())
    }

    /// Join an ephemeral session
    pub async fn join_session(
        &self,
        invite_token: &str,
        redemption_code: &str,
        display_name: String,
        device_info: DeviceInfo,
    ) -> Result<JoinSessionResponse, String> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| e.to_string())?
            .as_secs();

        // Validate invite
        let session_id = {
            let mut invites = self.invites.write().await;
            let invite = invites
                .get_mut(invite_token)
                .ok_or_else(|| "Invalid invite token".to_string())?;

            if now > invite.expires_at {
                return Err("Invite has expired".to_string());
            }

            if invite.current_uses >= invite.max_uses {
                return Err("Invite has reached maximum uses".to_string());
            }

            if invite.requires_code {
                let expected_code = invite
                    .redemption_code
                    .as_ref()
                    .ok_or_else(|| "Redemption code required".to_string())?;
                if redemption_code != expected_code {
                    return Err("Invalid redemption code".to_string());
                }
            }

            invite.current_uses += 1;

            // Find session ID from token
            let session_invites = self.session_invites.read().await;
            session_invites
                .iter()
                .find(|(_, t)| *t == invite_token)
                .map(|(s, _)| s.clone())
                .ok_or_else(|| "Session not found".to_string())?
        };

        // Get session and update participant count
        let session = {
            let mut sessions = self.sessions.write().await;
            let session = sessions
                .get_mut(&session_id)
                .ok_or_else(|| "Session not found".to_string())?;

            if session.status != SessionStatus::Active {
                return Err(format!("Session is {:?}", session.status));
            }

            if session.participant_count >= session.max_participants {
                session.status = SessionStatus::Full;
                return Err("Session is full".to_string());
            }

            session.participant_count += 1;
            session.clone()
        };

        // Create participant
        let participant = EphemeralParticipant {
            id: Uuid::new_v4().to_string(),
            display_name,
            joined_at: now,
            connection_quality: ConnectionQuality::Good,
            is_active: true,
            last_activity: now,
            current_activity: None,
        };

        // Add participant to session
        {
            let mut participants = self.participants.write().await;
            let session_participants = participants.entry(session_id).or_insert_with(Vec::new);
            session_participants.push(participant.clone());
        }

        // Generate access token
        let access_token = generate_url_safe_token(64);

        Ok(JoinSessionResponse {
            session,
            participant,
            access_token,
        })
    }

    /// Get participants in a session
    pub async fn get_presence(&self, session_id: &str) -> Result<Vec<EphemeralParticipant>, String> {
        let participants = self.participants.read().await;
        participants
            .get(session_id)
            .cloned()
            .ok_or_else(|| "Session not found".to_string())
    }

    /// Get a specific session
    pub async fn get_session(&self, session_id: &str) -> Result<EphemeralSession, String> {
        let sessions = self.sessions.read().await;
        sessions
            .get(session_id)
            .cloned()
            .ok_or_else(|| "Session not found".to_string())
    }

    /// List all active sessions
    pub async fn list_sessions(&self) -> Vec<EphemeralSession> {
        let sessions = self.sessions.read().await;
        sessions
            .values()
            .filter(|s| s.status == SessionStatus::Active)
            .cloned()
            .collect()
    }

    /// End an ephemeral session
    pub async fn end_session(&self, session_id: &str) -> Result<(), String> {
        let mut sessions = self.sessions.write().await;
        let session = sessions
            .get_mut(session_id)
            .ok_or_else(|| "Session not found".to_string())?;

        session.status = SessionStatus::Ended;

        // Clean up participants
        {
            let mut participants = self.participants.write().await;
            participants.remove(session_id);
        }

        Ok(())
    }

    /// Leave an ephemeral session
    pub async fn leave_session(
        &self,
        session_id: &str,
        participant_id: &str,
    ) -> Result<(), String> {
        // Remove participant
        {
            let mut participants = self.participants.write().await;
            if let Some(session_participants) = participants.get_mut(session_id) {
                session_participants.retain(|p| p.id != participant_id);
            }
        }

        // Update session participant count
        {
            let mut sessions = self.sessions.write().await;
            if let Some(session) = sessions.get_mut(session_id) {
                session.participant_count = session.participant_count.saturating_sub(1);
                if session.status == SessionStatus::Full {
                    session.status = SessionStatus::Active;
                }
            }
        }

        Ok(())
    }

    /// Update participant activity
    pub async fn update_activity(
        &self,
        session_id: &str,
        participant_id: &str,
        activity: Option<String>,
    ) -> Result<(), String> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| e.to_string())?
            .as_secs();

        let mut participants = self.participants.write().await;
        if let Some(session_participants) = participants.get_mut(session_id) {
            if let Some(participant) = session_participants
                .iter_mut()
                .find(|p| p.id == participant_id)
            {
                participant.last_activity = now;
                participant.current_activity = activity;
                participant.is_active = true;
            }
        }

        Ok(())
    }

    /// Process expired sessions (should be called periodically)
    pub async fn cleanup_expired(&self) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // Mark expired sessions
        {
            let mut sessions = self.sessions.write().await;
            for session in sessions.values_mut() {
                if session.status == SessionStatus::Active && now > session.expires_at {
                    session.status = SessionStatus::Expired;
                }
            }
        }

        // Clean up expired invites
        {
            let mut invites = self.invites.write().await;
            invites.retain(|_, invite| now <= invite.expires_at);
        }
    }

    /// Get session statistics
    pub async fn get_stats(&self) -> EphemeralStats {
        let sessions = self.sessions.read().await;
        let participants = self.participants.read().await;

        let active_sessions = sessions
            .values()
            .filter(|s| s.status == SessionStatus::Active)
            .count();

        let total_participants: usize = participants.values().map(|p| p.len()).sum();

        EphemeralStats {
            active_sessions: active_sessions as u32,
            total_participants: total_participants as u32,
            total_sessions_created: sessions.len() as u32,
        }
    }
}

impl Default for EphemeralBridge {
    fn default() -> Self {
        Self::new()
    }
}

/// Ephemeral system statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EphemeralStats {
    /// Number of active sessions
    pub active_sessions: u32,
    /// Total participants across all sessions
    pub total_participants: u32,
    /// Total sessions ever created
    pub total_sessions_created: u32,
}

// Helper functions

/// Generate a URL-safe random token
fn generate_url_safe_token(length: usize) -> String {
    use rand::Rng;
    const CHARSET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    let mut rng = rand::thread_rng();
    (0..length)
        .map(|_| {
            let idx = rng.gen_range(0..CHARSET.len());
            CHARSET[idx] as char
        })
        .collect()
}

/// Generate a human-readable redemption code (6 characters, alphanumeric, no confusing chars)
fn generate_redemption_code() -> String {
    use rand::Rng;
    // Exclude confusing characters: 0, O, I, l, 1
    const CHARSET: &[u8] = b"ABCDEFGHJKMNPQRSTUVWXYZ23456789";
    let mut rng = rand::thread_rng();
    (0..6)
        .map(|_| {
            let idx = rng.gen_range(0..CHARSET.len());
            CHARSET[idx] as char
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_create_session() {
        let bridge = EphemeralBridge::new();
        let response = bridge
            .create_session(
                SessionType::PairProgramming,
                "Test Session".to_string(),
                EphemeralPermissions::pair_programming(),
                60,
                5,
            )
            .await
            .unwrap();

        assert_eq!(response.session.name, "Test Session");
        assert_eq!(response.session.status, SessionStatus::Active);
        assert!(response.invite.requires_code);
        assert!(response.invite.redemption_code.is_some());
    }

    #[tokio::test]
    async fn test_join_session() {
        let bridge = EphemeralBridge::new();

        // Create session
        let create_response = bridge
            .create_session(
                SessionType::CodeReview,
                "Review Session".to_string(),
                EphemeralPermissions::read_only(),
                60,
                3,
            )
            .await
            .unwrap();

        let redemption_code = create_response.invite.redemption_code.unwrap();

        // Join session
        let join_response = bridge
            .join_session(
                &create_response.invite.token,
                &redemption_code,
                "Alice".to_string(),
                DeviceInfo {
                    device_type: "desktop".to_string(),
                    os: "macOS".to_string(),
                    client: "Horizon".to_string(),
                    fingerprint: None,
                },
            )
            .await
            .unwrap();

        assert_eq!(join_response.participant.display_name, "Alice");
        assert_eq!(join_response.session.participant_count, 1);
    }

    #[tokio::test]
    async fn test_get_presence() {
        let bridge = EphemeralBridge::new();

        let create_response = bridge
            .create_session(
                SessionType::PairProgramming,
                "Pair Session".to_string(),
                EphemeralPermissions::pair_programming(),
                60,
                5,
            )
            .await
            .unwrap();

        let redemption_code = create_response.invite.redemption_code.unwrap();

        // Join with two participants
        bridge
            .join_session(
                &create_response.invite.token,
                &redemption_code,
                "Alice".to_string(),
                DeviceInfo {
                    device_type: "desktop".to_string(),
                    os: "macOS".to_string(),
                    client: "Horizon".to_string(),
                    fingerprint: None,
                },
            )
            .await
            .unwrap();

        bridge
            .join_session(
                &create_response.invite.token,
                &redemption_code,
                "Bob".to_string(),
                DeviceInfo {
                    device_type: "laptop".to_string(),
                    os: "Windows".to_string(),
                    client: "Horizon".to_string(),
                    fingerprint: None,
                },
            )
            .await
            .unwrap();

        let presence = bridge
            .get_presence(&create_response.session.id)
            .await
            .unwrap();

        assert_eq!(presence.len(), 2);
    }
}
