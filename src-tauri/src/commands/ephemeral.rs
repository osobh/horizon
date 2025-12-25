//! Ephemeral Session Commands
//!
//! Tauri commands for managing ephemeral collaboration sessions.
//! These commands enable:
//! - Creating time-bounded sessions with specific permissions
//! - Generating shareable invite links
//! - Joining sessions with two-factor verification
//! - Tracking participant presence in real-time
//! - Managing session lifecycle

use crate::ephemeral_bridge::{
    ConnectionQuality, CreateSessionResponse, DeviceInfo, EphemeralParticipant,
    EphemeralPermissions, EphemeralSession, EphemeralStats, InviteLink, JoinSessionResponse,
    SessionType,
};
use crate::state::AppState;
use tauri::State;

/// Create a new ephemeral collaboration session.
///
/// Returns the session details and an invite link to share with collaborators.
#[tauri::command]
pub async fn create_ephemeral_session(
    session_type: String,
    name: String,
    permissions: Option<EphemeralPermissions>,
    ttl_minutes: Option<i64>,
    max_participants: Option<u32>,
    state: State<'_, AppState>,
) -> Result<CreateSessionResponse, String> {
    let session_type = match session_type.as_str() {
        "code_review" => SessionType::CodeReview,
        "pair_programming" => SessionType::PairProgramming,
        "training_observer" => SessionType::TrainingObserver,
        "model_inference" => SessionType::ModelInference,
        "data_exploration" => SessionType::DataExploration,
        other => SessionType::Custom(other.to_string()),
    };

    let permissions = permissions.unwrap_or_else(|| match &session_type {
        SessionType::CodeReview => EphemeralPermissions::read_only(),
        SessionType::PairProgramming => EphemeralPermissions::pair_programming(),
        SessionType::TrainingObserver => EphemeralPermissions::training_observer(),
        _ => EphemeralPermissions::read_only(),
    });

    state
        .ephemeral
        .create_session(
            session_type,
            name,
            permissions,
            ttl_minutes.unwrap_or(60),
            max_participants.unwrap_or(5),
        )
        .await
}

/// Get the invite URL for an existing session.
#[tauri::command]
pub async fn get_invite_url(
    session_id: String,
    state: State<'_, AppState>,
) -> Result<InviteLink, String> {
    state.ephemeral.get_invite_url(&session_id).await
}

/// Join an ephemeral session using an invite token and redemption code.
#[tauri::command]
pub async fn join_ephemeral_session(
    invite_token: String,
    redemption_code: String,
    display_name: String,
    device_info: DeviceInfo,
    state: State<'_, AppState>,
) -> Result<JoinSessionResponse, String> {
    state
        .ephemeral
        .join_session(&invite_token, &redemption_code, display_name, device_info)
        .await
}

/// Get the current presence (participants) in a session.
#[tauri::command]
pub async fn get_ephemeral_presence(
    session_id: String,
    state: State<'_, AppState>,
) -> Result<Vec<EphemeralParticipant>, String> {
    state.ephemeral.get_presence(&session_id).await
}

/// Get details of a specific session.
#[tauri::command]
pub async fn get_ephemeral_session(
    session_id: String,
    state: State<'_, AppState>,
) -> Result<EphemeralSession, String> {
    state.ephemeral.get_session(&session_id).await
}

/// List all active ephemeral sessions.
#[tauri::command]
pub async fn list_ephemeral_sessions(
    state: State<'_, AppState>,
) -> Result<Vec<EphemeralSession>, String> {
    Ok(state.ephemeral.list_sessions().await)
}

/// End an ephemeral session (host only).
#[tauri::command]
pub async fn end_ephemeral_session(
    session_id: String,
    state: State<'_, AppState>,
) -> Result<(), String> {
    state.ephemeral.end_session(&session_id).await
}

/// Leave an ephemeral session.
#[tauri::command]
pub async fn leave_ephemeral_session(
    session_id: String,
    participant_id: String,
    state: State<'_, AppState>,
) -> Result<(), String> {
    state
        .ephemeral
        .leave_session(&session_id, &participant_id)
        .await
}

/// Update participant activity status.
#[tauri::command]
pub async fn update_ephemeral_activity(
    session_id: String,
    participant_id: String,
    activity: Option<String>,
    state: State<'_, AppState>,
) -> Result<(), String> {
    state
        .ephemeral
        .update_activity(&session_id, &participant_id, activity)
        .await
}

/// Get ephemeral system statistics.
#[tauri::command]
pub async fn get_ephemeral_stats(state: State<'_, AppState>) -> Result<EphemeralStats, String> {
    Ok(state.ephemeral.get_stats().await)
}

/// Trigger cleanup of expired sessions (for maintenance).
#[tauri::command]
pub async fn cleanup_expired_sessions(state: State<'_, AppState>) -> Result<(), String> {
    state.ephemeral.cleanup_expired().await;
    Ok(())
}
