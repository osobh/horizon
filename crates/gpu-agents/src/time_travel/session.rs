//! Debug Session Management
//!
//! Handles debugging session lifecycle and state management

use super::EvolutionSnapshot;
use crate::time_travel::evolution_debugger::DebugSessionConfig;
use std::collections::HashMap;
use std::time::SystemTime;

/// Debug session for time-travel debugging
#[derive(Debug)]
pub struct DebugSession {
    pub session_id: String,
    pub session_name: String,
    pub start_time: SystemTime,
    pub is_active: bool,
    pub snapshots: HashMap<String, EvolutionSnapshot>,
    pub config: DebugSessionConfig,
}

impl DebugSession {
    /// Create new debug session
    pub fn new(session_name: &str, config: DebugSessionConfig) -> Self {
        Self {
            session_id: format!(
                "session_{}",
                SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)?
                    .as_millis()
            ),
            session_name: session_name.to_string(),
            start_time: SystemTime::now(),
            is_active: true,
            snapshots: HashMap::new(),
            config,
        }
    }

    /// Check if session is active
    pub fn is_active(&self) -> bool {
        self.is_active
    }

    /// Add snapshot to session
    pub fn add_snapshot(&mut self, snapshot: EvolutionSnapshot) {
        self.snapshots.insert(snapshot.id.clone(), snapshot);
    }

    /// Get snapshot by ID
    pub fn get_snapshot(&self, snapshot_id: &str) -> Option<&EvolutionSnapshot> {
        self.snapshots.get(snapshot_id)
    }

    /// End debugging session
    pub fn end_session(&mut self) {
        self.is_active = false;
    }
}
