use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::comparator::{ComparisonResult, StateComparator};
use crate::error::{Result, TimeDebuggerError};
use crate::event_log::{AgentEvent, EventLog};
use crate::navigator::{Breakpoint, TimeNavigator, TimePosition};
use crate::snapshot::{SnapshotManager, StateSnapshot};

/// Configuration for debug sessions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebugSessionConfig {
    pub max_session_duration: chrono::Duration,
    pub auto_save_interval: std::time::Duration,
    pub max_snapshots_per_session: usize,
    pub enable_auto_breakpoints: bool,
    pub compression_enabled: bool,
    pub persistent_storage: bool,
}

impl Default for DebugSessionConfig {
    fn default() -> Self {
        Self {
            max_session_duration: chrono::Duration::hours(24),
            auto_save_interval: std::time::Duration::from_secs(300), // 5 minutes
            max_snapshots_per_session: 1000,
            enable_auto_breakpoints: true,
            compression_enabled: true,
            persistent_storage: false,
        }
    }
}

/// State of a debug session
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SessionState {
    Created,
    Active,
    Paused,
    Completed,
    Error(String),
}

/// Metadata about a debug session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebugSessionMetadata {
    pub id: Uuid,
    pub agent_id: Uuid,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub state: SessionState,
    pub name: Option<String>,
    pub description: Option<String>,
    pub tags: Vec<String>,
    pub creator: Option<String>,
}

/// Export format for debug sessions
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExportFormat {
    Json,
    Binary,
    Archive, // Compressed archive with all data
}

/// Import/Export data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionExportData {
    pub metadata: DebugSessionMetadata,
    pub snapshots: Vec<StateSnapshot>,
    pub events: Vec<AgentEvent>,
    pub breakpoints: Vec<Breakpoint>,
    pub navigation_history: Vec<TimePosition>,
    pub comparisons: Vec<ComparisonResult>,
    pub config: DebugSessionConfig,
}

/// A complete debug session managing agent debugging
#[derive(Clone)]
pub struct DebugSession {
    pub metadata: DebugSessionMetadata,
    pub config: DebugSessionConfig,
    snapshot_manager: Arc<SnapshotManager>,
    event_log: Arc<EventLog>,
    navigator: Arc<TimeNavigator>,
    comparator: Arc<StateComparator>,
    auto_save_task: Arc<RwLock<Option<tokio::task::JoinHandle<()>>>>,
    custom_data: Arc<RwLock<HashMap<String, serde_json::Value>>>,
    // Note: TempDir is not Clone, so we'll handle persistence differently
}

impl DebugSession {
    pub async fn new(
        agent_id: Uuid,
        config: DebugSessionConfig,
        snapshot_manager: Arc<SnapshotManager>,
        event_log: Arc<EventLog>,
        navigator: Arc<TimeNavigator>,
        comparator: Arc<StateComparator>,
    ) -> Result<Self> {
        let session_id = Uuid::new_v4();
        let now = Utc::now();

        let metadata = DebugSessionMetadata {
            id: session_id,
            agent_id,
            created_at: now,
            updated_at: now,
            state: SessionState::Created,
            name: None,
            description: None,
            tags: Vec::new(),
            creator: None,
        };

        // Initialize navigation state for this session
        navigator
            .initialize_navigation(session_id, agent_id, None)
            .await?;

        Ok(Self {
            metadata,
            config,
            snapshot_manager,
            event_log,
            navigator,
            comparator,
            auto_save_task: Arc::new(RwLock::new(None)),
            custom_data: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Start the debug session
    pub async fn start(&mut self) -> Result<()> {
        if self.metadata.state != SessionState::Created {
            return Err(TimeDebuggerError::SessionNotFound {
                id: self.metadata.id,
            });
        }

        // Initialize navigation for this session
        self.navigator
            .initialize_navigation(self.metadata.id, self.metadata.agent_id, None)
            .await?;

        // Start auto-save if enabled
        if self.config.auto_save_interval.as_secs() > 0 {
            self.start_auto_save().await;
        }

        // Set up automatic breakpoints if enabled
        if self.config.enable_auto_breakpoints {
            self.setup_auto_breakpoints().await?;
        }

        self.metadata.state = SessionState::Active;
        self.metadata.updated_at = Utc::now();

        Ok(())
    }

    /// Pause the debug session
    pub async fn pause(&mut self) -> Result<()> {
        if self.metadata.state != SessionState::Active {
            return Err(TimeDebuggerError::ConcurrentAccess {
                operation: "pause".to_string(),
            });
        }

        self.metadata.state = SessionState::Paused;
        self.metadata.updated_at = Utc::now();

        // Stop auto-save temporarily
        self.stop_auto_save().await;

        Ok(())
    }

    /// Resume the debug session
    pub async fn resume(&mut self) -> Result<()> {
        if self.metadata.state != SessionState::Paused {
            return Err(TimeDebuggerError::ConcurrentAccess {
                operation: "resume".to_string(),
            });
        }

        self.metadata.state = SessionState::Active;
        self.metadata.updated_at = Utc::now();

        // Restart auto-save
        if self.config.auto_save_interval.as_secs() > 0 {
            self.start_auto_save().await;
        }

        Ok(())
    }

    /// Complete the debug session
    pub async fn complete(&mut self) -> Result<()> {
        if !matches!(
            self.metadata.state,
            SessionState::Active | SessionState::Paused
        ) {
            return Err(TimeDebuggerError::ConcurrentAccess {
                operation: "complete".to_string(),
            });
        }

        // Stop auto-save
        self.stop_auto_save().await;

        // Clean up navigation state
        self.navigator.cleanup_session(self.metadata.id).await?;

        self.metadata.state = SessionState::Completed;
        self.metadata.updated_at = Utc::now();

        Ok(())
    }

    /// Get current session state
    pub fn get_state(&self) -> SessionState {
        self.metadata.state.clone()
    }

    /// Update session metadata
    pub async fn update_metadata(
        &mut self,
        name: Option<String>,
        description: Option<String>,
        tags: Option<Vec<String>>,
    ) -> Result<()> {
        if let Some(name) = name {
            self.metadata.name = Some(name);
        }
        if let Some(description) = description {
            self.metadata.description = Some(description);
        }
        if let Some(tags) = tags {
            self.metadata.tags = tags;
        }

        self.metadata.updated_at = Utc::now();
        Ok(())
    }

    /// Take a snapshot of current agent state
    pub async fn take_snapshot(
        &self,
        state_data: serde_json::Value,
        memory_usage: u64,
        metadata: HashMap<String, String>,
    ) -> Result<Uuid> {
        if self.metadata.state != SessionState::Active {
            return Err(TimeDebuggerError::ConcurrentAccess {
                operation: "take_snapshot".to_string(),
            });
        }

        self.snapshot_manager
            .create_snapshot(self.metadata.agent_id, state_data, memory_usage, metadata)
            .await
    }

    /// Record an event in the session
    pub async fn record_event(
        &self,
        event_type: crate::event_log::EventType,
        event_data: serde_json::Value,
        metadata: HashMap<String, String>,
        causality_id: Option<Uuid>,
    ) -> Result<Uuid> {
        if self.metadata.state != SessionState::Active {
            return Err(TimeDebuggerError::ConcurrentAccess {
                operation: "record_event".to_string(),
            });
        }

        self.event_log
            .record_event(
                self.metadata.agent_id,
                event_type,
                event_data,
                metadata,
                causality_id,
            )
            .await
    }

    /// Navigate to a specific time position
    pub async fn navigate_to_time(&self, target_time: DateTime<Utc>) -> Result<TimePosition> {
        if !matches!(
            self.metadata.state,
            SessionState::Active | SessionState::Paused
        ) {
            return Err(TimeDebuggerError::ConcurrentAccess {
                operation: "navigate".to_string(),
            });
        }

        self.navigator
            .navigate_to_time(self.metadata.id, target_time)
            .await
    }

    /// Navigate to a specific event index
    pub async fn navigate_to_event_index(&self, event_index: usize) -> Result<TimePosition> {
        if !matches!(
            self.metadata.state,
            SessionState::Active | SessionState::Paused
        ) {
            return Err(TimeDebuggerError::ConcurrentAccess {
                operation: "navigate".to_string(),
            });
        }

        self.navigator
            .navigate_to_event_index(self.metadata.id, event_index)
            .await
    }

    /// Step through time
    pub async fn step(
        &self,
        direction: crate::navigator::NavigationDirection,
        step_size: crate::navigator::StepSize,
    ) -> Result<TimePosition> {
        if !matches!(
            self.metadata.state,
            SessionState::Active | SessionState::Paused
        ) {
            return Err(TimeDebuggerError::ConcurrentAccess {
                operation: "step".to_string(),
            });
        }

        self.navigator
            .step(self.metadata.id, direction, step_size)
            .await
    }

    /// Get current position in time
    pub async fn get_current_position(&self) -> Result<TimePosition> {
        self.navigator.get_current_position(self.metadata.id).await
    }

    /// Create a breakpoint
    pub async fn create_breakpoint(
        &self,
        condition: crate::navigator::BreakpointCondition,
        metadata: HashMap<String, String>,
    ) -> Result<Uuid> {
        self.navigator
            .create_breakpoint(self.metadata.agent_id, condition, metadata)
            .await
    }

    /// Compare two snapshots
    pub async fn compare_snapshots(
        &self,
        from_snapshot_id: Uuid,
        to_snapshot_id: Uuid,
        options: Option<crate::comparator::ComparisonOptions>,
    ) -> Result<ComparisonResult> {
        let from_snapshot = self.snapshot_manager.get_snapshot(from_snapshot_id).await?;
        let to_snapshot = self.snapshot_manager.get_snapshot(to_snapshot_id).await?;

        self.comparator
            .compare_snapshots(&from_snapshot, &to_snapshot, options)
            .await
    }

    /// Get session statistics
    pub async fn get_statistics(&self) -> Result<SessionStatistics> {
        // Get agent-specific snapshots instead of all snapshots
        let agent_snapshots = self
            .snapshot_manager
            .get_agent_snapshots(self.metadata.agent_id)
            .await?;
        let snapshot_count = agent_snapshots.len();
        let total_memory: u64 = agent_snapshots.iter().map(|s| s.memory_usage).sum();

        // Get agent-specific event count
        let agent_events = self
            .event_log
            .get_agent_events(self.metadata.agent_id, None, None)
            .await
            .unwrap_or_default();
        let total_events = agent_events.len();

        let navigation_history = self
            .navigator
            .get_navigation_history(self.metadata.id)
            .await?;
        let breakpoints = self
            .navigator
            .get_agent_breakpoints(self.metadata.agent_id)
            .await?;
        let custom_data = self.custom_data.read().await;

        let session_duration = Utc::now() - self.metadata.created_at;

        Ok(SessionStatistics {
            session_id: self.metadata.id,
            agent_id: self.metadata.agent_id,
            session_duration,
            snapshot_count,
            diff_count: 0, // Diff count is not tracked per agent
            total_memory_usage: total_memory,
            total_events,
            breakpoint_count: breakpoints.len(),
            navigation_steps: navigation_history.len(),
            custom_data_count: custom_data.len(),
            state: self.metadata.state.clone(),
        })
    }

    /// Store custom data in the session
    pub async fn set_custom_data(&self, key: String, value: serde_json::Value) -> Result<()> {
        let mut custom_data = self.custom_data.write().await;
        custom_data.insert(key, value);
        Ok(())
    }

    /// Retrieve custom data from the session
    pub async fn get_custom_data(&self, key: &str) -> Result<Option<serde_json::Value>> {
        let custom_data = self.custom_data.read().await;
        Ok(custom_data.get(key).cloned())
    }

    /// Export session data
    pub async fn export_session(&self, format: ExportFormat) -> Result<Vec<u8>> {
        let export_data = self.prepare_export_data().await?;

        match format {
            ExportFormat::Json => {
                let json = serde_json::to_string_pretty(&export_data)?;
                Ok(json.into_bytes())
            }
            ExportFormat::Binary => {
                // In a real implementation, you might use bincode or similar
                let json = serde_json::to_string(&export_data)?;
                Ok(json.into_bytes())
            }
            ExportFormat::Archive => {
                // In a real implementation, you'd create a compressed archive
                let json = serde_json::to_string_pretty(&export_data)?;
                Ok(json.into_bytes())
            }
        }
    }

    /// Import session data (creates a new session)
    pub async fn import_session(
        data: &[u8],
        format: ExportFormat,
        snapshot_manager: Arc<SnapshotManager>,
        event_log: Arc<EventLog>,
        navigator: Arc<TimeNavigator>,
        comparator: Arc<StateComparator>,
    ) -> Result<Self> {
        let export_data: SessionExportData = match format {
            ExportFormat::Json | ExportFormat::Binary | ExportFormat::Archive => {
                let json_str = String::from_utf8(data.to_vec()).map_err(|e| {
                    TimeDebuggerError::DeserializationError {
                        message: format!("Invalid UTF-8: {}", e),
                    }
                })?;
                serde_json::from_str(&json_str)?
            }
        };

        // Create new session with imported metadata
        let mut session = Self::new(
            export_data.metadata.agent_id,
            export_data.config,
            snapshot_manager,
            event_log,
            navigator,
            comparator,
        )
        .await?;

        // Update metadata but keep new ID and timestamps
        session.metadata.name = export_data.metadata.name;
        session.metadata.description = export_data.metadata.description;
        session.metadata.tags = export_data.metadata.tags;
        session.metadata.creator = export_data.metadata.creator;

        // Import snapshots
        for snapshot in export_data.snapshots {
            session
                .snapshot_manager
                .create_snapshot(
                    snapshot.agent_id,
                    snapshot.state_data,
                    snapshot.memory_usage,
                    snapshot.metadata,
                )
                .await?;
        }

        // Import events
        if !export_data.events.is_empty() {
            let events_json = serde_json::to_string(&export_data.events)?;
            session
                .event_log
                .import_events(session.metadata.agent_id, &events_json)
                .await?;
        }

        // Import breakpoints
        for breakpoint in export_data.breakpoints {
            session
                .navigator
                .create_breakpoint(
                    breakpoint.agent_id,
                    breakpoint.condition,
                    breakpoint.metadata,
                )
                .await?;
        }

        Ok(session)
    }

    // Private helper methods

    async fn start_auto_save(&self) {
        let mut auto_save_task = self.auto_save_task.write().await;
        if auto_save_task.is_some() {
            return; // Already running
        }

        let session_id = self.metadata.id;
        let interval = self.config.auto_save_interval;
        let snapshot_manager = Arc::clone(&self.snapshot_manager);

        let task = tokio::spawn(async move {
            let mut interval = tokio::time::interval(interval);
            loop {
                interval.tick().await;
                // In a real implementation, this would save session state
                tracing::debug!("Auto-saving session {}", session_id);

                // Could save to temp_dir or persistent storage
                let _ = snapshot_manager.get_memory_stats().await;
            }
        });

        *auto_save_task = Some(task);
    }

    async fn stop_auto_save(&self) {
        let mut auto_save_task = self.auto_save_task.write().await;
        if let Some(task) = auto_save_task.take() {
            task.abort();
        }
    }

    async fn setup_auto_breakpoints(&self) -> Result<()> {
        // Set up common breakpoints automatically
        use crate::event_log::EventType;
        use crate::navigator::BreakpointCondition;

        // Break on errors
        self.navigator
            .create_breakpoint(
                self.metadata.agent_id,
                BreakpointCondition::OnEventType(EventType::ErrorOccurred),
                [("auto".to_string(), "true".to_string())]
                    .into_iter()
                    .collect(),
            )
            .await?;

        // Break on decision points
        self.navigator
            .create_breakpoint(
                self.metadata.agent_id,
                BreakpointCondition::OnEventType(EventType::DecisionMade),
                [("auto".to_string(), "true".to_string())]
                    .into_iter()
                    .collect(),
            )
            .await?;

        Ok(())
    }

    async fn prepare_export_data(&self) -> Result<SessionExportData> {
        let snapshots = self
            .snapshot_manager
            .get_agent_snapshots(self.metadata.agent_id)
            .await?;

        let events = self
            .event_log
            .get_agent_events(self.metadata.agent_id, None, None)
            .await?;

        let breakpoints = self
            .navigator
            .get_agent_breakpoints(self.metadata.agent_id)
            .await?;

        let navigation_history = self
            .navigator
            .get_navigation_history(self.metadata.id)
            .await
            .unwrap_or_default();

        Ok(SessionExportData {
            metadata: self.metadata.clone(),
            snapshots,
            events,
            breakpoints,
            navigation_history,
            comparisons: Vec::new(), // Could store comparison history
            config: self.config.clone(),
        })
    }
}

impl Drop for DebugSession {
    fn drop(&mut self) {
        // Cancel auto-save task when dropping
        if let Ok(mut task) = self.auto_save_task.try_write() {
            if let Some(handle) = task.take() {
                handle.abort();
            }
        }
    }
}

/// Statistics about a debug session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionStatistics {
    pub session_id: Uuid,
    pub agent_id: Uuid,
    pub session_duration: chrono::Duration,
    pub snapshot_count: usize,
    pub diff_count: usize,
    pub total_memory_usage: u64,
    pub total_events: usize,
    pub breakpoint_count: usize,
    pub navigation_steps: usize,
    pub custom_data_count: usize,
    pub state: SessionState,
}

/// Manager for multiple debug sessions
pub struct SessionManager {
    sessions: Arc<RwLock<HashMap<Uuid, DebugSession>>>,
    snapshot_manager: Arc<SnapshotManager>,
    event_log: Arc<EventLog>,
    navigator: Arc<TimeNavigator>,
    comparator: Arc<StateComparator>,
    default_config: DebugSessionConfig,
}

impl SessionManager {
    pub fn new(
        snapshot_manager: Arc<SnapshotManager>,
        event_log: Arc<EventLog>,
        navigator: Arc<TimeNavigator>,
        comparator: Arc<StateComparator>,
        default_config: DebugSessionConfig,
    ) -> Self {
        Self {
            sessions: Arc::new(RwLock::new(HashMap::new())),
            snapshot_manager,
            event_log,
            navigator,
            comparator,
            default_config,
        }
    }

    /// Create a new debug session
    pub async fn create_session(
        &self,
        agent_id: Uuid,
        config: Option<DebugSessionConfig>,
    ) -> Result<Uuid> {
        let config = config.unwrap_or_else(|| self.default_config.clone());

        let session = DebugSession::new(
            agent_id,
            config,
            Arc::clone(&self.snapshot_manager),
            Arc::clone(&self.event_log),
            Arc::clone(&self.navigator),
            Arc::clone(&self.comparator),
        )
        .await?;

        let session_id = session.metadata.id;

        let mut sessions = self.sessions.write().await;
        sessions.insert(session_id, session);

        Ok(session_id)
    }

    /// Get a session by ID
    pub async fn get_session(&self, session_id: Uuid) -> Result<DebugSession> {
        let sessions = self.sessions.read().await;
        sessions
            .get(&session_id)
            .cloned()
            .ok_or(TimeDebuggerError::SessionNotFound { id: session_id })
    }

    /// Get all sessions for an agent
    pub async fn get_agent_sessions(&self, agent_id: Uuid) -> Vec<DebugSessionMetadata> {
        let sessions = self.sessions.read().await;
        sessions
            .values()
            .filter(|session| session.metadata.agent_id == agent_id)
            .map(|session| session.metadata.clone())
            .collect()
    }

    /// Remove a completed session
    pub async fn remove_session(&self, session_id: Uuid) -> Result<()> {
        let mut sessions = self.sessions.write().await;
        let session = sessions
            .get(&session_id)
            .ok_or(TimeDebuggerError::SessionNotFound { id: session_id })?;

        if !matches!(
            session.metadata.state,
            SessionState::Completed | SessionState::Error(_)
        ) {
            return Err(TimeDebuggerError::ConcurrentAccess {
                operation: "remove".to_string(),
            });
        }

        sessions.remove(&session_id);
        Ok(())
    }

    /// Get statistics for all sessions
    pub async fn get_all_statistics(&self) -> Vec<SessionStatistics> {
        let sessions = self.sessions.read().await;
        let mut stats = Vec::new();

        eprintln!("get_all_statistics: Found {} sessions", sessions.len());

        for session in sessions.values() {
            eprintln!(
                "Processing session {} with state {:?}",
                session.metadata.id, session.metadata.state
            );
            match session.get_statistics().await {
                Ok(session_stats) => stats.push(session_stats),
                Err(e) => {
                    // For completed sessions, navigation state might be cleaned up
                    // Create basic statistics without navigation data
                    if matches!(session.metadata.state, SessionState::Completed) {
                        eprintln!("Creating basic stats for completed session: {:?}", e);
                        let agent_snapshots = session
                            .snapshot_manager
                            .get_agent_snapshots(session.metadata.agent_id)
                            .await
                            .unwrap_or_default();
                        let agent_events = session
                            .event_log
                            .get_agent_events(session.metadata.agent_id, None, None)
                            .await
                            .unwrap_or_default();

                        let basic_stats = SessionStatistics {
                            session_id: session.metadata.id,
                            agent_id: session.metadata.agent_id,
                            session_duration: chrono::Utc::now() - session.metadata.created_at,
                            snapshot_count: agent_snapshots.len(),
                            diff_count: 0,
                            total_memory_usage: agent_snapshots
                                .iter()
                                .map(|s| s.memory_usage)
                                .sum(),
                            total_events: agent_events.len(),
                            breakpoint_count: 0, // Can't get this for completed sessions
                            navigation_steps: 0, // Navigation state is cleaned up
                            custom_data_count: session.custom_data.read().await.len(),
                            state: session.metadata.state.clone(),
                        };
                        stats.push(basic_stats);
                    }
                }
            }
        }

        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::comparator::ComparisonOptions;
    use crate::event_log::EventLogConfig;
    use crate::snapshot::SnapshotConfig;
    use serde_json::json;

    async fn create_test_components() -> (
        Arc<SnapshotManager>,
        Arc<EventLog>,
        Arc<TimeNavigator>,
        Arc<StateComparator>,
    ) {
        let snapshot_manager = Arc::new(SnapshotManager::new(SnapshotConfig::default()));
        let event_log = Arc::new(EventLog::new(EventLogConfig::default()));
        let navigator = Arc::new(TimeNavigator::new(
            Arc::clone(&snapshot_manager),
            Arc::clone(&event_log),
            100,
        ));
        let comparator = Arc::new(StateComparator::new(ComparisonOptions::default()));

        (snapshot_manager, event_log, navigator, comparator)
    }

    #[tokio::test]
    async fn test_session_creation_and_lifecycle() {
        let (snapshot_manager, event_log, navigator, comparator) = create_test_components().await;
        let agent_id = Uuid::new_v4();
        let config = DebugSessionConfig::default();

        let mut session = DebugSession::new(
            agent_id,
            config,
            snapshot_manager,
            event_log,
            navigator,
            comparator,
        )
        .await
        .unwrap();

        assert_eq!(session.get_state(), SessionState::Created);
        assert_eq!(session.metadata.agent_id, agent_id);

        // Start session
        session.start().await?;
        assert_eq!(session.get_state(), SessionState::Active);

        // Pause session
        session.pause().await?;
        assert_eq!(session.get_state(), SessionState::Paused);

        // Resume session
        session.resume().await?;
        assert_eq!(session.get_state(), SessionState::Active);

        // Complete session
        session.complete().await?;
        assert_eq!(session.get_state(), SessionState::Completed);
    }

    #[tokio::test]
    async fn test_session_metadata_updates() {
        let (snapshot_manager, event_log, navigator, comparator) = create_test_components().await;
        let agent_id = Uuid::new_v4();

        let mut session = DebugSession::new(
            agent_id,
            DebugSessionConfig::default(),
            snapshot_manager,
            event_log,
            navigator,
            comparator,
        )
        .await
        .unwrap();

        // Update metadata
        session
            .update_metadata(
                Some("Test Session".to_string()),
                Some("Testing session functionality".to_string()),
                Some(vec!["test".to_string(), "debug".to_string()]),
            )
            .await
            .unwrap();

        assert_eq!(session.metadata.name, Some("Test Session".to_string()));
        assert_eq!(
            session.metadata.description,
            Some("Testing session functionality".to_string())
        );
        assert_eq!(session.metadata.tags, vec!["test", "debug"]);
    }

    #[tokio::test]
    async fn test_session_snapshot_operations() {
        let (snapshot_manager, event_log, navigator, comparator) = create_test_components().await;
        let agent_id = Uuid::new_v4();

        let mut session = DebugSession::new(
            agent_id,
            DebugSessionConfig::default(),
            snapshot_manager,
            event_log,
            navigator,
            comparator,
        )
        .await
        .unwrap();

        session.start().await?;

        // Take a snapshot
        let state_data = json!({"health": 100, "level": 1});
        let snapshot_id = session
            .take_snapshot(state_data, 1024, HashMap::new())
            .await
            .unwrap();

        // Verify snapshot was created
        let snapshot = session
            .snapshot_manager
            .get_snapshot(snapshot_id)
            .await
            .unwrap();
        assert_eq!(snapshot.agent_id, agent_id);
        assert_eq!(snapshot.state_data["health"], json!(100));
    }

    #[tokio::test]
    async fn test_session_event_recording() {
        let (snapshot_manager, event_log, navigator, comparator) = create_test_components().await;
        let agent_id = Uuid::new_v4();

        let mut session = DebugSession::new(
            agent_id,
            DebugSessionConfig::default(),
            snapshot_manager,
            event_log,
            navigator,
            comparator,
        )
        .await
        .unwrap();

        session.start().await?;

        // Record an event
        let event_id = session
            .record_event(
                crate::event_log::EventType::StateChange,
                json!({"action": "move"}),
                HashMap::new(),
                None,
            )
            .await
            .unwrap();

        // Verify event was recorded
        let event = session.event_log.get_event(event_id).await?;
        assert_eq!(event.agent_id, agent_id);
        assert_eq!(event.event_data["action"], json!("move"));
    }

    #[tokio::test]
    async fn test_session_navigation() {
        let (snapshot_manager, event_log, navigator, comparator) = create_test_components().await;
        let agent_id = Uuid::new_v4();

        let mut session = DebugSession::new(
            agent_id,
            DebugSessionConfig::default(),
            snapshot_manager,
            event_log,
            navigator,
            comparator,
        )
        .await
        .unwrap();

        session.start().await?;

        // Record some events first
        for i in 0..3 {
            session
                .record_event(
                    crate::event_log::EventType::StateChange,
                    json!({"step": i}),
                    HashMap::new(),
                    None,
                )
                .await
                .unwrap();
        }

        // Test navigation
        let position = session
            .step(
                crate::navigator::NavigationDirection::Forward,
                crate::navigator::StepSize::Event,
            )
            .await
            .unwrap();

        assert_eq!(position.agent_id, agent_id);

        let current_position = session.get_current_position().await?;
        assert_eq!(current_position.agent_id, agent_id);
    }

    #[tokio::test]
    async fn test_session_breakpoints() {
        let (snapshot_manager, event_log, navigator, comparator) = create_test_components().await;
        let agent_id = Uuid::new_v4();

        let mut session = DebugSession::new(
            agent_id,
            DebugSessionConfig::default(),
            snapshot_manager,
            event_log,
            navigator,
            comparator,
        )
        .await
        .unwrap();

        session.start().await?;

        // Create a breakpoint
        let breakpoint_id = session
            .create_breakpoint(
                crate::navigator::BreakpointCondition::OnEventType(
                    crate::event_log::EventType::StateChange,
                ),
                HashMap::new(),
            )
            .await
            .unwrap();

        // Verify breakpoint was created
        let breakpoints = session
            .navigator
            .get_agent_breakpoints(agent_id)
            .await
            .unwrap();
        assert!(breakpoints.iter().any(|bp| bp.id == breakpoint_id));
    }

    #[tokio::test]
    async fn test_session_comparisons() {
        let (snapshot_manager, event_log, navigator, comparator) = create_test_components().await;
        let agent_id = Uuid::new_v4();

        let mut session = DebugSession::new(
            agent_id,
            DebugSessionConfig::default(),
            snapshot_manager,
            event_log,
            navigator,
            comparator,
        )
        .await
        .unwrap();

        session.start().await?;

        // Create two snapshots
        let snapshot1_id = session
            .take_snapshot(json!({"health": 100}), 1024, HashMap::new())
            .await
            .unwrap();

        let snapshot2_id = session
            .take_snapshot(json!({"health": 80}), 1024, HashMap::new())
            .await
            .unwrap();

        // Compare snapshots
        let comparison = session
            .compare_snapshots(snapshot1_id, snapshot2_id, None)
            .await
            .unwrap();

        assert_eq!(comparison.from_snapshot_id, snapshot1_id);
        assert_eq!(comparison.to_snapshot_id, snapshot2_id);
        assert!(!comparison.changes.is_empty());
    }

    #[tokio::test]
    async fn test_session_custom_data() {
        let (snapshot_manager, event_log, navigator, comparator) = create_test_components().await;
        let agent_id = Uuid::new_v4();

        let session = DebugSession::new(
            agent_id,
            DebugSessionConfig::default(),
            snapshot_manager,
            event_log,
            navigator,
            comparator,
        )
        .await
        .unwrap();

        // Store custom data
        session
            .set_custom_data("test_key".to_string(), json!({"custom": "value"}))
            .await
            .unwrap();

        // Retrieve custom data
        let data = session.get_custom_data("test_key").await?;
        assert_eq!(data, Some(json!({"custom": "value"})));

        let missing_data = session.get_custom_data("missing_key").await?;
        assert_eq!(missing_data, None);
    }

    #[tokio::test]
    async fn test_session_statistics() {
        let (snapshot_manager, event_log, navigator, comparator) = create_test_components().await;
        let agent_id = Uuid::new_v4();

        let mut session = DebugSession::new(
            agent_id,
            DebugSessionConfig::default(),
            snapshot_manager,
            event_log,
            navigator,
            comparator,
        )
        .await
        .unwrap();

        session.start().await?;

        // Add some data
        session
            .take_snapshot(json!({}), 1024, HashMap::new())
            .await
            .unwrap();
        session
            .record_event(
                crate::event_log::EventType::StateChange,
                json!({}),
                HashMap::new(),
                None,
            )
            .await
            .unwrap();
        session
            .create_breakpoint(
                crate::navigator::BreakpointCondition::OnEventType(
                    crate::event_log::EventType::StateChange,
                ),
                HashMap::new(),
            )
            .await
            .unwrap();

        let stats = session.get_statistics().await?;
        assert_eq!(stats.session_id, session.metadata.id);
        assert_eq!(stats.agent_id, agent_id);
        assert_eq!(stats.snapshot_count, 1);
        assert_eq!(stats.total_events, 1);
        assert_eq!(stats.breakpoint_count, 3); // 1 + 2 auto breakpoints
    }

    #[tokio::test]
    async fn test_session_export_import() {
        let (snapshot_manager, event_log, navigator, comparator) = create_test_components().await;
        let agent_id = Uuid::new_v4();

        let mut session = DebugSession::new(
            agent_id,
            DebugSessionConfig::default(),
            Arc::clone(&snapshot_manager),
            Arc::clone(&event_log),
            Arc::clone(&navigator),
            Arc::clone(&comparator),
        )
        .await
        .unwrap();

        session.start().await?;
        session
            .update_metadata(
                Some("Export Test".to_string()),
                None,
                Some(vec!["export".to_string()]),
            )
            .await
            .unwrap();

        // Add some data
        session
            .take_snapshot(json!({"value": 42}), 1024, HashMap::new())
            .await
            .unwrap();
        session
            .record_event(
                crate::event_log::EventType::StateChange,
                json!({"change": "test"}),
                HashMap::new(),
                None,
            )
            .await
            .unwrap();

        // Export session
        let exported_data = session.export_session(ExportFormat::Json).await?;
        assert!(!exported_data.is_empty());

        // Import session
        let imported_session = DebugSession::import_session(
            &exported_data,
            ExportFormat::Json,
            snapshot_manager,
            event_log,
            navigator,
            comparator,
        )
        .await
        .unwrap();

        assert_eq!(imported_session.metadata.agent_id, agent_id);
        assert_eq!(
            imported_session.metadata.name,
            Some("Export Test".to_string())
        );
        assert_eq!(imported_session.metadata.tags, vec!["export"]);
    }

    #[tokio::test]
    async fn test_session_manager() {
        let (snapshot_manager, event_log, navigator, comparator) = create_test_components().await;

        let manager = SessionManager::new(
            snapshot_manager,
            event_log,
            navigator,
            comparator,
            DebugSessionConfig::default(),
        );

        let agent_id = Uuid::new_v4();

        // Create a session
        let session_id = manager.create_session(agent_id, None).await?;

        // Get the session
        let session = manager.get_session(session_id).await?;
        assert_eq!(session.metadata.agent_id, agent_id);

        // Get agent sessions
        let agent_sessions = manager.get_agent_sessions(agent_id).await;
        assert_eq!(agent_sessions.len(), 1);
        assert_eq!(agent_sessions[0].id, session_id);

        // Get all statistics
        let all_stats = manager.get_all_statistics().await;
        assert_eq!(all_stats.len(), 1);
        assert_eq!(all_stats[0].session_id, session_id);
    }

    #[tokio::test]
    async fn test_session_error_handling() {
        let (snapshot_manager, event_log, navigator, comparator) = create_test_components().await;
        let agent_id = Uuid::new_v4();

        let session = DebugSession::new(
            agent_id,
            DebugSessionConfig::default(),
            snapshot_manager,
            event_log,
            navigator,
            comparator,
        )
        .await
        .unwrap();

        // Try operations on inactive session
        let result = session.take_snapshot(json!({}), 1024, HashMap::new()).await;
        assert!(matches!(
            result,
            Err(TimeDebuggerError::ConcurrentAccess { .. })
        ));

        let result = session
            .record_event(
                crate::event_log::EventType::StateChange,
                json!({}),
                HashMap::new(),
                None,
            )
            .await;
        assert!(matches!(
            result,
            Err(TimeDebuggerError::ConcurrentAccess { .. })
        ));

        let result = session.navigate_to_time(Utc::now()).await;
        assert!(matches!(
            result,
            Err(TimeDebuggerError::ConcurrentAccess { .. })
        ));
    }

    #[tokio::test]
    async fn test_session_auto_breakpoints() {
        let (snapshot_manager, event_log, navigator, comparator) = create_test_components().await;
        let agent_id = Uuid::new_v4();

        let config = DebugSessionConfig {
            enable_auto_breakpoints: true,
            ..Default::default()
        };

        let mut session = DebugSession::new(
            agent_id,
            config,
            snapshot_manager,
            event_log,
            navigator,
            comparator,
        )
        .await
        .unwrap();

        session.start().await?;

        // Should have auto-breakpoints created
        let breakpoints = session
            .navigator
            .get_agent_breakpoints(agent_id)
            .await
            .unwrap();
        assert!(!breakpoints.is_empty());

        // Check for auto-breakpoints
        let auto_breakpoints: Vec<_> = breakpoints
            .iter()
            .filter(|bp| bp.metadata.get("auto") == Some(&"true".to_string()))
            .collect();
        assert!(!auto_breakpoints.is_empty());
    }
}
