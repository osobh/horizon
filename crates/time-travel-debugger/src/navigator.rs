use chrono::{DateTime, Utc};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::error::{Result, TimeDebuggerError};
use crate::event_log::{AgentEvent, EventLog, EventType};
use crate::snapshot::{SnapshotManager, StateSnapshot};

/// Represents a position in time for navigation
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TimePosition {
    pub timestamp: DateTime<Utc>,
    pub event_index: usize,
    pub snapshot_id: Option<Uuid>,
    pub agent_id: Uuid,
}

/// Represents a breakpoint in time-travel debugging
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Breakpoint {
    pub id: Uuid,
    pub agent_id: Uuid,
    pub condition: BreakpointCondition,
    pub is_enabled: bool,
    pub hit_count: usize,
    pub created_at: DateTime<Utc>,
    pub metadata: HashMap<String, String>,
}

/// Conditions that can trigger a breakpoint
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum BreakpointCondition {
    /// Break at specific timestamp
    AtTimestamp(DateTime<Utc>),
    /// Break at specific event index
    AtEventIndex(usize),
    /// Break when event type occurs
    OnEventType(EventType),
    /// Break when state field matches value
    OnStateCondition {
        field_path: String,
        expected_value: serde_json::Value,
    },
    /// Break when causality chain reaches certain length
    OnCausalityDepth(usize),
    /// Custom condition (JavaScript-like expression)
    Custom(String),
}

/// Navigation direction for stepping through time
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NavigationDirection {
    Forward,
    Backward,
}

/// Step size for navigation
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StepSize {
    Event,                  // Single event
    Snapshot,               // To next/previous snapshot
    Time(chrono::Duration), // Fixed time interval
    Custom(usize),          // Custom number of events
}

/// Navigation state and history
#[derive(Debug, Clone)]
pub struct NavigationState {
    pub current_position: TimePosition,
    pub history: Vec<TimePosition>,
    pub history_index: usize,
    pub max_history_size: usize,
}

impl NavigationState {
    pub fn new(initial_position: TimePosition, max_history_size: usize) -> Self {
        Self {
            current_position: initial_position.clone(),
            history: vec![initial_position],
            history_index: 0,
            max_history_size,
        }
    }

    pub fn can_go_back(&self) -> bool {
        self.history_index > 0
    }

    pub fn can_go_forward(&self) -> bool {
        self.history_index < self.history.len() - 1
    }

    pub fn add_position(&mut self, position: TimePosition) {
        // Remove forward history if we're not at the end
        if self.history_index < self.history.len() - 1 {
            self.history.truncate(self.history_index + 1);
        }

        // Add new position
        self.history.push(position.clone());
        self.history_index = self.history.len() - 1;
        self.current_position = position;

        // Enforce max history size
        if self.history.len() > self.max_history_size {
            let remove_count = self.history.len() - self.max_history_size;
            self.history.drain(0..remove_count);
            self.history_index = self.history_index.saturating_sub(remove_count);
        }
    }

    pub fn go_back(&mut self) -> Result<TimePosition> {
        if !self.can_go_back() {
            return Err(TimeDebuggerError::NavigationOutOfBounds {
                index: self.history_index.saturating_sub(1),
            });
        }

        self.history_index -= 1;
        self.current_position = self.history[self.history_index].clone();
        Ok(self.current_position.clone())
    }

    pub fn go_forward(&mut self) -> Result<TimePosition> {
        if !self.can_go_forward() {
            return Err(TimeDebuggerError::NavigationOutOfBounds {
                index: self.history_index + 1,
            });
        }

        self.history_index += 1;
        self.current_position = self.history[self.history_index].clone();
        Ok(self.current_position.clone())
    }
}

/// Time-travel navigation manager
pub struct TimeNavigator {
    snapshot_manager: Arc<SnapshotManager>,
    event_log: Arc<EventLog>,
    navigation_states: Arc<DashMap<Uuid, NavigationState>>, // session_id -> state (lock-free)
    breakpoints: Arc<DashMap<Uuid, Breakpoint>>, // breakpoint_id -> breakpoint (lock-free)
    agent_breakpoints: Arc<DashMap<Uuid, Vec<Uuid>>>, // agent_id -> breakpoint_ids (lock-free)
    max_navigation_history: usize,
}

impl TimeNavigator {
    pub fn new(
        snapshot_manager: Arc<SnapshotManager>,
        event_log: Arc<EventLog>,
        max_navigation_history: usize,
    ) -> Self {
        Self {
            snapshot_manager,
            event_log,
            navigation_states: Arc::new(DashMap::new()),
            breakpoints: Arc::new(DashMap::new()),
            agent_breakpoints: Arc::new(DashMap::new()),
            max_navigation_history,
        }
    }

    /// Initialize navigation for a session at a specific time
    pub async fn initialize_navigation(
        &self,
        session_id: Uuid,
        agent_id: Uuid,
        target_time: Option<DateTime<Utc>>,
    ) -> Result<TimePosition> {
        let target_time = target_time.unwrap_or_else(Utc::now);

        // Get events up to target time
        let events = self
            .event_log
            .get_agent_events(agent_id, None, Some(target_time))
            .await?;

        let event_index = events.len().saturating_sub(1);

        // Find closest snapshot
        let snapshot_id = match self
            .snapshot_manager
            .get_snapshot_at_time(agent_id, target_time)
            .await
        {
            Ok(snapshot) => Some(snapshot.id),
            Err(_) => None,
        };

        let position = TimePosition {
            timestamp: target_time,
            event_index,
            snapshot_id,
            agent_id,
        };

        let nav_state = NavigationState::new(position.clone(), self.max_navigation_history);

        self.navigation_states.insert(session_id, nav_state);

        Ok(position)
    }

    /// Navigate to a specific time position
    pub async fn navigate_to_time(
        &self,
        session_id: Uuid,
        target_time: DateTime<Utc>,
    ) -> Result<TimePosition> {
        let nav_state_ref = self
            .navigation_states
            .get(&session_id)
            .ok_or(TimeDebuggerError::SessionNotFound { id: session_id })?;
        let agent_id = nav_state_ref.current_position.agent_id;
        drop(nav_state_ref); // Release read lock before async operations

        // Get events up to target time
        let events = self
            .event_log
            .get_agent_events(agent_id, None, Some(target_time))
            .await?;

        let event_index = events.len().saturating_sub(1);

        // Find closest snapshot
        let snapshot_id = match self
            .snapshot_manager
            .get_snapshot_at_time(agent_id, target_time)
            .await
        {
            Ok(snapshot) => Some(snapshot.id),
            Err(_) => None,
        };

        let position = TimePosition {
            timestamp: target_time,
            event_index,
            snapshot_id,
            agent_id,
        };

        // Get mutable access to update navigation state
        let mut nav_state_mut = self
            .navigation_states
            .get_mut(&session_id)
            .ok_or(TimeDebuggerError::SessionNotFound { id: session_id })?;
        nav_state_mut.add_position(position.clone());
        Ok(position)
    }

    /// Navigate to a specific event index
    pub async fn navigate_to_event_index(
        &self,
        session_id: Uuid,
        target_index: usize,
    ) -> Result<TimePosition> {
        let nav_state_ref = self
            .navigation_states
            .get(&session_id)
            .ok_or(TimeDebuggerError::SessionNotFound { id: session_id })?;
        let agent_id = nav_state_ref.current_position.agent_id;
        drop(nav_state_ref); // Release read lock before async operations

        // Get all events for the agent
        let events = self
            .event_log
            .get_agent_events(agent_id, None, None)
            .await?;

        if target_index >= events.len() {
            return Err(TimeDebuggerError::NavigationOutOfBounds {
                index: target_index,
            });
        }

        let event = &events[target_index];
        let timestamp = event.timestamp;

        // Find closest snapshot
        let snapshot_id = match self
            .snapshot_manager
            .get_snapshot_at_time(agent_id, timestamp)
            .await
        {
            Ok(snapshot) => Some(snapshot.id),
            Err(_) => None,
        };

        let position = TimePosition {
            timestamp,
            event_index: target_index,
            snapshot_id,
            agent_id,
        };

        // Get mutable access to update navigation state
        let mut nav_state_mut = self
            .navigation_states
            .get_mut(&session_id)
            .ok_or(TimeDebuggerError::SessionNotFound { id: session_id })?;
        nav_state_mut.add_position(position.clone());
        Ok(position)
    }

    /// Step through time by a specified amount
    pub async fn step(
        &self,
        session_id: Uuid,
        direction: NavigationDirection,
        step_size: StepSize,
    ) -> Result<TimePosition> {
        let nav_state_ref = self
            .navigation_states
            .get(&session_id)
            .ok_or(TimeDebuggerError::SessionNotFound { id: session_id })?;
        let current_position = nav_state_ref.current_position.clone();
        drop(nav_state_ref);

        match step_size {
            StepSize::Event => {
                let new_index = match direction {
                    NavigationDirection::Forward => current_position.event_index + 1,
                    NavigationDirection::Backward => current_position.event_index.saturating_sub(1),
                };
                self.navigate_to_event_index(session_id, new_index).await
            }
            StepSize::Custom(count) => {
                let new_index = match direction {
                    NavigationDirection::Forward => current_position.event_index + count,
                    NavigationDirection::Backward => {
                        current_position.event_index.saturating_sub(count)
                    }
                };
                self.navigate_to_event_index(session_id, new_index).await
            }
            StepSize::Time(duration) => {
                let new_time = match direction {
                    NavigationDirection::Forward => current_position.timestamp + duration,
                    NavigationDirection::Backward => current_position.timestamp - duration,
                };
                self.navigate_to_time(session_id, new_time).await
            }
            StepSize::Snapshot => {
                let agent_id = current_position.agent_id;
                let snapshots = self.snapshot_manager.get_agent_snapshots(agent_id).await?;

                let current_snapshot_index =
                    if let Some(current_snapshot_id) = current_position.snapshot_id {
                        snapshots.iter().position(|s| s.id == current_snapshot_id)
                    } else {
                        None
                    };

                let new_snapshot = match (direction, current_snapshot_index) {
                    (NavigationDirection::Forward, Some(idx)) => snapshots.get(idx + 1),
                    (NavigationDirection::Backward, Some(idx)) => {
                        if idx > 0 {
                            snapshots.get(idx - 1)
                        } else {
                            None
                        }
                    }
                    (NavigationDirection::Forward, None) => snapshots.first(),
                    (NavigationDirection::Backward, None) => snapshots.last(),
                };

                if let Some(snapshot) = new_snapshot {
                    self.navigate_to_time(session_id, snapshot.timestamp).await
                } else {
                    Err(TimeDebuggerError::NavigationOutOfBounds {
                        index: current_position.event_index,
                    })
                }
            }
        }
    }

    /// Navigate through history (browser-like back/forward)
    pub async fn navigate_history(
        &self,
        session_id: Uuid,
        direction: NavigationDirection,
    ) -> Result<TimePosition> {
        let mut nav_state = self
            .navigation_states
            .get_mut(&session_id)
            .ok_or(TimeDebuggerError::SessionNotFound { id: session_id })?;

        match direction {
            NavigationDirection::Backward => nav_state.go_back(),
            NavigationDirection::Forward => nav_state.go_forward(),
        }
    }

    /// Get current navigation position
    pub async fn get_current_position(&self, session_id: Uuid) -> Result<TimePosition> {
        let nav_state = self
            .navigation_states
            .get(&session_id)
            .ok_or(TimeDebuggerError::SessionNotFound { id: session_id })?;

        Ok(nav_state.current_position.clone())
    }

    /// Get navigation history
    pub async fn get_navigation_history(&self, session_id: Uuid) -> Result<Vec<TimePosition>> {
        let nav_state = self
            .navigation_states
            .get(&session_id)
            .ok_or(TimeDebuggerError::SessionNotFound { id: session_id })?;

        Ok(nav_state.history.clone())
    }

    /// Create a breakpoint
    pub async fn create_breakpoint(
        &self,
        agent_id: Uuid,
        condition: BreakpointCondition,
        metadata: HashMap<String, String>,
    ) -> Result<Uuid> {
        let breakpoint_id = Uuid::new_v4();
        let breakpoint = Breakpoint {
            id: breakpoint_id,
            agent_id,
            condition,
            is_enabled: true,
            hit_count: 0,
            created_at: Utc::now(),
            metadata,
        };

        self.breakpoints.insert(breakpoint_id, breakpoint);
        self.agent_breakpoints
            .entry(agent_id)
            .or_insert_with(Vec::new)
            .push(breakpoint_id);

        Ok(breakpoint_id)
    }

    /// Remove a breakpoint
    pub async fn remove_breakpoint(&self, breakpoint_id: Uuid) -> Result<()> {
        let (_, breakpoint) = self
            .breakpoints
            .remove(&breakpoint_id)
            .ok_or(TimeDebuggerError::BreakpointNotFound { id: breakpoint_id })?;

        if let Some(mut bp_list) = self.agent_breakpoints.get_mut(&breakpoint.agent_id) {
            bp_list.retain(|&id| id != breakpoint_id);
        }

        Ok(())
    }

    /// Enable/disable a breakpoint
    pub async fn set_breakpoint_enabled(&self, breakpoint_id: Uuid, enabled: bool) -> Result<()> {
        let mut breakpoint = self
            .breakpoints
            .get_mut(&breakpoint_id)
            .ok_or(TimeDebuggerError::BreakpointNotFound { id: breakpoint_id })?;

        breakpoint.is_enabled = enabled;
        Ok(())
    }

    /// Get all breakpoints for an agent
    pub async fn get_agent_breakpoints(&self, agent_id: Uuid) -> Result<Vec<Breakpoint>> {
        let breakpoint_ids = self
            .agent_breakpoints
            .get(&agent_id)
            .map(|r| r.clone())
            .unwrap_or_default();

        let mut result = Vec::new();
        for id in breakpoint_ids {
            if let Some(breakpoint) = self.breakpoints.get(&id) {
                result.push(breakpoint.clone());
            }
        }

        Ok(result)
    }

    /// Check if any breakpoints should trigger at current position
    pub async fn check_breakpoints(
        &self,
        session_id: Uuid,
        event: &AgentEvent,
        state_snapshot: Option<&StateSnapshot>,
    ) -> Result<Vec<Uuid>> {
        let position = self.get_current_position(session_id).await?;
        let agent_breakpoints = self.get_agent_breakpoints(position.agent_id).await?;

        let mut triggered_breakpoints = Vec::new();

        for breakpoint in agent_breakpoints {
            if !breakpoint.is_enabled {
                continue;
            }

            let should_trigger = match &breakpoint.condition {
                BreakpointCondition::AtTimestamp(target_time) => {
                    (event.timestamp - *target_time).num_milliseconds().abs() < 1000
                    // Within 1 second
                }
                BreakpointCondition::AtEventIndex(target_index) => {
                    position.event_index == *target_index
                }
                BreakpointCondition::OnEventType(target_type) => event.event_type == *target_type,
                BreakpointCondition::OnStateCondition {
                    field_path,
                    expected_value,
                } => {
                    if let Some(snapshot) = state_snapshot {
                        Self::check_state_condition(
                            &snapshot.state_data,
                            field_path,
                            expected_value,
                        )
                    } else {
                        false
                    }
                }
                BreakpointCondition::OnCausalityDepth(target_depth) => {
                    if let Ok(chain) = self.event_log.get_causality_chain(event.id).await {
                        chain.len() >= *target_depth
                    } else {
                        false
                    }
                }
                BreakpointCondition::Custom(_expression) => {
                    // For now, custom conditions are not implemented
                    // In a real implementation, you'd use a JavaScript engine or similar
                    false
                }
            };

            if should_trigger {
                triggered_breakpoints.push(breakpoint.id);

                // Increment hit count
                if let Some(mut bp) = self.breakpoints.get_mut(&breakpoint.id) {
                    bp.hit_count += 1;
                }
            }
        }

        Ok(triggered_breakpoints)
    }

    /// Remove navigation state for a session
    pub async fn cleanup_session(&self, session_id: Uuid) -> Result<()> {
        self.navigation_states.remove(&session_id);
        Ok(())
    }

    // Private helper methods

    fn check_state_condition(
        state_data: &serde_json::Value,
        field_path: &str,
        expected_value: &serde_json::Value,
    ) -> bool {
        let path_parts: Vec<&str> = field_path.split('.').collect();
        let mut current_value = state_data;

        for part in path_parts {
            match current_value {
                serde_json::Value::Object(obj) => {
                    if let Some(value) = obj.get(part) {
                        current_value = value;
                    } else {
                        return false;
                    }
                }
                _ => return false,
            }
        }

        current_value == expected_value
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::snapshot::SnapshotConfig;
    use serde_json::json;

    async fn create_test_navigator() -> (TimeNavigator, Uuid) {
        let snapshot_config = SnapshotConfig::default();
        let snapshot_manager = Arc::new(SnapshotManager::new(snapshot_config));

        let event_log_config = crate::event_log::EventLogConfig::default();
        let event_log = Arc::new(EventLog::new(event_log_config));

        let navigator = TimeNavigator::new(snapshot_manager, event_log, 100);
        let agent_id = Uuid::new_v4();

        (navigator, agent_id)
    }

    #[tokio::test]
    async fn test_navigation_state() {
        let agent_id = Uuid::new_v4();
        let initial_position = TimePosition {
            timestamp: Utc::now(),
            event_index: 0,
            snapshot_id: None,
            agent_id,
        };

        let mut nav_state = NavigationState::new(initial_position.clone(), 10);

        assert!(!nav_state.can_go_back());
        assert!(!nav_state.can_go_forward());

        // Add a new position
        let new_position = TimePosition {
            timestamp: Utc::now(),
            event_index: 1,
            snapshot_id: None,
            agent_id,
        };
        nav_state.add_position(new_position);

        assert!(nav_state.can_go_back());
        assert!(!nav_state.can_go_forward());

        // Go back
        let back_position = nav_state.go_back()?;
        assert_eq!(back_position.event_index, 0);
        assert!(!nav_state.can_go_back());
        assert!(nav_state.can_go_forward());

        // Go forward
        let forward_position = nav_state.go_forward()?;
        assert_eq!(forward_position.event_index, 1);
        assert!(nav_state.can_go_back());
        assert!(!nav_state.can_go_forward());
    }

    #[tokio::test]
    async fn test_initialize_navigation() {
        let (navigator, agent_id) = create_test_navigator().await;
        let session_id = Uuid::new_v4();

        // Record some events first
        navigator
            .event_log
            .record_event(
                agent_id,
                EventType::StateChange,
                json!({"state": "initial"}),
                HashMap::new(),
                None,
            )
            .await
            .unwrap();

        let position = navigator
            .initialize_navigation(session_id, agent_id, None)
            .await
            .unwrap();

        assert_eq!(position.agent_id, agent_id);
        assert_eq!(position.event_index, 0);
    }

    #[tokio::test]
    async fn test_navigate_to_event_index() {
        let (navigator, agent_id) = create_test_navigator().await;
        let session_id = Uuid::new_v4();

        // Record multiple events
        for i in 0..5 {
            navigator
                .event_log
                .record_event(
                    agent_id,
                    EventType::StateChange,
                    json!({"step": i}),
                    HashMap::new(),
                    None,
                )
                .await
                .unwrap();
        }

        // Initialize navigation
        navigator
            .initialize_navigation(session_id, agent_id, None)
            .await
            .unwrap();

        // Navigate to specific index
        let position = navigator
            .navigate_to_event_index(session_id, 2)
            .await
            .unwrap();

        assert_eq!(position.event_index, 2);
        assert_eq!(position.agent_id, agent_id);
    }

    #[tokio::test]
    async fn test_step_navigation() {
        let (navigator, agent_id) = create_test_navigator().await;
        let session_id = Uuid::new_v4();

        // Record events
        for i in 0..5 {
            navigator
                .event_log
                .record_event(
                    agent_id,
                    EventType::StateChange,
                    json!({"step": i}),
                    HashMap::new(),
                    None,
                )
                .await
                .unwrap();
        }

        // Initialize at event 2
        navigator
            .initialize_navigation(session_id, agent_id, None)
            .await
            .unwrap();
        navigator
            .navigate_to_event_index(session_id, 2)
            .await
            .unwrap();

        // Step forward by 1 event
        let position = navigator
            .step(session_id, NavigationDirection::Forward, StepSize::Event)
            .await
            .unwrap();
        assert_eq!(position.event_index, 3);

        // Step backward by 2 events
        let position = navigator
            .step(
                session_id,
                NavigationDirection::Backward,
                StepSize::Custom(2),
            )
            .await
            .unwrap();
        assert_eq!(position.event_index, 1);
    }

    #[tokio::test]
    async fn test_navigate_history() {
        let (navigator, agent_id) = create_test_navigator().await;
        let session_id = Uuid::new_v4();

        // Record events
        for i in 0..3 {
            navigator
                .event_log
                .record_event(
                    agent_id,
                    EventType::StateChange,
                    json!({"step": i}),
                    HashMap::new(),
                    None,
                )
                .await
                .unwrap();
        }

        // Initialize navigation
        navigator
            .initialize_navigation(session_id, agent_id, None)
            .await
            .unwrap();

        // Navigate to different positions to build history
        navigator
            .navigate_to_event_index(session_id, 1)
            .await
            .unwrap();
        navigator
            .navigate_to_event_index(session_id, 2)
            .await
            .unwrap();

        // Use history navigation
        let position = navigator
            .navigate_history(session_id, NavigationDirection::Backward)
            .await
            .unwrap();
        assert_eq!(position.event_index, 1);

        let position = navigator
            .navigate_history(session_id, NavigationDirection::Backward)
            .await
            .unwrap();
        assert_eq!(position.event_index, 2); // Initial position
    }

    #[tokio::test]
    async fn test_breakpoint_management() {
        let (navigator, agent_id) = create_test_navigator().await;

        // Create breakpoint
        let breakpoint_id = navigator
            .create_breakpoint(
                agent_id,
                BreakpointCondition::OnEventType(EventType::StateChange),
                HashMap::new(),
            )
            .await
            .unwrap();

        // Get breakpoints
        let breakpoints = navigator.get_agent_breakpoints(agent_id).await?;
        assert_eq!(breakpoints.len(), 1);
        assert_eq!(breakpoints[0].id, breakpoint_id);
        assert!(breakpoints[0].is_enabled);

        // Disable breakpoint
        navigator
            .set_breakpoint_enabled(breakpoint_id, false)
            .await
            .unwrap();

        let breakpoints = navigator.get_agent_breakpoints(agent_id).await?;
        assert!(!breakpoints[0].is_enabled);

        // Remove breakpoint
        navigator.remove_breakpoint(breakpoint_id).await?;

        let breakpoints = navigator.get_agent_breakpoints(agent_id).await?;
        assert_eq!(breakpoints.len(), 0);
    }

    #[tokio::test]
    async fn test_breakpoint_triggering() {
        let (navigator, agent_id) = create_test_navigator().await;
        let session_id = Uuid::new_v4();

        // Create breakpoint for StateChange events
        navigator
            .create_breakpoint(
                agent_id,
                BreakpointCondition::OnEventType(EventType::StateChange),
                HashMap::new(),
            )
            .await
            .unwrap();

        // Initialize navigation
        navigator
            .initialize_navigation(session_id, agent_id, None)
            .await
            .unwrap();

        // Create event
        let event = crate::event_log::AgentEvent {
            id: Uuid::new_v4(),
            agent_id,
            timestamp: Utc::now(),
            event_type: EventType::StateChange,
            event_data: json!({}),
            metadata: HashMap::new(),
            causality_id: None,
        };

        // Check breakpoints
        let triggered = navigator
            .check_breakpoints(session_id, &event, None)
            .await
            .unwrap();

        assert_eq!(triggered.len(), 1);
    }

    #[tokio::test]
    async fn test_state_condition_breakpoint() {
        let (navigator, agent_id) = create_test_navigator().await;
        let session_id = Uuid::new_v4();

        // Create snapshot with state
        let state_data = json!({
            "health": 50,
            "position": {"x": 10, "y": 20}
        });

        let snapshot_id = navigator
            .snapshot_manager
            .create_snapshot(agent_id, state_data, 1024, HashMap::new())
            .await
            .unwrap();

        let snapshot = navigator
            .snapshot_manager
            .get_snapshot(snapshot_id)
            .await
            .unwrap();

        // Create breakpoint for health condition
        navigator
            .create_breakpoint(
                agent_id,
                BreakpointCondition::OnStateCondition {
                    field_path: "health".to_string(),
                    expected_value: json!(50),
                },
                HashMap::new(),
            )
            .await
            .unwrap();

        navigator
            .initialize_navigation(session_id, agent_id, None)
            .await
            .unwrap();

        let event = crate::event_log::AgentEvent {
            id: Uuid::new_v4(),
            agent_id,
            timestamp: Utc::now(),
            event_type: EventType::StateChange,
            event_data: json!({}),
            metadata: HashMap::new(),
            causality_id: None,
        };

        // Check breakpoint with matching state
        let triggered = navigator
            .check_breakpoints(session_id, &event, Some(&snapshot))
            .await
            .unwrap();

        assert_eq!(triggered.len(), 1);

        // Create breakpoint for nested field
        navigator
            .create_breakpoint(
                agent_id,
                BreakpointCondition::OnStateCondition {
                    field_path: "position.x".to_string(),
                    expected_value: json!(10),
                },
                HashMap::new(),
            )
            .await
            .unwrap();

        let triggered = navigator
            .check_breakpoints(session_id, &event, Some(&snapshot))
            .await
            .unwrap();

        assert_eq!(triggered.len(), 2); // Both breakpoints should trigger
    }

    #[tokio::test]
    async fn test_step_by_timestamp() {
        let (navigator, agent_id) = create_test_navigator().await;
        let session_id = Uuid::new_v4();

        let base_time = Utc::now();

        // Record events with known timestamps
        for i in 0..3 {
            let _event_time = base_time + chrono::Duration::minutes(i * 10);
            // Note: In a real implementation, you'd want to control event timestamps
            navigator
                .event_log
                .record_event(
                    agent_id,
                    EventType::StateChange,
                    json!({"step": i}),
                    HashMap::new(),
                    None,
                )
                .await
                .unwrap();
        }

        navigator
            .initialize_navigation(session_id, agent_id, Some(base_time))
            .await
            .unwrap();

        // Step forward by 5 minutes
        let position = navigator
            .step(
                session_id,
                NavigationDirection::Forward,
                StepSize::Time(chrono::Duration::minutes(5)),
            )
            .await
            .unwrap();

        // The position should be updated with the new timestamp
        assert!(position.timestamp > base_time);
    }

    #[tokio::test]
    async fn test_cleanup_session() {
        let (navigator, agent_id) = create_test_navigator().await;
        let session_id = Uuid::new_v4();

        // Initialize navigation
        navigator
            .initialize_navigation(session_id, agent_id, None)
            .await
            .unwrap();

        // Verify session exists
        assert!(navigator.get_current_position(session_id).await.is_ok());

        // Cleanup session
        navigator.cleanup_session(session_id).await?;

        // Verify session is gone
        assert!(matches!(
            navigator.get_current_position(session_id).await,
            Err(TimeDebuggerError::SessionNotFound { .. })
        ));
    }

    #[tokio::test]
    async fn test_navigation_bounds() {
        let (navigator, agent_id) = create_test_navigator().await;
        let session_id = Uuid::new_v4();

        // Record only one event
        navigator
            .event_log
            .record_event(
                agent_id,
                EventType::StateChange,
                json!({"step": 0}),
                HashMap::new(),
                None,
            )
            .await
            .unwrap();

        navigator
            .initialize_navigation(session_id, agent_id, None)
            .await
            .unwrap();

        // Try to navigate to out-of-bounds index
        let result = navigator.navigate_to_event_index(session_id, 10).await;

        assert!(matches!(
            result,
            Err(TimeDebuggerError::NavigationOutOfBounds { .. })
        ));

        // Try to step backward from the beginning
        let result = navigator
            .step(session_id, NavigationDirection::Backward, StepSize::Event)
            .await;

        // Should handle gracefully (might succeed with index 0 due to saturating_sub)
        assert!(result.is_ok());
    }
}
