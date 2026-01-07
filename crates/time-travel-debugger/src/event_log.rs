use chrono::{DateTime, Utc};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::error::{Result, TimeDebuggerError};

/// Represents an event in the agent execution timeline
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AgentEvent {
    pub id: Uuid,
    pub agent_id: Uuid,
    pub timestamp: DateTime<Utc>,
    pub event_type: EventType,
    pub event_data: serde_json::Value,
    pub metadata: HashMap<String, String>,
    pub causality_id: Option<Uuid>, // Links to causing event
}

/// Types of events that can occur during agent execution
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum EventType {
    StateChange,
    ActionExecution,
    MessageReceived,
    MessageSent,
    ErrorOccurred,
    DecisionMade,
    MemoryUpdate,
    NetworkActivity,
    Custom(String),
}

/// Configuration for event logging
#[derive(Debug, Clone)]
pub struct EventLogConfig {
    pub max_events_per_agent: usize,
    pub enable_causality_tracking: bool,
    pub enable_compression: bool,
    pub flush_interval: std::time::Duration,
    pub enable_persistence: bool,
}

impl Default for EventLogConfig {
    fn default() -> Self {
        Self {
            max_events_per_agent: 10000,
            enable_causality_tracking: true,
            enable_compression: false,
            flush_interval: std::time::Duration::from_secs(30),
            enable_persistence: false,
        }
    }
}

/// Manages event logging and replay for time-travel debugging
pub struct EventLog {
    events: Arc<DashMap<Uuid, VecDeque<AgentEvent>>>, // agent_id -> events
    event_index: Arc<DashMap<Uuid, AgentEvent>>,      // event_id -> event
    causality_graph: Arc<DashMap<Uuid, Vec<Uuid>>>,   // event_id -> caused_events
    config: EventLogConfig,
    flush_task: Arc<RwLock<Option<tokio::task::JoinHandle<()>>>>,
}

impl EventLog {
    pub fn new(config: EventLogConfig) -> Self {
        Self {
            events: Arc::new(DashMap::new()),
            event_index: Arc::new(DashMap::new()),
            causality_graph: Arc::new(DashMap::new()),
            config,
            flush_task: Arc::new(RwLock::new(None)),
        }
    }

    /// Record a new event
    pub async fn record_event(
        &self,
        agent_id: Uuid,
        event_type: EventType,
        event_data: serde_json::Value,
        metadata: HashMap<String, String>,
        causality_id: Option<Uuid>,
    ) -> Result<Uuid> {
        let event_id = Uuid::new_v4();
        let timestamp = Utc::now();

        let event = AgentEvent {
            id: event_id,
            agent_id,
            timestamp,
            event_type,
            event_data,
            metadata,
            causality_id,
        };

        // Add to event index
        self.event_index.insert(event_id, event.clone());

        // Add to agent events
        let mut agent_events = self.events.entry(agent_id).or_insert_with(VecDeque::new);
        agent_events.push_back(event);

        // Enforce max events limit
        if agent_events.len() > self.config.max_events_per_agent {
            if let Some(removed_event) = agent_events.pop_front() {
                self.event_index.remove(&removed_event.id);
                // Clean up causality graph
                self.causality_graph.remove(&removed_event.id);
            }
        }

        // Update causality graph if enabled
        if self.config.enable_causality_tracking {
            if let Some(cause_id) = causality_id {
                self.causality_graph
                    .entry(cause_id)
                    .or_insert_with(Vec::new)
                    .push(event_id);
            }
        }

        Ok(event_id)
    }

    /// Get an event by ID
    pub async fn get_event(&self, event_id: Uuid) -> Result<AgentEvent> {
        self.event_index
            .get(&event_id)
            .map(|entry| entry.clone())
            .ok_or(TimeDebuggerError::EmptyEventLog)
    }

    /// Get all events for an agent, optionally filtered by time range
    pub async fn get_agent_events(
        &self,
        agent_id: Uuid,
        start_time: Option<DateTime<Utc>>,
        end_time: Option<DateTime<Utc>>,
    ) -> Result<Vec<AgentEvent>> {
        let events = self
            .events
            .get(&agent_id)
            .map(|events| events.clone())
            .unwrap_or_default();

        let filtered_events: Vec<AgentEvent> = events
            .into_iter()
            .filter(|event| {
                if let Some(start) = start_time {
                    if event.timestamp < start {
                        return false;
                    }
                }
                if let Some(end) = end_time {
                    if event.timestamp > end {
                        return false;
                    }
                }
                true
            })
            .collect();

        Ok(filtered_events)
    }

    /// Get events by type for an agent
    pub async fn get_events_by_type(
        &self,
        agent_id: Uuid,
        event_type: EventType,
    ) -> Result<Vec<AgentEvent>> {
        let events = self.get_agent_events(agent_id, None, None).await?;

        let filtered_events: Vec<AgentEvent> = events
            .into_iter()
            .filter(|event| event.event_type == event_type)
            .collect();

        Ok(filtered_events)
    }

    /// Get causality chain for an event
    pub async fn get_causality_chain(&self, event_id: Uuid) -> Result<Vec<AgentEvent>> {
        if !self.config.enable_causality_tracking {
            return Ok(vec![]);
        }

        let mut chain = Vec::new();
        let mut current_id = Some(event_id);

        while let Some(id) = current_id {
            if let Some(event) = self.event_index.get(&id) {
                chain.push(event.clone());
                current_id = event.causality_id;
            } else {
                break;
            }
        }

        // Reverse to get chronological order
        chain.reverse();
        Ok(chain)
    }

    /// Get events caused by a specific event
    pub async fn get_caused_events(&self, event_id: Uuid) -> Result<Vec<AgentEvent>> {
        if !self.config.enable_causality_tracking {
            return Ok(vec![]);
        }

        let caused_ids = self
            .causality_graph
            .get(&event_id)
            .map(|ids| ids.clone())
            .unwrap_or_default();

        let mut caused_events = Vec::new();
        for id in caused_ids {
            if let Some(event) = self.event_index.get(&id) {
                caused_events.push(event.clone());
            }
        }

        // Sort by timestamp
        caused_events.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));
        Ok(caused_events)
    }

    /// Replay events from a specific point in time
    pub async fn replay_from_time(
        &self,
        agent_id: Uuid,
        start_time: DateTime<Utc>,
        mut callback: impl FnMut(&AgentEvent) -> Result<()>,
    ) -> Result<usize> {
        let events = self
            .get_agent_events(agent_id, Some(start_time), None)
            .await?;

        let mut replayed_count = 0;
        for event in events {
            callback(&event).map_err(|e| TimeDebuggerError::ReplayFailed {
                index: replayed_count,
                reason: e.to_string(),
            })?;
            replayed_count += 1;
        }

        Ok(replayed_count)
    }

    /// Replay events up to a specific point in time
    pub async fn replay_until_time(
        &self,
        agent_id: Uuid,
        end_time: DateTime<Utc>,
        mut callback: impl FnMut(&AgentEvent) -> Result<()>,
    ) -> Result<usize> {
        let events = self
            .get_agent_events(agent_id, None, Some(end_time))
            .await?;

        let mut replayed_count = 0;
        for event in events {
            callback(&event).map_err(|e| TimeDebuggerError::ReplayFailed {
                index: replayed_count,
                reason: e.to_string(),
            })?;
            replayed_count += 1;
        }

        Ok(replayed_count)
    }

    /// Replay a specific number of events from an index
    pub async fn replay_events(
        &self,
        agent_id: Uuid,
        start_index: usize,
        count: usize,
        mut callback: impl FnMut(&AgentEvent) -> Result<()>,
    ) -> Result<usize> {
        let events = self.get_agent_events(agent_id, None, None).await?;

        if start_index >= events.len() {
            return Err(TimeDebuggerError::NavigationOutOfBounds { index: start_index });
        }

        let end_index = std::cmp::min(start_index + count, events.len());
        let mut replayed_count = 0;

        for (i, event) in events.iter().enumerate() {
            if i >= start_index && i < end_index {
                callback(event).map_err(|e| TimeDebuggerError::ReplayFailed {
                    index: i,
                    reason: e.to_string(),
                })?;
                replayed_count += 1;
            }
        }

        Ok(replayed_count)
    }

    /// Clear all events for an agent
    pub async fn clear_agent_events(&self, agent_id: Uuid) -> Result<usize> {
        let removed_count = self
            .events
            .get(&agent_id)
            .map(|events| events.len())
            .unwrap_or(0);

        // Remove from event index
        if let Some(events) = self.events.get(&agent_id) {
            for event in events.iter() {
                self.event_index.remove(&event.id);
                self.causality_graph.remove(&event.id);
            }
        }

        // Remove agent events
        self.events.remove(&agent_id);

        Ok(removed_count)
    }

    /// Get event statistics
    pub async fn get_statistics(&self) -> HashMap<String, serde_json::Value> {
        let mut stats = HashMap::new();

        let total_agents = self.events.len();
        let total_events: usize = self.events.iter().map(|entry| entry.value().len()).sum();

        let mut event_type_counts: HashMap<String, usize> = HashMap::new();
        for entry in self.event_index.iter() {
            let event = entry.value();
            let type_name = match &event.event_type {
                EventType::Custom(name) => name.clone(),
                other => format!("{:?}", other),
            };
            *event_type_counts.entry(type_name).or_insert(0) += 1;
        }

        stats.insert(
            "total_agents".to_string(),
            serde_json::Value::Number(total_agents.into()),
        );
        stats.insert(
            "total_events".to_string(),
            serde_json::Value::Number(total_events.into()),
        );
        stats.insert(
            "event_type_counts".to_string(),
            serde_json::to_value(event_type_counts).unwrap_or_default(),
        );

        if self.config.enable_causality_tracking {
            let causality_chains = self.causality_graph.len();
            stats.insert(
                "causality_chains".to_string(),
                serde_json::Value::Number(causality_chains.into()),
            );
        }

        stats
    }

    /// Export events to JSON
    pub async fn export_events(&self, agent_id: Uuid) -> Result<String> {
        let events = self.get_agent_events(agent_id, None, None).await?;
        serde_json::to_string_pretty(&events).map_err(|e| e.into())
    }

    /// Import events from JSON
    pub async fn import_events(&self, agent_id: Uuid, json_data: &str) -> Result<usize> {
        let events: Vec<AgentEvent> = serde_json::from_str(json_data)?;

        // Clear existing events first
        self.clear_agent_events(agent_id).await?;

        let mut imported_count = 0;
        for event in events {
            // Verify agent_id matches
            if event.agent_id != agent_id {
                return Err(TimeDebuggerError::ReplayFailed {
                    index: imported_count,
                    reason: "Agent ID mismatch".to_string(),
                });
            }

            // Add to event structures
            self.event_index.insert(event.id, event.clone());

            let mut agent_events = self.events.entry(agent_id).or_insert_with(VecDeque::new);
            agent_events.push_back(event.clone());

            // Update causality graph
            if self.config.enable_causality_tracking {
                if let Some(cause_id) = event.causality_id {
                    self.causality_graph
                        .entry(cause_id)
                        .or_insert_with(Vec::new)
                        .push(event.id);
                }
            }

            imported_count += 1;
        }

        Ok(imported_count)
    }

    /// Start periodic flush task
    pub async fn start_flush_task(&self) {
        let mut flush_task = self.flush_task.write().await;
        if flush_task.is_some() {
            return; // Already running
        }

        let events = Arc::clone(&self.events);
        let config = self.config.clone();

        let task = tokio::spawn(async move {
            let mut interval = tokio::time::interval(config.flush_interval);
            loop {
                interval.tick().await;
                Self::flush_events(&events, &config).await;
            }
        });

        *flush_task = Some(task);
    }

    /// Stop the flush task
    pub async fn stop_flush_task(&self) {
        let mut flush_task = self.flush_task.write().await;
        if let Some(task) = flush_task.take() {
            task.abort();
        }
    }

    // Private helper methods

    async fn flush_events(_events: &DashMap<Uuid, VecDeque<AgentEvent>>, _config: &EventLogConfig) {
        // Implementation would flush events to persistent storage
        // For now, this is a placeholder
        tracing::debug!("Flushing events to persistent storage");
    }
}

impl Drop for EventLog {
    fn drop(&mut self) {
        // Cancel flush task when dropping
        if let Ok(mut task) = self.flush_task.try_write() {
            if let Some(handle) = task.take() {
                handle.abort();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn create_test_event_log() -> EventLog {
        let config = EventLogConfig {
            max_events_per_agent: 5,
            enable_causality_tracking: true,
            enable_compression: false,
            flush_interval: std::time::Duration::from_millis(100),
            enable_persistence: false,
        };
        EventLog::new(config)
    }

    #[tokio::test]
    async fn test_record_and_get_event() {
        let event_log = create_test_event_log();
        let agent_id = Uuid::new_v4();
        let event_data = json!({"action": "move", "direction": "north"});
        let metadata = HashMap::new();

        let event_id = event_log
            .record_event(
                agent_id,
                EventType::ActionExecution,
                event_data.clone(),
                metadata,
                None,
            )
            .await
            .unwrap();

        let retrieved_event = event_log.get_event(event_id).await?;
        assert_eq!(retrieved_event.agent_id, agent_id);
        assert_eq!(retrieved_event.event_type, EventType::ActionExecution);
        assert_eq!(retrieved_event.event_data, event_data);
    }

    #[tokio::test]
    async fn test_get_agent_events() {
        let event_log = create_test_event_log();
        let agent_id = Uuid::new_v4();

        // Record multiple events
        for i in 0..3 {
            event_log
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

        let events = event_log
            .get_agent_events(agent_id, None, None)
            .await
            .unwrap();
        assert_eq!(events.len(), 3);

        // Events should be in chronological order
        for i in 0..2 {
            assert!(events[i].timestamp <= events[i + 1].timestamp);
        }
    }

    #[tokio::test]
    async fn test_event_limit_enforcement() {
        let event_log = create_test_event_log();
        let agent_id = Uuid::new_v4();

        // Record more events than the limit
        let mut event_ids = Vec::new();
        for i in 0..7 {
            let event_id = event_log
                .record_event(
                    agent_id,
                    EventType::StateChange,
                    json!({"step": i}),
                    HashMap::new(),
                    None,
                )
                .await
                .unwrap();
            event_ids.push(event_id);
        }

        let events = event_log
            .get_agent_events(agent_id, None, None)
            .await
            .unwrap();
        assert_eq!(events.len(), 5); // Should be limited to max_events_per_agent

        // First events should be removed
        assert!(event_log.get_event(event_ids[0]).await.is_err());
        assert!(event_log.get_event(event_ids[1]).await.is_err());

        // Last events should still exist
        assert!(event_log.get_event(event_ids[6]).await.is_ok());
    }

    #[tokio::test]
    async fn test_events_by_type() {
        let event_log = create_test_event_log();
        let agent_id = Uuid::new_v4();

        // Record different types of events
        event_log
            .record_event(
                agent_id,
                EventType::StateChange,
                json!({"state": "idle"}),
                HashMap::new(),
                None,
            )
            .await
            .unwrap();

        event_log
            .record_event(
                agent_id,
                EventType::ActionExecution,
                json!({"action": "move"}),
                HashMap::new(),
                None,
            )
            .await
            .unwrap();

        event_log
            .record_event(
                agent_id,
                EventType::StateChange,
                json!({"state": "moving"}),
                HashMap::new(),
                None,
            )
            .await
            .unwrap();

        let state_changes = event_log
            .get_events_by_type(agent_id, EventType::StateChange)
            .await
            .unwrap();
        assert_eq!(state_changes.len(), 2);

        let actions = event_log
            .get_events_by_type(agent_id, EventType::ActionExecution)
            .await
            .unwrap();
        assert_eq!(actions.len(), 1);
    }

    #[tokio::test]
    async fn test_time_range_filtering() {
        let event_log = create_test_event_log();
        let agent_id = Uuid::new_v4();

        let start_time = Utc::now();

        // Record first event
        event_log
            .record_event(
                agent_id,
                EventType::StateChange,
                json!({"step": 1}),
                HashMap::new(),
                None,
            )
            .await
            .unwrap();

        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        let mid_time = Utc::now();

        // Record second event
        event_log
            .record_event(
                agent_id,
                EventType::StateChange,
                json!({"step": 2}),
                HashMap::new(),
                None,
            )
            .await
            .unwrap();

        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        let end_time = Utc::now();

        // Test filtering
        let all_events = event_log
            .get_agent_events(agent_id, None, None)
            .await
            .unwrap();
        assert_eq!(all_events.len(), 2);

        let events_from_mid = event_log
            .get_agent_events(agent_id, Some(mid_time), None)
            .await
            .unwrap();
        assert_eq!(events_from_mid.len(), 1);

        let events_until_mid = event_log
            .get_agent_events(agent_id, None, Some(mid_time))
            .await
            .unwrap();
        assert_eq!(events_until_mid.len(), 1);

        let events_in_range = event_log
            .get_agent_events(agent_id, Some(start_time), Some(end_time))
            .await
            .unwrap();
        assert_eq!(events_in_range.len(), 2);
    }

    #[tokio::test]
    async fn test_causality_tracking() {
        let event_log = create_test_event_log();
        let agent_id = Uuid::new_v4();

        // Record root event
        let root_event_id = event_log
            .record_event(
                agent_id,
                EventType::MessageReceived,
                json!({"message": "start"}),
                HashMap::new(),
                None,
            )
            .await
            .unwrap();

        // Record caused event
        let caused_event_id = event_log
            .record_event(
                agent_id,
                EventType::DecisionMade,
                json!({"decision": "process"}),
                HashMap::new(),
                Some(root_event_id),
            )
            .await
            .unwrap();

        // Record another caused event
        let _final_event_id = event_log
            .record_event(
                agent_id,
                EventType::ActionExecution,
                json!({"action": "execute"}),
                HashMap::new(),
                Some(caused_event_id),
            )
            .await
            .unwrap();

        // Test causality chain
        let chain = event_log
            .get_causality_chain(caused_event_id)
            .await
            .unwrap();
        assert_eq!(chain.len(), 2); // Root + caused event

        // Test caused events
        let caused_events = event_log.get_caused_events(root_event_id).await?;
        assert_eq!(caused_events.len(), 1);
        assert_eq!(caused_events[0].id, caused_event_id);
    }

    #[tokio::test]
    async fn test_replay_from_time() {
        let event_log = create_test_event_log();
        let agent_id = Uuid::new_v4();

        // Record events
        event_log
            .record_event(
                agent_id,
                EventType::StateChange,
                json!({"step": 1}),
                HashMap::new(),
                None,
            )
            .await
            .unwrap();

        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        let replay_time = Utc::now();

        event_log
            .record_event(
                agent_id,
                EventType::StateChange,
                json!({"step": 2}),
                HashMap::new(),
                None,
            )
            .await
            .unwrap();

        event_log
            .record_event(
                agent_id,
                EventType::StateChange,
                json!({"step": 3}),
                HashMap::new(),
                None,
            )
            .await
            .unwrap();

        // Replay from middle time
        let mut replayed_events = Vec::new();
        let count = event_log
            .replay_from_time(agent_id, replay_time, |event| {
                replayed_events.push(event.clone());
                Ok(())
            })
            .await
            .unwrap();

        assert_eq!(count, 2); // Should replay last 2 events
        assert_eq!(replayed_events.len(), 2);
    }

    #[tokio::test]
    async fn test_replay_until_time() {
        let event_log = create_test_event_log();
        let agent_id = Uuid::new_v4();

        // Record events
        event_log
            .record_event(
                agent_id,
                EventType::StateChange,
                json!({"step": 1}),
                HashMap::new(),
                None,
            )
            .await
            .unwrap();

        event_log
            .record_event(
                agent_id,
                EventType::StateChange,
                json!({"step": 2}),
                HashMap::new(),
                None,
            )
            .await
            .unwrap();

        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        let cutoff_time = Utc::now();

        event_log
            .record_event(
                agent_id,
                EventType::StateChange,
                json!({"step": 3}),
                HashMap::new(),
                None,
            )
            .await
            .unwrap();

        // Replay until cutoff time
        let mut replayed_events = Vec::new();
        let count = event_log
            .replay_until_time(agent_id, cutoff_time, |event| {
                replayed_events.push(event.clone());
                Ok(())
            })
            .await
            .unwrap();

        assert_eq!(count, 2); // Should replay first 2 events
        assert_eq!(replayed_events.len(), 2);
    }

    #[tokio::test]
    async fn test_replay_events_by_index() {
        let event_log = create_test_event_log();
        let agent_id = Uuid::new_v4();

        // Record events
        for i in 0..5 {
            event_log
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

        // Replay events 1-3 (0-indexed)
        let mut replayed_events = Vec::new();
        let count = event_log
            .replay_events(agent_id, 1, 3, |event| {
                replayed_events.push(event.clone());
                Ok(())
            })
            .await
            .unwrap();

        assert_eq!(count, 3);
        assert_eq!(replayed_events.len(), 3);

        // Verify correct events were replayed
        for (i, event) in replayed_events.iter().enumerate() {
            let expected_step = i + 1; // Events 1, 2, 3
            assert_eq!(event.event_data["step"], expected_step);
        }
    }

    #[tokio::test]
    async fn test_clear_agent_events() {
        let event_log = create_test_event_log();
        let agent_id = Uuid::new_v4();

        // Record events
        for i in 0..3 {
            event_log
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

        // Verify events exist
        let events = event_log
            .get_agent_events(agent_id, None, None)
            .await
            .unwrap();
        assert_eq!(events.len(), 3);

        // Clear events
        let cleared_count = event_log.clear_agent_events(agent_id).await?;
        assert_eq!(cleared_count, 3);

        // Verify events are gone
        let events = event_log
            .get_agent_events(agent_id, None, None)
            .await
            .unwrap();
        assert_eq!(events.len(), 0);
    }

    #[tokio::test]
    async fn test_export_import_events() {
        let event_log = create_test_event_log();
        let agent_id = Uuid::new_v4();

        // Record events
        for i in 0..3 {
            event_log
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

        // Export events
        let exported_json = event_log.export_events(agent_id).await?;
        assert!(!exported_json.is_empty());

        // Clear events
        event_log.clear_agent_events(agent_id).await?;

        // Import events
        let imported_count = event_log
            .import_events(agent_id, &exported_json)
            .await
            .unwrap();
        assert_eq!(imported_count, 3);

        // Verify events are restored
        let events = event_log
            .get_agent_events(agent_id, None, None)
            .await
            .unwrap();
        assert_eq!(events.len(), 3);
    }

    #[tokio::test]
    async fn test_statistics() {
        let event_log = create_test_event_log();
        let agent1_id = Uuid::new_v4();
        let agent2_id = Uuid::new_v4();

        // Record different types of events
        event_log
            .record_event(
                agent1_id,
                EventType::StateChange,
                json!({}),
                HashMap::new(),
                None,
            )
            .await
            .unwrap();

        event_log
            .record_event(
                agent1_id,
                EventType::ActionExecution,
                json!({}),
                HashMap::new(),
                None,
            )
            .await
            .unwrap();

        event_log
            .record_event(
                agent2_id,
                EventType::StateChange,
                json!({}),
                HashMap::new(),
                None,
            )
            .await
            .unwrap();

        let stats = event_log.get_statistics().await;

        assert_eq!(stats["total_agents"], serde_json::Value::Number(2.into()));
        assert_eq!(stats["total_events"], serde_json::Value::Number(3.into()));

        let event_type_counts = stats["event_type_counts"].as_object()?;
        assert_eq!(
            event_type_counts["StateChange"],
            serde_json::Value::Number(2.into())
        );
        assert_eq!(
            event_type_counts["ActionExecution"],
            serde_json::Value::Number(1.into())
        );
    }

    #[tokio::test]
    async fn test_replay_error_handling() {
        let event_log = create_test_event_log();
        let agent_id = Uuid::new_v4();

        // Record an event
        event_log
            .record_event(
                agent_id,
                EventType::StateChange,
                json!({"step": 1}),
                HashMap::new(),
                None,
            )
            .await
            .unwrap();

        // Test replay with error in callback
        let result = event_log
            .replay_from_time(
                agent_id,
                Utc::now() - chrono::Duration::hours(1),
                |_event| {
                    Err(TimeDebuggerError::ReplayFailed {
                        index: 0,
                        reason: "Test error".to_string(),
                    })
                },
            )
            .await;

        assert!(matches!(
            result,
            Err(TimeDebuggerError::ReplayFailed { .. })
        ));
    }
}
