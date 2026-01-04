//! Temporal indexing and analysis for causal inference

use std::collections::{HashMap, VecDeque};

use super::types::TemporalEvent;

/// Temporal index for efficient causal queries
pub struct TemporalIndex {
    pub time_ordered_events: VecDeque<TemporalEvent>,
    pub event_to_nodes: HashMap<String, Vec<String>>,
    pub node_timelines: HashMap<String, Vec<TemporalEvent>>,
}

impl Default for TemporalIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl TemporalIndex {
    pub fn new() -> Self {
        Self {
            time_ordered_events: VecDeque::new(),
            event_to_nodes: HashMap::new(),
            node_timelines: HashMap::new(),
        }
    }

    pub fn add_event(&mut self, event: TemporalEvent) {
        // Insert event in temporal order
        let mut insert_pos = self.time_ordered_events.len();
        for (i, existing_event) in self.time_ordered_events.iter().enumerate().rev() {
            if existing_event.timestamp <= event.timestamp {
                insert_pos = i + 1;
                break;
            }
            if i == 0 {
                insert_pos = 0;
            }
        }

        self.time_ordered_events.insert(insert_pos, event.clone());

        // Update event to nodes mapping
        self.event_to_nodes
            .entry(event.event_id.clone())
            .or_default()
            .push(event.node_id.clone());

        // Update node timelines
        self.node_timelines
            .entry(event.node_id.clone())
            .or_default()
            .push(event);
    }

    pub fn get_events_for_node(&self, node_id: &str) -> Option<&Vec<TemporalEvent>> {
        self.node_timelines.get(node_id)
    }

    pub fn get_events_in_range(
        &self,
        start_time: chrono::DateTime<chrono::Utc>,
        end_time: chrono::DateTime<chrono::Utc>,
    ) -> Vec<&TemporalEvent> {
        self.time_ordered_events
            .iter()
            .filter(|event| event.timestamp >= start_time && event.timestamp <= end_time)
            .collect()
    }

    pub fn clear_old_events(&mut self, before_time: chrono::DateTime<chrono::Utc>) {
        while let Some(front_event) = self.time_ordered_events.front() {
            if front_event.timestamp < before_time {
                let removed_event = self.time_ordered_events.pop_front().unwrap();

                // Clean up mappings
                if let Some(nodes) = self.event_to_nodes.get_mut(&removed_event.event_id) {
                    nodes.retain(|node_id| node_id != &removed_event.node_id);
                    if nodes.is_empty() {
                        self.event_to_nodes.remove(&removed_event.event_id);
                    }
                }

                if let Some(timeline) = self.node_timelines.get_mut(&removed_event.node_id) {
                    timeline.retain(|event| event.event_id != removed_event.event_id);
                    if timeline.is_empty() {
                        self.node_timelines.remove(&removed_event.node_id);
                    }
                }
            } else {
                break;
            }
        }
    }

    pub fn event_count(&self) -> usize {
        self.time_ordered_events.len()
    }
}
