//! Query functionality for audit logs

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::audit::types::{AuditEventType, AuditOutcome, AuditSeverity};

/// Query parameters for audit log searches
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditQuery {
    /// Time range start
    pub start_time: Option<DateTime<Utc>>,
    /// Time range end
    pub end_time: Option<DateTime<Utc>>,
    /// Filter by event types
    pub event_types: Vec<AuditEventType>,
    /// Filter by severity levels
    pub severity_levels: Vec<AuditSeverity>,
    /// Filter by outcomes
    pub outcomes: Vec<AuditOutcome>,
    /// Filter by actor
    pub actor: Option<String>,
    /// Filter by target
    pub target: Option<String>,
    /// Search in description
    pub description_contains: Option<String>,
    /// Filter by correlation ID
    pub correlation_id: Option<Uuid>,
    /// Maximum results to return
    pub limit: Option<usize>,
    /// Offset for pagination
    pub offset: Option<usize>,
    /// Sort order
    pub sort_order: SortOrder,
}

/// Sort order for query results
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SortOrder {
    /// Sort by timestamp ascending (oldest first)
    TimestampAsc,
    /// Sort by timestamp descending (newest first)
    TimestampDesc,
    /// Sort by severity ascending
    SeverityAsc,
    /// Sort by severity descending
    SeverityDesc,
}

impl AuditQuery {
    /// Create a new empty query
    pub fn new() -> Self {
        Self {
            start_time: None,
            end_time: None,
            event_types: Vec::new(),
            severity_levels: Vec::new(),
            outcomes: Vec::new(),
            actor: None,
            target: None,
            description_contains: None,
            correlation_id: None,
            limit: None,
            offset: None,
            sort_order: SortOrder::TimestampDesc,
        }
    }

    /// Set time range
    pub fn with_time_range(mut self, start: DateTime<Utc>, end: DateTime<Utc>) -> Self {
        self.start_time = Some(start);
        self.end_time = Some(end);
        self
    }

    /// Filter by event type
    pub fn with_event_type(mut self, event_type: AuditEventType) -> Self {
        self.event_types.push(event_type);
        self
    }

    /// Filter by severity
    pub fn with_severity(mut self, severity: AuditSeverity) -> Self {
        self.severity_levels.push(severity);
        self
    }

    /// Filter by actor
    pub fn with_actor(mut self, actor: String) -> Self {
        self.actor = Some(actor);
        self
    }

    /// Filter by target
    pub fn with_target(mut self, target: String) -> Self {
        self.target = Some(target);
        self
    }

    /// Set pagination
    pub fn with_pagination(mut self, limit: usize, offset: usize) -> Self {
        self.limit = Some(limit);
        self.offset = Some(offset);
        self
    }

    /// Set sort order
    pub fn with_sort(mut self, sort_order: SortOrder) -> Self {
        self.sort_order = sort_order;
        self
    }

    /// Check if query has any filters
    pub fn has_filters(&self) -> bool {
        self.start_time.is_some()
            || self.end_time.is_some()
            || !self.event_types.is_empty()
            || !self.severity_levels.is_empty()
            || !self.outcomes.is_empty()
            || self.actor.is_some()
            || self.target.is_some()
            || self.description_contains.is_some()
            || self.correlation_id.is_some()
    }
}

impl Default for AuditQuery {
    fn default() -> Self {
        Self::new()
    }
}
