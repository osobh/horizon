//! Trace query functionality

use super::span::{SpanStatus, TraceSpan};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Query for filtering traces
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceQuery {
    pub trace_id: Option<Uuid>,
    pub kernel_id: Option<String>,
    pub container_id: Option<Uuid>,
    pub status_filter: Option<SpanStatus>,
    pub operation_filter: Option<String>,
    pub min_duration_ms: Option<u64>,
    pub max_duration_ms: Option<u64>,
    pub limit: Option<usize>,
}

impl TraceQuery {
    pub fn new() -> Self {
        Self {
            trace_id: None,
            kernel_id: None,
            container_id: None,
            status_filter: None,
            operation_filter: None,
            min_duration_ms: None,
            max_duration_ms: None,
            limit: None,
        }
    }

    pub fn with_trace_id(mut self, trace_id: Uuid) -> Self {
        self.trace_id = Some(trace_id);
        self
    }

    pub fn with_kernel_id(mut self, kernel_id: String) -> Self {
        self.kernel_id = Some(kernel_id);
        self
    }

    pub fn with_container_id(mut self, container_id: Uuid) -> Self {
        self.container_id = Some(container_id);
        self
    }

    pub fn with_status(mut self, status: SpanStatus) -> Self {
        self.status_filter = Some(status);
        self
    }

    pub fn with_operation(mut self, operation: String) -> Self {
        self.operation_filter = Some(operation);
        self
    }

    pub fn with_min_duration_ms(mut self, duration: u64) -> Self {
        self.min_duration_ms = Some(duration);
        self
    }

    pub fn with_max_duration_ms(mut self, duration: u64) -> Self {
        self.max_duration_ms = Some(duration);
        self
    }

    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Check if a span matches this query
    pub fn matches(&self, span: &TraceSpan) -> bool {
        if let Some(id) = self.trace_id {
            if span.context.trace_id != id {
                return false;
            }
        }

        if let Some(ref kernel_id) = self.kernel_id {
            if !span.context.kernel_id.contains(kernel_id) {
                return false;
            }
        }

        if let Some(container_id) = self.container_id {
            if span.context.container_id != container_id {
                return false;
            }
        }

        if let Some(status) = self.status_filter {
            if span.status != status {
                return false;
            }
        }

        if let Some(ref operation) = self.operation_filter {
            if !span.operation.contains(operation) {
                return false;
            }
        }

        if let Some(min_duration) = self.min_duration_ms {
            if let Some(duration) = span.duration_ms {
                if duration < min_duration {
                    return false;
                }
            } else {
                return false; // No duration means still running
            }
        }

        if let Some(max_duration) = self.max_duration_ms {
            if let Some(duration) = span.duration_ms {
                if duration > max_duration {
                    return false;
                }
            }
        }

        true
    }
}

impl Default for TraceQuery {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tracer::context::TraceContext;

    #[test]
    fn test_query_creation() {
        let query = TraceQuery::new();
        assert!(query.trace_id.is_none());
        assert!(query.kernel_id.is_none());
        assert!(query.container_id.is_none());
        assert!(query.status_filter.is_none());
        assert!(query.operation_filter.is_none());
        assert!(query.min_duration_ms.is_none());
        assert!(query.max_duration_ms.is_none());
        assert!(query.limit.is_none());
    }

    #[test]
    fn test_query_builder() {
        let trace_id = Uuid::new_v4();
        let container_id = Uuid::new_v4();

        let query = TraceQuery::new()
            .with_trace_id(trace_id)
            .with_kernel_id("test_kernel".to_string())
            .with_container_id(container_id)
            .with_status(SpanStatus::Success)
            .with_operation("test_op".to_string())
            .with_min_duration_ms(100)
            .with_max_duration_ms(1000)
            .with_limit(10);

        assert_eq!(query.trace_id, Some(trace_id));
        assert_eq!(query.kernel_id, Some("test_kernel".to_string()));
        assert_eq!(query.container_id, Some(container_id));
        assert_eq!(query.status_filter, Some(SpanStatus::Success));
        assert_eq!(query.operation_filter, Some("test_op".to_string()));
        assert_eq!(query.min_duration_ms, Some(100));
        assert_eq!(query.max_duration_ms, Some(1000));
        assert_eq!(query.limit, Some(10));
    }

    #[test]
    fn test_query_matches_trace_id() {
        let trace_id = Uuid::new_v4();
        let context = TraceContext {
            trace_id,
            kernel_id: "kernel".to_string(),
            container_id: Uuid::new_v4(),
            span_id: Uuid::new_v4(),
            parent_span_id: None,
            created_at: 0,
        };
        let span = TraceSpan::new(context);

        let query = TraceQuery::new().with_trace_id(trace_id);
        assert!(query.matches(&span));

        let query = TraceQuery::new().with_trace_id(Uuid::new_v4());
        assert!(!query.matches(&span));
    }

    #[test]
    fn test_query_matches_kernel_id() {
        let context = TraceContext::new("test_kernel_v1".to_string(), Uuid::new_v4());
        let span = TraceSpan::new(context);

        let query = TraceQuery::new().with_kernel_id("test_kernel".to_string());
        assert!(query.matches(&span));

        let query = TraceQuery::new().with_kernel_id("other_kernel".to_string());
        assert!(!query.matches(&span));
    }

    #[test]
    fn test_query_matches_status() {
        let context = TraceContext::new("kernel".to_string(), Uuid::new_v4());
        let mut span = TraceSpan::new(context);
        span.end();

        let query = TraceQuery::new().with_status(SpanStatus::Success);
        assert!(query.matches(&span));

        let query = TraceQuery::new().with_status(SpanStatus::Failed);
        assert!(!query.matches(&span));
    }

    #[test]
    fn test_query_matches_duration() {
        let context = TraceContext::new("kernel".to_string(), Uuid::new_v4());
        let mut span = TraceSpan::new(context);
        span.duration_ms = Some(500);

        let query = TraceQuery::new().with_min_duration_ms(100);
        assert!(query.matches(&span));

        let query = TraceQuery::new().with_min_duration_ms(600);
        assert!(!query.matches(&span));

        let query = TraceQuery::new().with_max_duration_ms(600);
        assert!(query.matches(&span));

        let query = TraceQuery::new().with_max_duration_ms(400);
        assert!(!query.matches(&span));
    }

    #[test]
    fn test_query_matches_multiple_criteria() {
        let container_id = Uuid::new_v4();
        let context = TraceContext::new("test_kernel".to_string(), container_id);
        let mut span = TraceSpan::new(context);
        span.end();
        // Manually set duration for consistent testing
        span.duration_ms = Some(500);

        let query = TraceQuery::new()
            .with_kernel_id("test_kernel".to_string())
            .with_container_id(container_id)
            .with_status(SpanStatus::Success)
            .with_min_duration_ms(400)
            .with_max_duration_ms(600);

        assert!(query.matches(&span));

        // Change one criteria to not match
        let query = TraceQuery::new()
            .with_kernel_id("test_kernel".to_string())
            .with_container_id(container_id)
            .with_status(SpanStatus::Failed) // This doesn't match
            .with_min_duration_ms(400)
            .with_max_duration_ms(600);

        assert!(!query.matches(&span));
    }
}
