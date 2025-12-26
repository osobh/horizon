//! Trace context for distributed operations

use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};
use uuid::Uuid;

/// Trace context for distributed operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceContext {
    /// Unique trace ID tied to originating agent + goal
    pub trace_id: Uuid,
    /// SHA256 hash of PTX/kernel code
    pub kernel_id: String,
    /// Runtime container identifier
    pub container_id: Uuid,
    /// Parent span ID for hierarchical tracing
    pub span_id: Uuid,
    /// Parent span ID (optional)
    pub parent_span_id: Option<Uuid>,
    /// Trace creation timestamp
    pub created_at: u64,
}

impl TraceContext {
    /// Create new trace context
    pub fn new(kernel_id: String, container_id: Uuid) -> Self {
        Self {
            trace_id: Uuid::new_v4(),
            kernel_id,
            container_id,
            span_id: Uuid::new_v4(),
            parent_span_id: None,
            created_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        }
    }

    /// Create child span
    pub fn child_span(&self, operation: &str) -> Self {
        Self {
            trace_id: self.trace_id,
            kernel_id: format!("{}_{}", self.kernel_id, operation),
            container_id: self.container_id,
            span_id: Uuid::new_v4(),
            parent_span_id: Some(self.span_id),
            created_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trace_context_creation() {
        let container_id = Uuid::new_v4();
        let context = TraceContext::new("test_kernel".to_string(), container_id);

        assert_eq!(context.kernel_id, "test_kernel");
        assert_eq!(context.container_id, container_id);
        assert!(context.parent_span_id.is_none());
        assert!(context.created_at > 0);
    }

    #[test]
    fn test_child_span_creation() {
        let container_id = Uuid::new_v4();
        let parent = TraceContext::new("test_kernel".to_string(), container_id);
        let child = parent.child_span("operation");

        assert_eq!(child.trace_id, parent.trace_id);
        assert_eq!(child.kernel_id, "test_kernel_operation");
        assert_eq!(child.container_id, parent.container_id);
        assert_eq!(child.parent_span_id, Some(parent.span_id));
        assert_ne!(child.span_id, parent.span_id);
    }

    #[test]
    fn test_serialization() {
        let context = TraceContext::new("test_kernel".to_string(), Uuid::new_v4());

        let serialized = serde_json::to_string(&context).unwrap();
        let deserialized: TraceContext = serde_json::from_str(&serialized).unwrap();

        assert_eq!(context.trace_id, deserialized.trace_id);
        assert_eq!(context.kernel_id, deserialized.kernel_id);
        assert_eq!(context.container_id, deserialized.container_id);
    }
}
