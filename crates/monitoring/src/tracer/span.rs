//! Trace span implementation

use super::context::TraceContext;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

/// Individual trace span
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceSpan {
    pub context: TraceContext,
    pub operation: String,
    pub start_time: u64,
    pub end_time: Option<u64>,
    pub duration_ms: Option<u64>,
    pub status: SpanStatus,
    pub attributes: HashMap<String, String>,
    pub events: Vec<SpanEvent>,
    pub logs: Vec<SpanLog>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpanStatus {
    Running,
    Success,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpanEvent {
    pub timestamp: u64,
    pub name: String,
    pub attributes: HashMap<String, String>,
}

/// Log entry within a span
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpanLog {
    pub timestamp: u64,
    pub level: String,
    pub message: String,
    pub fields: HashMap<String, String>,
}

impl SpanLog {
    pub fn new(level: String, message: String) -> Self {
        Self {
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_micros() as u64,
            level,
            message,
            fields: HashMap::new(),
        }
    }
}

impl TraceSpan {
    /// Create new trace span
    pub fn new(context: TraceContext) -> Self {
        Self {
            operation: context.kernel_id.clone(),
            context,
            start_time: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_micros() as u64,
            end_time: None,
            duration_ms: None,
            status: SpanStatus::Running,
            attributes: HashMap::new(),
            events: Vec::new(),
            logs: Vec::new(),
        }
    }

    /// End the span
    pub fn end(&mut self) {
        let end_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;

        self.end_time = Some(end_time);
        self.duration_ms = Some((end_time - self.start_time) / 1000);

        if self.status == SpanStatus::Running {
            self.status = SpanStatus::Success;
        }
    }

    /// Mark span as failed
    pub fn fail(&mut self, error: String) {
        self.status = SpanStatus::Failed;
        self.add_attribute("error", error);
        self.end();
    }

    /// Add attribute
    pub fn add_attribute(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.attributes.insert(key.into(), value.into());
    }

    /// Add event
    pub fn add_event(&mut self, name: String, attributes: HashMap<String, String>) {
        self.events.push(SpanEvent {
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_micros() as u64,
            name,
            attributes,
        });
    }

    /// Add log entry
    pub fn add_log(&mut self, log: SpanLog) {
        self.logs.push(log);
    }

    /// Get duration in milliseconds
    pub fn duration_ms(&self) -> Option<u64> {
        self.duration_ms
    }

    /// Check if span is complete
    pub fn is_complete(&self) -> bool {
        self.end_time.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    #[test]
    fn test_span_creation() {
        let context = TraceContext::new("test_kernel".to_string(), Uuid::new_v4());
        let span = TraceSpan::new(context.clone());

        assert_eq!(span.operation, "test_kernel");
        assert_eq!(span.status, SpanStatus::Running);
        assert!(span.end_time.is_none());
        assert!(span.duration_ms.is_none());
        assert!(span.events.is_empty());
        assert!(span.logs.is_empty());
    }

    #[test]
    fn test_span_end() {
        let context = TraceContext::new("test_kernel".to_string(), Uuid::new_v4());
        let mut span = TraceSpan::new(context);

        std::thread::sleep(std::time::Duration::from_millis(10));
        span.end();

        assert_eq!(span.status, SpanStatus::Success);
        assert!(span.end_time.is_some());
        assert!(span.duration_ms.is_some());
        assert!(span.duration_ms.unwrap() >= 10);
        assert!(span.is_complete());
    }

    #[test]
    fn test_span_fail() {
        let context = TraceContext::new("test_kernel".to_string(), Uuid::new_v4());
        let mut span = TraceSpan::new(context);

        span.fail("Test error".to_string());

        assert_eq!(span.status, SpanStatus::Failed);
        assert!(span.is_complete());
        assert_eq!(
            span.attributes.get("error"),
            Some(&"Test error".to_string())
        );
    }

    #[test]
    fn test_span_attributes() {
        let context = TraceContext::new("test_kernel".to_string(), Uuid::new_v4());
        let mut span = TraceSpan::new(context);

        span.add_attribute("key1", "value1");
        span.add_attribute("key2", "value2");

        assert_eq!(span.attributes.len(), 2);
        assert_eq!(span.attributes.get("key1"), Some(&"value1".to_string()));
        assert_eq!(span.attributes.get("key2"), Some(&"value2".to_string()));
    }

    #[test]
    fn test_span_events() {
        let context = TraceContext::new("test_kernel".to_string(), Uuid::new_v4());
        let mut span = TraceSpan::new(context);

        let mut attrs = HashMap::new();
        attrs.insert("detail".to_string(), "important".to_string());

        span.add_event("Event1".to_string(), HashMap::new());
        span.add_event("Event2".to_string(), attrs);

        assert_eq!(span.events.len(), 2);
        assert_eq!(span.events[0].name, "Event1");
        assert_eq!(span.events[1].name, "Event2");
        assert_eq!(
            span.events[1].attributes.get("detail"),
            Some(&"important".to_string())
        );
    }

    #[test]
    fn test_span_logs() {
        let context = TraceContext::new("test_kernel".to_string(), Uuid::new_v4());
        let mut span = TraceSpan::new(context);

        let log1 = SpanLog::new("INFO".to_string(), "Test message 1".to_string());
        let log2 = SpanLog::new("ERROR".to_string(), "Test message 2".to_string());

        span.add_log(log1);
        span.add_log(log2);

        assert_eq!(span.logs.len(), 2);
        assert_eq!(span.logs[0].level, "INFO");
        assert_eq!(span.logs[1].level, "ERROR");
    }

    #[test]
    fn test_span_log_creation() {
        let mut log = SpanLog::new("DEBUG".to_string(), "Debug message".to_string());
        log.fields.insert("key".to_string(), "value".to_string());

        assert_eq!(log.level, "DEBUG");
        assert_eq!(log.message, "Debug message");
        assert!(log.timestamp > 0);
        assert_eq!(log.fields.get("key"), Some(&"value".to_string()));
    }
}
