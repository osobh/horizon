//! Trace storage implementations

use super::query::TraceQuery;
use super::span::TraceSpan;
use crate::MonitoringError;
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Trait for trace storage backends
#[async_trait]
pub trait TraceStorage {
    /// Store a completed trace
    async fn store_trace(&self, span: TraceSpan) -> Result<(), MonitoringError>;

    /// Query traces based on criteria
    async fn query_traces(&self, query: TraceQuery) -> Result<Vec<TraceSpan>, MonitoringError>;

    /// Get trace by ID
    async fn get_trace(&self, trace_id: uuid::Uuid) -> Result<Option<TraceSpan>, MonitoringError>;

    /// Delete old traces
    async fn cleanup(&self, older_than_secs: u64) -> Result<usize, MonitoringError>;

    /// Clear all traces
    async fn clear(&self) -> Result<(), MonitoringError>;
}

/// In-memory trace storage
pub struct MemoryTraceStorage {
    traces: Arc<RwLock<HashMap<uuid::Uuid, TraceSpan>>>,
}

impl MemoryTraceStorage {
    pub fn new() -> Self {
        Self {
            traces: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn trace_count(&self) -> usize {
        self.traces.read().await.len()
    }
}

#[async_trait]
impl TraceStorage for MemoryTraceStorage {
    async fn store_trace(&self, span: TraceSpan) -> Result<(), MonitoringError> {
        let mut traces = self.traces.write().await;
        traces.insert(span.context.trace_id, span);
        Ok(())
    }

    async fn query_traces(&self, query: TraceQuery) -> Result<Vec<TraceSpan>, MonitoringError> {
        let traces = self.traces.read().await;
        let mut results: Vec<TraceSpan> = traces
            .values()
            .filter(|span| query.matches(span))
            .cloned()
            .collect();

        // Sort by start time (newest first)
        results.sort_by(|a, b| b.start_time.cmp(&a.start_time));

        // Apply limit
        if let Some(limit) = query.limit {
            results.truncate(limit);
        }

        Ok(results)
    }

    async fn get_trace(&self, trace_id: uuid::Uuid) -> Result<Option<TraceSpan>, MonitoringError> {
        let traces = self.traces.read().await;
        Ok(traces.get(&trace_id).cloned())
    }

    async fn cleanup(&self, older_than_secs: u64) -> Result<usize, MonitoringError> {
        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let mut traces = self.traces.write().await;
        let initial_count = traces.len();

        traces.retain(|_, span| {
            let age_secs = current_time.saturating_sub(span.start_time / 1_000_000);
            age_secs < older_than_secs
        });

        Ok(initial_count - traces.len())
    }

    async fn clear(&self) -> Result<(), MonitoringError> {
        self.traces.write().await.clear();
        Ok(())
    }
}

impl Default for MemoryTraceStorage {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tracer::context::TraceContext;
    use uuid::Uuid;

    #[tokio::test]
    async fn test_memory_storage_basic() {
        let storage = MemoryTraceStorage::new();

        // Create and store trace
        let context = TraceContext::new("test_kernel".to_string(), Uuid::new_v4());
        let mut span = TraceSpan::new(context.clone());
        span.end();

        storage.store_trace(span.clone()).await.unwrap();

        // Verify stored
        assert_eq!(storage.trace_count().await, 1);

        // Get by ID
        let retrieved = storage.get_trace(context.trace_id).await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().context.trace_id, context.trace_id);
    }

    #[tokio::test]
    async fn test_memory_storage_query() {
        let storage = MemoryTraceStorage::new();

        // Store multiple traces
        for i in 0..5 {
            let context = TraceContext::new(format!("kernel_{i}"), Uuid::new_v4());
            let mut span = TraceSpan::new(context);
            if i % 2 == 0 {
                span.add_attribute("type", "even");
            }
            span.end();
            storage.store_trace(span).await.unwrap();
        }

        // Query all
        let query = TraceQuery::new();
        let results = storage.query_traces(query).await.unwrap();
        assert_eq!(results.len(), 5);

        // Query with limit
        let query = TraceQuery::new().with_limit(3);
        let results = storage.query_traces(query).await.unwrap();
        assert_eq!(results.len(), 3);
    }

    #[tokio::test]
    async fn test_memory_storage_cleanup() {
        let storage = MemoryTraceStorage::new();

        // Store old trace
        let context = TraceContext::new("old_kernel".to_string(), Uuid::new_v4());
        let mut span = TraceSpan::new(context);
        span.start_time = 1000; // Very old timestamp
        span.end();
        storage.store_trace(span).await.unwrap();

        // Store recent trace
        let context = TraceContext::new("new_kernel".to_string(), Uuid::new_v4());
        let span = TraceSpan::new(context);
        storage.store_trace(span).await.unwrap();

        // Cleanup old traces
        let removed = storage.cleanup(60).await.unwrap(); // 60 seconds
        assert_eq!(removed, 1);
        assert_eq!(storage.trace_count().await, 1);
    }

    #[tokio::test]
    async fn test_memory_storage_clear() {
        let storage = MemoryTraceStorage::new();

        // Store multiple traces
        for i in 0..3 {
            let context = TraceContext::new(format!("kernel_{i}"), Uuid::new_v4());
            let span = TraceSpan::new(context);
            storage.store_trace(span).await.unwrap();
        }

        assert_eq!(storage.trace_count().await, 3);

        // Clear all
        storage.clear().await.unwrap();
        assert_eq!(storage.trace_count().await, 0);
    }

    #[tokio::test]
    async fn test_memory_storage_not_found() {
        let storage = MemoryTraceStorage::new();

        let fake_id = Uuid::new_v4();
        let result = storage.get_trace(fake_id).await.unwrap();
        assert!(result.is_none());
    }
}
