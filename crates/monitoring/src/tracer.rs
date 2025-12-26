//! Distributed tracing implementation for GPU containers and agent workflows

pub mod context;
pub mod query;
pub mod span;
pub mod storage;

pub use context::TraceContext;
pub use query::TraceQuery;
pub use span::{SpanLog, TraceSpan};
pub use storage::{MemoryTraceStorage, TraceStorage};

use crate::MonitoringError;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::info;
use uuid::Uuid;

/// Main distributed tracer for managing traces across GPU containers
pub struct DistributedTracer {
    storage: Arc<dyn TraceStorage + Send + Sync>,
    active_spans: Arc<RwLock<Vec<TraceSpan>>>,
}

impl DistributedTracer {
    /// Create new distributed tracer
    pub fn new(storage: Arc<dyn TraceStorage + Send + Sync>) -> Self {
        Self {
            storage,
            active_spans: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Start a new trace
    pub async fn start_trace(&self, context: TraceContext) -> Result<TraceSpan, MonitoringError> {
        let span = TraceSpan::new(context);

        // Add to active spans
        let mut active = self.active_spans.write().await;
        active.push(span.clone());

        info!(
            "Started trace: {} for kernel: {}",
            span.context.trace_id, span.context.kernel_id
        );

        Ok(span)
    }

    /// Add event to a trace
    pub async fn add_event(&self, trace_id: Uuid, event: String) -> Result<(), MonitoringError> {
        let mut active = self.active_spans.write().await;

        if let Some(span) = active.iter_mut().find(|s| s.context.trace_id == trace_id) {
            span.add_event(event, HashMap::new());
            Ok(())
        } else {
            Err(MonitoringError::TracingFailed {
                reason: format!("Trace {trace_id} not found"),
            })
        }
    }

    /// End a trace
    pub async fn end_trace(&self, trace_id: Uuid) -> Result<(), MonitoringError> {
        let mut active = self.active_spans.write().await;

        if let Some(index) = active.iter().position(|s| s.context.trace_id == trace_id) {
            let mut span = active.remove(index);
            span.end();

            // Store completed trace
            self.storage.store_trace(span).await?;

            info!("Ended trace: {}", trace_id);
            Ok(())
        } else {
            Err(MonitoringError::TracingFailed {
                reason: format!("Trace {trace_id} not found"),
            })
        }
    }

    /// Query traces
    pub async fn query_traces(&self, query: TraceQuery) -> Result<Vec<TraceSpan>, MonitoringError> {
        self.storage.query_traces(query).await
    }

    /// Get active spans
    pub async fn active_spans(&self) -> Vec<TraceSpan> {
        self.active_spans.read().await.clone()
    }

    /// Clear all traces
    pub async fn clear_traces(&self) -> Result<(), MonitoringError> {
        self.active_spans.write().await.clear();
        self.storage.clear().await
    }
}

use std::collections::HashMap;

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_distributed_tracer_basic() {
        let storage = Arc::new(MemoryTraceStorage::new());
        let tracer = DistributedTracer::new(storage);

        let context = TraceContext::new("test_kernel".to_string(), Uuid::new_v4());
        let span = tracer.start_trace(context.clone()).await.unwrap();

        // Verify active spans
        let active = tracer.active_spans().await;
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].context.trace_id, context.trace_id);

        // End trace
        tracer.end_trace(context.trace_id).await.unwrap();

        // Verify no active spans
        let active = tracer.active_spans().await;
        assert_eq!(active.len(), 0);

        // Query stored traces
        let query = TraceQuery::new();
        let traces = tracer.query_traces(query).await.unwrap();
        assert_eq!(traces.len(), 1);
    }

    #[tokio::test]
    async fn test_distributed_tracer_events() {
        let storage = Arc::new(MemoryTraceStorage::new());
        let tracer = DistributedTracer::new(storage);

        let context = TraceContext::new("test_kernel".to_string(), Uuid::new_v4());
        let _span = tracer.start_trace(context.clone()).await.unwrap();

        // Add events
        tracer
            .add_event(context.trace_id, "Event 1".to_string())
            .await
            .unwrap();
        tracer
            .add_event(context.trace_id, "Event 2".to_string())
            .await
            .unwrap();

        // Verify events in active span
        let active = tracer.active_spans().await;
        assert_eq!(active[0].events.len(), 2);

        // End trace
        tracer.end_trace(context.trace_id).await.unwrap();

        // Query and verify events persisted
        let query = TraceQuery::new();
        let traces = tracer.query_traces(query).await.unwrap();
        assert_eq!(traces[0].events.len(), 2);
    }

    #[tokio::test]
    async fn test_distributed_tracer_not_found() {
        let storage = Arc::new(MemoryTraceStorage::new());
        let tracer = DistributedTracer::new(storage);

        let fake_id = Uuid::new_v4();

        // Try to add event to non-existent trace
        let result = tracer.add_event(fake_id, "Event".to_string()).await;
        assert!(result.is_err());

        // Try to end non-existent trace
        let result = tracer.end_trace(fake_id).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_clear_traces() {
        let storage = Arc::new(MemoryTraceStorage::new());
        let tracer = DistributedTracer::new(storage);

        // Create multiple traces
        for i in 0..3 {
            let context = TraceContext::new(format!("kernel_{i}"), Uuid::new_v4());
            let _span = tracer.start_trace(context.clone()).await.unwrap();
            tracer.end_trace(context.trace_id).await.unwrap();
        }

        // Verify traces exist
        let query = TraceQuery::new();
        let traces = tracer.query_traces(query.clone()).await.unwrap();
        assert_eq!(traces.len(), 3);

        // Clear all traces
        tracer.clear_traces().await.unwrap();

        // Verify traces cleared
        let traces = tracer.query_traces(query).await.unwrap();
        assert_eq!(traces.len(), 0);
    }
}
