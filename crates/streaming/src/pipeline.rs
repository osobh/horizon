//! Streaming pipeline builder and orchestration

/// Pipeline for connecting sources, processors, and sinks
pub struct StreamingPipeline {
    name: String,
    // Implementation will be added in next steps
}

impl StreamingPipeline {
    /// Create a new streaming pipeline
    pub fn new(name: String) -> Self {
        Self { name }
    }

    /// Get pipeline name
    pub fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_creation() {
        let pipeline = StreamingPipeline::new("test-pipeline".to_string());
        assert_eq!(pipeline.name(), "test-pipeline");
    }
}
