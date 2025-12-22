//! Tests for streaming pipeline functionality

use crate::pipeline::*;

#[test]
fn test_pipeline_creation() {
    let pipeline = StreamingPipeline::new("test-pipeline".to_string());
    assert_eq!(pipeline.name(), "test-pipeline");
}

#[test]
fn test_pipeline_name_empty() {
    let pipeline = StreamingPipeline::new(String::new());
    assert_eq!(pipeline.name(), "");
}

#[test]
fn test_pipeline_name_with_special_chars() {
    let pipeline = StreamingPipeline::new("test-pipeline_123!@#".to_string());
    assert_eq!(pipeline.name(), "test-pipeline_123!@#");
}

#[test]
fn test_pipeline_name_unicode() {
    let pipeline = StreamingPipeline::new("тест-pipeline-🚀".to_string());
    assert_eq!(pipeline.name(), "тест-pipeline-🚀");
}

#[test]
fn test_multiple_pipelines() {
    let pipeline1 = StreamingPipeline::new("pipeline-1".to_string());
    let pipeline2 = StreamingPipeline::new("pipeline-2".to_string());

    assert_eq!(pipeline1.name(), "pipeline-1");
    assert_eq!(pipeline2.name(), "pipeline-2");
    assert_ne!(pipeline1.name(), pipeline2.name());
}

#[test]
fn test_pipeline_long_name() {
    let long_name = "a".repeat(1000);
    let pipeline = StreamingPipeline::new(long_name.clone());
    assert_eq!(pipeline.name(), &long_name);
    assert_eq!(pipeline.name().len(), 1000);
}
