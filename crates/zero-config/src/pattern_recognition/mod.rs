//! Pattern recognition for zero-config intelligence
//!
//! This module recognizes deployment patterns from code analysis and provides
//! similarity matching for configuration optimization.

pub mod pattern_recognizer;
pub mod pattern_types;
pub mod similarity_engine;

// Re-export main types
pub use pattern_recognizer::PatternRecognizer;
pub use pattern_types::{PatternType, RecognizedPattern};
pub use similarity_engine::SimilarityEngine;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::{CodebaseAnalysis, ComplexityMetrics};
    use crate::{Dependency, DependencyType, ResourceRequirements};

    #[tokio::test]
    async fn test_modular_pattern_recognition_integration() {
        let recognizer = PatternRecognizer::new();
        let similarity_engine = SimilarityEngine::new();

        // Test that all modules work together
        let analysis = CodebaseAnalysis {
            path: "/test/path".to_string(),
            language: "rust".to_string(),
            framework: Some("tokio".to_string()),
            dependencies: vec![Dependency {
                name: "tokio".to_string(),
                version: Some("1.0.0".to_string()),
                dependency_type: DependencyType::WebFramework,
            }],
            resources: ResourceRequirements {
                cpu_cores: 2.0,
                memory_gb: 4.0,
                gpu_units: 0.0,
                storage_gb: 10.0,
                network_bandwidth_mbps: 100.0,
            },
            complexity: ComplexityMetrics {
                total_lines: 1000,
                total_files: 20,
                function_count: 50,
                class_count: 10,
                import_count: 15,
                cyclomatic_complexity: 5.0,
                maintainability_index: 75.0,
            },
            file_count: 20,
            total_lines: 1000,
        };

        let patterns = recognizer.recognize_patterns(&analysis).await;
        assert!(patterns.is_ok(), "Pattern recognition should work");

        let patterns = patterns.unwrap();
        assert!(
            !patterns.is_empty(),
            "Should recognize at least one pattern"
        );
    }
}
