//! Configuration generation for zero-config intelligence
//!
//! This module generates optimal agent configurations based on code analysis
//! and learned patterns from similar deployments.

pub mod config_builder;
pub mod config_generator;
pub mod config_optimization;

// Re-export main types
pub use config_builder::AgentConfigBuilder;
pub use config_generator::ConfigGenerator;
pub use config_optimization::PatternOptimizer;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::{CodebaseAnalysis, ComplexityMetrics};
    use crate::learning::DeploymentPattern;
    use crate::{Dependency, DependencyType, ResourceRequirements};

    #[tokio::test]
    async fn test_modular_config_generation_integration() {
        let generator = ConfigGenerator::new();
        let builder = AgentConfigBuilder::new();
        let optimizer = PatternOptimizer::new();

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

        let config = generator.generate_config(analysis, vec![]).await;
        assert!(config.is_ok(), "Modular config generation should work");

        let config = config.unwrap();
        assert_eq!(config.language, "rust");
        assert_eq!(config.framework, Some("tokio".to_string()));
    }
}
