//! Main configuration generator implementation

use super::config_builder::AgentConfigBuilder;
use super::config_optimization::PatternOptimizer;
use crate::{analysis::CodebaseAnalysis, learning::DeploymentPattern, AgentConfiguration, Result};

/// Configuration generator that creates optimal agent configurations
pub struct ConfigGenerator {
    builder: AgentConfigBuilder,
    optimizer: PatternOptimizer,
}

impl ConfigGenerator {
    /// Create a new configuration generator
    pub fn new() -> Self {
        Self {
            builder: AgentConfigBuilder::new(),
            optimizer: PatternOptimizer::new(),
        }
    }

    /// Generate configuration based on analysis and learned patterns
    pub async fn generate_config(
        &self,
        analysis: CodebaseAnalysis,
        patterns: Vec<DeploymentPattern>,
    ) -> Result<AgentConfiguration> {
        // Start with base configuration from analysis
        let mut config = self.builder.from_analysis(&analysis).await?;

        // Apply insights from similar patterns
        if !patterns.is_empty() {
            config = self
                .optimizer
                .apply_pattern_insights(config, &patterns)
                .await?;
        }

        // Validate and optimize the final configuration
        self.validate_and_optimize(config).await
    }

    /// Validate and optimize the final configuration
    async fn validate_and_optimize(
        &self,
        mut config: AgentConfiguration,
    ) -> Result<AgentConfiguration> {
        // Validate resource constraints
        if config.resources.cpu_cores <= 0.0 {
            return Err(crate::ZeroConfigError::config_validation(
                "CPU cores must be positive",
            ));
        }
        if config.resources.memory_gb <= 0.0 {
            return Err(crate::ZeroConfigError::config_validation(
                "Memory must be positive",
            ));
        }

        // Optimize scaling policy based on resource requirements
        if config.resources.cpu_cores > 4.0 || config.resources.memory_gb > 8.0 {
            config.scaling.max_replicas = config.scaling.max_replicas.min(5);
        }

        // Ensure personality traits are within valid bounds
        config.personality.risk_tolerance = config.personality.risk_tolerance.clamp(0.0, 1.0);
        config.personality.cooperation = config.personality.cooperation.clamp(0.0, 1.0);
        config.personality.exploration = config.personality.exploration.clamp(0.0, 1.0);
        config.personality.efficiency_focus = config.personality.efficiency_focus.clamp(0.0, 1.0);
        config.personality.stability_preference =
            config.personality.stability_preference.clamp(0.0, 1.0);

        Ok(config)
    }
}

impl Default for ConfigGenerator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::{CodebaseAnalysis, ComplexityMetrics};
    use crate::{Dependency, DependencyType, ResourceRequirements};

    #[tokio::test]
    async fn test_config_generator_creation() {
        let generator = ConfigGenerator::new();
        assert!(true); // Basic instantiation test
    }

    #[tokio::test]
    async fn test_config_generation_validation() {
        let generator = ConfigGenerator::new();

        let analysis = CodebaseAnalysis {
            path: "/test/path".to_string(),
            language: "rust".to_string(),
            framework: Some("tokio".to_string()),
            dependencies: vec![],
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
        assert!(config.is_ok());

        let config = config.unwrap();
        assert_eq!(config.language, "rust");
        assert!(
            config.personality.risk_tolerance >= 0.0 && config.personality.risk_tolerance <= 1.0
        );
    }
}
