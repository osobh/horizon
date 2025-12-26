//! Configuration optimization using learned patterns

use crate::{learning::DeploymentPattern, AgentConfiguration, Result};

/// Pattern-based configuration optimizer
pub struct PatternOptimizer;

impl PatternOptimizer {
    /// Create a new pattern optimizer
    pub fn new() -> Self {
        Self
    }

    /// Apply insights from learned patterns to improve configuration
    pub async fn apply_pattern_insights(
        &self,
        mut config: AgentConfiguration,
        patterns: &[DeploymentPattern],
    ) -> Result<AgentConfiguration> {
        // Calculate weighted averages based on pattern confidence and success
        let mut total_weight = 0.0;
        let mut weighted_cpu = 0.0;
        let mut weighted_memory = 0.0;
        let mut weighted_replicas = 0.0;

        for pattern in patterns {
            let weight = pattern.confidence * if pattern.success { 1.0 } else { 0.5 };

            weighted_cpu += pattern.config.resources.cpu_cores * weight;
            weighted_memory += pattern.config.resources.memory_gb * weight;
            weighted_replicas += pattern.config.scaling.max_replicas as f32 * weight;
            total_weight += weight;
        }

        if total_weight > 0.0 {
            // Apply weighted insights with conservative blending
            let blend_factor = 0.3; // 30% pattern influence, 70% original analysis

            config.resources.cpu_cores = config.resources.cpu_cores * (1.0 - blend_factor)
                + (weighted_cpu / total_weight) * blend_factor;
            config.resources.memory_gb = config.resources.memory_gb * (1.0 - blend_factor)
                + (weighted_memory / total_weight) * blend_factor;
            config.scaling.max_replicas = ((config.scaling.max_replicas as f32)
                * (1.0 - blend_factor)
                + (weighted_replicas / total_weight) * blend_factor)
                .round() as u32;

            // Apply personality adjustments from successful patterns
            self.apply_personality_insights(&mut config, patterns)
                .await?;
        }

        Ok(config)
    }

    /// Apply personality insights from patterns
    async fn apply_personality_insights(
        &self,
        config: &mut AgentConfiguration,
        patterns: &[DeploymentPattern],
    ) -> Result<()> {
        let successful_patterns: Vec<_> = patterns
            .iter()
            .filter(|p| p.success && p.confidence > 0.7)
            .collect();

        if successful_patterns.is_empty() {
            return Ok(());
        }

        let mut avg_risk = 0.0;
        let mut avg_coop = 0.0;
        let mut avg_explore = 0.0;
        let mut avg_efficiency = 0.0;
        let mut avg_stability = 0.0;

        for pattern in &successful_patterns {
            avg_risk += pattern.config.personality.risk_tolerance;
            avg_coop += pattern.config.personality.cooperation;
            avg_explore += pattern.config.personality.exploration;
            avg_efficiency += pattern.config.personality.efficiency_focus;
            avg_stability += pattern.config.personality.stability_preference;
        }

        let count = successful_patterns.len() as f32;
        let influence = 0.2; // 20% influence from patterns

        config.personality.risk_tolerance =
            config.personality.risk_tolerance * (1.0 - influence) + (avg_risk / count) * influence;
        config.personality.cooperation =
            config.personality.cooperation * (1.0 - influence) + (avg_coop / count) * influence;
        config.personality.exploration =
            config.personality.exploration * (1.0 - influence) + (avg_explore / count) * influence;
        config.personality.efficiency_focus = config.personality.efficiency_focus
            * (1.0 - influence)
            + (avg_efficiency / count) * influence;
        config.personality.stability_preference = config.personality.stability_preference
            * (1.0 - influence)
            + (avg_stability / count) * influence;

        Ok(())
    }

    /// Calculate configuration confidence score
    pub fn calculate_confidence_score(
        &self,
        config: &AgentConfiguration,
        patterns: &[DeploymentPattern],
    ) -> f32 {
        if patterns.is_empty() {
            return 0.5; // Base confidence with no patterns
        }

        let mut similarity_scores = vec![];

        for pattern in patterns {
            let language_match = if pattern.language == config.language {
                1.0
            } else {
                0.0
            };
            let framework_match = match (&pattern.framework, &config.framework) {
                (Some(p), Some(c)) if p == c => 1.0,
                (None, None) => 1.0,
                _ => 0.5,
            };

            let resource_similarity =
                self.calculate_resource_similarity(&config.resources, &pattern.config.resources);

            let similarity = (language_match + framework_match + resource_similarity) / 3.0;
            similarity_scores.push(similarity * pattern.confidence);
        }

        // Return weighted average of similarities
        let total_similarity: f32 = similarity_scores.iter().sum();
        let max_possible: f32 = patterns.iter().map(|p| p.confidence).sum();

        if max_possible > 0.0 {
            (total_similarity / max_possible).clamp(0.0, 1.0)
        } else {
            0.5
        }
    }

    /// Calculate similarity between resource requirements
    fn calculate_resource_similarity(
        &self,
        res1: &crate::ResourceRequirements,
        res2: &crate::ResourceRequirements,
    ) -> f32 {
        let cpu_ratio = (res1.cpu_cores / res2.cpu_cores.max(0.1))
            .min(res2.cpu_cores / res1.cpu_cores.max(0.1));
        let memory_ratio = (res1.memory_gb / res2.memory_gb.max(0.1))
            .min(res2.memory_gb / res1.memory_gb.max(0.1));
        let gpu_ratio = if res1.gpu_units == 0.0 && res2.gpu_units == 0.0 {
            1.0
        } else {
            (res1.gpu_units / res2.gpu_units.max(0.1)).min(res2.gpu_units / res1.gpu_units.max(0.1))
        };

        (cpu_ratio + memory_ratio + gpu_ratio) / 3.0
    }
}

impl Default for PatternOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BackupFrequency, BackupPolicy};
    use crate::{
        NetworkConfiguration, PersonalityConfig, ResourceRequirements, ScalingPolicy,
        StorageConfiguration,
    };
    use std::collections::HashMap;

    fn create_test_config() -> AgentConfiguration {
        AgentConfiguration {
            agent_id: "test".to_string(),
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
            scaling: ScalingPolicy {
                min_replicas: 1,
                max_replicas: 5,
                target_cpu_percent: 70.0,
                scale_up_threshold: 80.0,
                scale_down_threshold: 30.0,
            },
            networking: NetworkConfiguration {
                expose_ports: vec![],
                ingress_rules: vec![],
                service_mesh: false,
            },
            storage: StorageConfiguration {
                persistent_volumes: vec![],
                temporary_storage_gb: 1.0,
                backup_policy: BackupPolicy {
                    enabled: false,
                    frequency: BackupFrequency::Never,
                    retention_days: 0,
                },
            },
            personality: PersonalityConfig {
                risk_tolerance: 0.5,
                cooperation: 0.7,
                exploration: 0.3,
                efficiency_focus: 0.6,
                stability_preference: 0.8,
            },
        }
    }

    #[tokio::test]
    async fn test_pattern_optimizer_creation() {
        let optimizer = PatternOptimizer::new();
        assert!(true); // Basic instantiation test
    }

    #[tokio::test]
    async fn test_confidence_score_calculation() {
        let optimizer = PatternOptimizer::new();
        let config = create_test_config();

        // Test with no patterns
        let score = optimizer.calculate_confidence_score(&config, &[]);
        assert_eq!(score, 0.5);

        // Test with empty patterns
        let patterns = vec![];
        let score = optimizer.calculate_confidence_score(&config, &patterns);
        assert_eq!(score, 0.5);
    }

    #[tokio::test]
    async fn test_resource_similarity_calculation() {
        let optimizer = PatternOptimizer::new();

        let res1 = ResourceRequirements {
            cpu_cores: 2.0,
            memory_gb: 4.0,
            gpu_units: 0.0,
            storage_gb: 10.0,
            network_bandwidth_mbps: 100.0,
        };

        let res2 = ResourceRequirements {
            cpu_cores: 2.0,
            memory_gb: 4.0,
            gpu_units: 0.0,
            storage_gb: 10.0,
            network_bandwidth_mbps: 100.0,
        };

        let similarity = optimizer.calculate_resource_similarity(&res1, &res2);
        assert_eq!(similarity, 1.0); // Identical resources should have perfect similarity
    }
}
