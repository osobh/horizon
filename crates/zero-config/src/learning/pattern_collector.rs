//! Pattern collection and feature extraction

use super::{
    DependencyFeatures, DeploymentOutcome, DeploymentPattern, FeatureVector, LanguageFeatures,
    PatternStore, PersonalityFeatures, ResourceFeatures, ScalingFeatures,
};
use crate::{AgentConfiguration, Result};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Pattern collection system
pub struct PatternCollector {
    pub(crate) pattern_store: Arc<RwLock<PatternStore>>,
}

impl PatternCollector {
    pub(crate) fn new(pattern_store: Arc<RwLock<PatternStore>>) -> Self {
        Self { pattern_store }
    }

    /// Collect a pattern from a deployment outcome
    pub async fn collect_pattern(&self, deployment: DeploymentOutcome) -> Result<()> {
        // Extract features from the deployment
        let features = self.extract_features(&deployment.config).await?;

        // Create deployment pattern
        let pattern = DeploymentPattern {
            id: uuid::Uuid::new_v4().to_string(),
            language: deployment.config.language.clone(),
            framework: deployment.config.framework.clone(),
            features,
            config: deployment.config.clone(),
            metrics: deployment.metrics,
            success: deployment.success,
            confidence: if deployment.success { 1.0 } else { 0.0 },
            usage_count: 1,
            last_used: chrono::Utc::now(),
            personality_adjustments: HashMap::new(),
            cpu_cores: deployment.config.resources.cpu_cores,
            memory_gb: deployment.config.resources.memory_gb,
            success_rate: if deployment.success { 1.0 } else { 0.0 },
        };

        // Store the pattern
        let mut store = self.pattern_store.write().await;
        store.add_pattern(pattern);

        Ok(())
    }

    /// Extract features from an agent configuration for pattern matching
    pub(crate) async fn extract_features(
        &self,
        config: &AgentConfiguration,
    ) -> Result<FeatureVector> {
        Ok(FeatureVector {
            language_features: self.extract_language_features(config),
            dependency_features: self.extract_dependency_features(config),
            resource_features: self.extract_resource_features(config),
            scaling_features: self.extract_scaling_features(config),
            personality_features: self.extract_personality_features(config),
        })
    }

    pub(crate) fn extract_language_features(
        &self,
        config: &AgentConfiguration,
    ) -> LanguageFeatures {
        LanguageFeatures {
            language: config.language.clone(),
            framework: config.framework.clone(),
            has_web_framework: config
                .dependencies
                .iter()
                .any(|d| matches!(d.dependency_type, crate::DependencyType::WebFramework)),
            has_database: config
                .dependencies
                .iter()
                .any(|d| matches!(d.dependency_type, crate::DependencyType::Database)),
            has_cache: config
                .dependencies
                .iter()
                .any(|d| matches!(d.dependency_type, crate::DependencyType::Cache)),
            has_ml_framework: config
                .dependencies
                .iter()
                .any(|d| matches!(d.dependency_type, crate::DependencyType::MLFramework)),
            has_message_queue: config
                .dependencies
                .iter()
                .any(|d| matches!(d.dependency_type, crate::DependencyType::MessageQueue)),
            dependency_count: config.dependencies.len(),
            total_lines: 1000, // Default value, would be set from actual analysis
            complexity_score: 0.5, // Default value, would be set from actual analysis
        }
    }

    pub(crate) fn extract_dependency_features(
        &self,
        config: &AgentConfiguration,
    ) -> DependencyFeatures {
        let mut db_count = 0;
        let mut cache_count = 0;
        let mut web_count = 0;
        let mut ml_count = 0;
        let mut message_queue_count = 0;

        for dep in &config.dependencies {
            match dep.dependency_type {
                crate::DependencyType::Database => db_count += 1,
                crate::DependencyType::Cache => cache_count += 1,
                crate::DependencyType::WebFramework => web_count += 1,
                crate::DependencyType::MLFramework => ml_count += 1,
                crate::DependencyType::MessageQueue => message_queue_count += 1,
                _ => {}
            }
        }

        DependencyFeatures {
            database_count: db_count,
            cache_count,
            web_framework_count: web_count,
            ml_framework_count: ml_count,
            message_queue_count,
            total_dependencies: config.dependencies.len(),
            web_frameworks: web_count,
            databases: db_count,
            caches: cache_count,
            ml_frameworks: ml_count,
            message_queues: message_queue_count,
        }
    }

    pub(crate) fn extract_resource_features(
        &self,
        config: &AgentConfiguration,
    ) -> ResourceFeatures {
        ResourceFeatures {
            cpu_cores: config.resources.cpu_cores,
            memory_gb: config.resources.memory_gb,
            gpu_units: config.resources.gpu_units,
            storage_gb: config.resources.storage_gb,
            network_bandwidth_mbps: config.resources.network_bandwidth_mbps,
            resource_intensity: self.calculate_resource_intensity(&config.resources),
        }
    }

    pub(crate) fn extract_scaling_features(&self, config: &AgentConfiguration) -> ScalingFeatures {
        ScalingFeatures {
            min_replicas: config.scaling.min_replicas,
            max_replicas: config.scaling.max_replicas,
            target_cpu_percent: config.scaling.target_cpu_percent,
            scale_up_threshold: config.scaling.scale_up_threshold,
            scale_down_threshold: config.scaling.scale_down_threshold,
            scaling_aggressiveness: self.calculate_scaling_aggressiveness(&config.scaling),
        }
    }

    pub(crate) fn extract_personality_features(
        &self,
        config: &AgentConfiguration,
    ) -> PersonalityFeatures {
        PersonalityFeatures {
            risk_tolerance: config.personality.risk_tolerance,
            cooperation: config.personality.cooperation,
            exploration: config.personality.exploration,
            efficiency_focus: config.personality.efficiency_focus,
            stability_preference: config.personality.stability_preference,
            personality_type: self.classify_personality_type(&config.personality),
        }
    }

    pub(crate) fn calculate_resource_intensity(
        &self,
        resources: &crate::ResourceRequirements,
    ) -> f32 {
        // Normalized resource intensity score
        let cpu_score = resources.cpu_cores / 8.0; // Normalize to 8 cores
        let memory_score = resources.memory_gb / 16.0; // Normalize to 16GB
        let gpu_score = resources.gpu_units; // Already normalized
        let storage_score = resources.storage_gb / 100.0; // Normalize to 100GB
        let network_score = resources.network_bandwidth_mbps / 1000.0; // Normalize to 1Gbps

        (cpu_score + memory_score + gpu_score + storage_score + network_score) / 5.0
    }

    pub(crate) fn calculate_scaling_aggressiveness(&self, scaling: &crate::ScalingPolicy) -> f32 {
        let replica_range = scaling.max_replicas - scaling.min_replicas;
        let threshold_gap = scaling.scale_up_threshold - scaling.scale_down_threshold;

        // More aggressive scaling if wider replica range and tighter thresholds
        let replica_factor = (replica_range as f32).min(10.0) / 10.0;
        let threshold_factor = (100.0 - threshold_gap) / 100.0;

        (replica_factor + threshold_factor) / 2.0
    }

    pub(crate) fn classify_personality_type(
        &self,
        personality: &crate::PersonalityConfig,
    ) -> String {
        // Simple personality classification based on dominant traits
        let traits = vec![
            ("risk_tolerant", personality.risk_tolerance),
            ("cooperative", personality.cooperation),
            ("explorer", personality.exploration),
            ("efficient", personality.efficiency_focus),
            ("stable", personality.stability_preference),
        ];

        traits
            .into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(name, _)| name.to_string())
            .unwrap_or_else(|| "balanced".to_string())
    }
}
