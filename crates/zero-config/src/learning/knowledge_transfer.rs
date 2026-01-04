//! Knowledge transfer and pattern similarity matching

use super::{
    DependencyFeatures, DeploymentPattern, FeatureVector, LanguageFeatures, PatternStore,
    PersonalityFeatures, ResourceFeatures, ScalingFeatures,
};
use crate::{analysis::CodebaseAnalysis, Result};
use std::sync::Arc;
use tokio::sync::RwLock;

/// Knowledge transfer system
pub struct KnowledgeTransfer {
    pub(crate) pattern_store: Arc<RwLock<PatternStore>>,
}

impl KnowledgeTransfer {
    pub(crate) fn new(pattern_store: Arc<RwLock<PatternStore>>) -> Self {
        Self { pattern_store }
    }

    /// Find similar patterns for a given codebase analysis
    pub async fn find_similar_patterns(
        &self,
        analysis: &CodebaseAnalysis,
    ) -> Result<Vec<DeploymentPattern>> {
        let store = self.pattern_store.read().await;
        let query_features = self.create_query_features(analysis).await?;

        let mut similarities = Vec::new();

        for pattern in &store.patterns {
            let similarity = self
                .calculate_similarity(&query_features, &pattern.features)
                .await?;
            if similarity > 0.3 {
                // Minimum similarity threshold
                similarities.push((pattern.clone(), similarity));
            }
        }

        // Sort by similarity (descending) and return top matches
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        Ok(similarities
            .into_iter()
            .take(10) // Return top 10 matches
            .map(|(pattern, _)| pattern)
            .collect())
    }

    /// Create query features from codebase analysis
    async fn create_query_features(&self, analysis: &CodebaseAnalysis) -> Result<FeatureVector> {
        Ok(FeatureVector {
            language_features: LanguageFeatures {
                language: analysis.language.clone(),
                framework: analysis.framework.clone(),
                has_web_framework: analysis
                    .dependencies
                    .iter()
                    .any(|d| matches!(d.dependency_type, crate::DependencyType::WebFramework)),
                has_database: analysis
                    .dependencies
                    .iter()
                    .any(|d| matches!(d.dependency_type, crate::DependencyType::Database)),
                has_cache: analysis
                    .dependencies
                    .iter()
                    .any(|d| matches!(d.dependency_type, crate::DependencyType::Cache)),
                has_ml_framework: analysis
                    .dependencies
                    .iter()
                    .any(|d| matches!(d.dependency_type, crate::DependencyType::MLFramework)),
                has_message_queue: analysis
                    .dependencies
                    .iter()
                    .any(|d| matches!(d.dependency_type, crate::DependencyType::MessageQueue)),
                dependency_count: analysis.dependencies.len(),
                total_lines: analysis.total_lines,
                complexity_score: analysis.complexity.cyclomatic_complexity,
            },
            dependency_features: self.create_dependency_features(&analysis.dependencies),
            resource_features: ResourceFeatures {
                cpu_cores: analysis.resources.cpu_cores,
                memory_gb: analysis.resources.memory_gb,
                gpu_units: analysis.resources.gpu_units,
                storage_gb: analysis.resources.storage_gb,
                network_bandwidth_mbps: analysis.resources.network_bandwidth_mbps,
                resource_intensity: self.calculate_query_resource_intensity(&analysis.resources),
            },
            scaling_features: ScalingFeatures {
                min_replicas: 1, // Default values for query
                max_replicas: 10,
                target_cpu_percent: 70.0,
                scale_up_threshold: 80.0,
                scale_down_threshold: 30.0,
                scaling_aggressiveness: 0.5,
            },
            personality_features: PersonalityFeatures {
                risk_tolerance: 0.5, // Default balanced personality for query
                cooperation: 0.7,
                exploration: 0.3,
                efficiency_focus: 0.6,
                stability_preference: 0.8,
                personality_type: "balanced".to_string(),
            },
        })
    }

    fn create_dependency_features(&self, dependencies: &[crate::Dependency]) -> DependencyFeatures {
        let mut db_count = 0;
        let mut cache_count = 0;
        let mut web_count = 0;
        let mut ml_count = 0;
        let mut message_queue_count = 0;

        for dep in dependencies {
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
            total_dependencies: dependencies.len(),
            web_frameworks: web_count,
            databases: db_count,
            caches: cache_count,
            ml_frameworks: ml_count,
            message_queues: message_queue_count,
        }
    }

    fn calculate_query_resource_intensity(&self, resources: &crate::ResourceRequirements) -> f32 {
        let cpu_score = resources.cpu_cores / 8.0;
        let memory_score = resources.memory_gb / 16.0;
        let gpu_score = resources.gpu_units;
        let storage_score = resources.storage_gb / 100.0;
        let network_score = resources.network_bandwidth_mbps / 1000.0;

        (cpu_score + memory_score + gpu_score + storage_score + network_score) / 5.0
    }

    /// Calculate similarity between query features and pattern features
    pub(crate) async fn calculate_similarity(
        &self,
        query: &FeatureVector,
        pattern: &FeatureVector,
    ) -> Result<f32> {
        let language_sim = self
            .calculate_language_similarity(&query.language_features, &pattern.language_features);
        let dependency_sim = self.calculate_dependency_similarity(
            &query.dependency_features,
            &pattern.dependency_features,
        );
        let resource_sim = self
            .calculate_resource_similarity(&query.resource_features, &pattern.resource_features);
        let scaling_sim =
            self.calculate_scaling_similarity(&query.scaling_features, &pattern.scaling_features);
        let personality_sim = self.calculate_personality_similarity(
            &query.personality_features,
            &pattern.personality_features,
        );

        // Weighted similarity calculation
        let weights = [0.3, 0.25, 0.2, 0.15, 0.1]; // Language, Dependencies, Resources, Scaling, Personality
        let similarities = [
            language_sim,
            dependency_sim,
            resource_sim,
            scaling_sim,
            personality_sim,
        ];

        let weighted_sum: f32 = weights
            .iter()
            .zip(similarities.iter())
            .map(|(w, s)| w * s)
            .sum();

        Ok(weighted_sum)
    }

    pub(crate) fn calculate_language_similarity(
        &self,
        query: &LanguageFeatures,
        pattern: &LanguageFeatures,
    ) -> f32 {
        let mut score = 0.0;

        // Exact language match is very important
        if query.language == pattern.language {
            score += 0.4;
        }

        // Framework match is also important
        if query.framework == pattern.framework {
            score += 0.2;
        }

        // Boolean feature matches
        let bool_matches = [
            (query.has_web_framework, pattern.has_web_framework),
            (query.has_database, pattern.has_database),
            (query.has_cache, pattern.has_cache),
            (query.has_ml_framework, pattern.has_ml_framework),
            (query.has_message_queue, pattern.has_message_queue),
        ];

        let bool_score: f32 = bool_matches
            .iter()
            .map(|(q, p)| if q == p { 0.08 } else { 0.0 })
            .sum();

        score += bool_score;

        score.min(1.0)
    }

    fn calculate_dependency_similarity(
        &self,
        query: &DependencyFeatures,
        pattern: &DependencyFeatures,
    ) -> f32 {
        // Calculate similarity based on dependency counts
        let features = [
            (query.database_count, pattern.database_count),
            (query.cache_count, pattern.cache_count),
            (query.web_framework_count, pattern.web_framework_count),
            (query.ml_framework_count, pattern.ml_framework_count),
            (query.message_queue_count, pattern.message_queue_count),
        ];

        let mut total_similarity = 0.0;
        for (q, p) in features {
            let max_val = q.max(p).max(1);
            let min_val = q.min(p);
            total_similarity += min_val as f32 / max_val as f32;
        }

        total_similarity / features.len() as f32
    }

    fn calculate_resource_similarity(
        &self,
        query: &ResourceFeatures,
        pattern: &ResourceFeatures,
    ) -> f32 {
        let features = [
            (query.cpu_cores, pattern.cpu_cores),
            (query.memory_gb, pattern.memory_gb),
            (query.gpu_units, pattern.gpu_units),
            (query.storage_gb, pattern.storage_gb),
            (
                query.network_bandwidth_mbps / 100.0,
                pattern.network_bandwidth_mbps / 100.0,
            ), // Scale down for comparison
        ];

        let mut total_similarity = 0.0;
        for (q, p) in features {
            let max_val = q.max(p).max(0.1);
            let min_val = q.min(p);
            total_similarity += min_val / max_val;
        }

        total_similarity / features.len() as f32
    }

    fn calculate_scaling_similarity(
        &self,
        query: &ScalingFeatures,
        pattern: &ScalingFeatures,
    ) -> f32 {
        let features = [
            (
                (query.min_replicas as f32).log2(),
                (pattern.min_replicas as f32).log2(),
            ),
            (
                (query.max_replicas as f32).log2(),
                (pattern.max_replicas as f32).log2(),
            ),
            (
                query.target_cpu_percent / 100.0,
                pattern.target_cpu_percent / 100.0,
            ),
            (query.scaling_aggressiveness, pattern.scaling_aggressiveness),
        ];

        let mut total_similarity = 0.0;
        for (q, p) in features {
            let diff = (q - p).abs();
            total_similarity += 1.0 - diff.min(1.0);
        }

        total_similarity / features.len() as f32
    }

    fn calculate_personality_similarity(
        &self,
        query: &PersonalityFeatures,
        pattern: &PersonalityFeatures,
    ) -> f32 {
        let features = [
            (query.risk_tolerance, pattern.risk_tolerance),
            (query.cooperation, pattern.cooperation),
            (query.exploration, pattern.exploration),
            (query.efficiency_focus, pattern.efficiency_focus),
            (query.stability_preference, pattern.stability_preference),
        ];

        let mut total_similarity = 0.0;
        for (q, p) in features {
            let diff = (q - p).abs();
            total_similarity += 1.0 - diff;
        }

        total_similarity / features.len() as f32
    }
}
