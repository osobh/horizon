//! Similarity calculation engine for pattern matching

use crate::{analysis::CodebaseAnalysis, Result};
use std::collections::HashMap;

/// Similarity calculation engine
pub struct SimilarityEngine;

impl SimilarityEngine {
    /// Create a new similarity engine
    pub fn new() -> Self {
        Self
    }

    /// Calculate similarity between codebase analysis and agent configuration
    pub async fn calculate_codebase_similarity(
        &self,
        analysis: &CodebaseAnalysis,
        config: &crate::AgentConfiguration,
    ) -> Result<f32> {
        let language_sim = self.calculate_language_similarity(&analysis.language, &config.language);
        let framework_sim =
            self.calculate_framework_similarity(&analysis.framework, &config.framework);
        let dependency_sim =
            self.calculate_dependency_similarity(&analysis.dependencies, &config.dependencies)?;
        let resource_sim =
            self.calculate_resource_similarity(&analysis.resources, &config.resources)?;

        // Weighted similarity calculation
        let weights = [0.3, 0.2, 0.3, 0.2]; // Language, Framework, Dependencies, Resources
        let similarities = [language_sim, framework_sim, dependency_sim, resource_sim];

        let weighted_sum: f32 = weights
            .iter()
            .zip(similarities.iter())
            .map(|(w, s)| w * s)
            .sum();

        Ok(weighted_sum)
    }

    /// Calculate language similarity
    fn calculate_language_similarity(&self, lang1: &str, lang2: &str) -> f32 {
        if lang1 == lang2 {
            1.0
        } else {
            // Check for language family similarities
            let similar_languages = [
                (vec!["javascript", "typescript"], 0.8),
                (vec!["c", "cpp", "c++"], 0.7),
                (vec!["python", "ruby"], 0.6),
            ];

            for (langs, sim) in similar_languages {
                if langs.contains(&lang1) && langs.contains(&lang2) {
                    return sim;
                }
            }

            0.0
        }
    }

    /// Calculate framework similarity
    fn calculate_framework_similarity(&self, fw1: &Option<String>, fw2: &Option<String>) -> f32 {
        match (fw1, fw2) {
            (Some(f1), Some(f2)) => {
                if f1 == f2 {
                    1.0
                } else {
                    // Check for similar framework types
                    let web_frameworks =
                        vec!["express", "fastapi", "django", "flask", "actix-web", "warp"];
                    let ml_frameworks = vec!["tensorflow", "pytorch", "scikit-learn"];

                    if web_frameworks.contains(&f1.as_str())
                        && web_frameworks.contains(&f2.as_str())
                    {
                        0.6
                    } else if ml_frameworks.contains(&f1.as_str())
                        && ml_frameworks.contains(&f2.as_str())
                    {
                        0.7
                    } else {
                        0.2
                    }
                }
            }
            (None, None) => 1.0,
            _ => 0.5,
        }
    }

    /// Calculate dependency similarity
    fn calculate_dependency_similarity(
        &self,
        deps1: &[crate::Dependency],
        deps2: &[crate::Dependency],
    ) -> Result<f32> {
        if deps1.is_empty() && deps2.is_empty() {
            return Ok(1.0);
        }

        let deps1_types: HashMap<crate::DependencyType, usize> = self.count_dependency_types(deps1);
        let deps2_types: HashMap<crate::DependencyType, usize> = self.count_dependency_types(deps2);

        let mut total_similarity = 0.0;
        let mut type_count = 0;

        let all_types = [
            crate::DependencyType::Database,
            crate::DependencyType::Cache,
            crate::DependencyType::WebFramework,
            crate::DependencyType::MLFramework,
            crate::DependencyType::MessageQueue,
        ];

        for dep_type in &all_types {
            let count1 = deps1_types.get(dep_type).unwrap_or(&0);
            let count2 = deps2_types.get(dep_type).unwrap_or(&0);

            let max_count = (*count1).max(*count2).max(1);
            let min_count = (*count1).min(*count2);

            total_similarity += min_count as f32 / max_count as f32;
            type_count += 1;
        }

        Ok(total_similarity / type_count as f32)
    }

    /// Count dependency types
    pub fn count_dependency_types(
        &self,
        deps: &[crate::Dependency],
    ) -> HashMap<crate::DependencyType, usize> {
        let mut counts = HashMap::new();
        for dep in deps {
            *counts.entry(dep.dependency_type.clone()).or_insert(0) += 1;
        }
        counts
    }

    /// Calculate resource similarity
    fn calculate_resource_similarity(
        &self,
        res1: &crate::ResourceRequirements,
        res2: &crate::ResourceRequirements,
    ) -> Result<f32> {
        let cpu_sim = self.calculate_numeric_similarity(res1.cpu_cores, res2.cpu_cores);
        let memory_sim = self.calculate_numeric_similarity(res1.memory_gb, res2.memory_gb);
        let gpu_sim = self.calculate_numeric_similarity(res1.gpu_units, res2.gpu_units);
        let storage_sim = self.calculate_numeric_similarity(res1.storage_gb, res2.storage_gb);
        let network_sim = self.calculate_numeric_similarity(
            res1.network_bandwidth_mbps / 100.0, // Scale down for comparison
            res2.network_bandwidth_mbps / 100.0,
        );

        Ok((cpu_sim + memory_sim + gpu_sim + storage_sim + network_sim) / 5.0)
    }

    /// Calculate similarity between two numeric values
    pub fn calculate_numeric_similarity(&self, val1: f32, val2: f32) -> f32 {
        // If both values are 0, they are perfectly similar
        if val1 == 0.0 && val2 == 0.0 {
            return 1.0;
        }

        let max_val = val1.max(val2).max(0.1);
        let min_val = val1.min(val2);
        min_val / max_val
    }
}

impl Default for SimilarityEngine {
    fn default() -> Self {
        Self::new()
    }
}
