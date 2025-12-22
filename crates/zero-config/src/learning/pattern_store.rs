//! Pattern storage and management

use super::types::DeploymentPattern;
use std::collections::HashMap;

/// Storage for learned deployment patterns
#[derive(Debug, Clone)]
pub struct PatternStore {
    pub patterns: Vec<DeploymentPattern>,
}

impl PatternStore {
    /// Create a new pattern store
    pub fn new() -> Self {
        Self {
            patterns: Vec::new(),
        }
    }

    /// Add a new pattern to the store
    pub fn add_pattern(&mut self, pattern: DeploymentPattern) {
        // Check if pattern already exists based on similar features
        let exists = self.patterns.iter().any(|p| {
            p.language == pattern.language
                && p.framework == pattern.framework
                && p.config.dependencies.len() == pattern.config.dependencies.len()
                && (p.cpu_cores - pattern.cpu_cores).abs() < 0.1
                && (p.memory_gb - pattern.memory_gb).abs() < 0.1
        });

        if !exists {
            self.patterns.push(pattern);
        }
    }

    /// Get patterns for a specific language
    pub fn get_patterns_by_language(&self, language: &str) -> Vec<&DeploymentPattern> {
        self.patterns
            .iter()
            .filter(|p| p.language == language)
            .collect()
    }

    /// Calculate overall success rate
    pub fn calculate_success_rate(&self) -> f32 {
        if self.patterns.is_empty() {
            return 0.0;
        }

        let successful = self.patterns.iter().filter(|p| p.success).count() as f32;
        successful / self.patterns.len() as f32
    }

    /// Calculate average confidence
    pub fn calculate_average_confidence(&self) -> f32 {
        if self.patterns.is_empty() {
            return 0.0;
        }

        let total_confidence: f32 = self.patterns.iter().map(|p| p.confidence).sum();
        total_confidence / self.patterns.len() as f32
    }

    /// Get language distribution
    pub fn get_language_distribution(&self) -> HashMap<String, usize> {
        let mut distribution = HashMap::new();
        for pattern in &self.patterns {
            *distribution.entry(pattern.language.clone()).or_insert(0) += 1;
        }
        distribution
    }

    /// Get framework distribution
    pub fn get_framework_distribution(&self) -> HashMap<String, usize> {
        let mut distribution = HashMap::new();
        for pattern in &self.patterns {
            if let Some(framework) = &pattern.framework {
                *distribution.entry(framework.clone()).or_insert(0) += 1;
            }
        }
        distribution
    }
}

impl Default for PatternStore {
    fn default() -> Self {
        Self::new()
    }
}
