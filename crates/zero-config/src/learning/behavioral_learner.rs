//! Main behavioral learning system

use super::{
    DeploymentOutcome, DeploymentPattern, KnowledgeTransfer, LearningStatistics, PatternCollector,
    PatternStore,
};
use crate::{analysis::CodebaseAnalysis, Result};
use std::sync::Arc;
use tokio::sync::RwLock;

/// Main behavioral learning system
pub struct BehavioralLearner {
    pub(crate) pattern_collector: PatternCollector,
    pub(crate) knowledge_transfer: KnowledgeTransfer,
    pub(crate) pattern_store: Arc<RwLock<PatternStore>>,
}

impl BehavioralLearner {
    /// Create a new behavioral learner
    pub fn new() -> Self {
        let pattern_store = Arc::new(RwLock::new(PatternStore::new()));

        Self {
            pattern_collector: PatternCollector::new(pattern_store.clone()),
            knowledge_transfer: KnowledgeTransfer::new(pattern_store.clone()),
            pattern_store,
        }
    }

    /// Record a successful deployment for learning
    pub async fn record_deployment(&mut self, deployment: DeploymentOutcome) -> Result<()> {
        self.pattern_collector.collect_pattern(deployment).await
    }

    /// Find similar patterns for a given codebase analysis
    pub async fn find_similar_patterns(
        &self,
        analysis: &CodebaseAnalysis,
    ) -> Result<Vec<DeploymentPattern>> {
        self.knowledge_transfer
            .find_similar_patterns(analysis)
            .await
    }

    /// Get learning statistics
    pub async fn get_statistics(&self) -> Result<LearningStatistics> {
        let store = self.pattern_store.read().await;
        Ok(LearningStatistics {
            total_patterns: store.patterns.len(),
            languages: store.get_language_distribution(),
            frameworks: store.get_framework_distribution(),
            success_rate: store.calculate_success_rate(),
            avg_confidence: store.calculate_average_confidence(),
        })
    }

    /// Clear all learned patterns (for testing or reset)
    pub async fn clear_patterns(&mut self) -> Result<()> {
        let mut store = self.pattern_store.write().await;
        store.patterns.clear();
        Ok(())
    }
}

impl Default for BehavioralLearner {
    fn default() -> Self {
        Self::new()
    }
}
