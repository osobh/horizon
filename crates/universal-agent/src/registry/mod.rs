//! Agent Registry for service discovery and version management

use crate::dna::{AgentDNA, DNAId, DNAVersion};
use crate::error::{Result, UniversalAgentError};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Registry for storing and discovering versioned AgentDNA
pub struct AgentRegistry {
    /// In-memory storage (can be replaced with persistent storage)
    storage: Arc<RwLock<HashMap<DNAId, AgentDNA>>>,
    /// Version history per DNA
    version_history: Arc<RwLock<HashMap<DNAId, Vec<DNAVersion>>>>,
    /// Capability index
    capability_index: Arc<RwLock<CapabilityIndex>>,
}

impl AgentRegistry {
    pub fn new() -> Self {
        Self {
            storage: Arc::new(RwLock::new(HashMap::new())),
            version_history: Arc::new(RwLock::new(HashMap::new())),
            capability_index: Arc::new(RwLock::new(CapabilityIndex::new())),
        }
    }

    /// Register a new DNA
    pub async fn register(&self, dna: AgentDNA) -> Result<DNAId> {
        let id = dna.id;

        // Validate DNA
        dna.validate()
            .map_err(|errors| UniversalAgentError::ValidationFailed(errors.join("; ")))?;

        // Store DNA
        self.storage.write().await.insert(id, dna.clone());

        // Update version history
        self.version_history
            .write()
            .await
            .entry(id)
            .or_default()
            .push(dna.version.clone());

        // Update capability index
        self.capability_index.write().await.index_dna(&dna);

        Ok(id)
    }

    /// Find the best DNA for a capability
    pub async fn find_best_for_capability(
        &self,
        capability_id: &str,
        requirements: &CapabilityRequirements,
    ) -> Result<Option<AgentDNA>> {
        let index = self.capability_index.read().await;
        let storage = self.storage.read().await;

        let candidates = index.find_by_capability(capability_id);
        if candidates.is_empty() {
            return Ok(None);
        }

        // Score and rank candidates
        let mut scored: Vec<(DNAId, f64)> = candidates
            .iter()
            .filter_map(|id| {
                storage
                    .get(id)
                    .map(|dna| (*id, self.score_dna(dna, requirements)))
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        if let Some((best_id, _)) = scored.first() {
            Ok(storage.get(best_id).cloned())
        } else {
            Ok(None)
        }
    }

    /// Find DNA by tags
    pub async fn find_by_tags(&self, tags: &[String]) -> Result<Vec<AgentDNA>> {
        let storage = self.storage.read().await;
        let results: Vec<AgentDNA> = storage
            .values()
            .filter(|dna| tags.iter().all(|tag| dna.tags.contains(tag)))
            .cloned()
            .collect();
        Ok(results)
    }

    /// Get the latest version of DNA
    pub async fn get_latest(&self, id: DNAId) -> Result<Option<AgentDNA>> {
        let storage = self.storage.read().await;
        Ok(storage.get(&id).cloned())
    }

    /// Get a specific version of DNA
    pub async fn get_version(&self, id: DNAId, version: &DNAVersion) -> Result<Option<AgentDNA>> {
        let storage = self.storage.read().await;
        if let Some(dna) = storage.get(&id) {
            if &dna.version == version {
                return Ok(Some(dna.clone()));
            }
        }
        // TODO: Implement historical version storage
        Ok(None)
    }

    /// Get version history for a DNA
    pub async fn get_version_history(&self, id: DNAId) -> Result<Vec<DNAVersion>> {
        let history = self.version_history.read().await;
        Ok(history.get(&id).cloned().unwrap_or_default())
    }

    /// Upgrade a DNA to a new version
    pub async fn upgrade(&self, id: DNAId, new_dna: AgentDNA) -> Result<DNAVersion> {
        let current = self
            .get_latest(id)
            .await?
            .ok_or(UniversalAgentError::DNANotFound(id.0))?;

        if new_dna.version.generation <= current.version.generation {
            return Err(UniversalAgentError::InvalidVersionUpgrade {
                current: current.version.clone(),
                proposed: new_dna.version.clone(),
            });
        }

        let version = new_dna.version.clone();

        // Store new version
        self.storage.write().await.insert(id, new_dna.clone());

        // Update history
        self.version_history
            .write()
            .await
            .entry(id)
            .or_default()
            .push(version.clone());

        // Re-index capabilities
        self.capability_index.write().await.index_dna(&new_dna);

        Ok(version)
    }

    /// List all registered DNA
    pub async fn list_all(&self) -> Result<Vec<AgentDNA>> {
        let storage = self.storage.read().await;
        Ok(storage.values().cloned().collect())
    }

    /// Score a DNA against requirements
    fn score_dna(&self, dna: &AgentDNA, requirements: &CapabilityRequirements) -> f64 {
        let mut score = 0.0;

        // Fitness contribution
        score += dna.fitness.overall * 0.4;

        // Benchmark evidence contribution
        if let Some(ref benchmark) = dna.benchmarks.latest {
            if benchmark.passed {
                score += 0.2;
            }
            score += (benchmark.total_score / 100.0).min(0.2);
        }

        // Capability quality score
        for cap in &dna.capabilities {
            if cap.id == requirements.capability_id {
                score += cap.quality_score * 0.2;
                if cap.avg_latency_ms <= requirements.max_latency_ms {
                    score += 0.1;
                }
            }
        }

        // Skill proficiency
        score += (dna.total_proficiency() / 10.0) * 0.1;

        score
    }
}

impl Default for AgentRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Requirements for finding a capable agent
#[derive(Debug, Clone)]
pub struct CapabilityRequirements {
    pub capability_id: String,
    pub min_quality_score: f64,
    pub max_latency_ms: u64,
    pub required_skills: Vec<String>,
    pub tags: Vec<String>,
}

impl CapabilityRequirements {
    pub fn new(capability_id: impl Into<String>) -> Self {
        Self {
            capability_id: capability_id.into(),
            min_quality_score: 0.5,
            max_latency_ms: 5000,
            required_skills: Vec::new(),
            tags: Vec::new(),
        }
    }

    pub fn with_min_quality(mut self, score: f64) -> Self {
        self.min_quality_score = score;
        self
    }

    pub fn with_max_latency(mut self, ms: u64) -> Self {
        self.max_latency_ms = ms;
        self
    }
}

/// Capability index for fast lookup
#[derive(Debug, Default)]
pub struct CapabilityIndex {
    by_capability: HashMap<String, Vec<DNAId>>,
    by_tag: HashMap<String, Vec<DNAId>>,
}

impl CapabilityIndex {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn index_dna(&mut self, dna: &AgentDNA) {
        // Remove old entries for this DNA
        self.remove_dna(dna.id);

        // Index by capability
        for cap in &dna.capabilities {
            self.by_capability
                .entry(cap.id.clone())
                .or_default()
                .push(dna.id);
        }

        // Index by tag
        for tag in &dna.tags {
            self.by_tag.entry(tag.clone()).or_default().push(dna.id);
        }
    }

    pub fn remove_dna(&mut self, id: DNAId) {
        for ids in self.by_capability.values_mut() {
            ids.retain(|i| *i != id);
        }
        for ids in self.by_tag.values_mut() {
            ids.retain(|i| *i != id);
        }
    }

    pub fn find_by_capability(&self, capability_id: &str) -> Vec<DNAId> {
        self.by_capability
            .get(capability_id)
            .cloned()
            .unwrap_or_default()
    }

    pub fn find_by_tag(&self, tag: &str) -> Vec<DNAId> {
        self.by_tag.get(tag).cloned().unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dna::AgentDNA;

    #[tokio::test]
    async fn test_registry_register_and_find() {
        let registry = AgentRegistry::new();
        let dna = AgentDNA::efficiency_hunter_template();
        let id = registry.register(dna).await.unwrap();

        let requirements = CapabilityRequirements::new("efficiency_analysis");
        let found = registry
            .find_best_for_capability("efficiency_analysis", &requirements)
            .await
            .unwrap();

        assert!(found.is_some());
        assert_eq!(found.unwrap().id, id);
    }

    #[tokio::test]
    async fn test_registry_find_by_tags() {
        let registry = AgentRegistry::new();
        let dna = AgentDNA::efficiency_hunter_template();
        registry.register(dna).await.unwrap();

        let results = registry
            .find_by_tags(&["efficiency".to_string()])
            .await
            .unwrap();
        assert_eq!(results.len(), 1);

        let results = registry
            .find_by_tags(&["nonexistent".to_string()])
            .await
            .unwrap();
        assert_eq!(results.len(), 0);
    }
}
