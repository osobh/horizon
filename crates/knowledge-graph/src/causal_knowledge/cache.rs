//! Caching system for causal inference results

use std::collections::HashMap;

use super::types::CausalRelationship;
use super::chains::CausalChain;
use super::evidence::CounterfactualAnalysis;
use super::patterns::TemporalCausalPattern;

/// Cache for causal inference results
pub struct InferenceCache {
    pub causal_relationships: HashMap<String, CausalRelationship>,
    pub causal_chains: HashMap<String, CausalChain>,
    pub counterfactuals: HashMap<String, CounterfactualAnalysis>,
    pub temporal_patterns: HashMap<String, TemporalCausalPattern>,
}

impl InferenceCache {
    pub fn new() -> Self {
        Self {
            causal_relationships: HashMap::new(),
            causal_chains: HashMap::new(),
            counterfactuals: HashMap::new(),
            temporal_patterns: HashMap::new(),
        }
    }

    pub fn cache_relationship(&mut self, relationship: CausalRelationship) {
        self.causal_relationships.insert(relationship.id.clone(), relationship);
    }

    pub fn get_relationship(&self, id: &str) -> Option<&CausalRelationship> {
        self.causal_relationships.get(id)
    }

    pub fn cache_chain(&mut self, chain: CausalChain) {
        self.causal_chains.insert(chain.id.clone(), chain);
    }

    pub fn get_chain(&self, id: &str) -> Option<&CausalChain> {
        self.causal_chains.get(id)
    }

    pub fn cache_counterfactual(&mut self, analysis: CounterfactualAnalysis, key: String) {
        self.counterfactuals.insert(key, analysis);
    }

    pub fn get_counterfactual(&self, key: &str) -> Option<&CounterfactualAnalysis> {
        self.counterfactuals.get(key)
    }

    pub fn cache_pattern(&mut self, pattern: TemporalCausalPattern) {
        self.temporal_patterns.insert(pattern.pattern_id.clone(), pattern);
    }

    pub fn get_pattern(&self, id: &str) -> Option<&TemporalCausalPattern> {
        self.temporal_patterns.get(id)
    }

    pub fn clear(&mut self) {
        self.causal_relationships.clear();
        self.causal_chains.clear();
        self.counterfactuals.clear();
        self.temporal_patterns.clear();
    }

    pub fn size(&self) -> usize {
        self.causal_relationships.len() + 
        self.causal_chains.len() + 
        self.counterfactuals.len() + 
        self.temporal_patterns.len()
    }
}
