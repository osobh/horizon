//! Multi-hop reasoning for knowledge graphs
//!
//! This module provides logical reasoning capabilities including:
//! - Multi-hop query execution
//! - Inference rule application  
//! - Logical reasoning chains
//! - Probabilistic reasoning

use super::{KnowledgeEdge, KnowledgeNode};
use anyhow::{anyhow, Result};
use cudarc::driver::{CudaContext, CudaSlice, DeviceSlice};
use dashmap::DashMap;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex};

/// Object in a logical fact
#[derive(Debug, Clone, PartialEq)]
pub enum FactObject {
    Node(u32),
    Value(String),
}

/// Logical fact (subject-predicate-object triple)
#[derive(Debug, Clone)]
pub struct LogicalFact {
    pub subject: u32,
    pub predicate: String,
    pub object: FactObject,
    pub confidence: f32,
}

/// Rule pattern for inference
#[derive(Debug, Clone)]
pub enum RulePattern {
    /// If A->B and B->C then A->C
    Transitive { relation_type: String },
    /// If A is_a B and B has_property P, then A has_property P
    PropertyInheritance {
        relation_type: String,
        property_type: String,
    },
    /// Detect contradictions
    Contradiction {
        predicate: String,
        conflicting_values: Vec<u32>,
    },
    /// Analogical reasoning
    Analogy {
        source_domain: String,
        target_domain: String,
        relation_mapping: Vec<(String, String)>,
    },
    /// Custom rule
    Custom { description: String },
}

/// Inference rule
#[derive(Debug, Clone)]
pub struct InferenceRule {
    pub id: u32,
    pub name: String,
    pub pattern: RulePattern,
    pub confidence: f32,
    pub priority: u32,
}

/// Query type for reasoning
#[derive(Debug, Clone)]
pub enum QueryType {
    /// What properties does entity have?
    WhatProperties { entity_id: u32 },
    /// What can entity do?
    WhatCan { entity_id: u32 },
    /// Is there a relation between entities?
    IsRelated {
        subject_id: u32,
        predicate: String,
        object_id: u32,
    },
    /// Explain how a fact can be inferred
    Explain { fact: LogicalFact },
}

/// Reasoning query
#[derive(Debug, Clone)]
pub struct ReasoningQuery {
    pub query_type: QueryType,
    pub max_hops: u32,
    pub min_confidence: f32,
    pub include_inferred: bool,
}

/// Reasoning result
#[derive(Debug, Clone)]
pub struct ReasoningResult {
    pub conclusions: Vec<LogicalFact>,
    pub explanation_chains: Vec<ExplanationChain>,
    pub confidence_scores: Vec<f32>,
}

/// Explanation chain for a conclusion
#[derive(Debug, Clone)]
pub struct ExplanationChain {
    pub conclusion: LogicalFact,
    pub steps: Vec<ReasoningStep>,
    pub total_confidence: f32,
}

/// Single step in reasoning
#[derive(Debug, Clone)]
pub struct ReasoningStep {
    pub fact: LogicalFact,
    pub rule_used: Option<String>,
    pub confidence: f32,
}

/// Contradiction in knowledge base
#[derive(Debug, Clone)]
pub struct Contradiction {
    pub entity_id: u32,
    pub predicate: String,
    pub conflicting_facts: Vec<LogicalFact>,
    pub confidence_difference: f32,
}

/// GPU reasoning data
struct GpuReasoningData {
    fact_subjects: CudaSlice<u32>,
    fact_predicates: CudaSlice<u32>, // Hashed predicates
    fact_objects: CudaSlice<u32>,    // Node IDs only
    fact_confidences: CudaSlice<f32>,
    rule_patterns: CudaSlice<u32>, // Encoded patterns
}

/// Reasoning engine for multi-hop queries
pub struct ReasoningEngine {
    device: Arc<CudaContext>,
    facts: Arc<Mutex<Vec<LogicalFact>>>,
    rules: Arc<Mutex<Vec<InferenceRule>>>,
    fact_index: Arc<DashMap<u32, Vec<usize>>>, // Subject -> fact indices
    gpu_data: Option<GpuReasoningData>,
    max_facts: usize,
}

impl ReasoningEngine {
    /// Create new reasoning engine
    pub fn new(device: Arc<CudaContext>, max_facts: usize) -> Result<Self> {
        Ok(Self {
            device,
            facts: Arc::new(Mutex::new(Vec::new())),
            rules: Arc::new(Mutex::new(Vec::new())),
            fact_index: Arc::new(DashMap::new()),
            gpu_data: None,
            max_facts,
        })
    }

    /// Add inference rule
    pub fn add_rule(&mut self, rule: InferenceRule) -> Result<()> {
        let mut rules = self.rules.lock().map_err(|e| anyhow!("Lock poisoned: {}", e))?;
        rules.push(rule);
        rules.sort_by_key(|r| std::cmp::Reverse(r.priority));
        Ok(())
    }

    /// Add logical fact
    pub fn add_fact(&mut self, fact: LogicalFact) -> Result<()> {
        let mut facts = self.facts.lock().map_err(|e| anyhow!("Lock poisoned: {}", e))?;
        let fact_idx = facts.len();

        self.fact_index
            .entry(fact.subject)
            .or_insert_with(Vec::new)
            .push(fact_idx);

        facts.push(fact);
        Ok(())
    }

    /// Get rule count
    pub fn rule_count(&self) -> usize {
        self.rules.lock().map(|r| r.len()).unwrap_or(0)
    }

    /// Get fact count
    pub fn fact_count(&self) -> usize {
        self.facts.lock().map(|f| f.len()).unwrap_or(0)
    }

    /// Perform reasoning
    pub fn reason(&self, query: ReasoningQuery) -> Result<ReasoningResult> {
        match query.query_type {
            QueryType::WhatProperties { entity_id } => self.reason_properties(
                entity_id,
                query.max_hops,
                query.min_confidence,
                query.include_inferred,
            ),
            QueryType::WhatCan { entity_id } => self.reason_capabilities(
                entity_id,
                query.max_hops,
                query.min_confidence,
                query.include_inferred,
            ),
            QueryType::IsRelated {
                subject_id,
                ref predicate,
                object_id,
            } => self.reason_relation(
                subject_id,
                predicate,
                object_id,
                query.max_hops,
                query.min_confidence,
            ),
            QueryType::Explain { ref fact } => self.explain_fact(fact, query.max_hops),
        }
    }

    /// Reason about entity properties
    fn reason_properties(
        &self,
        entity_id: u32,
        max_hops: u32,
        min_confidence: f32,
        include_inferred: bool,
    ) -> Result<ReasoningResult> {
        let mut conclusions = Vec::new();
        let explanation_chains = Vec::new();

        // Get direct facts
        let facts = self.facts.lock().map_err(|e| anyhow!("Lock poisoned: {}", e))?;

        if let Some(indices) = self.fact_index.get(&entity_id) {
            for &idx in indices.value() {
                let fact = &facts[idx];
                if fact.predicate == "has_property" && fact.confidence >= min_confidence {
                    conclusions.push(fact.clone());
                }
            }
        }

        if include_inferred {
            // Apply inference rules
            let rules = self.rules.lock().map_err(|e| anyhow!("Lock poisoned: {}", e))?;

            // Find parent classes
            let parent_chain = self.find_parent_chain(entity_id, max_hops)?;

            for rule in rules.iter() {
                match &rule.pattern {
                    RulePattern::PropertyInheritance {
                        relation_type: _,
                        property_type,
                    } => {
                        if property_type == "has_property" {
                            // Check each parent
                            for (parent_id, path_confidence) in &parent_chain {
                                if let Some(parent_indices) = self.fact_index.get(parent_id) {
                                    for &idx in parent_indices.value() {
                                        let parent_fact = &facts[idx];
                                        if parent_fact.predicate == *property_type {
                                            let inferred_confidence = parent_fact.confidence
                                                * path_confidence
                                                * rule.confidence;
                                            if inferred_confidence >= min_confidence {
                                                let inferred = LogicalFact {
                                                    subject: entity_id,
                                                    predicate: property_type.clone(),
                                                    object: parent_fact.object.clone(),
                                                    confidence: inferred_confidence,
                                                };
                                                conclusions.push(inferred);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        let confidence_scores = conclusions.iter().map(|f| f.confidence).collect();

        Ok(ReasoningResult {
            conclusions,
            explanation_chains,
            confidence_scores,
        })
    }

    /// Reason about entity capabilities
    fn reason_capabilities(
        &self,
        entity_id: u32,
        _max_hops: u32,
        min_confidence: f32,
        include_inferred: bool,
    ) -> Result<ReasoningResult> {
        let mut conclusions = Vec::new();
        let facts = self.facts.lock().map_err(|e| anyhow!("Lock poisoned: {}", e))?;

        if let Some(indices) = self.fact_index.get(&entity_id) {
            for &idx in indices.value() {
                let fact = &facts[idx];
                if fact.predicate == "can_do" && fact.confidence >= min_confidence {
                    conclusions.push(fact.clone());
                }
            }
        }

        if include_inferred {
            // Apply analogical reasoning
            let rules = self.rules.lock().map_err(|e| anyhow!("Lock poisoned: {}", e))?;
            for rule in rules.iter() {
                if let RulePattern::Analogy {
                    source_domain: _,
                    target_domain,
                    relation_mapping,
                } = &rule.pattern
                {
                    // Check if entity is in target domain
                    let empty_vec = vec![];
                    let indices = self
                        .fact_index
                        .get(&entity_id)
                        .map(|r| r.value().clone())
                        .unwrap_or(empty_vec);
                    for idx in indices {
                        let fact = &facts[idx];
                        if fact.predicate == "lives_in" {
                            if let FactObject::Value(domain) = &fact.object {
                                if domain == target_domain {
                                    // Find analogous capabilities
                                    for (_source_action, target_action) in relation_mapping {
                                        let inferred = LogicalFact {
                                            subject: entity_id,
                                            predicate: "can_do".to_string(),
                                            object: FactObject::Value(target_action.clone()),
                                            confidence: rule.confidence * 0.8,
                                        };
                                        if inferred.confidence >= min_confidence {
                                            conclusions.push(inferred);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        let confidence_scores = conclusions.iter().map(|f| f.confidence).collect();

        Ok(ReasoningResult {
            conclusions,
            explanation_chains: vec![],
            confidence_scores,
        })
    }

    /// Reason about relation between entities
    fn reason_relation(
        &self,
        subject_id: u32,
        predicate: &str,
        object_id: u32,
        max_hops: u32,
        min_confidence: f32,
    ) -> Result<ReasoningResult> {
        let mut conclusions = Vec::new();
        let facts = self.facts.lock().map_err(|e| anyhow!("Lock poisoned: {}", e))?;

        // Check direct relation
        if let Some(indices) = self.fact_index.get(&subject_id) {
            for &idx in indices.value() {
                let fact = &facts[idx];
                if fact.predicate == predicate && fact.object == FactObject::Node(object_id) {
                    conclusions.push(fact.clone());
                }
            }
        }

        // Apply transitive rules
        let rules = self.rules.lock().map_err(|e| anyhow!("Lock poisoned: {}", e))?;
        for rule in rules.iter() {
            if let RulePattern::Transitive { relation_type } = &rule.pattern {
                if relation_type == predicate {
                    // Find transitive path
                    if let Some(path) =
                        self.find_path(subject_id, object_id, predicate, max_hops)?
                    {
                        let path_confidence = path
                            .iter()
                            .map(|step| step.confidence)
                            .fold(1.0, |acc, c| acc * c)
                            * rule.confidence;

                        if path_confidence >= min_confidence {
                            let inferred = LogicalFact {
                                subject: subject_id,
                                predicate: predicate.to_string(),
                                object: FactObject::Node(object_id),
                                confidence: path_confidence,
                            };
                            conclusions.push(inferred);
                        }
                    }
                }
            }
        }

        let confidence_scores = conclusions.iter().map(|f| f.confidence).collect();

        Ok(ReasoningResult {
            conclusions,
            explanation_chains: vec![],
            confidence_scores,
        })
    }

    /// Explain how a fact can be inferred
    fn explain_fact(&self, target_fact: &LogicalFact, max_hops: u32) -> Result<ReasoningResult> {
        let mut explanation_chains = Vec::new();

        // Try to build explanation chain
        if let Some(chain) = self.build_explanation_chain(target_fact, max_hops)? {
            explanation_chains.push(chain);
        }

        Ok(ReasoningResult {
            conclusions: vec![target_fact.clone()],
            explanation_chains,
            confidence_scores: vec![target_fact.confidence],
        })
    }

    /// Find parent chain for entity
    fn find_parent_chain(&self, entity_id: u32, max_hops: u32) -> Result<Vec<(u32, f32)>> {
        let mut parents = Vec::new();
        let mut queue = VecDeque::new();
        let mut visited = HashSet::new();

        queue.push_back((entity_id, 1.0, 0));
        visited.insert(entity_id);

        let facts = self.facts.lock().map_err(|e| anyhow!("Lock poisoned: {}", e))?;

        while let Some((current, confidence, depth)) = queue.pop_front() {
            if depth >= max_hops {
                continue;
            }

            if let Some(indices) = self.fact_index.get(&current) {
                for &idx in indices.value() {
                    let fact = &facts[idx];
                    if fact.predicate == "is_a" {
                        if let FactObject::Node(parent_id) = fact.object {
                            if !visited.contains(&parent_id) {
                                visited.insert(parent_id);
                                let new_confidence = confidence * fact.confidence;
                                parents.push((parent_id, new_confidence));
                                queue.push_back((parent_id, new_confidence, depth + 1));
                            }
                        }
                    }
                }
            }
        }

        Ok(parents)
    }

    /// Find path between entities
    fn find_path(
        &self,
        start: u32,
        end: u32,
        predicate: &str,
        max_hops: u32,
    ) -> Result<Option<Vec<ReasoningStep>>> {
        let mut queue = VecDeque::new();
        let mut visited = HashSet::new();

        queue.push_back((start, vec![], 0));
        visited.insert(start);

        let facts = self.facts.lock().map_err(|e| anyhow!("Lock poisoned: {}", e))?;

        while let Some((current, path, depth)) = queue.pop_front() {
            if current == end {
                return Ok(Some(path));
            }

            if depth >= max_hops {
                continue;
            }

            if let Some(indices) = self.fact_index.get(&current) {
                for &idx in indices.value() {
                    let fact = &facts[idx];
                    if fact.predicate == predicate {
                        if let FactObject::Node(next) = fact.object {
                            if !visited.contains(&next) {
                                visited.insert(next);
                                let mut new_path = path.clone();
                                new_path.push(ReasoningStep {
                                    fact: fact.clone(),
                                    rule_used: None,
                                    confidence: fact.confidence,
                                });
                                queue.push_back((next, new_path, depth + 1));
                            }
                        }
                    }
                }
            }
        }

        Ok(None)
    }

    /// Build explanation chain
    fn build_explanation_chain(
        &self,
        target_fact: &LogicalFact,
        max_hops: u32,
    ) -> Result<Option<ExplanationChain>> {
        // Simple implementation - just show direct facts leading to conclusion
        let steps = vec![ReasoningStep {
            fact: target_fact.clone(),
            rule_used: Some("Direct observation".to_string()),
            confidence: target_fact.confidence,
        }];

        Ok(Some(ExplanationChain {
            conclusion: target_fact.clone(),
            steps,
            total_confidence: target_fact.confidence,
        }))
    }

    /// Find contradictions in knowledge base
    pub fn find_contradictions(&self) -> Result<Vec<Contradiction>> {
        let mut contradictions = Vec::new();
        let facts = self.facts.lock().map_err(|e| anyhow!("Lock poisoned: {}", e))?;
        let rules = self.rules.lock().map_err(|e| anyhow!("Lock poisoned: {}", e))?;

        // Check contradiction rules
        for rule in rules.iter() {
            if let RulePattern::Contradiction {
                predicate,
                conflicting_values,
            } = &rule.pattern
            {
                // Group facts by subject and predicate
                let mut entity_facts: HashMap<u32, Vec<&LogicalFact>> = HashMap::new();

                for fact in facts.iter() {
                    if fact.predicate == *predicate {
                        entity_facts
                            .entry(fact.subject)
                            .or_insert_with(Vec::new)
                            .push(fact);
                    }
                }

                // Check for conflicts
                for (entity_id, entity_fact_list) in entity_facts {
                    let mut conflicting = Vec::new();

                    for fact in entity_fact_list {
                        if let FactObject::Node(node_id) = fact.object {
                            if conflicting_values.contains(&node_id) {
                                conflicting.push(fact.clone());
                            }
                        }
                    }

                    if conflicting.len() > 1 {
                        let max_confidence = conflicting
                            .iter()
                            .map(|f| f.confidence)
                            .fold(0.0f32, f32::max);
                        let min_confidence = conflicting
                            .iter()
                            .map(|f| f.confidence)
                            .fold(1.0f32, f32::min);

                        contradictions.push(Contradiction {
                            entity_id,
                            predicate: predicate.clone(),
                            conflicting_facts: conflicting,
                            confidence_difference: max_confidence - min_confidence,
                        });
                    }
                }
            }
        }

        Ok(contradictions)
    }

    /// Sync to GPU for accelerated reasoning
    pub fn sync_to_gpu(&mut self) -> Result<()> {
        let facts = self.facts.lock().map_err(|e| anyhow!("Lock poisoned: {}", e))?;

        let mut subjects = Vec::new();
        let mut predicates = Vec::new();
        let mut objects = Vec::new();
        let mut confidences = Vec::new();

        for fact in facts.iter() {
            subjects.push(fact.subject);
            predicates.push(self.hash_predicate(&fact.predicate));

            // For simplicity, only handle node objects
            let obj_id = match &fact.object {
                FactObject::Node(id) => *id,
                FactObject::Value(_) => u32::MAX, // Placeholder
            };
            objects.push(obj_id);
            confidences.push(fact.confidence);
        }

        // Upload to GPU
        let stream = self.device.default_stream();
        let gpu_data = GpuReasoningData {
            fact_subjects: stream.clone_htod(&subjects)?,
            fact_predicates: stream.clone_htod(&predicates)?,
            fact_objects: stream.clone_htod(&objects)?,
            fact_confidences: stream.clone_htod(&confidences)?,
            rule_patterns: stream.alloc_zeros(1)?, // Placeholder
        };

        self.gpu_data = Some(gpu_data);
        Ok(())
    }

    /// Hash predicate for GPU
    fn hash_predicate(&self, predicate: &str) -> u32 {
        // Simple hash function
        predicate
            .bytes()
            .fold(0u32, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u32))
    }

    /// GPU multi-hop reasoning
    pub fn gpu_multi_hop_reasoning(&self, start: u32, end: u32, max_hops: u32) -> Result<Vec<u32>> {
        if self.gpu_data.is_none() {
            return Err(anyhow!("GPU data not synchronized"));
        }

        // In real implementation, would launch GPU kernel
        // For now, return placeholder
        Ok(vec![start, end])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_logical_fact_creation() {
        let fact = LogicalFact {
            subject: 1,
            predicate: "is_a".to_string(),
            object: FactObject::Node(2),
            confidence: 0.9,
        };

        assert_eq!(fact.subject, 1);
        assert_eq!(fact.predicate, "is_a");
        assert_eq!(fact.object, FactObject::Node(2));
        assert_eq!(fact.confidence, 0.9);
    }
}
