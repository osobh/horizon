//! Core Darwin archive implementation

use crate::{
    error::{EvolutionEngineError, EvolutionEngineResult},
    traits::{AgentGenome, Evolvable, EvolvableAgent},
};
use parking_lot::RwLock;
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use super::config::DarwinArchiveConfig;
use super::types::{ArchivedAgent, DiversityMetrics, SteppingStone};

/// Darwin-style archive for open-ended exploration
pub struct DarwinArchive {
    /// Configuration
    config: DarwinArchiveConfig,
    /// All discovered agents
    agents: Arc<RwLock<HashMap<String, ArchivedAgent>>>,
    /// Stepping stone relationships
    stepping_stones: Arc<RwLock<Vec<SteppingStone>>>,
    /// Parent selection weights (agent_id -> weight)
    selection_weights: Arc<RwLock<HashMap<String, f64>>>,
    /// Lineage tracking (agent_id -> root_ancestor_id)
    lineages: Arc<RwLock<HashMap<String, String>>>,
    /// Current generation
    current_generation: Arc<RwLock<u32>>,
    /// Random number generator
    rng: Arc<RwLock<StdRng>>,
}

impl DarwinArchive {
    /// Create new archive with given configuration
    pub fn new(config: DarwinArchiveConfig) -> Self {
        Self {
            config,
            agents: Arc::new(RwLock::new(HashMap::new())),
            stepping_stones: Arc::new(RwLock::new(Vec::new())),
            selection_weights: Arc::new(RwLock::new(HashMap::new())),
            lineages: Arc::new(RwLock::new(HashMap::new())),
            current_generation: Arc::new(RwLock::new(0)),
            rng: Arc::new(RwLock::new(StdRng::from_entropy())),
        }
    }

    /// Initialize archive with a single agent
    pub fn initialize(&self, initial_agent: EvolvableAgent) -> EvolutionEngineResult<()> {
        let agent_id = format!("agent_{}", uuid::Uuid::new_v4());

        let archived_agent = ArchivedAgent {
            id: agent_id.clone(),
            genome: initial_agent.genome.clone(),
            performance_score: 1.0, // Initial agent gets baseline score
            children_count: 0,
            has_editing_capability: true,
            discovery_generation: 0,
            parent_id: None,
            modification_type: "Initial".to_string(),
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        };

        self.agents.write().insert(agent_id.clone(), archived_agent);
        self.lineages
            .write()
            .insert(agent_id.clone(), agent_id.clone());
        self.update_selection_weights();

        if self.config.enable_logging {
            tracing::info!("Initialized Darwin archive with agent {}", agent_id);
        }

        Ok(())
    }

    /// Add a discovered agent to the archive
    pub fn add_agent(
        &self,
        agent: EvolvableAgent,
        performance: f64,
        parent_id: Option<String>,
        modification_type: String,
        has_editing_capability: bool,
    ) -> EvolutionEngineResult<String> {
        let agent_id = format!("agent_{}", uuid::Uuid::new_v4());
        let generation = *self.current_generation.read();

        let archived_agent = ArchivedAgent {
            id: agent_id.clone(),
            genome: agent.genome.clone(),
            performance_score: performance,
            children_count: 0,
            has_editing_capability,
            discovery_generation: generation,
            parent_id: parent_id.clone(),
            modification_type,
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        };

        // Update parent's children count
        if let Some(ref pid) = parent_id {
            if let Some(parent) = self.agents.write().get_mut(pid) {
                parent.children_count += 1;
            }
        }

        // Track lineage
        let lineage_root = if let Some(ref pid) = parent_id {
            self.lineages
                .read()
                .get(pid)
                .cloned()
                .unwrap_or_else(|| pid.clone())
        } else {
            agent_id.clone()
        };

        self.agents.write().insert(agent_id.clone(), archived_agent);
        self.lineages.write().insert(agent_id.clone(), lineage_root);

        // Detect stepping stones
        self.detect_stepping_stones(&agent_id, performance);

        // Update weights and prune if needed
        self.update_selection_weights();
        self.prune_if_needed();

        if self.config.enable_logging {
            tracing::info!(
                "Added agent {} with performance {} to archive",
                agent_id,
                performance
            );
        }

        Ok(agent_id)
    }

    /// Select a parent agent for modification using weighted selection
    pub fn select_parent(&self) -> EvolutionEngineResult<Option<ArchivedAgent>> {
        let agents = self.agents.read();
        let weights = self.selection_weights.read();

        // Filter to only agents with editing capability
        let eligible_agents: Vec<_> = agents
            .values()
            .filter(|a| a.has_editing_capability)
            .collect();

        if eligible_agents.is_empty() {
            return Ok(None);
        }

        // Get weights for eligible agents
        let agent_weights: Vec<f64> = eligible_agents
            .iter()
            .map(|a| weights.get(&a.id).copied().unwrap_or(1.0))
            .collect();

        // Weighted random selection
        let dist = WeightedIndex::new(&agent_weights).map_err(|e| {
            EvolutionEngineError::Other(format!("Failed to create weighted distribution: {}", e))
        })?;

        let selected_idx = dist.sample(&mut *self.rng.write());
        Ok(Some(eligible_agents[selected_idx].clone()))
    }

    /// Calculate diversity metrics for the archive
    pub fn calculate_diversity(&self) -> DiversityMetrics {
        let agents = self.agents.read();

        if agents.is_empty() {
            return DiversityMetrics {
                modification_diversity: 0.0,
                performance_variance: 0.0,
                genome_diversity: 0.0,
                lineage_count: 0,
            };
        }

        // Calculate modification type diversity
        let mut mod_types = HashSet::new();
        for agent in agents.values() {
            mod_types.insert(&agent.modification_type);
        }
        let modification_diversity = mod_types.len() as f64 / agents.len() as f64;

        // Calculate performance variance
        let performances: Vec<f64> = agents.values().map(|a| a.performance_score).collect();
        let mean = performances.iter().sum::<f64>() / performances.len() as f64;
        let variance = performances.iter().map(|p| (p - mean).powi(2)).sum::<f64>()
            / performances.len() as f64;

        // Calculate lineage diversity
        let unique_lineages = self.lineages.read().values().collect::<HashSet<_>>().len();

        // Simple genome diversity based on behavior parameters
        let mut genome_sum = 0.0;
        for (a1, a2) in agents.values().zip(agents.values().skip(1)) {
            genome_sum += self.calculate_genome_distance(&a1.genome, &a2.genome);
        }
        let genome_diversity = if agents.len() > 1 {
            genome_sum / (agents.len() * (agents.len() - 1)) as f64
        } else {
            0.0
        };

        DiversityMetrics {
            modification_diversity,
            performance_variance: variance,
            genome_diversity,
            lineage_count: unique_lineages,
        }
    }

    /// Get agents that served as stepping stones
    pub fn get_stepping_stones(&self) -> Vec<(String, Vec<String>)> {
        let stones = self.stepping_stones.read();
        let mut stone_map: HashMap<String, Vec<String>> = HashMap::new();

        for stone in stones.iter() {
            stone_map
                .entry(stone.from_agent.clone())
                .or_insert_with(Vec::new)
                .push(stone.to_agent.clone());
        }

        stone_map.into_iter().collect()
    }

    /// Update selection weights based on performance and children
    fn update_selection_weights(&self) {
        let agents = self.agents.read();
        let mut weights = self.selection_weights.write();

        weights.clear();

        for agent in agents.values() {
            if !agent.has_editing_capability {
                continue;
            }

            // Calculate weight based on performance, children, and diversity
            let performance_component = agent.performance_score * self.config.performance_weight;
            let children_component =
                (agent.children_count as f64 / 10.0).min(1.0) * self.config.children_weight;

            // Diversity component could be based on uniqueness of modification type
            let diversity_component = self.config.diversity_weight;

            let total_weight = performance_component + children_component + diversity_component;
            weights.insert(agent.id.clone(), total_weight.max(0.1)); // Minimum weight
        }
    }

    /// Detect stepping stone relationships
    fn detect_stepping_stones(&self, new_agent_id: &str, performance: f64) {
        let agents = self.agents.read();

        if let Some(new_agent) = agents.get(new_agent_id) {
            if let Some(ref parent_id) = new_agent.parent_id {
                // Check ancestors within stepping stone window
                let mut current_id = parent_id.clone();
                let mut generations_back = 0;

                while generations_back < self.config.stepping_stone_window {
                    if let Some(ancestor) = agents.get(&current_id) {
                        // If new agent significantly outperforms ancestor
                        if performance > ancestor.performance_score * 1.2 {
                            let stone = SteppingStone {
                                from_agent: current_id.clone(),
                                to_agent: new_agent_id.to_string(),
                                improvement: performance - ancestor.performance_score,
                                generation_gap: new_agent.discovery_generation
                                    - ancestor.discovery_generation,
                            };
                            self.stepping_stones.write().push(stone);
                        }

                        // Move to next ancestor
                        if let Some(ref next_parent) = ancestor.parent_id {
                            current_id = next_parent.clone();
                            generations_back += 1;
                        } else {
                            break;
                        }
                    } else {
                        break;
                    }
                }
            }
        }
    }

    /// Prune archive if it exceeds max size
    fn prune_if_needed(&self) {
        if self.config.max_size == 0 {
            return; // No limit
        }

        let mut agents = self.agents.write();

        if agents.len() <= self.config.max_size {
            return;
        }

        // Score agents for pruning (lower score = more likely to prune)
        let mut agent_scores: Vec<(String, f64)> = agents
            .iter()
            .map(|(id, agent)| {
                let mut score = agent.performance_score;

                // Boost score for agents with children
                score += (agent.children_count as f64) * 0.1;

                // Boost score for agents with editing capability
                if agent.has_editing_capability {
                    score += 0.2;
                }

                // Never prune the initial agent
                if agent.parent_id.is_none() {
                    score += 10.0;
                }

                (id.clone(), score)
            })
            .collect();

        // Sort by score (ascending)
        agent_scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Remove lowest scoring agents
        let to_remove = agents.len() - self.config.max_size;
        for (id, _) in agent_scores.iter().take(to_remove) {
            agents.remove(id);
            self.lineages.write().remove(id);
            self.selection_weights.write().remove(id);
        }
    }

    /// Calculate genome diversity between two agents
    fn calculate_genome_distance(&self, genome1: &AgentGenome, genome2: &AgentGenome) -> f64 {
        // Simple distance metric based on genome parameters
        let behavior_dist = (genome1.behavior.exploration_rate - genome2.behavior.exploration_rate)
            .abs()
            + (genome1.behavior.learning_rate - genome2.behavior.learning_rate).abs()
            + (genome1.behavior.risk_tolerance - genome2.behavior.risk_tolerance).abs();

        let architecture_dist = (genome1.architecture.memory_capacity as f64
            - genome2.architecture.memory_capacity as f64)
            .abs()
            / 1000.0
            + (genome1.architecture.processing_units as f64
                - genome2.architecture.processing_units as f64)
                .abs()
                / 10.0;

        let topology_dist = if genome1.architecture.network_topology.len()
            == genome2.architecture.network_topology.len()
        {
            genome1
                .architecture
                .network_topology
                .iter()
                .zip(genome2.architecture.network_topology.iter())
                .map(|(a, b)| (*a as f64 - *b as f64).abs() / 100.0)
                .sum::<f64>()
        } else {
            1.0
        };

        (behavior_dist + architecture_dist + topology_dist) / 3.0
    }

    /// Get all agents in archive
    pub fn get_all_agents(&self) -> Vec<ArchivedAgent> {
        self.agents.read().values().cloned().collect()
    }

    /// Get agent by ID
    pub fn get_agent(&self, id: &str) -> Option<ArchivedAgent> {
        self.agents.read().get(id).cloned()
    }

    /// Get current generation
    pub fn get_generation(&self) -> u32 {
        *self.current_generation.read()
    }

    /// Increment generation counter
    pub fn next_generation(&self) {
        *self.current_generation.write() += 1;
    }

    /// Get archive size
    pub fn size(&self) -> usize {
        self.agents.read().len()
    }

    /// Save archive to persistent storage
    pub async fn save(&self, path: &str) -> EvolutionEngineResult<()> {
        #[derive(Serialize)]
        struct ArchiveSnapshot {
            agents: Vec<ArchivedAgent>,
            generation: u32,
            stepping_stones: Vec<SteppingStone>,
        }

        let snapshot = ArchiveSnapshot {
            agents: self.agents.read().values().cloned().collect(),
            generation: *self.current_generation.read(),
            stepping_stones: self.stepping_stones.read().clone(),
        };

        let json = serde_json::to_string_pretty(&snapshot)
            .map_err(|e| EvolutionEngineError::SerializationError(e))?;

        tokio::fs::write(path, json)
            .await
            .map_err(|e| EvolutionEngineError::IoError(e))?;

        Ok(())
    }

    /// Load archive from persistent storage
    pub async fn load(path: &str, config: DarwinArchiveConfig) -> EvolutionEngineResult<Self> {
        #[derive(Deserialize)]
        struct ArchiveSnapshot {
            agents: Vec<ArchivedAgent>,
            generation: u32,
            stepping_stones: Vec<SteppingStone>,
        }

        let json = tokio::fs::read_to_string(path)
            .await
            .map_err(|e| EvolutionEngineError::IoError(e))?;

        let snapshot: ArchiveSnapshot =
            serde_json::from_str(&json).map_err(|e| EvolutionEngineError::SerializationError(e))?;

        let archive = Self::new(config);

        // Restore agents
        for agent in snapshot.agents {
            let lineage_root = if let Some(ref pid) = agent.parent_id {
                archive
                    .lineages
                    .read()
                    .get(pid)
                    .cloned()
                    .unwrap_or_else(|| agent.id.clone())
            } else {
                agent.id.clone()
            };

            archive
                .agents
                .write()
                .insert(agent.id.clone(), agent.clone());
            archive
                .lineages
                .write()
                .insert(agent.id.clone(), lineage_root);
        }

        // Restore generation and stepping stones
        *archive.current_generation.write() = snapshot.generation;
        *archive.stepping_stones.write() = snapshot.stepping_stones;

        // Update weights
        archive.update_selection_weights();

        Ok(archive)
    }
}
