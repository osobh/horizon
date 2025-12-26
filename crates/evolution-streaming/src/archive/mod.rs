//! Archive management for agent evolution storage

use crate::{AgentGenome, AgentId, EvolutionStreamingError, SelectionStrategy};
use dashmap::DashMap;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::RwLock;

pub mod storage;

pub use storage::*;

/// Agent archive for storing and managing evolved agents
#[derive(Debug)]
pub struct AgentArchive {
    agents: Arc<DashMap<AgentId, ArchivedAgent>>,
    fitness_index: Arc<RwLock<BTreeMap<OrderedFloat, AgentId>>>,
    diversity_index: Arc<RwLock<DashMap<String, Vec<AgentId>>>>,
    max_size: usize,
    stats: Arc<ArchiveStats>,
}

/// Wrapper for f64 to make it orderable in BTreeMap
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
struct OrderedFloat(f64);

impl Eq for OrderedFloat {}

impl Ord for OrderedFloat {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .partial_cmp(&other.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

impl From<f64> for OrderedFloat {
    fn from(f: f64) -> Self {
        OrderedFloat(f)
    }
}

/// Archived agent with metadata
#[derive(Debug, Clone)]
pub struct ArchivedAgent {
    pub genome: AgentGenome,
    pub fitness: f64,
    pub archive_time: u64,
    pub evaluation_count: u32,
    pub lineage_info: LineageInfo,
}

/// Lineage tracking information
#[derive(Debug, Clone)]
pub struct LineageInfo {
    pub ancestors: Vec<AgentId>,
    pub descendants: Vec<AgentId>,
    pub generation: u64,
    pub family_size: u32,
}

/// Archive statistics
#[derive(Debug, Default)]
struct ArchiveStats {
    total_agents: AtomicU64,
    additions: AtomicU64,
    removals: AtomicU64,
    diversity_score: AtomicU64, // Stored as u64 for atomic access
    best_fitness: AtomicU64,    // Stored as u64 bits
}

impl AgentArchive {
    /// Create a new agent archive
    pub fn new(max_size: usize) -> Self {
        Self {
            agents: Arc::new(DashMap::new()),
            fitness_index: Arc::new(RwLock::new(BTreeMap::new())),
            diversity_index: Arc::new(RwLock::new(DashMap::new())),
            max_size,
            stats: Arc::new(ArchiveStats::default()),
        }
    }

    /// Add agent to archive if novel enough
    pub async fn add_if_novel(
        &self,
        agent: AgentGenome,
        fitness: f64,
    ) -> Result<bool, EvolutionStreamingError> {
        // Check for novelty
        if !self.is_novel(&agent, fitness).await? {
            return Ok(false);
        }

        // Check if archive is full
        if self.agents.len() >= self.max_size {
            self.make_space_for_new_agent(fitness).await?;
        }

        // Create archived agent
        let archived = ArchivedAgent {
            genome: agent.clone(),
            fitness,
            archive_time: self.current_timestamp(),
            evaluation_count: 1,
            lineage_info: self.build_lineage_info(&agent).await,
        };

        // Add to main storage
        self.agents.insert(agent.id, archived.clone());

        // Update fitness index
        {
            let mut fitness_index = self.fitness_index.write().await;
            fitness_index.insert(OrderedFloat::from(fitness), agent.id);
        }

        // Update diversity index
        self.update_diversity_index(&agent).await;

        // Update statistics
        self.stats.total_agents.fetch_add(1, Ordering::Relaxed);
        self.stats.additions.fetch_add(1, Ordering::Relaxed);

        let best_fitness_bits = f64::to_bits(fitness);
        self.stats
            .best_fitness
            .fetch_max(best_fitness_bits, Ordering::Relaxed);

        Ok(true)
    }

    /// Check if agent is novel enough to be added
    async fn is_novel(
        &self,
        agent: &AgentGenome,
        fitness: f64,
    ) -> Result<bool, EvolutionStreamingError> {
        // Check if exact duplicate exists
        if self.agents.contains_key(&agent.id) {
            return Ok(false);
        }

        // Check behavioral diversity
        let diversity_threshold = 0.8; // 80% similarity threshold
        let behavior_key = self.compute_behavior_key(agent);

        let diversity_index = self.diversity_index.read().await;
        if let Some(similar_agents) = diversity_index.get(&behavior_key) {
            for &similar_id in similar_agents.iter() {
                if let Some(similar_agent) = self.agents.get(&similar_id) {
                    let similarity = agent.similarity(&similar_agent.genome);
                    if similarity > diversity_threshold {
                        // Only add if significantly better
                        return Ok(fitness > similar_agent.fitness * 1.1);
                    }
                }
            }
        }

        // Check fitness threshold
        let fitness_threshold = self.get_fitness_threshold().await;
        Ok(fitness >= fitness_threshold)
    }

    /// Make space for new agent by removing least fit
    async fn make_space_for_new_agent(
        &self,
        new_fitness: f64,
    ) -> Result<(), EvolutionStreamingError> {
        let fitness_index = self.fitness_index.read().await;

        // Find lowest fitness agent
        if let Some((&lowest_fitness, &lowest_id)) = fitness_index.iter().next() {
            if new_fitness > lowest_fitness.0 {
                drop(fitness_index); // Release read lock
                self.remove_agent(lowest_id).await?;
            }
        }

        Ok(())
    }

    /// Remove agent from archive
    pub async fn remove_agent(&self, agent_id: AgentId) -> Result<(), EvolutionStreamingError> {
        if let Some((_, archived_agent)) = self.agents.remove(&agent_id) {
            // Remove from fitness index
            {
                let mut fitness_index = self.fitness_index.write().await;
                fitness_index.remove(&OrderedFloat::from(archived_agent.fitness));
            }

            // Remove from diversity index
            let behavior_key = self.compute_behavior_key(&archived_agent.genome);
            let diversity_index = self.diversity_index.write().await;
            if let Some(mut agents) = diversity_index.get_mut(&behavior_key) {
                agents.retain(|&id| id != agent_id);
                if agents.is_empty() {
                    diversity_index.remove(&behavior_key);
                }
            }

            self.stats.removals.fetch_add(1, Ordering::Relaxed);
            self.stats.total_agents.fetch_sub(1, Ordering::Relaxed);
        }

        Ok(())
    }

    /// Update fitness for existing agent
    pub async fn update_fitness(
        &self,
        agent_id: AgentId,
        new_fitness: f64,
    ) -> Result<(), EvolutionStreamingError> {
        if let Some(mut archived_agent) = self.agents.get_mut(&agent_id) {
            let old_fitness = archived_agent.fitness;
            archived_agent.fitness = new_fitness;
            archived_agent.evaluation_count += 1;

            // Update fitness index
            {
                let mut fitness_index = self.fitness_index.write().await;
                fitness_index.remove(&OrderedFloat::from(old_fitness));
                fitness_index.insert(OrderedFloat::from(new_fitness), agent_id);
            }

            let best_fitness_bits = f64::to_bits(new_fitness);
            self.stats
                .best_fitness
                .fetch_max(best_fitness_bits, Ordering::Relaxed);
        }

        Ok(())
    }

    /// Select agents using specified strategy
    pub async fn select_agents(
        &self,
        count: usize,
        strategy: SelectionStrategy,
    ) -> Result<Vec<AgentGenome>, EvolutionStreamingError> {
        match strategy {
            SelectionStrategy::Elite { count: elite_count } => {
                self.select_elite(count.min(elite_count)).await
            }
            SelectionStrategy::Tournament { size } => self.select_tournament(count, size).await,
            SelectionStrategy::RouletteWheel => self.select_roulette_wheel(count).await,
            SelectionStrategy::RankBased => self.select_rank_based(count).await,
            SelectionStrategy::Random => self.select_random(count).await,
        }
    }

    /// Select elite agents (best fitness)
    async fn select_elite(
        &self,
        count: usize,
    ) -> Result<Vec<AgentGenome>, EvolutionStreamingError> {
        let fitness_index = self.fitness_index.read().await;
        let selected_ids: Vec<AgentId> = fitness_index
            .iter()
            .rev() // Start from highest fitness
            .take(count)
            .map(|(_, &id)| id)
            .collect();

        let mut selected = Vec::new();
        for id in selected_ids {
            if let Some(agent) = self.agents.get(&id) {
                selected.push(agent.genome.clone());
            }
        }

        Ok(selected)
    }

    /// Tournament selection
    async fn select_tournament(
        &self,
        count: usize,
        tournament_size: usize,
    ) -> Result<Vec<AgentGenome>, EvolutionStreamingError> {
        let mut selected = Vec::new();
        let agent_ids: Vec<AgentId> = self.agents.iter().map(|entry| *entry.key()).collect();

        if agent_ids.is_empty() {
            return Ok(selected);
        }

        for _ in 0..count {
            let mut best_fitness = f64::NEG_INFINITY;
            let mut best_agent = None;

            // Run tournament
            for _ in 0..tournament_size {
                let random_idx = fastrand::usize(..agent_ids.len());
                let agent_id = agent_ids[random_idx];

                if let Some(agent) = self.agents.get(&agent_id) {
                    if agent.fitness > best_fitness {
                        best_fitness = agent.fitness;
                        best_agent = Some(agent.genome.clone());
                    }
                }
            }

            if let Some(agent) = best_agent {
                selected.push(agent);
            }
        }

        Ok(selected)
    }

    /// Roulette wheel selection
    async fn select_roulette_wheel(
        &self,
        count: usize,
    ) -> Result<Vec<AgentGenome>, EvolutionStreamingError> {
        let agents: Vec<(AgentId, f64)> = self
            .agents
            .iter()
            .map(|entry| (*entry.key(), entry.value().fitness))
            .collect();

        if agents.is_empty() {
            return Ok(Vec::new());
        }

        // Normalize fitness to positive values
        let min_fitness = agents.iter().map(|(_, f)| *f).fold(f64::INFINITY, f64::min);
        let offset = if min_fitness < 0.0 {
            -min_fitness + 1.0
        } else {
            0.0
        };

        let fitness_sum: f64 = agents.iter().map(|(_, f)| f + offset).sum();
        let mut selected = Vec::new();

        for _ in 0..count {
            let mut random_value = fastrand::f64() * fitness_sum;

            for (agent_id, fitness) in &agents {
                random_value -= fitness + offset;
                if random_value <= 0.0 {
                    if let Some(agent) = self.agents.get(agent_id) {
                        selected.push(agent.genome.clone());
                    }
                    break;
                }
            }
        }

        Ok(selected)
    }

    /// Rank-based selection
    async fn select_rank_based(
        &self,
        count: usize,
    ) -> Result<Vec<AgentGenome>, EvolutionStreamingError> {
        let fitness_index = self.fitness_index.read().await;
        let agent_count = fitness_index.len();

        if agent_count == 0 {
            return Ok(Vec::new());
        }

        let mut selected = Vec::new();
        let rank_sum = (agent_count * (agent_count + 1)) / 2;

        for _ in 0..count {
            let mut random_value = fastrand::usize(..rank_sum);

            for (rank, (_, &agent_id)) in fitness_index.iter().enumerate() {
                let rank_weight = rank + 1;
                if random_value < rank_weight {
                    if let Some(agent) = self.agents.get(&agent_id) {
                        selected.push(agent.genome.clone());
                    }
                    break;
                }
                random_value -= rank_weight;
            }
        }

        Ok(selected)
    }

    /// Random selection
    async fn select_random(
        &self,
        count: usize,
    ) -> Result<Vec<AgentGenome>, EvolutionStreamingError> {
        let agent_ids: Vec<AgentId> = self.agents.iter().map(|entry| *entry.key()).collect();
        let mut selected = Vec::new();

        for _ in 0..count {
            if !agent_ids.is_empty() {
                let random_idx = fastrand::usize(..agent_ids.len());
                let agent_id = agent_ids[random_idx];

                if let Some(agent) = self.agents.get(&agent_id) {
                    selected.push(agent.genome.clone());
                }
            }
        }

        Ok(selected)
    }

    /// Get current size of archive
    pub fn size(&self) -> usize {
        self.agents.len()
    }

    /// Check if archive is empty
    pub fn is_empty(&self) -> bool {
        self.agents.is_empty()
    }

    /// Get best fitness in archive
    pub async fn best_fitness(&self) -> Option<f64> {
        let fitness_index = self.fitness_index.read().await;
        fitness_index.keys().last().map(|f| f.0)
    }

    /// Get diversity score
    pub fn diversity_score(&self) -> f64 {
        let bits = self.stats.diversity_score.load(Ordering::Relaxed);
        f64::from_bits(bits)
    }

    /// Get archive statistics
    pub fn get_stats(&self) -> ArchiveStatistics {
        ArchiveStatistics {
            total_agents: self.stats.total_agents.load(Ordering::Relaxed),
            additions: self.stats.additions.load(Ordering::Relaxed),
            removals: self.stats.removals.load(Ordering::Relaxed),
            current_size: self.agents.len() as u64,
            max_size: self.max_size as u64,
            best_fitness: f64::from_bits(self.stats.best_fitness.load(Ordering::Relaxed)),
            diversity_score: self.diversity_score(),
        }
    }

    // Helper methods

    fn current_timestamp(&self) -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }

    async fn build_lineage_info(&self, agent: &AgentGenome) -> LineageInfo {
        LineageInfo {
            ancestors: if let Some(parent_id) = agent.metadata.parent_id {
                vec![parent_id]
            } else {
                Vec::new()
            },
            descendants: Vec::new(),
            generation: agent.metadata.generation,
            family_size: 1,
        }
    }

    fn compute_behavior_key(&self, agent: &AgentGenome) -> String {
        // Simple behavior key based on code hash and parameter ranges
        let code_hash = format!("{:x}", md5::compute(&agent.code));
        let param_summary = if agent.parameters.is_empty() {
            "no_params".to_string()
        } else {
            let avg = agent.parameters.iter().sum::<f32>() / agent.parameters.len() as f32;
            format!("avg_{:.2}", avg)
        };

        format!("{}_{}", code_hash, param_summary)
    }

    async fn update_diversity_index(&self, agent: &AgentGenome) {
        let behavior_key = self.compute_behavior_key(agent);
        let diversity_index = self.diversity_index.write().await;

        diversity_index
            .entry(behavior_key)
            .or_insert_with(Vec::new)
            .push(agent.id);
    }

    async fn get_fitness_threshold(&self) -> f64 {
        if self.agents.is_empty() {
            return 0.0;
        }

        let fitness_index = self.fitness_index.read().await;
        if let Some((&median_fitness, _)) = fitness_index.iter().nth(fitness_index.len() / 2) {
            median_fitness.0 * 0.8 // 80% of median fitness
        } else {
            0.0
        }
    }
}

/// Archive statistics
#[derive(Debug, Clone)]
pub struct ArchiveStatistics {
    pub total_agents: u64,
    pub additions: u64,
    pub removals: u64,
    pub current_size: u64,
    pub max_size: u64,
    pub best_fitness: f64,
    pub diversity_score: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_archive_creation() {
        let archive = AgentArchive::new(100);
        assert_eq!(archive.size(), 0);
        assert!(archive.is_empty());
        assert_eq!(archive.max_size, 100);
    }

    #[tokio::test]
    async fn test_add_novel_agent() {
        let archive = AgentArchive::new(10);
        let agent = AgentGenome::new("fn test() { 42 }".to_string(), vec![1.0, 2.0]);

        let added = archive.add_if_novel(agent.clone(), 0.8).await.unwrap();
        assert!(added);
        assert_eq!(archive.size(), 1);

        let best_fitness = archive.best_fitness().await.unwrap();
        assert_eq!(best_fitness, 0.8);
    }

    #[tokio::test]
    async fn test_duplicate_agent_rejection() {
        let archive = AgentArchive::new(10);
        let agent = AgentGenome::new("fn test() { 42 }".to_string(), vec![1.0]);

        // Add first time
        let added1 = archive.add_if_novel(agent.clone(), 0.8).await.unwrap();
        assert!(added1);

        // Try to add same agent again
        let added2 = archive.add_if_novel(agent, 0.9).await.unwrap();
        assert!(!added2);
        assert_eq!(archive.size(), 1);
    }

    #[tokio::test]
    async fn test_fitness_update() {
        let archive = AgentArchive::new(10);
        let agent = AgentGenome::new("fn test() { 42 }".to_string(), vec![1.0]);

        archive.add_if_novel(agent.clone(), 0.5).await.unwrap();
        archive.update_fitness(agent.id, 0.9).await.unwrap();

        let best_fitness = archive.best_fitness().await.unwrap();
        assert_eq!(best_fitness, 0.9);
    }

    #[tokio::test]
    async fn test_agent_removal() {
        let archive = AgentArchive::new(10);
        let agent = AgentGenome::new("fn test() { 42 }".to_string(), vec![1.0]);

        archive.add_if_novel(agent.clone(), 0.8).await.unwrap();
        assert_eq!(archive.size(), 1);

        archive.remove_agent(agent.id).await.unwrap();
        assert_eq!(archive.size(), 0);
        assert!(archive.is_empty());
    }

    #[tokio::test]
    async fn test_elite_selection() {
        let archive = AgentArchive::new(10);

        // Add agents with different fitness values
        for i in 0..5 {
            let agent = AgentGenome::new(format!("fn test{} {{}}", i), vec![i as f32]);
            archive.add_if_novel(agent, i as f64 * 0.2).await.unwrap();
        }

        let selected = archive
            .select_agents(3, SelectionStrategy::Elite { count: 3 })
            .await
            .unwrap();
        assert_eq!(selected.len(), 3);

        // Should select the best 3 agents (highest fitness)
        // The agent with fitness 0.8 should be included
        assert!(selected.iter().any(|a| a.code.contains("test4")));
    }

    #[tokio::test]
    async fn test_tournament_selection() {
        let archive = AgentArchive::new(10);

        // Add some agents
        for i in 0..5 {
            let agent = AgentGenome::new(format!("fn test{} {{}}", i), vec![i as f32]);
            archive.add_if_novel(agent, i as f64 * 0.1).await.unwrap();
        }

        let selected = archive
            .select_agents(3, SelectionStrategy::Tournament { size: 2 })
            .await
            .unwrap();
        assert_eq!(selected.len(), 3);
    }

    #[tokio::test]
    async fn test_random_selection() {
        let archive = AgentArchive::new(10);

        // Add some agents
        for i in 0..3 {
            let agent = AgentGenome::new(format!("fn test{} {{}}", i), vec![i as f32]);
            archive.add_if_novel(agent, 0.5).await.unwrap();
        }

        let selected = archive
            .select_agents(2, SelectionStrategy::Random)
            .await
            .unwrap();
        assert_eq!(selected.len(), 2);
    }

    #[tokio::test]
    async fn test_archive_size_limit() {
        let archive = AgentArchive::new(2); // Small limit

        // Add more agents than the limit
        for i in 0..5 {
            let agent = AgentGenome::new(format!("fn test{} {{}}", i), vec![i as f32]);
            archive.add_if_novel(agent, i as f64).await.unwrap();
        }

        // Should not exceed max size
        assert!(archive.size() <= 2);

        // Best fitness should be from the latest, highest fitness agents
        let best_fitness = archive.best_fitness().await.unwrap();
        assert!(best_fitness >= 3.0); // Should keep high fitness agents
    }

    #[tokio::test]
    async fn test_archive_statistics() {
        let archive = AgentArchive::new(10);

        let agent1 = AgentGenome::new("fn test1() {}".to_string(), vec![1.0]);
        let agent2 = AgentGenome::new("fn test2() {}".to_string(), vec![2.0]);

        archive.add_if_novel(agent1.clone(), 0.7).await.unwrap();
        archive.add_if_novel(agent2, 0.9).await.unwrap();
        archive.remove_agent(agent1.id).await.unwrap();

        let stats = archive.get_stats();
        assert_eq!(stats.additions, 2);
        assert_eq!(stats.removals, 1);
        assert_eq!(stats.current_size, 1);
        assert_eq!(stats.max_size, 10);
        assert_eq!(stats.best_fitness, 0.9);
    }

    #[test]
    fn test_ordered_float() {
        let f1 = OrderedFloat::from(1.5);
        let f2 = OrderedFloat::from(2.5);
        let f3 = OrderedFloat::from(1.5);

        assert!(f1 < f2);
        assert!(f1 == f3);
        assert!(f2 > f1);
    }
}
