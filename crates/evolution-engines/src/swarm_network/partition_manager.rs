//! Particle partitioning management for distributed swarm

use super::types::{ParticleMetadata, PartitionStrategy};
use crate::error::EvolutionEngineResult;
use crate::swarm_distributed::DistributedSwarmConfig;
use std::collections::HashMap;

/// Manages particle partitioning across nodes
pub struct PartitionManager {
    /// Current particle assignments: node_id -> particle_ids
    pub(crate) assignments: HashMap<String, Vec<String>>,
    /// Particle metadata for migration decisions
    pub(crate) particle_metadata: HashMap<String, ParticleMetadata>,
    /// Partition strategy
    pub(crate) strategy: PartitionStrategy,
}

impl PartitionManager {
    /// Create new partition manager
    pub async fn new(_config: DistributedSwarmConfig) -> EvolutionEngineResult<Self> {
        Ok(Self {
            assignments: HashMap::new(),
            particle_metadata: HashMap::new(),
            strategy: PartitionStrategy::NetworkAware,
        })
    }

    /// Assign particle to node
    pub async fn assign_particle(
        &mut self,
        particle_id: String,
        node_id: String,
    ) -> EvolutionEngineResult<()> {
        // Add to assignments
        self.assignments
            .entry(node_id.clone())
            .or_insert_with(Vec::new)
            .push(particle_id.clone());

        // Create metadata
        self.particle_metadata.insert(
            particle_id.clone(),
            ParticleMetadata {
                id: particle_id,
                current_node: node_id,
                compute_cost: 1.0,
                communication_frequency: HashMap::new(),
                migration_count: 0,
                last_migration: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_millis() as u64,
            },
        );

        Ok(())
    }

    /// Get particles assigned to a node
    pub fn get_particles_on_node(&self, node_id: &str) -> Vec<String> {
        self.assignments.get(node_id).cloned().unwrap_or_default()
    }

    /// Migrate particle between nodes
    pub async fn migrate_particle(
        &mut self,
        particle_id: &str,
        from_node: &str,
        to_node: &str,
    ) -> EvolutionEngineResult<()> {
        // Remove from source node
        if let Some(particles) = self.assignments.get_mut(from_node) {
            particles.retain(|id| id != particle_id);
        }

        // Add to destination node
        self.assignments
            .entry(to_node.to_string())
            .or_insert_with(Vec::new)
            .push(particle_id.to_string());

        // Update metadata
        if let Some(metadata) = self.particle_metadata.get_mut(particle_id) {
            metadata.current_node = to_node.to_string();
            metadata.migration_count += 1;
            metadata.last_migration = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64;
        }

        Ok(())
    }

    /// Get all current assignments
    pub fn get_assignments(&self) -> &HashMap<String, Vec<String>> {
        &self.assignments
    }

    /// Get particle metadata
    pub fn get_particle_metadata(&self, particle_id: &str) -> Option<&ParticleMetadata> {
        self.particle_metadata.get(particle_id)
    }

    /// Update particle compute cost
    pub fn update_particle_cost(&mut self, particle_id: &str, cost: f64) {
        if let Some(metadata) = self.particle_metadata.get_mut(particle_id) {
            metadata.compute_cost = cost;
        }
    }

    /// Get total particles across all nodes
    pub fn total_particles(&self) -> usize {
        self.assignments.values().map(|v| v.len()).sum()
    }

    /// Get particles per node distribution
    pub fn get_distribution(&self) -> HashMap<String, usize> {
        self.assignments
            .iter()
            .map(|(node, particles)| (node.clone(), particles.len()))
            .collect()
    }
}
