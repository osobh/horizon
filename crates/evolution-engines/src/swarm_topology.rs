//! Swarm topology implementations for different neighbor selection strategies

use parking_lot::RwLock;
use rand::{rngs::StdRng, Rng};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Swarm topology types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SwarmTopology {
    /// Global best - all particles see the global best
    Global,
    /// Local best - particles only see neighbors
    Local,
    /// Ring topology
    Ring,
    /// Random topology
    Random,
    /// Dynamic topology that changes over time
    Dynamic,
}

impl Default for SwarmTopology {
    fn default() -> Self {
        SwarmTopology::Local
    }
}

/// Topology manager for neighbor selection
pub struct TopologyManager {
    /// Random number generator
    rng: Arc<RwLock<StdRng>>,
}

impl TopologyManager {
    /// Create new topology manager
    pub fn new(rng: Arc<RwLock<StdRng>>) -> Self {
        Self { rng }
    }

    /// Get neighbors for a particle based on topology
    pub fn get_neighbors(
        &self,
        topology: &SwarmTopology,
        particle_idx: usize,
        num_particles: usize,
        neighborhood_size: usize,
    ) -> Vec<usize> {
        match topology {
            SwarmTopology::Global => self.get_global_neighbors(num_particles),
            SwarmTopology::Local => {
                self.get_local_neighbors(particle_idx, num_particles, neighborhood_size)
            }
            SwarmTopology::Ring => self.get_ring_neighbors(particle_idx, num_particles),
            SwarmTopology::Random => self.get_random_neighbors(particle_idx, num_particles),
            SwarmTopology::Dynamic => {
                self.get_local_neighbors(particle_idx, num_particles, neighborhood_size)
            }
        }
    }

    /// Get all particles as neighbors (global topology)
    fn get_global_neighbors(&self, num_particles: usize) -> Vec<usize> {
        (0..num_particles).collect()
    }

    /// Get local neighbors around a particle
    fn get_local_neighbors(
        &self,
        particle_idx: usize,
        num_particles: usize,
        neighborhood_size: usize,
    ) -> Vec<usize> {
        let mut neighbors = Vec::new();
        let neighborhood_size = neighborhood_size.min(num_particles);

        for i in 0..neighborhood_size {
            let offset = i as i32 - (neighborhood_size as i32 / 2);
            let neighbor_idx = ((particle_idx as i32 + offset + num_particles as i32)
                % num_particles as i32) as usize;
            neighbors.push(neighbor_idx);
        }

        neighbors
    }

    /// Get ring topology neighbors (previous, self, next)
    fn get_ring_neighbors(&self, particle_idx: usize, num_particles: usize) -> Vec<usize> {
        vec![
            (particle_idx + num_particles - 1) % num_particles,
            particle_idx,
            (particle_idx + 1) % num_particles,
        ]
    }

    /// Get random neighbors
    fn get_random_neighbors(&self, particle_idx: usize, num_particles: usize) -> Vec<usize> {
        let mut rng = self.rng.write();
        let mut neighbors: Vec<usize> = (0..num_particles).collect();
        neighbors.retain(|&idx| idx != particle_idx || rng.gen_bool(0.3));
        neighbors
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn create_topology_manager() -> TopologyManager {
        let rng = Arc::new(RwLock::new(StdRng::seed_from_u64(42)));
        TopologyManager::new(rng)
    }

    #[test]
    fn test_global_topology() {
        let manager = create_topology_manager();
        let neighbors = manager.get_neighbors(&SwarmTopology::Global, 2, 10, 5);
        assert_eq!(neighbors.len(), 10);
        assert!(neighbors.contains(&0));
        assert!(neighbors.contains(&9));
    }

    #[test]
    fn test_local_topology() {
        let manager = create_topology_manager();
        let neighbors = manager.get_neighbors(&SwarmTopology::Local, 5, 10, 3);
        assert_eq!(neighbors.len(), 3);
        assert!(neighbors.contains(&4));
        assert!(neighbors.contains(&5));
        assert!(neighbors.contains(&6));
    }

    #[test]
    fn test_ring_topology() {
        let manager = create_topology_manager();
        let neighbors = manager.get_neighbors(&SwarmTopology::Ring, 3, 10, 5);
        assert_eq!(neighbors, vec![2, 3, 4]);

        // Test edge cases
        let neighbors_start = manager.get_neighbors(&SwarmTopology::Ring, 0, 10, 5);
        assert_eq!(neighbors_start, vec![9, 0, 1]);

        let neighbors_end = manager.get_neighbors(&SwarmTopology::Ring, 9, 10, 5);
        assert_eq!(neighbors_end, vec![8, 9, 0]);
    }

    #[test]
    fn test_random_topology() {
        let manager = create_topology_manager();
        let neighbors = manager.get_neighbors(&SwarmTopology::Random, 2, 10, 5);
        assert!(neighbors.len() <= 10);
    }

    #[test]
    fn test_dynamic_topology() {
        let manager = create_topology_manager();
        let neighbors = manager.get_neighbors(&SwarmTopology::Dynamic, 3, 10, 5);
        assert!(neighbors.len() <= 5);
    }

    #[test]
    fn test_topology_serialization() {
        let topologies = vec![
            SwarmTopology::Global,
            SwarmTopology::Local,
            SwarmTopology::Ring,
            SwarmTopology::Random,
            SwarmTopology::Dynamic,
        ];

        for topology in topologies {
            let json = serde_json::to_string(&topology)?;
            let _deserialized: SwarmTopology = serde_json::from_str(&json)?;
        }
    }

    #[test]
    fn test_default_topology() {
        let topology = SwarmTopology::default();
        match topology {
            SwarmTopology::Local => assert!(true),
            _ => panic!("Default topology should be Local"),
        }
    }

    #[test]
    fn test_local_neighbors_edge_cases() {
        let manager = create_topology_manager();

        // Test with neighborhood size 0
        let neighbors = manager.get_local_neighbors(5, 10, 0);
        assert_eq!(neighbors.len(), 0);

        // Test with neighborhood larger than population
        let neighbors = manager.get_local_neighbors(3, 8, 15);
        assert_eq!(neighbors.len(), 8);
    }

    #[test]
    fn test_ring_topology_wrapping() {
        let manager = create_topology_manager();

        // Test wrapping at start
        let neighbors = manager.get_ring_neighbors(0, 5);
        assert!(neighbors.contains(&4)); // Previous wraps to last
        assert!(neighbors.contains(&0)); // Self
        assert!(neighbors.contains(&1)); // Next

        // Test wrapping at end
        let neighbors = manager.get_ring_neighbors(4, 5);
        assert!(neighbors.contains(&3)); // Previous
        assert!(neighbors.contains(&4)); // Self
        assert!(neighbors.contains(&0)); // Next wraps to first
    }
}
