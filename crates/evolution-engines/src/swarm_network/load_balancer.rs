//! Load balancing for distributed swarm networks

use super::types::{LoadBalanceMetrics, Migration, MigrationPlan, NodeCapacity};
use crate::error::EvolutionEngineResult;
use crate::swarm_distributed::{DistributedSwarmConfig, LoadBalanceStrategy};
use std::collections::HashMap;

/// Load balancer for distributing computational load
pub struct LoadBalancer {
    /// Load balancing strategy
    pub(crate) strategy: LoadBalanceStrategy,
    /// Node capacity information
    pub(crate) node_capacities: HashMap<String, NodeCapacity>,
    /// Current load distribution
    pub(crate) load_distribution: HashMap<String, f64>,
    /// Migration threshold
    pub(crate) migration_threshold: f64,
}

impl LoadBalancer {
    /// Create new load balancer
    pub async fn new(config: DistributedSwarmConfig) -> EvolutionEngineResult<Self> {
        Ok(Self {
            strategy: config.load_balance_config.strategy,
            node_capacities: HashMap::new(),
            load_distribution: HashMap::new(),
            migration_threshold: config.load_balance_config.rebalance_threshold,
        })
    }

    /// Add node capacity information
    pub async fn add_node_capacity(
        &mut self,
        node_id: String,
        capacity: NodeCapacity,
    ) -> EvolutionEngineResult<()> {
        self.node_capacities.insert(node_id.clone(), capacity);
        self.load_distribution.insert(node_id, 0.0);
        Ok(())
    }

    /// Remove node
    pub async fn remove_node(&mut self, node_id: &str) -> EvolutionEngineResult<()> {
        self.node_capacities.remove(node_id);
        self.load_distribution.remove(node_id);
        Ok(())
    }

    /// Get load balance metrics
    pub async fn get_metrics(&self) -> EvolutionEngineResult<LoadBalanceMetrics> {
        let total_particles: usize = self
            .load_distribution
            .values()
            .map(|&load| load as usize)
            .sum();
        let active_nodes = self.node_capacities.len();
        let avg_particles = if active_nodes > 0 {
            total_particles as f64 / active_nodes as f64
        } else {
            0.0
        };

        // Calculate standard deviation
        let variance: f64 = self
            .load_distribution
            .values()
            .map(|&load| (load - avg_particles).powi(2))
            .sum::<f64>()
            / active_nodes.max(1) as f64;
        let std_dev = variance.sqrt();

        // Calculate imbalance factor
        let max_load = self
            .load_distribution
            .values()
            .fold(0.0f64, |a, &b| a.max(b));
        let min_load = self
            .load_distribution
            .values()
            .fold(f64::INFINITY, |a, &b| a.min(b));
        let imbalance_factor = if avg_particles > 0.0 {
            (max_load - min_load) / avg_particles
        } else {
            0.0
        };

        Ok(LoadBalanceMetrics {
            imbalance_factor,
            total_particles,
            active_nodes,
            avg_particles_per_node: avg_particles,
            distribution_std_dev: std_dev,
            pending_migrations: 0, // Stub implementation
        })
    }

    /// Create migration plan to balance load
    pub async fn create_migration_plan(&self) -> EvolutionEngineResult<MigrationPlan> {
        let mut migrations = Vec::new();

        if self.load_distribution.is_empty() {
            return Ok(MigrationPlan {
                migrations,
                expected_improvement: 0.0,
                migration_cost: 0.0,
            });
        }

        // Find overloaded and underloaded nodes
        let avg_load =
            self.load_distribution.values().sum::<f64>() / self.load_distribution.len() as f64;

        for (node_id, &current_load) in &self.load_distribution {
            if current_load > avg_load * (1.0 + self.migration_threshold) {
                // Node is overloaded - create migration from this node
                if let Some(target_node) = self.find_least_loaded_node() {
                    if target_node != *node_id {
                        migrations.push(Migration {
                            particle_id: format!("particle_from_{}", node_id),
                            from_node: node_id.clone(),
                            to_node: target_node,
                            priority: current_load - avg_load,
                            expected_benefit: (current_load - avg_load) * 0.5,
                        });
                    }
                }
            }
        }

        Ok(MigrationPlan {
            migrations,
            expected_improvement: 0.1, // Stub implementation
            migration_cost: 0.05,      // Stub implementation
        })
    }

    /// Create evacuation plan for node removal
    pub async fn create_evacuation_plan(
        &self,
        node_id: &str,
        particles: &[String],
    ) -> EvolutionEngineResult<MigrationPlan> {
        let mut migrations = Vec::new();

        for particle_id in particles {
            if let Some(target_node) = self.find_least_loaded_node_excluding(node_id) {
                migrations.push(Migration {
                    particle_id: particle_id.clone(),
                    from_node: node_id.to_string(),
                    to_node: target_node,
                    priority: 1.0, // High priority for evacuation
                    expected_benefit: 1.0,
                });
            }
        }

        Ok(MigrationPlan {
            migrations,
            expected_improvement: 1.0, // Full evacuation
            migration_cost: particles.len() as f64 * 0.1,
        })
    }

    /// Update node load
    pub fn update_node_load(&mut self, node_id: &str, load: f64) {
        self.load_distribution.insert(node_id.to_string(), load);
    }

    /// Get current load for a node
    pub fn get_node_load(&self, node_id: &str) -> f64 {
        self.load_distribution.get(node_id).copied().unwrap_or(0.0)
    }

    /// Find least loaded node
    fn find_least_loaded_node(&self) -> Option<String> {
        self.load_distribution
            .iter()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(node_id, _)| node_id.clone())
    }

    /// Find least loaded node excluding a specific node
    fn find_least_loaded_node_excluding(&self, exclude_node: &str) -> Option<String> {
        self.load_distribution
            .iter()
            .filter(|(node_id, _)| *node_id != exclude_node)
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(node_id, _)| node_id.clone())
    }

    /// Get all node capacities
    pub fn get_node_capacities(&self) -> &HashMap<String, NodeCapacity> {
        &self.node_capacities
    }

    /// Get load distribution
    pub fn get_load_distribution(&self) -> &HashMap<String, f64> {
        &self.load_distribution
    }
}
