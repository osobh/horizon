//! Type definitions for swarm network management

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Particle partitioning strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PartitionStrategy {
    /// Round-robin distribution
    RoundRobin,
    /// Hash-based partitioning
    HashBased,
    /// Range-based partitioning
    RangeBased,
    /// Network-aware partitioning
    NetworkAware,
    /// Load-aware partitioning
    LoadAware,
}

/// Metadata about a particle for migration decisions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParticleMetadata {
    /// Particle identifier
    pub id: String,
    /// Current node assignment
    pub current_node: String,
    /// Computational cost estimate
    pub compute_cost: f64,
    /// Communication frequency with other particles
    pub communication_frequency: HashMap<String, f64>,
    /// Migration count (for stability)
    pub migration_count: u32,
    /// Last migration timestamp
    pub last_migration: u64,
}

/// Node capacity information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeCapacity {
    /// Maximum particles this node can handle
    pub max_particles: usize,
    /// Computational capacity (relative units)
    pub compute_capacity: f64,
    /// Memory capacity in bytes
    pub memory_capacity: usize,
    /// Network bandwidth capacity
    pub bandwidth_capacity: f64,
    /// Current utilization (0.0 - 1.0)
    pub current_utilization: f64,
}

/// Edge weight information for network connections
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeWeight {
    /// Network latency in milliseconds
    pub latency: f64,
    /// Available bandwidth in Mbps
    pub bandwidth: f64,
    /// Connection reliability (0.0 - 1.0)
    pub reliability: f64,
    /// Connection cost/priority
    pub cost: f64,
}

/// Migration plan for rebalancing particles
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationPlan {
    /// Migrations to perform
    pub migrations: Vec<Migration>,
    /// Expected improvement in load balance
    pub expected_improvement: f64,
    /// Estimated migration cost
    pub migration_cost: f64,
}

/// Individual migration operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Migration {
    /// Particle to migrate
    pub particle_id: String,
    /// Source node
    pub from_node: String,
    /// Destination node
    pub to_node: String,
    /// Migration priority (higher = more urgent)
    pub priority: f64,
    /// Expected benefit of this migration
    pub expected_benefit: f64,
}

/// Network partition information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkPartition {
    /// Partition identifier
    pub id: String,
    /// Nodes in this partition
    pub nodes: Vec<String>,
    /// Particles assigned to this partition
    pub particles: Vec<String>,
    /// Partition load
    pub load: f64,
    /// Inter-partition communication cost
    pub communication_cost: f64,
}

/// Load balancing metrics
#[derive(Debug, Clone, Default)]
pub struct LoadBalanceMetrics {
    /// Load imbalance factor (0.0 = perfect balance, 1.0 = maximum imbalance)
    pub imbalance_factor: f64,
    /// Total number of particles
    pub total_particles: usize,
    /// Number of active nodes
    pub active_nodes: usize,
    /// Average particles per node
    pub avg_particles_per_node: f64,
    /// Standard deviation of particle distribution
    pub distribution_std_dev: f64,
    /// Number of pending migrations
    pub pending_migrations: usize,
}

impl Default for PartitionStrategy {
    fn default() -> Self {
        Self::LoadAware
    }
}

impl Default for NodeCapacity {
    fn default() -> Self {
        Self {
            max_particles: 100,
            compute_capacity: 1.0,
            memory_capacity: 1024 * 1024 * 1024, // 1GB
            bandwidth_capacity: 100.0,           // 100 Mbps
            current_utilization: 0.0,
        }
    }
}

impl Default for EdgeWeight {
    fn default() -> Self {
        Self {
            latency: 10.0,     // 10ms
            bandwidth: 100.0,  // 100 Mbps
            reliability: 0.99, // 99% reliable
            cost: 1.0,         // Default cost
        }
    }
}
