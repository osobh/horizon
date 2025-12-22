//! Network topology management for distributed SwarmAgentic systems
//!
//! This module handles particle distribution, load balancing, and network-aware
//! topology management across distributed nodes.

mod load_balancer;
mod network_graph;
mod partition_manager;
mod topology;
mod types;

#[cfg(test)]
mod tests;

pub use load_balancer::LoadBalancer;
pub use network_graph::NetworkGraph;
pub use partition_manager::PartitionManager;
pub use topology::NetworkTopology;
pub use types::{
    EdgeWeight, LoadBalanceMetrics, Migration, MigrationPlan, NetworkPartition, NodeCapacity,
    ParticleMetadata, PartitionStrategy,
};
