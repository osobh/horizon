//! Distributed SwarmAgentic runtime for multi-node particle swarm optimization
//!
//! This module implements distributed execution of SwarmAgentic algorithms across
//! multiple nodes in a network, providing scalability and fault tolerance.

mod config;
mod engine;
mod engine_actor;
mod message_bus;
mod messages;
mod network;
mod node_manager;
mod types;

#[cfg(test)]
mod tests;

pub use config::{
    DistributedSwarmConfig, FaultToleranceConfig, LoadBalanceConfig, LoadBalanceStrategy,
    NetworkConfig, RecoveryStrategy,
};
pub use engine::DistributedSwarmEngine;
// Re-export actor types for new code
pub use engine_actor::{
    create_distributed_swarm_actor, DistributedSwarmActor, DistributedSwarmHandle,
    SwarmEngineRequest,
};
pub use message_bus::{MessageBus, MessageHandler};
pub use messages::{CheckpointData, DistributedMessage, MigrationParticle};
pub use network::{Connection, ConnectionPool, ConnectionStatus, NetworkTransport};
pub use node_manager::NodeManager;
pub use types::{
    ClusterStatus, DiscoveryProtocol, DiscoveryRequest, DiscoveryService, Migration, MigrationPlan,
    NodeStatus, SwarmNode,
};
