//! Production Evolution Marketplace for Cross-Cluster Knowledge Transfer
//!
//! Implements distributed consensus validation, algorithm replication,
//! and secure cross-cluster evolution sharing with Byzantine fault tolerance.

pub mod types;
pub mod consensus;
pub mod transfer;
pub mod security;
pub mod marketplace;
pub mod stats;

// Re-export main types
pub use types::{
    EvolutionPackage, BenchmarkResults, CompatibilityMatrix, UsageStatistics,
    ValidationResult, ValidationMetadata, ResourceUsage, EvolutionParameters,
};
pub use consensus::DistributedConsensus;
pub use transfer::{KnowledgeTransferCoordinator, TransferRequest, TransferPriority};
pub use security::SecurityManager;
pub use marketplace::{EvolutionMarketplace, ClusterInfo, SyncCommand};
pub use stats::MarketplaceStats;

#[cfg(test)]
mod tests;