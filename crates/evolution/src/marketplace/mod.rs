//! Production Evolution Marketplace for Cross-Cluster Knowledge Transfer
//!
//! Implements distributed consensus validation, algorithm replication,
//! and secure cross-cluster evolution sharing with Byzantine fault tolerance.

pub mod consensus;
pub mod marketplace;
pub mod security;
pub mod stats;
pub mod transfer;
pub mod types;

// Re-export main types
pub use consensus::DistributedConsensus;
pub use marketplace::{ClusterInfo, EvolutionMarketplace, SyncCommand};
pub use security::SecurityManager;
pub use stats::MarketplaceStats;
pub use transfer::{KnowledgeTransferCoordinator, TransferPriority, TransferRequest};
pub use types::{
    BenchmarkResults, CompatibilityMatrix, EvolutionPackage, EvolutionParameters, ResourceUsage,
    UsageStatistics, ValidationMetadata, ValidationResult,
};

#[cfg(test)]
mod tests;
