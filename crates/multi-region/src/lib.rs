//! Multi-region deployment and data sovereignty management
//!
//! This crate provides comprehensive multi-region capabilities for:
//! - Data sovereignty enforcement
//! - Region-aware load balancing
//! - Cross-region replication
//! - Secure inter-region tunnels
//! - Region-specific compliance mapping

#![warn(missing_docs)]

pub mod compliance_mapping;
pub mod data_sovereignty;
pub mod error;
pub mod load_balancer;
pub mod region_manager;
pub mod replication;
pub mod tunnels;

pub use compliance_mapping::{
    ComplianceFramework, ComplianceMappingConfig, ComplianceMappingManager,
};
pub use data_sovereignty::{DataSovereignty, SovereigntyRule};
pub use error::{MultiRegionError, MultiRegionResult};
pub use load_balancer::{LoadBalancer, LoadBalancerConfig, LoadBalancingAlgorithm};
pub use region_manager::{RegionConfig, RegionManager};
pub use replication::{ConsistencyModel, ReplicationConfig, ReplicationManager};
pub use tunnels::{TlsConfiguration, TunnelConfig, TunnelManager};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_region_framework_creation() {
        // This will fail initially (RED phase)
        let config = RegionConfig::default();
        let manager = RegionManager::new(config);
        assert!(manager.is_ok());
    }
}
