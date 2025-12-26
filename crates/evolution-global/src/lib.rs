//! Evolution system global deployment with AI safety compliance and secure multi-party evolution
//!
//! This crate provides comprehensive global evolution system capabilities for:
//! - AI safety compliance across all regions
//! - Cross-region evolution coordination
//! - Secure multi-party evolution protocols
//! - Evolution intrusion detection and prevention
//! - Distributed evolution consensus mechanisms

#![warn(missing_docs)]

pub mod ai_safety_compliance;
pub mod consensus_engine;
pub mod cross_region_sync;
pub mod error;
pub mod evolution_coordinator;
pub mod evolution_monitor;
pub mod intrusion_detection;
pub mod secure_multiparty;

pub use error::{EvolutionGlobalError, EvolutionGlobalResult};
pub use evolution_coordinator::{EvolutionConfig, EvolutionCoordinator};

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use evolution_coordinator::EvolutionExecutor;
    use uuid::Uuid;

    struct MockTestExecutor;

    #[async_trait]
    impl EvolutionExecutor for MockTestExecutor {
        async fn execute_evolution(
            &self,
            _request: &evolution_coordinator::EvolutionRequest,
        ) -> EvolutionGlobalResult<Uuid> {
            Ok(Uuid::new_v4())
        }

        async fn validate_evolution(&self, _evolution_id: Uuid) -> EvolutionGlobalResult<f64> {
            Ok(0.98)
        }

        async fn rollback_evolution(
            &self,
            _snapshot: &evolution_coordinator::RollbackSnapshot,
        ) -> EvolutionGlobalResult<()> {
            Ok(())
        }
    }

    #[test]
    fn test_evolution_global_creation() {
        // This will fail initially (RED phase)
        let config = EvolutionConfig::default();
        let executor = std::sync::Arc::new(MockTestExecutor);
        let coordinator = EvolutionCoordinator::new(config, executor);
        assert!(coordinator.is_ok());
    }
}
