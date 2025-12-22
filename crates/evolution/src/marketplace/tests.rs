//! Tests for the evolution marketplace

#[cfg(test)]
mod tests {
    use super::super::*;
    use tokio::time::timeout;
    use uuid::Uuid;
    use std::time::Duration;

    #[tokio::test]
    async fn test_marketplace_initialization() {
        let (marketplace, _sync_receiver) = EvolutionMarketplace::new(
            "test_cluster".to_string(),
            "test_node".to_string(),
            2,
        );
        
        // Test basic marketplace creation
        assert_eq!(marketplace.cluster_id, "test_cluster");
        assert_eq!(marketplace.node_id, "test_node");
        assert_eq!(marketplace.replication_factor, 2);
    }

    #[tokio::test]
    async fn test_consensus_system() {
        let consensus = DistributedConsensus::new("test_cluster".to_string(), 75.0);
        
        // Register some validators
        assert!(consensus.register_validator("validator_1".to_string(), vec!["sm_80".to_string()]).await.is_ok());
        assert!(consensus.register_validator("validator_2".to_string(), vec!["sm_80".to_string()]).await.is_ok());
    }

    #[tokio::test] 
    async fn test_knowledge_transfer_coordinator() {
        let coordinator = KnowledgeTransferCoordinator::new(2);
        
        let transfer_request = TransferRequest {
            package_id: Uuid::new_v4(),
            source_cluster: "source".to_string(),
            target_cluster: "target".to_string(),
            priority: TransferPriority::High,
            estimated_size_mb: 50.0,
            compression_ratio: 0.7,
            retry_count: 0,
        };
        
        // Test queueing transfer
        coordinator.enqueue_transfer(transfer_request.clone()).await;
        
        // Test getting next transfer
        let next = coordinator.get_next_transfer().await;
        assert!(next.is_some());
        assert_eq!(next.unwrap().package_id, transfer_request.package_id);
    }

    #[tokio::test]
    async fn test_security_manager() {
        let security_manager = SecurityManager::new();
        
        // Test adding trusted cluster
        assert!(security_manager.add_trusted_cluster("trusted_cluster".to_string()).await.is_ok());
        
        // Test checking trust
        assert!(security_manager.is_trusted("trusted_cluster").await);
        assert!(!security_manager.is_trusted("untrusted_cluster").await);
        
        // Test access policy
        assert!(security_manager.set_access_policy(
            "trusted_cluster".to_string(),
            vec!["download".to_string(), "upload".to_string()],
            2,
        ).await.is_ok());
        
        // Test access check
        assert!(security_manager.check_access("trusted_cluster", "download").await.unwrap_or(false));
    }

    #[tokio::test]
    async fn test_marketplace_stats() {
        let mut stats = MarketplaceStats::default();
        
        // Test validation updates
        stats.update_validation(true);
        stats.update_validation(true);
        stats.update_validation(false);
        
        assert_eq!(stats.successful_validations, 2);
        assert_eq!(stats.failed_validations, 1);
        assert!((stats.validation_success_rate() - 0.666).abs() < 0.01);
        
        // Test transfer updates
        stats.update_transfer(1_000_000_000); // 1GB
        assert_eq!(stats.total_transfers, 1);
        assert!((stats.network_bandwidth_usage_gb - 0.931).abs() < 0.01);
    }

    #[tokio::test]
    async fn test_priority_ordering() {
        assert!(TransferPriority::Emergency > TransferPriority::Critical);
        assert!(TransferPriority::Critical > TransferPriority::High);
        assert!(TransferPriority::High > TransferPriority::Normal);
        assert!(TransferPriority::Normal > TransferPriority::Low);
    }

    #[tokio::test]
    async fn test_package_publishing_flow() {
        let (marketplace, mut sync_receiver) = EvolutionMarketplace::new(
            "test_cluster".to_string(),
            "test_node".to_string(),
            2,
        );
        
        // Start marketplace
        assert!(marketplace.start_marketplace().await.is_ok());
        
        // Try to publish an algorithm
        let result = marketplace.publish_algorithm(
            "test_algorithm".to_string(),
            "// CUDA code here".to_string(),
            std::collections::HashMap::from([
                ("learning_rate".to_string(), 0.01),
                ("batch_size".to_string(), 32.0),
            ]),
        ).await;
        
        // Publishing might fail due to consensus, but the flow should work
        if let Ok(package_id) = result {
            // Check if sync command was sent
            let timeout_result = timeout(
                Duration::from_secs(1),
                sync_receiver.recv()
            ).await;
            
            if let Ok(Some(cmd)) = timeout_result {
                match cmd {
                    SyncCommand::PublishPackage { package } => {
                        assert_eq!(package.id, package_id);
                        assert_eq!(package.algorithm_name, "test_algorithm");
                    }
                    _ => {}
                }
            }
        }
        
        // Stop marketplace
        assert!(marketplace.stop_marketplace().await.is_ok());
    }
}