//! Integration tests for evolution coordinator

use async_trait::async_trait;
use chrono::Utc;
use stratoswarm_evolution_global::{
    error::EvolutionGlobalResult,
    evolution_coordinator::{
        EvolutionConfig, EvolutionCoordinator, EvolutionExecutor, EvolutionPhase, EvolutionRequest,
        EvolutionStatus, RollbackSnapshot,
    },
};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;
use uuid::Uuid;

/// Mock executor that simulates real evolution operations
struct IntegrationTestExecutor {
    executions: Arc<Mutex<Vec<(Uuid, EvolutionRequest)>>>,
    validations: Arc<Mutex<Vec<Uuid>>>,
    rollbacks: Arc<Mutex<Vec<RollbackSnapshot>>>,
}

impl IntegrationTestExecutor {
    fn new() -> Self {
        Self {
            executions: Arc::new(Mutex::new(Vec::new())),
            validations: Arc::new(Mutex::new(Vec::new())),
            rollbacks: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

#[async_trait]
impl EvolutionExecutor for IntegrationTestExecutor {
    async fn execute_evolution(&self, request: &EvolutionRequest) -> EvolutionGlobalResult<Uuid> {
        let evolution_id = Uuid::new_v4();
        let mut executions = self.executions.lock().await;
        executions.push((evolution_id, request.clone()));
        Ok(evolution_id)
    }

    async fn validate_evolution(&self, evolution_id: Uuid) -> EvolutionGlobalResult<f64> {
        let mut validations = self.validations.lock().await;
        validations.push(evolution_id);
        Ok(0.98) // High validation score
    }

    async fn rollback_evolution(&self, snapshot: &RollbackSnapshot) -> EvolutionGlobalResult<()> {
        let mut rollbacks = self.rollbacks.lock().await;
        rollbacks.push(snapshot.clone());
        Ok(())
    }
}

fn create_test_request(region: &str, model_id: &str) -> EvolutionRequest {
    EvolutionRequest {
        region: region.to_string(),
        model_id: model_id.to_string(),
        evolution_type: "integration_test".to_string(),
        parameters: HashMap::new(),
        priority: 5,
        requires_validation: true,
        cross_region_sync: false,
    }
}

#[tokio::test]
async fn test_complete_evolution_workflow() {
    let config = EvolutionConfig::default();
    let executor = Arc::new(IntegrationTestExecutor::new());
    let coordinator = EvolutionCoordinator::new(config, executor.clone()).unwrap();

    // Start evolution
    let request = create_test_request("us-east-1", "model-123");
    let evolution_id = coordinator.start_evolution(request).await.unwrap();

    // Check status
    let status = coordinator
        .get_evolution_status(evolution_id)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(status.phase, EvolutionPhase::Execution);

    // Validate evolution
    let validation_score = coordinator.validate_evolution(evolution_id).await.unwrap();
    assert_eq!(validation_score, 0.98);

    // Check status after validation
    let status = coordinator
        .get_evolution_status(evolution_id)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(status.phase, EvolutionPhase::Completion);
    assert_eq!(status.validation_score, Some(0.98));
}

#[tokio::test]
async fn test_concurrent_evolutions() {
    let config = EvolutionConfig {
        max_concurrent_evolutions: 3,
        ..Default::default()
    };
    let executor = Arc::new(IntegrationTestExecutor::new());
    let coordinator = EvolutionCoordinator::new(config, executor).unwrap();

    // Start multiple evolutions concurrently
    let mut evolution_ids = Vec::new();
    for i in 0..3 {
        let request = create_test_request("us-east-1", &format!("model-{}", i));
        let id = coordinator.start_evolution(request).await.unwrap();
        evolution_ids.push(id);
    }

    // Verify all are active
    for id in &evolution_ids {
        let status = coordinator.get_evolution_status(*id).await.unwrap();
        assert!(status.is_some());
    }

    // Try to start one more - should fail due to limit
    let request = create_test_request("us-east-1", "model-extra");
    let result = coordinator.start_evolution(request).await;
    assert!(result.is_err());

    // Cancel one evolution
    coordinator
        .cancel_evolution(evolution_ids[0])
        .await
        .unwrap();

    // Now we should be able to start another
    let request = create_test_request("us-east-1", "model-new");
    let result = coordinator.start_evolution(request).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_evolution_history_tracking() {
    let config = EvolutionConfig::default();
    let executor = Arc::new(IntegrationTestExecutor::new());
    let coordinator = EvolutionCoordinator::new(config, executor).unwrap();

    // Create and complete several evolutions
    for i in 0..5 {
        let request = create_test_request("us-west-2", &format!("model-{}", i));
        let evolution_id = coordinator.start_evolution(request).await.unwrap();

        // Validate to move to completion
        coordinator.validate_evolution(evolution_id).await.unwrap();
    }

    // Clean up completed evolutions
    let cleaned = coordinator.cleanup_completed_evolutions().await.unwrap();
    assert_eq!(cleaned, 5);

    // Check history
    let history = coordinator
        .get_evolution_history(Some("us-west-2".to_string()), Some(3))
        .await
        .unwrap();
    assert_eq!(history.len(), 3); // Limited to 3
    assert!(history.iter().all(|h| h.success));
}

#[tokio::test]
async fn test_cross_region_evolution() {
    let config = EvolutionConfig {
        enable_cross_region: true,
        ..Default::default()
    };
    let executor = Arc::new(IntegrationTestExecutor::new());
    let coordinator = EvolutionCoordinator::new(config, executor).unwrap();

    let regions = vec!["us-east-1", "eu-west-1", "ap-south-1"];
    let requests: Vec<_> = regions
        .iter()
        .map(|region| create_test_request(region, "model-global"))
        .collect();

    let evolution_ids = coordinator
        .schedule_cross_region_evolution(requests)
        .await
        .unwrap();

    assert_eq!(evolution_ids.len(), 3);

    // Verify all regions have active evolutions
    for (i, region) in regions.iter().enumerate() {
        let count = coordinator.get_active_evolutions_count(region).await;
        assert_eq!(count, 1);
    }
}

#[tokio::test]
async fn test_rollback_functionality() {
    let config = EvolutionConfig {
        enable_rollback: true,
        ..Default::default()
    };
    let executor = Arc::new(IntegrationTestExecutor::new());
    let coordinator = EvolutionCoordinator::new(config, executor.clone()).unwrap();

    // Start evolution
    let request = create_test_request("us-east-1", "model-rollback");
    let evolution_id = coordinator.start_evolution(request).await.unwrap();

    // Create additional snapshots
    for _ in 0..3 {
        coordinator
            .create_rollback_snapshot(evolution_id)
            .await
            .unwrap();
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    }

    // Perform rollback
    coordinator.rollback_evolution(evolution_id).await.unwrap();

    // Verify rollback was executed
    let status = coordinator
        .get_evolution_status(evolution_id)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(status.phase, EvolutionPhase::Rollback);

    // Check executor received rollback
    let rollbacks = executor.rollbacks.lock().await;
    assert_eq!(rollbacks.len(), 1);
}

#[tokio::test]
async fn test_evolution_validation_failure() {
    let config = EvolutionConfig {
        validation_threshold: 0.95,
        ..Default::default()
    };

    // Create custom executor that returns low validation score
    struct LowScoreExecutor;

    #[async_trait]
    impl EvolutionExecutor for LowScoreExecutor {
        async fn execute_evolution(
            &self,
            _request: &EvolutionRequest,
        ) -> EvolutionGlobalResult<Uuid> {
            Ok(Uuid::new_v4())
        }

        async fn validate_evolution(&self, _evolution_id: Uuid) -> EvolutionGlobalResult<f64> {
            Ok(0.5) // Below threshold
        }

        async fn rollback_evolution(
            &self,
            _snapshot: &RollbackSnapshot,
        ) -> EvolutionGlobalResult<()> {
            Ok(())
        }
    }

    let coordinator = EvolutionCoordinator::new(config, Arc::new(LowScoreExecutor)).unwrap();

    // Start and validate evolution
    let request = create_test_request("us-east-1", "model-fail");
    let evolution_id = coordinator.start_evolution(request).await.unwrap();
    let validation_score = coordinator.validate_evolution(evolution_id).await.unwrap();

    // Check that evolution failed due to low score
    let status = coordinator
        .get_evolution_status(evolution_id)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(status.phase, EvolutionPhase::Failed);
    assert!(status.error.is_some());
    assert_eq!(validation_score, 0.5);
}

#[tokio::test]
async fn test_evolution_timeout_handling() {
    let config = EvolutionConfig {
        evolution_timeout_ms: 100, // Very short timeout
        ..Default::default()
    };
    let executor = Arc::new(IntegrationTestExecutor::new());
    let coordinator = EvolutionCoordinator::new(config, executor).unwrap();

    // Start evolution
    let request = create_test_request("us-east-1", "model-timeout");
    let evolution_id = coordinator.start_evolution(request).await.unwrap();

    // Wait for timeout
    tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;

    // Evolution should still be active (timeout handling would be implemented in real system)
    let status = coordinator
        .get_evolution_status(evolution_id)
        .await
        .unwrap();
    assert!(status.is_some());
}

#[tokio::test]
async fn test_disabled_features() {
    // Test with rollback disabled
    let config = EvolutionConfig {
        enable_rollback: false,
        enable_cross_region: false,
        ..Default::default()
    };
    let executor = Arc::new(IntegrationTestExecutor::new());
    let coordinator = EvolutionCoordinator::new(config, executor).unwrap();

    // Start evolution
    let request = create_test_request("us-east-1", "model-no-features");
    let evolution_id = coordinator.start_evolution(request).await.unwrap();

    // Try rollback - should fail
    let rollback_result = coordinator.rollback_evolution(evolution_id).await;
    assert!(rollback_result.is_err());

    // Try cross-region - should fail
    let requests = vec![
        create_test_request("us-east-1", "model-1"),
        create_test_request("eu-west-1", "model-2"),
    ];
    let cross_region_result = coordinator.schedule_cross_region_evolution(requests).await;
    assert!(cross_region_result.is_err());
}
