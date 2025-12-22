//! Tests for intent orchestrator

#[cfg(test)]
mod tests {
    use super::super::*;
    use candle_core::Device;

    #[tokio::test]
    async fn test_orchestrator_creation() -> Result<(), Box<dyn std::error::Error>> {
        let device = Device::Cpu;
        let config = config::OrchestratorConfig::default();
        let orchestrator = orchestrator::IntentOrchestrator::new(device, config).await;
        assert!(orchestrator.is_ok());
    }

    #[tokio::test]
    async fn test_intent_parsing() -> Result<(), Box<dyn std::error::Error>> {
        let device = Device::Cpu;
        let config = config::OrchestratorConfig::default();
        let orchestrator = orchestrator::IntentOrchestrator::new(device, config)
            .await
            .unwrap();
        
        let result = orchestrator.parse_intent("Create a new cluster", None).await;
        assert!(result.is_ok());
        
        let intent = result.unwrap();
        assert!(!intent.text.is_empty());
        assert!(intent.confidence >= 0.0 && intent.confidence <= 1.0);
    }

    #[tokio::test]
    async fn test_entity_extraction() -> Result<(), Box<dyn std::error::Error>> {
        let device = Device::Cpu;
        let config = config::OrchestratorConfig::default();
        let extractor = orchestrator::EntityExtractor::new(device, &config)
            .await
            .unwrap();
        
        let text = "Scale the cluster to 10 nodes";
        let tokens: Vec<String> = text.split_whitespace().map(|s| s.to_string()).collect();
        
        let entities = extractor.extract(text, &tokens).await?;
        assert!(!entities.is_empty());
        
        // Should find the number "10"
        let number_entity = entities.iter().find(|e| e.text == "10");
        assert!(number_entity.is_some());
    }

    #[tokio::test]
    async fn test_action_planning() -> Result<(), Box<dyn std::error::Error>> {
        let planner = planning::ActionPlanner::new().await?;
        
        let mut params = std::collections::HashMap::new();
        params.insert(
            "resource".to_string(),
            types::EntityValue::String("cluster".to_string()),
        );
        
        let steps = planner.create_action_steps("create", params).await?;
        assert!(!steps.is_empty());
        assert_eq!(steps.get(0).ok_or("Index out of bounds")?.action_type, execution::ActionType::Deploy);
    }

    #[tokio::test]
    async fn test_resource_allocation() -> Result<(), Box<dyn std::error::Error>> {
        let allocator = planning::ResourceAllocator::new().await?;
        let steps = vec![];
        
        let requirements = allocator.estimate_requirements(&steps).await?;
        assert!(requirements.cpu_cores > 0.0);
        assert!(requirements.memory_gb > 0.0);
        
        let allocation_id = allocator.allocate_resources(&requirements).await?;
        assert!(!allocation_id.is_empty());
        
        let release_result = allocator.release_resources(&allocation_id).await;
        assert!(release_result.is_ok());
    }

    #[tokio::test]
    async fn test_metrics_collection() -> Result<(), Box<dyn std::error::Error>> {
        let mut metrics = metrics::OrchestrationMetrics::default();
        
        // Record classification
        metrics.record_classification(
            &intents::IntentType::Query { query_type: "test".to_string() },
            true,
            100.0,
        );
        
        assert_eq!(metrics.total_intents, 1);
        assert_eq!(metrics.successful_classifications, 1);
        assert_eq!(metrics.avg_classification_time_ms, 100.0);
        
        // Record execution
        metrics.record_execution(execution::ExecutionStatus::Completed, 200.0);
        
        assert_eq!(metrics.total_executions, 1);
        assert_eq!(metrics.successful_executions, 1);
        assert_eq!(metrics.avg_execution_time_ms, 200.0);
    }

    #[tokio::test]
    async fn test_context_management() -> Result<(), Box<dyn std::error::Error>> {
        let session = context::SessionContext::default();
        let intent = intents::Intent {
            id: uuid::Uuid::new_v4().to_string(),
            text: "Test intent".to_string(),
            intent_type: intents::IntentType::Query { query_type: "test".to_string() },
            confidence: 0.9,
            entities: vec![],
            context: None,
            created_at: chrono::Utc::now(),
            metadata: std::collections::HashMap::new(),
        };
        
        let mut ctx = context::IntentContext::new(intent, session);
        
        // Test role checking
        assert!(ctx.has_role("user"));
        assert!(!ctx.has_role("admin"));
        
        // Test action allowance
        assert!(ctx.is_action_allowed("read"));
        ctx.ambient.environment = "production".to_string();
        assert!(!ctx.is_action_allowed("delete"));
    }

    #[tokio::test]
    async fn test_execution_record() -> Result<(), Box<dyn std::error::Error>> {
        let action_plan = execution::ActionPlan {
            id: uuid::Uuid::new_v4().to_string(),
            steps: vec![],
            resources: execution::ResourceRequirements::default(),
            retry_policy: execution::RetryPolicy::default(),
            success_criteria: vec![],
            metadata: std::collections::HashMap::new(),
        };
        
        let mut record = execution::ExecutionRecord::new(
            uuid::Uuid::new_v4().to_string(),
            action_plan,
        );
        
        assert_eq!(record.status, execution::ExecutionStatus::Pending);
        
        record.start();
        assert_eq!(record.status, execution::ExecutionStatus::Running);
        
        let result = execution::ExecutionResult {
            success: true,
            data: std::collections::HashMap::new(),
            error: None,
            metrics: std::collections::HashMap::new(),
            artifacts: vec![],
        };
        
        record.complete(result);
        assert_eq!(record.status, execution::ExecutionStatus::Completed);
        assert!(record.duration.is_some());
    }

    #[tokio::test]
    async fn test_config_validation() -> Result<(), Box<dyn std::error::Error>> {
        let mut config = config::OrchestratorConfig::default();
        assert!(config.validate().is_ok());
        
        // Test invalid threshold
        config.classification_threshold = 1.5;
        assert!(config.validate().is_err());
        
        // Test production config
        let prod_config = config::OrchestratorConfig::production();
        assert!(prod_config.validate().is_ok());
        assert_eq!(prod_config.classification_threshold, 0.85);
        
        // Test development config
        let dev_config = config::OrchestratorConfig::development();
        assert!(dev_config.validate().is_ok());
        assert!(dev_config.debug_mode);
    }
}
