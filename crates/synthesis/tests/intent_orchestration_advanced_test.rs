//! Intent Orchestration Advanced Features Tests - RED Phase

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

use tokio::sync::RwLock;
use uuid::Uuid;
use rand::prelude::*;
use candle_core::{Device, Tensor, DType, Shape};
use candle_nn::{Linear, Module, VarBuilder, linear, sequential, Seq, activation, embedding, Embedding};
use ndarray::{Array2, Array1, Array3};
use serde::{Deserialize, Serialize};
use regex::Regex;

use stratoswarm_synthesis::{Synthesizer, Pipeline, Executor};
use stratoswarm_agent_core::{Agent, AgentId};
use stratoswarm_memory::pool::MemoryPool;

// Import all the shared types from the core test file
pub use intent_orchestrator_test::*;

#[cfg(test)]
mod advanced_tests {
    use super::*;
    use tokio_test;

    #[tokio::test]
    async fn test_action_plan_execution() {
        let device = Device::Cpu;
        let config = OrchestratorConfig::default();
        let orchestrator = IntentOrchestrator::new(device, config).await.expect("Should initialize");
        
        let action_plan = ActionPlan {
            id: Uuid::new_v4(),
            steps: vec![
                ActionStep {
                    id: Uuid::new_v4(), action_type: ActionType::AllocateResource,
                    parameters: { let mut params = HashMap::new();
                        params.insert("cpu".to_string(), EntityValue::Integer(4));
                        params.insert("memory".to_string(), EntityValue::Integer(8)); params },
                    dependencies: vec![], timeout: Duration::from_secs(300),
                    retry_policy: RetryPolicy { max_attempts: 3,
                        backoff_strategy: BackoffStrategy::Exponential { initial_delay: Duration::from_secs(1), multiplier: 2.0 },
                        retry_conditions: vec![] }, parallel_group: None,
                },
                ActionStep {
                    id: Uuid::new_v4(), action_type: ActionType::CreateAgent,
                    parameters: { let mut params = HashMap::new();
                        params.insert("agent_type".to_string(), EntityValue::String("worker".to_string()));
                        params.insert("region".to_string(), EntityValue::String("us-west-1".to_string())); params },
                    dependencies: vec![], timeout: Duration::from_secs(600),
                    retry_policy: RetryPolicy { max_attempts: 2,
                        backoff_strategy: BackoffStrategy::Linear { delay: Duration::from_secs(5) },
                        retry_conditions: vec![] }, parallel_group: Some("initialization".to_string()),
                },
            ],
            estimated_duration: Duration::from_secs(900),
            required_resources: ResourceRequirements { cpu_cores: 4, memory_gb: 8, gpu_count: 0,
                storage_gb: 100, network_gbps: 1.0 },
            success_criteria: vec![SuccessCriterion {
                metric: "agent_count".to_string(), operator: ComparisonOperator::GreaterThanOrEqual,
                target_value: EntityValue::Integer(1), timeout: Duration::from_secs(300),
            }],
            rollback_plan: None,
        };
        
        let execution_record = orchestrator.execute_action_plan(&action_plan).await.expect("Should execute");
        
        assert_eq!(execution_record.action_plan.id, action_plan.id);
        assert!(matches!(execution_record.execution_status, ExecutionStatus::Completed | ExecutionStatus::Running));
        assert!(execution_record.end_time.is_some() || execution_record.execution_status == ExecutionStatus::Running);
        
        assert!(execution_record.resource_usage.cpu_seconds >= 0.0);
        assert!(execution_record.resource_usage.memory_peak_gb >= 0.0);
        
        assert!(!execution_record.outcomes.is_empty());
        for outcome in &execution_record.outcomes {
            assert!(!outcome.step_id.is_nil());
            assert!(action_plan.steps.iter().any(|s| s.id == outcome.step_id));
        }
    }

    #[tokio::test]
    async fn test_multimodal_intent_understanding() {
        let device = Device::Cpu;
        let config = OrchestratorConfig::default();
        let orchestrator = IntentOrchestrator::new(device, config).await.expect("Should initialize");
        
        let complex_intents = vec![
            "Create a 5-node GPU cluster in us-east-1 with 32GB RAM each, deploy the ML training service, and set up monitoring for memory usage above 80%",
            "If CPU usage exceeds 90% for more than 5 minutes, scale up by 2 instances, but don't exceed 10 total instances",
            "Migrate the user database from us-west-1 to eu-central-1, maintain read replicas in both regions, and update the connection strings",
            "Debug the intermittent connection timeouts in the API gateway, check network policies, and generate a report with recommendations",
        ];
        
        for (i, input) in complex_intents.iter().enumerate() {
            let context = IntentContext {
                conversation_history: VecDeque::new(),
                active_session: Some(SessionContext {
                    session_id: format!("session_{}", i), start_time: SystemTime::now() - Duration::from_hours(1),
                    last_activity: SystemTime::now(), active_resources: vec!["cluster-1".to_string(), "db-main".to_string()],
                    execution_history: vec![] }),
                user_preferences: { let mut prefs = HashMap::new();
                    prefs.insert("default_region".to_string(), "us-west-1".to_string());
                    prefs.insert("cost_optimization".to_string(), "true".to_string()); prefs },
                system_state: SystemState { active_agents: 15,
                    resource_utilization: { let mut util = HashMap::new();
                        util.insert("cpu".to_string(), 0.75); util.insert("memory".to_string(), 0.6);
                        util.insert("gpu".to_string(), 0.8); util },
                    cluster_health: { let mut health = HashMap::new();
                        health.insert("us-west-1".to_string(), 0.95); health.insert("us-east-1".to_string(), 0.88); health },
                    pending_operations: 3, error_rate: 0.01 },
                ambient_context: AmbientContext { time_of_day: 0.6, system_load: 0.7, network_conditions: 0.9,
                    maintenance_window: false, emergency_mode: false },
            };
            
            let intent = orchestrator.parse_intent(input, Some(context)).await.expect("Should parse complex intent");
            
            assert!(intent.entities.len() >= 3);
            let entities_with_relations: Vec<_> = intent.entities.iter().filter(|e| !e.relations.is_empty()).collect();
            assert!(!entities_with_relations.is_empty());
            
            if input.contains("If") || input.contains("exceed") {
                let condition_entities: Vec<_> = intent.entities.iter().filter(|e| e.entity_type == EntityType::Condition).collect();
            }
            
            let action_plan = orchestrator.orchestrate_intent(&intent).await.expect("Should orchestrate complex intent");
            assert!(action_plan.steps.len() >= 2);
            
            let steps_with_deps: Vec<_> = action_plan.steps.iter().filter(|s| !s.dependencies.is_empty()).collect();
            if action_plan.steps.len() > 2 {
                assert!(!steps_with_deps.is_empty());
            }
        }
    }

    #[tokio::test]
    async fn test_neural_model_training() {
        let device = Device::Cpu;
        let config = OrchestratorConfig {
            transformer_layers: 2, transformer_heads: 4, hidden_size: 128, vocab_size: 10000,
            max_sequence_length: 128, dropout_rate: 0.1, learning_rate: 0.01,
        };
        let mut orchestrator = IntentOrchestrator::new(device, config).await.expect("Should initialize");
        
        let mut training_data = Vec::new();
        let intent_templates = vec![
            ("create {} cluster", IntentType::CreateResource { resource_type: "cluster".to_string() }),
            ("scale up {}", IntentType::Scale { direction: ScaleDirection::Up }),
            ("scale down {}", IntentType::Scale { direction: ScaleDirection::Down }),
            ("deploy {} service", IntentType::Deploy { deployment_type: "service".to_string() }),
            ("monitor {} usage", IntentType::Monitor { metric: "usage".to_string() }),
            ("backup {} data", IntentType::Backup { scope: "data".to_string() }),
            ("debug {} issues", IntentType::Debug { component: "component".to_string() }),
        ];
        
        let resource_types = vec!["GPU", "CPU", "memory", "storage", "network"];
        let services = vec!["web", "api", "database", "cache", "queue"];
        let metrics = vec!["CPU", "memory", "disk", "network"];
        
        for _ in 0..500 {
            let mut rng = thread_rng();
            let (template, intent_type) = intent_templates.choose(&mut rng).unwrap();
            
            let text = match template {
                t if t.contains("cluster") => {
                    let resource = resource_types.choose(&mut rng).unwrap();
                    t.replace("{}", resource)
                },
                t if t.contains("service") => {
                    let service = services.choose(&mut rng).unwrap();
                    t.replace("{}", service)
                },
                t if t.contains("usage") => {
                    let metric = metrics.choose(&mut rng).unwrap();
                    t.replace("{}", metric)
                },
                t => t.replace("{}", "system"),
            };
            
            let entities = extract_entities_from_template(&text, intent_type);
            training_data.push(TrainingExample {
                text, intent_type: intent_type.clone(), entities, context: create_default_context(),
                expected_action_plan: create_sample_action_plan(&intent_type), execution_outcome: None,
            });
        }
        
        let performance = orchestrator.train_models(training_data).await.expect("Should train neural models");
        
        assert!(performance.intent_accuracy > 0.7);
        assert!(performance.entity_extraction_accuracy > 0.6);
        assert!(performance.total_intents_processed > 400);
        assert!(performance.average_processing_time < Duration::from_millis(200));
    }

    #[tokio::test]
    async fn test_knowledge_graph_integration() {
        let device = Device::Cpu;
        let config = OrchestratorConfig::default();
        let orchestrator = IntentOrchestrator::new(device, config).await.expect("Should initialize");
        
        let entities = vec![
            Entity {
                text: "gpu-cluster-1".to_string(), entity_type: EntityType::Resource, start: 0, end: 13,
                confidence: 0.95, value: EntityValue::String("gpu-cluster-1".to_string()),
                relations: vec![
                    EntityRelation { relation_type: RelationType::LocatedIn,
                        target_entity: "us-west-1".to_string(), confidence: 0.9 },
                    EntityRelation { relation_type: RelationType::Contains,
                        target_entity: "gpu-node".to_string(), confidence: 0.85 },
                ],
            },
            Entity {
                text: "ml-training-service".to_string(), entity_type: EntityType::Resource,
                start: 0, end: 19, confidence: 0.9, value: EntityValue::String("ml-training-service".to_string()),
                relations: vec![
                    EntityRelation { relation_type: RelationType::DependsOn,
                        target_entity: "gpu-cluster-1".to_string(), confidence: 0.95 },
                ],
            },
        ];
        
        let additional_relations = vec![
            EntityRelation { relation_type: RelationType::ConnectedTo,
                target_entity: "load-balancer".to_string(), confidence: 0.8 },
        ];
        
        orchestrator.update_knowledge_graph(entities.clone(), additional_relations).await.expect("Should update knowledge graph");
        
        let test_intent = "Scale up the ML training service that depends on gpu-cluster-1";
        let intent = orchestrator.parse_intent(test_intent, None).await.expect("Should parse intent with graph context");
        
        let service_entities: Vec<_> = intent.entities.iter().filter(|e| e.text.contains("training")).collect();
        assert!(!service_entities.is_empty());
        
        let cluster_entities: Vec<_> = intent.entities.iter().filter(|e| e.text.contains("cluster")).collect();
        assert!(!cluster_entities.is_empty());
        
        let action_plan = orchestrator.orchestrate_intent(&intent).await.expect("Should orchestrate with graph context");
        
        let scaling_steps: Vec<_> = action_plan.steps.iter().filter(|step| matches!(step.action_type, ActionType::ScaleCluster)).collect();
        assert!(!scaling_steps.is_empty());
        
        assert!(action_plan.required_resources.gpu_count > 0);
    }

    #[tokio::test]
    async fn test_realtime_orchestration_performance() {
        let device = Device::Cpu;
        let config = OrchestratorConfig {
            transformer_layers: 2, transformer_heads: 4, hidden_size: 256, vocab_size: 5000,
            max_sequence_length: 256, dropout_rate: 0.0, learning_rate: 0.001,
        };
        let orchestrator = IntentOrchestrator::new(device, config).await.expect("Should initialize");
        
        let test_intents = vec![
            "Create a GPU cluster",
            "Scale up production",
            "Deploy web service",
            "Monitor CPU usage",
            "Backup database",
            "Debug network issues",
            "Configure auto-scaling",
            "Optimize performance",
        ];
        
        let start_time = Instant::now();
        let mut tasks = Vec::new();
        
        for (i, intent_text) in test_intents.iter().enumerate() {
            let orchestrator_clone = orchestrator.clone();
            let text = intent_text.to_string();
            
            let task = tokio::spawn(async move {
                let start = Instant::now();
                let intent = orchestrator_clone.parse_intent(&text, None).await?;
                let parse_time = start.elapsed();
                let action_plan = orchestrator_clone.orchestrate_intent(&intent).await?;
                let orchestrate_time = start.elapsed() - parse_time;
                let execution_record = orchestrator_clone.execute_action_plan(&action_plan).await?;
                let execute_time = start.elapsed() - parse_time - orchestrate_time;
                Ok::<_, Box<dyn std::error::Error + Send + Sync>>((intent, action_plan, execution_record, parse_time, orchestrate_time, execute_time))
            });
            tasks.push(task);
        }
        
        let results: Vec<_> = futures::future::join_all(tasks).await;
        let total_duration = start_time.elapsed();
        
        assert!(total_duration < Duration::from_secs(30));
        
        let successful_results: Vec<_> = results.iter().filter(|r| r.is_ok() && r.as_ref().unwrap().is_ok()).collect();
        assert!(successful_results.len() >= 7);
        
        for result in successful_results {
            if let Ok(Ok((intent, plan, record, parse_time, orchestrate_time, execute_time))) = result {
                assert!(parse_time < Duration::from_millis(500));
                assert!(orchestrate_time < Duration::from_secs(2));
                assert!(execute_time < Duration::from_secs(5));
                
                assert!(intent.confidence > 0.5);
                assert!(!plan.steps.is_empty());
                assert!(matches!(record.execution_status, ExecutionStatus::Completed | ExecutionStatus::Running));
            }
        }
    }

    #[tokio::test]
    async fn test_orchestration_fault_tolerance() {
        let device = Device::Cpu;
        let config = OrchestratorConfig::default();
        let orchestrator = IntentOrchestrator::new(device, config).await.expect("Should initialize");
        
        let malformed_inputs = vec![
            "", "fjdsklfjdsklfjklds fjdskl fjdskl", "CREATE CREATE CREATE CLUSTER CLUSTER",
            "Scale up down left right diagonal", "DELETE EVERYTHING PERMANENTLY",
        ];
        
        for input in malformed_inputs {
            let result = orchestrator.parse_intent(input, None).await;
            match result {
                Ok(intent) => {
                    if !input.is_empty() {
                        assert!(intent.confidence < 0.7 || intent.intent_type == IntentType::Unknown);
                    }
                },
                Err(_) => {}
            }
        }
        
        let resource_intensive_intent = Intent {
            id: Uuid::new_v4(),
            text: "Create 1000 GPU clusters with unlimited resources".to_string(),
            intent_type: IntentType::CreateResource { resource_type: "cluster".to_string() },
            entities: vec![
                Entity { text: "1000".to_string(), entity_type: EntityType::Value,
                    start: 7, end: 11, confidence: 0.99, value: EntityValue::Integer(1000), relations: vec![] },
            ],
            confidence: 0.95,
            context: create_default_context(),
            timestamp: SystemTime::now(),
            user_id: Some("test_user".to_string()),
            session_id: Some("test_session".to_string()),
        };
        
        let result = orchestrator.orchestrate_intent(&resource_intensive_intent).await;
        match result {
            Ok(plan) => {
                assert!(plan.required_resources.gpu_count < 1000);
                let constraint_steps: Vec<_> = plan.steps.iter()
                    .filter(|step| step.parameters.contains_key("resource_limit") || step.parameters.contains_key("constraint"))
                    .collect();
            },
            Err(_) => {}
        }
        
        let failing_plan = ActionPlan {
            id: Uuid::new_v4(),
            steps: vec![
                ActionStep {
                    id: Uuid::new_v4(), action_type: ActionType::CreateAgent,
                    parameters: { let mut params = HashMap::new();
                        params.insert("invalid_param".to_string(), EntityValue::String("invalid".to_string())); params },
                    dependencies: vec![], timeout: Duration::from_millis(100),
                    retry_policy: RetryPolicy { max_attempts: 2,
                        backoff_strategy: BackoffStrategy::Fixed { delay: Duration::from_millis(10) },
                        retry_conditions: vec![] }, parallel_group: None,
                },
            ],
            estimated_duration: Duration::from_secs(60),
            required_resources: ResourceRequirements { cpu_cores: 1, memory_gb: 1, gpu_count: 0,
                storage_gb: 10, network_gbps: 0.1 },
            success_criteria: vec![],
            rollback_plan: Some(RollbackPlan {
                steps: vec![RollbackStep { id: Uuid::new_v4(), action_type: ActionType::TriggerAlert,
                    description: "Cleanup failed resources".to_string(), timeout: Duration::from_secs(300) }],
                trigger_conditions: vec!["execution_failed".to_string()],
            }),
        };
        
        let execution_result = orchestrator.execute_action_plan(&failing_plan).await;
        match execution_result {
            Ok(record) => {
                if record.execution_status == ExecutionStatus::Failed {
                    assert!(!record.outcomes.is_empty());
                    let retry_attempts: u32 = record.outcomes.iter().map(|outcome| outcome.retry_count).sum();
                    assert!(retry_attempts > 0);
                }
            },
            Err(_) => {}
        }
    }

    #[tokio::test]
    async fn test_advanced_orchestration_patterns() {
        let device = Device::Cpu;
        let config = OrchestratorConfig::default();
        let orchestrator = IntentOrchestrator::new(device, config).await.expect("Should initialize");

        let conditional_intent = Intent {
            id: Uuid::new_v4(),
            text: "If cluster CPU > 80%, then scale up by 2 nodes, otherwise optimize existing resources".to_string(),
            intent_type: IntentType::Scale { direction: ScaleDirection::Auto },
            entities: vec![
                Entity { text: "80%".to_string(), entity_type: EntityType::Value,
                    start: 18, end: 21, confidence: 0.99, value: EntityValue::Float(0.8), relations: vec![] },
                Entity { text: "2 nodes".to_string(), entity_type: EntityType::Value,
                    start: 39, end: 46, confidence: 0.95, value: EntityValue::Integer(2), relations: vec![] },
            ],
            confidence: 0.88,
            context: create_default_context(),
            timestamp: SystemTime::now(),
            user_id: Some("user123".to_string()),
            session_id: Some("session456".to_string()),
        };

        let action_plan = orchestrator.orchestrate_intent(&conditional_intent).await.expect("Should orchestrate conditional intent");

        assert!(action_plan.steps.len() >= 2);
        let evaluation_steps: Vec<_> = action_plan.steps.iter()
            .filter(|step| step.parameters.contains_key("condition") || step.parameters.contains_key("evaluation"))
            .collect();

        let cascading_intent = Intent {
            id: Uuid::new_v4(),
            text: "Deploy microservices A, B, C where B depends on A, and C depends on both A and B".to_string(),
            intent_type: IntentType::Deploy { deployment_type: "microservices".to_string() },
            entities: vec![
                Entity { text: "microservices A".to_string(), entity_type: EntityType::Resource,
                    start: 7, end: 22, confidence: 0.9, value: EntityValue::String("service-a".to_string()), relations: vec![] },
                Entity { text: "B".to_string(), entity_type: EntityType::Resource,
                    start: 24, end: 25, confidence: 0.9, value: EntityValue::String("service-b".to_string()),
                    relations: vec![EntityRelation { relation_type: RelationType::DependsOn,
                        target_entity: "service-a".to_string(), confidence: 0.95 }] },
                Entity { text: "C".to_string(), entity_type: EntityType::Resource,
                    start: 27, end: 28, confidence: 0.9, value: EntityValue::String("service-c".to_string()),
                    relations: vec![
                        EntityRelation { relation_type: RelationType::DependsOn, target_entity: "service-a".to_string(), confidence: 0.9 },
                        EntityRelation { relation_type: RelationType::DependsOn, target_entity: "service-b".to_string(), confidence: 0.9 },
                    ] },
            ],
            confidence: 0.92,
            context: create_default_context(),
            timestamp: SystemTime::now(),
            user_id: Some("user123".to_string()),
            session_id: Some("session456".to_string()),
        };

        let cascading_plan = orchestrator.orchestrate_intent(&cascading_intent).await.expect("Should orchestrate cascading deployment");

        assert!(cascading_plan.steps.len() >= 3);
        let service_steps: Vec<_> = cascading_plan.steps.iter()
            .filter(|step| matches!(step.action_type, ActionType::DeployContainer)).collect();
        assert!(service_steps.len() >= 3);

        let steps_with_deps: Vec<_> = cascading_plan.steps.iter().filter(|step| !step.dependencies.is_empty()).collect();
        assert!(!steps_with_deps.is_empty());
    }

    #[tokio::test]
    async fn test_resource_optimization_orchestration() {
        let device = Device::Cpu;
        let config = OrchestratorConfig::default();
        let orchestrator = IntentOrchestrator::new(device, config).await.expect("Should initialize");

        let constrained_context = IntentContext {
            conversation_history: VecDeque::new(),
            active_session: None,
            user_preferences: { let mut prefs = HashMap::new();
                prefs.insert("cost_optimization".to_string(), "aggressive".to_string());
                prefs.insert("performance_priority".to_string(), "cost".to_string()); prefs },
            system_state: SystemState { active_agents: 50,
                resource_utilization: { let mut util = HashMap::new();
                    util.insert("cpu".to_string(), 0.95); util.insert("memory".to_string(), 0.85);
                    util.insert("gpu".to_string(), 0.6); util },
                cluster_health: { let mut health = HashMap::new();
                    health.insert("us-west-1".to_string(), 0.7); health.insert("us-east-1".to_string(), 0.95); health },
                pending_operations: 15, error_rate: 0.05 },
            ambient_context: AmbientContext { time_of_day: 0.9, system_load: 0.9, network_conditions: 0.6,
                maintenance_window: false, emergency_mode: false },
        };

        let optimization_intent = Intent {
            id: Uuid::new_v4(),
            text: "Create a high-performance compute cluster for batch processing".to_string(),
            intent_type: IntentType::CreateResource { resource_type: "compute-cluster".to_string() },
            entities: vec![
                Entity { text: "high-performance".to_string(), entity_type: EntityType::Parameter,
                    start: 9, end: 25, confidence: 0.9, value: EntityValue::String("high-performance".to_string()), relations: vec![] },
                Entity { text: "batch processing".to_string(), entity_type: EntityType::Action,
                    start: 50, end: 66, confidence: 0.95, value: EntityValue::String("batch-processing".to_string()), relations: vec![] },
            ],
            confidence: 0.9,
            context: constrained_context,
            timestamp: SystemTime::now(),
            user_id: Some("user123".to_string()),
            session_id: Some("session456".to_string()),
        };

        let optimized_plan = orchestrator.orchestrate_intent(&optimization_intent).await.expect("Should create optimized orchestration plan");

        assert!(!optimized_plan.steps.is_empty());
        
        let region_specific_steps: Vec<_> = optimized_plan.steps.iter()
            .filter(|step| step.parameters.get("region")
                .map(|v| match v {
                    EntityValue::String(s) => s.contains("us-east-1"),
                    _ => false,
                }).unwrap_or(false)).collect();

        let cost_optimization_steps: Vec<_> = optimized_plan.steps.iter()
            .filter(|step| step.parameters.contains_key("cost_optimization") || step.parameters.contains_key("spot_instances") || step.parameters.contains_key("resource_sharing"))
            .collect();

        assert!(optimized_plan.required_resources.cpu_cores > 0);
        
        let total_resource_score = (optimized_plan.required_resources.cpu_cores as f32) * 0.3 +
                                 (optimized_plan.required_resources.memory_gb as f32) * 0.2 +
                                 (optimized_plan.required_resources.gpu_count as f32) * 0.5;
        assert!(total_resource_score < 1000.0);
    }

    fn extract_entities_from_template(text: &str, intent_type: &IntentType) -> Vec<Entity> {
        let mut entities = Vec::new();
        
        if text.contains("GPU") {
            entities.push(Entity {
                text: "GPU".to_string(), entity_type: EntityType::Resource,
                start: text.find("GPU")?, end: text.find("GPU")? + 3,
                confidence: 0.9, value: EntityValue::String("GPU".to_string()), relations: vec![],
            });
        }
        
        if text.contains("cluster") {
            entities.push(Entity {
                text: "cluster".to_string(), entity_type: EntityType::Resource,
                start: text.find("cluster").unwrap(), end: text.find("cluster").unwrap() + 7,
                confidence: 0.95, value: EntityValue::String("cluster".to_string()), relations: vec![],
            });
        }
        
        use regex::Regex;
        let num_regex = Regex::new(r"\d+").unwrap();
        for mat in num_regex.find_iter(text) {
            entities.push(Entity {
                text: mat.as_str().to_string(), entity_type: EntityType::Value,
                start: mat.start(), end: mat.end(), confidence: 0.95,
                value: EntityValue::Integer(mat.as_str().parse().unwrap()), relations: vec![],
            });
        }
        entities
    }
}