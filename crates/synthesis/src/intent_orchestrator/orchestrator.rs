//! Main intent orchestrator implementation

use candle_core::{Device, Tensor};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use super::config::OrchestratorConfig;
use super::context::{IntentContext, SessionContext};
use super::engine::{ExecutionPlanner, OrchestrationEngine};
use super::entities::Entity;
use super::execution::{ActionPlan, ExecutionRecord, ExecutionResult};
use super::intents::{ClassificationResult, Intent, IntentType};
use super::metrics::OrchestrationMetrics;
use super::models::{BertModel, TransformerModel};
use super::types::{AgentId, KnowledgeGraph};

/// Main intent orchestrator
pub struct IntentOrchestrator {
    /// Computation device
    pub device: Device,
    /// Configuration
    pub config: OrchestratorConfig,
    /// Intent classifier
    pub intent_classifier: Arc<RwLock<IntentClassifier>>,
    /// Entity extractor
    pub entity_extractor: Arc<RwLock<EntityExtractor>>,
    /// Orchestration engine
    pub orchestration_engine: OrchestrationEngine,
    /// Execution planner
    pub execution_planner: ExecutionPlanner,
    /// Knowledge graph
    pub knowledge_graph: Arc<RwLock<KnowledgeGraph>>,
    /// Metrics collector
    pub metrics: Arc<RwLock<OrchestrationMetrics>>,
    /// Active sessions
    pub active_sessions: Arc<RwLock<HashMap<String, SessionContext>>>,
}

/// Intent classifier
pub struct IntentClassifier {
    /// Device
    pub device: Device,
    /// Transformer model
    pub transformer: TransformerModel,
    /// Classification threshold
    pub threshold: f32,
}

/// Entity extractor
pub struct EntityExtractor {
    /// Device
    pub device: Device,
    /// BERT model
    pub bert_model: BertModel,
    /// Entity vocabulary
    pub entity_vocab: HashMap<String, usize>,
}

impl IntentOrchestrator {
    /// Create new orchestrator
    pub async fn new(
        device: Device,
        config: OrchestratorConfig,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // Validate config
        config.validate()?;
        
        // Create components
        let intent_classifier = Arc::new(RwLock::new(
            IntentClassifier::new(device.clone(), &config).await?
        ));
        
        let entity_extractor = Arc::new(RwLock::new(
            EntityExtractor::new(device.clone(), &config).await?
        ));
        
        let orchestration_engine = OrchestrationEngine::new(device.clone()).await?;
        let execution_planner = ExecutionPlanner::new(device.clone()).await?;
        
        Ok(Self {
            device,
            config,
            intent_classifier,
            entity_extractor,
            orchestration_engine,
            execution_planner,
            knowledge_graph: Arc::new(RwLock::new(KnowledgeGraph::default())),
            metrics: Arc::new(RwLock::new(OrchestrationMetrics::default())),
            active_sessions: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Parse and process intent
    pub async fn parse_intent(
        &self,
        text: &str,
        context: Option<IntentContext>,
    ) -> Result<Intent, Box<dyn std::error::Error>> {
        let start_time = Utc::now();
        
        // Tokenize text
        let tokens = self.tokenize_text(text).await?;
        
        // Classify intent
        let classifier = self.intent_classifier.read().await;
        let classification = classifier.classify(text, &tokens).await?;
        
        // Extract entities
        let extractor = self.entity_extractor.read().await;
        let entities = extractor.extract(text, &tokens).await?;
        
        // Create intent
        let mut metadata = HashMap::new();
        metadata.insert("id".to_string(), uuid::Uuid::new_v4().to_string());
        if let Some(ctx) = context {
            metadata.insert("context".to_string(), serde_json::to_string(&ctx).unwrap_or_default());
        }

        let intent = Intent {
            intent_type: classification.intent_type.clone(),
            confidence: classification.get_confidence(),
            entities,
            raw_text: text.to_string(),
            tokens: tokens.clone(),
            metadata,
            timestamp: start_time,
        };
        
        // Update metrics
        let duration_ms = (Utc::now() - start_time).num_milliseconds() as f64;
        let mut metrics = self.metrics.write().await;
        metrics.record_classification(&intent.intent_type, true, duration_ms);
        
        Ok(intent)
    }

    /// Execute intent
    pub async fn execute_intent(
        &self,
        intent: &Intent,
    ) -> Result<ExecutionRecord, Box<dyn std::error::Error>> {
        let start_time = Utc::now();
        
        // Create action plan
        let action_plan = self.create_action_plan(intent).await?;
        
        // Create execution record
        let intent_id = intent.metadata.get("id").cloned().unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        let mut record = ExecutionRecord::new(intent_id, action_plan.clone());
        record.start();
        
        // Execute plan
        let result = self.orchestration_engine
            .execute_plan(action_plan.steps, &mut record)
            .await?;
        
        // Complete execution
        record.complete(result);
        
        // Update metrics
        let duration_ms = (Utc::now() - start_time).num_milliseconds() as f64;
        let mut metrics = self.metrics.write().await;
        metrics.record_execution(record.status, duration_ms);
        
        Ok(record)
    }

    /// Create action plan from intent
    async fn create_action_plan(
        &self,
        intent: &Intent,
    ) -> Result<ActionPlan, Box<dyn std::error::Error>> {
        // Create action steps
        let steps = self.orchestration_engine.action_planner
            .create_action_steps(
                &format!("{:?}", intent.intent_type),
                HashMap::new(),
            )
            .await?;
        
        // Estimate resources
        let resources = self.orchestration_engine.resource_allocator
            .estimate_requirements(&steps)
            .await?;
        
        Ok(ActionPlan {
            id: uuid::Uuid::new_v4().to_string(),
            steps,
            resources,
            retry_policy: Default::default(),
            success_criteria: vec![],
            metadata: HashMap::new(),
        })
    }

    /// Tokenize text
    async fn tokenize_text(&self, text: &str) -> Result<Vec<String>, Box<dyn std::error::Error>> {
        // Simple tokenization
        let tokens: Vec<String> = text
            .to_lowercase()
            .split_whitespace()
            .map(|s| s.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
            .filter(|s| !s.is_empty())
            .collect();
        Ok(tokens)
    }

    /// Update knowledge graph
    pub async fn update_knowledge_graph(
        &self,
        entities: Vec<Entity>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut kg = self.knowledge_graph.write().await;
        
        for entity in entities {
            kg.entities.insert(entity.text.clone(), entity);
        }
        
        Ok(())
    }
}

impl IntentClassifier {
    /// Create new classifier
    async fn new(
        device: Device,
        config: &OrchestratorConfig,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let transformer = TransformerModel::new(config.model_config.clone().into());
        
        Ok(Self {
            device,
            transformer,
            threshold: config.classification_threshold,
        })
    }

    /// Classify intent
    async fn classify(
        &self,
        text: &str,
        tokens: &[String],
    ) -> Result<ClassificationResult, Box<dyn std::error::Error>> {
        // Simplified classification
        let confidence = 0.85;
        
        let intent_type = if text.contains("create") || text.contains("new") {
            IntentType::Deploy
        } else if text.contains("scale") {
            if text.contains("up") {
                IntentType::Scale(super::types::ScaleDirection::Up)
            } else if text.contains("down") {
                IntentType::Scale(super::types::ScaleDirection::Down)
            } else {
                IntentType::Scale(super::types::ScaleDirection::Out)
            }
        } else if text.contains("monitor") {
            IntentType::Monitor
        } else {
            IntentType::Query
        };
        
        let mut scores = HashMap::new();
        scores.insert(format!("{:?}", intent_type).to_lowercase(), confidence);

        Ok(ClassificationResult {
            intent_type: intent_type.clone(),
            scores,
            top_k: vec![(intent_type, confidence)],
            metadata: HashMap::new(),
        })
    }
}

impl EntityExtractor {
    /// Create new extractor
    async fn new(
        device: Device,
        config: &OrchestratorConfig,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let bert_model = BertModel::new(config.model_config.clone());
        
        Ok(Self {
            device,
            bert_model,
            entity_vocab: HashMap::new(),
        })
    }

    /// Extract entities
    async fn extract(
        &self,
        text: &str,
        tokens: &[String],
    ) -> Result<Vec<Entity>, Box<dyn std::error::Error>> {
        let mut entities = Vec::new();
        
        // Simple pattern-based extraction
        for (i, token) in tokens.iter().enumerate() {
            if token.parse::<i32>().is_ok() {
                entities.push(Entity {
                    text: token.clone(),
                    entity_type: super::entities::EntityType::Number,
                    value: super::entities::EntityValue::String(token.clone()),
                    start: i,
                    end: i + 1,
                    confidence: 0.95,
                    metadata: HashMap::new(),
                });
            }
        }
        
        Ok(entities)
    }
}
