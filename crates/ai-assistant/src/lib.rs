//! AI Assistant for StratoSwarm Developer Experience
//!
//! This module provides natural language interfaces for interacting with StratoSwarm,
//! enabling developers to use conversational commands to manage infrastructure.

use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

pub mod command_generator;
pub mod embeddings;
pub mod error;
pub mod learning;
pub mod parser;
pub mod query_engine;
pub mod templates;

pub use command_generator::{CommandGenerator, GeneratedCommand};
pub use error::{AssistantError, AssistantResult};
pub use learning::{LearningSystem, Pattern};
pub use parser::{Intent, NaturalLanguageParser, ParsedQuery};
pub use query_engine::{QueryEngine, QueryResult};

/// AI Assistant for natural language operations
#[derive(Clone)]
pub struct AiAssistant {
    parser: Arc<NaturalLanguageParser>,
    command_generator: Arc<CommandGenerator>,
    query_engine: Arc<QueryEngine>,
    learning_system: Arc<RwLock<LearningSystem>>,
    config: AssistantConfig,
}

/// Configuration for the AI Assistant
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssistantConfig {
    /// Maximum tokens for responses
    pub max_response_tokens: usize,
    /// Temperature for generation (0.0 - 1.0)
    pub temperature: f32,
    /// Whether to use local or cloud model
    pub use_local_model: bool,
    /// Path to local model if using local
    pub local_model_path: Option<String>,
    /// API key for cloud model
    pub cloud_api_key: Option<String>,
    /// Enable learning from interactions
    pub enable_learning: bool,
    /// Confidence threshold for command execution
    pub confidence_threshold: f32,
}

impl Default for AssistantConfig {
    fn default() -> Self {
        Self {
            max_response_tokens: 2048,
            temperature: 0.7,
            use_local_model: true,
            local_model_path: None,
            cloud_api_key: None,
            enable_learning: true,
            confidence_threshold: 0.8,
        }
    }
}

/// Result of processing a natural language input
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssistantResponse {
    /// The interpreted intent
    pub intent: Intent,
    /// Generated command if applicable
    pub command: Option<GeneratedCommand>,
    /// Query results if applicable
    pub query_results: Option<Vec<QueryResult>>,
    /// Natural language response
    pub response: String,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f32,
    /// Suggested follow-up actions
    pub suggestions: Vec<String>,
}

/// Trait for components that can handle natural language
#[async_trait]
pub trait NaturalLanguageHandler: Send + Sync {
    /// Process a natural language input
    async fn process(&self, input: &str) -> Result<AssistantResponse>;

    /// Get suggestions for the current context
    async fn get_suggestions(&self, context: &str) -> Result<Vec<String>>;

    /// Learn from user feedback
    async fn learn_from_feedback(
        &self,
        input: &str,
        response: &AssistantResponse,
        was_helpful: bool,
    ) -> Result<()>;
}

impl AiAssistant {
    /// Create a new AI Assistant
    pub async fn new(config: AssistantConfig) -> Result<Self> {
        let parser = Arc::new(NaturalLanguageParser::new()?);
        let command_generator = Arc::new(CommandGenerator::new()?);
        let query_engine = Arc::new(QueryEngine::new().await?);
        let learning_system = Arc::new(RwLock::new(LearningSystem::new(config.enable_learning)?));

        Ok(Self {
            parser,
            command_generator,
            query_engine,
            learning_system,
            config,
        })
    }

    /// Process a natural language input
    pub async fn process_input(&self, input: &str) -> Result<AssistantResponse> {
        // Parse the natural language input
        let parsed = self.parser.parse(input).await?;

        // Generate command if needed
        let command = if parsed.requires_command() {
            Some(self.command_generator.generate(&parsed).await?)
        } else {
            None
        };

        // Execute queries if needed
        let query_results = if parsed.requires_query() {
            Some(self.query_engine.execute(&parsed).await?)
        } else {
            None
        };

        // Generate natural language response
        let response = self
            .generate_response(&parsed, &command, &query_results)
            .await?;

        // Learn from this interaction if enabled
        if self.config.enable_learning {
            let mut learning = self.learning_system.write().await;
            learning
                .record_interaction(input, &parsed, &response.text)
                .await?;
        }

        // Get suggestions for follow-up
        let suggestions = self.get_contextual_suggestions(&parsed).await?;

        Ok(AssistantResponse {
            intent: parsed.intent,
            command,
            query_results,
            response: response.text,
            confidence: response.confidence,
            suggestions,
        })
    }

    /// Generate a natural language response
    async fn generate_response(
        &self,
        parsed: &ParsedQuery,
        command: &Option<GeneratedCommand>,
        query_results: &Option<Vec<QueryResult>>,
    ) -> Result<GeneratedResponse> {
        // This would integrate with a language model
        // For now, template-based responses
        let template = templates::get_template_for_intent(&parsed.intent)?;
        let response = templates::render_template(template, parsed, command, query_results)?;

        Ok(GeneratedResponse {
            text: response,
            confidence: parsed.confidence,
        })
    }

    /// Get contextual suggestions
    async fn get_contextual_suggestions(&self, parsed: &ParsedQuery) -> Result<Vec<String>> {
        let learning = self.learning_system.read().await;
        learning
            .get_suggestions_for_context(&parsed.intent)
            .await
            .map_err(|e| anyhow::anyhow!("Learning system error: {}", e))
    }

    /// Handle user feedback
    pub async fn handle_feedback(
        &self,
        input: &str,
        response: &AssistantResponse,
        was_helpful: bool,
    ) -> Result<()> {
        let mut learning = self.learning_system.write().await;
        learning
            .update_from_feedback(input, response, was_helpful)
            .await
            .map_err(|e| anyhow::anyhow!("Learning system error: {}", e))
    }
}

#[async_trait]
impl NaturalLanguageHandler for AiAssistant {
    async fn process(&self, input: &str) -> Result<AssistantResponse> {
        self.process_input(input).await
    }

    async fn get_suggestions(&self, context: &str) -> Result<Vec<String>> {
        let parsed = self.parser.parse(context).await?;
        self.get_contextual_suggestions(&parsed).await
    }

    async fn learn_from_feedback(
        &self,
        input: &str,
        response: &AssistantResponse,
        was_helpful: bool,
    ) -> Result<()> {
        self.handle_feedback(input, response, was_helpful).await
    }
}

/// Generated response with metadata
struct GeneratedResponse {
    text: String,
    confidence: f32,
}

/// Example intents that the assistant can handle
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CommonIntent {
    /// Deploy an application
    Deploy { app_name: String, source: String },
    /// Scale resources
    Scale {
        target: String,
        replicas: Option<u32>,
        resources: Option<String>,
    },
    /// Query infrastructure status
    QueryStatus {
        resource_type: Option<String>,
        filters: HashMap<String, String>,
    },
    /// Debug issues
    Debug {
        resource: String,
        issue_description: String,
    },
    /// Optimize performance
    Optimize { target: String, metric: String },
    /// General help
    Help { topic: Option<String> },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_assistant_creation() {
        let config = AssistantConfig::default();
        let assistant = AiAssistant::new(config).await.unwrap();
        assert!(assistant.config.enable_learning);
    }

    #[tokio::test]
    async fn test_process_deploy_command() {
        let config = AssistantConfig::default();
        let assistant = AiAssistant::new(config).await.unwrap();

        let response = assistant
            .process_input("deploy my app from github.com/user/repo")
            .await
            .unwrap();
        assert!(matches!(response.intent, Intent::Deploy { .. }));
        assert!(response.command.is_some());
        assert!(response.confidence > 0.8);
    }

    #[tokio::test]
    async fn test_process_query_command() {
        let config = AssistantConfig::default();
        let assistant = AiAssistant::new(config).await.unwrap();

        let response = assistant
            .process_input("show me all running agents")
            .await
            .unwrap();
        assert!(matches!(response.intent, Intent::Query { .. }));
        assert!(response.query_results.is_some());
    }

    #[tokio::test]
    async fn test_learning_from_feedback() {
        let config = AssistantConfig::default();
        let assistant = AiAssistant::new(config).await.unwrap();

        let response = assistant.process_input("deploy my app").await.unwrap();
        assistant
            .handle_feedback("deploy my app", &response, true)
            .await
            .unwrap();

        // Should improve confidence on similar queries
        let response2 = assistant.process_input("deploy another app").await.unwrap();
        assert!(response2.confidence >= response.confidence);
    }
}
