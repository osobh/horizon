//! Ollama LLM client for local model inference
//! Replaces OpenAI with local Ollama models for unlimited usage

use crate::error::{BusinessError, BusinessResult};
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tracing::{debug, info};

/// Ollama model selection based on task type
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum TaskType {
    /// Natural language goal parsing
    GoalParsing,
    /// Safety and risk validation
    SafetyValidation,
    /// Code generation for agents
    CodeGeneration,
    /// Complex reasoning and decision making
    Reasoning,
    /// User-friendly explanations
    Explanation,
}

/// Ollama model configuration
#[derive(Debug, Clone)]
pub struct OllamaConfig {
    /// Base URL for Ollama API
    pub base_url: String,
    /// Request timeout
    pub timeout: Duration,
    /// Default temperature for generation
    pub temperature: f32,
    /// Maximum tokens to generate
    pub max_tokens: u32,
}

impl Default for OllamaConfig {
    fn default() -> Self {
        Self {
            base_url: "http://localhost:11434".to_string(),
            timeout: Duration::from_secs(120),
            temperature: 0.7,
            max_tokens: 2048,
        }
    }
}

/// Ollama API request
#[derive(Debug, Serialize)]
struct OllamaRequest {
    model: String,
    prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    stream: bool,
}

/// Ollama API response
#[derive(Debug, Deserialize)]
struct OllamaResponse {
    model: String,
    created_at: String,
    response: String,
    done: bool,
    #[serde(default)]
    context: Vec<u32>,
    #[serde(default)]
    total_duration: u64,
    #[serde(default)]
    eval_count: u32,
}

/// Available models info
#[derive(Debug, Clone, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    size: u64,
    digest: String,
    #[serde(default)]
    details: ModelDetails,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct ModelDetails {
    #[serde(default)]
    format: String,
    #[serde(default)]
    family: String,
    #[serde(default)]
    parameter_size: String,
}

/// Ollama client for LLM operations
pub struct OllamaClient {
    client: Client,
    config: OllamaConfig,
}

impl OllamaClient {
    /// Create new Ollama client
    pub fn new(config: OllamaConfig) -> Self {
        let client = Client::builder()
            .timeout(config.timeout)
            .build()
            .expect("Failed to create HTTP client");

        Self { client, config }
    }

    /// Select best model for task type
    pub async fn select_model(&self, task: TaskType) -> BusinessResult<String> {
        // Get available models
        let models = self.list_models().await?;

        // Model selection logic based on task and available models
        let selected = match task {
            TaskType::GoalParsing => {
                // Prefer instruction-following models
                self.find_model(&models, &["wizardlm2", "mixtral", "llama3.1", "dolphin3"])
            }
            TaskType::SafetyValidation => {
                // Prefer reasoning and safety-focused models
                self.find_model(&models, &["deepseek-r1", "llama3.1:70b", "gemma3"])
            }
            TaskType::CodeGeneration => {
                // Prefer code-focused models
                self.find_model(&models, &["devstral", "deepseek-r1", "qwen3"])
            }
            TaskType::Reasoning => {
                // Prefer large reasoning models
                self.find_model(&models, &["deepseek-r1", "llama3.1:70b", "mixtral"])
            }
            TaskType::Explanation => {
                // Prefer general-purpose models
                self.find_model(&models, &["solar-pro", "dolphin3", "llama3.2-vision:11b"])
            }
        };

        selected.ok_or_else(|| {
            BusinessError::ConfigurationError(format!(
                "No suitable model found for task: {:?}",
                task
            ))
        })
    }

    /// Find first available model from preferences
    fn find_model(&self, models: &[ModelInfo], preferences: &[&str]) -> Option<String> {
        for pref in preferences {
            if let Some(model) = models.iter().find(|m| m.name.contains(pref)) {
                return Some(model.name.clone());
            }
        }
        // Fallback to first available model
        models.first().map(|m| m.name.clone())
    }

    /// List available models
    pub async fn list_models(&self) -> BusinessResult<Vec<ModelInfo>> {
        let url = format!("{}/api/tags", self.config.base_url);

        let response =
            self.client
                .get(&url)
                .send()
                .await
                .map_err(|e| BusinessError::LlmIntegrationError {
                    service: "Ollama".to_string(),
                    error: e.to_string(),
                })?;

        #[derive(Deserialize)]
        struct TagsResponse {
            models: Vec<ModelInfo>,
        }

        let tags: TagsResponse =
            response
                .json()
                .await
                .map_err(|e| BusinessError::LlmIntegrationError {
                    service: "Ollama".to_string(),
                    error: e.to_string(),
                })?;

        Ok(tags.models)
    }

    /// Generate completion for prompt
    pub async fn generate(
        &self,
        model: &str,
        prompt: &str,
        system: Option<&str>,
    ) -> BusinessResult<String> {
        let url = format!("{}/api/generate", self.config.base_url);

        let request = OllamaRequest {
            model: model.to_string(),
            prompt: prompt.to_string(),
            system: system.map(|s| s.to_string()),
            temperature: Some(self.config.temperature),
            max_tokens: Some(self.config.max_tokens),
            stream: false,
        };

        debug!(
            "Sending request to Ollama: model={}, prompt_len={}",
            model,
            prompt.len()
        );

        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(|e| BusinessError::LlmIntegrationError {
                service: "Ollama".to_string(),
                error: e.to_string(),
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(BusinessError::LlmIntegrationError {
                service: "Ollama".to_string(),
                error: format!("HTTP {}: {error_text}", status),
            });
        }

        let ollama_response: OllamaResponse =
            response
                .json()
                .await
                .map_err(|e| BusinessError::LlmIntegrationError {
                    service: "Ollama".to_string(),
                    error: e.to_string(),
                })?;

        info!(
            "Ollama response: model={}, eval_count={}, duration={}ms",
            ollama_response.model,
            ollama_response.eval_count,
            ollama_response.total_duration / 1_000_000
        );

        Ok(ollama_response.response)
    }

    /// Parse natural language goal
    pub async fn parse_goal(&self, goal_text: &str) -> BusinessResult<serde_json::Value> {
        let model = self.select_model(TaskType::GoalParsing).await?;

        let system_prompt = r#"You are an AI assistant that parses natural language goals into structured JSON.
Extract the following information:
- objective: The main goal to achieve
- constraints: Any limitations or requirements
- success_criteria: How to measure success
- priority: high/medium/low
- estimated_resources: Expected resource needs

Respond with valid JSON only."#;

        let prompt = format!("Parse this goal into structured JSON:\n\n{}", goal_text);

        let response = self.generate(&model, &prompt, Some(system_prompt)).await?;

        // Parse JSON response
        serde_json::from_str(&response).map_err(|e| BusinessError::GoalParsingFailed {
            message: format!("Invalid JSON from LLM: {}", e),
        })
    }

    /// Validate goal for safety
    pub async fn validate_safety(&self, goal: &serde_json::Value) -> BusinessResult<bool> {
        let model = self.select_model(TaskType::SafetyValidation).await?;

        let system_prompt = r#"You are a safety validation system. Analyze goals for potential risks.
Check for:
- Resource exhaustion risks
- Security vulnerabilities
- Ethical concerns
- System stability risks

Respond with JSON: {"safe": true/false, "concerns": ["list", "of", "concerns"]}"#;

        let prompt = format!(
            "Validate the safety of this goal:\n\n{}",
            serde_json::to_string_pretty(goal)?
        );

        let response = self.generate(&model, &prompt, Some(system_prompt)).await?;

        let result: serde_json::Value =
            serde_json::from_str(&response).map_err(|e| BusinessError::SafetyValidationFailed {
                reason: format!("Invalid JSON from LLM: {}", e),
            })?;

        Ok(result["safe"].as_bool().unwrap_or(false))
    }

    /// Generate code for agent
    pub async fn generate_code(
        &self,
        specification: &str,
        language: &str,
    ) -> BusinessResult<String> {
        let model = self.select_model(TaskType::CodeGeneration).await?;

        let system_prompt = format!(
            "You are an expert {} programmer. Generate clean, efficient, and well-documented code.",
            language
        );

        let prompt = format!(
            "Generate {} code for the following specification:\n\n{}",
            language, specification
        );

        self.generate(&model, &prompt, Some(&system_prompt)).await
    }

    /// Explain result to user
    pub async fn explain_result(
        &self,
        result: &serde_json::Value,
        context: &str,
    ) -> BusinessResult<String> {
        let model = self.select_model(TaskType::Explanation).await?;

        let system_prompt = "You are a helpful AI assistant. Explain technical results in clear, user-friendly language.";

        let prompt = format!(
            "Explain this result to a non-technical user:\n\nContext: {}\n\nResult: {}",
            context,
            serde_json::to_string_pretty(result)?
        );

        self.generate(&model, &prompt, Some(system_prompt)).await
    }
}

/// LLM provider trait for abstraction
#[async_trait]
pub trait LLMProvider: Send + Sync {
    /// Parse goal from natural language
    async fn parse_goal(&self, text: &str) -> BusinessResult<serde_json::Value>;

    /// Validate goal safety
    async fn validate_safety(&self, goal: &serde_json::Value) -> BusinessResult<bool>;

    /// Generate code
    async fn generate_code(&self, spec: &str, language: &str) -> BusinessResult<String>;

    /// Explain result
    async fn explain_result(
        &self,
        result: &serde_json::Value,
        context: &str,
    ) -> BusinessResult<String>;
}

#[async_trait]
impl LLMProvider for OllamaClient {
    async fn parse_goal(&self, text: &str) -> BusinessResult<serde_json::Value> {
        self.parse_goal(text).await
    }

    async fn validate_safety(&self, goal: &serde_json::Value) -> BusinessResult<bool> {
        self.validate_safety(goal).await
    }

    async fn generate_code(&self, spec: &str, language: &str) -> BusinessResult<String> {
        self.generate_code(spec, language).await
    }

    async fn explain_result(
        &self,
        result: &serde_json::Value,
        context: &str,
    ) -> BusinessResult<String> {
        self.explain_result(result, context).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_model_selection() {
        let client = OllamaClient::new(OllamaConfig::default());

        // Test model selection for different tasks
        let tasks = vec![
            TaskType::GoalParsing,
            TaskType::SafetyValidation,
            TaskType::CodeGeneration,
            TaskType::Reasoning,
            TaskType::Explanation,
        ];

        for task in tasks {
            match client.select_model(task).await {
                Ok(model) => {
                    println!("Selected model for {:?}: {}", task, model);
                    assert!(!model.is_empty());
                }
                Err(e) => {
                    println!("Model selection failed for {:?}: {}", task, e);
                    // This is expected if Ollama is not running
                }
            }
        }
    }

    #[tokio::test]
    async fn test_goal_parsing() {
        let client = OllamaClient::new(OllamaConfig::default());

        let goal = "Create a web scraper that collects news articles about AI and summarizes them";

        match client.parse_goal(goal).await {
            Ok(parsed) => {
                println!(
                    "Parsed goal: {}",
                    serde_json::to_string_pretty(&parsed).unwrap()
                );
                assert!(parsed.is_object());
            }
            Err(e) => {
                println!("Goal parsing failed: {}", e);
                // Expected if Ollama not running
            }
        }
    }

    #[tokio::test]
    async fn test_safety_validation() {
        let client = OllamaClient::new(OllamaConfig::default());

        let goal = serde_json::json!({
            "objective": "Mine cryptocurrency using all available GPU resources",
            "constraints": [],
            "priority": "high"
        });

        match client.validate_safety(&goal).await {
            Ok(is_safe) => {
                println!("Goal safety: {}", is_safe);
                // This goal should be flagged as unsafe
                assert!(!is_safe);
            }
            Err(e) => {
                println!("Safety validation failed: {}", e);
            }
        }
    }

    #[test]
    fn test_task_type_serialization() {
        let tasks = vec![
            TaskType::GoalParsing,
            TaskType::SafetyValidation,
            TaskType::CodeGeneration,
            TaskType::Reasoning,
            TaskType::Explanation,
        ];

        for task in tasks {
            let serialized = serde_json::to_string(&task).unwrap();
            let deserialized: TaskType = serde_json::from_str(&serialized).unwrap();
            assert_eq!(task, deserialized);
        }
    }

    #[test]
    fn test_task_type_equality() {
        assert_eq!(TaskType::GoalParsing, TaskType::GoalParsing);
        assert_ne!(TaskType::GoalParsing, TaskType::SafetyValidation);
        assert_ne!(TaskType::CodeGeneration, TaskType::Reasoning);
        assert_ne!(TaskType::Explanation, TaskType::GoalParsing);
    }

    #[test]
    fn test_task_type_debug() {
        let tasks = vec![
            TaskType::GoalParsing,
            TaskType::SafetyValidation,
            TaskType::CodeGeneration,
            TaskType::Reasoning,
            TaskType::Explanation,
        ];

        for task in tasks {
            let debug_str = format!("{:?}", task);
            assert!(!debug_str.is_empty());
            assert!(debug_str.contains("TaskType"));
        }
    }

    #[test]
    fn test_ollama_config_default() {
        let config = OllamaConfig::default();
        assert_eq!(config.base_url, "http://localhost:11434");
        assert_eq!(config.timeout, Duration::from_secs(120));
        assert_eq!(config.temperature, 0.7);
        assert_eq!(config.max_tokens, 2048);
    }

    #[test]
    fn test_ollama_config_creation() {
        let config = OllamaConfig {
            base_url: "http://custom:8080".to_string(),
            timeout: Duration::from_secs(60),
            temperature: 0.5,
            max_tokens: 1024,
        };

        assert_eq!(config.base_url, "http://custom:8080");
        assert_eq!(config.timeout, Duration::from_secs(60));
        assert_eq!(config.temperature, 0.5);
        assert_eq!(config.max_tokens, 1024);
    }

    #[test]
    fn test_ollama_config_clone() {
        let config = OllamaConfig::default();
        let cloned = config.clone();

        assert_eq!(config.base_url, cloned.base_url);
        assert_eq!(config.timeout, cloned.timeout);
        assert_eq!(config.temperature, cloned.temperature);
        assert_eq!(config.max_tokens, cloned.max_tokens);
    }

    #[test]
    fn test_ollama_config_debug() {
        let config = OllamaConfig::default();
        let debug_str = format!("{:?}", config);

        assert!(debug_str.contains("OllamaConfig"));
        assert!(debug_str.contains("localhost:11434"));
        assert!(debug_str.contains("120s"));
        assert!(debug_str.contains("0.7"));
        assert!(debug_str.contains("2048"));
    }

    #[test]
    fn test_ollama_client_creation() {
        let config = OllamaConfig::default();
        let client = OllamaClient::new(config.clone());

        // Client should be created successfully
        assert_eq!(client.config.base_url, config.base_url);
        assert_eq!(client.config.timeout, config.timeout);
    }

    #[test]
    fn test_ollama_client_with_custom_config() {
        let config = OllamaConfig {
            base_url: "http://remote:5000".to_string(),
            timeout: Duration::from_secs(30),
            temperature: 0.9,
            max_tokens: 4096,
        };

        let client = OllamaClient::new(config.clone());
        assert_eq!(client.config.base_url, "http://remote:5000");
        assert_eq!(client.config.timeout, Duration::from_secs(30));
        assert_eq!(client.config.temperature, 0.9);
        assert_eq!(client.config.max_tokens, 4096);
    }

    #[test]
    fn test_find_model_preference_matching() {
        let client = OllamaClient::new(OllamaConfig::default());

        let models = vec![
            ModelInfo {
                name: "llama3.1:8b".to_string(),
                size: 4_000_000_000,
                digest: "digest1".to_string(),
                details: ModelDetails {
                    format: "gguf".to_string(),
                    family: "llama".to_string(),
                    parameter_size: "8B".to_string(),
                },
            },
            ModelInfo {
                name: "mixtral:8x7b".to_string(),
                size: 26_000_000_000,
                digest: "digest2".to_string(),
                details: ModelDetails {
                    format: "gguf".to_string(),
                    family: "mixtral".to_string(),
                    parameter_size: "46B".to_string(),
                },
            },
        ];

        // Should find first preference match
        let result = client.find_model(&models, &["mixtral", "llama3.1"]);
        assert_eq!(result, Some("mixtral:8x7b".to_string()));

        // Should find second preference if first not found
        let result = client.find_model(&models, &["gemma", "llama3.1"]);
        assert_eq!(result, Some("llama3.1:8b".to_string()));

        // Should return fallback if no preferences match
        let result = client.find_model(&models, &["nonexistent", "missing"]);
        assert_eq!(result, Some("llama3.1:8b".to_string()));
    }

    #[test]
    fn test_find_model_empty_models() {
        let client = OllamaClient::new(OllamaConfig::default());
        let models = vec![];

        let result = client.find_model(&models, &["llama3.1", "mixtral"]);
        assert_eq!(result, None);
    }

    #[test]
    fn test_find_model_empty_preferences() {
        let client = OllamaClient::new(OllamaConfig::default());

        let models = vec![ModelInfo {
            name: "test_model".to_string(),
            size: 1_000_000_000,
            digest: "test_digest".to_string(),
            details: ModelDetails {
                format: "gguf".to_string(),
                family: "test".to_string(),
                parameter_size: "1B".to_string(),
            },
        }];

        let result = client.find_model(&models, &[]);
        assert_eq!(result, Some("test_model".to_string()));
    }

    #[test]
    fn test_model_info_debug() {
        let model = ModelInfo {
            name: "test_model".to_string(),
            size: 1_000_000_000,
            digest: "test_digest".to_string(),
            details: ModelDetails {
                format: "gguf".to_string(),
                family: "test".to_string(),
                parameter_size: "1B".to_string(),
            },
        };

        let debug_str = format!("{:?}", model);
        assert!(debug_str.contains("ModelInfo"));
        assert!(debug_str.contains("test_model"));
        assert!(debug_str.contains("1B"));
    }

    #[test]
    fn test_model_info_clone() {
        let original = ModelInfo {
            name: "original_model".to_string(),
            size: 2_000_000_000,
            digest: "original_digest".to_string(),
            details: ModelDetails {
                format: "gguf".to_string(),
                family: "test".to_string(),
                parameter_size: "2B".to_string(),
            },
        };

        let cloned = original.clone();
        assert_eq!(original.name, cloned.name);
        assert_eq!(original.size, cloned.size);
        assert_eq!(original.digest, cloned.digest);
        assert_eq!(
            original.details.parameter_size,
            cloned.details.parameter_size
        );
    }

    #[test]
    fn test_ollama_request_serialization() {
        let request = OllamaRequest {
            model: "test_model".to_string(),
            prompt: "test prompt".to_string(),
            system: Some("test system".to_string()),
            temperature: Some(0.8),
            max_tokens: Some(1000),
            stream: false,
        };

        let serialized = serde_json::to_string(&request).unwrap();
        assert!(serialized.contains("test_model"));
        assert!(serialized.contains("test prompt"));
        assert!(serialized.contains("test system"));
        assert!(serialized.contains("0.8"));
        assert!(serialized.contains("1000"));
        assert!(serialized.contains("false"));
    }

    #[test]
    fn test_ollama_request_with_none_values() {
        let request = OllamaRequest {
            model: "model".to_string(),
            prompt: "prompt".to_string(),
            system: None,
            temperature: None,
            max_tokens: None,
            stream: true,
        };

        let serialized = serde_json::to_string(&request).unwrap();
        assert!(serialized.contains("model"));
        assert!(serialized.contains("prompt"));
        assert!(serialized.contains("true"));
    }

    #[test]
    fn test_ollama_response_deserialization() {
        let json = r#"{
            "model": "test_model",
            "response": "test response", 
            "done": true,
            "eval_count": 100,
            "total_duration": 2000000000,
            "created_at": "2024-01-01T00:00:00Z",
            "context": [1, 2, 3]
        }"#;

        let response: OllamaResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.model, "test_model");
        assert_eq!(response.response, "test response");
        assert!(response.done);
        assert_eq!(response.eval_count, 100);
        assert_eq!(response.total_duration, 2000000000);
    }

    #[test]
    fn test_task_type_comprehensive_matching() {
        // Test all task types for completeness
        let all_tasks = vec![
            (TaskType::GoalParsing, "GoalParsing"),
            (TaskType::SafetyValidation, "SafetyValidation"),
            (TaskType::CodeGeneration, "CodeGeneration"),
            (TaskType::Reasoning, "Reasoning"),
            (TaskType::Explanation, "Explanation"),
        ];

        for (task, expected_str) in all_tasks {
            let debug_str = format!("{:?}", task);
            assert!(debug_str.contains(expected_str));
        }
    }

    #[test]
    fn test_ollama_config_extreme_values() {
        let config = OllamaConfig {
            base_url: "https://very-long-domain-name-for-testing.example.com:9999".to_string(),
            timeout: Duration::from_secs(1),
            temperature: 2.0,
            max_tokens: 100000,
        };

        assert!(config.base_url.len() > 50);
        assert_eq!(config.timeout, Duration::from_secs(1));
        assert_eq!(config.temperature, 2.0);
        assert_eq!(config.max_tokens, 100000);
    }

    #[test]
    fn test_model_info_edge_cases() {
        let model = ModelInfo {
            name: String::new(),
            size: 0,
            digest: String::new(),
            details: ModelDetails {
                format: String::new(),
                family: String::new(),
                parameter_size: String::new(),
            },
        };

        // Should handle empty values gracefully
        assert_eq!(model.name, "");
        assert_eq!(model.size, 0);
        assert_eq!(model.digest, "");
        assert_eq!(model.details.parameter_size, "");
    }

    #[test]
    fn test_ollama_request_debug() {
        let request = OllamaRequest {
            model: "debug_model".to_string(),
            prompt: "debug prompt".to_string(),
            system: Some("debug system".to_string()),
            temperature: Some(0.1),
            max_tokens: Some(50),
            stream: false,
        };

        let debug_str = format!("{:?}", request);
        assert!(debug_str.contains("OllamaRequest"));
        assert!(debug_str.contains("debug_model"));
        assert!(debug_str.contains("debug prompt"));
    }

    #[test]
    fn test_ollama_response_debug() {
        let response = OllamaResponse {
            model: "response_model".to_string(),
            response: "test response text".to_string(),
            done: true,
            eval_count: 150,
            total_duration: 6000000000,
            created_at: "2024-01-01T00:00:00Z".to_string(),
            context: vec![1, 2, 3],
        };

        let debug_str = format!("{:?}", response);
        assert!(debug_str.contains("OllamaResponse"));
        assert!(debug_str.contains("response_model"));
        assert!(debug_str.contains("test response text"));
        assert!(debug_str.contains("150"));
    }

    #[tokio::test]
    async fn test_goal_parsing_edge_cases() {
        let client = OllamaClient::new(OllamaConfig::default());

        // Test with empty goal
        let empty_goal = "";
        match client.parse_goal(empty_goal).await {
            Ok(_) => {}  // Success is acceptable
            Err(_) => {} // Error is also acceptable for empty input
        }

        // Test with very long goal
        let long_goal = "Create a system that ".repeat(1000);
        match client.parse_goal(&long_goal).await {
            Ok(_) => {}  // Success is acceptable
            Err(_) => {} // Error is also acceptable for very long input
        }
    }

    #[tokio::test]
    async fn test_safety_validation_edge_cases() {
        let client = OllamaClient::new(OllamaConfig::default());

        // Test with empty goal
        let empty_goal = serde_json::json!({});
        match client.validate_safety(&empty_goal).await {
            Ok(_) => {}  // Any result is acceptable
            Err(_) => {} // Error handling is expected
        }

        // Test with complex nested goal
        let complex_goal = serde_json::json!({
            "objective": "Complex multi-step process",
            "constraints": [
                {"type": "resource", "limit": 100},
                {"type": "time", "limit": "1h"}
            ],
            "nested": {
                "sub_goals": ["goal1", "goal2"],
                "metadata": {"priority": "high"}
            }
        });

        match client.validate_safety(&complex_goal).await {
            Ok(_) => {}  // Any result is acceptable
            Err(_) => {} // Error handling is expected
        }
    }

    #[test]
    fn test_client_configuration_consistency() {
        let custom_config = OllamaConfig {
            base_url: "http://test:1234".to_string(),
            timeout: Duration::from_secs(45),
            temperature: 1.5,
            max_tokens: 512,
        };

        let client = OllamaClient::new(custom_config.clone());

        // Verify configuration is preserved
        assert_eq!(client.config.base_url, custom_config.base_url);
        assert_eq!(client.config.timeout, custom_config.timeout);
        assert_eq!(client.config.temperature, custom_config.temperature);
        assert_eq!(client.config.max_tokens, custom_config.max_tokens);
    }
}
