//! LLM integration for natural language goal parsing

use crate::error::{BusinessError, BusinessResult};
use crate::goal::{
    BusinessGoal, Constraint, Criterion, GoalCategory, GoalPriority, ResourceLimits, SafetyLevel,
};
use crate::ollama_client::{OllamaClient, OllamaConfig};
use chrono::{Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::debug;

/// LLM-based goal parser using Ollama
pub struct LlmGoalParser {
    client: OllamaClient,
    system_prompt: String,
}

/// Parsed goal information from LLM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedGoalInfo {
    /// Cleaned and structured description
    pub description: String,
    /// Inferred goal category
    pub category: GoalCategory,
    /// Inferred priority
    pub priority: GoalPriority,
    /// Extracted constraints
    pub constraints: Vec<Constraint>,
    /// Extracted success criteria
    pub success_criteria: Vec<Criterion>,
    /// Estimated resource requirements
    pub resource_limits: ResourceLimits,
    /// Estimated duration in hours
    pub estimated_duration_hours: Option<f64>,
    /// Safety analysis
    pub safety_analysis: SafetyAnalysis,
    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Safety analysis from LLM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyAnalysis {
    /// Overall safety rating
    pub safety_rating: SafetyLevel,
    /// Identified risks
    pub risks: Vec<String>,
    /// Recommended mitigations
    pub mitigations: Vec<String>,
    /// Compliance requirements
    pub compliance_requirements: Vec<String>,
    /// Data sensitivity level
    pub data_sensitivity: String,
}

impl LlmGoalParser {
    /// Create a new LLM goal parser
    pub fn new(_api_key: Option<String>) -> BusinessResult<Self> {
        // API key is ignored for Ollama (local deployment)
        let config = OllamaConfig::default();
        let client = OllamaClient::new(config);

        let system_prompt = r#"
You are an expert AI system for parsing business goals and extracting structured information.
Your task is to analyze natural language goal descriptions and extract:

1. Goal category (DataAnalysis, MachineLearning, Research, etc.)
2. Priority level (Low, Medium, High, Critical, Emergency)
3. Constraints (time, resource, safety, compliance, budget, geographic, privacy)
4. Success criteria (performance, quality, completion, accuracy, efficiency)
5. Resource requirements (GPU memory, CPU, system memory, storage, time, cost)
6. Safety analysis (risks, mitigations, compliance)

Respond with a JSON object containing all extracted information.
Be conservative with resource estimates and prioritize safety.
If information is unclear, ask for clarification or make conservative assumptions.
"#
        .trim()
        .to_string();

        Ok(Self {
            client,
            system_prompt,
        })
    }

    /// Parse a natural language goal description
    pub async fn parse_goal(
        &self,
        description: &str,
        submitted_by: &str,
    ) -> BusinessResult<BusinessGoal> {
        debug!("Parsing goal: {}", description);

        let parsed_info = self.extract_goal_info(description).await?;

        let mut goal = BusinessGoal::new(parsed_info.description.clone(), submitted_by.to_string());

        // Apply parsed information
        goal.category = parsed_info.category;
        goal.priority = parsed_info.priority;
        goal.constraints = parsed_info.constraints;
        goal.success_criteria = parsed_info.success_criteria;
        goal.resource_limits = parsed_info.resource_limits;

        if let Some(hours) = parsed_info.estimated_duration_hours {
            goal.estimated_duration = Some(std::time::Duration::from_secs_f64(hours * 3600.0));
        }

        // Add safety metadata
        goal.metadata.insert(
            "safety_analysis".to_string(),
            serde_json::to_value(&parsed_info.safety_analysis)?,
        );

        // Add other metadata
        for (key, value) in parsed_info.metadata {
            goal.metadata.insert(key, value);
        }

        // Validate the parsed goal
        goal.validate()?;

        debug!("Successfully parsed goal: {}", goal.goal_id);
        Ok(goal)
    }

    /// Extract structured information from goal description
    async fn extract_goal_info(&self, description: &str) -> BusinessResult<ParsedGoalInfo> {
        #[cfg(feature = "mock")]
        {
            // Mock implementation for testing
            return Ok(self.create_mock_parsed_info(description));
        }

        // Use Ollama to parse the goal
        let parsed_json = self.client.parse_goal(description).await?;

        // The Ollama client returns a JSON value, convert to ParsedGoalInfo
        let parsed_info: ParsedGoalInfo =
            serde_json::from_value(parsed_json).map_err(|e| BusinessError::GoalParsingFailed {
                message: format!("Failed to parse LLM response: {}", e),
            })?;

        Ok(parsed_info)
    }

    #[cfg(feature = "mock")]
    fn create_mock_parsed_info(&self, description: &str) -> ParsedGoalInfo {
        let category = if description.contains("data") || description.contains("analysis") {
            GoalCategory::DataAnalysis
        } else if description.contains("machine learning") || description.contains("ML") {
            GoalCategory::MachineLearning
        } else if description.contains("research") {
            GoalCategory::Research
        } else {
            GoalCategory::Custom("General".to_string())
        };

        let priority = if description.contains("urgent") || description.contains("critical") {
            GoalPriority::High
        } else if description.contains("low priority") {
            GoalPriority::Low
        } else {
            GoalPriority::Medium
        };

        let mut constraints = Vec::new();
        let mut success_criteria = Vec::new();

        // Extract common patterns
        if description.contains("deadline") || description.contains("by ") {
            constraints.push(Constraint::TimeLimit {
                deadline: Utc::now() + Duration::days(7), // Default to 1 week
            });
        }

        if description.contains("budget") || description.contains("cost") {
            constraints.push(Constraint::BudgetLimit {
                currency: "USD".to_string(),
                max_amount: 1000.0,
            });
        }

        if description.contains("accuracy") || description.contains("accurate") {
            success_criteria.push(Criterion::Accuracy { min_accuracy: 0.95 });
        }

        if description.contains("complete") || description.contains("finish") {
            success_criteria.push(Criterion::Completion { percentage: 100.0 });
        }

        let resource_limits = if description.contains("large") || description.contains("big") {
            ResourceLimits {
                max_gpu_memory_mb: Some(16384), // 16GB
                max_cpu_cores: Some(16),
                max_memory_mb: Some(32768),  // 32GB
                max_storage_mb: Some(51200), // 50GB
                max_execution_time: Some(std::time::Duration::from_secs(4 * 3600)),
                max_cost_usd: Some(500.0),
                max_agents: Some(20),
                max_network_mbps: Some(2000.0),
            }
        } else {
            ResourceLimits::default()
        };

        let safety_rating = if description.contains("sensitive") || description.contains("private")
        {
            SafetyLevel::High
        } else if description.contains("public") {
            SafetyLevel::Low
        } else {
            SafetyLevel::Medium
        };

        let safety_analysis = SafetyAnalysis {
            safety_rating,
            risks: vec!["Data privacy".to_string(), "Resource usage".to_string()],
            mitigations: vec!["Encryption".to_string(), "Access controls".to_string()],
            compliance_requirements: vec!["GDPR".to_string()],
            data_sensitivity: "Medium".to_string(),
        };

        let estimated_duration_hours =
            if description.contains("quick") || description.contains("fast") {
                Some(1.0)
            } else if description.contains("complex") || description.contains("detailed") {
                Some(8.0)
            } else {
                Some(4.0)
            };

        ParsedGoalInfo {
            description: description.to_string(),
            category,
            priority,
            constraints,
            success_criteria,
            resource_limits,
            estimated_duration_hours,
            safety_analysis,
            metadata: HashMap::new(),
        }
    }

    /// Validate parsed goal information
    pub fn validate_parsed_info(&self, info: &ParsedGoalInfo) -> BusinessResult<()> {
        if info.description.is_empty() {
            return Err(BusinessError::GoalValidationFailed {
                reason: "Parsed description cannot be empty".to_string(),
            });
        }

        // Validate resource limits
        if let Some(memory) = info.resource_limits.max_gpu_memory_mb {
            if memory == 0 {
                return Err(BusinessError::GoalValidationFailed {
                    reason: "GPU memory limit cannot be zero".to_string(),
                });
            }
        }

        // Validate duration
        if let Some(hours) = info.estimated_duration_hours {
            if hours <= 0.0 {
                return Err(BusinessError::GoalValidationFailed {
                    reason: "Estimated duration must be positive".to_string(),
                });
            }
        }

        Ok(())
    }

    /// Set custom model (not used with Ollama - it auto-selects based on task)
    pub fn set_model(&mut self, _model: String) {
        // Model selection is handled automatically by OllamaClient based on task type
        debug!("Model selection is automatic with Ollama based on task type");
    }

    /// Set custom system prompt
    pub fn set_system_prompt(&mut self, prompt: String) {
        self.system_prompt = prompt;
    }

    /// Get current configuration
    pub fn get_config(&self) -> ParserConfig {
        ParserConfig {
            model: "auto-selected".to_string(), // Ollama auto-selects
            max_tokens: 2048,                   // Default value
            temperature: 0.7,                   // Default value
            system_prompt: self.system_prompt.clone(),
        }
    }
}

/// Parser configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParserConfig {
    pub model: String,
    pub max_tokens: u16,
    pub temperature: f32,
    pub system_prompt: String,
}

impl Default for LlmGoalParser {
    fn default() -> Self {
        Self::new(None).expect("Failed to create default LLM parser")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_parser() -> LlmGoalParser {
        LlmGoalParser::new(None).unwrap() // Ollama doesn't need API key
    }

    #[test]
    fn test_parser_creation() {
        let parser = create_test_parser();
        // Ollama auto-selects models, so we just verify the parser was created
        assert!(!parser.system_prompt.is_empty());
    }

    #[test]
    fn test_parser_creation_without_api_key() {
        let parser = LlmGoalParser::new(None);
        assert!(parser.is_ok());
    }

    #[test]
    fn test_set_model() {
        let mut parser = create_test_parser();
        parser.set_model("some-model".to_string());
        // Ollama auto-selects models, so this is a no-op
        let config = parser.get_config();
        assert_eq!(config.model, "auto-selected");
    }

    #[test]
    fn test_set_system_prompt() {
        let mut parser = create_test_parser();
        let new_prompt = "Custom prompt".to_string();
        parser.set_system_prompt(new_prompt.clone());
        assert_eq!(parser.system_prompt, new_prompt);
    }

    #[test]
    fn test_get_config() {
        let parser = create_test_parser();
        let config = parser.get_config();
        assert_eq!(config.model, "auto-selected");
        assert_eq!(config.max_tokens, 2048);
        assert_eq!(config.temperature, 0.7);
    }

    #[cfg(feature = "mock")]
    #[test]
    fn test_mock_parsed_info_data_analysis() {
        let parser = create_test_parser();
        let description = "Analyze customer data to find patterns";
        let info = parser.create_mock_parsed_info(description);

        assert_eq!(info.description, description);
        assert_eq!(info.category, GoalCategory::DataAnalysis);
        assert_eq!(info.priority, GoalPriority::Medium);
        assert!(!info.success_criteria.is_empty());
    }

    #[cfg(feature = "mock")]
    #[test]
    fn test_mock_parsed_info_machine_learning() {
        let parser = create_test_parser();
        let description = "Build a machine learning model for prediction";
        let info = parser.create_mock_parsed_info(description);

        assert_eq!(info.category, GoalCategory::MachineLearning);
    }

    #[cfg(feature = "mock")]
    #[test]
    fn test_mock_parsed_info_urgent() {
        let parser = create_test_parser();
        let description = "Urgent analysis needed for critical decision";
        let info = parser.create_mock_parsed_info(description);

        assert_eq!(info.priority, GoalPriority::High);
    }

    #[cfg(feature = "mock")]
    #[test]
    fn test_mock_parsed_info_with_deadline() {
        let parser = create_test_parser();
        let description = "Complete analysis by next week deadline";
        let info = parser.create_mock_parsed_info(description);

        assert!(!info.constraints.is_empty());
        assert!(info
            .constraints
            .iter()
            .any(|c| matches!(c, Constraint::TimeLimit { .. })));
    }

    #[cfg(feature = "mock")]
    #[test]
    fn test_mock_parsed_info_with_budget() {
        let parser = create_test_parser();
        let description = "Analysis within budget constraints of $500";
        let info = parser.create_mock_parsed_info(description);

        assert!(info
            .constraints
            .iter()
            .any(|c| matches!(c, Constraint::BudgetLimit { .. })));
    }

    #[cfg(feature = "mock")]
    #[test]
    fn test_mock_parsed_info_accuracy_requirement() {
        let parser = create_test_parser();
        let description = "High accuracy model needed for predictions";
        let info = parser.create_mock_parsed_info(description);

        assert!(info
            .success_criteria
            .iter()
            .any(|c| matches!(c, Criterion::Accuracy { .. })));
    }

    #[cfg(feature = "mock")]
    #[test]
    fn test_mock_parsed_info_large_scale() {
        let parser = create_test_parser();
        let description = "Large scale data processing for big dataset";
        let info = parser.create_mock_parsed_info(description);

        assert_eq!(info.resource_limits.max_gpu_memory_mb, Some(16384));
        assert_eq!(info.resource_limits.max_cpu_cores, Some(16));
    }

    #[cfg(feature = "mock")]
    #[test]
    fn test_mock_parsed_info_sensitive_data() {
        let parser = create_test_parser();
        let description = "Analysis of sensitive customer data";
        let info = parser.create_mock_parsed_info(description);

        assert_eq!(info.safety_analysis.safety_rating, SafetyLevel::High);
        assert!(!info.safety_analysis.risks.is_empty());
        assert!(!info.safety_analysis.mitigations.is_empty());
    }

    #[cfg(feature = "mock")]
    #[test]
    fn test_mock_parsed_info_quick_task() {
        let parser = create_test_parser();
        let description = "Quick analysis of recent data";
        let info = parser.create_mock_parsed_info(description);

        assert_eq!(info.estimated_duration_hours, Some(1.0));
    }

    #[cfg(feature = "mock")]
    #[test]
    fn test_mock_parsed_info_complex_task() {
        let parser = create_test_parser();
        let description = "Complex detailed analysis of multiple datasets";
        let info = parser.create_mock_parsed_info(description);

        assert_eq!(info.estimated_duration_hours, Some(8.0));
    }

    #[test]
    fn test_validate_parsed_info_success() {
        let parser = create_test_parser();
        let info = ParsedGoalInfo {
            description: "Valid description".to_string(),
            category: GoalCategory::DataAnalysis,
            priority: GoalPriority::Medium,
            constraints: Vec::new(),
            success_criteria: Vec::new(),
            resource_limits: ResourceLimits::default(),
            estimated_duration_hours: Some(2.0),
            safety_analysis: SafetyAnalysis {
                safety_rating: SafetyLevel::Medium,
                risks: Vec::new(),
                mitigations: Vec::new(),
                compliance_requirements: Vec::new(),
                data_sensitivity: "Medium".to_string(),
            },
            metadata: HashMap::new(),
        };

        assert!(parser.validate_parsed_info(&info).is_ok());
    }

    #[test]
    fn test_validate_parsed_info_empty_description() {
        let parser = create_test_parser();
        let info = ParsedGoalInfo {
            description: String::new(),
            category: GoalCategory::DataAnalysis,
            priority: GoalPriority::Medium,
            constraints: Vec::new(),
            success_criteria: Vec::new(),
            resource_limits: ResourceLimits::default(),
            estimated_duration_hours: Some(2.0),
            safety_analysis: SafetyAnalysis {
                safety_rating: SafetyLevel::Medium,
                risks: Vec::new(),
                mitigations: Vec::new(),
                compliance_requirements: Vec::new(),
                data_sensitivity: "Medium".to_string(),
            },
            metadata: HashMap::new(),
        };

        assert!(parser.validate_parsed_info(&info).is_err());
    }

    #[test]
    fn test_validate_parsed_info_zero_gpu_memory() {
        let parser = create_test_parser();
        let info = ParsedGoalInfo {
            description: "Valid description".to_string(),
            category: GoalCategory::DataAnalysis,
            priority: GoalPriority::Medium,
            constraints: Vec::new(),
            success_criteria: Vec::new(),
            resource_limits: ResourceLimits {
                max_gpu_memory_mb: Some(0),
                ..Default::default()
            },
            estimated_duration_hours: Some(2.0),
            safety_analysis: SafetyAnalysis {
                safety_rating: SafetyLevel::Medium,
                risks: Vec::new(),
                mitigations: Vec::new(),
                compliance_requirements: Vec::new(),
                data_sensitivity: "Medium".to_string(),
            },
            metadata: HashMap::new(),
        };

        assert!(parser.validate_parsed_info(&info).is_err());
    }

    #[test]
    fn test_validate_parsed_info_negative_duration() {
        let parser = create_test_parser();
        let info = ParsedGoalInfo {
            description: "Valid description".to_string(),
            category: GoalCategory::DataAnalysis,
            priority: GoalPriority::Medium,
            constraints: Vec::new(),
            success_criteria: Vec::new(),
            resource_limits: ResourceLimits::default(),
            estimated_duration_hours: Some(-1.0),
            safety_analysis: SafetyAnalysis {
                safety_rating: SafetyLevel::Medium,
                risks: Vec::new(),
                mitigations: Vec::new(),
                compliance_requirements: Vec::new(),
                data_sensitivity: "Medium".to_string(),
            },
            metadata: HashMap::new(),
        };

        assert!(parser.validate_parsed_info(&info).is_err());
    }

    #[test]
    fn test_safety_analysis_serialization() {
        let analysis = SafetyAnalysis {
            safety_rating: SafetyLevel::High,
            risks: vec!["Data breach".to_string(), "Privacy violation".to_string()],
            mitigations: vec!["Encryption".to_string(), "Access control".to_string()],
            compliance_requirements: vec!["GDPR".to_string(), "HIPAA".to_string()],
            data_sensitivity: "High".to_string(),
        };

        let serialized = serde_json::to_string(&analysis).unwrap();
        let deserialized: SafetyAnalysis = serde_json::from_str(&serialized).unwrap();

        assert_eq!(analysis.safety_rating, deserialized.safety_rating);
        assert_eq!(analysis.risks, deserialized.risks);
        assert_eq!(analysis.mitigations, deserialized.mitigations);
    }

    #[test]
    fn test_parsed_goal_info_serialization() {
        let info = ParsedGoalInfo {
            description: "Test description".to_string(),
            category: GoalCategory::DataAnalysis,
            priority: GoalPriority::High,
            constraints: vec![Constraint::TimeLimit {
                deadline: Utc::now() + Duration::hours(24),
            }],
            success_criteria: vec![Criterion::Accuracy { min_accuracy: 0.95 }],
            resource_limits: ResourceLimits::default(),
            estimated_duration_hours: Some(4.0),
            safety_analysis: SafetyAnalysis {
                safety_rating: SafetyLevel::Medium,
                risks: Vec::new(),
                mitigations: Vec::new(),
                compliance_requirements: Vec::new(),
                data_sensitivity: "Medium".to_string(),
            },
            metadata: HashMap::new(),
        };

        let serialized = serde_json::to_string(&info).unwrap();
        let deserialized: ParsedGoalInfo = serde_json::from_str(&serialized).unwrap();

        assert_eq!(info.description, deserialized.description);
        assert_eq!(info.category, deserialized.category);
        assert_eq!(info.priority, deserialized.priority);
    }

    #[tokio::test]
    async fn test_parse_goal_mock() {
        let parser = create_test_parser();
        let description = "Analyze customer data to improve recommendations";
        let result = parser.parse_goal(description, "test@example.com").await;

        assert!(result.is_ok());
        let goal = result.unwrap();
        assert_eq!(goal.description, description);
        assert_eq!(goal.submitted_by, "test@example.com");
        assert!(!goal.goal_id.is_empty());
    }

    #[test]
    fn test_parser_config_serialization() {
        let config = ParserConfig {
            model: "gpt-4".to_string(),
            max_tokens: 2048,
            temperature: 0.3,
            system_prompt: "Test prompt".to_string(),
        };

        let serialized = serde_json::to_string(&config).unwrap();
        let deserialized: ParserConfig = serde_json::from_str(&serialized).unwrap();

        assert_eq!(config.model, deserialized.model);
        assert_eq!(config.max_tokens, deserialized.max_tokens);
        assert_eq!(config.temperature, deserialized.temperature);
        assert_eq!(config.system_prompt, deserialized.system_prompt);
    }

    #[test]
    fn test_default_parser() {
        let parser = LlmGoalParser::default();
        // Just verify it was created successfully
        assert!(!parser.system_prompt.is_empty());
    }
}
