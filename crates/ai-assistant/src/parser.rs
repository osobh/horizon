//! Natural language parsing for StratoSwarm commands

use crate::error::AssistantResult;
use lazy_static::lazy_static;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

lazy_static! {
    // Common patterns for different intents
    static ref DEPLOY_PATTERN: Regex = Regex::new(r"(?i)(deploy|launch|start|run)\s+(\S+)(?:\s+from\s+(\S+))?").unwrap();
    static ref SCALE_PATTERN: Regex = Regex::new(r"(?i)(scale|resize)\s+(\S+)(?:\s+to\s+(\d+))?").unwrap();
    static ref QUERY_PATTERN: Regex = Regex::new(r"(?i)(show|list|get|find|query)\s+(?:me\s+)?(.+)").unwrap();
    static ref DEBUG_PATTERN: Regex = Regex::new(r"(?i)(debug|troubleshoot|diagnose|fix)\s+(\S+)").unwrap();
    static ref OPTIMIZE_PATTERN: Regex = Regex::new(r"(?i)(optimize|improve|enhance)\s+(\S+)(?:\s+for\s+(\S+))?").unwrap();
    static ref HELP_PATTERN: Regex = Regex::new(r"(?i)(help|how|what|explain)(?:\s+(?:me\s+)?(?:about|with)?\s*(.+))?").unwrap();
    static ref STATUS_PATTERN: Regex = Regex::new(r"(?i)(status|health|state)\s+(?:of\s+)?(\S+)?").unwrap();
    static ref LOGS_PATTERN: Regex = Regex::new(r"(?i)(logs?|tail)\s+(?:for\s+|from\s+)?(\S+)").unwrap();
    static ref ROLLBACK_PATTERN: Regex = Regex::new(r"(?i)(rollback|revert|undo)\s+(\S+)(?:\s+to\s+(\S+))?").unwrap();
    static ref EVOLUTION_PATTERN: Regex = Regex::new(r"(?i)(evolve|mutate|improve)\s+(\S+)").unwrap();
}

/// Intent parsed from natural language
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Intent {
    Deploy {
        target: String,
        source: Option<String>,
        config: HashMap<String, String>,
    },
    Scale {
        target: String,
        replicas: Option<u32>,
        resources: Option<ResourceSpec>,
    },
    Query {
        resource_type: String,
        filters: HashMap<String, String>,
        projection: Option<Vec<String>>,
    },
    Debug {
        target: String,
        symptoms: Vec<String>,
    },
    Optimize {
        target: String,
        metric: String,
        constraints: Vec<String>,
    },
    Status {
        target: Option<String>,
    },
    Logs {
        target: String,
        follow: bool,
        lines: Option<u32>,
    },
    Rollback {
        target: String,
        version: Option<String>,
    },
    Evolve {
        target: String,
        fitness_function: Option<String>,
    },
    Help {
        topic: Option<String>,
    },
    Unknown {
        raw_input: String,
    },
}

/// Resource specifications
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResourceSpec {
    pub cpu: Option<String>,
    pub memory: Option<String>,
    pub gpu: Option<String>,
}

/// Parsed query with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedQuery {
    pub intent: Intent,
    pub confidence: f32,
    pub entities: HashMap<String, String>,
    pub context: QueryContext,
    pub raw_input: String,
}

/// Context for the query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryContext {
    pub timestamp: i64,
    pub user_id: Option<String>,
    pub session_id: Option<String>,
    pub previous_intents: Vec<Intent>,
}

impl Default for QueryContext {
    fn default() -> Self {
        Self {
            timestamp: chrono::Utc::now().timestamp(),
            user_id: None,
            session_id: None,
            previous_intents: Vec::new(),
        }
    }
}

/// Natural language parser
pub struct NaturalLanguageParser {
    /// Patterns for intent detection
    patterns: HashMap<String, Regex>,
    /// Entity extractors
    entity_extractors: Vec<Box<dyn EntityExtractor>>,
}

impl NaturalLanguageParser {
    pub fn new() -> AssistantResult<Self> {
        Ok(Self {
            patterns: HashMap::new(),
            entity_extractors: vec![
                Box::new(UrlExtractor),
                Box::new(NumberExtractor),
                Box::new(ResourceExtractor),
            ],
        })
    }

    pub async fn parse(&self, input: &str) -> AssistantResult<ParsedQuery> {
        let input = input.trim();
        let (intent, confidence) = self.detect_intent(input)?;
        let entities = self.extract_entities(input)?;

        Ok(ParsedQuery {
            intent,
            confidence,
            entities,
            context: QueryContext::default(),
            raw_input: input.to_string(),
        })
    }

    fn detect_intent(&self, input: &str) -> AssistantResult<(Intent, f32)> {
        // Try each pattern in order of specificity
        if let Some(captures) = DEPLOY_PATTERN.captures(input) {
            let target = captures
                .get(2)
                .map(|m| m.as_str().to_string())
                .unwrap_or_default();
            let source = captures.get(3).map(|m| m.as_str().to_string());
            return Ok((
                Intent::Deploy {
                    target,
                    source,
                    config: HashMap::new(),
                },
                0.9,
            ));
        }

        if let Some(captures) = SCALE_PATTERN.captures(input) {
            let target = captures
                .get(2)
                .map(|m| m.as_str().to_string())
                .unwrap_or_default();
            let replicas = captures.get(3).and_then(|m| m.as_str().parse().ok());
            return Ok((
                Intent::Scale {
                    target,
                    replicas,
                    resources: None,
                },
                0.85,
            ));
        }

        if let Some(captures) = QUERY_PATTERN.captures(input) {
            let resource_type = captures
                .get(2)
                .map(|m| m.as_str().to_string())
                .unwrap_or_default();
            return Ok((
                Intent::Query {
                    resource_type,
                    filters: HashMap::new(),
                    projection: None,
                },
                0.85,
            ));
        }

        if let Some(captures) = DEBUG_PATTERN.captures(input) {
            let target = captures
                .get(2)
                .map(|m| m.as_str().to_string())
                .unwrap_or_default();
            return Ok((
                Intent::Debug {
                    target,
                    symptoms: vec![],
                },
                0.8,
            ));
        }

        if let Some(captures) = OPTIMIZE_PATTERN.captures(input) {
            let target = captures
                .get(2)
                .map(|m| m.as_str().to_string())
                .unwrap_or_default();
            let metric = captures
                .get(3)
                .map(|m| m.as_str().to_string())
                .unwrap_or("performance".to_string());
            return Ok((
                Intent::Optimize {
                    target,
                    metric,
                    constraints: vec![],
                },
                0.8,
            ));
        }

        if let Some(captures) = STATUS_PATTERN.captures(input) {
            let target = captures.get(2).map(|m| m.as_str().to_string());
            return Ok((Intent::Status { target }, 0.9));
        }

        if let Some(captures) = LOGS_PATTERN.captures(input) {
            let target = captures
                .get(2)
                .map(|m| m.as_str().to_string())
                .unwrap_or_default();
            let follow = input.contains("tail") || input.contains("follow");
            return Ok((
                Intent::Logs {
                    target,
                    follow,
                    lines: None,
                },
                0.9,
            ));
        }

        if let Some(captures) = ROLLBACK_PATTERN.captures(input) {
            let target = captures
                .get(2)
                .map(|m| m.as_str().to_string())
                .unwrap_or_default();
            let version = captures.get(3).map(|m| m.as_str().to_string());
            return Ok((Intent::Rollback { target, version }, 0.85));
        }

        if let Some(captures) = EVOLUTION_PATTERN.captures(input) {
            let target = captures
                .get(2)
                .map(|m| m.as_str().to_string())
                .unwrap_or_default();
            return Ok((
                Intent::Evolve {
                    target,
                    fitness_function: None,
                },
                0.8,
            ));
        }

        if let Some(captures) = HELP_PATTERN.captures(input) {
            let topic = captures.get(2).map(|m| m.as_str().to_string());
            return Ok((Intent::Help { topic }, 0.95));
        }

        // If no pattern matches, return unknown with low confidence
        Ok((
            Intent::Unknown {
                raw_input: input.to_string(),
            },
            0.3,
        ))
    }

    fn extract_entities(&self, input: &str) -> AssistantResult<HashMap<String, String>> {
        let mut entities = HashMap::new();

        for extractor in &self.entity_extractors {
            let extracted = extractor.extract(input);
            entities.extend(extracted);
        }

        Ok(entities)
    }
}

impl ParsedQuery {
    pub fn requires_command(&self) -> bool {
        matches!(
            self.intent,
            Intent::Deploy { .. }
                | Intent::Scale { .. }
                | Intent::Rollback { .. }
                | Intent::Evolve { .. }
        )
    }

    pub fn requires_query(&self) -> bool {
        matches!(
            self.intent,
            Intent::Query { .. } | Intent::Status { .. } | Intent::Logs { .. }
        )
    }
}

/// Trait for entity extraction
trait EntityExtractor: Send + Sync {
    fn extract(&self, input: &str) -> HashMap<String, String>;
}

/// URL extractor
struct UrlExtractor;

impl EntityExtractor for UrlExtractor {
    fn extract(&self, input: &str) -> HashMap<String, String> {
        let mut entities = HashMap::new();
        let url_pattern = Regex::new(r"https?://[^\s]+").unwrap();

        if let Some(capture) = url_pattern.find(input) {
            entities.insert("url".to_string(), capture.as_str().to_string());
        }

        entities
    }
}

/// Number extractor
struct NumberExtractor;

impl EntityExtractor for NumberExtractor {
    fn extract(&self, input: &str) -> HashMap<String, String> {
        let mut entities = HashMap::new();
        let number_pattern = Regex::new(r"\b(\d+)\b").unwrap();

        for (i, capture) in number_pattern.find_iter(input).enumerate() {
            entities.insert(format!("number_{}", i), capture.as_str().to_string());
        }

        entities
    }
}

/// Resource specification extractor
struct ResourceExtractor;

impl EntityExtractor for ResourceExtractor {
    fn extract(&self, input: &str) -> HashMap<String, String> {
        let mut entities = HashMap::new();

        // CPU pattern: 2 cores, 2.5 cpu, etc
        let cpu_pattern = Regex::new(r"(\d+(?:\.\d+)?)\s*(cpu|cores?)").unwrap();
        if let Some(capture) = cpu_pattern.captures(input) {
            entities.insert("cpu".to_string(), capture[1].to_string());
        }

        // Memory pattern: 4GB, 512Mi, etc
        let mem_pattern = Regex::new(r"(\d+(?:\.\d+)?)\s*(GB|GiB|MB|MiB|KB|KiB)").unwrap();
        if let Some(capture) = mem_pattern.captures(input) {
            entities.insert(
                "memory".to_string(),
                format!("{}{}", &capture[1], &capture[2]),
            );
        }

        // GPU pattern: 1 gpu, 0.5 GPU, etc
        let gpu_pattern = Regex::new(r"(\d+(?:\.\d+)?)\s*(gpu|GPU)").unwrap();
        if let Some(capture) = gpu_pattern.captures(input) {
            entities.insert("gpu".to_string(), capture[1].to_string());
        }

        entities
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_deploy_intent_parsing() {
        let parser = NaturalLanguageParser::new().unwrap();

        let query = parser
            .parse("deploy myapp from github.com/user/repo")
            .await
            .unwrap();
        assert!(matches!(query.intent, Intent::Deploy { .. }));
        assert_eq!(query.confidence, 0.9);

        if let Intent::Deploy { target, source, .. } = query.intent {
            assert_eq!(target, "myapp");
            assert_eq!(source, Some("github.com/user/repo".to_string()));
        }
    }

    #[tokio::test]
    async fn test_scale_intent_parsing() {
        let parser = NaturalLanguageParser::new().unwrap();

        let query = parser.parse("scale web-service to 5").await.unwrap();
        assert!(matches!(query.intent, Intent::Scale { .. }));

        if let Intent::Scale {
            target, replicas, ..
        } = query.intent
        {
            assert_eq!(target, "web-service");
            assert_eq!(replicas, Some(5));
        }
    }

    #[tokio::test]
    async fn test_query_intent_parsing() {
        let parser = NaturalLanguageParser::new().unwrap();

        let query = parser.parse("show me all running agents").await.unwrap();
        assert!(matches!(query.intent, Intent::Query { .. }));

        if let Intent::Query { resource_type, .. } = query.intent {
            assert_eq!(resource_type, "all running agents");
        }
    }

    #[tokio::test]
    async fn test_help_intent_parsing() {
        let parser = NaturalLanguageParser::new().unwrap();

        let query = parser.parse("help me with deployment").await.unwrap();
        assert!(matches!(query.intent, Intent::Help { .. }));

        if let Intent::Help { topic } = query.intent {
            assert_eq!(topic, Some("deployment".to_string()));
        }
    }

    #[tokio::test]
    async fn test_entity_extraction() {
        let parser = NaturalLanguageParser::new().unwrap();

        let query = parser
            .parse("deploy app with 2 cpu and 4GB memory")
            .await
            .unwrap();
        assert!(query.entities.contains_key("cpu"));
        assert!(query.entities.contains_key("memory"));
        assert_eq!(query.entities["cpu"], "2");
        assert_eq!(query.entities["memory"], "4GB");
    }

    #[tokio::test]
    async fn test_url_extraction() {
        let parser = NaturalLanguageParser::new().unwrap();

        let query = parser
            .parse("deploy from https://github.com/user/repo")
            .await
            .unwrap();
        assert!(query.entities.contains_key("url"));
        assert_eq!(query.entities["url"], "https://github.com/user/repo");
    }

    #[tokio::test]
    async fn test_unknown_intent() {
        let parser = NaturalLanguageParser::new().unwrap();

        let query = parser
            .parse("random gibberish that doesn't match anything")
            .await
            .unwrap();
        assert!(matches!(query.intent, Intent::Unknown { .. }));
        assert!(query.confidence < 0.5);
    }
}
