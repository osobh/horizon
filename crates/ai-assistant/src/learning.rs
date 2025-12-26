//! Learning system for the AI Assistant

use crate::error::AssistantResult;
use crate::parser::{Intent, ParsedQuery};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Learning system that improves from user interactions
pub struct LearningSystem {
    /// Whether learning is enabled
    enabled: bool,
    /// Patterns learned from interactions
    patterns: Vec<Pattern>,
    /// User feedback history
    feedback_history: Vec<FeedbackRecord>,
    /// Intent confidence adjustments
    confidence_adjustments: HashMap<String, f32>,
    /// Common query suggestions
    suggestion_cache: HashMap<String, Vec<String>>,
}

/// A learned pattern from user interactions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pattern {
    /// Input pattern that triggers this
    pub trigger: String,
    /// The intent it should map to
    pub intent: String,
    /// Confidence in this pattern
    pub confidence: f32,
    /// How many times this pattern was successful
    pub success_count: u32,
    /// How many times it failed
    pub failure_count: u32,
    /// When this pattern was learned
    pub learned_at: i64,
}

/// Record of user feedback
#[derive(Debug, Clone, Serialize, Deserialize)]
struct FeedbackRecord {
    input: String,
    intent: String,
    was_helpful: bool,
    timestamp: i64,
}

impl LearningSystem {
    pub fn new(enabled: bool) -> AssistantResult<Self> {
        Ok(Self {
            enabled,
            patterns: Vec::new(),
            feedback_history: Vec::new(),
            confidence_adjustments: HashMap::new(),
            suggestion_cache: HashMap::new(),
        })
    }

    /// Record an interaction for learning
    pub async fn record_interaction(
        &mut self,
        input: &str,
        parsed: &ParsedQuery,
        _response: &str,
    ) -> AssistantResult<()> {
        if !self.enabled {
            return Ok(());
        }

        // Extract patterns from successful interactions
        if parsed.confidence > 0.8 {
            self.extract_pattern(input, &parsed.intent).await?;
        }

        // Update suggestion cache
        self.update_suggestions(&parsed.intent).await?;

        Ok(())
    }

    /// Update from user feedback
    pub async fn update_from_feedback(
        &mut self,
        input: &str,
        response: &crate::AssistantResponse,
        was_helpful: bool,
    ) -> AssistantResult<()> {
        if !self.enabled {
            return Ok(());
        }

        // Record feedback
        let record = FeedbackRecord {
            input: input.to_string(),
            intent: format!("{:?}", response.intent),
            was_helpful,
            timestamp: chrono::Utc::now().timestamp(),
        };
        self.feedback_history.push(record);

        // Adjust confidence based on feedback
        let intent_key = format!("{:?}", response.intent);
        let adjustment = if was_helpful { 0.05 } else { -0.1 };

        let current = self.confidence_adjustments.get(&intent_key).unwrap_or(&0.0);
        self.confidence_adjustments
            .insert(intent_key, (current + adjustment).clamp(-0.5, 0.5));

        // Update patterns
        self.update_patterns_from_feedback(input, &response.intent, was_helpful)
            .await?;

        Ok(())
    }

    /// Get suggestions for a given context
    pub async fn get_suggestions_for_context(
        &self,
        intent: &Intent,
    ) -> AssistantResult<Vec<String>> {
        let intent_key = format!("{:?}", intent);

        if let Some(cached) = self.suggestion_cache.get(&intent_key) {
            return Ok(cached.clone());
        }

        // Generate context-appropriate suggestions
        let suggestions = match intent {
            Intent::Deploy { .. } => vec![
                "Scale the application if needed".to_string(),
                "Check deployment status".to_string(),
                "View application logs".to_string(),
                "Set up monitoring".to_string(),
            ],
            Intent::Scale { .. } => vec![
                "Monitor resource usage after scaling".to_string(),
                "Check application performance".to_string(),
                "View updated resource allocation".to_string(),
            ],
            Intent::Query { .. } => vec![
                "Filter results by status".to_string(),
                "Export results to file".to_string(),
                "Set up alerts for changes".to_string(),
            ],
            Intent::Debug { .. } => vec![
                "Check related logs".to_string(),
                "View resource metrics".to_string(),
                "Run performance diagnostics".to_string(),
                "Check network connectivity".to_string(),
            ],
            Intent::Optimize { .. } => vec![
                "Monitor optimization results".to_string(),
                "Compare before and after metrics".to_string(),
                "Schedule regular optimization".to_string(),
            ],
            Intent::Status { .. } => vec![
                "Drill down into specific resources".to_string(),
                "Set up monitoring dashboard".to_string(),
                "Check historical trends".to_string(),
            ],
            Intent::Logs { .. } => vec![
                "Filter logs by severity".to_string(),
                "Export logs for analysis".to_string(),
                "Set up log alerts".to_string(),
            ],
            Intent::Rollback { .. } => vec![
                "Verify rollback success".to_string(),
                "Check application health".to_string(),
                "Review what caused the issue".to_string(),
            ],
            Intent::Evolve { .. } => vec![
                "Monitor evolution progress".to_string(),
                "Check fitness improvements".to_string(),
                "Review generated variations".to_string(),
            ],
            Intent::Help { .. } => vec![
                "Try a specific command".to_string(),
                "Browse documentation".to_string(),
                "Join the community".to_string(),
            ],
            Intent::Unknown { .. } => vec![
                "Ask for help with your task".to_string(),
                "Try rephrasing your request".to_string(),
                "Check the documentation".to_string(),
            ],
        };

        Ok(suggestions)
    }

    /// Extract a pattern from successful interaction
    async fn extract_pattern(&mut self, input: &str, intent: &Intent) -> AssistantResult<()> {
        let intent_str = format!("{:?}", intent);

        // Look for existing pattern
        if let Some(pattern) = self.patterns.iter_mut().find(|p| p.trigger == input) {
            pattern.success_count += 1;
            pattern.confidence = (pattern.success_count as f32)
                / ((pattern.success_count + pattern.failure_count) as f32);
        } else {
            // Create new pattern
            let pattern = Pattern {
                trigger: input.to_string(),
                intent: intent_str,
                confidence: 1.0,
                success_count: 1,
                failure_count: 0,
                learned_at: chrono::Utc::now().timestamp(),
            };
            self.patterns.push(pattern);
        }

        Ok(())
    }

    /// Update suggestions cache
    async fn update_suggestions(&mut self, intent: &Intent) -> AssistantResult<()> {
        let suggestions = self.get_suggestions_for_context(intent).await?;
        let intent_key = format!("{:?}", intent);
        self.suggestion_cache.insert(intent_key, suggestions);
        Ok(())
    }

    /// Update patterns based on feedback
    async fn update_patterns_from_feedback(
        &mut self,
        input: &str,
        intent: &Intent,
        was_helpful: bool,
    ) -> AssistantResult<()> {
        if let Some(pattern) = self.patterns.iter_mut().find(|p| p.trigger == input) {
            if was_helpful {
                pattern.success_count += 1;
            } else {
                pattern.failure_count += 1;
            }

            // Recalculate confidence
            pattern.confidence = (pattern.success_count as f32)
                / ((pattern.success_count + pattern.failure_count) as f32);
        }

        Ok(())
    }

    /// Get confidence adjustment for an intent
    pub fn get_confidence_adjustment(&self, intent: &Intent) -> f32 {
        let intent_key = format!("{:?}", intent);
        self.confidence_adjustments
            .get(&intent_key)
            .copied()
            .unwrap_or(0.0)
    }

    /// Get learning statistics
    pub fn get_stats(&self) -> LearningStats {
        let total_feedback = self.feedback_history.len();
        let positive_feedback = self
            .feedback_history
            .iter()
            .filter(|f| f.was_helpful)
            .count();

        let success_rate = if total_feedback > 0 {
            positive_feedback as f32 / total_feedback as f32
        } else {
            0.0
        };

        LearningStats {
            total_patterns: self.patterns.len(),
            total_feedback: total_feedback,
            success_rate,
            enabled: self.enabled,
        }
    }

    /// Find similar patterns for better intent detection
    pub async fn find_similar_patterns(&self, input: &str) -> Vec<&Pattern> {
        let mut similar = Vec::new();

        for pattern in &self.patterns {
            if self.calculate_similarity(input, &pattern.trigger) > 0.7 {
                similar.push(pattern);
            }
        }

        // Sort by confidence
        similar.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        similar
    }

    /// Calculate similarity between two strings (simple word overlap)
    fn calculate_similarity(&self, a: &str, b: &str) -> f32 {
        let a_lower = a.to_lowercase();
        let b_lower = b.to_lowercase();
        let words_a: std::collections::HashSet<_> = a_lower.split_whitespace().collect();
        let words_b: std::collections::HashSet<_> = b_lower.split_whitespace().collect();

        let intersection = words_a.intersection(&words_b).count();
        let union = words_a.union(&words_b).count();

        if union == 0 {
            0.0
        } else {
            intersection as f32 / union as f32
        }
    }
}

/// Statistics about the learning system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningStats {
    pub total_patterns: usize,
    pub total_feedback: usize,
    pub success_rate: f32,
    pub enabled: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AssistantResponse;

    #[tokio::test]
    async fn test_learning_system_creation() {
        let system = LearningSystem::new(true).unwrap();
        assert!(system.enabled);
        assert_eq!(system.patterns.len(), 0);
    }

    #[tokio::test]
    async fn test_pattern_extraction() {
        let mut system = LearningSystem::new(true).unwrap();
        let intent = Intent::Deploy {
            target: "test-app".to_string(),
            source: None,
            config: HashMap::new(),
        };

        let parsed = ParsedQuery {
            intent: intent.clone(),
            confidence: 0.9,
            entities: HashMap::new(),
            context: Default::default(),
            raw_input: "deploy test-app".to_string(),
        };

        system
            .record_interaction("deploy test-app", &parsed, "Deploying...")
            .await
            .unwrap();

        assert_eq!(system.patterns.len(), 1);
        assert_eq!(system.patterns[0].trigger, "deploy test-app");
        assert_eq!(system.patterns[0].success_count, 1);
    }

    #[tokio::test]
    async fn test_feedback_learning() {
        let mut system = LearningSystem::new(true).unwrap();

        let response = AssistantResponse {
            intent: Intent::Deploy {
                target: "test-app".to_string(),
                source: None,
                config: HashMap::new(),
            },
            command: None,
            query_results: None,
            response: "Deployed successfully".to_string(),
            confidence: 0.8,
            suggestions: vec![],
        };

        system
            .update_from_feedback("deploy test-app", &response, true)
            .await
            .unwrap();

        assert_eq!(system.feedback_history.len(), 1);
        assert!(system.feedback_history[0].was_helpful);

        let intent_key = format!("{:?}", response.intent);
        assert!(system.confidence_adjustments.get(&intent_key).unwrap() > &0.0);
    }

    #[tokio::test]
    async fn test_suggestions_generation() {
        let system = LearningSystem::new(true).unwrap();

        let intent = Intent::Deploy {
            target: "test-app".to_string(),
            source: None,
            config: HashMap::new(),
        };

        let suggestions = system.get_suggestions_for_context(&intent).await.unwrap();
        assert!(!suggestions.is_empty());
        assert!(suggestions.iter().any(|s| s.contains("Scale")));
    }

    #[tokio::test]
    async fn test_pattern_similarity() {
        let system = LearningSystem::new(true).unwrap();

        let similarity = system.calculate_similarity("deploy my app", "deploy your app");
        assert!(similarity > 0.5);

        let similarity = system.calculate_similarity("deploy app", "scale service");
        assert!(similarity < 0.5);
    }

    #[tokio::test]
    async fn test_confidence_adjustment() {
        let mut system = LearningSystem::new(true).unwrap();

        let intent = Intent::Help { topic: None };
        let response = AssistantResponse {
            intent: intent.clone(),
            command: None,
            query_results: None,
            response: "Here's help".to_string(),
            confidence: 0.8,
            suggestions: vec![],
        };

        // Negative feedback should reduce confidence
        system
            .update_from_feedback("help", &response, false)
            .await
            .unwrap();
        let adjustment = system.get_confidence_adjustment(&intent);
        assert!(adjustment < 0.0);
    }

    #[tokio::test]
    async fn test_learning_stats() {
        let mut system = LearningSystem::new(true).unwrap();

        // Add some feedback
        let response = AssistantResponse {
            intent: Intent::Help { topic: None },
            command: None,
            query_results: None,
            response: "Help".to_string(),
            confidence: 0.8,
            suggestions: vec![],
        };

        system
            .update_from_feedback("help", &response, true)
            .await
            .unwrap();
        system
            .update_from_feedback("help me", &response, false)
            .await
            .unwrap();

        let stats = system.get_stats();
        assert_eq!(stats.total_feedback, 2);
        assert_eq!(stats.success_rate, 0.5);
        assert!(stats.enabled);
    }
}
