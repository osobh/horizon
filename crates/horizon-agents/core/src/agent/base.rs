use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

use crate::error::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AutonomyLevel {
    ReadOnly, // Observe only
    Low,      // Recommendations only
    Medium,   // Low-risk execution
    High,     // Auto-execute with thresholds
    Full,     // Full autonomy
}

impl AutonomyLevel {
    pub fn can_execute(&self) -> bool {
        matches!(self, Self::Medium | Self::High | Self::Full)
    }

    pub fn requires_approval(&self) -> bool {
        matches!(self, Self::ReadOnly | Self::Low)
    }

    pub fn can_transition_to(&self, target: AutonomyLevel) -> bool {
        use AutonomyLevel::*;
        match (self, target) {
            // Can always stay at same level
            (a, b) if a == &b => true,
            // Can always decrease autonomy
            (Low, ReadOnly) => true,
            (Medium, ReadOnly | Low) => true,
            (High, ReadOnly | Low | Medium) => true,
            (Full, ReadOnly | Low | Medium | High) => true,
            // Can increase by one level at a time
            (ReadOnly, Low) => true,
            (Low, Medium) => true,
            (Medium, High) => true,
            (High, Full) => true,
            // Cannot skip levels when increasing
            _ => false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentRequest {
    pub id: Uuid,
    pub content: String,
    pub metadata: HashMap<String, String>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl AgentRequest {
    pub fn new(content: String) -> Self {
        Self {
            id: Uuid::new_v4(),
            content,
            metadata: HashMap::new(),
            timestamp: chrono::Utc::now(),
        }
    }

    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentResponse {
    pub request_id: Uuid,
    pub content: String,
    pub actions_taken: Vec<String>,
    pub recommendations: Vec<String>,
    pub metadata: HashMap<String, String>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl AgentResponse {
    pub fn new(request_id: Uuid, content: String) -> Self {
        Self {
            request_id,
            content,
            actions_taken: Vec::new(),
            recommendations: Vec::new(),
            metadata: HashMap::new(),
            timestamp: chrono::Utc::now(),
        }
    }

    pub fn with_action(mut self, action: String) -> Self {
        self.actions_taken.push(action);
        self
    }

    pub fn with_recommendation(mut self, recommendation: String) -> Self {
        self.recommendations.push(recommendation);
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HealthStatus {
    Healthy,
    Degraded { reason: String },
    Unhealthy { reason: String },
}

impl HealthStatus {
    pub fn is_healthy(&self) -> bool {
        matches!(self, Self::Healthy)
    }

    pub fn is_degraded(&self) -> bool {
        matches!(self, Self::Degraded { .. })
    }

    pub fn is_unhealthy(&self) -> bool {
        matches!(self, Self::Unhealthy { .. })
    }
}

#[async_trait]
pub trait Agent: Send + Sync {
    /// Get the agent's name
    fn name(&self) -> &str;

    /// Initialize the agent
    async fn init(&mut self) -> Result<()>;

    /// Process a request
    async fn process(&self, request: AgentRequest) -> Result<AgentResponse>;

    /// Check agent health
    async fn health(&self) -> Result<HealthStatus>;

    /// Shutdown the agent
    async fn shutdown(&mut self) -> Result<()>;

    /// Get current autonomy level
    fn autonomy_level(&self) -> AutonomyLevel;

    /// Set autonomy level
    fn set_autonomy_level(&mut self, level: AutonomyLevel) -> Result<()>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_autonomy_level_can_execute() {
        assert!(!AutonomyLevel::ReadOnly.can_execute());
        assert!(!AutonomyLevel::Low.can_execute());
        assert!(AutonomyLevel::Medium.can_execute());
        assert!(AutonomyLevel::High.can_execute());
        assert!(AutonomyLevel::Full.can_execute());
    }

    #[test]
    fn test_autonomy_level_requires_approval() {
        assert!(AutonomyLevel::ReadOnly.requires_approval());
        assert!(AutonomyLevel::Low.requires_approval());
        assert!(!AutonomyLevel::Medium.requires_approval());
        assert!(!AutonomyLevel::High.requires_approval());
        assert!(!AutonomyLevel::Full.requires_approval());
    }

    #[test]
    fn test_autonomy_level_transitions() {
        // Can stay at same level
        assert!(AutonomyLevel::Low.can_transition_to(AutonomyLevel::Low));

        // Can decrease
        assert!(AutonomyLevel::Full.can_transition_to(AutonomyLevel::ReadOnly));
        assert!(AutonomyLevel::High.can_transition_to(AutonomyLevel::Medium));

        // Can increase by one level
        assert!(AutonomyLevel::ReadOnly.can_transition_to(AutonomyLevel::Low));
        assert!(AutonomyLevel::Low.can_transition_to(AutonomyLevel::Medium));
        assert!(AutonomyLevel::Medium.can_transition_to(AutonomyLevel::High));
        assert!(AutonomyLevel::High.can_transition_to(AutonomyLevel::Full));

        // Cannot skip levels
        assert!(!AutonomyLevel::ReadOnly.can_transition_to(AutonomyLevel::Medium));
        assert!(!AutonomyLevel::Low.can_transition_to(AutonomyLevel::High));
        assert!(!AutonomyLevel::ReadOnly.can_transition_to(AutonomyLevel::Full));
    }

    #[test]
    fn test_agent_request_creation() {
        let req = AgentRequest::new("test content".to_string());
        assert_eq!(req.content, "test content");
        assert!(req.metadata.is_empty());
    }

    #[test]
    fn test_agent_request_with_metadata() {
        let req = AgentRequest::new("test".to_string())
            .with_metadata("key1".to_string(), "value1".to_string())
            .with_metadata("key2".to_string(), "value2".to_string());

        assert_eq!(req.metadata.len(), 2);
        assert_eq!(req.metadata.get("key1").unwrap(), "value1");
    }

    #[test]
    fn test_agent_response_creation() {
        let req_id = Uuid::new_v4();
        let resp = AgentResponse::new(req_id, "response".to_string());
        assert_eq!(resp.request_id, req_id);
        assert_eq!(resp.content, "response");
        assert!(resp.actions_taken.is_empty());
        assert!(resp.recommendations.is_empty());
    }

    #[test]
    fn test_agent_response_with_actions() {
        let req_id = Uuid::new_v4();
        let resp = AgentResponse::new(req_id, "response".to_string())
            .with_action("action1".to_string())
            .with_action("action2".to_string());

        assert_eq!(resp.actions_taken.len(), 2);
        assert_eq!(resp.actions_taken[0], "action1");
    }

    #[test]
    fn test_agent_response_with_recommendations() {
        let req_id = Uuid::new_v4();
        let resp = AgentResponse::new(req_id, "response".to_string())
            .with_recommendation("rec1".to_string())
            .with_recommendation("rec2".to_string());

        assert_eq!(resp.recommendations.len(), 2);
        assert_eq!(resp.recommendations[0], "rec1");
    }

    #[test]
    fn test_health_status_checks() {
        let healthy = HealthStatus::Healthy;
        assert!(healthy.is_healthy());
        assert!(!healthy.is_degraded());
        assert!(!healthy.is_unhealthy());

        let degraded = HealthStatus::Degraded {
            reason: "slow".to_string(),
        };
        assert!(!degraded.is_healthy());
        assert!(degraded.is_degraded());
        assert!(!degraded.is_unhealthy());

        let unhealthy = HealthStatus::Unhealthy {
            reason: "down".to_string(),
        };
        assert!(!unhealthy.is_healthy());
        assert!(!unhealthy.is_degraded());
        assert!(unhealthy.is_unhealthy());
    }
}
