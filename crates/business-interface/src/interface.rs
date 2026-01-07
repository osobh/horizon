//! Main business interface for natural language goal submission and management

use crate::error::{BusinessError, BusinessResult};
use crate::explanation::{ExplainedResult, ResultExplainer};
use crate::goal::{BusinessGoal, GoalStatus};
use crate::llm_parser::LlmGoalParser;
use crate::progress::{GoalProgress, ProgressEvent, ProgressTracker};
use crate::safety::{SafetyValidationResult, SafetyValidator};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{broadcast, RwLock};
use tracing::{debug, info, warn};
use uuid::Uuid;

/// Main business interface for goal management
pub struct BusinessInterface {
    /// LLM-based goal parser
    goal_parser: Arc<LlmGoalParser>,
    /// Safety validation system
    safety_validator: Arc<SafetyValidator>,
    /// Progress tracking system
    progress_tracker: Arc<ProgressTracker>,
    /// Result explanation system
    result_explainer: Arc<ResultExplainer>,
    /// Active goals storage
    active_goals: DashMap<String, BusinessGoal>,
    /// Goal results storage
    goal_results: DashMap<String, GoalResult>,
    /// Interface configuration
    config: Arc<RwLock<InterfaceConfig>>,
    /// Event broadcaster
    event_sender: broadcast::Sender<InterfaceEvent>,
    /// Interface metrics
    metrics: Arc<InterfaceMetrics>,
}

/// Goal submission request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoalSubmissionRequest {
    /// Natural language goal description
    pub description: String,
    /// Submitter identifier
    pub submitted_by: String,
    /// Optional priority override
    pub priority_override: Option<crate::goal::GoalPriority>,
    /// Optional category override
    pub category_override: Option<crate::goal::GoalCategory>,
    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Goal submission response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoalSubmissionResponse {
    /// Generated goal ID
    pub goal_id: String,
    /// Parsed goal information
    pub parsed_goal: BusinessGoal,
    /// Safety validation result
    pub safety_validation: SafetyValidationResult,
    /// Submission status
    pub status: SubmissionStatus,
    /// Any warnings or recommendations
    pub messages: Vec<String>,
}

/// Goal execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoalResult {
    /// Goal ID
    pub goal_id: String,
    /// Execution status
    pub status: GoalStatus,
    /// Raw execution data
    pub execution_data: HashMap<String, serde_json::Value>,
    /// Explained results
    pub explanation: Option<ExplainedResult>,
    /// Result timestamp
    pub completed_at: chrono::DateTime<chrono::Utc>,
    /// Execution duration
    pub execution_duration: std::time::Duration,
    /// Success criteria met
    pub criteria_met: bool,
}

/// Interface configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterfaceConfig {
    /// Auto-approve goals below safety threshold
    pub auto_approve_threshold: f64,
    /// Maximum concurrent goals
    pub max_concurrent_goals: u32,
    /// Default goal timeout
    pub default_timeout_hours: f64,
    /// Enable automatic explanations
    pub auto_explain_results: bool,
    /// LLM model preferences
    pub llm_model: String,
    /// Safety validation strictness
    pub safety_strictness: SafetyStrictness,
}

/// Interface event for broadcasting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterfaceEvent {
    /// Event ID
    pub event_id: String,
    /// Event type
    pub event_type: InterfaceEventType,
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Associated goal ID
    pub goal_id: Option<String>,
    /// Event details
    pub details: String,
    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Interface metrics
///
/// Cache-line aligned (64 bytes) to prevent false sharing when
/// multiple threads update these counters concurrently.
#[repr(C, align(64))]
#[derive(Debug, Default)]
pub struct InterfaceMetrics {
    /// Total goals submitted
    pub total_submitted: std::sync::atomic::AtomicU64,
    /// Goals approved
    pub total_approved: std::sync::atomic::AtomicU64,
    /// Goals rejected
    pub total_rejected: std::sync::atomic::AtomicU64,
    /// Goals completed
    pub total_completed: std::sync::atomic::AtomicU64,
    /// Goals failed
    pub total_failed: std::sync::atomic::AtomicU64,
    /// Average processing time
    pub avg_processing_time: std::sync::atomic::AtomicU64,
    // Padding to fill cache line (6 * 8 = 48 bytes, need 16 more)
    _padding: [u8; 16],
}

// Enums
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SubmissionStatus {
    Accepted,
    Rejected,
    PendingApproval,
    SafetyReview,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum InterfaceEventType {
    GoalSubmitted,
    GoalApproved,
    GoalRejected,
    GoalStarted,
    GoalCompleted,
    GoalFailed,
    SafetyViolation,
    SystemError,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SafetyStrictness {
    Permissive,
    Balanced,
    Strict,
    Maximum,
}

impl BusinessInterface {
    /// Create a new business interface
    pub async fn new(api_key: Option<String>) -> BusinessResult<Self> {
        let goal_parser = Arc::new(LlmGoalParser::new(api_key.clone())?);
        let safety_validator = Arc::new(SafetyValidator::new());
        let progress_tracker = Arc::new(ProgressTracker::new());
        let result_explainer = Arc::new(ResultExplainer::new(api_key)?);

        let config = Arc::new(RwLock::new(InterfaceConfig::default()));
        let (event_sender, _) = broadcast::channel(1000);
        let metrics = Arc::new(InterfaceMetrics::default());

        Ok(Self {
            goal_parser,
            safety_validator,
            progress_tracker,
            result_explainer,
            active_goals: DashMap::new(),
            goal_results: DashMap::new(),
            config,
            event_sender,
            metrics,
        })
    }

    /// Submit a new business goal
    pub async fn submit_goal(
        &self,
        request: GoalSubmissionRequest,
    ) -> BusinessResult<GoalSubmissionResponse> {
        info!("Submitting goal: {}", request.description);

        self.metrics
            .total_submitted
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        // Parse the goal using LLM
        let mut parsed_goal = self
            .goal_parser
            .parse_goal(&request.description, &request.submitted_by)
            .await?;

        // Apply overrides if provided
        if let Some(priority) = request.priority_override {
            parsed_goal.priority = priority;
        }
        if let Some(category) = request.category_override {
            parsed_goal.category = category;
        }

        // Add request metadata
        for (key, value) in request.metadata {
            parsed_goal.metadata.insert(key, value);
        }

        // Validate safety
        let safety_validation = self.safety_validator.validate_goal(&parsed_goal)?;

        // Determine submission status
        let config = self.config.read().await;
        let status = if safety_validation.passed
            && safety_validation.safety_score >= config.auto_approve_threshold
        {
            SubmissionStatus::Accepted
        } else if safety_validation.passed {
            SubmissionStatus::PendingApproval
        } else {
            SubmissionStatus::Rejected
        };

        // Generate messages
        let mut messages = Vec::new();
        if !safety_validation.warnings.is_empty() {
            messages.extend(
                safety_validation
                    .warnings
                    .iter()
                    .map(|w| format!("Warning: {}", w)),
            );
        }
        if !safety_validation.errors.is_empty() {
            messages.extend(
                safety_validation
                    .errors
                    .iter()
                    .map(|e| format!("Error: {}", e)),
            );
        }
        if !safety_validation.mitigations.is_empty() {
            messages.push("Recommended mitigations:".to_string());
            messages.extend(safety_validation.mitigations.clone());
        }

        // Update metrics
        match status {
            SubmissionStatus::Accepted => {
                self.metrics
                    .total_approved
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                self.active_goals
                    .insert(parsed_goal.goal_id.clone(), parsed_goal.clone());
                self.progress_tracker.start_tracking(&parsed_goal)?;
            }
            SubmissionStatus::Rejected => {
                self.metrics
                    .total_rejected
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            _ => {}
        }

        // Broadcast event
        let event = InterfaceEvent {
            event_id: Uuid::new_v4().to_string(),
            event_type: match status {
                SubmissionStatus::Accepted => InterfaceEventType::GoalApproved,
                SubmissionStatus::Rejected => InterfaceEventType::GoalRejected,
                _ => InterfaceEventType::GoalSubmitted,
            },
            timestamp: chrono::Utc::now(),
            goal_id: Some(parsed_goal.goal_id.clone()),
            details: format!("Goal submitted: {}", request.description),
            metadata: HashMap::new(),
        };

        self.broadcast_event(event).await;

        let response = GoalSubmissionResponse {
            goal_id: parsed_goal.goal_id.clone(),
            parsed_goal,
            safety_validation,
            status,
            messages,
        };

        info!(
            "Goal submission completed: {} - {:?}",
            response.goal_id, response.status
        );
        Ok(response)
    }

    /// Approve a pending goal
    pub async fn approve_goal(&self, goal_id: &str) -> BusinessResult<()> {
        let goal =
            self.active_goals
                .get(goal_id)
                .ok_or_else(|| BusinessError::ProgressTrackingError {
                    goal_id: goal_id.to_string(),
                    operation: "Goal not found for approval".to_string(),
                })?;

        let mut goal = goal.clone();
        goal.update_status(GoalStatus::Approved);

        self.active_goals.insert(goal_id.to_string(), goal);
        self.progress_tracker
            .update_status(goal_id, GoalStatus::Approved)?;

        let event = InterfaceEvent {
            event_id: Uuid::new_v4().to_string(),
            event_type: InterfaceEventType::GoalApproved,
            timestamp: chrono::Utc::now(),
            goal_id: Some(goal_id.to_string()),
            details: "Goal manually approved".to_string(),
            metadata: HashMap::new(),
        };

        self.broadcast_event(event).await;

        info!("Goal approved: {}", goal_id);
        Ok(())
    }

    /// Reject a goal
    pub async fn reject_goal(&self, goal_id: &str, reason: &str) -> BusinessResult<()> {
        if let Some((_, mut goal)) = self.active_goals.remove(goal_id) {
            goal.update_status(GoalStatus::ValidationFailed {
                reason: reason.to_string(),
            });
            self.progress_tracker.stop_tracking(goal_id)?;
        }

        let event = InterfaceEvent {
            event_id: Uuid::new_v4().to_string(),
            event_type: InterfaceEventType::GoalRejected,
            timestamp: chrono::Utc::now(),
            goal_id: Some(goal_id.to_string()),
            details: format!("Goal rejected: {}", reason),
            metadata: HashMap::new(),
        };

        self.broadcast_event(event).await;

        info!("Goal rejected: {} - {}", goal_id, reason);
        Ok(())
    }

    /// Start goal execution
    pub async fn start_goal_execution(&self, goal_id: &str) -> BusinessResult<()> {
        let mut goal = self.active_goals.get_mut(goal_id).ok_or_else(|| {
            BusinessError::ProgressTrackingError {
                goal_id: goal_id.to_string(),
                operation: "Goal not found for execution start".to_string(),
            }
        })?;

        let executing_status = GoalStatus::Executing {
            started_at: chrono::Utc::now(),
        };
        goal.update_status(executing_status.clone());

        self.progress_tracker
            .update_status(goal_id, executing_status)?;

        let event = InterfaceEvent {
            event_id: Uuid::new_v4().to_string(),
            event_type: InterfaceEventType::GoalStarted,
            timestamp: chrono::Utc::now(),
            goal_id: Some(goal_id.to_string()),
            details: "Goal execution started".to_string(),
            metadata: HashMap::new(),
        };

        self.broadcast_event(event).await;

        info!("Goal execution started: {}", goal_id);
        Ok(())
    }

    /// Complete goal execution with results
    pub async fn complete_goal(
        &self,
        goal_id: &str,
        execution_data: HashMap<String, serde_json::Value>,
    ) -> BusinessResult<GoalResult> {
        let start_time = std::time::Instant::now();

        let goal =
            self.active_goals
                .get(goal_id)
                .ok_or_else(|| BusinessError::ProgressTrackingError {
                    goal_id: goal_id.to_string(),
                    operation: "Goal not found for completion".to_string(),
                })?;

        let goal = goal.clone();
        let completed_status = GoalStatus::Completed {
            completed_at: chrono::Utc::now(),
        };

        // Update goal status
        self.active_goals
            .get_mut(goal_id)
            .unwrap()
            .update_status(completed_status.clone());
        self.progress_tracker
            .update_status(goal_id, completed_status.clone())?;
        self.progress_tracker.update_progress(goal_id, 100.0)?;

        // Generate explanation if enabled
        let config = self.config.read().await;
        let explanation = if config.auto_explain_results {
            match self
                .result_explainer
                .explain_results(&goal, &execution_data)
                .await
            {
                Ok(explained) => Some(explained),
                Err(e) => {
                    warn!("Failed to generate explanation for goal {}: {}", goal_id, e);
                    None
                }
            }
        } else {
            None
        };

        // Check if success criteria were met
        let criteria_met = self.check_success_criteria(&goal, &execution_data);

        let execution_duration = start_time.elapsed();

        let result = GoalResult {
            goal_id: goal_id.to_string(),
            status: completed_status,
            execution_data,
            explanation,
            completed_at: chrono::Utc::now(),
            execution_duration,
            criteria_met,
        };

        // Store result
        self.goal_results
            .insert(goal_id.to_string(), result.clone());

        // Update metrics
        self.metrics
            .total_completed
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        // Broadcast event
        let event = InterfaceEvent {
            event_id: Uuid::new_v4().to_string(),
            event_type: InterfaceEventType::GoalCompleted,
            timestamp: chrono::Utc::now(),
            goal_id: Some(goal_id.to_string()),
            details: format!(
                "Goal completed with {} criteria met",
                if criteria_met { "all" } else { "some" }
            ),
            metadata: HashMap::new(),
        };

        self.broadcast_event(event).await;

        // Stop tracking
        self.progress_tracker.stop_tracking(goal_id)?;
        self.active_goals.remove(goal_id);

        info!(
            "Goal completed: {} - criteria met: {}",
            goal_id, criteria_met
        );
        Ok(result)
    }

    /// Mark goal as failed
    pub async fn fail_goal(
        &self,
        goal_id: &str,
        reason: &str,
        execution_data: Option<HashMap<String, serde_json::Value>>,
    ) -> BusinessResult<GoalResult> {
        let goal =
            self.active_goals
                .get(goal_id)
                .ok_or_else(|| BusinessError::ProgressTrackingError {
                    goal_id: goal_id.to_string(),
                    operation: "Goal not found for failure".to_string(),
                })?;

        let _goal = goal.clone();
        let failed_status = GoalStatus::Failed {
            reason: reason.to_string(),
            failed_at: chrono::Utc::now(),
        };

        // Update goal status
        self.active_goals
            .get_mut(goal_id)
            .unwrap()
            .update_status(failed_status.clone());
        self.progress_tracker
            .update_status(goal_id, failed_status)?;

        let result = GoalResult {
            goal_id: goal_id.to_string(),
            status: GoalStatus::Failed {
                reason: reason.to_string(),
                failed_at: chrono::Utc::now(),
            },
            execution_data: execution_data.unwrap_or_default(),
            explanation: None,
            completed_at: chrono::Utc::now(),
            execution_duration: std::time::Duration::from_secs(0),
            criteria_met: false,
        };

        // Store result
        self.goal_results
            .insert(goal_id.to_string(), result.clone());

        // Update metrics
        self.metrics
            .total_failed
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        // Broadcast event
        let event = InterfaceEvent {
            event_id: Uuid::new_v4().to_string(),
            event_type: InterfaceEventType::GoalFailed,
            timestamp: chrono::Utc::now(),
            goal_id: Some(goal_id.to_string()),
            details: format!("Goal failed: {}", reason),
            metadata: HashMap::new(),
        };

        self.broadcast_event(event).await;

        // Stop tracking
        self.progress_tracker.stop_tracking(goal_id)?;
        self.active_goals.remove(goal_id);

        warn!("Goal failed: {} - {}", goal_id, reason);
        Ok(result)
    }

    /// Get goal status
    pub fn get_goal_status(&self, goal_id: &str) -> Option<GoalStatus> {
        self.active_goals
            .get(goal_id)
            .map(|goal| goal.status.clone())
            .or_else(|| {
                self.goal_results
                    .get(goal_id)
                    .map(|result| result.status.clone())
            })
    }

    /// Get goal progress
    pub fn get_goal_progress(&self, goal_id: &str) -> Option<GoalProgress> {
        self.progress_tracker.get_progress(goal_id)
    }

    /// Get goal result
    pub fn get_goal_result(&self, goal_id: &str) -> Option<GoalResult> {
        self.goal_results.get(goal_id).map(|r| r.clone())
    }

    /// List active goals
    pub fn list_active_goals(&self) -> Vec<BusinessGoal> {
        self.active_goals
            .iter()
            .map(|entry| entry.value().clone())
            .collect()
    }

    /// Subscribe to interface events
    pub fn subscribe_to_events(&self) -> broadcast::Receiver<InterfaceEvent> {
        self.event_sender.subscribe()
    }

    /// Subscribe to progress events
    pub fn subscribe_to_progress(&self) -> broadcast::Receiver<ProgressEvent> {
        self.progress_tracker.subscribe()
    }

    /// Get interface metrics
    pub fn get_metrics(&self) -> InterfaceMetrics {
        InterfaceMetrics {
            total_submitted: std::sync::atomic::AtomicU64::new(
                self.metrics
                    .total_submitted
                    .load(std::sync::atomic::Ordering::Relaxed),
            ),
            total_approved: std::sync::atomic::AtomicU64::new(
                self.metrics
                    .total_approved
                    .load(std::sync::atomic::Ordering::Relaxed),
            ),
            total_rejected: std::sync::atomic::AtomicU64::new(
                self.metrics
                    .total_rejected
                    .load(std::sync::atomic::Ordering::Relaxed),
            ),
            total_completed: std::sync::atomic::AtomicU64::new(
                self.metrics
                    .total_completed
                    .load(std::sync::atomic::Ordering::Relaxed),
            ),
            total_failed: std::sync::atomic::AtomicU64::new(
                self.metrics
                    .total_failed
                    .load(std::sync::atomic::Ordering::Relaxed),
            ),
            avg_processing_time: std::sync::atomic::AtomicU64::new(
                self.metrics
                    .avg_processing_time
                    .load(std::sync::atomic::Ordering::Relaxed),
            ),
            _padding: [0u8; 16],
        }
    }

    /// Update interface configuration
    pub async fn update_config(&self, new_config: InterfaceConfig) -> BusinessResult<()> {
        let mut config = self.config.write().await;
        *config = new_config;

        info!("Interface configuration updated");
        Ok(())
    }

    /// Get current configuration
    pub async fn get_config(&self) -> InterfaceConfig {
        self.config.read().await.clone()
    }

    /// Check if success criteria were met
    fn check_success_criteria(
        &self,
        goal: &BusinessGoal,
        execution_data: &HashMap<String, serde_json::Value>,
    ) -> bool {
        for criterion in &goal.success_criteria {
            match criterion {
                crate::goal::Criterion::Accuracy { min_accuracy } => {
                    if let Some(accuracy) = execution_data.get("accuracy").and_then(|v| v.as_f64())
                    {
                        if accuracy < *min_accuracy {
                            return false;
                        }
                    } else {
                        return false;
                    }
                }
                crate::goal::Criterion::Completion { percentage } => {
                    if let Some(completion) = execution_data
                        .get("completion_percentage")
                        .and_then(|v| v.as_f64())
                    {
                        if completion < *percentage as f64 {
                            return false;
                        }
                    } else {
                        return false;
                    }
                }
                _ => {
                    // For other criteria types, assume they're met if not explicitly failed
                    // This can be extended with more specific logic
                }
            }
        }
        true
    }

    /// Broadcast interface event
    async fn broadcast_event(&self, event: InterfaceEvent) {
        if let Err(_) = self.event_sender.send(event.clone()) {
            debug!("No subscribers for interface event: {}", event.event_id);
        }

        debug!(
            "Broadcasted interface event: {:?} - {}",
            event.event_type, event.details
        );
    }
}

impl Default for InterfaceConfig {
    fn default() -> Self {
        Self {
            auto_approve_threshold: 0.8,
            max_concurrent_goals: 100,
            default_timeout_hours: 24.0,
            auto_explain_results: true,
            llm_model: "gpt-4".to_string(),
            safety_strictness: SafetyStrictness::Balanced,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::goal::{Criterion, GoalCategory, GoalPriority};

    async fn create_test_interface() -> BusinessInterface {
        BusinessInterface::new(Some("test-api-key".to_string()))
            .await
            .unwrap()
    }

    fn create_test_request() -> GoalSubmissionRequest {
        GoalSubmissionRequest {
            description: "Analyze customer data to find insights and patterns".to_string(),
            submitted_by: "test@example.com".to_string(),
            priority_override: None,
            category_override: None,
            metadata: HashMap::new(),
        }
    }

    #[tokio::test]
    async fn test_interface_creation() {
        let interface = create_test_interface().await;
        assert_eq!(interface.active_goals.len(), 0);
        assert_eq!(interface.goal_results.len(), 0);

        let config = interface.get_config().await;
        assert_eq!(config.auto_approve_threshold, 0.8);
        assert_eq!(config.max_concurrent_goals, 100);
    }

    #[tokio::test]
    async fn test_submit_goal() {
        let interface = create_test_interface().await;
        let request = create_test_request();

        let response = interface.submit_goal(request).await.unwrap();

        assert!(!response.goal_id.is_empty());
        assert_eq!(
            response.parsed_goal.description,
            "Analyze customer data to find insights and patterns"
        );
        assert_eq!(response.parsed_goal.submitted_by, "test@example.com");
        assert!(matches!(
            response.status,
            SubmissionStatus::Accepted | SubmissionStatus::PendingApproval
        ));

        let metrics = interface.get_metrics();
        assert_eq!(
            metrics
                .total_submitted
                .load(std::sync::atomic::Ordering::Relaxed),
            1
        );
    }

    #[tokio::test]
    async fn test_submit_goal_with_overrides() {
        let interface = create_test_interface().await;
        let mut request = create_test_request();
        request.priority_override = Some(GoalPriority::High);
        request.category_override = Some(GoalCategory::MachineLearning);
        request.metadata.insert(
            "custom_field".to_string(),
            serde_json::Value::String("test_value".to_string()),
        );

        let response = interface.submit_goal(request).await.unwrap();

        assert_eq!(response.parsed_goal.priority, GoalPriority::High);
        assert_eq!(response.parsed_goal.category, GoalCategory::MachineLearning);
        assert!(response.parsed_goal.metadata.contains_key("custom_field"));
    }

    #[tokio::test]
    async fn test_approve_goal() {
        let interface = create_test_interface().await;
        let request = create_test_request();

        let response = interface.submit_goal(request).await.unwrap();
        let goal_id = response.goal_id;

        assert!(interface.approve_goal(&goal_id).await.is_ok());

        let status = interface.get_goal_status(&goal_id);
        assert_eq!(status, Some(GoalStatus::Approved));
    }

    #[tokio::test]
    async fn test_reject_goal() {
        let interface = create_test_interface().await;
        let request = create_test_request();

        let response = interface.submit_goal(request).await.unwrap();
        let goal_id = response.goal_id;

        assert!(interface
            .reject_goal(&goal_id, "Safety concerns")
            .await
            .is_ok());

        let status = interface.get_goal_status(&goal_id);
        assert!(matches!(status, Some(GoalStatus::ValidationFailed { .. })));

        let metrics = interface.get_metrics();
        assert_eq!(
            metrics
                .total_rejected
                .load(std::sync::atomic::Ordering::Relaxed),
            1
        );
    }

    #[tokio::test]
    async fn test_start_goal_execution() {
        let interface = create_test_interface().await;
        let request = create_test_request();

        let response = interface.submit_goal(request).await.unwrap();
        let goal_id = response.goal_id;

        assert!(interface.start_goal_execution(&goal_id).await.is_ok());

        let status = interface.get_goal_status(&goal_id);
        assert!(matches!(status, Some(GoalStatus::Executing { .. })));
    }

    #[tokio::test]
    async fn test_complete_goal() {
        let interface = create_test_interface().await;
        let request = create_test_request();

        let response = interface.submit_goal(request).await.unwrap();
        let goal_id = response.goal_id;

        interface.start_goal_execution(&goal_id).await.unwrap();

        let mut execution_data = HashMap::new();
        execution_data.insert(
            "accuracy".to_string(),
            serde_json::Value::Number(serde_json::Number::from_f64(0.95).unwrap()),
        );
        execution_data.insert(
            "completion_percentage".to_string(),
            serde_json::Value::Number(serde_json::Number::from_f64(100.0).unwrap()),
        );

        let result = interface
            .complete_goal(&goal_id, execution_data)
            .await
            .unwrap();

        assert_eq!(result.goal_id, goal_id);
        assert!(matches!(result.status, GoalStatus::Completed { .. }));
        assert!(result.execution_duration > std::time::Duration::from_nanos(0));

        let metrics = interface.get_metrics();
        assert_eq!(
            metrics
                .total_completed
                .load(std::sync::atomic::Ordering::Relaxed),
            1
        );

        // Goal should be removed from active goals
        assert!(interface.get_goal_status(&goal_id).is_some()); // Still available in results
        assert_eq!(interface.list_active_goals().len(), 0);
    }

    #[tokio::test]
    async fn test_fail_goal() {
        let interface = create_test_interface().await;
        let request = create_test_request();

        let response = interface.submit_goal(request).await.unwrap();
        let goal_id = response.goal_id;

        interface.start_goal_execution(&goal_id).await.unwrap();

        let result = interface
            .fail_goal(&goal_id, "Insufficient resources", None)
            .await
            .unwrap();

        assert_eq!(result.goal_id, goal_id);
        assert!(matches!(result.status, GoalStatus::Failed { .. }));
        assert!(!result.criteria_met);

        let metrics = interface.get_metrics();
        assert_eq!(
            metrics
                .total_failed
                .load(std::sync::atomic::Ordering::Relaxed),
            1
        );
    }

    #[tokio::test]
    async fn test_get_goal_progress() {
        let interface = create_test_interface().await;
        let request = create_test_request();

        let response = interface.submit_goal(request).await.unwrap();
        let goal_id = response.goal_id;

        let progress = interface.get_goal_progress(&goal_id);
        assert!(progress.is_some());

        let progress = progress.unwrap();
        assert_eq!(progress.goal_id, goal_id);
        assert_eq!(progress.percentage, 0.0);
    }

    #[tokio::test]
    async fn test_list_active_goals() {
        let interface = create_test_interface().await;

        // Submit multiple goals
        for i in 0..3 {
            let mut request = create_test_request();
            request.description = format!("Goal {}", i);
            interface.submit_goal(request).await.unwrap();
        }

        let active_goals = interface.list_active_goals();
        assert_eq!(active_goals.len(), 3);
    }

    #[tokio::test]
    async fn test_event_subscription() {
        let interface = create_test_interface().await;
        let mut receiver = interface.subscribe_to_events();

        let request = create_test_request();
        interface.submit_goal(request).await.unwrap();

        // Should receive goal submission event
        let event = receiver.try_recv().unwrap();
        assert!(matches!(
            event.event_type,
            InterfaceEventType::GoalApproved | InterfaceEventType::GoalSubmitted
        ));
        assert!(event.goal_id.is_some());
    }

    #[tokio::test]
    async fn test_progress_subscription() {
        let interface = create_test_interface().await;
        let mut receiver = interface.subscribe_to_progress();

        let request = create_test_request();
        let response = interface.submit_goal(request).await.unwrap();

        // Should receive progress tracking started event
        let event = receiver.try_recv().unwrap();
        assert_eq!(event.goal_id, response.goal_id);
    }

    #[tokio::test]
    async fn test_update_config() {
        let interface = create_test_interface().await;

        let mut new_config = InterfaceConfig::default();
        new_config.auto_approve_threshold = 0.9;
        new_config.max_concurrent_goals = 50;
        new_config.llm_model = "gpt-3.5-turbo".to_string();

        assert!(interface.update_config(new_config.clone()).await.is_ok());

        let config = interface.get_config().await;
        assert_eq!(config.auto_approve_threshold, 0.9);
        assert_eq!(config.max_concurrent_goals, 50);
        assert_eq!(config.llm_model, "gpt-3.5-turbo");
    }

    #[tokio::test]
    async fn test_check_success_criteria() {
        let interface = create_test_interface().await;

        let mut goal = BusinessGoal::new("Test goal".to_string(), "test@example.com".to_string());
        goal.add_criterion(Criterion::Accuracy { min_accuracy: 0.9 });
        goal.add_criterion(Criterion::Completion { percentage: 95.0 });

        // Test with criteria met
        let mut execution_data = HashMap::new();
        execution_data.insert(
            "accuracy".to_string(),
            serde_json::Value::Number(serde_json::Number::from_f64(0.95).unwrap()),
        );
        execution_data.insert(
            "completion_percentage".to_string(),
            serde_json::Value::Number(serde_json::Number::from_f64(98.0).unwrap()),
        );

        assert!(interface.check_success_criteria(&goal, &execution_data));

        // Test with criteria not met
        execution_data.insert(
            "accuracy".to_string(),
            serde_json::Value::Number(serde_json::Number::from_f64(0.85).unwrap()),
        );
        assert!(!interface.check_success_criteria(&goal, &execution_data));
    }

    #[test]
    fn test_submission_status_serialization() {
        let statuses = vec![
            SubmissionStatus::Accepted,
            SubmissionStatus::Rejected,
            SubmissionStatus::PendingApproval,
            SubmissionStatus::SafetyReview,
        ];

        for status in statuses {
            let serialized = serde_json::to_string(&status).unwrap();
            let deserialized: SubmissionStatus = serde_json::from_str(&serialized).unwrap();
            assert_eq!(status, deserialized);
        }
    }

    #[test]
    fn test_interface_event_type_serialization() {
        let event_types = vec![
            InterfaceEventType::GoalSubmitted,
            InterfaceEventType::GoalApproved,
            InterfaceEventType::GoalRejected,
            InterfaceEventType::GoalStarted,
            InterfaceEventType::GoalCompleted,
            InterfaceEventType::GoalFailed,
            InterfaceEventType::SafetyViolation,
            InterfaceEventType::SystemError,
        ];

        for event_type in event_types {
            let serialized = serde_json::to_string(&event_type).unwrap();
            let deserialized: InterfaceEventType = serde_json::from_str(&serialized).unwrap();
            assert_eq!(event_type, deserialized);
        }
    }

    #[test]
    fn test_safety_strictness_serialization() {
        let strictness_levels = vec![
            SafetyStrictness::Permissive,
            SafetyStrictness::Balanced,
            SafetyStrictness::Strict,
            SafetyStrictness::Maximum,
        ];

        for strictness in strictness_levels {
            let serialized = serde_json::to_string(&strictness).unwrap();
            let deserialized: SafetyStrictness = serde_json::from_str(&serialized).unwrap();
            assert_eq!(strictness, deserialized);
        }
    }

    #[test]
    fn test_interface_config_default() {
        let config = InterfaceConfig::default();
        assert_eq!(config.auto_approve_threshold, 0.8);
        assert_eq!(config.max_concurrent_goals, 100);
        assert_eq!(config.default_timeout_hours, 24.0);
        assert!(config.auto_explain_results);
        assert_eq!(config.llm_model, "gpt-4");
        assert_eq!(config.safety_strictness, SafetyStrictness::Balanced);
    }

    #[test]
    fn test_goal_submission_request_serialization() {
        let mut metadata = HashMap::new();
        metadata.insert(
            "custom".to_string(),
            serde_json::Value::String("value".to_string()),
        );

        let request = GoalSubmissionRequest {
            description: "Test goal".to_string(),
            submitted_by: "user@test.com".to_string(),
            priority_override: Some(GoalPriority::High),
            category_override: Some(GoalCategory::DataAnalysis),
            metadata,
        };

        let serialized = serde_json::to_string(&request).unwrap();
        let deserialized: GoalSubmissionRequest = serde_json::from_str(&serialized).unwrap();

        assert_eq!(request.description, deserialized.description);
        assert_eq!(request.submitted_by, deserialized.submitted_by);
        assert_eq!(request.priority_override, deserialized.priority_override);
        assert_eq!(request.category_override, deserialized.category_override);
        assert_eq!(request.metadata.len(), deserialized.metadata.len());
    }

    #[test]
    fn test_goal_result_serialization() {
        let mut execution_data = HashMap::new();
        execution_data.insert(
            "result".to_string(),
            serde_json::Value::String("success".to_string()),
        );

        let result = GoalResult {
            goal_id: "test-goal".to_string(),
            status: GoalStatus::Completed {
                completed_at: chrono::Utc::now(),
            },
            execution_data,
            explanation: None,
            completed_at: chrono::Utc::now(),
            execution_duration: std::time::Duration::from_secs(300),
            criteria_met: true,
        };

        let serialized = serde_json::to_string(&result).unwrap();
        let deserialized: GoalResult = serde_json::from_str(&serialized).unwrap();

        assert_eq!(result.goal_id, deserialized.goal_id);
        assert_eq!(result.criteria_met, deserialized.criteria_met);
        assert_eq!(
            result.execution_data.len(),
            deserialized.execution_data.len()
        );
    }
}
