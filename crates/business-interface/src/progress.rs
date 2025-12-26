//! Progress tracking for business goals

use crate::error::{BusinessError, BusinessResult};
use crate::goal::{BusinessGoal, GoalStatus};
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::broadcast;
use tracing::{debug, info, warn};
use uuid::Uuid;

/// Progress tracking system for business goals
pub struct ProgressTracker {
    /// Active goal progress data
    goal_progress: DashMap<String, GoalProgress>,
    /// Progress event broadcaster
    event_sender: broadcast::Sender<ProgressEvent>,
    /// Progress metrics
    metrics: Arc<ProgressMetrics>,
}

/// Progress information for a business goal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoalProgress {
    /// Goal ID
    pub goal_id: String,
    /// Current execution percentage (0-100)
    pub percentage: f32,
    /// Current status
    pub status: GoalStatus,
    /// Progress milestones
    pub milestones: Vec<ProgressMilestone>,
    /// Resource usage metrics
    pub resource_usage: ResourceUsage,
    /// Progress history
    pub history: Vec<ProgressEvent>,
    /// Last update timestamp
    pub last_updated: DateTime<Utc>,
    /// Estimated completion time
    pub estimated_completion: Option<DateTime<Utc>>,
    /// Agent assignments
    pub assigned_agents: Vec<String>,
}

/// Progress milestone
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProgressMilestone {
    /// Milestone ID
    pub milestone_id: String,
    /// Milestone name
    pub name: String,
    /// Description
    pub description: String,
    /// Target percentage for this milestone
    pub target_percentage: f32,
    /// Whether this milestone is completed
    pub completed: bool,
    /// Completion timestamp
    pub completed_at: Option<DateTime<Utc>>,
    /// Dependencies on other milestones
    pub dependencies: Vec<String>,
}

/// Resource usage tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    /// Current GPU memory usage in MB
    pub gpu_memory_mb: u32,
    /// Current CPU usage percentage
    pub cpu_usage_percent: f32,
    /// Current system memory usage in MB
    pub memory_mb: u32,
    /// Current storage usage in MB
    pub storage_mb: u32,
    /// Current cost accrued in USD
    pub cost_usd: f64,
    /// Number of active agents
    pub active_agents: u32,
    /// Network bandwidth usage in Mbps
    pub network_mbps: f32,
    /// Last measurement timestamp
    pub measured_at: DateTime<Utc>,
}

/// Progress event for broadcasting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressEvent {
    /// Event ID
    pub event_id: String,
    /// Goal ID
    pub goal_id: String,
    /// Event type
    pub event_type: ProgressEventType,
    /// Event timestamp
    pub timestamp: DateTime<Utc>,
    /// Previous progress percentage
    pub previous_percentage: f32,
    /// New progress percentage
    pub new_percentage: f32,
    /// Event details
    pub details: String,
    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Types of progress events
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ProgressEventType {
    /// Goal execution started
    Started,
    /// Progress updated
    ProgressUpdate,
    /// Milestone completed
    MilestoneCompleted,
    /// Status changed
    StatusChanged,
    /// Resource usage updated
    ResourceUpdate,
    /// Goal completed
    Completed,
    /// Goal failed
    Failed,
    /// Goal paused
    Paused,
    /// Goal resumed
    Resumed,
    /// Agent assigned
    AgentAssigned,
    /// Agent released
    AgentReleased,
}

/// Progress tracking metrics
#[derive(Debug, Default)]
pub struct ProgressMetrics {
    /// Total goals tracked
    pub total_goals: std::sync::atomic::AtomicU64,
    /// Active goals
    pub active_goals: std::sync::atomic::AtomicU64,
    /// Completed goals
    pub completed_goals: std::sync::atomic::AtomicU64,
    /// Failed goals
    pub failed_goals: std::sync::atomic::AtomicU64,
    /// Total progress events
    pub total_events: std::sync::atomic::AtomicU64,
}

impl ProgressTracker {
    /// Create a new progress tracker
    pub fn new() -> Self {
        let (event_sender, _) = broadcast::channel(1000);

        Self {
            goal_progress: DashMap::new(),
            event_sender,
            metrics: Arc::new(ProgressMetrics::default()),
        }
    }

    /// Start tracking a goal
    pub fn start_tracking(&self, goal: &BusinessGoal) -> BusinessResult<()> {
        debug!("Starting tracking for goal: {}", goal.goal_id);

        let progress = GoalProgress {
            goal_id: goal.goal_id.clone(),
            percentage: goal.progress,
            status: goal.status.clone(),
            milestones: self.create_default_milestones(&goal.goal_id),
            resource_usage: ResourceUsage::default(),
            history: Vec::new(),
            last_updated: Utc::now(),
            estimated_completion: self.estimate_completion(goal),
            assigned_agents: Vec::new(),
        };

        self.goal_progress.insert(goal.goal_id.clone(), progress);

        let event = ProgressEvent {
            event_id: Uuid::new_v4().to_string(),
            goal_id: goal.goal_id.clone(),
            event_type: ProgressEventType::Started,
            timestamp: Utc::now(),
            previous_percentage: 0.0,
            new_percentage: goal.progress,
            details: "Goal tracking started".to_string(),
            metadata: HashMap::new(),
        };

        self.broadcast_event(event)?;

        self.metrics
            .total_goals
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.metrics
            .active_goals
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        info!("Started tracking goal: {}", goal.goal_id);
        Ok(())
    }

    /// Update goal progress
    pub fn update_progress(&self, goal_id: &str, new_percentage: f32) -> BusinessResult<()> {
        if !(0.0..=100.0).contains(&new_percentage) {
            return Err(BusinessError::ProgressTrackingError {
                goal_id: goal_id.to_string(),
                operation: format!("Invalid progress percentage: {}", new_percentage),
            });
        }

        let mut progress = self.goal_progress.get_mut(goal_id).ok_or_else(|| {
            BusinessError::ProgressTrackingError {
                goal_id: goal_id.to_string(),
                operation: "Goal not found for progress update".to_string(),
            }
        })?;

        let previous_percentage = progress.percentage;
        progress.percentage = new_percentage;
        progress.last_updated = Utc::now();

        // Check for completed milestones
        for milestone in &mut progress.milestones {
            if !milestone.completed && new_percentage >= milestone.target_percentage {
                milestone.completed = true;
                milestone.completed_at = Some(Utc::now());

                let milestone_event = ProgressEvent {
                    event_id: Uuid::new_v4().to_string(),
                    goal_id: goal_id.to_string(),
                    event_type: ProgressEventType::MilestoneCompleted,
                    timestamp: Utc::now(),
                    previous_percentage,
                    new_percentage,
                    details: format!("Milestone '{}' completed", milestone.name),
                    metadata: {
                        let mut meta = HashMap::new();
                        meta.insert(
                            "milestone_id".to_string(),
                            serde_json::Value::String(milestone.milestone_id.clone()),
                        );
                        meta
                    },
                };

                self.broadcast_event(milestone_event)?;
            }
        }

        let event = ProgressEvent {
            event_id: Uuid::new_v4().to_string(),
            goal_id: goal_id.to_string(),
            event_type: ProgressEventType::ProgressUpdate,
            timestamp: Utc::now(),
            previous_percentage,
            new_percentage,
            details: format!(
                "Progress updated from {:.1}% to {:.1}%",
                previous_percentage, new_percentage
            ),
            metadata: HashMap::new(),
        };

        progress.history.push(event.clone());
        self.broadcast_event(event)?;

        debug!(
            "Updated progress for goal {}: {:.1}%",
            goal_id, new_percentage
        );
        Ok(())
    }

    /// Update goal status
    pub fn update_status(&self, goal_id: &str, new_status: GoalStatus) -> BusinessResult<()> {
        let mut progress = self.goal_progress.get_mut(goal_id).ok_or_else(|| {
            BusinessError::ProgressTrackingError {
                goal_id: goal_id.to_string(),
                operation: "Goal not found for status update".to_string(),
            }
        })?;

        let old_status = progress.status.clone();
        progress.status = new_status.clone();
        progress.last_updated = Utc::now();

        // Update metrics based on status change
        match (&old_status, &new_status) {
            (_, GoalStatus::Completed { .. }) => {
                self.metrics
                    .active_goals
                    .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
                self.metrics
                    .completed_goals
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                progress.percentage = 100.0;
            }
            (_, GoalStatus::Failed { .. }) => {
                self.metrics
                    .active_goals
                    .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
                self.metrics
                    .failed_goals
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            _ => {}
        }

        let event = ProgressEvent {
            event_id: Uuid::new_v4().to_string(),
            goal_id: goal_id.to_string(),
            event_type: ProgressEventType::StatusChanged,
            timestamp: Utc::now(),
            previous_percentage: progress.percentage,
            new_percentage: progress.percentage,
            details: format!("Status changed from {:?} to {:?}", old_status, new_status),
            metadata: HashMap::new(),
        };

        progress.history.push(event.clone());
        self.broadcast_event(event)?;

        info!("Updated status for goal {}: {:?}", goal_id, new_status);
        Ok(())
    }

    /// Update resource usage
    pub fn update_resource_usage(&self, goal_id: &str, usage: ResourceUsage) -> BusinessResult<()> {
        let mut progress = self.goal_progress.get_mut(goal_id).ok_or_else(|| {
            BusinessError::ProgressTrackingError {
                goal_id: goal_id.to_string(),
                operation: "Goal not found for resource update".to_string(),
            }
        })?;

        progress.resource_usage = usage;
        progress.last_updated = Utc::now();

        let event = ProgressEvent {
            event_id: Uuid::new_v4().to_string(),
            goal_id: goal_id.to_string(),
            event_type: ProgressEventType::ResourceUpdate,
            timestamp: Utc::now(),
            previous_percentage: progress.percentage,
            new_percentage: progress.percentage,
            details: "Resource usage updated".to_string(),
            metadata: HashMap::new(),
        };

        progress.history.push(event.clone());
        self.broadcast_event(event)?;

        debug!("Updated resource usage for goal: {}", goal_id);
        Ok(())
    }

    /// Assign agent to goal
    pub fn assign_agent(&self, goal_id: &str, agent_id: &str) -> BusinessResult<()> {
        let mut progress = self.goal_progress.get_mut(goal_id).ok_or_else(|| {
            BusinessError::ProgressTrackingError {
                goal_id: goal_id.to_string(),
                operation: "Goal not found for agent assignment".to_string(),
            }
        })?;

        if !progress.assigned_agents.contains(&agent_id.to_string()) {
            progress.assigned_agents.push(agent_id.to_string());
            progress.last_updated = Utc::now();

            let event = ProgressEvent {
                event_id: Uuid::new_v4().to_string(),
                goal_id: goal_id.to_string(),
                event_type: ProgressEventType::AgentAssigned,
                timestamp: Utc::now(),
                previous_percentage: progress.percentage,
                new_percentage: progress.percentage,
                details: format!("Agent {} assigned", agent_id),
                metadata: {
                    let mut meta = HashMap::new();
                    meta.insert(
                        "agent_id".to_string(),
                        serde_json::Value::String(agent_id.to_string()),
                    );
                    meta
                },
            };

            progress.history.push(event.clone());
            self.broadcast_event(event)?;

            info!("Assigned agent {} to goal {}", agent_id, goal_id);
        }

        Ok(())
    }

    /// Release agent from goal
    pub fn release_agent(&self, goal_id: &str, agent_id: &str) -> BusinessResult<()> {
        let mut progress = self.goal_progress.get_mut(goal_id).ok_or_else(|| {
            BusinessError::ProgressTrackingError {
                goal_id: goal_id.to_string(),
                operation: "Goal not found for agent release".to_string(),
            }
        })?;

        if let Some(pos) = progress.assigned_agents.iter().position(|x| x == agent_id) {
            progress.assigned_agents.remove(pos);
            progress.last_updated = Utc::now();

            let event = ProgressEvent {
                event_id: Uuid::new_v4().to_string(),
                goal_id: goal_id.to_string(),
                event_type: ProgressEventType::AgentReleased,
                timestamp: Utc::now(),
                previous_percentage: progress.percentage,
                new_percentage: progress.percentage,
                details: format!("Agent {} released", agent_id),
                metadata: {
                    let mut meta = HashMap::new();
                    meta.insert(
                        "agent_id".to_string(),
                        serde_json::Value::String(agent_id.to_string()),
                    );
                    meta
                },
            };

            progress.history.push(event.clone());
            self.broadcast_event(event)?;

            info!("Released agent {} from goal {}", agent_id, goal_id);
        }

        Ok(())
    }

    /// Get goal progress
    pub fn get_progress(&self, goal_id: &str) -> Option<GoalProgress> {
        self.goal_progress.get(goal_id).map(|p| p.clone())
    }

    /// Get all active goals
    pub fn get_active_goals(&self) -> Vec<GoalProgress> {
        self.goal_progress
            .iter()
            .filter(|p| {
                matches!(
                    p.status,
                    GoalStatus::Executing { .. } | GoalStatus::Paused { .. }
                )
            })
            .map(|p| p.clone())
            .collect()
    }

    /// Subscribe to progress events
    pub fn subscribe(&self) -> broadcast::Receiver<ProgressEvent> {
        self.event_sender.subscribe()
    }

    /// Get tracking metrics
    pub fn get_metrics(&self) -> ProgressMetrics {
        ProgressMetrics {
            total_goals: std::sync::atomic::AtomicU64::new(
                self.metrics
                    .total_goals
                    .load(std::sync::atomic::Ordering::Relaxed),
            ),
            active_goals: std::sync::atomic::AtomicU64::new(
                self.metrics
                    .active_goals
                    .load(std::sync::atomic::Ordering::Relaxed),
            ),
            completed_goals: std::sync::atomic::AtomicU64::new(
                self.metrics
                    .completed_goals
                    .load(std::sync::atomic::Ordering::Relaxed),
            ),
            failed_goals: std::sync::atomic::AtomicU64::new(
                self.metrics
                    .failed_goals
                    .load(std::sync::atomic::Ordering::Relaxed),
            ),
            total_events: std::sync::atomic::AtomicU64::new(
                self.metrics
                    .total_events
                    .load(std::sync::atomic::Ordering::Relaxed),
            ),
        }
    }

    /// Stop tracking a goal
    pub fn stop_tracking(&self, goal_id: &str) -> BusinessResult<()> {
        if let Some((_, progress)) = self.goal_progress.remove(goal_id) {
            if matches!(
                progress.status,
                GoalStatus::Executing { .. } | GoalStatus::Paused { .. }
            ) {
                self.metrics
                    .active_goals
                    .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
            }

            info!("Stopped tracking goal: {}", goal_id);
        } else {
            warn!("Attempted to stop tracking non-existent goal: {}", goal_id);
        }

        Ok(())
    }

    /// Create default milestones for a goal
    fn create_default_milestones(&self, goal_id: &str) -> Vec<ProgressMilestone> {
        vec![
            ProgressMilestone {
                milestone_id: format!("{}-milestone-1", goal_id),
                name: "Initialization".to_string(),
                description: "Goal setup and resource allocation".to_string(),
                target_percentage: 10.0,
                completed: false,
                completed_at: None,
                dependencies: Vec::new(),
            },
            ProgressMilestone {
                milestone_id: format!("{}-milestone-2", goal_id),
                name: "First Quarter".to_string(),
                description: "25% completion milestone".to_string(),
                target_percentage: 25.0,
                completed: false,
                completed_at: None,
                dependencies: vec![format!("{}-milestone-1", goal_id)],
            },
            ProgressMilestone {
                milestone_id: format!("{}-milestone-3", goal_id),
                name: "Halfway Point".to_string(),
                description: "50% completion milestone".to_string(),
                target_percentage: 50.0,
                completed: false,
                completed_at: None,
                dependencies: vec![format!("{}-milestone-2", goal_id)],
            },
            ProgressMilestone {
                milestone_id: format!("{}-milestone-4", goal_id),
                name: "Final Quarter".to_string(),
                description: "75% completion milestone".to_string(),
                target_percentage: 75.0,
                completed: false,
                completed_at: None,
                dependencies: vec![format!("{}-milestone-3", goal_id)],
            },
            ProgressMilestone {
                milestone_id: format!("{}-milestone-5", goal_id),
                name: "Completion".to_string(),
                description: "Goal completion milestone".to_string(),
                target_percentage: 100.0,
                completed: false,
                completed_at: None,
                dependencies: vec![format!("{}-milestone-4", goal_id)],
            },
        ]
    }

    /// Estimate completion time for a goal
    fn estimate_completion(&self, goal: &BusinessGoal) -> Option<DateTime<Utc>> {
        goal.estimated_duration.map(|duration| {
            Utc::now() + chrono::Duration::from_std(duration).unwrap_or(chrono::Duration::hours(1))
        })
    }

    /// Broadcast progress event
    fn broadcast_event(&self, event: ProgressEvent) -> BusinessResult<()> {
        self.metrics
            .total_events
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        if let Err(_) = self.event_sender.send(event.clone()) {
            warn!("No subscribers for progress event: {}", event.event_id);
        }

        debug!(
            "Broadcasted progress event: {:?} for goal: {}",
            event.event_type, event.goal_id
        );
        Ok(())
    }
}

impl Default for ResourceUsage {
    fn default() -> Self {
        Self {
            gpu_memory_mb: 0,
            cpu_usage_percent: 0.0,
            memory_mb: 0,
            storage_mb: 0,
            cost_usd: 0.0,
            active_agents: 0,
            network_mbps: 0.0,
            measured_at: Utc::now(),
        }
    }
}

impl Default for ProgressTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::goal::{BusinessGoal, GoalCategory, GoalPriority};
    use chrono::Duration;

    fn create_test_goal() -> BusinessGoal {
        let mut goal = BusinessGoal::new(
            "Test data analysis goal".to_string(),
            "test@example.com".to_string(),
        );
        goal.category = GoalCategory::DataAnalysis;
        goal.priority = GoalPriority::Medium;
        goal.estimated_duration = Some(std::time::Duration::from_secs(4 * 3600));
        goal
    }

    fn create_test_tracker() -> ProgressTracker {
        ProgressTracker::new()
    }

    #[test]
    fn test_progress_tracker_creation() {
        let tracker = create_test_tracker();
        let metrics = tracker.get_metrics();
        assert_eq!(
            metrics
                .total_goals
                .load(std::sync::atomic::Ordering::Relaxed),
            0
        );
        assert_eq!(tracker.goal_progress.len(), 0);
    }

    #[test]
    fn test_start_tracking() {
        let tracker = create_test_tracker();
        let goal = create_test_goal();

        assert!(tracker.start_tracking(&goal).is_ok());

        let progress = tracker.get_progress(&goal.goal_id).unwrap();
        assert_eq!(progress.goal_id, goal.goal_id);
        assert_eq!(progress.percentage, 0.0);
        assert_eq!(progress.milestones.len(), 5);

        let metrics = tracker.get_metrics();
        assert_eq!(
            metrics
                .total_goals
                .load(std::sync::atomic::Ordering::Relaxed),
            1
        );
        assert_eq!(
            metrics
                .active_goals
                .load(std::sync::atomic::Ordering::Relaxed),
            1
        );
    }

    #[test]
    fn test_update_progress_valid() {
        let tracker = create_test_tracker();
        let goal = create_test_goal();

        tracker.start_tracking(&goal).unwrap();
        assert!(tracker.update_progress(&goal.goal_id, 25.0).is_ok());

        let progress = tracker.get_progress(&goal.goal_id).unwrap();
        assert_eq!(progress.percentage, 25.0);
        assert!(progress.milestones[0].completed); // 10% milestone
        assert!(progress.milestones[1].completed); // 25% milestone
        assert!(!progress.milestones[2].completed); // 50% milestone
    }

    #[test]
    fn test_update_progress_invalid() {
        let tracker = create_test_tracker();
        let goal = create_test_goal();

        tracker.start_tracking(&goal).unwrap();
        assert!(tracker.update_progress(&goal.goal_id, -10.0).is_err());
        assert!(tracker.update_progress(&goal.goal_id, 150.0).is_err());
    }

    #[test]
    fn test_update_progress_nonexistent_goal() {
        let tracker = create_test_tracker();
        assert!(tracker.update_progress("nonexistent", 50.0).is_err());
    }

    #[test]
    fn test_milestone_completion() {
        let tracker = create_test_tracker();
        let goal = create_test_goal();

        tracker.start_tracking(&goal).unwrap();
        tracker.update_progress(&goal.goal_id, 75.0).unwrap();

        let progress = tracker.get_progress(&goal.goal_id).unwrap();

        // Check that milestones up to 75% are completed
        assert!(progress.milestones[0].completed); // 10%
        assert!(progress.milestones[1].completed); // 25%
        assert!(progress.milestones[2].completed); // 50%
        assert!(progress.milestones[3].completed); // 75%
        assert!(!progress.milestones[4].completed); // 100%

        // Check completion timestamps
        assert!(progress.milestones[0].completed_at.is_some());
        assert!(progress.milestones[3].completed_at.is_some());
    }

    #[test]
    fn test_update_status() {
        let tracker = create_test_tracker();
        let goal = create_test_goal();

        tracker.start_tracking(&goal).unwrap();

        let new_status = GoalStatus::Executing {
            started_at: Utc::now(),
        };
        assert!(tracker
            .update_status(&goal.goal_id, new_status.clone())
            .is_ok());

        let progress = tracker.get_progress(&goal.goal_id).unwrap();
        assert_eq!(progress.status, new_status);
    }

    #[test]
    fn test_update_status_completion() {
        let tracker = create_test_tracker();
        let goal = create_test_goal();

        tracker.start_tracking(&goal).unwrap();

        let completed_status = GoalStatus::Completed {
            completed_at: Utc::now(),
        };
        tracker
            .update_status(&goal.goal_id, completed_status)
            .unwrap();

        let progress = tracker.get_progress(&goal.goal_id).unwrap();
        assert_eq!(progress.percentage, 100.0);

        let metrics = tracker.get_metrics();
        assert_eq!(
            metrics
                .active_goals
                .load(std::sync::atomic::Ordering::Relaxed),
            0
        );
        assert_eq!(
            metrics
                .completed_goals
                .load(std::sync::atomic::Ordering::Relaxed),
            1
        );
    }

    #[test]
    fn test_update_status_failure() {
        let tracker = create_test_tracker();
        let goal = create_test_goal();

        tracker.start_tracking(&goal).unwrap();

        let failed_status = GoalStatus::Failed {
            reason: "Test failure".to_string(),
            failed_at: Utc::now(),
        };
        tracker.update_status(&goal.goal_id, failed_status).unwrap();

        let metrics = tracker.get_metrics();
        assert_eq!(
            metrics
                .active_goals
                .load(std::sync::atomic::Ordering::Relaxed),
            0
        );
        assert_eq!(
            metrics
                .failed_goals
                .load(std::sync::atomic::Ordering::Relaxed),
            1
        );
    }

    #[test]
    fn test_update_resource_usage() {
        let tracker = create_test_tracker();
        let goal = create_test_goal();

        tracker.start_tracking(&goal).unwrap();

        let usage = ResourceUsage {
            gpu_memory_mb: 2048,
            cpu_usage_percent: 75.0,
            memory_mb: 4096,
            storage_mb: 1024,
            cost_usd: 10.50,
            active_agents: 3,
            network_mbps: 500.0,
            measured_at: Utc::now(),
        };

        assert!(tracker
            .update_resource_usage(&goal.goal_id, usage.clone())
            .is_ok());

        let progress = tracker.get_progress(&goal.goal_id).unwrap();
        assert_eq!(progress.resource_usage.gpu_memory_mb, 2048);
        assert_eq!(progress.resource_usage.cpu_usage_percent, 75.0);
        assert_eq!(progress.resource_usage.cost_usd, 10.50);
    }

    #[test]
    fn test_assign_agent() {
        let tracker = create_test_tracker();
        let goal = create_test_goal();

        tracker.start_tracking(&goal).unwrap();
        assert!(tracker.assign_agent(&goal.goal_id, "agent-123").is_ok());

        let progress = tracker.get_progress(&goal.goal_id).unwrap();
        assert_eq!(progress.assigned_agents.len(), 1);
        assert_eq!(progress.assigned_agents[0], "agent-123");
    }

    #[test]
    fn test_assign_duplicate_agent() {
        let tracker = create_test_tracker();
        let goal = create_test_goal();

        tracker.start_tracking(&goal).unwrap();
        tracker.assign_agent(&goal.goal_id, "agent-123").unwrap();
        tracker.assign_agent(&goal.goal_id, "agent-123").unwrap(); // Duplicate

        let progress = tracker.get_progress(&goal.goal_id).unwrap();
        assert_eq!(progress.assigned_agents.len(), 1); // Should not duplicate
    }

    #[test]
    fn test_release_agent() {
        let tracker = create_test_tracker();
        let goal = create_test_goal();

        tracker.start_tracking(&goal).unwrap();
        tracker.assign_agent(&goal.goal_id, "agent-123").unwrap();
        tracker.assign_agent(&goal.goal_id, "agent-456").unwrap();

        assert!(tracker.release_agent(&goal.goal_id, "agent-123").is_ok());

        let progress = tracker.get_progress(&goal.goal_id).unwrap();
        assert_eq!(progress.assigned_agents.len(), 1);
        assert_eq!(progress.assigned_agents[0], "agent-456");
    }

    #[test]
    fn test_release_nonexistent_agent() {
        let tracker = create_test_tracker();
        let goal = create_test_goal();

        tracker.start_tracking(&goal).unwrap();
        assert!(tracker.release_agent(&goal.goal_id, "nonexistent").is_ok()); // Should not error

        let progress = tracker.get_progress(&goal.goal_id).unwrap();
        assert_eq!(progress.assigned_agents.len(), 0);
    }

    #[test]
    fn test_get_active_goals() {
        let tracker = create_test_tracker();
        let goal1 = create_test_goal();
        let mut goal2 = create_test_goal();
        goal2.goal_id = "goal-2".to_string();

        tracker.start_tracking(&goal1).unwrap();
        tracker.start_tracking(&goal2).unwrap();

        // Set one goal to executing
        tracker
            .update_status(
                &goal1.goal_id,
                GoalStatus::Executing {
                    started_at: Utc::now(),
                },
            )
            .unwrap();

        // Set other goal to completed
        tracker
            .update_status(
                &goal2.goal_id,
                GoalStatus::Completed {
                    completed_at: Utc::now(),
                },
            )
            .unwrap();

        let active_goals = tracker.get_active_goals();
        assert_eq!(active_goals.len(), 1);
        assert_eq!(active_goals[0].goal_id, goal1.goal_id);
    }

    #[test]
    fn test_subscribe_to_events() {
        let tracker = create_test_tracker();
        let mut receiver = tracker.subscribe();
        let goal = create_test_goal();

        tracker.start_tracking(&goal).unwrap();

        // Should receive start event
        let event = receiver.try_recv().unwrap();
        assert_eq!(event.goal_id, goal.goal_id);
        assert_eq!(event.event_type, ProgressEventType::Started);
    }

    #[test]
    fn test_progress_event_history() {
        let tracker = create_test_tracker();
        let goal = create_test_goal();

        tracker.start_tracking(&goal).unwrap();
        tracker.update_progress(&goal.goal_id, 25.0).unwrap();
        tracker.update_progress(&goal.goal_id, 50.0).unwrap();

        let progress = tracker.get_progress(&goal.goal_id).unwrap();

        // Should have progress update events + milestone events
        assert!(progress.history.len() >= 2);

        let progress_events: Vec<_> = progress
            .history
            .iter()
            .filter(|e| e.event_type == ProgressEventType::ProgressUpdate)
            .collect();
        assert_eq!(progress_events.len(), 2);
    }

    #[test]
    fn test_stop_tracking() {
        let tracker = create_test_tracker();
        let goal = create_test_goal();

        tracker.start_tracking(&goal).unwrap();
        assert!(tracker.get_progress(&goal.goal_id).is_some());

        tracker.stop_tracking(&goal.goal_id).unwrap();
        assert!(tracker.get_progress(&goal.goal_id).is_none());
    }

    #[test]
    fn test_stop_tracking_nonexistent() {
        let tracker = create_test_tracker();
        assert!(tracker.stop_tracking("nonexistent").is_ok()); // Should not error
    }

    #[test]
    fn test_default_milestones() {
        let tracker = create_test_tracker();
        let milestones = tracker.create_default_milestones("test-goal");

        assert_eq!(milestones.len(), 5);
        assert_eq!(milestones[0].target_percentage, 10.0);
        assert_eq!(milestones[1].target_percentage, 25.0);
        assert_eq!(milestones[2].target_percentage, 50.0);
        assert_eq!(milestones[3].target_percentage, 75.0);
        assert_eq!(milestones[4].target_percentage, 100.0);

        // Check dependencies
        assert!(milestones[0].dependencies.is_empty());
        assert_eq!(milestones[1].dependencies.len(), 1);
        assert!(milestones[1]
            .dependencies
            .contains(&"test-goal-milestone-1".to_string()));
    }

    #[test]
    fn test_resource_usage_default() {
        let usage = ResourceUsage::default();
        assert_eq!(usage.gpu_memory_mb, 0);
        assert_eq!(usage.cpu_usage_percent, 0.0);
        assert_eq!(usage.cost_usd, 0.0);
        assert_eq!(usage.active_agents, 0);
    }

    #[test]
    fn test_progress_milestone_serialization() {
        let milestone = ProgressMilestone {
            milestone_id: "test-milestone".to_string(),
            name: "Test Milestone".to_string(),
            description: "Test description".to_string(),
            target_percentage: 50.0,
            completed: true,
            completed_at: Some(Utc::now()),
            dependencies: vec!["dep1".to_string()],
        };

        let serialized = serde_json::to_string(&milestone).unwrap();
        let deserialized: ProgressMilestone = serde_json::from_str(&serialized).unwrap();

        assert_eq!(milestone.milestone_id, deserialized.milestone_id);
        assert_eq!(milestone.target_percentage, deserialized.target_percentage);
        assert_eq!(milestone.completed, deserialized.completed);
    }

    #[test]
    fn test_progress_event_serialization() {
        let event = ProgressEvent {
            event_id: "event-123".to_string(),
            goal_id: "goal-456".to_string(),
            event_type: ProgressEventType::ProgressUpdate,
            timestamp: Utc::now(),
            previous_percentage: 25.0,
            new_percentage: 50.0,
            details: "Test event".to_string(),
            metadata: HashMap::new(),
        };

        let serialized = serde_json::to_string(&event).unwrap();
        let deserialized: ProgressEvent = serde_json::from_str(&serialized).unwrap();

        assert_eq!(event.event_id, deserialized.event_id);
        assert_eq!(event.goal_id, deserialized.goal_id);
        assert_eq!(event.event_type, deserialized.event_type);
        assert_eq!(event.previous_percentage, deserialized.previous_percentage);
    }

    #[test]
    fn test_progress_event_types() {
        let event_types = vec![
            ProgressEventType::Started,
            ProgressEventType::ProgressUpdate,
            ProgressEventType::MilestoneCompleted,
            ProgressEventType::StatusChanged,
            ProgressEventType::ResourceUpdate,
            ProgressEventType::Completed,
            ProgressEventType::Failed,
            ProgressEventType::Paused,
            ProgressEventType::Resumed,
            ProgressEventType::AgentAssigned,
            ProgressEventType::AgentReleased,
        ];

        for event_type in event_types {
            let serialized = serde_json::to_string(&event_type).unwrap();
            let deserialized: ProgressEventType = serde_json::from_str(&serialized).unwrap();
            assert_eq!(event_type, deserialized);
        }
    }

    #[test]
    fn test_goal_progress_serialization() {
        let progress = GoalProgress {
            goal_id: "goal-123".to_string(),
            percentage: 75.0,
            status: GoalStatus::Executing {
                started_at: Utc::now(),
            },
            milestones: Vec::new(),
            resource_usage: ResourceUsage::default(),
            history: Vec::new(),
            last_updated: Utc::now(),
            estimated_completion: Some(Utc::now() + Duration::hours(2)),
            assigned_agents: vec!["agent1".to_string(), "agent2".to_string()],
        };

        let serialized = serde_json::to_string(&progress).unwrap();
        let deserialized: GoalProgress = serde_json::from_str(&serialized).unwrap();

        assert_eq!(progress.goal_id, deserialized.goal_id);
        assert_eq!(progress.percentage, deserialized.percentage);
        assert_eq!(progress.assigned_agents, deserialized.assigned_agents);
    }
}
