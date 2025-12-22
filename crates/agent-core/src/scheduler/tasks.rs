//! Task management and scheduling logic

use super::resources::ResourceAllocation;
use crate::agent::AgentId;
use crate::goal::Goal;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::time::Duration;
use uuid::Uuid;

/// Task execution status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskStatus {
    /// Task is in the queue
    Queued,
    /// Task is currently running  
    Running,
    /// Task completed successfully
    Completed,
    /// Task failed with error message
    Failed(String),
    /// Task was cancelled
    Cancelled,
}

/// Scheduled task
#[derive(Debug, Clone)]
pub struct ScheduledTask {
    /// Task ID
    pub id: Uuid,
    /// Agent ID
    pub agent_id: AgentId,
    /// Goal
    pub goal: Goal,
    /// Required resources
    pub resources: ResourceAllocation,
    /// Scheduled time
    pub scheduled_at: DateTime<Utc>,
    /// Estimated duration
    pub estimated_duration: Duration,
    /// Current status
    pub status: TaskStatus,
}

impl ScheduledTask {
    /// Create a new scheduled task
    pub fn new(
        agent_id: AgentId,
        goal: Goal,
        resources: ResourceAllocation,
        estimated_duration: Duration,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            agent_id,
            goal,
            resources,
            scheduled_at: Utc::now(),
            estimated_duration,
            status: TaskStatus::Queued,
        }
    }

    /// Update task status
    pub fn update_status(&mut self, status: TaskStatus) {
        self.status = status;
    }

    /// Check if task is terminal (completed, failed, or cancelled)
    pub fn is_terminal(&self) -> bool {
        matches!(
            self.status,
            TaskStatus::Completed | TaskStatus::Failed(_) | TaskStatus::Cancelled
        )
    }

    /// Check if task is running
    pub fn is_running(&self) -> bool {
        matches!(self.status, TaskStatus::Running)
    }

    /// Check if task is queued
    pub fn is_queued(&self) -> bool {
        matches!(self.status, TaskStatus::Queued)
    }
}

impl PartialEq for ScheduledTask {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for ScheduledTask {}

impl PartialOrd for ScheduledTask {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScheduledTask {
    fn cmp(&self, other: &Self) -> Ordering {
        // Higher priority goals come first
        match self.goal.priority.cmp(&other.goal.priority) {
            Ordering::Equal => {
                // Earlier scheduled times come first
                self.scheduled_at.cmp(&other.scheduled_at)
            }
            ordering => ordering,
        }
    }
}

/// Queue statistics for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueStats {
    /// Number of queued tasks
    pub queued_tasks: usize,
    /// Number of running tasks
    pub running_tasks: usize,
    /// Total tasks in system
    pub total_tasks: usize,
    /// Average wait time
    pub average_wait_time: Duration,
}

impl QueueStats {
    /// Create new queue stats
    pub fn new(queued_tasks: usize, running_tasks: usize) -> Self {
        Self {
            queued_tasks,
            running_tasks,
            total_tasks: queued_tasks + running_tasks,
            average_wait_time: Duration::ZERO,
        }
    }

    /// Update with wait time information
    pub fn with_wait_time(mut self, average_wait_time: Duration) -> Self {
        self.average_wait_time = average_wait_time;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::goal::GoalPriority;

    #[test]
    fn test_scheduled_task_creation() {
        let agent_id = AgentId::new();
        let goal = Goal::new("test task".to_string(), GoalPriority::Normal);
        let resources = ResourceAllocation::empty();
        let duration = Duration::from_secs(60);

        let task = ScheduledTask::new(agent_id, goal, resources, duration);

        assert_eq!(task.agent_id, agent_id);
        assert_eq!(task.status, TaskStatus::Queued);
        assert_eq!(task.estimated_duration, duration);
        assert!(!task.id.is_nil());
    }

    #[test]
    fn test_scheduled_task_ordering() {
        let agent_id = AgentId::new();

        let high_priority_goal = Goal::new("high".to_string(), GoalPriority::High);
        let normal_priority_goal = Goal::new("normal".to_string(), GoalPriority::Normal);

        let task1 = ScheduledTask::new(
            agent_id,
            high_priority_goal,
            ResourceAllocation::empty(),
            Duration::from_secs(1),
        );

        let task2 = ScheduledTask::new(
            agent_id,
            normal_priority_goal,
            ResourceAllocation::empty(),
            Duration::from_secs(1),
        );

        // High priority should come first
        assert!(task1 > task2);
    }

    #[test]
    fn test_scheduled_task_equality() {
        let agent_id = AgentId::new();
        let goal = Goal::new("test".to_string(), GoalPriority::Normal);
        let id = Uuid::new_v4();

        let mut task1 = ScheduledTask::new(
            agent_id,
            goal.clone(),
            ResourceAllocation::empty(),
            Duration::from_secs(1),
        );
        task1.id = id;

        let mut task2 = ScheduledTask::new(
            agent_id,
            goal,
            ResourceAllocation::empty(),
            Duration::from_secs(10),
        );
        task2.id = id;

        // Tasks with same ID should be equal
        assert_eq!(task1, task2);
    }

    #[test]
    fn test_scheduled_task_ordering_same_priority() {
        let agent_id = AgentId::new();
        let goal = Goal::new("test".to_string(), GoalPriority::Normal);
        let now = Utc::now();

        let mut earlier_task = ScheduledTask::new(
            agent_id,
            goal.clone(),
            ResourceAllocation::empty(),
            Duration::from_secs(1),
        );
        earlier_task.scheduled_at = now;

        let mut later_task = ScheduledTask::new(
            agent_id,
            goal,
            ResourceAllocation::empty(),
            Duration::from_secs(1),
        );
        later_task.scheduled_at = now + chrono::Duration::seconds(10);

        // Earlier scheduled task should come first when priorities are equal
        assert!(earlier_task > later_task);
    }

    #[test]
    fn test_task_status_transitions() {
        let statuses = vec![
            TaskStatus::Queued,
            TaskStatus::Running,
            TaskStatus::Completed,
            TaskStatus::Failed("error".to_string()),
            TaskStatus::Cancelled,
        ];

        // Verify each status can be cloned and compared
        for status in statuses {
            let cloned = status.clone();
            assert_eq!(status, cloned);
        }
    }

    #[test]
    fn test_task_status_checks() {
        let mut task = ScheduledTask::new(
            AgentId::new(),
            Goal::new("test".to_string(), GoalPriority::Normal),
            ResourceAllocation::empty(),
            Duration::from_secs(60),
        );

        // Initial state
        assert!(task.is_queued());
        assert!(!task.is_running());
        assert!(!task.is_terminal());

        // Running state
        task.update_status(TaskStatus::Running);
        assert!(!task.is_queued());
        assert!(task.is_running());
        assert!(!task.is_terminal());

        // Completed state
        task.update_status(TaskStatus::Completed);
        assert!(!task.is_queued());
        assert!(!task.is_running());
        assert!(task.is_terminal());

        // Failed state
        task.update_status(TaskStatus::Failed("test error".to_string()));
        assert!(task.is_terminal());

        // Cancelled state
        task.update_status(TaskStatus::Cancelled);
        assert!(task.is_terminal());
    }

    #[test]
    fn test_task_status_error_scenarios() {
        let error_status = TaskStatus::Failed("Resource allocation failed".to_string());

        if let TaskStatus::Failed(message) = error_status {
            assert!(!message.is_empty());
            assert!(message.contains("allocation"));
        } else {
            panic!("Expected Failed status");
        }

        // Test other status variants
        assert!(matches!(TaskStatus::Queued, TaskStatus::Queued));
        assert!(matches!(TaskStatus::Running, TaskStatus::Running));
        assert!(matches!(TaskStatus::Completed, TaskStatus::Completed));
        assert!(matches!(TaskStatus::Cancelled, TaskStatus::Cancelled));
    }

    #[test]
    fn test_queue_stats() {
        let stats = QueueStats::new(5, 3);
        assert_eq!(stats.queued_tasks, 5);
        assert_eq!(stats.running_tasks, 3);
        assert_eq!(stats.total_tasks, 8);
        assert_eq!(stats.average_wait_time, Duration::ZERO);

        let stats_with_wait = stats.with_wait_time(Duration::from_secs(10));
        assert_eq!(stats_with_wait.average_wait_time, Duration::from_secs(10));
    }

    #[test]
    fn test_task_serialization() {
        let status = TaskStatus::Failed("test error".to_string());
        let json = serde_json::to_string(&status).unwrap();
        let deserialized: TaskStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(status, deserialized);

        let stats = QueueStats::new(10, 5);
        let json = serde_json::to_string(&stats).unwrap();
        let deserialized: QueueStats = serde_json::from_str(&json).unwrap();
        assert_eq!(stats.queued_tasks, deserialized.queued_tasks);
        assert_eq!(stats.running_tasks, deserialized.running_tasks);
    }
}
