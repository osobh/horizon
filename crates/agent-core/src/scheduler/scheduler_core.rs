//! Core scheduler implementation

use super::resources::ResourceAllocation;
use super::tasks::{QueueStats, ScheduledTask, TaskStatus};
use crate::agent::AgentId;
use crate::error::{AgentError, AgentResult};
use crate::goal::{Goal, GoalPriority};
use dashmap::DashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;
use stratoswarm_core::{PrioritySchedulerQueue, SchedulerPriority};
use tokio::sync::Semaphore;
use tracing;
use uuid::Uuid;

/// Convert GoalPriority to SchedulerPriority for branch-prediction-friendly queue.
impl From<GoalPriority> for SchedulerPriority {
    #[inline]
    fn from(priority: GoalPriority) -> Self {
        match priority {
            GoalPriority::Low => SchedulerPriority::Low,
            GoalPriority::Normal => SchedulerPriority::Normal,
            GoalPriority::High => SchedulerPriority::High,
            GoalPriority::Critical => SchedulerPriority::Critical,
        }
    }
}

/// Scheduling policy
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SchedulingPolicy {
    /// First in, first out
    FIFO,
    /// Priority-based scheduling
    #[default]
    Priority,
    /// Round-robin scheduling
    RoundRobin,
    /// Shortest job first
    ShortestJobFirst,
    /// Fair share scheduling
    FairShare,
    /// Custom scheduling policy
    Custom(String),
}

/// Core scheduler implementation
#[derive(Debug)]
pub struct SchedulerCore {
    /// Scheduling policy
    pub policy: SchedulingPolicy,
    /// Resource limits
    resource_limits: ResourceAllocation,
    /// Currently allocated resources
    allocated_resources: Arc<RwLock<ResourceAllocation>>,
    /// Branch-prediction-friendly priority queue using segmented VecDeques
    /// instead of BinaryHeap (O(1) enqueue/dequeue with predictable branches)
    task_queue: Arc<RwLock<PrioritySchedulerQueue<ScheduledTask>>>,
    /// Running tasks
    running_tasks: DashMap<Uuid, ScheduledTask>,
    /// Agent allocations
    agent_allocations: DashMap<AgentId, ResourceAllocation>,
    /// Concurrency limit
    pub max_concurrent_tasks: usize,
    /// Task semaphore
    _task_semaphore: Arc<Semaphore>,
}

impl SchedulerCore {
    /// Create new scheduler core
    pub fn new(
        policy: SchedulingPolicy,
        resource_limits: ResourceAllocation,
        max_concurrent_tasks: usize,
    ) -> Self {
        Self {
            policy,
            resource_limits,
            allocated_resources: Arc::new(RwLock::new(ResourceAllocation::empty())),
            task_queue: Arc::new(RwLock::new(PrioritySchedulerQueue::new())),
            running_tasks: DashMap::new(),
            agent_allocations: DashMap::new(),
            max_concurrent_tasks,
            _task_semaphore: Arc::new(Semaphore::new(max_concurrent_tasks)),
        }
    }

    /// Schedule a goal for an agent
    pub async fn schedule_goal(
        &self,
        agent_id: AgentId,
        goal: Goal,
        required_resources: ResourceAllocation,
        estimated_duration: Duration,
    ) -> AgentResult<Uuid> {
        // Check if resources fit within limits
        if !required_resources.fits_within(&self.resource_limits) {
            return Err(AgentError::ResourceLimitExceeded {
                resource: "Requested resources exceed system limits".to_string(),
            });
        }

        // Create scheduled task
        let priority: SchedulerPriority = goal.priority.into();
        let task = ScheduledTask::new(agent_id, goal, required_resources, estimated_duration);
        let task_id = task.id;

        // Add to queue
        {
            let mut queue = self.task_queue.write();
            queue.enqueue(task, priority);
        }

        // Try to run tasks
        self.try_run_tasks().await;

        Ok(task_id)
    }

    /// Schedule a generic task (convenience method)
    pub async fn schedule_task(
        &self,
        task_id: String,
        requirements: ResourceAllocation,
    ) -> AgentResult<Uuid> {
        use crate::goal::GoalPriority;

        // Map to existing schedule_goal method with dummy values
        let agent_id = AgentId::new();
        let goal = Goal::new(format!("Task: {}", task_id), GoalPriority::Normal);
        let duration = Duration::from_secs(60);

        self.schedule_goal(agent_id, goal, requirements, duration)
            .await
    }

    /// Cancel a task
    pub async fn cancel_task(&self, task_id: &str) -> AgentResult<()> {
        // Find task by matching the goal description
        let mut queue = self.task_queue.write();
        let mut temp_tasks = Vec::new();
        let mut found = false;

        while let Some(mut task) = queue.dequeue() {
            if task.goal.description.contains(task_id) {
                task.update_status(TaskStatus::Cancelled);
                found = true;
                // Don't put cancelled task back in queue
            } else {
                temp_tasks.push(task);
            }
        }

        // Put back non-cancelled tasks with their original priorities
        for task in temp_tasks {
            let priority: SchedulerPriority = task.goal.priority.into();
            queue.enqueue(task, priority);
        }

        if found {
            Ok(())
        } else {
            Err(AgentError::Other(format!("Task {} not found", task_id)))
        }
    }

    /// Get task status
    pub async fn get_task_status(&self, task_id: &str) -> Option<TaskStatus> {
        // Check running tasks first
        for running_task in self.running_tasks.iter() {
            if running_task.value().goal.description.contains(task_id) {
                return Some(running_task.value().status.clone());
            }
        }

        // Check queued tasks
        let queue = self.task_queue.read();
        for task in queue.iter() {
            if task.goal.description.contains(task_id) {
                return Some(task.status.clone());
            }
        }

        None
    }

    /// Get queue statistics
    pub async fn get_queue_stats(&self) -> QueueStats {
        QueueStats::new(self.queue_size(), self.running_tasks())
    }

    /// Try to run pending tasks
    async fn try_run_tasks(&self) {
        loop {
            // Check if we can run more tasks
            if self.running_tasks.len() >= self.max_concurrent_tasks {
                break;
            }

            // Get next task that fits resources
            let next_task = {
                let mut queue = self.task_queue.write();
                let allocated = self.allocated_resources.read();

                // Find task that fits available resources
                let mut temp_tasks = Vec::new();
                let mut selected_task = None;

                while let Some(task) = queue.dequeue() {
                    let mut test_allocation = allocated.clone();
                    test_allocation.add(&task.resources);

                    if test_allocation.fits_within(&self.resource_limits) {
                        selected_task = Some(task);
                        break;
                    } else {
                        temp_tasks.push(task);
                    }
                }

                // Put back tasks we couldn't run with their original priorities
                for task in temp_tasks {
                    let priority: SchedulerPriority = task.goal.priority.into();
                    queue.enqueue(task, priority);
                }

                selected_task
            };

            if let Some(mut task) = next_task {
                // Allocate resources
                {
                    let mut allocated = self.allocated_resources.write();
                    allocated.add(&task.resources);

                    self.agent_allocations
                        .entry(task.agent_id)
                        .and_modify(|alloc| alloc.add(&task.resources))
                        .or_insert_with(|| task.resources.clone());
                }

                // Update task status to running
                task.update_status(TaskStatus::Running);

                // Mark as running
                self.running_tasks.insert(task.id, task);

                // NOTE: Task execution logic is implemented in _execute_task() method below
                // Uncomment the following line to enable actual task spawning:
                // tokio::spawn(self._execute_task(task));
            } else {
                // No task can run with available resources
                break;
            }
        }
    }

    /// Execute a task (placeholder for future implementation)
    async fn _execute_task(&self, task: ScheduledTask) {
        // Acquire semaphore permit
        let _permit = match self._task_semaphore.acquire().await {
            Ok(permit) => permit,
            Err(e) => {
                tracing::error!(
                    task_id = %task.id,
                    error = ?e,
                    "Task semaphore closed - scheduler is shutting down"
                );
                return;
            }
        };

        // Simulate task execution
        tokio::time::sleep(task.estimated_duration).await;

        // Release resources
        {
            let mut allocated = self.allocated_resources.write();
            allocated.subtract(&task.resources);

            if let Some(mut agent_alloc) = self.agent_allocations.get_mut(&task.agent_id) {
                agent_alloc.subtract(&task.resources);
            }
        }

        // Remove from running
        self.running_tasks.remove(&task.id);

        // Try to run more tasks
        // self.try_run_tasks().await;
    }

    /// Get resource usage for an agent
    pub fn get_agent_resources(&self, agent_id: AgentId) -> Option<ResourceAllocation> {
        self.agent_allocations.get(&agent_id).map(|r| r.clone())
    }

    /// Get total allocated resources
    pub fn get_allocated_resources(&self) -> ResourceAllocation {
        self.allocated_resources.read().clone()
    }

    /// Get available resources
    pub fn get_available_resources(&self) -> ResourceAllocation {
        let allocated = self.allocated_resources.read();
        let mut available = self.resource_limits.clone();
        available.subtract(&allocated);
        available
    }

    /// Get queue size
    pub fn queue_size(&self) -> usize {
        self.task_queue.read().len()
    }

    /// Get running task count
    pub fn running_tasks(&self) -> usize {
        self.running_tasks.len()
    }

    /// Get resource limits
    pub fn resource_limits(&self) -> &ResourceAllocation {
        &self.resource_limits
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::goal::GoalPriority;

    #[tokio::test]
    async fn test_scheduler_core_creation() {
        let limits = ResourceAllocation {
            cpu_cores: 4.0,
            memory: 8192,
            gpu_compute: 2.0,
            gpu_memory: 4096,
            network_bandwidth: 1000,
            storage_iops: 10000,
        };

        let scheduler = SchedulerCore::new(SchedulingPolicy::Priority, limits, 10);

        assert_eq!(scheduler.policy, SchedulingPolicy::Priority);
        assert_eq!(scheduler.max_concurrent_tasks, 10);
        assert_eq!(scheduler.queue_size(), 0);
        assert_eq!(scheduler.running_tasks(), 0);
    }

    #[tokio::test]
    async fn test_resource_availability() {
        let limits = ResourceAllocation {
            cpu_cores: 4.0,
            memory: 8192,
            gpu_compute: 2.0,
            gpu_memory: 4096,
            network_bandwidth: 1000,
            storage_iops: 10000,
        };

        let scheduler = SchedulerCore::new(SchedulingPolicy::Priority, limits.clone(), 10);

        let available = scheduler.get_available_resources();
        assert_eq!(available.cpu_cores, limits.cpu_cores);
        assert_eq!(available.memory, limits.memory);
    }

    #[tokio::test]
    async fn test_schedule_goal_basic() {
        let limits = ResourceAllocation {
            cpu_cores: 4.0,
            memory: 8192,
            gpu_compute: 2.0,
            gpu_memory: 4096,
            network_bandwidth: 1000,
            storage_iops: 10000,
        };

        let scheduler = SchedulerCore::new(SchedulingPolicy::Priority, limits, 10);
        let agent_id = AgentId::new();
        let goal = Goal::new("test goal".to_string(), GoalPriority::Normal);

        let resources = ResourceAllocation {
            cpu_cores: 1.0,
            memory: 1024,
            gpu_compute: 0.5,
            gpu_memory: 512,
            network_bandwidth: 100,
            storage_iops: 100,
        };

        let task_id = scheduler
            .schedule_goal(agent_id, goal, resources, Duration::from_secs(60))
            .await
            .unwrap();

        assert!(!task_id.is_nil());
        assert_eq!(scheduler.queue_size(), 1);
    }

    #[tokio::test]
    async fn test_schedule_goal_exceeds_limits() {
        let limits = ResourceAllocation {
            cpu_cores: 2.0,
            memory: 4096,
            gpu_compute: 1.0,
            gpu_memory: 2048,
            network_bandwidth: 500,
            storage_iops: 5000,
        };

        let scheduler = SchedulerCore::new(SchedulingPolicy::Priority, limits, 10);
        let agent_id = AgentId::new();
        let goal = Goal::new("big goal".to_string(), GoalPriority::High);

        let excessive_resources = ResourceAllocation {
            cpu_cores: 10.0, // Exceeds limit
            memory: 1024,
            gpu_compute: 0.5,
            gpu_memory: 512,
            network_bandwidth: 100,
            storage_iops: 100,
        };

        let result = scheduler
            .schedule_goal(agent_id, goal, excessive_resources, Duration::from_secs(60))
            .await;

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            AgentError::ResourceLimitExceeded { .. }
        ));
    }

    #[tokio::test]
    async fn test_schedule_task_convenience_method() {
        let limits = ResourceAllocation {
            cpu_cores: 4.0,
            memory: 8192,
            gpu_compute: 2.0,
            gpu_memory: 4096,
            network_bandwidth: 1000,
            storage_iops: 10000,
        };

        let scheduler = SchedulerCore::new(SchedulingPolicy::Priority, limits, 10);

        let resources = ResourceAllocation {
            cpu_cores: 1.0,
            memory: 1024,
            gpu_compute: 0.5,
            gpu_memory: 512,
            network_bandwidth: 100,
            storage_iops: 100,
        };

        let task_id = scheduler
            .schedule_task("test-task-123".to_string(), resources)
            .await
            .unwrap();

        assert!(!task_id.is_nil());
        assert_eq!(scheduler.queue_size(), 1);
    }

    #[tokio::test]
    async fn test_cancel_task() {
        let limits = ResourceAllocation {
            cpu_cores: 4.0,
            memory: 8192,
            gpu_compute: 2.0,
            gpu_memory: 4096,
            network_bandwidth: 1000,
            storage_iops: 10000,
        };

        let scheduler = SchedulerCore::new(SchedulingPolicy::Priority, limits, 5);

        let resources = ResourceAllocation {
            cpu_cores: 1.0,
            memory: 1024,
            gpu_compute: 0.5,
            gpu_memory: 512,
            network_bandwidth: 100,
            storage_iops: 100,
        };

        // Schedule a task
        scheduler
            .schedule_task("cancellable-task".to_string(), resources)
            .await
            .unwrap();

        assert_eq!(scheduler.queue_size(), 1);

        // Cancel the task
        let result = scheduler.cancel_task("cancellable-task").await;
        assert!(result.is_ok());

        // Task should be removed from queue
        assert_eq!(scheduler.queue_size(), 0);
    }

    #[tokio::test]
    async fn test_get_task_status() {
        let limits = ResourceAllocation {
            cpu_cores: 4.0,
            memory: 8192,
            gpu_compute: 2.0,
            gpu_memory: 4096,
            network_bandwidth: 1000,
            storage_iops: 10000,
        };

        let scheduler = SchedulerCore::new(SchedulingPolicy::Priority, limits, 5);

        let resources = ResourceAllocation {
            cpu_cores: 1.0,
            memory: 1024,
            gpu_compute: 0.5,
            gpu_memory: 512,
            network_bandwidth: 100,
            storage_iops: 100,
        };

        // Schedule a task
        scheduler
            .schedule_task("status-test-task".to_string(), resources)
            .await
            .unwrap();

        // Check status
        let status = scheduler.get_task_status("status-test-task").await;
        assert!(status.is_some());
        // Task should be either Queued or Running depending on resource availability
        match status.unwrap() {
            TaskStatus::Queued | TaskStatus::Running => {}
            _ => panic!("Unexpected task status"),
        }
    }

    #[tokio::test]
    async fn test_queue_stats() {
        let limits = ResourceAllocation {
            cpu_cores: 2.0,
            memory: 4096,
            gpu_compute: 1.0,
            gpu_memory: 2048,
            network_bandwidth: 500,
            storage_iops: 5000,
        };

        let scheduler = SchedulerCore::new(SchedulingPolicy::Priority, limits, 3);

        // Get initial stats
        let initial_stats = scheduler.get_queue_stats().await;
        assert_eq!(initial_stats.queued_tasks, 0);
        assert_eq!(initial_stats.running_tasks, 0);
        assert_eq!(initial_stats.total_tasks, 0);

        // Schedule some tasks
        for i in 0..2 {
            let requirements = ResourceAllocation {
                cpu_cores: 0.5,
                memory: 512,
                gpu_compute: 0.25,
                gpu_memory: 256,
                network_bandwidth: 50,
                storage_iops: 500,
            };

            scheduler
                .schedule_task(format!("task-{}", i), requirements)
                .await
                .unwrap();
        }

        let after_stats = scheduler.get_queue_stats().await;
        assert_eq!(after_stats.total_tasks, 2);
        // queued_tasks + running_tasks should equal total_tasks
        assert_eq!(
            after_stats.queued_tasks + after_stats.running_tasks,
            after_stats.total_tasks
        );
    }

    #[tokio::test]
    async fn test_different_scheduling_policies() {
        let limits = ResourceAllocation {
            cpu_cores: 4.0,
            memory: 8192,
            gpu_compute: 2.0,
            gpu_memory: 4096,
            network_bandwidth: 1000,
            storage_iops: 10000,
        };

        let policies = vec![
            SchedulingPolicy::FIFO,
            SchedulingPolicy::Priority,
            SchedulingPolicy::ShortestJobFirst,
            SchedulingPolicy::RoundRobin,
            SchedulingPolicy::Custom("custom-policy".to_string()),
        ];

        for policy in policies {
            let scheduler = SchedulerCore::new(policy.clone(), limits.clone(), 5);

            // Verify scheduler was created with the correct policy
            assert_eq!(scheduler.policy, policy);
        }
    }

    #[test]
    fn test_scheduling_policy_serialization() {
        let policies = vec![
            SchedulingPolicy::FIFO,
            SchedulingPolicy::Priority,
            SchedulingPolicy::ShortestJobFirst,
            SchedulingPolicy::RoundRobin,
            SchedulingPolicy::Custom("test-policy".to_string()),
        ];

        for policy in policies {
            let json = serde_json::to_string(&policy).unwrap();
            let deserialized: SchedulingPolicy = serde_json::from_str(&json).unwrap();
            assert_eq!(policy, deserialized);
        }
    }

    #[test]
    fn test_scheduling_policy_default() {
        assert_eq!(SchedulingPolicy::default(), SchedulingPolicy::Priority);
    }
}
