//! Agent scheduling and resource management
//!
//! This module provides comprehensive scheduling capabilities for agents,
//! including resource allocation, task queuing, and execution coordination.
//!
//! ## Architecture
//!
//! The scheduler has been refactored into logical modules:
//! - `resources`: Resource types and allocation management
//! - `tasks`: Task representation and status tracking  
//! - `scheduler_core`: Core scheduling logic and algorithms
//!
//! ## Usage
//!
//! ```rust
//! use agent_core::scheduler::{Scheduler, SchedulingPolicy, ResourceAllocation};
//! use std::time::Duration;
//!
//! let limits = ResourceAllocation {
//!     cpu_cores: 4.0,
//!     memory: 8192,
//!     gpu_compute: 2.0,  
//!     gpu_memory: 4096,
//!     network_bandwidth: 1000,
//!     storage_iops: 10000,
//! };
//!
//! let scheduler = Scheduler::new(SchedulingPolicy::Priority, limits, 10);
//! ```

// Re-export everything from the modular implementation
mod resources;
mod scheduler_core;
mod tasks;

pub use resources::{ResourceAllocation, ResourceType};
pub use scheduler_core::{SchedulerCore, SchedulingPolicy};
pub use tasks::{QueueStats, ScheduledTask, TaskStatus};

// Export the main scheduler interface
use crate::agent::AgentId;
use crate::error::{AgentError, AgentResult};
use crate::goal::Goal;
use once_cell::sync::Lazy;
use parking_lot::RwLock;
use std::sync::Arc;
use std::time::Duration;
use uuid::Uuid;

/// Global scheduler instance
static SCHEDULER: Lazy<Arc<RwLock<Option<Arc<Scheduler>>>>> =
    Lazy::new(|| Arc::new(RwLock::new(None)));

/// Main scheduler that provides backward compatibility with the original API
#[derive(Debug)]
pub struct Scheduler {
    core: SchedulerCore,
}

impl Scheduler {
    /// Create new scheduler
    pub fn new(
        policy: SchedulingPolicy,
        resource_limits: ResourceAllocation,
        max_concurrent_tasks: usize,
    ) -> Self {
        Self {
            core: SchedulerCore::new(policy, resource_limits, max_concurrent_tasks),
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
        self.core
            .schedule_goal(agent_id, goal, required_resources, estimated_duration)
            .await
    }

    /// Schedule a generic task (convenience method)
    pub async fn schedule_task(
        &self,
        task_id: String,
        requirements: ResourceAllocation,
    ) -> AgentResult<Uuid> {
        self.core.schedule_task(task_id, requirements).await
    }

    /// Cancel a task
    pub async fn cancel_task(&self, task_id: &str) -> AgentResult<()> {
        self.core.cancel_task(task_id).await
    }

    /// Get task status
    pub async fn get_task_status(&self, task_id: &str) -> Option<TaskStatus> {
        self.core.get_task_status(task_id).await
    }

    /// Get queue statistics
    pub async fn get_queue_stats(&self) -> QueueStats {
        self.core.get_queue_stats().await
    }

    /// Get resource usage for an agent
    pub fn get_agent_resources(&self, agent_id: AgentId) -> Option<ResourceAllocation> {
        self.core.get_agent_resources(agent_id)
    }

    /// Get total allocated resources
    pub fn get_allocated_resources(&self) -> ResourceAllocation {
        self.core.get_allocated_resources()
    }

    /// Get available resources
    pub fn get_available_resources(&self) -> ResourceAllocation {
        self.core.get_available_resources()
    }

    /// Get queue size
    pub fn queue_size(&self) -> usize {
        self.core.queue_size()
    }

    /// Get running task count
    pub fn running_tasks(&self) -> usize {
        self.core.running_tasks()
    }

    /// Get scheduling policy
    pub fn policy(&self) -> &SchedulingPolicy {
        &self.core.policy
    }

    /// Get max concurrent tasks
    pub fn max_concurrent_tasks(&self) -> usize {
        self.core.max_concurrent_tasks
    }

    /// Get resource limits
    pub fn resource_limits(&self) -> &ResourceAllocation {
        self.core.resource_limits()
    }
}

/// Initialize global scheduler
pub async fn init_scheduler() -> AgentResult<()> {
    let resource_limits = ResourceAllocation {
        cpu_cores: 32.0,
        memory: 64 * 1024 * 1024 * 1024, // 64 GB
        gpu_compute: 8.0,
        gpu_memory: 24 * 1024 * 1024 * 1024,        // 24 GB
        network_bandwidth: 10 * 1024 * 1024 * 1024, // 10 Gbps
        storage_iops: 100000,
    };

    let scheduler = Arc::new(Scheduler::new(
        SchedulingPolicy::default(),
        resource_limits,
        100, // max concurrent tasks
    ));

    *SCHEDULER.write() = Some(scheduler);
    Ok(())
}

/// Get global scheduler instance
pub fn scheduler() -> AgentResult<Arc<Scheduler>> {
    SCHEDULER
        .read()
        .clone()
        .ok_or_else(|| AgentError::SchedulingError {
            message: "Scheduler not initialized".to_string(),
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::goal::GoalPriority;

    #[test]
    fn test_resource_allocation() {
        let mut alloc1 = ResourceAllocation {
            cpu_cores: 2.0,
            memory: 1024,
            gpu_compute: 1.0,
            gpu_memory: 512,
            network_bandwidth: 100,
            storage_iops: 1000,
        };

        let alloc2 = ResourceAllocation {
            cpu_cores: 1.0,
            memory: 512,
            gpu_compute: 0.5,
            gpu_memory: 256,
            network_bandwidth: 50,
            storage_iops: 500,
        };

        // Test fits_within
        let limits = ResourceAllocation {
            cpu_cores: 4.0,
            memory: 2048,
            gpu_compute: 2.0,
            gpu_memory: 1024,
            network_bandwidth: 200,
            storage_iops: 2000,
        };

        assert!(alloc1.fits_within(&limits));

        // Test add
        alloc1.add(&alloc2);
        assert_eq!(alloc1.cpu_cores, 3.0);
        assert_eq!(alloc1.memory, 1536);

        // Test subtract
        alloc1.subtract(&alloc2);
        assert_eq!(alloc1.cpu_cores, 2.0);
        assert_eq!(alloc1.memory, 1024);
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

    #[tokio::test]
    async fn test_scheduler_creation() {
        let limits = ResourceAllocation {
            cpu_cores: 4.0,
            memory: 8192,
            gpu_compute: 2.0,
            gpu_memory: 4096,
            network_bandwidth: 1000,
            storage_iops: 10000,
        };

        let scheduler = Scheduler::new(SchedulingPolicy::Priority, limits, 10);

        assert_eq!(*scheduler.policy(), SchedulingPolicy::Priority);
        assert_eq!(scheduler.max_concurrent_tasks(), 10);
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

        let scheduler = Scheduler::new(SchedulingPolicy::Priority, limits.clone(), 10);

        let available = scheduler.get_available_resources();
        assert_eq!(available.cpu_cores, limits.cpu_cores);
        assert_eq!(available.memory, limits.memory);
    }

    #[tokio::test]
    async fn test_global_scheduler_init() {
        assert!(init_scheduler().await.is_ok());

        // Should be able to get scheduler
        let result = scheduler();
        assert!(result.is_ok());
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

        let scheduler = Scheduler::new(SchedulingPolicy::Priority, limits, 10);
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

        let scheduler = Scheduler::new(SchedulingPolicy::Priority, limits, 10);
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
    async fn test_get_agent_resources() {
        let limits = ResourceAllocation {
            cpu_cores: 4.0,
            memory: 8192,
            gpu_compute: 2.0,
            gpu_memory: 4096,
            network_bandwidth: 1000,
            storage_iops: 10000,
        };

        let scheduler = Scheduler::new(SchedulingPolicy::Priority, limits, 10);
        let agent_id = AgentId::new();

        // Initially no resources allocated
        assert!(scheduler.get_agent_resources(agent_id).is_none());
    }

    #[tokio::test]
    async fn test_multiple_scheduling_policies() {
        let policies = vec![
            SchedulingPolicy::FIFO,
            SchedulingPolicy::Priority,
            SchedulingPolicy::RoundRobin,
            SchedulingPolicy::ShortestJobFirst,
            SchedulingPolicy::FairShare,
        ];

        let limits = ResourceAllocation {
            cpu_cores: 4.0,
            memory: 8192,
            gpu_compute: 2.0,
            gpu_memory: 4096,
            network_bandwidth: 1000,
            storage_iops: 10000,
        };

        for policy in policies {
            let scheduler = Scheduler::new(policy.clone(), limits.clone(), 10);
            assert_eq!(*scheduler.policy(), policy);
        }
    }

    #[tokio::test]
    async fn test_scheduler_queue_operations() {
        let limits = ResourceAllocation {
            cpu_cores: 4.0,
            memory: 8192,
            gpu_compute: 2.0,
            gpu_memory: 4096,
            network_bandwidth: 1000,
            storage_iops: 10000,
        };

        let scheduler = Scheduler::new(SchedulingPolicy::Priority, limits, 10);
        let agent_id = AgentId::new();

        // Schedule multiple goals with different priorities
        let priorities = vec![
            GoalPriority::Low,
            GoalPriority::Critical,
            GoalPriority::Normal,
            GoalPriority::High,
        ];

        for (i, priority) in priorities.into_iter().enumerate() {
            let goal = Goal::new(format!("goal_{}", i), priority);
            let resources = ResourceAllocation {
                cpu_cores: 0.5,
                memory: 512,
                gpu_compute: 0.25,
                gpu_memory: 256,
                network_bandwidth: 50,
                storage_iops: 50,
            };

            scheduler
                .schedule_goal(agent_id, goal, resources, Duration::from_secs(30))
                .await
                .unwrap();
        }

        assert_eq!(scheduler.queue_size(), 4);
    }

    #[test]
    fn test_resource_type_variants() {
        // Ensure all resource types are covered
        let types = vec![
            ResourceType::CPU,
            ResourceType::Memory,
            ResourceType::GPUCompute,
            ResourceType::GPUMemory,
            ResourceType::Network,
            ResourceType::Storage,
        ];

        // Just ensure they can be used
        for resource_type in types {
            match resource_type {
                ResourceType::CPU => assert_eq!(format!("{:?}", resource_type), "CPU"),
                ResourceType::Memory => assert_eq!(format!("{:?}", resource_type), "Memory"),
                ResourceType::GPUCompute => {
                    assert_eq!(format!("{:?}", resource_type), "GPUCompute")
                }
                ResourceType::GPUMemory => assert_eq!(format!("{:?}", resource_type), "GPUMemory"),
                ResourceType::Network => assert_eq!(format!("{:?}", resource_type), "Network"),
                ResourceType::Storage => assert_eq!(format!("{:?}", resource_type), "Storage"),
            }
        }
    }

    #[test]
    fn test_resource_allocation_empty() {
        let empty = ResourceAllocation::empty();
        assert_eq!(empty.cpu_cores, 0.0);
        assert_eq!(empty.memory, 0);
        assert_eq!(empty.gpu_compute, 0.0);
        assert_eq!(empty.gpu_memory, 0);
        assert_eq!(empty.network_bandwidth, 0);
        assert_eq!(empty.storage_iops, 0);
    }

    #[test]
    fn test_resource_allocation_fits_within_edge_cases() {
        let alloc = ResourceAllocation {
            cpu_cores: 2.0,
            memory: 1024,
            gpu_compute: 1.0,
            gpu_memory: 512,
            network_bandwidth: 100,
            storage_iops: 1000,
        };

        // Exact match should fit
        assert!(alloc.fits_within(&alloc));

        // Any resource exceeding limit should fail
        let tight_limits = ResourceAllocation {
            cpu_cores: 1.9,
            memory: 1024,
            gpu_compute: 1.0,
            gpu_memory: 512,
            network_bandwidth: 100,
            storage_iops: 1000,
        };
        assert!(!alloc.fits_within(&tight_limits));
    }

    #[test]
    fn test_resource_allocation_subtract_underflow() {
        let mut alloc = ResourceAllocation {
            cpu_cores: 1.0,
            memory: 500,
            gpu_compute: 0.5,
            gpu_memory: 256,
            network_bandwidth: 50,
            storage_iops: 100,
        };

        let larger = ResourceAllocation {
            cpu_cores: 2.0,
            memory: 1000,
            gpu_compute: 1.0,
            gpu_memory: 512,
            network_bandwidth: 100,
            storage_iops: 200,
        };

        // Subtract should clamp to 0, not underflow
        alloc.subtract(&larger);
        assert_eq!(alloc.cpu_cores, 0.0);
        assert_eq!(alloc.memory, 0);
        assert_eq!(alloc.gpu_compute, 0.0);
        assert_eq!(alloc.gpu_memory, 0);
        assert_eq!(alloc.network_bandwidth, 0);
        assert_eq!(alloc.storage_iops, 0);
    }

    #[test]
    fn test_scheduling_policy_default() {
        assert_eq!(SchedulingPolicy::default(), SchedulingPolicy::Priority);
    }

    #[tokio::test]
    async fn test_get_allocated_vs_available_resources() {
        let limits = ResourceAllocation {
            cpu_cores: 4.0,
            memory: 8192,
            gpu_compute: 2.0,
            gpu_memory: 4096,
            network_bandwidth: 1000,
            storage_iops: 10000,
        };

        let scheduler = Scheduler::new(SchedulingPolicy::Priority, limits.clone(), 10);

        let allocated = scheduler.get_allocated_resources();
        let available = scheduler.get_available_resources();

        // Initially, nothing allocated
        assert_eq!(allocated.cpu_cores, 0.0);
        assert_eq!(allocated.memory, 0);

        // All resources available
        assert_eq!(available.cpu_cores, limits.cpu_cores);
        assert_eq!(available.memory, limits.memory);
        assert_eq!(available.gpu_compute, limits.gpu_compute);
        assert_eq!(available.gpu_memory, limits.gpu_memory);
    }

    #[test]
    fn test_scheduling_policy_serialization() {
        let policies = vec![
            SchedulingPolicy::FIFO,
            SchedulingPolicy::Priority,
            SchedulingPolicy::RoundRobin,
            SchedulingPolicy::ShortestJobFirst,
            SchedulingPolicy::FairShare,
        ];

        for policy in policies {
            let json = serde_json::to_string(&policy).unwrap();
            let deserialized: SchedulingPolicy = serde_json::from_str(&json).unwrap();
            assert_eq!(policy, deserialized);
        }
    }

    #[tokio::test]
    async fn test_scheduler_error_propagation() {
        // Clear any existing scheduler
        *SCHEDULER.write() = None;

        // Try to get scheduler before initialization
        let result = scheduler();
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            AgentError::SchedulingError { .. }
        ));

        // Initialize and try again
        init_scheduler().await.unwrap();
        assert!(scheduler().is_ok());
    }
}
