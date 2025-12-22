use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;
use uuid::Uuid;

use super::resource::{GpuVendor, RequestPriority, ResourceRequest, StorageType, TpuVariant};
use crate::error::{HpcError, SchedulerErrorExt};
use crate::Result;

/// Job priority levels (with OpenAPI schema support)
pub type Priority = RequestPriority;

// Manually implement ToSchema for RequestPriority
#[derive(utoipa::ToSchema)]
#[aliases(Priority = RequestPriority)]
#[schema(as = RequestPriority)]
pub enum PrioritySchema {
    Low,
    Normal,
    High,
    Urgent,
}

/// Job state machine
#[derive(Debug, Clone, Copy, Serialize, Deserialize, ToSchema, PartialEq, Eq, PartialOrd, Ord)]
pub enum JobState {
    Queued = 0,
    Scheduled = 1,
    Running = 2,
    Preempted = 3,
    Completed = 4,
    Failed = 5,
    Cancelled = 6,
}

impl JobState {
    /// Validates if a state transition is allowed
    pub fn can_transition_to(&self, next: JobState) -> bool {
        use JobState::*;

        match (self, next) {
            // Forward transitions
            (Queued, Scheduled) => true,
            (Scheduled, Running) => true,
            (Running, Completed) => true,
            (Running, Failed) => true,
            (Running, Preempted) => true,

            // Preemption recovery
            (Preempted, Queued) => true,

            // Cancellation from any active state
            (Queued, Cancelled) => true,
            (Scheduled, Cancelled) => true,
            (Running, Cancelled) => true,
            (Preempted, Cancelled) => true,

            // No transitions from terminal states
            (Completed, _) | (Failed, _) | (Cancelled, _) => false,

            // Invalid backward transitions
            _ => false,
        }
    }
}

/// Job represents a workload to be scheduled
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema, PartialEq)]
pub struct Job {
    pub id: Uuid,
    pub user_id: String,
    pub job_name: Option<String>,
    pub state: JobState,
    pub priority: Priority,
    pub resources: ResourceRequest,
    pub command: Option<String>,
    pub working_dir: Option<String>,
    pub environment: Option<serde_json::Value>,
    pub checkpoint_path: Option<String>,
    pub created_at: DateTime<Utc>,
    pub scheduled_at: Option<DateTime<Utc>>,
    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
}

impl Job {
    /// Creates a new job builder
    pub fn builder() -> JobBuilder {
        JobBuilder::default()
    }

    /// Transitions the job to a new state with validation
    pub fn transition_to(&mut self, next_state: JobState) -> Result<()> {
        if !self.state.can_transition_to(next_state) {
            return Err(HpcError::invalid_state_transition(
                format!("{:?}", self.state),
                format!("{:?}", next_state),
            ));
        }

        self.state = next_state;

        // Update timestamps based on state
        match next_state {
            JobState::Scheduled => self.scheduled_at = Some(Utc::now()),
            JobState::Running => self.started_at = Some(Utc::now()),
            JobState::Completed | JobState::Failed | JobState::Cancelled => {
                self.completed_at = Some(Utc::now())
            }
            _ => {}
        }

        Ok(())
    }
}

/// Builder pattern for creating jobs
#[derive(Debug, Default)]
pub struct JobBuilder {
    user_id: Option<String>,
    job_name: Option<String>,
    resources: ResourceRequest,
    priority: Option<Priority>,
    command: Option<String>,
    working_dir: Option<String>,
    environment: Option<serde_json::Value>,
}

impl JobBuilder {
    pub fn user_id(mut self, user_id: impl Into<String>) -> Self {
        self.user_id = Some(user_id.into());
        self
    }

    pub fn job_name(mut self, name: impl Into<String>) -> Self {
        self.job_name = Some(name.into());
        self
    }

    // New generic resource methods
    pub fn resources(mut self, resources: ResourceRequest) -> Self {
        self.resources = resources;
        self
    }

    // Backward compatibility helpers - GPU-focused
    pub fn gpu_count(mut self, count: usize) -> Self {
        self.resources = self.resources.add_gpu(GpuVendor::Nvidia, "H100", count as f64);
        self
    }

    pub fn gpu_type(mut self, gpu_type: impl Into<String>) -> Self {
        // This is for backward compatibility - update existing GPU spec or add new one
        self.resources = self.resources.add_gpu(GpuVendor::Nvidia, gpu_type, 1.0);
        self
    }

    pub fn cpu_cores(mut self, cores: u32) -> Self {
        self.resources = self.resources.add_cpu_cores(cores as f64);
        self
    }

    pub fn memory_gb(mut self, memory: u64) -> Self {
        self.resources = self.resources.add_memory_gb(memory as f64);
        self
    }

    // New resource helpers
    pub fn add_gpu(mut self, vendor: GpuVendor, model: impl Into<String>, count: f64) -> Self {
        self.resources = self.resources.add_gpu(vendor, model, count);
        self
    }

    pub fn add_tpu(mut self, variant: TpuVariant, count: f64) -> Self {
        self.resources = self.resources.add_tpu(variant, count);
        self
    }

    pub fn add_storage(mut self, storage_type: StorageType, gb: f64) -> Self {
        self.resources = self.resources.add_storage(storage_type, gb);
        self
    }

    pub fn priority(mut self, priority: Priority) -> Self {
        self.priority = Some(priority);
        self
    }

    pub fn command(mut self, command: impl Into<String>) -> Self {
        self.command = Some(command.into());
        self
    }

    pub fn working_dir(mut self, dir: impl Into<String>) -> Self {
        self.working_dir = Some(dir.into());
        self
    }

    pub fn environment(mut self, env: serde_json::Value) -> Self {
        self.environment = Some(env);
        self
    }

    pub fn build(self) -> Result<Job> {
        // Validate required fields
        let user_id = self.user_id.ok_or_else(|| {
            HpcError::validation("user_id is required")
        })?;

        // Validate that we have at least one resource
        self.resources
            .validate()
            .map_err(|e| HpcError::validation(e))?;

        Ok(Job {
            id: Uuid::new_v4(),
            user_id,
            job_name: self.job_name,
            state: JobState::Queued,
            priority: self.priority.unwrap_or(Priority::Normal),
            resources: self.resources,
            command: self.command,
            working_dir: self.working_dir,
            environment: self.environment,
            checkpoint_path: None,
            created_at: Utc::now(),
            scheduled_at: None,
            started_at: None,
            completed_at: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_priority_ordering() {
        assert!(Priority::Low < Priority::Normal);
        assert!(Priority::Normal < Priority::High);
    }

    #[test]
    fn test_state_ordering() {
        assert!(JobState::Queued < JobState::Scheduled);
        assert!(JobState::Scheduled < JobState::Running);
    }

    #[test]
    fn test_job_builder_gpu_backward_compat() {
        let job = Job::builder()
            .user_id("test_user")
            .gpu_count(4)
            .build()
            .unwrap();

        assert_eq!(job.user_id, "test_user");
        assert!(job.resources.has_gpu());
        assert_eq!(job.state, JobState::Queued);
        assert_eq!(job.priority, Priority::Normal);
    }

    #[test]
    fn test_job_builder_multi_resource() {
        let job = Job::builder()
            .user_id("test_user")
            .add_gpu(GpuVendor::Nvidia, "H100", 4.0)
            .cpu_cores(64)
            .memory_gb(512)
            .build()
            .unwrap();

        assert_eq!(job.user_id, "test_user");
        assert!(job.resources.has_gpu());
        assert!(job.resources.has_cpu());
    }

    #[test]
    fn test_job_builder_cpu_only() {
        let job = Job::builder()
            .user_id("test_user")
            .cpu_cores(128)
            .memory_gb(1024)
            .build()
            .unwrap();

        assert!(!job.resources.has_gpu());
        assert!(job.resources.has_cpu());
    }

    #[test]
    fn test_job_builder_tpu() {
        let job = Job::builder()
            .user_id("test_user")
            .add_tpu(TpuVariant::GoogleV5p, 8.0)
            .memory_gb(512)
            .build()
            .unwrap();

        assert!(!job.resources.has_gpu());
        assert!(job.resources.inner.has_tpu());
    }

    #[test]
    fn test_job_builder_validation_failure() {
        // No resources specified
        let result = Job::builder()
            .user_id("test_user")
            .build();

        assert!(result.is_err());
    }

    #[test]
    fn test_state_transitions() {
        let mut job = Job::builder()
            .user_id("test")
            .cpu_cores(16)
            .build()
            .unwrap();

        // Valid transitions
        assert!(job.transition_to(JobState::Scheduled).is_ok());
        assert_eq!(job.state, JobState::Scheduled);

        assert!(job.transition_to(JobState::Running).is_ok());
        assert_eq!(job.state, JobState::Running);

        assert!(job.transition_to(JobState::Completed).is_ok());
        assert_eq!(job.state, JobState::Completed);
    }

    #[test]
    fn test_invalid_transition() {
        let mut job = Job::builder()
            .user_id("test")
            .cpu_cores(16)
            .build()
            .unwrap();

        // Try to jump to completed without going through scheduled/running
        let result = job.transition_to(JobState::Completed);
        assert!(result.is_err());
    }

    #[test]
    fn test_preemption_transition() {
        let mut job = Job::builder()
            .user_id("test")
            .add_gpu(GpuVendor::Amd, "MI300X", 2.0)
            .build()
            .unwrap();

        job.transition_to(JobState::Scheduled).unwrap();
        job.transition_to(JobState::Running).unwrap();

        // Can preempt running job
        assert!(job.transition_to(JobState::Preempted).is_ok());
        assert_eq!(job.state, JobState::Preempted);

        // Can requeue preempted job
        assert!(job.transition_to(JobState::Queued).is_ok());
        assert_eq!(job.state, JobState::Queued);
    }

    #[test]
    fn test_cancellation() {
        let mut job1 = Job::builder()
            .user_id("test")
            .cpu_cores(8)
            .build()
            .unwrap();
        assert!(job1.transition_to(JobState::Cancelled).is_ok());

        let mut job2 = Job::builder()
            .user_id("test")
            .add_gpu(GpuVendor::Intel, "Max-1550", 1.0)
            .build()
            .unwrap();
        job2.transition_to(JobState::Scheduled).unwrap();
        assert!(job2.transition_to(JobState::Cancelled).is_ok());

        let mut job3 = Job::builder()
            .user_id("test")
            .cpu_cores(16)
            .build()
            .unwrap();
        job3.transition_to(JobState::Scheduled).unwrap();
        job3.transition_to(JobState::Running).unwrap();
        assert!(job3.transition_to(JobState::Cancelled).is_ok());
    }
}
