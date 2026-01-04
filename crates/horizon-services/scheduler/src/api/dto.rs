use serde::{Deserialize, Serialize};
use utoipa::ToSchema;
use uuid::Uuid;

use crate::models::resource::{ComputeType, ResourceType};
use crate::models::{Job, JobState, Priority};

/// Request to submit a new job
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct SubmitJobRequest {
    pub user_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub job_name: Option<String>,
    pub gpu_count: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cpu_cores: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_gb: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub priority: Option<Priority>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub command: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub working_dir: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub environment: Option<serde_json::Value>,
}

impl SubmitJobRequest {
    pub fn into_job(self) -> crate::Result<Job> {
        let mut builder = Job::builder()
            .user_id(self.user_id)
            .gpu_count(self.gpu_count);

        if let Some(job_name) = self.job_name {
            builder = builder.job_name(job_name);
        }
        if let Some(gpu_type) = self.gpu_type {
            builder = builder.gpu_type(gpu_type);
        }
        if let Some(cpu_cores) = self.cpu_cores {
            builder = builder.cpu_cores(cpu_cores);
        }
        if let Some(memory_gb) = self.memory_gb {
            builder = builder.memory_gb(memory_gb);
        }
        if let Some(priority) = self.priority {
            builder = builder.priority(priority);
        }
        if let Some(command) = self.command {
            builder = builder.command(command);
        }
        if let Some(working_dir) = self.working_dir {
            builder = builder.working_dir(working_dir);
        }
        if let Some(environment) = self.environment {
            builder = builder.environment(environment);
        }

        builder.build()
    }
}

/// Response containing job details
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct JobResponse {
    pub id: Uuid,
    pub user_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub job_name: Option<String>,
    pub state: JobState,
    pub priority: Priority,
    pub resources: ResourceResponseDto,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub command: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub working_dir: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub environment: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub checkpoint_path: Option<String>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scheduled_at: Option<chrono::DateTime<chrono::Utc>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub started_at: Option<chrono::DateTime<chrono::Utc>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completed_at: Option<chrono::DateTime<chrono::Utc>>,
}

impl From<Job> for JobResponse {
    fn from(job: Job) -> Self {
        // Extract GPU info
        let gpu_spec = job.resources.get_gpu_spec();
        let gpu_count = gpu_spec.map(|s| s.amount as usize).unwrap_or(0);
        let gpu_type = gpu_spec
            .and_then(|s| s.constraints.as_ref())
            .and_then(|c| c.model.clone());

        // Extract CPU info
        let cpu_spec = job
            .resources
            .inner
            .resources
            .get(&ResourceType::Compute(ComputeType::Cpu));
        let cpu_cores = cpu_spec.map(|s| s.amount as u32);

        // Extract memory info
        let memory_spec = job.resources.inner.resources.get(&ResourceType::Memory);
        let memory_gb = memory_spec.map(|s| s.amount as u64);

        Self {
            id: job.id,
            user_id: job.user_id,
            job_name: job.job_name,
            state: job.state,
            priority: job.priority,
            resources: ResourceResponseDto {
                gpu_count,
                gpu_type,
                cpu_cores,
                memory_gb,
            },
            command: job.command,
            working_dir: job.working_dir,
            environment: job.environment,
            checkpoint_path: job.checkpoint_path,
            created_at: job.created_at,
            scheduled_at: job.scheduled_at,
            started_at: job.started_at,
            completed_at: job.completed_at,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ResourceResponseDto {
    pub gpu_count: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cpu_cores: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_gb: Option<u64>,
}

/// Response containing multiple jobs
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct JobListResponse {
    pub jobs: Vec<JobResponse>,
    pub total: usize,
}

/// Queue statistics response
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct QueueStatusResponse {
    pub total: usize,
    pub urgent_priority: usize,
    pub high_priority: usize,
    pub normal_priority: usize,
    pub low_priority: usize,
}

/// Health check response
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct HealthResponse {
    pub status: String,
}

/// Slurm sbatch request
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct SlurmSbatchRequest {
    pub user_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub job_name: Option<String>,
    pub script_content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpus: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cpus: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mem_gb: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub partition: Option<String>,
}
