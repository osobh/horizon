use crate::error::{HpcError, SchedulerErrorExt};
use crate::models::{Job, JobState, Priority};
use crate::Result;
use sqlx::{PgPool, Row};
use uuid::Uuid;

/// Database repository for job persistence
pub struct JobRepository {
    pool: PgPool,
}

impl JobRepository {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Create a new job in the database
    pub async fn create(&self, job: &Job) -> Result<Job> {
        use crate::models::resource::{ComputeType, ResourceType};

        // Extract GPU info from new resource model
        let gpu_spec = job.resources.get_gpu_spec();
        let gpu_count = gpu_spec.map(|s| s.amount as i32).unwrap_or(0);
        let gpu_type = gpu_spec
            .and_then(|s| s.constraints.as_ref())
            .and_then(|c| c.model.clone());

        // Extract CPU info
        let cpu_spec = job
            .resources
            .inner
            .resources
            .get(&ResourceType::Compute(ComputeType::Cpu));
        let cpu_cores = cpu_spec.map(|s| s.amount as i32);

        // Extract memory info
        let memory_spec = job.resources.inner.resources.get(&ResourceType::Memory);
        let memory_gb = memory_spec.map(|s| s.amount as i64);

        sqlx::query(
            r#"
            INSERT INTO jobs (id, user_id, job_name, state, priority, gpu_count, gpu_type,
                             cpu_cores, memory_gb, command, working_dir, submitted_at)
            VALUES ($1, $2, $3, $4::job_state, $5::priority_level, $6, $7, $8, $9, $10, $11, $12)
            "#,
        )
        .bind(job.id)
        .bind(&job.user_id)
        .bind(&job.job_name)
        .bind(Self::state_to_string(job.state))
        .bind(Self::priority_to_string(job.priority))
        .bind(gpu_count)
        .bind(gpu_type)
        .bind(cpu_cores)
        .bind(memory_gb)
        .bind(&job.command)
        .bind(&job.working_dir)
        .bind(job.created_at)
        .execute(&self.pool)
        .await?;

        Ok(job.clone())
    }

    /// Get job by ID
    pub async fn get_by_id(&self, job_id: Uuid) -> Result<Job> {
        let row = sqlx::query(
            r#"
            SELECT id, user_id, job_name, state::text as state, priority::text as priority,
                   gpu_count, gpu_type, cpu_cores, memory_gb, command, working_dir, submitted_at,
                   scheduled_at, started_at, completed_at
            FROM jobs WHERE id = $1
            "#,
        )
        .bind(job_id)
        .fetch_optional(&self.pool)
        .await?
        .ok_or_else(|| HpcError::job_not_found(job_id))?;

        self.row_to_job(row)
    }

    /// List jobs by state
    pub async fn list_by_state(&self, state: JobState) -> Result<Vec<Job>> {
        let rows = sqlx::query(
            r#"
            SELECT id, user_id, job_name, state::text as state, priority::text as priority,
                   gpu_count, gpu_type, cpu_cores, memory_gb, command, working_dir, submitted_at,
                   scheduled_at, started_at, completed_at
            FROM jobs WHERE state = $1::job_state
            ORDER BY submitted_at DESC
            "#,
        )
        .bind(Self::state_to_string(state))
        .fetch_all(&self.pool)
        .await?;

        rows.into_iter().map(|row| self.row_to_job(row)).collect()
    }

    /// List all jobs
    pub async fn list_all(&self) -> Result<Vec<Job>> {
        let rows = sqlx::query(
            r#"
            SELECT id, user_id, job_name, state::text as state, priority::text as priority,
                   gpu_count, gpu_type, cpu_cores, memory_gb, command, working_dir, submitted_at,
                   scheduled_at, started_at, completed_at
            FROM jobs
            ORDER BY submitted_at DESC
            "#,
        )
        .fetch_all(&self.pool)
        .await?;

        rows.into_iter().map(|row| self.row_to_job(row)).collect()
    }

    /// Update job
    pub async fn update(&self, job: &Job) -> Result<()> {
        sqlx::query(
            r#"
            UPDATE jobs
            SET state = $2::job_state, scheduled_at = $3, started_at = $4, completed_at = $5
            WHERE id = $1
            "#,
        )
        .bind(job.id)
        .bind(Self::state_to_string(job.state))
        .bind(job.scheduled_at)
        .bind(job.started_at)
        .bind(job.completed_at)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    fn row_to_job(&self, row: sqlx::postgres::PgRow) -> Result<Job> {
        use crate::models::resource::{GpuVendor, ResourceRequest};

        let gpu_count: i32 = row.try_get("gpu_count")?;
        let gpu_type: Option<String> = row.try_get("gpu_type").ok().flatten();
        let cpu_cores: Option<i32> = row.try_get("cpu_cores").ok().flatten();
        let memory_gb: Option<i64> = row.try_get("memory_gb").ok().flatten();

        // Build resources using the new API
        let mut resources = ResourceRequest::new();

        if gpu_count > 0 {
            let model = gpu_type.unwrap_or_else(|| "H100".to_string());
            resources = resources.add_gpu(GpuVendor::Nvidia, model, gpu_count as f64);
        }

        if let Some(cores) = cpu_cores {
            resources = resources.add_cpu_cores(cores as f64);
        }

        if let Some(mem) = memory_gb {
            resources = resources.add_memory_gb(mem as f64);
        }

        let state_str: String = row.try_get("state")?;
        let state = self.parse_job_state(&state_str)?;

        let priority_str: String = row.try_get("priority")?;
        let priority = self.parse_priority_string(&priority_str)?;

        Ok(Job {
            id: row.try_get("id")?,
            user_id: row.try_get("user_id")?,
            job_name: row.try_get("job_name").ok(),
            state,
            priority,
            resources,
            command: row.try_get("command").ok(),
            working_dir: row.try_get("working_dir").ok(),
            environment: None,
            checkpoint_path: None,
            created_at: row.try_get("submitted_at")?,
            scheduled_at: row.try_get("scheduled_at").ok(),
            started_at: row.try_get("started_at").ok(),
            completed_at: row.try_get("completed_at").ok(),
        })
    }

    fn parse_job_state(&self, state: &str) -> Result<JobState> {
        match state {
            "queued" => Ok(JobState::Queued),
            "scheduled" => Ok(JobState::Scheduled),
            "running" => Ok(JobState::Running),
            "preempted" => Ok(JobState::Preempted),
            "completed" => Ok(JobState::Completed),
            "failed" => Ok(JobState::Failed),
            "cancelled" => Ok(JobState::Cancelled),
            _ => Err(HpcError::internal(format!("Invalid job state: {}", state))),
        }
    }

    fn parse_priority_string(&self, priority: &str) -> Result<Priority> {
        match priority {
            "low" => Ok(Priority::Low),
            "normal" => Ok(Priority::Normal),
            "high" => Ok(Priority::High),
            "urgent" => Ok(Priority::Urgent),
            _ => Err(HpcError::internal(format!(
                "Invalid priority: {}",
                priority
            ))),
        }
    }

    fn state_to_string(state: JobState) -> String {
        match state {
            JobState::Queued => "queued".to_string(),
            JobState::Scheduled => "scheduled".to_string(),
            JobState::Running => "running".to_string(),
            JobState::Preempted => "preempted".to_string(),
            JobState::Completed => "completed".to_string(),
            JobState::Failed => "failed".to_string(),
            JobState::Cancelled => "cancelled".to_string(),
        }
    }

    fn priority_to_string(priority: Priority) -> String {
        match priority {
            Priority::Low => "low".to_string(),
            Priority::Normal => "normal".to_string(),
            Priority::High => "high".to_string(),
            Priority::Urgent => "urgent".to_string(),
        }
    }
}
