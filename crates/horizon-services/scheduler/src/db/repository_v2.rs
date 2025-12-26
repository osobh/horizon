use crate::error::{HpcError, SchedulerErrorExt};
use crate::models::{Job, JobState, Priority};
use crate::Result;
use sqlx::{PgPool, Row};
use uuid::Uuid;

/// Database repository for job persistence using new JSONB schema
pub struct JobRepositoryV2 {
    pool: PgPool,
}

impl JobRepositoryV2 {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Create a new job in the database using JSONB resource_specs
    pub async fn create(&self, job: &Job) -> Result<Job> {
        // Serialize the entire ResourceRequest to JSON
        let resource_specs = serde_json::to_value(&job.resources)?;

        sqlx::query(
            r#"
            INSERT INTO jobs (id, user_id, job_name, state, priority, resource_specs,
                             command, working_dir, submitted_at)
            VALUES ($1, $2, $3, $4::job_state, $5::priority_level, $6, $7, $8, $9)
            "#,
        )
        .bind(job.id)
        .bind(&job.user_id)
        .bind(&job.job_name)
        .bind(Self::state_to_string(job.state))
        .bind(Self::priority_to_string(job.priority))
        .bind(resource_specs)
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
                   resource_specs, command, working_dir, submitted_at,
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
                   resource_specs, command, working_dir, submitted_at,
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
                   resource_specs, command, working_dir, submitted_at,
                   scheduled_at, started_at, completed_at
            FROM jobs
            ORDER BY submitted_at DESC
            "#,
        )
        .fetch_all(&self.pool)
        .await?;

        rows.into_iter().map(|row| self.row_to_job(row)).collect()
    }

    /// Update job state and timestamps
    pub async fn update_state(&self, job: &Job) -> Result<()> {
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
        use crate::models::resource::ResourceRequest;

        // Deserialize JSONB resource_specs back to ResourceRequest
        let resource_specs_json: serde_json::Value = row.try_get("resource_specs")?;
        let resources: ResourceRequest = serde_json::from_value(resource_specs_json)
            .map_err(|e| HpcError::internal(format!("Failed to parse resource_specs: {}", e)))?;

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
            _ => Err(HpcError::internal(format!("Invalid priority: {}", priority))),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::resource::{GpuVendor, ResourceRequest};

    #[test]
    fn test_state_serialization() {
        assert_eq!(JobRepositoryV2::state_to_string(JobState::Queued), "queued");
        assert_eq!(JobRepositoryV2::state_to_string(JobState::Running), "running");
        assert_eq!(JobRepositoryV2::state_to_string(JobState::Completed), "completed");
    }

    #[test]
    fn test_priority_serialization() {
        assert_eq!(JobRepositoryV2::priority_to_string(Priority::Low), "low");
        assert_eq!(JobRepositoryV2::priority_to_string(Priority::Normal), "normal");
        assert_eq!(JobRepositoryV2::priority_to_string(Priority::High), "high");
        assert_eq!(JobRepositoryV2::priority_to_string(Priority::Urgent), "urgent");
    }

    #[test]
    fn test_resource_request_json_serialization() {
        // Test that ResourceRequest can be serialized to JSON and back
        let request = ResourceRequest::new()
            .add_gpu(GpuVendor::Nvidia, "H100", 4.0)
            .add_cpu_cores(64.0)
            .add_memory_gb(512.0);

        let json = serde_json::to_value(&request).unwrap();
        let deserialized: ResourceRequest = serde_json::from_value(json).unwrap();

        assert!(deserialized.has_gpu());
        assert!(deserialized.has_cpu());
    }
}
