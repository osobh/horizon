use axum::{
    extract::{Path, State},
    http::StatusCode,
    Json,
};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;
use uuid::Uuid;

use crate::{
    api::state::AppState,
    error::{HpcError, SchedulerErrorExt},
    models::JobState,
};

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct CheckpointRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub checkpoint_path: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct CheckpointResponse {
    pub job_id: Uuid,
    pub checkpoint_path: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub status: String,
}

/// Create a checkpoint for a running job
#[utoipa::path(
    post,
    path = "/api/jobs/{job_id}/checkpoint",
    params(
        ("job_id" = Uuid, Path, description = "Job ID")
    ),
    request_body = CheckpointRequest,
    responses(
        (status = 201, description = "Checkpoint created successfully", body = CheckpointResponse),
        (status = 400, description = "Invalid request"),
        (status = 404, description = "Job not found"),
        (status = 409, description = "Job not in running state")
    ),
    tag = "checkpoints"
)]
pub async fn create_checkpoint(
    State(state): State<AppState>,
    Path(job_id): Path<Uuid>,
    Json(request): Json<CheckpointRequest>,
) -> Result<(StatusCode, Json<CheckpointResponse>), crate::HpcError> {
    // Get the job
    let mut job = state.scheduler.get_job(job_id).await?;

    // Verify job is in a checkpointable state (Running or Preempted)
    if !matches!(job.state, JobState::Running | JobState::Preempted) {
        return Err(HpcError::validation(format!(
            "Job must be in Running or Preempted state to create checkpoint. Current state: {:?}",
            job.state
        )));
    }

    // Set checkpoint path if provided, otherwise generate one
    let checkpoint_path = request.checkpoint_path.unwrap_or_else(|| {
        format!("/checkpoints/{}/checkpoint-{}.tar.gz", job_id, chrono::Utc::now().timestamp())
    });

    // Update job with checkpoint path
    job.checkpoint_path = Some(checkpoint_path.clone());

    // For now, we just mark the checkpoint location in the job
    // In a full implementation, this would trigger actual checkpoint creation
    // by the job's execution environment (e.g., Kubernetes, Slurm)

    Ok((StatusCode::CREATED, Json(CheckpointResponse {
        job_id,
        checkpoint_path,
        created_at: chrono::Utc::now(),
        status: "checkpoint_requested".to_string(),
    })))
}

/// Get checkpoint information for a job
#[utoipa::path(
    get,
    path = "/api/jobs/{job_id}/checkpoint",
    params(
        ("job_id" = Uuid, Path, description = "Job ID")
    ),
    responses(
        (status = 200, description = "Checkpoint information", body = CheckpointResponse),
        (status = 404, description = "Job or checkpoint not found")
    ),
    tag = "checkpoints"
)]
pub async fn get_checkpoint(
    State(state): State<AppState>,
    Path(job_id): Path<Uuid>,
) -> Result<Json<CheckpointResponse>, crate::HpcError> {
    // Get the job
    let job = state.scheduler.get_job(job_id).await?;

    // Check if job has a checkpoint
    let checkpoint_path = job.checkpoint_path.ok_or_else(|| {
        HpcError::checkpoint_not_found(job_id.to_string())
    })?;

    Ok(Json(CheckpointResponse {
        job_id,
        checkpoint_path,
        created_at: job.created_at,
        status: "available".to_string(),
    }))
}
