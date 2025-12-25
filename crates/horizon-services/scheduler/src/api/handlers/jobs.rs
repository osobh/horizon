use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    Json,
};
use hpc_channels::SchedulerMessage;
use serde::Deserialize;
use utoipa::ToSchema;
use uuid::Uuid;

use crate::{
    api::{
        dto::{JobListResponse, JobResponse, SubmitJobRequest},
        state::AppState,
    },
    models::JobState as ModelJobState,
};

#[derive(Debug, Deserialize, ToSchema)]
pub struct ListJobsQuery {
    pub state: Option<String>,
    pub user_id: Option<String>,
    pub priority: Option<String>,
}

/// Submit a new job to the scheduler
#[utoipa::path(
    post,
    path = "/api/v1/jobs",
    request_body = SubmitJobRequest,
    responses(
        (status = 201, description = "Job submitted successfully", body = JobResponse),
        (status = 400, description = "Invalid request")
    ),
    tag = "jobs"
)]
pub async fn submit_job(
    State(state): State<AppState>,
    Json(request): Json<SubmitJobRequest>,
) -> Result<(StatusCode, Json<JobResponse>), crate::HpcError> {
    // Convert request to Job
    let job = request.into_job()?;

    // Submit job to scheduler
    let submitted_job = state.scheduler.submit_job(job).await?;

    // Publish job submitted event via hpc-channels
    state.publish_job_event(SchedulerMessage::JobSubmitted {
        job_id: submitted_job.id.to_string(),
        tenant_id: submitted_job.user_id.clone(),
    });

    // Return response
    Ok((StatusCode::CREATED, Json(submitted_job.into())))
}

/// Get job details by ID
#[utoipa::path(
    get,
    path = "/api/v1/jobs/{id}",
    params(
        ("id" = Uuid, Path, description = "Job ID")
    ),
    responses(
        (status = 200, description = "Job details", body = JobResponse),
        (status = 404, description = "Job not found")
    ),
    tag = "jobs"
)]
pub async fn get_job(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
) -> Result<Json<JobResponse>, crate::HpcError> {
    let job = state.scheduler.get_job(id).await?;
    Ok(Json(job.into()))
}

/// List all jobs with optional filtering
#[utoipa::path(
    get,
    path = "/api/v1/jobs",
    params(
        ("state" = Option<String>, Query, description = "Filter by job state"),
        ("user_id" = Option<String>, Query, description = "Filter by user ID"),
        ("priority" = Option<String>, Query, description = "Filter by priority")
    ),
    responses(
        (status = 200, description = "List of jobs", body = JobListResponse),
        (status = 400, description = "Invalid query parameters")
    ),
    tag = "jobs"
)]
pub async fn list_jobs(
    State(state): State<AppState>,
    Query(query): Query<ListJobsQuery>,
) -> Result<Json<JobListResponse>, crate::HpcError> {
    // Parse state filter if provided
    let state_filter = if let Some(state_str) = query.state {
        Some(parse_job_state(&state_str)?)
    } else {
        None
    };

    // Get jobs from scheduler
    let jobs = state.scheduler.list_jobs(state_filter).await?;

    // Apply additional filters
    let mut filtered_jobs = jobs;

    if let Some(user_id) = query.user_id {
        filtered_jobs.retain(|job| job.user_id == user_id);
    }

    if let Some(priority_str) = query.priority {
        let priority = parse_priority(&priority_str)?;
        filtered_jobs.retain(|job| job.priority == priority);
    }

    let total = filtered_jobs.len();
    let job_responses: Vec<JobResponse> = filtered_jobs.into_iter().map(Into::into).collect();

    Ok(Json(JobListResponse {
        jobs: job_responses,
        total,
    }))
}

/// Cancel a job
#[utoipa::path(
    delete,
    path = "/api/v1/jobs/{id}",
    params(
        ("id" = Uuid, Path, description = "Job ID")
    ),
    responses(
        (status = 200, description = "Job cancelled successfully", body = JobResponse),
        (status = 404, description = "Job not found"),
        (status = 400, description = "Job cannot be cancelled")
    ),
    tag = "jobs"
)]
pub async fn cancel_job(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
) -> Result<Json<JobResponse>, crate::HpcError> {
    let cancelled_job = state.scheduler.cancel_job(id).await?;

    // Publish job cancelled event via hpc-channels
    state.publish_job_event(SchedulerMessage::JobCancelled {
        job_id: cancelled_job.id.to_string(),
    });

    Ok(Json(cancelled_job.into()))
}

fn parse_job_state(state_str: &str) -> Result<ModelJobState, crate::HpcError> {
    use crate::SchedulerErrorExt;
    match state_str {
        "Queued" => Ok(ModelJobState::Queued),
        "Scheduled" => Ok(ModelJobState::Scheduled),
        "Running" => Ok(ModelJobState::Running),
        "Preempted" => Ok(ModelJobState::Preempted),
        "Completed" => Ok(ModelJobState::Completed),
        "Failed" => Ok(ModelJobState::Failed),
        "Cancelled" => Ok(ModelJobState::Cancelled),
        _ => Err(crate::HpcError::validation(format!(
            "Invalid job state: {}",
            state_str
        ))),
    }
}

fn parse_priority(priority_str: &str) -> Result<crate::models::Priority, crate::HpcError> {
    use crate::SchedulerErrorExt;
    match priority_str {
        "Low" => Ok(crate::models::Priority::Low),
        "Normal" => Ok(crate::models::Priority::Normal),
        "High" => Ok(crate::models::Priority::High),
        _ => Err(crate::HpcError::validation(format!(
            "Invalid priority: {}",
            priority_str
        ))),
    }
}

/// Submit a new job for a specific user (alternate path)
#[utoipa::path(
    post,
    path = "/api/users/{user_id}/jobs",
    request_body = SubmitJobRequest,
    params(
        ("user_id" = String, Path, description = "User ID")
    ),
    responses(
        (status = 201, description = "Job submitted successfully", body = JobResponse),
        (status = 400, description = "Invalid request")
    ),
    tag = "jobs"
)]
pub async fn submit_user_job(
    State(state): State<AppState>,
    Path(user_id): Path<String>,
    Json(mut request): Json<SubmitJobRequest>,
) -> Result<(StatusCode, Json<JobResponse>), crate::HpcError> {
    // Override user_id from path parameter
    request.user_id = user_id;

    // Use existing submit_job logic
    let job = request.into_job()?;
    let submitted_job = state.scheduler.submit_job(job).await?;

    // Publish job submitted event via hpc-channels
    state.publish_job_event(SchedulerMessage::JobSubmitted {
        job_id: submitted_job.id.to_string(),
        tenant_id: submitted_job.user_id.clone(),
    });

    Ok((StatusCode::CREATED, Json(submitted_job.into())))
}

/// List jobs for a specific user (alternate path)
#[utoipa::path(
    get,
    path = "/api/users/{user_id}/jobs",
    params(
        ("user_id" = String, Path, description = "User ID"),
        ("state" = Option<String>, Query, description = "Filter by job state"),
        ("priority" = Option<String>, Query, description = "Filter by priority")
    ),
    responses(
        (status = 200, description = "List of user jobs", body = JobListResponse),
        (status = 400, description = "Invalid query parameters")
    ),
    tag = "jobs"
)]
pub async fn list_user_jobs(
    State(state): State<AppState>,
    Path(user_id): Path<String>,
    Query(query): Query<ListJobsQuery>,
) -> Result<Json<JobListResponse>, crate::HpcError> {
    // Parse state filter if provided
    let state_filter = if let Some(state_str) = query.state {
        Some(parse_job_state(&state_str)?)
    } else {
        None
    };

    // Get jobs from scheduler
    let jobs = state.scheduler.list_jobs(state_filter).await?;

    // Filter by user_id from path
    let mut filtered_jobs: Vec<_> = jobs.into_iter()
        .filter(|job| job.user_id == user_id)
        .collect();

    // Apply additional filters
    if let Some(priority_str) = query.priority {
        let priority = parse_priority(&priority_str)?;
        filtered_jobs.retain(|job| job.priority == priority);
    }

    let total = filtered_jobs.len();
    let job_responses: Vec<JobResponse> = filtered_jobs.into_iter().map(Into::into).collect();

    Ok(Json(JobListResponse {
        jobs: job_responses,
        total,
    }))
}

/// Get user activity summary
#[utoipa::path(
    get,
    path = "/api/users/{user_id}/activity",
    params(
        ("user_id" = String, Path, description = "User ID")
    ),
    responses(
        (status = 200, description = "User activity summary"),
        (status = 404, description = "User not found")
    ),
    tag = "jobs"
)]
pub async fn get_user_activity(
    State(state): State<AppState>,
    Path(user_id): Path<String>,
) -> Result<Json<serde_json::Value>, crate::HpcError> {
    // Get all jobs for the user
    let all_jobs = state.scheduler.list_jobs(None).await?;
    let user_jobs: Vec<_> = all_jobs.into_iter()
        .filter(|job| job.user_id == user_id)
        .collect();

    // Calculate activity statistics
    let total_jobs = user_jobs.len();
    let running_jobs = user_jobs.iter()
        .filter(|j| matches!(j.state, ModelJobState::Running))
        .count();
    let queued_jobs = user_jobs.iter()
        .filter(|j| matches!(j.state, ModelJobState::Queued))
        .count();
    let completed_jobs = user_jobs.iter()
        .filter(|j| matches!(j.state, ModelJobState::Completed))
        .count();
    let failed_jobs = user_jobs.iter()
        .filter(|j| matches!(j.state, ModelJobState::Failed))
        .count();

    Ok(Json(serde_json::json!({
        "user_id": user_id,
        "total_jobs": total_jobs,
        "running_jobs": running_jobs,
        "queued_jobs": queued_jobs,
        "completed_jobs": completed_jobs,
        "failed_jobs": failed_jobs,
        "recent_jobs": user_jobs.iter()
            .take(10)
            .map(|j| JobResponse::from(j.clone()))
            .collect::<Vec<_>>()
    })))
}
