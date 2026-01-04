use crate::api::dto::{
    CreatePolicyRequest, PolicyResponse, PolicyVersionResponse, UpdatePolicyRequest,
};
use crate::api::routes::AppState;
use crate::error::Result;
use crate::models::Policy;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::Json;
use hpc_channels::GovernorMessage;
use std::sync::Arc;

#[utoipa::path(
    post,
    path = "/api/v1/policies",
    request_body = CreatePolicyRequest,
    responses(
        (status = 201, description = "Policy created successfully", body = PolicyResponse),
        (status = 400, description = "Invalid request"),
        (status = 409, description = "Policy already exists")
    )
)]
pub async fn create_policy(
    State(state): State<Arc<AppState>>,
    Json(request): Json<CreatePolicyRequest>,
) -> Result<(StatusCode, Json<PolicyResponse>)> {
    let created_by = request.created_by.clone();

    let policy = state
        .repo
        .create(
            &request.name,
            &request.content,
            request.description.as_deref(),
            &request.created_by,
        )
        .await?;

    // Publish policy created event
    state.publish_policy_event(GovernorMessage::PolicyCreated {
        policy_name: policy.name.clone(),
        version: policy.version,
        created_by,
    });

    Ok((StatusCode::CREATED, Json(policy_to_response(policy))))
}

#[utoipa::path(
    get,
    path = "/api/v1/policies",
    responses(
        (status = 200, description = "List of policies", body = Vec<PolicyResponse>)
    )
)]
pub async fn list_policies(
    State(state): State<Arc<AppState>>,
) -> Result<Json<Vec<PolicyResponse>>> {
    let policies = state.repo.list(false).await?;
    Ok(Json(policies.into_iter().map(policy_to_response).collect()))
}

#[utoipa::path(
    get,
    path = "/api/v1/policies/{name}",
    params(
        ("name" = String, Path, description = "Policy name")
    ),
    responses(
        (status = 200, description = "Policy found", body = PolicyResponse),
        (status = 404, description = "Policy not found")
    )
)]
pub async fn get_policy(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> Result<Json<PolicyResponse>> {
    let policy = state.repo.get_by_name(&name).await?;
    Ok(Json(policy_to_response(policy)))
}

#[utoipa::path(
    put,
    path = "/api/v1/policies/{name}",
    params(
        ("name" = String, Path, description = "Policy name")
    ),
    request_body = UpdatePolicyRequest,
    responses(
        (status = 200, description = "Policy updated successfully", body = PolicyResponse),
        (status = 404, description = "Policy not found")
    )
)]
pub async fn update_policy(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    Json(request): Json<UpdatePolicyRequest>,
) -> Result<Json<PolicyResponse>> {
    // Get the current version before updating
    let previous_policy = state.repo.get_by_name(&name).await?;
    let previous_version = previous_policy.version;
    let updated_by = request.created_by.clone();

    let policy = state
        .repo
        .update(
            &name,
            &request.content,
            request.description.as_deref(),
            &request.created_by,
        )
        .await?;

    // Publish policy updated event
    state.publish_policy_event(GovernorMessage::PolicyUpdated {
        policy_name: policy.name.clone(),
        version: policy.version,
        previous_version,
        updated_by,
    });

    Ok(Json(policy_to_response(policy)))
}

#[utoipa::path(
    delete,
    path = "/api/v1/policies/{name}",
    params(
        ("name" = String, Path, description = "Policy name")
    ),
    responses(
        (status = 204, description = "Policy deleted successfully"),
        (status = 404, description = "Policy not found")
    )
)]
pub async fn delete_policy(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> Result<StatusCode> {
    state.repo.delete(&name).await?;

    // Publish policy deleted event
    state.publish_policy_event(GovernorMessage::PolicyDeleted { policy_name: name });

    Ok(StatusCode::NO_CONTENT)
}

#[utoipa::path(
    get,
    path = "/api/v1/policies/{name}/versions",
    params(
        ("name" = String, Path, description = "Policy name")
    ),
    responses(
        (status = 200, description = "Policy versions", body = Vec<PolicyVersionResponse>),
        (status = 404, description = "Policy not found")
    )
)]
pub async fn get_policy_versions(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> Result<Json<Vec<PolicyVersionResponse>>> {
    let versions = state.repo.get_versions(&name).await?;
    Ok(Json(
        versions
            .into_iter()
            .map(|v| PolicyVersionResponse {
                id: v.id,
                policy_id: v.policy_id,
                version: v.version,
                content: v.content,
                created_at: v.created_at,
                created_by: v.created_by,
            })
            .collect(),
    ))
}

fn policy_to_response(policy: Policy) -> PolicyResponse {
    PolicyResponse {
        id: policy.id,
        name: policy.name,
        version: policy.version,
        content: policy.content,
        description: policy.description,
        created_at: policy.created_at,
        updated_at: policy.updated_at,
        created_by: policy.created_by,
        enabled: policy.enabled,
    }
}
