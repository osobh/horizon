use crate::api::dto::{EvaluateRequest, EvaluateResponse};
use crate::db::PolicyRepository;
use crate::error::{GovernorErrorExt, HpcError, Result};
use axum::extract::State;
use axum::Json;
use hpc_policy::{parse_policy, evaluate as evaluate_policy, Decision, EvaluationContext, PrincipalContext, ResourceContext};
use std::time::Instant;

#[utoipa::path(
    post,
    path = "/api/v1/evaluate",
    request_body = EvaluateRequest,
    responses(
        (status = 200, description = "Evaluation result", body = EvaluateResponse),
        (status = 400, description = "Invalid request")
    )
)]
pub async fn evaluate(
    State(repo): State<PolicyRepository>,
    Json(request): Json<EvaluateRequest>,
) -> Result<Json<EvaluateResponse>> {
    let start = Instant::now();

    let policies = repo.get_all_enabled_policies().await?;

    let mut matched_policy_name: Option<String> = None;
    let mut final_decision = Decision::Deny;

    for db_policy in &policies {
        let policy = parse_policy(&db_policy.content)
            .map_err(|e| HpcError::invalid_policy_content(format!("Failed to parse policy {}: {}", db_policy.name, e)))?;

        let principal = PrincipalContext::new(
            Some(request.principal.user_id.clone()),
            request.principal.roles.clone(),
            request.principal.teams.clone(),
        );

        let mut resource = ResourceContext::new(
            request.resource.resource_type.clone(),
            request.resource.id.clone(),
        );

        for (key, value) in &request.resource.attributes {
            resource = resource.with_attribute(key.clone(), value.clone());
        }

        let context = EvaluationContext::new(
            principal,
            resource,
            request.action.clone(),
        );

        let decision = evaluate_policy(&policy, &context)
            .map_err(|e| HpcError::evaluation_error(format!("Policy evaluation failed: {}", e)))?;

        if decision == Decision::Allow {
            final_decision = Decision::Allow;
            matched_policy_name = Some(db_policy.name.clone());
            break;
        }
    }

    let evaluation_time = start.elapsed();
    let evaluation_time_ms = evaluation_time.as_secs_f64() * 1000.0;

    let response = EvaluateResponse {
        decision: match final_decision {
            Decision::Allow => "allow".to_string(),
            Decision::Deny => "deny".to_string(),
        },
        matched_policy: matched_policy_name,
        evaluation_time_ms,
    };

    Ok(Json(response))
}
