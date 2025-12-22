use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use utoipa::ToSchema;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct CreatePolicyRequest {
    pub name: String,
    pub content: String,
    pub description: Option<String>,
    pub created_by: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct UpdatePolicyRequest {
    pub content: String,
    pub description: Option<String>,
    pub created_by: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct PolicyResponse {
    pub id: Uuid,
    pub name: String,
    pub version: i32,
    pub content: String,
    pub description: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub created_by: String,
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct PolicyVersionResponse {
    pub id: Uuid,
    pub policy_id: Uuid,
    pub version: i32,
    pub content: String,
    pub created_at: DateTime<Utc>,
    pub created_by: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct EvaluateRequest {
    pub principal: Principal,
    pub resource: Resource,
    pub action: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct Principal {
    pub user_id: String,
    #[serde(default)]
    pub roles: Vec<String>,
    #[serde(default)]
    pub teams: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct Resource {
    #[serde(rename = "type")]
    pub resource_type: String,
    pub id: String,
    #[serde(default)]
    pub attributes: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct EvaluateResponse {
    pub decision: String,
    pub matched_policy: Option<String>,
    pub evaluation_time_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
}
