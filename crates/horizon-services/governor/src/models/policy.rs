use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::FromRow;
use utoipa::ToSchema;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize, FromRow, ToSchema)]
pub struct Policy {
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

#[derive(Debug, Clone, Serialize, Deserialize, FromRow, ToSchema)]
pub struct PolicyVersion {
    pub id: Uuid,
    pub policy_id: Uuid,
    pub version: i32,
    pub content: String,
    pub created_at: DateTime<Utc>,
    pub created_by: String,
}
