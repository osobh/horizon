use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema, PartialEq)]
pub struct Checkpoint {
    pub id: Uuid,
    pub job_id: Uuid,
    pub state_data: serde_json::Value,
    pub storage_path: String,
    pub size_bytes: u64,
    pub created_at: DateTime<Utc>,
}

impl Checkpoint {
    pub fn new(
        job_id: Uuid,
        state_data: serde_json::Value,
        storage_path: String,
        size_bytes: u64,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            job_id,
            state_data,
            storage_path,
            size_bytes,
            created_at: Utc::now(),
        }
    }
}
