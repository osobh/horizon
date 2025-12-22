use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;
use uuid::Uuid;

use super::{AssetStatus, ChangeOperation};

#[derive(Debug, Clone, PartialEq, sqlx::FromRow, Serialize, Deserialize, ToSchema)]
pub struct AssetHistory {
    pub id: i64,
    pub asset_id: Uuid,
    pub operation: ChangeOperation,
    pub previous_status: Option<AssetStatus>,
    #[sqlx(json)]
    pub previous_metadata: Option<serde_json::Value>,
    pub new_status: Option<AssetStatus>,
    #[sqlx(json)]
    pub new_metadata: Option<serde_json::Value>,
    #[sqlx(json)]
    pub metadata_delta: Option<serde_json::Value>,
    pub changed_at: DateTime<Utc>,
    pub changed_by: String,
    pub reason: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_asset_history_serialization() {
        let history = AssetHistory {
            id: 1,
            asset_id: Uuid::new_v4(),
            operation: ChangeOperation::StatusChanged,
            previous_status: Some(AssetStatus::Available),
            previous_metadata: Some(serde_json::json!({})),
            new_status: Some(AssetStatus::Allocated),
            new_metadata: Some(serde_json::json!({"job_id": "job-123"})),
            metadata_delta: Some(serde_json::json!({"job_id": "job-123"})),
            changed_at: Utc::now(),
            changed_by: "scheduler".to_string(),
            reason: Some("Job allocation".to_string()),
        };

        let json = serde_json::to_string(&history).unwrap();
        let deserialized: AssetHistory = serde_json::from_str(&json).unwrap();

        assert_eq!(history.id, deserialized.id);
        assert_eq!(history.operation, deserialized.operation);
    }
}
