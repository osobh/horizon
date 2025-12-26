use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use sqlx::FromRow;
use utoipa::ToSchema;
use uuid::Uuid;

use super::ResourceType;

#[derive(Debug, Clone, FromRow, Serialize, Deserialize, ToSchema)]
pub struct Allocation {
    pub id: Uuid,
    pub quota_id: Uuid,
    pub job_id: Uuid,
    pub resource_type: ResourceType,
    pub allocated_value: Decimal,
    pub allocated_at: DateTime<Utc>,
    pub released_at: Option<DateTime<Utc>>,
    pub version: i32,
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct CreateAllocationRequest {
    pub quota_id: Uuid,
    pub job_id: Uuid,
    pub resource_type: ResourceType,
    pub allocated_value: Decimal,
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct AllocationCheckRequest {
    pub entity_type: String,
    pub entity_id: String,
    pub resource_type: ResourceType,
    pub requested_value: Decimal,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct AllocationCheckResponse {
    pub allowed: bool,
    pub available: Decimal,
    pub requested: Decimal,
    pub reason: Option<String>,
}

#[derive(Debug, Clone, FromRow, Serialize, Deserialize, ToSchema)]
pub struct UsageHistory {
    pub id: Uuid,
    pub quota_id: Uuid,
    pub entity_type: String,
    pub entity_id: String,
    pub resource_type: ResourceType,
    pub allocated_value: Decimal,
    pub operation: OperationType,
    pub job_id: Option<Uuid>,
    pub timestamp: DateTime<Utc>,
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, sqlx::Type, ToSchema)]
#[sqlx(type_name = "text", rename_all = "lowercase")]
#[serde(rename_all = "lowercase")]
pub enum OperationType {
    Allocate,
    Release,
}

impl OperationType {
    pub fn as_str(&self) -> &'static str {
        match self {
            OperationType::Allocate => "allocate",
            OperationType::Release => "release",
        }
    }
}

impl Allocation {
    pub fn is_active(&self) -> bool {
        self.released_at.is_none()
    }

    pub fn duration(&self) -> Option<chrono::Duration> {
        self.released_at
            .map(|released| released.signed_duration_since(self.allocated_at))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_allocation_is_active() {
        let mut allocation = create_test_allocation();
        assert!(allocation.is_active());

        allocation.released_at = Some(Utc::now());
        assert!(!allocation.is_active());
    }

    #[test]
    fn test_allocation_duration() {
        let mut allocation = create_test_allocation();
        assert!(allocation.duration().is_none());

        let allocated_at = Utc::now();
        let released_at = allocated_at + chrono::Duration::hours(2);
        allocation.allocated_at = allocated_at;
        allocation.released_at = Some(released_at);

        let duration = allocation.duration().unwrap();
        assert_eq!(duration.num_hours(), 2);
    }

    #[test]
    fn test_operation_type_as_str() {
        assert_eq!(OperationType::Allocate.as_str(), "allocate");
        assert_eq!(OperationType::Release.as_str(), "release");
    }

    fn create_test_allocation() -> Allocation {
        Allocation {
            id: Uuid::new_v4(),
            quota_id: Uuid::new_v4(),
            job_id: Uuid::new_v4(),
            resource_type: ResourceType::GpuHours,
            allocated_value: dec!(10.0),
            allocated_at: Utc::now(),
            released_at: None,
            version: 0,
            metadata: None,
        }
    }
}
