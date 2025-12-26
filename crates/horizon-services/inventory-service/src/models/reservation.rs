use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;
use uuid::Uuid;

#[derive(Debug, Clone, PartialEq, sqlx::FromRow, Serialize, Deserialize, ToSchema)]
pub struct AssetReservation {
    pub id: Uuid,
    pub asset_id: Uuid,
    pub job_id: Uuid,
    pub reserved_by: String,
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub active: bool,
    pub created_at: DateTime<Utc>,
}

impl AssetReservation {
    pub fn new(
        asset_id: Uuid,
        job_id: Uuid,
        reserved_by: String,
        start_time: DateTime<Utc>,
        end_time: DateTime<Utc>,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            asset_id,
            job_id,
            reserved_by,
            start_time,
            end_time,
            active: true,
            created_at: Utc::now(),
        }
    }

    pub fn is_active_at(&self, time: DateTime<Utc>) -> bool {
        self.active && self.start_time <= time && time < self.end_time
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;

    #[test]
    fn test_asset_reservation_new() {
        let asset_id = Uuid::new_v4();
        let job_id = Uuid::new_v4();
        let start_time = Utc::now();
        let end_time = start_time + Duration::hours(2);

        let reservation = AssetReservation::new(
            asset_id,
            job_id,
            "user@example.com".to_string(),
            start_time,
            end_time,
        );

        assert_eq!(reservation.asset_id, asset_id);
        assert_eq!(reservation.job_id, job_id);
        assert!(reservation.active);
    }

    #[test]
    fn test_asset_reservation_is_active_at() {
        let start_time = Utc::now();
        let end_time = start_time + Duration::hours(2);

        let reservation = AssetReservation::new(
            Uuid::new_v4(),
            Uuid::new_v4(),
            "user".to_string(),
            start_time,
            end_time,
        );

        assert!(reservation.is_active_at(start_time + Duration::hours(1)));
        assert!(!reservation.is_active_at(start_time - Duration::hours(1)));
        assert!(!reservation.is_active_at(end_time + Duration::hours(1)));
    }

    #[test]
    fn test_asset_reservation_serialization() {
        let reservation = AssetReservation::new(
            Uuid::new_v4(),
            Uuid::new_v4(),
            "user".to_string(),
            Utc::now(),
            Utc::now() + Duration::hours(2),
        );

        let json = serde_json::to_string(&reservation).unwrap();
        let deserialized: AssetReservation = serde_json::from_str(&json).unwrap();

        assert_eq!(reservation.id, deserialized.id);
        assert_eq!(reservation.asset_id, deserialized.asset_id);
    }
}
