use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use sqlx::FromRow;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct WasteDetection {
    pub id: Uuid,
    pub detection_type: String,
    pub resource_id: String,
    pub resource_type: String,
    pub detected_at: DateTime<Utc>,
    pub idle_duration: Option<String>,
    pub cost_impact_monthly: Option<Decimal>,
    pub root_cause: Option<String>,
    pub status: String,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct EfficiencyProposal {
    pub id: Uuid,
    pub detection_id: Option<Uuid>,
    pub proposal_type: String,
    pub description: String,
    pub estimated_savings_monthly: Option<Decimal>,
    pub roi_months: Option<Decimal>,
    pub implementation_cost: Option<Decimal>,
    pub status: String,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SavingsSummary {
    pub total_detections: i64,
    pub total_savings: Decimal,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_waste_detection_creation() {
        let det = WasteDetection {
            id: Uuid::new_v4(),
            detection_type: "idle".to_string(),
            resource_id: "test".to_string(),
            resource_type: "gpu".to_string(),
            detected_at: Utc::now(),
            idle_duration: None,
            cost_impact_monthly: Some(Decimal::new(100, 0)),
            root_cause: None,
            status: "detected".to_string(),
            created_at: Utc::now(),
        };
        assert_eq!(det.detection_type, "idle");
    }
}
