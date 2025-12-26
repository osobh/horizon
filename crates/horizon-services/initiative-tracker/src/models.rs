use chrono::{DateTime, NaiveDate, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use sqlx::FromRow;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct Initiative {
    pub id: Uuid,
    pub name: String,
    pub description: Option<String>,
    pub owner: String,
    pub category: String,
    pub projected_impact: Option<Decimal>,
    pub actual_impact: Option<Decimal>,
    pub status: String,
    pub priority: String,
    pub start_date: Option<NaiveDate>,
    pub target_date: Option<NaiveDate>,
    pub completion_date: Option<NaiveDate>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct Milestone {
    pub id: Uuid,
    pub initiative_id: Uuid,
    pub name: String,
    pub description: Option<String>,
    pub target_date: NaiveDate,
    pub completion_date: Option<NaiveDate>,
    pub status: String,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioSummary {
    pub total_initiatives: i64,
    pub in_progress: i64,
    pub completed: i64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initiative_creation() {
        let init = Initiative {
            id: Uuid::new_v4(),
            name: "Test".to_string(),
            description: None,
            owner: "test".to_string(),
            category: "cost_reduction".to_string(),
            projected_impact: None,
            actual_impact: None,
            status: "draft".to_string(),
            priority: "medium".to_string(),
            start_date: None,
            target_date: None,
            completion_date: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };
        assert_eq!(init.name, "Test");
    }
}
