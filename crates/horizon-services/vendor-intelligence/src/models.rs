use chrono::{DateTime, NaiveDate, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use sqlx::FromRow;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct Vendor {
    pub id: Uuid,
    pub name: String,
    pub type_: String,
    pub website: Option<String>,
    pub primary_contact: Option<String>,
    pub email: Option<String>,
    pub performance_score: Option<Decimal>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct Contract {
    pub id: Uuid,
    pub vendor_id: Option<Uuid>,
    pub contract_number: Option<String>,
    pub type_: String,
    pub start_date: NaiveDate,
    pub end_date: NaiveDate,
    pub renewal_date: Option<NaiveDate>,
    pub committed_amount: Option<Decimal>,
    pub total_value: Option<Decimal>,
    pub status: String,
    pub auto_renew: Option<bool>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VendorSummary {
    pub total_vendors: i64,
    pub active_contracts: i64,
    pub total_value: Decimal,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vendor_creation() {
        let vendor = Vendor {
            id: Uuid::new_v4(),
            name: "Test Vendor".to_string(),
            type_: "cloud_provider".to_string(),
            website: None,
            primary_contact: None,
            email: None,
            performance_score: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };
        assert_eq!(vendor.name, "Test Vendor");
    }
}
