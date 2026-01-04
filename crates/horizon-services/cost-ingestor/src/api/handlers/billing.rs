use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    Json,
};
use hpc_channels::CostMessage;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use uuid::Uuid;

use crate::api::state::AppState;
use crate::error::{HpcError, Result};
use crate::models::{BillingRecord, BillingRecordQuery, CreateBillingRecord, Provider};
use crate::normalize::RawBillingData;

#[derive(Debug, Serialize, Deserialize)]
pub struct IngestRequest {
    pub provider: Provider,
    pub data: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct IngestResponse {
    pub ingested_count: usize,
    pub records: Vec<BillingRecord>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct QueryParams {
    pub provider: Option<String>,
    pub account_id: Option<String>,
    pub service: Option<String>,
    pub resource_id: Option<String>,
    pub start_date: Option<String>,
    pub end_date: Option<String>,
    pub limit: Option<i64>,
    pub offset: Option<i64>,
}

pub async fn ingest_billing_data(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<IngestRequest>,
) -> Result<Json<IngestResponse>> {
    let provider_name = format!("{:?}", payload.provider).to_lowercase();

    let raw = RawBillingData {
        provider: payload.provider,
        data: payload.data,
    };

    let normalized = state.schema.normalize(&raw)?;

    let records = state.repository.create_batch(&normalized).await?;
    let ingested_count = records.len();

    // Calculate total amount from records
    let total_amount: f64 = records
        .iter()
        .map(|r| r.amount.try_into().unwrap_or(0.0))
        .sum();

    // Get current timestamp
    let timestamp_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);

    // Publish ingestion event
    state.publish_ingestion_event(CostMessage::BillingDataIngested {
        provider: provider_name,
        record_count: ingested_count,
        total_amount,
        timestamp_ms,
    });

    Ok(Json(IngestResponse {
        ingested_count,
        records,
    }))
}

pub async fn create_billing_record(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<CreateBillingRecord>,
) -> Result<(StatusCode, Json<BillingRecord>)> {
    let provider_name = format!("{:?}", payload.provider).to_lowercase();
    let amount: f64 = payload.amount.try_into().unwrap_or(0.0);

    payload.validate()?;
    let record = state.repository.create(&payload).await?;

    // Publish record created event
    state.publish_ingestion_event(CostMessage::BillingRecordCreated {
        record_id: record.id.to_string(),
        provider: provider_name,
        amount,
    });

    Ok((StatusCode::CREATED, Json(record)))
}

pub async fn get_billing_record(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> Result<Json<BillingRecord>> {
    let record = state.repository.find_by_id(id).await?.ok_or_else(|| {
        HpcError::not_found(
            "billing_record",
            format!("Billing record not found: {}", id),
        )
    })?;

    Ok(Json(record))
}

pub async fn query_billing_records(
    State(state): State<Arc<AppState>>,
    Query(params): Query<QueryParams>,
) -> Result<Json<Vec<BillingRecord>>> {
    let provider = if let Some(p) = params.provider {
        Some(p.parse::<Provider>()?)
    } else {
        None
    };

    let start_date = if let Some(s) = params.start_date {
        Some(crate::normalize::parse_iso_datetime(&s)?)
    } else {
        None
    };

    let end_date = if let Some(e) = params.end_date {
        Some(crate::normalize::parse_iso_datetime(&e)?)
    } else {
        None
    };

    let query = BillingRecordQuery {
        provider,
        account_id: params.account_id,
        service: params.service,
        resource_id: params.resource_id,
        start_date,
        end_date,
        limit: params.limit,
        offset: params.offset,
    };

    let records = state.repository.query(&query).await?;
    Ok(Json(records))
}

pub async fn delete_billing_record(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> Result<StatusCode> {
    let deleted = state.repository.delete_by_id(id).await?;
    if deleted {
        // Publish record deleted event
        state.publish_ingestion_event(CostMessage::BillingRecordDeleted {
            record_id: id.to_string(),
        });
        Ok(StatusCode::NO_CONTENT)
    } else {
        Err(HpcError::not_found(
            "billing_record",
            format!("Billing record not found: {}", id),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::BillingRepository;
    use crate::normalize::NormalizedBillingSchema;
    use chrono::Utc;
    use rust_decimal_macros::dec;

    #[test]
    fn test_ingest_request_serialization() {
        let request = IngestRequest {
            provider: Provider::Aws,
            data: serde_json::json!({"test": "data"}),
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("aws"));
    }

    #[test]
    fn test_query_params_parsing() {
        let params = QueryParams {
            provider: Some("aws".to_string()),
            account_id: Some("123".to_string()),
            service: None,
            resource_id: None,
            start_date: None,
            end_date: None,
            limit: Some(10),
            offset: Some(0),
        };

        assert_eq!(params.provider, Some("aws".to_string()));
        assert_eq!(params.limit, Some(10));
    }

    #[tokio::test]
    #[ignore]
    async fn test_create_billing_record_handler() {
        let database_url = std::env::var("TEST_DATABASE_URL")
            .unwrap_or_else(|_| "postgres://localhost/horizon_cost_test".to_string());

        let pool = sqlx::PgPool::connect(&database_url).await.unwrap();
        let repository = BillingRepository::new(pool.clone());
        let schema = NormalizedBillingSchema::new();

        let state = Arc::new(AppState::new(repository, schema));

        let now = Utc::now();
        let later = now + chrono::Duration::hours(1);
        let record = CreateBillingRecord::new(Provider::Aws, now, later, dec!(50.0));

        let result = create_billing_record(State(state), Json(record)).await;
        assert!(result.is_ok());

        sqlx::query("DELETE FROM raw_billing_records")
            .execute(&pool)
            .await
            .unwrap();
    }
}
