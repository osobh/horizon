use axum::{
    extract::{Query, State},
    http::{header, StatusCode},
    response::{IntoResponse, Response},
};
use chrono::{DateTime, Utc};
use serde::Deserialize;
use std::sync::Arc;

use crate::api::AppState;

#[derive(Debug, Deserialize)]
pub struct ExportQuery {
    pub start_date: DateTime<Utc>,
    pub end_date: DateTime<Utc>,
    pub team_id: Option<String>,
    pub user_id: Option<String>,
    pub customer_id: Option<String>,
}

pub async fn export_csv(
    State(state): State<Arc<AppState>>,
    Query(query): Query<ExportQuery>,
) -> Result<Response, (StatusCode, String)> {
    let attributions = state
        .repository
        .get_attributions(
            query.start_date,
            query.end_date,
            query.team_id.as_deref(),
            query.user_id.as_deref(),
            query.customer_id.as_deref(),
        )
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let csv_data = state
        .csv_exporter
        .export(&attributions)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok((
        [(header::CONTENT_TYPE, "text/csv")],
        csv_data,
    )
        .into_response())
}

pub async fn export_json(
    State(state): State<Arc<AppState>>,
    Query(query): Query<ExportQuery>,
) -> Result<Response, (StatusCode, String)> {
    let attributions = state
        .repository
        .get_attributions(
            query.start_date,
            query.end_date,
            query.team_id.as_deref(),
            query.user_id.as_deref(),
            query.customer_id.as_deref(),
        )
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let json_data = state
        .json_exporter
        .export(&attributions)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok((
        [(header::CONTENT_TYPE, "application/json")],
        json_data,
    )
        .into_response())
}

pub async fn export_markdown(
    State(state): State<Arc<AppState>>,
    Query(query): Query<ExportQuery>,
) -> Result<Response, (StatusCode, String)> {
    let attributions = state
        .repository
        .get_attributions(
            query.start_date,
            query.end_date,
            query.team_id.as_deref(),
            query.user_id.as_deref(),
            query.customer_id.as_deref(),
        )
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let markdown_data = state
        .markdown_exporter
        .export(&attributions)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok((
        [(header::CONTENT_TYPE, "text/markdown")],
        markdown_data,
    )
        .into_response())
}
