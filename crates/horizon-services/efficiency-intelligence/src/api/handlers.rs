use crate::db::EfficiencyRepository;
use crate::error::Result;
use crate::models::*;
use axum::{extract::State, Json};
use std::sync::Arc;

#[derive(Clone)]
pub struct AppState {
    pub repository: EfficiencyRepository,
}

pub async fn health() -> &'static str {
    "OK"
}

pub async fn list_detections(
    State(state): State<Arc<AppState>>,
) -> Result<Json<Vec<WasteDetection>>> {
    let detections = state.repository.list_detections().await?;
    Ok(Json(detections))
}

pub async fn get_summary(
    State(state): State<Arc<AppState>>,
) -> Result<Json<SavingsSummary>> {
    let summary = state.repository.get_savings_summary().await?;
    Ok(Json(summary))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_health() {
        assert_eq!(health().await, "OK");
    }
}
