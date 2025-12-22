use crate::db::VendorRepository;
use crate::error::Result;
use crate::models::*;
use axum::{extract::State, Json};
use std::sync::Arc;

#[derive(Clone)]
pub struct AppState {
    pub repository: VendorRepository,
}

pub async fn health() -> &'static str {
    "OK"
}

pub async fn list_vendors(
    State(state): State<Arc<AppState>>,
) -> Result<Json<Vec<Vendor>>> {
    let vendors = state.repository.list_vendors().await?;
    Ok(Json(vendors))
}

pub async fn get_summary(
    State(state): State<Arc<AppState>>,
) -> Result<Json<VendorSummary>> {
    let summary = state.repository.get_summary().await?;
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
