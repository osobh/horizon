use crate::db::InitiativeRepository;
use crate::error::Result;
use crate::models::*;
use axum::{extract::State, Json};
use std::sync::Arc;

#[derive(Clone)]
pub struct AppState {
    pub repository: InitiativeRepository,
}

pub async fn health() -> &'static str {
    "OK"
}

pub async fn list_initiatives(
    State(state): State<Arc<AppState>>,
) -> Result<Json<Vec<Initiative>>> {
    let initiatives = state.repository.list_initiatives().await?;
    Ok(Json(initiatives))
}

pub async fn get_portfolio(
    State(state): State<Arc<AppState>>,
) -> Result<Json<PortfolioSummary>> {
    let summary = state.repository.get_portfolio_summary().await?;
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
