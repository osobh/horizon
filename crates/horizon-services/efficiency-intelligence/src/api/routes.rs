use super::handlers::*;
use axum::{routing::get, Router};
use std::sync::Arc;

pub fn create_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/api/v1/detections", get(list_detections))
        .route("/api/v1/savings/summary", get(get_summary))
        .with_state(state)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_create_router() {
        let pool = sqlx::PgPool::connect_lazy("postgres://localhost/test").unwrap();
        let repo = crate::db::EfficiencyRepository::new(pool);
        let state = Arc::new(AppState { repository: repo });
        let _router = create_router(state);
    }
}
