use horizon_quota_manager::{create_router, Config, DbPool, QuotaRepository};
use std::net::SocketAddr;
use tracing::info;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let config = Config::default();

    info!("Connecting to database...");
    let db_pool = DbPool::new(&config.database).await?;

    info!("Running database migrations...");
    sqlx::migrate!("./migrations")
        .run(db_pool.inner())
        .await?;

    info!("Initializing quota repository...");
    let repository = QuotaRepository::new(db_pool.inner().clone());

    info!("Creating router...");
    let app = create_router(repository);

    let addr = SocketAddr::from(([0, 0, 0, 0], config.server.port));
    info!("Starting server on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
