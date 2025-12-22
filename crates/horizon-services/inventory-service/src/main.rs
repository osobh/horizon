use anyhow::Result;
use std::net::SocketAddr;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use inventory_service::{api::create_routes, config::Config, db::create_pool};

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "inventory_service=info,tower_http=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let config = Config::default();

    tracing::info!(
        database_url = %config.database_url,
        listen_addr = %config.listen_addr,
        "Starting inventory-service"
    );

    let pool = create_pool(&config.database_url).await?;

    tracing::info!("Running database migrations");
    sqlx::migrate!("./migrations").run(&pool).await?;

    tracing::info!("Database migrations completed");

    let app = create_routes(pool);

    let addr: SocketAddr = config.listen_addr.parse()?;
    tracing::info!(?addr, "Inventory service listening");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
