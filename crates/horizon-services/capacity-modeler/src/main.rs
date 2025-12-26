use anyhow::Result;
use capacity_modeler::{api::create_routes, api::handlers::AppState, Config, ForecastService};
use std::net::SocketAddr;
use std::sync::Arc;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "capacity_modeler=info,tower_http=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let config = Config::default();

    tracing::info!(
        listen_addr = %config.listen_addr,
        min_historical_days = config.min_historical_days,
        "Starting capacity-modeler service"
    );

    // Create forecast service and application state
    let service = ForecastService::new(config.min_historical_days);
    let state = Arc::new(AppState::new(service));

    // Create router with application state
    let app = create_routes().with_state(state);

    let addr: SocketAddr = config.listen_addr.parse()?;
    tracing::info!(?addr, "Capacity modeler listening");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
