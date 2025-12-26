use cost_attributor::{api, config::Config, db, error::Result};
use tracing::{info, warn};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_target(false)
        .compact()
        .init();

    info!("Starting cost-attributor service");

    // Load configuration
    let config = Config::from_env().unwrap_or_else(|e| {
        warn!("Failed to load configuration from environment: {}", e);
        info!("Using default configuration");
        Config::default()
    });
    config.validate()?;

    info!(
        "Loaded configuration: server={}:{}",
        config.server.host, config.server.port
    );

    // Create database pool
    let pool = match db::create_pool(&config.database.url, config.database.max_connections).await {
        Ok(pool) => {
            info!("Database connection pool created");
            pool
        }
        Err(e) => {
            warn!("Failed to create database pool: {}", e);
            warn!("Service will start but database operations will fail");
            // In production, you might want to exit here
            // For now, we'll continue to allow the service to start
            return Err(e);
        }
    };

    // Run migrations
    info!("Running database migrations");
    sqlx::migrate!("./migrations")
        .run(&pool)
        .await
        .map_err(|e| hpc_error::HpcError::internal(format!("Migration failed: {}", e)))?;

    // Create repository
    let repository = db::Repository::new(pool);

    // Create app state
    let state = api::AppState::new(repository, config.clone());

    // Create router
    let app = api::create_router(state);

    // Start server
    let addr = format!("{}:{}", config.server.host, config.server.port);
    info!("Starting server on {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .map_err(|e| hpc_error::HpcError::internal(e.to_string()))?;

    info!("Server listening on {}", addr);

    axum::serve(listener, app)
        .await
        .map_err(|e| hpc_error::HpcError::internal(e.to_string()))?;

    Ok(())
}
