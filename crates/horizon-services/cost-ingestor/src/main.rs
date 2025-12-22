use cost_ingestor::{
    api::{create_router, AppState},
    config::Config,
    db::{create_pool, run_migrations, BillingRepository},
    ingest::{AwsCurNormalizer, AzureEaNormalizer, GcpBillingNormalizer, OnPremNormalizer},
    models::Provider,
    normalize::NormalizedBillingSchema,
};
use std::sync::Arc;
use tracing::info;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    info!("Starting Cost Ingestor service");

    let config = Config::from_env().unwrap_or_else(|_| {
        info!("Using default configuration");
        Config::default()
    });

    info!("Connecting to database: {}", config.database.url);
    let pool = create_pool(&config).await?;

    info!("Running database migrations");
    run_migrations(&pool).await?;

    let repository = BillingRepository::new(pool);

    let mut schema = NormalizedBillingSchema::new();
    schema.register_normalizer(Provider::Aws, Box::new(AwsCurNormalizer::new()));
    schema.register_normalizer(Provider::Gcp, Box::new(GcpBillingNormalizer::new()));
    schema.register_normalizer(Provider::Azure, Box::new(AzureEaNormalizer::new()));
    schema.register_normalizer(Provider::OnPrem, Box::new(OnPremNormalizer::new()));

    let state = Arc::new(AppState {
        repository,
        schema,
    });

    let app = create_router(state);

    let addr = config.server_address();
    info!("Starting server on {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
