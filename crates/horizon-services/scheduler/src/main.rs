use scheduler::{api::create_router, config::Config, scheduler::Scheduler};
use sqlx::PgPool;
use std::net::SocketAddr;
use std::sync::Arc;
use tower_http::trace::TraceLayer;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "scheduler=debug,tower_http=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    tracing::info!("Starting Horizon Scheduler Service");

    // Load configuration
    let config = Config::from_env()?;
    tracing::info!(
        "Configuration loaded - Server: {}:{}, Database: {}",
        config.server.host,
        config.server.port,
        config.database.url
    );

    // Create database pool
    let pool = PgPool::connect(&config.database.url).await?;
    tracing::info!("Database connection pool established");

    // Run migrations
    sqlx::migrate!("./migrations").run(&pool).await?;
    tracing::info!("Database migrations completed");

    // Create scheduler
    let scheduler = Arc::new(Scheduler::new(config.clone(), pool.clone()).await?);
    tracing::info!("Scheduler initialized");

    // Create OpenAPI documentation
    #[derive(OpenApi)]
    #[openapi(
        paths(
            scheduler::api::handlers::jobs::submit_job,
            scheduler::api::handlers::jobs::get_job,
            scheduler::api::handlers::jobs::list_jobs,
            scheduler::api::handlers::jobs::cancel_job,
            scheduler::api::handlers::queue::get_queue_status,
            scheduler::api::handlers::health::health_check,
        ),
        components(
            schemas(
                scheduler::api::dto::SubmitJobRequest,
                scheduler::api::dto::JobResponse,
                scheduler::api::dto::JobListResponse,
                scheduler::api::dto::QueueStatusResponse,
                scheduler::api::dto::HealthResponse,
                scheduler::api::dto::ResourceResponseDto,
                scheduler::models::JobState,
                scheduler::models::job::PrioritySchema,
            )
        ),
        tags(
            (name = "jobs", description = "Job management endpoints"),
            (name = "queue", description = "Queue management endpoints"),
            (name = "health", description = "Health check endpoints")
        ),
        info(
            title = "Horizon Scheduler API",
            version = "1.0.0",
            description = "GPU job scheduling service with fair-share and preemption support",
            contact(
                name = "Horizon Team",
                email = "team@horizon.dev"
            )
        )
    )]
    struct ApiDoc;

    // Create router with all endpoints
    let api_router = create_router(scheduler.clone());

    // Add Swagger UI
    let app = api_router
        .merge(SwaggerUi::new("/swagger-ui").url("/api-docs/openapi.json", ApiDoc::openapi()))
        .layer(TraceLayer::new_for_http());

    // Start server
    let addr = SocketAddr::from(([0, 0, 0, 0], config.server.port));
    tracing::info!("Starting server on {}", addr);
    tracing::info!("Swagger UI available at http://{}/swagger-ui", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
