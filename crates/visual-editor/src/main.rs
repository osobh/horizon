use anyhow::Result;
use clap::Parser;
use stratoswarm_visual_editor::server::run_server;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[derive(Parser)]
#[command(name = "visual-editor")]
#[command(about = "StratoSwarm Visual Editor Server")]
#[command(version)]
struct Cli {
    /// Server bind address
    #[arg(long, default_value = "0.0.0.0")]
    host: String,

    /// Server port
    #[arg(long, default_value = "8080")]
    port: u16,

    /// Enable debug logging
    #[arg(long)]
    debug: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialize tracing
    let log_level = if cli.debug {
        tracing::Level::DEBUG
    } else {
        std::env::var("RUST_LOG")
            .unwrap_or_else(|_| "info".to_string())
            .parse()
            .unwrap_or(tracing::Level::INFO)
    };

    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| {
                    tracing_subscriber::EnvFilter::new(format!("stratoswarm_visual_editor={}", log_level))
                }),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    // Get configuration from environment or CLI
    let host = std::env::var("VISUAL_EDITOR_HOST").unwrap_or(cli.host);
    let port = std::env::var("VISUAL_EDITOR_PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(cli.port);

    let addr = format!("{}:{}", host, port);

    tracing::info!("Starting StratoSwarm Visual Editor");
    tracing::info!("Server will bind to: {}", addr);
    tracing::info!("Health endpoint: http://{}/health", addr);
    tracing::info!("GraphQL playground: http://{}/graphql/playground", addr);

    run_server(&addr).await?;

    Ok(())
}