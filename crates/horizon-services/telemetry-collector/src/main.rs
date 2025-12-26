use anyhow::Result;
use hpc_tracing::TracingConfig;
use telemetry_collector::{collector::TelemetryCollector, config::CollectorConfig};

#[tokio::main]
async fn main() -> Result<()> {
    // Load configuration from environment or config file
    let config: CollectorConfig = match hpc_config::load("telemetry-collector") {
        Ok(cfg) => cfg,
        Err(e) => {
            eprintln!("Failed to load configuration: {}", e);
            eprintln!("Using default configuration...");
            // Use default config but with environment-based overrides
            CollectorConfig::default()
        }
    };

    // Validate configuration
    if let Err(e) = config.validate() {
        eprintln!("Invalid configuration: {}", e);
        eprintln!("Please check your environment variables or config file");
        std::process::exit(1);
    }

    // Initialize tracing
    hpc_tracing::init(TracingConfig {
        service_name: "telemetry-collector".to_string(),
        log_level: config.observability.log_level.clone(),
        otlp_endpoint: None,
    })?;

    tracing::info!(
        "Starting telemetry-collector (version {})",
        env!("CARGO_PKG_VERSION")
    );
    tracing::info!("Listening on {}", config.server.listen_addr);

    // Create and run collector
    let collector = TelemetryCollector::new(config)?;
    collector.run().await
}
