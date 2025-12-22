//! Genesis Loader Binary
//!
//! This is the minimal bootstrap binary that initializes the ExoRust agent ecosystem
//! from nothing. It's designed to be as small and reliable as possible.

use anyhow::Result;
use clap::Parser;
use exorust_bootstrap::{initialize_bootstrap, BootstrapConfig, BootstrapPhase, GenesisLoader};
use std::path::PathBuf;
use tracing::{error, info};

#[derive(clap::Parser)]
#[command(
    name = "genesis-loader",
    about = "ExoRust Genesis Loader - Bootstrap the agent ecosystem",
    long_about = "Initializes the ExoRust agent ecosystem from scratch using template agents and evolutionary processes."
)]
struct Args {
    /// Configuration file path
    #[arg(short, long, default_value = "bootstrap.json")]
    config: PathBuf,

    /// Log level
    #[arg(short, long, default_value = "info")]
    log_level: String,

    /// Skip GPU initialization (use CPU-only mode)
    #[arg(long)]
    no_gpu: bool,

    /// Dry run mode (validate configuration without starting)
    #[arg(long)]
    dry_run: bool,

    /// Export default configuration to file
    #[arg(long)]
    export_default_config: Option<PathBuf>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // Initialize logging
    init_logging(&args.log_level)?;

    info!("ExoRust Genesis Loader starting...");
    info!("Version: {}", env!("CARGO_PKG_VERSION"));

    // Export default configuration if requested
    if let Some(config_path) = args.export_default_config {
        export_default_config(&config_path).await?;
        info!(
            "Default configuration exported to: {}",
            config_path.display()
        );
        return Ok(());
    }

    // Load configuration
    let config = load_configuration(&args.config).await?;

    // Validate configuration
    config.validate()?;
    info!("Configuration validated successfully");

    if args.dry_run {
        info!("Dry run completed successfully");
        return Ok(());
    }

    // Initialize core systems
    info!("Initializing core systems...");
    initialize_bootstrap().await?;

    if !args.no_gpu {
        info!("GPU mode enabled");
    } else {
        info!("CPU-only mode enabled");
    }

    // Create genesis loader
    let mut genesis_loader = GenesisLoader::with_config(config)?;

    // Execute bootstrap sequence
    info!("Starting bootstrap sequence...");
    let result = match genesis_loader.bootstrap().await {
        Ok(result) => result,
        Err(e) => {
            error!("Bootstrap failed: {}", e);
            std::process::exit(1);
        }
    };

    // Report results
    info!("Bootstrap sequence completed successfully!");
    info!("Final phase: {:?}", result.final_phase);
    info!("Agents created: {}", result.agents_created);
    info!("System is now self-sustaining");

    // Keep the system running
    if result.final_phase == BootstrapPhase::SelfSustaining {
        info!("Entering autonomous operation mode...");
        info!("Press Ctrl+C to shutdown");

        // Wait for shutdown signal
        tokio::signal::ctrl_c().await?;
        info!("Shutdown signal received, terminating...");
    }

    Ok(())
}

/// Initialize structured logging
fn init_logging(level: &str) -> Result<()> {
    let filter = match level {
        "trace" => "trace",
        "debug" => "debug",
        "info" => "info",
        "warn" => "warn",
        "error" => "error",
        _ => "info",
    };

    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(true)
        .with_thread_ids(true)
        .with_file(true)
        .with_line_number(true)
        .init();

    Ok(())
}

/// Load bootstrap configuration
async fn load_configuration(config_path: &PathBuf) -> Result<BootstrapConfig> {
    if config_path.exists() {
        info!("Loading configuration from: {}", config_path.display());
        BootstrapConfig::load_from_file(config_path.to_str()?)
    } else {
        info!("Configuration file not found, using defaults");
        Ok(BootstrapConfig::default())
    }
}

/// Export default configuration to file
async fn export_default_config(path: &PathBuf) -> Result<()> {
    let config = BootstrapConfig::default();
    config.save_to_file(path.to_str()?)?;
    Ok(())
}
