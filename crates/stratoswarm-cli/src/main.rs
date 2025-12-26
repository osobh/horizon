//! StratoSwarm CLI - The unified orchestration platform CLI

use clap::Parser;
use stratoswarm_cli::{Cli, Result};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_target(false)
        .init();

    // Parse CLI arguments
    let cli = Cli::parse();

    // Execute command
    match cli.command {
        Some(cmd) => cmd.execute().await,
        None => {
            // No command specified, start interactive shell
            stratoswarm_cli::shell::run_shell().await
        }
    }
}
