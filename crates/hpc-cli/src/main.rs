//! HPC-AI Unified CLI
//!
//! The most powerful tool for HPC-AI operators to query, deploy,
//! administrate, and interact with all platform components.
//!
//! Usage:
//!   hpc <subcommand> [options]
//!   hpc --tui                    # Launch TUI dashboard
//!
//! Available subcommands:
//!   rustg     - GPU-accelerated compiler (00-rust)
//!   channels  - IPC utilities (02-hpc-channels)
//!   parcode   - Lazy-loading storage (03-parcode)
//!   rmpi      - Message passing (04-rmpi)
//!   rnccl     - GPU collective communication (05-rnccl)
//!   slai      - GPU cluster management (06-slai)
//!   warp      - GPU-accelerated data transfer (07-warp)
//!   swarm     - StratoSwarm orchestration (08-stratoswarm)
//!   spark     - Distributed data processing (09-rustyspark)
//!   torch     - ML training (10-rustytorch)
//!   vortex    - Edge proxy (11-vortex)
//!   nebula    - Real-time communication (12-nebula)
//!   argus     - Observability platform (14-argus)
//!   stack     - Stack lifecycle management
//!   deploy    - Quick project deployment
//!   inventory - Node inventory management
//!   info      - System information
//!   version   - Show versions

mod cli;
mod commands;
mod core;
mod server;
mod tui;

use anyhow::Result;
use clap::Parser;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use cli::{Cli, Commands};
use core::config::AppConfig;

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialize tracing based on verbosity
    let filter = match cli.verbose {
        0 => "warn",
        1 => "info",
        2 => "debug",
        _ => "trace",
    };

    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| filter.into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    // Load configuration
    let config = if let Some(ref path) = cli.config {
        AppConfig::load_from(path)?
    } else {
        AppConfig::load().unwrap_or_default()
    };

    // Check for TUI mode
    if cli.tui {
        return tui::app::launch(config).await;
    }

    // Handle command or show help
    let Some(command) = cli.command else {
        println!("HPC-AI Unified CLI v{}", env!("CARGO_PKG_VERSION"));
        println!();
        println!("Use 'hpc --help' for usage information.");
        println!("Use 'hpc --tui' for interactive dashboard.");
        println!();
        println!("Quick start:");
        println!("  hpc info              Show system information");
        println!("  hpc deploy status     Show deployment status");
        println!("  hpc stack list        List available stacks");
        return Ok(());
    };

    match command {
        // Core Infrastructure (00-02)
        Commands::Rustg(cmd) => cmd.execute().await,
        Commands::Channels(cmd) => cmd.execute().await,

        // Storage & Communication (03-05)
        Commands::Parcode(cmd) => cmd.execute().await,
        Commands::Rmpi(cmd) => cmd.execute().await,
        Commands::Rnccl(cmd) => cmd.execute().await,

        // GPU Management (06)
        Commands::Slai(cmd) => cmd.execute().await,

        // Data Transfer & Orchestration (07-08)
        Commands::Warp(cmd) => cmd.execute().await,
        Commands::Swarm(cmd) => cmd.execute().await,

        // Data Processing & ML (09-10)
        Commands::Spark(cmd) => cmd.execute().await,
        Commands::Torch(cmd) => cmd.execute().await,

        // Networking (11-12)
        Commands::Vortex(cmd) => cmd.execute().await,
        Commands::Nebula(cmd) => cmd.execute().await,

        // Observability (14)
        Commands::Argus(cmd) => cmd.execute().await,

        // Deployment & Lifecycle
        Commands::Stack(cmd) => cmd.execute().await,
        Commands::Deploy(cmd) => cmd.execute().await,
        Commands::Inventory(cmd) => cmd.execute().await,

        // Meta commands
        Commands::Tui => tui::app::launch(config).await,
        Commands::Info => {
            print_system_info();
            Ok(())
        }
        Commands::Init => {
            init_config().await
        }
        Commands::Completions { shell } => {
            generate_completions(shell);
            Ok(())
        }
        Commands::Version => {
            print_versions();
            Ok(())
        }
    }
}

/// Initialize configuration directories
async fn init_config() -> Result<()> {
    println!("Initializing HPC-AI Configuration");
    println!("==================================");
    println!();

    AppConfig::init_dirs()?;

    let config_path = AppConfig::config_path();
    let stacks_dir = AppConfig::stacks_dir();

    println!("Created directories:");
    println!("  Config:  {:?}", config_path.parent().unwrap());
    println!("  Stacks:  {:?}", stacks_dir);
    println!("  State:   {:?}", AppConfig::state_dir());
    println!();

    if !config_path.exists() {
        let config = AppConfig::default();
        config.save()?;
        println!("Created default configuration: {:?}", config_path);
    } else {
        println!("Configuration already exists: {:?}", config_path);
    }

    println!();
    println!("Configuration complete!");
    println!();
    println!("Next steps:");
    println!("  hpc stack init ml-training    Create a stack");
    println!("  hpc deploy local              Deploy locally");
    println!("  hpc --tui                     Launch dashboard");

    Ok(())
}

fn print_system_info() {
    println!("HPC-AI System Information");
    println!("=========================");
    println!();
    println!("Platform: {} {}", std::env::consts::OS, std::env::consts::ARCH);
    println!();
    println!("Components (15 projects):");
    println!();
    println!("  Core Infrastructure:");
    println!("    00-rust:        (use 'hpc rustg info' for details)");
    println!("    02-hpc-channels: available");
    println!();
    println!("  Storage & Communication:");
    println!("    03-parcode:     (use 'hpc parcode cache stats' for details)");
    println!("    04-rmpi:        (use 'hpc rmpi info' for details)");
    println!("    05-rnccl:       (use 'hpc rnccl info' for details)");
    println!();
    println!("  GPU Management:");
    println!("    06-slai:        (use 'hpc slai detect' for details)");
    println!();
    println!("  Data Transfer & Orchestration:");
    println!("    07-warp:        (use 'hpc warp info' for details)");
    println!("    08-stratoswarm: (use 'hpc swarm status' for details)");
    println!();
    println!("  Data Processing & ML:");
    println!("    09-rustyspark:  (use 'hpc spark status' for details)");
    println!("    10-rustytorch:  (use 'hpc torch status' for details)");
    println!();
    println!("  Networking:");
    println!("    11-vortex:      (use 'hpc vortex status' for details)");
    println!("    12-nebula:      (use 'hpc nebula status' for details)");
    println!();
    println!("  UI & Observability:");
    println!("    13-horizon:     (desktop app - not in CLI)");
    println!("    14-argus:       (use 'hpc argus status' for details)");
    println!();
    println!("CLI version: v{}", env!("CARGO_PKG_VERSION"));
}

fn print_versions() {
    println!("HPC-AI Platform v{}", env!("CARGO_PKG_VERSION"));
    println!();
    println!("CLI Components:");
    println!("  hpc-ai-cli:  {}", env!("CARGO_PKG_VERSION"));
    println!();
    println!("Use 'hpc <component> --version' for individual component versions.");
}

fn generate_completions(shell: clap_complete::Shell) {
    use clap::CommandFactory;
    clap_complete::generate(shell, &mut Cli::command(), "hpc", &mut std::io::stdout());
}
