//! StratoSwarm CLI integration
//!
//! Unified orchestration platform commands.
//! This module delegates to the real stratoswarm-cli crate.

use clap::Subcommand;

// Re-export stratoswarm-cli commands for direct use
pub use stratoswarm_cli::commands::{
    deploy::DeployArgs, evolve::EvolveArgs, logs::LogsArgs, quickstart::QuickstartArgs,
    registry::RegistryArgs, scale::ScaleArgs, status::StatusArgs, token::TokenArgs,
};

/// StratoSwarm orchestration commands
///
/// These commands delegate to the stratoswarm-cli crate for full functionality.
#[derive(Subcommand)]
pub enum SwarmCommands {
    /// Deploy a swarm application
    Deploy(DeployArgs),

    /// Show status of deployed applications
    Status(StatusArgs),

    /// Show logs from agents
    Logs(LogsArgs),

    /// Scale agents
    Scale(ScaleArgs),

    /// Evolve agents through generations
    Evolve(EvolveArgs),

    /// Create a new project from template
    Quickstart(QuickstartArgs),

    /// Manage container images and registry
    Registry(RegistryArgs),

    /// Manage join tokens for node onboarding
    Token(TokenArgs),

    /// Enter interactive shell
    Shell,
}

impl SwarmCommands {
    /// Execute the swarm command by delegating to stratoswarm-cli
    pub async fn execute(self) -> anyhow::Result<()> {
        // Convert to stratoswarm_cli::Commands and execute
        let cmd = match self {
            Self::Deploy(args) => stratoswarm_cli::Commands::Deploy(args),
            Self::Status(args) => stratoswarm_cli::Commands::Status(args),
            Self::Logs(args) => stratoswarm_cli::Commands::Logs(args),
            Self::Scale(args) => stratoswarm_cli::Commands::Scale(args),
            Self::Evolve(args) => stratoswarm_cli::Commands::Evolve(args),
            Self::Quickstart(args) => stratoswarm_cli::Commands::Quickstart(args),
            Self::Registry(args) => stratoswarm_cli::Commands::Registry(args),
            Self::Token(args) => stratoswarm_cli::Commands::Token(args),
            Self::Shell => stratoswarm_cli::Commands::Shell,
        };

        cmd.execute().await.map_err(|e| anyhow::anyhow!("{}", e))
    }
}
