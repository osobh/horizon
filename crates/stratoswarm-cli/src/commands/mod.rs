//! Command implementations

pub mod deploy;
pub mod evolve;
pub mod logs;
pub mod quickstart;
pub mod registry;
pub mod scale;
pub mod status;

use crate::Result;
use clap::Subcommand;

#[derive(Debug, Clone, Subcommand)]
pub enum Commands {
    /// Deploy a swarm application
    Deploy(deploy::DeployArgs),

    /// Show status of deployed applications
    Status(status::StatusArgs),

    /// Show logs from agents
    Logs(logs::LogsArgs),

    /// Scale agents
    Scale(scale::ScaleArgs),

    /// Evolve agents through generations
    Evolve(evolve::EvolveArgs),

    /// Create a new project from template
    Quickstart(quickstart::QuickstartArgs),

    /// Manage container images and registry
    Registry(registry::RegistryArgs),

    /// Enter interactive shell
    Shell,
}

impl Commands {
    pub async fn execute(self) -> Result<()> {
        match self {
            Commands::Deploy(args) => deploy::execute(args).await,
            Commands::Status(args) => status::execute(args).await,
            Commands::Logs(args) => logs::execute(args).await,
            Commands::Scale(args) => scale::execute(args).await,
            Commands::Evolve(args) => evolve::execute(args).await,
            Commands::Quickstart(args) => quickstart::execute(args).await,
            Commands::Registry(args) => registry::execute(args).await,
            Commands::Shell => Box::pin(crate::shell::run_shell()).await,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[derive(Debug, Parser)]
    struct TestCli {
        #[command(subcommand)]
        command: Commands,
    }

    #[test]
    fn test_parse_deploy_command() {
        let cli = TestCli::parse_from(["test", "deploy", "app.swarm"]);
        assert!(matches!(cli.command, Commands::Deploy(_)));
    }

    #[test]
    fn test_parse_status_command() {
        let cli = TestCli::parse_from(["test", "status"]);
        assert!(matches!(cli.command, Commands::Status(_)));
    }

    #[test]
    fn test_parse_logs_command() {
        let cli = TestCli::parse_from(["test", "logs", "frontend"]);
        assert!(matches!(cli.command, Commands::Logs(_)));
    }

    #[test]
    fn test_parse_scale_command() {
        let cli = TestCli::parse_from(["test", "scale", "backend=5"]);
        assert!(matches!(cli.command, Commands::Scale(_)));
    }

    #[test]
    fn test_parse_evolve_command() {
        let cli = TestCli::parse_from(["test", "evolve", "backend", "--generations", "10"]);
        assert!(matches!(cli.command, Commands::Evolve(_)));
    }

    #[test]
    fn test_parse_shell_command() {
        let cli = TestCli::parse_from(["test", "shell"]);
        assert!(matches!(cli.command, Commands::Shell));
    }

    #[test]
    fn test_parse_registry_command() {
        let cli = TestCli::parse_from(["test", "registry", "build", "ubuntu:22.04"]);
        assert!(matches!(cli.command, Commands::Registry(_)));
    }
}
