//! CLI structure and argument parsing

use crate::commands::Commands;
use clap::Parser;

/// StratoSwarm CLI - The unified orchestration platform
#[derive(Debug, Parser)]
#[command(name = "stratoswarm")]
#[command(version, about, long_about = None)]
pub struct Cli {
    /// Commands
    #[command(subcommand)]
    pub command: Option<Commands>,

    /// Verbose output
    #[arg(short, long, global = true)]
    pub verbose: bool,

    /// Configuration file path
    #[arg(short, long, global = true)]
    pub config: Option<String>,
}
