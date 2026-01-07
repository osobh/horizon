//! HPC-AI Unified CLI structure
//!
//! Defines the root command structure with subcommands for each HPC-AI component.
//! Supports both CLI and TUI modes via --tui flag.

use clap::{Parser, Subcommand};

use crate::commands::{
    ArgusCommands, ChannelsCommands, DeployCommands, InventoryCommands, NebulaCommands,
    ParcodeCommands, RmpiCommands, RncclCommands, RustgCommands, SlaiCommands, SparkCommands,
    StackCommands, SwarmCommands, TorchCommands, VortexCommands, WarpCommands,
};

/// HPC-AI - Unified High-Performance Computing Platform CLI
///
/// The most powerful tool for HPC-AI operators to query, deploy,
/// administrate, and interact with all platform components.
#[derive(Parser)]
#[command(name = "hpc")]
#[command(version, about, long_about = None)]
#[command(propagate_version = true)]
#[command(author = "HPC-AI Team")]
pub struct Cli {
    /// Increase verbosity (-v, -vv, -vvv)
    #[arg(short, long, action = clap::ArgAction::Count, global = true)]
    pub verbose: u8,

    /// Output format
    #[arg(short, long, default_value = "table", global = true)]
    pub format: OutputFormat,

    /// Configuration file path
    #[arg(short, long, global = true)]
    pub config: Option<String>,

    /// Active profile (dev, staging, prod)
    #[arg(short, long, global = true)]
    pub profile: Option<String>,

    /// Launch Terminal UI mode
    #[arg(long)]
    pub tui: bool,

    #[command(subcommand)]
    pub command: Option<Commands>,
}

#[derive(Clone, Debug, Default, clap::ValueEnum)]
pub enum OutputFormat {
    #[default]
    Table,
    Json,
    Plain,
}

#[derive(Subcommand)]
pub enum Commands {
    // === Core Infrastructure (00-02) ===

    /// Rustg - GPU-accelerated Rust compiler (00-rust)
    #[command(subcommand)]
    Rustg(RustgCommands),

    /// HPC Channels - IPC and message passing utilities (02-hpc-channels)
    #[command(subcommand)]
    Channels(ChannelsCommands),

    // === Storage & Communication (03-05) ===

    /// Parcode - Lazy-loading object storage (03-parcode)
    #[command(subcommand)]
    Parcode(ParcodeCommands),

    /// RMPI - Rust MPI utilities (04-rmpi)
    #[command(subcommand)]
    Rmpi(RmpiCommands),

    /// RNCCL - Rust-native GPU collective communication (05-rnccl)
    #[command(subcommand)]
    Rnccl(RncclCommands),

    // === GPU Management (06) ===

    /// SLAI - GPU detection and cluster management (06-slai)
    #[command(subcommand)]
    Slai(SlaiCommands),

    // === Data Transfer & Orchestration (07-08) ===

    /// WARP - GPU-accelerated bulk data transfer (07-warp)
    #[command(subcommand)]
    Warp(WarpCommands),

    /// StratoSwarm - Unified orchestration platform (08-stratoswarm)
    #[command(subcommand)]
    Swarm(SwarmCommands),

    // === Data Processing & ML (09-10) ===

    /// RustySpark - Distributed data processing (09-rustyspark)
    #[command(subcommand)]
    Spark(SparkCommands),

    /// RustyTorch - GPU-accelerated ML training (10-rustytorch)
    #[command(subcommand)]
    Torch(TorchCommands),

    // === Networking (11-12) ===

    /// Vortex - Intelligent edge proxy (11-vortex)
    #[command(subcommand)]
    Vortex(VortexCommands),

    /// Nebula - Rust-native real-time communication (12-nebula)
    #[command(subcommand)]
    Nebula(NebulaCommands),

    // === Observability (14) ===

    /// Argus - Observability platform (14-argus)
    #[command(subcommand)]
    Argus(ArgusCommands),

    // === Deployment & Lifecycle ===

    /// Stack lifecycle management (init, up, down, status, promote)
    #[command(subcommand)]
    Stack(StackCommands),

    /// Quick deploy projects
    #[command(subcommand)]
    Deploy(DeployCommands),

    /// Node inventory management
    #[command(subcommand)]
    Inventory(InventoryCommands),

    // === Meta Commands ===

    /// Launch Terminal UI dashboard
    Tui,

    /// Show system information
    Info,

    /// Initialize HPC-AI configuration
    Init,

    /// Generate shell completions
    Completions {
        #[arg(value_enum)]
        shell: clap_complete::Shell,
    },

    /// Show project versions
    Version,
}
