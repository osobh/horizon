//! StratoSwarm Swarmlet - Lightweight node agent for easy cluster joining

use clap::{Parser, Subcommand};
use swarmlet::{
    agent::SwarmletAgent, discovery::ClusterDiscovery, error::SwarmletError, join::JoinProtocol,
    profile::HardwareProfiler,
};
use tracing::{debug, error, info};

#[derive(Parser)]
#[command(name = "swarmlet")]
#[command(version, about = "StratoSwarm lightweight node agent")]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Enable debug logging
    #[arg(short, long, global = true)]
    debug: bool,

    /// Configuration file path
    #[arg(short, long, global = true)]
    config: Option<String>,
}

#[derive(Subcommand)]
enum Commands {
    /// Join a StratoSwarm cluster
    Join {
        /// Join token from cluster coordinator
        #[arg(short, long)]
        token: String,

        /// Cluster coordinator address (IP:PORT)
        #[arg(short, long)]
        cluster: String,

        /// Node name (defaults to hostname)
        #[arg(short, long)]
        name: Option<String>,

        /// Data directory for persistent storage
        #[arg(short, long, default_value = "/data")]
        data_dir: String,
    },

    /// Automatically discover and join clusters on local network
    DiscoverAndJoin {
        /// Timeout for discovery in seconds
        #[arg(short, long, default_value = "30")]
        timeout: u64,

        /// Node name (defaults to hostname)
        #[arg(short, long)]
        name: Option<String>,
    },

    /// Profile local hardware capabilities
    ProfileHardware,

    /// Test connection to cluster
    TestConnection {
        /// Cluster coordinator address (IP:PORT)
        #[arg(short, long)]
        cluster: String,
    },

    /// Discover available clusters on local network
    Discover {
        /// Discovery timeout in seconds
        #[arg(short, long, default_value = "10")]
        timeout: u64,
    },

    /// Run as daemon (for Docker containers)
    Daemon {
        /// Configuration file path
        #[arg(short, long)]
        config: String,
    },
}

#[tokio::main]
async fn main() -> Result<(), SwarmletError> {
    let cli = Cli::parse();

    // Initialize logging
    let log_level = if cli.debug { "debug" } else { "info" };
    tracing_subscriber::fmt()
        .with_env_filter(format!("swarmlet={log_level},warn"))
        .with_target(false)
        .with_thread_ids(true)
        .init();

    info!(
        "Starting StratoSwarm Swarmlet v{}",
        env!("CARGO_PKG_VERSION")
    );

    match cli.command {
        Commands::Join {
            token,
            cluster,
            name,
            data_dir,
        } => join_cluster(token, cluster, name, data_dir).await,

        Commands::DiscoverAndJoin { timeout, name } => discover_and_join(timeout, name).await,

        Commands::ProfileHardware => profile_hardware().await,

        Commands::TestConnection { cluster } => test_connection(cluster).await,

        Commands::Discover { timeout } => discover_clusters(timeout).await,

        Commands::Daemon { config } => run_daemon(config).await,
    }
}

async fn join_cluster(
    token: String,
    cluster: String,
    name: Option<String>,
    data_dir: String,
) -> Result<(), SwarmletError> {
    info!("Joining cluster at {} with token", cluster);

    // Profile local hardware
    let mut profiler = HardwareProfiler::new();
    let profile = profiler.profile().await?;
    debug!("Hardware profile: {:#?}", profile);

    // Set up join protocol
    let mut join_protocol = JoinProtocol::new(token, cluster, profile);

    // Set node name
    let node_name = name.unwrap_or_else(|| {
        hostname::get()
            .unwrap_or_else(|_| "swarmlet".into())
            .to_string_lossy()
            .to_string()
    });
    join_protocol.set_node_name(node_name);

    // Perform join handshake
    info!("Initiating join handshake...");
    let join_result = join_protocol.join().await?;
    info!(
        "Successfully joined cluster! Node ID: {}",
        join_result.node_id
    );

    // Start swarmlet agent
    let agent = SwarmletAgent::new(join_result, data_dir).await?;
    info!("Starting swarmlet agent...");

    // Run agent (this blocks until shutdown)
    agent.run().await?;

    Ok(())
}

async fn discover_and_join(timeout: u64, _name: Option<String>) -> Result<(), SwarmletError> {
    info!("Discovering StratoSwarm clusters on local network...");

    let discovery = ClusterDiscovery::new();
    let clusters = discovery.discover_clusters(timeout).await?;

    if clusters.is_empty() {
        error!("No StratoSwarm clusters found on local network");
        return Err(SwarmletError::NoClusterFound);
    }

    info!("Found {} cluster(s):", clusters.len());
    for (i, cluster) in clusters.iter().enumerate() {
        println!("  {}. {} ({})", i + 1, cluster.name, cluster.address);
    }

    // For now, auto-join the first cluster found (in a real implementation,
    // this might prompt the user or use additional selection criteria)
    let cluster = &clusters[0];
    info!(
        "Auto-joining cluster: {} at {}",
        cluster.name, cluster.address
    );

    // This would need a token from the cluster's join service
    // For now, return an error indicating manual join is required
    error!("Auto-join not yet implemented. Please use 'join' command with a token.");
    Err(SwarmletError::NotImplemented(
        "Auto-join with discovered clusters".to_string(),
    ))
}

async fn profile_hardware() -> Result<(), SwarmletError> {
    info!("Profiling local hardware...");

    let mut profiler = HardwareProfiler::new();
    let profile = profiler.profile().await?;

    println!("Hardware Profile:");
    println!("{}", serde_json::to_string_pretty(&profile)?);

    Ok(())
}

async fn test_connection(cluster: String) -> Result<(), SwarmletError> {
    info!("Testing connection to cluster at {}", cluster);

    let discovery = ClusterDiscovery::new();
    match discovery.test_connection(&cluster).await {
        Ok(_) => {
            println!("✓ Connection successful");
            Ok(())
        }
        Err(e) => {
            println!("✗ Connection failed: {e}");
            Err(e)
        }
    }
}

async fn discover_clusters(timeout: u64) -> Result<(), SwarmletError> {
    info!(
        "Discovering StratoSwarm clusters (timeout: {}s)...",
        timeout
    );

    let discovery = ClusterDiscovery::new();
    let clusters = discovery.discover_clusters(timeout).await?;

    if clusters.is_empty() {
        println!("No StratoSwarm clusters found on local network");
    } else {
        println!("Found {} cluster(s):", clusters.len());
        for cluster in clusters {
            println!(
                "  • {} at {} ({})",
                cluster.name, cluster.address, cluster.node_class
            );
        }
    }

    Ok(())
}

async fn run_daemon(config_path: String) -> Result<(), SwarmletError> {
    info!("Starting swarmlet daemon with config: {}", config_path);

    // Load configuration
    let config = swarmlet::config::Config::load(&config_path).await?;
    debug!("Loaded configuration: {:#?}", config);

    // Create agent from configuration
    let agent = SwarmletAgent::from_config(config).await?;

    // Run agent daemon
    info!("Swarmlet daemon started");
    agent.run().await?;

    Ok(())
}
