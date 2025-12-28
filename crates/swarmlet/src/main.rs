//! StratoSwarm Swarmlet - Lightweight node agent for easy cluster joining

use clap::{Parser, Subcommand};
use swarmlet::{
    agent::SwarmletAgent, discovery::ClusterDiscovery, error::SwarmletError, join::JoinProtocol,
    profile::HardwareProfiler,
};
use tracing::{debug, error, info, warn};

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

async fn discover_and_join(timeout: u64, name: Option<String>) -> Result<(), SwarmletError> {
    info!("Discovering StratoSwarm clusters on local network...");

    let discovery = ClusterDiscovery::new();
    let clusters = discovery.discover_clusters(timeout).await?;

    if clusters.is_empty() {
        error!("No StratoSwarm clusters found on local network");
        return Err(SwarmletError::NoClusterFound);
    }

    info!("Found {} cluster(s):", clusters.len());
    for (i, cluster) in clusters.iter().enumerate() {
        let caps = cluster.capabilities.join(", ");
        println!(
            "  {}. {} ({}) - {} nodes, capabilities: [{}]",
            i + 1,
            cluster.name,
            cluster.address,
            cluster.nodes_count,
            caps
        );
    }

    // Find a cluster that supports auto-join (has "open_join" capability)
    let joinable_cluster = clusters
        .iter()
        .find(|c| c.capabilities.iter().any(|cap| cap == "open_join" || cap == "auto_join"));

    let cluster = match joinable_cluster {
        Some(c) => c,
        None => {
            // If no cluster has open_join, try the first cluster anyway
            warn!("No clusters with 'open_join' capability found, attempting first cluster");
            &clusters[0]
        }
    };

    info!(
        "Attempting to auto-join cluster: {} at {}",
        cluster.name, cluster.address
    );

    // Request an auto-join token from the cluster
    let token = request_auto_join_token(&cluster.address).await?;
    info!("Received auto-join token from cluster");

    // Use the default data directory
    let data_dir = swarmlet::defaults::DATA_DIR.to_string();

    // Perform the join using the received token
    join_cluster(token, cluster.address.clone(), name, data_dir).await
}

/// Request an auto-join token from a cluster's API
async fn request_auto_join_token(cluster_address: &str) -> Result<String, SwarmletError> {
    use serde::{Deserialize, Serialize};

    #[derive(Serialize)]
    struct AutoJoinRequest {
        node_type: String,
        capabilities_requested: Vec<String>,
    }

    #[derive(Deserialize)]
    struct AutoJoinResponse {
        token: String,
        expires_in_seconds: u64,
    }

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .map_err(|e| SwarmletError::Network(e))?;

    let url = format!("http://{}/api/v1/cluster/auto-join", cluster_address);

    let request_body = AutoJoinRequest {
        node_type: "swarmlet".to_string(),
        capabilities_requested: vec!["basic".to_string(), "workload_execution".to_string()],
    };

    debug!("Requesting auto-join token from {}", url);

    let response = client
        .post(&url)
        .json(&request_body)
        .send()
        .await
        .map_err(|e| {
            if e.is_timeout() {
                SwarmletError::Timeout
            } else if e.is_connect() {
                SwarmletError::Discovery(format!("Cannot connect to cluster: {}", e))
            } else {
                SwarmletError::Network(e)
            }
        })?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();

        if status.as_u16() == 403 {
            return Err(SwarmletError::ClusterRejection(
                "Cluster does not allow auto-join. Use 'join' command with a valid token.".to_string()
            ));
        } else if status.as_u16() == 404 {
            return Err(SwarmletError::Discovery(
                "Cluster does not support auto-join API. Use 'join' command with a valid token.".to_string()
            ));
        }

        return Err(SwarmletError::JoinProtocol(format!(
            "Auto-join request failed ({}): {}",
            status, body
        )));
    }

    let auto_join_response: AutoJoinResponse = response.json().await.map_err(|e| {
        SwarmletError::JoinProtocol(format!("Invalid auto-join response: {}", e))
    })?;

    info!(
        "Received auto-join token (expires in {} seconds)",
        auto_join_response.expires_in_seconds
    );

    Ok(auto_join_response.token)
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
