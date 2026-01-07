//! Inventory management commands
//!
//! CLI commands for managing the node inventory.

use anyhow::{Context, Result};
use clap::{Args, Subcommand};
use console::style;

use crate::core::inventory::{
    CredentialRef, CredentialStore, InventoryStore, NodeDetector, NodeInfo, NodeMode, NodeStatus,
    SshAuth, SshClient,
};

/// Inventory management commands
#[derive(Subcommand)]
pub enum InventoryCommands {
    /// Add a new node to inventory
    Add(AddNodeArgs),

    /// List all nodes in inventory
    List {
        /// Filter by status (connected, pending, failed, unreachable)
        #[arg(long)]
        status: Option<String>,

        /// Filter by tag
        #[arg(long)]
        tag: Option<String>,

        /// Output format (table, json, wide)
        #[arg(short, long, default_value = "table")]
        output: String,
    },

    /// Remove a node from inventory
    Remove {
        /// Node ID or name
        node: String,

        /// Force removal without confirmation
        #[arg(long)]
        force: bool,
    },

    /// Show detailed node status
    Status {
        /// Node ID or name (shows all if omitted)
        node: Option<String>,
    },

    /// Bootstrap or reinstall agent on a node
    Bootstrap {
        /// Node ID or name
        node: String,

        /// Force reinstall even if agent exists
        #[arg(long)]
        force: bool,
    },

    /// Run health check on nodes
    Check {
        /// Node ID or name (checks all if omitted)
        node: Option<String>,
    },

    /// Manage SSH keys
    #[command(subcommand)]
    Keys(KeyCommands),

    /// Import nodes from file
    Import {
        /// Path to inventory file (TOML/JSON)
        file: String,
    },

    /// Export inventory to file
    Export {
        /// Output file path
        #[arg(short, long)]
        output: Option<String>,

        /// Format (toml, json)
        #[arg(short, long, default_value = "json")]
        format: String,
    },
}

/// Arguments for adding a node
#[derive(Args)]
pub struct AddNodeArgs {
    /// Node IP address or hostname
    pub address: String,

    /// SSH username
    #[arg(short, long)]
    pub user: String,

    /// SSH password (will prompt if not provided and no key specified)
    #[arg(short = 'P', long)]
    pub password: Option<String>,

    /// Path to SSH private key
    #[arg(short, long)]
    pub key: Option<String>,

    /// Use SSH agent for authentication
    #[arg(long)]
    pub agent: bool,

    /// SSH port
    #[arg(long, default_value = "22")]
    pub port: u16,

    /// Deployment mode (docker or binary)
    #[arg(short, long, value_enum)]
    pub mode: NodeModeArg,

    /// Custom node name (auto-generated if not provided)
    #[arg(short, long)]
    pub name: Option<String>,

    /// Tags for the node (can be specified multiple times)
    #[arg(long)]
    pub tag: Vec<String>,

    /// Skip bootstrap (just add to inventory)
    #[arg(long)]
    pub no_bootstrap: bool,

    /// Generate new SSH key for this node
    #[arg(long)]
    pub generate_key: bool,
}

/// Node mode argument enum
#[derive(Clone, clap::ValueEnum)]
pub enum NodeModeArg {
    Docker,
    Binary,
}

impl From<NodeModeArg> for NodeMode {
    fn from(arg: NodeModeArg) -> Self {
        match arg {
            NodeModeArg::Docker => NodeMode::Docker,
            NodeModeArg::Binary => NodeMode::Binary,
        }
    }
}

/// SSH key management subcommands
#[derive(Subcommand)]
pub enum KeyCommands {
    /// List all stored SSH keys
    List,

    /// Generate a new SSH key pair
    Generate {
        /// Key name/identifier
        name: String,
    },

    /// Import an existing SSH key
    Import {
        /// Path to private key file
        path: String,

        /// Key name/identifier
        #[arg(short, long)]
        name: Option<String>,
    },

    /// Delete an SSH key
    Delete {
        /// Key name/identifier
        name: String,

        /// Skip confirmation
        #[arg(long)]
        force: bool,
    },

    /// Show public key for a node
    Show {
        /// Node ID or key name
        name: String,
    },
}

impl InventoryCommands {
    pub async fn execute(self) -> Result<()> {
        match self {
            Self::Add(args) => add_node(args).await,
            Self::List { status, tag, output } => list_nodes(status, tag, output).await,
            Self::Remove { node, force } => remove_node(node, force).await,
            Self::Status { node } => show_status(node).await,
            Self::Bootstrap { node, force } => bootstrap_node(node, force).await,
            Self::Check { node } => check_nodes(node).await,
            Self::Keys(cmd) => cmd.execute().await,
            Self::Import { file } => import_inventory(file).await,
            Self::Export { output, format } => export_inventory(output, format).await,
        }
    }
}

impl KeyCommands {
    pub async fn execute(self) -> Result<()> {
        match self {
            Self::List => list_keys().await,
            Self::Generate { name } => generate_key(name).await,
            Self::Import { path, name } => import_key(path, name).await,
            Self::Delete { name, force } => delete_key(name, force).await,
            Self::Show { name } => show_public_key(name).await,
        }
    }
}

// === Command Implementations ===

async fn add_node(args: AddNodeArgs) -> Result<()> {
    let mut store = InventoryStore::new()?;
    let cred_store = CredentialStore::new()?;

    println!();
    println!("{}", style("Adding Node to Inventory").cyan().bold());
    println!("{}", style("=".repeat(40)).dim());
    println!();

    // Generate node name if not provided
    let node_name = args.name.unwrap_or_else(|| store.generate_node_name(&args.address));

    // Determine credential type
    let credential_ref = if args.generate_key {
        // Generate a new SSH key for this node
        println!("Generating SSH key for node...");
        let temp_id = uuid::Uuid::new_v4().to_string();
        cred_store.generate_ssh_key(&temp_id)?
    } else if let Some(key_path) = &args.key {
        // Use provided key file
        CredentialRef::SshKey {
            path: std::path::PathBuf::from(key_path),
        }
    } else if args.agent {
        // Use SSH agent
        CredentialRef::SshAgent
    } else if let Some(password) = &args.password {
        // Store password in keyring
        let temp_id = uuid::Uuid::new_v4().to_string();
        cred_store.store_password(&temp_id, password)?
    } else {
        // Prompt for password
        let password = dialoguer::Password::new()
            .with_prompt("SSH password")
            .interact()?;

        let temp_id = uuid::Uuid::new_v4().to_string();
        cred_store.store_password(&temp_id, &password)?
    };

    // Create node
    let mut node = NodeInfo::new(
        node_name.clone(),
        args.address.clone(),
        args.port,
        args.user.clone(),
        credential_ref,
        args.mode.into(),
    );

    node.tags = args.tag;

    println!("Node:     {}", style(&node_name).green());
    println!("Address:  {}:{}", style(&args.address).green(), args.port);
    println!("User:     {}", style(&args.user).green());
    println!("Mode:     {}", style(node.mode.as_str()).green());

    if !node.tags.is_empty() {
        println!("Tags:     {}", style(node.tags.join(", ")).green());
    }

    println!();

    // Add to inventory
    store.add_node(node)?;

    println!(
        "{} Node '{}' added to inventory",
        style("✓").green().bold(),
        node_name
    );

    if args.no_bootstrap {
        println!();
        println!(
            "{}",
            style("Bootstrap skipped. Run 'hpc inventory bootstrap' to install agent.").yellow()
        );
    } else {
        println!();
        println!("To bootstrap this node, run:");
        println!("  {} hpc inventory bootstrap {}", style("$").dim(), node_name);
    }

    Ok(())
}

async fn list_nodes(status_filter: Option<String>, tag_filter: Option<String>, output: String) -> Result<()> {
    let store = InventoryStore::new()?;

    let nodes: Vec<_> = store
        .list_nodes()
        .into_iter()
        .filter(|n| {
            // Filter by status
            if let Some(ref status) = status_filter {
                if n.status.as_str() != status {
                    return false;
                }
            }
            // Filter by tag
            if let Some(ref tag) = tag_filter {
                if !n.tags.contains(tag) {
                    return false;
                }
            }
            true
        })
        .collect();

    if nodes.is_empty() {
        println!();
        println!("{}", style("No nodes in inventory.").yellow());
        println!();
        println!("Add a node with: hpc inventory add <IP> --user <user> --mode docker");
        return Ok(());
    }

    match output.as_str() {
        "json" => {
            let json = serde_json::to_string_pretty(&nodes)?;
            println!("{}", json);
        }
        "wide" => {
            print_nodes_wide(&nodes);
        }
        _ => {
            print_nodes_table(&nodes);
        }
    }

    // Print summary
    let summary = store.summary();
    println!();
    println!(
        "Total: {} nodes ({} connected, {} pending, {} failed)",
        style(summary.total_nodes).bold(),
        style(summary.connected).green(),
        style(summary.pending).yellow(),
        style(summary.failed).red()
    );

    Ok(())
}

fn print_nodes_table(nodes: &[&NodeInfo]) {
    println!();
    println!(
        "{:20} {:20} {:12} {:15} {:8}",
        style("NAME").bold().underlined(),
        style("ADDRESS").bold().underlined(),
        style("STATUS").bold().underlined(),
        style("PLATFORM").bold().underlined(),
        style("MODE").bold().underlined()
    );

    for node in nodes {
        let status_styled = match node.status {
            NodeStatus::Connected => style(format!("{} {}", node.status.symbol(), node.status)).green(),
            NodeStatus::Pending | NodeStatus::Connecting | NodeStatus::Bootstrapping => {
                style(format!("{} {}", node.status.symbol(), node.status)).yellow()
            }
            NodeStatus::Failed => style(format!("{} {}", node.status.symbol(), node.status)).red(),
            NodeStatus::Unreachable => style(format!("{} {}", node.status.symbol(), node.status)).red().dim(),
            NodeStatus::Offline => style(format!("{} {}", node.status.symbol(), node.status)).dim(),
        };

        println!(
            "{:20} {:20} {:12} {:15} {:8}",
            node.name,
            format!("{}:{}", node.address, node.port),
            status_styled,
            node.platform_str(),
            node.mode.as_str()
        );
    }
}

fn print_nodes_wide(nodes: &[&NodeInfo]) {
    println!();
    println!(
        "{:36} {:15} {:20} {:12} {:15} {:8} {:20}",
        style("ID").bold().underlined(),
        style("NAME").bold().underlined(),
        style("ADDRESS").bold().underlined(),
        style("STATUS").bold().underlined(),
        style("PLATFORM").bold().underlined(),
        style("MODE").bold().underlined(),
        style("LAST SEEN").bold().underlined()
    );

    for node in nodes {
        let last_seen = node
            .last_heartbeat
            .map(|t| t.format("%Y-%m-%d %H:%M").to_string())
            .unwrap_or_else(|| "-".to_string());

        println!(
            "{:36} {:15} {:20} {:12} {:15} {:8} {:20}",
            &node.id[..36.min(node.id.len())],
            node.name,
            format!("{}:{}", node.address, node.port),
            format!("{} {}", node.status.symbol(), node.status),
            node.platform_str(),
            node.mode.as_str(),
            last_seen
        );
    }
}

async fn remove_node(node_id: String, force: bool) -> Result<()> {
    let mut store = InventoryStore::new()?;
    let cred_store = CredentialStore::new()?;

    let node = store
        .find_node(&node_id)
        .context(format!("Node '{}' not found", node_id))?;

    let node_name = node.name.clone();
    let node_actual_id = node.id.clone();

    if !force {
        let confirm = dialoguer::Confirm::new()
            .with_prompt(format!("Remove node '{}'?", node_name))
            .default(false)
            .interact()?;

        if !confirm {
            println!("Cancelled.");
            return Ok(());
        }
    }

    // Remove credentials
    cred_store.delete_credentials(&node_actual_id)?;

    // Remove from inventory
    store.remove_node(&node_actual_id)?;

    println!(
        "{} Node '{}' removed from inventory",
        style("✓").green().bold(),
        node_name
    );

    Ok(())
}

async fn show_status(node_id: Option<String>) -> Result<()> {
    let store = InventoryStore::new()?;

    match node_id {
        Some(id) => {
            let node = store
                .find_node(&id)
                .context(format!("Node '{}' not found", id))?;

            print_node_details(node);
        }
        None => {
            let summary = store.summary();

            println!();
            println!("{}", style("Inventory Status").cyan().bold());
            println!("{}", style("=".repeat(40)).dim());
            println!();
            println!("Total Nodes:  {}", style(summary.total_nodes).bold());
            println!(
                "Connected:    {}",
                style(summary.connected).green().bold()
            );
            println!("Pending:      {}", style(summary.pending).yellow());
            println!("Unreachable:  {}", style(summary.unreachable).red().dim());
            println!("Failed:       {}", style(summary.failed).red());
            println!();
            println!("Resources:");
            println!("  CPUs:   {}", summary.total_cpus);
            println!("  GPUs:   {}", summary.total_gpus);
            println!("  Memory: {:.1} GB", summary.total_memory_gb);
        }
    }

    Ok(())
}

fn print_node_details(node: &NodeInfo) {
    println!();
    println!("{}", style(format!("Node: {}", node.name)).cyan().bold());
    println!("{}", style("=".repeat(40)).dim());
    println!();
    println!("ID:         {}", node.id);
    println!("Address:    {}:{}", node.address, node.port);
    println!("Username:   {}", node.username);
    println!(
        "Status:     {} {}",
        node.status.symbol(),
        style(node.status.as_str()).bold()
    );
    println!("Mode:       {}", node.mode);
    println!("Platform:   {}", node.platform_str());

    if !node.tags.is_empty() {
        println!("Tags:       {}", node.tags.join(", "));
    }

    println!();
    println!("Timestamps:");
    println!("  Created:  {}", node.created_at.format("%Y-%m-%d %H:%M:%S"));
    println!("  Updated:  {}", node.updated_at.format("%Y-%m-%d %H:%M:%S"));

    if let Some(hb) = node.last_heartbeat {
        println!("  Last HB:  {}", hb.format("%Y-%m-%d %H:%M:%S"));
    }

    if let Some(hw) = &node.hardware {
        println!();
        println!("Hardware:");
        println!("  CPU:      {} ({} cores)", hw.cpu_model, hw.cpu_cores);
        println!("  Memory:   {:.1} GB", hw.memory_gb);
        println!("  Storage:  {:.1} GB", hw.storage_gb);

        if !hw.gpus.is_empty() {
            println!("  GPUs:");
            for gpu in &hw.gpus {
                println!(
                    "    [{}] {} ({} MB)",
                    gpu.index, gpu.name, gpu.memory_mb
                );
            }
        }
    }

    if let Some(error) = &node.error {
        println!();
        println!("{}: {}", style("Error").red().bold(), error);
    }
}

async fn bootstrap_node(node_id: String, force: bool) -> Result<()> {
    let mut store = InventoryStore::new()?;
    let cred_store = CredentialStore::new()?;

    // Get node info
    let node = store
        .find_node(&node_id)
        .context(format!("Node '{}' not found", node_id))?
        .clone();

    println!();
    println!(
        "{} Bootstrapping node '{}'...",
        style("→").cyan(),
        node.name
    );
    println!("{}", style("=".repeat(50)).dim());
    println!();

    // Update status to connecting
    if let Some(n) = store.find_node_mut(&node_id) {
        n.set_status(NodeStatus::Connecting);
    }
    store.save()?;

    // Determine SSH auth
    let auth = match &node.credential_ref {
        CredentialRef::SshKey { path } => SshAuth::PrivateKey(path.clone()),
        CredentialRef::Password { key_id } => {
            let password = cred_store.get_password(key_id)?;
            SshAuth::Password(password)
        }
        CredentialRef::SshAgent => SshAuth::Agent,
    };

    println!(
        "{} Connecting to {}:{}...",
        style("1.").cyan(),
        node.address,
        node.port
    );

    // Connect via SSH
    let client = SshClient::new(node.address.clone(), node.port, node.username.clone(), auth);
    let session = match client.connect().await {
        Ok(session) => {
            println!("   {} Connected", style("✓").green());
            session
        }
        Err(e) => {
            if let Some(n) = store.find_node_mut(&node_id) {
                n.set_error(format!("SSH connection failed: {}", e));
            }
            store.save()?;
            return Err(e.context("Failed to connect via SSH"));
        }
    };

    // Update status to bootstrapping
    if let Some(n) = store.find_node_mut(&node_id) {
        n.set_status(NodeStatus::Bootstrapping);
    }
    store.save()?;

    // Detect platform
    println!("{} Detecting platform...", style("2.").cyan());
    let detector = NodeDetector::new(&session);

    let (os, arch) = match detector.detect_platform().await {
        Ok((os, arch)) => {
            println!("   {} Detected: {}/{}", style("✓").green(), os, arch);
            (Some(os), Some(arch))
        }
        Err(e) => {
            println!("   {} Detection failed: {}", style("!").yellow(), e);
            (None, None)
        }
    };

    // Update node with detected platform
    if let Some(n) = store.find_node_mut(&node_id) {
        n.os = os.clone();
        n.arch = arch.clone();
    }
    store.save()?;

    // Check system requirements
    println!("{} Checking requirements...", style("3.").cyan());
    let requirements = detector.check_requirements().await?;

    if requirements.has_docker {
        println!("   {} Docker available", style("✓").green());
    } else {
        println!("   {} Docker not available", style("○").yellow());
    }

    if requirements.has_agent && !force {
        println!("   {} Agent already installed", style("✓").green());
        println!();
        println!(
            "{}",
            style("Node already has agent installed. Use --force to reinstall.").yellow()
        );

        // Mark as connected since agent is present
        if let Some(n) = store.find_node_mut(&node_id) {
            n.set_status(NodeStatus::Connected);
        }
        store.save()?;

        session.disconnect().await?;
        return Ok(());
    }

    if !requirements.issues.is_empty() {
        println!("   {} Issues found:", style("!").yellow());
        for issue in &requirements.issues {
            println!("     - {}", issue);
        }
    }

    // Detect hardware profile
    println!("{} Detecting hardware...", style("4.").cyan());
    match detector.detect_hardware().await {
        Ok(hw) => {
            println!(
                "   CPU: {} ({} cores)",
                style(&hw.cpu_model).cyan(),
                hw.cpu_cores
            );
            println!("   Memory: {:.1} GB", hw.memory_gb);
            println!("   Storage: {:.1} GB", hw.storage_gb);
            if !hw.gpus.is_empty() {
                println!("   GPUs: {}", hw.gpus.len());
                for gpu in &hw.gpus {
                    println!("     [{}] {} ({} MB)", gpu.index, gpu.name, gpu.memory_mb);
                }
            }

            // Save hardware profile
            if let Some(n) = store.find_node_mut(&node_id) {
                n.hardware = Some(hw);
            }
            store.save()?;
        }
        Err(e) => {
            println!("   {} Hardware detection failed: {}", style("!").yellow(), e);
        }
    }

    // Bootstrap step - for now just show instructions
    println!("{} Bootstrap instructions:", style("5.").cyan());
    println!();

    let mode_str = node.mode.as_str();
    let platform = requirements.platform_str();

    println!(
        "   To complete bootstrap, run on the remote node:"
    );
    println!();

    if requirements.has_curl {
        println!(
            "   curl -sSL http://<hpc-server>/bootstrap?mode={}&platform={} | bash",
            mode_str, platform
        );
    } else if requirements.has_wget {
        println!(
            "   wget -qO- http://<hpc-server>/bootstrap?mode={}&platform={} | bash",
            mode_str, platform
        );
    }

    println!();
    println!(
        "{}",
        style("Note: Full automated bootstrap requires Phase 4 (HTTP server).").dim()
    );

    // For now, mark as pending (waiting for agent)
    if let Some(n) = store.find_node_mut(&node_id) {
        n.set_status(NodeStatus::Pending);
    }
    store.save()?;

    // Disconnect
    session.disconnect().await?;

    println!();
    println!(
        "{} Bootstrap preparation complete for '{}'",
        style("✓").green().bold(),
        node.name
    );

    Ok(())
}

async fn check_nodes(node_id: Option<String>) -> Result<()> {
    let store = InventoryStore::new()?;

    println!();
    println!("{}", style("Health Check").cyan().bold());
    println!("{}", style("=".repeat(40)).dim());
    println!();

    let nodes: Vec<_> = match node_id {
        Some(id) => {
            vec![store
                .find_node(&id)
                .context(format!("Node '{}' not found", id))?]
        }
        None => store.list_nodes(),
    };

    for node in nodes {
        let status_icon = match node.status {
            NodeStatus::Connected => style("✓").green(),
            NodeStatus::Pending => style("○").yellow(),
            _ => style("✗").red(),
        };

        println!(
            "{} {:20} {:12} {}",
            status_icon,
            node.name,
            node.status.as_str(),
            node.address
        );
    }

    println!();
    println!(
        "{}",
        style("Note: Full health check requires Phase 5 (heartbeat system).").dim()
    );

    Ok(())
}

async fn list_keys() -> Result<()> {
    let cred_store = CredentialStore::new()?;
    let keys = cred_store.list_keys()?;

    if keys.is_empty() {
        println!();
        println!("{}", style("No SSH keys stored.").yellow());
        println!();
        println!("Generate a key with: hpc inventory keys generate <name>");
        return Ok(());
    }

    println!();
    println!(
        "{:30} {:10} {:40}",
        style("NAME").bold().underlined(),
        style("PUB KEY").bold().underlined(),
        style("PATH").bold().underlined()
    );

    for key in keys {
        let has_pub = if key.has_public_key { "yes" } else { "no" };
        println!(
            "{:30} {:10} {:40}",
            key.name,
            has_pub,
            key.path.display()
        );
    }

    Ok(())
}

async fn generate_key(name: String) -> Result<()> {
    let cred_store = CredentialStore::new()?;

    println!();
    println!("Generating SSH key pair: {}", style(&name).cyan());

    let cred_ref = cred_store.generate_ssh_key(&name)?;

    if let CredentialRef::SshKey { path } = &cred_ref {
        println!();
        println!("{} SSH key generated", style("✓").green().bold());
        println!("  Private key: {}", path.display());
        println!(
            "  Public key:  {}.pub",
            path.display()
        );
        println!();

        // Show public key content
        if let Ok(pub_key) = cred_store.get_public_key(&name) {
            println!("Public key (add to remote authorized_keys):");
            println!("{}", style(&pub_key).dim());
        }
    }

    Ok(())
}

async fn import_key(path: String, name: Option<String>) -> Result<()> {
    let cred_store = CredentialStore::new()?;

    let key_name = name.unwrap_or_else(|| {
        std::path::Path::new(&path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("imported")
            .to_string()
    });

    println!();
    println!("Importing SSH key as: {}", style(&key_name).cyan());

    cred_store.import_ssh_key(&key_name, &path)?;

    println!("{} SSH key imported", style("✓").green().bold());

    Ok(())
}

async fn delete_key(name: String, force: bool) -> Result<()> {
    let cred_store = CredentialStore::new()?;

    if !cred_store.key_exists(&name) {
        anyhow::bail!("Key '{}' not found", name);
    }

    if !force {
        let confirm = dialoguer::Confirm::new()
            .with_prompt(format!("Delete SSH key '{}'?", name))
            .default(false)
            .interact()?;

        if !confirm {
            println!("Cancelled.");
            return Ok(());
        }
    }

    cred_store.delete_credentials(&name)?;

    println!("{} SSH key '{}' deleted", style("✓").green().bold(), name);

    Ok(())
}

async fn show_public_key(name: String) -> Result<()> {
    let cred_store = CredentialStore::new()?;

    let pub_key = cred_store.get_public_key(&name)?;

    println!("{}", pub_key);

    Ok(())
}

async fn import_inventory(file: String) -> Result<()> {
    let content = std::fs::read_to_string(&file)
        .with_context(|| format!("Failed to read file: {}", file))?;

    let nodes: Vec<NodeInfo> = if file.ends_with(".json") {
        serde_json::from_str(&content)?
    } else if file.ends_with(".toml") {
        toml::from_str(&content)?
    } else {
        anyhow::bail!("Unsupported file format. Use .json or .toml");
    };

    let mut store = InventoryStore::new()?;
    let mut imported = 0;

    for node in nodes {
        if store.find_by_address(&node.address).is_some() {
            println!(
                "{} Skipping {} (address already exists)",
                style("!").yellow(),
                node.name
            );
            continue;
        }

        store.add_node(node)?;
        imported += 1;
    }

    println!();
    println!(
        "{} Imported {} nodes from {}",
        style("✓").green().bold(),
        imported,
        file
    );

    Ok(())
}

async fn export_inventory(output: Option<String>, format: String) -> Result<()> {
    let store = InventoryStore::new()?;
    let nodes: Vec<_> = store.list_nodes().into_iter().cloned().collect();

    let content = match format.as_str() {
        "toml" => toml::to_string_pretty(&nodes)?,
        _ => serde_json::to_string_pretty(&nodes)?,
    };

    match output {
        Some(path) => {
            std::fs::write(&path, &content)?;
            println!(
                "{} Exported {} nodes to {}",
                style("✓").green().bold(),
                nodes.len(),
                path
            );
        }
        None => {
            println!("{}", content);
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_mode_conversion() {
        let docker: NodeMode = NodeModeArg::Docker.into();
        assert_eq!(docker, NodeMode::Docker);

        let binary: NodeMode = NodeModeArg::Binary.into();
        assert_eq!(binary, NodeMode::Binary);
    }
}
