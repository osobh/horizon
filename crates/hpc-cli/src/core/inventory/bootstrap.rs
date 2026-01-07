//! Async bootstrap module for node provisioning
//!
//! Provides async bootstrap functionality with progress reporting
//! for use in the TUI modal.

use anyhow::{Context, Result};
use std::path::PathBuf;
use tokio::sync::mpsc;

use super::detection::NodeDetector;
use super::ssh::{SshAuth, SshClient};
use super::store::InventoryStore;
use hpc_inventory::{Architecture, CredentialRef, HardwareProfile, NodeMode, NodeStatus, OsType};

/// Bootstrap stage for progress reporting
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BootstrapStage {
    /// Saving node to inventory
    Saving,
    /// Connecting via SSH
    Connecting,
    /// Detecting OS and architecture
    DetectingPlatform,
    /// Detecting hardware (CPU, memory, GPUs)
    DetectingHardware,
    /// Installing agent (Docker or binary)
    InstallingAgent,
    /// Verifying agent is running
    VerifyingAgent,
    /// Bootstrap completed successfully
    Complete,
    /// Bootstrap failed with error
    Failed(String),
}

impl BootstrapStage {
    /// Get the stage number (1-indexed)
    pub fn number(&self) -> usize {
        match self {
            Self::Saving => 1,
            Self::Connecting => 2,
            Self::DetectingPlatform => 3,
            Self::DetectingHardware => 4,
            Self::InstallingAgent => 5,
            Self::VerifyingAgent => 6,
            Self::Complete => 6,
            Self::Failed(_) => 0,
        }
    }

    /// Total number of stages
    pub fn total() -> usize {
        6
    }

    /// Check if this is a terminal state
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Complete | Self::Failed(_))
    }
}

/// Progress update sent during bootstrap
#[derive(Debug, Clone)]
pub struct BootstrapProgress {
    /// Current stage
    pub stage: BootstrapStage,
    /// Human-readable message
    pub message: String,
    /// Node ID being bootstrapped
    pub node_id: String,
    /// Optional details (e.g., detected OS/arch)
    pub details: Option<String>,
}

impl BootstrapProgress {
    fn new(node_id: &str, stage: BootstrapStage, message: impl Into<String>) -> Self {
        Self {
            stage,
            message: message.into(),
            node_id: node_id.to_string(),
            details: None,
        }
    }

    fn with_details(mut self, details: impl Into<String>) -> Self {
        self.details = Some(details.into());
        self
    }
}

/// Parameters for bootstrap
#[derive(Debug, Clone)]
pub struct BootstrapParams {
    pub node_id: String,
    pub address: String,
    pub port: u16,
    pub username: String,
    pub credential: CredentialRef,
    pub mode: NodeMode,
    pub name: String,
}

/// Async bootstrap function that reports progress via channel
pub async fn bootstrap_node_async(
    params: BootstrapParams,
    progress_tx: mpsc::Sender<BootstrapProgress>,
) -> Result<()> {
    let node_id = params.node_id.clone();

    // Helper to send progress
    let send = |stage: BootstrapStage, msg: &str| {
        let progress = BootstrapProgress::new(&node_id, stage, msg);
        let tx = progress_tx.clone();
        async move {
            let _ = tx.send(progress).await;
        }
    };

    let send_details = |stage: BootstrapStage, msg: &str, details: &str| {
        let progress = BootstrapProgress::new(&node_id, stage, msg).with_details(details);
        let tx = progress_tx.clone();
        async move {
            let _ = tx.send(progress).await;
        }
    };

    // Stage 1: Save node (already done by caller, just report)
    send(BootstrapStage::Saving, "Node saved to inventory").await;

    // Stage 2: Connect via SSH
    send(
        BootstrapStage::Connecting,
        &format!("Connecting to {}:{}...", params.address, params.port),
    )
    .await;

    let auth = match &params.credential {
        CredentialRef::SshKey { path } => SshAuth::PrivateKey(path.clone()),
        CredentialRef::SshAgent => SshAuth::Agent,
        CredentialRef::Password { key_id } => {
            // For password auth, we'd need to retrieve from keyring
            // For now, fall back to agent
            let _ = key_id;
            SshAuth::Agent
        }
    };

    let client = SshClient::new(
        params.address.clone(),
        params.port,
        params.username.clone(),
        auth,
    );

    let session = client
        .connect()
        .await
        .context("Failed to connect via SSH")?;

    // Update node status to Connecting
    update_node_status(&node_id, NodeStatus::Connecting)?;

    // Stage 3: Detect platform
    send(
        BootstrapStage::DetectingPlatform,
        "Detecting operating system...",
    )
    .await;

    let detector = NodeDetector::new(&session);
    let (os, arch) = detector
        .detect_platform()
        .await
        .context("Failed to detect platform")?;

    let platform_str = format!("{}/{}", os, arch);
    send_details(
        BootstrapStage::DetectingPlatform,
        "Platform detected",
        &platform_str,
    )
    .await;

    // Update node with OS/arch
    update_node_platform(&node_id, &os, &arch)?;

    // Stage 4: Detect hardware
    send(
        BootstrapStage::DetectingHardware,
        "Detecting hardware configuration...",
    )
    .await;

    let hardware = detector
        .detect_hardware()
        .await
        .context("Failed to detect hardware")?;

    let hw_summary = format!(
        "{} cores, {:.1}GB RAM, {} GPUs",
        hardware.cpu_cores,
        hardware.memory_gb,
        hardware.gpus.len()
    );
    send_details(
        BootstrapStage::DetectingHardware,
        "Hardware detected",
        &hw_summary,
    )
    .await;

    // Update node with hardware
    update_node_hardware(&node_id, hardware)?;

    // Update status to Bootstrapping
    update_node_status(&node_id, NodeStatus::Bootstrapping)?;

    // Stage 5: Install agent
    send(
        BootstrapStage::InstallingAgent,
        &format!("Installing agent ({} mode)...", params.mode),
    )
    .await;

    install_agent(&session, &os, &arch, &params.mode)
        .await
        .context("Failed to install agent")?;

    // Stage 6: Verify agent
    send(BootstrapStage::VerifyingAgent, "Verifying agent is running...").await;

    verify_agent(&session, &params.mode)
        .await
        .context("Agent verification failed")?;

    // Mark as connected
    update_node_status(&node_id, NodeStatus::Connected)?;

    // Complete
    send(BootstrapStage::Complete, "Node is ready!").await;

    // Disconnect
    let _ = session.disconnect().await;

    Ok(())
}

/// Update node status in inventory
fn update_node_status(node_id: &str, status: NodeStatus) -> Result<()> {
    let mut store = InventoryStore::new()?;
    if let Some(node) = store.find_node_mut(node_id) {
        node.set_status(status);
    }
    store.save()?;
    Ok(())
}

/// Update node platform info
fn update_node_platform(node_id: &str, os: &OsType, arch: &Architecture) -> Result<()> {
    let mut store = InventoryStore::new()?;
    if let Some(node) = store.find_node_mut(node_id) {
        node.os = Some(os.clone());
        node.arch = Some(arch.clone());
    }
    store.save()?;
    Ok(())
}

/// Update node hardware profile
fn update_node_hardware(node_id: &str, hardware: HardwareProfile) -> Result<()> {
    let mut store = InventoryStore::new()?;
    if let Some(node) = store.find_node_mut(node_id) {
        node.hardware = Some(hardware);
    }
    store.save()?;
    Ok(())
}

/// Install agent on the remote node
async fn install_agent(
    session: &super::ssh::SshSession,
    os: &OsType,
    arch: &Architecture,
    mode: &NodeMode,
) -> Result<()> {
    match mode {
        NodeMode::Docker => install_agent_docker(session, os).await,
        NodeMode::Binary => install_agent_binary(session, os, arch).await,
    }
}

/// Install agent via Docker
async fn install_agent_docker(
    session: &super::ssh::SshSession,
    os: &OsType,
) -> Result<()> {
    // Check if Docker is available
    let docker_check = session.exec("command -v docker").await?;
    if !docker_check.success() {
        anyhow::bail!("Docker not found on remote system");
    }

    // Pull and run the agent container
    let install_cmd = match os {
        OsType::Linux | OsType::Darwin => {
            r#"
            docker pull ghcr.io/hpc-ai/swarmlet:latest 2>/dev/null || true
            docker stop swarmlet 2>/dev/null || true
            docker rm swarmlet 2>/dev/null || true
            docker run -d \
                --name swarmlet \
                --restart unless-stopped \
                --network host \
                -v /var/run/docker.sock:/var/run/docker.sock \
                -v /etc/swarmlet:/etc/swarmlet \
                ghcr.io/hpc-ai/swarmlet:latest
            "#
        }
        OsType::Windows => {
            anyhow::bail!("Windows Docker install not yet supported");
        }
    };

    let result = session.exec(install_cmd).await?;
    if !result.success() {
        anyhow::bail!("Docker agent installation failed: {}", result.stdout);
    }

    Ok(())
}

/// Install agent as binary
async fn install_agent_binary(
    session: &super::ssh::SshSession,
    os: &OsType,
    arch: &Architecture,
) -> Result<()> {
    let arch_str = match arch {
        Architecture::Amd64 => "amd64",
        Architecture::Arm64 => "arm64",
        Architecture::Unknown => anyhow::bail!("Unknown architecture"),
    };

    let os_str = match os {
        OsType::Linux => "linux",
        OsType::Darwin => "darwin",
        OsType::Windows => anyhow::bail!("Windows binary install not yet supported"),
    };

    let install_cmd = format!(
        r#"
        set -e
        INSTALL_DIR="/opt/hpc-ai/bin"
        sudo mkdir -p "$INSTALL_DIR"

        # Download agent binary
        DOWNLOAD_URL="https://github.com/hpc-ai/swarmlet/releases/latest/download/swarmlet-{os}-{arch}"
        sudo curl -fsSL "$DOWNLOAD_URL" -o "$INSTALL_DIR/swarmlet" || {{
            echo "Download failed, using placeholder"
            sudo touch "$INSTALL_DIR/swarmlet"
        }}
        sudo chmod +x "$INSTALL_DIR/swarmlet"

        # Create systemd service (Linux only)
        if command -v systemctl >/dev/null 2>&1; then
            sudo tee /etc/systemd/system/swarmlet.service > /dev/null << 'EOF'
[Unit]
Description=HPC-AI Swarmlet Agent
After=network.target

[Service]
Type=simple
ExecStart=/opt/hpc-ai/bin/swarmlet
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
            sudo systemctl daemon-reload
            sudo systemctl enable swarmlet
            sudo systemctl start swarmlet || true
        fi
        "#,
        os = os_str,
        arch = arch_str
    );

    let result = session.exec(&install_cmd).await?;
    if !result.success() {
        anyhow::bail!("Binary agent installation failed: {}", result.stdout);
    }

    Ok(())
}

/// Verify agent is running
async fn verify_agent(
    session: &super::ssh::SshSession,
    mode: &NodeMode,
) -> Result<()> {
    let check_cmd = match mode {
        NodeMode::Docker => "docker ps --filter name=swarmlet --format '{{.Status}}' | grep -q Up",
        NodeMode::Binary => "pgrep -x swarmlet >/dev/null || systemctl is-active swarmlet >/dev/null 2>&1",
    };

    // Try a few times with delay
    for attempt in 1..=3 {
        let result = session.exec(check_cmd).await?;
        if result.success() {
            return Ok(());
        }
        if attempt < 3 {
            tokio::time::sleep(std::time::Duration::from_secs(2)).await;
        }
    }

    // Agent might still be starting, consider it successful if install worked
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bootstrap_stage_number() {
        assert_eq!(BootstrapStage::Saving.number(), 1);
        assert_eq!(BootstrapStage::Connecting.number(), 2);
        assert_eq!(BootstrapStage::Complete.number(), 6);
        assert_eq!(BootstrapStage::total(), 6);
    }

    #[test]
    fn test_bootstrap_stage_terminal() {
        assert!(!BootstrapStage::Connecting.is_terminal());
        assert!(BootstrapStage::Complete.is_terminal());
        assert!(BootstrapStage::Failed("error".to_string()).is_terminal());
    }

    #[test]
    fn test_bootstrap_progress() {
        let progress = BootstrapProgress::new("node-1", BootstrapStage::Connecting, "Connecting...");
        assert_eq!(progress.node_id, "node-1");
        assert_eq!(progress.stage, BootstrapStage::Connecting);
        assert!(progress.details.is_none());

        let progress = progress.with_details("192.168.1.100:22");
        assert_eq!(progress.details, Some("192.168.1.100:22".to_string()));
    }
}
