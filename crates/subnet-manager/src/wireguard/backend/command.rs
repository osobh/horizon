//! Command-line WireGuard backend
//!
//! Uses `wg` and `ip` commands to manage WireGuard interfaces.
//! This is the most portable backend, working on any system with
//! WireGuard tools installed.

use super::traits::{BackendType, InterfaceStats, PeerStats, WireGuardBackend};
use crate::wireguard::{InterfaceConfig, PeerConfig};
use crate::{Error, Result};
use async_trait::async_trait;
use std::net::SocketAddr;
use std::process::Stdio;
use std::time::Duration;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;
use tracing::{debug, error, instrument, trace, warn};

/// Command-line WireGuard backend
///
/// Executes `wg` and `ip` commands to manage WireGuard interfaces.
/// Requires WireGuard tools to be installed on the system.
pub struct CommandBackend {
    /// Path to wg command (default: "wg")
    wg_path: String,
    /// Path to ip command (default: "ip")
    ip_path: String,
    /// Cached availability check
    available: bool,
}

impl CommandBackend {
    /// Create a new command backend with default paths
    pub fn new() -> Self {
        let available = Self::check_availability();
        Self {
            wg_path: "wg".to_string(),
            ip_path: "ip".to_string(),
            available,
        }
    }

    /// Create with custom command paths
    pub fn with_paths(wg_path: String, ip_path: String) -> Self {
        let mut backend = Self {
            wg_path,
            ip_path,
            available: false,
        };
        backend.available = backend.check_wg_available() && backend.check_ip_available();
        backend
    }

    /// Check if wg command is available
    fn check_availability() -> bool {
        std::process::Command::new("which")
            .arg("wg")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    }

    fn check_wg_available(&self) -> bool {
        std::process::Command::new("which")
            .arg(&self.wg_path)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    }

    fn check_ip_available(&self) -> bool {
        std::process::Command::new("which")
            .arg(&self.ip_path)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    }

    /// Write private key to a temporary file and return the path
    ///
    /// The file is created with restricted permissions (0600).
    async fn write_temp_key(&self, private_key: &str) -> Result<tempfile::NamedTempFile> {
        use std::io::Write;

        let mut file = tempfile::NamedTempFile::new().map_err(|e| {
            Error::WireGuardConfig(format!("Failed to create temp file for key: {}", e))
        })?;

        // Set restrictive permissions (Unix only)
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let perms = std::fs::Permissions::from_mode(0o600);
            std::fs::set_permissions(file.path(), perms).map_err(|e| {
                Error::WireGuardConfig(format!("Failed to set key file permissions: {}", e))
            })?;
        }

        file.write_all(private_key.as_bytes()).map_err(|e| {
            Error::WireGuardConfig(format!("Failed to write key to temp file: {}", e))
        })?;

        Ok(file)
    }

    /// Execute a command and return stdout
    async fn exec_command(&self, cmd: &str, args: &[&str]) -> Result<String> {
        trace!(cmd = cmd, args = ?args, "Executing command");

        let output = Command::new(cmd)
            .args(args)
            .output()
            .await
            .map_err(|e| Error::WireGuardConfig(format!("Failed to execute {}: {}", cmd, e)))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(Error::WireGuardConfig(format!(
                "Command {} {} failed: {}",
                cmd,
                args.join(" "),
                stderr
            )));
        }

        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }

    /// Execute a command that may require elevated privileges
    async fn exec_privileged(&self, cmd: &str, args: &[&str]) -> Result<String> {
        // First try without sudo
        match self.exec_command(cmd, args).await {
            Ok(output) => Ok(output),
            Err(_) => {
                // Try with sudo
                debug!("Retrying with sudo: {} {:?}", cmd, args);
                let mut sudo_args = vec![cmd];
                sudo_args.extend(args);
                self.exec_command("sudo", &sudo_args).await
            }
        }
    }

    /// Parse `wg show` output for interface stats
    fn parse_interface_stats(&self, output: &str, name: &str) -> InterfaceStats {
        let mut stats = InterfaceStats {
            name: name.to_string(),
            ..Default::default()
        };

        for line in output.lines() {
            let line = line.trim();
            if line.starts_with("public key:") {
                stats.public_key = line.trim_start_matches("public key:").trim().to_string();
            } else if line.starts_with("listening port:") {
                if let Ok(port) = line
                    .trim_start_matches("listening port:")
                    .trim()
                    .parse::<u16>()
                {
                    stats.listen_port = port;
                }
            } else if line.starts_with("peer:") {
                stats.peer_count += 1;
            } else if line.starts_with("transfer:") {
                // Parse "transfer: X received, Y sent"
                let transfer = line.trim_start_matches("transfer:").trim();
                if let Some((rx, tx)) = transfer.split_once(',') {
                    stats.rx_bytes = Self::parse_bytes(rx.trim());
                    stats.tx_bytes = Self::parse_bytes(tx.trim());
                }
            }
        }

        stats
    }

    /// Parse `wg show` output for peer stats
    fn parse_peer_stats(&self, output: &str, target_key: &str) -> Option<PeerStats> {
        let mut current_peer: Option<PeerStats> = None;
        let mut found_target = false;

        for line in output.lines() {
            let line = line.trim();

            if line.starts_with("peer:") {
                // Save previous peer if it was the target
                if found_target {
                    return current_peer;
                }

                let key = line.trim_start_matches("peer:").trim();
                if key == target_key {
                    found_target = true;
                    current_peer = Some(PeerStats {
                        public_key: key.to_string(),
                        ..Default::default()
                    });
                }
            } else if found_target {
                if let Some(ref mut peer) = current_peer {
                    if line.starts_with("endpoint:") {
                        let endpoint_str = line.trim_start_matches("endpoint:").trim();
                        peer.endpoint = endpoint_str.parse().ok();
                    } else if line.starts_with("allowed ips:") {
                        let ips = line.trim_start_matches("allowed ips:").trim();
                        peer.allowed_ips = ips.split(',').map(|s| s.trim().to_string()).collect();
                    } else if line.starts_with("latest handshake:") {
                        let handshake = line.trim_start_matches("latest handshake:").trim();
                        peer.last_handshake = Self::parse_duration(handshake);
                    } else if line.starts_with("transfer:") {
                        let transfer = line.trim_start_matches("transfer:").trim();
                        if let Some((rx, tx)) = transfer.split_once(',') {
                            peer.rx_bytes = Self::parse_bytes(rx.trim());
                            peer.tx_bytes = Self::parse_bytes(tx.trim());
                        }
                    } else if line.starts_with("persistent keepalive:") {
                        let keepalive = line.trim_start_matches("persistent keepalive:").trim();
                        if keepalive != "off" {
                            if let Some(secs) = keepalive.strip_suffix(" seconds") {
                                peer.persistent_keepalive = secs.parse().ok();
                            }
                        }
                    }
                }
            }
        }

        if found_target {
            current_peer
        } else {
            None
        }
    }

    /// Parse bytes from human-readable format (e.g., "1.23 MiB")
    fn parse_bytes(s: &str) -> u64 {
        let s = s.trim();
        let (num_str, unit) = s
            .split_once(' ')
            .unwrap_or((s.trim_end_matches(|c: char| c.is_alphabetic()), "B"));

        let num: f64 = num_str.parse().unwrap_or(0.0);
        let multiplier: u64 = match unit.trim().to_uppercase().as_str() {
            "B" => 1,
            "KIB" | "KB" => 1024,
            "MIB" | "MB" => 1024 * 1024,
            "GIB" | "GB" => 1024 * 1024 * 1024,
            "TIB" | "TB" => 1024 * 1024 * 1024 * 1024,
            "RECEIVED" => 1, // Handle "X received" format
            "SENT" => 1,     // Handle "Y sent" format
            _ => 1,
        };

        (num * multiplier as f64) as u64
    }

    /// Parse duration from human-readable format (e.g., "1 minute, 30 seconds ago")
    fn parse_duration(s: &str) -> Option<Duration> {
        let s = s.trim().trim_end_matches(" ago");
        let mut total_secs: u64 = 0;

        for part in s.split(',') {
            let part = part.trim();
            if let Some(mins) = part.strip_suffix(" minutes").or(part.strip_suffix(" minute")) {
                total_secs += mins.trim().parse::<u64>().unwrap_or(0) * 60;
            } else if let Some(secs) = part.strip_suffix(" seconds").or(part.strip_suffix(" second"))
            {
                total_secs += secs.trim().parse::<u64>().unwrap_or(0);
            } else if let Some(hours) = part.strip_suffix(" hours").or(part.strip_suffix(" hour")) {
                total_secs += hours.trim().parse::<u64>().unwrap_or(0) * 3600;
            } else if let Some(days) = part.strip_suffix(" days").or(part.strip_suffix(" day")) {
                total_secs += days.trim().parse::<u64>().unwrap_or(0) * 86400;
            }
        }

        if total_secs > 0 {
            Some(Duration::from_secs(total_secs))
        } else {
            None
        }
    }
}

impl Default for CommandBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl WireGuardBackend for CommandBackend {
    fn backend_type(&self) -> BackendType {
        BackendType::Command
    }

    fn is_available(&self) -> bool {
        self.available
    }

    #[instrument(skip(self, config), fields(interface = %config.name))]
    async fn create_interface(&self, config: &InterfaceConfig) -> Result<()> {
        debug!("Creating WireGuard interface");

        // 1. Create the interface
        self.exec_privileged(
            &self.ip_path,
            &["link", "add", &config.name, "type", "wireguard"],
        )
        .await?;

        // 2. Write private key to temp file
        let key_file = self.write_temp_key(&config.private_key).await?;
        let key_path = key_file.path().to_string_lossy().to_string();

        // 3. Set private key and listen port
        let port_str = config.listen_port.to_string();
        self.exec_privileged(
            &self.wg_path,
            &[
                "set",
                &config.name,
                "private-key",
                &key_path,
                "listen-port",
                &port_str,
            ],
        )
        .await?;

        // 4. Assign IP address
        let addr_str = config.address.to_string();
        self.exec_privileged(
            &self.ip_path,
            &["address", "add", &addr_str, "dev", &config.name],
        )
        .await?;

        // 5. Set MTU if specified
        if let Some(mtu) = config.mtu {
            let mtu_str = mtu.to_string();
            self.exec_privileged(
                &self.ip_path,
                &["link", "set", &config.name, "mtu", &mtu_str],
            )
            .await?;
        }

        // 6. Bring interface up
        self.exec_privileged(&self.ip_path, &["link", "set", &config.name, "up"])
            .await?;

        debug!("WireGuard interface created successfully");
        Ok(())
    }

    #[instrument(skip(self))]
    async fn destroy_interface(&self, name: &str) -> Result<()> {
        debug!("Destroying WireGuard interface");
        self.exec_privileged(&self.ip_path, &["link", "delete", name])
            .await?;
        Ok(())
    }

    async fn interface_exists(&self, name: &str) -> Result<bool> {
        match self
            .exec_command(&self.ip_path, &["link", "show", name])
            .await
        {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }

    #[instrument(skip(self, peer))]
    async fn add_peer(&self, interface: &str, peer: &PeerConfig) -> Result<()> {
        debug!(public_key = %peer.public_key, "Adding peer");

        let mut args = vec!["set", interface, "peer", &peer.public_key];

        // Build allowed-ips string
        let allowed_ips: Vec<String> = peer.allowed_ips.iter().map(|ip| ip.to_string()).collect();
        let allowed_ips_str = allowed_ips.join(",");
        if !allowed_ips_str.is_empty() {
            args.push("allowed-ips");
            args.push(&allowed_ips_str);
        }

        // Add endpoint if specified
        let endpoint_str;
        if let Some(endpoint) = &peer.endpoint {
            endpoint_str = endpoint.to_string();
            args.push("endpoint");
            args.push(&endpoint_str);
        }

        // Add preshared key if specified
        let psk_file;
        let psk_path;
        if let Some(psk) = &peer.preshared_key {
            psk_file = self.write_temp_key(psk).await?;
            psk_path = psk_file.path().to_string_lossy().to_string();
            args.push("preshared-key");
            args.push(&psk_path);
        }

        // Add persistent keepalive if specified
        let keepalive_str;
        if let Some(keepalive) = peer.persistent_keepalive {
            keepalive_str = keepalive.to_string();
            args.push("persistent-keepalive");
            args.push(&keepalive_str);
        }

        self.exec_privileged(&self.wg_path, &args).await?;
        Ok(())
    }

    #[instrument(skip(self))]
    async fn remove_peer(&self, interface: &str, public_key: &str) -> Result<()> {
        debug!("Removing peer");
        self.exec_privileged(&self.wg_path, &["set", interface, "peer", public_key, "remove"])
            .await?;
        Ok(())
    }

    async fn update_endpoint(
        &self,
        interface: &str,
        public_key: &str,
        endpoint: SocketAddr,
    ) -> Result<()> {
        let endpoint_str = endpoint.to_string();
        self.exec_privileged(
            &self.wg_path,
            &["set", interface, "peer", public_key, "endpoint", &endpoint_str],
        )
        .await?;
        Ok(())
    }

    async fn get_interface_stats(&self, interface: &str) -> Result<InterfaceStats> {
        let output = self
            .exec_command(&self.wg_path, &["show", interface])
            .await?;
        Ok(self.parse_interface_stats(&output, interface))
    }

    async fn get_peer_stats(&self, interface: &str, public_key: &str) -> Result<PeerStats> {
        let output = self
            .exec_command(&self.wg_path, &["show", interface])
            .await?;
        self.parse_peer_stats(&output, public_key)
            .ok_or_else(|| Error::WireGuardConfig(format!("Peer {} not found", public_key)))
    }

    async fn list_peers(&self, interface: &str) -> Result<Vec<PeerStats>> {
        let output = self
            .exec_command(&self.wg_path, &["show", interface])
            .await?;

        let mut peers = Vec::new();
        let mut current_peer: Option<PeerStats> = None;

        for line in output.lines() {
            let line = line.trim();

            if line.starts_with("peer:") {
                // Save previous peer
                if let Some(peer) = current_peer.take() {
                    peers.push(peer);
                }

                let key = line.trim_start_matches("peer:").trim();
                current_peer = Some(PeerStats {
                    public_key: key.to_string(),
                    ..Default::default()
                });
            } else if let Some(ref mut peer) = current_peer {
                if line.starts_with("endpoint:") {
                    let endpoint_str = line.trim_start_matches("endpoint:").trim();
                    peer.endpoint = endpoint_str.parse().ok();
                } else if line.starts_with("allowed ips:") {
                    let ips = line.trim_start_matches("allowed ips:").trim();
                    peer.allowed_ips = ips.split(',').map(|s| s.trim().to_string()).collect();
                } else if line.starts_with("latest handshake:") {
                    let handshake = line.trim_start_matches("latest handshake:").trim();
                    peer.last_handshake = Self::parse_duration(handshake);
                } else if line.starts_with("transfer:") {
                    let transfer = line.trim_start_matches("transfer:").trim();
                    if let Some((rx, tx)) = transfer.split_once(',') {
                        peer.rx_bytes = Self::parse_bytes(rx.trim());
                        peer.tx_bytes = Self::parse_bytes(tx.trim());
                    }
                } else if line.starts_with("persistent keepalive:") {
                    let keepalive = line.trim_start_matches("persistent keepalive:").trim();
                    if keepalive != "off" {
                        if let Some(secs) = keepalive.strip_suffix(" seconds") {
                            peer.persistent_keepalive = secs.parse().ok();
                        }
                    }
                }
            }
        }

        // Don't forget the last peer
        if let Some(peer) = current_peer {
            peers.push(peer);
        }

        Ok(peers)
    }

    async fn set_mtu(&self, interface: &str, mtu: u16) -> Result<()> {
        let mtu_str = mtu.to_string();
        self.exec_privileged(&self.ip_path, &["link", "set", interface, "mtu", &mtu_str])
            .await?;
        Ok(())
    }

    async fn bring_up(&self, interface: &str) -> Result<()> {
        self.exec_privileged(&self.ip_path, &["link", "set", interface, "up"])
            .await?;
        Ok(())
    }

    async fn bring_down(&self, interface: &str) -> Result<()> {
        self.exec_privileged(&self.ip_path, &["link", "set", interface, "down"])
            .await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_bytes() {
        assert_eq!(CommandBackend::parse_bytes("0 B"), 0);
        assert_eq!(CommandBackend::parse_bytes("1024 B"), 1024);
        assert_eq!(CommandBackend::parse_bytes("1 KiB"), 1024);
        assert_eq!(CommandBackend::parse_bytes("1 MiB"), 1024 * 1024);
        assert_eq!(CommandBackend::parse_bytes("1.5 MiB"), 1572864);
    }

    #[test]
    fn test_parse_duration() {
        assert_eq!(
            CommandBackend::parse_duration("30 seconds ago"),
            Some(Duration::from_secs(30))
        );
        assert_eq!(
            CommandBackend::parse_duration("1 minute, 30 seconds ago"),
            Some(Duration::from_secs(90))
        );
        assert_eq!(
            CommandBackend::parse_duration("2 hours, 5 minutes ago"),
            Some(Duration::from_secs(2 * 3600 + 5 * 60))
        );
    }

    #[test]
    fn test_backend_type() {
        let backend = CommandBackend::new();
        assert_eq!(backend.backend_type(), BackendType::Command);
    }
}
