//! SSH client for remote node management
//!
//! Provides SSH connectivity for executing commands on remote nodes,
//! uploading/downloading files, and managing the bootstrap process.

use anyhow::{Context, Result};
use async_trait::async_trait;
use russh::client;
use russh_keys::key::PublicKey;
use std::net::SocketAddr;
use std::path::Path;
use std::sync::Arc;
use tokio::net::TcpStream;

use hpc_inventory::CredentialRef;

/// Command execution result
#[derive(Debug, Clone)]
pub struct CommandOutput {
    /// Standard output
    pub stdout: String,
    /// Standard error
    pub stderr: String,
    /// Exit code
    pub exit_code: Option<u32>,
}

impl CommandOutput {
    pub fn success(&self) -> bool {
        self.exit_code == Some(0)
    }
}

/// SSH authentication method
#[derive(Debug, Clone)]
pub enum SshAuth {
    /// Password authentication
    Password(String),
    /// Private key authentication
    PrivateKey(std::path::PathBuf),
    /// SSH agent
    Agent,
}

impl From<&CredentialRef> for SshAuth {
    fn from(cred: &CredentialRef) -> Self {
        match cred {
            CredentialRef::SshKey { path } => SshAuth::PrivateKey(path.clone()),
            CredentialRef::Password { key_id: _ } => {
                // Password would need to be retrieved from keyring
                // For now, fall back to agent
                SshAuth::Agent
            }
            CredentialRef::SshAgent => SshAuth::Agent,
        }
    }
}

/// Trait for remote command execution
#[async_trait]
pub trait RemoteExecutor: Send + Sync {
    /// Execute a command and return output
    async fn execute(&self, command: &str) -> Result<CommandOutput>;

    /// Execute command with sudo
    async fn execute_sudo(&self, command: &str, password: Option<&str>) -> Result<CommandOutput>;

    /// Check if connection is alive
    async fn is_connected(&self) -> bool;

    /// Close the connection
    async fn disconnect(&mut self) -> Result<()>;
}

/// SSH client handler for russh
struct SshHandler {
    /// Collected stdout
    stdout: Vec<u8>,
    /// Collected stderr
    stderr: Vec<u8>,
    /// Exit status
    exit_status: Option<u32>,
}

impl SshHandler {
    fn new() -> Self {
        Self {
            stdout: Vec::new(),
            stderr: Vec::new(),
            exit_status: None,
        }
    }
}

#[async_trait]
impl client::Handler for SshHandler {
    type Error = russh::Error;

    async fn check_server_key(
        &mut self,
        _server_public_key: &PublicKey,
    ) -> Result<bool, Self::Error> {
        // Accept all server keys for now
        // TODO: Implement known_hosts checking
        Ok(true)
    }

    async fn data(
        &mut self,
        _channel: russh::ChannelId,
        data: &[u8],
        _session: &mut client::Session,
    ) -> Result<(), Self::Error> {
        self.stdout.extend_from_slice(data);
        Ok(())
    }

    async fn extended_data(
        &mut self,
        _channel: russh::ChannelId,
        ext: u32,
        data: &[u8],
        _session: &mut client::Session,
    ) -> Result<(), Self::Error> {
        if ext == 1 {
            // stderr
            self.stderr.extend_from_slice(data);
        }
        Ok(())
    }

    async fn exit_status(
        &mut self,
        _channel: russh::ChannelId,
        exit_status: u32,
        _session: &mut client::Session,
    ) -> Result<(), Self::Error> {
        self.exit_status = Some(exit_status);
        Ok(())
    }
}

/// SSH client for connecting to remote nodes
pub struct SshClient {
    /// Remote host address
    host: String,
    /// Remote port
    port: u16,
    /// Username
    username: String,
    /// Authentication method
    auth: SshAuth,
    /// Connection timeout in seconds
    timeout_secs: u64,
}

impl SshClient {
    /// Create a new SSH client
    pub fn new(host: String, port: u16, username: String, auth: SshAuth) -> Self {
        Self {
            host,
            port,
            username,
            auth,
            timeout_secs: 30,
        }
    }

    /// Set connection timeout
    pub fn with_timeout(mut self, secs: u64) -> Self {
        self.timeout_secs = secs;
        self
    }

    /// Connect to the remote host
    pub async fn connect(&self) -> Result<SshSession> {
        let config = Arc::new(client::Config {
            inactivity_timeout: Some(std::time::Duration::from_secs(self.timeout_secs)),
            ..Default::default()
        });

        let addr: SocketAddr = format!("{}:{}", self.host, self.port)
            .parse()
            .with_context(|| format!("Invalid address: {}:{}", self.host, self.port))?;

        let stream = tokio::time::timeout(
            std::time::Duration::from_secs(self.timeout_secs),
            TcpStream::connect(&addr),
        )
        .await
        .with_context(|| format!("Connection timeout to {}", addr))?
        .with_context(|| format!("Failed to connect to {}", addr))?;

        let handler = SshHandler::new();
        let mut session = client::connect_stream(config, stream, handler).await?;

        // Authenticate
        let authenticated = match &self.auth {
            SshAuth::Password(password) => {
                session
                    .authenticate_password(&self.username, password)
                    .await?
            }
            SshAuth::PrivateKey(key_path) => {
                let key = load_private_key(key_path).await?;
                session
                    .authenticate_publickey(&self.username, Arc::new(key))
                    .await?
            }
            SshAuth::Agent => {
                // Try common SSH key paths
                let home = dirs::home_dir().ok_or_else(|| anyhow::anyhow!("Cannot find home directory"))?;
                let key_paths = [
                    home.join(".ssh/id_ed25519"),
                    home.join(".ssh/id_rsa"),
                    home.join(".ssh/id_ecdsa"),
                ];

                let mut auth_success = false;
                for key_path in &key_paths {
                    if key_path.exists() {
                        if let Ok(key) = load_private_key(key_path).await {
                            match session
                                .authenticate_publickey(&self.username, Arc::new(key))
                                .await
                            {
                                Ok(true) => {
                                    auth_success = true;
                                    break;
                                }
                                Ok(false) | Err(_) => continue,
                            }
                        }
                    }
                }
                auth_success
            }
        };

        if !authenticated {
            return Err(anyhow::anyhow!(
                "Authentication failed for user '{}'",
                self.username
            ));
        }

        Ok(SshSession {
            session,
            host: self.host.clone(),
        })
    }
}

/// Active SSH session
pub struct SshSession {
    session: client::Handle<SshHandler>,
    host: String,
}

impl SshSession {
    /// Execute a command on the remote host
    pub async fn exec(&self, command: &str) -> Result<CommandOutput> {
        let mut channel = self.session.channel_open_session().await?;

        channel.exec(true, command).await?;

        // Wait for the channel to close and collect output
        let mut stdout = Vec::new();
        let mut stderr = Vec::new();
        let mut exit_status = None;

        loop {
            match channel.wait().await {
                Some(russh::ChannelMsg::Data { data }) => {
                    stdout.extend_from_slice(&data);
                }
                Some(russh::ChannelMsg::ExtendedData { data, ext }) => {
                    if ext == 1 {
                        stderr.extend_from_slice(&data);
                    }
                }
                Some(russh::ChannelMsg::ExitStatus { exit_status: status }) => {
                    exit_status = Some(status);
                }
                Some(russh::ChannelMsg::Eof) => {}
                Some(russh::ChannelMsg::Close) | None => break,
                _ => {}
            }
        }

        Ok(CommandOutput {
            stdout: String::from_utf8_lossy(&stdout).to_string(),
            stderr: String::from_utf8_lossy(&stderr).to_string(),
            exit_code: exit_status,
        })
    }

    /// Execute command with sudo
    pub async fn exec_sudo(&self, command: &str, password: Option<&str>) -> Result<CommandOutput> {
        let sudo_cmd = if let Some(pwd) = password {
            format!("echo '{}' | sudo -S {}", pwd, command)
        } else {
            format!("sudo {}", command)
        };

        self.exec(&sudo_cmd).await
    }

    /// Upload content to a remote file
    pub async fn upload_content(&self, content: &[u8], remote_path: &str) -> Result<()> {
        let mut channel = self.session.channel_open_session().await?;

        // Use cat to write content
        let cmd = format!("cat > '{}'", remote_path);
        channel.exec(true, cmd).await?;

        // Write content to stdin
        channel.data(content).await?;
        channel.eof().await?;

        // Wait for completion
        loop {
            match channel.wait().await {
                Some(russh::ChannelMsg::Close) | None => break,
                _ => {}
            }
        }

        Ok(())
    }

    /// Download file content from remote
    pub async fn download_content(&self, remote_path: &str) -> Result<Vec<u8>> {
        let cmd = format!("cat '{}'", remote_path);
        let output = self.exec(&cmd).await?;

        if output.success() {
            Ok(output.stdout.into_bytes())
        } else {
            Err(anyhow::anyhow!(
                "Failed to download {}: {}",
                remote_path,
                output.stderr
            ))
        }
    }

    /// Check if a command exists on the remote host
    pub async fn command_exists(&self, cmd: &str) -> Result<bool> {
        let output = self.exec(&format!("command -v {} >/dev/null 2>&1 && echo yes", cmd)).await?;
        Ok(output.stdout.trim() == "yes")
    }

    /// Get the remote hostname
    pub async fn hostname(&self) -> Result<String> {
        let output = self.exec("hostname").await?;
        Ok(output.stdout.trim().to_string())
    }

    /// Check connection status
    pub fn is_connected(&self) -> bool {
        !self.session.is_closed()
    }

    /// Get the host address
    pub fn host(&self) -> &str {
        &self.host
    }

    /// Close the session
    pub async fn disconnect(self) -> Result<()> {
        self.session
            .disconnect(russh::Disconnect::ByApplication, "", "en")
            .await?;
        Ok(())
    }
}

/// Load a private key from file
async fn load_private_key(path: &Path) -> Result<russh_keys::key::KeyPair> {
    let content = tokio::fs::read_to_string(path)
        .await
        .with_context(|| format!("Failed to read key file: {:?}", path))?;

    russh_keys::decode_secret_key(&content, None)
        .with_context(|| format!("Failed to decode private key: {:?}", path))
}

/// Quick helper to execute a single command on a remote host
pub async fn quick_exec(
    host: &str,
    port: u16,
    username: &str,
    auth: SshAuth,
    command: &str,
) -> Result<CommandOutput> {
    let client = SshClient::new(host.to_string(), port, username.to_string(), auth);
    let session = client.connect().await?;
    let output = session.exec(command).await?;
    session.disconnect().await?;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_command_output_success() {
        let output = CommandOutput {
            stdout: "hello".to_string(),
            stderr: String::new(),
            exit_code: Some(0),
        };
        assert!(output.success());

        let failed = CommandOutput {
            stdout: String::new(),
            stderr: "error".to_string(),
            exit_code: Some(1),
        };
        assert!(!failed.success());
    }

    #[test]
    fn test_ssh_auth_from_credential_ref() {
        let key_ref = CredentialRef::SshKey {
            path: "/path/to/key".into(),
        };
        let auth: SshAuth = (&key_ref).into();
        assert!(matches!(auth, SshAuth::PrivateKey(_)));

        let agent_ref = CredentialRef::SshAgent;
        let auth: SshAuth = (&agent_ref).into();
        assert!(matches!(auth, SshAuth::Agent));
    }

    #[test]
    fn test_ssh_client_builder() {
        let client = SshClient::new(
            "192.168.1.100".to_string(),
            22,
            "admin".to_string(),
            SshAuth::Agent,
        )
        .with_timeout(60);

        assert_eq!(client.host, "192.168.1.100");
        assert_eq!(client.port, 22);
        assert_eq!(client.username, "admin");
        assert_eq!(client.timeout_secs, 60);
    }
}
