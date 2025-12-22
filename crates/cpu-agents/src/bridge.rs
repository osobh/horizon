//! CPU-GPU Bridge for message-based communication
//!
//! Enables communication between CPU and GPU agents via shared storage

use crate::{CpuAgentError, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::Duration;
use tokio::fs;
use tokio::time::interval;

/// Message types for CPU-GPU communication
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MessageType {
    /// Request task execution
    TaskRequest,
    /// Return task results
    TaskResult,
    /// Transfer large data
    DataTransfer,
    /// Status update
    StatusUpdate,
    /// Error report
    ErrorReport,
    /// Shutdown signal
    Shutdown,
}

/// Message for CPU-GPU communication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuGpuMessage {
    /// Unique message identifier
    pub id: String,
    /// Type of message
    pub message_type: MessageType,
    /// Source agent identifier
    pub source: String,
    /// Destination agent identifier
    pub destination: String,
    /// Message payload (JSON)
    pub payload: serde_json::Value,
    /// Message timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Message priority (1-10, 10 highest)
    pub priority: u8,
}

/// Bridge configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeConfig {
    /// Shared storage path for communication
    pub shared_storage_path: PathBuf,
    /// Inbox directory name (CPU receives from GPU)
    pub inbox_dir: String,
    /// Outbox directory name (CPU sends to GPU)
    pub outbox_dir: String,
    /// Message retention time in seconds
    pub message_retention_seconds: u64,
    /// Maximum message size in MB
    pub max_message_size_mb: usize,
    /// Polling interval in milliseconds
    pub polling_interval_ms: u64,
}

impl Default for BridgeConfig {
    fn default() -> Self {
        Self {
            shared_storage_path: PathBuf::from("/tmp/exorust-bridge"),
            inbox_dir: "inbox".to_string(),
            outbox_dir: "outbox".to_string(),
            message_retention_seconds: 3600, // 1 hour
            max_message_size_mb: 10,
            polling_interval_ms: 100,
        }
    }
}

/// CPU-GPU communication bridge
pub struct CpuGpuBridge {
    config: BridgeConfig,
    inbox_path: PathBuf,
    outbox_path: PathBuf,
    stats: BridgeStats,
    cleanup_task: Option<tokio::task::JoinHandle<()>>,
}

/// Bridge operation statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BridgeStats {
    pub messages_sent: u64,
    pub messages_received: u64,
    pub messages_failed: u64,
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub cleanup_operations: u64,
}

impl CpuGpuBridge {
    /// Create new CPU-GPU bridge
    pub async fn new(config: BridgeConfig) -> Result<Self> {
        let inbox_path = config.shared_storage_path.join(&config.inbox_dir);
        let outbox_path = config.shared_storage_path.join(&config.outbox_dir);

        // Create directories
        fs::create_dir_all(&inbox_path).await.map_err(|e| {
            CpuAgentError::BridgeError(format!("Failed to create inbox directory: {}", e))
        })?;

        fs::create_dir_all(&outbox_path).await.map_err(|e| {
            CpuAgentError::BridgeError(format!("Failed to create outbox directory: {}", e))
        })?;

        log::info!(
            "CPU-GPU Bridge initialized: inbox={}, outbox={}",
            inbox_path.display(),
            outbox_path.display()
        );

        Ok(Self {
            config,
            inbox_path,
            outbox_path,
            stats: BridgeStats::default(),
            cleanup_task: None,
        })
    }

    /// Get bridge configuration
    pub fn config(&self) -> &BridgeConfig {
        &self.config
    }

    /// Get bridge statistics
    pub fn stats(&self) -> &BridgeStats {
        &self.stats
    }

    /// Start bridge background tasks
    pub async fn start(&mut self) -> Result<()> {
        // Start cleanup task
        let cleanup_handle = self.start_cleanup_task().await?;
        self.cleanup_task = Some(cleanup_handle);

        log::info!("CPU-GPU Bridge started");
        Ok(())
    }

    /// Stop bridge and cleanup
    pub async fn stop(&mut self) -> Result<()> {
        if let Some(handle) = self.cleanup_task.take() {
            handle.abort();
        }

        log::info!("CPU-GPU Bridge stopped");
        Ok(())
    }

    /// Send message to GPU
    pub async fn send_to_gpu(&mut self, message: CpuGpuMessage) -> Result<()> {
        self.validate_message(&message)?;

        let filename = format!("{}.json", message.id);
        let filepath = self.outbox_path.join(&filename);

        let json_content = serde_json::to_string_pretty(&message).map_err(|e| {
            CpuAgentError::MessageError(format!("Failed to serialize message: {}", e))
        })?;

        // Check message size
        let size_mb = json_content.len() as f64 / (1024.0 * 1024.0);
        if size_mb > self.config.max_message_size_mb as f64 {
            return Err(CpuAgentError::MessageError(format!(
                "Message too large: {:.2}MB > {}MB limit",
                size_mb, self.config.max_message_size_mb
            )));
        }

        fs::write(&filepath, &json_content).await.map_err(|e| {
            CpuAgentError::BridgeError(format!("Failed to write message file: {}", e))
        })?;

        self.stats.messages_sent += 1;
        self.stats.bytes_sent += json_content.len() as u64;

        log::debug!(
            "Message sent to GPU: {} ({:?})",
            message.id,
            message.message_type
        );
        Ok(())
    }

    /// Receive messages from GPU
    pub async fn receive_from_gpu(&mut self) -> Result<Vec<CpuGpuMessage>> {
        let mut messages = Vec::new();

        let mut entries = fs::read_dir(&self.inbox_path).await.map_err(|e| {
            CpuAgentError::BridgeError(format!("Failed to read inbox directory: {}", e))
        })?;

        while let Some(entry) = entries.next_entry().await.map_err(|e| {
            CpuAgentError::BridgeError(format!("Failed to read directory entry: {}", e))
        })? {
            let path = entry.path();

            if path.extension().and_then(|s| s.to_str()) == Some("json") {
                match self.read_message(&path).await {
                    Ok(message) => {
                        messages.push(message);

                        // Remove processed message
                        if let Err(e) = fs::remove_file(&path).await {
                            log::warn!(
                                "Failed to remove processed message {}: {}",
                                path.display(),
                                e
                            );
                        }
                    }
                    Err(e) => {
                        log::error!("Failed to read message from {}: {}", path.display(), e);
                        self.stats.messages_failed += 1;

                        // Move corrupted message to error directory
                        self.handle_corrupted_message(&path).await;
                    }
                }
            }
        }

        // Sort by priority (highest first) and timestamp
        messages.sort_by(|a, b| {
            b.priority
                .cmp(&a.priority)
                .then(a.timestamp.cmp(&b.timestamp))
        });

        self.stats.messages_received += messages.len() as u64;

        if !messages.is_empty() {
            log::debug!("Received {} messages from GPU", messages.len());
        }

        Ok(messages)
    }

    /// Read message from file
    async fn read_message(&mut self, path: &Path) -> Result<CpuGpuMessage> {
        let content = fs::read_to_string(path).await.map_err(|e| {
            CpuAgentError::BridgeError(format!("Failed to read message file: {}", e))
        })?;

        let message: CpuGpuMessage = serde_json::from_str(&content).map_err(|e| {
            CpuAgentError::MessageError(format!("Failed to deserialize message: {}", e))
        })?;

        self.stats.bytes_received += content.len() as u64;
        Ok(message)
    }

    /// Handle corrupted message file
    async fn handle_corrupted_message(&self, path: &Path) {
        let error_dir = self.config.shared_storage_path.join("error");

        if let Err(e) = fs::create_dir_all(&error_dir).await {
            log::error!("Failed to create error directory: {}", e);
            return;
        }

        if let Some(filename) = path.file_name() {
            let error_path = error_dir.join(filename);
            if let Err(e) = fs::rename(path, &error_path).await {
                log::error!("Failed to move corrupted message to error directory: {}", e);
                // Try to delete the corrupted file
                let _ = fs::remove_file(path).await;
            } else {
                log::info!("Moved corrupted message to: {}", error_path.display());
            }
        }
    }

    /// Validate message before sending
    fn validate_message(&self, message: &CpuGpuMessage) -> Result<()> {
        if message.id.is_empty() {
            return Err(CpuAgentError::MessageError(
                "Message ID cannot be empty".to_string(),
            ));
        }

        if message.source.is_empty() {
            return Err(CpuAgentError::MessageError(
                "Message source cannot be empty".to_string(),
            ));
        }

        if message.destination.is_empty() {
            return Err(CpuAgentError::MessageError(
                "Message destination cannot be empty".to_string(),
            ));
        }

        if message.priority == 0 || message.priority > 10 {
            return Err(CpuAgentError::MessageError(
                "Message priority must be between 1 and 10".to_string(),
            ));
        }

        Ok(())
    }

    /// Start cleanup task for old messages
    async fn start_cleanup_task(&self) -> Result<tokio::task::JoinHandle<()>> {
        let outbox_path = self.outbox_path.clone();
        let inbox_path = self.inbox_path.clone();
        let retention_duration = Duration::from_secs(self.config.message_retention_seconds);
        let cleanup_interval = Duration::from_secs(60); // Cleanup every minute

        let handle = tokio::spawn(async move {
            let mut cleanup_timer = interval(cleanup_interval);

            loop {
                cleanup_timer.tick().await;

                // Cleanup outbox
                if let Err(e) = Self::cleanup_directory(&outbox_path, retention_duration).await {
                    log::error!("Failed to cleanup outbox: {}", e);
                }

                // Cleanup inbox
                if let Err(e) = Self::cleanup_directory(&inbox_path, retention_duration).await {
                    log::error!("Failed to cleanup inbox: {}", e);
                }
            }
        });

        Ok(handle)
    }

    /// Cleanup old files in directory
    async fn cleanup_directory(dir_path: &Path, retention_duration: Duration) -> Result<()> {
        let mut entries = fs::read_dir(dir_path).await.map_err(|e| {
            CpuAgentError::BridgeError(format!("Failed to read directory for cleanup: {}", e))
        })?;

        let cutoff_time = std::time::SystemTime::now() - retention_duration;

        while let Some(entry) = entries.next_entry().await.map_err(|e| {
            CpuAgentError::BridgeError(format!("Failed to read directory entry: {}", e))
        })? {
            let path = entry.path();

            if let Ok(metadata) = entry.metadata().await {
                if let Ok(modified) = metadata.modified() {
                    if modified < cutoff_time {
                        if let Err(e) = fs::remove_file(&path).await {
                            log::warn!("Failed to cleanup old file {}: {}", path.display(), e);
                        } else {
                            log::debug!("Cleaned up old file: {}", path.display());
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

impl Drop for CpuGpuBridge {
    fn drop(&mut self) {
        if let Some(handle) = self.cleanup_task.take() {
            handle.abort();
        }
    }
}
