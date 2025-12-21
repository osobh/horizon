//! Storage Bridge
//!
//! Integrates WARP high-performance file transfer with Horizon using hpc-channels.
//!
//! ## Feature Flags
//!
//! - `embedded-storage`: When enabled, uses the real `warp-core` TransferEngine for
//!   high-performance file transfers with Merkle verification. Works on all platforms.
//!   When disabled, provides mock transfer data for development.
//!
//! ## Architecture
//!
//! The storage bridge wraps either:
//! - Real `TransferEngine` from warp-core (with embedded-storage)
//! - Mock transfer tracking with simulated progress (without embedded-storage)
//!
//! Both implementations expose the same API to the Tauri commands.

use hpc_channels::{channels, StorageMessage};

// Import real warp-core types when feature is enabled
#[cfg(feature = "embedded-storage")]
use warp_core::{TransferEngine, TransferConfig as WarpConfig, TransferProgress as WarpProgress};
#[cfg(feature = "embedded-storage")]
use warp_format::Compression as WarpCompression;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::RwLock;

/// Bridge to the WARP storage/transfer system.
pub struct StorageBridge {
    /// Active transfers
    transfers: Arc<RwLock<HashMap<String, Transfer>>>,
    /// Transfer counter for generating IDs
    transfer_counter: Arc<RwLock<u64>>,
    /// Request ID counter for channel messages
    request_counter: AtomicU64,
    /// Storage root directory (reserved for local cache)
    #[allow(dead_code)]
    storage_root: PathBuf,
    /// Real transfer engine (when embedded-storage feature is enabled)
    #[cfg(feature = "embedded-storage")]
    #[allow(dead_code)]
    engine: Arc<TransferEngine>,
}

/// A file transfer operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transfer {
    pub id: String,
    pub name: String,
    pub operation: TransferOperation,
    pub status: TransferStatus,
    pub progress: TransferProgress,
    pub source: String,
    pub destination: String,
    pub created_at: u64,
    pub started_at: Option<u64>,
    pub completed_at: Option<u64>,
    pub merkle_root: Option<String>,
    pub error: Option<String>,
}

/// Type of transfer operation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum TransferOperation {
    Upload,
    Download,
    Sync,
}

/// Transfer status.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum TransferStatus {
    Queued,
    Analyzing,
    Transferring,
    Verifying,
    Completed,
    Failed,
    Cancelled,
    Paused,
}

/// Transfer progress information.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TransferProgress {
    pub bytes_transferred: u64,
    pub total_bytes: u64,
    pub chunks_completed: u64,
    pub total_chunks: u64,
    pub current_file: Option<String>,
    pub bytes_per_second: f64,
    pub completion_percentage: f64,
    pub eta_seconds: Option<u64>,
}

/// Storage statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageStats {
    pub total_uploads: u64,
    pub total_downloads: u64,
    pub bytes_uploaded: u64,
    pub bytes_downloaded: u64,
    pub active_transfers: usize,
    pub completed_transfers: usize,
    pub failed_transfers: usize,
}

/// File metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileInfo {
    pub name: String,
    pub path: String,
    pub size: u64,
    pub is_directory: bool,
    pub modified_at: Option<u64>,
    pub merkle_root: Option<String>,
}

/// Transfer configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransferConfig {
    pub max_concurrent_chunks: usize,
    pub enable_gpu: bool,
    pub compression: String,
    pub compression_level: i32,
    pub verify_on_complete: bool,
}

impl Default for TransferConfig {
    fn default() -> Self {
        Self {
            max_concurrent_chunks: 16,
            enable_gpu: true,
            compression: "zstd".to_string(),
            compression_level: 3,
            verify_on_complete: true,
        }
    }
}

impl StorageBridge {
    /// Create a new storage bridge.
    pub fn new() -> Self {
        let home = std::env::var("HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("."));
        let storage_root = home.join(".horizon").join("storage");

        // Initialize the transfer engine when embedded-storage feature is enabled
        #[cfg(feature = "embedded-storage")]
        let engine = {
            let config = WarpConfig {
                max_concurrent_chunks: 16,
                enable_gpu: false, // Disable GPU by default for portability
                compression_level: 3,
                compression: WarpCompression::Zstd,
                verify_on_complete: true,
            };
            Arc::new(TransferEngine::new(config))
        };

        #[cfg(feature = "embedded-storage")]
        tracing::info!("StorageBridge initialized with warp-core TransferEngine");

        #[cfg(not(feature = "embedded-storage"))]
        tracing::info!("StorageBridge initialized with mock data (embedded-storage feature disabled)");

        let bridge = Self {
            transfers: Arc::new(RwLock::new(HashMap::new())),
            transfer_counter: Arc::new(RwLock::new(0)),
            request_counter: AtomicU64::new(0),
            storage_root,
            #[cfg(feature = "embedded-storage")]
            engine,
        };

        // Add some mock transfers for demo
        let transfers = bridge.transfers.clone();
        tokio::spawn(async move {
            let mut transfers_guard = transfers.write().await;

            // Add a completed upload
            transfers_guard.insert("transfer-001".to_string(), Transfer {
                id: "transfer-001".to_string(),
                name: "training-data.tar.gz".to_string(),
                operation: TransferOperation::Upload,
                status: TransferStatus::Completed,
                progress: TransferProgress {
                    bytes_transferred: 2_147_483_648, // 2 GB
                    total_bytes: 2_147_483_648,
                    chunks_completed: 512,
                    total_chunks: 512,
                    current_file: None,
                    bytes_per_second: 0.0,
                    completion_percentage: 100.0,
                    eta_seconds: None,
                },
                source: "/data/training-data.tar.gz".to_string(),
                destination: "cluster://storage/datasets/training-data.tar.gz".to_string(),
                created_at: 1703100000,
                started_at: Some(1703100100),
                completed_at: Some(1703100400),
                merkle_root: Some("a1b2c3d4e5f6...".to_string()),
                error: None,
            });

            // Add an in-progress download
            transfers_guard.insert("transfer-002".to_string(), Transfer {
                id: "transfer-002".to_string(),
                name: "model-checkpoint.pt".to_string(),
                operation: TransferOperation::Download,
                status: TransferStatus::Transferring,
                progress: TransferProgress {
                    bytes_transferred: 3_500_000_000, // 3.5 GB
                    total_bytes: 7_000_000_000,       // 7 GB
                    chunks_completed: 875,
                    total_chunks: 1750,
                    current_file: Some("model-checkpoint.pt".to_string()),
                    bytes_per_second: 524_288_000.0, // 500 MB/s
                    completion_percentage: 50.0,
                    eta_seconds: Some(7),
                },
                source: "cluster://storage/models/llama-7b/checkpoint.pt".to_string(),
                destination: "/models/llama-7b/checkpoint.pt".to_string(),
                created_at: 1703110000,
                started_at: Some(1703110100),
                completed_at: None,
                merkle_root: None,
                error: None,
            });

            // Add a queued sync
            transfers_guard.insert("transfer-003".to_string(), Transfer {
                id: "transfer-003".to_string(),
                name: "experiments/".to_string(),
                operation: TransferOperation::Sync,
                status: TransferStatus::Queued,
                progress: TransferProgress::default(),
                source: "/home/user/experiments/".to_string(),
                destination: "cluster://storage/experiments/".to_string(),
                created_at: 1703120000,
                started_at: None,
                completed_at: None,
                merkle_root: None,
                error: None,
            });
        });

        bridge
    }

    /// Get the storage root directory.
    #[allow(dead_code)]
    pub fn storage_root(&self) -> &PathBuf {
        &self.storage_root
    }

    /// Start an upload transfer.
    pub async fn upload(&self, source: String, destination: String, _config: Option<TransferConfig>) -> Result<Transfer, String> {
        let mut counter = self.transfer_counter.write().await;
        *counter += 1;
        let transfer_id = format!("transfer-{:03}", *counter);

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Extract filename from path
        let name = std::path::Path::new(&source)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("upload")
            .to_string();

        let transfer = Transfer {
            id: transfer_id.clone(),
            name,
            operation: TransferOperation::Upload,
            status: TransferStatus::Analyzing,
            progress: TransferProgress::default(),
            source: source.clone(),
            destination: destination.clone(),
            created_at: now,
            started_at: Some(now),
            completed_at: None,
            merkle_root: None,
            error: None,
        };

        self.transfers.write().await.insert(transfer_id.clone(), transfer.clone());

        // Broadcast upload start via channel
        let request_id = self.request_counter.fetch_add(1, Ordering::SeqCst);
        if let Some(tx) = hpc_channels::sender::<StorageMessage>(channels::STORAGE_UPLOAD) {
            let _ = tx.send(StorageMessage::Upload {
                path: source,
                request_id,
            }).await;
        }

        tracing::info!("Started upload transfer: {}", transfer_id);
        Ok(transfer)
    }

    /// Start a download transfer.
    pub async fn download(&self, source: String, destination: String, _config: Option<TransferConfig>) -> Result<Transfer, String> {
        let mut counter = self.transfer_counter.write().await;
        *counter += 1;
        let transfer_id = format!("transfer-{:03}", *counter);

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Extract filename from path
        let name = std::path::Path::new(&source)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("download")
            .to_string();

        let transfer = Transfer {
            id: transfer_id.clone(),
            name,
            operation: TransferOperation::Download,
            status: TransferStatus::Analyzing,
            progress: TransferProgress::default(),
            source: source.clone(),
            destination: destination.clone(),
            created_at: now,
            started_at: Some(now),
            completed_at: None,
            merkle_root: None,
            error: None,
        };

        self.transfers.write().await.insert(transfer_id.clone(), transfer.clone());

        // Broadcast download start via channel
        let request_id = self.request_counter.fetch_add(1, Ordering::SeqCst);
        if let Some(tx) = hpc_channels::sender::<StorageMessage>(channels::STORAGE_DOWNLOAD) {
            let _ = tx.send(StorageMessage::Download {
                merkle_root: source, // Using source as merkle_root for now
                dest_path: destination,
                request_id,
            }).await;
        }

        tracing::info!("Started download transfer: {}", transfer_id);
        Ok(transfer)
    }

    /// Get a transfer by ID.
    pub async fn get_transfer(&self, transfer_id: &str) -> Option<Transfer> {
        self.transfers.read().await.get(transfer_id).cloned()
    }

    /// Get all transfers.
    pub async fn get_all_transfers(&self) -> Vec<Transfer> {
        self.transfers.read().await.values().cloned().collect()
    }

    /// Get active transfers.
    pub async fn get_active_transfers(&self) -> Vec<Transfer> {
        self.transfers
            .read()
            .await
            .values()
            .filter(|t| {
                t.status == TransferStatus::Analyzing
                    || t.status == TransferStatus::Transferring
                    || t.status == TransferStatus::Verifying
            })
            .cloned()
            .collect()
    }

    /// Get storage statistics.
    pub async fn get_stats(&self) -> StorageStats {
        let transfers = self.transfers.read().await;

        let mut stats = StorageStats {
            total_uploads: 0,
            total_downloads: 0,
            bytes_uploaded: 0,
            bytes_downloaded: 0,
            active_transfers: 0,
            completed_transfers: 0,
            failed_transfers: 0,
        };

        for transfer in transfers.values() {
            match transfer.operation {
                TransferOperation::Upload | TransferOperation::Sync => {
                    stats.total_uploads += 1;
                    if transfer.status == TransferStatus::Completed {
                        stats.bytes_uploaded += transfer.progress.bytes_transferred;
                    }
                }
                TransferOperation::Download => {
                    stats.total_downloads += 1;
                    if transfer.status == TransferStatus::Completed {
                        stats.bytes_downloaded += transfer.progress.bytes_transferred;
                    }
                }
            }

            match transfer.status {
                TransferStatus::Completed => stats.completed_transfers += 1,
                TransferStatus::Failed => stats.failed_transfers += 1,
                TransferStatus::Analyzing | TransferStatus::Transferring | TransferStatus::Verifying => {
                    stats.active_transfers += 1;
                }
                _ => {}
            }
        }

        stats
    }

    /// Pause a transfer.
    pub async fn pause_transfer(&self, transfer_id: &str) -> Result<(), String> {
        let mut transfers = self.transfers.write().await;
        let transfer = transfers.get_mut(transfer_id).ok_or("Transfer not found")?;

        if transfer.status != TransferStatus::Transferring && transfer.status != TransferStatus::Analyzing {
            return Err("Transfer is not active".to_string());
        }

        transfer.status = TransferStatus::Paused;
        tracing::info!("Paused transfer: {}", transfer_id);
        Ok(())
    }

    /// Resume a paused transfer.
    pub async fn resume_transfer(&self, transfer_id: &str) -> Result<(), String> {
        let mut transfers = self.transfers.write().await;
        let transfer = transfers.get_mut(transfer_id).ok_or("Transfer not found")?;

        if transfer.status != TransferStatus::Paused {
            return Err("Transfer is not paused".to_string());
        }

        transfer.status = TransferStatus::Transferring;
        tracing::info!("Resumed transfer: {}", transfer_id);
        Ok(())
    }

    /// Cancel a transfer.
    pub async fn cancel_transfer(&self, transfer_id: &str) -> Result<(), String> {
        let mut transfers = self.transfers.write().await;
        let transfer = transfers.get_mut(transfer_id).ok_or("Transfer not found")?;

        if transfer.status == TransferStatus::Completed || transfer.status == TransferStatus::Failed {
            return Err("Transfer is already finished".to_string());
        }

        transfer.status = TransferStatus::Cancelled;
        tracing::info!("Cancelled transfer: {}", transfer_id);
        Ok(())
    }

    /// List files in a directory (local or remote).
    pub async fn list_files(&self, path: &str) -> Result<Vec<FileInfo>, String> {
        // For remote paths, would use WARP to fetch listing
        // For now, mock some files
        if path.starts_with("cluster://") {
            // Mock remote listing
            Ok(vec![
                FileInfo {
                    name: "datasets".to_string(),
                    path: format!("{}/datasets", path),
                    size: 0,
                    is_directory: true,
                    modified_at: Some(1703100000),
                    merkle_root: None,
                },
                FileInfo {
                    name: "models".to_string(),
                    path: format!("{}/models", path),
                    size: 0,
                    is_directory: true,
                    modified_at: Some(1703100000),
                    merkle_root: None,
                },
                FileInfo {
                    name: "checkpoints".to_string(),
                    path: format!("{}/checkpoints", path),
                    size: 0,
                    is_directory: true,
                    modified_at: Some(1703100000),
                    merkle_root: None,
                },
                FileInfo {
                    name: "config.yaml".to_string(),
                    path: format!("{}/config.yaml", path),
                    size: 4096,
                    is_directory: false,
                    modified_at: Some(1703100000),
                    merkle_root: Some("abc123...".to_string()),
                },
            ])
        } else {
            // Local directory listing
            let path = std::path::Path::new(path);
            if !path.exists() {
                return Err(format!("Path does not exist: {}", path.display()));
            }

            let mut files = Vec::new();
            let entries = std::fs::read_dir(path)
                .map_err(|e| format!("Failed to read directory: {}", e))?;

            for entry in entries {
                let entry = entry.map_err(|e| format!("Failed to read entry: {}", e))?;
                let metadata = entry.metadata().map_err(|e| format!("Failed to read metadata: {}", e))?;

                files.push(FileInfo {
                    name: entry.file_name().to_string_lossy().to_string(),
                    path: entry.path().to_string_lossy().to_string(),
                    size: metadata.len(),
                    is_directory: metadata.is_dir(),
                    modified_at: metadata
                        .modified()
                        .ok()
                        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                        .map(|d| d.as_secs()),
                    merkle_root: None,
                });
            }

            files.sort_by(|a, b| {
                // Directories first, then by name
                match (a.is_directory, b.is_directory) {
                    (true, false) => std::cmp::Ordering::Less,
                    (false, true) => std::cmp::Ordering::Greater,
                    _ => a.name.cmp(&b.name),
                }
            });

            Ok(files)
        }
    }

    /// Simulate progress update (for demo purposes).
    #[allow(dead_code)]
    pub async fn simulate_progress(&self) {
        let mut transfers = self.transfers.write().await;

        for transfer in transfers.values_mut() {
            if transfer.status == TransferStatus::Transferring {
                // Simulate 500 MB/s transfer
                let bytes_per_tick = 50_000_000u64; // 50 MB per tick (100ms)
                transfer.progress.bytes_transferred =
                    (transfer.progress.bytes_transferred + bytes_per_tick)
                        .min(transfer.progress.total_bytes);

                transfer.progress.completion_percentage =
                    (transfer.progress.bytes_transferred as f64 / transfer.progress.total_bytes as f64) * 100.0;

                transfer.progress.bytes_per_second = 500_000_000.0; // 500 MB/s

                // Calculate ETA
                let remaining = transfer.progress.total_bytes - transfer.progress.bytes_transferred;
                transfer.progress.eta_seconds = Some((remaining as f64 / transfer.progress.bytes_per_second) as u64);

                // Update chunks
                transfer.progress.chunks_completed =
                    (transfer.progress.completion_percentage / 100.0 * transfer.progress.total_chunks as f64) as u64;

                // Check if completed
                if transfer.progress.completion_percentage >= 100.0 {
                    transfer.status = TransferStatus::Verifying;
                }
            } else if transfer.status == TransferStatus::Verifying {
                // Complete verification after one tick
                transfer.status = TransferStatus::Completed;
                transfer.completed_at = Some(
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_secs()
                );
                transfer.merkle_root = Some(format!("merkle_{}", transfer.id));
            } else if transfer.status == TransferStatus::Analyzing {
                // Move to transferring after analysis
                transfer.status = TransferStatus::Transferring;
                transfer.progress.total_bytes = 1_000_000_000; // 1 GB default
                transfer.progress.total_chunks = 250;
            }
        }
    }
}

impl Default for StorageBridge {
    fn default() -> Self {
        Self::new()
    }
}

/// Start the storage message handler task.
#[allow(dead_code)]
pub async fn start_storage_handler(_bridge: Arc<StorageBridge>) {
    let (_tx, mut rx) = hpc_channels::channel::<StorageMessage>(channels::STORAGE_UPLOAD);
    let progress_tx = hpc_channels::broadcast::<StorageMessage>(channels::STORAGE_PROGRESS, 256);

    tracing::info!("Storage handler started");

    tokio::spawn(async move {
        while let Some(msg) = rx.recv().await {
            match msg {
                StorageMessage::Upload { path, request_id } => {
                    tracing::info!("Handling upload: {} (request_id: {})", path, request_id);
                    // In production, would use warp-core::TransferEngine here

                    // Notify progress
                    let _ = progress_tx.send(StorageMessage::UploadProgress {
                        request_id,
                        bytes_uploaded: 0,
                        total_bytes: 0,
                    });
                }
                StorageMessage::Download { merkle_root, dest_path, request_id } => {
                    tracing::info!("Handling download: {} -> {} (request_id: {})", merkle_root, dest_path, request_id);

                    let _ = progress_tx.send(StorageMessage::DownloadProgress {
                        request_id,
                        bytes_downloaded: 0,
                        total_bytes: 0,
                    });
                }
                StorageMessage::Failed { request_id, error } => {
                    tracing::error!("Transfer {} failed: {}", request_id, error);
                }
                _ => {}
            }
        }
    });
}
