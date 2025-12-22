//! I/O Manager for file and network operations
//!
//! Handles all I/O operations without GPU dependencies

use crate::{CpuAgentError, Result};
use flate2::{read::GzDecoder, write::GzEncoder, Compression};
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use tokio::fs;

/// I/O operation types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IoOperation {
    /// Read file from filesystem
    ReadFile { path: PathBuf },
    /// Write file to filesystem
    WriteFile {
        path: PathBuf,
        content: Vec<u8>,
        create_dirs: bool,
    },
    /// Delete file from filesystem
    DeleteFile { path: PathBuf },
    /// Create directory
    CreateDirectory { path: PathBuf, recursive: bool },
    /// List directory contents
    ListDirectory { path: PathBuf, recursive: bool },
    /// Copy file
    CopyFile {
        source: PathBuf,
        destination: PathBuf,
    },
    /// Move file
    MoveFile {
        source: PathBuf,
        destination: PathBuf,
    },
    /// Write compressed file
    WriteCompressed { path: PathBuf, content: Vec<u8> },
    /// Read compressed file
    ReadCompressed { path: PathBuf },
    /// Batch write multiple files
    BatchWrite { operations: Vec<(PathBuf, Vec<u8>)> },
}

/// Result of I/O operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IoResult {
    /// Operation completed successfully
    Success {
        message: String,
        bytes_processed: usize,
    },
    /// File content read
    FileContent(Vec<u8>),
    /// Directory listing
    DirectoryListing(Vec<PathBuf>),
    /// Batch operation results
    BatchResults(Vec<IoResult>),
}

/// I/O configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IoConfig {
    /// Maximum file size in MB
    pub max_file_size_mb: usize,
    /// Buffer size for I/O operations in KB
    pub buffer_size_kb: usize,
    /// Enable compression for large files
    pub enable_compression: bool,
    /// Temporary directory for operations
    pub temp_dir: PathBuf,
}

impl Default for IoConfig {
    fn default() -> Self {
        Self {
            max_file_size_mb: 100,
            buffer_size_kb: 64,
            enable_compression: true,
            temp_dir: PathBuf::from("/tmp/exorust-io"),
        }
    }
}

/// I/O Manager for handling file operations
pub struct IoManager {
    config: IoConfig,
    stats: IoStats,
}

/// I/O operation statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IoStats {
    pub files_read: u64,
    pub files_written: u64,
    pub files_deleted: u64,
    pub bytes_read: u64,
    pub bytes_written: u64,
    pub operations_failed: u64,
    pub total_operation_time_ms: f64,
}

impl IoManager {
    /// Create new I/O manager
    pub async fn new(config: IoConfig) -> Result<Self> {
        // Ensure temp directory exists
        if let Err(e) = fs::create_dir_all(&config.temp_dir).await {
            return Err(CpuAgentError::IoError(format!(
                "Failed to create temp directory: {}",
                e
            )));
        }

        Ok(Self {
            config,
            stats: IoStats::default(),
        })
    }

    /// Get configuration
    pub fn config(&self) -> &IoConfig {
        &self.config
    }

    /// Get I/O statistics
    pub fn stats(&self) -> &IoStats {
        &self.stats
    }

    /// Execute single I/O operation
    pub async fn execute(&mut self, operation: IoOperation) -> Result<IoResult> {
        let start = std::time::Instant::now();

        let result = match operation {
            IoOperation::ReadFile { path } => self.read_file(&path).await,
            IoOperation::WriteFile {
                path,
                content,
                create_dirs,
            } => self.write_file(&path, &content, create_dirs).await,
            IoOperation::DeleteFile { path } => self.delete_file(&path).await,
            IoOperation::CreateDirectory { path, recursive } => {
                self.create_directory(&path, recursive).await
            }
            IoOperation::ListDirectory { path, recursive } => {
                self.list_directory(&path, recursive).await
            }
            IoOperation::CopyFile {
                source,
                destination,
            } => self.copy_file(&source, &destination).await,
            IoOperation::MoveFile {
                source,
                destination,
            } => self.move_file(&source, &destination).await,
            IoOperation::WriteCompressed { path, content } => {
                self.write_compressed(&path, &content).await
            }
            IoOperation::ReadCompressed { path } => self.read_compressed(&path).await,
            IoOperation::BatchWrite { operations } => self.batch_write(operations).await,
        };

        let duration = start.elapsed();
        self.stats.total_operation_time_ms += duration.as_millis() as f64;

        match &result {
            Ok(_) => {}
            Err(_) => self.stats.operations_failed += 1,
        }

        result
    }

    /// Execute batch of I/O operations
    pub async fn execute_batch(&mut self, operations: Vec<IoOperation>) -> Result<Vec<IoResult>> {
        let mut results = Vec::with_capacity(operations.len());

        for operation in operations {
            let result = self.execute(operation).await;
            results.push(result?);
        }

        Ok(results)
    }

    /// Read file from filesystem
    async fn read_file(&mut self, path: &Path) -> Result<IoResult> {
        self.validate_file_size(path).await?;

        match fs::read(path).await {
            Ok(content) => {
                self.stats.files_read += 1;
                self.stats.bytes_read += content.len() as u64;
                Ok(IoResult::FileContent(content))
            }
            Err(e) => Err(CpuAgentError::IoError(format!(
                "Failed to read file {}: {}",
                path.display(),
                e
            ))),
        }
    }

    /// Write file to filesystem
    async fn write_file(
        &mut self,
        path: &Path,
        content: &[u8],
        create_dirs: bool,
    ) -> Result<IoResult> {
        self.validate_content_size(content)?;

        if create_dirs {
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent).await.map_err(|e| {
                    CpuAgentError::IoError(format!("Failed to create directories: {}", e))
                })?;
            }
        }

        match fs::write(path, content).await {
            Ok(_) => {
                self.stats.files_written += 1;
                self.stats.bytes_written += content.len() as u64;
                Ok(IoResult::Success {
                    message: format!("File written: {}", path.display()),
                    bytes_processed: content.len(),
                })
            }
            Err(e) => Err(CpuAgentError::IoError(format!(
                "Failed to write file {}: {}",
                path.display(),
                e
            ))),
        }
    }

    /// Delete file from filesystem
    async fn delete_file(&mut self, path: &Path) -> Result<IoResult> {
        match fs::remove_file(path).await {
            Ok(_) => {
                self.stats.files_deleted += 1;
                Ok(IoResult::Success {
                    message: format!("File deleted: {}", path.display()),
                    bytes_processed: 0,
                })
            }
            Err(e) => Err(CpuAgentError::IoError(format!(
                "Failed to delete file {}: {}",
                path.display(),
                e
            ))),
        }
    }

    /// Create directory
    async fn create_directory(&mut self, path: &Path, recursive: bool) -> Result<IoResult> {
        let result = if recursive {
            fs::create_dir_all(path).await
        } else {
            fs::create_dir(path).await
        };

        match result {
            Ok(_) => Ok(IoResult::Success {
                message: format!("Directory created: {}", path.display()),
                bytes_processed: 0,
            }),
            Err(e) => Err(CpuAgentError::IoError(format!(
                "Failed to create directory {}: {}",
                path.display(),
                e
            ))),
        }
    }

    /// List directory contents
    async fn list_directory(&mut self, path: &Path, recursive: bool) -> Result<IoResult> {
        if recursive {
            self.list_directory_recursive(path).await
        } else {
            self.list_directory_simple(path).await
        }
    }

    /// Simple directory listing
    async fn list_directory_simple(&mut self, path: &Path) -> Result<IoResult> {
        match fs::read_dir(path).await {
            Ok(mut entries) => {
                let mut files = Vec::new();
                while let Some(entry) = entries.next_entry().await.map_err(|e| {
                    CpuAgentError::IoError(format!("Failed to read directory entry: {}", e))
                })? {
                    files.push(entry.path());
                }
                Ok(IoResult::DirectoryListing(files))
            }
            Err(e) => Err(CpuAgentError::IoError(format!(
                "Failed to list directory {}: {}",
                path.display(),
                e
            ))),
        }
    }

    /// Recursive directory listing
    async fn list_directory_recursive(&mut self, path: &Path) -> Result<IoResult> {
        let mut files = Vec::new();
        self.collect_files_recursive(path, &mut files).await?;
        Ok(IoResult::DirectoryListing(files))
    }

    /// Recursively collect files
    fn collect_files_recursive<'a>(
        &'a self,
        path: &'a Path,
        files: &'a mut Vec<PathBuf>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            let mut entries = fs::read_dir(path).await.map_err(|e| {
                CpuAgentError::IoError(format!(
                    "Failed to read directory {}: {}",
                    path.display(),
                    e
                ))
            })?;

            while let Some(entry) = entries.next_entry().await.map_err(|e| {
                CpuAgentError::IoError(format!("Failed to read directory entry: {}", e))
            })? {
                let entry_path = entry.path();
                if entry_path.is_dir() {
                    self.collect_files_recursive(&entry_path, files).await?;
                } else {
                    files.push(entry_path);
                }
            }

            Ok(())
        })
    }

    /// Copy file
    async fn copy_file(&mut self, source: &Path, destination: &Path) -> Result<IoResult> {
        self.validate_file_size(source).await?;

        match fs::copy(source, destination).await {
            Ok(bytes_copied) => {
                self.stats.bytes_read += bytes_copied;
                self.stats.bytes_written += bytes_copied;
                Ok(IoResult::Success {
                    message: format!(
                        "File copied: {} -> {}",
                        source.display(),
                        destination.display()
                    ),
                    bytes_processed: bytes_copied as usize,
                })
            }
            Err(e) => Err(CpuAgentError::IoError(format!(
                "Failed to copy file {} to {}: {}",
                source.display(),
                destination.display(),
                e
            ))),
        }
    }

    /// Move file
    async fn move_file(&mut self, source: &Path, destination: &Path) -> Result<IoResult> {
        match fs::rename(source, destination).await {
            Ok(_) => Ok(IoResult::Success {
                message: format!(
                    "File moved: {} -> {}",
                    source.display(),
                    destination.display()
                ),
                bytes_processed: 0,
            }),
            Err(e) => Err(CpuAgentError::IoError(format!(
                "Failed to move file {} to {}: {}",
                source.display(),
                destination.display(),
                e
            ))),
        }
    }

    /// Write compressed file
    async fn write_compressed(&mut self, path: &Path, content: &[u8]) -> Result<IoResult> {
        self.validate_content_size(content)?;

        let compressed = tokio::task::spawn_blocking({
            let content = content.to_vec();
            move || -> Result<Vec<u8>> {
                let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
                encoder
                    .write_all(&content)
                    .map_err(|e| CpuAgentError::IoError(format!("Compression failed: {}", e)))?;
                encoder.finish().map_err(|e| {
                    CpuAgentError::IoError(format!("Compression finish failed: {}", e))
                })
            }
        })
        .await
        .map_err(|e| CpuAgentError::IoError(format!("Compression task failed: {}", e)))??;

        match fs::write(path, &compressed).await {
            Ok(_) => {
                self.stats.files_written += 1;
                self.stats.bytes_written += compressed.len() as u64;
                Ok(IoResult::Success {
                    message: format!(
                        "Compressed file written: {} ({}->{})",
                        path.display(),
                        content.len(),
                        compressed.len()
                    ),
                    bytes_processed: compressed.len(),
                })
            }
            Err(e) => Err(CpuAgentError::IoError(format!(
                "Failed to write compressed file {}: {}",
                path.display(),
                e
            ))),
        }
    }

    /// Read compressed file
    async fn read_compressed(&mut self, path: &Path) -> Result<IoResult> {
        let compressed = fs::read(path).await.map_err(|e| {
            CpuAgentError::IoError(format!(
                "Failed to read compressed file {}: {}",
                path.display(),
                e
            ))
        })?;

        let decompressed = tokio::task::spawn_blocking(move || -> Result<Vec<u8>> {
            let mut decoder = GzDecoder::new(&compressed[..]);
            let mut content = Vec::new();
            decoder
                .read_to_end(&mut content)
                .map_err(|e| CpuAgentError::IoError(format!("Decompression failed: {}", e)))?;
            Ok(content)
        })
        .await
        .map_err(|e| CpuAgentError::IoError(format!("Decompression task failed: {}", e)))??;

        self.stats.files_read += 1;
        self.stats.bytes_read += decompressed.len() as u64;

        Ok(IoResult::FileContent(decompressed))
    }

    /// Batch write multiple files
    async fn batch_write(&mut self, operations: Vec<(PathBuf, Vec<u8>)>) -> Result<IoResult> {
        let mut results = Vec::with_capacity(operations.len());

        for (path, content) in operations {
            let result = self.write_file(&path, &content, false).await?;
            results.push(result);
        }

        Ok(IoResult::BatchResults(results))
    }

    /// Validate file size before reading
    async fn validate_file_size(&self, path: &Path) -> Result<()> {
        match fs::metadata(path).await {
            Ok(metadata) => {
                let size_mb = metadata.len() as f64 / (1024.0 * 1024.0);
                if size_mb > self.config.max_file_size_mb as f64 {
                    return Err(CpuAgentError::IoError(format!(
                        "File too large: {:.2}MB > {}MB limit",
                        size_mb, self.config.max_file_size_mb
                    )));
                }
                Ok(())
            }
            Err(e) => Err(CpuAgentError::IoError(format!(
                "Failed to get file metadata: {}",
                e
            ))),
        }
    }

    /// Validate content size before writing
    fn validate_content_size(&self, content: &[u8]) -> Result<()> {
        let size_mb = content.len() as f64 / (1024.0 * 1024.0);
        if size_mb > self.config.max_file_size_mb as f64 {
            return Err(CpuAgentError::IoError(format!(
                "Content too large: {:.2}MB > {}MB limit",
                size_mb, self.config.max_file_size_mb
            )));
        }
        Ok(())
    }
}
