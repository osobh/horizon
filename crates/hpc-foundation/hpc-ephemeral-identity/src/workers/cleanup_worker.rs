//! Identity Cleanup Worker
//!
//! Background worker that processes expired ephemeral identities,
//! cleans up unused invitations, and maintains the token revocation list.

use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

/// Configuration for the cleanup worker
#[derive(Debug, Clone)]
pub struct CleanupWorkerConfig {
    /// Interval between cleanup runs
    pub interval: Duration,
    /// Batch size for processing expired items
    pub batch_size: usize,
    /// How long to retain expired identities before purging (in hours)
    pub identity_retention_hours: u32,
    /// How long to keep expired invitations (in hours)
    pub invitation_retention_hours: u32,
    /// Whether to send notifications before expiry
    pub send_expiry_warnings: bool,
    /// How many minutes before expiry to send warning
    pub warning_minutes: u32,
}

impl Default for CleanupWorkerConfig {
    fn default() -> Self {
        Self {
            interval: Duration::from_secs(60), // Every minute
            batch_size: 100,
            identity_retention_hours: 24 * 30,  // 30 days
            invitation_retention_hours: 24 * 7, // 7 days
            send_expiry_warnings: true,
            warning_minutes: 15,
        }
    }
}

/// Statistics from a cleanup run
#[derive(Debug, Clone, Default)]
pub struct CleanupStats {
    /// Number of identities expired
    pub identities_expired: u32,
    /// Number of identities purged (permanently deleted)
    pub identities_purged: u32,
    /// Number of invitations expired
    pub invitations_expired: u32,
    /// Number of invitations purged
    pub invitations_purged: u32,
    /// Number of tokens added to revocation list
    pub tokens_revoked: u32,
    /// Number of expiry warnings sent
    pub warnings_sent: u32,
    /// Duration of the cleanup run
    pub duration_ms: u64,
    /// Any errors encountered
    pub errors: Vec<String>,
}

/// Cleanup worker for processing expired identities and invitations
pub struct CleanupWorker {
    config: CleanupWorkerConfig,
    running: Arc<RwLock<bool>>,
    last_stats: Arc<RwLock<Option<CleanupStats>>>,
}

impl CleanupWorker {
    /// Create a new cleanup worker with default configuration
    pub fn new() -> Self {
        Self::with_config(CleanupWorkerConfig::default())
    }

    /// Create a new cleanup worker with custom configuration
    pub fn with_config(config: CleanupWorkerConfig) -> Self {
        Self {
            config,
            running: Arc::new(RwLock::new(false)),
            last_stats: Arc::new(RwLock::new(None)),
        }
    }

    /// Start the cleanup worker
    pub async fn start(&self) -> CleanupWorkerHandle {
        let running = Arc::clone(&self.running);
        let last_stats = Arc::clone(&self.last_stats);
        let config = self.config.clone();

        // Mark as running
        {
            let mut is_running = running.write().await;
            *is_running = true;
        }

        let running_clone = Arc::clone(&running);
        let handle = tokio::spawn(async move {
            info!(
                "Identity cleanup worker started with interval {:?}",
                config.interval
            );

            loop {
                // Check if we should stop
                {
                    let is_running = running_clone.read().await;
                    if !*is_running {
                        info!("Identity cleanup worker stopping");
                        break;
                    }
                }

                // Run cleanup processing
                let stats = Self::process_cleanup(&config).await;

                // Store stats
                {
                    let mut last = last_stats.write().await;
                    *last = Some(stats.clone());
                }

                if stats.errors.is_empty() {
                    if stats.identities_expired > 0 || stats.invitations_expired > 0 {
                        info!(
                            "Cleanup completed: {} identities expired, {} invitations expired",
                            stats.identities_expired, stats.invitations_expired
                        );
                    } else {
                        debug!("Cleanup completed: no items to process");
                    }
                } else {
                    warn!("Cleanup completed with {} errors", stats.errors.len());
                }

                // Wait for next interval
                tokio::time::sleep(config.interval).await;
            }
        });

        CleanupWorkerHandle {
            running,
            task: handle,
        }
    }

    /// Process all cleanup tasks
    async fn process_cleanup(config: &CleanupWorkerConfig) -> CleanupStats {
        let start = std::time::Instant::now();
        let mut stats = CleanupStats::default();

        // Send expiry warnings first
        if config.send_expiry_warnings {
            match Self::send_expiry_warnings(config.warning_minutes, config.batch_size).await {
                Ok(count) => stats.warnings_sent = count,
                Err(e) => stats.errors.push(format!("Warning send error: {}", e)),
            }
        }

        // Expire identities that have passed their expiry time
        match Self::expire_identities(config.batch_size).await {
            Ok(count) => stats.identities_expired = count,
            Err(e) => stats.errors.push(format!("Identity expiry error: {}", e)),
        }

        // Expire unused/old invitations
        match Self::expire_invitations(config.batch_size).await {
            Ok(count) => stats.invitations_expired = count,
            Err(e) => stats.errors.push(format!("Invitation expiry error: {}", e)),
        }

        // Revoke tokens for expired identities
        match Self::revoke_expired_tokens(config.batch_size).await {
            Ok(count) => stats.tokens_revoked = count,
            Err(e) => stats.errors.push(format!("Token revocation error: {}", e)),
        }

        // Purge old expired records
        if config.identity_retention_hours > 0 {
            match Self::purge_old_identities(config.identity_retention_hours).await {
                Ok(count) => stats.identities_purged = count,
                Err(e) => stats.errors.push(format!("Identity purge error: {}", e)),
            }
        }

        if config.invitation_retention_hours > 0 {
            match Self::purge_old_invitations(config.invitation_retention_hours).await {
                Ok(count) => stats.invitations_purged = count,
                Err(e) => stats.errors.push(format!("Invitation purge error: {}", e)),
            }
        }

        stats.duration_ms = start.elapsed().as_millis() as u64;
        stats
    }

    /// Send warnings to identities about to expire
    async fn send_expiry_warnings(warning_minutes: u32, batch_size: usize) -> Result<u32, String> {
        // Query identities expiring within warning_minutes that haven't been warned
        // Send notification (webhook, email, etc.)
        // Mark as warned to avoid duplicate notifications
        //
        // Example:
        // SELECT * FROM ephemeral_identities
        // WHERE status = 'active'
        //   AND expires_at BETWEEN NOW() AND NOW() + INTERVAL 'warning_minutes minutes'
        //   AND NOT expiry_warning_sent
        // LIMIT batch_size

        debug!(
            "Sending expiry warnings (within {} minutes, batch: {})",
            warning_minutes, batch_size
        );

        // Placeholder
        Ok(0)
    }

    /// Expire identities that have passed their expiry time
    async fn expire_identities(batch_size: usize) -> Result<u32, String> {
        // UPDATE ephemeral_identities
        // SET status = 'expired', updated_at = NOW()
        // WHERE status = 'active' AND expires_at < NOW()
        // LIMIT batch_size

        debug!(
            "Processing identity expirations (batch_size: {})",
            batch_size
        );

        // Placeholder
        Ok(0)
    }

    /// Expire invitations that are past their expiry or have reached max uses
    async fn expire_invitations(batch_size: usize) -> Result<u32, String> {
        // UPDATE invitation_links
        // SET status = 'expired', updated_at = NOW()
        // WHERE status = 'pending' AND (
        //     expires_at < NOW()
        //     OR (max_uses IS NOT NULL AND current_uses >= max_uses)
        // )
        // LIMIT batch_size

        debug!(
            "Processing invitation expirations (batch_size: {})",
            batch_size
        );

        // Placeholder
        Ok(0)
    }

    /// Add tokens for expired identities to the revocation list
    async fn revoke_expired_tokens(batch_size: usize) -> Result<u32, String> {
        // For each newly expired identity, add their active tokens to revocation list
        // This ensures tokens can't be used even if they haven't expired yet

        debug!("Processing token revocations (batch_size: {})", batch_size);

        // Placeholder
        Ok(0)
    }

    /// Purge identities that have been expired for longer than retention period
    async fn purge_old_identities(retention_hours: u32) -> Result<u32, String> {
        // DELETE FROM ephemeral_identities
        // WHERE status = 'expired'
        //   AND updated_at < NOW() - INTERVAL 'retention_hours hours'
        //
        // Should also clean up related records (tokens, sessions, etc.)

        debug!("Purging identities older than {} hours", retention_hours);

        // Placeholder
        Ok(0)
    }

    /// Purge invitations that have been expired/used for longer than retention period
    async fn purge_old_invitations(retention_hours: u32) -> Result<u32, String> {
        // DELETE FROM invitation_links
        // WHERE status IN ('expired', 'redeemed', 'revoked')
        //   AND updated_at < NOW() - INTERVAL 'retention_hours hours'

        debug!("Purging invitations older than {} hours", retention_hours);

        // Placeholder
        Ok(0)
    }

    /// Get the last cleanup run statistics
    pub async fn last_stats(&self) -> Option<CleanupStats> {
        let stats = self.last_stats.read().await;
        stats.clone()
    }

    /// Check if the worker is currently running
    pub async fn is_running(&self) -> bool {
        let running = self.running.read().await;
        *running
    }
}

impl Default for CleanupWorker {
    fn default() -> Self {
        Self::new()
    }
}

/// Handle for controlling the cleanup worker
pub struct CleanupWorkerHandle {
    running: Arc<RwLock<bool>>,
    task: tokio::task::JoinHandle<()>,
}

impl CleanupWorkerHandle {
    /// Stop the cleanup worker gracefully
    pub async fn stop(self) {
        {
            let mut running = self.running.write().await;
            *running = false;
        }

        // Wait for the task to complete
        if let Err(e) = self.task.await {
            error!("Error waiting for cleanup worker to stop: {}", e);
        }

        info!("Identity cleanup worker stopped");
    }

    /// Check if the worker is still running
    pub async fn is_running(&self) -> bool {
        let running = self.running.read().await;
        *running
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_worker_lifecycle() {
        let worker = CleanupWorker::with_config(CleanupWorkerConfig {
            interval: Duration::from_millis(100),
            ..Default::default()
        });

        assert!(!worker.is_running().await);

        let handle = worker.start().await;

        // Give it time to run
        tokio::time::sleep(Duration::from_millis(150)).await;

        assert!(handle.is_running().await);

        // Check stats were recorded
        let stats = worker.last_stats().await;
        assert!(stats.is_some());

        // Stop
        handle.stop().await;
    }

    #[tokio::test]
    async fn test_process_cleanup() {
        let config = CleanupWorkerConfig::default();
        let stats = CleanupWorker::process_cleanup(&config).await;

        // Should complete without errors
        assert!(stats.errors.is_empty());
        assert!(stats.duration_ms > 0);
    }

    #[test]
    fn test_default_config() {
        let config = CleanupWorkerConfig::default();

        assert_eq!(config.interval, Duration::from_secs(60));
        assert_eq!(config.batch_size, 100);
        assert_eq!(config.identity_retention_hours, 24 * 30);
        assert!(config.send_expiry_warnings);
    }
}
