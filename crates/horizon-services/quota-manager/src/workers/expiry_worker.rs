//! Expiry Worker
//!
//! Background worker that processes expired ephemeral quotas, pool allocations,
//! and resource pools. Runs periodically to ensure timely cleanup.

use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

/// Configuration for the expiry worker
#[derive(Debug, Clone)]
pub struct ExpiryWorkerConfig {
    /// Interval between expiry checks
    pub interval: Duration,
    /// Batch size for processing expired items
    pub batch_size: usize,
    /// How long to retain expired records before purging (in hours)
    pub retention_hours: u32,
    /// Whether to send notifications on expiry
    pub send_notifications: bool,
}

impl Default for ExpiryWorkerConfig {
    fn default() -> Self {
        Self {
            interval: Duration::from_secs(60), // Every minute
            batch_size: 100,
            retention_hours: 24 * 7, // 1 week
            send_notifications: true,
        }
    }
}

/// Statistics from an expiry run
#[derive(Debug, Clone, Default)]
pub struct ExpiryStats {
    /// Number of ephemeral quotas expired
    pub quotas_expired: u32,
    /// Number of pool allocations expired
    pub allocations_expired: u32,
    /// Number of resource pools expired
    pub pools_expired: u32,
    /// Number of old records purged
    pub records_purged: u32,
    /// Duration of the expiry run
    pub duration_ms: u64,
    /// Any errors encountered
    pub errors: Vec<String>,
}

/// Expiry worker for processing expired resources
pub struct ExpiryWorker {
    config: ExpiryWorkerConfig,
    running: Arc<RwLock<bool>>,
    last_stats: Arc<RwLock<Option<ExpiryStats>>>,
}

impl ExpiryWorker {
    /// Create a new expiry worker with default configuration
    pub fn new() -> Self {
        Self::with_config(ExpiryWorkerConfig::default())
    }

    /// Create a new expiry worker with custom configuration
    pub fn with_config(config: ExpiryWorkerConfig) -> Self {
        Self {
            config,
            running: Arc::new(RwLock::new(false)),
            last_stats: Arc::new(RwLock::new(None)),
        }
    }

    /// Start the expiry worker
    ///
    /// This spawns a background task that runs periodically.
    /// Returns a handle that can be used to stop the worker.
    pub async fn start(&self) -> ExpiryWorkerHandle {
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
            info!("Expiry worker started with interval {:?}", config.interval);

            loop {
                // Check if we should stop
                {
                    let is_running = running_clone.read().await;
                    if !*is_running {
                        info!("Expiry worker stopping");
                        break;
                    }
                }

                // Run expiry processing
                let stats = Self::process_expirations(&config).await;

                // Store stats
                {
                    let mut last = last_stats.write().await;
                    *last = Some(stats.clone());
                }

                if stats.errors.is_empty() {
                    debug!(
                        "Expiry run completed: {} quotas, {} allocations, {} pools expired",
                        stats.quotas_expired, stats.allocations_expired, stats.pools_expired
                    );
                } else {
                    warn!("Expiry run completed with {} errors", stats.errors.len());
                }

                // Wait for next interval
                tokio::time::sleep(config.interval).await;
            }
        });

        ExpiryWorkerHandle {
            running,
            task: handle,
        }
    }

    /// Process all expirations
    async fn process_expirations(config: &ExpiryWorkerConfig) -> ExpiryStats {
        let start = std::time::Instant::now();
        let mut stats = ExpiryStats::default();

        // Process ephemeral quota expirations
        match Self::expire_ephemeral_quotas(config.batch_size).await {
            Ok(count) => stats.quotas_expired = count,
            Err(e) => stats.errors.push(format!("Quota expiry error: {}", e)),
        }

        // Process pool allocation expirations
        match Self::expire_pool_allocations(config.batch_size).await {
            Ok(count) => stats.allocations_expired = count,
            Err(e) => stats.errors.push(format!("Allocation expiry error: {}", e)),
        }

        // Process resource pool expirations
        match Self::expire_resource_pools(config.batch_size).await {
            Ok(count) => stats.pools_expired = count,
            Err(e) => stats.errors.push(format!("Pool expiry error: {}", e)),
        }

        // Purge old expired records
        if config.retention_hours > 0 {
            match Self::purge_old_records(config.retention_hours).await {
                Ok(count) => stats.records_purged = count,
                Err(e) => stats.errors.push(format!("Purge error: {}", e)),
            }
        }

        stats.duration_ms = start.elapsed().as_millis() as u64;
        stats
    }

    /// Expire ephemeral quotas that have passed their expiry time
    async fn expire_ephemeral_quotas(batch_size: usize) -> Result<u32, String> {
        // In a real implementation, this would query the database
        // and update status to 'expired' for matching records
        //
        // Example SQL:
        // UPDATE ephemeral_quotas
        // SET status = 'expired', updated_at = NOW()
        // WHERE status = 'active' AND expires_at < NOW()
        // LIMIT batch_size
        // RETURNING id;

        debug!(
            "Processing ephemeral quota expirations (batch_size: {})",
            batch_size
        );

        // Placeholder - would connect to actual database
        Ok(0)
    }

    /// Expire pool allocations that have passed their expiry time
    async fn expire_pool_allocations(batch_size: usize) -> Result<u32, String> {
        // UPDATE pool_allocations
        // SET status = 'expired', updated_at = NOW()
        // WHERE status IN ('approved', 'active') AND expires_at < NOW()
        // LIMIT batch_size;
        //
        // Also need to update the parent pool's allocated/current_users counts

        debug!(
            "Processing pool allocation expirations (batch_size: {})",
            batch_size
        );

        // Placeholder
        Ok(0)
    }

    /// Expire resource pools that have passed their expiry time
    async fn expire_resource_pools(batch_size: usize) -> Result<u32, String> {
        // UPDATE resource_pools
        // SET status = 'expired', updated_at = NOW()
        // WHERE status = 'active' AND expires_at IS NOT NULL AND expires_at < NOW()
        // LIMIT batch_size;

        debug!(
            "Processing resource pool expirations (batch_size: {})",
            batch_size
        );

        // Placeholder
        Ok(0)
    }

    /// Purge old expired records beyond retention period
    async fn purge_old_records(retention_hours: u32) -> Result<u32, String> {
        // DELETE FROM ephemeral_quota_usage
        // WHERE timestamp < NOW() - INTERVAL 'retention_hours hours';
        //
        // Could also archive to cold storage instead of deleting

        debug!("Purging records older than {} hours", retention_hours);

        // Placeholder
        Ok(0)
    }

    /// Get the last expiry run statistics
    pub async fn last_stats(&self) -> Option<ExpiryStats> {
        let stats = self.last_stats.read().await;
        stats.clone()
    }

    /// Check if the worker is currently running
    pub async fn is_running(&self) -> bool {
        let running = self.running.read().await;
        *running
    }
}

impl Default for ExpiryWorker {
    fn default() -> Self {
        Self::new()
    }
}

/// Handle for controlling the expiry worker
pub struct ExpiryWorkerHandle {
    running: Arc<RwLock<bool>>,
    task: tokio::task::JoinHandle<()>,
}

impl ExpiryWorkerHandle {
    /// Stop the expiry worker gracefully
    pub async fn stop(self) {
        {
            let mut running = self.running.write().await;
            *running = false;
        }

        // Wait for the task to complete
        if let Err(e) = self.task.await {
            error!("Error waiting for expiry worker to stop: {}", e);
        }

        info!("Expiry worker stopped");
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
    async fn test_worker_start_stop() {
        let worker = ExpiryWorker::with_config(ExpiryWorkerConfig {
            interval: Duration::from_millis(100),
            ..Default::default()
        });

        let handle = worker.start().await;

        // Give it time to run a cycle
        tokio::time::sleep(Duration::from_millis(150)).await;

        assert!(handle.is_running().await);

        // Check stats were recorded
        let stats = worker.last_stats().await;
        assert!(stats.is_some());

        // Stop the worker
        handle.stop().await;
    }

    #[tokio::test]
    async fn test_process_expirations() {
        let config = ExpiryWorkerConfig::default();
        let stats = ExpiryWorker::process_expirations(&config).await;

        // Should complete without errors (using placeholder implementations)
        assert!(stats.errors.is_empty());
    }
}
