//! Migration coordinator for zero-downtime subnet migrations
//!
//! Coordinates node migrations between subnets using dual-stack IP addressing
//! and gradual peer updates, integrating with nebula-traverse for connectivity
//! verification.

use super::subnet_aware::{SubnetAwareWireGuard, SubnetPeer, SubnetWireGuardError};
use crate::events::SubnetEventPublisher;
use crate::migration::{Migration, MigrationStep};
use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::Ipv4Addr;
use std::sync::Arc;
use thiserror::Error;
use tracing::{debug, info, instrument, warn};
use uuid::Uuid;

#[cfg(feature = "quality-probing")]
use std::net::SocketAddr;

/// Errors for migration coordination
#[derive(Debug, Error)]
pub enum MigrationCoordError {
    #[error("Migration not found: {0}")]
    NotFound(Uuid),

    #[error("Migration already in progress for node: {0}")]
    AlreadyInProgress(Uuid),

    #[error("Invalid migration state transition: {0} -> {1}")]
    InvalidTransition(String, String),

    #[error("Connectivity probe failed: {0}")]
    ProbeFailed(String),

    #[error("Peer update failed: {0}")]
    PeerUpdateFailed(String),

    #[error("Timeout waiting for migration step")]
    Timeout,

    #[error("Rollback failed: {0}")]
    RollbackFailed(String),

    #[error("WireGuard error: {0}")]
    WireGuardError(#[from] SubnetWireGuardError),
}

/// Configuration for migration probing
#[derive(Debug, Clone)]
pub struct ProbeConfig {
    /// Timeout for individual probe
    pub probe_timeout_ms: u64,
    /// Number of probes to send
    pub probe_count: u32,
    /// Required success rate (0.0 - 1.0)
    pub required_success_rate: f64,
    /// Interval between probes
    pub probe_interval_ms: u64,
}

impl Default for ProbeConfig {
    fn default() -> Self {
        Self {
            probe_timeout_ms: 1000,
            probe_count: 5,
            required_success_rate: 0.8,
            probe_interval_ms: 200,
        }
    }
}

/// Result of connectivity probing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbeResult {
    pub success: bool,
    pub probes_sent: u32,
    pub probes_received: u32,
    pub avg_latency_ms: Option<f64>,
    pub min_latency_ms: Option<f64>,
    pub max_latency_ms: Option<f64>,
}

/// State of an active migration
#[derive(Debug)]
struct ActiveMigration {
    migration: Migration,
    step: MigrationStep,
    started_at: DateTime<Utc>,
    dual_stack_enabled: bool,
    peers_notified: usize,
    peers_acknowledged: usize,
    probe_results: Vec<ProbeResult>,
}

/// Coordinator for zero-downtime subnet migrations
pub struct MigrationCoordinator {
    /// Active migrations by migration ID
    active_migrations: RwLock<HashMap<Uuid, ActiveMigration>>,
    /// Migrations by node ID (for lookup)
    migrations_by_node: RwLock<HashMap<Uuid, Uuid>>,
    /// Subnet-aware WireGuard layer
    wireguard: Arc<SubnetAwareWireGuard>,
    /// Event publisher
    event_publisher: Arc<SubnetEventPublisher>,
    /// Probe configuration
    probe_config: ProbeConfig,
    /// Maximum concurrent migrations
    max_concurrent: usize,
}

impl MigrationCoordinator {
    /// Create a new migration coordinator
    pub fn new(
        wireguard: Arc<SubnetAwareWireGuard>,
        event_publisher: Arc<SubnetEventPublisher>,
    ) -> Self {
        Self {
            active_migrations: RwLock::new(HashMap::new()),
            migrations_by_node: RwLock::new(HashMap::new()),
            wireguard,
            event_publisher,
            probe_config: ProbeConfig::default(),
            max_concurrent: 10,
        }
    }

    /// Set probe configuration
    pub fn with_probe_config(mut self, config: ProbeConfig) -> Self {
        self.probe_config = config;
        self
    }

    /// Set maximum concurrent migrations
    pub fn with_max_concurrent(mut self, max: usize) -> Self {
        self.max_concurrent = max;
        self
    }

    /// Start a migration
    #[instrument(skip(self, migration), fields(
        migration_id = %migration.id,
        node_id = %migration.node_id,
        source = %migration.source_subnet_id,
        target = %migration.target_subnet_id
    ))]
    pub async fn start_migration(
        &self,
        migration: Migration,
    ) -> Result<(), MigrationCoordError> {
        // Check if migration already in progress for this node
        if self.migrations_by_node.read().contains_key(&migration.node_id) {
            return Err(MigrationCoordError::AlreadyInProgress(migration.node_id));
        }

        // Check concurrent migration limit
        if self.active_migrations.read().len() >= self.max_concurrent {
            warn!("Maximum concurrent migrations reached");
            // Could queue instead of error
        }

        info!("Starting subnet migration");

        let active = ActiveMigration {
            migration: migration.clone(),
            step: MigrationStep::NotStarted,
            started_at: Utc::now(),
            dual_stack_enabled: false,
            peers_notified: 0,
            peers_acknowledged: 0,
            probe_results: Vec::new(),
        };

        // Register migration
        {
            let mut migrations = self.active_migrations.write();
            let mut by_node = self.migrations_by_node.write();
            migrations.insert(migration.id, active);
            by_node.insert(migration.node_id, migration.id);
        }

        // Publish start event
        let _ = self.event_publisher.migration_started(&migration).await;

        Ok(())
    }

    /// Enable dual-stack for a migration (step 2)
    #[instrument(skip(self), fields(migration_id = %migration_id))]
    pub async fn enable_dual_stack(
        &self,
        migration_id: Uuid,
    ) -> Result<(), MigrationCoordError> {
        let migration = {
            let migrations = self.active_migrations.read();
            migrations
                .get(&migration_id)
                .ok_or(MigrationCoordError::NotFound(migration_id))?
                .migration
                .clone()
        };

        info!("Enabling dual-stack for migration");

        // Get the target IP (should be set during planning)
        let target_ip = migration.target_ip.ok_or_else(|| {
            MigrationCoordError::PeerUpdateFailed("No target IP allocated".to_string())
        })?;

        // Add peer to target subnet with new IP
        let new_peer = SubnetPeer::new(
            migration.node_id,
            &self.get_node_public_key(migration.node_id).await?,
            target_ip,
            migration.target_subnet_id,
        );

        self.wireguard.add_peer(new_peer).await?;

        // Update state
        {
            let mut migrations = self.active_migrations.write();
            if let Some(active) = migrations.get_mut(&migration_id) {
                active.dual_stack_enabled = true;
                active.step = MigrationStep::PropagatingToPeers;
            }
        }

        debug!("Dual-stack enabled, node responding on both IPs");
        Ok(())
    }

    /// Notify all peers of the new address (step 3)
    #[instrument(skip(self), fields(migration_id = %migration_id))]
    pub async fn notify_peers(
        &self,
        migration_id: Uuid,
    ) -> Result<usize, MigrationCoordError> {
        let migration = {
            let migrations = self.active_migrations.read();
            migrations
                .get(&migration_id)
                .ok_or(MigrationCoordError::NotFound(migration_id))?
                .migration
                .clone()
        };

        info!("Notifying peers of address change");

        // Get all peers that need to be notified
        // In production, this would send actual WireGuard config updates
        let source_peers = self.wireguard.get_subnet_peers(migration.source_subnet_id);
        let target_peers = self.wireguard.get_subnet_peers(migration.target_subnet_id);

        let total_peers = source_peers.len() + target_peers.len();

        // Update state
        {
            let mut migrations = self.active_migrations.write();
            if let Some(active) = migrations.get_mut(&migration_id) {
                active.peers_notified = total_peers;
                active.step = MigrationStep::VerifyingConnectivity;
            }
        }

        debug!(peers_notified = total_peers, "Peer notification complete");
        Ok(total_peers)
    }

    /// Verify connectivity through new address (step 4)
    #[instrument(skip(self), fields(migration_id = %migration_id))]
    pub async fn verify_connectivity(
        &self,
        migration_id: Uuid,
    ) -> Result<ProbeResult, MigrationCoordError> {
        let migration = {
            let migrations = self.active_migrations.read();
            migrations
                .get(&migration_id)
                .ok_or(MigrationCoordError::NotFound(migration_id))?
                .migration
                .clone()
        };

        info!("Verifying connectivity through new address");

        // Get target IP
        let target_ip = migration.target_ip.ok_or_else(|| {
            MigrationCoordError::ProbeFailed("No target IP allocated".to_string())
        })?;

        // Simulate probing (in production, would use actual connectivity tests)
        let result = self.probe_connectivity(target_ip).await?;

        // Update state
        {
            let mut migrations = self.active_migrations.write();
            if let Some(active) = migrations.get_mut(&migration_id) {
                active.probe_results.push(result.clone());
                if result.success {
                    active.step = MigrationStep::CuttingOver;
                }
            }
        }

        if result.success {
            debug!(
                avg_latency = ?result.avg_latency_ms,
                "Connectivity verified successfully"
            );
        } else {
            warn!("Connectivity verification failed");
        }

        Ok(result)
    }

    /// Complete the migration (step 5 & 6)
    #[instrument(skip(self), fields(migration_id = %migration_id))]
    pub async fn complete_migration(
        &self,
        migration_id: Uuid,
    ) -> Result<(), MigrationCoordError> {
        let active = {
            let mut migrations = self.active_migrations.write();
            let mut by_node = self.migrations_by_node.write();

            let active = migrations
                .remove(&migration_id)
                .ok_or(MigrationCoordError::NotFound(migration_id))?;

            by_node.remove(&active.migration.node_id);
            active
        };

        let migration = active.migration;
        let duration_ms = (Utc::now() - active.started_at).num_milliseconds() as u64;

        info!(duration_ms, "Completing migration");

        // Remove from source subnet
        self.wireguard.remove_peer(
            migration.source_subnet_id,
            &self.get_node_public_key(migration.node_id).await?,
        )?;

        // Publish completion event
        let _ = self
            .event_publisher
            .migration_completed(&migration, duration_ms)
            .await;

        info!("Migration completed successfully");
        Ok(())
    }

    /// Rollback a migration
    #[instrument(skip(self), fields(migration_id = %migration_id))]
    pub async fn rollback_migration(
        &self,
        migration_id: Uuid,
        reason: &str,
    ) -> Result<(), MigrationCoordError> {
        let active = {
            let mut migrations = self.active_migrations.write();
            let mut by_node = self.migrations_by_node.write();

            let active = migrations
                .remove(&migration_id)
                .ok_or(MigrationCoordError::NotFound(migration_id))?;

            by_node.remove(&active.migration.node_id);
            active
        };

        let migration = active.migration;

        warn!(reason, "Rolling back migration");

        // Remove from target subnet if dual-stack was enabled
        if active.dual_stack_enabled {
            let _ = self.wireguard.remove_peer(
                migration.target_subnet_id,
                &self.get_node_public_key(migration.node_id).await?,
            );
        }

        // Publish failure event
        let _ = self
            .event_publisher
            .migration_failed(&migration, reason, true)
            .await;

        info!("Migration rolled back");
        Ok(())
    }

    /// Get migration status
    pub fn get_status(&self, migration_id: Uuid) -> Option<MigrationCoordStatus> {
        self.active_migrations.read().get(&migration_id).map(|m| {
            MigrationCoordStatus {
                migration_id: m.migration.id,
                node_id: m.migration.node_id,
                source_subnet_id: m.migration.source_subnet_id,
                target_subnet_id: m.migration.target_subnet_id,
                current_step: m.step,
                started_at: m.started_at,
                dual_stack_enabled: m.dual_stack_enabled,
                peers_notified: m.peers_notified,
                peers_acknowledged: m.peers_acknowledged,
                probe_count: m.probe_results.len(),
                last_probe_success: m.probe_results.last().map(|p| p.success),
            }
        })
    }

    /// Get all active migrations
    pub fn get_active_migrations(&self) -> Vec<MigrationCoordStatus> {
        self.active_migrations
            .read()
            .values()
            .map(|m| MigrationCoordStatus {
                migration_id: m.migration.id,
                node_id: m.migration.node_id,
                source_subnet_id: m.migration.source_subnet_id,
                target_subnet_id: m.migration.target_subnet_id,
                current_step: m.step,
                started_at: m.started_at,
                dual_stack_enabled: m.dual_stack_enabled,
                peers_notified: m.peers_notified,
                peers_acknowledged: m.peers_acknowledged,
                probe_count: m.probe_results.len(),
                last_probe_success: m.probe_results.last().map(|p| p.success),
            })
            .collect()
    }

    /// Check if a node has an active migration
    pub fn has_active_migration(&self, node_id: Uuid) -> bool {
        self.migrations_by_node.read().contains_key(&node_id)
    }

    // ========================================================================
    // Internal helpers
    // ========================================================================

    /// Get a node's WireGuard public key (would be from registry in production)
    async fn get_node_public_key(&self, node_id: Uuid) -> Result<String, MigrationCoordError> {
        // In production, would look up from node registry or assignment
        // For now, return a placeholder
        Ok(format!("key-{}", node_id))
    }

    /// Probe connectivity to an IP using quality tracking
    #[cfg(feature = "quality-probing")]
    async fn probe_connectivity(&self, ip: Ipv4Addr) -> Result<ProbeResult, MigrationCoordError> {
        use super::quality::ConnectionQuality;

        let config = &self.probe_config;
        let target_addr = SocketAddr::new(ip.into(), 51820); // WireGuard default port

        // Create a quality tracker for this probe session
        let mut quality = ConnectionQuality::new();

        let mut successful_probes = 0u32;
        let mut latencies: Vec<f64> = Vec::new();

        debug!(
            target = %ip,
            probe_count = config.probe_count,
            "Starting connectivity probe with quality tracking"
        );

        // Perform probes with configured interval
        for i in 0..config.probe_count {
            if i > 0 {
                tokio::time::sleep(tokio::time::Duration::from_millis(
                    config.probe_interval_ms,
                ))
                .await;
            }

            let start = std::time::Instant::now();

            // Send probe packet (UDP to WireGuard port)
            match tokio::time::timeout(
                tokio::time::Duration::from_millis(config.probe_timeout_ms),
                self.send_probe_packet(target_addr),
            )
            .await
            {
                Ok(Ok(())) => {
                    let rtt = start.elapsed();
                    let rtt_ms = rtt.as_secs_f64() * 1000.0;

                    // Record in quality tracker
                    let seq = quality.record_send();
                    quality.record_ack(seq, rtt);

                    successful_probes += 1;
                    latencies.push(rtt_ms);
                    debug!(probe = i, rtt_ms = rtt_ms, "Probe succeeded");
                }
                Ok(Err(e)) => {
                    // Record send without ack for packet loss tracking
                    let _ = quality.record_send();
                    debug!(probe = i, error = %e, "Probe failed");
                }
                Err(_) => {
                    // Record send without ack for packet loss tracking
                    let _ = quality.record_send();
                    debug!(probe = i, "Probe timed out");
                }
            }
        }

        // Calculate statistics
        let success_rate = successful_probes as f64 / config.probe_count as f64;

        let (avg, min, max) = if !latencies.is_empty() {
            let sum: f64 = latencies.iter().sum();
            (
                Some(sum / latencies.len() as f64),
                Some(latencies.iter().cloned().fold(f64::INFINITY, f64::min)),
                Some(latencies.iter().cloned().fold(f64::NEG_INFINITY, f64::max)),
            )
        } else {
            (None, None, None)
        };

        // Check quality metrics from nebula-traverse
        let metrics = quality.metrics();
        let quality_acceptable = quality.is_acceptable(); // RTT < 300ms, jitter < 50ms, loss < 10%

        debug!(
            success_rate,
            rtt_ms = ?metrics.rtt.as_millis(),
            jitter_ms = ?metrics.jitter.as_millis(),
            packet_loss = metrics.packet_loss,
            quality_acceptable,
            "Probe complete"
        );

        Ok(ProbeResult {
            success: success_rate >= config.required_success_rate && quality_acceptable,
            probes_sent: config.probe_count,
            probes_received: successful_probes,
            avg_latency_ms: avg,
            min_latency_ms: min,
            max_latency_ms: max,
        })
    }

    /// Probe connectivity to an IP (stub implementation for development/testing)
    #[cfg(not(feature = "quality-probing"))]
    async fn probe_connectivity(&self, _ip: Ipv4Addr) -> Result<ProbeResult, MigrationCoordError> {
        // Simulate probing - use real network I/O when quality-probing feature enabled
        let config = &self.probe_config;
        let mut successes = 0u32;
        let mut latencies = Vec::new();

        for _ in 0..config.probe_count {
            // Simulate probe with random success/latency
            let success = rand::random::<f64>() > 0.1; // 90% success rate
            if success {
                successes += 1;
                latencies.push(rand::random::<f64>() * 50.0 + 5.0); // 5-55ms
            }
            tokio::time::sleep(tokio::time::Duration::from_millis(
                config.probe_interval_ms,
            ))
            .await;
        }

        let success_rate = successes as f64 / config.probe_count as f64;
        let success = success_rate >= config.required_success_rate;

        let (avg, min, max) = if !latencies.is_empty() {
            let sum: f64 = latencies.iter().sum();
            let avg = sum / latencies.len() as f64;
            let min = latencies.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = latencies.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            (Some(avg), Some(min), Some(max))
        } else {
            (None, None, None)
        };

        Ok(ProbeResult {
            success,
            probes_sent: config.probe_count,
            probes_received: successes,
            avg_latency_ms: avg,
            min_latency_ms: min,
            max_latency_ms: max,
        })
    }

    /// Send a probe packet to the target address
    #[cfg(feature = "quality-probing")]
    async fn send_probe_packet(&self, target: SocketAddr) -> Result<(), MigrationCoordError> {
        use tokio::net::UdpSocket;

        let socket = UdpSocket::bind("0.0.0.0:0")
            .await
            .map_err(|e| MigrationCoordError::ProbeFailed(format!("Socket bind failed: {}", e)))?;

        // Send WireGuard keepalive-style probe (minimal packet)
        let probe_data = [0u8; 32];
        socket
            .send_to(&probe_data, target)
            .await
            .map_err(|e| MigrationCoordError::ProbeFailed(format!("Send failed: {}", e)))?;

        // Wait for response (with short timeout handled by caller)
        let mut buf = [0u8; 128];
        socket
            .recv_from(&mut buf)
            .await
            .map_err(|e| MigrationCoordError::ProbeFailed(format!("Recv failed: {}", e)))?;

        Ok(())
    }
}

/// Status of a migration in the coordinator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationCoordStatus {
    pub migration_id: Uuid,
    pub node_id: Uuid,
    pub source_subnet_id: Uuid,
    pub target_subnet_id: Uuid,
    pub current_step: MigrationStep,
    pub started_at: DateTime<Utc>,
    pub dual_stack_enabled: bool,
    pub peers_notified: usize,
    pub peers_acknowledged: usize,
    pub probe_count: usize,
    pub last_probe_success: Option<bool>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::InMemoryTransport;
    use crate::migration::MigrationReason;

    fn create_test_coordinator() -> MigrationCoordinator {
        let wireguard = Arc::new(SubnetAwareWireGuard::new());
        let transport = Arc::new(InMemoryTransport::new());
        let publisher = Arc::new(SubnetEventPublisher::with_transport(transport));

        MigrationCoordinator::new(wireguard, publisher)
    }

    fn create_test_migration() -> Migration {
        Migration::new(
            Uuid::new_v4(),
            Uuid::new_v4(),
            Uuid::new_v4(),
            Ipv4Addr::new(10, 100, 0, 5),
            MigrationReason::PolicyChange,
        ).with_target_ip(Ipv4Addr::new(10, 101, 0, 5))
    }

    #[tokio::test]
    async fn test_start_migration() {
        let coordinator = create_test_coordinator();
        let migration = create_test_migration();

        coordinator.start_migration(migration.clone()).await.unwrap();

        assert!(coordinator.has_active_migration(migration.node_id));

        let status = coordinator.get_status(migration.id).unwrap();
        assert_eq!(status.current_step, MigrationStep::NotStarted);
    }

    #[tokio::test]
    async fn test_duplicate_migration_rejected() {
        let coordinator = create_test_coordinator();
        let migration = create_test_migration();

        coordinator.start_migration(migration.clone()).await.unwrap();

        // Second migration for same node should fail
        let result = coordinator.start_migration(migration).await;
        assert!(matches!(result, Err(MigrationCoordError::AlreadyInProgress(_))));
    }

    #[tokio::test]
    async fn test_probe_config() {
        let config = ProbeConfig {
            probe_count: 3,
            probe_timeout_ms: 500,
            required_success_rate: 0.5,
            probe_interval_ms: 100,
        };

        let wireguard = Arc::new(SubnetAwareWireGuard::new());
        let transport = Arc::new(InMemoryTransport::new());
        let publisher = Arc::new(SubnetEventPublisher::with_transport(transport));

        let coordinator = MigrationCoordinator::new(wireguard, publisher).with_probe_config(config);

        assert_eq!(coordinator.probe_config.probe_count, 3);
    }

    #[tokio::test]
    async fn test_rollback() {
        let coordinator = create_test_coordinator();
        let migration = create_test_migration();

        coordinator.start_migration(migration.clone()).await.unwrap();

        coordinator
            .rollback_migration(migration.id, "test failure")
            .await
            .unwrap();

        assert!(!coordinator.has_active_migration(migration.node_id));
    }
}
