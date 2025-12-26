//! WireGuard configuration synchronization service
//!
//! Handles distribution of WireGuard configurations to nodes,
//! tracking sync status, and managing configuration updates.

use crate::models::{Subnet, SubnetAssignment};
use crate::{Error, Result};
use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

/// Sync status for a node's WireGuard configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SyncStatus {
    /// Configuration not yet synced
    Pending,
    /// Sync in progress
    Syncing,
    /// Successfully synced
    Synced,
    /// Sync failed
    Failed,
    /// Configuration outdated, needs resync
    Stale,
}

/// Sync event types for tracking configuration changes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SyncEvent {
    /// Node added to subnet
    NodeAdded {
        node_id: Uuid,
        subnet_id: Uuid,
        timestamp: DateTime<Utc>,
    },
    /// Node removed from subnet
    NodeRemoved {
        node_id: Uuid,
        subnet_id: Uuid,
        timestamp: DateTime<Utc>,
    },
    /// Peer configuration updated
    PeerUpdated {
        node_id: Uuid,
        peer_id: Uuid,
        timestamp: DateTime<Utc>,
    },
    /// Subnet configuration changed
    SubnetConfigChanged {
        subnet_id: Uuid,
        timestamp: DateTime<Utc>,
    },
    /// Cross-subnet route added
    RouteAdded {
        source_subnet_id: Uuid,
        destination_subnet_id: Uuid,
        timestamp: DateTime<Utc>,
    },
    /// Cross-subnet route removed
    RouteRemoved {
        source_subnet_id: Uuid,
        destination_subnet_id: Uuid,
        timestamp: DateTime<Utc>,
    },
    /// Full resync triggered
    FullResync {
        subnet_id: Uuid,
        timestamp: DateTime<Utc>,
    },
}

/// Configuration change tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigChange {
    /// Unique change ID
    pub id: Uuid,
    /// The event that triggered this change
    pub event: SyncEvent,
    /// Nodes affected by this change
    pub affected_nodes: Vec<Uuid>,
    /// Whether this change has been fully propagated
    pub propagated: bool,
    /// Number of retry attempts
    pub retry_count: u32,
    /// Last error message if failed
    pub last_error: Option<String>,
}

impl ConfigChange {
    /// Create a new configuration change
    pub fn new(event: SyncEvent, affected_nodes: Vec<Uuid>) -> Self {
        Self {
            id: Uuid::new_v4(),
            event,
            affected_nodes,
            propagated: false,
            retry_count: 0,
            last_error: None,
        }
    }

    /// Mark the change as propagated
    pub fn mark_propagated(&mut self) {
        self.propagated = true;
    }

    /// Record a failed sync attempt
    pub fn record_failure(&mut self, error: String) {
        self.retry_count += 1;
        self.last_error = Some(error);
    }
}

/// Node sync state tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeSyncState {
    /// Node ID
    pub node_id: Uuid,
    /// Current sync status
    pub status: SyncStatus,
    /// Configuration version (hash of config)
    pub config_version: Option<String>,
    /// Last successful sync timestamp
    pub last_sync: Option<DateTime<Utc>>,
    /// Last sync attempt timestamp
    pub last_attempt: Option<DateTime<Utc>>,
    /// Number of consecutive failures
    pub failure_count: u32,
    /// Last error message
    pub last_error: Option<String>,
    /// Node's endpoint for config delivery
    pub endpoint: Option<String>,
}

impl NodeSyncState {
    /// Create new sync state for a node
    pub fn new(node_id: Uuid) -> Self {
        Self {
            node_id,
            status: SyncStatus::Pending,
            config_version: None,
            last_sync: None,
            last_attempt: None,
            failure_count: 0,
            last_error: None,
            endpoint: None,
        }
    }

    /// Update state after successful sync
    pub fn mark_synced(&mut self, config_version: String) {
        let now = Utc::now();
        self.status = SyncStatus::Synced;
        self.config_version = Some(config_version);
        self.last_sync = Some(now);
        self.last_attempt = Some(now);
        self.failure_count = 0;
        self.last_error = None;
    }

    /// Update state after failed sync
    pub fn mark_failed(&mut self, error: String) {
        self.status = SyncStatus::Failed;
        self.last_attempt = Some(Utc::now());
        self.failure_count += 1;
        self.last_error = Some(error);
    }

    /// Mark configuration as stale (needs resync)
    pub fn mark_stale(&mut self) {
        if self.status == SyncStatus::Synced {
            self.status = SyncStatus::Stale;
        }
    }

    /// Check if sync should be retried
    pub fn should_retry(&self, max_retries: u32) -> bool {
        matches!(self.status, SyncStatus::Failed | SyncStatus::Stale)
            && self.failure_count < max_retries
    }
}

/// Configuration sync service for WireGuard
pub struct ConfigSyncService {
    /// Node sync states by node ID
    node_states: Arc<RwLock<HashMap<Uuid, NodeSyncState>>>,
    /// Pending configuration changes
    pending_changes: Arc<RwLock<Vec<ConfigChange>>>,
    /// Subnet configurations cache
    subnet_configs: Arc<RwLock<HashMap<Uuid, SubnetSyncConfig>>>,
    /// Maximum retry attempts
    max_retries: u32,
    /// Sync timeout in seconds
    sync_timeout_secs: u64,
}

/// Cached subnet sync configuration
#[derive(Debug, Clone)]
pub struct SubnetSyncConfig {
    /// Subnet information
    pub subnet: Subnet,
    /// Current configuration version
    pub config_version: String,
    /// Last update timestamp
    pub last_updated: DateTime<Utc>,
    /// Node assignments in this subnet
    pub assignments: Vec<SubnetAssignment>,
}

impl ConfigSyncService {
    /// Create a new configuration sync service
    pub fn new() -> Self {
        Self {
            node_states: Arc::new(RwLock::new(HashMap::new())),
            pending_changes: Arc::new(RwLock::new(Vec::new())),
            subnet_configs: Arc::new(RwLock::new(HashMap::new())),
            max_retries: 3,
            sync_timeout_secs: 30,
        }
    }

    /// Create with custom configuration
    pub fn with_config(max_retries: u32, sync_timeout_secs: u64) -> Self {
        Self {
            node_states: Arc::new(RwLock::new(HashMap::new())),
            pending_changes: Arc::new(RwLock::new(Vec::new())),
            subnet_configs: Arc::new(RwLock::new(HashMap::new())),
            max_retries,
            sync_timeout_secs,
        }
    }

    /// Register a node for sync tracking
    pub fn register_node(&self, node_id: Uuid, endpoint: Option<String>) {
        let mut states = self.node_states.write();
        let state = states.entry(node_id).or_insert_with(|| NodeSyncState::new(node_id));
        state.endpoint = endpoint;
    }

    /// Unregister a node from sync tracking
    pub fn unregister_node(&self, node_id: Uuid) {
        self.node_states.write().remove(&node_id);
    }

    /// Get sync status for a node
    pub fn get_node_status(&self, node_id: Uuid) -> Option<NodeSyncState> {
        self.node_states.read().get(&node_id).cloned()
    }

    /// Get all nodes with a specific sync status
    pub fn get_nodes_by_status(&self, status: SyncStatus) -> Vec<Uuid> {
        self.node_states
            .read()
            .iter()
            .filter(|(_, state)| state.status == status)
            .map(|(id, _)| *id)
            .collect()
    }

    /// Queue a configuration change for sync
    pub fn queue_change(&self, event: SyncEvent, affected_nodes: Vec<Uuid>) {
        let change = ConfigChange::new(event, affected_nodes.clone());

        // Mark affected nodes as stale
        let mut states = self.node_states.write();
        for node_id in &affected_nodes {
            if let Some(state) = states.get_mut(node_id) {
                state.mark_stale();
            }
        }

        self.pending_changes.write().push(change);
    }

    /// Process pending changes and return nodes needing sync
    pub fn process_pending_changes(&self) -> Vec<(Uuid, Vec<ConfigChange>)> {
        let mut changes = self.pending_changes.write();
        let states = self.node_states.read();

        // Group changes by node
        let mut node_changes: HashMap<Uuid, Vec<ConfigChange>> = HashMap::new();

        for change in changes.iter().filter(|c| !c.propagated) {
            for node_id in &change.affected_nodes {
                if let Some(state) = states.get(node_id) {
                    if state.should_retry(self.max_retries) || state.status == SyncStatus::Pending {
                        node_changes
                            .entry(*node_id)
                            .or_default()
                            .push(change.clone());
                    }
                }
            }
        }

        node_changes.into_iter().collect()
    }

    /// Mark a node's sync as complete
    pub fn mark_node_synced(&self, node_id: Uuid, config_version: String) {
        if let Some(state) = self.node_states.write().get_mut(&node_id) {
            state.mark_synced(config_version);
        }

        // Update change propagation status
        self.update_change_propagation();
    }

    /// Mark a node's sync as failed
    pub fn mark_node_failed(&self, node_id: Uuid, error: String) {
        if let Some(state) = self.node_states.write().get_mut(&node_id) {
            state.mark_failed(error);
        }
    }

    /// Update propagation status of changes
    fn update_change_propagation(&self) {
        let states = self.node_states.read();
        let mut changes = self.pending_changes.write();

        for change in changes.iter_mut() {
            if change.propagated {
                continue;
            }

            // Check if all affected nodes are synced
            let all_synced = change.affected_nodes.iter().all(|node_id| {
                states
                    .get(node_id)
                    .map(|s| s.status == SyncStatus::Synced)
                    .unwrap_or(true) // If node was removed, consider it done
            });

            if all_synced {
                change.mark_propagated();
            }
        }
    }

    /// Cleanup old propagated changes
    pub fn cleanup_propagated_changes(&self, max_age_secs: i64) {
        let cutoff = Utc::now() - chrono::Duration::seconds(max_age_secs);
        let mut changes = self.pending_changes.write();

        changes.retain(|change| {
            if !change.propagated {
                return true;
            }

            // Keep if change is recent
            match &change.event {
                SyncEvent::NodeAdded { timestamp, .. }
                | SyncEvent::NodeRemoved { timestamp, .. }
                | SyncEvent::PeerUpdated { timestamp, .. }
                | SyncEvent::SubnetConfigChanged { timestamp, .. }
                | SyncEvent::RouteAdded { timestamp, .. }
                | SyncEvent::RouteRemoved { timestamp, .. }
                | SyncEvent::FullResync { timestamp, .. } => *timestamp > cutoff,
            }
        });
    }

    /// Update cached subnet configuration
    pub fn update_subnet_config(
        &self,
        subnet: Subnet,
        assignments: Vec<SubnetAssignment>,
    ) -> String {
        let config_version = Self::compute_config_version(&subnet, &assignments);

        let config = SubnetSyncConfig {
            subnet: subnet.clone(),
            config_version: config_version.clone(),
            last_updated: Utc::now(),
            assignments,
        };

        self.subnet_configs.write().insert(subnet.id, config);

        // Queue sync event for all nodes in subnet
        let affected_nodes: Vec<Uuid> = self
            .subnet_configs
            .read()
            .get(&subnet.id)
            .map(|c| c.assignments.iter().map(|a| a.node_id).collect())
            .unwrap_or_default();

        if !affected_nodes.is_empty() {
            self.queue_change(
                SyncEvent::SubnetConfigChanged {
                    subnet_id: subnet.id,
                    timestamp: Utc::now(),
                },
                affected_nodes,
            );
        }

        config_version
    }

    /// Compute a version hash for the configuration
    fn compute_config_version(subnet: &Subnet, assignments: &[SubnetAssignment]) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        subnet.id.hash(&mut hasher);
        subnet.cidr.to_string().hash(&mut hasher);
        subnet.wg_public_key.hash(&mut hasher);

        for assignment in assignments {
            assignment.node_id.hash(&mut hasher);
            assignment.assigned_ip.hash(&mut hasher);
            assignment.wg_public_key.hash(&mut hasher);
        }

        format!("{:016x}", hasher.finish())
    }

    /// Get cached subnet configuration
    pub fn get_subnet_config(&self, subnet_id: Uuid) -> Option<SubnetSyncConfig> {
        self.subnet_configs.read().get(&subnet_id).cloned()
    }

    /// Trigger full resync for a subnet
    pub fn trigger_full_resync(&self, subnet_id: Uuid) -> Result<()> {
        let configs = self.subnet_configs.read();
        let config = configs
            .get(&subnet_id)
            .ok_or_else(|| Error::SubnetNotFound(subnet_id))?;

        let affected_nodes: Vec<Uuid> = config.assignments.iter().map(|a| a.node_id).collect();

        if affected_nodes.is_empty() {
            return Ok(());
        }

        drop(configs);

        // Mark all nodes as pending
        {
            let mut states = self.node_states.write();
            for node_id in &affected_nodes {
                if let Some(state) = states.get_mut(node_id) {
                    state.status = SyncStatus::Pending;
                    state.failure_count = 0;
                }
            }
        }

        self.queue_change(
            SyncEvent::FullResync {
                subnet_id,
                timestamp: Utc::now(),
            },
            affected_nodes,
        );

        Ok(())
    }

    /// Get sync statistics
    pub fn get_stats(&self) -> SyncStats {
        let states = self.node_states.read();
        let changes = self.pending_changes.read();

        let mut stats = SyncStats::default();
        stats.total_nodes = states.len();

        for state in states.values() {
            match state.status {
                SyncStatus::Pending => stats.pending_nodes += 1,
                SyncStatus::Syncing => stats.syncing_nodes += 1,
                SyncStatus::Synced => stats.synced_nodes += 1,
                SyncStatus::Failed => stats.failed_nodes += 1,
                SyncStatus::Stale => stats.stale_nodes += 1,
            }
        }

        stats.total_changes = changes.len();
        stats.pending_changes = changes.iter().filter(|c| !c.propagated).count();
        stats.propagated_changes = changes.iter().filter(|c| c.propagated).count();

        stats
    }

    /// Get nodes that need immediate sync attention
    pub fn get_nodes_needing_sync(&self) -> Vec<NodeSyncState> {
        self.node_states
            .read()
            .values()
            .filter(|state| {
                // Pending nodes always need sync
                state.status == SyncStatus::Pending
                    // Failed/Stale nodes need sync if retry is allowed
                    || (matches!(state.status, SyncStatus::Stale | SyncStatus::Failed)
                        && state.should_retry(self.max_retries))
            })
            .cloned()
            .collect()
    }

    /// Check if all nodes in a subnet are synced
    pub fn is_subnet_synced(&self, subnet_id: Uuid) -> bool {
        let configs = self.subnet_configs.read();
        let Some(config) = configs.get(&subnet_id) else {
            return false;
        };

        let states = self.node_states.read();
        config.assignments.iter().all(|assignment| {
            states
                .get(&assignment.node_id)
                .map(|s| s.status == SyncStatus::Synced)
                .unwrap_or(false)
        })
    }
}

impl Default for ConfigSyncService {
    fn default() -> Self {
        Self::new()
    }
}

/// Sync statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SyncStats {
    /// Total tracked nodes
    pub total_nodes: usize,
    /// Nodes pending sync
    pub pending_nodes: usize,
    /// Nodes currently syncing
    pub syncing_nodes: usize,
    /// Successfully synced nodes
    pub synced_nodes: usize,
    /// Nodes with failed sync
    pub failed_nodes: usize,
    /// Nodes with stale config
    pub stale_nodes: usize,
    /// Total configuration changes
    pub total_changes: usize,
    /// Pending changes
    pub pending_changes: usize,
    /// Propagated changes
    pub propagated_changes: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{SubnetPurpose, SubnetStatus};
    use std::net::Ipv4Addr;

    fn create_test_subnet() -> Subnet {
        Subnet {
            id: Uuid::new_v4(),
            name: "test-subnet".to_string(),
            description: None,
            cidr: "10.100.0.0/24".parse().unwrap(),
            purpose: SubnetPurpose::Tenant,
            status: SubnetStatus::Active,
            tenant_id: Some(Uuid::new_v4()),
            node_type: None,
            region: None,
            resource_pool_id: None,
            wg_interface: "wg-test".to_string(),
            wg_listen_port: 51820,
            wg_public_key: Some("test-pub-key".to_string()),
            wg_private_key: Some("test-priv-key".to_string()),
            max_nodes: Some(100),
            current_nodes: 0,
            template_id: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            created_by: None,
            metadata: None,
        }
    }

    fn create_test_assignment(subnet_id: Uuid) -> SubnetAssignment {
        SubnetAssignment {
            id: Uuid::new_v4(),
            node_id: Uuid::new_v4(),
            subnet_id,
            assigned_ip: Ipv4Addr::new(10, 100, 0, 10),
            wg_public_key: "node-pub-key".to_string(),
            assigned_at: Utc::now(),
            assignment_method: "manual".to_string(),
            policy_id: None,
            is_migration_temp: false,
        }
    }

    #[test]
    fn test_node_registration() {
        let service = ConfigSyncService::new();
        let node_id = Uuid::new_v4();

        service.register_node(node_id, Some("10.0.0.1:51820".to_string()));

        let status = service.get_node_status(node_id);
        assert!(status.is_some());

        let state = status.unwrap();
        assert_eq!(state.status, SyncStatus::Pending);
        assert_eq!(state.endpoint, Some("10.0.0.1:51820".to_string()));
    }

    #[test]
    fn test_node_unregistration() {
        let service = ConfigSyncService::new();
        let node_id = Uuid::new_v4();

        service.register_node(node_id, None);
        assert!(service.get_node_status(node_id).is_some());

        service.unregister_node(node_id);
        assert!(service.get_node_status(node_id).is_none());
    }

    #[test]
    fn test_sync_status_transitions() {
        let service = ConfigSyncService::new();
        let node_id = Uuid::new_v4();

        service.register_node(node_id, None);

        // Initially pending
        let state = service.get_node_status(node_id).unwrap();
        assert_eq!(state.status, SyncStatus::Pending);

        // Mark synced
        service.mark_node_synced(node_id, "v1".to_string());
        let state = service.get_node_status(node_id).unwrap();
        assert_eq!(state.status, SyncStatus::Synced);
        assert_eq!(state.config_version, Some("v1".to_string()));

        // Queue a change (should mark stale)
        service.queue_change(
            SyncEvent::PeerUpdated {
                node_id,
                peer_id: Uuid::new_v4(),
                timestamp: Utc::now(),
            },
            vec![node_id],
        );

        let state = service.get_node_status(node_id).unwrap();
        assert_eq!(state.status, SyncStatus::Stale);
    }

    #[test]
    fn test_mark_node_failed() {
        let service = ConfigSyncService::new();
        let node_id = Uuid::new_v4();

        service.register_node(node_id, None);
        service.mark_node_failed(node_id, "Connection refused".to_string());

        let state = service.get_node_status(node_id).unwrap();
        assert_eq!(state.status, SyncStatus::Failed);
        assert_eq!(state.failure_count, 1);
        assert_eq!(state.last_error, Some("Connection refused".to_string()));
    }

    #[test]
    fn test_retry_logic() {
        let service = ConfigSyncService::with_config(3, 30);
        let node_id = Uuid::new_v4();

        service.register_node(node_id, None);

        // Should retry after first failure
        service.mark_node_failed(node_id, "Error 1".to_string());
        let state = service.get_node_status(node_id).unwrap();
        assert!(state.should_retry(3));

        // Should retry after second failure
        service.mark_node_failed(node_id, "Error 2".to_string());
        let state = service.get_node_status(node_id).unwrap();
        assert!(state.should_retry(3));

        // Should not retry after third failure (max_retries=3)
        service.mark_node_failed(node_id, "Error 3".to_string());
        let state = service.get_node_status(node_id).unwrap();
        assert!(!state.should_retry(3));
    }

    #[test]
    fn test_queue_and_process_changes() {
        let service = ConfigSyncService::new();
        let node1 = Uuid::new_v4();
        let node2 = Uuid::new_v4();
        let subnet_id = Uuid::new_v4();

        service.register_node(node1, None);
        service.register_node(node2, None);

        service.queue_change(
            SyncEvent::SubnetConfigChanged {
                subnet_id,
                timestamp: Utc::now(),
            },
            vec![node1, node2],
        );

        let pending = service.process_pending_changes();
        assert_eq!(pending.len(), 2); // Both nodes need sync
    }

    #[test]
    fn test_change_propagation() {
        let service = ConfigSyncService::new();
        let node_id = Uuid::new_v4();

        service.register_node(node_id, None);

        service.queue_change(
            SyncEvent::NodeAdded {
                node_id,
                subnet_id: Uuid::new_v4(),
                timestamp: Utc::now(),
            },
            vec![node_id],
        );

        // Change should be pending
        let stats = service.get_stats();
        assert_eq!(stats.pending_changes, 1);

        // Mark node synced
        service.mark_node_synced(node_id, "v1".to_string());

        // Change should be propagated
        let stats = service.get_stats();
        assert_eq!(stats.propagated_changes, 1);
        assert_eq!(stats.pending_changes, 0);
    }

    #[test]
    fn test_subnet_config_update() {
        let service = ConfigSyncService::new();
        let subnet = create_test_subnet();
        let assignment = create_test_assignment(subnet.id);

        service.register_node(assignment.node_id, None);

        let version = service.update_subnet_config(subnet.clone(), vec![assignment.clone()]);
        assert!(!version.is_empty());

        let config = service.get_subnet_config(subnet.id);
        assert!(config.is_some());
        assert_eq!(config.unwrap().config_version, version);
    }

    #[test]
    fn test_get_nodes_by_status() {
        let service = ConfigSyncService::new();
        let node1 = Uuid::new_v4();
        let node2 = Uuid::new_v4();
        let node3 = Uuid::new_v4();

        service.register_node(node1, None);
        service.register_node(node2, None);
        service.register_node(node3, None);

        service.mark_node_synced(node1, "v1".to_string());
        service.mark_node_failed(node2, "Error".to_string());
        // node3 stays pending

        let synced = service.get_nodes_by_status(SyncStatus::Synced);
        assert_eq!(synced.len(), 1);
        assert!(synced.contains(&node1));

        let failed = service.get_nodes_by_status(SyncStatus::Failed);
        assert_eq!(failed.len(), 1);
        assert!(failed.contains(&node2));

        let pending = service.get_nodes_by_status(SyncStatus::Pending);
        assert_eq!(pending.len(), 1);
        assert!(pending.contains(&node3));
    }

    #[test]
    fn test_get_stats() {
        let service = ConfigSyncService::new();

        for i in 0..5 {
            let node_id = Uuid::new_v4();
            service.register_node(node_id, None);

            if i < 2 {
                service.mark_node_synced(node_id, format!("v{}", i));
            } else if i < 4 {
                service.mark_node_failed(node_id, "Error".to_string());
            }
            // Last one stays pending
        }

        let stats = service.get_stats();
        assert_eq!(stats.total_nodes, 5);
        assert_eq!(stats.synced_nodes, 2);
        assert_eq!(stats.failed_nodes, 2);
        assert_eq!(stats.pending_nodes, 1);
    }

    #[test]
    fn test_nodes_needing_sync() {
        let service = ConfigSyncService::new();
        let node1 = Uuid::new_v4();
        let node2 = Uuid::new_v4();
        let node3 = Uuid::new_v4();

        service.register_node(node1, None);
        service.register_node(node2, None);
        service.register_node(node3, None);

        service.mark_node_synced(node1, "v1".to_string());
        service.mark_node_failed(node2, "Error".to_string());
        // node3 stays pending

        let needing_sync = service.get_nodes_needing_sync();
        assert_eq!(needing_sync.len(), 2); // node2 (failed) and node3 (pending)

        let ids: Vec<Uuid> = needing_sync.iter().map(|s| s.node_id).collect();
        assert!(ids.contains(&node2));
        assert!(ids.contains(&node3));
        assert!(!ids.contains(&node1));
    }

    #[test]
    fn test_is_subnet_synced() {
        let service = ConfigSyncService::new();
        let subnet = create_test_subnet();
        let assignment1 = create_test_assignment(subnet.id);
        let assignment2 = create_test_assignment(subnet.id);

        service.register_node(assignment1.node_id, None);
        service.register_node(assignment2.node_id, None);

        service.update_subnet_config(
            subnet.clone(),
            vec![assignment1.clone(), assignment2.clone()],
        );

        // Initially not all synced
        assert!(!service.is_subnet_synced(subnet.id));

        // Sync one node
        service.mark_node_synced(assignment1.node_id, "v1".to_string());
        assert!(!service.is_subnet_synced(subnet.id));

        // Sync both nodes
        service.mark_node_synced(assignment2.node_id, "v1".to_string());
        assert!(service.is_subnet_synced(subnet.id));
    }

    #[test]
    fn test_full_resync() {
        let service = ConfigSyncService::new();
        let subnet = create_test_subnet();
        let assignment = create_test_assignment(subnet.id);

        service.register_node(assignment.node_id, None);
        service.mark_node_synced(assignment.node_id, "v1".to_string());

        service.update_subnet_config(subnet.clone(), vec![assignment.clone()]);

        // Verify initially synced
        let state = service.get_node_status(assignment.node_id).unwrap();
        assert_eq!(state.status, SyncStatus::Stale); // Stale due to update_subnet_config

        // Sync again
        service.mark_node_synced(assignment.node_id, "v2".to_string());

        // Trigger full resync
        service.trigger_full_resync(subnet.id).unwrap();

        // Node should be pending again
        let state = service.get_node_status(assignment.node_id).unwrap();
        assert_eq!(state.status, SyncStatus::Pending);
        assert_eq!(state.failure_count, 0);
    }

    #[test]
    fn test_cleanup_propagated_changes() {
        let service = ConfigSyncService::new();
        let node_id = Uuid::new_v4();

        service.register_node(node_id, None);

        // Queue a change
        service.queue_change(
            SyncEvent::NodeAdded {
                node_id,
                subnet_id: Uuid::new_v4(),
                timestamp: Utc::now(),
            },
            vec![node_id],
        );

        // Sync and propagate
        service.mark_node_synced(node_id, "v1".to_string());

        let stats = service.get_stats();
        assert_eq!(stats.propagated_changes, 1);

        // Cleanup with 0 max age (should remove all propagated)
        service.cleanup_propagated_changes(0);

        let stats = service.get_stats();
        assert_eq!(stats.total_changes, 0);
    }

    #[test]
    fn test_config_change_creation() {
        let event = SyncEvent::NodeAdded {
            node_id: Uuid::new_v4(),
            subnet_id: Uuid::new_v4(),
            timestamp: Utc::now(),
        };

        let change = ConfigChange::new(event, vec![Uuid::new_v4()]);

        assert!(!change.propagated);
        assert_eq!(change.retry_count, 0);
        assert!(change.last_error.is_none());
    }

    #[test]
    fn test_config_change_record_failure() {
        let event = SyncEvent::NodeRemoved {
            node_id: Uuid::new_v4(),
            subnet_id: Uuid::new_v4(),
            timestamp: Utc::now(),
        };

        let mut change = ConfigChange::new(event, vec![]);

        change.record_failure("Connection timeout".to_string());
        assert_eq!(change.retry_count, 1);
        assert_eq!(change.last_error, Some("Connection timeout".to_string()));

        change.record_failure("Connection refused".to_string());
        assert_eq!(change.retry_count, 2);
        assert_eq!(change.last_error, Some("Connection refused".to_string()));
    }
}
