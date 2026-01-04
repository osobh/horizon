//! WireGuard configuration synchronization service
//!
//! Handles distribution of WireGuard configurations to nodes,
//! tracking sync status, and managing configuration updates.

use crate::models::{Subnet, SubnetAssignment};
use crate::wireguard::WireGuardConfigGenerator;
use crate::{Error, Result};
use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use tracing::{debug, info, warn};
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
    /// Node private keys (for initial config push)
    node_private_keys: Arc<RwLock<HashMap<Uuid, String>>>,
    /// Node endpoints cache
    node_endpoints: Arc<RwLock<HashMap<Uuid, SocketAddr>>>,
    /// HTTP client for node communication
    http_client: Arc<NodeHttpClient>,
    /// Config generator
    config_generator: Arc<WireGuardConfigGenerator>,
    /// Maximum retry attempts
    max_retries: u32,
    /// Sync timeout in seconds
    sync_timeout_secs: u64,
    /// Signing key for config signatures (base64 encoded)
    signing_key: Option<String>,
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

/// WireGuard configuration request sent to swarmlet nodes
/// This matches the swarmlet's WireGuardConfigRequest
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeWireGuardConfigRequest {
    /// Interface name (e.g., "wg-swarm0")
    pub interface_name: String,
    /// Private key (base64 encoded) - only sent during initial setup
    #[serde(skip_serializing_if = "Option::is_none")]
    pub private_key: Option<String>,
    /// Listen port for WireGuard
    pub listen_port: u16,
    /// Assigned IP address with CIDR (e.g., "10.0.1.5/24")
    pub address: String,
    /// MTU (optional, defaults to 1420)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mtu: Option<u16>,
    /// Peers to configure
    pub peers: Vec<NodePeerConfig>,
    /// Configuration version (for change tracking)
    pub config_version: String,
    /// Signature from coordinator (for verification)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,
}

/// Peer configuration sent to nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodePeerConfig {
    /// Peer public key (base64)
    pub public_key: String,
    /// Preshared key (optional, base64)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub preshared_key: Option<String>,
    /// Allowed IPs for this peer
    pub allowed_ips: Vec<String>,
    /// Endpoint (host:port)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub endpoint: Option<String>,
    /// Persistent keepalive interval in seconds
    #[serde(skip_serializing_if = "Option::is_none")]
    pub persistent_keepalive: Option<u16>,
}

/// Response from swarmlet after config application
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeConfigResponse {
    /// Whether the configuration was applied successfully
    pub success: bool,
    /// Error message if failed
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Current configuration version
    pub config_version: String,
}

/// HTTP client wrapper for node communication
pub struct NodeHttpClient {
    client: reqwest::Client,
    timeout: Duration,
}

impl NodeHttpClient {
    /// Create a new HTTP client for node communication
    pub fn new(timeout_secs: u64) -> Self {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(timeout_secs))
            .build()
            .unwrap_or_default();

        Self {
            client,
            timeout: Duration::from_secs(timeout_secs),
        }
    }

    /// Push WireGuard configuration to a node
    pub async fn push_config(
        &self,
        endpoint: &str,
        config: &NodeWireGuardConfigRequest,
    ) -> Result<NodeConfigResponse> {
        let url = format!(
            "{}/api/v1/wireguard/configure",
            endpoint.trim_end_matches('/')
        );

        debug!("Pushing WireGuard config to {}", url);

        let response = self
            .client
            .post(&url)
            .json(config)
            .send()
            .await
            .map_err(|e| Error::WireGuardSync(format!("Failed to connect to node: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(Error::WireGuardSync(format!(
                "Node returned error {}: {}",
                status, body
            )));
        }

        response
            .json::<NodeConfigResponse>()
            .await
            .map_err(|e| Error::WireGuardSync(format!("Failed to parse response: {}", e)))
    }

    /// Check if a node is reachable
    pub async fn health_check(&self, endpoint: &str) -> bool {
        let url = format!("{}/health", endpoint.trim_end_matches('/'));

        self.client
            .get(&url)
            .timeout(Duration::from_secs(5))
            .send()
            .await
            .map(|r| r.status().is_success())
            .unwrap_or(false)
    }
}

impl Default for NodeHttpClient {
    fn default() -> Self {
        Self::new(30)
    }
}

impl ConfigSyncService {
    /// Create a new configuration sync service
    pub fn new() -> Self {
        Self {
            node_states: Arc::new(RwLock::new(HashMap::new())),
            pending_changes: Arc::new(RwLock::new(Vec::new())),
            subnet_configs: Arc::new(RwLock::new(HashMap::new())),
            node_private_keys: Arc::new(RwLock::new(HashMap::new())),
            node_endpoints: Arc::new(RwLock::new(HashMap::new())),
            http_client: Arc::new(NodeHttpClient::new(30)),
            config_generator: Arc::new(WireGuardConfigGenerator::new()),
            max_retries: 3,
            sync_timeout_secs: 30,
            signing_key: None,
        }
    }

    /// Create with custom configuration
    pub fn with_config(max_retries: u32, sync_timeout_secs: u64) -> Self {
        Self {
            node_states: Arc::new(RwLock::new(HashMap::new())),
            pending_changes: Arc::new(RwLock::new(Vec::new())),
            subnet_configs: Arc::new(RwLock::new(HashMap::new())),
            node_private_keys: Arc::new(RwLock::new(HashMap::new())),
            node_endpoints: Arc::new(RwLock::new(HashMap::new())),
            http_client: Arc::new(NodeHttpClient::new(sync_timeout_secs)),
            config_generator: Arc::new(WireGuardConfigGenerator::new()),
            max_retries,
            sync_timeout_secs,
            signing_key: None,
        }
    }

    /// Set the signing key for config signatures
    pub fn with_signing_key(mut self, signing_key: String) -> Self {
        self.signing_key = Some(signing_key);
        self
    }

    /// Sign a configuration payload using ed25519
    fn sign_config(&self, config: &NodeWireGuardConfigRequest) -> Option<String> {
        use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
        use base64::Engine;
        use ed25519_dalek::{Signer, SigningKey};

        let signing_key_b64 = self.signing_key.as_ref()?;

        // Decode the signing key
        let key_bytes = BASE64_STANDARD.decode(signing_key_b64).ok()?;
        if key_bytes.len() != 32 {
            warn!(
                "Invalid signing key length: expected 32, got {}",
                key_bytes.len()
            );
            return None;
        }

        let mut key_array = [0u8; 32];
        key_array.copy_from_slice(&key_bytes);
        let signing_key = SigningKey::from_bytes(&key_array);

        // Create the message to sign (config version + address + peers info)
        let mut message = format!(
            "{}:{}:{}:{}",
            config.config_version, config.address, config.listen_port, config.interface_name
        );

        // Add peer public keys to the message
        for peer in &config.peers {
            message.push(':');
            message.push_str(&peer.public_key);
        }

        // Sign the message
        let signature = signing_key.sign(message.as_bytes());
        Some(BASE64_STANDARD.encode(signature.to_bytes()))
    }

    /// Register a node for sync tracking
    pub fn register_node(&self, node_id: Uuid, endpoint: Option<String>) {
        let mut states = self.node_states.write();
        let state = states
            .entry(node_id)
            .or_insert_with(|| NodeSyncState::new(node_id));
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
        let changes = self.pending_changes.write();
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
            .ok_or(Error::SubnetNotFound(subnet_id))?;

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

    // ============== Config Push Methods ==============

    /// Register a node's private key for config push
    pub fn register_node_private_key(&self, node_id: Uuid, private_key: String) {
        self.node_private_keys.write().insert(node_id, private_key);
    }

    /// Register a node's WireGuard endpoint
    pub fn register_node_wg_endpoint(&self, node_id: Uuid, endpoint: SocketAddr) {
        self.node_endpoints.write().insert(node_id, endpoint);
    }

    /// Generate WireGuard config for a specific node
    pub fn generate_node_config(
        &self,
        node_id: Uuid,
        subnet_id: Uuid,
    ) -> Result<NodeWireGuardConfigRequest> {
        let configs = self.subnet_configs.read();
        let subnet_config = configs
            .get(&subnet_id)
            .ok_or(Error::SubnetNotFound(subnet_id))?;

        // Find the node's assignment
        let assignment = subnet_config
            .assignments
            .iter()
            .find(|a| a.node_id == node_id)
            .ok_or_else(|| {
                Error::WireGuardSync(format!("Node {} not assigned to subnet", node_id))
            })?;

        // Get node's private key (optional - node may already have one)
        let private_key = self.node_private_keys.read().get(&node_id).cloned();

        // Build peer list (all other nodes in subnet + subnet gateway)
        let mut peers = Vec::new();

        // Add subnet gateway as peer (if it has a public key)
        if let Some(ref gateway_pubkey) = subnet_config.subnet.wg_public_key {
            if subnet_config.subnet.gateway_ip().is_some() {
                peers.push(NodePeerConfig {
                    public_key: gateway_pubkey.clone(),
                    preshared_key: None,
                    allowed_ips: vec![subnet_config.subnet.cidr.to_string()],
                    endpoint: None, // Gateway endpoint would be configured separately
                    persistent_keepalive: Some(25),
                });
            }
        }

        // Add other nodes in subnet as peers
        let node_endpoints = self.node_endpoints.read();
        for other_assignment in &subnet_config.assignments {
            if other_assignment.node_id == node_id {
                continue; // Skip self
            }

            let endpoint = node_endpoints
                .get(&other_assignment.node_id)
                .map(|e| e.to_string());

            peers.push(NodePeerConfig {
                public_key: other_assignment.wg_public_key.clone(),
                preshared_key: None,
                allowed_ips: vec![format!("{}/32", other_assignment.assigned_ip)],
                endpoint,
                persistent_keepalive: Some(25),
            });
        }

        // Build the config request
        let address = format!(
            "{}/{}",
            assignment.assigned_ip,
            subnet_config.subnet.cidr.prefix_len()
        );

        let mut config = NodeWireGuardConfigRequest {
            interface_name: subnet_config.subnet.wg_interface.clone(),
            private_key,
            listen_port: subnet_config.subnet.wg_listen_port,
            address,
            mtu: Some(1420),
            peers,
            config_version: subnet_config.config_version.clone(),
            signature: None,
        };

        // Sign the configuration if signing key is available
        config.signature = self.sign_config(&config);

        Ok(config)
    }

    /// Push WireGuard config to a single node
    pub async fn push_config_to_node(&self, node_id: Uuid, subnet_id: Uuid) -> Result<()> {
        // Get node endpoint
        let endpoint = {
            let states = self.node_states.read();
            states
                .get(&node_id)
                .and_then(|s| s.endpoint.clone())
                .ok_or_else(|| Error::WireGuardSync(format!("No endpoint for node {}", node_id)))?
        };

        // Mark as syncing
        {
            let mut states = self.node_states.write();
            if let Some(state) = states.get_mut(&node_id) {
                state.status = SyncStatus::Syncing;
                state.last_attempt = Some(Utc::now());
            }
        }

        // Generate config
        let config = match self.generate_node_config(node_id, subnet_id) {
            Ok(c) => c,
            Err(e) => {
                self.mark_node_failed(node_id, e.to_string());
                return Err(e);
            }
        };

        let config_version = config.config_version.clone();

        // Push to node
        info!(
            "Pushing WireGuard config v{} to node {} at {}",
            config_version, node_id, endpoint
        );

        match self.http_client.push_config(&endpoint, &config).await {
            Ok(response) => {
                if response.success {
                    info!("Successfully synced node {}", node_id);
                    self.mark_node_synced(node_id, config_version);
                    Ok(())
                } else {
                    let error = response
                        .error
                        .unwrap_or_else(|| "Unknown error".to_string());
                    warn!("Node {} rejected config: {}", node_id, error);
                    self.mark_node_failed(node_id, error.clone());
                    Err(Error::WireGuardSync(error))
                }
            }
            Err(e) => {
                warn!("Failed to push config to node {}: {}", node_id, e);
                self.mark_node_failed(node_id, e.to_string());
                Err(e)
            }
        }
    }

    /// Sync all nodes that need syncing in a subnet
    pub async fn sync_subnet(&self, subnet_id: Uuid) -> SyncResult {
        let configs = self.subnet_configs.read();
        let Some(config) = configs.get(&subnet_id) else {
            return SyncResult {
                total: 0,
                succeeded: 0,
                failed: 0,
                skipped: 0,
                errors: vec![format!("Subnet {} not found", subnet_id)],
            };
        };

        let node_ids: Vec<Uuid> = config.assignments.iter().map(|a| a.node_id).collect();
        drop(configs);

        self.sync_nodes(&node_ids, subnet_id).await
    }

    /// Sync specific nodes
    pub async fn sync_nodes(&self, node_ids: &[Uuid], subnet_id: Uuid) -> SyncResult {
        let mut result = SyncResult::default();
        result.total = node_ids.len();

        for node_id in node_ids {
            // Check if node needs sync
            let needs_sync = {
                let states = self.node_states.read();
                states
                    .get(node_id)
                    .map(|s| {
                        s.status == SyncStatus::Pending
                            || s.status == SyncStatus::Stale
                            || (s.status == SyncStatus::Failed && s.should_retry(self.max_retries))
                    })
                    .unwrap_or(false)
            };

            if !needs_sync {
                result.skipped += 1;
                continue;
            }

            match self.push_config_to_node(*node_id, subnet_id).await {
                Ok(()) => result.succeeded += 1,
                Err(e) => {
                    result.failed += 1;
                    result.errors.push(format!("Node {}: {}", node_id, e));
                }
            }
        }

        result
    }

    /// Sync all pending nodes across all subnets
    pub async fn sync_all_pending(&self) -> SyncResult {
        let nodes_needing_sync = self.get_nodes_needing_sync();
        let mut result = SyncResult::default();
        result.total = nodes_needing_sync.len();

        // Group nodes by subnet
        let mut nodes_by_subnet: HashMap<Uuid, Vec<Uuid>> = HashMap::new();
        {
            let configs = self.subnet_configs.read();
            for node_state in &nodes_needing_sync {
                // Find which subnet this node belongs to
                for (subnet_id, config) in configs.iter() {
                    if config
                        .assignments
                        .iter()
                        .any(|a| a.node_id == node_state.node_id)
                    {
                        nodes_by_subnet
                            .entry(*subnet_id)
                            .or_default()
                            .push(node_state.node_id);
                        break;
                    }
                }
            }
        }

        // Sync each subnet's nodes
        for (subnet_id, node_ids) in nodes_by_subnet {
            let subnet_result = self.sync_nodes(&node_ids, subnet_id).await;
            result.succeeded += subnet_result.succeeded;
            result.failed += subnet_result.failed;
            result.skipped += subnet_result.skipped;
            result.errors.extend(subnet_result.errors);
        }

        result
    }

    /// Propagate a new peer to all existing nodes in a subnet
    pub async fn propagate_peer_added(&self, new_node_id: Uuid, subnet_id: Uuid) -> Result<()> {
        info!(
            "Propagating new peer {} to subnet {}",
            new_node_id, subnet_id
        );

        // Queue change for all existing nodes (except the new one)
        let affected_nodes: Vec<Uuid> = {
            let configs = self.subnet_configs.read();
            configs
                .get(&subnet_id)
                .map(|c| {
                    c.assignments
                        .iter()
                        .filter(|a| a.node_id != new_node_id)
                        .map(|a| a.node_id)
                        .collect()
                })
                .unwrap_or_default()
        };

        if !affected_nodes.is_empty() {
            self.queue_change(
                SyncEvent::NodeAdded {
                    node_id: new_node_id,
                    subnet_id,
                    timestamp: Utc::now(),
                },
                affected_nodes,
            );
        }

        Ok(())
    }

    /// Propagate peer removal to all nodes in a subnet
    pub async fn propagate_peer_removed(
        &self,
        removed_node_id: Uuid,
        subnet_id: Uuid,
    ) -> Result<()> {
        info!(
            "Propagating peer removal {} from subnet {}",
            removed_node_id, subnet_id
        );

        // Queue change for all remaining nodes
        let affected_nodes: Vec<Uuid> = {
            let configs = self.subnet_configs.read();
            configs
                .get(&subnet_id)
                .map(|c| {
                    c.assignments
                        .iter()
                        .filter(|a| a.node_id != removed_node_id)
                        .map(|a| a.node_id)
                        .collect()
                })
                .unwrap_or_default()
        };

        if !affected_nodes.is_empty() {
            self.queue_change(
                SyncEvent::NodeRemoved {
                    node_id: removed_node_id,
                    subnet_id,
                    timestamp: Utc::now(),
                },
                affected_nodes,
            );
        }

        // Remove the node from tracking
        self.unregister_node(removed_node_id);

        Ok(())
    }

    /// Run the sync loop (for background task)
    pub async fn run_sync_loop(
        self: Arc<Self>,
        mut shutdown: tokio::sync::watch::Receiver<bool>,
        sync_interval: Duration,
    ) {
        info!("Starting WireGuard config sync loop");
        let mut interval = tokio::time::interval(sync_interval);

        loop {
            tokio::select! {
                _ = interval.tick() => {
                    // Sync pending nodes
                    let result = self.sync_all_pending().await;
                    if result.total > 0 {
                        info!(
                            "Sync cycle complete: {}/{} succeeded, {} failed, {} skipped",
                            result.succeeded, result.total, result.failed, result.skipped
                        );
                    }

                    // Cleanup old changes
                    self.cleanup_propagated_changes(3600); // Keep for 1 hour
                }
                _ = shutdown.changed() => {
                    info!("Sync loop shutting down");
                    break;
                }
            }
        }
    }
}

/// Result of a sync operation
#[derive(Debug, Clone, Default)]
pub struct SyncResult {
    /// Total nodes attempted
    pub total: usize,
    /// Successfully synced
    pub succeeded: usize,
    /// Failed to sync
    pub failed: usize,
    /// Skipped (already synced)
    pub skipped: usize,
    /// Error messages
    pub errors: Vec<String>,
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
