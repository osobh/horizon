//! State synchronization for consensus protocol

use crate::error::{ConsensusError, ConsensusResult};
use crate::validator::ValidatorId;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// State synchronization manager
pub struct SyncManager {
    /// This validator's ID
    validator_id: ValidatorId,
    /// Current local height
    local_height: u64,
    /// Known heights of other validators
    peer_heights: HashMap<ValidatorId, u64>,
    /// Pending sync requests
    pending_requests: HashMap<u64, SyncRequest>,
    /// Sync timeout duration
    sync_timeout: Duration,
    /// Maximum concurrent sync requests
    max_concurrent_syncs: usize,
    /// Sync request queue
    request_queue: VecDeque<SyncRequest>,
}

/// State synchronization request
#[derive(Debug, Clone)]
pub struct SyncRequest {
    /// Target height to sync to
    pub target_height: u64,
    /// Validator to sync from
    pub peer_id: ValidatorId,
    /// Request timestamp
    pub requested_at: u64,
    /// Request ID
    pub request_id: String,
}

/// Synchronization status
#[derive(Debug, Clone, PartialEq)]
pub enum SyncStatus {
    /// Up to date with network
    UpToDate,
    /// Syncing with peers
    Syncing { target_height: u64, progress: f64 },
    /// Failed to sync
    Failed { reason: String },
    /// No peers available for sync
    NoPeers,
}

/// State synchronization data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateSync {
    /// Height of the state
    pub height: u64,
    /// State data
    pub state_data: Vec<u8>,
    /// Merkle proof for verification
    pub proof: Vec<u8>,
    /// Validator who provided the state
    pub provider_id: ValidatorId,
    /// Timestamp when state was created
    pub timestamp: u64,
}

impl SyncManager {
    /// Create new sync manager
    #[must_use = "SyncManager must be stored to manage state synchronization"]
    pub fn new(validator_id: ValidatorId) -> Self {
        Self {
            validator_id,
            local_height: 0,
            peer_heights: HashMap::new(),
            pending_requests: HashMap::new(),
            sync_timeout: Duration::from_secs(30),
            max_concurrent_syncs: 3,
            request_queue: VecDeque::new(),
        }
    }

    /// Update local height
    pub fn update_local_height(&mut self, height: u64) {
        self.local_height = height;
    }

    /// Update peer height information
    pub fn update_peer_height(&mut self, peer_id: ValidatorId, height: u64) {
        self.peer_heights.insert(peer_id, height);
    }

    /// Get current sync status
    #[must_use = "sync status indicates whether node needs synchronization"]
    pub fn sync_status(&self) -> SyncStatus {
        let max_peer_height = self.peer_heights.values().max().copied().unwrap_or(0);

        if self.peer_heights.is_empty() {
            return SyncStatus::NoPeers;
        }

        if self.local_height >= max_peer_height {
            return SyncStatus::UpToDate;
        }

        // Check if currently syncing
        if !self.pending_requests.is_empty() {
            let target_height = self
                .pending_requests
                .values()
                .map(|req| req.target_height)
                .max()
                .unwrap_or(max_peer_height);

            let progress = if target_height > self.local_height {
                self.local_height as f64 / target_height as f64
            } else {
                1.0
            };

            return SyncStatus::Syncing {
                target_height,
                progress,
            };
        }

        // Need to sync but not currently syncing
        SyncStatus::UpToDate // Will be updated when sync starts
    }

    /// Check if synchronization is needed
    #[must_use = "ignoring sync requirement can cause node to fall behind"]
    pub fn needs_sync(&self) -> bool {
        let max_peer_height = self.peer_heights.values().max().copied().unwrap_or(0);
        max_peer_height > self.local_height
    }

    /// Start synchronization process
    #[must_use = "sync requests must be processed to complete synchronization"]
    pub fn start_sync(&mut self) -> ConsensusResult<Vec<SyncRequest>> {
        if !self.needs_sync() {
            return Ok(vec![]);
        }

        let max_peer_height = self.peer_heights.values().max().copied().unwrap_or(0);
        let mut requests = Vec::new();

        // Find best peers to sync from
        let mut best_peers: Vec<_> = self
            .peer_heights
            .iter()
            .filter(|(_, height)| **height > self.local_height)
            .collect();

        best_peers.sort_by(|a, b| b.1.cmp(a.1)); // Sort by height descending

        // Create sync requests for different height ranges
        let mut current_height = self.local_height + 1;
        let chunk_size = 100; // Sync in chunks of 100 blocks

        for (peer_id, peer_height) in best_peers.iter().take(self.max_concurrent_syncs) {
            if current_height > max_peer_height {
                break;
            }

            let target_height = (current_height + chunk_size - 1).min(**peer_height);
            let request = SyncRequest {
                target_height,
                peer_id: (*peer_id).clone(),
                requested_at: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_millis() as u64
                    / 1000, // Store as seconds but use millis precision in timeout
                request_id: format!("sync_{}_{}", current_height, target_height),
            };

            self.pending_requests.insert(target_height, request.clone());
            requests.push(request);
            current_height = target_height + 1;
        }

        Ok(requests)
    }

    /// Process received state data
    pub fn process_state_sync(&mut self, state_sync: StateSync) -> ConsensusResult<bool> {
        // Verify state sync is for a pending request
        let request = self
            .pending_requests
            .remove(&state_sync.height)
            .ok_or_else(|| {
                ConsensusError::StateSyncError(format!(
                    "No pending request for height {}",
                    state_sync.height
                ))
            })?;

        // Verify the provider is the expected peer
        if request.peer_id != state_sync.provider_id {
            return Err(ConsensusError::StateSyncError(
                "State sync from unexpected peer".to_string(),
            ));
        }

        // Verify the state proof (simplified verification)
        if !self.verify_state_proof(&state_sync) {
            return Err(ConsensusError::StateSyncError(
                "Invalid state proof".to_string(),
            ));
        }

        // Apply the state
        self.local_height = self.local_height.max(state_sync.height);

        // Check if sync is complete
        let sync_complete = !self.needs_sync() && self.pending_requests.is_empty();

        Ok(sync_complete)
    }

    /// Verify state proof (mock implementation)
    fn verify_state_proof(&self, _state_sync: &StateSync) -> bool {
        // Mock verification - in real implementation would verify Merkle proof
        true
    }

    /// Handle sync timeout
    pub fn handle_timeouts(&mut self) -> Vec<SyncRequest> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        let mut timed_out_requests = Vec::new();
        let timeout_millis = self.sync_timeout.as_millis() as u64;

        self.pending_requests.retain(|_, request| {
            let request_time_millis = request.requested_at * 1000; // Convert to millis
            if now.saturating_sub(request_time_millis) > timeout_millis {
                timed_out_requests.push(request.clone());
                false
            } else {
                true
            }
        });

        // Requeue timed out requests
        for request in &timed_out_requests {
            self.request_queue.push_back(request.clone());
        }

        timed_out_requests
    }

    /// Get next sync request from queue
    pub fn next_sync_request(&mut self) -> Option<SyncRequest> {
        if self.pending_requests.len() >= self.max_concurrent_syncs {
            return None;
        }

        if let Some(mut request) = self.request_queue.pop_front() {
            request.requested_at = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64
                / 1000;

            self.pending_requests
                .insert(request.target_height, request.clone());
            Some(request)
        } else {
            None
        }
    }

    /// Get sync statistics
    pub fn sync_stats(&self) -> SyncStats {
        let max_peer_height = self.peer_heights.values().max().copied().unwrap_or(0);
        let behind_blocks = max_peer_height.saturating_sub(self.local_height);

        SyncStats {
            local_height: self.local_height,
            max_peer_height,
            behind_blocks,
            pending_requests: self.pending_requests.len(),
            queued_requests: self.request_queue.len(),
            known_peers: self.peer_heights.len(),
            sync_status: self.sync_status(),
        }
    }

    /// Reset synchronization state
    pub fn reset(&mut self) {
        self.pending_requests.clear();
        self.request_queue.clear();
        self.peer_heights.clear();
    }

    /// Set sync timeout
    pub fn set_sync_timeout(&mut self, timeout: Duration) {
        self.sync_timeout = timeout;
    }

    /// Set maximum concurrent syncs
    pub fn set_max_concurrent_syncs(&mut self, max_syncs: usize) {
        self.max_concurrent_syncs = max_syncs;
    }

    /// Get local height
    pub fn local_height(&self) -> u64 {
        self.local_height
    }

    /// Get known peer heights
    pub fn peer_heights(&self) -> &HashMap<ValidatorId, u64> {
        &self.peer_heights
    }

    /// Check if peer is known
    pub fn has_peer(&self, peer_id: &ValidatorId) -> bool {
        self.peer_heights.contains_key(peer_id)
    }

    /// Remove peer
    pub fn remove_peer(&mut self, peer_id: &ValidatorId) {
        self.peer_heights.remove(peer_id);

        // Cancel pending requests from this peer
        self.pending_requests
            .retain(|_, request| request.peer_id != *peer_id);

        // Remove queued requests from this peer
        self.request_queue
            .retain(|request| request.peer_id != *peer_id);
    }
}

/// Synchronization statistics
#[derive(Debug, Clone)]
pub struct SyncStats {
    /// Current local height
    pub local_height: u64,
    /// Maximum height among peers
    pub max_peer_height: u64,
    /// Number of blocks behind
    pub behind_blocks: u64,
    /// Number of pending sync requests
    pub pending_requests: usize,
    /// Number of queued sync requests
    pub queued_requests: usize,
    /// Number of known peers
    pub known_peers: usize,
    /// Current sync status
    pub sync_status: SyncStatus,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_validator_id() -> ValidatorId {
        ValidatorId::new()
    }

    #[test]
    fn test_sync_manager_creation() {
        let validator_id = create_test_validator_id();
        let sync_manager = SyncManager::new(validator_id.clone());

        assert_eq!(sync_manager.validator_id, validator_id);
        assert_eq!(sync_manager.local_height, 0);
        assert!(sync_manager.peer_heights.is_empty());
    }

    #[test]
    fn test_update_local_height() {
        let validator_id = create_test_validator_id();
        let mut sync_manager = SyncManager::new(validator_id);

        sync_manager.update_local_height(10);
        assert_eq!(sync_manager.local_height(), 10);
    }

    #[test]
    fn test_update_peer_height() {
        let validator_id = create_test_validator_id();
        let peer_id = create_test_validator_id();
        let mut sync_manager = SyncManager::new(validator_id);

        sync_manager.update_peer_height(peer_id.clone(), 15);
        assert_eq!(sync_manager.peer_heights().get(&peer_id), Some(&15));
    }

    #[test]
    fn test_needs_sync() {
        let validator_id = create_test_validator_id();
        let peer_id = create_test_validator_id();
        let mut sync_manager = SyncManager::new(validator_id);

        // No peers, no sync needed
        assert!(!sync_manager.needs_sync());

        // Peer at same height, no sync needed
        sync_manager.update_peer_height(peer_id.clone(), 0);
        assert!(!sync_manager.needs_sync());

        // Peer ahead, sync needed
        sync_manager.update_peer_height(peer_id, 10);
        assert!(sync_manager.needs_sync());
    }

    #[test]
    fn test_sync_status_no_peers() {
        let validator_id = create_test_validator_id();
        let sync_manager = SyncManager::new(validator_id);

        assert_eq!(sync_manager.sync_status(), SyncStatus::NoPeers);
    }

    #[test]
    fn test_sync_status_up_to_date() {
        let validator_id = create_test_validator_id();
        let peer_id = create_test_validator_id();
        let mut sync_manager = SyncManager::new(validator_id);

        sync_manager.update_local_height(10);
        sync_manager.update_peer_height(peer_id, 10);

        assert_eq!(sync_manager.sync_status(), SyncStatus::UpToDate);
    }

    #[test]
    fn test_sync_status_syncing() {
        let validator_id = create_test_validator_id();
        let peer_id = create_test_validator_id();
        let mut sync_manager = SyncManager::new(validator_id);

        sync_manager.update_local_height(5);
        sync_manager.update_peer_height(peer_id.clone(), 15);

        // Start sync
        let requests = sync_manager.start_sync().unwrap();
        assert!(!requests.is_empty());

        // Should be syncing now
        if let SyncStatus::Syncing {
            target_height,
            progress,
        } = sync_manager.sync_status()
        {
            assert!(target_height > 5);
            assert!(progress >= 0.0 && progress <= 1.0);
        } else {
            panic!("Expected Syncing status");
        }
    }

    #[test]
    fn test_start_sync() {
        let validator_id = create_test_validator_id();
        let peer_id = create_test_validator_id();
        let mut sync_manager = SyncManager::new(validator_id);

        sync_manager.update_local_height(0);
        sync_manager.update_peer_height(peer_id.clone(), 50);

        let requests = sync_manager.start_sync().unwrap();
        assert!(!requests.is_empty());

        let request = &requests[0];
        assert_eq!(request.peer_id, peer_id);
        assert!(request.target_height > 0);
    }

    #[test]
    fn test_start_sync_no_sync_needed() {
        let validator_id = create_test_validator_id();
        let peer_id = create_test_validator_id();
        let mut sync_manager = SyncManager::new(validator_id);

        sync_manager.update_local_height(10);
        sync_manager.update_peer_height(peer_id, 10);

        let requests = sync_manager.start_sync().unwrap();
        assert!(requests.is_empty());
    }

    #[test]
    fn test_process_state_sync() {
        let validator_id = create_test_validator_id();
        let peer_id = create_test_validator_id();
        let mut sync_manager = SyncManager::new(validator_id);

        // Set up pending request
        sync_manager.update_local_height(0);
        sync_manager.update_peer_height(peer_id.clone(), 10);
        let requests = sync_manager.start_sync().unwrap();
        let request = &requests[0];

        // Process state sync
        let state_sync = StateSync {
            height: request.target_height,
            state_data: vec![1, 2, 3, 4],
            proof: vec![5, 6, 7, 8],
            provider_id: peer_id,
            timestamp: 123456789,
        };

        let result = sync_manager.process_state_sync(state_sync);
        assert!(result.is_ok());
        assert_eq!(sync_manager.local_height(), request.target_height);
    }

    #[test]
    fn test_process_state_sync_no_pending_request() {
        let validator_id = create_test_validator_id();
        let peer_id = create_test_validator_id();
        let mut sync_manager = SyncManager::new(validator_id);

        let state_sync = StateSync {
            height: 10,
            state_data: vec![1, 2, 3, 4],
            proof: vec![5, 6, 7, 8],
            provider_id: peer_id,
            timestamp: 123456789,
        };

        let result = sync_manager.process_state_sync(state_sync);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ConsensusError::StateSyncError(_)
        ));
    }

    #[test]
    fn test_process_state_sync_wrong_peer() {
        let validator_id = create_test_validator_id();
        let peer_id = create_test_validator_id();
        let wrong_peer_id = create_test_validator_id();
        let mut sync_manager = SyncManager::new(validator_id);

        // Set up pending request
        sync_manager.update_local_height(0);
        sync_manager.update_peer_height(peer_id, 10);
        let requests = sync_manager.start_sync().unwrap();
        let request = &requests[0];

        // Process state sync from wrong peer
        let state_sync = StateSync {
            height: request.target_height,
            state_data: vec![1, 2, 3, 4],
            proof: vec![5, 6, 7, 8],
            provider_id: wrong_peer_id, // Wrong peer
            timestamp: 123456789,
        };

        let result = sync_manager.process_state_sync(state_sync);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ConsensusError::StateSyncError(_)
        ));
    }

    #[test]
    fn test_handle_timeouts() {
        let validator_id = create_test_validator_id();
        let peer_id = create_test_validator_id();
        let mut sync_manager = SyncManager::new(validator_id);

        sync_manager.set_sync_timeout(Duration::from_millis(5)); // Very short timeout

        // Set up pending request
        sync_manager.update_local_height(0);
        sync_manager.update_peer_height(peer_id, 10);
        let _requests = sync_manager.start_sync().unwrap();

        // Wait for timeout with buffer
        std::thread::sleep(Duration::from_millis(50));

        let timed_out = sync_manager.handle_timeouts();
        assert!(!timed_out.is_empty());
        assert!(sync_manager.pending_requests.is_empty());
        assert!(!sync_manager.request_queue.is_empty());
    }

    #[test]
    fn test_next_sync_request() {
        let validator_id = create_test_validator_id();
        let mut sync_manager = SyncManager::new(validator_id);

        // Add request to queue manually
        let request = SyncRequest {
            target_height: 10,
            peer_id: create_test_validator_id(),
            requested_at: 0,
            request_id: "test_request".to_string(),
        };
        sync_manager.request_queue.push_back(request.clone());

        let next_request = sync_manager.next_sync_request();
        assert!(next_request.is_some());
        assert_eq!(next_request.unwrap().request_id, "test_request");
        assert!(!sync_manager.pending_requests.is_empty());
    }

    #[test]
    fn test_next_sync_request_max_concurrent() {
        let validator_id = create_test_validator_id();
        let mut sync_manager = SyncManager::new(validator_id);

        sync_manager.set_max_concurrent_syncs(1);

        // Add pending request (at max capacity)
        let pending_request = SyncRequest {
            target_height: 5,
            peer_id: create_test_validator_id(),
            requested_at: 123456789,
            request_id: "pending_request".to_string(),
        };
        sync_manager.pending_requests.insert(5, pending_request);

        // Add queued request
        let queued_request = SyncRequest {
            target_height: 10,
            peer_id: create_test_validator_id(),
            requested_at: 0,
            request_id: "queued_request".to_string(),
        };
        sync_manager.request_queue.push_back(queued_request);

        // Should not return request due to max concurrent limit
        let next_request = sync_manager.next_sync_request();
        assert!(next_request.is_none());
    }

    #[test]
    fn test_sync_stats() {
        let validator_id = create_test_validator_id();
        let peer_id = create_test_validator_id();
        let mut sync_manager = SyncManager::new(validator_id);

        sync_manager.update_local_height(5);
        sync_manager.update_peer_height(peer_id, 15);

        let stats = sync_manager.sync_stats();
        assert_eq!(stats.local_height, 5);
        assert_eq!(stats.max_peer_height, 15);
        assert_eq!(stats.behind_blocks, 10);
        assert_eq!(stats.known_peers, 1);
    }

    #[test]
    fn test_reset() {
        let validator_id = create_test_validator_id();
        let peer_id = create_test_validator_id();
        let mut sync_manager = SyncManager::new(validator_id);

        sync_manager.update_peer_height(peer_id, 10);
        let _requests = sync_manager.start_sync().unwrap();

        assert!(!sync_manager.peer_heights.is_empty());
        assert!(!sync_manager.pending_requests.is_empty());

        sync_manager.reset();

        assert!(sync_manager.peer_heights.is_empty());
        assert!(sync_manager.pending_requests.is_empty());
        assert!(sync_manager.request_queue.is_empty());
    }

    #[test]
    fn test_remove_peer() {
        let validator_id = create_test_validator_id();
        let peer_id = create_test_validator_id();
        let mut sync_manager = SyncManager::new(validator_id);

        sync_manager.update_peer_height(peer_id.clone(), 10);
        let _requests = sync_manager.start_sync().unwrap();

        assert!(sync_manager.has_peer(&peer_id));
        assert!(!sync_manager.pending_requests.is_empty());

        sync_manager.remove_peer(&peer_id);

        assert!(!sync_manager.has_peer(&peer_id));
        assert!(sync_manager.pending_requests.is_empty());
    }

    #[test]
    fn test_state_sync_serialization() {
        let state_sync = StateSync {
            height: 10,
            state_data: vec![1, 2, 3, 4],
            proof: vec![5, 6, 7, 8],
            provider_id: create_test_validator_id(),
            timestamp: 123456789,
        };

        let serialized = serde_json::to_string(&state_sync).unwrap();
        let deserialized: StateSync = serde_json::from_str(&serialized).unwrap();

        assert_eq!(state_sync.height, deserialized.height);
        assert_eq!(state_sync.state_data, deserialized.state_data);
        assert_eq!(state_sync.proof, deserialized.proof);
        assert_eq!(state_sync.provider_id, deserialized.provider_id);
        assert_eq!(state_sync.timestamp, deserialized.timestamp);
    }
}
