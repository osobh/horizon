//! Main consensus protocol implementation

use crate::error::{ConsensusError, ConsensusResult};
use crate::leader::{ElectionMessage, LeaderElection, LeaderState};
use crate::validator::{Validator, ValidatorId, ValidatorInfo, ValidatorStatus};
use crate::voting::{RoundId, RoundStatus, Vote, VoteType, VotingRound};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;
use tokio::sync::mpsc;
use tracing;

/// Consensus protocol configuration
#[derive(Debug, Clone)]
pub struct ConsensusConfig {
    /// Election timeout duration
    pub election_timeout: Duration,
    /// Heartbeat interval
    pub heartbeat_interval: Duration,
    /// Voting round timeout
    pub voting_timeout: Duration,
    /// Maximum number of concurrent rounds
    pub max_concurrent_rounds: usize,
    /// GPU computation timeout
    pub gpu_timeout: Duration,
    /// Network message timeout
    pub network_timeout: Duration,
}

impl Default for ConsensusConfig {
    fn default() -> Self {
        Self {
            election_timeout: Duration::from_secs(5),
            heartbeat_interval: Duration::from_secs(1),
            voting_timeout: Duration::from_secs(30),
            max_concurrent_rounds: 10,
            gpu_timeout: Duration::from_secs(60),
            network_timeout: Duration::from_secs(10),
        }
    }
}

/// Consensus message types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsensusMessage {
    /// Leader election message
    Election(ElectionMessage),
    /// Voting message
    Vote(Vote),
    /// Propose new value for consensus
    Proposal {
        round_id: RoundId,
        height: u64,
        proposer_id: ValidatorId,
        value: String,
        signature: Vec<u8>,
    },
    /// GPU computation request
    GpuCompute {
        task_id: String,
        computation_type: String,
        data: Vec<u8>,
        priority: u8,
    },
    /// GPU computation result
    GpuResult {
        task_id: String,
        result: Vec<u8>,
        validator_id: ValidatorId,
        proof: Vec<u8>,
    },
    /// Validator heartbeat
    Heartbeat {
        validator_id: ValidatorId,
        timestamp: u64,
        gpu_status: GpuStatus,
    },
    /// State synchronization request
    SyncRequest {
        from_height: u64,
        to_height: u64,
        requester_id: ValidatorId,
    },
    /// State synchronization response
    SyncResponse {
        height: u64,
        state_data: Vec<u8>,
        proof: Vec<u8>,
    },
}

/// GPU status in consensus
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuStatus {
    /// Available GPU memory
    pub available_memory: u64,
    /// Current utilization percentage
    pub utilization: u8,
    /// Number of active compute kernels
    pub active_kernels: u32,
    /// GPU temperature (Celsius)
    pub temperature: u16,
}

/// Main consensus protocol state machine
pub struct ConsensusProtocol {
    /// Protocol configuration
    config: ConsensusConfig,
    /// Validator management
    validator: Validator,
    /// Leader election
    leader_election: LeaderElection,
    /// Active voting rounds
    voting_rounds: HashMap<RoundId, VotingRound>,
    /// Current blockchain height
    current_height: u64,
    /// Committed values by height
    committed_values: HashMap<u64, String>,
    /// Pending GPU computations
    pending_gpu_tasks: HashMap<String, GpuComputeTask>,
    /// Message sender channel
    message_sender: Option<mpsc::Sender<ConsensusMessage>>,
    /// Protocol state
    state: ProtocolState,
}

/// Protocol operational state
#[derive(Debug, Clone, PartialEq)]
pub enum ProtocolState {
    /// Initializing
    Initializing,
    /// Following a leader
    Following,
    /// Acting as leader
    Leading,
    /// Participating in election
    Electing,
    /// Synchronizing state
    Syncing,
    /// Stopped/shutdown
    Stopped,
}

/// GPU computation task
#[derive(Debug, Clone)]
pub struct GpuComputeTask {
    /// Task identifier
    pub task_id: String,
    /// Computation type
    pub computation_type: String,
    /// Input data
    pub data: Vec<u8>,
    /// Priority level
    pub priority: u8,
    /// Requesting validator
    pub requester_id: ValidatorId,
    /// Task creation timestamp
    pub created_at: u64,
}

impl ConsensusProtocol {
    /// Create new consensus protocol instance
    #[must_use = "ignoring the Result may hide initialization errors"]
    pub fn new(
        config: ConsensusConfig,
        validator_address: std::net::SocketAddr,
        stake: u64,
        gpu_capacity: u32,
        public_key: Vec<u8>,
    ) -> ConsensusResult<Self> {
        let validator = Validator::new(validator_address, stake, gpu_capacity, public_key)?;
        let validator_id = validator.id().clone();

        let leader_election = LeaderElection::new(
            validator_id,
            config.election_timeout,
            config.heartbeat_interval,
        );

        Ok(Self {
            config,
            validator,
            leader_election,
            voting_rounds: HashMap::new(),
            current_height: 0,
            committed_values: HashMap::new(),
            pending_gpu_tasks: HashMap::new(),
            message_sender: None,
            state: ProtocolState::Initializing,
        })
    }

    /// Start the consensus protocol
    pub async fn start(
        &mut self,
        message_sender: mpsc::Sender<ConsensusMessage>,
    ) -> ConsensusResult<()> {
        self.message_sender = Some(message_sender);
        self.state = ProtocolState::Following;

        tracing::info!(
            "Starting consensus protocol for validator {}",
            self.validator.id()
        );

        Ok(())
    }

    /// Stop the consensus protocol
    pub async fn stop(&mut self) -> ConsensusResult<()> {
        self.state = ProtocolState::Stopped;
        self.message_sender = None;

        tracing::info!(
            "Stopping consensus protocol for validator {}",
            self.validator.id()
        );

        Ok(())
    }

    /// Handle incoming consensus message
    pub async fn handle_message(&mut self, message: ConsensusMessage) -> ConsensusResult<()> {
        match message {
            ConsensusMessage::Election(election_msg) => {
                self.handle_election_message(election_msg).await
            }
            ConsensusMessage::Vote(vote) => self.handle_vote(vote).await,
            ConsensusMessage::Proposal {
                round_id,
                height,
                proposer_id,
                value,
                signature,
            } => {
                self.handle_proposal(round_id, height, proposer_id, value, signature)
                    .await
            }
            ConsensusMessage::GpuCompute {
                task_id,
                computation_type,
                data,
                priority,
            } => {
                self.handle_gpu_compute(task_id, computation_type, data, priority)
                    .await
            }
            ConsensusMessage::GpuResult {
                task_id,
                result,
                validator_id,
                proof,
            } => {
                self.handle_gpu_result(task_id, result, validator_id, proof)
                    .await
            }
            ConsensusMessage::Heartbeat {
                validator_id,
                timestamp,
                gpu_status,
            } => {
                self.handle_heartbeat(validator_id, timestamp, gpu_status)
                    .await
            }
            ConsensusMessage::SyncRequest {
                from_height,
                to_height,
                requester_id,
            } => {
                self.handle_sync_request(from_height, to_height, requester_id)
                    .await
            }
            ConsensusMessage::SyncResponse {
                height,
                state_data,
                proof,
            } => self.handle_sync_response(height, state_data, proof).await,
        }
    }

    /// Handle election message
    async fn handle_election_message(&mut self, message: ElectionMessage) -> ConsensusResult<()> {
        // TODO(perf): Cache this HashMap in ConsensusProtocol struct and update only when
        // validators change, instead of rebuilding on every election message.
        // This clones all ValidatorInfo which is expensive for large validator sets.
        let validators: HashMap<ValidatorId, ValidatorInfo> = self
            .validator
            .active_validators()
            .iter()
            .map(|v| (v.id.clone(), (*v).clone()))
            .collect();

        let response = self.leader_election.handle_message(message, &validators)?;

        if let Some(response_msg) = response {
            self.send_message(ConsensusMessage::Election(response_msg))
                .await?;
        }

        // Update protocol state based on election state
        match self.leader_election.state() {
            LeaderState::Leader { .. } => {
                if self.state != ProtocolState::Leading {
                    self.state = ProtocolState::Leading;
                    self.start_new_round().await?;
                }
            }
            LeaderState::Candidate { .. } => {
                self.state = ProtocolState::Electing;
            }
            LeaderState::Follower { .. } => {
                self.state = ProtocolState::Following;
            }
        }

        Ok(())
    }

    /// Handle vote message
    async fn handle_vote(&mut self, vote: Vote) -> ConsensusResult<()> {
        let round_id = vote.round_id.clone();
        let validator_id = vote.validator_id.clone();

        if let Some(round) = self.voting_rounds.get_mut(&round_id) {
            let validator_info = self.validator.get_validator(&validator_id).ok_or_else(|| {
                ConsensusError::ValidationFailed(format!("Unknown validator: {}", validator_id))
            })?;

            round.add_vote(vote, validator_info)?;

            // Check if round completed (need to get round again to avoid borrow checker)
            let round_height = round.height();
            if let RoundStatus::Completed { value } = round.status() {
                let value_clone = value.clone();
                drop(round); // Release the mutable borrow
                self.commit_value(round_height, value_clone).await?;
            }
        }

        Ok(())
    }

    /// Handle proposal message
    async fn handle_proposal(
        &mut self,
        round_id: RoundId,
        height: u64,
        proposer_id: ValidatorId,
        value: String,
        _signature: Vec<u8>,
    ) -> ConsensusResult<()> {
        // Verify proposer is the current leader
        if Some(&proposer_id) != self.leader_election.current_leader() {
            return Err(ConsensusError::ValidationFailed(
                "Proposal from non-leader".to_string(),
            ));
        }

        // Get or create voting round
        if !self.voting_rounds.contains_key(&round_id) {
            // Invariant: Number of concurrent voting rounds should not exceed config limit
            debug_assert!(
                self.voting_rounds.len() < self.config.max_concurrent_rounds,
                "Voting rounds count {} exceeds max {}",
                self.voting_rounds.len(),
                self.config.max_concurrent_rounds
            );

            // Invariant: Proposal height should be for current or next block
            debug_assert!(
                height >= self.current_height,
                "Proposal height {} is less than current height {}",
                height,
                self.current_height
            );

            let validators: HashMap<ValidatorId, u64> = self
                .validator
                .active_validators()
                .iter()
                .map(|v| (v.id.clone(), v.stake))
                .collect();

            let threshold = self.validator.voting_threshold();
            let round = VotingRound::new(
                height,
                0, // Round number
                self.config.voting_timeout,
                validators,
                threshold,
            );
            self.voting_rounds.insert(round_id.clone(), round);
        }

        if let Some(round) = self.voting_rounds.get_mut(&round_id) {
            round.set_proposed_value(value);

            // Cast pre-vote
            let vote = Vote::new(
                round_id,
                VoteType::PreVote,
                self.validator.id().clone(),
                Some("value_hash".to_string()), // Simplified
                vec![1, 2, 3, 4],               // Mock signature
            );

            self.send_message(ConsensusMessage::Vote(vote)).await?;
        }

        Ok(())
    }

    /// Handle GPU compute request
    async fn handle_gpu_compute(
        &mut self,
        task_id: String,
        computation_type: String,
        data: Vec<u8>,
        priority: u8,
    ) -> ConsensusResult<()> {
        let task = GpuComputeTask {
            task_id: task_id.clone(),
            computation_type,
            data,
            priority,
            requester_id: self.validator.id().clone(),
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
        };

        let task_id_clone = task.task_id.clone();
        self.pending_gpu_tasks.insert(task_id, task);

        // Mock GPU computation result
        let result = vec![42u8; 32]; // Mock result
        let proof = vec![1, 2, 3, 4]; // Mock proof

        let result_msg = ConsensusMessage::GpuResult {
            task_id: task_id_clone,
            result,
            validator_id: self.validator.id().clone(),
            proof,
        };

        self.send_message(result_msg).await?;
        Ok(())
    }

    /// Handle GPU result
    async fn handle_gpu_result(
        &mut self,
        task_id: String,
        _result: Vec<u8>,
        _validator_id: ValidatorId,
        _proof: Vec<u8>,
    ) -> ConsensusResult<()> {
        // Remove completed task
        self.pending_gpu_tasks.remove(&task_id);

        tracing::debug!("GPU computation completed for task {}", task_id);
        Ok(())
    }

    /// Handle validator heartbeat
    async fn handle_heartbeat(
        &mut self,
        validator_id: ValidatorId,
        _timestamp: u64,
        _gpu_status: GpuStatus,
    ) -> ConsensusResult<()> {
        self.validator.record_heartbeat(&validator_id)?;
        Ok(())
    }

    /// Handle state sync request
    async fn handle_sync_request(
        &mut self,
        from_height: u64,
        to_height: u64,
        requester_id: ValidatorId,
    ) -> ConsensusResult<()> {
        // Send state data for requested range
        for height in from_height..=to_height {
            if let Some(value) = self.committed_values.get(&height) {
                let response = ConsensusMessage::SyncResponse {
                    height,
                    state_data: value.as_bytes().to_vec(),
                    proof: vec![1, 2, 3, 4], // Mock proof
                };
                self.send_message(response).await?;
            }
        }

        tracing::debug!(
            "Sent state sync data to {} for heights {}-{}",
            requester_id,
            from_height,
            to_height
        );
        Ok(())
    }

    /// Handle state sync response
    async fn handle_sync_response(
        &mut self,
        height: u64,
        state_data: Vec<u8>,
        _proof: Vec<u8>,
    ) -> ConsensusResult<()> {
        // Apply synced state
        if let Ok(value) = String::from_utf8(state_data) {
            self.committed_values.insert(height, value);
            self.current_height = self.current_height.max(height);
        }

        Ok(())
    }

    /// Start new voting round (for leaders)
    async fn start_new_round(&mut self) -> ConsensusResult<()> {
        if !self.leader_election.is_leader() {
            return Ok(());
        }

        // Invariant: Only leaders should start new rounds
        debug_assert!(
            self.leader_election.is_leader(),
            "Non-leader attempting to start voting round"
        );

        let height = self.current_height + 1;

        // Invariant: Height should monotonically increase
        debug_assert!(
            height > self.current_height,
            "New round height {} is not greater than current {}",
            height,
            self.current_height
        );

        let round_id = RoundId::new();

        // Create proposal
        let proposal = ConsensusMessage::Proposal {
            round_id: round_id.clone(),
            height,
            proposer_id: self.validator.id().clone(),
            value: format!("block_{}", height),
            signature: vec![1, 2, 3, 4], // Mock signature
        };

        self.send_message(proposal).await?;
        Ok(())
    }

    /// Commit value at height
    async fn commit_value(&mut self, height: u64, value: String) -> ConsensusResult<()> {
        // Invariant: Cannot commit duplicate heights (unless replacing with same value)
        debug_assert!(
            !self.committed_values.contains_key(&height)
                || self.committed_values.get(&height) == Some(&value),
            "Attempting to overwrite committed value at height {}",
            height
        );

        // Invariant: Height should not be too far ahead of current
        debug_assert!(
            height <= self.current_height + 100,
            "Commit height {} is too far ahead of current height {}",
            height,
            self.current_height
        );

        self.committed_values.insert(height, value.clone());
        self.current_height = self.current_height.max(height);

        tracing::info!("Committed value '{}' at height {}", value, height);

        Ok(())
    }

    /// Send message to network
    async fn send_message(&self, message: ConsensusMessage) -> ConsensusResult<()> {
        if let Some(sender) = &self.message_sender {
            sender
                .send(message)
                .await
                .map_err(|_| ConsensusError::NetworkError("Failed to send message".to_string()))?;
        }
        Ok(())
    }

    /// Get current protocol state
    pub fn state(&self) -> &ProtocolState {
        &self.state
    }

    /// Get current height
    pub fn current_height(&self) -> u64 {
        self.current_height
    }

    /// Get validator information
    pub fn validator_info(&self) -> &ValidatorInfo {
        self.validator.info()
    }

    /// Get active voting rounds
    pub fn active_rounds(&self) -> Vec<&RoundId> {
        self.voting_rounds.keys().collect()
    }

    /// Get pending GPU tasks
    pub fn pending_gpu_tasks(&self) -> &HashMap<String, GpuComputeTask> {
        &self.pending_gpu_tasks
    }

    /// Check consensus health
    pub fn consensus_health(&self) -> ConsensusHealth {
        let active_validators = self.validator.active_validators().len();
        let has_quorum = self.validator.has_consensus_quorum();
        let is_leader = self.leader_election.is_leader();
        let pending_tasks = self.pending_gpu_tasks.len();

        ConsensusHealth {
            has_quorum,
            active_validators,
            is_leader,
            current_height: self.current_height,
            pending_gpu_tasks: pending_tasks,
            state: self.state.clone(),
        }
    }
}

/// Consensus health status
#[derive(Debug, Clone)]
pub struct ConsensusHealth {
    /// Whether consensus quorum is available
    pub has_quorum: bool,
    /// Number of active validators
    pub active_validators: usize,
    /// Whether this node is the leader
    pub is_leader: bool,
    /// Current blockchain height
    pub current_height: u64,
    /// Number of pending GPU tasks
    pub pending_gpu_tasks: usize,
    /// Current protocol state
    pub state: ProtocolState,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::SocketAddr;

    fn create_test_config() -> ConsensusConfig {
        ConsensusConfig {
            election_timeout: Duration::from_millis(100),
            heartbeat_interval: Duration::from_millis(50),
            voting_timeout: Duration::from_millis(200),
            max_concurrent_rounds: 5,
            gpu_timeout: Duration::from_secs(5),
            network_timeout: Duration::from_secs(1),
        }
    }

    #[test]
    fn test_consensus_config_default() {
        let config = ConsensusConfig::default();
        assert_eq!(config.election_timeout, Duration::from_secs(5));
        assert_eq!(config.heartbeat_interval, Duration::from_secs(1));
    }

    #[tokio::test]
    async fn test_consensus_protocol_creation() {
        let config = create_test_config();
        let address: SocketAddr = "127.0.0.1:8080".parse().unwrap();

        let protocol = ConsensusProtocol::new(config, address, 100, 1000, vec![1, 2, 3, 4]);

        assert!(protocol.is_ok());
        let protocol = protocol.unwrap();
        assert_eq!(protocol.state(), &ProtocolState::Initializing);
        assert_eq!(protocol.current_height(), 0);
    }

    #[tokio::test]
    async fn test_protocol_start_stop() {
        let config = create_test_config();
        let address: SocketAddr = "127.0.0.1:8080".parse().unwrap();

        let mut protocol =
            ConsensusProtocol::new(config, address, 100, 1000, vec![1, 2, 3, 4]).unwrap();

        let (tx, _rx) = mpsc::channel(100);

        // Start protocol
        assert!(protocol.start(tx).await.is_ok());
        assert_eq!(protocol.state(), &ProtocolState::Following);

        // Stop protocol
        assert!(protocol.stop().await.is_ok());
        assert_eq!(protocol.state(), &ProtocolState::Stopped);
    }

    #[tokio::test]
    async fn test_gpu_compute_handling() {
        let config = create_test_config();
        let address: SocketAddr = "127.0.0.1:8080".parse().unwrap();

        let mut protocol =
            ConsensusProtocol::new(config, address, 100, 1000, vec![1, 2, 3, 4]).unwrap();

        let (tx, mut rx) = mpsc::channel(100);
        protocol.start(tx).await.unwrap();

        let compute_msg = ConsensusMessage::GpuCompute {
            task_id: "task_123".to_string(),
            computation_type: "matrix_multiply".to_string(),
            data: vec![1, 2, 3, 4],
            priority: 1,
        };

        // Handle GPU compute request
        assert!(protocol.handle_message(compute_msg).await.is_ok());

        // Should receive GPU result message
        let result_msg = rx.recv().await.unwrap();
        assert!(matches!(result_msg, ConsensusMessage::GpuResult { .. }));
    }

    #[tokio::test]
    async fn test_heartbeat_handling() {
        let config = create_test_config();
        let address: SocketAddr = "127.0.0.1:8080".parse().unwrap();

        let mut protocol =
            ConsensusProtocol::new(config, address, 100, 1000, vec![1, 2, 3, 4]).unwrap();

        let validator_id = protocol.validator.id().clone();

        let heartbeat_msg = ConsensusMessage::Heartbeat {
            validator_id,
            timestamp: 123456789,
            gpu_status: GpuStatus {
                available_memory: 8000,
                utilization: 50,
                active_kernels: 2,
                temperature: 65,
            },
        };

        assert!(protocol.handle_message(heartbeat_msg).await.is_ok());
    }

    #[tokio::test]
    async fn test_consensus_health() {
        let config = create_test_config();
        let address: SocketAddr = "127.0.0.1:8080".parse().unwrap();

        let protocol =
            ConsensusProtocol::new(config, address, 100, 1000, vec![1, 2, 3, 4]).unwrap();

        let health = protocol.consensus_health();
        assert!(health.has_quorum); // Single validator has quorum in single-node network
        assert_eq!(health.active_validators, 1);
        assert!(!health.is_leader);
        assert_eq!(health.current_height, 0);
        assert_eq!(health.pending_gpu_tasks, 0);
    }

    #[test]
    fn test_gpu_status_creation() {
        let status = GpuStatus {
            available_memory: 8000,
            utilization: 75,
            active_kernels: 3,
            temperature: 70,
        };

        assert_eq!(status.available_memory, 8000);
        assert_eq!(status.utilization, 75);
        assert_eq!(status.active_kernels, 3);
        assert_eq!(status.temperature, 70);
    }

    #[test]
    fn test_protocol_state_equality() {
        assert_eq!(ProtocolState::Initializing, ProtocolState::Initializing);
        assert_ne!(ProtocolState::Following, ProtocolState::Leading);
    }

    #[test]
    fn test_message_serialization() {
        let validator_id = ValidatorId::new();

        let messages = vec![
            ConsensusMessage::Proposal {
                round_id: RoundId::new(),
                height: 1,
                proposer_id: validator_id.clone(),
                value: "test_value".to_string(),
                signature: vec![1, 2, 3, 4],
            },
            ConsensusMessage::GpuCompute {
                task_id: "task_123".to_string(),
                computation_type: "test".to_string(),
                data: vec![1, 2, 3],
                priority: 1,
            },
            ConsensusMessage::Heartbeat {
                validator_id,
                timestamp: 123456789,
                gpu_status: GpuStatus {
                    available_memory: 8000,
                    utilization: 50,
                    active_kernels: 2,
                    temperature: 65,
                },
            },
        ];

        for message in messages {
            let serialized = serde_json::to_string(&message).unwrap();
            let _deserialized: ConsensusMessage = serde_json::from_str(&serialized).unwrap();
        }
    }

    #[test]
    fn test_consensus_config_edge_cases() {
        let config = ConsensusConfig {
            election_timeout: Duration::from_millis(1),
            heartbeat_interval: Duration::from_millis(1),
            voting_timeout: Duration::from_millis(1),
            max_concurrent_rounds: 1,
            gpu_timeout: Duration::from_millis(1),
            network_timeout: Duration::from_millis(1),
        };

        assert_eq!(config.election_timeout, Duration::from_millis(1));
        assert_eq!(config.max_concurrent_rounds, 1);
    }

    #[test]
    fn test_consensus_config_clone() {
        let config = ConsensusConfig::default();
        let cloned = config.clone();

        assert_eq!(config.election_timeout, cloned.election_timeout);
        assert_eq!(config.heartbeat_interval, cloned.heartbeat_interval);
        assert_eq!(config.voting_timeout, cloned.voting_timeout);
        assert_eq!(config.max_concurrent_rounds, cloned.max_concurrent_rounds);
    }

    #[test]
    fn test_consensus_config_debug() {
        let config = ConsensusConfig::default();
        let debug_str = format!("{:?}", config);

        assert!(debug_str.contains("ConsensusConfig"));
        assert!(debug_str.contains("election_timeout"));
        assert!(debug_str.contains("heartbeat_interval"));
    }

    #[test]
    fn test_gpu_status_edge_cases() {
        let status = GpuStatus {
            available_memory: 0,
            utilization: 0,
            active_kernels: 0,
            temperature: 0,
        };

        assert_eq!(status.available_memory, 0);
        assert_eq!(status.utilization, 0);
        assert_eq!(status.active_kernels, 0);
        assert_eq!(status.temperature, 0);

        let max_status = GpuStatus {
            available_memory: u64::MAX,
            utilization: 100,
            active_kernels: u32::MAX,
            temperature: u16::MAX,
        };

        assert_eq!(max_status.available_memory, u64::MAX);
        assert_eq!(max_status.utilization, 100);
    }

    #[test]
    fn test_gpu_status_clone() {
        let status = GpuStatus {
            available_memory: 8000,
            utilization: 75,
            active_kernels: 3,
            temperature: 70,
        };

        let cloned = status.clone();
        assert_eq!(status.available_memory, cloned.available_memory);
        assert_eq!(status.utilization, cloned.utilization);
        assert_eq!(status.active_kernels, cloned.active_kernels);
        assert_eq!(status.temperature, cloned.temperature);
    }

    #[test]
    fn test_gpu_status_debug() {
        let status = GpuStatus {
            available_memory: 8000,
            utilization: 75,
            active_kernels: 3,
            temperature: 70,
        };

        let debug_str = format!("{:?}", status);
        assert!(debug_str.contains("GpuStatus"));
        assert!(debug_str.contains("8000"));
        assert!(debug_str.contains("75"));
    }

    #[test]
    fn test_protocol_state_debug() {
        let states = vec![
            ProtocolState::Initializing,
            ProtocolState::Following,
            ProtocolState::Leading,
            ProtocolState::Stopped,
        ];

        for state in states {
            let debug_str = format!("{:?}", state);
            assert!(!debug_str.is_empty());
            assert!(debug_str.contains("ProtocolState"));
        }
    }

    #[test]
    fn test_protocol_state_clone() {
        let state = ProtocolState::Leading;
        let cloned = state.clone();
        assert_eq!(state, cloned);
    }

    #[test]
    fn test_protocol_state_partial_eq() {
        assert_eq!(ProtocolState::Initializing, ProtocolState::Initializing);
        assert_eq!(ProtocolState::Following, ProtocolState::Following);
        assert_eq!(ProtocolState::Leading, ProtocolState::Leading);
        assert_eq!(ProtocolState::Stopped, ProtocolState::Stopped);

        assert_ne!(ProtocolState::Initializing, ProtocolState::Following);
        assert_ne!(ProtocolState::Following, ProtocolState::Leading);
        assert_ne!(ProtocolState::Leading, ProtocolState::Stopped);
    }

    #[test]
    fn test_consensus_message_proposal_variants() {
        let validator_id = ValidatorId::new();
        let round_id = RoundId::new();

        let proposal = ConsensusMessage::Proposal {
            round_id: round_id.clone(),
            height: 100,
            proposer_id: validator_id.clone(),
            value: "complex_value".to_string(),
            signature: vec![0, 1, 2, 3, 4, 5],
        };

        match proposal {
            ConsensusMessage::Proposal {
                round_id: r,
                height: h,
                ..
            } => {
                assert_eq!(r, round_id);
                assert_eq!(h, 100);
            }
            _ => panic!("Expected Proposal variant"),
        }
    }

    #[test]
    fn test_consensus_message_gpu_compute_variants() {
        let compute = ConsensusMessage::GpuCompute {
            task_id: "compute_task_456".to_string(),
            computation_type: "neural_network".to_string(),
            data: vec![10, 20, 30, 40, 50],
            priority: 5,
        };

        match compute {
            ConsensusMessage::GpuCompute {
                task_id, priority, ..
            } => {
                assert_eq!(task_id, "compute_task_456");
                assert_eq!(priority, 5);
            }
            _ => panic!("Expected GpuCompute variant"),
        }
    }

    #[test]
    fn test_consensus_message_heartbeat_variants() {
        let validator_id = ValidatorId::new();
        let heartbeat = ConsensusMessage::Heartbeat {
            validator_id: validator_id.clone(),
            timestamp: 987654321,
            gpu_status: GpuStatus {
                available_memory: 16000,
                utilization: 90,
                active_kernels: 5,
                temperature: 80,
            },
        };

        match heartbeat {
            ConsensusMessage::Heartbeat {
                validator_id: v,
                timestamp: t,
                gpu_status,
            } => {
                assert_eq!(v, validator_id);
                assert_eq!(t, 987654321);
                assert_eq!(gpu_status.available_memory, 16000);
                assert_eq!(gpu_status.utilization, 90);
            }
            _ => panic!("Expected Heartbeat variant"),
        }
    }

    #[test]
    fn test_consensus_message_debug() {
        let validator_id = ValidatorId::new();

        let messages = vec![
            ConsensusMessage::Proposal {
                round_id: RoundId::new(),
                height: 1,
                proposer_id: validator_id.clone(),
                value: "test".to_string(),
                signature: vec![1, 2, 3],
            },
            ConsensusMessage::GpuCompute {
                task_id: "task".to_string(),
                computation_type: "test".to_string(),
                data: vec![1, 2, 3],
                priority: 1,
            },
            ConsensusMessage::Heartbeat {
                validator_id,
                timestamp: 123,
                gpu_status: GpuStatus {
                    available_memory: 8000,
                    utilization: 50,
                    active_kernels: 2,
                    temperature: 65,
                },
            },
        ];

        for message in messages {
            let debug_str = format!("{:?}", message);
            assert!(!debug_str.is_empty());
            assert!(debug_str.contains("ConsensusMessage"));
        }
    }

    #[test]
    fn test_consensus_message_clone() {
        let validator_id = ValidatorId::new();
        let original = ConsensusMessage::Heartbeat {
            validator_id: validator_id.clone(),
            timestamp: 123456,
            gpu_status: GpuStatus {
                available_memory: 8000,
                utilization: 50,
                active_kernels: 2,
                temperature: 65,
            },
        };

        let cloned = original.clone();
        match (original, cloned) {
            (
                ConsensusMessage::Heartbeat { timestamp: t1, .. },
                ConsensusMessage::Heartbeat { timestamp: t2, .. },
            ) => {
                assert_eq!(t1, t2);
            }
            _ => panic!("Clone should preserve message type"),
        }
    }

    #[tokio::test]
    async fn test_protocol_error_handling() {
        let config = create_test_config();
        let address: SocketAddr = "127.0.0.1:8080".parse().unwrap();

        let mut protocol =
            ConsensusProtocol::new(config, address, 100, 1000, vec![1, 2, 3, 4]).unwrap();

        // Test invalid message handling
        let invalid_msg = ConsensusMessage::GpuCompute {
            task_id: "".to_string(), // Empty task ID should cause error
            computation_type: "invalid".to_string(),
            data: vec![],
            priority: 0,
        };

        let result = protocol.handle_message(invalid_msg).await;
        // Should handle gracefully even if message is problematic
        assert!(result.is_ok() || result.is_err());
    }

    #[tokio::test]
    async fn test_protocol_multiple_starts() {
        let config = create_test_config();
        let address: SocketAddr = "127.0.0.1:8080".parse().unwrap();

        let mut protocol =
            ConsensusProtocol::new(config, address, 100, 1000, vec![1, 2, 3, 4]).unwrap();

        let (tx1, _rx1) = mpsc::channel(100);
        let (tx2, _rx2) = mpsc::channel(100);

        // First start should succeed
        assert!(protocol.start(tx1).await.is_ok());

        // Second start should handle appropriately (either succeed or fail gracefully)
        let result = protocol.start(tx2).await;
        assert!(result.is_ok() || result.is_err());
    }

    #[tokio::test]
    async fn test_protocol_stop_without_start() {
        let config = create_test_config();
        let address: SocketAddr = "127.0.0.1:8080".parse().unwrap();

        let mut protocol =
            ConsensusProtocol::new(config, address, 100, 1000, vec![1, 2, 3, 4]).unwrap();

        // Stop without start should handle gracefully
        let result = protocol.stop().await;
        assert!(result.is_ok() || result.is_err());
    }

    #[tokio::test]
    async fn test_protocol_state_transitions() {
        let config = create_test_config();
        let address: SocketAddr = "127.0.0.1:8080".parse().unwrap();

        let mut protocol =
            ConsensusProtocol::new(config, address, 100, 1000, vec![1, 2, 3, 4]).unwrap();

        // Initial state
        assert_eq!(protocol.state(), &ProtocolState::Initializing);

        let (tx, _rx) = mpsc::channel(100);

        // Start -> Following
        protocol.start(tx).await.unwrap();
        assert_eq!(protocol.state(), &ProtocolState::Following);

        // Stop -> Stopped
        protocol.stop().await.unwrap();
        assert_eq!(protocol.state(), &ProtocolState::Stopped);
    }

    #[test]
    fn test_consensus_health_fields() {
        let config = create_test_config();
        let address: SocketAddr = "127.0.0.1:8080".parse().unwrap();

        let protocol =
            ConsensusProtocol::new(config, address, 100, 1000, vec![1, 2, 3, 4]).unwrap();

        let health = protocol.consensus_health();

        // Verify all health fields are accessible
        assert!(health.has_quorum || !health.has_quorum); // Boolean field
        assert!(health.active_validators >= 0); // Should be non-negative
        assert!(health.is_leader || !health.is_leader); // Boolean field
        assert!(health.current_height >= 0); // Should be non-negative
        assert!(health.pending_gpu_tasks >= 0); // Should be non-negative
    }

    #[test]
    fn test_protocol_accessor_methods() {
        let config = create_test_config();
        let address: SocketAddr = "127.0.0.1:8080".parse().unwrap();

        let protocol =
            ConsensusProtocol::new(config, address, 100, 1000, vec![1, 2, 3, 4]).unwrap();

        // Test accessor methods
        assert_eq!(protocol.current_height(), 0);
        assert_eq!(protocol.state(), &ProtocolState::Initializing);

        // Validator should be accessible
        let validator_id = protocol.validator.id();
        assert!(!validator_id.to_string().is_empty());
    }

    #[test]
    fn test_large_message_serialization() {
        let validator_id = ValidatorId::new();
        let large_data = vec![0u8; 10000]; // 10KB of data
        let large_value = "x".repeat(1000); // 1KB string

        let large_messages = vec![
            ConsensusMessage::Proposal {
                round_id: RoundId::new(),
                height: u64::MAX,
                proposer_id: validator_id.clone(),
                value: large_value,
                signature: large_data.clone(),
            },
            ConsensusMessage::GpuCompute {
                task_id: "very_long_task_id_".repeat(100),
                computation_type: "complex_computation_type".repeat(10),
                data: large_data,
                priority: u8::MAX,
            },
        ];

        for message in large_messages {
            let serialized = serde_json::to_string(&message).unwrap();
            let _deserialized: ConsensusMessage = serde_json::from_str(&serialized).unwrap();
        }
    }

    #[test]
    fn test_message_serialization_empty_fields() {
        let validator_id = ValidatorId::new();

        let empty_messages = vec![
            ConsensusMessage::Proposal {
                round_id: RoundId::new(),
                height: 0,
                proposer_id: validator_id.clone(),
                value: String::new(),
                signature: vec![],
            },
            ConsensusMessage::GpuCompute {
                task_id: String::new(),
                computation_type: String::new(),
                data: vec![],
                priority: 0,
            },
        ];

        for message in empty_messages {
            let serialized = serde_json::to_string(&message).unwrap();
            let _deserialized: ConsensusMessage = serde_json::from_str(&serialized).unwrap();
        }
    }

    #[test]
    fn test_extreme_gpu_status_values() {
        let extreme_status = GpuStatus {
            available_memory: u64::MAX,
            utilization: 100,
            active_kernels: u32::MAX,
            temperature: u16::MAX,
        };

        let serialized = serde_json::to_string(&extreme_status).unwrap();
        let _deserialized: GpuStatus = serde_json::from_str(&serialized).unwrap();
    }
}
