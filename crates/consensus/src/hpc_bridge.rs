//! HPC Channels integration for consensus leader election events.
//!
//! This module bridges consensus and leader election events to the hpc-channels
//! message bus, enabling real-time monitoring of cluster consensus state.
//!
//! # Channels Used
//!
//! - `hpc.consensus.proposal` - Consensus proposal events
//!
//! # Example
//!
//! ```rust,ignore
//! use stratoswarm_consensus::hpc_bridge::ConsensusChannelBridge;
//!
//! let bridge = ConsensusChannelBridge::new();
//!
//! // Publish election started event
//! bridge.publish_election_started(5, "validator-123");
//!
//! // Subscribe to consensus events
//! let mut rx = bridge.subscribe();
//! while let Ok(event) = rx.recv().await {
//!     println!("Consensus event: {:?}", event);
//! }
//! ```

use std::sync::Arc;
use tokio::sync::broadcast;

use crate::ValidatorId;

/// Consensus events published to hpc-channels.
#[derive(Clone, Debug)]
pub enum ConsensusEvent {
    /// Election has started for a new term.
    ElectionStarted {
        /// Election term number.
        term: u64,
        /// Candidate validator ID.
        candidate_id: String,
        /// Timestamp (epoch ms).
        timestamp_ms: u64,
    },
    /// Vote was received from a validator.
    VoteReceived {
        /// Election term.
        term: u64,
        /// Voter validator ID.
        voter_id: String,
        /// Whether vote was granted.
        vote_granted: bool,
        /// Timestamp (epoch ms).
        timestamp_ms: u64,
    },
    /// A new leader was elected.
    LeaderElected {
        /// Election term.
        term: u64,
        /// New leader validator ID.
        leader_id: String,
        /// Timestamp (epoch ms).
        timestamp_ms: u64,
    },
    /// Leader stepped down.
    LeaderSteppedDown {
        /// Previous term.
        term: u64,
        /// Previous leader validator ID.
        leader_id: String,
        /// Timestamp (epoch ms).
        timestamp_ms: u64,
    },
    /// Quorum was reached for an election.
    QuorumReached {
        /// Election term.
        term: u64,
        /// Number of votes received.
        votes_received: u32,
        /// Total validators.
        total_validators: u32,
        /// Timestamp (epoch ms).
        timestamp_ms: u64,
    },
}

/// Bridge between consensus events and hpc-channels.
pub struct ConsensusChannelBridge {
    /// Broadcast sender for consensus events.
    events_tx: broadcast::Sender<ConsensusEvent>,
}

impl ConsensusChannelBridge {
    /// Create a new consensus channel bridge.
    ///
    /// Registers channels with the hpc-channels global registry.
    pub fn new() -> Self {
        let events_tx = hpc_channels::broadcast::<ConsensusEvent>(
            hpc_channels::channels::CONSENSUS_PROPOSAL,
            512,
        );

        Self { events_tx }
    }

    fn now_ms() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0)
    }

    /// Publish an election started event.
    pub fn publish_election_started(&self, term: u64, candidate_id: &ValidatorId) {
        let _ = self.events_tx.send(ConsensusEvent::ElectionStarted {
            term,
            candidate_id: candidate_id.to_string(),
            timestamp_ms: Self::now_ms(),
        });
    }

    /// Publish a vote received event.
    pub fn publish_vote_received(&self, term: u64, voter_id: &ValidatorId, vote_granted: bool) {
        let _ = self.events_tx.send(ConsensusEvent::VoteReceived {
            term,
            voter_id: voter_id.to_string(),
            vote_granted,
            timestamp_ms: Self::now_ms(),
        });
    }

    /// Publish a leader elected event.
    pub fn publish_leader_elected(&self, term: u64, leader_id: &ValidatorId) {
        let _ = self.events_tx.send(ConsensusEvent::LeaderElected {
            term,
            leader_id: leader_id.to_string(),
            timestamp_ms: Self::now_ms(),
        });
    }

    /// Publish a leader stepped down event.
    pub fn publish_leader_stepped_down(&self, term: u64, leader_id: &ValidatorId) {
        let _ = self.events_tx.send(ConsensusEvent::LeaderSteppedDown {
            term,
            leader_id: leader_id.to_string(),
            timestamp_ms: Self::now_ms(),
        });
    }

    /// Publish a quorum reached event.
    pub fn publish_quorum_reached(&self, term: u64, votes_received: u32, total_validators: u32) {
        let _ = self.events_tx.send(ConsensusEvent::QuorumReached {
            term,
            votes_received,
            total_validators,
            timestamp_ms: Self::now_ms(),
        });
    }

    /// Subscribe to consensus events.
    pub fn subscribe(&self) -> broadcast::Receiver<ConsensusEvent> {
        self.events_tx.subscribe()
    }
}

impl Default for ConsensusChannelBridge {
    fn default() -> Self {
        Self::new()
    }
}

/// Shared channel bridge type.
pub type SharedConsensusChannelBridge = Arc<ConsensusChannelBridge>;

/// Create a new shared channel bridge.
#[must_use]
pub fn shared_channel_bridge() -> SharedConsensusChannelBridge {
    Arc::new(ConsensusChannelBridge::new())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bridge_creation() {
        let bridge = ConsensusChannelBridge::new();
        assert!(hpc_channels::exists(
            hpc_channels::channels::CONSENSUS_PROPOSAL
        ));
        let _ = bridge;
    }

    #[tokio::test]
    async fn test_election_started_event() {
        let bridge = ConsensusChannelBridge::new();
        let mut rx = bridge.subscribe();

        let validator_id = ValidatorId::new();
        bridge.publish_election_started(1, &validator_id);

        let event = rx.recv().await.expect("Should receive event");
        match event {
            ConsensusEvent::ElectionStarted { term, .. } => {
                assert_eq!(term, 1);
            }
            _ => panic!("Expected ElectionStarted event"),
        }
    }

    #[tokio::test]
    async fn test_leader_elected_event() {
        let bridge = ConsensusChannelBridge::new();
        let mut rx = bridge.subscribe();

        let leader_id = ValidatorId::new();
        bridge.publish_leader_elected(2, &leader_id);

        let event = rx.recv().await.expect("Should receive event");
        match event {
            ConsensusEvent::LeaderElected { term, .. } => {
                assert_eq!(term, 2);
            }
            _ => panic!("Expected LeaderElected event"),
        }
    }

    #[tokio::test]
    async fn test_quorum_reached_event() {
        let bridge = ConsensusChannelBridge::new();
        let mut rx = bridge.subscribe();

        bridge.publish_quorum_reached(3, 5, 7);

        let event = rx.recv().await.expect("Should receive event");
        match event {
            ConsensusEvent::QuorumReached {
                term,
                votes_received,
                total_validators,
                ..
            } => {
                assert_eq!(term, 3);
                assert_eq!(votes_received, 5);
                assert_eq!(total_validators, 7);
            }
            _ => panic!("Expected QuorumReached event"),
        }
    }
}
