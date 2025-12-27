//! Leader election for consensus protocol

use crate::error::{ConsensusError, ConsensusResult};
use crate::validator::{ValidatorId, ValidatorInfo, ValidatorStatus};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

#[cfg(feature = "hpc-channels")]
use crate::hpc_bridge::{SharedConsensusChannelBridge, shared_channel_bridge};

/// Leader election state
#[derive(Debug, Clone, PartialEq)]
pub enum LeaderState {
    /// Following another leader
    Follower { leader_id: Option<ValidatorId> },
    /// Candidate for leadership
    Candidate { term: u64, votes_received: u32 },
    /// Current leader
    Leader { term: u64, last_heartbeat: u64 },
}

/// Leader election manager
pub struct LeaderElection {
    /// This validator's ID
    validator_id: ValidatorId,
    /// Current state
    state: LeaderState,
    /// Current term
    current_term: u64,
    /// Last vote cast in current term
    voted_for: Option<ValidatorId>,
    /// Election timeout duration
    election_timeout: Duration,
    /// Heartbeat interval for leaders
    heartbeat_interval: Duration,
    /// Last time a heartbeat was received
    last_heartbeat_received: SystemTime,
    /// Random election timeout offset
    election_timeout_offset: Duration,
    /// HPC-Channels event bridge for publishing consensus events
    #[cfg(feature = "hpc-channels")]
    event_bridge: SharedConsensusChannelBridge,
}

/// Election message types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ElectionMessage {
    /// Request vote for candidacy
    RequestVote {
        term: u64,
        candidate_id: ValidatorId,
        last_log_index: u64,
        last_log_term: u64,
    },
    /// Vote response
    VoteResponse {
        term: u64,
        vote_granted: bool,
        voter_id: ValidatorId,
    },
    /// Heartbeat from leader
    Heartbeat {
        term: u64,
        leader_id: ValidatorId,
        prev_log_index: u64,
        prev_log_term: u64,
        entries: Vec<String>, // Simplified log entries
        leader_commit: u64,
    },
    /// Heartbeat response
    HeartbeatResponse {
        term: u64,
        success: bool,
        follower_id: ValidatorId,
        match_index: u64,
    },
}

impl LeaderElection {
    /// Create new leader election manager
    pub fn new(
        validator_id: ValidatorId,
        election_timeout: Duration,
        heartbeat_interval: Duration,
    ) -> Self {
        Self {
            validator_id: validator_id.clone(),
            state: LeaderState::Follower { leader_id: None },
            current_term: 0,
            voted_for: None,
            election_timeout,
            heartbeat_interval,
            last_heartbeat_received: SystemTime::now(),
            election_timeout_offset: Duration::from_millis(
                (validator_id.as_uuid().as_u128() % 1000) as u64,
            ),
            #[cfg(feature = "hpc-channels")]
            event_bridge: shared_channel_bridge(),
        }
    }

    /// Get the event bridge for subscribing to consensus events
    #[cfg(feature = "hpc-channels")]
    pub fn event_bridge(&self) -> &SharedConsensusChannelBridge {
        &self.event_bridge
    }

    /// Get current state
    pub fn state(&self) -> &LeaderState {
        &self.state
    }

    /// Get current term
    pub fn term(&self) -> u64 {
        self.current_term
    }

    /// Check if this validator is the leader
    pub fn is_leader(&self) -> bool {
        matches!(self.state, LeaderState::Leader { .. })
    }

    /// Check if this validator is a candidate
    pub fn is_candidate(&self) -> bool {
        matches!(self.state, LeaderState::Candidate { .. })
    }

    /// Check if this validator is a follower
    pub fn is_follower(&self) -> bool {
        matches!(self.state, LeaderState::Follower { .. })
    }

    /// Get current leader ID if known
    pub fn current_leader(&self) -> Option<&ValidatorId> {
        match &self.state {
            LeaderState::Follower { leader_id } => leader_id.as_ref(),
            LeaderState::Leader { .. } => Some(&self.validator_id),
            LeaderState::Candidate { .. } => None,
        }
    }

    /// Check if election timeout has occurred
    pub fn is_election_timeout(&self) -> bool {
        let timeout = self.election_timeout + self.election_timeout_offset;
        SystemTime::now()
            .duration_since(self.last_heartbeat_received)
            .unwrap_or(Duration::ZERO)
            > timeout
    }

    /// Start election as candidate
    #[must_use = "ignoring the Result may hide election start failures"]
    pub fn start_election(&mut self) -> ConsensusResult<ElectionMessage> {
        self.current_term += 1;
        self.voted_for = Some(self.validator_id.clone());
        self.state = LeaderState::Candidate {
            term: self.current_term,
            votes_received: 1, // Vote for self
        };

        tracing::info!(
            "Starting election for term {} as candidate {}",
            self.current_term,
            self.validator_id
        );

        // Publish election started event
        #[cfg(feature = "hpc-channels")]
        self.event_bridge.publish_election_started(self.current_term, &self.validator_id);

        Ok(ElectionMessage::RequestVote {
            term: self.current_term,
            candidate_id: self.validator_id.clone(),
            last_log_index: 0, // Simplified
            last_log_term: 0,  // Simplified
        })
    }

    /// Handle incoming election message
    #[must_use = "ignoring the Result may hide message handling errors"]
    pub fn handle_message(
        &mut self,
        message: ElectionMessage,
        validators: &HashMap<ValidatorId, ValidatorInfo>,
    ) -> ConsensusResult<Option<ElectionMessage>> {
        match message {
            ElectionMessage::RequestVote {
                term,
                candidate_id,
                last_log_index: _,
                last_log_term: _,
            } => self.handle_vote_request(term, candidate_id),

            ElectionMessage::VoteResponse {
                term,
                vote_granted,
                voter_id,
            } => self.handle_vote_response(term, vote_granted, voter_id, validators),

            ElectionMessage::Heartbeat {
                term,
                leader_id,
                prev_log_index: _,
                prev_log_term: _,
                entries: _,
                leader_commit: _,
            } => self.handle_heartbeat(term, leader_id),

            ElectionMessage::HeartbeatResponse {
                term,
                success: _,
                follower_id: _,
                match_index: _,
            } => self.handle_heartbeat_response(term),
        }
    }

    /// Handle vote request
    fn handle_vote_request(
        &mut self,
        term: u64,
        candidate_id: ValidatorId,
    ) -> ConsensusResult<Option<ElectionMessage>> {
        // If term is older, reject
        if term < self.current_term {
            return Ok(Some(ElectionMessage::VoteResponse {
                term: self.current_term,
                vote_granted: false,
                voter_id: self.validator_id.clone(),
            }));
        }

        // If term is newer, become follower
        if term > self.current_term {
            self.current_term = term;
            self.voted_for = None;
            self.state = LeaderState::Follower { leader_id: None };
        }

        // Grant vote if haven't voted yet or already voted for this candidate
        let vote_granted =
            self.voted_for.is_none() || self.voted_for.as_ref() == Some(&candidate_id);

        if vote_granted {
            self.voted_for = Some(candidate_id);
        }

        Ok(Some(ElectionMessage::VoteResponse {
            term: self.current_term,
            vote_granted,
            voter_id: self.validator_id.clone(),
        }))
    }

    /// Handle vote response
    fn handle_vote_response(
        &mut self,
        term: u64,
        vote_granted: bool,
        _voter_id: ValidatorId,
        validators: &HashMap<ValidatorId, ValidatorInfo>,
    ) -> ConsensusResult<Option<ElectionMessage>> {
        // Only candidates care about vote responses
        if let LeaderState::Candidate {
            term: current_term,
            votes_received,
        } = &mut self.state
        {
            if term == *current_term && vote_granted {
                *votes_received += 1;

                // Check if won election (majority of validators)
                let total_validators = validators
                    .values()
                    .filter(|v| v.status == ValidatorStatus::Active)
                    .count();
                let majority = (total_validators / 2) + 1;

                if *votes_received >= majority as u32 {
                    self.become_leader()?;
                    return Ok(Some(self.create_heartbeat()));
                }
            }
        }

        Ok(None)
    }

    /// Handle heartbeat from leader
    fn handle_heartbeat(
        &mut self,
        term: u64,
        leader_id: ValidatorId,
    ) -> ConsensusResult<Option<ElectionMessage>> {
        // If term is newer, update and become follower
        if term >= self.current_term {
            self.current_term = term;
            self.voted_for = None;
            self.state = LeaderState::Follower {
                leader_id: Some(leader_id.clone()),
            };
            self.last_heartbeat_received = SystemTime::now();

            return Ok(Some(ElectionMessage::HeartbeatResponse {
                term: self.current_term,
                success: true,
                follower_id: self.validator_id.clone(),
                match_index: 0, // Simplified
            }));
        }

        // If term is older, reject
        Ok(Some(ElectionMessage::HeartbeatResponse {
            term: self.current_term,
            success: false,
            follower_id: self.validator_id.clone(),
            match_index: 0,
        }))
    }

    /// Handle heartbeat response
    fn handle_heartbeat_response(&mut self, term: u64) -> ConsensusResult<Option<ElectionMessage>> {
        // If response has newer term, step down
        if term > self.current_term {
            self.current_term = term;
            self.voted_for = None;
            self.state = LeaderState::Follower { leader_id: None };
        }

        Ok(None)
    }

    /// Become leader
    fn become_leader(&mut self) -> ConsensusResult<()> {
        self.state = LeaderState::Leader {
            term: self.current_term,
            last_heartbeat: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
        };

        tracing::info!(
            "Became leader for term {} as validator {}",
            self.current_term,
            self.validator_id
        );

        // Publish leader elected event
        #[cfg(feature = "hpc-channels")]
        self.event_bridge.publish_leader_elected(self.current_term, &self.validator_id);

        Ok(())
    }

    /// Create heartbeat message (for leaders)
    pub fn create_heartbeat(&self) -> ElectionMessage {
        ElectionMessage::Heartbeat {
            term: self.current_term,
            leader_id: self.validator_id.clone(),
            prev_log_index: 0, // Simplified
            prev_log_term: 0,  // Simplified
            entries: vec![],   // No new entries
            leader_commit: 0,  // Simplified
        }
    }

    /// Check if should send heartbeat (for leaders)
    pub fn should_send_heartbeat(&self) -> bool {
        if let LeaderState::Leader { last_heartbeat, .. } = &self.state {
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64;

            // last_heartbeat is already in milliseconds
            (now - last_heartbeat) >= self.heartbeat_interval.as_millis() as u64
        } else {
            false
        }
    }

    /// Update last heartbeat time (for leaders)
    pub fn update_heartbeat_time(&mut self) -> ConsensusResult<()> {
        if let LeaderState::Leader { last_heartbeat, .. } = &mut self.state {
            *last_heartbeat = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64;
        }
        Ok(())
    }

    /// Step down from leadership
    pub fn step_down(&mut self) {
        // Publish leader stepped down event before changing state
        #[cfg(feature = "hpc-channels")]
        self.event_bridge.publish_leader_stepped_down(self.current_term, &self.validator_id);

        self.state = LeaderState::Follower { leader_id: None };
        tracing::info!("Stepped down from leadership in term {}", self.current_term);
    }

    /// Reset election timeout
    pub fn reset_election_timeout(&mut self) {
        self.last_heartbeat_received = SystemTime::now();
    }

    /// Get election statistics
    pub fn get_election_stats(&self) -> ElectionStats {
        let votes_received = match &self.state {
            LeaderState::Candidate { votes_received, .. } => *votes_received,
            _ => 0,
        };

        let time_in_state = SystemTime::now()
            .duration_since(self.last_heartbeat_received)
            .unwrap_or(Duration::ZERO);

        ElectionStats {
            current_term: self.current_term,
            state: match &self.state {
                LeaderState::Follower { .. } => "Follower".to_string(),
                LeaderState::Candidate { .. } => "Candidate".to_string(),
                LeaderState::Leader { .. } => "Leader".to_string(),
            },
            votes_received,
            time_in_state,
            voted_for: self.voted_for.clone(),
            current_leader: self.current_leader().cloned(),
        }
    }
}

/// Election statistics
#[derive(Debug)]
pub struct ElectionStats {
    /// Current term number
    pub current_term: u64,
    /// Current state as string
    pub state: String,
    /// Number of votes received (if candidate)
    pub votes_received: u32,
    /// Time spent in current state
    pub time_in_state: Duration,
    /// Who this validator voted for in current term
    pub voted_for: Option<ValidatorId>,
    /// Current known leader
    pub current_leader: Option<ValidatorId>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::validator::ValidatorInfo;

    fn create_test_validator_info(id: ValidatorId) -> ValidatorInfo {
        ValidatorInfo {
            id,
            address: "127.0.0.1:8080".parse().unwrap(),
            stake: 100,
            gpu_capacity: 1000,
            status: ValidatorStatus::Active,
            last_heartbeat: 0,
            public_key: vec![1, 2, 3, 4],
        }
    }

    #[test]
    fn test_leader_election_creation() {
        let validator_id = ValidatorId::new();
        let election = LeaderElection::new(
            validator_id.clone(),
            Duration::from_secs(5),
            Duration::from_secs(1),
        );

        assert!(election.is_follower());
        assert_eq!(election.term(), 0);
        assert!(election.current_leader().is_none());
    }

    #[test]
    fn test_start_election() {
        let validator_id = ValidatorId::new();
        let mut election = LeaderElection::new(
            validator_id.clone(),
            Duration::from_secs(5),
            Duration::from_secs(1),
        );

        let message = election.start_election().unwrap();

        assert!(election.is_candidate());
        assert_eq!(election.term(), 1);

        if let ElectionMessage::RequestVote {
            term, candidate_id, ..
        } = message
        {
            assert_eq!(term, 1);
            assert_eq!(candidate_id, validator_id);
        } else {
            panic!("Expected RequestVote message");
        }
    }

    #[test]
    fn test_vote_request_handling() {
        let validator_id = ValidatorId::new();
        let candidate_id = ValidatorId::new();
        let mut election = LeaderElection::new(
            validator_id.clone(),
            Duration::from_secs(5),
            Duration::from_secs(1),
        );

        let vote_request = ElectionMessage::RequestVote {
            term: 1,
            candidate_id: candidate_id.clone(),
            last_log_index: 0,
            last_log_term: 0,
        };

        let validators = HashMap::new();
        let response = election.handle_message(vote_request, &validators).unwrap();

        assert!(response.is_some());
        if let Some(ElectionMessage::VoteResponse {
            term,
            vote_granted,
            voter_id,
        }) = response
        {
            assert_eq!(term, 1);
            assert!(vote_granted);
            assert_eq!(voter_id, validator_id);
        }
    }

    #[test]
    fn test_vote_request_older_term() {
        let validator_id = ValidatorId::new();
        let candidate_id = ValidatorId::new();
        let mut election = LeaderElection::new(
            validator_id.clone(),
            Duration::from_secs(5),
            Duration::from_secs(1),
        );

        // Set current term to 2
        election.current_term = 2;

        let vote_request = ElectionMessage::RequestVote {
            term: 1, // Older term
            candidate_id,
            last_log_index: 0,
            last_log_term: 0,
        };

        let validators = HashMap::new();
        let response = election.handle_message(vote_request, &validators).unwrap();

        assert!(response.is_some());
        if let Some(ElectionMessage::VoteResponse {
            term, vote_granted, ..
        }) = response
        {
            assert_eq!(term, 2);
            assert!(!vote_granted);
        }
    }

    #[test]
    fn test_double_voting_prevention() {
        let validator_id = ValidatorId::new();
        let candidate1 = ValidatorId::new();
        let candidate2 = ValidatorId::new();
        let mut election =
            LeaderElection::new(validator_id, Duration::from_secs(5), Duration::from_secs(1));

        let validators = HashMap::new();

        // First vote request
        let vote_request1 = ElectionMessage::RequestVote {
            term: 1,
            candidate_id: candidate1,
            last_log_index: 0,
            last_log_term: 0,
        };
        let response1 = election.handle_message(vote_request1, &validators).unwrap();

        if let Some(ElectionMessage::VoteResponse { vote_granted, .. }) = response1 {
            assert!(vote_granted);
        }

        // Second vote request from different candidate in same term
        let vote_request2 = ElectionMessage::RequestVote {
            term: 1,
            candidate_id: candidate2,
            last_log_index: 0,
            last_log_term: 0,
        };
        let response2 = election.handle_message(vote_request2, &validators).unwrap();

        if let Some(ElectionMessage::VoteResponse { vote_granted, .. }) = response2 {
            assert!(!vote_granted); // Should not grant vote to second candidate
        }
    }

    #[test]
    fn test_election_win() {
        let validator_id = ValidatorId::new();
        let mut election = LeaderElection::new(
            validator_id.clone(),
            Duration::from_secs(5),
            Duration::from_secs(1),
        );

        // Create 3 validators (need 2 votes to win)
        let mut validators = HashMap::new();
        for i in 0..3 {
            let id = ValidatorId::new();
            validators.insert(id.clone(), create_test_validator_info(id));
        }

        // Start election
        election.start_election().unwrap();

        // Receive vote from another validator
        let vote_response = ElectionMessage::VoteResponse {
            term: 1,
            vote_granted: true,
            voter_id: ValidatorId::new(),
        };

        let response = election.handle_message(vote_response, &validators).unwrap();

        // Should become leader and send heartbeat
        assert!(election.is_leader());
        assert!(response.is_some());
        assert!(matches!(
            response.unwrap(),
            ElectionMessage::Heartbeat { .. }
        ));
    }

    #[test]
    fn test_heartbeat_handling() {
        let validator_id = ValidatorId::new();
        let leader_id = ValidatorId::new();
        let mut election = LeaderElection::new(
            validator_id.clone(),
            Duration::from_secs(5),
            Duration::from_secs(1),
        );

        let heartbeat = ElectionMessage::Heartbeat {
            term: 1,
            leader_id: leader_id.clone(),
            prev_log_index: 0,
            prev_log_term: 0,
            entries: vec![],
            leader_commit: 0,
        };

        let validators = HashMap::new();
        let response = election.handle_message(heartbeat, &validators).unwrap();

        assert!(election.is_follower());
        assert_eq!(election.current_leader(), Some(&leader_id));

        assert!(response.is_some());
        if let Some(ElectionMessage::HeartbeatResponse { success, .. }) = response {
            assert!(success);
        }
    }

    #[test]
    fn test_heartbeat_older_term() {
        let validator_id = ValidatorId::new();
        let leader_id = ValidatorId::new();
        let mut election =
            LeaderElection::new(validator_id, Duration::from_secs(5), Duration::from_secs(1));

        // Set current term to 2
        election.current_term = 2;

        let heartbeat = ElectionMessage::Heartbeat {
            term: 1, // Older term
            leader_id,
            prev_log_index: 0,
            prev_log_term: 0,
            entries: vec![],
            leader_commit: 0,
        };

        let validators = HashMap::new();
        let response = election.handle_message(heartbeat, &validators).unwrap();

        if let Some(ElectionMessage::HeartbeatResponse { success, .. }) = response {
            assert!(!success);
        }
    }

    #[test]
    fn test_leader_step_down() {
        let validator_id = ValidatorId::new();
        let mut election =
            LeaderElection::new(validator_id, Duration::from_secs(5), Duration::from_secs(1));

        // Become leader
        election.current_term = 1;
        election.become_leader().unwrap();
        assert!(election.is_leader());

        // Step down
        election.step_down();
        assert!(election.is_follower());
    }

    #[test]
    fn test_heartbeat_timing() {
        let validator_id = ValidatorId::new();
        let mut election = LeaderElection::new(
            validator_id,
            Duration::from_secs(5),
            Duration::from_millis(50), // Shorter heartbeat interval
        );

        // Become leader
        election.current_term = 1;
        election.become_leader().unwrap();

        // Initially should not need heartbeat
        assert!(!election.should_send_heartbeat());

        // Wait for heartbeat interval plus buffer
        std::thread::sleep(Duration::from_millis(100));
        assert!(election.should_send_heartbeat());

        // Update heartbeat time
        election.update_heartbeat_time().unwrap();
        assert!(!election.should_send_heartbeat());
    }

    #[test]
    fn test_election_timeout() {
        let validator_id = ValidatorId::new();
        let mut election = LeaderElection::new(
            validator_id,
            Duration::from_millis(50), // Very short timeout
            Duration::from_secs(1),
        );

        // Initially no timeout
        assert!(!election.is_election_timeout());

        // Wait for timeout plus random offset (up to 1000ms)
        std::thread::sleep(Duration::from_millis(1200)); // Ensure we exceed timeout + max offset
        assert!(election.is_election_timeout());

        // Reset timeout
        election.reset_election_timeout();
        assert!(!election.is_election_timeout());
    }

    #[test]
    fn test_election_stats() {
        let validator_id = ValidatorId::new();
        let mut election = LeaderElection::new(
            validator_id.clone(),
            Duration::from_secs(5),
            Duration::from_secs(1),
        );

        let stats = election.get_election_stats();
        assert_eq!(stats.current_term, 0);
        assert_eq!(stats.state, "Follower");
        assert_eq!(stats.votes_received, 0);
        assert!(stats.voted_for.is_none());
        assert!(stats.current_leader.is_none());

        // Start election
        election.start_election().unwrap();

        let stats = election.get_election_stats();
        assert_eq!(stats.current_term, 1);
        assert_eq!(stats.state, "Candidate");
        assert_eq!(stats.votes_received, 1);
        assert_eq!(stats.voted_for, Some(validator_id.clone()));

        // Become leader
        election.become_leader().unwrap();

        let stats = election.get_election_stats();
        assert_eq!(stats.state, "Leader");
        assert_eq!(stats.current_leader, Some(validator_id));
    }

    #[test]
    fn test_message_serialization() {
        let validator_id = ValidatorId::new();

        let messages = vec![
            ElectionMessage::RequestVote {
                term: 1,
                candidate_id: validator_id.clone(),
                last_log_index: 0,
                last_log_term: 0,
            },
            ElectionMessage::VoteResponse {
                term: 1,
                vote_granted: true,
                voter_id: validator_id.clone(),
            },
            ElectionMessage::Heartbeat {
                term: 1,
                leader_id: validator_id.clone(),
                prev_log_index: 0,
                prev_log_term: 0,
                entries: vec!["entry1".to_string()],
                leader_commit: 0,
            },
            ElectionMessage::HeartbeatResponse {
                term: 1,
                success: true,
                follower_id: validator_id,
                match_index: 0,
            },
        ];

        for message in messages {
            let serialized = serde_json::to_string(&message).unwrap();
            let _deserialized: ElectionMessage = serde_json::from_str(&serialized).unwrap();
        }
    }
}
