//! Voting mechanism for consensus protocol

use crate::error::{ConsensusError, ConsensusResult};
use crate::validator::{ValidatorId, ValidatorInfo};
use rayon::prelude::*;
use ring::signature::{Ed25519KeyPair, KeyPair, UnparsedPublicKey, ED25519};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tracing;
use uuid::Uuid;

/// Get current timestamp in seconds, logging an error if system clock is invalid.
#[inline]
fn current_timestamp_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or_else(|e| {
            tracing::error!(
                error = ?e,
                "System clock is before UNIX epoch - consensus timestamps may be invalid"
            );
            0
        })
}

/// Unique identifier for voting rounds
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RoundId(pub Uuid);

impl RoundId {
    /// Create new round ID
    #[inline]
    #[must_use = "RoundId must be stored for round tracking"]
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    /// Create from UUID
    #[inline]
    #[must_use = "RoundId must be stored for round tracking"]
    pub fn from_uuid(uuid: Uuid) -> Self {
        Self(uuid)
    }

    /// Get underlying UUID
    #[inline]
    #[must_use]
    pub fn as_uuid(&self) -> Uuid {
        self.0
    }
}

impl Default for RoundId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for RoundId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Type of vote in consensus
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VoteType {
    /// Pre-vote phase
    PreVote,
    /// Pre-commit phase
    PreCommit,
    /// Final commit vote
    Commit,
}

/// Individual vote in consensus round
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vote {
    /// Voting round identifier
    pub round_id: RoundId,
    /// Type of vote
    pub vote_type: VoteType,
    /// Validator who cast the vote
    pub validator_id: ValidatorId,
    /// Proposed value hash (None for nil vote)
    pub value_hash: Option<String>,
    /// Timestamp when vote was cast
    pub timestamp: u64,
    /// Digital signature of the vote
    pub signature: Vec<u8>,
}

impl Vote {
    /// Create new vote
    pub fn new(
        round_id: RoundId,
        vote_type: VoteType,
        validator_id: ValidatorId,
        value_hash: Option<String>,
        signature: Vec<u8>,
    ) -> Self {
        Self {
            round_id,
            vote_type,
            validator_id,
            value_hash,
            timestamp: current_timestamp_secs(),
            signature,
        }
    }

    /// Check if this is a nil vote (no value proposed)
    pub fn is_nil(&self) -> bool {
        self.value_hash.is_none()
    }

    /// Compute the signing payload for this vote
    ///
    /// The payload includes all vote fields that should be signed:
    /// - round_id, vote_type, validator_id, value_hash, timestamp
    fn signing_payload(&self) -> Vec<u8> {
        let mut payload = Vec::new();

        // Include round ID
        payload.extend_from_slice(self.round_id.0.as_bytes());

        // Include vote type as a discriminant byte
        let vote_type_byte = match self.vote_type {
            VoteType::PreVote => 0u8,
            VoteType::PreCommit => 1u8,
            VoteType::Commit => 2u8,
        };
        payload.push(vote_type_byte);

        // Include validator ID
        payload.extend_from_slice(self.validator_id.0.as_bytes());

        // Include value hash (or empty marker)
        match &self.value_hash {
            Some(hash) => {
                payload.push(1u8); // Has value marker
                payload.extend_from_slice(hash.as_bytes());
            }
            None => {
                payload.push(0u8); // Nil vote marker
            }
        }

        // Include timestamp
        payload.extend_from_slice(&self.timestamp.to_le_bytes());

        payload
    }

    /// Verify vote signature using Ed25519
    ///
    /// Uses ring's Ed25519 implementation for cryptographic verification.
    /// The signature must be a valid Ed25519 signature over the vote's signing payload.
    pub fn verify_signature(&self, public_key: &[u8]) -> bool {
        // Validate public key length (Ed25519 public keys are 32 bytes)
        if public_key.len() != 32 {
            return false;
        }

        // Validate signature length (Ed25519 signatures are 64 bytes)
        if self.signature.len() != 64 {
            return false;
        }

        // Compute the signing payload
        let payload = self.signing_payload();

        // Verify the signature using ring's Ed25519 implementation
        let peer_public_key = UnparsedPublicKey::new(&ED25519, public_key);

        peer_public_key.verify(&payload, &self.signature).is_ok()
    }

    /// Sign this vote with an Ed25519 key pair
    ///
    /// Returns the signature bytes that should be stored in the vote.
    pub fn sign(
        round_id: RoundId,
        vote_type: VoteType,
        validator_id: ValidatorId,
        value_hash: Option<String>,
        key_pair: &Ed25519KeyPair,
    ) -> Self {
        let timestamp = current_timestamp_secs();

        // Create vote without signature first to compute payload
        let mut vote = Self {
            round_id,
            vote_type,
            validator_id,
            value_hash,
            timestamp,
            signature: Vec::new(),
        };

        // Compute payload and sign
        let payload = vote.signing_payload();
        let signature = key_pair.sign(&payload);
        vote.signature = signature.as_ref().to_vec();

        debug_assert!(
            vote.signature.len() == 64,
            "Ed25519 signature must be 64 bytes"
        );

        vote
    }
}

/// Voting round state and management
pub struct VotingRound {
    /// Round identifier
    round_id: RoundId,
    /// Current height/block number
    height: u64,
    /// Round number within this height
    round_number: u32,
    /// Round timeout
    timeout: Duration,
    /// Round start time
    start_time: SystemTime,
    /// All votes received in this round
    votes: HashMap<(ValidatorId, VoteType), Vote>,
    /// Active validators and their stakes
    validators: HashMap<ValidatorId, u64>,
    /// Required voting threshold
    voting_threshold: u64,
    /// Current proposed value
    proposed_value: Option<String>,
    /// Round status
    status: RoundStatus,
}

/// Status of a voting round
#[derive(Debug, Clone, PartialEq)]
pub enum RoundStatus {
    /// Round is active and accepting votes
    Active,
    /// Pre-vote phase completed
    PreVoteComplete,
    /// Pre-commit phase completed
    PreCommitComplete,
    /// Round completed with consensus
    Completed { value: String },
    /// Round timed out without consensus
    TimedOut,
    /// Round failed due to Byzantine behavior
    Failed { reason: String },
}

impl VotingRound {
    /// Create new voting round
    pub fn new(
        height: u64,
        round_number: u32,
        timeout: Duration,
        validators: HashMap<ValidatorId, u64>,
        voting_threshold: u64,
    ) -> Self {
        Self {
            round_id: RoundId::new(),
            height,
            round_number,
            timeout,
            start_time: SystemTime::now(),
            votes: HashMap::new(),
            validators,
            voting_threshold,
            proposed_value: None,
            status: RoundStatus::Active,
        }
    }

    /// Get round ID
    #[inline]
    pub fn round_id(&self) -> &RoundId {
        &self.round_id
    }

    /// Get current height
    #[inline]
    pub fn height(&self) -> u64 {
        self.height
    }

    /// Get round number
    #[inline]
    pub fn round_number(&self) -> u32 {
        self.round_number
    }

    /// Get round status
    #[inline]
    pub fn status(&self) -> &RoundStatus {
        &self.status
    }

    /// Check if round has timed out
    #[inline]
    pub fn is_timed_out(&self) -> bool {
        SystemTime::now()
            .duration_since(self.start_time)
            .unwrap_or(Duration::ZERO)
            > self.timeout
    }

    /// Add vote to the round
    #[must_use = "ignoring the Result may hide vote validation failures"]
    pub fn add_vote(&mut self, vote: Vote, validator_info: &ValidatorInfo) -> ConsensusResult<()> {
        // Validate vote belongs to this round
        if vote.round_id != self.round_id {
            return Err(ConsensusError::InvalidMessage(
                "Vote belongs to different round".to_string(),
            ));
        }

        // Check if validator is authorized
        if !self.validators.contains_key(&vote.validator_id) {
            return Err(ConsensusError::ValidationFailed(
                "Validator not authorized for this round".to_string(),
            ));
        }

        // Verify signature
        if !vote.verify_signature(&validator_info.public_key) {
            return Err(ConsensusError::ValidationFailed(
                "Invalid vote signature".to_string(),
            ));
        }

        // Check for double voting
        let vote_key = (vote.validator_id.clone(), vote.vote_type.clone());
        if self.votes.contains_key(&vote_key) {
            return Err(ConsensusError::ByzantineBehavior {
                validator_id: vote.validator_id.to_string(),
                reason: "Double voting detected".to_string(),
            });
        }

        // Add vote
        self.votes.insert(vote_key, vote);

        // Check if we've reached consensus
        self.check_consensus()?;

        Ok(())
    }

    /// Set proposed value for this round
    pub fn set_proposed_value(&mut self, value: String) {
        self.proposed_value = Some(value);
    }

    /// Get proposed value
    pub fn proposed_value(&self) -> Option<&String> {
        self.proposed_value.as_ref()
    }

    /// Check if consensus has been reached
    fn check_consensus(&mut self) -> ConsensusResult<()> {
        match self.status {
            RoundStatus::Active => {
                if self.has_pre_vote_majority()? {
                    self.status = RoundStatus::PreVoteComplete;
                }
            }
            RoundStatus::PreVoteComplete => {
                if self.has_pre_commit_majority()? {
                    self.status = RoundStatus::PreCommitComplete;
                }
            }
            RoundStatus::PreCommitComplete => {
                if let Some(value) = self.has_commit_majority()? {
                    self.status = RoundStatus::Completed { value };
                }
            }
            _ => {} // Terminal states
        }

        Ok(())
    }

    /// Check if we have majority for pre-vote phase
    fn has_pre_vote_majority(&self) -> ConsensusResult<bool> {
        debug_assert!(
            self.voting_threshold > 0,
            "Voting threshold must be positive for consensus"
        );
        let stake = self.count_votes_by_type(VoteType::PreVote);
        Ok(stake >= self.voting_threshold)
    }

    /// Check if we have majority for pre-commit phase
    fn has_pre_commit_majority(&self) -> ConsensusResult<bool> {
        let stake = self.count_votes_by_type(VoteType::PreCommit);
        Ok(stake >= self.voting_threshold)
    }

    /// Check if we have majority for commit phase
    fn has_commit_majority(&self) -> ConsensusResult<Option<String>> {
        let votes_by_value = self.group_commit_votes_by_value();

        for (value_hash, stake) in votes_by_value {
            if stake >= self.voting_threshold {
                return Ok(value_hash);
            }
        }

        Ok(None)
    }

    /// Count total stake for votes of given type (parallelized with Rayon)
    fn count_votes_by_type(&self, vote_type: VoteType) -> u64 {
        // Collect votes into a Vec for parallel iteration
        let votes: Vec<_> = self.votes.iter().collect();

        // Use Rayon parallel fold/reduce for stake counting
        votes
            .par_iter()
            .filter(|((_, v_type), _)| *v_type == vote_type)
            .map(|((validator_id, _), _)| self.validators.get(validator_id).copied().unwrap_or(0))
            .sum()
    }

    /// Group commit votes by their value hash (parallelized with Rayon)
    fn group_commit_votes_by_value(&self) -> HashMap<Option<String>, u64> {
        // Collect votes into a Vec for parallel iteration
        let votes: Vec<_> = self.votes.iter().collect();

        // Use Rayon parallel fold/reduce for grouping
        votes
            .par_iter()
            .filter(|((_, vote_type), _)| *vote_type == VoteType::Commit)
            .fold(
                HashMap::new,
                |mut acc: HashMap<Option<String>, u64>, ((validator_id, _), vote)| {
                    if let Some(stake) = self.validators.get(validator_id) {
                        *acc.entry(vote.value_hash.clone()).or_insert(0) += stake;
                    }
                    acc
                },
            )
            .reduce(HashMap::new, |mut a, b| {
                for (k, v) in b {
                    *a.entry(k).or_insert(0) += v;
                }
                a
            })
    }

    /// Get all votes for a specific type
    pub fn get_votes_by_type(&self, vote_type: VoteType) -> Vec<&Vote> {
        self.votes
            .iter()
            .filter_map(|((_, v_type), vote)| {
                if *v_type == vote_type {
                    Some(vote)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get vote from specific validator and type
    pub fn get_vote(&self, validator_id: &ValidatorId, vote_type: VoteType) -> Option<&Vote> {
        self.votes.get(&(validator_id.clone(), vote_type))
    }

    /// Check timeout and update status
    pub fn check_timeout(&mut self) {
        if self.is_timed_out()
            && matches!(
                self.status,
                RoundStatus::Active | RoundStatus::PreVoteComplete | RoundStatus::PreCommitComplete
            )
        {
            self.status = RoundStatus::TimedOut;
        }
    }

    /// Mark round as failed
    pub fn mark_failed(&mut self, reason: String) {
        self.status = RoundStatus::Failed { reason };
    }

    /// Get vote statistics
    pub fn get_vote_stats(&self) -> VoteStats {
        let mut stats = VoteStats::default();

        for vote_type in [VoteType::PreVote, VoteType::PreCommit, VoteType::Commit] {
            let stake = self.count_votes_by_type(vote_type.clone());
            let count = self.get_votes_by_type(vote_type.clone()).len();

            match vote_type {
                VoteType::PreVote => {
                    stats.pre_vote_stake = stake;
                    stats.pre_vote_count = count;
                }
                VoteType::PreCommit => {
                    stats.pre_commit_stake = stake;
                    stats.pre_commit_count = count;
                }
                VoteType::Commit => {
                    stats.commit_stake = stake;
                    stats.commit_count = count;
                }
            }
        }

        stats.total_validators = self.validators.len();
        stats.voting_threshold = self.voting_threshold;
        stats
    }
}

/// Vote statistics for a round
#[derive(Debug, Default)]
pub struct VoteStats {
    /// Pre-vote stake total
    pub pre_vote_stake: u64,
    /// Pre-vote count
    pub pre_vote_count: usize,
    /// Pre-commit stake total
    pub pre_commit_stake: u64,
    /// Pre-commit count
    pub pre_commit_count: usize,
    /// Commit stake total
    pub commit_stake: u64,
    /// Commit count
    pub commit_count: usize,
    /// Total validators in round
    pub total_validators: usize,
    /// Required voting threshold
    pub voting_threshold: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::validator::ValidatorInfo;
    use ring::rand::SystemRandom;
    use ring::signature::Ed25519KeyPair;
    use std::net::SocketAddr;

    /// Test key pair generator
    fn generate_test_keypair() -> (Ed25519KeyPair, Vec<u8>) {
        let rng = SystemRandom::new();
        let pkcs8_bytes = Ed25519KeyPair::generate_pkcs8(&rng).unwrap();
        let key_pair = Ed25519KeyPair::from_pkcs8(pkcs8_bytes.as_ref()).unwrap();
        let public_key = key_pair.public_key().as_ref().to_vec();
        (key_pair, public_key)
    }

    fn create_test_validator_info(id: ValidatorId, stake: u64, public_key: Vec<u8>) -> ValidatorInfo {
        ValidatorInfo {
            id,
            address: "127.0.0.1:8080".parse().unwrap(),
            stake,
            gpu_capacity: 1000,
            status: crate::validator::ValidatorStatus::Active,
            last_heartbeat: 0,
            public_key,
        }
    }

    fn create_test_vote(
        round_id: RoundId,
        vote_type: VoteType,
        validator_id: ValidatorId,
        value_hash: Option<String>,
        key_pair: &Ed25519KeyPair,
    ) -> Vote {
        Vote::sign(round_id, vote_type, validator_id, value_hash, key_pair)
    }

    #[test]
    fn test_round_id_creation() {
        let id1 = RoundId::new();
        let id2 = RoundId::new();
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_round_id_from_uuid() {
        let uuid = Uuid::new_v4();
        let id = RoundId::from_uuid(uuid);
        assert_eq!(id.as_uuid(), uuid);
    }

    #[test]
    fn test_vote_creation() {
        let (key_pair, _public_key) = generate_test_keypair();
        let round_id = RoundId::new();
        let validator_id = ValidatorId::new();
        let vote = create_test_vote(
            round_id.clone(),
            VoteType::PreVote,
            validator_id.clone(),
            Some("hash123".to_string()),
            &key_pair,
        );

        assert_eq!(vote.round_id, round_id);
        assert_eq!(vote.vote_type, VoteType::PreVote);
        assert_eq!(vote.validator_id, validator_id);
        assert_eq!(vote.value_hash, Some("hash123".to_string()));
        assert!(!vote.is_nil());
    }

    #[test]
    fn test_nil_vote() {
        let (key_pair, _) = generate_test_keypair();
        let vote = create_test_vote(RoundId::new(), VoteType::PreVote, ValidatorId::new(), None, &key_pair);
        assert!(vote.is_nil());
    }

    #[test]
    fn test_vote_signature_verification() {
        let (key_pair, public_key) = generate_test_keypair();
        let vote = create_test_vote(
            RoundId::new(),
            VoteType::PreVote,
            ValidatorId::new(),
            Some("hash123".to_string()),
            &key_pair,
        );

        // Valid signature should verify
        assert!(vote.verify_signature(&public_key));

        // Wrong public key should fail
        let (_, wrong_key) = generate_test_keypair();
        assert!(!vote.verify_signature(&wrong_key));

        // Empty key should fail
        assert!(!vote.verify_signature(&[]));

        // Wrong length key should fail
        assert!(!vote.verify_signature(&[1, 2, 3, 4]));
    }

    #[test]
    fn test_vote_signature_tampering() {
        let (key_pair, public_key) = generate_test_keypair();
        let mut vote = create_test_vote(
            RoundId::new(),
            VoteType::PreVote,
            ValidatorId::new(),
            Some("hash123".to_string()),
            &key_pair,
        );

        // Valid signature should verify
        assert!(vote.verify_signature(&public_key));

        // Tamper with the value hash
        vote.value_hash = Some("tampered".to_string());

        // Should fail verification after tampering
        assert!(!vote.verify_signature(&public_key));
    }

    #[test]
    fn test_voting_round_creation() {
        let mut validators = HashMap::new();
        let validator_id = ValidatorId::new();
        validators.insert(validator_id, 100);

        let round = VotingRound::new(
            1,
            0,
            Duration::from_secs(30),
            validators,
            67, // 2/3 threshold
        );

        assert_eq!(round.height(), 1);
        assert_eq!(round.round_number(), 0);
        assert!(matches!(round.status(), RoundStatus::Active));
    }

    #[test]
    fn test_add_valid_vote() {
        let (key_pair, public_key) = generate_test_keypair();
        let validator_id = ValidatorId::new();
        let validator_info = create_test_validator_info(validator_id.clone(), 100, public_key);

        let mut validators = HashMap::new();
        validators.insert(validator_id.clone(), 100);

        let mut round = VotingRound::new(1, 0, Duration::from_secs(30), validators, 67);

        let vote = create_test_vote(
            round.round_id().clone(),
            VoteType::PreVote,
            validator_id.clone(),
            Some("hash123".to_string()),
            &key_pair,
        );

        assert!(round.add_vote(vote, &validator_info).is_ok());
    }

    #[test]
    fn test_add_vote_wrong_round() {
        let (key_pair, public_key) = generate_test_keypair();
        let validator_id = ValidatorId::new();
        let validator_info = create_test_validator_info(validator_id.clone(), 100, public_key);

        let mut validators = HashMap::new();
        validators.insert(validator_id.clone(), 100);

        let mut round = VotingRound::new(1, 0, Duration::from_secs(30), validators, 67);

        let wrong_vote = create_test_vote(
            RoundId::new(), // Different round ID
            VoteType::PreVote,
            validator_id,
            Some("hash123".to_string()),
            &key_pair,
        );

        let result = round.add_vote(wrong_vote, &validator_info);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ConsensusError::InvalidMessage(_)
        ));
    }

    #[test]
    fn test_add_vote_unauthorized_validator() {
        let (key_pair, public_key) = generate_test_keypair();
        let validator_id = ValidatorId::new();
        let unauthorized_id = ValidatorId::new();
        let validator_info = create_test_validator_info(unauthorized_id.clone(), 100, public_key);

        let mut validators = HashMap::new();
        validators.insert(validator_id, 100); // Only this validator is authorized

        let mut round = VotingRound::new(1, 0, Duration::from_secs(30), validators, 67);

        let vote = create_test_vote(
            round.round_id().clone(),
            VoteType::PreVote,
            unauthorized_id,
            Some("hash123".to_string()),
            &key_pair,
        );

        let result = round.add_vote(vote, &validator_info);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ConsensusError::ValidationFailed(_)
        ));
    }

    #[test]
    fn test_double_voting_detection() {
        let (key_pair, public_key) = generate_test_keypair();
        let validator_id = ValidatorId::new();
        let validator_info = create_test_validator_info(validator_id.clone(), 100, public_key);

        let mut validators = HashMap::new();
        validators.insert(validator_id.clone(), 100);

        let mut round = VotingRound::new(1, 0, Duration::from_secs(30), validators, 67);

        // First vote
        let vote1 = create_test_vote(
            round.round_id().clone(),
            VoteType::PreVote,
            validator_id.clone(),
            Some("hash123".to_string()),
            &key_pair,
        );
        assert!(round.add_vote(vote1, &validator_info).is_ok());

        // Second vote of same type - should be detected as double voting
        let vote2 = create_test_vote(
            round.round_id().clone(),
            VoteType::PreVote,
            validator_id,
            Some("hash456".to_string()),
            &key_pair,
        );

        let result = round.add_vote(vote2, &validator_info);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ConsensusError::ByzantineBehavior { .. }
        ));
    }

    #[test]
    fn test_consensus_progression() {
        // Create 3 validators with equal stake
        let mut validators = HashMap::new();
        let mut validator_infos = Vec::new();
        let mut key_pairs = Vec::new();

        for _ in 0..3 {
            let id = ValidatorId::new();
            let (key_pair, public_key) = generate_test_keypair();
            let info = create_test_validator_info(id.clone(), 100, public_key);
            validators.insert(id.clone(), 100);
            validator_infos.push((id, info));
            key_pairs.push(key_pair);
        }

        let mut round = VotingRound::new(
            1,
            0,
            Duration::from_secs(30),
            validators,
            200, // Need 2/3 of 300 total stake
        );

        // Initially active
        assert!(matches!(round.status(), RoundStatus::Active));

        // Add pre-votes from 2 validators (enough for majority)
        for i in 0..2 {
            let (id, info) = &validator_infos[i];
            let vote = create_test_vote(
                round.round_id().clone(),
                VoteType::PreVote,
                id.clone(),
                Some("hash123".to_string()),
                &key_pairs[i],
            );
            round.add_vote(vote, info).unwrap();
        }

        // Should progress to PreVoteComplete
        assert!(matches!(round.status(), RoundStatus::PreVoteComplete));

        // Add pre-commits from 2 validators
        for i in 0..2 {
            let (id, info) = &validator_infos[i];
            let vote = create_test_vote(
                round.round_id().clone(),
                VoteType::PreCommit,
                id.clone(),
                Some("hash123".to_string()),
                &key_pairs[i],
            );
            round.add_vote(vote, info).unwrap();
        }

        // Should progress to PreCommitComplete
        assert!(matches!(round.status(), RoundStatus::PreCommitComplete));

        // Add commits from 2 validators
        for i in 0..2 {
            let (id, info) = &validator_infos[i];
            let vote = create_test_vote(
                round.round_id().clone(),
                VoteType::Commit,
                id.clone(),
                Some("hash123".to_string()),
                &key_pairs[i],
            );
            round.add_vote(vote, info).unwrap();
        }

        // Should complete with consensus
        assert!(matches!(round.status(), RoundStatus::Completed { .. }));
    }

    #[test]
    fn test_timeout_detection() {
        let validators = HashMap::new();
        let mut round = VotingRound::new(
            1,
            0,
            Duration::from_millis(10), // Very short timeout
            validators,
            67,
        );

        assert!(!round.is_timed_out());

        std::thread::sleep(Duration::from_millis(20));

        assert!(round.is_timed_out());

        round.check_timeout();
        assert!(matches!(round.status(), RoundStatus::TimedOut));
    }

    #[test]
    fn test_vote_statistics() {
        let mut validators = HashMap::new();
        let mut validator_infos = Vec::new();
        let mut key_pairs = Vec::new();

        for _ in 0..3 {
            let id = ValidatorId::new();
            let (key_pair, public_key) = generate_test_keypair();
            let info = create_test_validator_info(id.clone(), 100, public_key);
            validators.insert(id.clone(), 100);
            validator_infos.push((id, info));
            key_pairs.push(key_pair);
        }

        let mut round = VotingRound::new(1, 0, Duration::from_secs(30), validators, 200);

        // Add some votes
        let (id1, info1) = &validator_infos[0];
        let vote1 = create_test_vote(
            round.round_id().clone(),
            VoteType::PreVote,
            id1.clone(),
            Some("hash123".to_string()),
            &key_pairs[0],
        );
        round.add_vote(vote1, info1).unwrap();

        let (id2, info2) = &validator_infos[1];
        let vote2 = create_test_vote(
            round.round_id().clone(),
            VoteType::PreCommit,
            id2.clone(),
            Some("hash123".to_string()),
            &key_pairs[1],
        );
        round.add_vote(vote2, info2).unwrap();

        let stats = round.get_vote_stats();
        assert_eq!(stats.pre_vote_stake, 100);
        assert_eq!(stats.pre_vote_count, 1);
        assert_eq!(stats.pre_commit_stake, 100);
        assert_eq!(stats.pre_commit_count, 1);
        assert_eq!(stats.commit_stake, 0);
        assert_eq!(stats.commit_count, 0);
        assert_eq!(stats.total_validators, 3);
        assert_eq!(stats.voting_threshold, 200);
    }

    #[test]
    fn test_proposed_value() {
        let validators = HashMap::new();
        let mut round = VotingRound::new(1, 0, Duration::from_secs(30), validators, 67);

        assert!(round.proposed_value().is_none());

        round.set_proposed_value("test_value".to_string());
        assert_eq!(round.proposed_value(), Some(&"test_value".to_string()));
    }

    #[test]
    fn test_mark_failed() {
        let validators = HashMap::new();
        let mut round = VotingRound::new(1, 0, Duration::from_secs(30), validators, 67);

        round.mark_failed("Test failure".to_string());
        assert!(matches!(round.status(), RoundStatus::Failed { .. }));
    }

    #[test]
    fn test_get_votes_by_type() {
        let (key_pair, public_key) = generate_test_keypair();
        let validator_id = ValidatorId::new();
        let validator_info = create_test_validator_info(validator_id.clone(), 100, public_key);

        let mut validators = HashMap::new();
        validators.insert(validator_id.clone(), 100);

        let mut round = VotingRound::new(1, 0, Duration::from_secs(30), validators, 67);

        // Add different types of votes
        let vote1 = create_test_vote(
            round.round_id().clone(),
            VoteType::PreVote,
            validator_id.clone(),
            Some("hash123".to_string()),
            &key_pair,
        );
        round.add_vote(vote1, &validator_info).unwrap();

        let pre_votes = round.get_votes_by_type(VoteType::PreVote);
        assert_eq!(pre_votes.len(), 1);

        let pre_commits = round.get_votes_by_type(VoteType::PreCommit);
        assert_eq!(pre_commits.len(), 0);
    }

    #[test]
    fn test_get_specific_vote() {
        let (key_pair, public_key) = generate_test_keypair();
        let validator_id = ValidatorId::new();
        let validator_info = create_test_validator_info(validator_id.clone(), 100, public_key);

        let mut validators = HashMap::new();
        validators.insert(validator_id.clone(), 100);

        let mut round = VotingRound::new(1, 0, Duration::from_secs(30), validators, 67);

        let vote = create_test_vote(
            round.round_id().clone(),
            VoteType::PreVote,
            validator_id.clone(),
            Some("hash123".to_string()),
            &key_pair,
        );
        round.add_vote(vote, &validator_info).unwrap();

        let retrieved_vote = round.get_vote(&validator_id, VoteType::PreVote);
        assert!(retrieved_vote.is_some());

        let no_vote = round.get_vote(&validator_id, VoteType::PreCommit);
        assert!(no_vote.is_none());
    }

    #[test]
    fn test_vote_type_serialization() {
        let vote_types = vec![VoteType::PreVote, VoteType::PreCommit, VoteType::Commit];

        for vote_type in vote_types {
            let serialized = serde_json::to_string(&vote_type).unwrap();
            let deserialized: VoteType = serde_json::from_str(&serialized).unwrap();
            assert_eq!(vote_type, deserialized);
        }
    }

    #[test]
    fn test_round_id_display() {
        let id = RoundId::new();
        let display = format!("{id}");
        assert_eq!(display, id.as_uuid().to_string());
    }
}
