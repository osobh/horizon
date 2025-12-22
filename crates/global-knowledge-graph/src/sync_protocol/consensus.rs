//! Consensus mechanisms for distributed synchronization

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;
use uuid::Uuid;

use crate::sync_protocol::types::KnowledgeOperation;

/// Consensus proposal for voting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusProposal {
    pub proposal_id: Uuid,
    pub operations: Vec<KnowledgeOperation>,
    pub proposer: String,
    pub timestamp: DateTime<Utc>,
    pub round: usize,
}

/// Vote on a consensus proposal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusVote {
    pub vote_id: Uuid,
    pub proposal_id: Uuid,
    pub voter: String,
    pub vote: Vote,
    pub timestamp: DateTime<Utc>,
}

/// Vote types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Vote {
    Accept,
    Reject,
    Abstain,
}

/// Result of consensus round
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusResult {
    pub proposal_id: Uuid,
    pub accepted: bool,
    pub vote_count: HashMap<Vote, usize>,
    pub duration: Duration,
}

/// Trait for consensus engines
#[async_trait::async_trait]
pub trait ConsensusEngine: Send + Sync {
    async fn propose(&self, operations: Vec<KnowledgeOperation>) -> ConsensusProposal;
    async fn vote(&self, proposal: &ConsensusProposal) -> ConsensusVote;
    async fn tally_votes(&self, votes: Vec<ConsensusVote>) -> ConsensusResult;
    async fn finalize(&self, result: ConsensusResult);
}

/// PBFT consensus engine implementation
pub struct PBFTConsensusEngine {
    node_id: String,
    view_number: tokio::sync::RwLock<usize>,
    proposal_history: tokio::sync::RwLock<Vec<ConsensusProposal>>,
    vote_threshold: f32,
}

impl PBFTConsensusEngine {
    pub fn new(node_id: String, vote_threshold: f32) -> Self {
        Self {
            node_id,
            view_number: tokio::sync::RwLock::new(0),
            proposal_history: tokio::sync::RwLock::new(Vec::new()),
            vote_threshold,
        }
    }
}

#[async_trait::async_trait]
impl ConsensusEngine for PBFTConsensusEngine {
    async fn propose(&self, operations: Vec<KnowledgeOperation>) -> ConsensusProposal {
        let view = *self.view_number.read().await;
        let proposal = ConsensusProposal {
            proposal_id: Uuid::new_v4(),
            operations,
            proposer: self.node_id.clone(),
            timestamp: Utc::now(),
            round: view,
        };
        
        let mut history = self.proposal_history.write().await;
        history.push(proposal.clone());
        
        proposal
    }

    async fn vote(&self, proposal: &ConsensusProposal) -> ConsensusVote {
        // Simple voting logic - in production would validate proposal
        let vote = if proposal.operations.len() > 0 {
            Vote::Accept
        } else {
            Vote::Reject
        };
        
        ConsensusVote {
            vote_id: Uuid::new_v4(),
            proposal_id: proposal.proposal_id,
            voter: self.node_id.clone(),
            vote,
            timestamp: Utc::now(),
        }
    }

    async fn tally_votes(&self, votes: Vec<ConsensusVote>) -> ConsensusResult {
        let mut vote_count = HashMap::new();
        
        for vote in &votes {
            *vote_count.entry(vote.vote.clone()).or_insert(0) += 1;
        }
        
        let accept_count = vote_count.get(&Vote::Accept).copied().unwrap_or(0);
        let total_votes = votes.len();
        let accepted = accept_count as f32 / total_votes as f32 >= self.vote_threshold;
        
        ConsensusResult {
            proposal_id: votes.first().map(|v| v.proposal_id).unwrap_or_else(Uuid::new_v4),
            accepted,
            vote_count,
            duration: Duration::from_millis(100), // Mock duration
        }
    }

    async fn finalize(&self, result: ConsensusResult) {
        if result.accepted {
            // Increment view number for next round
            let mut view = self.view_number.write().await;
            *view += 1;
        }
    }
}