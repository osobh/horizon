//! Audit chain management for immutable audit trails

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use uuid::Uuid;

use crate::audit::entry::AuditLogEntry;
use crate::error::ComplianceResult;

/// Immutable audit chain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditChain {
    /// Chain identifier
    pub chain_id: Uuid,
    /// Chain creation timestamp
    pub created_at: DateTime<Utc>,
    /// Chain entries
    pub entries: VecDeque<AuditLogEntry>,
    /// Maximum chain length before rotation
    pub max_entries: usize,
    /// Chain metadata
    pub metadata: std::collections::HashMap<String, String>,
    /// Chain state
    pub state: ChainState,
    /// Previous chain ID (for rotation)
    pub previous_chain_id: Option<Uuid>,
    /// Next chain ID (for rotation)
    pub next_chain_id: Option<Uuid>,
}

/// Chain state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChainState {
    /// Chain is active and accepting new entries
    Active,
    /// Chain is sealed and read-only
    Sealed,
    /// Chain is archived
    Archived,
    /// Chain is corrupted
    Corrupted,
}

impl AuditChain {
    /// Create a new audit chain
    pub fn new(max_entries: usize) -> Self {
        Self {
            chain_id: Uuid::new_v4(),
            created_at: Utc::now(),
            entries: VecDeque::new(),
            max_entries,
            metadata: std::collections::HashMap::new(),
            state: ChainState::Active,
            previous_chain_id: None,
            next_chain_id: None,
        }
    }

    /// Add an entry to the chain
    pub fn add_entry(&mut self, mut entry: AuditLogEntry) -> ComplianceResult<()> {
        if self.state != ChainState::Active {
            return Err(crate::error::ComplianceError::AuditLogError(
                "Cannot add entry to non-active chain".to_string(),
            ));
        }

        // Set previous hash from last entry
        if let Some(last_entry) = self.entries.back() {
            entry.set_previous_hash(last_entry.hash.clone());
        }

        self.entries.push_back(entry);

        // Check if rotation is needed
        if self.entries.len() >= self.max_entries {
            self.state = ChainState::Sealed;
        }

        Ok(())
    }

    /// Verify the integrity of the entire chain
    pub fn verify_integrity(&self) -> bool {
        if self.entries.is_empty() {
            return true;
        }

        let mut previous_hash: Option<String> = None;

        for entry in &self.entries {
            // Verify individual entry integrity
            if !entry.verify_integrity() {
                return false;
            }

            // Verify chain linkage
            if entry.previous_hash != previous_hash {
                return false;
            }

            previous_hash = Some(entry.hash.clone());
        }

        true
    }

    /// Seal the chain (make it read-only)
    pub fn seal(&mut self) -> ComplianceResult<()> {
        if self.state != ChainState::Active {
            return Err(crate::error::ComplianceError::AuditLogError(
                "Chain is not active".to_string(),
            ));
        }

        self.state = ChainState::Sealed;
        Ok(())
    }

    /// Archive the chain
    pub fn archive(&mut self) -> ComplianceResult<()> {
        if self.state != ChainState::Sealed {
            return Err(crate::error::ComplianceError::AuditLogError(
                "Chain must be sealed before archiving".to_string(),
            ));
        }

        self.state = ChainState::Archived;
        Ok(())
    }

    /// Get the number of entries in the chain
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// Check if the chain is full
    pub fn is_full(&self) -> bool {
        self.entries.len() >= self.max_entries
    }

    /// Get entries within a time range
    pub fn get_entries_in_range(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Vec<&AuditLogEntry> {
        self.entries
            .iter()
            .filter(|entry| entry.timestamp >= start && entry.timestamp <= end)
            .collect()
    }

    /// Create a continuation chain
    pub fn create_continuation(&self) -> Self {
        let mut new_chain = Self::new(self.max_entries);
        new_chain.previous_chain_id = Some(self.chain_id);
        new_chain
    }
}
