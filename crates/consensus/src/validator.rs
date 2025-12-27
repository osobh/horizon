//! Validator management for consensus protocol

use crate::error::{ConsensusError, ConsensusResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use uuid::Uuid;

/// Unique identifier for validators
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ValidatorId(pub Uuid);

impl ValidatorId {
    /// Create new validator ID
    #[must_use = "ValidatorId must be stored for validator identification"]
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    /// Create from UUID
    #[must_use = "ValidatorId must be stored for validator identification"]
    pub fn from_uuid(uuid: Uuid) -> Self {
        Self(uuid)
    }

    /// Get underlying UUID
    pub fn as_uuid(&self) -> Uuid {
        self.0
    }
}

impl Default for ValidatorId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for ValidatorId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Validator information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatorInfo {
    /// Validator ID
    pub id: ValidatorId,
    /// Network address
    pub address: SocketAddr,
    /// Stake weight (for weighted voting)
    pub stake: u64,
    /// GPU compute capacity
    pub gpu_capacity: u32,
    /// Current status
    pub status: ValidatorStatus,
    /// Last heartbeat timestamp
    pub last_heartbeat: u64,
    /// Public key for signature verification
    pub public_key: Vec<u8>,
}

/// Validator operational status
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ValidatorStatus {
    /// Active and participating
    Active,
    /// Temporarily offline
    Offline,
    /// Suspected Byzantine behavior
    Suspected,
    /// Confirmed Byzantine, excluded from consensus
    Byzantine,
    /// Voluntarily leaving the network
    Leaving,
}

/// Validator registry and management
#[derive(Debug)]
pub struct Validator {
    /// This validator's info
    info: ValidatorInfo,
    /// Registry of known validators
    validators: HashMap<ValidatorId, ValidatorInfo>,
    /// Minimum stake required
    min_stake: u64,
    /// Maximum offline duration before marking as suspicious
    max_offline_duration: Duration,
}

impl Validator {
    /// Create new validator
    #[must_use = "ignoring the Result may hide validator creation errors"]
    pub fn new(
        address: SocketAddr,
        stake: u64,
        gpu_capacity: u32,
        public_key: Vec<u8>,
    ) -> ConsensusResult<Self> {
        if stake == 0 {
            return Err(ConsensusError::ConfigError(
                "Stake must be greater than zero".to_string(),
            ));
        }

        let info = ValidatorInfo {
            id: ValidatorId::new(),
            address,
            stake,
            gpu_capacity,
            status: ValidatorStatus::Active,
            last_heartbeat: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            public_key,
        };

        Ok(Self {
            info: info.clone(),
            validators: {
                let mut map = HashMap::new();
                map.insert(info.id.clone(), info);
                map
            },
            min_stake: 1,
            max_offline_duration: Duration::from_secs(300), // 5 minutes
        })
    }

    /// Get this validator's ID
    pub fn id(&self) -> &ValidatorId {
        &self.info.id
    }

    /// Get this validator's info
    pub fn info(&self) -> &ValidatorInfo {
        &self.info
    }

    /// Register a new validator
    pub fn register_validator(&mut self, validator_info: ValidatorInfo) -> ConsensusResult<()> {
        if validator_info.stake < self.min_stake {
            return Err(ConsensusError::ConfigError(format!(
                "Insufficient stake: {} < {}",
                validator_info.stake, self.min_stake
            )));
        }

        if self.validators.contains_key(&validator_info.id) {
            return Err(ConsensusError::ConfigError(
                "Validator already registered".to_string(),
            ));
        }

        self.validators
            .insert(validator_info.id.clone(), validator_info);
        Ok(())
    }

    /// Remove validator from registry
    pub fn remove_validator(&mut self, validator_id: &ValidatorId) -> ConsensusResult<()> {
        if validator_id == &self.info.id {
            return Err(ConsensusError::ConfigError(
                "Cannot remove self from registry".to_string(),
            ));
        }

        self.validators.remove(validator_id);
        Ok(())
    }

    /// Update validator status
    pub fn update_validator_status(
        &mut self,
        validator_id: &ValidatorId,
        status: ValidatorStatus,
    ) -> ConsensusResult<()> {
        if let Some(validator) = self.validators.get_mut(validator_id) {
            validator.status = status;
            Ok(())
        } else {
            Err(ConsensusError::ValidationFailed(format!(
                "Validator not found: {}",
                validator_id
            )))
        }
    }

    /// Record heartbeat from validator
    pub fn record_heartbeat(&mut self, validator_id: &ValidatorId) -> ConsensusResult<()> {
        if let Some(validator) = self.validators.get_mut(validator_id) {
            validator.last_heartbeat = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs();

            // Update status if was offline
            if validator.status == ValidatorStatus::Offline {
                validator.status = ValidatorStatus::Active;
            }

            Ok(())
        } else {
            Err(ConsensusError::ValidationFailed(format!(
                "Validator not found: {}",
                validator_id
            )))
        }
    }

    /// Get all active validators
    pub fn active_validators(&self) -> Vec<&ValidatorInfo> {
        self.validators
            .values()
            .filter(|v| v.status == ValidatorStatus::Active)
            .collect()
    }

    /// Get validator by ID
    pub fn get_validator(&self, validator_id: &ValidatorId) -> Option<&ValidatorInfo> {
        self.validators.get(validator_id)
    }

    /// Check for offline validators and update their status
    pub fn check_offline_validators(&mut self) -> Vec<ValidatorId> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let max_offline_secs = self.max_offline_duration.as_secs();

        let mut offline_validators = Vec::new();

        for (id, validator) in self.validators.iter_mut() {
            if validator.status == ValidatorStatus::Active {
                let time_since_heartbeat = now.saturating_sub(validator.last_heartbeat);
                if time_since_heartbeat > max_offline_secs {
                    validator.status = ValidatorStatus::Offline;
                    offline_validators.push(id.clone());
                }
            }
        }

        offline_validators
    }

    /// Calculate total stake of active validators
    pub fn total_active_stake(&self) -> u64 {
        self.active_validators().iter().map(|v| v.stake).sum()
    }

    /// Calculate voting threshold (2/3 + 1 of total stake)
    pub fn voting_threshold(&self) -> u64 {
        let total = self.total_active_stake();
        (total * 2) / 3 + 1
    }

    /// Check if enough validators are active for consensus
    pub fn has_consensus_quorum(&self) -> bool {
        let active_count = self.active_validators().len();
        let total_count = self.validators.len();

        // For single validator networks, always have quorum
        if total_count <= 1 {
            return active_count >= 1;
        }

        // Need more than 2/3 of total validators active (Byzantine fault tolerance)
        // This means we can tolerate up to 1/3 Byzantine validators
        let required = (total_count * 2 + 2) / 3; // Ceiling of (2/3 * total)
        active_count >= required
    }

    /// Mark validator as Byzantine
    pub fn mark_byzantine(
        &mut self,
        validator_id: &ValidatorId,
        reason: String,
    ) -> ConsensusResult<()> {
        if let Some(validator) = self.validators.get_mut(validator_id) {
            validator.status = ValidatorStatus::Byzantine;
            tracing::warn!("Validator {} marked as Byzantine: {}", validator_id, reason);
            Ok(())
        } else {
            Err(ConsensusError::ValidationFailed(format!(
                "Validator not found: {}",
                validator_id
            )))
        }
    }

    /// Get validator count by status
    pub fn validator_count_by_status(&self, status: ValidatorStatus) -> usize {
        self.validators
            .values()
            .filter(|v| v.status == status)
            .count()
    }

    /// Set minimum stake requirement
    pub fn set_min_stake(&mut self, min_stake: u64) {
        self.min_stake = min_stake;
    }

    /// Set maximum offline duration
    pub fn set_max_offline_duration(&mut self, duration: Duration) {
        self.max_offline_duration = duration;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_validator() -> Validator {
        Validator::new(
            "127.0.0.1:8080".parse().unwrap(),
            100,
            1000,
            vec![1, 2, 3, 4],
        )
        .unwrap()
    }

    #[test]
    fn test_validator_id_creation() {
        let id1 = ValidatorId::new();
        let id2 = ValidatorId::new();
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_validator_id_from_uuid() {
        let uuid = Uuid::new_v4();
        let id = ValidatorId::from_uuid(uuid);
        assert_eq!(id.as_uuid(), uuid);
    }

    #[test]
    fn test_validator_creation() {
        let validator = create_test_validator();
        assert_eq!(validator.info().stake, 100);
        assert_eq!(validator.info().gpu_capacity, 1000);
        assert_eq!(validator.info().status, ValidatorStatus::Active);
    }

    #[test]
    fn test_validator_creation_zero_stake() {
        let result = Validator::new(
            "127.0.0.1:8080".parse().unwrap(),
            0, // Zero stake should fail
            1000,
            vec![1, 2, 3, 4],
        );
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ConsensusError::ConfigError(_)
        ));
    }

    #[test]
    fn test_register_validator() {
        let mut validator = create_test_validator();

        let new_validator = ValidatorInfo {
            id: ValidatorId::new(),
            address: "127.0.0.1:8081".parse().unwrap(),
            stake: 50,
            gpu_capacity: 500,
            status: ValidatorStatus::Active,
            last_heartbeat: 0,
            public_key: vec![5, 6, 7, 8],
        };

        assert!(validator.register_validator(new_validator.clone()).is_ok());
        assert_eq!(validator.active_validators().len(), 2);
    }

    #[test]
    fn test_register_validator_insufficient_stake() {
        let mut validator = create_test_validator();
        validator.set_min_stake(100);

        let new_validator = ValidatorInfo {
            id: ValidatorId::new(),
            address: "127.0.0.1:8081".parse().unwrap(),
            stake: 50, // Below minimum
            gpu_capacity: 500,
            status: ValidatorStatus::Active,
            last_heartbeat: 0,
            public_key: vec![5, 6, 7, 8],
        };

        let result = validator.register_validator(new_validator);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ConsensusError::ConfigError(_)
        ));
    }

    #[test]
    fn test_register_duplicate_validator() {
        let mut validator = create_test_validator();

        let duplicate = validator.info().clone();
        let result = validator.register_validator(duplicate);

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ConsensusError::ConfigError(_)
        ));
    }

    #[test]
    fn test_remove_validator() {
        let mut validator = create_test_validator();

        let new_id = ValidatorId::new();
        let new_validator = ValidatorInfo {
            id: new_id.clone(),
            address: "127.0.0.1:8081".parse().unwrap(),
            stake: 50,
            gpu_capacity: 500,
            status: ValidatorStatus::Active,
            last_heartbeat: 0,
            public_key: vec![5, 6, 7, 8],
        };

        validator.register_validator(new_validator).unwrap();
        assert_eq!(validator.active_validators().len(), 2);

        validator.remove_validator(&new_id).unwrap();
        assert_eq!(validator.active_validators().len(), 1);
    }

    #[test]
    fn test_remove_self_validator() {
        let mut validator = create_test_validator();
        let self_id = validator.id().clone();

        let result = validator.remove_validator(&self_id);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ConsensusError::ConfigError(_)
        ));
    }

    #[test]
    fn test_update_validator_status() {
        let mut validator = create_test_validator();
        let validator_id = validator.id().clone();

        validator
            .update_validator_status(&validator_id, ValidatorStatus::Offline)
            .unwrap();

        let info = validator.get_validator(&validator_id).unwrap();
        assert_eq!(info.status, ValidatorStatus::Offline);
    }

    #[test]
    fn test_record_heartbeat() {
        let mut validator = create_test_validator();
        let validator_id = validator.id().clone();

        // Mark as offline first
        validator
            .update_validator_status(&validator_id, ValidatorStatus::Offline)
            .unwrap();

        // Record heartbeat should make it active again
        validator.record_heartbeat(&validator_id).unwrap();

        let info = validator.get_validator(&validator_id).unwrap();
        assert_eq!(info.status, ValidatorStatus::Active);
    }

    #[test]
    fn test_voting_threshold() {
        let mut validator = create_test_validator();

        // Add more validators
        for i in 0..3 {
            let new_validator = ValidatorInfo {
                id: ValidatorId::new(),
                address: format!("127.0.0.1:808{}", i + 1).parse().unwrap(),
                stake: 100,
                gpu_capacity: 500,
                status: ValidatorStatus::Active,
                last_heartbeat: 0,
                public_key: vec![i as u8],
            };
            validator.register_validator(new_validator).unwrap();
        }

        let total_stake = validator.total_active_stake();
        let threshold = validator.voting_threshold();

        assert_eq!(total_stake, 400); // 4 validators * 100 stake
        assert_eq!(threshold, 267); // (400 * 2) / 3 + 1
    }

    #[test]
    fn test_consensus_quorum() {
        let mut validator = create_test_validator();

        // Add 2 more validators (total 3)
        for i in 0..2 {
            let new_validator = ValidatorInfo {
                id: ValidatorId::new(),
                address: format!("127.0.0.1:808{}", i + 1).parse().unwrap(),
                stake: 100,
                gpu_capacity: 500,
                status: ValidatorStatus::Active,
                last_heartbeat: 0,
                public_key: vec![i as u8],
            };
            validator.register_validator(new_validator).unwrap();
        }

        assert!(validator.has_consensus_quorum()); // 3/3 active

        // Mark one as offline
        let validators: Vec<_> = validator
            .active_validators()
            .iter()
            .map(|v| v.id.clone())
            .collect();
        validator
            .update_validator_status(&validators[1], ValidatorStatus::Offline)
            .unwrap();

        assert!(validator.has_consensus_quorum()); // 2/3 active, still quorum

        // Mark another as offline
        validator
            .update_validator_status(&validators[2], ValidatorStatus::Offline)
            .unwrap();

        assert!(!validator.has_consensus_quorum()); // 1/3 active, no quorum
    }

    #[test]
    fn test_mark_byzantine() {
        let mut validator = create_test_validator();
        let validator_id = validator.id().clone();

        validator
            .mark_byzantine(&validator_id, "Double voting detected".to_string())
            .unwrap();

        let info = validator.get_validator(&validator_id).unwrap();
        assert_eq!(info.status, ValidatorStatus::Byzantine);
    }

    #[test]
    fn test_validator_count_by_status() {
        let mut validator = create_test_validator();

        // Add validators with different statuses
        for (i, status) in [ValidatorStatus::Offline, ValidatorStatus::Byzantine]
            .iter()
            .enumerate()
        {
            let new_validator = ValidatorInfo {
                id: ValidatorId::new(),
                address: format!("127.0.0.1:808{}", i + 1).parse().unwrap(),
                stake: 100,
                gpu_capacity: 500,
                status: status.clone(),
                last_heartbeat: 0,
                public_key: vec![i as u8],
            };
            validator.register_validator(new_validator).unwrap();
        }

        assert_eq!(
            validator.validator_count_by_status(ValidatorStatus::Active),
            1
        );
        assert_eq!(
            validator.validator_count_by_status(ValidatorStatus::Offline),
            1
        );
        assert_eq!(
            validator.validator_count_by_status(ValidatorStatus::Byzantine),
            1
        );
    }

    #[test]
    fn test_check_offline_validators() {
        let mut validator = create_test_validator();
        validator.set_max_offline_duration(Duration::from_millis(100));

        // Add another validator
        let new_id = ValidatorId::new();
        let mut new_validator = ValidatorInfo {
            id: new_id.clone(),
            address: "127.0.0.1:8081".parse().unwrap(),
            stake: 100,
            gpu_capacity: 500,
            status: ValidatorStatus::Active,
            last_heartbeat: 0, // Old timestamp
            public_key: vec![1],
        };

        validator.register_validator(new_validator).unwrap();

        // Wait for timeout
        std::thread::sleep(Duration::from_millis(150));

        let offline = validator.check_offline_validators();
        assert!(offline.contains(&new_id));

        let info = validator.get_validator(&new_id).unwrap();
        assert_eq!(info.status, ValidatorStatus::Offline);
    }

    #[test]
    fn test_validator_id_display() {
        let id = ValidatorId::new();
        let display = format!("{id}");
        assert_eq!(display, id.as_uuid().to_string());
    }

    #[test]
    fn test_validator_status_serialization() {
        let statuses = vec![
            ValidatorStatus::Active,
            ValidatorStatus::Offline,
            ValidatorStatus::Suspected,
            ValidatorStatus::Byzantine,
            ValidatorStatus::Leaving,
        ];

        for status in statuses {
            let serialized = serde_json::to_string(&status).unwrap();
            let deserialized: ValidatorStatus = serde_json::from_str(&serialized).unwrap();
            assert_eq!(status, deserialized);
        }
    }
}
