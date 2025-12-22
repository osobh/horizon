//! Integration tests for secure multi-party protocols

use async_trait::async_trait;
use chrono::Utc;
use exorust_evolution_global::{
    error::EvolutionGlobalResult,
    secure_multiparty::{
        CryptographicProvider, EncryptedEvolutionData, EncryptionScheme, MPCParticipant,
        MPCProtocol, SecureAggregationResult, SecureAggregator, SecureMultiPartyConfig,
        SecureMultiPartyManager, SessionStatus, ZKProof, ZKProofType,
    },
};
use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

/// Mock cryptographic provider for integration tests
struct IntegrationCryptoProvider;

#[async_trait]
impl CryptographicProvider for IntegrationCryptoProvider {
    async fn encrypt_data(
        &self,
        data: &[u8],
        scheme: EncryptionScheme,
        _key: &[u8],
    ) -> EvolutionGlobalResult<Vec<u8>> {
        // Simple mock encryption: reverse bytes and add scheme marker
        let mut encrypted = data.to_vec();
        encrypted.reverse();
        encrypted.push(match scheme {
            EncryptionScheme::AES256GCM => 1,
            EncryptionScheme::ChaCha20Poly1305 => 2,
            _ => 0,
        });
        Ok(encrypted)
    }

    async fn decrypt_data(
        &self,
        encrypted_data: &[u8],
        _scheme: EncryptionScheme,
        _key: &[u8],
    ) -> EvolutionGlobalResult<Vec<u8>> {
        // Simple mock decryption: remove marker and reverse
        let mut data = encrypted_data[..encrypted_data.len() - 1].to_vec();
        data.reverse();
        Ok(data)
    }

    async fn generate_zk_proof(
        &self,
        proof_type: ZKProofType,
        private_inputs: &[u8],
        public_inputs: &[u8],
    ) -> EvolutionGlobalResult<ZKProof> {
        Ok(ZKProof {
            proof_type,
            proof_data: private_inputs.to_vec(),
            public_inputs: public_inputs.to_vec(),
            verification_key: vec![1, 2, 3, 4],
            created_at: Utc::now(),
        })
    }

    async fn verify_zk_proof(&self, _proof: &ZKProof) -> EvolutionGlobalResult<bool> {
        Ok(true) // Always valid in tests
    }
}

/// Mock secure aggregator for integration tests
struct IntegrationSecureAggregator;

#[async_trait]
impl SecureAggregator for IntegrationSecureAggregator {
    async fn aggregate_encrypted_data(
        &self,
        encrypted_inputs: Vec<EncryptedEvolutionData>,
        aggregation_function: String,
    ) -> EvolutionGlobalResult<SecureAggregationResult> {
        // Simple mock aggregation
        let aggregated_data = vec![encrypted_inputs.len() as u8];

        Ok(SecureAggregationResult {
            aggregation_id: Uuid::new_v4(),
            session_id: Uuid::new_v4(),
            aggregated_data,
            participant_count: encrypted_inputs.len(),
            noise_level: 0.1,
            privacy_budget: 1.0,
            validation_proofs: vec![],
            timestamp: Utc::now(),
        })
    }

    async fn add_differential_privacy_noise(
        &self,
        data: &[u8],
        epsilon: f64,
        _delta: f64,
    ) -> EvolutionGlobalResult<Vec<u8>> {
        // Simple mock: add small noise based on epsilon
        let mut noisy_data = data.to_vec();
        for byte in &mut noisy_data {
            *byte = byte.wrapping_add((epsilon * 10.0) as u8);
        }
        Ok(noisy_data)
    }
}

fn create_test_participant(id: &str, trust_score: f64) -> MPCParticipant {
    MPCParticipant {
        participant_id: Uuid::new_v4(),
        public_key: vec![1, 2, 3, 4],
        endpoint: format!("https://{}.example.com", id),
        capabilities: vec![
            MPCProtocol::SecretSharing,
            MPCProtocol::HomomorphicEncryption,
            MPCProtocol::SecureAggregation,
        ],
        trust_score,
        last_seen: Utc::now(),
        active: true,
    }
}

#[tokio::test]
async fn test_participant_registration_workflow() {
    let config = SecureMultiPartyConfig::default();
    let crypto_provider = Arc::new(IntegrationCryptoProvider);
    let secure_aggregator = Arc::new(IntegrationSecureAggregator);

    let manager = SecureMultiPartyManager::new(config, crypto_provider, secure_aggregator).unwrap();

    // Register multiple participants
    let participants = vec![
        create_test_participant("alice", 0.9),
        create_test_participant("bob", 0.85),
        create_test_participant("charlie", 0.8),
    ];

    for participant in &participants {
        manager
            .register_participant(participant.clone())
            .await
            .unwrap();
    }

    // Get active participants
    let active = manager.get_active_participants().await.unwrap();
    assert_eq!(active.len(), 3);

    // Try to register participant with low trust score
    let untrusted = create_test_participant("mallory", 0.5);
    let result = manager.register_participant(untrusted).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_secure_session_creation() {
    let config = SecureMultiPartyConfig::default();
    let crypto_provider = Arc::new(IntegrationCryptoProvider);
    let secure_aggregator = Arc::new(IntegrationSecureAggregator);

    let manager = SecureMultiPartyManager::new(config, crypto_provider, secure_aggregator).unwrap();

    // Register participants
    let mut participant_ids = Vec::new();
    for i in 0..4 {
        let participant = create_test_participant(&format!("participant{}", i), 0.8);
        participant_ids.push(participant.participant_id);
        manager.register_participant(participant).await.unwrap();
    }

    // Create session with sufficient participants
    let session_id = manager
        .create_secure_session(MPCProtocol::SecretSharing, participant_ids[..3].to_vec())
        .await
        .unwrap();

    // Check session status
    let status = manager.get_session_status(session_id).await.unwrap();
    assert_eq!(status, Some(SessionStatus::Initializing));
}

#[tokio::test]
async fn test_insufficient_participants_error() {
    let config = SecureMultiPartyConfig {
        min_participants: 5,
        ..Default::default()
    };
    let crypto_provider = Arc::new(IntegrationCryptoProvider);
    let secure_aggregator = Arc::new(IntegrationSecureAggregator);

    let manager = SecureMultiPartyManager::new(config, crypto_provider, secure_aggregator).unwrap();

    // Try to create session with insufficient participants
    let participants = vec![Uuid::new_v4(), Uuid::new_v4(), Uuid::new_v4()];
    let result = manager
        .create_secure_session(MPCProtocol::SecureAggregation, participants)
        .await;

    assert!(result.is_err());
}

#[tokio::test]
async fn test_participant_removal() {
    let config = SecureMultiPartyConfig::default();
    let crypto_provider = Arc::new(IntegrationCryptoProvider);
    let secure_aggregator = Arc::new(IntegrationSecureAggregator);

    let manager = SecureMultiPartyManager::new(config, crypto_provider, secure_aggregator).unwrap();

    // Register participant
    let participant = create_test_participant("removable", 0.9);
    let participant_id = participant.participant_id;
    manager.register_participant(participant).await.unwrap();

    // Verify participant exists
    let active = manager.get_active_participants().await.unwrap();
    assert_eq!(active.len(), 1);

    // Remove participant
    manager.remove_participant(participant_id).await.unwrap();

    // Verify participant removed
    let active = manager.get_active_participants().await.unwrap();
    assert_eq!(active.len(), 0);

    // Try to remove non-existent participant
    let result = manager.remove_participant(Uuid::new_v4()).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_disabled_multiparty_system() {
    let config = SecureMultiPartyConfig {
        enabled: false,
        ..Default::default()
    };
    let crypto_provider = Arc::new(IntegrationCryptoProvider);
    let secure_aggregator = Arc::new(IntegrationSecureAggregator);

    let manager = SecureMultiPartyManager::new(config, crypto_provider, secure_aggregator).unwrap();

    // Try to register participant with disabled system
    let participant = create_test_participant("test", 0.9);
    let result = manager.register_participant(participant).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_custom_configuration() {
    let config = SecureMultiPartyConfig {
        min_participants: 2,
        max_participants: 20,
        session_timeout_minutes: 120,
        key_rotation_interval_hours: 6,
        require_zk_proofs: false,
        differential_privacy_enabled: false,
        default_epsilon: 0.5,
        min_trust_score: 0.6,
        ..Default::default()
    };
    let crypto_provider = Arc::new(IntegrationCryptoProvider);
    let secure_aggregator = Arc::new(IntegrationSecureAggregator);

    let manager = SecureMultiPartyManager::new(config, crypto_provider, secure_aggregator).unwrap();

    // Register participant with lower trust score (above new minimum)
    let participant = create_test_participant("low-trust", 0.65);
    manager.register_participant(participant).await.unwrap();

    // Create session with only 2 participants
    let participants = vec![Uuid::new_v4(), Uuid::new_v4()];
    let result = manager
        .create_secure_session(MPCProtocol::SecretSharing, participants)
        .await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_multiple_protocol_support() {
    let config = SecureMultiPartyConfig::default();
    let crypto_provider = Arc::new(IntegrationCryptoProvider);
    let secure_aggregator = Arc::new(IntegrationSecureAggregator);

    let manager = SecureMultiPartyManager::new(config, crypto_provider, secure_aggregator).unwrap();

    // Register participants with different capabilities
    let protocols = vec![
        MPCProtocol::SecretSharing,
        MPCProtocol::HomomorphicEncryption,
        MPCProtocol::GarbledCircuits,
        MPCProtocol::ObliviousTransfer,
        MPCProtocol::PrivateSetIntersection,
        MPCProtocol::SecureAggregation,
    ];

    let mut participant_ids = Vec::new();
    for (i, protocol) in protocols.iter().enumerate() {
        let mut participant = create_test_participant(&format!("participant{}", i), 0.8);
        participant.capabilities = vec![protocol.clone()];
        participant_ids.push(participant.participant_id);
        manager.register_participant(participant).await.unwrap();
    }

    // Create sessions with different protocols
    for protocol in &protocols[..3] {
        let session_id = manager
            .create_secure_session(protocol.clone(), participant_ids[..3].to_vec())
            .await
            .unwrap();
        assert!(session_id != Uuid::nil());
    }
}
