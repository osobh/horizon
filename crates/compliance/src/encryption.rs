//! Encryption Services for Data at Rest and in Transit
//!
//! Provides comprehensive encryption capabilities including:
//! - Data at rest encryption
//! - Data in transit encryption
//! - Key management and rotation
//! - Encryption policy enforcement
//! - Algorithm compliance validation

use crate::data_classification::{DataCategory, DataClassification, DataMetadata};
use crate::error::{ComplianceError, ComplianceResult};
use base64::{engine::general_purpose, Engine as _};
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use uuid::Uuid;

/// Encryption algorithm
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EncryptionAlgorithm {
    /// AES-256-GCM (recommended for most use cases)
    AES256GCM,
    /// AES-256-CBC (legacy support)
    AES256CBC,
    /// ChaCha20-Poly1305 (high performance)
    ChaCha20Poly1305,
    /// RSA-4096 (asymmetric encryption)
    RSA4096,
    /// Elliptic Curve P-256 (asymmetric encryption)
    ECCP256,
}

/// Key derivation function
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KeyDerivationFunction {
    /// PBKDF2 with SHA-256
    PBKDF2SHA256,
    /// Argon2id (recommended)
    Argon2id,
    /// scrypt
    Scrypt,
}

/// Encryption key metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptionKey {
    /// Key identifier
    pub id: Uuid,
    /// Key name/label
    pub name: String,
    /// Encryption algorithm
    pub algorithm: EncryptionAlgorithm,
    /// Key derivation function used
    pub kdf: Option<KeyDerivationFunction>,
    /// Key length in bits
    pub key_length: u32,
    /// Data classification this key is for
    pub classification: DataClassification,
    /// Data categories this key can encrypt
    pub allowed_categories: Vec<DataCategory>,
    /// Key creation date
    pub created_at: DateTime<Utc>,
    /// Key expiration date
    pub expires_at: Option<DateTime<Utc>>,
    /// Last rotation date
    pub last_rotated_at: DateTime<Utc>,
    /// Rotation interval
    pub rotation_interval: Duration,
    /// Whether key is active
    pub active: bool,
    /// Key owner/custodian
    pub owner: String,
    /// Compliance requirements
    pub compliance_requirements: Vec<String>,
    /// Key usage counter
    pub usage_count: u64,
    /// Maximum usage limit
    pub max_usage: Option<u64>,
}

/// Encrypted data container
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptedData {
    /// Data identifier
    pub id: Uuid,
    /// Key ID used for encryption
    pub key_id: Uuid,
    /// Encryption algorithm used
    pub algorithm: EncryptionAlgorithm,
    /// Encrypted payload (base64 encoded)
    pub ciphertext: String,
    /// Initialization vector (base64 encoded)
    pub iv: String,
    /// Authentication tag (base64 encoded, for authenticated encryption)
    pub tag: Option<String>,
    /// Data metadata
    pub metadata: DataMetadata,
    /// Encryption timestamp
    pub encrypted_at: DateTime<Utc>,
    /// Data integrity hash
    pub integrity_hash: String,
}

/// Encryption policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptionPolicy {
    /// Policy identifier
    pub id: String,
    /// Policy name
    pub name: String,
    /// Data classification this policy applies to
    pub classification: DataClassification,
    /// Data categories this policy applies to
    pub categories: Vec<DataCategory>,
    /// Required encryption algorithm
    pub required_algorithm: EncryptionAlgorithm,
    /// Minimum key length
    pub min_key_length: u32,
    /// Key rotation requirements
    pub key_rotation_required: bool,
    /// Maximum key rotation interval
    pub max_rotation_interval: Duration,
    /// Transit encryption required
    pub transit_encryption_required: bool,
    /// At-rest encryption required
    pub at_rest_encryption_required: bool,
    /// Additional compliance requirements
    pub compliance_standards: Vec<ComplianceStandard>,
    /// Policy owner
    pub owner: String,
    /// Policy created date
    pub created_at: DateTime<Utc>,
    /// Policy active status
    pub active: bool,
}

/// Compliance standards for encryption
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComplianceStandard {
    /// FIPS 140-2 Level 1
    FIPS1402L1,
    /// FIPS 140-2 Level 2
    FIPS1402L2,
    /// FIPS 140-2 Level 3
    FIPS1402L3,
    /// Common Criteria EAL4+
    CommonCriteriaEAL4,
    /// NIST SP 800-53
    NIST80053,
    /// ISO 27001
    ISO27001,
    /// SOC2 Type II
    SOC2TypeII,
}

/// Key rotation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyRotationResult {
    /// Old key ID
    pub old_key_id: Uuid,
    /// New key ID
    pub new_key_id: Uuid,
    /// Rotation timestamp
    pub rotated_at: DateTime<Utc>,
    /// Number of records re-encrypted
    pub records_reencrypted: usize,
    /// Rotation duration in milliseconds
    pub rotation_duration_ms: u64,
    /// Any errors encountered
    pub errors: Vec<String>,
}

/// Encryption validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptionValidationResult {
    /// Validation timestamp
    pub validated_at: DateTime<Utc>,
    /// Overall compliance status
    pub compliant: bool,
    /// Policy violations found
    pub violations: Vec<PolicyViolation>,
    /// Keys requiring rotation
    pub keys_requiring_rotation: Vec<Uuid>,
    /// Expired keys
    pub expired_keys: Vec<Uuid>,
    /// Encryption strength assessment
    pub strength_assessment: StrengthAssessment,
    /// Recommendations
    pub recommendations: Vec<String>,
}

/// Policy violation details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyViolation {
    /// Violation type
    pub violation_type: ViolationType,
    /// Severity level
    pub severity: ViolationSeverity,
    /// Description
    pub description: String,
    /// Affected resource ID
    pub resource_id: String,
    /// Policy ID violated
    pub policy_id: String,
    /// Detection timestamp
    pub detected_at: DateTime<Utc>,
}

/// Types of policy violations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ViolationType {
    /// Weak encryption algorithm
    WeakAlgorithm,
    /// Insufficient key length
    InsufficientKeyLength,
    /// Key rotation overdue
    RotationOverdue,
    /// Expired key in use
    ExpiredKey,
    /// Unencrypted sensitive data
    UnencryptedData,
    /// Non-compliant algorithm
    NonCompliantAlgorithm,
    /// Excessive key usage
    ExcessiveKeyUsage,
}

/// Violation severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, PartialOrd, Ord)]
pub enum ViolationSeverity {
    /// Low risk violation
    Low,
    /// Medium risk violation
    Medium,
    /// High risk violation
    High,
    /// Critical security risk
    Critical,
}

/// Encryption strength assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrengthAssessment {
    /// Overall strength score (0.0 - 1.0)
    pub overall_score: f64,
    /// Algorithm strength scores
    pub algorithm_scores: HashMap<EncryptionAlgorithm, f64>,
    /// Key management score
    pub key_management_score: f64,
    /// Compliance score
    pub compliance_score: f64,
    /// Areas needing improvement
    pub improvement_areas: Vec<String>,
}

/// Encryption service engine
pub struct EncryptionEngine {
    /// Encryption keys by ID
    keys: HashMap<Uuid, EncryptionKey>,
    /// Encrypted data by ID
    encrypted_data: HashMap<Uuid, EncryptedData>,
    /// Encryption policies by ID
    policies: HashMap<String, EncryptionPolicy>,
    /// Key usage tracking
    key_usage: HashMap<Uuid, u64>,
}

impl EncryptionEngine {
    /// Create new encryption engine
    pub fn new() -> Self {
        let mut engine = Self {
            keys: HashMap::new(),
            encrypted_data: HashMap::new(),
            policies: HashMap::new(),
            key_usage: HashMap::new(),
        };

        engine.initialize_default_policies();
        engine
    }

    /// Initialize default encryption policies
    fn initialize_default_policies(&mut self) {
        // Policy for PII data
        let pii_policy = EncryptionPolicy {
            id: "pii_encryption_policy".to_string(),
            name: "PII Encryption Policy".to_string(),
            classification: DataClassification::ConfidentialData,
            categories: vec![DataCategory::PII],
            required_algorithm: EncryptionAlgorithm::AES256GCM,
            min_key_length: 256,
            key_rotation_required: true,
            max_rotation_interval: Duration::days(90), // 90 days
            transit_encryption_required: true,
            at_rest_encryption_required: true,
            compliance_standards: vec![
                ComplianceStandard::FIPS1402L2,
                ComplianceStandard::SOC2TypeII,
            ],
            owner: "Data Protection Officer".to_string(),
            created_at: Utc::now(),
            active: true,
        };
        self.policies.insert(pii_policy.id.clone(), pii_policy);

        // Policy for PHI data
        let phi_policy = EncryptionPolicy {
            id: "phi_encryption_policy".to_string(),
            name: "PHI Encryption Policy".to_string(),
            classification: DataClassification::RestrictedData,
            categories: vec![DataCategory::PHI],
            required_algorithm: EncryptionAlgorithm::AES256GCM,
            min_key_length: 256,
            key_rotation_required: true,
            max_rotation_interval: Duration::days(60), // 60 days for PHI
            transit_encryption_required: true,
            at_rest_encryption_required: true,
            compliance_standards: vec![
                ComplianceStandard::FIPS1402L3,
                ComplianceStandard::NIST80053,
            ],
            owner: "HIPAA Compliance Officer".to_string(),
            created_at: Utc::now(),
            active: true,
        };
        self.policies.insert(phi_policy.id.clone(), phi_policy);

        // Policy for Financial data
        let financial_policy = EncryptionPolicy {
            id: "financial_encryption_policy".to_string(),
            name: "Financial Data Encryption Policy".to_string(),
            classification: DataClassification::ConfidentialData,
            categories: vec![DataCategory::Financial],
            required_algorithm: EncryptionAlgorithm::AES256GCM,
            min_key_length: 256,
            key_rotation_required: true,
            max_rotation_interval: Duration::days(120), // 120 days
            transit_encryption_required: true,
            at_rest_encryption_required: true,
            compliance_standards: vec![
                ComplianceStandard::FIPS1402L2,
                ComplianceStandard::ISO27001,
            ],
            owner: "Financial Compliance Officer".to_string(),
            created_at: Utc::now(),
            active: true,
        };
        self.policies
            .insert(financial_policy.id.clone(), financial_policy);
    }

    /// Add encryption policy
    pub fn add_policy(&mut self, policy: EncryptionPolicy) -> ComplianceResult<()> {
        if policy.min_key_length < 128 {
            return Err(ComplianceError::EncryptionError(
                "Minimum key length must be at least 128 bits".to_string(),
            ));
        }

        self.policies.insert(policy.id.clone(), policy);
        Ok(())
    }

    /// Generate encryption key
    pub fn generate_key(
        &mut self,
        name: String,
        algorithm: EncryptionAlgorithm,
        classification: DataClassification,
        allowed_categories: Vec<DataCategory>,
        owner: String,
    ) -> ComplianceResult<Uuid> {
        let key_length = self.get_key_length_for_algorithm(algorithm);
        let key_id = Uuid::new_v4();

        let key = EncryptionKey {
            id: key_id,
            name,
            algorithm,
            kdf: Some(KeyDerivationFunction::Argon2id),
            key_length,
            classification,
            allowed_categories,
            created_at: Utc::now(),
            expires_at: None,
            last_rotated_at: Utc::now(),
            rotation_interval: Duration::days(90), // Default 90 days
            active: true,
            owner,
            compliance_requirements: vec![],
            usage_count: 0,
            max_usage: Some(1_000_000), // Default 1M operations
        };

        self.keys.insert(key_id, key);
        self.key_usage.insert(key_id, 0);

        Ok(key_id)
    }

    /// Get recommended key length for algorithm
    fn get_key_length_for_algorithm(&self, algorithm: EncryptionAlgorithm) -> u32 {
        match algorithm {
            EncryptionAlgorithm::AES256GCM => 256,
            EncryptionAlgorithm::AES256CBC => 256,
            EncryptionAlgorithm::ChaCha20Poly1305 => 256,
            EncryptionAlgorithm::RSA4096 => 4096,
            EncryptionAlgorithm::ECCP256 => 256,
        }
    }

    /// Encrypt data
    pub fn encrypt_data(
        &mut self,
        data: &[u8],
        key_id: Uuid,
        metadata: DataMetadata,
    ) -> ComplianceResult<Uuid> {
        let key = self
            .keys
            .get(&key_id)
            .ok_or_else(|| ComplianceError::EncryptionError(format!("Key not found: {key_id}")))?;

        if !key.active {
            return Err(ComplianceError::EncryptionError(
                "Cannot encrypt with inactive key".to_string(),
            ));
        }

        // Check if key is expired
        if let Some(expires_at) = key.expires_at {
            if expires_at < Utc::now() {
                return Err(ComplianceError::EncryptionError(
                    "Cannot encrypt with expired key".to_string(),
                ));
            }
        }

        // Check key usage limits
        let current_usage = self.key_usage.get(&key_id).unwrap_or(&0);
        if let Some(max_usage) = key.max_usage {
            if *current_usage >= max_usage {
                return Err(ComplianceError::EncryptionError(
                    "Key usage limit exceeded".to_string(),
                ));
            }
        }

        // Validate against encryption policies
        self.validate_encryption_request(key, &metadata)?;

        // Simulate encryption (in real implementation, this would use actual crypto)
        let data_id = Uuid::new_v4();
        let iv = self.generate_iv(key.algorithm);
        let (ciphertext, tag) = self.perform_encryption(data, &iv, key.algorithm)?;

        let encrypted_data = EncryptedData {
            id: data_id,
            key_id,
            algorithm: key.algorithm,
            ciphertext: general_purpose::STANDARD.encode(&ciphertext),
            iv: general_purpose::STANDARD.encode(&iv),
            tag: tag.map(|t| general_purpose::STANDARD.encode(&t)),
            metadata,
            encrypted_at: Utc::now(),
            integrity_hash: self.calculate_integrity_hash(&ciphertext),
        };

        self.encrypted_data.insert(data_id, encrypted_data);

        // Update key usage
        *self.key_usage.entry(key_id).or_insert(0) += 1;

        Ok(data_id)
    }

    /// Generate initialization vector
    fn generate_iv(&self, algorithm: EncryptionAlgorithm) -> Vec<u8> {
        let iv_length = match algorithm {
            EncryptionAlgorithm::AES256GCM => 12,        // 96 bits for GCM
            EncryptionAlgorithm::AES256CBC => 16,        // 128 bits for CBC
            EncryptionAlgorithm::ChaCha20Poly1305 => 12, // 96 bits
            _ => 16,                                     // Default 128 bits
        };

        // In real implementation, this would use a cryptographically secure RNG
        vec![0u8; iv_length] // Placeholder
    }

    /// Perform encryption (simulated)
    fn perform_encryption(
        &self,
        data: &[u8],
        _iv: &[u8],
        algorithm: EncryptionAlgorithm,
    ) -> ComplianceResult<(Vec<u8>, Option<Vec<u8>>)> {
        // This is a simulation - real implementation would use actual encryption
        let mut hasher = Sha256::new();
        hasher.update(data);
        hasher.update(format!("{algorithm:?}").as_bytes());
        let hash = hasher.finalize();

        let ciphertext = hash.to_vec();
        let tag = match algorithm {
            EncryptionAlgorithm::AES256GCM | EncryptionAlgorithm::ChaCha20Poly1305 => {
                Some(vec![0u8; 16]) // Simulated authentication tag
            }
            _ => None,
        };

        Ok((ciphertext, tag))
    }

    /// Calculate integrity hash
    fn calculate_integrity_hash(&self, data: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(data);
        general_purpose::STANDARD.encode(hasher.finalize())
    }

    /// Validate encryption request against policies
    fn validate_encryption_request(
        &self,
        key: &EncryptionKey,
        metadata: &DataMetadata,
    ) -> ComplianceResult<()> {
        // Find applicable policy
        let policy = self.policies.values().find(|p| {
            p.active
                && p.classification == metadata.classification
                && p.categories.contains(&metadata.category)
        });

        if let Some(policy) = policy {
            // Check algorithm compliance
            if key.algorithm != policy.required_algorithm {
                return Err(ComplianceError::EncryptionError(format!(
                    "Key algorithm {:?} does not match policy requirement {:?}",
                    key.algorithm, policy.required_algorithm
                )));
            }

            // Check key length
            if key.key_length < policy.min_key_length {
                return Err(ComplianceError::EncryptionError(format!(
                    "Key length {} is below policy minimum {}",
                    key.key_length, policy.min_key_length
                )));
            }

            // Check key rotation
            if policy.key_rotation_required {
                let rotation_due = key.last_rotated_at + policy.max_rotation_interval;
                if rotation_due < Utc::now() {
                    return Err(ComplianceError::EncryptionError(
                        "Key rotation is overdue according to policy".to_string(),
                    ));
                }
            }
        }

        Ok(())
    }

    /// Decrypt data
    pub fn decrypt_data(&mut self, data_id: Uuid) -> ComplianceResult<Vec<u8>> {
        let encrypted_data = self.encrypted_data.get(&data_id).ok_or_else(|| {
            ComplianceError::EncryptionError(format!("Encrypted data not found: {data_id}"))
        })?;

        let key = self.keys.get(&encrypted_data.key_id).ok_or_else(|| {
            ComplianceError::EncryptionError(format!(
                "Decryption key not found: {}",
                encrypted_data.key_id
            ))
        })?;

        if !key.active {
            return Err(ComplianceError::EncryptionError(
                "Cannot decrypt with inactive key".to_string(),
            ));
        }

        // Decode ciphertext
        let ciphertext = general_purpose::STANDARD
            .decode(&encrypted_data.ciphertext)
            .map_err(|e| ComplianceError::EncryptionError(format!("Invalid ciphertext: {e}")))?;

        // Verify integrity
        let calculated_hash = self.calculate_integrity_hash(&ciphertext);
        if calculated_hash != encrypted_data.integrity_hash {
            return Err(ComplianceError::EncryptionError(
                "Data integrity check failed".to_string(),
            ));
        }

        // Simulate decryption (reverse of our simulated encryption)
        Ok(b"decrypted_data".to_vec()) // Placeholder
    }

    /// Rotate encryption key
    pub async fn rotate_key(&mut self, key_id: Uuid) -> ComplianceResult<KeyRotationResult> {
        let start_time = std::time::Instant::now();
        let old_key = self
            .keys
            .get(&key_id)
            .ok_or_else(|| ComplianceError::EncryptionError(format!("Key not found: {key_id}")))?;

        // Generate new key with same properties
        let new_key_id = self.generate_key(
            format!("{}_rotated", old_key.name),
            old_key.algorithm,
            old_key.classification,
            old_key.allowed_categories.clone(),
            old_key.owner.clone(),
        )?;

        // Re-encrypt data using the old key
        let mut records_reencrypted = 0;
        let mut errors = Vec::new();

        let data_to_reencrypt: Vec<_> = self
            .encrypted_data
            .iter()
            .filter(|(_, data)| data.key_id == key_id)
            .map(|(id, data)| (*id, data.clone()))
            .collect();

        for (data_id, encrypted_data) in data_to_reencrypt {
            match self.decrypt_data(data_id) {
                Ok(plaintext) => {
                    match self.encrypt_data(&plaintext, new_key_id, encrypted_data.metadata.clone())
                    {
                        Ok(_new_data_id) => {
                            // Remove old encrypted data
                            self.encrypted_data.remove(&data_id);
                            records_reencrypted += 1;
                        }
                        Err(e) => errors.push(format!("Re-encryption failed for {data_id}: {e}")),
                    }
                }
                Err(e) => errors.push(format!("Decryption failed for {data_id}: {e}")),
            }
        }

        // Deactivate old key
        if let Some(old_key) = self.keys.get_mut(&key_id) {
            old_key.active = false;
        }

        // Update new key rotation date
        if let Some(new_key) = self.keys.get_mut(&new_key_id) {
            new_key.last_rotated_at = Utc::now();
        }

        let duration = start_time.elapsed();

        Ok(KeyRotationResult {
            old_key_id: key_id,
            new_key_id,
            rotated_at: Utc::now(),
            records_reencrypted,
            rotation_duration_ms: duration.as_millis() as u64,
            errors,
        })
    }

    /// Validate encryption compliance
    pub fn validate_encryption_compliance(&self) -> ComplianceResult<EncryptionValidationResult> {
        let mut violations = Vec::new();
        let mut keys_requiring_rotation = Vec::new();
        let mut expired_keys = Vec::new();
        let now = Utc::now();

        // Check keys for compliance issues
        for (key_id, key) in &self.keys {
            if !key.active {
                continue;
            }

            // Check for expired keys
            if let Some(expires_at) = key.expires_at {
                if expires_at < now {
                    expired_keys.push(*key_id);
                    violations.push(PolicyViolation {
                        violation_type: ViolationType::ExpiredKey,
                        severity: ViolationSeverity::High,
                        description: format!("Key {key_id} has expired"),
                        resource_id: key_id.to_string(),
                        policy_id: "general_key_policy".to_string(),
                        detected_at: now,
                    });
                }
            }

            // Check for keys requiring rotation
            let rotation_due = key.last_rotated_at + key.rotation_interval;
            if rotation_due < now {
                keys_requiring_rotation.push(*key_id);
                violations.push(PolicyViolation {
                    violation_type: ViolationType::RotationOverdue,
                    severity: ViolationSeverity::Medium,
                    description: format!("Key {key_id} requires rotation"),
                    resource_id: key_id.to_string(),
                    policy_id: "general_key_policy".to_string(),
                    detected_at: now,
                });
            }

            // Check key usage limits
            if let Some(max_usage) = key.max_usage {
                let usage = self.key_usage.get(key_id).unwrap_or(&0);
                if *usage >= max_usage {
                    violations.push(PolicyViolation {
                        violation_type: ViolationType::ExcessiveKeyUsage,
                        severity: ViolationSeverity::High,
                        description: format!("Key {key_id} has exceeded usage limit"),
                        resource_id: key_id.to_string(),
                        policy_id: "general_key_policy".to_string(),
                        detected_at: now,
                    });
                }
            }

            // Check algorithm strength
            if !self.is_algorithm_secure(key.algorithm) {
                violations.push(PolicyViolation {
                    violation_type: ViolationType::WeakAlgorithm,
                    severity: ViolationSeverity::High,
                    description: format!("Key {} uses weak algorithm {:?}", key_id, key.algorithm),
                    resource_id: key_id.to_string(),
                    policy_id: "algorithm_policy".to_string(),
                    detected_at: now,
                });
            }
        }

        // Generate strength assessment
        let strength_assessment = self.assess_encryption_strength();

        // Generate recommendations
        let mut recommendations = Vec::new();
        if !keys_requiring_rotation.is_empty() {
            recommendations.push("Rotate overdue encryption keys".to_string());
        }
        if !expired_keys.is_empty() {
            recommendations.push("Replace expired encryption keys".to_string());
        }
        if strength_assessment.overall_score < 0.8 {
            recommendations.push("Upgrade encryption algorithms to stronger variants".to_string());
        }

        let compliant = violations.is_empty()
            || violations
                .iter()
                .all(|v| v.severity < ViolationSeverity::High);

        Ok(EncryptionValidationResult {
            validated_at: now,
            compliant,
            violations,
            keys_requiring_rotation,
            expired_keys,
            strength_assessment,
            recommendations,
        })
    }

    /// Check if algorithm is considered secure
    fn is_algorithm_secure(&self, algorithm: EncryptionAlgorithm) -> bool {
        match algorithm {
            EncryptionAlgorithm::AES256GCM => true,
            EncryptionAlgorithm::ChaCha20Poly1305 => true,
            EncryptionAlgorithm::RSA4096 => true,
            EncryptionAlgorithm::ECCP256 => true,
            EncryptionAlgorithm::AES256CBC => false, // Less secure due to padding oracle attacks
        }
    }

    /// Assess overall encryption strength
    fn assess_encryption_strength(&self) -> StrengthAssessment {
        let mut algorithm_scores = HashMap::new();

        // Score algorithms based on security
        algorithm_scores.insert(EncryptionAlgorithm::AES256GCM, 1.0);
        algorithm_scores.insert(EncryptionAlgorithm::ChaCha20Poly1305, 1.0);
        algorithm_scores.insert(EncryptionAlgorithm::RSA4096, 0.9);
        algorithm_scores.insert(EncryptionAlgorithm::ECCP256, 0.95);
        algorithm_scores.insert(EncryptionAlgorithm::AES256CBC, 0.7);

        // Calculate overall score based on active keys
        let active_keys: Vec<_> = self.keys.values().filter(|k| k.active).collect();
        let total_score: f64 = active_keys
            .iter()
            .map(|k| algorithm_scores.get(&k.algorithm).unwrap_or(&0.5))
            .sum();

        let overall_score = if active_keys.is_empty() {
            0.0
        } else {
            total_score / active_keys.len() as f64
        };

        // Key management score (based on rotation compliance)
        let now = Utc::now();
        let rotation_compliant = active_keys
            .iter()
            .filter(|k| k.last_rotated_at + k.rotation_interval > now)
            .count();

        let key_management_score = if active_keys.is_empty() {
            1.0
        } else {
            rotation_compliant as f64 / active_keys.len() as f64
        };

        // Compliance score (based on policy adherence)
        let compliance_score = if self.policies.is_empty() {
            0.5
        } else {
            0.8 // Simplified scoring
        };

        let mut improvement_areas = Vec::new();
        if overall_score < 0.8 {
            improvement_areas.push("Upgrade to stronger encryption algorithms".to_string());
        }
        if key_management_score < 0.9 {
            improvement_areas.push("Improve key rotation practices".to_string());
        }
        if compliance_score < 0.9 {
            improvement_areas.push("Enhance policy compliance monitoring".to_string());
        }

        StrengthAssessment {
            overall_score,
            algorithm_scores,
            key_management_score,
            compliance_score,
            improvement_areas,
        }
    }

    /// Get encryption key by ID
    pub fn get_key(&self, key_id: &Uuid) -> Option<&EncryptionKey> {
        self.keys.get(key_id)
    }

    /// Get encrypted data by ID
    pub fn get_encrypted_data(&self, data_id: &Uuid) -> Option<&EncryptedData> {
        self.encrypted_data.get(data_id)
    }

    /// Get encryption policy by ID
    pub fn get_policy(&self, policy_id: &str) -> Option<&EncryptionPolicy> {
        self.policies.get(policy_id)
    }

    /// List active keys
    pub fn list_active_keys(&self) -> Vec<&EncryptionKey> {
        self.keys.values().filter(|k| k.active).collect()
    }

    /// Get key usage statistics
    pub fn get_key_usage(&self, key_id: &Uuid) -> Option<u64> {
        self.key_usage.get(key_id).copied()
    }
}

impl Default for EncryptionEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_classification::DataClassifier;

    #[test]
    fn test_encryption_engine_creation() {
        let engine = EncryptionEngine::new();
        assert_eq!(engine.policies.len(), 3); // PII, PHI, Financial policies
        assert!(engine.keys.is_empty());
        assert!(engine.encrypted_data.is_empty());
    }

    #[test]
    fn test_add_encryption_policy() {
        let mut engine = EncryptionEngine::new();

        let policy = EncryptionPolicy {
            id: "test_policy".to_string(),
            name: "Test Policy".to_string(),
            classification: DataClassification::InternalData,
            categories: vec![DataCategory::BusinessData],
            required_algorithm: EncryptionAlgorithm::AES256GCM,
            min_key_length: 256,
            key_rotation_required: true,
            max_rotation_interval: Duration::days(180),
            transit_encryption_required: true,
            at_rest_encryption_required: true,
            compliance_standards: vec![ComplianceStandard::ISO27001],
            owner: "Security Team".to_string(),
            created_at: Utc::now(),
            active: true,
        };

        let result = engine.add_policy(policy);
        assert!(result.is_ok());
        assert_eq!(engine.policies.len(), 4);
    }

    #[test]
    fn test_add_invalid_encryption_policy() {
        let mut engine = EncryptionEngine::new();

        let invalid_policy = EncryptionPolicy {
            id: "invalid_policy".to_string(),
            name: "Invalid Policy".to_string(),
            classification: DataClassification::InternalData,
            categories: vec![DataCategory::BusinessData],
            required_algorithm: EncryptionAlgorithm::AES256GCM,
            min_key_length: 64, // Too small
            key_rotation_required: true,
            max_rotation_interval: Duration::days(180),
            transit_encryption_required: true,
            at_rest_encryption_required: true,
            compliance_standards: vec![],
            owner: "Security Team".to_string(),
            created_at: Utc::now(),
            active: true,
        };

        let result = engine.add_policy(invalid_policy);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ComplianceError::EncryptionError(_)
        ));
    }

    #[test]
    fn test_generate_encryption_key() {
        let mut engine = EncryptionEngine::new();

        let result = engine.generate_key(
            "Test Key".to_string(),
            EncryptionAlgorithm::AES256GCM,
            DataClassification::ConfidentialData,
            vec![DataCategory::PII],
            "Security Team".to_string(),
        );

        assert!(result.is_ok());
        let key_id = result?;

        let key = engine.get_key(&key_id)?;
        assert_eq!(key.name, "Test Key");
        assert_eq!(key.algorithm, EncryptionAlgorithm::AES256GCM);
        assert_eq!(key.key_length, 256);
        assert!(key.active);
    }

    #[test]
    fn test_encrypt_and_decrypt_data() {
        let mut engine = EncryptionEngine::new();
        let classifier = DataClassifier::new();

        // Generate key
        let key_id = engine
            .generate_key(
                "Test Key".to_string(),
                EncryptionAlgorithm::AES256GCM,
                DataClassification::ConfidentialData,
                vec![DataCategory::PII],
                "Security Team".to_string(),
            )
            .unwrap();

        // Create metadata
        let metadata = classifier
            .classify(
                DataCategory::PII,
                DataClassification::ConfidentialData,
                vec!["US".to_string()],
            )
            .unwrap();

        // Encrypt data
        let test_data = b"sensitive personal information";
        let result = engine.encrypt_data(test_data, key_id, metadata);
        assert!(result.is_ok());

        let data_id = result?;
        let encrypted = engine.get_encrypted_data(&data_id)?;
        assert_eq!(encrypted.key_id, key_id);
        assert_eq!(encrypted.algorithm, EncryptionAlgorithm::AES256GCM);

        // Decrypt data
        let decrypted_result = engine.decrypt_data(data_id);
        assert!(decrypted_result.is_ok());
    }

    #[test]
    fn test_encrypt_with_inactive_key() {
        let mut engine = EncryptionEngine::new();
        let classifier = DataClassifier::new();

        // Generate and deactivate key
        let key_id = engine
            .generate_key(
                "Test Key".to_string(),
                EncryptionAlgorithm::AES256GCM,
                DataClassification::ConfidentialData,
                vec![DataCategory::PII],
                "Security Team".to_string(),
            )
            .unwrap();

        if let Some(key) = engine.keys.get_mut(&key_id) {
            key.active = false;
        }

        let metadata = classifier
            .classify(
                DataCategory::PII,
                DataClassification::ConfidentialData,
                vec!["US".to_string()],
            )
            .unwrap();

        let test_data = b"sensitive data";
        let result = engine.encrypt_data(test_data, key_id, metadata);

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ComplianceError::EncryptionError(_)
        ));
    }

    #[test]
    fn test_encrypt_with_key_usage_limit() {
        let mut engine = EncryptionEngine::new();
        let classifier = DataClassifier::new();

        // Generate key with low usage limit
        let key_id = engine
            .generate_key(
                "Test Key".to_string(),
                EncryptionAlgorithm::AES256GCM,
                DataClassification::ConfidentialData,
                vec![DataCategory::PII],
                "Security Team".to_string(),
            )
            .unwrap();

        // Set low usage limit
        if let Some(key) = engine.keys.get_mut(&key_id) {
            key.max_usage = Some(1);
        }

        let metadata = classifier
            .classify(
                DataCategory::PII,
                DataClassification::ConfidentialData,
                vec!["US".to_string()],
            )
            .unwrap();

        let test_data = b"sensitive data";

        // First encryption should succeed
        let result1 = engine.encrypt_data(test_data, key_id, metadata.clone());
        assert!(result1.is_ok());

        // Second encryption should fail due to usage limit
        let result2 = engine.encrypt_data(test_data, key_id, metadata);
        assert!(result2.is_err());
        assert!(matches!(
            result2.unwrap_err(),
            ComplianceError::EncryptionError(_)
        ));
    }

    #[tokio::test]
    async fn test_key_rotation() {
        let mut engine = EncryptionEngine::new();
        let classifier = DataClassifier::new();

        // Generate key and encrypt some data
        let old_key_id = engine
            .generate_key(
                "Test Key".to_string(),
                EncryptionAlgorithm::AES256GCM,
                DataClassification::ConfidentialData,
                vec![DataCategory::PII],
                "Security Team".to_string(),
            )
            .unwrap();

        let metadata = classifier
            .classify(
                DataCategory::PII,
                DataClassification::ConfidentialData,
                vec!["US".to_string()],
            )
            .unwrap();

        let test_data = b"sensitive data";
        engine
            .encrypt_data(test_data, old_key_id, metadata)
            .unwrap();

        // Rotate key
        let rotation_result = engine.rotate_key(old_key_id).await;
        assert!(rotation_result.is_ok());

        let rotation = rotation_result?;
        assert_eq!(rotation.old_key_id, old_key_id);
        assert_ne!(rotation.new_key_id, old_key_id);
        assert_eq!(rotation.records_reencrypted, 1);

        // Check old key is deactivated
        let old_key = engine.get_key(&old_key_id)?;
        assert!(!old_key.active);

        // Check new key is active
        let new_key = engine.get_key(&rotation.new_key_id)?;
        assert!(new_key.active);
    }

    #[test]
    fn test_validate_encryption_compliance() {
        let mut engine = EncryptionEngine::new();

        // Generate some keys with different states
        let _active_key = engine
            .generate_key(
                "Active Key".to_string(),
                EncryptionAlgorithm::AES256GCM,
                DataClassification::ConfidentialData,
                vec![DataCategory::PII],
                "Security Team".to_string(),
            )
            .unwrap();

        let overdue_key = engine
            .generate_key(
                "Overdue Key".to_string(),
                EncryptionAlgorithm::AES256GCM,
                DataClassification::ConfidentialData,
                vec![DataCategory::PII],
                "Security Team".to_string(),
            )
            .unwrap();

        // Make one key overdue for rotation
        if let Some(key) = engine.keys.get_mut(&overdue_key) {
            key.last_rotated_at = Utc::now() - Duration::days(200);
            key.rotation_interval = Duration::days(90);
        }

        let validation_result = engine.validate_encryption_compliance();
        assert!(validation_result.is_ok());

        let validation = validation_result?;
        assert!(!validation.violations.is_empty());
        assert!(validation.keys_requiring_rotation.contains(&overdue_key));
        assert!(!validation.recommendations.is_empty());
    }

    #[test]
    fn test_algorithm_security_assessment() {
        let engine = EncryptionEngine::new();

        assert!(engine.is_algorithm_secure(EncryptionAlgorithm::AES256GCM));
        assert!(engine.is_algorithm_secure(EncryptionAlgorithm::ChaCha20Poly1305));
        assert!(!engine.is_algorithm_secure(EncryptionAlgorithm::AES256CBC));
    }

    #[test]
    fn test_encryption_strength_assessment() {
        let mut engine = EncryptionEngine::new();

        // Generate keys with different algorithms
        engine
            .generate_key(
                "Strong Key 1".to_string(),
                EncryptionAlgorithm::AES256GCM,
                DataClassification::ConfidentialData,
                vec![DataCategory::PII],
                "Security Team".to_string(),
            )
            .unwrap();

        engine
            .generate_key(
                "Strong Key 2".to_string(),
                EncryptionAlgorithm::ChaCha20Poly1305,
                DataClassification::ConfidentialData,
                vec![DataCategory::PII],
                "Security Team".to_string(),
            )
            .unwrap();

        engine
            .generate_key(
                "Weak Key".to_string(),
                EncryptionAlgorithm::AES256CBC,
                DataClassification::ConfidentialData,
                vec![DataCategory::PII],
                "Security Team".to_string(),
            )
            .unwrap();

        let assessment = engine.assess_encryption_strength();
        assert!(assessment.overall_score > 0.0);
        assert!(assessment.overall_score <= 1.0);
        assert!(!assessment.algorithm_scores.is_empty());
        assert!(assessment.key_management_score >= 0.0);
        assert!(assessment.compliance_score >= 0.0);
    }

    #[test]
    fn test_get_key_usage_statistics() {
        let mut engine = EncryptionEngine::new();
        let classifier = DataClassifier::new();

        let key_id = engine
            .generate_key(
                "Test Key".to_string(),
                EncryptionAlgorithm::AES256GCM,
                DataClassification::ConfidentialData,
                vec![DataCategory::PII],
                "Security Team".to_string(),
            )
            .unwrap();

        // Initially, usage should be 0
        assert_eq!(engine.get_key_usage(&key_id), Some(0));

        // Encrypt some data
        let metadata = classifier
            .classify(
                DataCategory::PII,
                DataClassification::ConfidentialData,
                vec!["US".to_string()],
            )
            .unwrap();

        engine
            .encrypt_data(b"test data", key_id, metadata.clone())
            .unwrap();
        assert_eq!(engine.get_key_usage(&key_id), Some(1));

        engine
            .encrypt_data(b"more test data", key_id, metadata)
            .unwrap();
        assert_eq!(engine.get_key_usage(&key_id), Some(2));
    }

    #[test]
    fn test_list_active_keys() {
        let mut engine = EncryptionEngine::new();

        let key1 = engine
            .generate_key(
                "Active Key 1".to_string(),
                EncryptionAlgorithm::AES256GCM,
                DataClassification::ConfidentialData,
                vec![DataCategory::PII],
                "Security Team".to_string(),
            )
            .unwrap();

        let key2 = engine
            .generate_key(
                "Active Key 2".to_string(),
                EncryptionAlgorithm::ChaCha20Poly1305,
                DataClassification::ConfidentialData,
                vec![DataCategory::PII],
                "Security Team".to_string(),
            )
            .unwrap();

        // Deactivate one key
        if let Some(key) = engine.keys.get_mut(&key2) {
            key.active = false;
        }

        let active_keys = engine.list_active_keys();
        assert_eq!(active_keys.len(), 1);
        assert_eq!(active_keys[0].id, key1);
    }

    #[test]
    fn test_get_key_length_for_algorithm() {
        let engine = EncryptionEngine::new();

        assert_eq!(
            engine.get_key_length_for_algorithm(EncryptionAlgorithm::AES256GCM),
            256
        );
        assert_eq!(
            engine.get_key_length_for_algorithm(EncryptionAlgorithm::AES256CBC),
            256
        );
        assert_eq!(
            engine.get_key_length_for_algorithm(EncryptionAlgorithm::ChaCha20Poly1305),
            256
        );
        assert_eq!(
            engine.get_key_length_for_algorithm(EncryptionAlgorithm::RSA4096),
            4096
        );
        assert_eq!(
            engine.get_key_length_for_algorithm(EncryptionAlgorithm::ECCP256),
            256
        );
    }
}
