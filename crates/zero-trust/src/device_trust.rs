//! Device attestation, trust scoring, certificate validation, and device compliance checking
//!
//! This module implements comprehensive device trust management following zero-trust principles:
//! - Hardware and software attestation
//! - Device trust scoring based on multiple factors
//! - Certificate-based device authentication
//! - Continuous compliance monitoring
//! - Device health assessment

use crate::error::{ZeroTrustError, ZeroTrustResult};
use async_trait::async_trait;
use chrono::{DateTime, Duration, Utc};
use dashmap::DashMap;
use ring::digest::{self, SHA256};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use uuid::Uuid;
use x509_parser::prelude::*;

/// Device trust configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceTrustConfig {
    /// Minimum trust score required
    pub min_trust_score: f64,
    /// Certificate validation strictness
    pub cert_validation_strict: bool,
    /// Attestation requirement level
    pub attestation_level: AttestationLevel,
    /// Compliance check interval
    pub compliance_check_interval: Duration,
    /// Device health check interval
    pub health_check_interval: Duration,
    /// Enable continuous monitoring
    pub continuous_monitoring: bool,
    /// Trust score decay rate (per hour)
    pub trust_decay_rate: f64,
}

impl Default for DeviceTrustConfig {
    fn default() -> Self {
        Self {
            min_trust_score: 0.7,
            cert_validation_strict: true,
            attestation_level: AttestationLevel::Full,
            compliance_check_interval: Duration::hours(1),
            health_check_interval: Duration::minutes(15),
            continuous_monitoring: true,
            trust_decay_rate: 0.05,
        }
    }
}

/// Attestation requirement levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttestationLevel {
    /// No attestation required
    None,
    /// Basic software attestation
    Basic,
    /// Full hardware and software attestation
    Full,
    /// Enhanced attestation with secure enclave
    Enhanced,
}

/// Device information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Device {
    /// Unique device ID
    pub device_id: Uuid,
    /// Device name
    pub name: String,
    /// Device type
    pub device_type: DeviceType,
    /// Operating system
    pub os: OperatingSystem,
    /// Hardware identifier
    pub hardware_id: String,
    /// Registration timestamp
    pub registered_at: DateTime<Utc>,
    /// Last seen timestamp
    pub last_seen: DateTime<Utc>,
    /// Current trust score
    pub trust_score: f64,
    /// Device state
    pub state: DeviceState,
    /// Compliance status
    pub compliance_status: ComplianceStatus,
    /// Device attributes
    pub attributes: serde_json::Value,
}

/// Device types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeviceType {
    Desktop,
    Laptop,
    Mobile,
    Tablet,
    Server,
    IoT,
    Virtual,
}

/// Operating system information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperatingSystem {
    /// OS name
    pub name: String,
    /// OS version
    pub version: String,
    /// Build number
    pub build: String,
    /// Architecture
    pub arch: String,
}

/// Device states
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeviceState {
    /// Device is trusted and active
    Trusted,
    /// Device is pending verification
    PendingVerification,
    /// Device is quarantined due to issues
    Quarantined,
    /// Device is blocked
    Blocked,
    /// Device is decommissioned
    Decommissioned,
}

/// Compliance status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceStatus {
    /// Overall compliance
    pub compliant: bool,
    /// Last check timestamp
    pub last_check: DateTime<Utc>,
    /// Failed policies
    pub failed_policies: Vec<String>,
    /// Compliance score
    pub score: f64,
}

/// Device attestation data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceAttestation {
    /// Attestation type
    pub attestation_type: AttestationType,
    /// Hardware attestation data
    pub hardware_attestation: Option<HardwareAttestation>,
    /// Software attestation data
    pub software_attestation: Option<SoftwareAttestation>,
    /// Attestation timestamp
    pub timestamp: DateTime<Utc>,
    /// Attestation signature
    pub signature: Vec<u8>,
    /// Nonce used for attestation
    pub nonce: Vec<u8>,
}

/// Attestation types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttestationType {
    Hardware,
    Software,
    Combined,
    SecureEnclave,
}

/// Hardware attestation data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareAttestation {
    /// TPM version
    pub tpm_version: String,
    /// PCR values
    pub pcr_values: Vec<Vec<u8>>,
    /// Secure boot state
    pub secure_boot_enabled: bool,
    /// Hardware root of trust
    pub hardware_root_of_trust: Vec<u8>,
    /// Measured boot log
    pub measured_boot_log: Vec<u8>,
}

/// Software attestation data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SoftwareAttestation {
    /// OS measurements
    pub os_measurements: Vec<u8>,
    /// Installed software hash
    pub software_inventory_hash: Vec<u8>,
    /// Configuration hash
    pub config_hash: Vec<u8>,
    /// Security patches applied
    pub security_patches: Vec<String>,
    /// Antivirus status
    pub antivirus_enabled: bool,
    /// Firewall status
    pub firewall_enabled: bool,
}

/// Device certificate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceCertificate {
    /// Certificate ID
    pub cert_id: Uuid,
    /// Device ID
    pub device_id: Uuid,
    /// Certificate data (DER encoded)
    pub cert_data: Vec<u8>,
    /// Issuer
    pub issuer: String,
    /// Subject
    pub subject: String,
    /// Not before
    pub not_before: DateTime<Utc>,
    /// Not after
    pub not_after: DateTime<Utc>,
    /// Certificate chain
    pub cert_chain: Vec<Vec<u8>>,
}

/// Trust factors for scoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrustFactors {
    /// Certificate validity
    pub cert_valid: bool,
    /// Attestation passed
    pub attestation_passed: bool,
    /// Compliance score
    pub compliance_score: f64,
    /// Time since last update
    pub hours_since_update: f64,
    /// Security patches up to date
    pub patches_current: bool,
    /// Antivirus enabled
    pub antivirus_enabled: bool,
    /// Firewall enabled
    pub firewall_enabled: bool,
    /// Encryption enabled
    pub encryption_enabled: bool,
    /// Known vulnerabilities
    pub known_vulnerabilities: u32,
}

/// Device trust manager trait
#[async_trait]
pub trait DeviceTrustManagerTrait: Send + Sync {
    /// Register a new device
    async fn register_device(
        &self,
        name: String,
        device_type: DeviceType,
        hardware_id: String,
    ) -> ZeroTrustResult<Device>;

    /// Perform device attestation
    async fn attest_device(
        &self,
        device_id: Uuid,
        attestation: DeviceAttestation,
    ) -> ZeroTrustResult<bool>;

    /// Calculate device trust score
    async fn calculate_trust_score(&self, device_id: Uuid) -> ZeroTrustResult<f64>;

    /// Validate device certificate
    async fn validate_certificate(
        &self,
        device_id: Uuid,
        cert_data: Vec<u8>,
    ) -> ZeroTrustResult<bool>;

    /// Check device compliance
    async fn check_compliance(&self, device_id: Uuid) -> ZeroTrustResult<ComplianceStatus>;

    /// Update device state
    async fn update_device_state(&self, device_id: Uuid, state: DeviceState)
        -> ZeroTrustResult<()>;

    /// Get device by ID
    async fn get_device(&self, device_id: Uuid) -> ZeroTrustResult<Device>;

    /// Enroll device certificate
    async fn enroll_certificate(
        &self,
        device_id: Uuid,
        cert_data: Vec<u8>,
    ) -> ZeroTrustResult<DeviceCertificate>;

    /// Revoke device
    async fn revoke_device(&self, device_id: Uuid) -> ZeroTrustResult<()>;
}

/// Device trust manager implementation
pub struct DeviceTrustManager {
    config: DeviceTrustConfig,
    devices: Arc<DashMap<Uuid, Device>>,
    attestations: Arc<DashMap<Uuid, DeviceAttestation>>,
    certificates: Arc<DashMap<Uuid, DeviceCertificate>>,
    trust_factors: Arc<DashMap<Uuid, TrustFactors>>,
}

impl DeviceTrustManager {
    /// Create new device trust manager
    pub fn new(config: DeviceTrustConfig) -> ZeroTrustResult<Self> {
        Ok(Self {
            config,
            devices: Arc::new(DashMap::new()),
            attestations: Arc::new(DashMap::new()),
            certificates: Arc::new(DashMap::new()),
            trust_factors: Arc::new(DashMap::new()),
        })
    }

    /// Hash device data
    fn hash_device_data(&self, data: &[u8]) -> Vec<u8> {
        digest::digest(&SHA256, data).as_ref().to_vec()
    }

    /// Verify attestation signature
    fn verify_attestation_signature(&self, attestation: &DeviceAttestation) -> bool {
        // In production, implement proper signature verification
        // For now, check signature is not empty
        !attestation.signature.is_empty()
    }

    /// Calculate trust score from factors
    fn calculate_score_from_factors(&self, factors: &TrustFactors) -> f64 {
        let mut score = 0.0;
        let mut weight_sum = 0.0;

        // Certificate validity (weight: 0.2)
        if factors.cert_valid {
            score += 0.2;
        }
        weight_sum += 0.2;

        // Attestation (weight: 0.25)
        if factors.attestation_passed {
            score += 0.25;
        }
        weight_sum += 0.25;

        // Compliance (weight: 0.2)
        score += factors.compliance_score * 0.2;
        weight_sum += 0.2;

        // Security features (weight: 0.15)
        let security_score = (factors.patches_current as u8
            + factors.antivirus_enabled as u8
            + factors.firewall_enabled as u8
            + factors.encryption_enabled as u8) as f64
            / 4.0;
        score += security_score * 0.15;
        weight_sum += 0.15;

        // Vulnerabilities (weight: 0.1)
        let vuln_score = match factors.known_vulnerabilities {
            0 => 1.0,
            1..=2 => 0.7,
            3..=5 => 0.4,
            _ => 0.0,
        };
        score += vuln_score * 0.1;
        weight_sum += 0.1;

        // Time decay (weight: 0.1)
        let decay_score =
            (1.0 - (factors.hours_since_update * self.config.trust_decay_rate / 100.0)).max(0.0);
        score += decay_score * 0.1;
        weight_sum += 0.1;

        score / weight_sum
    }

    /// Validate X.509 certificate and return validity info
    fn validate_certificate_data(
        &self,
        cert_data: &[u8],
    ) -> ZeroTrustResult<(DateTime<Utc>, DateTime<Utc>, String, String)> {
        let (_, cert) = parse_x509_certificate(cert_data).map_err(|e| {
            ZeroTrustError::CertificateValidationFailed {
                reason: format!("Failed to parse certificate: {e:?}"),
            }
        })?;

        let not_before = DateTime::from_timestamp(cert.validity().not_before.timestamp(), 0)
            .ok_or_else(|| ZeroTrustError::CertificateValidationFailed {
                reason: "Invalid not_before timestamp".to_string(),
            })?;
        let not_after = DateTime::from_timestamp(cert.validity().not_after.timestamp(), 0)
            .ok_or_else(|| ZeroTrustError::CertificateValidationFailed {
                reason: "Invalid not_after timestamp".to_string(),
            })?;

        Ok((
            not_before,
            not_after,
            cert.issuer().to_string(),
            cert.subject().to_string(),
        ))
    }
}

#[async_trait]
impl DeviceTrustManagerTrait for DeviceTrustManager {
    async fn register_device(
        &self,
        name: String,
        device_type: DeviceType,
        hardware_id: String,
    ) -> ZeroTrustResult<Device> {
        let device_id = Uuid::new_v4();
        let now = Utc::now();

        let device = Device {
            device_id,
            name: name.clone(),
            device_type,
            os: OperatingSystem {
                name: "Unknown".to_string(),
                version: "Unknown".to_string(),
                build: "Unknown".to_string(),
                arch: "Unknown".to_string(),
            },
            hardware_id,
            registered_at: now,
            last_seen: now,
            trust_score: 0.5, // Start with neutral score
            state: DeviceState::PendingVerification,
            compliance_status: ComplianceStatus {
                compliant: false,
                last_check: now,
                failed_policies: vec!["Initial registration".to_string()],
                score: 0.0,
            },
            attributes: serde_json::json!({}),
        };

        self.devices.insert(device_id, device.clone());

        // Initialize trust factors
        let factors = TrustFactors {
            cert_valid: false,
            attestation_passed: false,
            compliance_score: 0.0,
            hours_since_update: 0.0,
            patches_current: false,
            antivirus_enabled: false,
            firewall_enabled: false,
            encryption_enabled: false,
            known_vulnerabilities: 0,
        };
        self.trust_factors.insert(device_id, factors);

        Ok(device)
    }

    async fn attest_device(
        &self,
        device_id: Uuid,
        attestation: DeviceAttestation,
    ) -> ZeroTrustResult<bool> {
        let mut device =
            self.devices
                .get_mut(&device_id)
                .ok_or_else(|| ZeroTrustError::DeviceTrustFailed {
                    reason: "Device not found".to_string(),
                })?;

        // Verify attestation signature
        if !self.verify_attestation_signature(&attestation) {
            return Err(ZeroTrustError::AttestationFailed {
                component: "signature".to_string(),
                reason: "Invalid attestation signature".to_string(),
            });
        }

        // Validate based on attestation level
        let valid = match self.config.attestation_level {
            AttestationLevel::None => true,
            AttestationLevel::Basic => attestation.software_attestation.is_some(),
            AttestationLevel::Full => {
                attestation.hardware_attestation.is_some()
                    && attestation.software_attestation.is_some()
            }
            AttestationLevel::Enhanced => {
                matches!(attestation.attestation_type, AttestationType::SecureEnclave)
                    && attestation.hardware_attestation.is_some()
                    && attestation.software_attestation.is_some()
            }
        };

        if valid {
            // Store attestation
            self.attestations.insert(device_id, attestation);

            // Update trust factors
            if let Some(mut factors) = self.trust_factors.get_mut(&device_id) {
                factors.attestation_passed = true;
            }

            // Update last seen
            device.last_seen = Utc::now();
        }

        Ok(valid)
    }

    async fn calculate_trust_score(&self, device_id: Uuid) -> ZeroTrustResult<f64> {
        let factors = self.trust_factors.get(&device_id).ok_or_else(|| {
            ZeroTrustError::DeviceTrustFailed {
                reason: "Device trust factors not found".to_string(),
            }
        })?;

        let score = self.calculate_score_from_factors(&factors);

        // Update device trust score
        if let Some(mut device) = self.devices.get_mut(&device_id) {
            device.trust_score = score;
        }

        Ok(score)
    }

    async fn validate_certificate(
        &self,
        device_id: Uuid,
        cert_data: Vec<u8>,
    ) -> ZeroTrustResult<bool> {
        let (not_before, not_after, _issuer, _subject) =
            self.validate_certificate_data(&cert_data)?;

        // Check certificate validity period
        let now = Utc::now();

        if now < not_before || now > not_after {
            return Ok(false);
        }

        // In production, verify certificate chain, revocation status, etc.

        // Update trust factors
        if let Some(mut factors) = self.trust_factors.get_mut(&device_id) {
            factors.cert_valid = true;
        }

        Ok(true)
    }

    async fn check_compliance(&self, device_id: Uuid) -> ZeroTrustResult<ComplianceStatus> {
        let device =
            self.devices
                .get(&device_id)
                .ok_or_else(|| ZeroTrustError::DeviceTrustFailed {
                    reason: "Device not found".to_string(),
                })?;

        let mut failed_policies = Vec::new();
        let mut score: f64 = 1.0;

        // Check attestation
        if self.attestations.get(&device_id).is_none() {
            failed_policies.push("Missing attestation".to_string());
            score -= 0.3;
        }

        // Check certificate
        if self.certificates.get(&device_id).is_none() {
            failed_policies.push("Missing certificate".to_string());
            score -= 0.2;
        }

        // Check trust score
        if device.trust_score < self.config.min_trust_score {
            failed_policies.push(format!(
                "Trust score {} below minimum {}",
                device.trust_score, self.config.min_trust_score
            ));
            score -= 0.2;
        }

        // Check last seen time
        let hours_since_seen = (Utc::now() - device.last_seen).num_hours() as f64;
        if hours_since_seen > 24.0 {
            failed_policies.push(format!("Device not seen for {hours_since_seen} hours"));
            score -= 0.1;
        }

        let compliant = failed_policies.is_empty();
        let status = ComplianceStatus {
            compliant,
            last_check: Utc::now(),
            failed_policies,
            score: score.max(0.0),
        };

        // Update device compliance status
        if let Some(mut device) = self.devices.get_mut(&device_id) {
            device.compliance_status = status.clone();
        }

        // Update trust factors
        if let Some(mut factors) = self.trust_factors.get_mut(&device_id) {
            factors.compliance_score = status.score;
            factors.hours_since_update = hours_since_seen;
        }

        Ok(status)
    }

    async fn update_device_state(
        &self,
        device_id: Uuid,
        state: DeviceState,
    ) -> ZeroTrustResult<()> {
        let mut device =
            self.devices
                .get_mut(&device_id)
                .ok_or_else(|| ZeroTrustError::DeviceTrustFailed {
                    reason: "Device not found".to_string(),
                })?;

        device.state = state;
        Ok(())
    }

    async fn get_device(&self, device_id: Uuid) -> ZeroTrustResult<Device> {
        self.devices
            .get(&device_id)
            .map(|entry| entry.clone())
            .ok_or_else(|| ZeroTrustError::DeviceTrustFailed {
                reason: "Device not found".to_string(),
            })
    }

    async fn enroll_certificate(
        &self,
        device_id: Uuid,
        cert_data: Vec<u8>,
    ) -> ZeroTrustResult<DeviceCertificate> {
        let (not_before, not_after, issuer, subject) =
            self.validate_certificate_data(&cert_data)?;
        let cert_id = Uuid::new_v4();

        let device_cert = DeviceCertificate {
            cert_id,
            device_id,
            cert_data: cert_data.clone(),
            issuer,
            subject,
            not_before,
            not_after,
            cert_chain: vec![],
        };

        self.certificates.insert(device_id, device_cert.clone());

        // Validate the certificate
        self.validate_certificate(device_id, cert_data).await?;

        Ok(device_cert)
    }

    async fn revoke_device(&self, device_id: Uuid) -> ZeroTrustResult<()> {
        self.update_device_state(device_id, DeviceState::Decommissioned)
            .await?;

        // Remove associated data
        self.attestations.remove(&device_id);
        self.certificates.remove(&device_id);
        self.trust_factors.remove(&device_id);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_device_registration() {
        let config = DeviceTrustConfig::default();
        let manager = DeviceTrustManager::new(config)?;

        let device = manager
            .register_device(
                "test-laptop".to_string(),
                DeviceType::Laptop,
                "HW123456".to_string(),
            )
            .await
            .unwrap();

        assert_eq!(device.name, "test-laptop");
        assert_eq!(device.device_type, DeviceType::Laptop);
        assert_eq!(device.hardware_id, "HW123456");
        assert_eq!(device.state, DeviceState::PendingVerification);
        assert_eq!(device.trust_score, 0.5);
    }

    #[tokio::test]
    async fn test_device_attestation() {
        let config = DeviceTrustConfig::default();
        let manager = DeviceTrustManager::new(config)?;

        let device = manager
            .register_device(
                "test-device".to_string(),
                DeviceType::Desktop,
                "HW789".to_string(),
            )
            .await
            .unwrap();

        let attestation = DeviceAttestation {
            attestation_type: AttestationType::Combined,
            hardware_attestation: Some(HardwareAttestation {
                tpm_version: "2.0".to_string(),
                pcr_values: vec![vec![0x12, 0x34, 0x56]],
                secure_boot_enabled: true,
                hardware_root_of_trust: vec![0xAB, 0xCD, 0xEF],
                measured_boot_log: vec![0x01, 0x02, 0x03],
            }),
            software_attestation: Some(SoftwareAttestation {
                os_measurements: vec![0x11, 0x22, 0x33],
                software_inventory_hash: vec![0x44, 0x55, 0x66],
                config_hash: vec![0x77, 0x88, 0x99],
                security_patches: vec!["KB123456".to_string()],
                antivirus_enabled: true,
                firewall_enabled: true,
            }),
            timestamp: Utc::now(),
            signature: vec![0xDE, 0xAD, 0xBE, 0xEF],
            nonce: vec![0x12, 0x34, 0x56, 0x78],
        };

        let result = manager
            .attest_device(device.device_id, attestation)
            .await
            .unwrap();
        assert!(result);
    }

    #[tokio::test]
    async fn test_trust_score_calculation() {
        let config = DeviceTrustConfig::default();
        let manager = DeviceTrustManager::new(config)?;

        let device = manager
            .register_device(
                "test-device".to_string(),
                DeviceType::Mobile,
                "HW456".to_string(),
            )
            .await
            .unwrap();

        // Set up trust factors
        manager.trust_factors.insert(
            device.device_id,
            TrustFactors {
                cert_valid: true,
                attestation_passed: true,
                compliance_score: 0.9,
                hours_since_update: 2.0,
                patches_current: true,
                antivirus_enabled: true,
                firewall_enabled: true,
                encryption_enabled: true,
                known_vulnerabilities: 0,
            },
        );

        let score = manager
            .calculate_trust_score(device.device_id)
            .await
            .unwrap();
        assert!(
            score > 0.8,
            "Trust score should be high with all positive factors"
        );
    }

    #[tokio::test]
    async fn test_certificate_validation() {
        let config = DeviceTrustConfig::default();
        let manager = DeviceTrustManager::new(config)?;

        let device = manager
            .register_device(
                "test-device".to_string(),
                DeviceType::Server,
                "HW999".to_string(),
            )
            .await
            .unwrap();

        // For testing, create a dummy certificate
        // In production, use a real X.509 certificate
        let cert_data = include_bytes!("../test_data/test_cert.der");

        // This will fail as we don't have a real certificate
        let result = manager
            .validate_certificate(device.device_id, cert_data.to_vec())
            .await;
        assert!(result.is_err() || !result.unwrap());
    }

    #[tokio::test]
    async fn test_compliance_checking() {
        let config = DeviceTrustConfig::default();
        let manager = DeviceTrustManager::new(config)?;

        let device = manager
            .register_device(
                "test-device".to_string(),
                DeviceType::Laptop,
                "HW111".to_string(),
            )
            .await
            .unwrap();

        let compliance = manager.check_compliance(device.device_id).await?;
        assert!(!compliance.compliant);
        assert!(!compliance.failed_policies.is_empty());
        assert!(compliance.score < 1.0);
    }

    #[tokio::test]
    async fn test_device_state_transitions() {
        let config = DeviceTrustConfig::default();
        let manager = DeviceTrustManager::new(config)?;

        let device = manager
            .register_device(
                "test-device".to_string(),
                DeviceType::Desktop,
                "HW222".to_string(),
            )
            .await
            .unwrap();

        // Transition to trusted
        manager
            .update_device_state(device.device_id, DeviceState::Trusted)
            .await
            .unwrap();
        let updated = manager.get_device(device.device_id).await?;
        assert_eq!(updated.state, DeviceState::Trusted);

        // Transition to quarantined
        manager
            .update_device_state(device.device_id, DeviceState::Quarantined)
            .await
            .unwrap();
        let quarantined = manager.get_device(device.device_id).await?;
        assert_eq!(quarantined.state, DeviceState::Quarantined);
    }

    #[tokio::test]
    async fn test_device_revocation() {
        let config = DeviceTrustConfig::default();
        let manager = DeviceTrustManager::new(config)?;

        let device = manager
            .register_device(
                "test-device".to_string(),
                DeviceType::Mobile,
                "HW333".to_string(),
            )
            .await
            .unwrap();

        // Add some data
        let attestation = DeviceAttestation {
            attestation_type: AttestationType::Software,
            hardware_attestation: None,
            software_attestation: Some(SoftwareAttestation {
                os_measurements: vec![0x01],
                software_inventory_hash: vec![0x02],
                config_hash: vec![0x03],
                security_patches: vec![],
                antivirus_enabled: false,
                firewall_enabled: false,
            }),
            timestamp: Utc::now(),
            signature: vec![0x01],
            nonce: vec![0x02],
        };
        manager
            .attest_device(device.device_id, attestation)
            .await
            .unwrap();

        // Revoke device
        manager.revoke_device(device.device_id).await?;

        // Check state
        let revoked = manager.get_device(device.device_id).await?;
        assert_eq!(revoked.state, DeviceState::Decommissioned);

        // Check data is removed
        assert!(manager.attestations.get(&device.device_id).is_none());
        assert!(manager.trust_factors.get(&device.device_id).is_none());
    }

    #[tokio::test]
    async fn test_attestation_levels() {
        // Test with no attestation required
        let mut config = DeviceTrustConfig::default();
        config.attestation_level = AttestationLevel::None;
        let manager = DeviceTrustManager::new(config)?;

        let device = manager
            .register_device(
                "test-device".to_string(),
                DeviceType::IoT,
                "HW444".to_string(),
            )
            .await
            .unwrap();

        let attestation = DeviceAttestation {
            attestation_type: AttestationType::Software,
            hardware_attestation: None,
            software_attestation: None,
            timestamp: Utc::now(),
            signature: vec![0x01],
            nonce: vec![0x02],
        };

        let result = manager
            .attest_device(device.device_id, attestation)
            .await
            .unwrap();
        assert!(result);

        // Test with enhanced attestation required
        let mut config = DeviceTrustConfig::default();
        config.attestation_level = AttestationLevel::Enhanced;
        let manager = DeviceTrustManager::new(config)?;

        let device = manager
            .register_device(
                "secure-device".to_string(),
                DeviceType::Server,
                "HW555".to_string(),
            )
            .await
            .unwrap();

        let basic_attestation = DeviceAttestation {
            attestation_type: AttestationType::Software,
            hardware_attestation: None,
            software_attestation: Some(SoftwareAttestation {
                os_measurements: vec![0x01],
                software_inventory_hash: vec![0x02],
                config_hash: vec![0x03],
                security_patches: vec![],
                antivirus_enabled: true,
                firewall_enabled: true,
            }),
            timestamp: Utc::now(),
            signature: vec![0x01],
            nonce: vec![0x02],
        };

        let result = manager
            .attest_device(device.device_id, basic_attestation)
            .await
            .unwrap();
        assert!(
            !result,
            "Basic attestation should fail with enhanced requirements"
        );
    }

    #[tokio::test]
    async fn test_trust_score_decay() {
        let mut config = DeviceTrustConfig::default();
        config.trust_decay_rate = 0.1; // 10% per hour
        let manager = DeviceTrustManager::new(config)?;

        let device = manager
            .register_device(
                "test-device".to_string(),
                DeviceType::Desktop,
                "HW666".to_string(),
            )
            .await
            .unwrap();

        // Set up trust factors with 10 hours since update
        manager.trust_factors.insert(
            device.device_id,
            TrustFactors {
                cert_valid: true,
                attestation_passed: true,
                compliance_score: 1.0,
                hours_since_update: 10.0,
                patches_current: true,
                antivirus_enabled: true,
                firewall_enabled: true,
                encryption_enabled: true,
                known_vulnerabilities: 0,
            },
        );

        let score = manager
            .calculate_trust_score(device.device_id)
            .await
            .unwrap();
        assert!(score < 0.95, "Trust score should decay over time");
    }

    #[tokio::test]
    async fn test_vulnerability_impact() {
        let config = DeviceTrustConfig::default();
        let manager = DeviceTrustManager::new(config)?;

        let device = manager
            .register_device(
                "vulnerable-device".to_string(),
                DeviceType::Laptop,
                "HW777".to_string(),
            )
            .await
            .unwrap();

        // Set up trust factors with vulnerabilities
        manager.trust_factors.insert(
            device.device_id,
            TrustFactors {
                cert_valid: true,
                attestation_passed: true,
                compliance_score: 1.0,
                hours_since_update: 0.0,
                patches_current: false,
                antivirus_enabled: true,
                firewall_enabled: true,
                encryption_enabled: true,
                known_vulnerabilities: 10,
            },
        );

        let score = manager
            .calculate_trust_score(device.device_id)
            .await
            .unwrap();
        assert!(
            score < 0.8,
            "High vulnerability count should reduce trust score"
        );
    }

    #[tokio::test]
    async fn test_concurrent_device_operations() {
        let config = DeviceTrustConfig::default();
        let manager = Arc::new(DeviceTrustManager::new(config).unwrap());

        // Register multiple devices concurrently
        let mut handles = vec![];
        for i in 0..10 {
            let manager_clone = manager.clone();
            let handle = tokio::spawn(async move {
                manager_clone
                    .register_device(
                        format!("device-{}", i),
                        DeviceType::Virtual,
                        format!("HW{:03}", i),
                    )
                    .await
            });
            handles.push(handle);
        }

        // All should succeed
        for handle in handles {
            let result = handle.await?;
            assert!(result.is_ok());
        }
    }
}
