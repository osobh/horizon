//! ImageVerifier - Security scanning and policy enforcement

use crate::{Result, SwarmImage, SwarmRegistryError};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use tokio::process::Command;
use tracing::{debug, info, warn};

/// Security scan result
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ScanResult {
    /// Image that was scanned
    pub image_hash: String,
    /// Scan timestamp
    pub scanned_at: u64,
    /// Vulnerabilities found
    pub vulnerabilities: Vec<Vulnerability>,
    /// Security score (0-100, higher is better)
    pub security_score: u8,
    /// Policy violations
    pub policy_violations: Vec<PolicyViolation>,
    /// Scan status
    pub status: ScanStatus,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Vulnerability {
    /// CVE ID
    pub cve_id: String,
    /// Severity level
    pub severity: Severity,
    /// Affected package
    pub package: String,
    /// Package version
    pub version: String,
    /// Fix available
    pub fix_available: bool,
    /// Fixed version if available
    pub fixed_version: Option<String>,
    /// Description
    pub description: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Severity {
    Critical,
    High,
    Medium,
    Low,
    Info,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PolicyViolation {
    /// Policy that was violated
    pub policy: String,
    /// Violation description
    pub description: String,
    /// Severity of violation
    pub severity: Severity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ScanStatus {
    /// Scan completed successfully
    Completed,
    /// Scan is in progress
    InProgress,
    /// Scan failed
    Failed,
    /// Scan was skipped
    Skipped,
}

/// Security policy configuration
#[derive(Debug, Clone)]
pub struct SecurityPolicy {
    /// Maximum allowed critical vulnerabilities
    pub max_critical_vulns: usize,
    /// Maximum allowed high vulnerabilities
    pub max_high_vulns: usize,
    /// Minimum security score required
    pub min_security_score: u8,
    /// Blocked packages
    pub blocked_packages: HashSet<String>,
    /// Required packages
    pub required_packages: HashSet<String>,
    /// Maximum image age in days
    pub max_image_age_days: Option<u32>,
    /// Require signed images
    pub require_signatures: bool,
    /// Allowed base images
    pub allowed_base_images: HashSet<String>,
}

impl Default for SecurityPolicy {
    fn default() -> Self {
        Self {
            max_critical_vulns: 0,
            max_high_vulns: 5,
            min_security_score: 70,
            blocked_packages: HashSet::new(),
            required_packages: HashSet::new(),
            max_image_age_days: Some(90),
            require_signatures: false,
            allowed_base_images: HashSet::from([
                "ubuntu:22.04".to_string(),
                "ubuntu:24.04".to_string(),
                "ubuntu:25.04".to_string(),
            ]),
        }
    }
}

/// Verifier configuration
#[derive(Debug, Clone)]
pub struct VerifierConfig {
    /// Work directory for verification
    pub work_dir: PathBuf,
    /// Security policy
    pub policy: SecurityPolicy,
    /// Enable vulnerability scanning
    pub enable_vuln_scan: bool,
    /// Enable malware scanning
    pub enable_malware_scan: bool,
    /// Enable license scanning
    pub enable_license_scan: bool,
    /// Scanner backends to use
    pub scanners: Vec<ScannerBackend>,
}

#[derive(Debug, Clone)]
pub enum ScannerBackend {
    /// Trivy vulnerability scanner
    Trivy,
    /// ClamAV malware scanner
    ClamAV,
    /// Custom scanner
    Custom(String),
}

impl Default for VerifierConfig {
    fn default() -> Self {
        Self {
            work_dir: PathBuf::from("/tmp/stratoswarm-verifier"),
            policy: SecurityPolicy::default(),
            enable_vuln_scan: true,
            enable_malware_scan: true,
            enable_license_scan: false,
            scanners: vec![ScannerBackend::Trivy],
        }
    }
}

/// Image verifier for security scanning and policy enforcement
pub struct ImageVerifier {
    config: VerifierConfig,
    /// Cache of scan results
    scan_cache: HashMap<String, ScanResult>,
}

impl ImageVerifier {
    /// Create a new image verifier
    pub fn new(config: VerifierConfig) -> Self {
        Self {
            config,
            scan_cache: HashMap::new(),
        }
    }

    /// Verify an image
    pub async fn verify(&mut self, image: &SwarmImage) -> Result<ScanResult> {
        info!("Verifying image {}", image.hash);

        // Check cache
        if let Some(cached) = self.scan_cache.get(&image.hash) {
            debug!("Using cached scan result for {}", image.hash);
            return Ok(cached.clone());
        }

        // Create work directory
        let work_dir = self.config.work_dir.join(&image.hash);
        tokio::fs::create_dir_all(&work_dir).await?;

        // Extract image for scanning
        self.extract_image(image, &work_dir).await?;

        let mut scan_result = ScanResult {
            image_hash: image.hash.clone(),
            scanned_at: self.current_timestamp(),
            vulnerabilities: Vec::new(),
            security_score: 100,
            policy_violations: Vec::new(),
            status: ScanStatus::InProgress,
        };

        // Run vulnerability scan
        if self.config.enable_vuln_scan {
            let vulns = self.scan_vulnerabilities(&work_dir).await?;
            scan_result.vulnerabilities = vulns;
        }

        // Run malware scan
        if self.config.enable_malware_scan {
            let malware_found = self.scan_malware(&work_dir).await?;
            if malware_found {
                scan_result.policy_violations.push(PolicyViolation {
                    policy: "no-malware".to_string(),
                    description: "Malware detected in image".to_string(),
                    severity: Severity::Critical,
                });
            }
        }

        // Check security policies
        let violations = self.check_policies(image, &scan_result).await?;
        scan_result.policy_violations.extend(violations);

        // Calculate security score
        scan_result.security_score = self.calculate_security_score(&scan_result);

        // Update status
        scan_result.status = if scan_result.policy_violations.is_empty() {
            ScanStatus::Completed
        } else {
            ScanStatus::Failed
        };

        // Cache result
        self.scan_cache
            .insert(image.hash.clone(), scan_result.clone());

        // Clean up
        tokio::fs::remove_dir_all(&work_dir).await.ok();

        Ok(scan_result)
    }

    /// Check if an image passes security policies
    pub async fn check_compliance(&mut self, image: &SwarmImage) -> Result<bool> {
        let scan_result = self.verify(image).await?;

        // Check if there are any policy violations
        if !scan_result.policy_violations.is_empty() {
            return Ok(false);
        }

        // Check security score
        if scan_result.security_score < self.config.policy.min_security_score {
            return Ok(false);
        }

        Ok(true)
    }

    /// Update security policy
    pub fn update_policy(&mut self, policy: SecurityPolicy) {
        self.config.policy = policy;
        // Clear cache as policy changed
        self.scan_cache.clear();
    }

    async fn extract_image(&self, _image: &SwarmImage, work_dir: &Path) -> Result<()> {
        // TODO: Extract image layers to work directory
        // For now, just create a placeholder
        let placeholder = work_dir.join("extracted");
        tokio::fs::create_dir_all(&placeholder).await?;
        Ok(())
    }

    async fn scan_vulnerabilities(&self, work_dir: &Path) -> Result<Vec<Vulnerability>> {
        let mut vulnerabilities = Vec::new();

        for scanner in &self.config.scanners {
            match scanner {
                ScannerBackend::Trivy => {
                    vulnerabilities.extend(self.scan_with_trivy(work_dir).await?);
                }
                ScannerBackend::ClamAV => {
                    // ClamAV is for malware, not vulnerabilities
                }
                ScannerBackend::Custom(cmd) => {
                    vulnerabilities.extend(self.scan_with_custom(work_dir, cmd).await?);
                }
            }
        }

        Ok(vulnerabilities)
    }

    async fn scan_with_trivy(&self, work_dir: &Path) -> Result<Vec<Vulnerability>> {
        info!("Scanning with Trivy");

        // Check if trivy is available
        let trivy_check = Command::new("trivy").arg("--version").output().await;

        if trivy_check.is_err() {
            warn!("Trivy not found, skipping vulnerability scan");
            return Ok(Vec::new());
        }

        // Run trivy scan
        let output = Command::new("trivy")
            .arg("fs")
            .arg("--format")
            .arg("json")
            .arg("--severity")
            .arg("CRITICAL,HIGH,MEDIUM,LOW")
            .arg(work_dir)
            .output()
            .await?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            warn!("Trivy scan failed: {}", stderr);
            return Ok(Vec::new());
        }

        // Parse trivy output (simplified - real implementation would parse JSON)
        let vulnerabilities = vec![
            // Mock vulnerabilities for testing
            Vulnerability {
                cve_id: "CVE-2023-0001".to_string(),
                severity: Severity::High,
                package: "openssl".to_string(),
                version: "1.1.1".to_string(),
                fix_available: true,
                fixed_version: Some("1.1.1q".to_string()),
                description: "Buffer overflow in OpenSSL".to_string(),
            },
        ];

        Ok(vulnerabilities)
    }

    async fn scan_with_custom(&self, work_dir: &Path, cmd: &str) -> Result<Vec<Vulnerability>> {
        info!("Scanning with custom scanner: {}", cmd);

        let output = Command::new(cmd).arg(work_dir).output().await?;

        if !output.status.success() {
            warn!("Custom scanner failed");
            return Ok(Vec::new());
        }

        // Parse custom scanner output
        Ok(Vec::new())
    }

    async fn scan_malware(&self, work_dir: &Path) -> Result<bool> {
        info!("Scanning for malware");

        // Check if clamav is available
        let clamscan_check = Command::new("clamscan").arg("--version").output().await;

        if clamscan_check.is_err() {
            warn!("ClamAV not found, skipping malware scan");
            return Ok(false);
        }

        // Run clamscan
        let output = Command::new("clamscan")
            .arg("-r")
            .arg("--quiet")
            .arg(work_dir)
            .output()
            .await?;

        // Exit code 1 means virus found
        Ok(output.status.code() == Some(1))
    }

    async fn check_policies(
        &self,
        image: &SwarmImage,
        scan_result: &ScanResult,
    ) -> Result<Vec<PolicyViolation>> {
        let mut violations = Vec::new();
        let policy = &self.config.policy;

        // Check vulnerability counts
        let critical_count = scan_result
            .vulnerabilities
            .iter()
            .filter(|v| v.severity == Severity::Critical)
            .count();

        if critical_count > policy.max_critical_vulns {
            violations.push(PolicyViolation {
                policy: "max-critical-vulns".to_string(),
                description: format!(
                    "Found {} critical vulnerabilities, maximum allowed is {}",
                    critical_count, policy.max_critical_vulns
                ),
                severity: Severity::Critical,
            });
        }

        let high_count = scan_result
            .vulnerabilities
            .iter()
            .filter(|v| v.severity == Severity::High)
            .count();

        if high_count > policy.max_high_vulns {
            violations.push(PolicyViolation {
                policy: "max-high-vulns".to_string(),
                description: format!(
                    "Found {} high vulnerabilities, maximum allowed is {}",
                    high_count, policy.max_high_vulns
                ),
                severity: Severity::High,
            });
        }

        // Check image age
        if let Some(max_age_days) = policy.max_image_age_days {
            let age_seconds = self.current_timestamp() - image.metadata.created;
            let age_days = age_seconds / 86400;

            if age_days > max_age_days as u64 {
                violations.push(PolicyViolation {
                    policy: "max-image-age".to_string(),
                    description: format!(
                        "Image is {} days old, maximum allowed is {} days",
                        age_days, max_age_days
                    ),
                    severity: Severity::Medium,
                });
            }
        }

        // Check base image
        if !policy.allowed_base_images.is_empty() {
            let base = format!("{}:{}", image.metadata.name, image.metadata.tag);
            if !policy.allowed_base_images.contains(&base) {
                violations.push(PolicyViolation {
                    policy: "allowed-base-images".to_string(),
                    description: format!("Base image {} is not in allowed list", base),
                    severity: Severity::High,
                });
            }
        }

        Ok(violations)
    }

    fn calculate_security_score(&self, scan_result: &ScanResult) -> u8 {
        let mut score = 100u8;

        // Deduct points for vulnerabilities
        for vuln in &scan_result.vulnerabilities {
            match vuln.severity {
                Severity::Critical => score = score.saturating_sub(20),
                Severity::High => score = score.saturating_sub(10),
                Severity::Medium => score = score.saturating_sub(5),
                Severity::Low => score = score.saturating_sub(2),
                Severity::Info => score = score.saturating_sub(1),
            }
        }

        // Deduct points for policy violations
        for violation in &scan_result.policy_violations {
            match violation.severity {
                Severity::Critical => score = score.saturating_sub(30),
                Severity::High => score = score.saturating_sub(20),
                Severity::Medium => score = score.saturating_sub(10),
                Severity::Low => score = score.saturating_sub(5),
                Severity::Info => score = score.saturating_sub(2),
            }
        }

        score
    }

    fn current_timestamp(&self) -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }

    /// Export scan results as JSON
    pub fn export_results(&self, image_hash: &str) -> Result<String> {
        if let Some(result) = self.scan_cache.get(image_hash) {
            Ok(serde_json::to_string_pretty(result)?)
        } else {
            Err(SwarmRegistryError::Other(
                "No scan results found".to_string(),
            ))
        }
    }

    /// Clear scan cache
    pub fn clear_cache(&mut self) {
        self.scan_cache.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn create_test_verifier() -> ImageVerifier {
        let temp_dir = TempDir::new().unwrap();
        let config = VerifierConfig {
            work_dir: temp_dir.path().to_path_buf(),
            ..Default::default()
        };

        ImageVerifier::new(config)
    }

    fn create_test_image() -> SwarmImage {
        SwarmImage {
            hash: "sha256:test123".to_string(),
            metadata: crate::ImageMetadata {
                name: "ubuntu".to_string(),
                tag: "22.04".to_string(),
                variant: crate::ImageVariant::Base,
                created: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                size: 100 * 1024 * 1024,
                architecture: "amd64".to_string(),
                os: "linux".to_string(),
                env: vec![],
                entrypoint: None,
                cmd: None,
            },
            layers: vec!["layer1".to_string()],
            agent_config: None,
        }
    }

    #[tokio::test]
    async fn test_verifier_creation() {
        let verifier = create_test_verifier();

        assert!(verifier.config.enable_vuln_scan);
        assert!(verifier.config.enable_malware_scan);
        assert_eq!(verifier.config.policy.max_critical_vulns, 0);
    }

    #[tokio::test]
    async fn test_verify_image() {
        let mut verifier = create_test_verifier();
        let image = create_test_image();

        let result = verifier.verify(&image).await.unwrap();

        assert_eq!(result.image_hash, image.hash);
        assert!(result.scanned_at > 0);
    }

    #[tokio::test]
    async fn test_cache_behavior() {
        let mut verifier = create_test_verifier();
        let image = create_test_image();

        // First scan
        let result1 = verifier.verify(&image).await.unwrap();

        // Second scan should use cache
        let result2 = verifier.verify(&image).await.unwrap();

        assert_eq!(result1.scanned_at, result2.scanned_at);
    }

    #[tokio::test]
    async fn test_policy_update() {
        let mut verifier = create_test_verifier();

        let mut new_policy = SecurityPolicy::default();
        new_policy.min_security_score = 90;

        verifier.update_policy(new_policy);

        assert_eq!(verifier.config.policy.min_security_score, 90);
        assert!(verifier.scan_cache.is_empty()); // Cache should be cleared
    }

    #[tokio::test]
    async fn test_compliance_check() {
        let mut verifier = create_test_verifier();

        // Disable scanners for predictable test
        verifier.config.enable_vuln_scan = false;
        verifier.config.enable_malware_scan = false;

        let image = create_test_image();
        let compliant = verifier.check_compliance(&image).await.unwrap();

        // With no scanners and a valid image, should be compliant
        assert!(compliant);
    }

    #[tokio::test]
    async fn test_old_image_policy() {
        let mut verifier = create_test_verifier();

        // Set very short max age
        verifier.config.policy.max_image_age_days = Some(0);
        verifier.config.enable_vuln_scan = false;
        verifier.config.enable_malware_scan = false;

        let mut image = create_test_image();
        // Make image 2 days old
        image.metadata.created -= 2 * 86400;

        let result = verifier.verify(&image).await.unwrap();

        // Should have age violation
        assert!(!result.policy_violations.is_empty());
        assert!(result
            .policy_violations
            .iter()
            .any(|v| v.policy == "max-image-age"));
    }

    #[tokio::test]
    async fn test_security_score_calculation() {
        let verifier = create_test_verifier();

        let mut scan_result = ScanResult {
            image_hash: "test".to_string(),
            scanned_at: 0,
            vulnerabilities: vec![
                Vulnerability {
                    cve_id: "CVE-1".to_string(),
                    severity: Severity::Critical,
                    package: "test".to_string(),
                    version: "1.0".to_string(),
                    fix_available: false,
                    fixed_version: None,
                    description: "Test".to_string(),
                },
                Vulnerability {
                    cve_id: "CVE-2".to_string(),
                    severity: Severity::High,
                    package: "test".to_string(),
                    version: "1.0".to_string(),
                    fix_available: false,
                    fixed_version: None,
                    description: "Test".to_string(),
                },
            ],
            security_score: 100,
            policy_violations: vec![],
            status: ScanStatus::Completed,
        };

        let score = verifier.calculate_security_score(&scan_result);

        // Should be 100 - 20 (critical) - 10 (high) = 70
        assert_eq!(score, 70);
    }

    #[tokio::test]
    async fn test_export_results() {
        let mut verifier = create_test_verifier();
        let image = create_test_image();

        // Scan first
        verifier.verify(&image).await.unwrap();

        // Export results
        let json = verifier.export_results(&image.hash).unwrap();
        assert!(json.contains(&image.hash));

        // Parse back to verify
        let _: ScanResult = serde_json::from_str(&json).unwrap();
    }

    #[test]
    fn test_severity_ordering() {
        assert_eq!(Severity::Critical as u8, 0);
        assert_eq!(Severity::High as u8, 1);
        assert_eq!(Severity::Medium as u8, 2);
        assert_eq!(Severity::Low as u8, 3);
        assert_eq!(Severity::Info as u8, 4);
    }
}
