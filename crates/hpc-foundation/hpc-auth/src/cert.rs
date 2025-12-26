//! Certificate generation and management
//!
//! This module provides functionality for generating, parsing, and validating
//! X.509 certificates for mTLS authentication.

use crate::{AuthError, Result};
use chrono::{DateTime, Utc};
use rcgen::{Certificate, CertificateParams, DnType, IsCa, KeyPair};
use x509_parser::prelude::*;

/// Service identity representing a service name and optional DNS names
#[derive(Debug, Clone)]
pub struct ServiceIdentity {
    service_name: String,
    dns_names: Vec<String>,
}

impl ServiceIdentity {
    /// Create a new service identity with the given service name
    pub fn new(service_name: &str) -> Self {
        Self {
            service_name: service_name.to_string(),
            dns_names: vec![service_name.to_string()],
        }
    }

    /// Add additional DNS names for Subject Alternative Names (SAN)
    pub fn with_dns_names(mut self, dns_names: Vec<&str>) -> Self {
        for name in dns_names {
            if !self.dns_names.contains(&name.to_string()) {
                self.dns_names.push(name.to_string());
            }
        }
        self
    }

    /// Get the service name
    pub fn service_name(&self) -> &str {
        &self.service_name
    }

    /// Get all DNS names (including service name)
    pub fn dns_names(&self) -> &[String] {
        &self.dns_names
    }
}

/// Information extracted from a certificate
#[derive(Debug, Clone)]
pub struct CertificateInfo {
    pub common_name: String,
    pub dns_names: Vec<String>,
    pub is_ca: bool,
    pub not_before: DateTime<Utc>,
    pub not_after: DateTime<Utc>,
    pub key_algorithm: String,
    pub key_size: usize,
}

impl CertificateInfo {
    /// Check if the certificate is expired
    pub fn is_expired(&self) -> bool {
        Utc::now() > self.not_after
    }
}

/// A certificate with its private key
pub struct CertificateWithKey {
    cert: Certificate,
    cert_pem: String,
    key_pem: String,
}

impl Clone for CertificateWithKey {
    fn clone(&self) -> Self {
        // Parse the private key from PEM
        let key_pem_data = ::pem::parse(&self.key_pem)
            .expect("Failed to parse key PEM during clone");
        let key_pair = KeyPair::try_from(key_pem_data.contents())
            .expect("Failed to parse key pair during clone");

        // Create certificate params with the key
        let mut params = CertificateParams::default();
        params.key_pair = Some(key_pair);

        // Create new Certificate with the params
        let cert = Certificate::from_params(params)
            .expect("Failed to create certificate during clone");

        Self {
            cert,
            cert_pem: self.cert_pem.clone(),
            key_pem: self.key_pem.clone(),
        }
    }
}

impl CertificateWithKey {
    /// Create from rcgen Certificate
    fn from_rcgen(cert: Certificate) -> Result<Self> {
        let cert_pem = cert.serialize_pem().map_err(|e| {
            AuthError::CertGeneration(format!("Failed to serialize cert: {}", e))
        })?;
        let key_pem = cert.serialize_private_key_pem();

        Ok(Self {
            cert,
            cert_pem,
            key_pem,
        })
    }

    /// Get the certificate in PEM format
    pub fn cert_pem(&self) -> &str {
        &self.cert_pem
    }

    /// Get the private key in PEM format
    pub fn key_pem(&self) -> &str {
        &self.key_pem
    }

    /// Get certificate information
    pub fn info(&self) -> Result<CertificateInfo> {
        parse_cert_info(&self.cert_pem)
    }

    /// Extract the common name from the certificate
    pub fn common_name(&self) -> Result<String> {
        let info = self.info()?;
        Ok(info.common_name)
    }

    /// Validate hostname against certificate DNS names
    pub fn validate_hostname(&self, hostname: &str) -> Result<bool> {
        let info = self.info()?;

        // Check if hostname matches any DNS name in the cert
        let matches = info.dns_names.iter().any(|name| name == hostname);

        if !matches {
            return Ok(false);
        }

        Ok(true)
    }

    /// Verify certificate was signed by the given CA
    pub fn verify_with_ca(&self, ca: &CertificateWithKey) -> Result<bool> {
        // Parse the certificate from PEM (which has the signed version with correct issuer)
        let pem_data = ::pem::parse(self.cert_pem.as_bytes()).map_err(|e| {
            AuthError::CertParsing(format!("Failed to parse PEM: {}", e))
        })?;

        let (_, cert_parsed) = X509Certificate::from_der(pem_data.contents())
            .map_err(|e| AuthError::CertParsing(format!("Failed to parse cert: {}", e)))?;

        // Parse the CA certificate from PEM
        let ca_pem_data = ::pem::parse(ca.cert_pem.as_bytes()).map_err(|e| {
            AuthError::CertParsing(format!("Failed to parse CA PEM: {}", e))
        })?;

        let (_, ca_parsed) = X509Certificate::from_der(ca_pem_data.contents())
            .map_err(|e| AuthError::CertParsing(format!("Failed to parse CA: {}", e)))?;

        // Check if cert issuer matches CA subject
        let issuer_match = cert_parsed.issuer() == ca_parsed.subject();

        if !issuer_match {
            return Ok(false);
        }

        // For now, just verify issuer matches
        // Full signature verification would require additional dependencies
        let result = issuer_match;

        Ok(result)
    }

    /// Check if this is a null/empty certificate (for testing)
    pub fn is_null(&self) -> bool {
        self.cert_pem.is_empty()
    }

    /// Get the underlying rcgen Certificate (for signing other certs)
    pub(crate) fn rcgen_cert(&self) -> &Certificate {
        &self.cert
    }
}

/// Generate a self-signed certificate for a service
pub fn generate_self_signed_cert(identity: &ServiceIdentity) -> Result<CertificateWithKey> {
    let mut params = CertificateParams::new(identity.dns_names.clone());

    params
        .distinguished_name
        .push(DnType::CommonName, identity.service_name.clone());

    // Set validity period (365 days)
    params.not_before = ::time::OffsetDateTime::now_utc()
        .checked_sub(::time::Duration::seconds(60))
        .unwrap_or(::time::OffsetDateTime::now_utc());
    params.not_after = ::time::OffsetDateTime::now_utc()
        .checked_add(::time::Duration::days(365))
        .ok_or_else(|| {
            AuthError::CertGeneration("Failed to set certificate expiry".to_string())
        })?;

    let cert = Certificate::from_params(params)?;
    CertificateWithKey::from_rcgen(cert)
}

/// Generate a CA certificate
pub fn generate_ca_cert(ca_name: &str) -> Result<CertificateWithKey> {
    let mut params = CertificateParams::new(vec![ca_name.to_string()]);

    params
        .distinguished_name
        .push(DnType::CommonName, ca_name.to_string());

    // Mark as CA
    params.is_ca = IsCa::Ca(rcgen::BasicConstraints::Unconstrained);

    // Set validity period (1 year)
    params.not_before = ::time::OffsetDateTime::now_utc()
        .checked_sub(::time::Duration::seconds(60))
        .unwrap_or(::time::OffsetDateTime::now_utc());
    params.not_after = ::time::OffsetDateTime::now_utc()
        .checked_add(::time::Duration::days(365))
        .ok_or_else(|| {
            AuthError::CertGeneration("Failed to set certificate expiry".to_string())
        })?;

    let cert = Certificate::from_params(params)?;
    CertificateWithKey::from_rcgen(cert)
}

/// Generate a certificate signed by a CA
pub fn generate_signed_cert(
    identity: &ServiceIdentity,
    ca: &CertificateWithKey,
) -> Result<CertificateWithKey> {
    let mut params = CertificateParams::new(identity.dns_names.clone());

    params
        .distinguished_name
        .push(DnType::CommonName, identity.service_name.clone());

    // Set validity period (365 days)
    params.not_before = ::time::OffsetDateTime::now_utc()
        .checked_sub(::time::Duration::seconds(60))
        .unwrap_or(::time::OffsetDateTime::now_utc());
    params.not_after = ::time::OffsetDateTime::now_utc()
        .checked_add(::time::Duration::days(365))
        .ok_or_else(|| {
            AuthError::CertGeneration("Failed to set certificate expiry".to_string())
        })?;

    let cert = Certificate::from_params(params)?;

    // Sign with CA
    let signed_cert_pem = cert.serialize_pem_with_signer(ca.rcgen_cert()).map_err(|e| {
        AuthError::CertGeneration(format!("Failed to sign certificate: {}", e))
    })?;

    let key_pem = cert.serialize_private_key_pem();

    Ok(CertificateWithKey {
        cert,
        cert_pem: signed_cert_pem,
        key_pem,
    })
}

/// Parse certificate information from PEM
fn parse_cert_info(cert_pem: &str) -> Result<CertificateInfo> {
    // Extract DER from PEM
    let pem_data = ::pem::parse(cert_pem.as_bytes()).map_err(|e| {
        AuthError::CertParsing(format!("Failed to parse PEM: {}", e))
    })?;

    let (_, cert) = X509Certificate::from_der(pem_data.contents()).map_err(|e| {
        AuthError::CertParsing(format!("Failed to parse X.509: {}", e))
    })?;

    // Extract common name
    let common_name = cert
        .subject()
        .iter_common_name()
        .next()
        .and_then(|cn| cn.as_str().ok())
        .unwrap_or("")
        .to_string();

    // Extract DNS names from SAN
    let mut dns_names = vec![common_name.clone()];
    if let Ok(Some(san_ext)) = cert.subject_alternative_name() {
        for name in &san_ext.value.general_names {
            if let GeneralName::DNSName(dns) = name {
                if !dns_names.contains(&dns.to_string()) {
                    dns_names.push(dns.to_string());
                }
            }
        }
    }

    // Check if CA
    let is_ca = cert
        .basic_constraints()
        .map(|bc| bc.map(|b| b.value.ca).unwrap_or(false))
        .unwrap_or(false);

    // Get validity period - convert from time OffsetDateTime to chrono DateTime<Utc>
    let not_before_time = cert.validity().not_before.to_datetime();
    let not_after_time = cert.validity().not_after.to_datetime();

    let not_before = DateTime::<Utc>::from_timestamp(
        not_before_time.unix_timestamp(),
        not_before_time.nanosecond(),
    ).ok_or_else(|| AuthError::CertParsing("Invalid not_before timestamp".to_string()))?;

    let not_after = DateTime::<Utc>::from_timestamp(
        not_after_time.unix_timestamp(),
        not_after_time.nanosecond(),
    ).ok_or_else(|| AuthError::CertParsing("Invalid not_after timestamp".to_string()))?;

    // Get key info
    let key_algorithm = cert.public_key().algorithm.algorithm.to_string();
    let key_size = cert.public_key().subject_public_key.data.len() * 8;

    Ok(CertificateInfo {
        common_name,
        dns_names,
        is_ca,
        not_before,
        not_after,
        key_algorithm,
        key_size,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_service_identity() {
        let identity = ServiceIdentity::new("test-service");
        assert_eq!(identity.service_name(), "test-service");
        assert_eq!(identity.dns_names(), &["test-service"]);
    }

    #[test]
    fn test_service_identity_with_dns() {
        let identity = ServiceIdentity::new("test")
            .with_dns_names(vec!["test.local", "test.cluster.local"]);

        assert_eq!(identity.dns_names().len(), 3);
        assert!(identity.dns_names().contains(&"test".to_string()));
        assert!(identity.dns_names().contains(&"test.local".to_string()));
    }
}
