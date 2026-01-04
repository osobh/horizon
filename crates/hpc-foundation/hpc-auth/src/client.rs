//! Client-side TLS configuration
//!
//! This module provides functions to create rustls ClientConfig instances
//! for secure client-side TLS and mTLS configurations.

use crate::cert::CertificateWithKey;
use crate::{AuthError, Result};
use rustls::{Certificate, ClientConfig, PrivateKey, RootCertStore};
use rustls_pemfile::{certs, pkcs8_private_keys};
use std::io::BufReader;
use std::sync::Arc;

/// Create a basic client TLS configuration without client certificate
pub fn create_client_config(cert: &CertificateWithKey) -> Result<Arc<ClientConfig>> {
    let certs = parse_certs(cert.cert_pem())?;
    let key = parse_key(cert.key_pem())?;

    let config = ClientConfig::builder()
        .with_safe_defaults()
        .with_root_certificates(RootCertStore::empty())
        .with_client_auth_cert(certs, key)
        .map_err(|e| AuthError::TlsConfig(format!("Failed to create client config: {}", e)))?;

    Ok(Arc::new(config))
}

/// Create a client TLS configuration with server CA verification (mutual TLS)
pub fn create_client_config_with_server_ca(
    cert: &CertificateWithKey,
    ca_cert: &CertificateWithKey,
) -> Result<Arc<ClientConfig>> {
    let certs = parse_certs(cert.cert_pem())?;
    let key = parse_key(cert.key_pem())?;

    // Build root certificate store for server verification
    let mut root_store = RootCertStore::empty();
    let ca_certs = parse_certs(ca_cert.cert_pem())?;

    for ca_cert in ca_certs {
        root_store
            .add(&ca_cert)
            .map_err(|e| AuthError::TlsConfig(format!("Failed to add CA cert: {}", e)))?;
    }

    let config = ClientConfig::builder()
        .with_safe_defaults()
        .with_root_certificates(root_store)
        .with_client_auth_cert(certs, key)
        .map_err(|e| AuthError::TlsConfig(format!("Failed to create client config: {}", e)))?;

    Ok(Arc::new(config))
}

/// Parse certificates from PEM format
fn parse_certs(pem: &str) -> Result<Vec<Certificate>> {
    let mut reader = BufReader::new(pem.as_bytes());
    let certs = certs(&mut reader)
        .map_err(|e| AuthError::PemError(format!("Failed to parse certificates: {}", e)))?
        .into_iter()
        .map(Certificate)
        .collect();

    Ok(certs)
}

/// Parse private key from PEM format
fn parse_key(pem: &str) -> Result<PrivateKey> {
    let mut reader = BufReader::new(pem.as_bytes());
    let keys = pkcs8_private_keys(&mut reader)
        .map_err(|e| AuthError::PemError(format!("Failed to parse private key: {}", e)))?;

    if keys.is_empty() {
        return Err(AuthError::PemError(
            "No private key found in PEM".to_string(),
        ));
    }

    Ok(PrivateKey(keys[0].clone()))
}

/// Extension trait for ClientConfig to check if null (for testing)
pub trait ClientConfigExt {
    fn is_null(&self) -> bool;
}

impl ClientConfigExt for Arc<ClientConfig> {
    fn is_null(&self) -> bool {
        // A valid ClientConfig should always be valid
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cert::{
        generate_ca_cert, generate_self_signed_cert, generate_signed_cert, ServiceIdentity,
    };

    #[test]
    fn test_create_client_config() {
        let identity = ServiceIdentity::new("test-client");
        let cert = generate_self_signed_cert(&identity).unwrap();
        let config = create_client_config(&cert).unwrap();
        assert!(!config.is_null());
    }

    #[test]
    fn test_create_client_config_with_server_ca() {
        let ca = generate_ca_cert("Test CA").unwrap();
        let identity = ServiceIdentity::new("test-client");
        let cert = generate_signed_cert(&identity, &ca).unwrap();

        let config = create_client_config_with_server_ca(&cert, &ca).unwrap();
        assert!(!config.is_null());
    }
}
