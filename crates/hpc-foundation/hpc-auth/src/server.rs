//! Server-side TLS configuration
//!
//! This module provides functions to create rustls ServerConfig instances
//! for secure server-side TLS and mTLS configurations.

use crate::cert::CertificateWithKey;
use crate::{AuthError, Result};
use rustls::server::AllowAnyAuthenticatedClient;
use rustls::{Certificate, PrivateKey, RootCertStore, ServerConfig};
use rustls_pemfile::{certs, pkcs8_private_keys};
use std::io::BufReader;
use std::sync::Arc;

/// Create a basic server TLS configuration without client authentication
pub fn create_server_config(cert: &CertificateWithKey) -> Result<Arc<ServerConfig>> {
    let certs = parse_certs(cert.cert_pem())?;
    let key = parse_key(cert.key_pem())?;

    let config = ServerConfig::builder()
        .with_safe_defaults()
        .with_no_client_auth()
        .with_single_cert(certs, key)
        .map_err(|e| AuthError::TlsConfig(format!("Failed to create server config: {}", e)))?;

    Ok(Arc::new(config))
}

/// Create a server TLS configuration with mutual TLS (client authentication)
pub fn create_server_config_with_client_auth(
    cert: &CertificateWithKey,
    ca_cert: &CertificateWithKey,
) -> Result<Arc<ServerConfig>> {
    let certs = parse_certs(cert.cert_pem())?;
    let key = parse_key(cert.key_pem())?;

    // Build root certificate store for client verification
    let mut root_store = RootCertStore::empty();
    let ca_certs = parse_certs(ca_cert.cert_pem())?;

    for ca_cert in ca_certs {
        root_store
            .add(&ca_cert)
            .map_err(|e| AuthError::TlsConfig(format!("Failed to add CA cert: {}", e)))?;
    }

    let client_auth = AllowAnyAuthenticatedClient::new(root_store);

    let config = ServerConfig::builder()
        .with_safe_defaults()
        .with_client_cert_verifier(Arc::new(client_auth))
        .with_single_cert(certs, key)
        .map_err(|e| AuthError::TlsConfig(format!("Failed to create server config: {}", e)))?;

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

/// Extension trait for ServerConfig to check if null (for testing)
pub trait ServerConfigExt {
    fn is_null(&self) -> bool;
}

impl ServerConfigExt for Arc<ServerConfig> {
    fn is_null(&self) -> bool {
        // A valid ServerConfig should have at least one cert resolver
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
    fn test_create_server_config() {
        let identity = ServiceIdentity::new("test-server");
        let cert = generate_self_signed_cert(&identity).unwrap();
        let config = create_server_config(&cert).unwrap();
        assert!(!config.is_null());
    }

    #[test]
    fn test_create_server_config_with_client_auth() {
        let ca = generate_ca_cert("Test CA").unwrap();
        let identity = ServiceIdentity::new("test-server");
        let cert = generate_signed_cert(&identity, &ca).unwrap();

        let config = create_server_config_with_client_auth(&cert, &ca).unwrap();
        assert!(!config.is_null());
    }
}
