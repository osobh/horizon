//! QUIC endpoint creation and management

use crate::error::{RpcError, Result};
use hpc_auth::cert::CertificateWithKey;
use quinn::{ClientConfig, Connection, Endpoint, ServerConfig};
use std::net::SocketAddr;
use std::sync::Arc;
use tracing::{debug, info};

/// QUIC endpoint for client and server connections
///
/// Wraps quinn::Endpoint with Horizon-specific configuration and TLS integration.
pub struct QuicEndpoint {
    endpoint: Endpoint,
}

impl QuicEndpoint {
    /// Create a client endpoint
    ///
    /// # Arguments
    ///
    /// * `bind_addr` - Local address to bind to (use "0.0.0.0:0" for automatic port)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use hpc_rpc::quic::QuicEndpoint;
    /// use std::net::SocketAddr;
    ///
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let addr: SocketAddr = "127.0.0.1:0".parse()?;
    /// let endpoint = QuicEndpoint::client(addr)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn client(bind_addr: SocketAddr) -> Result<Self> {
        debug!(?bind_addr, "Creating QUIC client endpoint");

        let mut endpoint = Endpoint::client(bind_addr)?;

        // Set default client config (no certificate verification for now)
        let crypto = rustls::ClientConfig::builder()
            .with_safe_defaults()
            .with_custom_certificate_verifier(Arc::new(SkipServerVerification))
            .with_no_client_auth();

        let mut client_config = ClientConfig::new(Arc::new(crypto));

        // Configure transport parameters
        let mut transport = quinn::TransportConfig::default();
        transport.max_concurrent_bidi_streams(100u32.into());
        transport.max_concurrent_uni_streams(100u32.into());
        transport.keep_alive_interval(Some(std::time::Duration::from_secs(5)));
        client_config.transport_config(Arc::new(transport));

        endpoint.set_default_client_config(client_config);

        info!("QUIC client endpoint created");
        Ok(Self { endpoint })
    }

    /// Create a server endpoint with TLS
    ///
    /// # Arguments
    ///
    /// * `bind_addr` - Address to listen on
    /// * `cert` - Server certificate (must include private key)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use hpc_rpc::quic::QuicEndpoint;
    /// use hpc_auth::cert::{generate_ca_cert, generate_signed_cert, ServiceIdentity};
    /// use std::net::SocketAddr;
    ///
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let ca = generate_ca_cert("Horizon CA")?;
    /// let identity = ServiceIdentity::new("quic-server");
    /// let cert = generate_signed_cert(&identity, &ca)?;
    ///
    /// let addr: SocketAddr = "127.0.0.1:6000".parse()?;
    /// let endpoint = QuicEndpoint::server(addr, cert)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn server(bind_addr: SocketAddr, cert: CertificateWithKey) -> Result<Self> {
        debug!(?bind_addr, "Creating QUIC server endpoint");

        // Convert CertificateWithKey to rustls format
        let cert_pem = cert.cert_pem();
        let key_pem = cert.key_pem();

        let certs = rustls_pemfile::certs(&mut cert_pem.as_bytes())
            .map_err(|e| RpcError::Config(format!("Failed to parse certificate: {}", e)))?
            .into_iter()
            .map(rustls::Certificate)
            .collect();

        let keys = rustls_pemfile::pkcs8_private_keys(&mut key_pem.as_bytes())
            .map_err(|e| RpcError::Config(format!("Failed to parse private key: {}", e)))?;

        if keys.is_empty() {
            return Err(RpcError::Config("No private key found".to_string()));
        }

        let key = rustls::PrivateKey(keys[0].clone());

        // Build rustls server config
        let crypto = rustls::ServerConfig::builder()
            .with_safe_defaults()
            .with_no_client_auth()
            .with_single_cert(certs, key)
            .map_err(|e| RpcError::Config(format!("TLS config error: {}", e)))?;

        let mut server_config = ServerConfig::with_crypto(Arc::new(crypto));

        // Configure transport parameters
        let mut transport = quinn::TransportConfig::default();
        transport.max_concurrent_bidi_streams(100u32.into());
        transport.max_concurrent_uni_streams(100u32.into());
        transport.keep_alive_interval(Some(std::time::Duration::from_secs(5)));
        server_config.transport_config(Arc::new(transport));

        let endpoint = Endpoint::server(server_config, bind_addr)?;

        info!("QUIC server endpoint created");
        Ok(Self { endpoint })
    }

    /// Connect to a QUIC server (client-side)
    ///
    /// # Arguments
    ///
    /// * `server_addr` - Address of the server to connect to
    /// * `server_name` - Expected server name (for SNI)
    /// * `server_ca` - CA certificate for server verification
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use hpc_rpc::quic::QuicEndpoint;
    /// use hpc_auth::cert::generate_ca_cert;
    /// use std::net::SocketAddr;
    ///
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let ca = generate_ca_cert("Horizon CA")?;
    /// let client_addr: SocketAddr = "127.0.0.1:0".parse()?;
    /// let endpoint = QuicEndpoint::client(client_addr)?;
    ///
    /// let server_addr: SocketAddr = "127.0.0.1:6000".parse()?;
    /// let conn = endpoint.connect(server_addr, "quic-server", ca).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn connect(
        &self,
        server_addr: SocketAddr,
        server_name: &str,
        server_ca: CertificateWithKey,
    ) -> Result<Connection> {
        debug!(?server_addr, server_name, "Connecting to QUIC server");

        // Create custom client config with CA verification
        let ca_pem = server_ca.cert_pem();
        let ca_certs = rustls_pemfile::certs(&mut ca_pem.as_bytes())
            .map_err(|e| RpcError::Config(format!("Failed to parse CA certificate: {}", e)))?;

        let mut root_store = rustls::RootCertStore::empty();
        for cert in ca_certs {
            root_store
                .add(&rustls::Certificate(cert))
                .map_err(|e| RpcError::Config(format!("Failed to add CA cert: {}", e)))?;
        }

        let crypto = rustls::ClientConfig::builder()
            .with_safe_defaults()
            .with_root_certificates(root_store)
            .with_no_client_auth();

        let mut client_config = ClientConfig::new(Arc::new(crypto));

        // Configure transport parameters
        let mut transport = quinn::TransportConfig::default();
        transport.max_concurrent_bidi_streams(100u32.into());
        transport.max_concurrent_uni_streams(100u32.into());
        transport.keep_alive_interval(Some(std::time::Duration::from_secs(5)));
        client_config.transport_config(Arc::new(transport));

        let connecting = self.endpoint.connect_with(client_config, server_addr, server_name)?;
        let connection = connecting.await?;

        info!("QUIC connection established");
        Ok(connection)
    }

    /// Accept incoming connections (server-side)
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use hpc_rpc::quic::QuicEndpoint;
    /// use hpc_auth::cert::{generate_ca_cert, generate_signed_cert, ServiceIdentity};
    /// use std::net::SocketAddr;
    ///
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let ca = generate_ca_cert("Horizon CA")?;
    /// let identity = ServiceIdentity::new("quic-server");
    /// let cert = generate_signed_cert(&identity, &ca)?;
    ///
    /// let addr: SocketAddr = "127.0.0.1:6000".parse()?;
    /// let endpoint = QuicEndpoint::server(addr, cert)?;
    ///
    /// let incoming = endpoint.accept().await.expect("No incoming connection");
    /// let conn = incoming.await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn accept(&self) -> Option<quinn::Connecting> {
        self.endpoint.accept().await
    }

    /// Get the local address this endpoint is bound to
    pub fn local_addr(&self) -> Result<SocketAddr> {
        Ok(self.endpoint.local_addr()?)
    }

    /// Close the endpoint gracefully
    pub fn close(&self, error_code: quinn::VarInt, reason: &[u8]) {
        self.endpoint.close(error_code, reason);
    }
}

/// Custom certificate verifier that skips verification (for testing/development)
///
/// WARNING: This is insecure and should only be used in development.
struct SkipServerVerification;

impl rustls::client::ServerCertVerifier for SkipServerVerification {
    fn verify_server_cert(
        &self,
        _end_entity: &rustls::Certificate,
        _intermediates: &[rustls::Certificate],
        _server_name: &rustls::ServerName,
        _scts: &mut dyn Iterator<Item = &[u8]>,
        _ocsp_response: &[u8],
        _now: std::time::SystemTime,
    ) -> std::result::Result<rustls::client::ServerCertVerified, rustls::Error> {
        Ok(rustls::client::ServerCertVerified::assertion())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_client_endpoint_creation() {
        let addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
        let endpoint = QuicEndpoint::client(addr).unwrap();
        assert!(endpoint.local_addr().is_ok());
    }

    #[tokio::test]
    async fn test_server_endpoint_creation() {
        use hpc_auth::cert::{generate_ca_cert, generate_signed_cert, ServiceIdentity};

        let ca = generate_ca_cert("Test CA").unwrap();
        let identity = ServiceIdentity::new("test-server");
        let cert = generate_signed_cert(&identity, &ca).unwrap();

        let addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
        let endpoint = QuicEndpoint::server(addr, cert).unwrap();
        assert!(endpoint.local_addr().is_ok());
    }
}
