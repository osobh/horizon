//! gRPC client builder with connection pooling and timeout support

use crate::error::{RpcError, Result};
use hpc_auth::cert::CertificateWithKey;
use std::time::Duration;
use tonic::transport::{Certificate, Channel, ClientTlsConfig, Endpoint, Identity};
use tracing::{debug, info};

/// Builder for creating gRPC clients with optional TLS/mTLS and connection pooling
///
/// # Examples
///
/// ## Plaintext client
///
/// ```rust,no_run
/// use hpc_rpc::grpc::GrpcClientBuilder;
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     // let client: MyServiceClient<_> = GrpcClientBuilder::new("http://localhost:50051")
///     //     .connect()
///     //     .await?;
///     Ok(())
/// }
/// ```
///
/// ## TLS client
///
/// ```rust,no_run
/// use hpc_rpc::grpc::GrpcClientBuilder;
/// use hpc_auth::cert::{generate_ca_cert, generate_signed_cert, ServiceIdentity};
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let ca = generate_ca_cert("Horizon CA")?;
///
///     // let client: MyServiceClient<_> = GrpcClientBuilder::new("https://localhost:50051")
///     //     .with_server_ca(ca)
///     //     .connect()
///     //     .await?;
///     Ok(())
/// }
/// ```
///
/// ## mTLS client with connection pooling
///
/// ```rust,no_run
/// use hpc_rpc::grpc::GrpcClientBuilder;
/// use hpc_auth::cert::{generate_ca_cert, generate_signed_cert, ServiceIdentity};
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let ca = generate_ca_cert("Horizon CA")?;
///     let client_identity = ServiceIdentity::new("my-client");
///     let client_cert = generate_signed_cert(&client_identity, &ca)?;
///
///     // let client: MyServiceClient<_> = GrpcClientBuilder::new("https://localhost:50051")
///     //     .with_server_ca(ca)
///     //     .with_client_cert(client_cert)
///     //     .with_pool_size(5)
///     //     .with_timeout(Duration::from_secs(10))
///     //     .connect()
///     //     .await?;
///     Ok(())
/// }
/// ```
pub struct GrpcClientBuilder {
    endpoint_url: String,
    server_ca: Option<CertificateWithKey>,
    client_cert: Option<CertificateWithKey>,
    pool_size: Option<usize>,
    timeout: Option<Duration>,
    connect_timeout: Option<Duration>,
}

impl GrpcClientBuilder {
    /// Create a new gRPC client builder
    ///
    /// # Arguments
    ///
    /// * `endpoint_url` - The URL of the gRPC server (e.g., "http://localhost:50051" or "https://localhost:50051")
    pub fn new(endpoint_url: impl Into<String>) -> Self {
        let url = endpoint_url.into();
        debug!(?url, "Creating gRPC client builder");
        Self {
            endpoint_url: url,
            server_ca: None,
            client_cert: None,
            pool_size: None,
            timeout: None,
            connect_timeout: None,
        }
    }

    /// Set the CA certificate for verifying the server's certificate
    pub fn with_server_ca(mut self, ca: CertificateWithKey) -> Self {
        debug!("Setting server CA for TLS verification");
        self.server_ca = Some(ca);
        self
    }

    /// Set the client certificate for mTLS authentication
    pub fn with_client_cert(mut self, cert: CertificateWithKey) -> Self {
        debug!("Setting client certificate for mTLS");
        self.client_cert = Some(cert);
        self
    }

    /// Set the connection pool size (default: 1)
    ///
    /// A pool size > 1 enables concurrent requests over multiple connections.
    pub fn with_pool_size(mut self, size: usize) -> Self {
        debug!(pool_size = size, "Setting connection pool size");
        self.pool_size = Some(size);
        self
    }

    /// Set request timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        debug!(?timeout, "Setting request timeout");
        self.timeout = Some(timeout);
        self
    }

    /// Set connection timeout
    pub fn with_connect_timeout(mut self, timeout: Duration) -> Self {
        debug!(?timeout, "Setting connection timeout");
        self.connect_timeout = Some(timeout);
        self
    }

    /// Connect to the gRPC server and return a Channel
    ///
    /// The Channel can be used to construct any gRPC client stub.
    pub async fn connect<T>(self) -> Result<T>
    where
        T: From<Channel>,
    {
        info!(endpoint = ?self.endpoint_url, "Connecting to gRPC server");

        let mut endpoint = Endpoint::from_shared(self.endpoint_url.clone())
            .map_err(|e| RpcError::Config(format!("Invalid endpoint URL: {}", e)))?;

        // Configure TLS if CA provided
        if let Some(ca) = self.server_ca {
            debug!("Configuring TLS for client");
            let ca_pem = ca.cert_pem();
            let ca_cert = Certificate::from_pem(ca_pem);

            let mut tls_config = ClientTlsConfig::new().ca_certificate(ca_cert);

            // Configure client certificate if provided (mTLS)
            if let Some(client_cert) = self.client_cert {
                debug!("Configuring client certificate (mTLS)");
                let cert_pem = client_cert.cert_pem();
                let key_pem = client_cert.key_pem();
                let identity = Identity::from_pem(cert_pem, key_pem);
                tls_config = tls_config.identity(identity);
            }

            endpoint = endpoint.tls_config(tls_config)
                .map_err(|e| RpcError::Config(format!("TLS config error: {}", e)))?;
        }

        // Configure connection pool
        if let Some(size) = self.pool_size {
            endpoint = endpoint.concurrency_limit(size);
        }

        // Configure timeouts
        if let Some(timeout) = self.timeout {
            endpoint = endpoint.timeout(timeout);
        }

        if let Some(connect_timeout) = self.connect_timeout {
            endpoint = endpoint.connect_timeout(connect_timeout);
        }

        // Connect to the endpoint
        let channel = endpoint.connect().await?;

        debug!("gRPC client connected successfully");
        Ok(T::from(channel))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_creation() {
        let builder = GrpcClientBuilder::new("http://localhost:50051");
        assert_eq!(builder.endpoint_url, "http://localhost:50051");
        assert!(builder.server_ca.is_none());
        assert!(builder.client_cert.is_none());
        assert!(builder.pool_size.is_none());
    }

    #[test]
    fn test_builder_with_tls() {
        use hpc_auth::cert::generate_ca_cert;

        let ca = generate_ca_cert("Test CA").unwrap();

        let builder = GrpcClientBuilder::new("https://localhost:50051")
            .with_server_ca(ca);

        assert!(builder.server_ca.is_some());
    }

    #[test]
    fn test_builder_with_mtls() {
        use hpc_auth::cert::{generate_ca_cert, generate_signed_cert, ServiceIdentity};

        let ca = generate_ca_cert("Test CA").unwrap();
        let identity = ServiceIdentity::new("test-client");
        let cert = generate_signed_cert(&identity, &ca).unwrap();

        let builder = GrpcClientBuilder::new("https://localhost:50051")
            .with_server_ca(ca)
            .with_client_cert(cert);

        assert!(builder.server_ca.is_some());
        assert!(builder.client_cert.is_some());
    }

    #[test]
    fn test_builder_with_pool() {
        let builder = GrpcClientBuilder::new("http://localhost:50051")
            .with_pool_size(10)
            .with_timeout(Duration::from_secs(30));

        assert_eq!(builder.pool_size, Some(10));
        assert_eq!(builder.timeout, Some(Duration::from_secs(30)));
    }
}
