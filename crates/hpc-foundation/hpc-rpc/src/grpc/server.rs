//! gRPC server builder with TLS and mTLS support
//!
//! This module handles the complex type transitions in tonic's API:
//! - Server::builder() returns Server
//! - Server::add_service() returns Router (NOT Server!)
//! - Router::add_service() returns Router
//! - Router::serve() is the final step
//!
//! We use an internal enum to track whether we have a Server or Router.

use crate::error::{RpcError, Result};
use hpc_auth::cert::CertificateWithKey;
use std::convert::Infallible;
use std::net::SocketAddr;
use tonic::codegen::{http, Service};
use tonic::server::NamedService;
use tonic::transport::{server::Router, Certificate, Identity, Server, ServerTlsConfig};
use tracing::{debug, info};

/// Internal enum to handle Server -> Router type transition
enum ServerState {
    /// Before any services are added
    Server(Server),
    /// After first service is added
    Router(Router),
}

/// TLS configuration state
struct TlsState {
    identity: Option<Identity>,
    client_ca: Option<Certificate>,
}

/// Builder for creating gRPC servers with optional TLS/mTLS
///
/// # Usage Pattern
///
/// Due to tonic's API design, TLS must be configured BEFORE adding services:
/// ```no_run
/// # use hpc_rpc::grpc::GrpcServerBuilder;
/// # use std::net::SocketAddr;
/// # let addr: SocketAddr = "127.0.0.1:50051".parse().unwrap();
/// let server = GrpcServerBuilder::new(addr)
///     // .with_tls(cert)?           // Optional: configure TLS first
///     // .with_client_auth(ca)?     // Optional: require client certs
///     // .add_service(service1)     // Then add services
///     // .add_service(service2)     // Can add multiple services
///     .build()?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct GrpcServerBuilder {
    state: ServerState,
    addr: SocketAddr,
    tls_state: TlsState,
}

impl GrpcServerBuilder {
    /// Create a new gRPC server builder
    pub fn new(addr: SocketAddr) -> Self {
        debug!(?addr, "Creating gRPC server builder");
        Self {
            state: ServerState::Server(Server::builder()),
            addr,
            tls_state: TlsState {
                identity: None,
                client_ca: None,
            },
        }
    }

    /// Enable TLS with server certificate
    ///
    /// Must be called BEFORE add_service()
    pub fn with_tls(mut self, cert: CertificateWithKey) -> Result<Self> {
        debug!("Enabling TLS for gRPC server");

        // Can only configure TLS before adding services
        if !matches!(self.state, ServerState::Server(_)) {
            return Err(RpcError::Config(
                "Cannot configure TLS after adding services. Call with_tls() before add_service()".to_string()
            ));
        }

        let cert_pem = cert.cert_pem();
        let key_pem = cert.key_pem();
        let identity = Identity::from_pem(cert_pem, key_pem);

        // Store identity for potential reconstruction
        self.tls_state.identity = Some(identity.clone());

        // Apply TLS config
        self.apply_tls_config()?;

        Ok(self)
    }

    /// Enable mTLS by requiring client certificates signed by the given CA
    ///
    /// Must be called AFTER with_tls() and BEFORE add_service()
    pub fn with_client_auth(mut self, ca: CertificateWithKey) -> Result<Self> {
        debug!("Enabling client authentication (mTLS)");

        // Can only configure mTLS before adding services
        if !matches!(self.state, ServerState::Server(_)) {
            return Err(RpcError::Config(
                "Cannot configure mTLS after adding services. Call with_client_auth() before add_service()".to_string()
            ));
        }

        // Require that TLS was configured first
        if self.tls_state.identity.is_none() {
            return Err(RpcError::Config(
                "Must call with_tls() before with_client_auth()".to_string()
            ));
        }

        // Get CA certificate
        let ca_pem = ca.cert_pem();
        let ca_cert = Certificate::from_pem(ca_pem);

        // Store client CA for reconstruction
        self.tls_state.client_ca = Some(ca_cert);

        // Apply TLS config with both identity and client CA
        self.apply_tls_config()?;

        Ok(self)
    }

    /// Apply current TLS configuration to the server
    fn apply_tls_config(&mut self) -> Result<()> {
        let server = match &self.state {
            ServerState::Server(_) => {
                // Extract server temporarily
                let ServerState::Server(server) = std::mem::replace(
                    &mut self.state,
                    ServerState::Server(Server::builder())
                ) else {
                    unreachable!()
                };
                server
            }
            ServerState::Router(_) => {
                return Err(RpcError::Config(
                    "Cannot apply TLS config after adding services".to_string()
                ));
            }
        };

        // Build TLS config
        let mut tls_config = ServerTlsConfig::new();

        if let Some(identity) = &self.tls_state.identity {
            tls_config = tls_config.identity(identity.clone());
        }

        if let Some(ca) = &self.tls_state.client_ca {
            tls_config = tls_config.client_ca_root(ca.clone());
        }

        // Apply config to server
        let configured_server = server
            .tls_config(tls_config)
            .map_err(|e| RpcError::Config(format!("TLS config error: {}", e)))?;

        self.state = ServerState::Server(configured_server);

        Ok(())
    }

    /// Add a gRPC service to the server
    ///
    /// Can be called multiple times to add multiple services.
    /// After the first call, the internal state transitions from Server to Router.
    ///
    /// # Type Parameters
    ///
    /// The service must implement tonic's required traits. This is automatically
    /// satisfied by services generated by tonic-build.
    pub fn add_service<S>(self, service: S) -> Self
    where
        S: Service<http::Request<tonic::transport::Body>,
                   Response = http::Response<tonic::body::BoxBody>,
                   Error = Infallible>
            + NamedService
            + Clone
            + Send
            + 'static,
        S::Future: Send + 'static,
    {
        // We need to match on state to handle Server -> Router transition
        // The issue is that we can't easily move the trait bounds through the match
        // Let's use a different approach: always keep a Router after the first service

        let new_state = match self.state {
            ServerState::Server(mut server) => {
                // First service: Server -> Router
                // Note: add_service takes &mut self, not self
                ServerState::Router(server.add_service(service))
            }
            ServerState::Router(router) => {
                // Subsequent services: Router -> Router
                ServerState::Router(router.add_service(service))
            }
        };

        Self {
            state: new_state,
            addr: self.addr,
            tls_state: self.tls_state,
        }
    }

    /// Build the gRPC server
    ///
    /// Returns a GrpcServer that can be started with `.serve()`
    pub fn build(self) -> Result<GrpcServer> {
        info!(?self.addr, "Building gRPC server");

        match self.state {
            ServerState::Router(router) => {
                Ok(GrpcServer {
                    router,
                    addr: self.addr,
                })
            }
            ServerState::Server(_) => {
                Err(RpcError::Config(
                    "No services added to server. Call add_service() at least once before build()".to_string()
                ))
            }
        }
    }
}

/// gRPC server instance ready to serve requests
pub struct GrpcServer {
    router: Router,
    addr: SocketAddr,
}

impl GrpcServer {
    /// Start serving gRPC requests
    ///
    /// This method will block until the server is shut down.
    pub async fn serve(self) -> Result<()> {
        info!(?self.addr, "Starting gRPC server");

        self.router
            .serve(self.addr)
            .await
            .map_err(RpcError::GrpcTransport)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_creation() {
        let addr: SocketAddr = "127.0.0.1:50051".parse().unwrap();
        let _builder = GrpcServerBuilder::new(addr);
    }
}
