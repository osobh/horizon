//! # Horizon AuthX
//!
//! Zero-trust mTLS authentication and service identity for the Horizon platform.
//!
//! ## Features
//!
//! - **Certificate Generation**: Self-signed and CA-signed certificates via rcgen
//! - **rustls Integration**: ServerConfig and ClientConfig builders
//! - **mTLS Support**: Mutual authentication with client certificate verification
//! - **Service Identity**: Extract and validate service names from certificates
//! - **PEM Support**: Read/write certificates and keys in PEM format
//! - **Validation**: Expiry checks, signature verification, chain validation
//!
//! ## Security Considerations
//!
//! - All certificates use strong defaults (RSA 2048+ or EC)
//! - TLS 1.3 with secure cipher suites
//! - Validate certificate expiry before use
//! - Proper hostname/service name validation
//! - Clear error messages for auth failures
//!
//! ## Example: Generate mTLS certificates
//!
//! ```rust
//! use hpc_auth::cert::{generate_ca_cert, generate_signed_cert, ServiceIdentity};
//! use hpc_auth::server::create_server_config_with_client_auth;
//! use hpc_auth::client::create_client_config_with_server_ca;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // 1. Generate CA certificate
//! let ca = generate_ca_cert("Horizon CA")?;
//!
//! // 2. Generate server certificate (signed by CA)
//! let server_identity = ServiceIdentity::new("telemetry-collector");
//! let server_cert = generate_signed_cert(&server_identity, &ca)?;
//!
//! // 3. Generate client certificate (signed by CA)
//! let client_identity = ServiceIdentity::new("node-agent");
//! let client_cert = generate_signed_cert(&client_identity, &ca)?;
//!
//! // 4. Create mTLS server config
//! let server_config = create_server_config_with_client_auth(&server_cert, &ca)?;
//!
//! // 5. Create mTLS client config
//! let client_config = create_client_config_with_server_ca(&client_cert, &ca)?;
//! # Ok(())
//! # }
//! ```

pub mod cert;
pub mod client;
pub mod server;

mod error;

pub use client::ClientConfigExt;
pub use error::{AuthError, Result};
pub use server::ServerConfigExt;
