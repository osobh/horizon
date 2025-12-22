# Horizon AuthX

Zero-trust mTLS authentication and service identity for the Horizon platform.

## Features

- **Certificate Generation**: Self-signed and CA-signed certificates via rcgen
- **rustls Integration**: ServerConfig and ClientConfig builders for TLS 1.3
- **mTLS Support**: Mutual authentication with client certificate verification
- **Service Identity**: Extract and validate service names from certificates
- **PEM Support**: Read/write certificates and keys in PEM format
- **Validation**: Expiry checks, hostname verification, chain validation

## Quick Start

```rust
use horizon_authx::cert::{generate_ca_cert, generate_signed_cert, ServiceIdentity};
use horizon_authx::server::create_server_config_with_client_auth;
use horizon_authx::client::create_client_config_with_server_ca;

// Generate CA certificate
let ca = generate_ca_cert("Horizon CA")?;

// Generate server certificate (signed by CA)
let server_identity = ServiceIdentity::new("telemetry-collector");
let server_cert = generate_signed_cert(&server_identity, &ca)?;

// Generate client certificate (signed by CA)
let client_identity = ServiceIdentity::new("node-agent");
let client_cert = generate_signed_cert(&client_identity, &ca)?;

// Create mTLS server config (requires client certificates)
let server_config = create_server_config_with_client_auth(&server_cert, &ca)?;

// Create mTLS client config (verifies server with CA)
let client_config = create_client_config_with_server_ca(&client_cert, &ca)?;
```

## Examples

Run the mTLS demo to see the complete workflow:

```bash
cargo run --example mtls_demo
```

## Security Considerations

- **TLS 1.3**: Uses secure cipher suites by default
- **Strong Keys**: RSA 2048+ or EC keys (rcgen defaults)
- **Expiry Validation**: Certificates are checked for expiration
- **Hostname Validation**: Proper DNS/service name verification
- **Thread Safety**: All operations are thread-safe

## Architecture

### Modules

- **cert**: Certificate generation, parsing, and validation
- **server**: rustls ServerConfig builders
- **client**: rustls ClientConfig builders
- **error**: Comprehensive error types

### Dependencies

- `rustls` (0.21): TLS library
- `rcgen` (0.11): Certificate generation
- `x509-parser` (0.15): X.509 parsing
- `rustls-pemfile` (1.0): PEM parsing

## Performance

Benchmark results on standard hardware:

| Operation | Time |
|-----------|------|
| Certificate generation | ~24μs |
| TLS config creation | ~7μs |
| Hostname validation | ~2.5μs |

## Testing

```bash
# Run all tests
cargo test -p horizon-authx

# Run benchmarks
cargo bench -p horizon-authx

# Run specific test
cargo test -p horizon-authx test_cert_generation
```

## Known Limitations

- Certificate verification uses issuer matching (not full cryptographic signature verification)
- Certificate revocation (CRL/OCSP) not implemented
- SPIFFE/SPIRE integration planned for Phase 10

## Future Enhancements

- Full cryptographic signature verification
- Certificate rotation automation
- HSM integration
- SPIFFE/SPIRE support
