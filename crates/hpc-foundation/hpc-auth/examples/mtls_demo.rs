//! mTLS Certificate Generation Demo
//!
//! This example demonstrates the complete mTLS certificate workflow:
//! 1. Generate a CA certificate
//! 2. Generate server and client certificates signed by the CA
//! 3. Create mTLS server and client configurations
//! 4. Demonstrate certificate verification
//! 5. Show service identity extraction

use hpc_auth::cert::{
    generate_ca_cert, generate_self_signed_cert, generate_signed_cert, ServiceIdentity,
};
use hpc_auth::client::create_client_config_with_server_ca;
use hpc_auth::server::create_server_config_with_client_auth;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Horizon mTLS Certificate Generation Demo ===\n");

    // Step 1: Generate CA Certificate
    println!("Step 1: Generating CA Certificate...");
    let ca = generate_ca_cert("Horizon Root CA")?;
    let ca_info = ca.info()?;
    println!("  ✓ CA Generated:");
    println!("    - Common Name: {}", ca_info.common_name);
    println!("    - Is CA: {}", ca_info.is_ca);
    println!("    - Valid From: {}", ca_info.not_before);
    println!("    - Valid Until: {}", ca_info.not_after);
    println!("    - Key Algorithm: {}", ca_info.key_algorithm);
    println!("    - Key Size: {} bits\n", ca_info.key_size);

    // Step 2: Generate Server Certificate (signed by CA)
    println!("Step 2: Generating Server Certificate...");
    let server_identity = ServiceIdentity::new("telemetry-collector")
        .with_dns_names(vec!["localhost", "telemetry.horizon.local"]);
    let server_cert = generate_signed_cert(&server_identity, &ca)?;
    let server_info = server_cert.info()?;
    println!("  ✓ Server Certificate Generated:");
    println!("    - Common Name: {}", server_info.common_name);
    println!("    - DNS Names: {:?}", server_info.dns_names);
    println!("    - Is CA: {}", server_info.is_ca);
    println!();

    // Step 3: Generate Client Certificate (signed by CA)
    println!("Step 3: Generating Client Certificate...");
    let client_identity =
        ServiceIdentity::new("node-agent").with_dns_names(vec!["agent.horizon.local"]);
    let client_cert = generate_signed_cert(&client_identity, &ca)?;
    let client_info = client_cert.info()?;
    println!("  ✓ Client Certificate Generated:");
    println!("    - Common Name: {}", client_info.common_name);
    println!("    - DNS Names: {:?}", client_info.dns_names);
    println!();

    // Step 4: Create mTLS Server Configuration
    println!("Step 4: Creating mTLS Server Configuration...");
    let _server_config = create_server_config_with_client_auth(&server_cert, &ca)?;
    println!("  ✓ Server Config Created (requires client certificates)\n");

    // Step 5: Create mTLS Client Configuration
    println!("Step 5: Creating mTLS Client Configuration...");
    let _client_config = create_client_config_with_server_ca(&client_cert, &ca)?;
    println!("  ✓ Client Config Created (verifies server with CA)\n");

    // Step 6: Demonstrate Certificate Verification
    println!("Step 6: Certificate Verification...");

    // The current implementation validates issuer matching
    // Full cryptographic signature verification would require additional work
    let server_valid = server_cert.verify_with_ca(&ca)?;
    let client_valid = client_cert.verify_with_ca(&ca)?;

    println!("  ✓ Server Cert Valid (issuer check): {}", server_valid);
    println!("  ✓ Client Cert Valid (issuer check): {}", client_valid);
    println!();

    // Step 7: Service Identity Extraction
    println!("Step 7: Service Identity Extraction...");
    let server_cn = server_cert.common_name()?;
    let client_cn = client_cert.common_name()?;
    println!("  ✓ Server Service: {}", server_cn);
    println!("  ✓ Client Service: {}", client_cn);
    println!();

    // Step 8: Hostname Validation
    println!("Step 8: Hostname Validation...");
    let valid_server = server_cert.validate_hostname("telemetry-collector")?;
    let valid_localhost = server_cert.validate_hostname("localhost")?;
    let invalid = server_cert.validate_hostname("unknown-service")?;

    println!("  ✓ 'telemetry-collector' validates: {}", valid_server);
    println!("  ✓ 'localhost' validates: {}", valid_localhost);
    println!("  ✓ 'unknown-service' validates: {}", invalid);
    println!();

    // Step 9: Self-Signed Certificate (for development)
    println!("Step 9: Generating Self-Signed Certificate (dev only)...");
    let dev_identity = ServiceIdentity::new("dev-service");
    let dev_cert = generate_self_signed_cert(&dev_identity)?;
    let dev_info = dev_cert.info()?;
    println!("  ✓ Dev Certificate Generated:");
    println!("    - Common Name: {}", dev_info.common_name);
    println!("    - Is CA: {}", dev_info.is_ca);
    println!("    - Self-signed (not for production!)");
    println!();

    // Step 10: PEM Export (for storage/transmission)
    println!("Step 10: PEM Export...");
    println!("  ✓ CA Certificate PEM (first 100 chars):");
    println!("    {}", &ca.cert_pem()[..100.min(ca.cert_pem().len())]);
    println!("  ...");
    println!();

    println!("=== Demo Complete ===");
    println!("\nSummary:");
    println!("  - Generated CA, server, and client certificates");
    println!("  - Created mTLS configurations for both sides");
    println!("  - Demonstrated certificate verification");
    println!("  - Showed service identity extraction");
    println!("  - Validated hostnames");
    println!("\nThese certificates can be used for secure service-to-service");
    println!("communication in the Horizon platform with zero-trust mTLS.");

    Ok(())
}
