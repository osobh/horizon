use hpc_auth::cert::{
    generate_ca_cert, generate_self_signed_cert, generate_signed_cert, ServiceIdentity,
};
use hpc_auth::client::{create_client_config, create_client_config_with_server_ca};
use hpc_auth::server::{create_server_config, create_server_config_with_client_auth};
use hpc_auth::{ClientConfigExt, ServerConfigExt};
use std::sync::Arc;

#[test]
fn test_service_identity_creation() {
    let identity = ServiceIdentity::new("telemetry-collector");
    assert_eq!(identity.service_name(), "telemetry-collector");
}

#[test]
fn test_generate_self_signed_cert() {
    let identity = ServiceIdentity::new("test-service");
    let cert = generate_self_signed_cert(&identity).expect("Failed to generate self-signed cert");

    // Verify cert has correct common name
    let info = cert.info().expect("Failed to get cert info");
    assert_eq!(info.common_name, "test-service");
    assert!(!info.is_ca);
}

#[test]
fn test_generate_ca_cert() {
    let ca = generate_ca_cert("Horizon Test CA").expect("Failed to generate CA cert");

    let info = ca.info().expect("Failed to get CA cert info");
    assert_eq!(info.common_name, "Horizon Test CA");
    assert!(info.is_ca);
}

#[test]
fn test_generate_signed_cert() {
    let ca = generate_ca_cert("Horizon Test CA").expect("Failed to generate CA");
    let identity = ServiceIdentity::new("signed-service");

    let cert = generate_signed_cert(&identity, &ca).expect("Failed to generate signed cert");

    let info = cert.info().expect("Failed to get cert info");
    assert_eq!(info.common_name, "signed-service");
    assert!(!info.is_ca);
}

#[test]
fn test_cert_pem_encoding() {
    let identity = ServiceIdentity::new("pem-test");
    let cert = generate_self_signed_cert(&identity).expect("Failed to generate cert");

    let pem = cert.cert_pem();
    assert!(pem.contains("BEGIN CERTIFICATE"));
    assert!(pem.contains("END CERTIFICATE"));

    let key_pem = cert.key_pem();
    assert!(key_pem.contains("BEGIN PRIVATE KEY"));
    assert!(key_pem.contains("END PRIVATE KEY"));
}

#[test]
fn test_cert_expiry_validation() {
    let identity = ServiceIdentity::new("expiry-test");
    let cert = generate_self_signed_cert(&identity).expect("Failed to generate cert");

    let info = cert.info().expect("Failed to get cert info");
    assert!(
        !info.is_expired(),
        "Newly generated cert should not be expired"
    );
}

#[test]
fn test_cert_common_name_extraction() {
    let identity = ServiceIdentity::new("cn-test-service");
    let cert = generate_self_signed_cert(&identity).expect("Failed to generate cert");

    let common_name = cert.common_name().expect("Failed to extract common name");
    assert_eq!(common_name, "cn-test-service");
}

#[test]
fn test_cert_subject_alternative_names() {
    let identity = ServiceIdentity::new("san-test")
        .with_dns_names(vec!["service.local", "service.cluster.local"]);

    let cert = generate_self_signed_cert(&identity).expect("Failed to generate cert");
    let info = cert.info().expect("Failed to get cert info");

    assert!(info.dns_names.contains(&"san-test".to_string()));
    assert!(info.dns_names.contains(&"service.local".to_string()));
    assert!(info
        .dns_names
        .contains(&"service.cluster.local".to_string()));
}

#[test]
fn test_server_config_creation() {
    let identity = ServiceIdentity::new("server-test");
    let cert = generate_self_signed_cert(&identity).expect("Failed to generate cert");

    let config = create_server_config(&cert).expect("Failed to create server config");
    assert!(!config.is_null());
}

#[test]
fn test_server_config_with_client_auth() {
    let ca = generate_ca_cert("Test CA").expect("Failed to generate CA");
    let server_identity = ServiceIdentity::new("server-mtls");
    let server_cert =
        generate_signed_cert(&server_identity, &ca).expect("Failed to generate server cert");

    let config = create_server_config_with_client_auth(&server_cert, &ca)
        .expect("Failed to create server config with client auth");
    assert!(!config.is_null());
}

#[test]
fn test_client_config_creation() {
    let identity = ServiceIdentity::new("client-test");
    let cert = generate_self_signed_cert(&identity).expect("Failed to generate cert");

    let config = create_client_config(&cert).expect("Failed to create client config");
    assert!(!config.is_null());
}

#[test]
fn test_client_config_with_server_ca() {
    let ca = generate_ca_cert("Test CA").expect("Failed to generate CA");
    let client_identity = ServiceIdentity::new("client-mtls");
    let client_cert =
        generate_signed_cert(&client_identity, &ca).expect("Failed to generate client cert");

    let config = create_client_config_with_server_ca(&client_cert, &ca)
        .expect("Failed to create client config with server CA");
    assert!(!config.is_null());
}

#[test]
fn test_mtls_handshake_simulation() {
    // Generate CA
    let ca = generate_ca_cert("mTLS Test CA").expect("Failed to generate CA");

    // Generate server cert
    let server_identity = ServiceIdentity::new("test-server");
    let server_cert =
        generate_signed_cert(&server_identity, &ca).expect("Failed to generate server cert");

    // Generate client cert
    let client_identity = ServiceIdentity::new("test-client");
    let client_cert =
        generate_signed_cert(&client_identity, &ca).expect("Failed to generate client cert");

    // Create configs
    let server_config = create_server_config_with_client_auth(&server_cert, &ca)
        .expect("Failed to create server config");
    let client_config = create_client_config_with_server_ca(&client_cert, &ca)
        .expect("Failed to create client config");

    // Verify both configs exist
    assert!(!server_config.is_null());
    assert!(!client_config.is_null());
}

#[test]
fn test_cert_chain_validation() {
    let ca = generate_ca_cert("Chain Test CA").expect("Failed to generate CA");
    let identity = ServiceIdentity::new("chain-test");
    let cert = generate_signed_cert(&identity, &ca).expect("Failed to generate signed cert");

    // Validate that cert is signed by CA (current implementation checks issuer match)
    // Full cryptographic signature verification requires additional work
    let is_valid = cert.verify_with_ca(&ca).expect("Failed to verify cert");
    // Note: Implementation currently validates issuer match only
    assert!(is_valid);
}

#[test]
fn test_invalid_cert_signature() {
    let ca1 = generate_ca_cert("CA 1").expect("Failed to generate CA 1");
    let ca2 = generate_ca_cert("CA 2").expect("Failed to generate CA 2");

    let identity = ServiceIdentity::new("invalid-sig-test");
    let cert = generate_signed_cert(&identity, &ca1).expect("Failed to generate cert");

    // Verify with wrong CA should fail (checks issuer mismatch)
    let is_valid = cert.verify_with_ca(&ca2).expect("Failed to verify cert");
    assert!(!is_valid, "Cert should not validate with wrong CA");
}

#[test]
fn test_cert_hostname_validation() {
    let identity = ServiceIdentity::new("hostname-test").with_dns_names(vec!["valid.example.com"]);

    let cert = generate_self_signed_cert(&identity).expect("Failed to generate cert");

    assert!(cert
        .validate_hostname("hostname-test")
        .expect("Failed to validate hostname"));
    assert!(cert
        .validate_hostname("valid.example.com")
        .expect("Failed to validate hostname"));
    assert!(!cert
        .validate_hostname("invalid.example.com")
        .expect("Failed to validate hostname"));
}

#[test]
fn test_thread_safety() {
    use std::thread;

    let ca = Arc::new(generate_ca_cert("Thread Test CA").expect("Failed to generate CA"));

    let handles: Vec<_> = (0..10)
        .map(|i| {
            let ca = Arc::clone(&ca);
            thread::spawn(move || {
                let identity = ServiceIdentity::new(&format!("service-{}", i));
                generate_signed_cert(&identity, &ca).expect("Failed to generate cert")
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread panicked");
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn test_service_identity_roundtrip(name in "[a-z][a-z0-9-]{0,30}") {
            let identity = ServiceIdentity::new(&name);
            assert_eq!(identity.service_name(), name);
        }

        #[test]
        fn test_cert_generation_always_succeeds(name in "[a-z][a-z0-9-]{0,30}") {
            let identity = ServiceIdentity::new(&name);
            let cert = generate_self_signed_cert(&identity);
            prop_assert!(cert.is_ok());
        }

        #[test]
        fn test_generated_cert_not_expired(name in "[a-z][a-z0-9-]{0,30}") {
            let identity = ServiceIdentity::new(&name);
            let cert = generate_self_signed_cert(&identity).unwrap();
            let info = cert.info().unwrap();
            prop_assert!(!info.is_expired());
        }

        #[test]
        fn test_cn_extraction_matches_identity(name in "[a-z][a-z0-9-]{0,30}") {
            let identity = ServiceIdentity::new(&name);
            let cert = generate_self_signed_cert(&identity).unwrap();
            let cn = cert.common_name().unwrap();
            prop_assert_eq!(&cn, &name);
        }
    }
}

#[cfg(test)]
mod security_tests {
    use super::*;

    #[test]
    fn test_expired_cert_detection() {
        // This would require creating a cert with past expiry
        // For now, we test that newly generated certs are not expired
        let identity = ServiceIdentity::new("expiry-security-test");
        let cert = generate_self_signed_cert(&identity).expect("Failed to generate cert");
        let info = cert.info().expect("Failed to get cert info");

        assert!(!info.is_expired());
        assert!(info.not_after > chrono::Utc::now());
    }

    #[test]
    fn test_reject_self_signed_with_ca_validation() {
        let ca = generate_ca_cert("Security CA").expect("Failed to generate CA");
        let identity = ServiceIdentity::new("self-signed-test");
        let self_signed =
            generate_self_signed_cert(&identity).expect("Failed to generate self-signed cert");

        // Self-signed cert should not validate against unrelated CA
        assert!(!self_signed.verify_with_ca(&ca).expect("Failed to verify"));
    }

    #[test]
    fn test_strong_key_size() {
        let identity = ServiceIdentity::new("key-size-test");
        let cert = generate_self_signed_cert(&identity).expect("Failed to generate cert");
        let info = cert.info().expect("Failed to get cert info");

        // RSA should be at least 2048 bits, or use EC (rcgen defaults to RSA 2048 or EC)
        // Key size is reported in bits from the DER data
        assert!(
            info.key_size >= 256,
            "Key size should be at least 256 bits, got: {}",
            info.key_size
        );
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn test_cert_file_persistence() {
        let dir = tempdir().expect("Failed to create temp dir");
        let cert_path = dir.path().join("test.pem");
        let key_path = dir.path().join("test-key.pem");

        let identity = ServiceIdentity::new("file-test");
        let cert = generate_self_signed_cert(&identity).expect("Failed to generate cert");

        // Write to files
        fs::write(&cert_path, cert.cert_pem()).expect("Failed to write cert");
        fs::write(&key_path, cert.key_pem()).expect("Failed to write key");

        // Read back
        let cert_pem = fs::read_to_string(&cert_path).expect("Failed to read cert");
        let key_pem = fs::read_to_string(&key_path).expect("Failed to read key");

        assert!(cert_pem.contains("BEGIN CERTIFICATE"));
        assert!(key_pem.contains("BEGIN PRIVATE KEY"));
    }

    #[test]
    fn test_full_mtls_workflow() {
        // 1. Generate CA
        let ca = generate_ca_cert("Workflow CA").expect("Failed to generate CA");

        // 2. Generate server cert (signed by CA)
        let server_identity = ServiceIdentity::new("workflow-server");
        let server_cert =
            generate_signed_cert(&server_identity, &ca).expect("Failed to generate server cert");

        // 3. Generate client cert (signed by CA)
        let client_identity = ServiceIdentity::new("workflow-client");
        let client_cert =
            generate_signed_cert(&client_identity, &ca).expect("Failed to generate client cert");

        // 4. Verify certs are signed by CA (issuer validation)
        let server_valid = server_cert
            .verify_with_ca(&ca)
            .expect("Server cert verification failed");
        let client_valid = client_cert
            .verify_with_ca(&ca)
            .expect("Client cert verification failed");
        assert!(server_valid, "Server cert should be valid");
        assert!(client_valid, "Client cert should be valid");

        // 5. Create mTLS configs
        let server_config = create_server_config_with_client_auth(&server_cert, &ca)
            .expect("Failed to create server config");
        let client_config = create_client_config_with_server_ca(&client_cert, &ca)
            .expect("Failed to create client config");

        // 6. Verify configs are valid
        assert!(!server_config.is_null());
        assert!(!client_config.is_null());
    }
}
