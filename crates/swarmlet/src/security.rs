//! Security utilities for swarmlet

use crate::{Result, SwarmletError};
use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine};
use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

/// Join token for authenticating with clusters
#[derive(Debug, Clone)]
pub struct JoinToken {
    raw_token: String,
    parsed_token: Option<ParsedToken>,
}

/// Parsed join token structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedToken {
    pub cluster_id: String,
    pub expires_at: u64, // Unix timestamp
    pub capabilities: Vec<String>,
    #[serde(default)]
    pub signature: String,
}

/// Node certificate for ongoing authentication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeCertificate {
    pub node_id: String,
    pub cluster_id: String,
    pub public_key: String,
    pub expires_at: u64,
    pub signature: String,
}

impl JoinToken {
    /// Create a new join token
    pub fn new(token: String) -> Self {
        let parsed_token = Self::parse_token(&token).ok();

        Self {
            raw_token: token,
            parsed_token,
        }
    }

    /// Get the raw token string
    pub fn raw_token(&self) -> &str {
        &self.raw_token
    }

    /// Get the parsed token if valid
    pub fn parsed(&self) -> Option<&ParsedToken> {
        self.parsed_token.as_ref()
    }

    /// Check if the token is expired
    pub fn is_expired(&self) -> bool {
        if let Some(parsed) = &self.parsed_token {
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs();

            now > parsed.expires_at
        } else {
            true // Invalid tokens are considered expired
        }
    }

    /// Validate the token format and signature
    pub fn validate(&self) -> Result<()> {
        if self.raw_token.is_empty() {
            return Err(SwarmletError::InvalidToken("Empty token".to_string()));
        }

        if !self.raw_token.starts_with("swarm_join_") {
            return Err(SwarmletError::InvalidToken(
                "Invalid token prefix".to_string(),
            ));
        }

        match &self.parsed_token {
            Some(parsed) => {
                if self.is_expired() {
                    return Err(SwarmletError::InvalidToken("Token expired".to_string()));
                }

                // Basic format validation
                if parsed.cluster_id.is_empty() {
                    return Err(SwarmletError::InvalidToken("Empty cluster ID".to_string()));
                }

                if parsed.signature.is_empty() {
                    return Err(SwarmletError::InvalidToken("Empty signature".to_string()));
                }

                // In a real implementation, this would verify the cryptographic signature
                self.verify_signature(parsed)?;

                Ok(())
            }
            None => Err(SwarmletError::InvalidToken(
                "Could not parse token".to_string(),
            )),
        }
    }

    /// Parse a token string into structured data
    fn parse_token(token: &str) -> Result<ParsedToken> {
        if !token.starts_with("swarm_join_") {
            return Err(SwarmletError::InvalidToken(
                "Invalid token format".to_string(),
            ));
        }

        // Remove prefix
        let token_data = &token[11..]; // Remove "swarm_join_"

        // In a real implementation, this would use proper base64 decoding
        // and cryptographic verification. For now, we'll use a simplified format.

        // Expected format: base64(json_payload).signature
        let parts: Vec<&str> = token_data.split('.').collect();
        if parts.len() != 2 {
            return Err(SwarmletError::InvalidToken(
                "Invalid token structure".to_string(),
            ));
        }

        let payload_b64 = parts[0];
        let signature = parts[1];

        // Decode base64 payload
        let payload_json = Self::decode_base64(payload_b64)?;

        let mut parsed: ParsedToken = serde_json::from_str(&payload_json)?;
        parsed.signature = signature.to_string();

        Ok(parsed)
    }

    /// Verify the token signature
    fn verify_signature(&self, parsed: &ParsedToken) -> Result<()> {
        // Basic format validation
        if parsed.signature.len() < 32 {
            return Err(SwarmletError::InvalidToken(
                "Invalid signature length".to_string(),
            ));
        }

        // For basic validation when no public key is available,
        // just check signature is valid base64
        if let Err(_) = Self::decode_base64_bytes(&parsed.signature) {
            return Err(SwarmletError::InvalidToken(
                "Invalid signature encoding".to_string(),
            ));
        }

        Ok(())
    }

    /// Verify the token signature with a known cluster public key
    pub fn verify_signature_with_key(&self, cluster_public_key: &[u8; 32]) -> Result<()> {
        let parsed = self
            .parsed_token
            .as_ref()
            .ok_or_else(|| SwarmletError::InvalidToken("Token not parsed".to_string()))?;

        // Reconstruct the payload that was signed
        let payload = format!(
            "{}:{}:{}",
            parsed.cluster_id,
            parsed.expires_at,
            parsed.capabilities.join(",")
        );

        // Decode the signature from base64
        let signature_bytes = Self::decode_base64_bytes(&parsed.signature)?;
        if signature_bytes.len() != 64 {
            return Err(SwarmletError::InvalidToken(format!(
                "Invalid signature length: expected 64, got {}",
                signature_bytes.len()
            )));
        }

        let signature_array: [u8; 64] = signature_bytes
            .try_into()
            .map_err(|_| SwarmletError::InvalidToken("Invalid signature format".to_string()))?;

        let signature = Signature::from_bytes(&signature_array);

        // Verify using the cluster's public key
        let verifying_key = VerifyingKey::from_bytes(cluster_public_key)
            .map_err(|e| SwarmletError::InvalidToken(format!("Invalid public key: {}", e)))?;

        verifying_key
            .verify(payload.as_bytes(), &signature)
            .map_err(|_| SwarmletError::InvalidToken("Signature verification failed".to_string()))
    }

    /// Decode base64-encoded string
    fn decode_base64(input: &str) -> Result<String> {
        let bytes = BASE64_STANDARD
            .decode(input)
            .map_err(|e| SwarmletError::InvalidToken(format!("Invalid base64 encoding: {}", e)))?;

        String::from_utf8(bytes)
            .map_err(|e| SwarmletError::InvalidToken(format!("Invalid UTF-8 in token: {}", e)))
    }

    /// Decode base64 to raw bytes
    fn decode_base64_bytes(input: &str) -> Result<Vec<u8>> {
        BASE64_STANDARD
            .decode(input)
            .map_err(|e| SwarmletError::InvalidToken(format!("Invalid base64 encoding: {}", e)))
    }
}

impl NodeCertificate {
    /// Create a new node certificate from PEM data
    pub fn from_pem(pem_data: &str) -> Result<Self> {
        // In a real implementation, this would parse X.509 certificates
        // For now, return a mock certificate

        if pem_data.is_empty() {
            return Err(SwarmletError::InvalidToken(
                "Empty certificate data".to_string(),
            ));
        }

        Ok(NodeCertificate {
            node_id: "mock-node-id".to_string(),
            cluster_id: "mock-cluster-id".to_string(),
            public_key: "mock-public-key".to_string(),
            expires_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs()
                + (30 * 24 * 60 * 60), // 30 days
            signature: "mock-signature".to_string(),
        })
    }

    /// Check if the certificate is expired
    pub fn is_expired(&self) -> bool {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        now > self.expires_at
    }

    /// Validate the certificate
    pub fn validate(&self) -> Result<()> {
        if self.is_expired() {
            return Err(SwarmletError::Authentication(
                "Certificate expired".to_string(),
            ));
        }

        if self.node_id.is_empty() || self.cluster_id.is_empty() {
            return Err(SwarmletError::Authentication(
                "Invalid certificate data".to_string(),
            ));
        }

        // In a real implementation, would verify signature against cluster CA
        Ok(())
    }

    /// Export certificate to PEM format
    pub fn to_pem(&self) -> String {
        let json = serde_json::to_string(self).unwrap_or_default();
        let encoded = BASE64_STANDARD.encode(json.as_bytes());
        format!(
            "-----BEGIN SWARMLET CERTIFICATE-----\n{}\n-----END SWARMLET CERTIFICATE-----",
            encoded
        )
    }
}

/// Generate a secure random string for various security purposes
pub fn generate_secure_random(length: usize) -> String {
    use rand::Rng;

    const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ\
                          abcdefghijklmnopqrstuvwxyz\
                          0123456789";

    let mut rng = rand::thread_rng();
    (0..length)
        .map(|_| {
            let idx = rng.gen_range(0..CHARS.len());
            CHARS[idx] as char
        })
        .collect()
}

/// Hash a password or sensitive data (simplified implementation)
pub fn hash_data(data: &str) -> String {
    // In a real implementation, would use a proper cryptographic hash like Argon2
    // For now, use a simple hash for testing
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    data.hash(&mut hasher);
    format!("{:x}", hasher.finish())
}

/// Encode data to base64
pub fn base64_encode(data: &[u8]) -> String {
    BASE64_STANDARD.encode(data)
}

/// Decode base64 to bytes
pub fn base64_decode(input: &str) -> Result<Vec<u8>> {
    BASE64_STANDARD
        .decode(input)
        .map_err(|e| SwarmletError::Crypto(format!("Invalid base64: {}", e)))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a valid test token with proper base64 encoding
    fn create_test_token() -> String {
        use std::time::{SystemTime, UNIX_EPOCH};

        let expires_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
            + 86400; // 24 hours from now

        let payload = format!(
            r#"{{"cluster_id":"test-cluster","expires_at":{},"capabilities":["basic","workload_execution"]}}"#,
            expires_at
        );

        let payload_b64 = BASE64_STANDARD.encode(payload.as_bytes());
        // Create a fake signature (64 bytes of zeros, base64 encoded)
        let fake_sig = BASE64_STANDARD.encode(&[0u8; 64]);

        format!("swarm_join_{}.{}", payload_b64, fake_sig)
    }

    #[test]
    fn test_join_token_creation() {
        let token_str = create_test_token();
        let token = JoinToken::new(token_str.clone());
        assert_eq!(token.raw_token(), token_str);
    }

    #[test]
    fn test_invalid_token_format() {
        let token = JoinToken::new("invalid_token".to_string());
        assert!(token.validate().is_err());
    }

    #[test]
    fn test_valid_token_format() {
        let token = JoinToken::new(create_test_token());
        // Should parse successfully and pass basic validation
        assert!(token.parsed().is_some(), "Token should be parsed");
        assert!(token.validate().is_ok());
    }

    #[test]
    fn test_token_not_expired() {
        let token = JoinToken::new(create_test_token());
        assert!(!token.is_expired());
    }

    #[test]
    fn test_secure_random_generation() {
        let random1 = generate_secure_random(32);
        let random2 = generate_secure_random(32);

        assert_eq!(random1.len(), 32);
        assert_eq!(random2.len(), 32);
        assert_ne!(random1, random2); // Should be different
    }

    #[test]
    fn test_data_hashing() {
        let hash1 = hash_data("test_data");
        let hash2 = hash_data("test_data");
        let hash3 = hash_data("different_data");

        assert_eq!(hash1, hash2); // Same input should produce same hash
        assert_ne!(hash1, hash3); // Different input should produce different hash
    }

    #[test]
    fn test_node_certificate_creation() {
        let cert = NodeCertificate::from_pem(
            "-----BEGIN CERTIFICATE-----\ntest\n-----END CERTIFICATE-----",
        );
        assert!(cert.is_ok());

        let cert = cert.unwrap();
        assert!(!cert.is_expired());
        assert!(cert.validate().is_ok());
    }

    #[test]
    fn test_certificate_pem_export() {
        let cert = NodeCertificate::from_pem("test").unwrap();
        let pem = cert.to_pem();
        assert!(pem.contains("BEGIN SWARMLET CERTIFICATE"));
        assert!(pem.contains("END SWARMLET CERTIFICATE"));
    }

    #[test]
    fn test_base64_encode_decode_roundtrip() {
        let original = b"Hello, World!";
        let encoded = base64_encode(original);
        let decoded = base64_decode(&encoded).unwrap();
        assert_eq!(original.to_vec(), decoded);
    }

    #[test]
    fn test_base64_decode_invalid() {
        let result = base64_decode("not valid base64 !!!");
        assert!(result.is_err());
    }
}
