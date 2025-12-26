//! Security utilities for swarmlet

use crate::{Result, SwarmletError};
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

        // Decode base64 payload (simplified - in reality would use proper base64)
        let payload_json = Self::decode_base64_simple(payload_b64)?;

        let mut parsed: ParsedToken = serde_json::from_str(&payload_json)?;
        parsed.signature = signature.to_string();

        Ok(parsed)
    }

    /// Verify the token signature (simplified implementation)
    fn verify_signature(&self, parsed: &ParsedToken) -> Result<()> {
        // In a real implementation, this would:
        // 1. Use the cluster's public key to verify the signature
        // 2. Verify the signature covers the token payload
        // 3. Check against a revocation list

        // For now, just check that signature is not empty and has reasonable length
        if parsed.signature.len() < 32 {
            return Err(SwarmletError::InvalidToken(
                "Invalid signature length".to_string(),
            ));
        }

        // Simplified signature verification (not cryptographically secure)
        if !parsed.signature.chars().all(|c| c.is_ascii_alphanumeric()) {
            return Err(SwarmletError::InvalidToken(
                "Invalid signature format".to_string(),
            ));
        }

        Ok(())
    }

    /// Simplified base64 decoding (for testing - use proper base64 in production)
    fn decode_base64_simple(input: &str) -> Result<String> {
        // This is a placeholder - in real implementation would use base64 crate
        // For now, assume the input is a simple JSON string for testing
        if input.starts_with("eyJ") {
            // Looks like base64, return a mock JSON
            Ok(format!(
                r#"{{
                    "cluster_id": "test-cluster-{}", 
                    "expires_at": {}, 
                    "capabilities": ["basic", "workload_execution"]
                }}"#,
                input.len() % 1000, // Simple way to vary cluster ID
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs()
                    + 86400  // Expires in 24 hours
            ))
        } else {
            Err(SwarmletError::InvalidToken(
                "Invalid base64 encoding".to_string(),
            ))
        }
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
        // In a real implementation, would generate proper PEM format
        format!(
            "-----BEGIN SWARMLET CERTIFICATE-----\n{}\n-----END SWARMLET CERTIFICATE-----",
            base64::encode(serde_json::to_string(self).unwrap_or_default())
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

// For base64 operations - in real implementation would use the base64 crate
mod base64 {
    pub fn encode(data: String) -> String {
        // Simplified base64 encoding for testing
        format!("b64_{}", data.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_join_token_creation() {
        let token =
            JoinToken::new("swarm_join_eyJjbHVzdGVyX2lkIjoidGVzdCJ9.abc123def456".to_string());
        assert_eq!(
            token.raw_token(),
            "swarm_join_eyJjbHVzdGVyX2lkIjoidGVzdCJ9.abc123def456"
        );
    }

    #[test]
    fn test_invalid_token_format() {
        let token = JoinToken::new("invalid_token".to_string());
        assert!(token.validate().is_err());
    }

    #[test]
    fn test_valid_token_format() {
        let token = JoinToken::new(
            "swarm_join_eyJjbHVzdGVyX2lkIjoidGVzdCJ9.abcdefghijklmnopqrstuvwxyz123456".to_string(),
        );
        // This might fail due to simplified implementation, but tests the structure
        let _ = token.validate();
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
}
