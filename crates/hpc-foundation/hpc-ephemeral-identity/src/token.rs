//! Ephemeral access tokens with cryptographic signing and encryption.
//!
//! Tokens are signed using Ed25519 and optionally encrypted using AES-256-GCM.
//! They contain claims about the identity's permissions and expiration.

use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use chrono::{DateTime, Utc};
use hpc_crypto::{hash, KeyPair, PublicKey, Signature};
use hpc_vault::{MasterKey, VaultEncryption};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::capabilities::CapabilitySet;
use crate::error::{EphemeralError, Result};

/// Claims embedded in an ephemeral token.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenClaims {
    /// Unique token identifier.
    pub jti: Uuid,
    /// Subject (ephemeral identity ID).
    pub sub: Uuid,
    /// Issuer (service that created the token).
    pub iss: String,
    /// Audience (intended recipient).
    #[serde(default)]
    pub aud: Vec<String>,
    /// Issued at timestamp.
    pub iat: DateTime<Utc>,
    /// Expiration timestamp.
    pub exp: DateTime<Utc>,
    /// Not before timestamp.
    #[serde(default)]
    pub nbf: Option<DateTime<Utc>>,
    /// Sponsor identity ID.
    pub sponsor_id: Uuid,
    /// Tenant ID.
    pub tenant_id: Uuid,
    /// Capabilities granted by this token.
    pub capabilities: CapabilitySet,
    /// Device fingerprint (for binding).
    #[serde(default)]
    pub device_fingerprint: Option<String>,
    /// Parent token ID (for derived tokens).
    #[serde(default)]
    pub parent_token_id: Option<Uuid>,
    /// Custom metadata.
    #[serde(default)]
    pub metadata: serde_json::Value,
}

impl TokenClaims {
    /// Creates new token claims.
    #[must_use]
    pub fn new(
        identity_id: Uuid,
        sponsor_id: Uuid,
        tenant_id: Uuid,
        capabilities: CapabilitySet,
        issuer: impl Into<String>,
        ttl: chrono::Duration,
    ) -> Self {
        let now = Utc::now();
        Self {
            jti: Uuid::new_v4(),
            sub: identity_id,
            iss: issuer.into(),
            aud: Vec::new(),
            iat: now,
            exp: now + ttl,
            nbf: None,
            sponsor_id,
            tenant_id,
            capabilities,
            device_fingerprint: None,
            parent_token_id: None,
            metadata: serde_json::Value::Null,
        }
    }

    /// Sets the audience.
    #[must_use]
    pub fn with_audience(mut self, aud: Vec<String>) -> Self {
        self.aud = aud;
        self
    }

    /// Sets the not-before time.
    #[must_use]
    pub fn with_not_before(mut self, nbf: DateTime<Utc>) -> Self {
        self.nbf = Some(nbf);
        self
    }

    /// Sets the device fingerprint.
    #[must_use]
    pub fn with_device_fingerprint(mut self, fingerprint: impl Into<String>) -> Self {
        self.device_fingerprint = Some(fingerprint.into());
        self
    }

    /// Sets the parent token ID for derived tokens.
    #[must_use]
    pub fn with_parent(mut self, parent_id: Uuid) -> Self {
        self.parent_token_id = Some(parent_id);
        self
    }

    /// Sets custom metadata.
    #[must_use]
    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = metadata;
        self
    }

    /// Checks if the token is expired.
    #[must_use]
    pub fn is_expired(&self) -> bool {
        Utc::now() >= self.exp
    }

    /// Checks if the token is not yet valid.
    #[must_use]
    pub fn is_not_yet_valid(&self) -> bool {
        if let Some(nbf) = self.nbf {
            Utc::now() < nbf
        } else {
            false
        }
    }

    /// Returns time remaining until expiration.
    #[must_use]
    pub fn time_remaining(&self) -> chrono::Duration {
        let remaining = self.exp - Utc::now();
        if remaining < chrono::Duration::zero() {
            chrono::Duration::zero()
        } else {
            remaining
        }
    }
}

/// An ephemeral access token.
#[derive(Debug, Clone)]
pub struct EphemeralToken {
    /// The token claims.
    pub claims: TokenClaims,
    /// The encoded token string (header.payload.signature).
    encoded: String,
}

impl EphemeralToken {
    /// Creates and signs a new token.
    ///
    /// # Errors
    ///
    /// Returns an error if serialization fails.
    pub fn create(claims: TokenClaims, keypair: &KeyPair) -> Result<Self> {
        let encoded = Self::encode_and_sign(&claims, keypair)?;
        Ok(Self { claims, encoded })
    }

    /// Creates an encrypted token (for sensitive claims).
    ///
    /// # Errors
    ///
    /// Returns an error if encryption or serialization fails.
    pub fn create_encrypted(
        claims: TokenClaims,
        keypair: &KeyPair,
        encryption_key: MasterKey,
    ) -> Result<Self> {
        let encoded = Self::encode_sign_and_encrypt(&claims, keypair, encryption_key)?;
        Ok(Self { claims, encoded })
    }

    /// Returns the encoded token string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.encoded
    }

    /// Encodes claims and creates a signed token.
    fn encode_and_sign(claims: &TokenClaims, keypair: &KeyPair) -> Result<String> {
        // Create header
        let header = TokenHeader {
            alg: "EdDSA".to_string(),
            typ: "EPH+jwt".to_string(),
            enc: None,
        };

        let header_json = serde_json::to_vec(&header)?;
        let header_b64 = URL_SAFE_NO_PAD.encode(&header_json);

        let claims_json = serde_json::to_vec(claims)?;
        let claims_b64 = URL_SAFE_NO_PAD.encode(&claims_json);

        // Sign header.claims
        let signing_input = format!("{header_b64}.{claims_b64}");
        let signature = keypair.sign(signing_input.as_bytes());
        let sig_b64 = URL_SAFE_NO_PAD.encode(signature.as_bytes());

        Ok(format!("{signing_input}.{sig_b64}"))
    }

    /// Encodes, signs, and encrypts claims.
    fn encode_sign_and_encrypt(
        claims: &TokenClaims,
        keypair: &KeyPair,
        encryption_key: MasterKey,
    ) -> Result<String> {
        // First create the signed token
        let signed = Self::encode_and_sign(claims, keypair)?;

        // Encrypt the signed token
        let vault = VaultEncryption::new(encryption_key);
        let encrypted = vault
            .encrypt(signed.as_bytes())
            .map_err(|e| EphemeralError::CryptoError(e.to_string()))?;

        // Create encrypted header
        let header = TokenHeader {
            alg: "EdDSA".to_string(),
            typ: "EPH+jwt".to_string(),
            enc: Some("A256GCM".to_string()),
        };

        let header_json = serde_json::to_vec(&header)?;
        let header_b64 = URL_SAFE_NO_PAD.encode(&header_json);

        // Encode encrypted payload
        let encrypted_json = serde_json::to_vec(&encrypted)?;
        let encrypted_b64 = URL_SAFE_NO_PAD.encode(&encrypted_json);

        Ok(format!("{header_b64}.{encrypted_b64}"))
    }
}

/// Token header for identifying algorithm and type.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct TokenHeader {
    /// Algorithm used for signing.
    alg: String,
    /// Token type.
    typ: String,
    /// Encryption algorithm (if encrypted).
    #[serde(skip_serializing_if = "Option::is_none")]
    enc: Option<String>,
}

/// Result of validating a token.
#[derive(Debug, Clone)]
pub struct TokenValidation {
    /// The validated claims.
    pub claims: TokenClaims,
    /// Whether the signature was valid.
    pub signature_valid: bool,
    /// Whether the token is currently valid (not expired, not before).
    pub is_valid: bool,
    /// Validation warnings (e.g., nearing expiration).
    pub warnings: Vec<String>,
}

impl TokenValidation {
    /// Creates a successful validation result.
    #[must_use]
    pub fn valid(claims: TokenClaims) -> Self {
        Self {
            claims,
            signature_valid: true,
            is_valid: true,
            warnings: Vec::new(),
        }
    }

    /// Adds a warning to the validation result.
    #[must_use]
    pub fn with_warning(mut self, warning: impl Into<String>) -> Self {
        self.warnings.push(warning.into());
        self
    }
}

/// Token validator for verifying ephemeral tokens.
#[derive(Debug)]
pub struct TokenValidator {
    /// Public key for signature verification.
    public_key: PublicKey,
    /// Encryption key stored as base64 (MasterKey doesn't implement Clone).
    encryption_key_b64: Option<String>,
    /// Expected issuer (if any).
    expected_issuer: Option<String>,
    /// Expected audience (if any).
    expected_audience: Option<String>,
    /// Clock skew tolerance in seconds.
    clock_skew_seconds: i64,
}

impl TokenValidator {
    /// Creates a new token validator.
    #[must_use]
    pub fn new(public_key: PublicKey) -> Self {
        Self {
            public_key,
            encryption_key_b64: None,
            expected_issuer: None,
            expected_audience: None,
            clock_skew_seconds: 60, // 1 minute tolerance
        }
    }

    /// Sets the encryption key for decrypting tokens.
    #[must_use]
    pub fn with_encryption_key(mut self, key: MasterKey) -> Self {
        self.encryption_key_b64 = Some(key.to_base64());
        self
    }

    /// Gets the encryption key (recreated from stored base64).
    fn encryption_key(&self) -> Option<MasterKey> {
        self.encryption_key_b64.as_ref().map(|b64| {
            MasterKey::from_base64(b64).expect("Stored encryption key should always be valid")
        })
    }

    /// Sets the expected issuer.
    #[must_use]
    pub fn with_expected_issuer(mut self, issuer: impl Into<String>) -> Self {
        self.expected_issuer = Some(issuer.into());
        self
    }

    /// Sets the expected audience.
    #[must_use]
    pub fn with_expected_audience(mut self, audience: impl Into<String>) -> Self {
        self.expected_audience = Some(audience.into());
        self
    }

    /// Sets the clock skew tolerance.
    #[must_use]
    pub fn with_clock_skew(mut self, seconds: i64) -> Self {
        self.clock_skew_seconds = seconds;
        self
    }

    /// Validates a token string.
    ///
    /// # Errors
    ///
    /// Returns an error if the token is invalid, expired, or has a bad signature.
    pub fn validate(&self, token: &str) -> Result<TokenValidation> {
        // Split token into parts
        let parts: Vec<&str> = token.split('.').collect();
        if parts.len() < 2 {
            return Err(EphemeralError::TokenValidationFailed(
                "Invalid token format".to_string(),
            ));
        }

        // Decode header
        let header_bytes = URL_SAFE_NO_PAD
            .decode(parts[0])
            .map_err(|e| EphemeralError::TokenValidationFailed(format!("Invalid header: {e}")))?;
        let header: TokenHeader = serde_json::from_slice(&header_bytes)?;

        // Check if token is encrypted
        let (claims, signature_valid) = if header.enc.is_some() {
            self.validate_encrypted(&parts, &header)?
        } else if parts.len() == 3 {
            self.validate_signed(&parts)?
        } else {
            return Err(EphemeralError::TokenValidationFailed(
                "Invalid token format".to_string(),
            ));
        };

        // Validate time constraints
        let now = Utc::now();
        let skew = chrono::Duration::seconds(self.clock_skew_seconds);

        if claims.exp + skew < now {
            return Err(EphemeralError::TokenExpired);
        }

        if let Some(nbf) = claims.nbf {
            if nbf - skew > now {
                return Err(EphemeralError::TokenValidationFailed(
                    "Token not yet valid".to_string(),
                ));
            }
        }

        // Validate issuer
        if let Some(ref expected) = self.expected_issuer {
            if &claims.iss != expected {
                return Err(EphemeralError::TokenValidationFailed(format!(
                    "Invalid issuer: expected {expected}, got {}",
                    claims.iss
                )));
            }
        }

        // Validate audience
        if let Some(ref expected) = self.expected_audience {
            if !claims.aud.contains(expected) {
                return Err(EphemeralError::TokenValidationFailed(
                    "Invalid audience".to_string(),
                ));
            }
        }

        let mut validation = TokenValidation::valid(claims);
        validation.signature_valid = signature_valid;

        // Add warnings
        let remaining = validation.claims.time_remaining();
        if remaining < chrono::Duration::minutes(5) {
            validation = validation.with_warning("Token expires in less than 5 minutes");
        }

        Ok(validation)
    }

    /// Validates a signed (non-encrypted) token.
    fn validate_signed(&self, parts: &[&str]) -> Result<(TokenClaims, bool)> {
        let header_b64 = parts[0];
        let claims_b64 = parts[1];
        let sig_b64 = parts[2];

        // Decode claims
        let claims_bytes = URL_SAFE_NO_PAD.decode(claims_b64).map_err(|e| {
            EphemeralError::TokenValidationFailed(format!("Invalid claims: {e}"))
        })?;
        let claims: TokenClaims = serde_json::from_slice(&claims_bytes)?;

        // Verify signature
        let signing_input = format!("{header_b64}.{claims_b64}");
        let sig_bytes = URL_SAFE_NO_PAD.decode(sig_b64).map_err(|e| {
            EphemeralError::TokenValidationFailed(format!("Invalid signature encoding: {e}"))
        })?;

        let signature = Signature::from_bytes(&sig_bytes)
            .map_err(|e| EphemeralError::CryptoError(e.to_string()))?;

        let signature_valid = self
            .public_key
            .verify(signing_input.as_bytes(), &signature)
            .is_ok();

        if !signature_valid {
            return Err(EphemeralError::InvalidSignature);
        }

        Ok((claims, true))
    }

    /// Validates an encrypted token.
    fn validate_encrypted(&self, parts: &[&str], _header: &TokenHeader) -> Result<(TokenClaims, bool)> {
        if parts.len() != 2 {
            return Err(EphemeralError::TokenValidationFailed(
                "Invalid encrypted token format".to_string(),
            ));
        }

        let encryption_key = self.encryption_key().ok_or_else(|| {
            EphemeralError::TokenValidationFailed("No encryption key provided".to_string())
        })?;

        // Decode encrypted payload
        let encrypted_bytes = URL_SAFE_NO_PAD.decode(parts[1]).map_err(|e| {
            EphemeralError::TokenValidationFailed(format!("Invalid encrypted payload: {e}"))
        })?;

        let encrypted: hpc_vault::EncryptedCredential =
            serde_json::from_slice(&encrypted_bytes)?;

        // Decrypt
        let vault = VaultEncryption::new(encryption_key);
        let decrypted = vault
            .decrypt(&encrypted)
            .map_err(|e| EphemeralError::DecryptionFailed(e.to_string()))?;

        // Parse the inner signed token
        let inner_token = std::str::from_utf8(decrypted.expose_secret())
            .map_err(|e| EphemeralError::TokenValidationFailed(format!("Invalid UTF-8: {e}")))?;

        let inner_parts: Vec<&str> = inner_token.split('.').collect();
        if inner_parts.len() != 3 {
            return Err(EphemeralError::TokenValidationFailed(
                "Invalid inner token format".to_string(),
            ));
        }

        self.validate_signed(&inner_parts)
    }
}

/// Generates a cryptographic fingerprint for tokens.
#[must_use]
pub fn token_fingerprint(token: &str) -> String {
    let hash = hash(token.as_bytes());
    hash.to_hex()[..16].to_string()
}

/// Creates a token hash for storage (doesn't expose full token).
#[must_use]
pub fn token_hash(token: &str) -> String {
    let hash = hash(token.as_bytes());
    hash.to_hex()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capabilities::Capability;

    fn create_test_keypair() -> KeyPair {
        KeyPair::generate()
    }

    fn create_test_claims() -> TokenClaims {
        let capabilities = CapabilitySet::new()
            .with_capability(Capability::new("read", "notebooks"))
            .with_capability(Capability::new("write", "notebooks"));

        TokenClaims::new(
            Uuid::new_v4(),
            Uuid::new_v4(),
            Uuid::new_v4(),
            capabilities,
            "test-issuer",
            chrono::Duration::hours(1),
        )
    }

    // === TokenClaims Tests ===

    #[test]
    fn test_claims_new() {
        let claims = create_test_claims();

        assert!(!claims.is_expired());
        assert!(!claims.is_not_yet_valid());
        assert_eq!(claims.iss, "test-issuer");
    }

    #[test]
    fn test_claims_with_audience() {
        let claims = create_test_claims().with_audience(vec!["api".to_string()]);

        assert!(claims.aud.contains(&"api".to_string()));
    }

    #[test]
    fn test_claims_with_not_before() {
        let future = Utc::now() + chrono::Duration::hours(1);
        let claims = create_test_claims().with_not_before(future);

        assert!(claims.is_not_yet_valid());
    }

    #[test]
    fn test_claims_with_device_fingerprint() {
        let claims = create_test_claims().with_device_fingerprint("fp-123");

        assert_eq!(claims.device_fingerprint, Some("fp-123".to_string()));
    }

    #[test]
    fn test_claims_time_remaining() {
        let claims = create_test_claims();
        let remaining = claims.time_remaining();

        // Should be close to 1 hour
        assert!(remaining.num_minutes() >= 59);
    }

    #[test]
    fn test_claims_is_expired() {
        let mut claims = create_test_claims();
        claims.exp = Utc::now() - chrono::Duration::hours(1);

        assert!(claims.is_expired());
    }

    // === EphemeralToken Tests ===

    #[test]
    fn test_token_create() {
        let keypair = create_test_keypair();
        let claims = create_test_claims();

        let token = EphemeralToken::create(claims, &keypair).unwrap();

        let encoded = token.as_str();
        let parts: Vec<&str> = encoded.split('.').collect();
        assert_eq!(parts.len(), 3); // header.payload.signature
    }

    #[test]
    fn test_token_create_encrypted() {
        let keypair = create_test_keypair();
        let encryption_key = MasterKey::generate();
        let claims = create_test_claims();

        let token = EphemeralToken::create_encrypted(claims, &keypair, encryption_key).unwrap();

        let encoded = token.as_str();
        let parts: Vec<&str> = encoded.split('.').collect();
        assert_eq!(parts.len(), 2); // header.encrypted_payload
    }

    // === TokenValidator Tests ===

    #[test]
    fn test_validator_validate_success() {
        let keypair = create_test_keypair();
        let claims = create_test_claims();
        let token = EphemeralToken::create(claims.clone(), &keypair).unwrap();

        let validator = TokenValidator::new(keypair.public_key());
        let result = validator.validate(token.as_str()).unwrap();

        assert!(result.signature_valid);
        assert!(result.is_valid);
        assert_eq!(result.claims.sub, claims.sub);
    }

    #[test]
    fn test_validator_validate_expired() {
        let keypair = create_test_keypair();
        let mut claims = create_test_claims();
        claims.exp = Utc::now() - chrono::Duration::hours(1);
        let token = EphemeralToken::create(claims, &keypair).unwrap();

        let validator = TokenValidator::new(keypair.public_key());
        let result = validator.validate(token.as_str());

        assert!(matches!(result, Err(EphemeralError::TokenExpired)));
    }

    #[test]
    fn test_validator_validate_wrong_key() {
        let keypair = create_test_keypair();
        let wrong_keypair = create_test_keypair();
        let claims = create_test_claims();
        let token = EphemeralToken::create(claims, &keypair).unwrap();

        let validator = TokenValidator::new(wrong_keypair.public_key());
        let result = validator.validate(token.as_str());

        assert!(matches!(result, Err(EphemeralError::InvalidSignature)));
    }

    #[test]
    fn test_validator_validate_encrypted() {
        let keypair = create_test_keypair();
        let encryption_key = MasterKey::generate();
        let key_b64 = encryption_key.to_base64();
        let claims = create_test_claims();
        let token =
            EphemeralToken::create_encrypted(claims.clone(), &keypair, encryption_key).unwrap();

        // Recreate key from base64 for validation
        let key_for_validation = MasterKey::from_base64(&key_b64).unwrap();
        let validator = TokenValidator::new(keypair.public_key())
            .with_encryption_key(key_for_validation);
        let result = validator.validate(token.as_str()).unwrap();

        assert!(result.signature_valid);
        assert_eq!(result.claims.sub, claims.sub);
    }

    #[test]
    fn test_validator_validate_encrypted_wrong_key() {
        let keypair = create_test_keypair();
        let encryption_key = MasterKey::generate();
        let wrong_key = MasterKey::generate();
        let claims = create_test_claims();
        let token =
            EphemeralToken::create_encrypted(claims, &keypair, encryption_key).unwrap();

        let validator = TokenValidator::new(keypair.public_key())
            .with_encryption_key(wrong_key);
        let result = validator.validate(token.as_str());

        assert!(matches!(result, Err(EphemeralError::DecryptionFailed(_))));
    }

    #[test]
    fn test_validator_with_expected_issuer() {
        let keypair = create_test_keypair();
        let claims = create_test_claims();
        let token = EphemeralToken::create(claims, &keypair).unwrap();

        let validator = TokenValidator::new(keypair.public_key())
            .with_expected_issuer("test-issuer");
        let result = validator.validate(token.as_str());

        assert!(result.is_ok());
    }

    #[test]
    fn test_validator_wrong_issuer() {
        let keypair = create_test_keypair();
        let claims = create_test_claims();
        let token = EphemeralToken::create(claims, &keypair).unwrap();

        let validator = TokenValidator::new(keypair.public_key())
            .with_expected_issuer("wrong-issuer");
        let result = validator.validate(token.as_str());

        assert!(matches!(
            result,
            Err(EphemeralError::TokenValidationFailed(_))
        ));
    }

    #[test]
    fn test_validator_with_clock_skew() {
        let keypair = create_test_keypair();
        let mut claims = create_test_claims();
        // Token expired 30 seconds ago
        claims.exp = Utc::now() - chrono::Duration::seconds(30);
        let token = EphemeralToken::create(claims, &keypair).unwrap();

        // With 60 second skew, should still be valid
        let validator = TokenValidator::new(keypair.public_key())
            .with_clock_skew(60);
        let result = validator.validate(token.as_str());

        assert!(result.is_ok());
    }

    #[test]
    fn test_validator_expiring_soon_warning() {
        let keypair = create_test_keypair();
        let mut claims = create_test_claims();
        claims.exp = Utc::now() + chrono::Duration::minutes(3);
        let token = EphemeralToken::create(claims, &keypair).unwrap();

        let validator = TokenValidator::new(keypair.public_key());
        let result = validator.validate(token.as_str()).unwrap();

        assert!(!result.warnings.is_empty());
        assert!(result.warnings[0].contains("5 minutes"));
    }

    // === Utility Function Tests ===

    #[test]
    fn test_token_fingerprint() {
        let token = "test.token.value";
        let fingerprint = token_fingerprint(token);

        assert_eq!(fingerprint.len(), 16);
        // Same token should produce same fingerprint
        assert_eq!(fingerprint, token_fingerprint(token));
    }

    #[test]
    fn test_token_hash() {
        let token = "test.token.value";
        let hash = token_hash(token);

        assert_eq!(hash.len(), 64); // 32 bytes = 64 hex chars
    }

    // === Serialization Tests ===

    #[test]
    fn test_claims_serialization() {
        let claims = create_test_claims();
        let json = serde_json::to_string(&claims).unwrap();
        let restored: TokenClaims = serde_json::from_str(&json).unwrap();

        assert_eq!(claims.sub, restored.sub);
        assert_eq!(claims.iss, restored.iss);
    }
}
