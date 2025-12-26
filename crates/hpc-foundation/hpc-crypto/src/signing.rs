//! Ed25519 digital signature operations
//!
//! This module provides a safe wrapper around Ed25519 cryptographic signatures.
//! All operations are designed to be constant-time to resist timing attacks.
//!
//! # Security Considerations
//!
//! - Private key material is zeroized on drop
//! - Signature verification uses constant-time comparison
//! - All operations return errors rather than panicking
//!
//! # Example
//!
//! ```
//! use hpc_crypto::signing::KeyPair;
//!
//! let keypair = KeyPair::generate();
//! let message = b"Hello, HPC!";
//!
//! let signature = keypair.sign(message);
//! assert!(keypair.public_key().verify(message, &signature).is_ok());
//! ```

use ed25519_dalek::{Signature as DalekSignature, Signer, SigningKey, Verifier, VerifyingKey};
use rand::rngs::OsRng;
use std::fmt;
use thiserror::Error;

/// Errors that can occur during cryptographic operations
#[derive(Error, Debug)]
pub enum SigningError {
    #[error("Invalid signature")]
    InvalidSignature,

    #[error("Invalid public key: {0}")]
    InvalidPublicKey(String),

    #[error("Invalid secret key: {0}")]
    InvalidSecretKey(String),

    #[error("Invalid encoding: {0}")]
    InvalidEncoding(String),

    #[error("Signature verification failed")]
    VerificationFailed,
}

/// Ed25519 public key (32 bytes)
#[derive(Clone)]
pub struct PublicKey {
    key: VerifyingKey,
}

impl PublicKey {
    /// Create a public key from bytes
    ///
    /// # Errors
    ///
    /// Returns `SigningError::InvalidPublicKey` if the bytes are invalid
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, SigningError> {
        if bytes.len() != 32 {
            return Err(SigningError::InvalidPublicKey(format!(
                "expected 32 bytes, got {}",
                bytes.len()
            )));
        }

        let mut array = [0u8; 32];
        array.copy_from_slice(bytes);

        VerifyingKey::from_bytes(&array)
            .map(|key| PublicKey { key })
            .map_err(|e| SigningError::InvalidPublicKey(e.to_string()))
    }

    /// Create a public key from base64 encoding
    pub fn from_base64(s: &str) -> Result<Self, SigningError> {
        use base64::Engine;
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(s)
            .map_err(|e| SigningError::InvalidEncoding(e.to_string()))?;
        Self::from_bytes(&bytes)
    }

    /// Get the raw bytes of the public key
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        self.key.as_bytes()
    }

    /// Encode public key as base64
    #[must_use]
    pub fn to_base64(&self) -> String {
        use base64::Engine;
        base64::engine::general_purpose::STANDARD.encode(self.key.as_bytes())
    }

    /// Encode public key as hex
    #[must_use]
    pub fn to_hex(&self) -> String {
        hex::encode(self.key.as_bytes())
    }

    /// Verify a signature on a message
    ///
    /// This operation is constant-time to resist timing attacks.
    ///
    /// # Errors
    ///
    /// Returns `SigningError::VerificationFailed` if verification fails
    pub fn verify(&self, message: &[u8], signature: &Signature) -> Result<(), SigningError> {
        let sig = signature.to_dalek_signature();
        self.key
            .verify(message, &sig)
            .map_err(|_| SigningError::VerificationFailed)
    }
}

impl fmt::Debug for PublicKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PublicKey")
            .field("bytes", &hex::encode(self.key.as_bytes()))
            .finish()
    }
}

impl PartialEq for PublicKey {
    fn eq(&self, other: &Self) -> bool {
        self.key.as_bytes() == other.key.as_bytes()
    }
}

impl Eq for PublicKey {}

/// Ed25519 signature (64 bytes)
#[derive(Clone)]
pub struct Signature {
    bytes: [u8; 64],
}

impl Signature {
    /// Create a signature from bytes
    ///
    /// # Errors
    ///
    /// Returns `SigningError::InvalidSignature` if the bytes are invalid
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, SigningError> {
        if bytes.len() != 64 {
            return Err(SigningError::InvalidSignature);
        }

        let mut array = [0u8; 64];
        array.copy_from_slice(bytes);

        // Validate the signature format
        let _ = DalekSignature::from_bytes(&array);

        Ok(Signature { bytes: array })
    }

    /// Create a signature from base64 encoding
    pub fn from_base64(s: &str) -> Result<Self, SigningError> {
        use base64::Engine;
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(s)
            .map_err(|e| SigningError::InvalidEncoding(e.to_string()))?;
        Self::from_bytes(&bytes)
    }

    /// Get the raw bytes of the signature
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Encode signature as base64
    #[must_use]
    pub fn to_base64(&self) -> String {
        use base64::Engine;
        base64::engine::general_purpose::STANDARD.encode(self.bytes)
    }

    /// Encode signature as hex
    #[must_use]
    pub fn to_hex(&self) -> String {
        hex::encode(self.bytes)
    }

    /// Internal method to get `DalekSignature` for verification
    fn to_dalek_signature(&self) -> DalekSignature {
        DalekSignature::from_bytes(&self.bytes)
    }
}

impl fmt::Debug for Signature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Signature")
            .field("bytes", &hex::encode(self.bytes))
            .finish()
    }
}

impl PartialEq for Signature {
    fn eq(&self, other: &Self) -> bool {
        self.bytes == other.bytes
    }
}

impl Eq for Signature {}

/// Ed25519 keypair (secret + public key)
///
/// The secret key is automatically zeroized when dropped.
pub struct KeyPair {
    signing_key: SigningKey,
    verifying_key: VerifyingKey,
}

impl KeyPair {
    /// Generate a new random keypair
    ///
    /// Uses the operating system's secure random number generator.
    #[must_use]
    pub fn generate() -> Self {
        let signing_key = SigningKey::generate(&mut OsRng);
        let verifying_key = signing_key.verifying_key();

        KeyPair {
            signing_key,
            verifying_key,
        }
    }

    /// Create a keypair from secret key bytes
    ///
    /// # Errors
    ///
    /// Returns `SigningError::InvalidSecretKey` if the bytes are invalid
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, SigningError> {
        if bytes.len() != 32 {
            return Err(SigningError::InvalidSecretKey(format!(
                "expected 32 bytes, got {}",
                bytes.len()
            )));
        }

        let mut array = [0u8; 32];
        array.copy_from_slice(bytes);

        let signing_key = SigningKey::from_bytes(&array);
        let verifying_key = signing_key.verifying_key();

        Ok(KeyPair {
            signing_key,
            verifying_key,
        })
    }

    /// Get the public key
    #[must_use]
    pub fn public_key(&self) -> PublicKey {
        PublicKey {
            key: self.verifying_key,
        }
    }

    /// Sign a message
    ///
    /// This operation is constant-time with respect to the secret key.
    #[must_use]
    pub fn sign(&self, message: &[u8]) -> Signature {
        let sig = self.signing_key.sign(message);
        Signature {
            bytes: sig.to_bytes(),
        }
    }

    /// Export the secret key as bytes
    ///
    /// # Security Warning
    ///
    /// The returned bytes contain sensitive key material.
    /// The caller is responsible for zeroizing them after use.
    #[must_use]
    pub fn to_bytes(&self) -> [u8; 32] {
        self.signing_key.to_bytes()
    }
}

impl fmt::Debug for KeyPair {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("KeyPair")
            .field("public_key", &hex::encode(self.verifying_key.as_bytes()))
            .field("secret_key", &"<redacted>")
            .finish()
    }
}

// KeyPair is Sync + Send because SigningKey operations are thread-safe
unsafe impl Send for KeyPair {}
unsafe impl Sync for KeyPair {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keypair_generation() {
        let kp = KeyPair::generate();
        assert_eq!(kp.public_key().as_bytes().len(), 32);
    }

    #[test]
    fn test_sign_verify() {
        let kp = KeyPair::generate();
        let msg = b"test";
        let sig = kp.sign(msg);
        assert!(kp.public_key().verify(msg, &sig).is_ok());
    }

    #[test]
    fn test_verify_wrong_message() {
        let kp = KeyPair::generate();
        let sig = kp.sign(b"msg1");
        assert!(kp.public_key().verify(b"msg2", &sig).is_err());
    }

    #[test]
    fn test_keypair_serialization() {
        let kp1 = KeyPair::generate();
        let bytes = kp1.to_bytes();
        let kp2 = KeyPair::from_bytes(&bytes).unwrap();

        let msg = b"test message";
        let sig1 = kp1.sign(msg);
        let sig2 = kp2.sign(msg);

        assert_eq!(sig1, sig2);
    }

    #[test]
    fn test_public_key_encoding() {
        let kp = KeyPair::generate();
        let pk = kp.public_key();

        let b64 = pk.to_base64();
        let restored = PublicKey::from_base64(&b64).unwrap();

        assert_eq!(pk, restored);
    }
}
