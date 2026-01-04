//! WireGuard key generation and management
//!
//! Provides utilities for generating and managing WireGuard key pairs
//! using Curve25519 cryptography via the x25519-dalek library.

use crate::{Error, Result};
use rand::RngCore;
use rand_core::OsRng;
use serde::{Deserialize, Serialize};
use std::fmt;
use x25519_dalek::{PublicKey, StaticSecret};

/// WireGuard public key (32 bytes, base64 encoded for storage)
pub const KEY_LENGTH: usize = 32;

/// A WireGuard key pair (private + public)
#[derive(Clone)]
pub struct KeyPair {
    /// Private key (32 bytes)
    private_key: [u8; KEY_LENGTH],
    /// Public key (32 bytes)
    public_key: [u8; KEY_LENGTH],
}

impl KeyPair {
    /// Generate a new random key pair
    ///
    /// Uses the system's cryptographically secure random number generator
    /// to generate a Curve25519 key pair via x25519-dalek.
    pub fn generate() -> Self {
        // Generate a cryptographically secure random private key
        let secret = StaticSecret::random_from_rng(OsRng);
        let public = PublicKey::from(&secret);

        Self {
            private_key: secret.to_bytes(),
            public_key: public.to_bytes(),
        }
    }

    /// Create a key pair from an existing private key
    ///
    /// Derives the public key from the private key using proper
    /// Curve25519 scalar multiplication via x25519-dalek.
    pub fn from_private_key(private_key: [u8; KEY_LENGTH]) -> Self {
        let secret = StaticSecret::from(private_key);
        let public = PublicKey::from(&secret);
        Self {
            private_key,
            public_key: public.to_bytes(),
        }
    }

    /// Create a key pair from base64-encoded private key
    pub fn from_private_key_base64(private_key_b64: &str) -> Result<Self> {
        let bytes = base64_decode(private_key_b64)?;
        if bytes.len() != KEY_LENGTH {
            return Err(Error::WireGuardConfig(format!(
                "Invalid private key length: expected {}, got {}",
                KEY_LENGTH,
                bytes.len()
            )));
        }

        let mut private_key = [0u8; KEY_LENGTH];
        private_key.copy_from_slice(&bytes);
        Ok(Self::from_private_key(private_key))
    }

    /// Get the private key bytes
    pub fn private_key(&self) -> &[u8; KEY_LENGTH] {
        &self.private_key
    }

    /// Get the public key bytes
    pub fn public_key(&self) -> &[u8; KEY_LENGTH] {
        &self.public_key
    }

    /// Get the private key as base64
    pub fn private_key_base64(&self) -> String {
        base64_encode(&self.private_key)
    }

    /// Get the public key as base64
    pub fn public_key_base64(&self) -> String {
        base64_encode(&self.public_key)
    }
}

impl fmt::Debug for KeyPair {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("KeyPair")
            .field("public_key", &self.public_key_base64())
            .field("private_key", &"[REDACTED]")
            .finish()
    }
}

/// WireGuard key management utilities
pub struct WireGuardKeys;

impl WireGuardKeys {
    /// Generate a new key pair
    pub fn generate() -> KeyPair {
        KeyPair::generate()
    }

    /// Generate a preshared key for additional security
    pub fn generate_preshared_key() -> String {
        let mut key = [0u8; KEY_LENGTH];
        rand::thread_rng().fill_bytes(&mut key);
        base64_encode(&key)
    }

    /// Validate a base64-encoded public key
    pub fn validate_public_key(key: &str) -> Result<()> {
        let bytes = base64_decode(key)?;
        if bytes.len() != KEY_LENGTH {
            return Err(Error::WireGuardConfig(format!(
                "Invalid public key length: expected {}, got {}",
                KEY_LENGTH,
                bytes.len()
            )));
        }
        Ok(())
    }

    /// Parse a public key from base64 to bytes
    pub fn parse_public_key(key: &str) -> Result<[u8; KEY_LENGTH]> {
        let bytes = base64_decode(key)?;
        if bytes.len() != KEY_LENGTH {
            return Err(Error::WireGuardConfig(format!(
                "Invalid public key length: expected {}, got {}",
                KEY_LENGTH,
                bytes.len()
            )));
        }
        let mut result = [0u8; KEY_LENGTH];
        result.copy_from_slice(&bytes);
        Ok(result)
    }

    /// Format a public key for display (first 8 chars)
    pub fn format_key_short(key: &str) -> String {
        if key.len() >= 8 {
            format!("{}...", &key[..8])
        } else {
            key.to_string()
        }
    }
}

/// Encode bytes to base64
fn base64_encode(bytes: &[u8]) -> String {
    use std::io::Write;
    let mut buf = Vec::with_capacity(bytes.len() * 4 / 3 + 4);
    {
        let mut encoder = Base64Encoder::new(&mut buf);
        encoder.write_all(bytes).unwrap();
        encoder.finish().unwrap();
    }
    String::from_utf8(buf).unwrap()
}

/// Decode base64 to bytes
fn base64_decode(s: &str) -> Result<Vec<u8>> {
    let s = s.trim();
    if s.is_empty() {
        return Err(Error::WireGuardConfig("Empty base64 string".to_string()));
    }

    let mut result = Vec::with_capacity(s.len() * 3 / 4);
    let mut buffer = 0u32;
    let mut bits = 0u32;

    for c in s.bytes() {
        let value = match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            b'=' => continue,                         // Padding
            b' ' | b'\n' | b'\r' | b'\t' => continue, // Whitespace
            _ => {
                return Err(Error::WireGuardConfig(format!(
                    "Invalid base64 character: {}",
                    c as char
                )))
            }
        };

        buffer = (buffer << 6) | value as u32;
        bits += 6;

        if bits >= 8 {
            bits -= 8;
            result.push((buffer >> bits) as u8);
            buffer &= (1 << bits) - 1;
        }
    }

    Ok(result)
}

/// Simple base64 encoder
struct Base64Encoder<'a, W: std::io::Write> {
    writer: &'a mut W,
    buffer: u32,
    bits: u32,
}

impl<'a, W: std::io::Write> Base64Encoder<'a, W> {
    fn new(writer: &'a mut W) -> Self {
        Self {
            writer,
            buffer: 0,
            bits: 0,
        }
    }

    fn finish(mut self) -> std::io::Result<()> {
        if self.bits > 0 {
            self.buffer <<= 6 - self.bits;
            self.writer
                .write_all(&[Self::encode_char(self.buffer as u8)])?;
            let padding = (6 - self.bits) / 2;
            for _ in 0..padding {
                self.writer.write_all(b"=")?;
            }
        }
        Ok(())
    }

    fn encode_char(value: u8) -> u8 {
        const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        ALPHABET[(value & 0x3F) as usize]
    }
}

impl<'a, W: std::io::Write> std::io::Write for Base64Encoder<'a, W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        for &byte in buf {
            self.buffer = (self.buffer << 8) | byte as u32;
            self.bits += 8;

            while self.bits >= 6 {
                self.bits -= 6;
                let value = (self.buffer >> self.bits) as u8;
                self.writer.write_all(&[Self::encode_char(value)])?;
                self.buffer &= (1 << self.bits) - 1;
            }
        }
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.writer.flush()
    }
}

/// Serializable key pair for storage
#[derive(Clone, Serialize, Deserialize)]
pub struct StoredKeyPair {
    /// Private key (base64)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub private_key: Option<String>,
    /// Public key (base64)
    pub public_key: String,
}

impl From<&KeyPair> for StoredKeyPair {
    fn from(kp: &KeyPair) -> Self {
        Self {
            private_key: Some(kp.private_key_base64()),
            public_key: kp.public_key_base64(),
        }
    }
}

impl StoredKeyPair {
    /// Create from public key only (for peer storage)
    pub fn public_only(public_key: String) -> Self {
        Self {
            private_key: None,
            public_key,
        }
    }

    /// Convert to KeyPair (requires private key)
    pub fn to_key_pair(&self) -> Result<KeyPair> {
        let private_key = self
            .private_key
            .as_ref()
            .ok_or_else(|| Error::WireGuardConfig("Missing private key".to_string()))?;
        KeyPair::from_private_key_base64(private_key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_key_generation() {
        let kp = KeyPair::generate();

        // Keys should be 32 bytes
        assert_eq!(kp.private_key().len(), KEY_LENGTH);
        assert_eq!(kp.public_key().len(), KEY_LENGTH);

        // Keys should be different
        assert_ne!(kp.private_key(), kp.public_key());

        // Base64 should be valid
        let pub_b64 = kp.public_key_base64();
        let priv_b64 = kp.private_key_base64();
        assert!(!pub_b64.is_empty());
        assert!(!priv_b64.is_empty());
    }

    #[test]
    fn test_key_from_private() {
        let kp1 = KeyPair::generate();
        let kp2 = KeyPair::from_private_key(*kp1.private_key());

        // Public keys should match
        assert_eq!(kp1.public_key(), kp2.public_key());
    }

    #[test]
    fn test_base64_roundtrip() {
        let original = [1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let encoded = base64_encode(&original);
        let decoded = base64_decode(&encoded).unwrap();
        assert_eq!(&original[..], &decoded[..]);
    }

    #[test]
    fn test_validate_public_key() {
        let kp = KeyPair::generate();
        assert!(WireGuardKeys::validate_public_key(&kp.public_key_base64()).is_ok());

        // Invalid key
        assert!(WireGuardKeys::validate_public_key("invalid").is_err());
        assert!(WireGuardKeys::validate_public_key("").is_err());
    }

    #[test]
    fn test_preshared_key() {
        let psk1 = WireGuardKeys::generate_preshared_key();
        let psk2 = WireGuardKeys::generate_preshared_key();

        // Should be unique
        assert_ne!(psk1, psk2);

        // Should be valid base64
        assert!(base64_decode(&psk1).is_ok());
    }

    #[test]
    fn test_stored_key_pair() {
        let kp = KeyPair::generate();
        let stored: StoredKeyPair = (&kp).into();

        assert!(stored.private_key.is_some());
        assert_eq!(stored.public_key, kp.public_key_base64());

        // Round trip
        let restored = stored.to_key_pair().unwrap();
        assert_eq!(restored.public_key(), kp.public_key());
    }

    #[test]
    fn test_format_key_short() {
        let kp = KeyPair::generate();
        let short = WireGuardKeys::format_key_short(&kp.public_key_base64());
        assert!(short.ends_with("..."));
        assert!(short.len() < kp.public_key_base64().len());
    }
}
