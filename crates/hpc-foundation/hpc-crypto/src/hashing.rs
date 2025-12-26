//! Blake3 cryptographic hashing
//!
//! This module provides fast, secure hashing using the Blake3 algorithm.
//! Blake3 is faster than SHA-256 and provides 256-bit security.
//!
//! # Features
//!
//! - Regular hashing
//! - Keyed hashing (HMAC-like)
//! - Streaming API for large data
//! - Serialization to hex and base64
//!
//! # Example
//!
//! ```
//! use hpc_crypto::hashing::hash;
//!
//! let data = b"Hello, HPC!";
//! let hash_value = hash(data);
//! println!("Hash: {}", hash_value.to_hex());
//! ```

use serde::{Deserialize, Serialize};
use std::fmt;
use thiserror::Error;

/// Errors that can occur during hashing operations
#[derive(Error, Debug)]
pub enum HashError {
    #[error("Invalid hash encoding: {0}")]
    InvalidEncoding(String),

    #[error("Invalid hash length: expected 32 bytes, got {0}")]
    InvalidLength(usize),

    #[error("Invalid key length: expected 32 bytes, got {0}")]
    InvalidKeyLength(usize),
}

/// A 256-bit hash value (32 bytes)
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HashValue {
    bytes: [u8; 32],
}

impl HashValue {
    /// Create a hash value from bytes
    ///
    /// # Errors
    ///
    /// Returns `HashError::InvalidLength` if not exactly 32 bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, HashError> {
        if bytes.len() != 32 {
            return Err(HashError::InvalidLength(bytes.len()));
        }

        let mut array = [0u8; 32];
        array.copy_from_slice(bytes);

        Ok(HashValue { bytes: array })
    }

    /// Create a hash value from hex encoding
    pub fn from_hex(s: &str) -> Result<Self, HashError> {
        let bytes = hex::decode(s).map_err(|e| HashError::InvalidEncoding(e.to_string()))?;
        Self::from_bytes(&bytes)
    }

    /// Create a hash value from base64 encoding
    pub fn from_base64(s: &str) -> Result<Self, HashError> {
        use base64::Engine;
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(s)
            .map_err(|e| HashError::InvalidEncoding(e.to_string()))?;
        Self::from_bytes(&bytes)
    }

    /// Get the raw bytes of the hash
    #[must_use]
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.bytes
    }

    /// Encode hash as hexadecimal string
    #[must_use]
    pub fn to_hex(&self) -> String {
        hex::encode(self.bytes)
    }

    /// Encode hash as base64 string
    #[must_use]
    pub fn to_base64(&self) -> String {
        use base64::Engine;
        base64::engine::general_purpose::STANDARD.encode(self.bytes)
    }
}

impl fmt::Debug for HashValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HashValue")
            .field("hex", &self.to_hex())
            .finish()
    }
}

impl fmt::Display for HashValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_hex())
    }
}

impl AsRef<[u8]> for HashValue {
    fn as_ref(&self) -> &[u8] {
        &self.bytes
    }
}

/// Hash data using Blake3
///
/// # Returns
///
/// A 32-byte hash value
#[must_use]
pub fn hash(data: &[u8]) -> HashValue {
    let hash_bytes = blake3::hash(data);
    HashValue {
        bytes: *hash_bytes.as_bytes(),
    }
}

/// Hash data with a key (keyed hashing)
///
/// This provides HMAC-like properties. The same key must be used
/// to verify the hash.
#[must_use]
pub fn hash_with_key(data: &[u8], key: &[u8]) -> HashValue {
    let mut key_array = [0u8; 32];
    let copy_len = key.len().min(32);
    key_array[..copy_len].copy_from_slice(&key[..copy_len]);

    let hash_bytes = blake3::keyed_hash(&key_array, data);
    HashValue {
        bytes: *hash_bytes.as_bytes(),
    }
}

/// Streaming hasher for large data
///
/// This allows you to hash data in chunks, which is useful for
/// large files or streaming data.
pub struct StreamHasher {
    hasher: blake3::Hasher,
}

impl StreamHasher {
    /// Create a new stream hasher
    #[must_use]
    pub fn new() -> Self {
        StreamHasher {
            hasher: blake3::Hasher::new(),
        }
    }

    /// Create a new keyed stream hasher
    #[must_use]
    pub fn new_keyed(key: &[u8]) -> Self {
        let mut key_array = [0u8; 32];
        let copy_len = key.len().min(32);
        key_array[..copy_len].copy_from_slice(&key[..copy_len]);

        StreamHasher {
            hasher: blake3::Hasher::new_keyed(&key_array),
        }
    }

    /// Update the hasher with more data
    pub fn update(&mut self, data: &[u8]) {
        self.hasher.update(data);
    }

    /// Finalize the hash and return the result
    ///
    /// This consumes the hasher.
    #[must_use]
    pub fn finalize(self) -> HashValue {
        let hash_bytes = self.hasher.finalize();
        HashValue {
            bytes: *hash_bytes.as_bytes(),
        }
    }

    /// Get the current hash without consuming the hasher
    pub fn finalize_reset(&mut self) -> HashValue {
        let mut hash_bytes = self.hasher.finalize_xof();
        let mut bytes = [0u8; 32];
        hash_bytes.fill(&mut bytes);

        // Reset for continued use
        self.hasher.reset();

        HashValue { bytes }
    }
}

impl Default for StreamHasher {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for StreamHasher {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StreamHasher").finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_deterministic() {
        let data = b"test";
        let h1 = hash(data);
        let h2 = hash(data);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_hash_different_inputs() {
        let h1 = hash(b"data1");
        let h2 = hash(b"data2");
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_keyed_hash() {
        let data = b"test";
        let key = b"0123456789abcdef0123456789abcdef";

        let h1 = hash(data);
        let h2 = hash_with_key(data, key);

        assert_ne!(h1, h2);
    }

    #[test]
    fn test_stream_hasher() {
        let mut hasher = StreamHasher::new();
        hasher.update(b"Hello, ");
        hasher.update(b"World!");
        let h1 = hasher.finalize();

        let h2 = hash(b"Hello, World!");

        assert_eq!(h1, h2);
    }

    #[test]
    fn test_hash_value_hex() {
        let data = b"test";
        let hash_val = hash(data);
        let hex = hash_val.to_hex();
        let restored = HashValue::from_hex(&hex).unwrap();
        assert_eq!(hash_val, restored);
    }

    #[test]
    fn test_hash_value_base64() {
        let data = b"test";
        let hash_val = hash(data);
        let b64 = hash_val.to_base64();
        let restored = HashValue::from_base64(&b64).unwrap();
        assert_eq!(hash_val, restored);
    }
}
