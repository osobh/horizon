use crate::error::{Result, VaultError};
use argon2::{
    password_hash::PasswordHasher,
    password_hash::SaltString,
    Argon2,
};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use rand::rngs::OsRng;
use zeroize::{Zeroize, ZeroizeOnDrop};

const KEY_SIZE: usize = 32; // 256 bits for AES-256

/// Master encryption key (never persisted to disk)
///
/// This key is used to encrypt/decrypt all credential data.
/// It must be derived from a passphrase or loaded from a secure source.
#[derive(Zeroize, ZeroizeOnDrop)]
pub struct MasterKey {
    key: [u8; KEY_SIZE],
}

impl MasterKey {
    /// Derive a master key from a passphrase using Argon2
    ///
    /// # Arguments
    /// * `passphrase` - The passphrase to derive the key from
    /// * `salt` - Salt for key derivation (must be at least 16 bytes)
    ///
    /// # Security
    /// The salt should be unique and randomly generated for each master key.
    /// Store the salt alongside encrypted data (it's not secret).
    pub fn from_passphrase(passphrase: &str, salt: &[u8]) -> Result<Self> {
        if salt.len() < 16 {
            return Err(VaultError::KeyDerivationFailed(
                "Salt must be at least 16 bytes".to_string(),
            ));
        }

        let argon2 = Argon2::default();

        // Create a salt string from the provided bytes
        let salt_str = SaltString::encode_b64(salt)
            .map_err(|e| VaultError::KeyDerivationFailed(e.to_string()))?;

        // Hash the passphrase
        let password_hash = argon2
            .hash_password(passphrase.as_bytes(), &salt_str)
            .map_err(|e| VaultError::KeyDerivationFailed(e.to_string()))?;

        // Extract the hash bytes
        let hash_bytes = password_hash
            .hash
            .ok_or_else(|| VaultError::KeyDerivationFailed("No hash produced".to_string()))?;

        let hash_slice = hash_bytes.as_bytes();

        if hash_slice.len() < KEY_SIZE {
            return Err(VaultError::KeyDerivationFailed(format!(
                "Hash too short: {} bytes",
                hash_slice.len()
            )));
        }

        let mut key = [0u8; KEY_SIZE];
        key.copy_from_slice(&hash_slice[..KEY_SIZE]);

        Ok(Self { key })
    }

    /// Load master key from environment variable
    ///
    /// Expects `VAULT_MASTER_KEY` environment variable containing
    /// a base64-encoded 32-byte key.
    pub fn from_env() -> Result<Self> {
        let key_base64 = std::env::var("VAULT_MASTER_KEY")
            .map_err(|_| VaultError::InvalidMasterKeyFormat)?;

        Self::from_base64(&key_base64)
    }

    /// Load master key from a base64-encoded string
    pub fn from_base64(encoded: &str) -> Result<Self> {
        let key_bytes = STANDARD.decode(encoded)?;

        if key_bytes.len() != KEY_SIZE {
            return Err(VaultError::InvalidKeyLength {
                expected: KEY_SIZE,
                actual: key_bytes.len(),
            });
        }

        let mut key = [0u8; KEY_SIZE];
        key.copy_from_slice(&key_bytes);

        Ok(Self { key })
    }

    /// Create a master key from raw bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() != KEY_SIZE {
            return Err(VaultError::InvalidKeyLength {
                expected: KEY_SIZE,
                actual: bytes.len(),
            });
        }

        let mut key = [0u8; KEY_SIZE];
        key.copy_from_slice(bytes);

        Ok(Self { key })
    }

    /// Generate a random master key
    ///
    /// # Security
    /// Uses cryptographically secure random number generator.
    /// The generated key should be stored securely.
    pub fn generate() -> Self {
        let mut key = [0u8; KEY_SIZE];
        rand::Rng::fill(&mut OsRng, &mut key);
        Self { key }
    }

    /// Export the key as base64 (for secure storage)
    ///
    /// # Security
    /// The returned string contains the secret key.
    /// It should be stored securely (e.g., environment variable, secrets manager).
    pub fn to_base64(&self) -> String {
        STANDARD.encode(self.key)
    }

    /// Get the raw key bytes (use carefully!)
    pub(crate) fn as_bytes(&self) -> &[u8; KEY_SIZE] {
        &self.key
    }
}

impl std::fmt::Debug for MasterKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MasterKey([REDACTED])")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_passphrase() {
        let salt = b"sixteen_byte_salt!!!!";
        let key = MasterKey::from_passphrase("test-password", salt).unwrap();

        // Verify key is correct size
        assert_eq!(key.as_bytes().len(), KEY_SIZE);
    }

    #[test]
    fn test_from_passphrase_deterministic() {
        let salt = b"sixteen_byte_salt!!!!";
        let key1 = MasterKey::from_passphrase("test-password", salt).unwrap();
        let key2 = MasterKey::from_passphrase("test-password", salt).unwrap();

        // Same passphrase and salt should produce same key
        assert_eq!(key1.as_bytes(), key2.as_bytes());
    }

    #[test]
    fn test_from_passphrase_different_salts() {
        let salt1 = b"sixteen_byte_salt!!!!";
        let salt2 = b"different_salt!!!!!!";
        let key1 = MasterKey::from_passphrase("test-password", salt1).unwrap();
        let key2 = MasterKey::from_passphrase("test-password", salt2).unwrap();

        // Different salts should produce different keys
        assert_ne!(key1.as_bytes(), key2.as_bytes());
    }

    #[test]
    fn test_from_passphrase_salt_too_short() {
        let salt = b"short"; // Less than 16 bytes
        let result = MasterKey::from_passphrase("test-password", salt);

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), VaultError::KeyDerivationFailed(_)));
    }

    #[test]
    fn test_from_base64() {
        // Generate a key and encode it
        let key1 = MasterKey::generate();
        let encoded = key1.to_base64();

        // Decode it back
        let key2 = MasterKey::from_base64(&encoded).unwrap();

        assert_eq!(key1.as_bytes(), key2.as_bytes());
    }

    #[test]
    fn test_from_base64_invalid_length() {
        let encoded = STANDARD.encode(b"too short");
        let result = MasterKey::from_base64(&encoded);

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), VaultError::InvalidKeyLength { .. }));
    }

    #[test]
    fn test_from_base64_invalid_encoding() {
        let result = MasterKey::from_base64("not-valid-base64!!!");

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), VaultError::Base64Error(_)));
    }

    #[test]
    fn test_from_bytes() {
        let bytes = [42u8; KEY_SIZE];
        let key = MasterKey::from_bytes(&bytes).unwrap();

        assert_eq!(key.as_bytes(), &bytes);
    }

    #[test]
    fn test_from_bytes_wrong_length() {
        let bytes = [42u8; 16]; // Wrong length
        let result = MasterKey::from_bytes(&bytes);

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), VaultError::InvalidKeyLength { .. }));
    }

    #[test]
    fn test_generate() {
        let key1 = MasterKey::generate();
        let key2 = MasterKey::generate();

        // Generated keys should be different
        assert_ne!(key1.as_bytes(), key2.as_bytes());

        // Generated keys should be correct size
        assert_eq!(key1.as_bytes().len(), KEY_SIZE);
    }

    #[test]
    fn test_to_base64() {
        let key = MasterKey::generate();
        let encoded = key.to_base64();

        // Should be valid base64
        assert!(STANDARD.decode(&encoded).is_ok());

        // Should decode to correct length
        let decoded = STANDARD.decode(&encoded).unwrap();
        assert_eq!(decoded.len(), KEY_SIZE);
    }

    #[test]
    fn test_debug_does_not_expose_key() {
        let key = MasterKey::generate();
        let debug_str = format!("{:?}", key);

        assert!(debug_str.contains("REDACTED"));
        assert!(!debug_str.contains(&key.to_base64()));
    }

    #[test]
    fn test_key_size_constant() {
        assert_eq!(KEY_SIZE, 32); // 256 bits
    }
}
