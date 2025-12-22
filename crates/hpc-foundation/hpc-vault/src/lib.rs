//! Horizon Vault - Secure credential encryption and storage
//!
//! This crate provides secure encryption primitives for storing sensitive
//! credentials like cloud provider API keys, database passwords, and SSH keys.
//!
//! # Features
//! - AES-256-GCM encryption
//! - Argon2 key derivation
//! - Secure memory zeroing (zeroize)
//! - Key rotation support
//!
//! # Example
//! ```rust
//! use hpc_vault::{MasterKey, VaultEncryption, SecretString};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Derive a master key from a passphrase
//! let master_key = MasterKey::from_passphrase(
//!     "my-secure-passphrase",
//!     b"unique-salt-1234",
//! )?;
//!
//! // Create encryption engine
//! let vault = VaultEncryption::new(master_key);
//!
//! // Encrypt sensitive data
//! let plaintext = b"aws-secret-key-12345";
//! let encrypted = vault.encrypt(plaintext)?;
//!
//! // Decrypt when needed
//! let decrypted = vault.decrypt(&encrypted)?;
//! assert_eq!(plaintext, decrypted.expose_secret());
//! # Ok(())
//! # }
//! ```

mod encryption;
mod error;
mod key;
mod secret;

// Re-export public API
pub use encryption::{EncryptedCredential, VaultEncryption};
pub use error::{Result, VaultError};
pub use key::MasterKey;
pub use secret::SecretString;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_full_workflow() {
        // Generate a new master key
        let master_key = MasterKey::generate();

        // Create vault
        let vault = VaultEncryption::new(master_key);

        // Encrypt some data
        let plaintext = b"sensitive-credential-data";
        let encrypted = vault.encrypt(plaintext).unwrap();

        // Decrypt it back
        let decrypted = vault.decrypt(&encrypted).unwrap();

        assert_eq!(plaintext, decrypted.expose_secret());
    }

    #[test]
    fn test_key_derivation_workflow() {
        // Derive key from passphrase
        let salt = b"application-salt-1234567890";
        let master_key = MasterKey::from_passphrase("user-passphrase", salt).unwrap();

        // Create vault
        let vault = VaultEncryption::new(master_key);

        // Use it
        let encrypted = vault.encrypt(b"test data").unwrap();
        let decrypted = vault.decrypt(&encrypted).unwrap();

        assert_eq!(b"test data", decrypted.expose_secret());
    }

    #[test]
    fn test_base64_workflow() {
        // Generate and export key
        let key1 = MasterKey::generate();
        let encoded = key1.to_base64();

        // Later, load it back
        let key2 = MasterKey::from_base64(&encoded).unwrap();

        // Should work the same
        let vault1 = VaultEncryption::new(key1);
        let vault2 = VaultEncryption::new(key2);

        let encrypted = vault1.encrypt(b"test").unwrap();
        let decrypted = vault2.decrypt(&encrypted).unwrap();

        assert_eq!(b"test", decrypted.expose_secret());
    }

    #[test]
    fn test_version_compatibility() {
        // Create two vaults with same key but different versions
        let salt = b"test-salt-1234567890123456";
        let key1 = MasterKey::from_passphrase("same-passphrase", salt).unwrap();
        let key2 = MasterKey::from_passphrase("same-passphrase", salt).unwrap();

        // Encrypt with version 1
        let vault_v1 = VaultEncryption::with_version(key1, 1);
        let encrypted_v1 = vault_v1.encrypt(b"data").unwrap();
        assert_eq!(encrypted_v1.version, 1);

        // Decrypt with version 2 vault (should still work with same key)
        let vault_v2 = VaultEncryption::with_version(key2, 2);
        let decrypted = vault_v2.decrypt(&encrypted_v1).unwrap();
        assert_eq!(b"data", decrypted.expose_secret());
    }

    #[test]
    fn test_secret_string_security() {
        let secret = SecretString::from_string("password123");

        // Debug output should not reveal secret
        let debug = format!("{:?}", secret);
        assert!(!debug.contains("password123"));
        assert!(debug.contains("REDACTED"));
    }

    #[test]
    fn test_master_key_security() {
        let key = MasterKey::generate();

        // Debug output should not reveal key
        let debug = format!("{:?}", key);
        assert!(!debug.contains(&key.to_base64()));
        assert!(debug.contains("REDACTED"));
    }
}
