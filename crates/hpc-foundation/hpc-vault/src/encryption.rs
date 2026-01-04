use crate::error::{Result, VaultError};
use crate::key::MasterKey;
use crate::secret::SecretString;
use aes_gcm::{
    aead::{Aead, KeyInit, Payload},
    Aes256Gcm, Nonce,
};
use rand::RngCore;
use serde::{Deserialize, Serialize};

const NONCE_SIZE: usize = 12; // 96 bits for AES-GCM

/// Encrypted credential envelope
///
/// Contains ciphertext, nonce, and encryption version for versioning support.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptedCredential {
    /// The encrypted data
    pub ciphertext: Vec<u8>,

    /// The nonce used for encryption (12 bytes for AES-GCM)
    pub nonce: Vec<u8>,

    /// Encryption version (for key rotation support)
    pub version: u32,

    /// Optional additional authenticated data
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aad: Option<Vec<u8>>,
}

/// Vault encryption engine
///
/// Provides AES-256-GCM encryption and decryption using a master key.
pub struct VaultEncryption {
    master_key: MasterKey,
    version: u32,
}

impl VaultEncryption {
    /// Create a new vault encryption engine
    pub fn new(master_key: MasterKey) -> Self {
        Self {
            master_key,
            version: 1,
        }
    }

    /// Create with a specific version (for key rotation)
    pub fn with_version(master_key: MasterKey, version: u32) -> Self {
        Self {
            master_key,
            version,
        }
    }

    /// Encrypt plaintext data
    ///
    /// # Arguments
    /// * `plaintext` - The data to encrypt
    ///
    /// # Returns
    /// An `EncryptedCredential` containing the ciphertext and nonce
    pub fn encrypt(&self, plaintext: &[u8]) -> Result<EncryptedCredential> {
        self.encrypt_with_aad(plaintext, None)
    }

    /// Encrypt plaintext data with additional authenticated data (AAD)
    ///
    /// # Arguments
    /// * `plaintext` - The data to encrypt
    /// * `aad` - Additional authenticated data (not encrypted, but authenticated)
    ///
    /// # Returns
    /// An `EncryptedCredential` containing the ciphertext, nonce, and AAD
    pub fn encrypt_with_aad(
        &self,
        plaintext: &[u8],
        aad: Option<&[u8]>,
    ) -> Result<EncryptedCredential> {
        // Create cipher
        let cipher = Aes256Gcm::new(self.master_key.as_bytes().into());

        // Generate random nonce
        let mut nonce_bytes = [0u8; NONCE_SIZE];
        rand::thread_rng().fill_bytes(&mut nonce_bytes);
        let nonce = Nonce::from_slice(&nonce_bytes);

        // Prepare payload
        let payload = if let Some(aad_data) = aad {
            Payload {
                msg: plaintext,
                aad: aad_data,
            }
        } else {
            Payload {
                msg: plaintext,
                aad: b"",
            }
        };

        // Encrypt
        let ciphertext = cipher
            .encrypt(nonce, payload)
            .map_err(|e| VaultError::EncryptionFailed(e.to_string()))?;

        Ok(EncryptedCredential {
            ciphertext,
            nonce: nonce_bytes.to_vec(),
            version: self.version,
            aad: aad.map(|a| a.to_vec()),
        })
    }

    /// Decrypt encrypted credential
    ///
    /// # Arguments
    /// * `encrypted` - The encrypted credential envelope
    ///
    /// # Returns
    /// A `SecretString` containing the decrypted data
    pub fn decrypt(&self, encrypted: &EncryptedCredential) -> Result<SecretString> {
        // Validate nonce size
        if encrypted.nonce.len() != NONCE_SIZE {
            return Err(VaultError::InvalidNonceLength {
                expected: NONCE_SIZE,
                actual: encrypted.nonce.len(),
            });
        }

        // Create cipher
        let cipher = Aes256Gcm::new(self.master_key.as_bytes().into());

        // Create nonce
        let nonce = Nonce::from_slice(&encrypted.nonce);

        // Prepare payload
        let payload = if let Some(ref aad_data) = encrypted.aad {
            Payload {
                msg: &encrypted.ciphertext,
                aad: aad_data,
            }
        } else {
            Payload {
                msg: &encrypted.ciphertext,
                aad: b"",
            }
        };

        // Decrypt
        let plaintext = cipher
            .decrypt(nonce, payload)
            .map_err(|e| VaultError::DecryptionFailed(e.to_string()))?;

        Ok(SecretString::new(plaintext))
    }

    /// Get the current encryption version
    pub fn version(&self) -> u32 {
        self.version
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_master_key() -> MasterKey {
        MasterKey::from_passphrase("test-password", b"test-salt-16byte").unwrap()
    }

    #[test]
    fn test_encrypt_decrypt_roundtrip() {
        let key = test_master_key();
        let vault = VaultEncryption::new(key);

        let plaintext = b"sensitive credential data";
        let encrypted = vault.encrypt(plaintext).unwrap();
        let decrypted = vault.decrypt(&encrypted).unwrap();

        assert_eq!(plaintext, decrypted.expose_secret());
    }

    #[test]
    fn test_encrypt_produces_different_ciphertexts() {
        let key = test_master_key();
        let vault = VaultEncryption::new(key);

        let plaintext = b"test data";
        let encrypted1 = vault.encrypt(plaintext).unwrap();
        let encrypted2 = vault.encrypt(plaintext).unwrap();

        // Same plaintext should produce different ciphertexts (different nonces)
        assert_ne!(encrypted1.ciphertext, encrypted2.ciphertext);
        assert_ne!(encrypted1.nonce, encrypted2.nonce);
    }

    #[test]
    fn test_encrypt_with_aad() {
        let key = test_master_key();
        let vault = VaultEncryption::new(key);

        let plaintext = b"secret data";
        let aad = b"metadata";
        let encrypted = vault.encrypt_with_aad(plaintext, Some(aad)).unwrap();

        assert_eq!(encrypted.aad, Some(aad.to_vec()));

        let decrypted = vault.decrypt(&encrypted).unwrap();
        assert_eq!(plaintext, decrypted.expose_secret());
    }

    #[test]
    fn test_decrypt_with_wrong_aad_fails() {
        let key = test_master_key();
        let vault = VaultEncryption::new(key);

        let plaintext = b"secret data";
        let aad = b"metadata";
        let mut encrypted = vault.encrypt_with_aad(plaintext, Some(aad)).unwrap();

        // Tamper with AAD
        encrypted.aad = Some(b"wrong metadata".to_vec());

        // Decryption should fail
        let result = vault.decrypt(&encrypted);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            VaultError::DecryptionFailed(_)
        ));
    }

    #[test]
    fn test_decrypt_with_wrong_key_fails() {
        let key1 = MasterKey::from_passphrase("password1", b"salt1-16byte-min").unwrap();
        let key2 = MasterKey::from_passphrase("password2", b"salt2-16byte-min").unwrap();

        let vault1 = VaultEncryption::new(key1);
        let vault2 = VaultEncryption::new(key2);

        let plaintext = b"secret data";
        let encrypted = vault1.encrypt(plaintext).unwrap();

        // Decryption with wrong key should fail
        let result = vault2.decrypt(&encrypted);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            VaultError::DecryptionFailed(_)
        ));
    }

    #[test]
    fn test_decrypt_with_tampered_ciphertext_fails() {
        let key = test_master_key();
        let vault = VaultEncryption::new(key);

        let plaintext = b"secret data";
        let mut encrypted = vault.encrypt(plaintext).unwrap();

        // Tamper with ciphertext
        if !encrypted.ciphertext.is_empty() {
            encrypted.ciphertext[0] ^= 0xFF;
        }

        // Decryption should fail
        let result = vault.decrypt(&encrypted);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            VaultError::DecryptionFailed(_)
        ));
    }

    #[test]
    fn test_decrypt_with_invalid_nonce_length() {
        let key = test_master_key();
        let vault = VaultEncryption::new(key);

        let plaintext = b"secret data";
        let mut encrypted = vault.encrypt(plaintext).unwrap();

        // Invalid nonce length
        encrypted.nonce = vec![0u8; 16]; // Wrong size

        let result = vault.decrypt(&encrypted);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            VaultError::InvalidNonceLength { .. }
        ));
    }

    #[test]
    fn test_encrypted_credential_version() {
        let key = test_master_key();
        let vault = VaultEncryption::new(key);

        let plaintext = b"test";
        let encrypted = vault.encrypt(plaintext).unwrap();

        assert_eq!(encrypted.version, 1);
    }

    #[test]
    fn test_vault_with_custom_version() {
        let key = test_master_key();
        let vault = VaultEncryption::with_version(key, 42);

        assert_eq!(vault.version(), 42);

        let plaintext = b"test";
        let encrypted = vault.encrypt(plaintext).unwrap();

        assert_eq!(encrypted.version, 42);
    }

    #[test]
    fn test_nonce_size_constant() {
        assert_eq!(NONCE_SIZE, 12); // 96 bits for AES-GCM
    }

    #[test]
    fn test_encrypt_empty_data() {
        let key = test_master_key();
        let vault = VaultEncryption::new(key);

        let plaintext = b"";
        let encrypted = vault.encrypt(plaintext).unwrap();
        let decrypted = vault.decrypt(&encrypted).unwrap();

        assert_eq!(plaintext, decrypted.expose_secret());
    }

    #[test]
    fn test_encrypt_large_data() {
        let key = test_master_key();
        let vault = VaultEncryption::new(key);

        // 1 MB of data
        let plaintext = vec![42u8; 1024 * 1024];
        let encrypted = vault.encrypt(&plaintext).unwrap();
        let decrypted = vault.decrypt(&encrypted).unwrap();

        assert_eq!(plaintext.as_slice(), decrypted.expose_secret());
    }

    #[test]
    fn test_encrypted_credential_serialization() {
        let encrypted = EncryptedCredential {
            ciphertext: vec![1, 2, 3, 4],
            nonce: vec![5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            version: 1,
            aad: Some(vec![17, 18, 19]),
        };

        let json = serde_json::to_string(&encrypted).unwrap();
        let deserialized: EncryptedCredential = serde_json::from_str(&json).unwrap();

        assert_eq!(encrypted.ciphertext, deserialized.ciphertext);
        assert_eq!(encrypted.nonce, deserialized.nonce);
        assert_eq!(encrypted.version, deserialized.version);
        assert_eq!(encrypted.aad, deserialized.aad);
    }
}
