use thiserror::Error;

/// Errors that can occur during vault operations
#[derive(Error, Debug)]
pub enum VaultError {
    #[error("Invalid key length: expected {expected}, got {actual}")]
    InvalidKeyLength { expected: usize, actual: usize },

    #[error("Invalid master key format")]
    InvalidMasterKeyFormat,

    #[error("Encryption failed: {0}")]
    EncryptionFailed(String),

    #[error("Decryption failed: {0}")]
    DecryptionFailed(String),

    #[error("Invalid nonce length: expected {expected}, got {actual}")]
    InvalidNonceLength { expected: usize, actual: usize },

    #[error("Key derivation failed: {0}")]
    KeyDerivationFailed(String),

    #[error("Base64 decode error: {0}")]
    Base64Error(#[from] base64::DecodeError),

    #[error("Password hash error: {0}")]
    PasswordHashError(String),
}

/// Result type for vault operations
pub type Result<T> = std::result::Result<T, VaultError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = VaultError::InvalidKeyLength {
            expected: 32,
            actual: 16,
        };
        assert_eq!(
            err.to_string(),
            "Invalid key length: expected 32, got 16"
        );
    }

    #[test]
    fn test_error_from_base64() {
        use base64::{engine::general_purpose::STANDARD, Engine as _};
        let result: Result<Vec<u8>> = STANDARD.decode("invalid!!!")
            .map_err(VaultError::from);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), VaultError::Base64Error(_)));
    }

    #[test]
    fn test_encryption_failed_error() {
        let err = VaultError::EncryptionFailed("test error".to_string());
        assert_eq!(err.to_string(), "Encryption failed: test error");
    }

    #[test]
    fn test_decryption_failed_error() {
        let err = VaultError::DecryptionFailed("bad key".to_string());
        assert_eq!(err.to_string(), "Decryption failed: bad key");
    }
}
