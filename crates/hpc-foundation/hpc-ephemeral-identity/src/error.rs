//! Error types for ephemeral identity operations.

use thiserror::Error;
use uuid::Uuid;

/// Result type for ephemeral identity operations.
pub type Result<T> = std::result::Result<T, EphemeralError>;

/// Errors that can occur during ephemeral identity operations.
#[derive(Error, Debug)]
pub enum EphemeralError {
    /// Identity not found in the system.
    #[error("Ephemeral identity not found: {0}")]
    IdentityNotFound(Uuid),

    /// Identity has expired and can no longer be used.
    #[error("Ephemeral identity expired: {0}")]
    IdentityExpired(Uuid),

    /// Identity has been revoked.
    #[error("Ephemeral identity revoked: {0}")]
    IdentityRevoked(Uuid),

    /// Identity is in an invalid state for the requested operation.
    #[error("Invalid identity state: expected {expected}, found {found}")]
    InvalidState {
        /// Expected state for the operation.
        expected: String,
        /// Actual state found.
        found: String,
    },

    /// Token validation failed.
    #[error("Token validation failed: {0}")]
    TokenValidationFailed(String),

    /// Token has expired.
    #[error("Token expired")]
    TokenExpired,

    /// Invalid token signature.
    #[error("Invalid token signature")]
    InvalidSignature,

    /// Token decryption failed.
    #[error("Token decryption failed: {0}")]
    DecryptionFailed(String),

    /// Invitation not found.
    #[error("Invitation not found: {0}")]
    InvitationNotFound(Uuid),

    /// Invitation has expired.
    #[error("Invitation expired: {0}")]
    InvitationExpired(Uuid),

    /// Invitation has already been redeemed.
    #[error("Invitation already redeemed: {0}")]
    InvitationAlreadyRedeemed(Uuid),

    /// Invalid redemption code.
    #[error("Invalid redemption code")]
    InvalidRedemptionCode,

    /// Maximum redemption attempts exceeded.
    #[error("Maximum redemption attempts exceeded for invitation: {0}")]
    MaxRedemptionAttemptsExceeded(Uuid),

    /// Capability not allowed for this identity.
    #[error("Capability denied: {0}")]
    CapabilityDenied(String),

    /// Rate limit exceeded.
    #[error("Rate limit exceeded: {resource} (limit: {limit}/min, used: {used})")]
    RateLimitExceeded {
        /// Resource that was rate limited.
        resource: String,
        /// Maximum allowed per minute.
        limit: u32,
        /// Current usage count.
        used: u32,
    },

    /// Operation not allowed during current time window.
    #[error("Operation not allowed during current time window")]
    TimeWindowViolation,

    /// Sponsor identity not found or invalid.
    #[error("Sponsor identity not found: {0}")]
    SponsorNotFound(Uuid),

    /// Maximum ephemeral identities per sponsor exceeded.
    #[error("Maximum ephemeral identities per sponsor exceeded (limit: {0})")]
    MaxIdentitiesExceeded(usize),

    /// Device binding mismatch.
    #[error("Device binding mismatch: expected {expected}, found {found}")]
    DeviceBindingMismatch {
        /// Expected device fingerprint.
        expected: String,
        /// Actual device fingerprint.
        found: String,
    },

    /// Cryptographic operation failed.
    #[error("Cryptographic error: {0}")]
    CryptoError(String),

    /// Serialization or deserialization failed.
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// Internal service error.
    #[error("Internal error: {0}")]
    Internal(String),
}

impl From<serde_json::Error> for EphemeralError {
    fn from(err: serde_json::Error) -> Self {
        EphemeralError::SerializationError(err.to_string())
    }
}

impl From<base64::DecodeError> for EphemeralError {
    fn from(err: base64::DecodeError) -> Self {
        EphemeralError::DecryptionFailed(format!("Base64 decode error: {err}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let id = Uuid::new_v4();
        let err = EphemeralError::IdentityNotFound(id);
        assert!(err.to_string().contains(&id.to_string()));
    }

    #[test]
    fn test_rate_limit_error() {
        let err = EphemeralError::RateLimitExceeded {
            resource: "api_calls".to_string(),
            limit: 100,
            used: 150,
        };
        let msg = err.to_string();
        assert!(msg.contains("api_calls"));
        assert!(msg.contains("100"));
        assert!(msg.contains("150"));
    }

    #[test]
    fn test_device_binding_mismatch() {
        let err = EphemeralError::DeviceBindingMismatch {
            expected: "device-123".to_string(),
            found: "device-456".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("device-123"));
        assert!(msg.contains("device-456"));
    }

    #[test]
    fn test_from_serde_error() {
        let json_err = serde_json::from_str::<String>("invalid").unwrap_err();
        let err: EphemeralError = json_err.into();
        matches!(err, EphemeralError::SerializationError(_));
    }
}
