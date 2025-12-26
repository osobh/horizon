use thiserror::Error;

pub type ProviderResult<T> = Result<T, ProviderError>;

#[derive(Error, Debug)]
pub enum ProviderError {
    #[error("Invalid request: {0}")]
    InvalidRequest(String),

    #[error("Provider unavailable: {0}")]
    Unavailable(String),

    #[error("Quota exceeded: {0}")]
    QuotaExceeded(String),

    #[error("Resource not found: {0}")]
    NotFound(String),

    #[error("Authentication failed: {0}")]
    AuthenticationFailed(String),

    #[error("Rate limit exceeded: {0}")]
    RateLimitExceeded(String),

    #[error("Provisioning failed: {0}")]
    ProvisioningFailed(String),

    #[error("Deprovisioning failed: {0}")]
    DeprovisioningFailed(String),

    #[error("Network error: {0}")]
    NetworkError(String),

    #[error("Internal error: {0}")]
    InternalError(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_invalid_request() {
        let err = ProviderError::InvalidRequest("test".to_string());
        assert_eq!(err.to_string(), "Invalid request: test");
    }

    #[test]
    fn test_error_unavailable() {
        let err = ProviderError::Unavailable("service down".to_string());
        assert_eq!(err.to_string(), "Provider unavailable: service down");
    }

    #[test]
    fn test_error_quota_exceeded() {
        let err = ProviderError::QuotaExceeded("GPU limit reached".to_string());
        assert_eq!(err.to_string(), "Quota exceeded: GPU limit reached");
    }

    #[test]
    fn test_error_not_found() {
        let err = ProviderError::NotFound("instance-123".to_string());
        assert_eq!(err.to_string(), "Resource not found: instance-123");
    }

    #[test]
    fn test_error_authentication_failed() {
        let err = ProviderError::AuthenticationFailed("invalid credentials".to_string());
        assert_eq!(
            err.to_string(),
            "Authentication failed: invalid credentials"
        );
    }

    #[test]
    fn test_error_rate_limit() {
        let err = ProviderError::RateLimitExceeded("too many requests".to_string());
        assert_eq!(err.to_string(), "Rate limit exceeded: too many requests");
    }

    #[test]
    fn test_error_provisioning_failed() {
        let err = ProviderError::ProvisioningFailed("capacity unavailable".to_string());
        assert_eq!(
            err.to_string(),
            "Provisioning failed: capacity unavailable"
        );
    }

    #[test]
    fn test_error_deprovisioning_failed() {
        let err = ProviderError::DeprovisioningFailed("instance stuck".to_string());
        assert_eq!(err.to_string(), "Deprovisioning failed: instance stuck");
    }

    #[test]
    fn test_error_network() {
        let err = ProviderError::NetworkError("connection timeout".to_string());
        assert_eq!(err.to_string(), "Network error: connection timeout");
    }

    #[test]
    fn test_error_internal() {
        let err = ProviderError::InternalError("unexpected state".to_string());
        assert_eq!(err.to_string(), "Internal error: unexpected state");
    }
}
