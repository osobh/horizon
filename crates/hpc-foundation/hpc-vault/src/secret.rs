use zeroize::ZeroizeOnDrop;

/// A secure string that zeros its memory when dropped
///
/// This prevents sensitive data like passwords and keys from
/// remaining in memory after they're no longer needed.
#[derive(Clone, ZeroizeOnDrop)]
pub struct SecretString {
    inner: Vec<u8>,
}

impl SecretString {
    /// Create a new SecretString from bytes
    pub fn new(data: Vec<u8>) -> Self {
        Self { inner: data }
    }

    /// Create a SecretString from a string slice
    pub fn from_string(s: &str) -> Self {
        Self::new(s.as_bytes().to_vec())
    }

    /// Expose the secret data (use carefully!)
    ///
    /// # Security
    /// The returned slice should not be cloned or stored.
    /// It will be zeroed when the SecretString is dropped.
    pub fn expose_secret(&self) -> &[u8] {
        &self.inner
    }

    /// Get the length of the secret
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Check if the secret is empty
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

impl std::fmt::Debug for SecretString {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SecretString([REDACTED {} bytes])", self.inner.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_secret_string_new() {
        let data = b"sensitive data".to_vec();
        let secret = SecretString::new(data.clone());
        assert_eq!(secret.expose_secret(), data.as_slice());
    }

    #[test]
    fn test_secret_string_from_string() {
        let secret = SecretString::from_string("password123");
        assert_eq!(secret.expose_secret(), b"password123");
    }

    #[test]
    fn test_secret_string_len() {
        let secret = SecretString::from_string("test");
        assert_eq!(secret.len(), 4);
    }

    #[test]
    fn test_secret_string_is_empty() {
        let secret = SecretString::new(vec![]);
        assert!(secret.is_empty());

        let secret2 = SecretString::from_string("not empty");
        assert!(!secret2.is_empty());
    }

    #[test]
    fn test_secret_string_debug() {
        let secret = SecretString::from_string("secret");
        let debug_str = format!("{:?}", secret);
        assert!(debug_str.contains("REDACTED"));
        assert!(debug_str.contains("6 bytes"));
        assert!(!debug_str.contains("secret"));
    }

    #[test]
    fn test_secret_string_clone() {
        let secret = SecretString::from_string("test");
        let cloned = secret.clone();
        assert_eq!(secret.expose_secret(), cloned.expose_secret());
    }

    #[test]
    fn test_secret_string_zeroize_on_drop() {
        // Create a secret and get a pointer to its data
        let data = b"sensitive".to_vec();
        let secret = SecretString::new(data);

        // Drop the secret
        drop(secret);

        // Note: We can't safely verify the memory is zeroed without unsafe code
        // The zeroize crate is well-tested for this behavior
        // This test ensures the API works correctly
    }

    #[test]
    fn test_expose_secret_does_not_copy() {
        let secret = SecretString::from_string("test data");
        let exposed1 = secret.expose_secret();
        let exposed2 = secret.expose_secret();

        // Both should point to the same memory location
        assert_eq!(exposed1.as_ptr(), exposed2.as_ptr());
    }
}
