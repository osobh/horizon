//! Application state for the cluster coordinator API.

use crate::install_script::BinaryChecksums;
use chrono::{DateTime, Duration, Utc};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, Ordering};
use uuid::Uuid;

/// Application state shared across all handlers.
#[derive(Debug)]
pub struct AppState {
    /// Cluster hostname or IP address
    pub cluster_host: String,
    /// Cluster port
    pub cluster_port: u16,
    /// Swarmlet version to install
    pub swarmlet_version: String,
    /// Docker image for container deployment
    pub docker_image: String,
    /// Base URL for binary releases
    pub releases_base_url: String,
    /// Binary checksums for verification
    pub binary_checksums: BinaryChecksums,
    /// Token store for validation
    tokens: DashMap<String, TokenInfo>,
    /// Whether the service is ready
    ready: AtomicBool,
}

/// Information about a stored token.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenInfo {
    /// Token ID
    pub id: Uuid,
    /// When the token expires
    pub expires_at: DateTime<Utc>,
    /// Maximum number of uses (0 = unlimited)
    pub max_uses: u32,
    /// Current number of uses
    pub current_uses: u32,
    /// Capabilities granted by this token
    pub capabilities: Vec<String>,
}

impl TokenInfo {
    /// Check if the token has expired.
    pub fn is_expired(&self) -> bool {
        Utc::now() > self.expires_at
    }

    /// Check if the token can still be used.
    pub fn is_valid(&self) -> bool {
        !self.is_expired() && (self.max_uses == 0 || self.current_uses < self.max_uses)
    }
}

/// Configuration for creating AppState.
#[derive(Debug, Clone)]
pub struct AppStateConfig {
    pub cluster_host: String,
    pub cluster_port: u16,
    pub swarmlet_version: Option<String>,
    pub docker_image: Option<String>,
    pub releases_base_url: Option<String>,
    pub binary_checksums: Option<BinaryChecksums>,
}

impl Default for AppStateConfig {
    fn default() -> Self {
        Self {
            cluster_host: "localhost".to_string(),
            cluster_port: 7946,
            swarmlet_version: None,
            docker_image: None,
            releases_base_url: None,
            binary_checksums: None,
        }
    }
}

impl AppStateConfig {
    /// Load configuration from environment variables.
    ///
    /// Reads from .env file if present, then checks environment variables:
    /// - `CLUSTER_COORDINATOR_HOST` (default: coordinator.stratoswarm.com)
    /// - `CLUSTER_COORDINATOR_PORT` (default: 7946)
    /// - `SWARMLET_VERSION` (default: crate version)
    /// - `SWARMLET_DOCKER_IMAGE` (default: stratoswarm/swarmlet)
    /// - `SWARMLET_RELEASES_URL` (default: https://releases.stratoswarm.com/swarmlet)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use stratoswarm_cluster_mesh::api::AppStateConfig;
    ///
    /// // Set environment or create .env file
    /// std::env::set_var("CLUSTER_COORDINATOR_HOST", "my-cluster.local");
    ///
    /// let config = AppStateConfig::from_env();
    /// assert_eq!(config.cluster_host, "my-cluster.local");
    /// ```
    pub fn from_env() -> Self {
        // Load .env file if present (silently ignore if missing)
        let _ = dotenvy::dotenv();

        Self {
            cluster_host: std::env::var("CLUSTER_COORDINATOR_HOST")
                .unwrap_or_else(|_| "coordinator.stratoswarm.com".to_string()),
            cluster_port: std::env::var("CLUSTER_COORDINATOR_PORT")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(7946),
            swarmlet_version: std::env::var("SWARMLET_VERSION").ok(),
            docker_image: std::env::var("SWARMLET_DOCKER_IMAGE").ok(),
            releases_base_url: std::env::var("SWARMLET_RELEASES_URL").ok(),
            binary_checksums: None,
        }
    }
}

impl AppState {
    /// Create a new AppState with the given configuration.
    pub fn new(config: AppStateConfig) -> Self {
        Self {
            cluster_host: config.cluster_host,
            cluster_port: config.cluster_port,
            swarmlet_version: config
                .swarmlet_version
                .unwrap_or_else(|| env!("CARGO_PKG_VERSION").to_string()),
            docker_image: config
                .docker_image
                .unwrap_or_else(|| "stratoswarm/swarmlet".to_string()),
            releases_base_url: config
                .releases_base_url
                .unwrap_or_else(|| "https://releases.stratoswarm.com/swarmlet".to_string()),
            binary_checksums: config.binary_checksums.unwrap_or_default(),
            tokens: DashMap::new(),
            ready: AtomicBool::new(true),
        }
    }

    /// Check if the service is ready.
    pub fn is_ready(&self) -> bool {
        self.ready.load(Ordering::Relaxed)
    }

    /// Set the ready state.
    pub fn set_ready(&self, ready: bool) {
        self.ready.store(ready, Ordering::Relaxed);
    }

    /// Add a token to the store.
    pub fn add_token(&self, token: String, info: TokenInfo) {
        self.tokens.insert(token, info);
    }

    /// Validate a token and return its info.
    pub async fn validate_token(&self, token: &str) -> Result<TokenInfo, TokenError> {
        // First, check if the token exists in our store
        if let Some(mut entry) = self.tokens.get_mut(token) {
            let info = entry.value().clone();

            // Check if token is valid
            if info.is_expired() {
                return Err(TokenError::Expired);
            }

            if info.max_uses > 0 && info.current_uses >= info.max_uses {
                return Err(TokenError::MaxUsesExceeded);
            }

            // Increment usage counter
            entry.current_uses += 1;

            return Ok(info);
        }

        // For development/testing: accept any token that looks like a valid format
        // In production, this should be removed and all tokens should be pre-registered
        if token.len() >= 32 {
            // Accept long tokens as valid for development
            let info = TokenInfo {
                id: Uuid::new_v4(),
                expires_at: Utc::now() + Duration::hours(24),
                max_uses: 0, // unlimited
                current_uses: 0,
                capabilities: vec![],
            };

            tracing::warn!(
                "Accepting unregistered token for development (length={})",
                token.len()
            );

            return Ok(info);
        }

        Err(TokenError::Invalid)
    }

    /// Remove a token from the store.
    pub fn revoke_token(&self, token: &str) -> bool {
        self.tokens.remove(token).is_some()
    }

    /// Generate a new token with the given parameters.
    pub fn generate_token(
        &self,
        ttl: Duration,
        max_uses: u32,
        capabilities: Vec<String>,
    ) -> (String, TokenInfo) {
        let token_bytes: [u8; 32] = rand::random();
        let token = base64::Engine::encode(
            &base64::engine::general_purpose::URL_SAFE_NO_PAD,
            token_bytes,
        );

        let info = TokenInfo {
            id: Uuid::new_v4(),
            expires_at: Utc::now() + ttl,
            max_uses,
            current_uses: 0,
            capabilities,
        };

        self.tokens.insert(token.clone(), info.clone());

        (token, info)
    }

    /// Get statistics about tokens.
    pub fn token_stats(&self) -> TokenStats {
        let total = self.tokens.len();
        let expired = self
            .tokens
            .iter()
            .filter(|e| e.value().is_expired())
            .count();
        let exhausted = self
            .tokens
            .iter()
            .filter(|e| {
                let v = e.value();
                v.max_uses > 0 && v.current_uses >= v.max_uses
            })
            .count();

        TokenStats {
            total,
            active: total - expired - exhausted,
            expired,
            exhausted,
        }
    }

    /// Clean up expired tokens.
    pub fn cleanup_expired_tokens(&self) -> usize {
        let before = self.tokens.len();
        self.tokens.retain(|_, v| !v.is_expired());
        before - self.tokens.len()
    }
}

/// Token validation errors.
#[derive(Debug, thiserror::Error)]
pub enum TokenError {
    #[error("Invalid token")]
    Invalid,
    #[error("Token has expired")]
    Expired,
    #[error("Token has exceeded maximum uses")]
    MaxUsesExceeded,
}

/// Statistics about tokens.
#[derive(Debug, Clone, Serialize)]
pub struct TokenStats {
    pub total: usize,
    pub active: usize,
    pub expired: usize,
    pub exhausted: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_app_state_creation() {
        let config = AppStateConfig {
            cluster_host: "test.local".to_string(),
            cluster_port: 8080,
            ..Default::default()
        };

        let state = AppState::new(config);
        assert_eq!(state.cluster_host, "test.local");
        assert_eq!(state.cluster_port, 8080);
        assert!(state.is_ready());
    }

    #[test]
    fn test_token_generation() {
        let state = AppState::new(AppStateConfig::default());

        let (token, info) = state.generate_token(Duration::hours(1), 5, vec![]);

        assert!(!token.is_empty());
        assert_eq!(info.max_uses, 5);
        assert_eq!(info.current_uses, 0);
        assert!(!info.is_expired());
    }

    #[tokio::test]
    async fn test_token_validation() {
        let state = AppState::new(AppStateConfig::default());

        let (token, _) = state.generate_token(Duration::hours(1), 5, vec![]);

        let result = state.validate_token(&token).await;
        assert!(result.is_ok());

        let info = result.unwrap();
        assert_eq!(info.current_uses, 1); // Should be incremented
    }

    #[tokio::test]
    async fn test_invalid_token() {
        let state = AppState::new(AppStateConfig::default());

        let result = state.validate_token("short").await;
        assert!(result.is_err());
    }

    #[test]
    fn test_token_revocation() {
        let state = AppState::new(AppStateConfig::default());

        let (token, _) = state.generate_token(Duration::hours(1), 5, vec![]);

        assert!(state.revoke_token(&token));
        assert!(!state.revoke_token(&token)); // Already revoked
    }

    #[test]
    fn test_cleanup_expired_tokens() {
        let state = AppState::new(AppStateConfig::default());

        // Add an expired token manually
        state.tokens.insert(
            "expired-token".to_string(),
            TokenInfo {
                id: Uuid::new_v4(),
                expires_at: Utc::now() - Duration::hours(1), // Already expired
                max_uses: 0,
                current_uses: 0,
                capabilities: vec![],
            },
        );

        // Add a valid token
        state.generate_token(Duration::hours(1), 0, vec![]);

        let cleaned = state.cleanup_expired_tokens();
        assert_eq!(cleaned, 1);
    }
}
