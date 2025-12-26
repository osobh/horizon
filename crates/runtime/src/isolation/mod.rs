//! Container isolation and security
//!
//! This module provides isolation primitives for GPU container execution,
//! including memory quotas, kernel verification, and security enforcement.

mod container;
mod namespace;
mod sandbox;
mod security;

use std::collections::HashMap;
use std::sync::Mutex;

use crate::{ContainerConfig, RuntimeError};

pub use container::IsolatedContainer;
pub use namespace::Namespace;
pub use sandbox::Sandbox;
pub use security::SecurityPolicy;

/// Result of an isolation operation
#[derive(Debug, Clone, PartialEq)]
pub enum IsolationResult {
    /// Operation succeeded with return value (e.g., memory address)
    Success(u64),
    /// Operation failed due to security policy violation
    SecurityViolation(String),
    /// Operation failed because quota was exceeded
    QuotaExceeded(String),
    /// Operation failed because resource is unavailable
    ResourceUnavailable(String),
}

/// Signature for kernel verification
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct KernelSignature {
    /// Hash of the prompt that generated the kernel
    pub prompt_hash: String,
    /// Hash of the PTX code
    pub ptx_hash: String,
    /// Agent ID that created this kernel
    pub agent_id: Option<String>,
    /// Cryptographic signature (optional for unsigned kernels)
    pub signature: Option<String>,
    /// Unix timestamp when signature was created
    pub created_at: u64,
}

impl Default for KernelSignature {
    fn default() -> Self {
        Self {
            prompt_hash: String::new(),
            ptx_hash: String::new(),
            agent_id: None,
            signature: None,
            created_at: 0,
        }
    }
}

/// Statistics for a container's isolation context
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct IsolationStats {
    /// Container ID this stats belong to
    pub container_id: String,
    /// Total memory allocated in bytes
    pub memory_allocated: usize,
    /// Memory quota limit in bytes
    pub memory_quota: usize,
    /// Number of kernels launched
    pub kernels_launched: u64,
    /// Number of security violations detected
    pub security_violations: u64,
    /// Whether isolation is currently active
    pub is_active: bool,
    /// Timestamp of last activity
    pub last_activity: Option<u64>,
}

impl Default for IsolationStats {
    fn default() -> Self {
        Self {
            container_id: String::new(),
            memory_allocated: 0,
            memory_quota: 0,
            kernels_launched: 0,
            security_violations: 0,
            is_active: false,
            last_activity: None,
        }
    }
}

/// Internal isolation context for a container
struct IsolationContext {
    container_id: String,
    memory_quota: usize,
    memory_allocated: usize,
    kernels_launched: u64,
    security_violations: u64,
    is_active: bool,
}

/// Manages isolation contexts for containers
///
/// The `IsolationManager` is responsible for:
/// - Creating and managing isolation contexts per container
/// - Enforcing memory quotas
/// - Verifying kernel signatures before launch
/// - Tracking isolation statistics
pub struct IsolationManager {
    /// Active isolation contexts by container ID
    contexts: Mutex<HashMap<String, IsolationContext>>,
    /// Global memory counter for address allocation
    next_address: Mutex<u64>,
}

impl IsolationManager {
    /// Create a new isolation manager
    #[must_use]
    pub fn new() -> Self {
        Self {
            contexts: Mutex::new(HashMap::new()),
            next_address: Mutex::new(0x1000), // Start at 4KB
        }
    }

    /// Create isolation context for a container
    ///
    /// # Errors
    /// Returns `RuntimeError` if the lock cannot be acquired
    pub async fn create_isolation(
        &self,
        container_id: String,
        config: &ContainerConfig,
    ) -> Result<IsolationResult, RuntimeError> {
        let mut contexts = self.contexts.lock().map_err(|e| RuntimeError::StartupFailed {
            reason: format!("Failed to acquire contexts lock: {e}"),
        })?;

        if contexts.contains_key(&container_id) {
            return Ok(IsolationResult::ResourceUnavailable(format!(
                "Isolation context already exists for {}",
                container_id
            )));
        }

        contexts.insert(
            container_id.clone(),
            IsolationContext {
                container_id,
                memory_quota: config.memory_limit_bytes,
                memory_allocated: 0,
                kernels_launched: 0,
                security_violations: 0,
                is_active: true,
            },
        );

        Ok(IsolationResult::Success(0))
    }

    /// Launch a kernel with security verification
    ///
    /// # Errors
    /// Returns `RuntimeError` if the container is not found or lock fails
    pub async fn launch_kernel(
        &self,
        container_id: &str,
        kernel_signature: KernelSignature,
    ) -> Result<IsolationResult, RuntimeError> {
        let mut contexts = self.contexts.lock().map_err(|e| RuntimeError::StartupFailed {
            reason: format!("Failed to acquire contexts lock: {e}"),
        })?;

        let context = contexts.get_mut(container_id).ok_or_else(|| RuntimeError::InvalidConfig {
            reason: format!("Container {} not found", container_id),
        })?;

        // Security check: empty prompt hash is invalid
        if kernel_signature.prompt_hash.is_empty() {
            context.security_violations += 1;
            return Ok(IsolationResult::SecurityViolation(
                "Empty prompt hash is not allowed".to_string(),
            ));
        }

        context.kernels_launched += 1;
        Ok(IsolationResult::Success(context.kernels_launched))
    }

    /// Allocate memory for a container with quota enforcement
    ///
    /// # Errors
    /// Returns `RuntimeError` if the container is not found or lock fails
    pub async fn allocate_memory(
        &self,
        container_id: &str,
        size_bytes: usize,
    ) -> Result<IsolationResult, RuntimeError> {
        let mut contexts = self.contexts.lock().map_err(|e| RuntimeError::StartupFailed {
            reason: format!("Failed to acquire contexts lock: {e}"),
        })?;

        let context = contexts.get_mut(container_id).ok_or_else(|| RuntimeError::InvalidConfig {
            reason: format!("Container {} not found", container_id),
        })?;

        // Check quota
        if context.memory_allocated + size_bytes > context.memory_quota {
            return Ok(IsolationResult::QuotaExceeded(format!(
                "Allocation of {} bytes would exceed quota of {} bytes (current: {} bytes)",
                size_bytes, context.memory_quota, context.memory_allocated
            )));
        }

        // Allocate address
        let mut next_addr = self.next_address.lock().map_err(|e| RuntimeError::StartupFailed {
            reason: format!("Failed to acquire address lock: {e}"),
        })?;

        let address = *next_addr;
        *next_addr += size_bytes as u64;
        context.memory_allocated += size_bytes;

        Ok(IsolationResult::Success(address))
    }

    /// Terminate a container's isolation context
    ///
    /// # Errors
    /// Returns `RuntimeError` if the container is not found or lock fails
    pub async fn terminate_container(&self, container_id: &str) -> Result<(), RuntimeError> {
        let mut contexts = self.contexts.lock().map_err(|e| RuntimeError::StartupFailed {
            reason: format!("Failed to acquire contexts lock: {e}"),
        })?;

        contexts.remove(container_id).ok_or_else(|| RuntimeError::InvalidConfig {
            reason: format!("Container {} not found", container_id),
        })?;

        Ok(())
    }

    /// Get isolation statistics for a container
    ///
    /// # Errors
    /// Returns `RuntimeError` if the container is not found or lock fails
    pub async fn get_isolation_stats(&self, container_id: &str) -> Result<IsolationStats, RuntimeError> {
        let contexts = self.contexts.lock().map_err(|e| RuntimeError::StartupFailed {
            reason: format!("Failed to acquire contexts lock: {e}"),
        })?;

        let context = contexts.get(container_id).ok_or_else(|| RuntimeError::ContainerNotFound {
            id: container_id.to_string(),
        })?;

        Ok(IsolationStats {
            container_id: context.container_id.clone(),
            memory_allocated: context.memory_allocated,
            memory_quota: context.memory_quota,
            kernels_launched: context.kernels_launched,
            security_violations: context.security_violations,
            is_active: context.is_active,
            last_activity: None,
        })
    }

    /// List all active isolation contexts
    ///
    /// # Errors
    /// Returns `RuntimeError` if the lock cannot be acquired
    pub async fn list_contexts(&self) -> Result<Vec<String>, RuntimeError> {
        let contexts = self.contexts.lock().map_err(|e| RuntimeError::StartupFailed {
            reason: format!("Failed to acquire contexts lock: {e}"),
        })?;

        Ok(contexts.keys().cloned().collect())
    }
}

impl Default for IsolationManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_isolation_result_variants() {
        let success = IsolationResult::Success(42);
        assert_eq!(success, IsolationResult::Success(42));

        let violation = IsolationResult::SecurityViolation("test".to_string());
        assert_eq!(violation, IsolationResult::SecurityViolation("test".to_string()));

        let quota = IsolationResult::QuotaExceeded("limit".to_string());
        assert_eq!(quota, IsolationResult::QuotaExceeded("limit".to_string()));

        let unavailable = IsolationResult::ResourceUnavailable("busy".to_string());
        assert_eq!(unavailable, IsolationResult::ResourceUnavailable("busy".to_string()));
    }

    #[test]
    fn test_kernel_signature_default() {
        let sig = KernelSignature::default();
        assert!(sig.prompt_hash.is_empty());
        assert!(sig.ptx_hash.is_empty());
        assert!(sig.agent_id.is_none());
        assert!(sig.signature.is_none());
        assert_eq!(sig.created_at, 0);
    }

    #[test]
    fn test_isolation_stats_default() {
        let stats = IsolationStats::default();
        assert!(stats.container_id.is_empty());
        assert_eq!(stats.memory_allocated, 0);
        assert_eq!(stats.memory_quota, 0);
        assert_eq!(stats.kernels_launched, 0);
        assert_eq!(stats.security_violations, 0);
        assert!(!stats.is_active);
        assert!(stats.last_activity.is_none());
    }

    #[test]
    fn test_isolation_manager_new() {
        let manager = IsolationManager::new();
        // Verify it can be created without panicking
        drop(manager);
    }

    #[tokio::test]
    async fn test_isolation_manager_create_and_terminate() {
        let manager = IsolationManager::new();
        let config = ContainerConfig::default();

        // Create isolation
        let result = manager.create_isolation("test-1".to_string(), &config).await;
        assert!(result.is_ok());
        assert!(matches!(result.unwrap(), IsolationResult::Success(_)));

        // Terminate
        let term_result = manager.terminate_container("test-1").await;
        assert!(term_result.is_ok());
    }

    #[tokio::test]
    async fn test_isolation_manager_duplicate_create() {
        let manager = IsolationManager::new();
        let config = ContainerConfig::default();

        // Create first
        let _ = manager.create_isolation("dup-test".to_string(), &config).await;

        // Try to create duplicate
        let result = manager.create_isolation("dup-test".to_string(), &config).await;
        assert!(result.is_ok());
        assert!(matches!(result.unwrap(), IsolationResult::ResourceUnavailable(_)));
    }

    #[tokio::test]
    async fn test_isolation_manager_memory_allocation() {
        let manager = IsolationManager::new();
        let config = ContainerConfig {
            memory_limit_bytes: 1024,
            ..Default::default()
        };

        let _ = manager.create_isolation("mem-test".to_string(), &config).await;

        // Allocate within quota
        let result = manager.allocate_memory("mem-test", 512).await;
        assert!(result.is_ok());
        assert!(matches!(result.unwrap(), IsolationResult::Success(_)));

        // Allocate exceeding quota
        let result = manager.allocate_memory("mem-test", 1024).await;
        assert!(result.is_ok());
        assert!(matches!(result.unwrap(), IsolationResult::QuotaExceeded(_)));
    }

    #[tokio::test]
    async fn test_isolation_manager_kernel_launch() {
        let manager = IsolationManager::new();
        let config = ContainerConfig::default();

        let _ = manager.create_isolation("kernel-test".to_string(), &config).await;

        // Valid kernel signature
        let sig = KernelSignature {
            prompt_hash: "abc123".to_string(),
            ptx_hash: "def456".to_string(),
            agent_id: None,
            signature: None,
            created_at: 1234567890,
        };
        let result = manager.launch_kernel("kernel-test", sig).await;
        assert!(result.is_ok());
        assert!(matches!(result.unwrap(), IsolationResult::Success(_)));

        // Empty prompt hash should fail
        let invalid_sig = KernelSignature::default();
        let result = manager.launch_kernel("kernel-test", invalid_sig).await;
        assert!(result.is_ok());
        assert!(matches!(result.unwrap(), IsolationResult::SecurityViolation(_)));
    }

    #[tokio::test]
    async fn test_isolation_manager_get_stats() {
        let manager = IsolationManager::new();
        let config = ContainerConfig {
            memory_limit_bytes: 2048,
            ..Default::default()
        };

        let _ = manager.create_isolation("stats-test".to_string(), &config).await;
        let _ = manager.allocate_memory("stats-test", 1024).await;

        let stats = manager.get_isolation_stats("stats-test").await;
        assert!(stats.is_ok());
        let stats = stats.unwrap();
        assert_eq!(stats.container_id, "stats-test");
        assert_eq!(stats.memory_allocated, 1024);
        assert_eq!(stats.memory_quota, 2048);
        assert!(stats.is_active);
    }

    #[tokio::test]
    async fn test_isolation_manager_list_contexts() {
        let manager = IsolationManager::new();
        let config = ContainerConfig::default();

        let _ = manager.create_isolation("list-1".to_string(), &config).await;
        let _ = manager.create_isolation("list-2".to_string(), &config).await;

        let contexts = manager.list_contexts().await;
        assert!(contexts.is_ok());
        let contexts = contexts.unwrap();
        assert_eq!(contexts.len(), 2);
        assert!(contexts.contains(&"list-1".to_string()));
        assert!(contexts.contains(&"list-2".to_string()));
    }
}
