//! Stub build backend
//!
//! A placeholder backend used when no real backend (Docker or Linux native)
//! is available. This allows the agent to start but build jobs will fail
//! with a clear error message.

use super::{BackendCapabilities, BackendType, BuildBackend, BuildContext};
use crate::build_job::{BuildResult, CargoCommand};
use crate::{Result, SwarmletError};
use async_trait::async_trait;
use std::path::PathBuf;

/// Stub backend that returns errors for all operations
pub struct StubBackend {
    /// Reason why no real backend is available
    reason: String,
}

impl StubBackend {
    /// Create a new stub backend with a reason message
    pub fn new(reason: impl Into<String>) -> Self {
        Self {
            reason: reason.into(),
        }
    }

    /// Create with default reason
    pub fn unavailable() -> Self {
        Self::new("No build backend available. Please install Docker or run on Linux with appropriate permissions.")
    }
}

#[async_trait]
impl BuildBackend for StubBackend {
    async fn execute_cargo(
        &self,
        _command: &CargoCommand,
        _context: &BuildContext,
    ) -> Result<BuildResult> {
        Err(SwarmletError::NotImplemented(self.reason.clone()))
    }

    async fn create_workspace(&self, _job_id: &str) -> Result<PathBuf> {
        Err(SwarmletError::NotImplemented(self.reason.clone()))
    }

    async fn cleanup_workspace(&self, _workspace: &PathBuf) -> Result<()> {
        // Cleanup is a no-op for stub
        Ok(())
    }

    async fn is_available(&self) -> bool {
        false
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            full_namespace_isolation: false,
            cgroups_v2: false,
            gpu_passthrough: false,
            seccomp: false,
            user_namespace: false,
            max_containers: 0,
        }
    }

    fn backend_type(&self) -> BackendType {
        BackendType::Stub
    }

    fn name(&self) -> &str {
        "stub"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_stub_backend_not_available() {
        let backend = StubBackend::unavailable();
        assert!(!backend.is_available().await);
    }

    #[tokio::test]
    async fn test_stub_backend_execute_fails() {
        let backend = StubBackend::unavailable();
        let context = BuildContext::new(PathBuf::from("/workspace"), PathBuf::from("/toolchain"));
        let result = backend.execute_cargo(&CargoCommand::Build, &context).await;
        assert!(result.is_err());
    }

    #[test]
    fn test_stub_backend_capabilities() {
        let backend = StubBackend::unavailable();
        let caps = backend.capabilities();
        assert_eq!(caps.max_containers, 0);
        assert!(!caps.cgroups_v2);
    }
}
