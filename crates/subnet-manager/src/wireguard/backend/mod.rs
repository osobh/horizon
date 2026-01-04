//! WireGuard backend implementations
//!
//! Provides multiple WireGuard backends with automatic selection:
//!
//! 1. **Netlink** (Linux only) - Direct kernel interface, fastest
//! 2. **Command** - Shell out to `wg`/`ip` tools, most portable
//! 3. **Userspace** - boringtun, cross-platform, no kernel module needed
//!
//! # Usage
//!
//! ```ignore
//! use subnet_manager::wireguard::backend::AutoSelectBackend;
//!
//! let backend = AutoSelectBackend::new();
//! let selected = backend.select();
//! println!("Using {} backend", selected.backend_type());
//! ```

mod traits;

#[cfg(feature = "wg-command")]
pub mod command;

#[cfg(all(target_os = "linux", feature = "wg-netlink"))]
pub mod netlink;

#[cfg(feature = "wg-userspace")]
pub mod userspace;

pub use traits::{BackendCapabilities, BackendType, InterfaceStats, PeerStats, WireGuardBackend};

#[cfg(feature = "wg-command")]
pub use command::CommandBackend;

#[cfg(all(target_os = "linux", feature = "wg-netlink"))]
pub use netlink::NetlinkBackend;

#[cfg(feature = "wg-userspace")]
pub use userspace::UserspaceBackend;

use std::sync::Arc;
use tracing::{debug, info, warn};

/// Automatic backend selection with fallback chain
///
/// Selects the best available backend based on:
/// 1. Platform (Linux gets netlink priority)
/// 2. Availability (is the tool/module present?)
/// 3. User preference (if specified)
pub struct AutoSelectBackend {
    /// User-preferred backend type (optional)
    preferred: Option<BackendType>,
    /// Available backends in priority order
    available: Vec<Arc<dyn WireGuardBackend>>,
}

impl AutoSelectBackend {
    /// Create a new auto-selecting backend
    ///
    /// Discovers available backends and orders them by priority.
    pub fn new() -> Self {
        let mut backends: Vec<Arc<dyn WireGuardBackend>> = vec![];

        // Priority 1: Netlink (Linux only, fastest)
        #[cfg(all(target_os = "linux", feature = "wg-netlink"))]
        {
            let netlink = NetlinkBackend::new();
            if netlink.is_available() {
                info!("Netlink WireGuard backend available");
                backends.push(Arc::new(netlink));
            } else {
                debug!("Netlink WireGuard backend not available");
            }
        }

        // Priority 2: Command-line (requires wg tool installed)
        #[cfg(feature = "wg-command")]
        {
            let command = CommandBackend::new();
            if command.is_available() {
                info!("Command WireGuard backend available");
                backends.push(Arc::new(command));
            } else {
                debug!("Command WireGuard backend not available (wg not found)");
            }
        }

        // Priority 3: Userspace (always available, cross-platform)
        #[cfg(feature = "wg-userspace")]
        {
            let userspace = UserspaceBackend::new();
            if userspace.is_available() {
                info!("Userspace WireGuard backend available");
                backends.push(Arc::new(userspace));
            }
        }

        if backends.is_empty() {
            warn!("No WireGuard backends available! Enable at least one backend feature.");
        }

        Self {
            preferred: None,
            available: backends,
        }
    }

    /// Create with a preferred backend type
    pub fn with_preference(preferred: BackendType) -> Self {
        let mut backend = Self::new();
        backend.preferred = Some(preferred);
        backend
    }

    /// Set the preferred backend type
    pub fn set_preference(&mut self, preferred: BackendType) {
        self.preferred = Some(preferred);
    }

    /// Select the best available backend
    ///
    /// Returns the preferred backend if available, otherwise the highest
    /// priority available backend.
    pub fn select(&self) -> Option<Arc<dyn WireGuardBackend>> {
        // Try preferred backend first
        if let Some(pref) = &self.preferred {
            if let Some(backend) = self.available.iter().find(|b| &b.backend_type() == pref) {
                return Some(Arc::clone(backend));
            }
            warn!("Preferred backend {:?} not available, falling back", pref);
        }

        // Return first available (highest priority)
        self.available.first().cloned()
    }

    /// Get the selected backend, panicking if none available
    ///
    /// Use this when WireGuard is required for operation.
    pub fn select_or_panic(&self) -> Arc<dyn WireGuardBackend> {
        self.select()
            .expect("No WireGuard backend available. Enable at least one backend feature.")
    }

    /// List all available backends
    pub fn available_backends(&self) -> Vec<BackendType> {
        self.available.iter().map(|b| b.backend_type()).collect()
    }

    /// Check if any backend is available
    pub fn has_backend(&self) -> bool {
        !self.available.is_empty()
    }

    /// Check if a specific backend type is available
    pub fn has_backend_type(&self, backend_type: BackendType) -> bool {
        self.available
            .iter()
            .any(|b| b.backend_type() == backend_type)
    }
}

impl Default for AutoSelectBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auto_select_creation() {
        let backend = AutoSelectBackend::new();
        // At minimum, should not panic
        let _ = backend.available_backends();
    }

    #[test]
    fn test_preference_setting() {
        let mut backend = AutoSelectBackend::new();
        backend.set_preference(BackendType::Command);
        assert_eq!(backend.preferred, Some(BackendType::Command));
    }

    #[test]
    fn test_with_preference() {
        let backend = AutoSelectBackend::with_preference(BackendType::Userspace);
        assert_eq!(backend.preferred, Some(BackendType::Userspace));
    }
}
