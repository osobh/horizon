//! Kernel Module Interface
//!
//! This module provides a userspace interface to the swarm_guard kernel module.
//! It communicates with the kernel module via ioctl calls on /dev/swarm_guard.
//!
//! ## Usage
//!
//! ```ignore
//! use swarmlet::kernel_interface::KernelInterface;
//!
//! let ki = KernelInterface::open()?;
//! let agent_id = ki.create_agent("build-job-123", 4096, 100)?;
//! ki.setup_mount_isolation(agent_id, &config)?;
//! // ... run build ...
//! ki.teardown_mount_isolation(agent_id)?;
//! ki.destroy_agent(agent_id)?;
//! ```

use crate::{Result, SwarmletError};
use std::fs::{File, OpenOptions};
use std::path::Path;
use tracing::info;

#[cfg(target_os = "linux")]
use std::os::unix::io::AsRawFd;

#[cfg(target_os = "linux")]
use tracing::{debug, warn};

/// Path to the swarm_guard device
const SWARM_GUARD_DEVICE: &str = "/dev/swarm_guard";

/// Maximum path length for mount paths
pub const MAX_PATH_LEN: usize = 256;

/// Maximum number of bind mounts
pub const MAX_BIND_MOUNTS: usize = 16;

// IOCTL constants - only used on Linux
#[cfg(target_os = "linux")]
mod ioctl_constants {
    /// IOCTL magic number (must match kernel header)
    pub const SWARM_IOC_MAGIC: u8 = b'S';

    // IOCTL command numbers (must match kernel header)
    pub const SWARM_AGENT_CREATE: u8 = 10;
    pub const SWARM_AGENT_DESTROY: u8 = 11;
    #[allow(dead_code)]
    pub const SWARM_AGENT_QUERY: u8 = 12;
    pub const SWARM_MOUNT_SETUP: u8 = 14;
    pub const SWARM_MOUNT_TEARDOWN: u8 = 15;
    pub const SWARM_MOUNT_ADD_BIND: u8 = 16;
    pub const SWARM_MOUNT_OVERLAYFS: u8 = 17;
}

/// Mount flags
pub mod mount_flags {
    pub const READONLY: u32 = 0x01;
    pub const NOSUID: u32 = 0x02;
    pub const NOEXEC: u32 = 0x04;
    pub const NODEV: u32 = 0x08;
}

/// Agent configuration for creation
#[repr(C)]
#[derive(Debug, Clone)]
pub struct AgentConfig {
    pub name: [u8; 64],
    pub memory_limit: u64,
    pub cpu_quota: u32,
    pub gpu_memory_limit: u64,
    pub namespace_flags: u32,
    pub agent_id: u64, // Output
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            name: [0; 64],
            memory_limit: 0,
            cpu_quota: 0,
            gpu_memory_limit: 0,
            namespace_flags: 0,
            agent_id: 0,
        }
    }
}

impl AgentConfig {
    pub fn new(name: &str, memory_limit_mb: u64, cpu_quota_percent: u32) -> Self {
        let mut config = Self::default();
        let name_bytes = name.as_bytes();
        let len = name_bytes.len().min(63);
        config.name[..len].copy_from_slice(&name_bytes[..len]);
        config.memory_limit = memory_limit_mb * 1024 * 1024;
        config.cpu_quota = cpu_quota_percent;
        config
    }
}

/// Bind mount configuration
#[repr(C)]
#[derive(Debug, Clone)]
pub struct BindMountConfig {
    pub agent_id: u64,
    pub source: [u8; MAX_PATH_LEN],
    pub target: [u8; MAX_PATH_LEN],
    pub flags: u32,
}

impl Default for BindMountConfig {
    fn default() -> Self {
        Self {
            agent_id: 0,
            source: [0; MAX_PATH_LEN],
            target: [0; MAX_PATH_LEN],
            flags: 0,
        }
    }
}

impl BindMountConfig {
    pub fn new(agent_id: u64, source: &str, target: &str, flags: u32) -> Self {
        let mut config = Self::default();
        config.agent_id = agent_id;
        copy_path(&mut config.source, source);
        copy_path(&mut config.target, target);
        config.flags = flags;
        config
    }
}

/// OverlayFS configuration
#[repr(C)]
#[derive(Debug, Clone)]
pub struct OverlayFsConfig {
    pub agent_id: u64,
    pub lower_dir: [u8; MAX_PATH_LEN],
    pub upper_dir: [u8; MAX_PATH_LEN],
    pub work_dir: [u8; MAX_PATH_LEN],
    pub merged_dir: [u8; MAX_PATH_LEN],
}

impl Default for OverlayFsConfig {
    fn default() -> Self {
        Self {
            agent_id: 0,
            lower_dir: [0; MAX_PATH_LEN],
            upper_dir: [0; MAX_PATH_LEN],
            work_dir: [0; MAX_PATH_LEN],
            merged_dir: [0; MAX_PATH_LEN],
        }
    }
}

impl OverlayFsConfig {
    pub fn new(agent_id: u64, lower: &str, upper: &str, work: &str, merged: &str) -> Self {
        let mut config = Self::default();
        config.agent_id = agent_id;
        copy_path(&mut config.lower_dir, lower);
        copy_path(&mut config.upper_dir, upper);
        copy_path(&mut config.work_dir, work);
        copy_path(&mut config.merged_dir, merged);
        config
    }
}

/// Single bind mount entry in mount config
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BindMountEntry {
    pub source: [u8; MAX_PATH_LEN],
    pub target: [u8; MAX_PATH_LEN],
    pub flags: u32,
}

impl Default for BindMountEntry {
    fn default() -> Self {
        Self {
            source: [0; MAX_PATH_LEN],
            target: [0; MAX_PATH_LEN],
            flags: 0,
        }
    }
}

/// Complete mount isolation configuration
#[repr(C)]
#[derive(Debug, Clone)]
pub struct MountIsolationConfig {
    pub agent_id: u64,
    pub lower_dir: [u8; MAX_PATH_LEN],
    pub upper_dir: [u8; MAX_PATH_LEN],
    pub work_dir: [u8; MAX_PATH_LEN],
    pub merged_dir: [u8; MAX_PATH_LEN],
    pub bind_mounts: [BindMountEntry; MAX_BIND_MOUNTS],
    pub num_bind_mounts: u32,
    pub old_root: [u8; MAX_PATH_LEN],
    pub use_overlayfs: u32,
    pub private_mounts: u32,
}

impl Default for MountIsolationConfig {
    fn default() -> Self {
        Self {
            agent_id: 0,
            lower_dir: [0; MAX_PATH_LEN],
            upper_dir: [0; MAX_PATH_LEN],
            work_dir: [0; MAX_PATH_LEN],
            merged_dir: [0; MAX_PATH_LEN],
            bind_mounts: [BindMountEntry::default(); MAX_BIND_MOUNTS],
            num_bind_mounts: 0,
            old_root: [0; MAX_PATH_LEN],
            use_overlayfs: 0,
            private_mounts: 1,
        }
    }
}

impl MountIsolationConfig {
    pub fn new(agent_id: u64) -> Self {
        let mut config = Self::default();
        config.agent_id = agent_id;
        config
    }

    pub fn with_overlayfs(mut self, lower: &str, upper: &str, work: &str, merged: &str) -> Self {
        copy_path(&mut self.lower_dir, lower);
        copy_path(&mut self.upper_dir, upper);
        copy_path(&mut self.work_dir, work);
        copy_path(&mut self.merged_dir, merged);
        self.use_overlayfs = 1;
        self
    }

    pub fn with_old_root(mut self, path: &str) -> Self {
        copy_path(&mut self.old_root, path);
        self
    }

    pub fn add_bind_mount(&mut self, source: &str, target: &str, flags: u32) -> bool {
        if self.num_bind_mounts as usize >= MAX_BIND_MOUNTS {
            return false;
        }
        let idx = self.num_bind_mounts as usize;
        copy_path(&mut self.bind_mounts[idx].source, source);
        copy_path(&mut self.bind_mounts[idx].target, target);
        self.bind_mounts[idx].flags = flags;
        self.num_bind_mounts += 1;
        true
    }
}

/// Helper to copy a path string into a fixed-size buffer
fn copy_path(buf: &mut [u8], path: &str) {
    let bytes = path.as_bytes();
    let len = bytes.len().min(buf.len() - 1);
    buf[..len].copy_from_slice(&bytes[..len]);
    buf[len] = 0;
}

/// Kernel module interface
pub struct KernelInterface {
    #[allow(dead_code)] // Only used on Linux
    device: File,
}

impl KernelInterface {
    /// Check if the kernel module is available
    pub fn is_available() -> bool {
        Path::new(SWARM_GUARD_DEVICE).exists()
            || Path::new("/sys/module/swarm_guard").exists()
            || Path::new("/proc/swarm").exists()
    }

    /// Open the kernel module interface
    pub fn open() -> Result<Self> {
        if !Path::new(SWARM_GUARD_DEVICE).exists() {
            return Err(SwarmletError::System(
                "swarm_guard device not found at /dev/swarm_guard".to_string(),
            ));
        }

        let device = OpenOptions::new()
            .read(true)
            .write(true)
            .open(SWARM_GUARD_DEVICE)
            .map_err(|e| {
                SwarmletError::System(format!("Failed to open swarm_guard device: {}", e))
            })?;

        info!("Opened swarm_guard kernel module interface");
        Ok(Self { device })
    }

    /// Create a new agent in the kernel module
    #[cfg(target_os = "linux")]
    pub fn create_agent(&self, config: &mut AgentConfig) -> Result<u64> {
        use ioctl_constants::SWARM_AGENT_CREATE;
        use nix::libc::ioctl;

        let cmd = make_ioctl_cmd(
            IoctlDir::ReadWrite,
            SWARM_AGENT_CREATE,
            std::mem::size_of::<AgentConfig>(),
        );

        // SAFETY: ioctl is called with a valid file descriptor (self.device is an open File),
        // a properly constructed ioctl command number, and a pointer to a valid AgentConfig
        // struct. The kernel module expects this exact struct layout and will write the
        // agent_id back into it. The pointer remains valid for the duration of the ioctl call.
        let ret = unsafe { ioctl(self.device.as_raw_fd(), cmd, config as *mut AgentConfig) };

        if ret < 0 {
            let errno = std::io::Error::last_os_error();
            return Err(SwarmletError::System(format!(
                "IOCTL SWARM_AGENT_CREATE failed: {}",
                errno
            )));
        }

        debug!("Created agent with ID {}", config.agent_id);
        Ok(config.agent_id)
    }

    #[cfg(not(target_os = "linux"))]
    pub fn create_agent(&self, _config: &mut AgentConfig) -> Result<u64> {
        Err(SwarmletError::NotImplemented(
            "Kernel interface only available on Linux".to_string(),
        ))
    }

    /// Destroy an agent
    #[cfg(target_os = "linux")]
    pub fn destroy_agent(&self, agent_id: u64) -> Result<()> {
        use ioctl_constants::SWARM_AGENT_DESTROY;
        use nix::libc::ioctl;

        let cmd = make_ioctl_cmd(
            IoctlDir::Write,
            SWARM_AGENT_DESTROY,
            std::mem::size_of::<u64>(),
        );

        // SAFETY: ioctl is called with a valid file descriptor (self.device is an open File),
        // a properly constructed ioctl command number, and a pointer to a valid u64 on the
        // stack. The kernel module reads the agent_id value to identify which agent to destroy.
        let ret = unsafe { ioctl(self.device.as_raw_fd(), cmd, &agent_id as *const u64) };

        if ret < 0 {
            let errno = std::io::Error::last_os_error();
            return Err(SwarmletError::System(format!(
                "IOCTL SWARM_AGENT_DESTROY failed: {}",
                errno
            )));
        }

        debug!("Destroyed agent {}", agent_id);
        Ok(())
    }

    #[cfg(not(target_os = "linux"))]
    pub fn destroy_agent(&self, _agent_id: u64) -> Result<()> {
        Err(SwarmletError::NotImplemented(
            "Kernel interface only available on Linux".to_string(),
        ))
    }

    /// Setup mount isolation for an agent
    #[cfg(target_os = "linux")]
    pub fn setup_mount_isolation(&self, config: &MountIsolationConfig) -> Result<()> {
        use ioctl_constants::SWARM_MOUNT_SETUP;
        use nix::libc::ioctl;

        let cmd = make_ioctl_cmd(
            IoctlDir::Write,
            SWARM_MOUNT_SETUP,
            std::mem::size_of::<MountIsolationConfig>(),
        );

        // SAFETY: ioctl is called with a valid file descriptor (self.device is an open File),
        // a properly constructed ioctl command number, and a pointer to a valid MountIsolationConfig
        // struct. The #[repr(C)] layout matches the kernel's expected struct. The kernel reads
        // the config to set up mount namespaces, bind mounts, and overlayfs for the agent.
        let ret = unsafe {
            ioctl(
                self.device.as_raw_fd(),
                cmd,
                config as *const MountIsolationConfig,
            )
        };

        if ret < 0 {
            let errno = std::io::Error::last_os_error();
            return Err(SwarmletError::System(format!(
                "IOCTL SWARM_MOUNT_SETUP failed: {}",
                errno
            )));
        }

        info!("Mount isolation setup for agent {}", config.agent_id);
        Ok(())
    }

    #[cfg(not(target_os = "linux"))]
    pub fn setup_mount_isolation(&self, _config: &MountIsolationConfig) -> Result<()> {
        Err(SwarmletError::NotImplemented(
            "Kernel interface only available on Linux".to_string(),
        ))
    }

    /// Teardown mount isolation for an agent
    #[cfg(target_os = "linux")]
    pub fn teardown_mount_isolation(&self, agent_id: u64) -> Result<()> {
        use ioctl_constants::SWARM_MOUNT_TEARDOWN;
        use nix::libc::ioctl;

        let cmd = make_ioctl_cmd(
            IoctlDir::Write,
            SWARM_MOUNT_TEARDOWN,
            std::mem::size_of::<u64>(),
        );

        // SAFETY: ioctl is called with a valid file descriptor (self.device is an open File),
        // a properly constructed ioctl command number, and a pointer to a valid u64 on the
        // stack. The kernel reads the agent_id to identify which agent's mounts to tear down.
        let ret = unsafe { ioctl(self.device.as_raw_fd(), cmd, &agent_id as *const u64) };

        if ret < 0 {
            let errno = std::io::Error::last_os_error();
            return Err(SwarmletError::System(format!(
                "IOCTL SWARM_MOUNT_TEARDOWN failed: {}",
                errno
            )));
        }

        info!("Mount isolation teardown for agent {}", agent_id);
        Ok(())
    }

    #[cfg(not(target_os = "linux"))]
    pub fn teardown_mount_isolation(&self, _agent_id: u64) -> Result<()> {
        Err(SwarmletError::NotImplemented(
            "Kernel interface only available on Linux".to_string(),
        ))
    }

    /// Add a bind mount to an agent
    #[cfg(target_os = "linux")]
    pub fn add_bind_mount(&self, config: &BindMountConfig) -> Result<()> {
        use ioctl_constants::SWARM_MOUNT_ADD_BIND;
        use nix::libc::ioctl;

        let cmd = make_ioctl_cmd(
            IoctlDir::Write,
            SWARM_MOUNT_ADD_BIND,
            std::mem::size_of::<BindMountConfig>(),
        );

        // SAFETY: ioctl is called with a valid file descriptor (self.device is an open File),
        // a properly constructed ioctl command number, and a pointer to a valid BindMountConfig
        // struct. The #[repr(C)] layout matches the kernel's expected struct. The kernel reads
        // source/target paths and flags to create a new bind mount for the agent.
        let ret = unsafe {
            ioctl(
                self.device.as_raw_fd(),
                cmd,
                config as *const BindMountConfig,
            )
        };

        if ret < 0 {
            let errno = std::io::Error::last_os_error();
            return Err(SwarmletError::System(format!(
                "IOCTL SWARM_MOUNT_ADD_BIND failed: {}",
                errno
            )));
        }

        debug!("Added bind mount for agent {}", config.agent_id);
        Ok(())
    }

    #[cfg(not(target_os = "linux"))]
    pub fn add_bind_mount(&self, _config: &BindMountConfig) -> Result<()> {
        Err(SwarmletError::NotImplemented(
            "Kernel interface only available on Linux".to_string(),
        ))
    }

    /// Setup overlayfs for an agent
    #[cfg(target_os = "linux")]
    pub fn setup_overlayfs(&self, config: &OverlayFsConfig) -> Result<()> {
        use ioctl_constants::SWARM_MOUNT_OVERLAYFS;
        use nix::libc::ioctl;

        let cmd = make_ioctl_cmd(
            IoctlDir::Write,
            SWARM_MOUNT_OVERLAYFS,
            std::mem::size_of::<OverlayFsConfig>(),
        );

        // SAFETY: ioctl is called with a valid file descriptor (self.device is an open File),
        // a properly constructed ioctl command number, and a pointer to a valid OverlayFsConfig
        // struct. The #[repr(C)] layout matches the kernel's expected struct. The kernel reads
        // lower/upper/work/merged paths to configure the overlayfs mount for the agent.
        let ret = unsafe {
            ioctl(
                self.device.as_raw_fd(),
                cmd,
                config as *const OverlayFsConfig,
            )
        };

        if ret < 0 {
            let errno = std::io::Error::last_os_error();
            return Err(SwarmletError::System(format!(
                "IOCTL SWARM_MOUNT_OVERLAYFS failed: {}",
                errno
            )));
        }

        debug!("OverlayFS setup for agent {}", config.agent_id);
        Ok(())
    }

    #[cfg(not(target_os = "linux"))]
    pub fn setup_overlayfs(&self, _config: &OverlayFsConfig) -> Result<()> {
        Err(SwarmletError::NotImplemented(
            "Kernel interface only available on Linux".to_string(),
        ))
    }
}

// IoctlDir and make_ioctl_cmd are only used on Linux

/// IOCTL direction flags
#[cfg(target_os = "linux")]
#[derive(Clone, Copy)]
enum IoctlDir {
    #[allow(dead_code)]
    None = 0,
    Write = 1,
    #[allow(dead_code)]
    Read = 2,
    ReadWrite = 3,
}

/// Build an ioctl command number
/// Format: direction (2 bits) | size (14 bits) | type (8 bits) | number (8 bits)
#[cfg(target_os = "linux")]
fn make_ioctl_cmd(dir: IoctlDir, nr: u8, size: usize) -> nix::libc::c_ulong {
    use ioctl_constants::SWARM_IOC_MAGIC;
    let dir_bits = (dir as u32) << 30;
    let size_bits = ((size as u32) & 0x3FFF) << 16;
    let type_bits = (SWARM_IOC_MAGIC as u32) << 8;
    let nr_bits = nr as u32;
    (dir_bits | size_bits | type_bits | nr_bits) as nix::libc::c_ulong
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_interface_availability() {
        // Will return false unless running on a system with swarm_guard
        let available = KernelInterface::is_available();
        let _ = available;
    }

    #[test]
    fn test_agent_config() {
        let config = AgentConfig::new("test-build", 4096, 100);
        assert_eq!(&config.name[..10], b"test-build");
        assert_eq!(config.memory_limit, 4096 * 1024 * 1024);
        assert_eq!(config.cpu_quota, 100);
    }

    #[test]
    fn test_mount_isolation_config() {
        let mut config = MountIsolationConfig::new(123)
            .with_overlayfs("/lower", "/upper", "/work", "/merged")
            .with_old_root("/.oldroot");

        assert_eq!(config.agent_id, 123);
        assert_eq!(config.use_overlayfs, 1);

        config.add_bind_mount("/host/cache", "/container/cache", mount_flags::READONLY);
        assert_eq!(config.num_bind_mounts, 1);
    }

    #[test]
    fn test_copy_path() {
        let mut buf = [0u8; 256];
        copy_path(&mut buf, "/test/path");
        assert_eq!(&buf[..10], b"/test/path");
        assert_eq!(buf[10], 0);
    }
}
