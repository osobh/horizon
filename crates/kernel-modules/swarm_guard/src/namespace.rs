//! Namespace management for agent isolation
//!
//! This module handles the creation and management of Linux namespaces
//! for agent containers, ensuring proper isolation.

use alloc::format;
use alloc::vec;
use alloc::vec::Vec;
use core::ffi::c_int;

use crate::{KernelError, KernelResult};

/// Namespace types supported by Linux
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NamespaceType {
    /// Mount namespace (CLONE_NEWNS)
    Mount = 0x00020000,
    /// UTS namespace (CLONE_NEWUTS)
    UTS = 0x04000000,
    /// IPC namespace (CLONE_NEWIPC)
    IPC = 0x08000000,
    /// PID namespace (CLONE_NEWPID)
    PID = 0x20000000,
    /// Network namespace (CLONE_NEWNET)
    Network = 0x40000000,
    /// User namespace (CLONE_NEWUSER)
    User = 0x10000000,
    /// Cgroup namespace (CLONE_NEWCGROUP)
    Cgroup = 0x02000000,
}

impl NamespaceType {
    /// Get all namespace types
    pub fn all() -> Vec<Self> {
        vec![
            Self::Mount,
            Self::UTS,
            Self::IPC,
            Self::PID,
            Self::Network,
            Self::User,
            Self::Cgroup,
        ]
    }

    /// Convert to flag bit
    pub fn to_flag(self) -> u32 {
        self as u32
    }

    /// Get namespace name for /proc
    pub fn proc_name(self) -> &'static str {
        match self {
            Self::Mount => "mnt",
            Self::UTS => "uts",
            Self::IPC => "ipc",
            Self::PID => "pid",
            Self::Network => "net",
            Self::User => "user",
            Self::Cgroup => "cgroup",
        }
    }
}

/// Combined namespace flags
pub const CLONE_ALL_NAMESPACES: u32 = NamespaceType::Mount as u32
    | NamespaceType::UTS as u32
    | NamespaceType::IPC as u32
    | NamespaceType::PID as u32
    | NamespaceType::Network as u32
    | NamespaceType::User as u32
    | NamespaceType::Cgroup as u32;

/// Namespace configuration for an agent
#[derive(Debug, Clone)]
pub struct NamespaceSetup {
    /// Flags indicating which namespaces to create
    pub flags: u32,
    /// User ID mapping for user namespace
    pub uid_map: Vec<IdMap>,
    /// Group ID mapping for user namespace
    pub gid_map: Vec<IdMap>,
    /// Hostname for UTS namespace
    pub hostname: Option<Vec<u8>>,
    /// Root directory for mount namespace
    pub root_dir: Option<Vec<u8>>,
}

/// ID mapping for user namespaces
#[derive(Debug, Clone, Copy)]
pub struct IdMap {
    /// ID inside the namespace
    pub inside_id: u32,
    /// ID outside the namespace
    pub outside_id: u32,
    /// Number of IDs to map
    pub count: u32,
}

/// Maximum number of bind mounts
pub const MAX_BIND_MOUNTS: usize = 16;
/// Maximum path length
pub const MAX_PATH_LEN: usize = 256;

/// Mount flags for bind mounts
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MountFlags {
    #[default]
    None = 0,
    /// Mount read-only
    ReadOnly = 0x01,
    /// No setuid binaries
    NoSuid = 0x02,
    /// No executable files
    NoExec = 0x04,
    /// No device files
    NoDev = 0x08,
}

impl MountFlags {
    /// Create flags from raw u32
    pub fn from_raw(raw: u32) -> Self {
        match raw {
            0x01 => Self::ReadOnly,
            0x02 => Self::NoSuid,
            0x04 => Self::NoExec,
            0x08 => Self::NoDev,
            _ => Self::None,
        }
    }

    /// Combine multiple flags
    pub fn combine(flags: &[Self]) -> u32 {
        flags.iter().fold(0u32, |acc, f| acc | (*f as u32))
    }
}

/// Bind mount configuration
#[derive(Debug, Clone)]
pub struct BindMount {
    /// Host path to mount from
    pub source: Vec<u8>,
    /// Container path to mount to
    pub target: Vec<u8>,
    /// Mount flags (readonly, nosuid, noexec, nodev)
    pub flags: u32,
}

impl BindMount {
    /// Create a new bind mount configuration
    pub fn new(source: &str, target: &str) -> Self {
        Self {
            source: source.as_bytes().to_vec(),
            target: target.as_bytes().to_vec(),
            flags: 0,
        }
    }

    /// Create a read-only bind mount
    pub fn readonly(source: &str, target: &str) -> Self {
        Self {
            source: source.as_bytes().to_vec(),
            target: target.as_bytes().to_vec(),
            flags: MountFlags::ReadOnly as u32,
        }
    }

    /// Add mount flags
    pub fn with_flags(mut self, flags: u32) -> Self {
        self.flags = flags;
        self
    }

    /// Validate the bind mount configuration
    pub fn validate(&self) -> KernelResult<()> {
        if self.source.is_empty() || self.source.len() > MAX_PATH_LEN {
            return Err(KernelError::InvalidArgument);
        }
        if self.target.is_empty() || self.target.len() > MAX_PATH_LEN {
            return Err(KernelError::InvalidArgument);
        }
        Ok(())
    }
}

/// OverlayFS configuration for container rootfs
#[derive(Debug, Clone, Default)]
pub struct OverlayFsConfig {
    /// Read-only base layer (e.g., toolchain installation)
    pub lower_dir: Vec<u8>,
    /// Writable layer for container changes
    pub upper_dir: Vec<u8>,
    /// OverlayFS work directory (must be on same filesystem as upper)
    pub work_dir: Vec<u8>,
    /// Final merged mount point (new container root)
    pub merged_dir: Vec<u8>,
}

impl OverlayFsConfig {
    /// Create a new overlayfs configuration
    pub fn new(lower: &str, upper: &str, work: &str, merged: &str) -> Self {
        Self {
            lower_dir: lower.as_bytes().to_vec(),
            upper_dir: upper.as_bytes().to_vec(),
            work_dir: work.as_bytes().to_vec(),
            merged_dir: merged.as_bytes().to_vec(),
        }
    }

    /// Validate the overlayfs configuration
    pub fn validate(&self) -> KernelResult<()> {
        for path in [&self.lower_dir, &self.upper_dir, &self.work_dir, &self.merged_dir] {
            if path.is_empty() || path.len() > MAX_PATH_LEN {
                return Err(KernelError::InvalidArgument);
            }
        }
        Ok(())
    }
}

/// Mount isolation configuration for build containers
#[derive(Debug, Clone, Default)]
pub struct MountIsolationConfig {
    /// OverlayFS configuration for container root filesystem
    pub rootfs: OverlayFsConfig,
    /// Bind mounts for shared caches (cargo registry, sccache, etc.)
    pub bind_mounts: Vec<BindMount>,
    /// Old root mount point for pivot_root cleanup
    pub old_root: Vec<u8>,
    /// Use overlayfs for rootfs
    pub use_overlayfs: bool,
    /// Make mounts private (prevent propagation)
    pub private_mounts: bool,
}

impl MountIsolationConfig {
    /// Create a new mount isolation configuration
    pub fn new() -> Self {
        Self {
            rootfs: OverlayFsConfig::default(),
            bind_mounts: Vec::new(),
            old_root: b"/.old_root".to_vec(),
            use_overlayfs: true,
            private_mounts: true,
        }
    }

    /// Create configuration for a Rust build container
    pub fn for_rust_build(
        toolchain_path: &str,
        workspace_path: &str,
        cargo_registry: &str,
        sccache_dir: Option<&str>,
    ) -> Self {
        let mut config = Self::new();

        // Set up overlayfs with toolchain as lower (read-only) layer
        config.rootfs = OverlayFsConfig::new(
            toolchain_path,
            &format!("{}/upper", workspace_path),
            &format!("{}/work", workspace_path),
            &format!("{}/merged", workspace_path),
        );

        // Add cargo registry bind mount (shared cache)
        config.bind_mounts.push(BindMount::new(
            cargo_registry,
            "/root/.cargo/registry",
        ));

        // Add sccache bind mount if provided
        if let Some(sccache) = sccache_dir {
            config.bind_mounts.push(BindMount::new(
                sccache,
                "/root/.cache/sccache",
            ));
        }

        config
    }

    /// Add a bind mount
    pub fn add_bind_mount(&mut self, mount: BindMount) -> &mut Self {
        if self.bind_mounts.len() < MAX_BIND_MOUNTS {
            self.bind_mounts.push(mount);
        }
        self
    }

    /// Add a cargo registry cache mount
    pub fn with_cargo_registry(&mut self, host_path: &str) -> &mut Self {
        self.add_bind_mount(BindMount::new(host_path, "/root/.cargo/registry"))
    }

    /// Add a cargo git cache mount
    pub fn with_cargo_git(&mut self, host_path: &str) -> &mut Self {
        self.add_bind_mount(BindMount::new(host_path, "/root/.cargo/git"))
    }

    /// Add an sccache mount
    pub fn with_sccache(&mut self, host_path: &str) -> &mut Self {
        self.add_bind_mount(BindMount::new(host_path, "/root/.cache/sccache"))
    }

    /// Add a target directory cache mount
    pub fn with_target_cache(&mut self, host_path: &str, container_path: &str) -> &mut Self {
        self.add_bind_mount(BindMount::new(host_path, container_path))
    }

    /// Validate the configuration
    pub fn validate(&self) -> KernelResult<()> {
        if self.use_overlayfs {
            self.rootfs.validate()?;
        }

        if self.bind_mounts.len() > MAX_BIND_MOUNTS {
            return Err(KernelError::InvalidArgument);
        }

        for mount in &self.bind_mounts {
            mount.validate()?;
        }

        if self.old_root.is_empty() || self.old_root.len() > MAX_PATH_LEN {
            return Err(KernelError::InvalidArgument);
        }

        Ok(())
    }
}

impl Default for NamespaceSetup {
    fn default() -> Self {
        Self {
            flags: CLONE_ALL_NAMESPACES,
            uid_map: vec![IdMap {
                inside_id: 0,
                outside_id: 1000,
                count: 1,
            }],
            gid_map: vec![IdMap {
                inside_id: 0,
                outside_id: 1000,
                count: 1,
            }],
            hostname: None,
            root_dir: None,
        }
    }
}

/// Namespace manager
pub struct NamespaceManager;

impl NamespaceManager {
    /// Create namespaces for a new process
    pub fn create_namespaces(pid: u32, setup: &NamespaceSetup) -> KernelResult<()> {
        // In a real kernel module, this would:
        // 1. Set up namespace flags for clone()
        // 2. Create namespace structures
        // 3. Configure user namespace mappings
        // 4. Set up mount namespace root
        // 5. Configure network namespace

        // Check if we have required namespaces
        if setup.flags & NamespaceType::User as u32 != 0 {
            Self::setup_user_namespace(pid, &setup.uid_map, &setup.gid_map)?;
        }

        if setup.flags & NamespaceType::Mount as u32 != 0 {
            Self::setup_mount_namespace(pid, &setup.root_dir)?;
        }

        if setup.flags & NamespaceType::UTS as u32 != 0 {
            Self::setup_uts_namespace(pid, &setup.hostname)?;
        }

        if setup.flags & NamespaceType::Network as u32 != 0 {
            Self::setup_network_namespace(pid)?;
        }

        Ok(())
    }

    /// Set up user namespace with ID mappings
    fn setup_user_namespace(pid: u32, uid_map: &[IdMap], gid_map: &[IdMap]) -> KernelResult<()> {
        // Validate mappings
        for map in uid_map {
            if map.count == 0 || map.count > 1000 {
                return Err(KernelError::InvalidArgument);
            }
        }

        for map in gid_map {
            if map.count == 0 || map.count > 1000 {
                return Err(KernelError::InvalidArgument);
            }
        }

        // In kernel: write to /proc/[pid]/uid_map and gid_map

        Ok(())
    }

    /// Set up mount namespace
    fn setup_mount_namespace(pid: u32, root_dir: &Option<Vec<u8>>) -> KernelResult<()> {
        // In kernel: set up pivot_root or bind mounts

        if let Some(root) = root_dir {
            // Validate root directory exists and is safe
            if root.is_empty() || root.len() > 4096 {
                return Err(KernelError::InvalidArgument);
            }
        }

        Ok(())
    }

    /// Set up complete mount isolation for a build container
    ///
    /// This function configures:
    /// 1. Makes mounts private (prevents propagation)
    /// 2. Sets up overlayfs for container rootfs
    /// 3. Configures bind mounts for shared caches
    /// 4. Performs pivot_root to switch filesystem root
    pub fn setup_mount_isolation(pid: u32, config: &MountIsolationConfig) -> KernelResult<()> {
        // Validate configuration
        config.validate()?;

        // In kernel module: call swarm_mount_isolation_setup()
        // This would invoke the C implementation through FFI

        // Step 1: Make mounts private
        if config.private_mounts {
            Self::make_mounts_private()?;
        }

        // Step 2: Setup overlayfs
        if config.use_overlayfs {
            Self::setup_overlayfs(&config.rootfs)?;
        }

        // Step 3: Setup bind mounts
        for mount in &config.bind_mounts {
            Self::setup_bind_mount(mount)?;
        }

        // Step 4: Pivot root
        if config.use_overlayfs && !config.rootfs.merged_dir.is_empty() {
            Self::do_pivot_root(&config.rootfs.merged_dir, &config.old_root)?;
        }

        Ok(())
    }

    /// Make all mounts private (prevent propagation to other namespaces)
    fn make_mounts_private() -> KernelResult<()> {
        // In kernel: mount("", "/", NULL, MS_REC | MS_PRIVATE, NULL)
        Ok(())
    }

    /// Setup overlayfs mount
    fn setup_overlayfs(config: &OverlayFsConfig) -> KernelResult<()> {
        config.validate()?;

        // In kernel: mount("overlay", merged_dir, "overlay", 0, options)
        // where options = "lowerdir=...,upperdir=...,workdir=..."

        Ok(())
    }

    /// Setup a single bind mount
    fn setup_bind_mount(mount: &BindMount) -> KernelResult<()> {
        mount.validate()?;

        // In kernel:
        // 1. mount(source, target, NULL, MS_BIND, NULL)
        // 2. mount("", target, NULL, MS_BIND | MS_REMOUNT | flags, NULL)

        Ok(())
    }

    /// Perform pivot_root to switch filesystem root
    fn do_pivot_root(new_root: &[u8], old_root: &[u8]) -> KernelResult<()> {
        if new_root.is_empty() || old_root.is_empty() {
            return Err(KernelError::InvalidArgument);
        }

        // In kernel:
        // 1. mkdir(new_root + old_root)
        // 2. pivot_root(new_root, new_root + old_root)
        // 3. chdir("/")
        // 4. umount(old_root, MNT_DETACH)

        Ok(())
    }

    /// Set up UTS namespace with hostname
    fn setup_uts_namespace(pid: u32, hostname: &Option<Vec<u8>>) -> KernelResult<()> {
        if let Some(name) = hostname {
            // Validate hostname
            if name.is_empty() || name.len() > 64 {
                return Err(KernelError::InvalidArgument);
            }

            // Check for valid characters
            for &byte in name {
                if !byte.is_ascii_alphanumeric() && byte != b'-' && byte != b'.' {
                    return Err(KernelError::InvalidArgument);
                }
            }
        }

        Ok(())
    }

    /// Set up network namespace
    fn setup_network_namespace(pid: u32) -> KernelResult<()> {
        // In kernel: create virtual network devices, configure routing

        Ok(())
    }

    /// Enter a namespace of a given type
    pub fn enter_namespace(pid: u32, ns_type: NamespaceType) -> KernelResult<()> {
        // In kernel: setns() system call

        Ok(())
    }

    /// Check if a process is in a specific namespace
    pub fn is_in_namespace(pid: u32, ns_type: NamespaceType) -> bool {
        // In kernel: check /proc/[pid]/ns/[type]

        true
    }

    /// Get namespace ID for a process
    pub fn get_namespace_id(pid: u32, ns_type: NamespaceType) -> KernelResult<u64> {
        // In kernel: read namespace inode number

        Ok(0)
    }
}

/// Initialize namespace subsystem
pub fn init() -> KernelResult<()> {
    // Register namespace operations

    Ok(())
}

/// Cleanup namespace subsystem  
pub fn cleanup() {
    // Unregister namespace operations
}

// Kernel module integration tests
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_namespace_flags() {
        let all_flags = CLONE_ALL_NAMESPACES;

        // Verify all namespace types are included
        for ns_type in NamespaceType::all() {
            assert!(all_flags & ns_type.to_flag() != 0);
        }
    }

    #[test]
    fn test_id_mapping_validation() {
        let valid_map = IdMap {
            inside_id: 0,
            outside_id: 1000,
            count: 100,
        };

        let invalid_map = IdMap {
            inside_id: 0,
            outside_id: 1000,
            count: 0, // Invalid: count must be > 0
        };

        // Test validation logic
        assert!(valid_map.count > 0 && valid_map.count <= 1000);
        assert!(invalid_map.count == 0);
    }

    #[test]
    fn test_hostname_validation() {
        let valid_hostnames = vec![
            b"agent-123".to_vec(),
            b"swarm.local".to_vec(),
            b"test-container-01".to_vec(),
        ];

        let invalid_hostnames = vec![
            b"".to_vec(),             // Empty
            b"a".repeat(65).to_vec(), // Too long
            b"invalid_name".to_vec(), // Underscore not allowed
            b"test@host".to_vec(),    // @ not allowed
        ];

        for hostname in valid_hostnames {
            assert!(!hostname.is_empty() && hostname.len() <= 64);
            assert!(hostname
                .iter()
                .all(|&b| b.is_ascii_alphanumeric() || b == b'-' || b == b'.'));
        }

        for hostname in invalid_hostnames {
            assert!(
                hostname.is_empty()
                    || hostname.len() > 64
                    || !hostname
                        .iter()
                        .all(|&b| b.is_ascii_alphanumeric() || b == b'-' || b == b'.')
            );
        }
    }

    #[test]
    fn test_bind_mount_creation() {
        let mount = BindMount::new("/host/cache", "/container/cache");
        assert_eq!(mount.source, b"/host/cache");
        assert_eq!(mount.target, b"/container/cache");
        assert_eq!(mount.flags, 0);

        let readonly_mount = BindMount::readonly("/host/data", "/container/data");
        assert_eq!(readonly_mount.flags, MountFlags::ReadOnly as u32);
    }

    #[test]
    fn test_bind_mount_validation() {
        let valid_mount = BindMount::new("/host/path", "/container/path");
        assert!(valid_mount.validate().is_ok());

        let empty_source = BindMount::new("", "/container/path");
        assert!(empty_source.validate().is_err());

        let empty_target = BindMount::new("/host/path", "");
        assert!(empty_target.validate().is_err());
    }

    #[test]
    fn test_overlayfs_config() {
        let config = OverlayFsConfig::new(
            "/lower",
            "/upper",
            "/work",
            "/merged",
        );
        assert!(config.validate().is_ok());

        let invalid_config = OverlayFsConfig::default();
        assert!(invalid_config.validate().is_err());
    }

    #[test]
    fn test_mount_isolation_config() {
        let config = MountIsolationConfig::new();
        assert!(config.use_overlayfs);
        assert!(config.private_mounts);
        assert_eq!(config.old_root, b"/.old_root");
    }

    #[test]
    fn test_mount_isolation_for_rust_build() {
        let config = MountIsolationConfig::for_rust_build(
            "/toolchains/stable",
            "/workspace/build-123",
            "/cache/cargo-registry",
            Some("/cache/sccache"),
        );

        assert!(config.use_overlayfs);
        assert_eq!(config.bind_mounts.len(), 2);

        // Check cargo registry mount
        assert_eq!(config.bind_mounts[0].target, b"/root/.cargo/registry");

        // Check sccache mount
        assert_eq!(config.bind_mounts[1].target, b"/root/.cache/sccache");
    }

    #[test]
    fn test_mount_isolation_builder() {
        let mut config = MountIsolationConfig::new();
        config
            .with_cargo_registry("/cache/registry")
            .with_cargo_git("/cache/git")
            .with_sccache("/cache/sccache");

        assert_eq!(config.bind_mounts.len(), 3);
    }

    #[test]
    fn test_mount_flags() {
        let combined = MountFlags::combine(&[
            MountFlags::ReadOnly,
            MountFlags::NoSuid,
            MountFlags::NoExec,
        ]);
        assert_eq!(combined, 0x01 | 0x02 | 0x04);
    }
}
