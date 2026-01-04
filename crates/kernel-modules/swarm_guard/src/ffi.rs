//! FFI bridge between kernel C code and Rust implementation
//!
//! This module provides the C-compatible interface for the kernel module.

use alloc::boxed::Box;
use alloc::ffi::CString;
use core::ffi::{c_char, c_int, c_uint};
use core::ptr;

use crate::{cleanup, init, KernelResult};

// ==================== Mount Isolation FFI Structures ====================

/// Maximum path length for mount paths (matches C MAX_PATH_LEN)
pub const MAX_PATH_LEN: usize = 256;

/// Maximum number of bind mounts (matches C MAX_BIND_MOUNTS)
pub const MAX_BIND_MOUNTS: usize = 16;

/// Mount flags for bind mounts
pub mod mount_flags {
    pub const SWARM_MOUNT_READONLY: u32 = 0x01;
    pub const SWARM_MOUNT_NOSUID: u32 = 0x02;
    pub const SWARM_MOUNT_NOEXEC: u32 = 0x04;
    pub const SWARM_MOUNT_NODEV: u32 = 0x08;
}

/// Bind mount configuration (C-compatible)
#[repr(C)]
#[derive(Clone)]
pub struct SwarmBindMount {
    /// Host path to mount from
    pub source: [c_char; MAX_PATH_LEN],
    /// Container path to mount to
    pub target: [c_char; MAX_PATH_LEN],
    /// SWARM_MOUNT_* flags
    pub flags: c_uint,
}

impl Default for SwarmBindMount {
    fn default() -> Self {
        Self {
            source: [0; MAX_PATH_LEN],
            target: [0; MAX_PATH_LEN],
            flags: 0,
        }
    }
}

impl SwarmBindMount {
    /// Create a new bind mount from source and target paths
    pub fn new(source: &str, target: &str, flags: u32) -> Self {
        let mut mount = Self::default();
        copy_str_to_array(source, &mut mount.source);
        copy_str_to_array(target, &mut mount.target);
        mount.flags = flags;
        mount
    }
}

/// OverlayFS configuration for container rootfs (C-compatible)
#[repr(C)]
#[derive(Clone)]
pub struct SwarmOverlayFsConfig {
    /// Read-only base layer (e.g., toolchain)
    pub lower_dir: [c_char; MAX_PATH_LEN],
    /// Writable layer for changes
    pub upper_dir: [c_char; MAX_PATH_LEN],
    /// OverlayFS work directory
    pub work_dir: [c_char; MAX_PATH_LEN],
    /// Final merged mount point
    pub merged_dir: [c_char; MAX_PATH_LEN],
}

impl Default for SwarmOverlayFsConfig {
    fn default() -> Self {
        Self {
            lower_dir: [0; MAX_PATH_LEN],
            upper_dir: [0; MAX_PATH_LEN],
            work_dir: [0; MAX_PATH_LEN],
            merged_dir: [0; MAX_PATH_LEN],
        }
    }
}

impl SwarmOverlayFsConfig {
    /// Create a new overlayfs config
    pub fn new(lower: &str, upper: &str, work: &str, merged: &str) -> Self {
        let mut config = Self::default();
        copy_str_to_array(lower, &mut config.lower_dir);
        copy_str_to_array(upper, &mut config.upper_dir);
        copy_str_to_array(work, &mut config.work_dir);
        copy_str_to_array(merged, &mut config.merged_dir);
        config
    }
}

/// Mount isolation configuration for build containers (C-compatible)
#[repr(C)]
pub struct SwarmMountIsolationConfig {
    /// OverlayFS for container root filesystem
    pub rootfs: SwarmOverlayFsConfig,
    /// Bind mounts for shared caches
    pub bind_mounts: [SwarmBindMount; MAX_BIND_MOUNTS],
    /// Number of bind mounts
    pub num_bind_mounts: c_uint,
    /// Old root mount point for pivot_root cleanup
    pub old_root: [c_char; MAX_PATH_LEN],
    /// Use overlayfs for rootfs
    pub use_overlayfs: c_uint,
    /// Make mounts private (MS_PRIVATE)
    pub private_mounts: c_uint,
}

impl Default for SwarmMountIsolationConfig {
    fn default() -> Self {
        Self {
            rootfs: SwarmOverlayFsConfig::default(),
            bind_mounts: core::array::from_fn(|_| SwarmBindMount::default()),
            num_bind_mounts: 0,
            old_root: [0; MAX_PATH_LEN],
            use_overlayfs: 0,
            private_mounts: 1, // Default to private mounts
        }
    }
}

impl SwarmMountIsolationConfig {
    /// Set the old_root path
    pub fn set_old_root(&mut self, path: &str) {
        copy_str_to_array(path, &mut self.old_root);
    }

    /// Add a bind mount
    pub fn add_bind_mount(&mut self, source: &str, target: &str, flags: u32) -> bool {
        if self.num_bind_mounts as usize >= MAX_BIND_MOUNTS {
            return false;
        }
        self.bind_mounts[self.num_bind_mounts as usize] =
            SwarmBindMount::new(source, target, flags);
        self.num_bind_mounts += 1;
        true
    }
}

/// Helper to copy a string into a fixed-size char array
fn copy_str_to_array(s: &str, arr: &mut [c_char]) {
    let bytes = s.as_bytes();
    let len = bytes.len().min(arr.len() - 1);
    for (i, &b) in bytes[..len].iter().enumerate() {
        arr[i] = b as c_char;
    }
    arr[len] = 0; // Null terminate
}

// ==================== External C Functions (Kernel Module) ====================

extern "C" {
    /// Setup mount isolation for an agent
    pub fn swarm_mount_isolation_setup(
        agent_id: u64,
        config: *const SwarmMountIsolationConfig,
    ) -> c_int;

    /// Teardown mount isolation for an agent
    pub fn swarm_mount_isolation_teardown(agent_id: u64) -> c_int;

    /// Add a bind mount to an existing agent
    pub fn swarm_mount_add_bind(agent_id: u64, mount: *const SwarmBindMount) -> c_int;

    /// Setup overlayfs for an agent
    pub fn swarm_mount_setup_overlayfs(agent_id: u64, config: *const SwarmOverlayFsConfig)
        -> c_int;

    /// Perform pivot_root for an agent
    pub fn swarm_mount_pivot_root(
        agent_id: u64,
        new_root: *const c_char,
        old_root: *const c_char,
    ) -> c_int;
}

// ==================== Safe Rust Wrappers ====================

/// Result type for mount operations
pub type MountResult<T> = Result<T, MountError>;

/// Mount operation errors
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MountError {
    /// Permission denied
    PermissionDenied,
    /// Invalid argument
    InvalidArgument,
    /// No such file or directory
    NotFound,
    /// I/O error
    IoError,
    /// Operation not supported
    NotSupported,
    /// Agent not found
    AgentNotFound,
    /// Unknown error with errno
    Unknown(i32),
}

impl MountError {
    /// Convert errno to MountError
    pub fn from_errno(errno: i32) -> Self {
        match errno {
            -1 => MountError::PermissionDenied,
            -2 => MountError::NotFound,
            -5 => MountError::IoError,
            -22 => MountError::InvalidArgument,
            -95 => MountError::NotSupported,
            -3 => MountError::AgentNotFound, // ESRCH
            e => MountError::Unknown(e),
        }
    }
}

/// Safe wrapper for mount isolation setup
pub fn setup_mount_isolation(agent_id: u64, config: &SwarmMountIsolationConfig) -> MountResult<()> {
    // SAFETY: config is a valid reference to SwarmMountIsolationConfig. The kernel
    // function reads the config during this call and does not retain the pointer.
    let ret = unsafe { swarm_mount_isolation_setup(agent_id, config as *const _) };
    if ret == 0 {
        Ok(())
    } else {
        Err(MountError::from_errno(ret))
    }
}

/// Safe wrapper for mount isolation teardown
pub fn teardown_mount_isolation(agent_id: u64) -> MountResult<()> {
    // SAFETY: This is a simple kernel FFI call with no pointer arguments.
    // The kernel function cleans up resources associated with agent_id.
    let ret = unsafe { swarm_mount_isolation_teardown(agent_id) };
    if ret == 0 {
        Ok(())
    } else {
        Err(MountError::from_errno(ret))
    }
}

/// Safe wrapper for adding a bind mount
pub fn add_bind_mount(agent_id: u64, mount: &SwarmBindMount) -> MountResult<()> {
    // SAFETY: mount is a valid reference to SwarmBindMount. The kernel function
    // reads the mount config during this call and does not retain the pointer.
    let ret = unsafe { swarm_mount_add_bind(agent_id, mount as *const _) };
    if ret == 0 {
        Ok(())
    } else {
        Err(MountError::from_errno(ret))
    }
}

/// Safe wrapper for overlayfs setup
pub fn setup_overlayfs(agent_id: u64, config: &SwarmOverlayFsConfig) -> MountResult<()> {
    // SAFETY: config is a valid reference to SwarmOverlayFsConfig. The kernel
    // function reads the config during this call and does not retain the pointer.
    let ret = unsafe { swarm_mount_setup_overlayfs(agent_id, config as *const _) };
    if ret == 0 {
        Ok(())
    } else {
        Err(MountError::from_errno(ret))
    }
}

/// Safe wrapper for pivot_root
pub fn pivot_root(agent_id: u64, new_root: &str, old_root: &str) -> MountResult<()> {
    let mut new_root_buf = [0i8; MAX_PATH_LEN];
    let mut old_root_buf = [0i8; MAX_PATH_LEN];
    copy_str_to_array(new_root, &mut new_root_buf);
    copy_str_to_array(old_root, &mut old_root_buf);

    // SAFETY: new_root_buf and old_root_buf are stack-allocated null-terminated
    // strings. The kernel function reads these during the call and does not
    // retain the pointers. Both buffers are valid for the duration of the call.
    let ret =
        unsafe { swarm_mount_pivot_root(agent_id, new_root_buf.as_ptr(), old_root_buf.as_ptr()) };
    if ret == 0 {
        Ok(())
    } else {
        Err(MountError::from_errno(ret))
    }
}

// ==================== End Mount Isolation FFI ====================

/// Initialize SwarmGuard subsystems
#[no_mangle]
pub extern "C" fn swarm_guard_init() -> c_int {
    match init() {
        Ok(()) => 0,
        Err(_) => -1,
    }
}

/// Cleanup SwarmGuard subsystems
#[no_mangle]
pub extern "C" fn swarm_guard_cleanup() {
    cleanup();
}

/// Get status string for /proc/swarm/status
#[no_mangle]
pub extern "C" fn swarm_guard_get_status() -> *mut c_char {
    match crate::proc::ProcOps::read(crate::proc::ProcFile::Status, 0, &mut []) {
        Ok(content) => {
            // In real kernel, would use kernel allocator
            if let Ok(cstring) = CString::new(content) {
                cstring.into_raw()
            } else {
                ptr::null_mut()
            }
        }
        Err(_) => ptr::null_mut(),
    }
}

/// Create agent from JSON configuration
#[no_mangle]
pub extern "C" fn swarm_guard_create_agent(config: *const c_char) -> c_int {
    if config.is_null() {
        return -22; // -EINVAL
    }

    // Convert C string to Rust string
    // SAFETY: config is non-null (checked above). libc::strlen finds the null
    // terminator, ensuring len bytes are valid. from_raw_parts creates a slice
    // of exactly len bytes which is then validated as UTF-8.
    let config_str = unsafe {
        let len = libc::strlen(config);
        let slice = core::slice::from_raw_parts(config as *const u8, len);
        match core::str::from_utf8(slice) {
            Ok(s) => s,
            Err(_) => return -22, // -EINVAL
        }
    };

    // Parse and create agent
    match create_agent_from_config(config_str) {
        Ok(()) => 0,
        Err(e) => kernel_error_to_errno(e),
    }
}

/// Create agent from configuration string
fn create_agent_from_config(config: &str) -> KernelResult<()> {
    // Parse JSON configuration
    // Create agent with policy
    // Add to registry

    Ok(())
}

/// Convert kernel error to errno
fn kernel_error_to_errno(error: crate::KernelError) -> c_int {
    use crate::KernelError;

    match error {
        KernelError::PermissionDenied => -1,       // -EPERM
        KernelError::ResourceLimitExceeded => -12, // -ENOMEM
        KernelError::InvalidAgent => -22,          // -EINVAL
        KernelError::NamespaceError => -22,        // -EINVAL
        KernelError::CgroupError => -5,            // -EIO
        KernelError::SyscallError => -14,          // -EFAULT
        KernelError::OutOfMemory => -12,           // -ENOMEM
        KernelError::InvalidArgument => -22,       // -EINVAL
    }
}

// Mock libc functions for kernel environment
mod libc {
    use core::ffi::c_char;

    /// Calculate length of a null-terminated C string
    ///
    /// # Safety
    /// - `s` must be a valid pointer to a null-terminated C string
    /// - The memory from `s` to the null terminator must be readable
    pub unsafe fn strlen(s: *const c_char) -> usize {
        let mut len = 0;
        let mut ptr = s;
        // SAFETY: Caller guarantees s points to a null-terminated string.
        // We iterate until finding null, incrementing ptr within valid memory.
        while *ptr != 0 {
            len += 1;
            ptr = ptr.add(1);
        }
        len
    }
}
