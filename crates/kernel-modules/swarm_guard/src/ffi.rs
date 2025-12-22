//! FFI bridge between kernel C code and Rust implementation
//!
//! This module provides the C-compatible interface for the kernel module.

use alloc::boxed::Box;
use alloc::ffi::CString;
use core::ffi::{c_char, c_int};
use core::ptr;

use crate::{cleanup, init, KernelResult};

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

    pub unsafe fn strlen(s: *const c_char) -> usize {
        let mut len = 0;
        let mut ptr = s;
        while *ptr != 0 {
            len += 1;
            ptr = ptr.add(1);
        }
        len
    }
}
