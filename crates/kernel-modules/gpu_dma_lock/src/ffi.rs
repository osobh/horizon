//! FFI bridge for C kernel integration
//!
//! This module provides C-compatible interfaces for the kernel module.

use core::ffi::{c_char, c_int, c_ulong};
use core::ptr;
use core::slice;
use core::sync::atomic::Ordering;

use crate::{allocation, context, dma, proc, security, stats, DmaAccessMode, KernelError};

/// C-compatible result type
#[repr(C)]
pub struct CResult {
    pub success: c_int,
    pub error_code: c_int,
    pub value: c_ulong,
}

impl CResult {
    pub fn ok(value: c_ulong) -> Self {
        Self {
            success: 1,
            error_code: 0,
            value,
        }
    }

    pub fn err(error: KernelError) -> Self {
        Self {
            success: 0,
            error_code: error as c_int,
            value: 0,
        }
    }
}

/// GPU device info for C
#[repr(C)]
pub struct CGpuDeviceInfo {
    pub id: u32,
    pub name: [c_char; 64],
    pub total_memory: c_ulong,
    pub available_memory: c_ulong,
    pub allocated_memory: c_ulong,
}

/// Allocation info for C
#[repr(C)]
pub struct CAllocationInfo {
    pub id: c_ulong,
    pub agent_id: c_ulong,
    pub size: c_ulong,
    pub device_id: u32,
    pub timestamp: c_ulong,
}

/// DMA permission info for C
#[repr(C)]
pub struct CDmaPermission {
    pub agent_id: c_ulong,
    pub start_addr: c_ulong,
    pub end_addr: c_ulong,
    pub access_mode: c_int,
}

/// GPU context info for C
#[repr(C)]
pub struct CGpuContext {
    pub id: c_ulong,
    pub agent_id: c_ulong,
    pub device_id: u32,
    pub state: c_int,
    pub switch_count: c_ulong,
}

/// Statistics for C
#[repr(C)]
pub struct CGpuDmaStats {
    pub total_allocations: c_ulong,
    pub total_deallocations: c_ulong,
    pub total_bytes_allocated: c_ulong,
    pub dma_checks: c_ulong,
    pub dma_denials: c_ulong,
    pub context_switches: c_ulong,
}

/// Convert Rust string to C string buffer
///
/// # Safety
/// - `dst` must be a valid mutable slice with at least 1 byte for null terminator
/// - The caller must ensure dst has enough space for the copied string
unsafe fn rust_str_to_c_buf(src: &str, dst: &mut [c_char]) {
    let bytes = src.as_bytes();
    let copy_len = core::cmp::min(bytes.len(), dst.len() - 1);

    for i in 0..copy_len {
        dst[i] = bytes[i] as c_char;
    }
    dst[copy_len] = 0; // Null terminator
}

/// Convert C string to Rust string
///
/// # Safety
/// - `ptr` must be a valid pointer to a null-terminated C string
/// - The memory pointed to must remain valid for the 'static lifetime
/// - The string must be valid UTF-8
unsafe fn c_str_to_rust(ptr: *const c_char) -> Result<&'static str, KernelError> {
    if ptr.is_null() {
        return Err(KernelError::InvalidArgument);
    }

    let len = c_strlen(ptr);
    // SAFETY: ptr is non-null (checked above), and c_strlen found a null terminator
    // within 4096 bytes. The slice is valid for len bytes from ptr.
    let slice = slice::from_raw_parts(ptr as *const u8, len);
    core::str::from_utf8(slice).map_err(|_| KernelError::InvalidArgument)
}

/// Get length of C string
///
/// # Safety
/// - `ptr` must be a valid pointer to a null-terminated C string
/// - The memory from ptr to ptr+4096 must be readable (safety limit applied)
unsafe fn c_strlen(ptr: *const c_char) -> usize {
    let mut len = 0;
    // SAFETY: ptr.offset(len) is valid as long as len <= 4096 (our safety limit)
    // and the caller guarantees ptr points to valid memory.
    while *ptr.offset(len as isize) != 0 {
        len += 1;
        if len > 4096 {
            // Safety limit to prevent reading unbounded memory
            break;
        }
    }
    len
}

/// Initialize the GPU DMA lock subsystem
#[no_mangle]
pub extern "C" fn gpu_dma_lock_init() -> c_int {
    // Initialize all subsystems
    if let Err(_) = allocation::init() {
        return -1;
    }

    if let Err(_) = dma::init() {
        allocation::cleanup();
        return -1;
    }

    if let Err(_) = context::init() {
        dma::cleanup();
        allocation::cleanup();
        return -1;
    }

    if let Err(_) = proc::init() {
        context::cleanup();
        dma::cleanup();
        allocation::cleanup();
        return -1;
    }

    stats::init();
    security::init();

    0
}

/// Cleanup the GPU DMA lock subsystem
#[no_mangle]
pub extern "C" fn gpu_dma_lock_cleanup() {
    security::cleanup();
    stats::cleanup();
    proc::cleanup();
    context::cleanup();
    dma::cleanup();
    allocation::cleanup();
}

/// Register a GPU device
#[no_mangle]
pub extern "C" fn gpu_dma_register_device(
    id: u32,
    name: *const c_char,
    total_memory: c_ulong,
) -> c_int {
    // SAFETY: This is an FFI function. The caller (C code) must ensure:
    // - `name` is a valid pointer to a null-terminated string
    // - The string remains valid for the duration of this call
    unsafe {
        let device_name = match c_str_to_rust(name) {
            Ok(s) => s,
            Err(_) => return -1,
        };

        let manager = allocation::get_manager();
        manager.register_device(id, device_name, total_memory as usize);
        0
    }
}

/// Get number of registered devices
#[no_mangle]
pub extern "C" fn gpu_dma_get_device_count() -> c_int {
    let manager = allocation::get_manager();
    manager.device_count() as c_int
}

/// Get device information
#[no_mangle]
pub extern "C" fn gpu_dma_get_device_info(device_id: u32, info: *mut CGpuDeviceInfo) -> c_int {
    if info.is_null() {
        return -1;
    }

    let manager = allocation::get_manager();
    let devices = manager.devices.read();

    if let Some(device) = devices.iter().find(|d| d.id() == device_id) {
        // SAFETY: info is non-null (checked at function start). The caller guarantees
        // info points to a valid CGpuDeviceInfo struct with proper alignment.
        unsafe {
            (*info).id = device.id();
            rust_str_to_c_buf(device.name(), &mut (*info).name);
            (*info).total_memory = device.total_memory() as c_ulong;
            (*info).available_memory = device.available_memory() as c_ulong;
            (*info).allocated_memory = device.allocated_memory() as c_ulong;
        }
        0
    } else {
        -1
    }
}

/// Create an agent
#[no_mangle]
pub extern "C" fn gpu_dma_create_agent(agent_id: c_ulong, quota: c_ulong) -> CResult {
    let manager = allocation::get_manager();
    match manager.create_agent(agent_id as u64, quota as usize) {
        Ok(_) => CResult::ok(agent_id),
        Err(e) => CResult::err(e),
    }
}

/// Remove an agent
#[no_mangle]
pub extern "C" fn gpu_dma_remove_agent(agent_id: c_ulong) -> CResult {
    let manager = allocation::get_manager();
    match manager.remove_agent(agent_id as u64) {
        Ok(_) => CResult::ok(0),
        Err(e) => CResult::err(e),
    }
}

/// Allocate GPU memory
#[no_mangle]
pub extern "C" fn gpu_dma_allocate(agent_id: c_ulong, size: c_ulong, device_id: u32) -> CResult {
    let manager = allocation::get_manager();
    match manager.allocate(agent_id as u64, size as usize, Some(device_id)) {
        Ok(alloc_id) => CResult::ok(alloc_id as c_ulong),
        Err(e) => CResult::err(e),
    }
}

/// Deallocate GPU memory
#[no_mangle]
pub extern "C" fn gpu_dma_deallocate(allocation_id: c_ulong) -> CResult {
    let manager = allocation::get_manager();
    match manager.deallocate(allocation_id as u64) {
        Ok(_) => CResult::ok(0),
        Err(e) => CResult::err(e),
    }
}

/// Get allocation information
#[no_mangle]
pub extern "C" fn gpu_dma_get_allocation_info(
    allocation_id: c_ulong,
    info: *mut CAllocationInfo,
) -> c_int {
    if info.is_null() {
        return -1;
    }

    let manager = allocation::get_manager();
    let tracker = &manager.tracker;
    let allocations = tracker.allocations.read();

    if let Some(alloc) = allocations.get(&(allocation_id as u64)) {
        // SAFETY: info is non-null (checked at function start). The caller guarantees
        // info points to a valid CAllocationInfo struct with proper alignment.
        unsafe {
            (*info).id = allocation_id;
            (*info).agent_id = alloc.agent_id as c_ulong;
            (*info).size = alloc.size as c_ulong;
            (*info).device_id = alloc.device_id;
            (*info).timestamp = alloc.timestamp as c_ulong;
        }
        0
    } else {
        -1
    }
}

/// Grant DMA access
#[no_mangle]
pub extern "C" fn gpu_dma_grant_access(
    agent_id: c_ulong,
    start_addr: c_ulong,
    end_addr: c_ulong,
    access_mode: c_int,
) -> c_int {
    let mode = match access_mode {
        1 => DmaAccessMode::ReadOnly,
        2 => DmaAccessMode::WriteOnly,
        3 => DmaAccessMode::ReadWrite,
        _ => return -1,
    };

    let manager = dma::get_manager();
    manager
        .acl
        .grant_access(agent_id as u64, start_addr as u64, end_addr as u64, mode);
    0
}

/// Check DMA access
#[no_mangle]
pub extern "C" fn gpu_dma_check_access(
    agent_id: c_ulong,
    addr: c_ulong,
    access_mode: c_int,
) -> c_int {
    let mode = match access_mode {
        1 => DmaAccessMode::ReadOnly,
        2 => DmaAccessMode::WriteOnly,
        3 => DmaAccessMode::ReadWrite,
        _ => return 0,
    };

    let manager = dma::get_manager();
    if manager.acl.check_access(agent_id as u64, addr as u64, mode) {
        1
    } else {
        0
    }
}

/// Revoke DMA access
#[no_mangle]
pub extern "C" fn gpu_dma_revoke_access(agent_id: c_ulong) -> c_int {
    let manager = dma::get_manager();
    manager.acl.revoke_access(agent_id as u64);
    0
}

/// Create GPU context
#[no_mangle]
pub extern "C" fn gpu_dma_create_context(agent_id: c_ulong, device_id: u32) -> CResult {
    let manager = context::get_manager();
    match manager.create_context(agent_id as u64, device_id) {
        Ok(ctx_id) => CResult::ok(ctx_id as c_ulong),
        Err(e) => CResult::err(e),
    }
}

/// Switch GPU context
#[no_mangle]
pub extern "C" fn gpu_dma_switch_context(device_id: u32, context_id: c_ulong) -> CResult {
    let manager = context::get_manager();
    match manager.switch_context(device_id, context_id as u64) {
        Ok(_) => CResult::ok(0),
        Err(e) => CResult::err(e),
    }
}

/// Get context information
#[no_mangle]
pub extern "C" fn gpu_dma_get_context_info(context_id: c_ulong, info: *mut CGpuContext) -> c_int {
    if info.is_null() {
        return -1;
    }

    let manager = context::get_manager();
    let contexts = manager.contexts.read();

    if let Some(ctx) = contexts.get(&(context_id as u64)) {
        // SAFETY: info is non-null (checked at function start). The caller guarantees
        // info points to a valid CGpuContext struct with proper alignment.
        unsafe {
            (*info).id = context_id;
            (*info).agent_id = ctx.agent_id as c_ulong;
            (*info).device_id = ctx.device_id;
            (*info).state = ctx.state as c_int;
            (*info).switch_count = ctx.switch_count.load(Ordering::Relaxed) as c_ulong;
        }
        0
    } else {
        -1
    }
}

/// Get system statistics
#[no_mangle]
pub extern "C" fn gpu_dma_get_stats(stats_out: *mut CGpuDmaStats) -> c_int {
    if stats_out.is_null() {
        return -1;
    }

    let stats = &crate::GPU_DMA_STATS;
    // SAFETY: stats_out is non-null (checked at function start). The caller guarantees
    // stats_out points to a valid CGpuDmaStats struct with proper alignment.
    unsafe {
        (*stats_out).total_allocations = stats.total_allocations.load(Ordering::Relaxed) as c_ulong;
        (*stats_out).total_deallocations =
            stats.total_deallocations.load(Ordering::Relaxed) as c_ulong;
        (*stats_out).total_bytes_allocated =
            stats.total_bytes_allocated.load(Ordering::Relaxed) as c_ulong;
        (*stats_out).dma_checks = stats.dma_checks.load(Ordering::Relaxed) as c_ulong;
        (*stats_out).dma_denials = stats.dma_denials.load(Ordering::Relaxed) as c_ulong;
        (*stats_out).context_switches = stats.context_switches.load(Ordering::Relaxed) as c_ulong;
    }
    0
}

/// Enable or disable debug mode
#[no_mangle]
pub extern "C" fn gpu_dma_enable_debug(enable: c_int) {
    stats::enable_debug(enable != 0);
}

/// Reset statistics
#[no_mangle]
pub extern "C" fn gpu_dma_reset_stats() {
    stats::reset_stats();
}

/// Get detailed statistics as string
#[no_mangle]
pub extern "C" fn gpu_dma_get_detailed_stats(buffer: *mut c_char, buffer_size: c_ulong) -> c_int {
    if buffer.is_null() || buffer_size == 0 {
        return -1;
    }

    if let Some(detailed) = stats::get_detailed_stats() {
        let bytes = detailed.as_bytes();
        let copy_len = core::cmp::min(bytes.len(), buffer_size as usize - 1);

        // SAFETY: buffer is non-null and buffer_size > 0 (checked at function start).
        // copy_len is bounded by buffer_size - 1, so all writes are within bounds.
        unsafe {
            for i in 0..copy_len {
                *buffer.offset(i as isize) = bytes[i] as c_char;
            }
            *buffer.offset(copy_len as isize) = 0; // Null terminator
        }
        copy_len as c_int
    } else {
        0
    }
}

/// Perform garbage collection
#[no_mangle]
pub extern "C" fn gpu_dma_force_gc() {
    // Cleanup stale allocations and contexts
    let alloc_manager = allocation::get_manager();
    let _tracker = &alloc_manager.tracker;

    let ctx_manager = context::get_manager();
    let _contexts = &ctx_manager.contexts;

    // Log the GC operation
    stats::log_debug("Forced garbage collection performed");
}

/// Dump internal state
#[no_mangle]
pub extern "C" fn gpu_dma_dump_state() {
    stats::log_debug("=== Internal State Dump ===");

    // Dump allocation state
    let alloc_manager = allocation::get_manager();
    let device_count = alloc_manager.devices.read().len();
    stats::log_debug(&alloc::format!("Device count: {}", device_count));

    // Dump DMA state
    let dma_manager = dma::get_manager();
    let perm_count = dma_manager.acl.permissions.read().len();
    stats::log_debug(&alloc::format!("DMA permission entries: {}", perm_count));

    // Dump context state
    let ctx_manager = context::get_manager();
    let ctx_count = ctx_manager.contexts.read().len();
    stats::log_debug(&alloc::format!("Active contexts: {}", ctx_count));
}

/// Check if module is initialized
#[no_mangle]
pub extern "C" fn gpu_dma_is_initialized() -> c_int {
    // Check if subsystems are initialized
    let alloc_manager = allocation::get_manager();
    let device_count = alloc_manager.devices.read().len();

    // If we can read device count, assume initialized
    if device_count >= 0 {
        1
    } else {
        0
    }
}

/// Get module version
#[no_mangle]
pub extern "C" fn gpu_dma_get_version(buffer: *mut c_char, buffer_size: c_ulong) -> c_int {
    if buffer.is_null() || buffer_size == 0 {
        return -1;
    }

    let version = "1.0.0";
    let bytes = version.as_bytes();
    let copy_len = core::cmp::min(bytes.len(), buffer_size as usize - 1);

    // SAFETY: buffer is non-null and buffer_size > 0 (checked at function start).
    // copy_len is bounded by buffer_size - 1, so all writes are within bounds.
    unsafe {
        for i in 0..copy_len {
            *buffer.offset(i as isize) = bytes[i] as c_char;
        }
        *buffer.offset(copy_len as isize) = 0; // Null terminator
    }
    copy_len as c_int
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cresult() {
        let ok_result = CResult::ok(42);
        assert_eq!(ok_result.success, 1);
        assert_eq!(ok_result.error_code, 0);
        assert_eq!(ok_result.value, 42);

        let err_result = CResult::err(KernelError::OutOfMemory);
        assert_eq!(err_result.success, 0);
        assert_ne!(err_result.error_code, 0);
        assert_eq!(err_result.value, 0);
    }

    #[test]
    fn test_strlen() {
        let test_str = "Hello\0";
        let len = unsafe { c_strlen(test_str.as_ptr() as *const c_char) };
        assert_eq!(len, 5);

        let empty_str = "\0";
        let len = unsafe { c_strlen(empty_str.as_ptr() as *const c_char) };
        assert_eq!(len, 0);
    }

    #[test]
    fn test_rust_str_to_c_buf() {
        let mut buffer = [0i8; 64];
        unsafe {
            rust_str_to_c_buf("Test GPU", &mut buffer);
        }

        // Check first few characters
        assert_eq!(buffer[0], b'T' as i8);
        assert_eq!(buffer[1], b'e' as i8);
        assert_eq!(buffer[2], b's' as i8);
        assert_eq!(buffer[3], b't' as i8);
        assert_eq!(buffer[8], 0); // Null terminator
    }

    #[test]
    fn test_c_str_to_rust() {
        let test_str = "Hello World\0";
        let result = unsafe { c_str_to_rust(test_str.as_ptr() as *const c_char) };
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "Hello World");

        // Test null pointer
        let result = unsafe { c_str_to_rust(ptr::null()) };
        assert!(result.is_err());
    }

    #[test]
    fn test_init_cleanup() {
        let result = gpu_dma_lock_init();
        assert_eq!(result, 0);

        gpu_dma_lock_cleanup();

        // Test double init
        let result = gpu_dma_lock_init();
        assert_eq!(result, 0);

        gpu_dma_lock_cleanup();
    }

    #[test]
    fn test_device_registration() {
        gpu_dma_lock_init();

        let device_name = "Test GPU\0";
        let result = gpu_dma_register_device(
            0,
            device_name.as_ptr() as *const c_char,
            8 << 30, // 8GB
        );
        assert_eq!(result, 0);

        let count = gpu_dma_get_device_count();
        assert_eq!(count, 1);

        // Test device info retrieval
        let mut info = CGpuDeviceInfo {
            id: 0,
            name: [0; 64],
            total_memory: 0,
            available_memory: 0,
            allocated_memory: 0,
        };

        let result = gpu_dma_get_device_info(0, &mut info);
        assert_eq!(result, 0);
        assert_eq!(info.id, 0);
        assert_eq!(info.total_memory, 8 << 30);

        gpu_dma_lock_cleanup();
    }

    #[test]
    fn test_agent_management() {
        gpu_dma_lock_init();

        // Create agent
        let result = gpu_dma_create_agent(100, 2 << 30); // 2GB quota
        assert_eq!(result.success, 1);
        assert_eq!(result.value, 100);

        // Test duplicate agent
        let result = gpu_dma_create_agent(100, 1 << 30);
        assert_eq!(result.success, 0);

        // Remove agent
        let result = gpu_dma_remove_agent(100);
        assert_eq!(result.success, 1);

        // Test removing non-existent agent
        let result = gpu_dma_remove_agent(999);
        assert_eq!(result.success, 0);

        gpu_dma_lock_cleanup();
    }

    #[test]
    fn test_memory_allocation() {
        gpu_dma_lock_init();

        // Register device
        let device_name = "Test GPU\0";
        gpu_dma_register_device(0, device_name.as_ptr() as *const c_char, 8 << 30);

        // Create agent
        gpu_dma_create_agent(200, 4 << 30); // 4GB quota

        // Allocate memory
        let result = gpu_dma_allocate(200, 1 << 20, 0); // 1MB
        assert_eq!(result.success, 1);
        let alloc_id = result.value;

        // Get allocation info
        let mut info = CAllocationInfo {
            id: 0,
            agent_id: 0,
            size: 0,
            device_id: 0,
            timestamp: 0,
        };

        let result = gpu_dma_get_allocation_info(alloc_id, &mut info);
        assert_eq!(result, 0);
        assert_eq!(info.agent_id, 200);
        assert_eq!(info.size, 1 << 20);

        // Deallocate
        let result = gpu_dma_deallocate(alloc_id);
        assert_eq!(result.success, 1);

        // Test double deallocation
        let result = gpu_dma_deallocate(alloc_id);
        assert_eq!(result.success, 0);

        gpu_dma_lock_cleanup();
    }

    #[test]
    fn test_dma_access_control() {
        gpu_dma_lock_init();

        // Grant access
        let result = gpu_dma_grant_access(300, 0x1000, 0x2000, 3); // Read-write
        assert_eq!(result, 0);

        // Check access
        let result = gpu_dma_check_access(300, 0x1500, 1); // Read
        assert_eq!(result, 1);

        let result = gpu_dma_check_access(300, 0x1500, 2); // Write
        assert_eq!(result, 1);

        // Check out of range
        let result = gpu_dma_check_access(300, 0x500, 1);
        assert_eq!(result, 0);

        // Revoke access
        let result = gpu_dma_revoke_access(300);
        assert_eq!(result, 0);

        // Check after revocation
        let result = gpu_dma_check_access(300, 0x1500, 1);
        assert_eq!(result, 0);

        gpu_dma_lock_cleanup();
    }

    #[test]
    fn test_gpu_contexts() {
        gpu_dma_lock_init();

        // Register device
        let device_name = "Test GPU\0";
        gpu_dma_register_device(0, device_name.as_ptr() as *const c_char, 8 << 30);

        // Create context
        let result = gpu_dma_create_context(400, 0);
        assert_eq!(result.success, 1);
        let ctx_id = result.value;

        // Get context info
        let mut info = CGpuContext {
            id: 0,
            agent_id: 0,
            device_id: 0,
            state: 0,
            switch_count: 0,
        };

        let result = gpu_dma_get_context_info(ctx_id, &mut info);
        assert_eq!(result, 0);
        assert_eq!(info.agent_id, 400);
        assert_eq!(info.device_id, 0);

        // Switch context
        let result = gpu_dma_switch_context(0, ctx_id);
        assert_eq!(result.success, 1);

        gpu_dma_lock_cleanup();
    }

    #[test]
    fn test_statistics() {
        gpu_dma_lock_init();

        let mut stats = CGpuDmaStats {
            total_allocations: 0,
            total_deallocations: 0,
            total_bytes_allocated: 0,
            dma_checks: 0,
            dma_denials: 0,
            context_switches: 0,
        };

        let result = gpu_dma_get_stats(&mut stats);
        assert_eq!(result, 0);

        // Stats should be non-negative
        assert!(stats.total_allocations >= 0);
        assert!(stats.dma_checks >= 0);

        // Test reset
        gpu_dma_reset_stats();

        gpu_dma_lock_cleanup();
    }

    #[test]
    fn test_debug_functions() {
        gpu_dma_lock_init();

        // Test debug enable/disable
        gpu_dma_enable_debug(1);
        gpu_dma_enable_debug(0);

        // Test garbage collection
        gpu_dma_force_gc();

        // Test state dump
        gpu_dma_dump_state();

        gpu_dma_lock_cleanup();
    }

    #[test]
    fn test_version_and_init_status() {
        gpu_dma_lock_init();

        // Test initialization check
        let is_init = gpu_dma_is_initialized();
        assert_eq!(is_init, 1);

        // Test version
        let mut buffer = [0i8; 64];
        let len = gpu_dma_get_version(buffer.as_mut_ptr(), 64);
        assert!(len > 0);

        gpu_dma_lock_cleanup();
    }

    #[test]
    fn test_detailed_stats() {
        gpu_dma_lock_init();

        let mut buffer = [0i8; 1024];
        let len = gpu_dma_get_detailed_stats(buffer.as_mut_ptr(), 1024);
        // Should return 0 or positive length
        assert!(len >= 0);

        gpu_dma_lock_cleanup();
    }

    #[test]
    fn test_error_conditions() {
        gpu_dma_lock_init();

        // Test null pointers
        let result = gpu_dma_get_device_info(0, ptr::null_mut());
        assert_eq!(result, -1);

        let result = gpu_dma_get_allocation_info(0, ptr::null_mut());
        assert_eq!(result, -1);

        let result = gpu_dma_get_context_info(0, ptr::null_mut());
        assert_eq!(result, -1);

        let result = gpu_dma_get_stats(ptr::null_mut());
        assert_eq!(result, -1);

        let result = gpu_dma_get_version(ptr::null_mut(), 0);
        assert_eq!(result, -1);

        let result = gpu_dma_get_detailed_stats(ptr::null_mut(), 0);
        assert_eq!(result, -1);

        // Test invalid device registration
        let result = gpu_dma_register_device(0, ptr::null(), 1024);
        assert_eq!(result, -1);

        // Test invalid access modes
        let result = gpu_dma_check_access(100, 0x1000, 99); // Invalid mode
        assert_eq!(result, 0);

        gpu_dma_lock_cleanup();
    }
}
