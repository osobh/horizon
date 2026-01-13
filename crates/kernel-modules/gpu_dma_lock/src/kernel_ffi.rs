//! Minimal FFI interface for kernel module
//! 
//! This module provides the absolute minimum FFI interface to avoid GOT relocations

#![allow(non_camel_case_types)]

use core::ffi::{c_char, c_int, c_ulong};

/// Initialize the GPU DMA lock subsystem
#[unsafe(no_mangle)]
pub extern "C" fn gpu_dma_lock_init() -> c_int {
    // Minimal initialization
    0
}

/// Cleanup the GPU DMA lock subsystem
#[unsafe(no_mangle)]
pub extern "C" fn gpu_dma_lock_cleanup() {
    // Minimal cleanup
}

/// Register a GPU device
#[unsafe(no_mangle)]
pub extern "C" fn gpu_dma_register_device(
    id: u32,
    _name: *const c_char,
    _total_memory: c_ulong,
) -> c_int {
    // Store device info
    if id < 8 {
        0
    } else {
        -1
    }
}

/// Enable debug mode
#[unsafe(no_mangle)]
pub extern "C" fn gpu_dma_enable_debug(_enable: c_int) {
    // Set debug flag
}

/// CUDA allocation hook
#[unsafe(no_mangle)]
pub extern "C" fn gpu_dma_cuda_alloc_hook(
    agent_id: c_ulong,
    size: c_ulong,
    device_id: u32,
) -> c_ulong {
    // Simple allocation tracking
    if device_id < 8 && size > 0 && agent_id > 0 {
        // Return a fake allocation ID
        (agent_id << 32) | (size & 0xFFFFFFFF)
    } else {
        0
    }
}

/// CUDA free hook
#[unsafe(no_mangle)]
pub extern "C" fn gpu_dma_cuda_free_hook(alloc_id: c_ulong) -> c_int {
    if alloc_id != 0 {
        0
    } else {
        -1
    }
}