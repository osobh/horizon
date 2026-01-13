//! Absolutely minimal kernel module implementation
//! No external dependencies, no allocations, no complex features

#![no_std]
#![no_main]

// Simple GPU DMA init - returns 0 for success
#[unsafe(no_mangle)]
pub extern "C" fn gpu_dma_lock_init() -> i32 {
    0
}

// Simple cleanup - does nothing
#[unsafe(no_mangle)]
pub extern "C" fn gpu_dma_lock_cleanup() {
    // Nothing to clean up
}

// Register device - just check ID is valid
#[unsafe(no_mangle)]
pub extern "C" fn gpu_dma_register_device(id: u32, _name: *const u8, _total_memory: u64) -> i32 {
    if id < 8 {
        0
    } else {
        -1
    }
}

// Enable debug - does nothing
#[unsafe(no_mangle)]
pub extern "C" fn gpu_dma_enable_debug(_enable: i32) {
    // No debug functionality
}

// CUDA alloc hook - return fake allocation ID
#[unsafe(no_mangle)]
pub extern "C" fn gpu_dma_cuda_alloc_hook(agent_id: u64, size: u64, device_id: u32) -> u64 {
    if device_id < 8 && size > 0 && agent_id > 0 {
        // Simple fake allocation ID
        (agent_id << 32) | (size & 0xFFFFFFFF)
    } else {
        0
    }
}

// CUDA free hook - just validate ID
#[unsafe(no_mangle)]
pub extern "C" fn gpu_dma_cuda_free_hook(alloc_id: u64) -> i32 {
    if alloc_id != 0 {
        0
    } else {
        -1
    }
}

// Kernel panic handler
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    // SAFETY: This is the panic handler for a no_std kernel module. The inline
    // assembly creates an infinite loop (jmp to self) which is the only safe
    // thing to do when panicking in kernel space without external dependencies.
    // The noreturn option correctly marks this as a diverging function.
    unsafe {
        core::arch::asm!("2: jmp 2b", options(noreturn));
    }
}