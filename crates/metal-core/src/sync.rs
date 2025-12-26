//! Synchronization primitives for Metal.
//!
//! Provides CPU-GPU synchronization mechanisms.

use crate::error::Result;

/// Trait for Metal synchronization primitives.
pub trait MetalSync: Send + Sync {
    /// Wait for the sync point to be signaled.
    fn wait(&self) -> Result<()>;

    /// Wait with a timeout in milliseconds.
    fn wait_timeout(&self, timeout_ms: u64) -> Result<bool>;

    /// Check if the sync point is signaled without blocking.
    fn is_signaled(&self) -> bool;
}

/// A fence for ordering GPU work.
pub trait MetalFence: MetalSync {
    /// Get the current fence value.
    fn value(&self) -> u64;
}

/// A shared event for CPU-GPU synchronization.
///
/// Used in Metal 4 for efficient frame synchronization.
pub trait MetalSharedEvent: MetalSync {
    /// Get the current event value.
    fn signaled_value(&self) -> u64;

    /// Wait until the event reaches the specified value.
    fn wait_until(&self, value: u64) -> Result<()>;

    /// Wait with timeout until the event reaches the specified value.
    fn wait_until_timeout(&self, value: u64, timeout_ms: u64) -> Result<bool>;
}

/// Semaphore for limiting concurrent GPU work.
#[derive(Debug)]
pub struct GpuSemaphore {
    value: std::sync::atomic::AtomicU64,
    max_value: u64,
}

impl GpuSemaphore {
    /// Create a new semaphore with the given initial value.
    pub fn new(initial_value: u64, max_value: u64) -> Self {
        Self {
            value: std::sync::atomic::AtomicU64::new(initial_value),
            max_value,
        }
    }

    /// Get the current value.
    pub fn value(&self) -> u64 {
        self.value.load(std::sync::atomic::Ordering::Acquire)
    }

    /// Signal (increment) the semaphore.
    pub fn signal(&self) {
        self.value.fetch_add(1, std::sync::atomic::Ordering::Release);
    }

    /// Wait (decrement) the semaphore.
    pub fn wait(&self) {
        loop {
            let current = self.value.load(std::sync::atomic::Ordering::Acquire);
            if current > 0 {
                // Use compare_exchange_weak in loop - more efficient on ARM (LL/SC)
                // Spurious failures are acceptable since we retry anyway
                if self
                    .value
                    .compare_exchange_weak(
                        current,
                        current - 1,
                        std::sync::atomic::Ordering::AcqRel,
                        std::sync::atomic::Ordering::Relaxed,
                    )
                    .is_ok()
                {
                    return;
                }
            }
            std::hint::spin_loop();
        }
    }

    /// Try to wait without blocking.
    pub fn try_wait(&self) -> bool {
        let current = self.value.load(std::sync::atomic::Ordering::Acquire);
        if current > 0 {
            self.value
                .compare_exchange(
                    current,
                    current - 1,
                    std::sync::atomic::Ordering::AcqRel,
                    std::sync::atomic::Ordering::Relaxed,
                )
                .is_ok()
        } else {
            false
        }
    }
}

/// Frame synchronization helper.
///
/// Manages a pool of command buffers for triple-buffering.
pub struct FrameSync {
    current_frame: std::sync::atomic::AtomicU64,
    max_frames_in_flight: u64,
}

impl FrameSync {
    /// Create a new frame sync with the given number of frames in flight.
    pub fn new(max_frames_in_flight: u64) -> Self {
        Self {
            current_frame: std::sync::atomic::AtomicU64::new(0),
            max_frames_in_flight,
        }
    }

    /// Get the current frame index.
    pub fn current_frame(&self) -> u64 {
        self.current_frame.load(std::sync::atomic::Ordering::Acquire)
    }

    /// Get the buffer index for the current frame.
    pub fn buffer_index(&self) -> usize {
        (self.current_frame() % self.max_frames_in_flight) as usize
    }

    /// Advance to the next frame.
    pub fn next_frame(&self) -> u64 {
        self.current_frame
            .fetch_add(1, std::sync::atomic::Ordering::AcqRel)
            + 1
    }

    /// Get the maximum frames in flight.
    pub fn max_frames_in_flight(&self) -> u64 {
        self.max_frames_in_flight
    }
}
