//! Metal 4 command allocator simulation.
//!
//! Provides per-frame memory management that mirrors MTL4CommandAllocator.
//! On Metal 4 (macOS 26+), this will wrap the native allocator.
//! On earlier systems, it provides a compatible abstraction using buffer pools.
//!
//! # Example (Apple's triple-buffering pattern)
//!
//! ```ignore
//! const MAX_FRAMES_IN_FLIGHT: usize = 3;
//! let allocator = Metal4CommandAllocator::new(&device, MAX_FRAMES_IN_FLIGHT)?;
//!
//! // In render loop:
//! allocator.reset(); // Reclaim memory from completed frames
//! let cmd = queue.create_command_buffer_with_allocator(&allocator)?;
//! ```

use crate::error::{MetalError, Result};
use crate::metal4::Metal4Device;

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Command allocator for efficient per-frame memory management.
///
/// Metal 4 introduces MTL4CommandAllocator for managing command buffer memory.
/// This type provides a compatible abstraction:
/// - On Metal 4: Will wrap MTL4CommandAllocator
/// - On Metal 3: Tracks frame lifecycle for memory management patterns
///
/// # Frame Management
///
/// The allocator tracks frames in a ring buffer pattern. Call `reset()`
/// at the start of each frame to reclaim memory from completed frames.
pub struct Metal4CommandAllocator {
    /// Reference to the device
    device: Arc<Metal4Device>,
    /// Maximum number of frames that can be in flight simultaneously
    max_frames_in_flight: usize,
    /// Current frame counter (monotonically increasing)
    frame_counter: AtomicU64,
    /// Number of command buffers created this frame
    buffers_this_frame: AtomicU64,
}

impl Metal4CommandAllocator {
    /// Create a new command allocator.
    ///
    /// # Arguments
    ///
    /// * `device` - The Metal 4 device
    /// * `max_frames_in_flight` - Maximum concurrent frames (typically 2-3)
    ///
    /// # Errors
    ///
    /// Returns an error if `max_frames_in_flight` is 0.
    pub fn new(device: &Arc<Metal4Device>, max_frames_in_flight: usize) -> Result<Self> {
        if max_frames_in_flight == 0 {
            return Err(MetalError::creation_failed(
                "Metal4CommandAllocator",
                "max_frames_in_flight must be at least 1",
            ));
        }

        Ok(Self {
            device: Arc::clone(device),
            max_frames_in_flight,
            frame_counter: AtomicU64::new(0),
            buffers_this_frame: AtomicU64::new(0),
        })
    }

    /// Reset the allocator for a new frame.
    ///
    /// Call this at the start of each frame to:
    /// - Advance the frame counter
    /// - Reclaim memory from frames that have completed
    ///
    /// On Metal 4, this will call `MTL4CommandAllocator::reset()`.
    pub fn reset(&self) {
        self.frame_counter.fetch_add(1, Ordering::SeqCst);
        self.buffers_this_frame.store(0, Ordering::SeqCst);

        // When Metal 4 is available:
        // self.raw.reset();
    }

    /// Get the current frame number.
    ///
    /// This is a monotonically increasing counter that wraps around.
    pub fn current_frame(&self) -> u64 {
        self.frame_counter.load(Ordering::SeqCst)
    }

    /// Get the current frame index (for ring buffer access).
    ///
    /// Returns a value in the range `[0, max_frames_in_flight)`.
    pub fn current_frame_index(&self) -> usize {
        (self.current_frame() as usize) % self.max_frames_in_flight
    }

    /// Get the maximum number of frames in flight.
    pub fn max_frames_in_flight(&self) -> usize {
        self.max_frames_in_flight
    }

    /// Get the number of command buffers created this frame.
    pub fn buffers_this_frame(&self) -> u64 {
        self.buffers_this_frame.load(Ordering::SeqCst)
    }

    /// Record that a command buffer was created.
    ///
    /// Called internally when creating command buffers with this allocator.
    pub(crate) fn record_buffer_created(&self) {
        self.buffers_this_frame.fetch_add(1, Ordering::SeqCst);
    }

    /// Get a reference to the device.
    pub fn device(&self) -> &Arc<Metal4Device> {
        &self.device
    }

    /// Check if this allocator can be used to create more command buffers.
    ///
    /// Metal 4 allocators have memory limits. This checks if we're within budget.
    pub fn has_capacity(&self) -> bool {
        // On Metal 3, there's no hard limit
        // On Metal 4, would check allocator memory budget
        true
    }
}

// SAFETY: Metal4CommandAllocator is Send because:
// 1. Arc<Metal4Device> is Send + Sync
// 2. AtomicU64 operations are thread-safe
// 3. All methods use atomic operations for state updates
unsafe impl Send for Metal4CommandAllocator {}

// SAFETY: Metal4CommandAllocator is Sync because:
// 1. All state is accessed through atomic operations
// 2. No mutable borrows are required for any operation
// 3. reset() and record_buffer_created() are safe to call concurrently
unsafe impl Sync for Metal4CommandAllocator {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allocator_creation() {
        if let Ok(device) = Metal4Device::system_default() {
            let device = Arc::new(device);
            let allocator =
                Metal4CommandAllocator::new(&device, 3).expect("Failed to create allocator");

            assert_eq!(allocator.max_frames_in_flight(), 3);
            assert_eq!(allocator.current_frame(), 0);
            assert_eq!(allocator.current_frame_index(), 0);
            assert_eq!(allocator.buffers_this_frame(), 0);
        }
    }

    #[test]
    fn test_allocator_zero_frames_fails() {
        if let Ok(device) = Metal4Device::system_default() {
            let device = Arc::new(device);
            let result = Metal4CommandAllocator::new(&device, 0);
            assert!(result.is_err());
        }
    }

    #[test]
    fn test_allocator_reset() {
        if let Ok(device) = Metal4Device::system_default() {
            let device = Arc::new(device);
            let allocator =
                Metal4CommandAllocator::new(&device, 3).expect("Failed to create allocator");

            assert_eq!(allocator.current_frame(), 0);

            allocator.reset();
            assert_eq!(allocator.current_frame(), 1);

            allocator.reset();
            assert_eq!(allocator.current_frame(), 2);

            allocator.reset();
            assert_eq!(allocator.current_frame(), 3);
            assert_eq!(allocator.current_frame_index(), 0); // Wraps around
        }
    }

    #[test]
    fn test_allocator_buffer_tracking() {
        if let Ok(device) = Metal4Device::system_default() {
            let device = Arc::new(device);
            let allocator =
                Metal4CommandAllocator::new(&device, 3).expect("Failed to create allocator");

            assert_eq!(allocator.buffers_this_frame(), 0);

            allocator.record_buffer_created();
            assert_eq!(allocator.buffers_this_frame(), 1);

            allocator.record_buffer_created();
            assert_eq!(allocator.buffers_this_frame(), 2);

            // Reset clears the counter
            allocator.reset();
            assert_eq!(allocator.buffers_this_frame(), 0);
        }
    }

    #[test]
    fn test_allocator_capacity() {
        if let Ok(device) = Metal4Device::system_default() {
            let device = Arc::new(device);
            let allocator =
                Metal4CommandAllocator::new(&device, 3).expect("Failed to create allocator");

            // On Metal 3, always has capacity
            assert!(allocator.has_capacity());
        }
    }

    #[test]
    fn test_allocator_thread_safety() {
        use std::thread;

        if let Ok(device) = Metal4Device::system_default() {
            let device = Arc::new(device);
            let allocator = Arc::new(
                Metal4CommandAllocator::new(&device, 3).expect("Failed to create allocator"),
            );

            let allocator1 = Arc::clone(&allocator);
            let allocator2 = Arc::clone(&allocator);

            let handle1 = thread::spawn(move || {
                for _ in 0..100 {
                    allocator1.record_buffer_created();
                }
            });

            let handle2 = thread::spawn(move || {
                for _ in 0..100 {
                    allocator2.record_buffer_created();
                }
            });

            handle1.join().unwrap();
            handle2.join().unwrap();

            // Both threads should have recorded their buffers
            assert_eq!(allocator.buffers_this_frame(), 200);
        }
    }
}
