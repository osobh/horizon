//! Test helpers for memory crate tests

#[cfg(test)]
pub mod tests {
    use crate::GpuMemoryHandle;
    use std::collections::VecDeque;
    use std::sync::{Arc, Mutex};
    use std::thread;

    /// Create a poisoned mutex for testing
    pub fn create_poisoned_mutex<T: Send + Default + 'static>() -> Arc<Mutex<T>> {
        let mutex = Arc::new(Mutex::new(T::default()));
        let mutex_clone = mutex.clone();

        // Spawn a thread that will panic while holding the lock
        let handle = thread::spawn(move || {
            let _guard = mutex_clone.lock().unwrap();
            panic!("Intentionally poisoning mutex for testing");
        });

        // Wait for the thread to finish (and poison the mutex)
        let _ = handle.join();

        mutex
    }

    /// Pool wrapper that uses a pre-poisoned mutex for testing
    pub struct PoisonedMemoryPool {
        pub block_size: usize,
        pub pool: Arc<Mutex<VecDeque<GpuMemoryHandle>>>,
        pub max_blocks: usize,
    }

    impl PoisonedMemoryPool {
        pub fn new(block_size: usize, max_blocks: usize) -> Self {
            Self {
                block_size,
                pool: create_poisoned_mutex(),
                max_blocks,
            }
        }
    }
}
