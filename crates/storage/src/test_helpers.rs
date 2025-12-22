//! Test helpers for storage tests

#[cfg(test)]
pub mod tests {
    use std::collections::HashMap;
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

    /// Storage wrapper that uses a pre-poisoned mutex for testing
    pub struct PoisonedMemoryStorage {
        pub data: Arc<Mutex<HashMap<String, Vec<u8>>>>,
        pub total_capacity: u64,
    }

    impl PoisonedMemoryStorage {
        pub fn new(capacity: u64) -> Self {
            Self {
                data: create_poisoned_mutex(),
                total_capacity: capacity,
            }
        }
    }
}
