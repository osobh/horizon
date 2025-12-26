//! Test helpers for evolution crate tests

#[cfg(test)]
pub mod tests {
    use crate::EvolutionStats;
    use std::sync::{Arc, Mutex};
    use std::thread;

    /// Create a poisoned mutex for testing
    pub fn create_poisoned_mutex<T: Send + Default + 'static>() -> Arc<Mutex<T>> {
        let mutex = Arc::new(Mutex::new(T::default()));
        let mutex_clone = mutex.clone();

        // Spawn a thread that will panic while holding the lock
        let handle = thread::spawn(move || {
            let _guard = mutex_clone.lock()?;
            panic!("Intentionally poisoning mutex for testing");
        });

        // Wait for the thread to finish (and poison the mutex)
        let _ = handle.join();

        mutex
    }

    /// EvolutionEngine wrapper that uses pre-poisoned mutexes for testing
    pub struct PoisonedEvolutionEngine {
        pub stats: Arc<Mutex<EvolutionStats>>,
    }

    impl PoisonedEvolutionEngine {
        pub fn new() -> Self {
            Self {
                stats: create_poisoned_mutex(),
            }
        }
    }
}
