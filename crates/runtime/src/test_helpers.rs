//! Test helpers for runtime crate tests

#[cfg(test)]
pub mod tests {
    use crate::{ContainerState, GpuContainer};
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

    /// Container wrapper that uses pre-poisoned mutexes for testing
    pub struct PoisonedContainer {
        pub state: Arc<Mutex<ContainerState>>,
    }

    impl PoisonedContainer {
        pub fn new() -> Self {
            Self {
                state: create_poisoned_mutex(),
            }
        }
    }

    /// ContainerLifecycle wrapper that uses pre-poisoned mutexes for testing
    pub struct PoisonedContainerLifecycle {
        pub containers: Arc<Mutex<HashMap<String, Arc<GpuContainer>>>>,
    }

    impl PoisonedContainerLifecycle {
        pub fn new() -> Self {
            Self {
                containers: create_poisoned_mutex(),
            }
        }
    }
}
