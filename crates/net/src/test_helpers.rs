//! Test helpers for network crate tests

#[cfg(test)]
pub mod tests {
    use crate::{Message, NetworkStats};
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

    /// MemoryNetwork wrapper that uses pre-poisoned mutexes for testing
    pub struct PoisonedMemoryNetwork {
        pub messages: Arc<Mutex<Vec<(String, Message)>>>,
        pub sent_messages: Arc<Mutex<Vec<(String, Message)>>>,
        pub stats: Arc<Mutex<NetworkStats>>,
    }

    impl PoisonedMemoryNetwork {
        pub fn new_messages_poisoned() -> Self {
            Self {
                messages: create_poisoned_mutex(),
                sent_messages: Arc::new(Mutex::new(Vec::new())),
                stats: Arc::new(Mutex::new(NetworkStats::default())),
            }
        }

        pub fn new_sent_poisoned() -> Self {
            Self {
                messages: Arc::new(Mutex::new(Vec::new())),
                sent_messages: create_poisoned_mutex(),
                stats: Arc::new(Mutex::new(NetworkStats::default())),
            }
        }

        pub fn new_stats_poisoned() -> Self {
            Self {
                messages: Arc::new(Mutex::new(Vec::new())),
                sent_messages: Arc::new(Mutex::new(Vec::new())),
                stats: create_poisoned_mutex(),
            }
        }
    }

    /// ZeroCopyTransport wrapper that uses pre-poisoned mutexes for testing
    pub struct PoisonedZeroCopyTransport {
        pub buffer_pools: Arc<Mutex<std::collections::HashMap<String, Vec<u8>>>>,
        pub stats: Arc<Mutex<NetworkStats>>,
        pub message_queue: Arc<Mutex<Vec<(String, Message)>>>,
    }

    impl PoisonedZeroCopyTransport {
        pub fn new_buffer_pools_poisoned() -> Self {
            Self {
                buffer_pools: create_poisoned_mutex(),
                stats: Arc::new(Mutex::new(NetworkStats::default())),
                message_queue: Arc::new(Mutex::new(Vec::new())),
            }
        }

        pub fn new_stats_poisoned() -> Self {
            Self {
                buffer_pools: Arc::new(Mutex::new(std::collections::HashMap::new())),
                stats: create_poisoned_mutex(),
                message_queue: Arc::new(Mutex::new(Vec::new())),
            }
        }

        pub fn new_message_queue_poisoned() -> Self {
            Self {
                buffer_pools: Arc::new(Mutex::new(std::collections::HashMap::new())),
                stats: Arc::new(Mutex::new(NetworkStats::default())),
                message_queue: create_poisoned_mutex(),
            }
        }
    }
}
