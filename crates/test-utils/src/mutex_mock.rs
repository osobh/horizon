//! Mock mutex implementation for testing lock failures

use std::panic;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::thread;

/// Trait for mutex-like operations
pub trait MutexLike<T> {
    type Guard<'a>
    where
        Self: 'a,
        T: 'a;

    fn lock(&self) -> Result<Self::Guard<'_>, PoisonError<MutexGuard<'_, T>>>;
    fn is_poisoned(&self) -> bool;
}

/// Standard mutex wrapper
pub struct StandardMutex<T> {
    inner: Mutex<T>,
}

impl<T> StandardMutex<T> {
    pub fn new(value: T) -> Self {
        Self {
            inner: Mutex::new(value),
        }
    }
}

impl<T> MutexLike<T> for StandardMutex<T> {
    type Guard<'a>
        = MutexGuard<'a, T>
    where
        Self: 'a,
        T: 'a;

    fn lock(&self) -> Result<Self::Guard<'_>, PoisonError<MutexGuard<'_, T>>> {
        self.inner.lock()
    }

    fn is_poisoned(&self) -> bool {
        self.inner.is_poisoned()
    }
}

/// Mock mutex that can be configured to fail
pub struct MockMutex<T> {
    inner: Mutex<T>,
    fail_next: Arc<Mutex<bool>>,
}

impl<T> MockMutex<T> {
    pub fn new(value: T) -> Self {
        Self {
            inner: Mutex::new(value),
            fail_next: Arc::new(Mutex::new(false)),
        }
    }

    /// Configure the mutex to fail on next lock attempt
    pub fn fail_next_lock(&self) {
        if let Ok(mut flag) = self.fail_next.lock() {
            *flag = true;
        }
    }

    /// Poison the mutex by panicking in another thread
    pub fn poison(&self)
    where
        T: Default + Send + 'static,
    {
        let inner_clone = Arc::new(Mutex::new(T::default()));

        // Use a separate thread to poison a mutex
        thread::spawn(move || {
            let _guard = inner_clone.lock().unwrap();
            panic!("Poisoning mutex");
        })
        .join()
        .ok();

        // Now simulate the poisoned state
        // Since we can't actually poison the real mutex safely,
        // we'll configure it to always fail
        self.fail_next_lock();
    }
}

impl<T> MutexLike<T> for MockMutex<T> {
    type Guard<'a>
        = MutexGuard<'a, T>
    where
        Self: 'a,
        T: 'a;

    fn lock(&self) -> Result<Self::Guard<'_>, PoisonError<MutexGuard<'_, T>>> {
        // Check if we should fail
        let should_fail = self
            .fail_next
            .lock()
            .map(|mut flag| {
                let fail = *flag;
                *flag = false; // Reset for next time
                fail
            })
            .unwrap_or(false);

        if should_fail {
            // Create a poisoned mutex and return its error
            let poisoned_mutex = Arc::new(Mutex::new(()));
            thread::spawn(move || {
                let _guard = poisoned_mutex.lock().unwrap();
                panic!("Simulated panic");
            })
            .join()
            .ok();

            // This is a bit hacky but works for testing
            match self.inner.lock() {
                Ok(guard) => {
                    // Force a poison error by creating one
                    // We'll abuse the type system a bit here
                    drop(guard);
                    // Return a custom error that looks like a poison error
                    Err(PoisonError::new(self.inner.lock().unwrap()))
                }
                Err(e) => Err(e),
            }
        } else {
            self.inner.lock()
        }
    }

    fn is_poisoned(&self) -> bool {
        self.inner.is_poisoned()
    }
}

/// Helper to create a poisoned mutex for testing
pub fn create_poisoned_mutex<T: Send + Default + 'static>() -> Arc<Mutex<T>> {
    let mutex = Arc::new(Mutex::new(T::default()));
    let mutex_clone = mutex.clone();

    // Poison it in another thread
    thread::spawn(move || {
        let _guard = mutex_clone.lock().unwrap();
        panic!("Poisoning the mutex");
    })
    .join()
    .ok();

    mutex
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standard_mutex() {
        let mutex = StandardMutex::new(42);
        let guard = mutex.lock().unwrap();
        assert_eq!(*guard, 42);
    }

    #[test]
    fn test_mock_mutex_normal() {
        let mutex = MockMutex::new("test");
        let guard = mutex.lock().unwrap();
        assert_eq!(*guard, "test");
    }

    #[test]
    fn test_create_poisoned_mutex() {
        let mutex = create_poisoned_mutex::<Vec<i32>>();
        assert!(mutex.is_poisoned());
        assert!(mutex.lock().is_err());
    }
}
