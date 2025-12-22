//! Test utilities for ExoRust project
//!
//! This module provides utilities for testing edge cases like mutex poisoning

pub mod mutex_mock;

#[cfg(test)]
mod comprehensive_tests;

#[cfg(test)]
mod edge_case_tests;

use std::ops::{Deref, DerefMut};
use std::panic;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

/// A wrapper around Mutex that can be configured to simulate poisoning
pub struct TestMutex<T> {
    inner: Mutex<T>,
    should_poison: Arc<Mutex<bool>>,
    poison_on_lock: Arc<Mutex<bool>>,
}

impl<T> TestMutex<T> {
    /// Create a new TestMutex
    pub fn new(value: T) -> Self {
        Self {
            inner: Mutex::new(value),
            should_poison: Arc::new(Mutex::new(false)),
            poison_on_lock: Arc::new(Mutex::new(false)),
        }
    }

    /// Configure the mutex to poison on the next unlock
    pub fn poison_next(&self) {
        if let Ok(mut flag) = self.should_poison.lock() {
            *flag = true;
        }
    }

    /// Configure the mutex to panic when lock is called
    pub fn panic_on_lock(&self) {
        if let Ok(mut flag) = self.poison_on_lock.lock() {
            *flag = true;
        }
    }

    /// Lock the mutex
    pub fn lock(&self) -> Result<TestMutexGuard<'_, T>, PoisonError<MutexGuard<'_, T>>> {
        // Check if we should panic on lock
        if let Ok(flag) = self.poison_on_lock.lock() {
            if *flag {
                panic!("Simulated panic on lock!");
            }
        }

        match self.inner.lock() {
            Ok(guard) => Ok(TestMutexGuard {
                guard: Some(guard),
                should_poison: self.should_poison.clone(),
            }),
            Err(e) => Err(e),
        }
    }

    /// Try to lock the mutex
    pub fn try_lock(
        &self,
    ) -> Result<TestMutexGuard<'_, T>, std::sync::TryLockError<MutexGuard<'_, T>>> {
        match self.inner.try_lock() {
            Ok(guard) => Ok(TestMutexGuard {
                guard: Some(guard),
                should_poison: self.should_poison.clone(),
            }),
            Err(e) => Err(e),
        }
    }

    /// Get mutable reference to inner value (for testing)
    pub fn get_mut(&mut self) -> Option<&mut T> {
        self.inner.get_mut().ok()
    }

    /// Check if the mutex is poisoned
    pub fn is_poisoned(&self) -> bool {
        self.inner.is_poisoned()
    }
}

/// Guard for TestMutex that can simulate poisoning on drop
pub struct TestMutexGuard<'a, T> {
    guard: Option<MutexGuard<'a, T>>,
    should_poison: Arc<Mutex<bool>>,
}

impl<'a, T> Deref for TestMutexGuard<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.guard.as_ref().unwrap()
    }
}

impl<'a, T> DerefMut for TestMutexGuard<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.guard.as_mut().unwrap()
    }
}

impl<'a, T> Drop for TestMutexGuard<'a, T> {
    fn drop(&mut self) {
        // Check if we should poison the mutex
        let should_poison = self.should_poison.lock().map(|flag| *flag).unwrap_or(false);

        if should_poison {
            // Reset the flag
            if let Ok(mut flag) = self.should_poison.lock() {
                *flag = false;
            }

            // Drop the guard first
            drop(self.guard.take());

            // Panic to poison the mutex
            panic!("Simulated panic to poison mutex!");
        }
    }
}

/// Helper function to test code that should handle mutex poisoning
pub fn with_poisoned_mutex<T, F>(value: T, test_fn: F)
where
    T: Send + 'static,
    F: FnOnce(Arc<TestMutex<T>>),
{
    let mutex = Arc::new(TestMutex::new(value));
    let mutex_clone = mutex.clone();

    // Spawn a thread that will poison the mutex
    let handle = std::thread::spawn(move || {
        let _guard = mutex_clone.lock().unwrap();
        panic!("Poisoning the mutex!");
    });

    // Wait for the thread to panic and poison the mutex
    let _ = handle.join();

    // Now the mutex is poisoned, run the test
    test_fn(mutex);
}

/// Macro to inject a TestMutex in place of a regular Mutex for testing
#[macro_export]
macro_rules! test_mutex {
    ($value:expr) => {
        $crate::TestMutex::new($value)
    };
}

/// Test helper to run a closure and capture panics
pub fn catch_panic<F, R>(f: F) -> Result<R, String>
where
    F: FnOnce() -> R + panic::UnwindSafe,
{
    match panic::catch_unwind(f) {
        Ok(result) => Ok(result),
        Err(e) => {
            let msg = if let Some(s) = e.downcast_ref::<&str>() {
                s.to_string()
            } else if let Some(s) = e.downcast_ref::<String>() {
                s.clone()
            } else {
                "Unknown panic".to_string()
            };
            Err(msg)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normal_mutex_operations() {
        let mutex = TestMutex::new(42);

        {
            let mut guard = mutex.lock().unwrap();
            *guard = 100;
        }

        let guard = mutex.lock().unwrap();
        assert_eq!(*guard, 100);
    }

    #[test]
    fn test_mutex_poisoning() {
        use std::sync::Mutex as StdMutex;

        let mutex = Arc::new(StdMutex::new(vec![1, 2, 3]));
        let mutex_clone = mutex.clone();

        // Create a thread that will panic and poison the mutex
        let handle = std::thread::spawn(move || {
            let _guard = mutex_clone.lock().unwrap();
            panic!("Intentional panic to poison mutex");
        });

        // Wait for the thread to panic
        let _ = handle.join();

        assert!(mutex.is_poisoned());

        // Next lock should return PoisonError
        let result = mutex.lock();
        assert!(result.is_err());
    }

    #[test]
    fn test_panic_on_lock() {
        let mutex = TestMutex::new("test");

        mutex.panic_on_lock();

        let result = catch_panic(|| {
            let _guard = mutex.lock();
        });

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Simulated panic on lock"));
    }

    #[test]
    fn test_with_poisoned_mutex() {
        with_poisoned_mutex(vec![1, 2, 3], |mutex| {
            assert!(mutex.is_poisoned());

            let result = mutex.lock();
            assert!(result.is_err());
        });
    }
}
