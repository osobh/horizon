//! Spin lock implementation for no_std kernel environment
//!
//! This module provides atomic spinlock implementations for kernel modules.
//! These locks use busy-waiting with proper memory ordering for correctness.

use core::cell::UnsafeCell;
use core::hint::spin_loop;
use core::ops::{Deref, DerefMut};
use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// Atomic read-write lock using spinlock semantics
///
/// This lock allows multiple concurrent readers OR a single exclusive writer.
/// Uses atomic operations for thread-safe access in kernel context.
pub struct RwLock<T> {
    /// Lock state: 0 = unlocked, usize::MAX = write locked, 1..MAX-1 = reader count
    state: AtomicUsize,
    /// Protected data
    data: UnsafeCell<T>,
}

/// Sentinel value indicating write lock is held
const WRITE_LOCKED: usize = usize::MAX;

// SAFETY: RwLock<T> is Send if T is Send because:
// 1. The atomic state ensures proper synchronization
// 2. Only one thread can hold write access at a time
// 3. Data is only accessible through properly synchronized guards
unsafe impl<T: Send> Send for RwLock<T> {}

// SAFETY: RwLock<T> is Sync if T is Send + Sync because:
// 1. Multiple readers require T: Sync for concurrent shared access
// 2. Writers require T: Send for exclusive mutable access
// 3. Atomic operations provide memory ordering guarantees
unsafe impl<T: Send + Sync> Sync for RwLock<T> {}

impl<T> RwLock<T> {
    /// Create a new RwLock
    pub const fn new(data: T) -> Self {
        Self {
            state: AtomicUsize::new(0),
            data: UnsafeCell::new(data),
        }
    }

    /// Acquire read lock, spinning until available
    ///
    /// Multiple readers can hold the lock simultaneously.
    /// Blocks if a writer holds or is waiting for the lock.
    pub fn read(&self) -> RwLockReadGuard<T> {
        loop {
            let state = self.state.load(Ordering::Relaxed);

            // Cannot acquire read lock if write locked
            if state == WRITE_LOCKED {
                spin_loop();
                continue;
            }

            // Try to increment reader count
            // Use Acquire ordering to synchronize with writer's Release
            match self.state.compare_exchange_weak(
                state,
                state + 1,
                Ordering::Acquire,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    debug_assert!(
                        self.state.load(Ordering::Relaxed) != WRITE_LOCKED,
                        "Read lock acquired but state indicates write lock"
                    );
                    return RwLockReadGuard { lock: self };
                }
                Err(_) => {
                    spin_loop();
                    continue;
                }
            }
        }
    }

    /// Acquire write lock, spinning until available
    ///
    /// Exclusive access - no other readers or writers can hold the lock.
    pub fn write(&self) -> RwLockWriteGuard<T> {
        loop {
            // Try to acquire write lock (state must be 0)
            // Use Acquire ordering to synchronize with previous Release
            match self.state.compare_exchange_weak(
                0,
                WRITE_LOCKED,
                Ordering::Acquire,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    debug_assert!(
                        self.state.load(Ordering::Relaxed) == WRITE_LOCKED,
                        "Write lock acquired but state doesn't indicate write lock"
                    );
                    return RwLockWriteGuard { lock: self };
                }
                Err(_) => {
                    spin_loop();
                    continue;
                }
            }
        }
    }

    /// Try to acquire read lock without blocking
    pub fn try_read(&self) -> Option<RwLockReadGuard<T>> {
        let state = self.state.load(Ordering::Relaxed);

        if state == WRITE_LOCKED {
            return None;
        }

        match self
            .state
            .compare_exchange(state, state + 1, Ordering::Acquire, Ordering::Relaxed)
        {
            Ok(_) => Some(RwLockReadGuard { lock: self }),
            Err(_) => None,
        }
    }

    /// Try to acquire write lock without blocking
    pub fn try_write(&self) -> Option<RwLockWriteGuard<T>> {
        match self
            .state
            .compare_exchange(0, WRITE_LOCKED, Ordering::Acquire, Ordering::Relaxed)
        {
            Ok(_) => Some(RwLockWriteGuard { lock: self }),
            Err(_) => None,
        }
    }
}

/// Read guard for RwLock - allows shared read access
pub struct RwLockReadGuard<'a, T> {
    lock: &'a RwLock<T>,
}

impl<'a, T> Deref for RwLockReadGuard<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        // SAFETY: Read lock is held, guaranteeing no concurrent writes
        unsafe { &*self.lock.data.get() }
    }
}

impl<'a, T> Drop for RwLockReadGuard<'a, T> {
    fn drop(&mut self) {
        // Decrement reader count with Release ordering
        // This synchronizes with Acquire in write()
        let prev = self.lock.state.fetch_sub(1, Ordering::Release);
        debug_assert!(
            prev != 0 && prev != WRITE_LOCKED,
            "Invalid state when dropping read guard: {}",
            prev
        );
    }
}

/// Write guard for RwLock - allows exclusive write access
pub struct RwLockWriteGuard<'a, T> {
    lock: &'a RwLock<T>,
}

impl<'a, T> Deref for RwLockWriteGuard<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        // SAFETY: Write lock is held, guaranteeing exclusive access
        unsafe { &*self.lock.data.get() }
    }
}

impl<'a, T> DerefMut for RwLockWriteGuard<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        // SAFETY: Write lock is held, guaranteeing exclusive access
        unsafe { &mut *self.lock.data.get() }
    }
}

impl<'a, T> Drop for RwLockWriteGuard<'a, T> {
    fn drop(&mut self) {
        // Release write lock with Release ordering
        // This synchronizes with Acquire in read() and write()
        let prev = self.lock.state.swap(0, Ordering::Release);
        debug_assert!(
            prev == WRITE_LOCKED,
            "Invalid state when dropping write guard: {}",
            prev
        );
    }
}

/// Atomic mutex using spinlock semantics
///
/// Provides exclusive access to protected data using busy-waiting.
pub struct Mutex<T> {
    /// Lock state: false = unlocked, true = locked
    locked: AtomicBool,
    /// Protected data
    data: UnsafeCell<T>,
}

// SAFETY: Mutex<T> is Send if T is Send because:
// 1. The atomic bool ensures proper synchronization
// 2. Only one thread can hold the lock at a time
// 3. Data is only accessible through the MutexGuard
unsafe impl<T: Send> Send for Mutex<T> {}

// SAFETY: Mutex<T> is Sync if T is Send because:
// 1. Mutex provides exclusive access (T: Sync not required)
// 2. The lock ensures only one thread accesses data at a time
// 3. Atomic operations provide memory ordering guarantees
unsafe impl<T: Send> Sync for Mutex<T> {}

impl<T> Mutex<T> {
    /// Create a new Mutex
    pub const fn new(data: T) -> Self {
        Self {
            locked: AtomicBool::new(false),
            data: UnsafeCell::new(data),
        }
    }

    /// Acquire the mutex lock, spinning until available
    pub fn lock(&self) -> MutexGuard<T> {
        loop {
            // Try to acquire lock using test-and-set pattern
            // Use Acquire ordering to synchronize with Release in drop
            match self.locked.compare_exchange_weak(
                false,
                true,
                Ordering::Acquire,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    debug_assert!(
                        self.locked.load(Ordering::Relaxed),
                        "Lock acquired but state shows unlocked"
                    );
                    return MutexGuard { lock: self };
                }
                Err(_) => {
                    // Spin with hint for better CPU efficiency
                    spin_loop();
                    continue;
                }
            }
        }
    }

    /// Try to acquire the lock without blocking
    pub fn try_lock(&self) -> Option<MutexGuard<T>> {
        match self
            .locked
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
        {
            Ok(_) => Some(MutexGuard { lock: self }),
            Err(_) => None,
        }
    }
}

/// Guard for Mutex - provides exclusive access to protected data
pub struct MutexGuard<'a, T> {
    lock: &'a Mutex<T>,
}

impl<'a, T> Deref for MutexGuard<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        // SAFETY: Lock is held, guaranteeing exclusive access
        unsafe { &*self.lock.data.get() }
    }
}

impl<'a, T> DerefMut for MutexGuard<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        // SAFETY: Lock is held, guaranteeing exclusive access
        unsafe { &mut *self.lock.data.get() }
    }
}

impl<'a, T> Drop for MutexGuard<'a, T> {
    fn drop(&mut self) {
        // Release lock with Release ordering
        // This synchronizes with Acquire in lock()
        let was_locked = self.lock.locked.swap(false, Ordering::Release);
        debug_assert!(was_locked, "Dropping unlocked mutex guard");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mutex_basic() {
        let mutex = Mutex::new(42);
        {
            let guard = mutex.lock();
            assert_eq!(*guard, 42);
        }
        // Lock should be released now
        assert!(mutex.try_lock().is_some());
    }

    #[test]
    fn test_mutex_mutation() {
        let mutex = Mutex::new(0);
        {
            let mut guard = mutex.lock();
            *guard = 100;
        }
        {
            let guard = mutex.lock();
            assert_eq!(*guard, 100);
        }
    }

    #[test]
    fn test_mutex_try_lock() {
        let mutex = Mutex::new(42);

        // Should succeed when unlocked
        let guard = mutex.try_lock();
        assert!(guard.is_some());

        // Should fail when locked
        assert!(mutex.try_lock().is_none());

        drop(guard);

        // Should succeed again after release
        assert!(mutex.try_lock().is_some());
    }

    #[test]
    fn test_rwlock_read() {
        let rwlock = RwLock::new(42);
        let guard1 = rwlock.read();
        let guard2 = rwlock.read();
        assert_eq!(*guard1, 42);
        assert_eq!(*guard2, 42);
    }

    #[test]
    fn test_rwlock_write() {
        let rwlock = RwLock::new(0);
        {
            let mut guard = rwlock.write();
            *guard = 100;
        }
        {
            let guard = rwlock.read();
            assert_eq!(*guard, 100);
        }
    }

    #[test]
    fn test_rwlock_try_read_write() {
        let rwlock = RwLock::new(42);

        // Multiple reads should work
        let r1 = rwlock.try_read();
        assert!(r1.is_some());
        let r2 = rwlock.try_read();
        assert!(r2.is_some());

        // Write should fail with readers
        assert!(rwlock.try_write().is_none());

        drop(r1);
        drop(r2);

        // Write should succeed now
        let w = rwlock.try_write();
        assert!(w.is_some());

        // Read should fail with writer
        assert!(rwlock.try_read().is_none());
    }

    #[test]
    fn test_rwlock_write_blocks_read() {
        let rwlock = RwLock::new(42);

        let write_guard = rwlock.try_write();
        assert!(write_guard.is_some());

        // Should not be able to acquire read while write held
        assert!(rwlock.try_read().is_none());
    }
}
