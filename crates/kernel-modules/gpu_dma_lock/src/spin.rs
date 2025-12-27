//! Spin lock implementation for no_std kernel environment
//!
//! This module provides simple spinlock implementations for the kernel module.
//! In a real kernel module, these would be replaced with actual kernel spinlocks.

use core::cell::UnsafeCell;
use core::ops::{Deref, DerefMut};

/// Read-write lock
pub struct RwLock<T> {
    data: UnsafeCell<T>,
}

// SAFETY: RwLock<T> is Send because:
// 1. The `UnsafeCell<T>` only grants access through the lock guards
// 2. If T: Send, then transferring ownership of the lock is safe
// 3. The lock itself has no thread-local state
// 4. In a real kernel, this would use kernel spinlock primitives
// 5. Currently a no-op lock for kernel module skeleton
unsafe impl<T: Send> Send for RwLock<T> {}

// SAFETY: RwLock<T> is Sync because:
// 1. RwLock provides synchronized access to the inner data
// 2. Only one writer OR multiple readers can access at a time (semantically)
// 3. T: Send is required to ensure T can be accessed from any thread
// 4. NOTE: This is a placeholder implementation - in production kernel code,
//    actual spinlock synchronization must be implemented
// 5. Multiple threads can safely hold &RwLock because access is controlled
unsafe impl<T: Send> Sync for RwLock<T> {}

impl<T> RwLock<T> {
    pub const fn new(data: T) -> Self {
        Self {
            data: UnsafeCell::new(data),
        }
    }

    pub fn read(&self) -> RwLockReadGuard<T> {
        RwLockReadGuard { lock: self }
    }

    pub fn write(&self) -> RwLockWriteGuard<T> {
        RwLockWriteGuard { lock: self }
    }
}

pub struct RwLockReadGuard<'a, T> {
    lock: &'a RwLock<T>,
}

impl<'a, T> Deref for RwLockReadGuard<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        unsafe { &*self.lock.data.get() }
    }
}

pub struct RwLockWriteGuard<'a, T> {
    lock: &'a RwLock<T>,
}

impl<'a, T> Deref for RwLockWriteGuard<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        unsafe { &*self.lock.data.get() }
    }
}

impl<'a, T> DerefMut for RwLockWriteGuard<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { &mut *self.lock.data.get() }
    }
}

/// Mutex lock
pub struct Mutex<T> {
    data: UnsafeCell<T>,
}

// SAFETY: Mutex<T> is Send because:
// 1. The `UnsafeCell<T>` only grants access through the MutexGuard
// 2. If T: Send, then transferring ownership of the lock is safe
// 3. The lock itself has no thread-local state
// 4. In a real kernel, this would use kernel spinlock primitives
// 5. Currently a no-op lock for kernel module skeleton
unsafe impl<T: Send> Send for Mutex<T> {}

// SAFETY: Mutex<T> is Sync because:
// 1. Mutex provides exclusive synchronized access to the inner data
// 2. Only one thread can hold the lock at a time (semantically)
// 3. T: Send is required to ensure T can be accessed from any thread
// 4. NOTE: This is a placeholder implementation - in production kernel code,
//    actual spinlock synchronization must be implemented
// 5. Multiple threads can safely hold &Mutex because access is controlled
unsafe impl<T: Send> Sync for Mutex<T> {}

impl<T> Mutex<T> {
    pub const fn new(data: T) -> Self {
        Self {
            data: UnsafeCell::new(data),
        }
    }

    pub fn lock(&self) -> MutexGuard<T> {
        MutexGuard { lock: self }
    }
}

pub struct MutexGuard<'a, T> {
    lock: &'a Mutex<T>,
}

impl<'a, T> Deref for MutexGuard<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        unsafe { &*self.lock.data.get() }
    }
}

impl<'a, T> DerefMut for MutexGuard<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { &mut *self.lock.data.get() }
    }
}
