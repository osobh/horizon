//! Storage module for GPU agents
//!
//! Provides high-performance storage capabilities including GPUDirect Storage support
//! for direct GPU-to-storage data transfers.

mod gpudirect;
#[cfg(test)]
mod gpudirect_tests;
#[cfg(test)]
mod storage_tests;

pub use gpudirect::{
    GpuDirectConfig, GpuDirectManager, GpuIoBuffer, GdsBatchOperation,
    IoResult, GdsAvailabilityChecker,
};

// Note: Storage types are defined in the parent storage.rs file