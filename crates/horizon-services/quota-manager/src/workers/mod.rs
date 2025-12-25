//! Background Workers
//!
//! Periodic tasks for quota and pool management:
//! - Expiry processing for ephemeral quotas
//! - Pool allocation cleanup
//! - Usage statistics aggregation

mod expiry_worker;

pub use expiry_worker::ExpiryWorker;
