//! Background Workers
//!
//! Periodic tasks for ephemeral identity management:
//! - Identity expiration processing
//! - Invitation cleanup
//! - Token revocation list maintenance

mod cleanup_worker;

pub use cleanup_worker::{CleanupStats, CleanupWorker, CleanupWorkerConfig, CleanupWorkerHandle};
