//! Stream sink implementations

pub mod memory;
pub mod network;
pub mod storage;

pub use memory::MemoryStreamSink;
pub use storage::{StorageBackend, StorageStreamSink};
