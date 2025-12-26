//! Stratoswarm Core - High-performance channel infrastructure for distributed agent systems.
//!
//! This crate provides the foundational communication infrastructure for the stratoswarm
//! distributed agent system. It implements a zero-copy, type-safe channel architecture
//! for inter-component communication with support for:
//!
//! - Point-to-point messaging (mpsc)
//! - Broadcast messaging for events
//! - Request/response patterns with configurable timeouts
//! - Backpressure control for GPU operations
//! - Zero-copy data transfer using `bytes::Bytes`
//!
//! # Architecture
//!
//! The core abstraction is the [`ChannelRegistry`](channels::ChannelRegistry), which manages
//! all system channels and provides strongly-typed access to senders and receivers.
//!
//! Channel types are organized by their communication pattern:
//!
//! ## Point-to-Point Channels (mpsc)
//!
//! These channels provide directed communication to specific components:
//!
//! - **GPU Channel** (buffer: 100): GPU commands with backpressure control
//! - **Evolution Channel** (buffer: 10000): High-throughput evolutionary algorithm messages
//! - **Cost Channel** (buffer: 1000): Cost optimization requests/responses
//! - **Efficiency Channel** (buffer: 1000): Efficiency intelligence queries
//! - **Scheduler Channel** (buffer: 1000): Task scheduling operations
//! - **Governor Channel** (buffer: 1000): Resource allocation governance
//! - **Knowledge Channel** (buffer: 1000): Knowledge graph operations
//!
//! ## Broadcast Channels
//!
//! These channels distribute events to multiple subscribers:
//!
//! - **Events Channel** (buffer: 1000): System events (agent spawned, fitness improved, etc.)
//!
//! # Example Usage
//!
//! ```rust
//! use stratoswarm_core::channels::{ChannelRegistry, GpuCommand};
//! use bytes::Bytes;
//!
//! #[tokio::main]
//! async fn main() {
//!     // Create the registry
//!     let registry = ChannelRegistry::new();
//!
//!     // Get senders and receivers
//!     let gpu_tx = registry.gpu_sender();
//!     let mut gpu_rx = registry.subscribe_gpu();
//!
//!     // Send a GPU command with zero-copy data
//!     let data = Bytes::from(vec![1, 2, 3, 4]);
//!     let cmd = GpuCommand::TransferToDevice {
//!         buffer_id: "my_buffer".to_string(),
//!         data,
//!         offset: 0,
//!     };
//!
//!     gpu_tx.send(cmd).await.unwrap();
//!
//!     // Receive the command
//!     let received = gpu_rx.recv().await.unwrap();
//! }
//! ```
//!
//! # Request/Response Pattern
//!
//! ```rust,no_run
//! use stratoswarm_core::channels::{ChannelRegistry, CostMessage, request_with_timeout};
//! use std::time::Duration;
//!
//! #[tokio::main]
//! async fn main() {
//!     let registry = ChannelRegistry::new();
//!     let cost_tx = registry.cost_sender();
//!
//!     // Send a request with timeout
//!     let response = request_with_timeout(
//!         &cost_tx,
//!         CostMessage::QueryCost,
//!         Duration::from_secs(5)
//!     ).await.unwrap();
//! }
//! ```
//!
//! # Performance Characteristics
//!
//! - **Zero-copy transfers**: Uses `bytes::Bytes` for efficient data sharing
//! - **Bounded channels**: Provides backpressure to prevent memory exhaustion
//! - **Lock-free**: Built on tokio's lock-free async channels
//! - **Type-safe**: Compile-time verification of message types
//!
//! # Safety
//!
//! This crate contains no unsafe code and relies entirely on Rust's type system
//! and tokio's well-tested async primitives for safety guarantees.

#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

pub mod channels;
pub mod error;
pub mod evolution;
pub mod gpu;
pub mod priority_queue;

#[cfg(test)]
mod tests;

// Re-export main types for convenience
pub use channels::ChannelRegistry;
pub use error::{ChannelError, Result};
pub use evolution::{EvolutionConfig, EvolutionService};
pub use gpu::{GpuConfig, GpuMetrics, GpuRuntime};
pub use priority_queue::{PrioritySchedulerQueue, SchedulerPriority};
