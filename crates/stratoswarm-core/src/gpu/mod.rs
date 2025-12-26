//! GPU service module for channel-based GPU communication.
//!
//! This module provides a GPU runtime service that processes commands through
//! channels, replacing the Mutex+VecDeque pattern with a more efficient
//! async channel-based approach.
//!
//! # Architecture
//!
//! - **GpuRuntime**: Main service that processes GPU commands from a channel
//! - **GpuDevice**: Trait abstraction for GPU operations (enables testing)
//! - **UnifiedMemoryPool**: Zero-copy memory management using `bytes::Bytes`
//! - **CommandProcessor**: Logic for processing different GPU commands
//!
//! # Example Usage
//!
//! ```rust
//! use stratoswarm_core::gpu::{GpuRuntime, GpuConfig, MockDevice};
//! use stratoswarm_core::channels::{ChannelRegistry, GpuCommand};
//! use bytes::Bytes;
//!
//! #[tokio::main]
//! async fn main() {
//!     // Create channel registry
//!     let registry = ChannelRegistry::new();
//!
//!     // Create GPU runtime with mock device for testing
//!     let config = GpuConfig::default();
//!     let device = MockDevice::new(0, 1024 * 1024 * 1024); // 1GB device
//!     let runtime = GpuRuntime::new(config, device);
//!
//!     // Get channels
//!     let gpu_rx = registry.subscribe_gpu();
//!     let events_tx = registry.event_sender();
//!     let gpu_tx = registry.gpu_sender();
//!
//!     // Spawn the GPU service
//!     tokio::spawn(async move {
//!         runtime.run(gpu_rx, events_tx).await
//!     });
//!
//!     // Send GPU commands via the mpsc sender
//!     let cmd = GpuCommand::LaunchKernel {
//!         kernel_id: "my_kernel".to_string(),
//!         grid_dim: (1, 1, 1),
//!         block_dim: (256, 1, 1),
//!         params: Bytes::from(vec![1, 2, 3, 4]),
//!     };
//!
//!     gpu_tx.send(cmd).await.expect("Failed to send GPU command");
//!
//!     // Subscribe to events
//!     let mut event_rx = registry.subscribe_events();
//!     if let Ok(event) = event_rx.recv().await {
//!         println!("Received event: {:?}", event);
//!     }
//! }
//! ```

pub mod command;
pub mod memory;
pub mod runtime;

// Re-export main types
pub use command::{CommandProcessor, GpuDevice, MockDevice};
pub use memory::{MemoryStats, UnifiedMemoryPool};
pub use runtime::{GpuConfig, GpuMetrics, GpuRuntime};

#[cfg(test)]
mod tests;
