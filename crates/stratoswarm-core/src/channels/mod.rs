//! Channel infrastructure for stratoswarm inter-component communication.
//!
//! This module provides a comprehensive channel system for message passing between
//! different components in the stratoswarm system. It includes:
//!
//! - Strongly-typed message definitions for all component types
//! - Channel registry for centralized channel management
//! - Request/response patterns with timeout support
//! - Both point-to-point (mpsc) and broadcast communication patterns
//! - Zero-copy message passing using `bytes::Bytes`
//!
//! # Architecture
//!
//! The channel system is built on tokio's async channels and provides:
//!
//! - **Point-to-point channels** (mpsc): For directed communication to specific components
//!   - GPU commands
//!   - Evolution messages
//!   - Cost optimization requests
//!   - Efficiency queries
//!   - Scheduler tasks
//!   - Governor resource allocation
//!   - Knowledge graph operations
//!
//! - **Broadcast channels**: For events that need to reach multiple subscribers
//!   - System events (agent spawned, fitness improved, etc.)
//!
//! # Example
//!
//! ```rust
//! use stratoswarm_core::channels::registry::ChannelRegistry;
//! use stratoswarm_core::channels::messages::GpuCommand;
//!
//! #[tokio::main]
//! async fn main() {
//!     // Create the central registry
//!     let registry = ChannelRegistry::new();
//!
//!     // Get a GPU command sender
//!     let gpu_tx = registry.gpu_sender();
//!
//!     // Subscribe to GPU commands
//!     let mut gpu_rx = registry.subscribe_gpu();
//!
//!     // Send a synchronization command
//!     gpu_tx.send(GpuCommand::Synchronize { stream_id: None })
//!         .await
//!         .unwrap();
//!
//!     // Receive the command
//!     let cmd = gpu_rx.recv().await.unwrap();
//! }
//! ```

pub mod messages;
pub mod patterns;
pub mod registry;

// Re-export commonly used types
pub use messages::{
    CostMessage, EfficiencyMessage, EvolutionMessage, GovernorMessage, GpuCommand,
    KnowledgeMessage, SchedulerMessage, SystemEvent,
};
pub use patterns::{request_with_timeout, QueryResponse, Request, Responder};
pub use registry::ChannelRegistry;
