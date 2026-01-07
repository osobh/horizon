//! HPC Inventory - Shared inventory types and storage for HPC-AI platform
//!
//! This crate provides unified node inventory management types and storage
//! backends for the HPC-AI platform. It enables sharing inventory data
//! between the CLI and web services.
//!
//! # Features
//!
//! - `postgres` - Enable PostgreSQL storage backend (requires `sqlx`)
//!
//! # Example
//!
//! ```no_run
//! use hpc_inventory::{NodeInfo, NodeMode, CredentialRef, InMemoryStore, InventoryStore};
//!
//! #[tokio::main]
//! async fn main() -> hpc_inventory::Result<()> {
//!     let mut store = InMemoryStore::new()?;
//!
//!     let node = NodeInfo::new(
//!         "gpu-node-1".to_string(),
//!         "192.168.1.100".to_string(),
//!         22,
//!         "admin".to_string(),
//!         CredentialRef::SshAgent,
//!         NodeMode::Docker,
//!     );
//!
//!     store.add_node(node).await?;
//!
//!     let nodes = store.list_nodes().await?;
//!     println!("Total nodes: {}", nodes.len());
//!
//!     Ok(())
//! }
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

pub mod error;
pub mod node;
pub mod store;

// Re-export commonly used types
pub use error::{InventoryError, Result};
pub use node::{
    Architecture, CredentialRef, GpuInfo, HardwareProfile, InventorySummary, NodeInfo, NodeMode,
    NodeStatus, OsType,
};
pub use store::{InMemoryStore, InventoryData, InventoryStore};
