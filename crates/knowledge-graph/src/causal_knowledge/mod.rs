//! Causal Knowledge Chain Inference Engine
//!
//! GPU-accelerated causal inference system for discovering, analyzing, and reasoning about
//! causal relationships within knowledge graphs. Provides advanced capabilities for:
//! - Real-time causal relationship discovery
//! - Multi-hop causal chain inference with uncertainty quantification
//! - GPU-accelerated causal graph neural networks
//! - Temporal causal analysis and counterfactual reasoning
//! - Distributed causal consistency across knowledge graphs

pub mod config;
pub mod types;
pub mod evidence;
pub mod chains;
pub mod gpu;
pub mod graphs;
pub mod cache;
pub mod temporal;
pub mod patterns;
pub mod engine;

pub use config::*;
pub use types::*;
pub use evidence::*;
pub use chains::*;
pub use gpu::*;
pub use graphs::*;
pub use cache::*;
pub use temporal::*;
pub use patterns::*;
pub use engine::*;
