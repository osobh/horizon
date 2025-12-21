//! Horizon Library
//!
//! Core library for the Horizon unified HPC-AI platform.

pub mod cluster_bridge;
pub mod commands;
pub mod gpu_compiler_bridge;
pub mod kernel_bridge;
pub mod state;
pub mod storage_bridge;
pub mod training_bridge;

pub use cluster_bridge::ClusterBridge;
pub use gpu_compiler_bridge::GpuCompilerBridge;
pub use kernel_bridge::KernelBridge;
pub use state::AppState;
pub use storage_bridge::StorageBridge;
pub use training_bridge::TrainingBridge;
