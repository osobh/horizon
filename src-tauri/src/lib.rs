//! Horizon Library
//!
//! Core library for the Horizon unified HPC-AI platform.

pub mod cluster_bridge;
pub mod commands;
pub mod data_pipeline_bridge;
pub mod edge_proxy_bridge;
pub mod evolution_bridge;
pub mod gpu_compiler_bridge;
pub mod kernel_bridge;
pub mod nebula_bridge;
pub mod state;
pub mod storage_bridge;
pub mod tensor_mesh_bridge;
pub mod training_bridge;

pub use cluster_bridge::ClusterBridge;
pub use data_pipeline_bridge::DataPipelineBridge;
pub use edge_proxy_bridge::EdgeProxyBridge;
pub use evolution_bridge::EvolutionBridge;
pub use gpu_compiler_bridge::GpuCompilerBridge;
pub use kernel_bridge::KernelBridge;
pub use nebula_bridge::NebulaBridge;
pub use state::AppState;
pub use storage_bridge::StorageBridge;
pub use tensor_mesh_bridge::TensorMeshBridge;
pub use training_bridge::TrainingBridge;
