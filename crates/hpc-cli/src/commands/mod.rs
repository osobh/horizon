//! Command modules for each HPC-AI component.

// Project commands
mod argus;
mod channels;
mod nebula;
mod parcode;
mod rmpi;
mod rnccl;
mod rustg;
mod slai;
mod spark;
mod stratoswarm;
mod torch;
mod vortex;
mod warp;

// Deployment and lifecycle commands
mod deploy;
mod inventory;
pub mod picker;
mod stack;

pub use argus::ArgusCommands;
pub use channels::ChannelsCommands;
pub use deploy::DeployCommands;
pub use inventory::InventoryCommands;
pub use nebula::NebulaCommands;
pub use parcode::ParcodeCommands;
pub use rmpi::RmpiCommands;
pub use rnccl::RncclCommands;
pub use rustg::RustgCommands;
pub use slai::SlaiCommands;
pub use spark::SparkCommands;
pub use stack::StackCommands;
pub use stratoswarm::SwarmCommands;
pub use torch::TorchCommands;
pub use vortex::VortexCommands;
pub use warp::WarpCommands;
