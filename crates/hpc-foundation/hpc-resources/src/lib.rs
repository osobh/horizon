// ResourceX - Universal resource abstraction for Horizon capacity planning
//
// This crate provides a vendor-agnostic, extensible resource model that supports:
// - Multi-resource requests (CPU, GPU, TPU, Memory, Storage, Network, Custom)
// - Vendor-specific constraints (Nvidia, AMD, Intel, Apple GPUs; Google TPUs)
// - Cloud provider discovery and pricing abstractions
// - Resource allocation tracking with audit trails
//
// Key modules:
// - types: Core resource type system and specifications
// - request: Resource request builder with multi-resource support
// - allocation: Resource allocation and assignment tracking
// - provider: Cloud provider discovery and pricing traits

pub mod allocation;
pub mod provider;
pub mod request;
pub mod types;

// Re-export commonly used types for convenience
pub use allocation::{ResourceAllocation, ResourceAssignment};
pub use provider::{
    AvailableResource, BillingModel, CloudProvider, CommitmentTerm, PricingError, PricingInfo,
    ResourceDiscovery, ResourceDiscoveryError, ResourcePricing, SpotPricePoint,
};
pub use request::{RequestPriority, ResourceRequest};
pub use types::{
    ComputeType, GpuVendor, NetworkType, ResourceConstraints, ResourceSpec, ResourceType,
    ResourceUnit, StorageType, TpuVariant,
};
