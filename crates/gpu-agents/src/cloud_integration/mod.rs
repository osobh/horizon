//! Cloud Integration Module
//!
//! Provides cloud provider abstractions for AWS, GCP, and Alibaba Cloud
//! with auto-provisioning, multi-cloud orchestration, and cost optimization.

pub mod aws;
pub mod gcp;
pub mod alibaba;
pub mod core;
pub mod provisioner;
pub mod cost_optimizer;

#[cfg(test)]
mod tests;

// Re-export main types
pub use core::{CloudProvider, CloudResource, InstanceType, CloudConfig};
pub use provisioner::{CloudProvisioner, ProvisioningRequest, ProvisioningResult};
pub use cost_optimizer::{CostOptimizer, CostStrategy};
pub use aws::AwsProvider;
pub use gcp::GcpProvider;
pub use alibaba::AlibabaProvider;