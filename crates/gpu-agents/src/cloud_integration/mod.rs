//! Cloud Integration Module
//!
//! Provides cloud provider abstractions for AWS, GCP, and Alibaba Cloud
//! with auto-provisioning, multi-cloud orchestration, and cost optimization.

pub mod alibaba;
pub mod aws;
pub mod core;
pub mod cost_optimizer;
pub mod gcp;
pub mod provisioner;

#[cfg(test)]
mod tests;

// Re-export main types
pub use alibaba::AlibabaProvider;
pub use aws::AwsProvider;
pub use core::{CloudConfig, CloudProvider, CloudResource, InstanceType};
pub use cost_optimizer::{CostOptimizer, CostStrategy};
pub use gcp::GcpProvider;
pub use provisioner::{CloudProvisioner, ProvisioningRequest, ProvisioningResult};
