//! AWS cloud provider implementation for GPU capacity management.

pub mod config;
pub mod ec2;
pub mod pricing;
pub mod quota;
pub mod spot;

pub use config::AwsConfig;
pub use ec2::AwsProvider;

// Re-export common types from providerx
pub use hpc_provider::{
    Availability, CapacityProvider, HealthStatus, Instance, InstanceState, ProviderError,
    ProviderResult, ProvisionResult, ProvisionSpec, Quote, QuoteRequest, ServiceQuotas, SpotPrices,
};
