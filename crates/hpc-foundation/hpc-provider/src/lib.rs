pub mod error;
pub mod testing;
pub mod traits;
pub mod types;

pub use error::{ProviderError, ProviderResult};
pub use traits::CapacityProvider;
pub use types::{
    Availability, HealthStatus, Instance, InstanceState, ProvisionResult, ProvisionSpec, Quote,
    QuoteRequest, ServiceQuotas, SpotPrices,
};
