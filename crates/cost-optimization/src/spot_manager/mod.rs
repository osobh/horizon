//! Spot instance management for cost optimization

mod bidding;
mod config;
mod manager;
mod market;
mod types;

pub use bidding::{BiddingStrategy, BidCalculator};
pub use config::SpotManagerConfig;
pub use manager::SpotManager;
pub use market::{SpotMarketAnalysis, SpotPrice};
pub use types::{FallbackStrategy, SpotInstance, SpotInstanceRequest, SpotInstanceState};
