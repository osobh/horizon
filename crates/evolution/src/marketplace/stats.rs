//! Marketplace statistics and metrics

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketplaceStats {
    pub total_packages: u64,
    pub total_clusters: u32,
    pub total_transfers: u64,
    pub successful_validations: u64,
    pub failed_validations: u64,
    pub average_package_rating: f64,
    pub total_download_count: u64,
    pub average_consensus_time_ms: u64,
    pub network_bandwidth_usage_gb: f64,
    pub storage_usage_gb: f64,
}

impl Default for MarketplaceStats {
    fn default() -> Self {
        Self {
            total_packages: 0,
            total_clusters: 0,
            total_transfers: 0,
            successful_validations: 0,
            failed_validations: 0,
            average_package_rating: 0.0,
            total_download_count: 0,
            average_consensus_time_ms: 0,
            network_bandwidth_usage_gb: 0.0,
            storage_usage_gb: 0.0,
        }
    }
}

impl MarketplaceStats {
    pub fn update_validation(&mut self, success: bool) {
        if success {
            self.successful_validations += 1;
        } else {
            self.failed_validations += 1;
        }
    }
    
    pub fn update_transfer(&mut self, bytes: u64) {
        self.total_transfers += 1;
        self.network_bandwidth_usage_gb += bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    }
    
    pub fn validation_success_rate(&self) -> f64 {
        let total = self.successful_validations + self.failed_validations;
        if total == 0 {
            0.0
        } else {
            self.successful_validations as f64 / total as f64
        }
    }
}