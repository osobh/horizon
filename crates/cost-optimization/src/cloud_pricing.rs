//! Cloud pricing module for multi-cloud cost calculation and provider comparison
//!
//! This module provides comprehensive cloud pricing capabilities including:
//! - Multi-cloud pricing models (AWS, Azure, GCP, OCI)
//! - Real-time pricing data retrieval and caching
//! - Cost calculation with regional variations
//! - Provider comparison and arbitrage opportunities

use crate::error::{CostOptimizationError, CostOptimizationResult};
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use ordered_float::OrderedFloat;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;
use tracing::{debug, error, info, warn};

/// Cloud provider enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CloudProvider {
    /// Amazon Web Services
    Aws,
    /// Microsoft Azure
    Azure,
    /// Google Cloud Platform
    Gcp,
    /// Oracle Cloud Infrastructure
    Oci,
    /// Other cloud provider
    Other,
}

impl std::fmt::Display for CloudProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CloudProvider::Aws => write!(f, "AWS"),
            CloudProvider::Azure => write!(f, "Azure"),
            CloudProvider::Gcp => write!(f, "GCP"),
            CloudProvider::Oci => write!(f, "OCI"),
            CloudProvider::Other => write!(f, "Other"),
        }
    }
}

/// Instance type classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InstanceClass {
    /// General purpose instances
    GeneralPurpose,
    /// Compute optimized instances
    ComputeOptimized,
    /// Memory optimized instances
    MemoryOptimized,
    /// GPU instances
    GpuAccelerated,
    /// Storage optimized instances
    StorageOptimized,
    /// Burstable instances
    Burstable,
}

/// Pricing model type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PricingModel {
    /// On-demand pricing
    OnDemand,
    /// Reserved instances (1 year)
    Reserved1Year,
    /// Reserved instances (3 year)
    Reserved3Year,
    /// Spot/preemptible instances
    Spot,
    /// Savings plans
    SavingsPlan,
}

/// Instance pricing information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstancePricing {
    /// Provider
    pub provider: CloudProvider,
    /// Region
    pub region: String,
    /// Instance type (e.g., "p4d.24xlarge", "Standard_NC96ads_A100_v4")
    pub instance_type: String,
    /// Instance class
    pub instance_class: InstanceClass,
    /// vCPUs
    pub vcpus: u32,
    /// Memory in GB
    pub memory_gb: f64,
    /// GPU count
    pub gpu_count: u32,
    /// GPU model
    pub gpu_model: Option<String>,
    /// Storage in GB
    pub storage_gb: f64,
    /// Network performance
    pub network_performance: String,
    /// Pricing by model
    pub pricing: HashMap<PricingModel, PriceInfo>,
    /// Last updated
    pub last_updated: DateTime<Utc>,
}

/// Price information with currency
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceInfo {
    /// Price per hour
    pub hourly_rate: f64,
    /// Currency (USD, EUR, etc.)
    pub currency: String,
    /// Minimum commitment hours
    pub min_commitment: Option<u32>,
    /// Upfront cost
    pub upfront_cost: Option<f64>,
    /// Discount percentage from on-demand
    pub discount_percent: f64,
}

/// Regional pricing data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionalPricing {
    /// Provider
    pub provider: CloudProvider,
    /// Region code
    pub region: String,
    /// Region display name
    pub region_name: String,
    /// Data transfer costs
    pub data_transfer: DataTransferPricing,
    /// Storage costs
    pub storage: StoragePricing,
    /// Additional services
    pub additional_services: HashMap<String, f64>,
}

/// Data transfer pricing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataTransferPricing {
    /// Inbound data transfer (per GB)
    pub inbound_per_gb: f64,
    /// Outbound data transfer within region (per GB)
    pub outbound_same_region_per_gb: f64,
    /// Outbound data transfer to internet (per GB)
    pub outbound_internet_per_gb: f64,
    /// Outbound data transfer to other regions (per GB)
    pub outbound_cross_region_per_gb: f64,
}

/// Storage pricing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoragePricing {
    /// Standard storage (per GB per month)
    pub standard_per_gb_month: f64,
    /// SSD storage (per GB per month)
    pub ssd_per_gb_month: f64,
    /// Archive storage (per GB per month)
    pub archive_per_gb_month: f64,
    /// IOPS pricing
    pub iops_per_thousand: f64,
}

/// Cost calculation request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostCalculationRequest {
    /// Provider
    pub provider: CloudProvider,
    /// Region
    pub region: String,
    /// Instance type
    pub instance_type: String,
    /// Instance count
    pub instance_count: u32,
    /// Duration in hours
    pub duration_hours: f64,
    /// Pricing model
    pub pricing_model: PricingModel,
    /// Storage requirements in GB
    pub storage_gb: f64,
    /// Data transfer estimates
    pub data_transfer: DataTransferEstimate,
}

/// Data transfer estimate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataTransferEstimate {
    /// Inbound GB
    pub inbound_gb: f64,
    /// Outbound to internet GB
    pub outbound_internet_gb: f64,
    /// Outbound to same region GB
    pub outbound_same_region_gb: f64,
    /// Outbound to other regions GB
    pub outbound_cross_region_gb: f64,
}

/// Cost calculation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostCalculationResult {
    /// Request details
    pub request: CostCalculationRequest,
    /// Compute cost
    pub compute_cost: f64,
    /// Storage cost
    pub storage_cost: f64,
    /// Data transfer cost
    pub data_transfer_cost: f64,
    /// Total cost
    pub total_cost: f64,
    /// Cost breakdown
    pub breakdown: CostBreakdown,
    /// Savings compared to on-demand
    pub savings_vs_ondemand: f64,
    /// Alternative recommendations
    pub recommendations: Vec<CostRecommendation>,
}

/// Detailed cost breakdown
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostBreakdown {
    /// Hourly compute rate
    pub hourly_rate: f64,
    /// Total compute hours
    pub total_hours: f64,
    /// Upfront costs
    pub upfront_cost: f64,
    /// Storage cost per month
    pub storage_monthly: f64,
    /// Data transfer breakdown
    pub data_transfer_breakdown: HashMap<String, f64>,
}

/// Cost optimization recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostRecommendation {
    /// Recommendation type
    pub recommendation_type: RecommendationType,
    /// Description
    pub description: String,
    /// Potential savings
    pub potential_savings: f64,
    /// Savings percentage
    pub savings_percent: f64,
    /// Implementation effort
    pub effort: ImplementationEffort,
    /// Alternative configuration
    pub alternative: Option<CostCalculationRequest>,
}

/// Recommendation type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecommendationType {
    /// Use different instance type
    InstanceTypeChange,
    /// Use different pricing model
    PricingModelChange,
    /// Use different region
    RegionChange,
    /// Use different provider
    ProviderChange,
    /// Optimize resource allocation
    ResourceOptimization,
}

/// Implementation effort level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImplementationEffort {
    /// Low effort
    Low,
    /// Medium effort
    Medium,
    /// High effort
    High,
}

/// Cloud pricing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudPricingConfig {
    /// Enable real-time pricing updates
    pub enable_realtime_updates: bool,
    /// Pricing cache duration
    pub cache_duration: Duration,
    /// Default currency
    pub default_currency: String,
    /// API endpoints by provider
    pub api_endpoints: HashMap<CloudProvider, String>,
    /// API keys by provider
    pub api_keys: HashMap<CloudProvider, String>,
}

impl Default for CloudPricingConfig {
    fn default() -> Self {
        let mut api_endpoints = HashMap::new();
        api_endpoints.insert(
            CloudProvider::Aws,
            "https://pricing.us-east-1.amazonaws.com".to_string(),
        );
        api_endpoints.insert(
            CloudProvider::Azure,
            "https://prices.azure.com/api/retail/prices".to_string(),
        );
        api_endpoints.insert(
            CloudProvider::Gcp,
            "https://cloudbilling.googleapis.com/v1/services".to_string(),
        );

        Self {
            enable_realtime_updates: true,
            cache_duration: Duration::from_secs(3600), // 1 hour
            default_currency: "USD".to_string(),
            api_endpoints,
            api_keys: HashMap::new(),
        }
    }
}

/// Cloud pricing service for multi-cloud cost management
pub struct CloudPricingService {
    /// Configuration
    config: Arc<CloudPricingConfig>,
    /// Pricing cache by provider and region
    pricing_cache: Arc<DashMap<(CloudProvider, String), Vec<InstancePricing>>>,
    /// Regional pricing cache
    regional_cache: Arc<DashMap<(CloudProvider, String), RegionalPricing>>,
    /// Last cache update times
    cache_timestamps: Arc<DashMap<(CloudProvider, String), DateTime<Utc>>>,
    /// Update mutex to prevent concurrent updates
    update_mutex: Arc<Mutex<()>>,
}

impl CloudPricingService {
    /// Create a new cloud pricing service
    pub fn new(config: CloudPricingConfig) -> CostOptimizationResult<Self> {
        Ok(Self {
            config: Arc::new(config),
            pricing_cache: Arc::new(DashMap::new()),
            regional_cache: Arc::new(DashMap::new()),
            cache_timestamps: Arc::new(DashMap::new()),
            update_mutex: Arc::new(Mutex::new(())),
        })
    }

    /// Get instance pricing for a provider and region
    pub async fn get_instance_pricing(
        &self,
        provider: CloudProvider,
        region: &str,
    ) -> CostOptimizationResult<Vec<InstancePricing>> {
        // Check cache first
        if let Some(cached) = self.get_cached_pricing(provider, region) {
            return Ok(cached);
        }

        // Update pricing data
        self.update_pricing_data(provider, region).await?;

        // Return from cache
        self.get_cached_pricing(provider, region).ok_or_else(|| {
            CostOptimizationError::PricingUnavailable {
                provider: provider.to_string(),
                region: region.to_string(),
            }
        })
    }

    /// Calculate costs for a workload
    pub async fn calculate_cost(
        &self,
        request: CostCalculationRequest,
    ) -> CostOptimizationResult<CostCalculationResult> {
        info!(
            "Calculating cost for {} in {}",
            request.instance_type, request.region
        );

        // Get instance pricing
        let pricing_data = self
            .get_instance_pricing(request.provider, &request.region)
            .await?;

        let instance_pricing = pricing_data
            .iter()
            .find(|p| p.instance_type == request.instance_type)
            .ok_or_else(|| CostOptimizationError::PricingUnavailable {
                provider: request.provider.to_string(),
                region: format!("{}/{}", request.region, request.instance_type),
            })?;

        // Get regional pricing
        let regional_pricing = self
            .get_regional_pricing(request.provider, &request.region)
            .await?;

        // Calculate compute cost
        let price_info = instance_pricing
            .pricing
            .get(&request.pricing_model)
            .ok_or_else(|| CostOptimizationError::CalculationError {
                details: format!("Pricing model {:?} not available", request.pricing_model),
            })?;

        let compute_cost =
            price_info.hourly_rate * request.duration_hours * request.instance_count as f64;
        let upfront_cost = price_info.upfront_cost.unwrap_or(0.0) * request.instance_count as f64;

        // Calculate storage cost
        let storage_months = request.duration_hours / 730.0; // Average hours per month
        let storage_cost =
            regional_pricing.storage.ssd_per_gb_month * request.storage_gb * storage_months;

        // Calculate data transfer cost
        let data_transfer_cost = self
            .calculate_data_transfer_cost(&regional_pricing.data_transfer, &request.data_transfer);

        let total_cost = compute_cost + upfront_cost + storage_cost + data_transfer_cost;

        // Calculate savings vs on-demand
        let on_demand_rate = instance_pricing
            .pricing
            .get(&PricingModel::OnDemand)
            .map(|p| p.hourly_rate)
            .unwrap_or(price_info.hourly_rate);

        let on_demand_cost =
            on_demand_rate * request.duration_hours * request.instance_count as f64;
        let savings_vs_ondemand = on_demand_cost - compute_cost;

        // Generate recommendations
        let recommendations = self
            .generate_recommendations(&request, &instance_pricing, total_cost)
            .await?;

        Ok(CostCalculationResult {
            request: request.clone(),
            compute_cost: compute_cost + upfront_cost,
            storage_cost,
            data_transfer_cost,
            total_cost,
            breakdown: CostBreakdown {
                hourly_rate: price_info.hourly_rate,
                total_hours: request.duration_hours * request.instance_count as f64,
                upfront_cost,
                storage_monthly: regional_pricing.storage.ssd_per_gb_month * request.storage_gb,
                data_transfer_breakdown: HashMap::from([
                    (
                        "inbound".to_string(),
                        regional_pricing.data_transfer.inbound_per_gb
                            * request.data_transfer.inbound_gb,
                    ),
                    (
                        "outbound_internet".to_string(),
                        regional_pricing.data_transfer.outbound_internet_per_gb
                            * request.data_transfer.outbound_internet_gb,
                    ),
                    (
                        "outbound_same_region".to_string(),
                        regional_pricing.data_transfer.outbound_same_region_per_gb
                            * request.data_transfer.outbound_same_region_gb,
                    ),
                    (
                        "outbound_cross_region".to_string(),
                        regional_pricing.data_transfer.outbound_cross_region_per_gb
                            * request.data_transfer.outbound_cross_region_gb,
                    ),
                ]),
            },
            savings_vs_ondemand,
            recommendations,
        })
    }

    /// Compare costs across providers
    pub async fn compare_providers(
        &self,
        instance_class: InstanceClass,
        vcpus: u32,
        memory_gb: f64,
        gpu_count: u32,
        regions: Vec<(CloudProvider, String)>,
        duration_hours: f64,
    ) -> CostOptimizationResult<Vec<ProviderComparison>> {
        let mut comparisons = Vec::new();

        for (provider, region) in regions {
            let pricing_data = self.get_instance_pricing(provider, &region).await?;

            // Find matching instances
            let matching_instances: Vec<_> = pricing_data
                .iter()
                .filter(|p| {
                    p.instance_class == instance_class
                        && p.vcpus >= vcpus
                        && p.memory_gb >= memory_gb
                        && p.gpu_count >= gpu_count
                })
                .collect();

            if matching_instances.is_empty() {
                continue;
            }

            // Find best price
            let best_instance = matching_instances
                .iter()
                .min_by(|a, b| {
                    let a_price = a
                        .pricing
                        .get(&PricingModel::OnDemand)
                        .map(|p| p.hourly_rate)
                        .unwrap_or(f64::MAX);
                    let b_price = b
                        .pricing
                        .get(&PricingModel::OnDemand)
                        .map(|p| p.hourly_rate)
                        .unwrap_or(f64::MAX);
                    a_price.partial_cmp(&b_price).unwrap()
                })
                .unwrap();

            comparisons.push(ProviderComparison {
                provider,
                region: region.clone(),
                instance_type: best_instance.instance_type.clone(),
                hourly_rate: best_instance
                    .pricing
                    .get(&PricingModel::OnDemand)
                    .map(|p| p.hourly_rate)
                    .unwrap_or(0.0),
                total_cost: best_instance
                    .pricing
                    .get(&PricingModel::OnDemand)
                    .map(|p| p.hourly_rate * duration_hours)
                    .unwrap_or(0.0),
                specifications: InstanceSpecs {
                    vcpus: best_instance.vcpus,
                    memory_gb: best_instance.memory_gb,
                    gpu_count: best_instance.gpu_count,
                    gpu_model: best_instance.gpu_model.clone(),
                },
            });
        }

        // Sort by total cost
        comparisons.sort_by(|a, b| a.total_cost.partial_cmp(&b.total_cost).unwrap());

        Ok(comparisons)
    }

    /// Get cached pricing data
    fn get_cached_pricing(
        &self,
        provider: CloudProvider,
        region: &str,
    ) -> Option<Vec<InstancePricing>> {
        let key = (provider, region.to_string());

        // Check cache age
        if let Some(timestamp) = self.cache_timestamps.get(&key) {
            if Utc::now() - *timestamp.value()
                > chrono::Duration::from_std(self.config.cache_duration)?
            {
                return None; // Cache expired
            }
        }

        self.pricing_cache
            .get(&key)
            .map(|entry| entry.value().clone())
    }

    /// Update pricing data from provider
    async fn update_pricing_data(
        &self,
        provider: CloudProvider,
        region: &str,
    ) -> CostOptimizationResult<()> {
        let _lock = self.update_mutex.lock().await;

        info!("Updating pricing data for {} in {}", provider, region);

        // In a real implementation, this would call provider APIs
        // For now, we'll use mock data
        let pricing_data = self.mock_pricing_data(provider, region);
        let regional_data = self.mock_regional_data(provider, region);

        let key = (provider, region.to_string());
        self.pricing_cache.insert(key.clone(), pricing_data);
        self.regional_cache.insert(key.clone(), regional_data);
        self.cache_timestamps.insert(key, Utc::now());

        Ok(())
    }

    /// Get regional pricing data
    async fn get_regional_pricing(
        &self,
        provider: CloudProvider,
        region: &str,
    ) -> CostOptimizationResult<RegionalPricing> {
        let key = (provider, region.to_string());

        if let Some(cached) = self.regional_cache.get(&key) {
            return Ok(cached.value().clone());
        }

        self.update_pricing_data(provider, region).await?;

        self.regional_cache
            .get(&key)
            .map(|entry| entry.value().clone())
            .ok_or_else(|| CostOptimizationError::PricingUnavailable {
                provider: provider.to_string(),
                region: region.to_string(),
            })
    }

    /// Calculate data transfer cost
    fn calculate_data_transfer_cost(
        &self,
        pricing: &DataTransferPricing,
        estimate: &DataTransferEstimate,
    ) -> f64 {
        pricing.inbound_per_gb * estimate.inbound_gb
            + pricing.outbound_internet_per_gb * estimate.outbound_internet_gb
            + pricing.outbound_same_region_per_gb * estimate.outbound_same_region_gb
            + pricing.outbound_cross_region_per_gb * estimate.outbound_cross_region_gb
    }

    /// Generate cost optimization recommendations
    async fn generate_recommendations(
        &self,
        request: &CostCalculationRequest,
        instance_pricing: &InstancePricing,
        current_cost: f64,
    ) -> CostOptimizationResult<Vec<CostRecommendation>> {
        let mut recommendations = Vec::new();

        // Check for better pricing models
        for (model, price_info) in &instance_pricing.pricing {
            if model != &request.pricing_model {
                let alt_cost =
                    price_info.hourly_rate * request.duration_hours * request.instance_count as f64
                        + price_info.upfront_cost.unwrap_or(0.0) * request.instance_count as f64;

                if alt_cost < current_cost {
                    let savings = current_cost - alt_cost;
                    recommendations.push(CostRecommendation {
                        recommendation_type: RecommendationType::PricingModelChange,
                        description: format!("Switch to {:?} pricing model", model),
                        potential_savings: savings,
                        savings_percent: (savings / current_cost) * 100.0,
                        effort: match model {
                            PricingModel::Spot => ImplementationEffort::High,
                            PricingModel::Reserved1Year | PricingModel::Reserved3Year => {
                                ImplementationEffort::Medium
                            }
                            _ => ImplementationEffort::Low,
                        },
                        alternative: Some(CostCalculationRequest {
                            pricing_model: *model,
                            ..request.clone()
                        }),
                    });
                }
            }
        }

        // Check for instance right-sizing
        let pricing_data = self
            .get_instance_pricing(request.provider, &request.region)
            .await?;

        for alt_instance in &pricing_data {
            if alt_instance.instance_type != instance_pricing.instance_type &&
               alt_instance.vcpus >= instance_pricing.vcpus * 8 / 10 && // Allow 20% less vCPUs
               alt_instance.memory_gb >= instance_pricing.memory_gb * 0.8 && // Allow 20% less memory
               alt_instance.gpu_count == instance_pricing.gpu_count
            {
                if let Some(price_info) = alt_instance.pricing.get(&request.pricing_model) {
                    let alt_cost = price_info.hourly_rate
                        * request.duration_hours
                        * request.instance_count as f64;

                    if alt_cost < current_cost * 0.9 {
                        // At least 10% savings
                        let savings = current_cost - alt_cost;
                        recommendations.push(CostRecommendation {
                            recommendation_type: RecommendationType::InstanceTypeChange,
                            description: format!(
                                "Switch to {} instance type",
                                alt_instance.instance_type
                            ),
                            potential_savings: savings,
                            savings_percent: (savings / current_cost) * 100.0,
                            effort: ImplementationEffort::Medium,
                            alternative: Some(CostCalculationRequest {
                                instance_type: alt_instance.instance_type.clone(),
                                ..request.clone()
                            }),
                        });
                    }
                }
            }
        }

        // Sort by savings
        recommendations.sort_by(|a, b| {
            b.potential_savings
                .partial_cmp(&a.potential_savings)
                .unwrap()
        });

        Ok(recommendations)
    }

    /// Mock pricing data for testing
    fn mock_pricing_data(&self, provider: CloudProvider, region: &str) -> Vec<InstancePricing> {
        match provider {
            CloudProvider::Aws => vec![
                InstancePricing {
                    provider: CloudProvider::Aws,
                    region: region.to_string(),
                    instance_type: "p4d.24xlarge".to_string(),
                    instance_class: InstanceClass::GpuAccelerated,
                    vcpus: 96,
                    memory_gb: 1152.0,
                    gpu_count: 8,
                    gpu_model: Some("A100".to_string()),
                    storage_gb: 8000.0,
                    network_performance: "400 Gbps".to_string(),
                    pricing: HashMap::from([
                        (
                            PricingModel::OnDemand,
                            PriceInfo {
                                hourly_rate: 32.77,
                                currency: "USD".to_string(),
                                min_commitment: None,
                                upfront_cost: None,
                                discount_percent: 0.0,
                            },
                        ),
                        (
                            PricingModel::Reserved1Year,
                            PriceInfo {
                                hourly_rate: 20.37,
                                currency: "USD".to_string(),
                                min_commitment: Some(8760),
                                upfront_cost: Some(89200.0),
                                discount_percent: 38.0,
                            },
                        ),
                        (
                            PricingModel::Spot,
                            PriceInfo {
                                hourly_rate: 9.83,
                                currency: "USD".to_string(),
                                min_commitment: None,
                                upfront_cost: None,
                                discount_percent: 70.0,
                            },
                        ),
                    ]),
                    last_updated: Utc::now(),
                },
                InstancePricing {
                    provider: CloudProvider::Aws,
                    region: region.to_string(),
                    instance_type: "g4dn.xlarge".to_string(),
                    instance_class: InstanceClass::GpuAccelerated,
                    vcpus: 4,
                    memory_gb: 16.0,
                    gpu_count: 1,
                    gpu_model: Some("T4".to_string()),
                    storage_gb: 125.0,
                    network_performance: "Up to 25 Gbps".to_string(),
                    pricing: HashMap::from([(
                        PricingModel::OnDemand,
                        PriceInfo {
                            hourly_rate: 0.526,
                            currency: "USD".to_string(),
                            min_commitment: None,
                            upfront_cost: None,
                            discount_percent: 0.0,
                        },
                    )]),
                    last_updated: Utc::now(),
                },
            ],
            CloudProvider::Azure => vec![InstancePricing {
                provider: CloudProvider::Azure,
                region: region.to_string(),
                instance_type: "Standard_NC96ads_A100_v4".to_string(),
                instance_class: InstanceClass::GpuAccelerated,
                vcpus: 96,
                memory_gb: 880.0,
                gpu_count: 8,
                gpu_model: Some("A100".to_string()),
                storage_gb: 2800.0,
                network_performance: "80 Gbps".to_string(),
                pricing: HashMap::from([(
                    PricingModel::OnDemand,
                    PriceInfo {
                        hourly_rate: 27.20,
                        currency: "USD".to_string(),
                        min_commitment: None,
                        upfront_cost: None,
                        discount_percent: 0.0,
                    },
                )]),
                last_updated: Utc::now(),
            }],
            CloudProvider::Gcp => vec![InstancePricing {
                provider: CloudProvider::Gcp,
                region: region.to_string(),
                instance_type: "a2-highgpu-8g".to_string(),
                instance_class: InstanceClass::GpuAccelerated,
                vcpus: 96,
                memory_gb: 680.0,
                gpu_count: 8,
                gpu_model: Some("A100".to_string()),
                storage_gb: 3000.0,
                network_performance: "100 Gbps".to_string(),
                pricing: HashMap::from([(
                    PricingModel::OnDemand,
                    PriceInfo {
                        hourly_rate: 29.35,
                        currency: "USD".to_string(),
                        min_commitment: None,
                        upfront_cost: None,
                        discount_percent: 0.0,
                    },
                )]),
                last_updated: Utc::now(),
            }],
            _ => vec![],
        }
    }

    /// Mock regional pricing data
    fn mock_regional_data(&self, provider: CloudProvider, region: &str) -> RegionalPricing {
        RegionalPricing {
            provider,
            region: region.to_string(),
            region_name: match region {
                "us-east-1" => "US East (N. Virginia)".to_string(),
                "us-west-2" => "US West (Oregon)".to_string(),
                "eu-west-1" => "Europe (Ireland)".to_string(),
                _ => region.to_string(),
            },
            data_transfer: DataTransferPricing {
                inbound_per_gb: 0.0,
                outbound_same_region_per_gb: 0.01,
                outbound_internet_per_gb: 0.09,
                outbound_cross_region_per_gb: 0.02,
            },
            storage: StoragePricing {
                standard_per_gb_month: 0.10,
                ssd_per_gb_month: 0.17,
                archive_per_gb_month: 0.004,
                iops_per_thousand: 0.065,
            },
            additional_services: HashMap::new(),
        }
    }
}

/// Provider comparison result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderComparison {
    /// Provider
    pub provider: CloudProvider,
    /// Region
    pub region: String,
    /// Best matching instance type
    pub instance_type: String,
    /// Hourly rate
    pub hourly_rate: f64,
    /// Total cost for duration
    pub total_cost: f64,
    /// Instance specifications
    pub specifications: InstanceSpecs,
}

/// Instance specifications
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstanceSpecs {
    /// vCPUs
    pub vcpus: u32,
    /// Memory in GB
    pub memory_gb: f64,
    /// GPU count
    pub gpu_count: u32,
    /// GPU model
    pub gpu_model: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_cloud_provider_display() {
        assert_eq!(CloudProvider::Aws.to_string(), "AWS");
        assert_eq!(CloudProvider::Azure.to_string(), "Azure");
        assert_eq!(CloudProvider::Gcp.to_string(), "GCP");
        assert_eq!(CloudProvider::Oci.to_string(), "OCI");
    }

    #[test]
    fn test_cloud_pricing_service_creation() {
        let config = CloudPricingConfig::default();
        let service = CloudPricingService::new(config);
        assert!(service.is_ok());
    }

    #[tokio::test]
    async fn test_get_instance_pricing() {
        let config = CloudPricingConfig::default();
        let service = CloudPricingService::new(config).unwrap();

        let result = service
            .get_instance_pricing(CloudProvider::Aws, "us-east-1")
            .await;
        assert!(result.is_ok());

        let pricing = result?;
        assert!(!pricing.is_empty());
        assert!(pricing.iter().any(|p| p.gpu_count > 0));
    }

    #[tokio::test]
    async fn test_cost_calculation() {
        let config = CloudPricingConfig::default();
        let service = CloudPricingService::new(config).unwrap();

        let request = CostCalculationRequest {
            provider: CloudProvider::Aws,
            region: "us-east-1".to_string(),
            instance_type: "p4d.24xlarge".to_string(),
            instance_count: 2,
            duration_hours: 24.0,
            pricing_model: PricingModel::OnDemand,
            storage_gb: 1000.0,
            data_transfer: DataTransferEstimate {
                inbound_gb: 100.0,
                outbound_internet_gb: 50.0,
                outbound_same_region_gb: 200.0,
                outbound_cross_region_gb: 0.0,
            },
        };

        let result = service.calculate_cost(request).await;
        assert!(result.is_ok());

        let cost = result.unwrap();
        assert!(cost.total_cost > 0.0);
        assert!(cost.compute_cost > 0.0);
        assert!(cost.storage_cost > 0.0);
        assert!(cost.data_transfer_cost > 0.0);
    }

    #[tokio::test]
    async fn test_spot_pricing_savings() {
        let config = CloudPricingConfig::default();
        let service = CloudPricingService::new(config).unwrap();

        let on_demand_request = CostCalculationRequest {
            provider: CloudProvider::Aws,
            region: "us-east-1".to_string(),
            instance_type: "p4d.24xlarge".to_string(),
            instance_count: 1,
            duration_hours: 100.0,
            pricing_model: PricingModel::OnDemand,
            storage_gb: 0.0,
            data_transfer: DataTransferEstimate {
                inbound_gb: 0.0,
                outbound_internet_gb: 0.0,
                outbound_same_region_gb: 0.0,
                outbound_cross_region_gb: 0.0,
            },
        };

        let spot_request = CostCalculationRequest {
            pricing_model: PricingModel::Spot,
            ..on_demand_request.clone()
        };

        let on_demand_cost = service.calculate_cost(on_demand_request).await.unwrap();
        let spot_cost = service.calculate_cost(spot_request).await.unwrap();

        assert!(spot_cost.total_cost < on_demand_cost.total_cost);
        assert!(spot_cost.savings_vs_ondemand > 0.0);
    }

    #[tokio::test]
    async fn test_provider_comparison() {
        let config = CloudPricingConfig::default();
        let service = CloudPricingService::new(config).unwrap();

        let regions = vec![
            (CloudProvider::Aws, "us-east-1".to_string()),
            (CloudProvider::Azure, "eastus".to_string()),
            (CloudProvider::Gcp, "us-central1".to_string()),
        ];

        let result = service
            .compare_providers(InstanceClass::GpuAccelerated, 96, 880.0, 8, regions, 24.0)
            .await;

        assert!(result.is_ok());
        let comparisons = result.unwrap();
        assert!(!comparisons.is_empty());

        // Check that results are sorted by cost
        for i in 1..comparisons.len() {
            assert!(comparisons[i].total_cost >= comparisons[i - 1].total_cost);
        }
    }

    #[tokio::test]
    async fn test_recommendations_generation() {
        let config = CloudPricingConfig::default();
        let service = CloudPricingService::new(config).unwrap();

        let request = CostCalculationRequest {
            provider: CloudProvider::Aws,
            region: "us-east-1".to_string(),
            instance_type: "p4d.24xlarge".to_string(),
            instance_count: 1,
            duration_hours: 8760.0, // 1 year
            pricing_model: PricingModel::OnDemand,
            storage_gb: 1000.0,
            data_transfer: DataTransferEstimate {
                inbound_gb: 1000.0,
                outbound_internet_gb: 500.0,
                outbound_same_region_gb: 2000.0,
                outbound_cross_region_gb: 100.0,
            },
        };

        let result = service.calculate_cost(request).await.unwrap();
        assert!(!result.recommendations.is_empty());

        // Should recommend reserved instances for long-term usage
        assert!(result.recommendations.iter().any(|r| matches!(
            r.recommendation_type,
            RecommendationType::PricingModelChange
        )));
    }

    #[test]
    fn test_data_transfer_calculation() {
        let config = CloudPricingConfig::default();
        let service = CloudPricingService::new(config).unwrap();

        let pricing = DataTransferPricing {
            inbound_per_gb: 0.0,
            outbound_same_region_per_gb: 0.01,
            outbound_internet_per_gb: 0.09,
            outbound_cross_region_per_gb: 0.02,
        };

        let estimate = DataTransferEstimate {
            inbound_gb: 1000.0,
            outbound_internet_gb: 100.0,
            outbound_same_region_gb: 500.0,
            outbound_cross_region_gb: 200.0,
        };

        let cost = service.calculate_data_transfer_cost(&pricing, &estimate);
        let expected = 0.0 * 1000.0 + 0.09 * 100.0 + 0.01 * 500.0 + 0.02 * 200.0;
        assert_relative_eq!(cost, expected, epsilon = 0.01);
    }

    #[tokio::test]
    async fn test_cache_expiration() {
        let mut config = CloudPricingConfig::default();
        config.cache_duration = Duration::from_millis(100); // Very short cache

        let service = CloudPricingService::new(config).unwrap();

        // First call should update cache
        let result1 = service
            .get_instance_pricing(CloudProvider::Aws, "us-east-1")
            .await;
        assert!(result1.is_ok());

        // Wait for cache to expire
        tokio::time::sleep(Duration::from_millis(200)).await;

        // Second call should trigger cache update
        let result2 = service
            .get_instance_pricing(CloudProvider::Aws, "us-east-1")
            .await;
        assert!(result2.is_ok());
    }

    #[test]
    fn test_pricing_model_attributes() {
        let price_info = PriceInfo {
            hourly_rate: 10.0,
            currency: "USD".to_string(),
            min_commitment: Some(8760),
            upfront_cost: Some(10000.0),
            discount_percent: 40.0,
        };

        assert_eq!(price_info.hourly_rate, 10.0);
        assert_eq!(price_info.min_commitment?, 8760);
        assert_eq!(price_info.upfront_cost.unwrap(), 10000.0);
        assert_eq!(price_info.discount_percent, 40.0);
    }

    #[test]
    fn test_instance_class_variants() {
        let classes = vec![
            InstanceClass::GeneralPurpose,
            InstanceClass::ComputeOptimized,
            InstanceClass::MemoryOptimized,
            InstanceClass::GpuAccelerated,
            InstanceClass::StorageOptimized,
            InstanceClass::Burstable,
        ];

        // Ensure all variants are distinct
        for (i, class1) in classes.iter().enumerate() {
            for (j, class2) in classes.iter().enumerate() {
                if i != j {
                    assert_ne!(class1, class2);
                }
            }
        }
    }

    #[test]
    fn test_recommendation_effort_levels() {
        let low = ImplementationEffort::Low;
        let medium = ImplementationEffort::Medium;
        let high = ImplementationEffort::High;

        // Ensure effort levels are distinct
        assert_ne!(low, medium);
        assert_ne!(medium, high);
        assert_ne!(low, high);
    }

    #[tokio::test]
    async fn test_concurrent_pricing_updates() {
        let config = CloudPricingConfig::default();
        let service = Arc::new(CloudPricingService::new(config).unwrap());

        // Spawn multiple concurrent requests
        let mut handles = vec![];

        for i in 0..5 {
            let service_clone = service.clone();
            let handle = tokio::spawn(async move {
                service_clone
                    .get_instance_pricing(CloudProvider::Aws, "us-east-1")
                    .await
            });
            handles.push(handle);
        }

        // All should succeed without conflicts
        for handle in handles {
            let result = handle.await.unwrap();
            assert!(result.is_ok());
        }
    }
}
