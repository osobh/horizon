//! Core cloud integration types and traits
//!
//! Defines the common interfaces and types used across all cloud providers.

use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Cloud provider trait that all providers must implement
#[async_trait]
pub trait CloudProvider: Send + Sync {
    /// Get provider name
    fn name(&self) -> &str;
    
    /// List available regions
    async fn list_regions(&self) -> Result<Vec<CloudRegion>>;
    
    /// List available instance types
    async fn list_instance_types(&self, region: &str) -> Result<Vec<InstanceType>>;
    
    /// Provision resources
    async fn provision(
        &self,
        request: &ProvisionRequest,
    ) -> Result<ProvisionResponse>;
    
    /// Deprovision resources
    async fn deprovision(&self, resource_id: &str) -> Result<()>;
    
    /// Get resource status
    async fn get_status(&self, resource_id: &str) -> Result<ResourceStatus>;
    
    /// Estimate costs
    async fn estimate_cost(
        &self,
        instance_type: &InstanceType,
        duration: Duration,
    ) -> Result<CostEstimate>;
    
    /// Check spot/preemptible availability
    async fn check_spot_availability(
        &self,
        region: &str,
        instance_type: &str,
    ) -> Result<SpotAvailability>;
}

/// Cloud region information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudRegion {
    pub id: String,
    pub name: String,
    pub location: String,
    pub availability_zones: Vec<String>,
    pub gpu_available: bool,
    pub latency_ms: f64,
}

/// Instance type specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstanceType {
    pub id: String,
    pub name: String,
    pub vcpus: u32,
    pub memory_gb: u32,
    pub gpu_count: u32,
    pub gpu_type: Option<String>,
    pub gpu_memory_gb: Option<u32>,
    pub network_performance: String,
    pub storage_gb: u32,
    pub price_per_hour: f64,
}

/// Provisioning request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvisionRequest {
    pub region: String,
    pub instance_type: String,
    pub count: u32,
    pub use_spot: bool,
    pub tags: HashMap<String, String>,
    pub user_data: Option<String>,
    pub security_groups: Vec<String>,
    pub key_pair: Option<String>,
}

/// Provisioning response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvisionResponse {
    pub resource_id: String,
    pub instances: Vec<CloudInstance>,
    pub total_cost_estimate: f64,
    pub provisioning_time_ms: u64,
}

/// Cloud instance details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudInstance {
    pub instance_id: String,
    pub public_ip: Option<String>,
    pub private_ip: String,
    pub state: InstanceState,
    pub launch_time: Duration,
}

/// Instance state
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum InstanceState {
    Pending,
    Running,
    Stopping,
    Stopped,
    Terminating,
    Terminated,
}

/// Resource status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceStatus {
    pub resource_id: String,
    pub state: ResourceState,
    pub instances: Vec<CloudInstance>,
    pub health_status: HealthStatus,
    pub metrics: ResourceMetrics,
}

/// Resource state
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ResourceState {
    Creating,
    Active,
    Updating,
    Deleting,
    Deleted,
    Failed,
}

/// Health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatus {
    pub healthy_instances: u32,
    pub unhealthy_instances: u32,
    pub health_checks_passed: bool,
}

/// Resource metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceMetrics {
    pub cpu_utilization: f32,
    pub memory_utilization: f32,
    pub gpu_utilization: f32,
    pub network_in_mbps: f64,
    pub network_out_mbps: f64,
}

/// Cost estimate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostEstimate {
    pub instance_cost: f64,
    pub storage_cost: f64,
    pub network_cost: f64,
    pub total_cost: f64,
    pub currency: String,
    pub discount_applied: Option<f64>,
}

/// Spot/preemptible availability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpotAvailability {
    pub available: bool,
    pub current_price: f64,
    pub savings_percentage: f64,
    pub interruption_rate: f64,
    pub recommended: bool,
}

/// Cloud resource abstraction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudResource {
    pub id: String,
    pub provider: String,
    pub region: String,
    pub resource_type: ResourceType,
    pub created_at: Duration,
    pub tags: HashMap<String, String>,
}

/// Resource types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ResourceType {
    ComputeInstance,
    GpuCluster,
    StorageVolume,
    NetworkInterface,
    LoadBalancer,
    SecurityGroup,
}

/// Cloud configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudConfig {
    pub provider: String,
    pub credentials: CloudCredentials,
    pub default_region: String,
    pub default_tags: HashMap<String, String>,
    pub cost_optimization: CostOptimizationConfig,
}

/// Cloud credentials (simplified for example)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CloudCredentials {
    Aws {
        access_key_id: String,
        secret_access_key: String,
        session_token: Option<String>,
    },
    Gcp {
        project_id: String,
        service_account_json: String,
    },
    Alibaba {
        access_key_id: String,
        access_key_secret: String,
    },
}

/// Cost optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostOptimizationConfig {
    pub use_spot_instances: bool,
    pub spot_price_threshold: f64,
    pub enable_auto_shutdown: bool,
    pub shutdown_after_hours: u32,
    pub enable_right_sizing: bool,
}

/// Multi-cloud orchestration request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiCloudRequest {
    pub providers: Vec<String>,
    pub distribution_strategy: DistributionStrategy,
    pub total_instances: u32,
    pub requirements: ResourceRequirements,
}

/// Distribution strategy for multi-cloud
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DistributionStrategy {
    EvenDistribution,
    CostOptimized,
    LatencyOptimized,
    AvailabilityOptimized,
    Custom(HashMap<String, u32>),
}

/// Resource requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirements {
    pub min_vcpus: u32,
    pub min_memory_gb: u32,
    pub min_gpu_count: u32,
    pub gpu_type: Option<String>,
    pub max_price_per_hour: Option<f64>,
    pub preferred_regions: Vec<String>,
}