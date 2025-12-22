use crate::types::*;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Represents a discovered resource available from a provider
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AvailableResource {
    pub id: String,
    pub resource_type: ResourceType,
    pub provider: CloudProvider,
    pub region: String,
    pub availability_zone: Option<String>,
    pub capacity: ResourceSpec,
    pub pricing: Option<PricingInfo>,
    pub metadata: HashMap<String, String>,
}

/// Cloud provider classification
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CloudProvider {
    Aws,
    Gcp,
    Azure,
    Custom(String),
}

/// Pricing information for a resource
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PricingInfo {
    pub price_per_hour: f64,
    pub currency: String,
    pub billing_model: BillingModel,
    pub minimum_commitment: Option<CommitmentTerm>,
}

/// Billing model for resource pricing
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BillingModel {
    OnDemand,
    Reserved,
    Spot,
    Preemptible,
    Custom(String),
}

/// Commitment term for reserved pricing
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CommitmentTerm {
    OneYear,
    ThreeYears,
    Custom(String),
}

/// Trait for discovering resources from cloud providers
#[async_trait]
pub trait ResourceDiscovery: Send + Sync {
    /// Discover all available resources in a specific region
    async fn discover_resources(
        &self,
        region: &str,
    ) -> Result<Vec<AvailableResource>, ResourceDiscoveryError>;

    /// Discover resources of a specific type
    async fn discover_by_type(
        &self,
        region: &str,
        resource_type: &ResourceType,
    ) -> Result<Vec<AvailableResource>, ResourceDiscoveryError>;

    /// Get detailed information about a specific resource
    async fn get_resource_details(
        &self,
        resource_id: &str,
    ) -> Result<AvailableResource, ResourceDiscoveryError>;

    /// List all available regions for this provider
    async fn list_regions(&self) -> Result<Vec<String>, ResourceDiscoveryError>;

    /// Check if a specific resource type is available in a region
    async fn is_resource_available(
        &self,
        region: &str,
        resource_type: &ResourceType,
    ) -> Result<bool, ResourceDiscoveryError>;
}

/// Trait for retrieving pricing information from cloud providers
#[async_trait]
pub trait ResourcePricing: Send + Sync {
    /// Get pricing for a specific resource type
    async fn get_pricing(
        &self,
        region: &str,
        resource_type: &ResourceType,
    ) -> Result<PricingInfo, PricingError>;

    /// Get pricing for multiple resource types (batch operation)
    async fn get_bulk_pricing(
        &self,
        region: &str,
        resource_types: &[ResourceType],
    ) -> Result<HashMap<ResourceType, PricingInfo>, PricingError>;

    /// Calculate total cost for a resource request over a duration
    async fn calculate_cost(
        &self,
        region: &str,
        resource_type: &ResourceType,
        spec: &ResourceSpec,
        duration_hours: f64,
    ) -> Result<f64, PricingError>;

    /// Get spot/preemptible pricing history
    async fn get_spot_price_history(
        &self,
        region: &str,
        resource_type: &ResourceType,
        lookback_hours: u32,
    ) -> Result<Vec<SpotPricePoint>, PricingError>;
}

/// Spot pricing data point
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpotPricePoint {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub price_per_hour: f64,
    pub availability_zone: String,
}

/// Errors that can occur during resource discovery
#[derive(Debug, thiserror::Error)]
pub enum ResourceDiscoveryError {
    #[error("Provider authentication failed: {0}")]
    AuthenticationFailed(String),

    #[error("Region {0} not found")]
    RegionNotFound(String),

    #[error("Resource type {0} not supported by provider")]
    UnsupportedResourceType(String),

    #[error("Network error: {0}")]
    NetworkError(String),

    #[error("Provider API error: {0}")]
    ProviderApiError(String),

    #[error("Resource {0} not found")]
    ResourceNotFound(String),

    #[error("Internal error: {0}")]
    InternalError(String),
}

/// Errors that can occur during pricing operations
#[derive(Debug, thiserror::Error)]
pub enum PricingError {
    #[error("Pricing information not available for {0}")]
    PricingNotAvailable(String),

    #[error("Invalid billing model: {0}")]
    InvalidBillingModel(String),

    #[error("Provider API error: {0}")]
    ProviderApiError(String),

    #[error("Network error: {0}")]
    NetworkError(String),

    #[error("Internal error: {0}")]
    InternalError(String),
}

impl CloudProvider {
    pub fn as_str(&self) -> &str {
        match self {
            CloudProvider::Aws => "aws",
            CloudProvider::Gcp => "gcp",
            CloudProvider::Azure => "azure",
            CloudProvider::Custom(name) => name,
        }
    }
}

impl std::fmt::Display for CloudProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CloudProvider::Aws => write!(f, "AWS"),
            CloudProvider::Gcp => write!(f, "GCP"),
            CloudProvider::Azure => write!(f, "Azure"),
            CloudProvider::Custom(name) => write!(f, "{}", name),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cloud_provider_display() {
        assert_eq!(CloudProvider::Aws.to_string(), "AWS");
        assert_eq!(CloudProvider::Gcp.to_string(), "GCP");
        assert_eq!(CloudProvider::Azure.to_string(), "Azure");
        assert_eq!(
            CloudProvider::Custom("OCI".to_string()).to_string(),
            "OCI"
        );
    }

    #[test]
    fn test_cloud_provider_as_str() {
        assert_eq!(CloudProvider::Aws.as_str(), "aws");
        assert_eq!(CloudProvider::Gcp.as_str(), "gcp");
        assert_eq!(CloudProvider::Azure.as_str(), "azure");
    }

    #[test]
    fn test_pricing_info_creation() {
        let pricing = PricingInfo {
            price_per_hour: 3.50,
            currency: "USD".to_string(),
            billing_model: BillingModel::OnDemand,
            minimum_commitment: None,
        };

        assert_eq!(pricing.price_per_hour, 3.50);
        assert_eq!(pricing.currency, "USD");
        assert_eq!(pricing.billing_model, BillingModel::OnDemand);
        assert!(pricing.minimum_commitment.is_none());
    }

    #[test]
    fn test_pricing_info_with_commitment() {
        let pricing = PricingInfo {
            price_per_hour: 2.10,
            currency: "USD".to_string(),
            billing_model: BillingModel::Reserved,
            minimum_commitment: Some(CommitmentTerm::OneYear),
        };

        assert_eq!(pricing.minimum_commitment, Some(CommitmentTerm::OneYear));
    }

    #[test]
    fn test_available_resource_creation() {
        let resource = AvailableResource {
            id: "i-1234567890abcdef0".to_string(),
            resource_type: ResourceType::Compute(ComputeType::Gpu),
            provider: CloudProvider::Aws,
            region: "us-west-2".to_string(),
            availability_zone: Some("us-west-2a".to_string()),
            capacity: ResourceSpec::new(8.0, ResourceUnit::Count)
                .with_vendor(GpuVendor::Nvidia)
                .with_model("H100"),
            pricing: Some(PricingInfo {
                price_per_hour: 32.77,
                currency: "USD".to_string(),
                billing_model: BillingModel::OnDemand,
                minimum_commitment: None,
            }),
            metadata: HashMap::new(),
        };

        assert_eq!(resource.provider, CloudProvider::Aws);
        assert_eq!(resource.region, "us-west-2");
        assert!(resource.pricing.is_some());
    }

    #[test]
    fn test_spot_price_point() {
        let now = chrono::Utc::now();
        let spot_price = SpotPricePoint {
            timestamp: now,
            price_per_hour: 12.50,
            availability_zone: "us-east-1a".to_string(),
        };

        assert_eq!(spot_price.price_per_hour, 12.50);
        assert_eq!(spot_price.availability_zone, "us-east-1a");
    }

    #[test]
    fn test_billing_model_serialization() {
        let on_demand = BillingModel::OnDemand;
        let json = serde_json::to_string(&on_demand).unwrap();
        let deserialized: BillingModel = serde_json::from_str(&json).unwrap();
        assert_eq!(on_demand, deserialized);
    }

    #[test]
    fn test_commitment_term_equality() {
        assert_eq!(CommitmentTerm::OneYear, CommitmentTerm::OneYear);
        assert_ne!(CommitmentTerm::OneYear, CommitmentTerm::ThreeYears);
    }

    #[test]
    fn test_resource_discovery_error_display() {
        let error = ResourceDiscoveryError::RegionNotFound("invalid-region".to_string());
        assert_eq!(error.to_string(), "Region invalid-region not found");
    }

    #[test]
    fn test_pricing_error_display() {
        let error = PricingError::PricingNotAvailable("GPU".to_string());
        assert_eq!(
            error.to_string(),
            "Pricing information not available for GPU"
        );
    }

    #[test]
    fn test_available_resource_with_metadata() {
        let mut metadata = HashMap::new();
        metadata.insert("instance_type".to_string(), "p5.48xlarge".to_string());
        metadata.insert("vram_gb".to_string(), "640".to_string());

        let resource = AvailableResource {
            id: "test-resource".to_string(),
            resource_type: ResourceType::Compute(ComputeType::Gpu),
            provider: CloudProvider::Aws,
            region: "us-west-2".to_string(),
            availability_zone: None,
            capacity: ResourceSpec::new(8.0, ResourceUnit::Count),
            pricing: None,
            metadata,
        };

        assert_eq!(resource.metadata.len(), 2);
        assert_eq!(
            resource.metadata.get("instance_type"),
            Some(&"p5.48xlarge".to_string())
        );
    }

    #[test]
    fn test_serialization_available_resource() {
        let resource = AvailableResource {
            id: "test-resource".to_string(),
            resource_type: ResourceType::Memory,
            provider: CloudProvider::Gcp,
            region: "us-central1".to_string(),
            availability_zone: None,
            capacity: ResourceSpec::new(512.0, ResourceUnit::Gigabytes),
            pricing: None,
            metadata: HashMap::new(),
        };

        let json = serde_json::to_string(&resource).unwrap();
        let deserialized: AvailableResource = serde_json::from_str(&json).unwrap();

        assert_eq!(resource.id, deserialized.id);
        assert_eq!(resource.provider, deserialized.provider);
        assert_eq!(resource.region, deserialized.region);
    }
}
