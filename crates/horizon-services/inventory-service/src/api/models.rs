use serde::{Deserialize, Serialize};
use utoipa::ToSchema;
use uuid::Uuid;
use validator::Validate;

use crate::models::{Asset, AssetHistory, AssetStatus, AssetType, ProviderType};

#[derive(Debug, Clone, Serialize, Deserialize, Validate, ToSchema)]
pub struct CreateAssetRequest {
    pub asset_type: AssetType,
    pub provider: ProviderType,
    pub provider_id: Option<String>,
    pub parent_id: Option<Uuid>,
    #[validate(length(min = 1, max = 255))]
    pub hostname: Option<String>,
    pub status: Option<AssetStatus>,
    #[validate(length(max = 255))]
    pub location: Option<String>,
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Validate, ToSchema)]
pub struct UpdateAssetRequest {
    pub status: Option<AssetStatus>,
    #[validate(length(max = 255))]
    pub location: Option<String>,
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ListAssetsQuery {
    pub asset_type: Option<AssetType>,
    pub status: Option<AssetStatus>,
    pub provider: Option<ProviderType>,
    pub location: Option<String>,
    pub page: Option<i64>,
    pub page_size: Option<i64>,
    pub sort: Option<String>,
    pub order: Option<String>,
}

impl ListAssetsQuery {
    pub fn page(&self) -> i64 {
        self.page.unwrap_or(1).max(1)
    }

    pub fn page_size(&self) -> i64 {
        self.page_size.unwrap_or(50).clamp(1, 1000)
    }

    pub fn offset(&self) -> i64 {
        (self.page() - 1) * self.page_size()
    }

    pub fn sort_field(&self) -> &str {
        self.sort.as_deref().unwrap_or("created_at")
    }

    pub fn sort_order(&self) -> &str {
        self.order.as_deref().unwrap_or("desc")
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ListAssetsResponse {
    pub data: Vec<Asset>,
    pub pagination: Pagination,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct Pagination {
    pub page: i64,
    pub page_size: i64,
    pub total_items: i64,
    pub total_pages: i64,
}

impl Pagination {
    pub fn new(page: i64, page_size: i64, total_items: i64) -> Self {
        let total_pages = if total_items == 0 {
            0
        } else {
            (total_items + page_size - 1) / page_size
        };

        Self {
            page,
            page_size,
            total_items,
            total_pages,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ListHistoryResponse {
    pub data: Vec<AssetHistory>,
    pub pagination: Pagination,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct DiscoverAssetsRequest {
    pub node: DiscoverNodeRequest,
    pub gpus: Vec<DiscoverGpuRequest>,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct DiscoverNodeRequest {
    pub hostname: String,
    pub provider: ProviderType,
    pub provider_id: Option<String>,
    pub location: Option<String>,
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct DiscoverGpuRequest {
    pub gpu_uuid: String,
    pub metadata: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct DiscoverAssetsResponse {
    pub node_id: Uuid,
    pub gpu_ids: Vec<Uuid>,
    pub created: usize,
    pub updated: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
    pub database: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_list_assets_query_defaults() {
        let query = ListAssetsQuery {
            asset_type: None,
            status: None,
            provider: None,
            location: None,
            page: None,
            page_size: None,
            sort: None,
            order: None,
        };

        assert_eq!(query.page(), 1);
        assert_eq!(query.page_size(), 50);
        assert_eq!(query.offset(), 0);
        assert_eq!(query.sort_field(), "created_at");
        assert_eq!(query.sort_order(), "desc");
    }

    #[test]
    fn test_list_assets_query_custom() {
        let query = ListAssetsQuery {
            asset_type: Some(AssetType::Gpu),
            status: Some(AssetStatus::Available),
            provider: None,
            location: None,
            page: Some(2),
            page_size: Some(100),
            sort: Some("hostname".to_string()),
            order: Some("asc".to_string()),
        };

        assert_eq!(query.page(), 2);
        assert_eq!(query.page_size(), 100);
        assert_eq!(query.offset(), 100);
        assert_eq!(query.sort_field(), "hostname");
        assert_eq!(query.sort_order(), "asc");
    }

    #[test]
    fn test_pagination_calculation() {
        let pagination = Pagination::new(1, 50, 150);
        assert_eq!(pagination.page, 1);
        assert_eq!(pagination.page_size, 50);
        assert_eq!(pagination.total_items, 150);
        assert_eq!(pagination.total_pages, 3);

        let pagination = Pagination::new(1, 50, 0);
        assert_eq!(pagination.total_pages, 0);
    }

    #[test]
    fn test_create_asset_request_validation() {
        let request = CreateAssetRequest {
            asset_type: AssetType::Gpu,
            provider: ProviderType::Baremetal,
            provider_id: None,
            parent_id: None,
            hostname: Some("gpu-node-01".to_string()),
            status: None,
            location: Some("us-west-1a".to_string()),
            metadata: Some(serde_json::json!({"gpu_model": "H100"})),
        };

        assert!(request.validate().is_ok());
    }
}
