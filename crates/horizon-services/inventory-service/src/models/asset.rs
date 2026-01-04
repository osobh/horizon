use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;
use uuid::Uuid;

use super::{AssetStatus, AssetType, ProviderType};

#[derive(Debug, Clone, PartialEq, sqlx::FromRow, Serialize, Deserialize, ToSchema)]
pub struct Asset {
    pub id: Uuid,
    pub asset_type: AssetType,
    pub provider: ProviderType,
    pub provider_id: Option<String>,
    pub parent_id: Option<Uuid>,
    pub hostname: Option<String>,
    pub status: AssetStatus,
    pub location: Option<String>,
    #[sqlx(json)]
    pub metadata: serde_json::Value,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub decommissioned_at: Option<DateTime<Utc>>,
    pub created_by: String,
}

impl Asset {
    pub fn new(
        asset_type: AssetType,
        provider: ProviderType,
        hostname: Option<String>,
        created_by: String,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            asset_type,
            provider,
            provider_id: None,
            parent_id: None,
            hostname,
            status: AssetStatus::Provisioning,
            location: None,
            metadata: serde_json::json!({}),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            decommissioned_at: None,
            created_by,
        }
    }

    pub fn with_provider_id(mut self, provider_id: String) -> Self {
        self.provider_id = Some(provider_id);
        self
    }

    pub fn with_parent_id(mut self, parent_id: Uuid) -> Self {
        self.parent_id = Some(parent_id);
        self
    }

    pub fn with_location(mut self, location: String) -> Self {
        self.location = Some(location);
        self
    }

    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = metadata;
        self
    }

    pub fn with_status(mut self, status: AssetStatus) -> Self {
        self.status = status;
        self
    }

    pub fn is_decommissioned(&self) -> bool {
        self.status == AssetStatus::Decommissioned || self.decommissioned_at.is_some()
    }

    pub fn is_available(&self) -> bool {
        self.status == AssetStatus::Available
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_asset_new() {
        let asset = Asset::new(
            AssetType::Gpu,
            ProviderType::Baremetal,
            Some("gpu-node-01".to_string()),
            "system".to_string(),
        );

        assert_eq!(asset.asset_type, AssetType::Gpu);
        assert_eq!(asset.provider, ProviderType::Baremetal);
        assert_eq!(asset.hostname, Some("gpu-node-01".to_string()));
        assert_eq!(asset.status, AssetStatus::Provisioning);
        assert_eq!(asset.created_by, "system");
        assert!(!asset.is_decommissioned());
    }

    #[test]
    fn test_asset_builder_pattern() {
        let asset = Asset::new(
            AssetType::Node,
            ProviderType::Aws,
            Some("node-01".to_string()),
            "system".to_string(),
        )
        .with_provider_id("i-1234567890abcdef0".to_string())
        .with_location("us-west-2a".to_string())
        .with_metadata(serde_json::json!({"instance_type": "p5.48xlarge"}))
        .with_status(AssetStatus::Available);

        assert_eq!(asset.provider_id, Some("i-1234567890abcdef0".to_string()));
        assert_eq!(asset.location, Some("us-west-2a".to_string()));
        assert_eq!(asset.status, AssetStatus::Available);
        assert!(asset.is_available());
    }

    #[test]
    fn test_asset_is_decommissioned() {
        let mut asset = Asset::new(
            AssetType::Gpu,
            ProviderType::Baremetal,
            Some("gpu-01".to_string()),
            "system".to_string(),
        );

        assert!(!asset.is_decommissioned());

        asset.status = AssetStatus::Decommissioned;
        assert!(asset.is_decommissioned());

        asset.status = AssetStatus::Available;
        asset.decommissioned_at = Some(Utc::now());
        assert!(asset.is_decommissioned());
    }

    #[test]
    fn test_asset_serialization() {
        let asset = Asset::new(
            AssetType::Gpu,
            ProviderType::Baremetal,
            Some("gpu-01".to_string()),
            "system".to_string(),
        );

        let json = serde_json::to_string(&asset).unwrap();
        let deserialized: Asset = serde_json::from_str(&json).unwrap();

        assert_eq!(asset.id, deserialized.id);
        assert_eq!(asset.asset_type, deserialized.asset_type);
        assert_eq!(asset.hostname, deserialized.hostname);
    }
}
