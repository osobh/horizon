use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

#[derive(Debug, Clone, Copy, PartialEq, Eq, sqlx::Type, Serialize, Deserialize, ToSchema)]
#[sqlx(type_name = "asset_type", rename_all = "lowercase")]
#[serde(rename_all = "lowercase")]
pub enum AssetType {
    Node,
    Gpu,
    Cpu,
    Nic,
    Switch,
    Rack,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, sqlx::Type, Serialize, Deserialize, ToSchema)]
#[sqlx(type_name = "asset_status", rename_all = "lowercase")]
#[serde(rename_all = "lowercase")]
pub enum AssetStatus {
    Provisioning,
    Available,
    Allocated,
    Maintenance,
    Degraded,
    Decommissioned,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, sqlx::Type, Serialize, Deserialize, ToSchema)]
#[sqlx(type_name = "provider_type", rename_all = "lowercase")]
#[serde(rename_all = "lowercase")]
pub enum ProviderType {
    Baremetal,
    Aws,
    Gcp,
    Azure,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, sqlx::Type, Serialize, Deserialize, ToSchema)]
#[sqlx(type_name = "change_operation", rename_all = "snake_case")]
#[serde(rename_all = "snake_case")]
pub enum ChangeOperation {
    Created,
    Updated,
    StatusChanged,
    MetadataChanged,
    Decommissioned,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_asset_type_serialization() {
        let asset_type = AssetType::Gpu;
        let json = serde_json::to_string(&asset_type).unwrap();
        assert_eq!(json, "\"gpu\"");

        let deserialized: AssetType = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, AssetType::Gpu);
    }

    #[test]
    fn test_asset_status_serialization() {
        let status = AssetStatus::Available;
        let json = serde_json::to_string(&status).unwrap();
        assert_eq!(json, "\"available\"");

        let deserialized: AssetStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, AssetStatus::Available);
    }

    #[test]
    fn test_provider_type_serialization() {
        let provider = ProviderType::Baremetal;
        let json = serde_json::to_string(&provider).unwrap();
        assert_eq!(json, "\"baremetal\"");

        let deserialized: ProviderType = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, ProviderType::Baremetal);
    }

    #[test]
    fn test_change_operation_serialization() {
        let operation = ChangeOperation::StatusChanged;
        let json = serde_json::to_string(&operation).unwrap();
        assert_eq!(json, "\"status_changed\"");

        let deserialized: ChangeOperation = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, ChangeOperation::StatusChanged);
    }
}
