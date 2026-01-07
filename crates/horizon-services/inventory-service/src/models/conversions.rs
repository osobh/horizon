//! Type conversions between hpc-inventory and inventory-service models
//!
//! This module provides conversion utilities for bridging the CLI-focused
//! hpc-inventory types with the enterprise inventory-service types.

use super::{Asset, AssetStatus, AssetType, ProviderType};
use hpc_inventory::{GpuInfo, HardwareProfile, NodeInfo, NodeStatus};
use uuid::Uuid;

/// Convert a CLI NodeInfo to an inventory-service Asset
impl From<&NodeInfo> for Asset {
    fn from(node: &NodeInfo) -> Self {
        let status = match node.status {
            NodeStatus::Pending | NodeStatus::Connecting | NodeStatus::Bootstrapping => {
                AssetStatus::Provisioning
            }
            NodeStatus::Connected => AssetStatus::Available,
            NodeStatus::Unreachable | NodeStatus::Failed => AssetStatus::Degraded,
            NodeStatus::Offline => AssetStatus::Maintenance,
        };

        // Build metadata from hardware profile
        let metadata = if let Some(hw) = &node.hardware {
            serde_json::json!({
                "cpu_model": hw.cpu_model,
                "cpu_cores": hw.cpu_cores,
                "memory_gb": hw.memory_gb,
                "storage_gb": hw.storage_gb,
                "gpu_count": hw.gpus.len(),
                "gpus": hw.gpus.iter().map(|g| serde_json::json!({
                    "index": g.index,
                    "name": g.name,
                    "memory_mb": g.memory_mb,
                    "vendor": g.vendor,
                })).collect::<Vec<_>>(),
                "os": node.os.as_ref().map(|o| o.to_string()),
                "arch": node.arch.as_ref().map(|a| a.to_string()),
                "ssh_port": node.port,
                "ssh_username": node.username,
                "tags": node.tags,
            })
        } else {
            serde_json::json!({
                "os": node.os.as_ref().map(|o| o.to_string()),
                "arch": node.arch.as_ref().map(|a| a.to_string()),
                "ssh_port": node.port,
                "ssh_username": node.username,
                "tags": node.tags,
            })
        };

        Asset {
            id: Uuid::parse_str(&node.id).unwrap_or_else(|_| Uuid::new_v4()),
            asset_type: AssetType::Node,
            provider: ProviderType::Baremetal,
            provider_id: Some(node.address.clone()),
            parent_id: None,
            hostname: Some(node.name.clone()),
            status,
            location: None,
            metadata,
            created_at: node.created_at,
            updated_at: node.updated_at,
            decommissioned_at: None,
            created_by: "hpc-cli".to_string(),
        }
    }
}

/// Convert an inventory-service Asset to a CLI NodeInfo (partial - no credentials)
impl TryFrom<&Asset> for NodeInfo {
    type Error = &'static str;

    fn try_from(asset: &Asset) -> Result<Self, Self::Error> {
        if asset.asset_type != AssetType::Node {
            return Err("Can only convert Node assets to NodeInfo");
        }

        let status = match asset.status {
            AssetStatus::Provisioning => NodeStatus::Pending,
            AssetStatus::Available => NodeStatus::Connected,
            AssetStatus::Allocated => NodeStatus::Connected,
            AssetStatus::Maintenance => NodeStatus::Offline,
            AssetStatus::Degraded => NodeStatus::Unreachable,
            AssetStatus::Decommissioned => NodeStatus::Offline,
        };

        // Extract hardware profile from metadata
        let hardware = if let Some(cpu_cores) = asset.metadata.get("cpu_cores") {
            Some(HardwareProfile {
                cpu_model: asset
                    .metadata
                    .get("cpu_model")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
                cpu_cores: cpu_cores.as_u64().unwrap_or(0) as u32,
                memory_gb: asset
                    .metadata
                    .get("memory_gb")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0) as f32,
                storage_gb: asset
                    .metadata
                    .get("storage_gb")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0) as f32,
                gpus: asset
                    .metadata
                    .get("gpus")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|g| {
                                Some(GpuInfo {
                                    index: g.get("index")?.as_u64()? as u32,
                                    name: g.get("name")?.as_str()?.to_string(),
                                    memory_mb: g.get("memory_mb")?.as_u64()?,
                                    vendor: g.get("vendor")?.as_str()?.to_string(),
                                })
                            })
                            .collect()
                    })
                    .unwrap_or_default(),
            })
        } else {
            None
        };

        Ok(NodeInfo {
            id: asset.id.to_string(),
            name: asset.hostname.clone().unwrap_or_else(|| asset.id.to_string()),
            address: asset.provider_id.clone().unwrap_or_default(),
            port: asset
                .metadata
                .get("ssh_port")
                .and_then(|v| v.as_u64())
                .unwrap_or(22) as u16,
            username: asset
                .metadata
                .get("ssh_username")
                .and_then(|v| v.as_str())
                .unwrap_or("root")
                .to_string(),
            credential_ref: hpc_inventory::CredentialRef::SshAgent,
            mode: hpc_inventory::NodeMode::Docker,
            os: asset
                .metadata
                .get("os")
                .and_then(|v| v.as_str())
                .and_then(|s| match s {
                    "linux" => Some(hpc_inventory::OsType::Linux),
                    "darwin" => Some(hpc_inventory::OsType::Darwin),
                    "windows" => Some(hpc_inventory::OsType::Windows),
                    _ => None,
                }),
            arch: asset
                .metadata
                .get("arch")
                .and_then(|v| v.as_str())
                .map(hpc_inventory::Architecture::from_uname),
            status,
            hardware,
            last_heartbeat: None,
            quic_endpoint: None,
            created_at: asset.created_at,
            updated_at: asset.updated_at,
            error: None,
            tags: asset
                .metadata
                .get("tags")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default(),
        })
    }
}

/// Extension trait for batch conversions
pub trait NodeInfoBatch {
    /// Convert multiple NodeInfo to Assets
    fn to_assets(&self) -> Vec<Asset>;
}

impl NodeInfoBatch for [NodeInfo] {
    fn to_assets(&self) -> Vec<Asset> {
        self.iter().map(Asset::from).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hpc_inventory::{CredentialRef, NodeMode};

    #[test]
    fn test_node_info_to_asset() {
        let node = NodeInfo::new(
            "test-node".to_string(),
            "192.168.1.100".to_string(),
            22,
            "admin".to_string(),
            CredentialRef::SshAgent,
            NodeMode::Docker,
        );

        let asset = Asset::from(&node);

        assert_eq!(asset.asset_type, AssetType::Node);
        assert_eq!(asset.provider, ProviderType::Baremetal);
        assert_eq!(asset.hostname, Some("test-node".to_string()));
        assert_eq!(asset.provider_id, Some("192.168.1.100".to_string()));
        assert_eq!(asset.status, AssetStatus::Provisioning);
    }

    #[test]
    fn test_asset_to_node_info() {
        let asset = Asset::new(
            AssetType::Node,
            ProviderType::Baremetal,
            Some("gpu-node-01".to_string()),
            "system".to_string(),
        )
        .with_provider_id("10.0.0.50".to_string())
        .with_status(AssetStatus::Available)
        .with_metadata(serde_json::json!({
            "ssh_port": 22,
            "ssh_username": "ubuntu",
            "os": "linux",
            "arch": "amd64",
        }));

        let node = NodeInfo::try_from(&asset).unwrap();

        assert_eq!(node.name, "gpu-node-01");
        assert_eq!(node.address, "10.0.0.50");
        assert_eq!(node.port, 22);
        assert_eq!(node.username, "ubuntu");
        assert_eq!(node.status, NodeStatus::Connected);
    }

    #[test]
    fn test_non_node_asset_conversion_fails() {
        let asset = Asset::new(
            AssetType::Gpu,
            ProviderType::Baremetal,
            Some("gpu-01".to_string()),
            "system".to_string(),
        );

        let result = NodeInfo::try_from(&asset);
        assert!(result.is_err());
    }

    #[test]
    fn test_hardware_profile_roundtrip() {
        let mut node = NodeInfo::new(
            "hw-test".to_string(),
            "1.2.3.4".to_string(),
            22,
            "user".to_string(),
            CredentialRef::SshAgent,
            NodeMode::Binary,
        );
        node.hardware = Some(HardwareProfile {
            cpu_model: "AMD EPYC 7763".to_string(),
            cpu_cores: 128,
            memory_gb: 512.0,
            storage_gb: 2000.0,
            gpus: vec![
                GpuInfo {
                    index: 0,
                    name: "NVIDIA H100".to_string(),
                    memory_mb: 81920,
                    vendor: "nvidia".to_string(),
                },
                GpuInfo {
                    index: 1,
                    name: "NVIDIA H100".to_string(),
                    memory_mb: 81920,
                    vendor: "nvidia".to_string(),
                },
            ],
        });
        node.status = NodeStatus::Connected;

        let asset = Asset::from(&node);
        let roundtrip = NodeInfo::try_from(&asset).unwrap();

        assert_eq!(roundtrip.hardware.as_ref().unwrap().cpu_cores, 128);
        assert_eq!(roundtrip.hardware.as_ref().unwrap().gpus.len(), 2);
        assert_eq!(
            roundtrip.hardware.as_ref().unwrap().gpus[0].name,
            "NVIDIA H100"
        );
    }
}
