//! SwarmFS Registry - Distributed container image registry for StratoSwarm
//!
//! This crate provides a P2P-based container image registry that supports:
//! - Building rootfs images from scratch
//! - Converting Docker/OCI images to SwarmFS format
//! - Content-addressable storage with deduplication
//! - Distributed registry across cluster mesh
//! - Progressive image loading for fast container starts

pub mod docker;
pub mod error;
pub mod mocks;
pub mod registry;
pub mod rootfs;
pub mod store;
pub mod streamer;
pub mod verifier;

pub use error::{Result, SwarmRegistryError};

// Re-export main types
pub use docker::DockerConverter;
pub use registry::DistributedRegistry;
pub use rootfs::RootfsBuilder;
pub use store::ContentAddressableStore;
pub use streamer::ImageStreamer;
pub use verifier::ImageVerifier;

/// SwarmFS image format with agent enhancement support
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SwarmImage {
    /// Unique content hash of the image
    pub hash: String,
    /// Image metadata
    pub metadata: ImageMetadata,
    /// Layer hashes in order
    pub layers: Vec<String>,
    /// Agent configuration if applicable
    pub agent_config: Option<AgentConfig>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ImageMetadata {
    /// Image name (e.g., "ubuntu")
    pub name: String,
    /// Image tag (e.g., "25.04-gpu")
    pub tag: String,
    /// Image variant (e.g., "base", "dev", "gpu", "ai")
    pub variant: ImageVariant,
    /// Creation timestamp
    pub created: u64,
    /// Image size in bytes
    pub size: u64,
    /// Architecture (e.g., "amd64", "arm64")
    pub architecture: String,
    /// OS (e.g., "linux")
    pub os: String,
    /// Environment variables
    pub env: Vec<String>,
    /// Entrypoint command
    pub entrypoint: Option<Vec<String>>,
    /// Default command
    pub cmd: Option<Vec<String>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ImageVariant {
    /// Minimal rootfs with just essentials
    Base,
    /// Development tools included
    Dev,
    /// CUDA drivers and GPU support
    Gpu,
    /// AI/ML frameworks pre-installed
    Ai,
    /// Custom variant
    Custom,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AgentConfig {
    /// Agent personality traits
    pub personality: AgentPersonality,
    /// Evolution policy
    pub evolution_enabled: bool,
    /// Initial memory allocations
    pub memory_config: MemoryConfig,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AgentPersonality {
    pub risk_tolerance: f32,
    pub cooperation: f32,
    pub exploration: f32,
    pub efficiency_focus: f32,
    pub stability_preference: f32,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MemoryConfig {
    pub working_memory: String,
    pub episodic_memory: String,
    pub semantic_memory: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_image_variant_serialization() {
        let variant = ImageVariant::Gpu;
        let json = serde_json::to_string(&variant).unwrap();
        assert_eq!(json, r#""Gpu""#);

        let deserialized: ImageVariant = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, variant);
    }

    #[test]
    fn test_swarm_image_creation() {
        let image = SwarmImage {
            hash: "sha256:abcdef".to_string(),
            metadata: ImageMetadata {
                name: "ubuntu".to_string(),
                tag: "25.04".to_string(),
                variant: ImageVariant::Base,
                created: 1234567890,
                size: 1024 * 1024 * 100, // 100MB
                architecture: "amd64".to_string(),
                os: "linux".to_string(),
                env: vec!["PATH=/usr/bin".to_string()],
                entrypoint: Some(vec!["/bin/bash".to_string()]),
                cmd: None,
            },
            layers: vec!["layer1".to_string(), "layer2".to_string()],
            agent_config: None,
        };

        assert_eq!(image.metadata.name, "ubuntu");
        assert_eq!(image.layers.len(), 2);
    }
}
