//! Simplified DistributedRegistry without complex libp2p dependencies

use crate::mocks::cluster_mesh::{ClusterMesh, Message};
use crate::{Result, SwarmImage, SwarmRegistryError};
use dashmap::DashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tracing::{debug, info};

/// Registry configuration
#[derive(Debug, Clone)]
pub struct RegistryConfig {
    /// Local storage path
    pub storage_path: PathBuf,
    /// Maximum concurrent transfers
    pub max_transfers: usize,
    /// Replication factor
    pub replication_factor: usize,
}

impl Default for RegistryConfig {
    fn default() -> Self {
        Self {
            storage_path: PathBuf::from("/var/lib/stratoswarm/registry"),
            max_transfers: 10,
            replication_factor: 3,
        }
    }
}

/// Simplified distributed registry
pub struct DistributedRegistry {
    /// Registry configuration
    config: RegistryConfig,
    /// Cluster mesh connection
    cluster_mesh: Arc<ClusterMesh>,
    /// Local image store
    local_images: Arc<DashMap<String, SwarmImage>>,
    /// Remote image locations
    remote_images: Arc<DashMap<String, Vec<String>>>,
}

impl DistributedRegistry {
    /// Create a new distributed registry
    pub async fn new(config: RegistryConfig, cluster_mesh: Arc<ClusterMesh>) -> Result<Self> {
        // Create storage directory
        tokio::fs::create_dir_all(&config.storage_path).await?;

        let registry = Self {
            config,
            cluster_mesh,
            local_images: Arc::new(DashMap::new()),
            remote_images: Arc::new(DashMap::new()),
        };

        Ok(registry)
    }

    /// Publish a local image to the registry
    pub async fn publish(&self, image: SwarmImage) -> Result<()> {
        let image_hash = image.hash.clone();
        info!("Publishing image {} to distributed registry", image_hash);

        // Store locally
        self.local_images.insert(image_hash.clone(), image.clone());

        // Store image data
        let image_path = self.config.storage_path.join(&image_hash);
        let image_data = bincode::serialize(&image)?;
        tokio::fs::write(&image_path, image_data).await?;

        // Announce to cluster mesh
        let announcement = serde_json::json!({
            "type": "image_announcement",
            "image_hash": image_hash,
            "name": image.metadata.name,
            "tag": image.metadata.tag,
            "node_id": "local",
        });

        let message = Message::Custom(serde_json::to_vec(&announcement)?);
        self.cluster_mesh
            .broadcast(message)
            .await
            .map_err(|e| SwarmRegistryError::Network(e.to_string()))?;

        Ok(())
    }

    /// Pull an image from the registry
    pub async fn pull(&self, image_ref: &str) -> Result<SwarmImage> {
        info!("Pulling image {} from distributed registry", image_ref);

        // Check if we have it locally
        if let Some(image) = self.local_images.get(image_ref).map(|r| r.clone()) {
            debug!("Image {} found locally", image_ref);
            return Ok(image);
        }

        // Check if file exists on disk
        let image_path = self.config.storage_path.join(image_ref);
        if image_path.exists() {
            let image_data = tokio::fs::read(&image_path).await?;
            let image: SwarmImage = bincode::deserialize(&image_data)?;

            // Cache in memory
            self.local_images.insert(image_ref.to_string(), image.clone());

            return Ok(image);
        }

        Err(SwarmRegistryError::ImageNotFound(image_ref.to_string()))
    }

    /// List all available images
    pub async fn list(&self) -> Result<Vec<(String, String, String)>> {
        let images: Vec<_> = self.local_images
            .iter()
            .map(|entry| {
                let hash = entry.key().clone();
                let image = entry.value();
                (hash, image.metadata.name.clone(), image.metadata.tag.clone())
            })
            .collect();

        Ok(images)
    }

    /// Search for images by name
    pub async fn search(&self, query: &str) -> Result<Vec<SwarmImage>> {
        let results: Vec<_> = self.local_images
            .iter()
            .filter(|entry| {
                let image = entry.value();
                image.metadata.name.contains(query) || image.metadata.tag.contains(query)
            })
            .map(|entry| entry.value().clone())
            .collect();

        Ok(results)
    }
}
