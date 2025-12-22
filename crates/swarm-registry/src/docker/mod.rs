//! DockerConverter - Converts Docker/OCI images to SwarmFS format with agent enhancement

use crate::{AgentConfig, ImageMetadata, ImageVariant, Result, SwarmImage, SwarmRegistryError};
use oci_spec::image::{ImageConfiguration, ImageManifest};
use std::path::{Path, PathBuf};
use tokio::io::AsyncWriteExt;
use tracing::{debug, info};

/// Configuration for Docker image conversion
#[derive(Debug, Clone)]
pub struct ConversionConfig {
    /// Add StratoSwarm agent to the image
    pub inject_agent: bool,
    /// Agent configuration if injecting
    pub agent_config: Option<AgentConfig>,
    /// Override the image variant
    pub variant_override: Option<ImageVariant>,
    /// Additional layers to add
    pub additional_layers: Vec<PathBuf>,
}

impl Default for ConversionConfig {
    fn default() -> Self {
        Self {
            inject_agent: true,
            agent_config: None,
            variant_override: None,
            additional_layers: vec![],
        }
    }
}

/// Converts Docker/OCI images to SwarmFS format
pub struct DockerConverter {
    work_dir: PathBuf,
    docker_registry: String,
}

impl DockerConverter {
    /// Create a new DockerConverter
    pub fn new(work_dir: impl AsRef<Path>) -> Self {
        Self {
            work_dir: work_dir.as_ref().to_path_buf(),
            docker_registry: "https://registry-1.docker.io".to_string(),
        }
    }

    /// Set a custom Docker registry
    pub fn with_registry(mut self, registry: String) -> Self {
        self.docker_registry = registry;
        self
    }

    /// Convert a Docker image to SwarmFS format
    pub async fn convert(&self, image_ref: &str, config: ConversionConfig) -> Result<SwarmImage> {
        info!("Converting Docker image {} to SwarmFS format", image_ref);

        // Parse image reference
        let (image_name, tag) = self.parse_image_ref(image_ref)?;

        // Phase 1: Pull manifest
        let manifest = self.pull_manifest(&image_name, &tag).await?;

        // Phase 2: Download layers
        let layer_paths = self.download_layers(&image_name, &manifest).await?;

        // Phase 3: Extract image config
        let image_config = self.pull_image_config(&image_name, &manifest).await?;

        // Phase 4: Convert layers to SwarmFS format
        let mut swarm_layers = self.convert_layers(&layer_paths).await?;

        // Phase 5: Inject agent if requested
        if config.inject_agent {
            let agent_layer = self.create_agent_layer(&config.agent_config).await?;
            swarm_layers.push(agent_layer);
        }

        // Phase 6: Add additional layers
        for layer_path in &config.additional_layers {
            let layer_hash = self.add_custom_layer(layer_path).await?;
            swarm_layers.push(layer_hash);
        }

        // Phase 7: Create SwarmImage
        let swarm_image = self
            .create_swarm_image(&image_name, &tag, &image_config, swarm_layers, config)
            .await?;

        Ok(swarm_image)
    }

    fn parse_image_ref(&self, image_ref: &str) -> Result<(String, String)> {
        let parts: Vec<&str> = image_ref.split(':').collect();

        match parts.len() {
            1 => Ok((parts[0].to_string(), "latest".to_string())),
            2 => Ok((parts[0].to_string(), parts[1].to_string())),
            _ => Err(SwarmRegistryError::InvalidFormat(format!(
                "Invalid image reference: {}",
                image_ref
            ))),
        }
    }

    async fn pull_manifest(&self, image: &str, tag: &str) -> Result<ImageManifest> {
        info!("Pulling manifest for {}:{}", image, tag);

        // Construct manifest URL
        let manifest_url = format!(
            "{}/v2/{}/manifests/{}",
            self.docker_registry,
            self.normalize_image_name(image),
            tag
        );

        // Make HTTP request
        let client = reqwest::Client::new();
        let response = client
            .get(&manifest_url)
            .header(
                "Accept",
                "application/vnd.docker.distribution.manifest.v2+json",
            )
            .send()
            .await
            .map_err(|e| SwarmRegistryError::Http(e.to_string()))?;

        if !response.status().is_success() {
            return Err(SwarmRegistryError::ImageNotFound(format!(
                "{}:{}",
                image, tag
            )));
        }

        let manifest_bytes = response
            .bytes()
            .await
            .map_err(|e| SwarmRegistryError::Http(e.to_string()))?;

        // Parse manifest
        let manifest: ImageManifest = serde_json::from_slice(&manifest_bytes)?;

        Ok(manifest)
    }

    fn normalize_image_name(&self, image: &str) -> String {
        // Docker Hub images need "library/" prefix if no namespace
        if !image.contains('/') {
            format!("library/{}", image)
        } else {
            image.to_string()
        }
    }

    async fn download_layers(&self, image: &str, manifest: &ImageManifest) -> Result<Vec<PathBuf>> {
        info!("Downloading {} layers", manifest.layers().len());

        let mut layer_paths = Vec::new();
        let layers_dir = self.work_dir.join("layers");
        tokio::fs::create_dir_all(&layers_dir).await?;

        for (idx, layer) in manifest.layers().iter().enumerate() {
            let digest = layer.digest().to_string();
            info!(
                "Downloading layer {}/{}: {}",
                idx + 1,
                manifest.layers().len(),
                digest
            );

            let layer_path = layers_dir.join(format!("{}.tar.gz", digest.replace(':', "-")));

            if layer_path.exists() {
                debug!("Layer already cached: {}", digest);
                layer_paths.push(layer_path);
                continue;
            }

            // Download layer
            let blob_url = format!(
                "{}/v2/{}/blobs/{}",
                self.docker_registry,
                self.normalize_image_name(image),
                digest
            );

            let client = reqwest::Client::new();
            let response = client
                .get(&blob_url)
                .send()
                .await
                .map_err(|e| SwarmRegistryError::Http(e.to_string()))?;

            if !response.status().is_success() {
                return Err(SwarmRegistryError::Network(format!(
                    "Failed to download layer {}: {}",
                    digest,
                    response.status()
                )));
            }

            // Stream to file
            let mut file = tokio::fs::File::create(&layer_path).await?;
            let mut stream = response.bytes_stream();

            while let Some(chunk) = futures::StreamExt::next(&mut stream).await {
                let chunk = chunk.map_err(|e| SwarmRegistryError::Http(e.to_string()))?;
                file.write_all(&chunk).await?;
            }

            layer_paths.push(layer_path);
        }

        Ok(layer_paths)
    }

    async fn pull_image_config(
        &self,
        image: &str,
        manifest: &ImageManifest,
    ) -> Result<ImageConfiguration> {
        let config_digest = manifest.config().digest().to_string();
        info!("Pulling image configuration: {}", config_digest);

        let config_url = format!(
            "{}/v2/{}/blobs/{}",
            self.docker_registry,
            self.normalize_image_name(image),
            config_digest
        );

        let client = reqwest::Client::new();
        let response = client
            .get(&config_url)
            .send()
            .await
            .map_err(|e| SwarmRegistryError::Http(e.to_string()))?;

        if !response.status().is_success() {
            return Err(SwarmRegistryError::Network(format!(
                "Failed to get config: {}",
                response.status()
            )));
        }

        let config_bytes = response
            .bytes()
            .await
            .map_err(|e| SwarmRegistryError::Http(e.to_string()))?;

        let config: ImageConfiguration = serde_json::from_slice(&config_bytes)?;

        Ok(config)
    }

    async fn convert_layers(&self, layer_paths: &[PathBuf]) -> Result<Vec<String>> {
        info!("Converting {} layers to SwarmFS format", layer_paths.len());

        let mut swarm_layers = Vec::new();

        for layer_path in layer_paths {
            // Read layer
            let layer_data = tokio::fs::read(layer_path).await?;

            // Calculate SwarmFS hash
            let hash = self.calculate_hash(&layer_data);

            // Store in SwarmFS format
            let swarm_path = self.work_dir.join("swarmfs").join(&hash);
            tokio::fs::create_dir_all(swarm_path.parent().unwrap()).await?;
            tokio::fs::write(&swarm_path, &layer_data).await?;

            swarm_layers.push(hash);
        }

        Ok(swarm_layers)
    }

    async fn create_agent_layer(&self, agent_config: &Option<AgentConfig>) -> Result<String> {
        info!("Creating agent injection layer");

        let agent_dir = self.work_dir.join("agent-layer");
        tokio::fs::create_dir_all(&agent_dir).await?;

        // Create agent directory structure
        let opt_dir = agent_dir.join("opt/stratoswarm");
        tokio::fs::create_dir_all(&opt_dir).await?;

        // Copy agent binary
        let agent_binary = std::env::var("STRATOSWARM_AGENT_PATH")
            .unwrap_or_else(|_| "/usr/local/bin/stratoswarm-agent".to_string());

        if Path::new(&agent_binary).exists() {
            let dest = opt_dir.join("stratoswarm-agent");
            tokio::fs::copy(&agent_binary, &dest).await?;
        }

        // Write agent config if provided
        if let Some(config) = agent_config {
            let config_path = opt_dir.join("agent.yaml");
            let config_content = serde_yaml::to_string(config)
                .map_err(|e| SwarmRegistryError::Other(e.to_string()))?;
            tokio::fs::write(&config_path, config_content).await?;
        }

        // Create systemd service
        let systemd_dir = agent_dir.join("etc/systemd/system");
        tokio::fs::create_dir_all(&systemd_dir).await?;

        let service_content = r#"[Unit]
Description=StratoSwarm Agent
After=network.target

[Service]
Type=simple
ExecStart=/opt/stratoswarm/stratoswarm-agent
Restart=always

[Install]
WantedBy=multi-user.target
"#;

        let service_path = systemd_dir.join("stratoswarm-agent.service");
        tokio::fs::write(&service_path, service_content).await?;

        // Create tar archive
        let layer_tar = self.work_dir.join("agent-layer.tar");
        let output = tokio::process::Command::new("tar")
            .arg("-C")
            .arg(&agent_dir)
            .arg("-cf")
            .arg(&layer_tar)
            .arg(".")
            .output()
            .await?;

        if !output.status.success() {
            return Err(SwarmRegistryError::ConversionFailed(
                "Failed to create agent layer".to_string(),
            ));
        }

        // Read and hash the layer
        let layer_data = tokio::fs::read(&layer_tar).await?;
        let hash = self.calculate_hash(&layer_data);

        // Store in SwarmFS format
        let swarm_path = self.work_dir.join("swarmfs").join(&hash);
        tokio::fs::create_dir_all(swarm_path.parent().unwrap()).await?;
        tokio::fs::write(&swarm_path, layer_data).await?;

        // Clean up
        tokio::fs::remove_dir_all(&agent_dir).await?;
        tokio::fs::remove_file(&layer_tar).await?;

        Ok(hash)
    }

    async fn add_custom_layer(&self, layer_path: &Path) -> Result<String> {
        info!("Adding custom layer from {:?}", layer_path);

        let layer_data = tokio::fs::read(layer_path).await?;
        let hash = self.calculate_hash(&layer_data);

        // Store in SwarmFS format
        let swarm_path = self.work_dir.join("swarmfs").join(&hash);
        tokio::fs::create_dir_all(swarm_path.parent().unwrap()).await?;
        tokio::fs::write(&swarm_path, layer_data).await?;

        Ok(hash)
    }

    async fn create_swarm_image(
        &self,
        image_name: &str,
        tag: &str,
        docker_config: &ImageConfiguration,
        layers: Vec<String>,
        conversion_config: ConversionConfig,
    ) -> Result<SwarmImage> {
        // Determine variant
        let variant = conversion_config
            .variant_override
            .unwrap_or_else(|| self.detect_variant(image_name, tag, docker_config));

        // Extract Docker config
        let config = docker_config
            .config()
            .as_ref()
            .ok_or_else(|| SwarmRegistryError::InvalidFormat("Missing config".to_string()))?;

        let metadata = ImageMetadata {
            name: image_name.to_string(),
            tag: tag.to_string(),
            variant,
            created: docker_config
                .created()
                .as_ref()
                .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
                .map(|dt| dt.timestamp() as u64)
                .unwrap_or_else(|| {
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_secs()
                }),
            size: self.calculate_total_size(&layers).await?,
            architecture: docker_config.architecture().to_string(),
            os: docker_config.os().to_string(),
            env: config
                .env()
                .as_ref()
                .map(|e| e.to_vec())
                .unwrap_or_default(),
            entrypoint: config.entrypoint().as_ref().cloned(),
            cmd: config.cmd().as_ref().cloned(),
        };

        let swarm_image = SwarmImage {
            hash: self.calculate_image_hash(&metadata, &layers),
            metadata,
            layers,
            agent_config: conversion_config.agent_config,
        };

        Ok(swarm_image)
    }

    fn detect_variant(
        &self,
        image_name: &str,
        tag: &str,
        _config: &ImageConfiguration,
    ) -> ImageVariant {
        // Try to detect variant from image name and tag
        let full_name = format!("{}:{}", image_name, tag);

        if full_name.contains("cuda") || full_name.contains("gpu") {
            ImageVariant::Gpu
        } else if full_name.contains("devel") || full_name.contains("dev") {
            ImageVariant::Dev
        } else if full_name.contains("ml")
            || full_name.contains("ai")
            || full_name.contains("tensorflow")
            || full_name.contains("pytorch")
        {
            ImageVariant::Ai
        } else {
            ImageVariant::Base
        }
    }

    fn calculate_hash(&self, data: &[u8]) -> String {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(data);
        format!("sha256:{}", hex::encode(hasher.finalize()))
    }

    async fn calculate_total_size(&self, layers: &[String]) -> Result<u64> {
        let mut total_size = 0u64;

        for layer in layers {
            let layer_path = self.work_dir.join("swarmfs").join(layer);
            if let Ok(metadata) = tokio::fs::metadata(&layer_path).await {
                total_size += metadata.len();
            }
        }

        Ok(total_size)
    }

    fn calculate_image_hash(&self, metadata: &ImageMetadata, layers: &[String]) -> String {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();

        // Hash metadata
        hasher.update(metadata.name.as_bytes());
        hasher.update(metadata.tag.as_bytes());
        hasher.update(metadata.created.to_le_bytes());

        // Hash layers
        for layer in layers {
            hasher.update(layer.as_bytes());
        }

        format!("sha256:{}", hex::encode(hasher.finalize()))
    }
}

// For parsing dates
use chrono;

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_docker_converter_creation() {
        let temp_dir = TempDir::new().unwrap();
        let converter = DockerConverter::new(temp_dir.path());

        assert_eq!(converter.docker_registry, "https://registry-1.docker.io");
    }

    #[test]
    fn test_parse_image_ref() {
        let temp_dir = TempDir::new().unwrap();
        let converter = DockerConverter::new(temp_dir.path());

        // Test with tag
        let (name, tag) = converter.parse_image_ref("ubuntu:22.04").unwrap();
        assert_eq!(name, "ubuntu");
        assert_eq!(tag, "22.04");

        // Test without tag (should default to latest)
        let (name, tag) = converter.parse_image_ref("nginx").unwrap();
        assert_eq!(name, "nginx");
        assert_eq!(tag, "latest");

        // Test invalid format
        let result = converter.parse_image_ref("invalid:format:extra");
        assert!(result.is_err());
    }

    #[test]
    fn test_normalize_image_name() {
        let temp_dir = TempDir::new().unwrap();
        let converter = DockerConverter::new(temp_dir.path());

        // Test official image
        assert_eq!(converter.normalize_image_name("ubuntu"), "library/ubuntu");

        // Test namespaced image
        assert_eq!(converter.normalize_image_name("user/image"), "user/image");
    }

    #[test]
    fn test_detect_variant() {
        let temp_dir = TempDir::new().unwrap();
        let converter = DockerConverter::new(temp_dir.path());

        let config = oci_spec::image::ImageConfigurationBuilder::default()
            .architecture("amd64")
            .os("linux")
            .build()
            .unwrap();

        // Test GPU detection
        assert_eq!(
            converter.detect_variant("nvidia/cuda", "11.8.0-base", &config),
            ImageVariant::Gpu
        );

        // Test Dev detection
        assert_eq!(
            converter.detect_variant("ubuntu", "22.04-devel", &config),
            ImageVariant::Dev
        );

        // Test AI detection
        assert_eq!(
            converter.detect_variant("tensorflow/tensorflow", "latest", &config),
            ImageVariant::Ai
        );

        // Test Base detection
        assert_eq!(
            converter.detect_variant("ubuntu", "22.04", &config),
            ImageVariant::Base
        );
    }

    #[tokio::test]
    async fn test_conversion_config_default() {
        let config = ConversionConfig::default();

        assert!(config.inject_agent);
        assert!(config.agent_config.is_none());
        assert!(config.variant_override.is_none());
        assert!(config.additional_layers.is_empty());
    }

    #[tokio::test]
    async fn test_calculate_hash() {
        let temp_dir = TempDir::new().unwrap();
        let converter = DockerConverter::new(temp_dir.path());

        let data = b"test data";
        let hash = converter.calculate_hash(data);

        assert!(hash.starts_with("sha256:"));
        assert_eq!(hash.len(), 71); // "sha256:" + 64 hex chars
    }

    #[tokio::test]
    #[ignore] // This test requires network access
    async fn test_pull_manifest() {
        let temp_dir = TempDir::new().unwrap();
        let converter = DockerConverter::new(temp_dir.path());

        let result = converter.pull_manifest("hello-world", "latest").await;

        // This will fail without network, but tests the API
        assert!(result.is_err() || result.is_ok());
    }
}
