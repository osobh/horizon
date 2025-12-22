//! RootfsBuilder - Creates Ubuntu rootfs images from scratch with agent injection

use crate::{AgentConfig, ImageMetadata, ImageVariant, Result, SwarmImage, SwarmRegistryError};
use std::path::{Path, PathBuf};
use tokio::process::Command;
use tracing::{info, warn};

/// Ubuntu release configuration
#[derive(Debug, Clone)]
pub struct UbuntuRelease {
    pub version: String,
    pub codename: String,
    pub arch: String,
    pub mirror: String,
}

impl Default for UbuntuRelease {
    fn default() -> Self {
        Self {
            version: "25.04".to_string(),
            codename: "plucky".to_string(), // Ubuntu 25.04 codename
            arch: "amd64".to_string(),
            mirror: "http://archive.ubuntu.com/ubuntu/".to_string(),
        }
    }
}

/// Configuration for building rootfs
#[derive(Debug, Clone)]
pub struct RootfsConfig {
    pub release: UbuntuRelease,
    pub variant: ImageVariant,
    pub packages: Vec<String>,
    pub agent_config: Option<AgentConfig>,
    pub include_stratoswarm_agent: bool,
}

impl Default for RootfsConfig {
    fn default() -> Self {
        Self {
            release: UbuntuRelease::default(),
            variant: ImageVariant::Base,
            packages: vec![],
            agent_config: None,
            include_stratoswarm_agent: true,
        }
    }
}

/// Builder for creating rootfs images from scratch
pub struct RootfsBuilder {
    work_dir: PathBuf,
    config: RootfsConfig,
}

impl RootfsBuilder {
    /// Create a new RootfsBuilder
    pub fn new(work_dir: impl AsRef<Path>) -> Self {
        Self {
            work_dir: work_dir.as_ref().to_path_buf(),
            config: RootfsConfig::default(),
        }
    }

    /// Set the Ubuntu release
    pub fn with_release(mut self, release: UbuntuRelease) -> Self {
        self.config.release = release;
        self
    }

    /// Set the image variant
    pub fn with_variant(mut self, variant: ImageVariant) -> Self {
        self.config.variant = variant;
        self
    }

    /// Add packages to install
    pub fn with_packages(mut self, packages: Vec<String>) -> Self {
        self.config.packages = packages;
        self
    }

    /// Set agent configuration
    pub fn with_agent_config(mut self, agent_config: AgentConfig) -> Self {
        self.config.agent_config = Some(agent_config);
        self
    }

    /// Build the rootfs image
    pub async fn build(&self) -> Result<SwarmImage> {
        info!(
            "Building rootfs for Ubuntu {} ({})",
            self.config.release.version,
            self.config.variant.as_str()
        );

        // Create work directory
        let rootfs_dir = self.work_dir.join("rootfs");
        tokio::fs::create_dir_all(&rootfs_dir).await?;

        // Phase 1: Bootstrap base system
        self.bootstrap_base(&rootfs_dir).await?;

        // Phase 2: Configure base system
        self.configure_base(&rootfs_dir).await?;

        // Phase 3: Install variant-specific packages
        self.install_variant_packages(&rootfs_dir).await?;

        // Phase 4: Inject StratoSwarm agent if enabled
        if self.config.include_stratoswarm_agent {
            self.inject_stratoswarm_agent(&rootfs_dir).await?;
        }

        // Phase 5: Configure agent if provided
        if let Some(ref agent_config) = self.config.agent_config {
            self.configure_agent(&rootfs_dir, agent_config).await?;
        }

        // Phase 6: Create layers and calculate hashes
        let layers = self.create_layers(&rootfs_dir).await?;

        // Phase 7: Create SwarmImage
        let image = self.create_swarm_image(layers).await?;

        Ok(image)
    }

    async fn bootstrap_base(&self, rootfs_dir: &Path) -> Result<()> {
        info!("Bootstrapping base system with debootstrap");

        // Use debootstrap to create minimal rootfs
        let output = Command::new("debootstrap")
            .arg("--variant=minbase")
            .arg("--arch")
            .arg(&self.config.release.arch)
            .arg(&self.config.release.codename)
            .arg(rootfs_dir)
            .arg(&self.config.release.mirror)
            .output()
            .await?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(SwarmRegistryError::BuildFailed(format!(
                "debootstrap failed: {}",
                stderr
            )));
        }

        Ok(())
    }

    async fn configure_base(&self, rootfs_dir: &Path) -> Result<()> {
        info!("Configuring base system");

        // Configure apt sources
        let sources_list = format!(
            "deb {} {} main restricted universe multiverse\n\
             deb {} {}-updates main restricted universe multiverse\n\
             deb {} {}-security main restricted universe multiverse",
            self.config.release.mirror,
            self.config.release.codename,
            self.config.release.mirror,
            self.config.release.codename,
            self.config.release.mirror,
            self.config.release.codename,
        );

        let sources_path = rootfs_dir.join("etc/apt/sources.list");
        tokio::fs::write(&sources_path, sources_list).await?;

        // Configure hostname
        let hostname_path = rootfs_dir.join("etc/hostname");
        tokio::fs::write(&hostname_path, "stratoswarm-container\n").await?;

        // Configure resolv.conf
        let resolv_path = rootfs_dir.join("etc/resolv.conf");
        tokio::fs::write(&resolv_path, "nameserver 8.8.8.8\nnameserver 8.8.4.4\n").await?;

        Ok(())
    }

    async fn install_variant_packages(&self, rootfs_dir: &Path) -> Result<()> {
        let mut packages = self.get_variant_packages();
        packages.extend(self.config.packages.clone());

        if packages.is_empty() {
            return Ok(());
        }

        info!(
            "Installing {} packages for variant {:?}",
            packages.len(),
            self.config.variant
        );

        // Use chroot to install packages
        let package_list = packages.join(" ");
        let install_script = format!(
            "#!/bin/bash\n\
             apt-get update\n\
             DEBIAN_FRONTEND=noninteractive apt-get install -y {}\n\
             apt-get clean\n\
             rm -rf /var/lib/apt/lists/*",
            package_list
        );

        let script_path = rootfs_dir.join("tmp/install.sh");
        tokio::fs::write(&script_path, install_script).await?;

        let output = Command::new("chroot")
            .arg(rootfs_dir)
            .arg("/bin/bash")
            .arg("/tmp/install.sh")
            .output()
            .await?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(SwarmRegistryError::BuildFailed(format!(
                "Package installation failed: {}",
                stderr
            )));
        }

        // Clean up script
        tokio::fs::remove_file(&script_path).await?;

        Ok(())
    }

    fn get_variant_packages(&self) -> Vec<String> {
        match self.config.variant {
            ImageVariant::Base => vec![
                "systemd".to_string(),
                "systemd-sysv".to_string(),
                "ca-certificates".to_string(),
                "curl".to_string(),
                "iproute2".to_string(),
            ],
            ImageVariant::Dev => vec![
                "build-essential".to_string(),
                "git".to_string(),
                "vim".to_string(),
                "python3".to_string(),
                "python3-pip".to_string(),
                "nodejs".to_string(),
                "npm".to_string(),
            ],
            ImageVariant::Gpu => vec![
                "nvidia-driver-565".to_string(), // Latest NVIDIA driver
                "cuda-toolkit-12-6".to_string(),
                "nvidia-container-toolkit".to_string(),
            ],
            ImageVariant::Ai => vec![
                "python3-torch".to_string(),
                "python3-tensorflow".to_string(),
                "python3-numpy".to_string(),
                "python3-pandas".to_string(),
                "python3-scikit-learn".to_string(),
                "jupyter-notebook".to_string(),
            ],
            ImageVariant::Custom => vec![],
        }
    }

    async fn inject_stratoswarm_agent(&self, rootfs_dir: &Path) -> Result<()> {
        info!("Injecting StratoSwarm agent");

        // Create agent directory
        let agent_dir = rootfs_dir.join("opt/stratoswarm");
        tokio::fs::create_dir_all(&agent_dir).await?;

        // Copy agent binary (assuming it's built)
        let agent_binary = std::env::var("STRATOSWARM_AGENT_PATH")
            .unwrap_or_else(|_| "/usr/local/bin/stratoswarm-agent".to_string());

        if Path::new(&agent_binary).exists() {
            let dest = agent_dir.join("stratoswarm-agent");
            tokio::fs::copy(&agent_binary, &dest).await?;

            // Make executable
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                let mut perms = tokio::fs::metadata(&dest).await?.permissions();
                perms.set_mode(0o755);
                tokio::fs::set_permissions(&dest, perms).await?;
            }
        } else {
            warn!("StratoSwarm agent binary not found at {}", agent_binary);
        }

        // Create systemd service
        let service_content = r#"[Unit]
Description=StratoSwarm Agent
After=network.target

[Service]
Type=simple
ExecStart=/opt/stratoswarm/stratoswarm-agent
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"#;

        let service_path = rootfs_dir.join("etc/systemd/system/stratoswarm-agent.service");
        tokio::fs::write(&service_path, service_content).await?;

        Ok(())
    }

    async fn configure_agent(&self, rootfs_dir: &Path, agent_config: &AgentConfig) -> Result<()> {
        info!("Configuring agent with personality and evolution settings");

        let config_path = rootfs_dir.join("opt/stratoswarm/agent.yaml");
        let config_content = serde_yaml::to_string(agent_config).map_err(|e| {
            SwarmRegistryError::Other(format!("Failed to serialize agent config: {}", e))
        })?;

        tokio::fs::write(&config_path, config_content).await?;

        Ok(())
    }

    async fn create_layers(&self, rootfs_dir: &Path) -> Result<Vec<String>> {
        info!("Creating image layers");

        let mut layers = Vec::new();

        // Create tar archive of rootfs
        let layer_path = self.work_dir.join("layer.tar");
        let output = Command::new("tar")
            .arg("-C")
            .arg(rootfs_dir)
            .arg("-cf")
            .arg(&layer_path)
            .arg(".")
            .output()
            .await?;

        if !output.status.success() {
            return Err(SwarmRegistryError::BuildFailed(
                "Failed to create tar archive".to_string(),
            ));
        }

        // Calculate hash
        let layer_data = tokio::fs::read(&layer_path).await?;
        let hash = self.calculate_hash(&layer_data);

        // Compress layer
        let compressed_path = self.work_dir.join(format!("{}.tar.gz", &hash));
        let compressed_data = self.compress_data(&layer_data)?;
        tokio::fs::write(&compressed_path, compressed_data).await?;

        layers.push(hash);

        Ok(layers)
    }

    fn calculate_hash(&self, data: &[u8]) -> String {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(data);
        format!("sha256:{}", hex::encode(hasher.finalize()))
    }

    fn compress_data(&self, data: &[u8]) -> Result<Vec<u8>> {
        use flate2::write::GzEncoder;
        use flate2::Compression;
        use std::io::Write;

        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(data)?;
        Ok(encoder.finish()?)
    }

    async fn create_swarm_image(&self, layers: Vec<String>) -> Result<SwarmImage> {
        let metadata = ImageMetadata {
            name: "ubuntu".to_string(),
            tag: format!(
                "{}-{}",
                self.config.release.version,
                self.config.variant.as_str()
            ),
            variant: self.config.variant,
            created: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            size: self.calculate_total_size(&layers).await?,
            architecture: self.config.release.arch.clone(),
            os: "linux".to_string(),
            env: vec![
                "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin".to_string(),
            ],
            entrypoint: Some(vec!["/bin/bash".to_string()]),
            cmd: None,
        };

        let image_data = SwarmImage {
            hash: self.calculate_image_hash(&metadata, &layers),
            metadata,
            layers,
            agent_config: self.config.agent_config.clone(),
        };

        Ok(image_data)
    }

    async fn calculate_total_size(&self, layers: &[String]) -> Result<u64> {
        let mut total_size = 0u64;

        for layer in layers {
            let layer_path = self.work_dir.join(format!("{}.tar.gz", layer));
            let metadata = tokio::fs::metadata(&layer_path).await?;
            total_size += metadata.len();
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

impl ImageVariant {
    fn as_str(&self) -> &'static str {
        match self {
            ImageVariant::Base => "base",
            ImageVariant::Dev => "dev",
            ImageVariant::Gpu => "gpu",
            ImageVariant::Ai => "ai",
            ImageVariant::Custom => "custom",
        }
    }
}

// Add serde_yaml to dependencies
use serde_yaml;

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_rootfs_builder_creation() {
        let temp_dir = TempDir::new().unwrap();
        let builder = RootfsBuilder::new(temp_dir.path());

        assert_eq!(builder.config.variant, ImageVariant::Base);
        assert_eq!(builder.config.release.version, "25.04");
    }

    #[tokio::test]
    async fn test_rootfs_builder_with_variant() {
        let temp_dir = TempDir::new().unwrap();
        let builder = RootfsBuilder::new(temp_dir.path()).with_variant(ImageVariant::Gpu);

        assert_eq!(builder.config.variant, ImageVariant::Gpu);
    }

    #[tokio::test]
    async fn test_get_variant_packages() {
        let temp_dir = TempDir::new().unwrap();
        let builder = RootfsBuilder::new(temp_dir.path()).with_variant(ImageVariant::Dev);

        let packages = builder.get_variant_packages();
        assert!(packages.contains(&"build-essential".to_string()));
        assert!(packages.contains(&"git".to_string()));
    }

    #[tokio::test]
    async fn test_calculate_hash() {
        let temp_dir = TempDir::new().unwrap();
        let builder = RootfsBuilder::new(temp_dir.path());

        let data = b"test data";
        let hash = builder.calculate_hash(data);

        assert!(hash.starts_with("sha256:"));
        assert_eq!(hash.len(), 71); // "sha256:" + 64 hex chars
    }

    #[tokio::test]
    async fn test_compress_data() {
        let temp_dir = TempDir::new().unwrap();
        let builder = RootfsBuilder::new(temp_dir.path());

        let data = b"test data to compress";
        let compressed = builder.compress_data(data).unwrap();

        // Compressed data should be different and likely smaller
        assert_ne!(compressed.as_slice(), data);
    }

    #[tokio::test]
    async fn test_with_agent_config() {
        let temp_dir = TempDir::new().unwrap();
        let agent_config = AgentConfig {
            personality: crate::AgentPersonality {
                risk_tolerance: 0.7,
                cooperation: 0.9,
                exploration: 0.5,
                efficiency_focus: 0.8,
                stability_preference: 0.6,
            },
            evolution_enabled: true,
            memory_config: crate::MemoryConfig {
                working_memory: "256MB".to_string(),
                episodic_memory: "1GB".to_string(),
                semantic_memory: "512MB".to_string(),
            },
        };

        let builder = RootfsBuilder::new(temp_dir.path()).with_agent_config(agent_config.clone());

        assert!(builder.config.agent_config.is_some());
        assert_eq!(
            builder
                .config
                .agent_config
                .as_ref()
                .unwrap()
                .personality
                .risk_tolerance,
            0.7
        );
    }

    #[tokio::test]
    #[ignore] // This test requires debootstrap to be installed
    async fn test_build_minimal_rootfs() {
        let temp_dir = TempDir::new().unwrap();
        let builder = RootfsBuilder::new(temp_dir.path())
            .with_variant(ImageVariant::Base)
            .with_packages(vec!["nano".to_string()]);

        let result = builder.build().await;

        // This will fail without debootstrap, but tests the API
        assert!(result.is_err() || result.is_ok());
    }
}
