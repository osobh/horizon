//! Environment profiles for deployment
//!
//! Manages dev, staging, and production profiles with
//! environment-specific configuration.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Deployment environment
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Environment {
    #[default]
    Dev,
    Staging,
    Prod,
}

impl Environment {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Dev => "dev",
            Self::Staging => "staging",
            Self::Prod => "prod",
        }
    }

    /// Parse from string
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "dev" | "development" => Some(Self::Dev),
            "staging" | "stage" => Some(Self::Staging),
            "prod" | "production" => Some(Self::Prod),
            _ => None,
        }
    }

    /// Get next environment in promotion chain
    pub fn next(&self) -> Option<Self> {
        match self {
            Self::Dev => Some(Self::Staging),
            Self::Staging => Some(Self::Prod),
            Self::Prod => None,
        }
    }

    /// Check if promotion to target is valid
    pub fn can_promote_to(&self, target: &Environment) -> bool {
        match (self, target) {
            (Self::Dev, Self::Staging) => true,
            (Self::Staging, Self::Prod) => true,
            _ => false,
        }
    }
}

impl std::fmt::Display for Environment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl std::str::FromStr for Environment {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::from_str(s).ok_or_else(|| format!("Invalid environment: {}", s))
    }
}

/// Environment-specific settings
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ProfileSettings {
    /// StratoSwarm cluster endpoint
    pub stratoswarm_endpoint: Option<String>,
    /// Argus observability endpoint
    pub argus_endpoint: Option<String>,
    /// Docker compose file path for local deployment
    pub docker_compose_file: Option<PathBuf>,
    /// Default number of replicas
    pub replicas: u32,
    /// GPU resources per service
    pub gpu_resources: u32,
    /// Additional environment variables
    pub env_vars: std::collections::HashMap<String, String>,
}

impl Default for ProfileSettings {
    fn default() -> Self {
        Self {
            stratoswarm_endpoint: None,
            argus_endpoint: None,
            docker_compose_file: None,
            replicas: 1,
            gpu_resources: 0,
            env_vars: std::collections::HashMap::new(),
        }
    }
}

impl ProfileSettings {
    /// Create settings for dev environment
    pub fn dev() -> Self {
        Self {
            replicas: 1,
            gpu_resources: 0, // No GPU in dev by default
            ..Default::default()
        }
    }

    /// Create settings for staging environment
    pub fn staging() -> Self {
        Self {
            replicas: 2,
            gpu_resources: 1,
            ..Default::default()
        }
    }

    /// Create settings for production environment
    pub fn prod() -> Self {
        Self {
            replicas: 4,
            gpu_resources: 4,
            ..Default::default()
        }
    }
}

/// Profile configuration containing all environments
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Profile {
    /// Profile name (e.g., "ml-training", "data-processing")
    pub name: String,
    /// Description
    pub description: Option<String>,
    /// Projects included in this profile
    pub projects: Vec<String>,
    /// Dev environment settings
    pub dev: ProfileSettings,
    /// Staging environment settings
    pub staging: ProfileSettings,
    /// Production environment settings
    pub prod: ProfileSettings,
}

impl Default for Profile {
    fn default() -> Self {
        Self {
            name: "default".to_string(),
            description: None,
            projects: Vec::new(),
            dev: ProfileSettings::dev(),
            staging: ProfileSettings::staging(),
            prod: ProfileSettings::prod(),
        }
    }
}

impl Profile {
    /// Create a new profile
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            ..Default::default()
        }
    }

    /// Get settings for a specific environment
    pub fn get_settings(&self, env: Environment) -> &ProfileSettings {
        match env {
            Environment::Dev => &self.dev,
            Environment::Staging => &self.staging,
            Environment::Prod => &self.prod,
        }
    }

    /// Get mutable settings for a specific environment
    pub fn get_settings_mut(&mut self, env: Environment) -> &mut ProfileSettings {
        match env {
            Environment::Dev => &mut self.dev,
            Environment::Staging => &mut self.staging,
            Environment::Prod => &mut self.prod,
        }
    }
}

/// Preset profiles for common use cases
pub fn get_preset_profiles() -> Vec<Profile> {
    vec![
        Profile {
            name: "ml-training".to_string(),
            description: Some("Full ML training stack with GPU support".to_string()),
            projects: vec![
                "rnccl".to_string(),
                "slai".to_string(),
                "torch".to_string(),
                "argus".to_string(),
            ],
            dev: ProfileSettings::dev(),
            staging: ProfileSettings::staging(),
            prod: ProfileSettings::prod(),
        },
        Profile {
            name: "data-processing".to_string(),
            description: Some("Distributed data processing stack".to_string()),
            projects: vec![
                "rmpi".to_string(),
                "spark".to_string(),
                "warp".to_string(),
                "argus".to_string(),
            ],
            dev: ProfileSettings::dev(),
            staging: ProfileSettings::staging(),
            prod: ProfileSettings::prod(),
        },
        Profile {
            name: "full-stack".to_string(),
            description: Some("Complete HPC-AI platform".to_string()),
            projects: vec![
                "rnccl".to_string(),
                "slai".to_string(),
                "torch".to_string(),
                "spark".to_string(),
                "warp".to_string(),
                "stratoswarm".to_string(),
                "nebula".to_string(),
                "vortex".to_string(),
                "argus".to_string(),
            ],
            dev: ProfileSettings::dev(),
            staging: ProfileSettings::staging(),
            prod: ProfileSettings::prod(),
        },
        Profile {
            name: "monitoring".to_string(),
            description: Some("Observability and monitoring only".to_string()),
            projects: vec!["argus".to_string(), "slai".to_string()],
            dev: ProfileSettings::dev(),
            staging: ProfileSettings::staging(),
            prod: ProfileSettings::prod(),
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_environment_promotion() {
        assert!(Environment::Dev.can_promote_to(&Environment::Staging));
        assert!(Environment::Staging.can_promote_to(&Environment::Prod));
        assert!(!Environment::Prod.can_promote_to(&Environment::Dev));
        assert!(!Environment::Dev.can_promote_to(&Environment::Prod));
    }

    #[test]
    fn test_environment_next() {
        assert_eq!(Environment::Dev.next(), Some(Environment::Staging));
        assert_eq!(Environment::Staging.next(), Some(Environment::Prod));
        assert_eq!(Environment::Prod.next(), None);
    }

    #[test]
    fn test_preset_profiles() {
        let presets = get_preset_profiles();
        assert!(presets.len() >= 3);
        assert!(presets.iter().any(|p| p.name == "ml-training"));
    }
}
