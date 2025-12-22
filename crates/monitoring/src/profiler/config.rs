//! Profiler configuration

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Configuration for profiling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileConfig {
    pub output_dir: PathBuf,
    pub enable_nsight: bool,
    pub enable_nvtx: bool,
    pub sampling_rate_hz: u32,
    pub max_profile_duration_secs: u64,
}

impl Default for ProfileConfig {
    fn default() -> Self {
        Self {
            output_dir: PathBuf::from("/tmp/exorust_profiles"),
            enable_nsight: false,
            enable_nvtx: false,
            sampling_rate_hz: 100,
            max_profile_duration_secs: 300,
        }
    }
}

impl ProfileConfig {
    /// Create a new profile configuration
    pub fn new(output_dir: PathBuf) -> Self {
        Self {
            output_dir,
            ..Default::default()
        }
    }

    /// Enable Nsight Compute profiling
    pub fn with_nsight(mut self, enable: bool) -> Self {
        self.enable_nsight = enable;
        self
    }

    /// Enable NVTX markers
    pub fn with_nvtx(mut self, enable: bool) -> Self {
        self.enable_nvtx = enable;
        self
    }

    /// Set sampling rate
    pub fn with_sampling_rate(mut self, rate_hz: u32) -> Self {
        self.sampling_rate_hz = rate_hz;
        self
    }

    /// Set maximum profile duration
    pub fn with_max_duration(mut self, duration_secs: u64) -> Self {
        self.max_profile_duration_secs = duration_secs;
        self
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.sampling_rate_hz == 0 {
            return Err("Sampling rate must be greater than 0".to_string());
        }

        if self.max_profile_duration_secs == 0 {
            return Err("Max profile duration must be greater than 0".to_string());
        }

        if !self.output_dir.is_absolute() {
            return Err("Output directory must be an absolute path".to_string());
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profile_config_default() {
        let config = ProfileConfig::default();

        assert_eq!(config.output_dir, PathBuf::from("/tmp/exorust_profiles"));
        assert!(!config.enable_nsight);
        assert!(!config.enable_nvtx);
        assert_eq!(config.sampling_rate_hz, 100);
        assert_eq!(config.max_profile_duration_secs, 300);
    }

    #[test]
    fn test_profile_config_builder() {
        let config = ProfileConfig::new(PathBuf::from("/var/profiles"))
            .with_nsight(true)
            .with_nvtx(true)
            .with_sampling_rate(200)
            .with_max_duration(600);

        assert_eq!(config.output_dir, PathBuf::from("/var/profiles"));
        assert!(config.enable_nsight);
        assert!(config.enable_nvtx);
        assert_eq!(config.sampling_rate_hz, 200);
        assert_eq!(config.max_profile_duration_secs, 600);
    }

    #[test]
    fn test_profile_config_validation_success() {
        let config = ProfileConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_profile_config_validation_zero_sampling_rate() {
        let config = ProfileConfig::default().with_sampling_rate(0);

        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Sampling rate"));
    }

    #[test]
    fn test_profile_config_validation_zero_duration() {
        let config = ProfileConfig::default().with_max_duration(0);

        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Max profile duration"));
    }

    #[test]
    fn test_profile_config_validation_relative_path() {
        let config = ProfileConfig {
            output_dir: PathBuf::from("relative/path"),
            ..Default::default()
        };

        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("absolute path"));
    }
}
