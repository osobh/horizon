//! Type definitions for behavioral learning system

use crate::{AgentConfiguration, DeploymentMetrics};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A learned deployment pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentPattern {
    pub id: String,
    pub language: String,
    pub framework: Option<String>,
    pub features: FeatureVector,
    pub config: AgentConfiguration,
    pub metrics: DeploymentMetrics,
    pub success: bool,
    pub confidence: f32,
    pub usage_count: u32,
    pub last_used: chrono::DateTime<chrono::Utc>,
    pub personality_adjustments: HashMap<String, f32>,
    pub cpu_cores: f32,
    pub memory_gb: f32,
    pub success_rate: f32,
}

/// Feature vector for pattern matching
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureVector {
    pub language_features: LanguageFeatures,
    pub dependency_features: DependencyFeatures,
    pub resource_features: ResourceFeatures,
    pub scaling_features: ScalingFeatures,
    pub personality_features: PersonalityFeatures,
}

/// Language-specific features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageFeatures {
    pub language: String,
    pub framework: Option<String>,
    pub has_web_framework: bool,
    pub has_database: bool,
    pub has_cache: bool,
    pub has_ml_framework: bool,
    pub has_message_queue: bool,
    pub dependency_count: usize,
    pub total_lines: usize,
    pub complexity_score: f32,
}

/// Dependency-related features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyFeatures {
    pub database_count: usize,
    pub cache_count: usize,
    pub web_framework_count: usize,
    pub ml_framework_count: usize,
    pub message_queue_count: usize,
    pub total_dependencies: usize,
    pub web_frameworks: usize,
    pub databases: usize,
    pub caches: usize,
    pub ml_frameworks: usize,
    pub message_queues: usize,
}

/// Resource requirement features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceFeatures {
    pub cpu_cores: f32,
    pub memory_gb: f32,
    pub gpu_units: f32,
    pub storage_gb: f32,
    pub network_bandwidth_mbps: f32,
    pub resource_intensity: f32,
}

/// Scaling behavior features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingFeatures {
    pub min_replicas: u32,
    pub max_replicas: u32,
    pub target_cpu_percent: f32,
    pub scale_up_threshold: f32,
    pub scale_down_threshold: f32,
    pub scaling_aggressiveness: f32,
}

/// Personality trait features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonalityFeatures {
    pub personality_type: String,
    pub risk_tolerance: f32,
    pub cooperation: f32,
    pub exploration: f32,
    pub efficiency_focus: f32,
    pub stability_preference: f32,
}

/// Statistics about the learning system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningStatistics {
    pub total_patterns: usize,
    pub languages: HashMap<String, usize>,
    pub frameworks: HashMap<String, usize>,
    pub success_rate: f32,
    pub avg_confidence: f32,
}

/// Outcome of a deployment for learning
#[derive(Debug, Clone)]
pub struct DeploymentOutcome {
    pub config: AgentConfiguration,
    pub metrics: DeploymentMetrics,
    pub success: bool,
    pub issues: Vec<String>,
    pub improvements: Vec<String>,
}

// Default implementations
impl Default for DeploymentPattern {
    fn default() -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            language: "unknown".to_string(),
            framework: None,
            features: FeatureVector::default(),
            config: AgentConfiguration {
                agent_id: "default".to_string(),
                language: "unknown".to_string(),
                framework: None,
                dependencies: vec![],
                resources: crate::ResourceRequirements {
                    cpu_cores: 1.0,
                    memory_gb: 2.0,
                    gpu_units: 0.0,
                    storage_gb: 5.0,
                    network_bandwidth_mbps: 50.0,
                },
                scaling: crate::ScalingPolicy {
                    min_replicas: 1,
                    max_replicas: 3,
                    target_cpu_percent: 70.0,
                    scale_up_threshold: 80.0,
                    scale_down_threshold: 30.0,
                },
                networking: crate::NetworkConfiguration {
                    expose_ports: vec![],
                    ingress_rules: vec![],
                    service_mesh: false,
                },
                storage: crate::StorageConfiguration {
                    persistent_volumes: vec![],
                    temporary_storage_gb: 1.0,
                    backup_policy: crate::BackupPolicy {
                        enabled: false,
                        frequency: crate::BackupFrequency::Never,
                        retention_days: 0,
                    },
                },
                personality: crate::PersonalityConfig {
                    risk_tolerance: 0.5,
                    cooperation: 0.7,
                    exploration: 0.3,
                    efficiency_focus: 0.6,
                    stability_preference: 0.8,
                },
            },
            metrics: crate::DeploymentMetrics {
                startup_time_ms: 1000,
                avg_cpu_usage: 0.5,
                avg_memory_usage_gb: 1.0,
                error_rate: 0.01,
                throughput_rps: 100.0,
                latency_p99_ms: 100,
            },
            success: true,
            confidence: 0.5,
            usage_count: 0,
            last_used: chrono::Utc::now(),
            personality_adjustments: HashMap::new(),
            cpu_cores: 1.0,
            memory_gb: 2.0,
            success_rate: 0.5,
        }
    }
}

impl Default for FeatureVector {
    fn default() -> Self {
        Self {
            language_features: LanguageFeatures::default(),
            dependency_features: DependencyFeatures::default(),
            resource_features: ResourceFeatures::default(),
            scaling_features: ScalingFeatures::default(),
            personality_features: PersonalityFeatures::default(),
        }
    }
}

impl Default for LanguageFeatures {
    fn default() -> Self {
        Self {
            language: "unknown".to_string(),
            framework: None,
            has_web_framework: false,
            has_database: false,
            has_cache: false,
            has_ml_framework: false,
            has_message_queue: false,
            dependency_count: 0,
            total_lines: 1000,
            complexity_score: 0.5,
        }
    }
}

impl Default for DependencyFeatures {
    fn default() -> Self {
        Self {
            database_count: 0,
            cache_count: 0,
            web_framework_count: 0,
            ml_framework_count: 0,
            message_queue_count: 0,
            total_dependencies: 0,
            web_frameworks: 0,
            databases: 0,
            caches: 0,
            ml_frameworks: 0,
            message_queues: 0,
        }
    }
}

impl Default for ResourceFeatures {
    fn default() -> Self {
        Self {
            cpu_cores: 1.0,
            memory_gb: 2.0,
            gpu_units: 0.0,
            storage_gb: 5.0,
            network_bandwidth_mbps: 50.0,
            resource_intensity: 0.5,
        }
    }
}

impl Default for ScalingFeatures {
    fn default() -> Self {
        Self {
            min_replicas: 1,
            max_replicas: 3,
            target_cpu_percent: 70.0,
            scale_up_threshold: 80.0,
            scale_down_threshold: 30.0,
            scaling_aggressiveness: 0.5,
        }
    }
}

impl Default for PersonalityFeatures {
    fn default() -> Self {
        Self {
            personality_type: "balanced".to_string(),
            risk_tolerance: 0.5,
            cooperation: 0.7,
            exploration: 0.3,
            efficiency_focus: 0.6,
            stability_preference: 0.8,
        }
    }
}
