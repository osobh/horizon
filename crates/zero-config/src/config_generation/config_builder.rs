//! Agent configuration builder implementation

use crate::{analysis::CodebaseAnalysis, AgentConfiguration, Result};
use crate::{BackupFrequency, BackupPolicy, IngressRule, PersistentVolume, StorageClass};
use crate::{NetworkConfiguration, PersonalityConfig, ScalingPolicy, StorageConfiguration};

/// Builder for creating agent configurations
pub struct AgentConfigBuilder;

impl AgentConfigBuilder {
    /// Create a new config builder
    pub fn new() -> Self {
        Self
    }

    /// Create a configuration from codebase analysis
    pub async fn from_analysis(&self, analysis: &CodebaseAnalysis) -> Result<AgentConfiguration> {
        let agent_id = format!("agent-{}", &uuid::Uuid::new_v4().to_string()[..8]);
        Ok(AgentConfiguration {
            agent_id,
            language: analysis.language.clone(),
            framework: analysis.framework.clone(),
            dependencies: analysis.dependencies.clone(),
            resources: analysis.resources.clone(),
            scaling: self.create_default_scaling_policy(&analysis.language),
            networking: self.create_network_config(&analysis.dependencies),
            storage: self
                .create_storage_config(&analysis.dependencies, analysis.resources.storage_gb),
            personality: self.create_personality_config(&analysis.language, &analysis.framework),
        })
    }

    /// Create default scaling policy based on language
    fn create_default_scaling_policy(&self, language: &str) -> ScalingPolicy {
        match language {
            "rust" | "go" => ScalingPolicy {
                min_replicas: 1,
                max_replicas: 10,
                target_cpu_percent: 80.0,
                scale_up_threshold: 85.0,
                scale_down_threshold: 30.0,
            },
            "python" | "javascript" | "typescript" => ScalingPolicy {
                min_replicas: 2,
                max_replicas: 15,
                target_cpu_percent: 60.0,
                scale_up_threshold: 75.0,
                scale_down_threshold: 25.0,
            },
            "java" => ScalingPolicy {
                min_replicas: 2,
                max_replicas: 8,
                target_cpu_percent: 70.0,
                scale_up_threshold: 80.0,
                scale_down_threshold: 35.0,
            },
            _ => ScalingPolicy {
                min_replicas: 1,
                max_replicas: 5,
                target_cpu_percent: 70.0,
                scale_up_threshold: 80.0,
                scale_down_threshold: 40.0,
            },
        }
    }

    /// Create network configuration based on dependencies
    fn create_network_config(&self, dependencies: &[crate::Dependency]) -> NetworkConfiguration {
        let mut network_config = NetworkConfiguration {
            expose_ports: vec![],
            ingress_rules: vec![],
            service_mesh: false,
        };

        // Check for web frameworks and add appropriate ports
        let has_web_framework = dependencies
            .iter()
            .any(|d| matches!(d.dependency_type, crate::DependencyType::WebFramework));

        if has_web_framework {
            network_config.expose_ports.push(8080);
            network_config.ingress_rules.push(IngressRule {
                path: "/".to_string(),
                methods: vec!["GET".to_string(), "POST".to_string()],
                rate_limit: Some(1000),
            });
        }

        // Enable service mesh for complex deployments
        if dependencies.len() > 5 {
            network_config.service_mesh = true;
        }

        network_config
    }

    /// Create storage configuration based on dependencies and requirements
    fn create_storage_config(
        &self,
        dependencies: &[crate::Dependency],
        storage_gb: f32,
    ) -> StorageConfiguration {
        let mut storage_config = StorageConfiguration {
            persistent_volumes: vec![],
            temporary_storage_gb: (storage_gb * 0.2).max(1.0), // 20% for temp storage
            backup_policy: BackupPolicy {
                enabled: false,
                frequency: BackupFrequency::Never,
                retention_days: 0,
            },
        };

        // Add persistent storage for databases
        let has_database = dependencies
            .iter()
            .any(|d| matches!(d.dependency_type, crate::DependencyType::Database));

        if has_database {
            storage_config.persistent_volumes.push(PersistentVolume {
                mount_path: "/data".to_string(),
                size_gb: (storage_gb * 0.6).max(5.0), // 60% for database
                storage_class: StorageClass::Ssd,
            });

            // Enable backups for databases
            storage_config.backup_policy = BackupPolicy {
                enabled: true,
                frequency: BackupFrequency::Daily,
                retention_days: 7,
            };
        }

        storage_config
    }

    /// Create personality configuration based on language and framework
    fn create_personality_config(
        &self,
        language: &str,
        framework: &Option<String>,
    ) -> PersonalityConfig {
        let mut personality = PersonalityConfig {
            risk_tolerance: 0.5,
            cooperation: 0.7,
            exploration: 0.3,
            efficiency_focus: 0.6,
            stability_preference: 0.8,
        };

        // Language-specific personality adjustments
        match language {
            "rust" => {
                personality.stability_preference = 0.9;
                personality.efficiency_focus = 0.8;
                personality.risk_tolerance = 0.3;
            }
            "python" => {
                personality.exploration = 0.7;
                personality.cooperation = 0.8;
                personality.risk_tolerance = 0.6;
            }
            "javascript" | "typescript" => {
                personality.exploration = 0.8;
                personality.risk_tolerance = 0.7;
                personality.stability_preference = 0.5;
            }
            _ => {}
        }

        // Framework-specific adjustments
        if let Some(fw) = framework {
            match fw.as_str() {
                "react" | "vue" | "angular" => {
                    personality.exploration += 0.1;
                    personality.risk_tolerance += 0.1;
                }
                "express" | "fastapi" | "django" => {
                    personality.cooperation += 0.1;
                    personality.stability_preference += 0.1;
                }
                _ => {}
            }
        }

        // Clamp all values to valid ranges
        personality.risk_tolerance = personality.risk_tolerance.clamp(0.0, 1.0);
        personality.cooperation = personality.cooperation.clamp(0.0, 1.0);
        personality.exploration = personality.exploration.clamp(0.0, 1.0);
        personality.efficiency_focus = personality.efficiency_focus.clamp(0.0, 1.0);
        personality.stability_preference = personality.stability_preference.clamp(0.0, 1.0);

        personality
    }

    /// Optimize configuration for specific dependencies
    pub async fn optimize_for_dependencies(
        &self,
        mut config: AgentConfiguration,
        dependencies: &[crate::Dependency],
    ) -> Result<AgentConfiguration> {
        // Analyze dependencies to optimize resource allocation
        let db_count = dependencies
            .iter()
            .filter(|d| matches!(d.dependency_type, crate::DependencyType::Database))
            .count();
        let cache_count = dependencies
            .iter()
            .filter(|d| matches!(d.dependency_type, crate::DependencyType::Cache))
            .count();
        let web_count = dependencies
            .iter()
            .filter(|d| matches!(d.dependency_type, crate::DependencyType::WebFramework))
            .count();

        // Adjust resources based on dependencies
        if db_count > 0 {
            config.resources.memory_gb *= 1.5; // Increase memory for database operations
            config.resources.cpu_cores += 0.5; // Add CPU for database processing
        }

        if cache_count > 0 {
            config.resources.memory_gb *= 1.2; // Increase memory for caching
        }

        if web_count > 0 {
            config.resources.network_bandwidth_mbps *= 2.0; // Increase bandwidth for web services
            config.scaling.max_replicas += 2; // Allow more replicas for web services
        }

        Ok(config)
    }
}

impl Default for AgentConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::{CodebaseAnalysis, ComplexityMetrics};
    use crate::{Dependency, DependencyType, ResourceRequirements};

    #[tokio::test]
    async fn test_config_builder_creation() {
        let builder = AgentConfigBuilder::new();
        assert!(true); // Basic instantiation test
    }

    #[tokio::test]
    async fn test_rust_scaling_policy() {
        let builder = AgentConfigBuilder::new();
        let policy = builder.create_default_scaling_policy("rust");

        assert_eq!(policy.min_replicas, 1);
        assert_eq!(policy.max_replicas, 10);
        assert_eq!(policy.target_cpu_percent, 80.0);
    }

    #[tokio::test]
    async fn test_network_config_with_web_framework() {
        let builder = AgentConfigBuilder::new();
        let deps = vec![Dependency {
            name: "express".to_string(),
            version: None,
            dependency_type: DependencyType::WebFramework,
        }];

        let network_config = builder.create_network_config(&deps);
        assert!(network_config.expose_ports.contains(&8080));
        assert_eq!(network_config.ingress_rules.len(), 1);
    }
}
