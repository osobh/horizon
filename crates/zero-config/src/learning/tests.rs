//! Tests for the learning module

use super::*;

#[tokio::test]
async fn test_modular_learning_integration() {
    // Test that all modules work together
    let learner = BehavioralLearner::new();
    let _collector = PatternCollector::new(learner.pattern_store.clone());
    let _knowledge = KnowledgeTransfer::new(learner.pattern_store.clone());

    // Verify basic functionality
    let stats = learner.get_statistics().await.unwrap();
    assert_eq!(stats.total_patterns, 0);

    // Test pattern store
    let store = PatternStore::new();
    assert_eq!(store.patterns.len(), 0);
}

#[cfg(test)]
mod tests {
    use super::super::*;
    use crate::{
        AgentConfiguration, BackupFrequency, BackupPolicy, Dependency, DependencyType,
        DeploymentMetrics, NetworkConfiguration, PersonalityConfig, ResourceRequirements,
        ScalingPolicy, StorageConfiguration,
    };
    use std::collections::HashMap;
    use std::sync::Arc;
    use tokio::sync::RwLock;

    fn create_test_config() -> AgentConfiguration {
        AgentConfiguration {
            agent_id: "test-001".to_string(),
            language: "rust".to_string(),
            framework: Some("tokio".to_string()),
            dependencies: vec![
                Dependency {
                    name: "tokio".to_string(),
                    version: Some("1.0".to_string()),
                    dependency_type: DependencyType::WebFramework,
                },
                Dependency {
                    name: "sqlx".to_string(),
                    version: Some("0.7".to_string()),
                    dependency_type: DependencyType::Database,
                },
            ],
            resources: ResourceRequirements {
                cpu_cores: 2.0,
                memory_gb: 4.0,
                gpu_units: 0.0,
                storage_gb: 10.0,
                network_bandwidth_mbps: 100.0,
            },
            scaling: ScalingPolicy {
                min_replicas: 2,
                max_replicas: 10,
                target_cpu_percent: 70.0,
                scale_up_threshold: 80.0,
                scale_down_threshold: 30.0,
            },
            networking: NetworkConfiguration {
                expose_ports: vec![8080],
                ingress_rules: vec![],
                service_mesh: false,
            },
            storage: StorageConfiguration {
                persistent_volumes: vec![],
                temporary_storage_gb: 1.0,
                backup_policy: BackupPolicy {
                    enabled: false,
                    frequency: BackupFrequency::Never,
                    retention_days: 0,
                },
            },
            personality: PersonalityConfig {
                risk_tolerance: 0.5,
                cooperation: 0.7,
                exploration: 0.3,
                efficiency_focus: 0.6,
                stability_preference: 0.8,
            },
        }
    }

    fn create_test_deployment() -> DeploymentOutcome {
        DeploymentOutcome {
            config: create_test_config(),
            metrics: DeploymentMetrics {
                startup_time_ms: 2000,
                avg_cpu_usage: 0.4,
                avg_memory_usage_gb: 1.2,
                error_rate: 0.01,
                throughput_rps: 1000.0,
                latency_p99_ms: 50,
            },
            success: true,
            issues: vec![],
            improvements: vec!["Could optimize memory usage".to_string()],
        }
    }

    #[tokio::test]
    async fn test_behavioral_learner_creation() {
        let learner = BehavioralLearner::new();
        let stats = learner.get_statistics().await.unwrap();
        assert_eq!(stats.total_patterns, 0);
    }

    #[tokio::test]
    async fn test_pattern_collection() {
        let mut learner = BehavioralLearner::new();
        let deployment = create_test_deployment();

        let result = learner.record_deployment(deployment).await;
        assert!(result.is_ok());

        let stats = learner.get_statistics().await.unwrap();
        assert_eq!(stats.total_patterns, 1);
        assert_eq!(stats.success_rate, 1.0);
    }

    #[tokio::test]
    async fn test_multiple_pattern_collection() {
        let mut learner = BehavioralLearner::new();

        // Add successful deployment
        let successful_deployment = create_test_deployment();
        learner
            .record_deployment(successful_deployment)
            .await
            .unwrap();

        // Add failed deployment with different configuration to avoid deduplication
        let mut failed_deployment = create_test_deployment();
        failed_deployment.success = false;
        failed_deployment.config.resources.cpu_cores = 4.0; // Make it different
        failed_deployment.config.resources.memory_gb = 8.0; // Make it different
        learner.record_deployment(failed_deployment).await.unwrap();

        let stats = learner.get_statistics().await.unwrap();
        assert_eq!(stats.total_patterns, 2);
        assert_eq!(stats.success_rate, 0.5);
    }

    #[tokio::test]
    async fn test_feature_extraction() {
        let collector = PatternCollector::new(Arc::new(RwLock::new(PatternStore::new())));
        let config = create_test_config();

        let features = collector.extract_features(&config).await.unwrap();

        assert_eq!(features.language_features.language, "rust");
        assert_eq!(
            features.language_features.framework,
            Some("tokio".to_string())
        );
        assert_eq!(features.language_features.dependency_count, 2);
        assert!(features.language_features.has_web_framework);
        assert!(features.language_features.has_database);
        assert!(!features.language_features.has_cache);
        assert!(!features.language_features.has_ml_framework);
    }

    #[tokio::test]
    async fn test_dependency_features_extraction() {
        let collector = PatternCollector::new(Arc::new(RwLock::new(PatternStore::new())));
        let config = create_test_config();

        let dep_features = collector.extract_dependency_features(&config);

        assert_eq!(dep_features.database_count, 1);
        assert_eq!(dep_features.web_framework_count, 1);
        assert_eq!(dep_features.cache_count, 0);
        assert_eq!(dep_features.ml_framework_count, 0);
        assert_eq!(dep_features.message_queue_count, 0);
        assert_eq!(dep_features.total_dependencies, 2);
    }

    #[tokio::test]
    async fn test_clear_patterns() {
        let mut learner = BehavioralLearner::new();

        // Add a pattern
        learner
            .record_deployment(create_test_deployment())
            .await
            .unwrap();

        let stats_before = learner.get_statistics().await.unwrap();
        assert_eq!(stats_before.total_patterns, 1);

        // Clear patterns
        learner.clear_patterns().await.unwrap();

        let stats_after = learner.get_statistics().await.unwrap();
        assert_eq!(stats_after.total_patterns, 0);
    }

    #[tokio::test]
    async fn test_pattern_similarity_calculation() {
        let learner = BehavioralLearner::new();
        let knowledge_transfer = KnowledgeTransfer::new(learner.pattern_store.clone());

        // Create two similar feature vectors
        let features1 = FeatureVector {
            language_features: LanguageFeatures {
                language: "rust".to_string(),
                framework: Some("tokio".to_string()),
                has_web_framework: true,
                has_database: true,
                has_cache: false,
                has_ml_framework: false,
                has_message_queue: false,
                dependency_count: 2,
                total_lines: 1000,
                complexity_score: 5.0,
            },
            dependency_features: DependencyFeatures {
                database_count: 1,
                cache_count: 0,
                web_framework_count: 1,
                ml_framework_count: 0,
                message_queue_count: 0,
                total_dependencies: 2,
                web_frameworks: 1,
                databases: 1,
                caches: 0,
                ml_frameworks: 0,
                message_queues: 0,
            },
            resource_features: ResourceFeatures {
                cpu_cores: 2.0,
                memory_gb: 4.0,
                gpu_units: 0.0,
                storage_gb: 10.0,
                network_bandwidth_mbps: 100.0,
                resource_intensity: 0.3,
            },
            scaling_features: ScalingFeatures {
                min_replicas: 2,
                max_replicas: 10,
                target_cpu_percent: 70.0,
                scale_up_threshold: 80.0,
                scale_down_threshold: 30.0,
                scaling_aggressiveness: 0.5,
            },
            personality_features: PersonalityFeatures {
                risk_tolerance: 0.5,
                cooperation: 0.7,
                exploration: 0.3,
                efficiency_focus: 0.6,
                stability_preference: 0.8,
                personality_type: "stable".to_string(),
            },
        };
        let features2 = features1.clone(); // Identical features

        let similarity = knowledge_transfer
            .calculate_similarity(&features1, &features2)
            .await
            .unwrap();
        assert!(similarity > 0.8); // Should be very similar
    }

    #[tokio::test]
    async fn test_language_similarity_calculation() {
        let knowledge_transfer = KnowledgeTransfer::new(Arc::new(RwLock::new(PatternStore::new())));

        let lang1 = LanguageFeatures {
            language: "rust".to_string(),
            framework: Some("tokio".to_string()),
            has_web_framework: true,
            has_database: true,
            has_cache: false,
            has_ml_framework: false,
            has_message_queue: false,
            dependency_count: 2,
            total_lines: 1000,
            complexity_score: 0.6,
        };

        let lang2 = lang1.clone();

        let lang3 = LanguageFeatures {
            language: "python".to_string(),
            framework: Some("fastapi".to_string()),
            has_web_framework: true,
            has_database: false,
            has_cache: true,
            has_ml_framework: true,
            has_message_queue: false,
            dependency_count: 3,
            total_lines: 1500,
            complexity_score: 0.8,
        };

        let similarity_same = knowledge_transfer.calculate_language_similarity(&lang1, &lang2);
        let similarity_different = knowledge_transfer.calculate_language_similarity(&lang1, &lang3);

        assert!(similarity_same > similarity_different);
        assert!(similarity_same > 0.5);
        assert!(similarity_different < 0.5);
    }

    #[tokio::test]
    async fn test_pattern_store_operations() {
        let mut store = PatternStore::new();

        let pattern = DeploymentPattern {
            id: "test-001".to_string(),
            language: "rust".to_string(),
            framework: Some("tokio".to_string()),
            features: FeatureVector {
                language_features: LanguageFeatures {
                    language: "rust".to_string(),
                    framework: Some("tokio".to_string()),
                    has_web_framework: true,
                    has_database: false,
                    has_cache: false,
                    has_ml_framework: false,
                    has_message_queue: false,
                    dependency_count: 1,
                    total_lines: 1000,
                    complexity_score: 5.0,
                },
                dependency_features: DependencyFeatures {
                    database_count: 0,
                    cache_count: 0,
                    web_framework_count: 1,
                    ml_framework_count: 0,
                    message_queue_count: 0,
                    total_dependencies: 1,
                    web_frameworks: 1,
                    databases: 0,
                    caches: 0,
                    ml_frameworks: 0,
                    message_queues: 0,
                },
                resource_features: ResourceFeatures {
                    cpu_cores: 1.0,
                    memory_gb: 2.0,
                    gpu_units: 0.0,
                    storage_gb: 5.0,
                    network_bandwidth_mbps: 50.0,
                    resource_intensity: 0.2,
                },
                scaling_features: ScalingFeatures {
                    min_replicas: 1,
                    max_replicas: 5,
                    target_cpu_percent: 70.0,
                    scale_up_threshold: 80.0,
                    scale_down_threshold: 30.0,
                    scaling_aggressiveness: 0.3,
                },
                personality_features: PersonalityFeatures {
                    risk_tolerance: 0.5,
                    cooperation: 0.7,
                    exploration: 0.3,
                    efficiency_focus: 0.6,
                    stability_preference: 0.8,
                    personality_type: "stable".to_string(),
                },
            },
            config: create_test_config(),
            metrics: DeploymentMetrics {
                startup_time_ms: 1000,
                avg_cpu_usage: 0.3,
                avg_memory_usage_gb: 0.8,
                error_rate: 0.0,
                throughput_rps: 500.0,
                latency_p99_ms: 30,
            },
            success: true,
            confidence: 1.0,
            usage_count: 1,
            last_used: chrono::Utc::now(),
            personality_adjustments: HashMap::new(),
            cpu_cores: 1.0,
            memory_gb: 2.0,
            success_rate: 1.0,
        };

        store.add_pattern(pattern);
        assert_eq!(store.patterns.len(), 1);

        let distribution = store.get_language_distribution();
        assert_eq!(distribution.get("rust"), Some(&1));

        let framework_dist = store.get_framework_distribution();
        assert_eq!(framework_dist.get("tokio"), Some(&1));

        assert_eq!(store.calculate_success_rate(), 1.0);
        assert_eq!(store.calculate_average_confidence(), 1.0);
    }
}
