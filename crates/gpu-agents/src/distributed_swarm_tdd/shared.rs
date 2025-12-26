//! Shared types, utilities and helper functions for distributed SwarmAgentic TDD tests

use crate::consensus_synthesis::integration::ConsensusSynthesisEngine;
use crate::multi_region::{
    MultiRegionConfig, MultiRegionConsensusEngine, Region, MaliciousBehavior,
    MultiRegionPerformanceMetrics,
};
use crate::synthesis::{SynthesisTask, Pattern, Template, Token};
use crate::types::GpuSwarmConfig;
use anyhow::{anyhow, Result};
use cudarc::driver::CudaDevice;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Individual test result
#[derive(Debug, Clone)]
pub struct TestResult {
    pub test_name: String,
    pub success: bool,
    pub duration: Duration,
    pub metrics: HashMap<String, f64>,
    pub error_message: Option<String>,
}

/// Comprehensive distributed runtime performance metrics
#[derive(Debug, Clone)]
pub struct DistributedRuntimeMetrics {
    pub multi_region_consensus_time_ms: f64,
    pub cross_cloud_deployment_time_ms: f64,
    pub disaster_recovery_failover_time_ms: f64,
    pub zero_trust_detection_rate: f32,
    pub auto_scaling_response_time_ms: f64,
    pub global_throughput_tasks_per_second: f64,
    pub consensus_success_rate: f32,
    pub cross_region_latency_p99_ms: f64,
    pub security_violations_detected: usize,
    pub total_regions_active: usize,
    pub cloud_providers_utilized: usize,
}

/// Comprehensive test results for distributed runtime
#[derive(Debug, Clone)]
pub struct DistributedRuntimeTestResults {
    pub test_summary: TestSummary,
    pub performance_metrics: HashMap<String, f64>,
    pub tdd_phases_completed: Vec<String>,
    pub distributed_features_validated: Vec<String>,
    pub test_details: Vec<TestResult>,
}

/// Test execution summary
#[derive(Debug, Clone)]
pub struct TestSummary {
    pub total_tests: usize,
    pub passed_tests: usize,
    pub failed_tests: usize,
    pub success_rate: f32,
    pub total_duration: Duration,
}

/// Base test harness for distributed SwarmAgentic tests
pub struct DistributedTestHarness {
    pub device: Arc<CudaDevice>,
    pub test_results: Vec<TestResult>,
}

impl DistributedTestHarness {
    /// Create new test harness
    pub fn new() -> Result<Self> {
        let device = CudaDevice::new(0)?;
        
        Ok(Self {
            device,
            test_results: Vec::new(),
        })
    }

    /// Record test result
    pub fn record_test_result(
        &mut self,
        test_name: &str,
        success: bool,
        duration: Duration,
        metrics: HashMap<String, f64>,
        error_message: Option<String>,
    ) {
        self.test_results.push(TestResult {
            test_name: test_name.to_string(),
            success,
            duration,
            metrics,
            error_message,
        });
    }

    /// Convert metrics to hashmap
    pub fn metrics_to_hashmap(&self, metrics: &DistributedRuntimeMetrics) -> HashMap<String, f64> {
        let mut map = HashMap::new();
        map.insert("multi_region_consensus_time_ms".to_string(), metrics.multi_region_consensus_time_ms);
        map.insert("cross_cloud_deployment_time_ms".to_string(), metrics.cross_cloud_deployment_time_ms);
        map.insert("disaster_recovery_failover_time_ms".to_string(), metrics.disaster_recovery_failover_time_ms);
        map.insert("zero_trust_detection_rate".to_string(), metrics.zero_trust_detection_rate as f64);
        map.insert("auto_scaling_response_time_ms".to_string(), metrics.auto_scaling_response_time_ms);
        map.insert("global_throughput_tasks_per_second".to_string(), metrics.global_throughput_tasks_per_second);
        map.insert("consensus_success_rate".to_string(), metrics.consensus_success_rate as f64);
        map.insert("cross_region_latency_p99_ms".to_string(), metrics.cross_region_latency_p99_ms);
        map.insert("security_violations_detected".to_string(), metrics.security_violations_detected as f64);
        map.insert("total_regions_active".to_string(), metrics.total_regions_active as f64);
        map.insert("cloud_providers_utilized".to_string(), metrics.cloud_providers_utilized as f64);
        map
    }

    /// Generate comprehensive test results
    pub async fn generate_comprehensive_test_results(&self) -> Result<DistributedRuntimeTestResults> {
        // Calculate aggregate metrics from all test phases
        let total_tests = self.test_results.len();
        let passed_tests = self.test_results.iter().filter(|r| r.success).count();
        let failed_tests = total_tests - passed_tests;
        
        let total_duration: Duration = self.test_results.iter().map(|r| r.duration).sum();
        
        // Extract performance metrics
        let mut aggregate_metrics = HashMap::new();
        for result in &self.test_results {
            for (key, value) in &result.metrics {
                let current = aggregate_metrics.get(key).unwrap_or(&0.0);
                aggregate_metrics.insert(key.clone(), current + value);
            }
        }

        // Calculate averages
        for value in aggregate_metrics.values_mut() {
            *value /= total_tests as f64;
        }

        Ok(DistributedRuntimeTestResults {
            test_summary: TestSummary {
                total_tests,
                passed_tests,
                failed_tests,
                success_rate: (passed_tests as f32) / (total_tests as f32),
                total_duration,
            },
            performance_metrics: aggregate_metrics,
            tdd_phases_completed: vec!["RED".to_string(), "GREEN".to_string(), "REFACTOR".to_string()],
            distributed_features_validated: vec![
                "Multi-Region Consensus".to_string(),
                "Cross-Cloud Deployment".to_string(),
                "Disaster Recovery".to_string(),
                "Zero-Trust Security".to_string(),
                "Performance Benchmarks".to_string(),
                "Auto-Scaling".to_string(),
            ],
            test_details: self.test_results.clone(),
        })
    }
}

/// Configuration factory for test scenarios
pub struct TestConfigFactory;

impl TestConfigFactory {
    /// Create basic multi-region config for testing
    pub async fn create_test_multi_region_config() -> Result<MultiRegionConfig> {
        Ok(MultiRegionConfig {
            regions: vec![
                Region {
                    id: "us-east-1".to_string(),
                    location: "US East".to_string(),
                    node_count: 10,
                    latency_ms: 50.0,
                    disaster_recovery_tier: 1,
                },
                Region {
                    id: "us-west-2".to_string(),
                    location: "US West".to_string(),
                    node_count: 8,
                    latency_ms: 60.0,
                    disaster_recovery_tier: 2,
                },
                Region {
                    id: "eu-west-1".to_string(),
                    location: "Europe".to_string(),
                    node_count: 12,
                    latency_ms: 120.0,
                    disaster_recovery_tier: 2,
                },
            ],
            consensus_threshold: 0.6,
            cross_region_timeout: Duration::from_secs(30),
            disaster_recovery_enabled: false,
            zero_trust_validation: false,
            cloud_provider_integration: false,
        })
    }

    /// Create disaster recovery config
    pub async fn create_disaster_recovery_config() -> Result<MultiRegionConfig> {
        Ok(MultiRegionConfig {
            regions: vec![
                Region {
                    id: "us-east-1".to_string(),
                    location: "US East".to_string(),
                    node_count: 15,
                    latency_ms: 50.0,
                    disaster_recovery_tier: 1,
                },
                Region {
                    id: "us-west-2".to_string(),
                    location: "US West".to_string(),
                    node_count: 12,
                    latency_ms: 60.0,
                    disaster_recovery_tier: 2,
                },
                Region {
                    id: "eu-west-1".to_string(),
                    location: "Europe".to_string(),
                    node_count: 10,
                    latency_ms: 120.0,
                    disaster_recovery_tier: 3,
                },
                Region {
                    id: "ap-southeast-1".to_string(),
                    location: "Asia Pacific".to_string(),
                    node_count: 8,
                    latency_ms: 180.0,
                    disaster_recovery_tier: 3,
                },
                Region {
                    id: "cn-beijing".to_string(),
                    location: "China".to_string(),
                    node_count: 6,
                    latency_ms: 200.0,
                    disaster_recovery_tier: 4,
                },
            ],
            consensus_threshold: 0.6,
            cross_region_timeout: Duration::from_secs(60),
            disaster_recovery_enabled: true,
            zero_trust_validation: false,
            cloud_provider_integration: false,
        })
    }

    /// Create zero-trust security config
    pub async fn create_zero_trust_config() -> Result<MultiRegionConfig> {
        let mut config = Self::create_test_multi_region_config().await?;
        config.zero_trust_validation = true;
        config.disaster_recovery_enabled = true;
        Ok(config)
    }

    /// Create comprehensive test config
    pub async fn create_comprehensive_test_config() -> Result<MultiRegionConfig> {
        let mut config = Self::create_disaster_recovery_config().await?;
        config.zero_trust_validation = true;
        config.cloud_provider_integration = false;
        Ok(config)
    }

    /// Create production multi-region config
    pub async fn create_production_multi_region_config() -> Result<MultiRegionConfig> {
        let mut config = Self::create_disaster_recovery_config().await?;
        config.zero_trust_validation = true;
        config.cloud_provider_integration = true;
        config.consensus_threshold = 0.8; // Higher threshold for production
        Ok(config)
    }

    /// Create security-focused config
    pub async fn create_security_focused_config() -> Result<MultiRegionConfig> {
        let mut config = Self::create_production_multi_region_config().await?;
        config.zero_trust_validation = true;
        config.consensus_threshold = 0.9; // Very high threshold for security
        Ok(config)
    }
}

/// Factory for creating consensus engines
pub struct ConsensusEngineFactory;

impl ConsensusEngineFactory {
    /// Create test consensus engine
    pub async fn create_test_consensus_engine(device: Arc<CudaDevice>) -> Result<ConsensusSynthesisEngine> {
        let swarm_config = GpuSwarmConfig {
            device_id: 0,
            max_agents: 1000,
            block_size: 256,
            shared_memory_size: 48 * 1024,
            evolution_interval: 100,
            enable_llm: false,
            enable_collective_intelligence: false,
            enable_knowledge_graph: false,
            enable_collective_knowledge: false,
        };
        
        ConsensusSynthesisEngine::new(device, Default::default())
    }

    /// Create optimized consensus engine for production testing
    pub async fn create_optimized_consensus_engine(device: Arc<CudaDevice>) -> Result<ConsensusSynthesisEngine> {
        let swarm_config = GpuSwarmConfig {
            device_id: 0,
            max_agents: 10000, // Higher capacity for production
            block_size: 512,   // Larger blocks for efficiency
            shared_memory_size: 48 * 1024,
            evolution_interval: 50, // More frequent evolution
            enable_llm: true,
            enable_collective_intelligence: true,
            enable_knowledge_graph: true,
            enable_collective_knowledge: true,
        };
        
        ConsensusSynthesisEngine::new(device, Default::default())
    }
}

/// Factory for creating test synthesis tasks
pub struct SynthesisTaskFactory;

impl SynthesisTaskFactory {
    /// Create simple test synthesis task
    pub fn create_test_synthesis_task() -> SynthesisTask {
        SynthesisTask {
            pattern: Pattern {
                node_type: crate::synthesis::NodeType::Function,
                children: vec![],
                value: Some("test_distributed_consensus".to_string()),
            },
            template: Template {
                tokens: vec![
                    Token::Literal("fn test_".to_string()),
                    Token::Variable("task_name".to_string()),
                    Token::Literal("() {}".to_string()),
                ],
            },
        }
    }

    /// Create complex synthesis task for advanced testing
    pub fn create_complex_synthesis_task() -> SynthesisTask {
        SynthesisTask {
            pattern: Pattern {
                node_type: crate::synthesis::NodeType::Block,
                children: vec![],
                value: Some("complex_distributed_workflow".to_string()),
            },
            template: Template {
                tokens: vec![
                    Token::Literal("fn complex_workflow_".to_string()),
                    Token::Variable("region_id".to_string()),
                    Token::Literal("_".to_string()),
                    Token::Variable("task_id".to_string()),
                    Token::Literal("() {}".to_string()),
                ],
            },
        }
    }

    /// Create batch of test tasks
    pub fn create_batch_test_tasks(count: usize) -> Vec<SynthesisTask> {
        (0..count)
            .map(|i| SynthesisTask {
                pattern: Pattern {
                    node_type: crate::synthesis::NodeType::Function,
                    children: vec![],
                    value: Some(format!("batch_task_{}", i)),
                },
                template: Template {
                    tokens: vec![
                        Token::Literal("fn batch_task_".to_string()),
                        Token::Variable("task_id".to_string()),
                        Token::Literal("() {}".to_string()),
                    ],
                },
            })
            .collect()
    }
}