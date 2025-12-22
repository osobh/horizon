//! Tests for GPU/CPU agent integration with shared storage
//! 
//! Tests the complete integration including:
//! - Job submission from CPU to GPU agents
//! - GPU agent job processing
//! - Result return from GPU to CPU
//! - Resource isolation verification

use super::*;
use shared_storage::*;
use crate::GpuAgent;
use cpu_agents::{CpuAgent, IoManager, Orchestrator};
use std::sync::Arc;
use std::time::Duration;
use tempfile::tempdir;
use tokio::time::sleep;

// =============================================================================
// Test Helpers
// =============================================================================

async fn create_test_environment() -> anyhow::Result<TestEnvironment> {
    let temp_dir = tempdir()?;
    
    let storage_config = SharedStorageConfig {
        base_path: temp_dir.path().to_path_buf(),
        max_job_size: 10 * 1024 * 1024, // 10MB
        cleanup_interval: Duration::from_secs(60),
        job_ttl: Duration::from_secs(300),
        enable_compression: false,
        max_concurrent_jobs: 100,
        inbox_dir: "inbox".to_string(),
        outbox_dir: "outbox".to_string(),
        processing_dir: "processing".to_string(),
    };
    
    let storage_manager = Arc::new(SharedStorageManager::new(storage_config).await?);
    
    let integration_config = IntegrationConfig {
        storage_manager: storage_manager.clone(),
        gpu_poll_interval: Duration::from_millis(100),
        cpu_poll_interval: Duration::from_millis(100),
        max_batch_size: 10,
        enable_gpu_direct: false,
    };
    
    Ok(TestEnvironment {
        temp_dir,
        storage_manager,
        integration_config,
    })
}

struct TestEnvironment {
    #[allow(dead_code)]
    temp_dir: tempfile::TempDir,
    storage_manager: Arc<SharedStorageManager>,
    integration_config: IntegrationConfig,
}

// =============================================================================
// Unit Tests
// =============================================================================

#[test]
fn test_integration_config_creation() {
    let config = IntegrationConfig::default();
    assert_eq!(config.gpu_poll_interval, Duration::from_millis(50));
    assert_eq!(config.cpu_poll_interval, Duration::from_millis(100));
    assert_eq!(config.max_batch_size, 32);
    assert!(!config.enable_gpu_direct);
}

#[tokio::test]
async fn test_gpu_agent_connector_creation() -> anyhow::Result<()> {
    let env = create_test_environment().await?;
    
    let connector = GpuAgentConnector::new(
        0, // GPU ID
        env.storage_manager.clone(),
        env.integration_config.gpu_poll_interval,
    );
    
    assert_eq!(connector.gpu_id(), 0);
    assert!(connector.is_running());
    
    connector.shutdown().await?;
    assert!(!connector.is_running());
    
    Ok(())
}

#[tokio::test]
async fn test_cpu_agent_connector_creation() -> anyhow::Result<()> {
    let env = create_test_environment().await?;
    
    let connector = CpuAgentConnector::new(
        1, // CPU ID
        env.storage_manager.clone(),
        env.integration_config.cpu_poll_interval,
    );
    
    assert_eq!(connector.cpu_id(), 1);
    assert!(connector.is_running());
    
    connector.shutdown().await?;
    assert!(!connector.is_running());
    
    Ok(())
}

#[tokio::test]
async fn test_job_submission_pipeline() -> anyhow::Result<()> {
    let env = create_test_environment().await?;
    
    let pipeline = JobSubmissionPipeline::new(env.storage_manager.clone());
    
    // Create test job
    let data = vec![1, 2, 3, 4, 5];
    let job_request = JobRequest {
        job_type: JobType::ComputeRequest,
        target: TargetAgent::Gpu(0),
        data: data.clone(),
        priority: JobPriority::Normal,
        metadata: Default::default(),
    };
    
    // Submit job
    let job_id = pipeline.submit_job(1, job_request).await?;
    
    // Verify job was submitted
    let job = env.storage_manager.get_job(&job_id).await?;
    assert!(job.is_some());
    
    let retrieved_job = job.unwrap();
    assert_eq!(retrieved_job.data, data);
    assert_eq!(retrieved_job.source_agent, AgentId::CpuAgent(1));
    assert_eq!(retrieved_job.target_agent, AgentId::GpuAgent(0));
    
    Ok(())
}

#[tokio::test]
async fn test_data_flow_pattern_compute() -> anyhow::Result<()> {
    let env = create_test_environment().await?;
    
    let pattern = DataFlowPattern::Compute {
        preprocessing: Some(PreprocessingStep::Normalize),
        postprocessing: Some(PostprocessingStep::Aggregate),
    };
    
    let handler = pattern.create_handler();
    
    // Test preprocessing
    let input_data = vec![0u8, 128, 255];
    let preprocessed = handler.preprocess(&input_data)?;
    
    // Should normalize to [0.0, 0.5, 1.0] encoded as bytes
    assert_eq!(preprocessed.len(), 12); // 3 * 4 bytes for f32
    
    Ok(())
}

#[tokio::test]
async fn test_data_flow_pattern_streaming() -> anyhow::Result<()> {
    let pattern = DataFlowPattern::Streaming {
        chunk_size: 1024,
        window_size: 4096,
    };
    
    let handler = pattern.create_handler();
    
    // Test chunking
    let large_data = vec![42u8; 3000];
    let chunks = handler.chunk_data(&large_data)?;
    
    assert_eq!(chunks.len(), 3); // 3000 / 1024 = 2.93, so 3 chunks
    assert_eq!(chunks[0].len(), 1024);
    assert_eq!(chunks[1].len(), 1024);
    assert_eq!(chunks[2].len(), 952); // Remaining bytes
    
    Ok(())
}

#[tokio::test]
async fn test_resource_isolation_verifier() -> anyhow::Result<()> {
    let verifier = ResourceIsolationVerifier::new();
    
    // Start monitoring
    verifier.start_monitoring()?;
    
    // Simulate CPU agent activity (should not use GPU)
    let cpu_agent = MockCpuAgent::new(1);
    cpu_agent.do_work()?;
    
    // Check isolation
    let cpu_metrics = verifier.get_cpu_agent_metrics(1)?;
    assert_eq!(cpu_metrics.gpu_memory_used, 0);
    assert_eq!(cpu_metrics.gpu_compute_used, 0.0);
    
    // Simulate GPU agent activity (should not do I/O)
    let gpu_agent = MockGpuAgent::new(0);
    gpu_agent.do_compute()?;
    
    let gpu_metrics = verifier.get_gpu_agent_metrics(0)?;
    assert_eq!(gpu_metrics.io_operations, 0);
    assert!(gpu_metrics.compute_utilization > 0.0);
    
    verifier.stop_monitoring()?;
    
    Ok(())
}

// =============================================================================
// Integration Tests
// =============================================================================

#[tokio::test]
async fn test_cpu_to_gpu_workflow() -> anyhow::Result<()> {
    let env = create_test_environment().await?;
    
    // Create integration manager
    let mut integration = IntegrationManager::new(env.integration_config);
    
    // Register agents
    let cpu_connector = integration.register_cpu_agent(1).await?;
    let gpu_connector = integration.register_gpu_agent(0).await?;
    
    // Start integration
    integration.start().await?;
    
    // CPU submits job
    let job_data = vec![1, 2, 3, 4, 5];
    let job_id = cpu_connector.submit_compute_job(
        job_data.clone(),
        TargetAgent::Gpu(0),
        JobPriority::High,
    ).await?;
    
    // Wait for GPU to process
    sleep(Duration::from_millis(500)).await;
    
    // GPU should have received and processed the job
    let gpu_stats = gpu_connector.get_stats().await?;
    assert!(gpu_stats.jobs_processed > 0);
    
    // Check for result
    let result = cpu_connector.poll_results(10).await?;
    assert!(!result.is_empty());
    
    let job_result = &result[0];
    assert_eq!(job_result.original_job_id, job_id);
    assert_eq!(job_result.status, JobStatus::Completed);
    
    integration.shutdown().await?;
    
    Ok(())
}

#[tokio::test]
async fn test_gpu_to_cpu_result_return() -> anyhow::Result<()> {
    let env = create_test_environment().await?;
    
    let mut integration = IntegrationManager::new(env.integration_config);
    
    let cpu_connector = integration.register_cpu_agent(1).await?;
    let gpu_connector = integration.register_gpu_agent(0).await?;
    
    integration.start().await?;
    
    // GPU completes computation and sends result
    let result_data = vec![5, 4, 3, 2, 1];
    let original_job_id = uuid::Uuid::new_v4();
    
    gpu_connector.submit_result(
        original_job_id,
        result_data.clone(),
        AgentId::CpuAgent(1),
    ).await?;
    
    // CPU should receive result
    sleep(Duration::from_millis(300)).await;
    
    let results = cpu_connector.poll_results(10).await?;
    assert!(!results.is_empty());
    
    let result = &results[0];
    assert_eq!(result.data, result_data);
    assert_eq!(result.source_agent, AgentId::GpuAgent(0));
    
    integration.shutdown().await?;
    
    Ok(())
}

#[tokio::test]
async fn test_batch_job_processing() -> anyhow::Result<()> {
    let env = create_test_environment().await?;
    
    let mut integration = IntegrationManager::new(env.integration_config);
    let cpu_connector = integration.register_cpu_agent(1).await?;
    let gpu_connector = integration.register_gpu_agent(0).await?;
    
    integration.start().await?;
    
    // Submit batch of jobs
    let mut job_ids = vec![];
    for i in 0..5 {
        let data = vec![i; 100];
        let job_id = cpu_connector.submit_compute_job(
            data,
            TargetAgent::Gpu(0),
            JobPriority::Normal,
        ).await?;
        job_ids.push(job_id);
    }
    
    // Wait for batch processing
    sleep(Duration::from_millis(1000)).await;
    
    // Check GPU processed all jobs
    let gpu_stats = gpu_connector.get_stats().await?;
    assert_eq!(gpu_stats.jobs_processed, 5);
    
    // Check all results returned
    let results = cpu_connector.poll_results(10).await?;
    assert_eq!(results.len(), 5);
    
    // Verify all job IDs match
    let result_ids: Vec<_> = results.iter().map(|r| r.original_job_id).collect();
    for job_id in job_ids {
        assert!(result_ids.contains(&job_id));
    }
    
    integration.shutdown().await?;
    
    Ok(())
}

#[tokio::test]
async fn test_resource_isolation_enforcement() -> anyhow::Result<()> {
    let env = create_test_environment().await?;
    
    let mut integration = IntegrationManager::new(env.integration_config);
    integration.enable_isolation_monitoring()?;
    
    let cpu_connector = integration.register_cpu_agent(1).await?;
    let gpu_connector = integration.register_gpu_agent(0).await?;
    
    integration.start().await?;
    
    // Run workload
    for _ in 0..10 {
        cpu_connector.submit_compute_job(
            vec![1; 1024],
            TargetAgent::Gpu(0),
            JobPriority::Normal,
        ).await?;
    }
    
    sleep(Duration::from_secs(1)).await;
    
    // Check isolation metrics
    let isolation_report = integration.get_isolation_report()?;
    
    // CPU agents should have 0% GPU usage
    assert_eq!(isolation_report.cpu_gpu_violations, 0);
    assert_eq!(isolation_report.max_cpu_gpu_usage, 0.0);
    
    // GPU agents should have minimal I/O
    assert!(isolation_report.gpu_io_operations < 100); // Only shared storage ops
    assert!(isolation_report.gpu_io_percentage < 5.0);
    
    integration.shutdown().await?;
    
    Ok(())
}

// =============================================================================
// Mock Implementations for Testing
// =============================================================================

struct MockCpuAgent {
    id: usize,
}

impl MockCpuAgent {
    fn new(id: usize) -> Self {
        Self { id }
    }
    
    fn do_work(&self) -> anyhow::Result<()> {
        // Simulate CPU work without GPU usage
        std::thread::sleep(Duration::from_millis(10));
        Ok(())
    }
}

struct MockGpuAgent {
    id: usize,
}

impl MockGpuAgent {
    fn new(id: usize) -> Self {
        Self { id }
    }
    
    fn do_compute(&self) -> anyhow::Result<()> {
        // Simulate GPU compute without I/O
        std::thread::sleep(Duration::from_millis(10));
        Ok(())
    }
}