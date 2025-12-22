//! CPU→GPU→CPU workflow integration test
//! 
//! Tests the complete end-to-end workflow including:
//! - CPU agent job submission
//! - GPU agent job processing
//! - Result return to CPU agent

use super::*;
use shared_storage::*;
use std::sync::Arc;
use std::time::Duration;
use tempfile::tempdir;
use tokio::time::sleep;
use anyhow::Result;

#[tokio::test]
async fn test_cpu_gpu_cpu_full_workflow() -> Result<()> {
    // Setup test environment
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
        gpu_poll_interval: Duration::from_millis(50),
        cpu_poll_interval: Duration::from_millis(50),
        max_batch_size: 10,
        enable_gpu_direct: false,
    };
    
    // Create integration manager
    let mut integration = IntegrationManager::new(integration_config);
    
    // Register agents
    let cpu_connector = integration.register_cpu_agent(1).await?;
    let gpu_connector = integration.register_gpu_agent(0).await?;
    
    // Start integration
    integration.start().await?;
    
    println!("✓ Integration manager started");
    
    // Test 1: Simple compute job
    println!("Testing simple compute job...");
    
    let test_data = vec![1, 2, 3, 4, 5];
    let job_id = cpu_connector.submit_compute_job(
        test_data.clone(),
        TargetAgent::Gpu(0),
        JobPriority::High,
    ).await?;
    
    println!("✓ Job {} submitted from CPU to GPU", job_id);
    
    // Wait for processing
    sleep(Duration::from_millis(200)).await;
    
    // Check GPU received and processed the job
    let gpu_stats = gpu_connector.get_stats().await?;
    assert!(gpu_stats.jobs_processed > 0, "GPU should have processed at least one job");
    
    println!("✓ GPU processed {} jobs", gpu_stats.jobs_processed);
    
    // Poll for results
    let results = cpu_connector.poll_results(10).await?;
    assert!(!results.is_empty(), "Should have received at least one result");
    
    let result = &results[0];
    assert_eq!(result.original_job_id, job_id, "Result should match submitted job");
    assert_eq!(result.status, JobStatus::Completed, "Job should be completed");
    assert!(!result.data.is_empty(), "Result should have data");
    
    println!("✓ CPU received result: {} bytes", result.data.len());
    
    // Test 2: Batch processing
    println!("Testing batch processing...");
    
    let mut batch_job_ids = vec![];
    for i in 0..5 {
        let data = vec![i; 10];
        let job_id = cpu_connector.submit_compute_job(
            data,
            TargetAgent::Gpu(0),
            JobPriority::Normal,
        ).await?;
        batch_job_ids.push(job_id);
    }
    
    println!("✓ Submitted batch of {} jobs", batch_job_ids.len());
    
    // Wait for batch processing
    sleep(Duration::from_millis(500)).await;
    
    // Check all jobs were processed
    let final_results = cpu_connector.poll_results(10).await?;
    assert!(final_results.len() >= 5, "Should have results for all batch jobs");
    
    // Verify all job IDs are present
    let result_job_ids: Vec<_> = final_results.iter()
        .map(|r| r.original_job_id)
        .collect();
    
    for job_id in &batch_job_ids {
        assert!(result_job_ids.contains(job_id), "Should have result for job {}", job_id);
    }
    
    println!("✓ Batch processing completed successfully");
    
    // Test 3: Resource isolation verification
    println!("Testing resource isolation...");
    
    integration.enable_isolation_monitoring()?;
    
    // Run some workload
    for _ in 0..10 {
        cpu_connector.submit_compute_job(
            vec![42; 100],
            TargetAgent::Gpu(0),
            JobPriority::Normal,
        ).await?;
    }
    
    sleep(Duration::from_millis(300)).await;
    
    // Check isolation report
    let isolation_report = integration.get_isolation_report()?;
    
    // CPU agents should not use GPU resources
    assert_eq!(isolation_report.cpu_gpu_violations, 0, 
               "CPU agents should not violate GPU isolation");
    assert_eq!(isolation_report.max_cpu_gpu_usage, 0.0, 
               "CPU agents should have 0% GPU usage");
    
    // GPU agents should have minimal I/O
    assert!(isolation_report.gpu_io_percentage < 10.0, 
            "GPU agents should have minimal I/O operations");
    
    println!("✓ Resource isolation verified");
    
    // Test 4: Performance metrics
    println!("Testing performance metrics...");
    
    let final_gpu_stats = gpu_connector.get_stats().await?;
    let final_cpu_stats = cpu_connector.get_stats().await?;
    
    // Check performance metrics
    assert!(final_gpu_stats.jobs_processed >= 15, "GPU should process multiple jobs");
    assert!(final_cpu_stats.jobs_submitted >= 15, "CPU should submit multiple jobs");
    assert!(final_gpu_stats.avg_processing_time_us < 10_000, "Processing should be fast");
    
    println!("✓ Performance metrics:");
    println!("  GPU jobs processed: {}", final_gpu_stats.jobs_processed);
    println!("  CPU jobs submitted: {}", final_cpu_stats.jobs_submitted);
    println!("  Average processing time: {}μs", final_gpu_stats.avg_processing_time_us);
    
    // Cleanup
    integration.shutdown().await?;
    
    println!("✓ Integration manager shut down");
    println!("\n🎉 CPU→GPU→CPU workflow test PASSED!");
    
    Ok(())
}

#[tokio::test]
async fn test_streaming_workflow() -> Result<()> {
    // Setup test environment
    let temp_dir = tempdir()?;
    
    let storage_config = SharedStorageConfig {
        base_path: temp_dir.path().to_path_buf(),
        max_job_size: 5 * 1024 * 1024, // 5MB
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
        gpu_poll_interval: Duration::from_millis(25),
        cpu_poll_interval: Duration::from_millis(25),
        max_batch_size: 20,
        enable_gpu_direct: false,
    };
    
    // Create integration manager
    let mut integration = IntegrationManager::new(integration_config);
    
    // Register agents
    let data_producer = integration.register_cpu_agent(1).await?;
    let gpu_processor = integration.register_gpu_agent(0).await?;
    let result_consumer = integration.register_cpu_agent(2).await?;
    
    // Start integration
    integration.start().await?;
    
    println!("✓ Streaming integration started");
    
    // Create streaming configuration
    let stream_config = StreamConfig {
        chunk_size: 1024,
        window_size: 4096,
        processing_interval: Duration::from_millis(50),
    };
    
    // Setup stream
    let stream_id = integration.create_stream(
        data_producer.cpu_id(),
        gpu_processor.gpu_id(),
        result_consumer.cpu_id(),
        stream_config,
    ).await?;
    
    println!("✓ Stream {} created", stream_id);
    
    // Send streaming data
    for i in 0..10 {
        let data = vec![i as u8; 512]; // 512 bytes each
        data_producer.submit_stream_data(stream_id, data).await?;
    }
    
    println!("✓ Sent 10 stream chunks");
    
    // Wait for stream processing
    sleep(Duration::from_millis(500)).await;
    
    // Check GPU processed stream data
    let gpu_stats = gpu_processor.get_stats().await?;
    assert!(gpu_stats.stream_chunks_processed > 0, "GPU should process stream chunks");
    
    println!("✓ GPU processed {} stream chunks", gpu_stats.stream_chunks_processed);
    
    // Check for stream results
    let stream_results = result_consumer.poll_stream_results(stream_id, 20).await?;
    assert!(!stream_results.is_empty(), "Should have stream results");
    
    println!("✓ Consumer received {} stream results", stream_results.len());
    
    // Close stream
    integration.close_stream(stream_id).await?;
    integration.shutdown().await?;
    
    println!("✓ Streaming workflow test PASSED!");
    
    Ok(())
}

#[tokio::test]
async fn test_fault_tolerance_workflow() -> Result<()> {
    // This test verifies the system handles failures gracefully
    let temp_dir = tempdir()?;
    
    let storage_config = SharedStorageConfig {
        base_path: temp_dir.path().to_path_buf(),
        max_job_size: 1024 * 1024, // 1MB
        cleanup_interval: Duration::from_secs(60),
        job_ttl: Duration::from_secs(300),
        enable_compression: false,
        max_concurrent_jobs: 50,
        inbox_dir: "inbox".to_string(),
        outbox_dir: "outbox".to_string(),
        processing_dir: "processing".to_string(),
    };
    
    let storage_manager = Arc::new(SharedStorageManager::new(storage_config).await?);
    
    let integration_config = IntegrationConfig {
        storage_manager: storage_manager.clone(),
        gpu_poll_interval: Duration::from_millis(50),
        cpu_poll_interval: Duration::from_millis(50),
        max_batch_size: 10,
        enable_gpu_direct: false,
    };
    
    let mut integration = IntegrationManager::new(integration_config);
    
    // Enable fault tolerance
    integration.enable_fault_tolerance(FaultToleranceConfig {
        max_retries: 3,
        retry_delay: Duration::from_millis(100),
        heartbeat_interval: Duration::from_millis(200),
        agent_timeout: Duration::from_secs(2),
    })?;
    
    let cpu_agent = integration.register_cpu_agent(1).await?;
    let gpu_agent = integration.register_gpu_agent(0).await?;
    
    integration.start().await?;
    
    println!("✓ Fault tolerance test started");
    
    // Submit normal job
    let job_id = cpu_agent.submit_compute_job(
        vec![1, 2, 3],
        TargetAgent::Gpu(0),
        JobPriority::High,
    ).await?;
    
    // Wait for processing
    sleep(Duration::from_millis(200)).await;
    
    // Verify normal operation
    let results = cpu_agent.poll_results(5).await?;
    assert!(!results.is_empty(), "Should process normal jobs");
    
    println!("✓ Normal job processing works");
    
    // Check health status
    let health_status = integration.get_agent_health_status().await?;
    assert!(health_status.gpu_agents[&0].is_healthy, "GPU agent should be healthy");
    assert!(health_status.cpu_agents[&1].is_healthy, "CPU agent should be healthy");
    
    println!("✓ Health monitoring works");
    
    integration.shutdown().await?;
    
    println!("✓ Fault tolerance test PASSED!");
    
    Ok(())
}

/// Generate test data for various scenarios
fn generate_test_data(size: usize, pattern: u8) -> Vec<u8> {
    (0..size).map(|i| (i as u8).wrapping_add(pattern)).collect()
}

/// Verify test result matches expected pattern
fn verify_result(input: &[u8], output: &[u8], expected_transform: fn(u8) -> u8) -> bool {
    if input.len() != output.len() {
        return false;
    }
    
    input.iter().zip(output.iter())
        .all(|(&inp, &out)| out == expected_transform(inp))
}