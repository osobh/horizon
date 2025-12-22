//! Integration tests for GPU/CPU agent communication
//! 
//! Tests end-to-end scenarios including:
//! - Multi-agent coordination
//! - Large-scale data processing
//! - Fault tolerance
//! - Performance under load

use super::*;
use shared_storage::*;
use crate::{GpuAgent, GpuEvolutionEngine};
use cpu_agents::{CpuAgent, IoManager, Orchestrator};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use tempfile::tempdir;
use tokio::time::sleep;

// =============================================================================
// Integration Test Scenarios
// =============================================================================

#[tokio::test]
async fn test_evolution_computation_pipeline() -> anyhow::Result<()> {
    let env = create_test_environment().await?;
    let mut integration = IntegrationManager::new(env.integration_config);
    
    // Register multiple CPU and GPU agents
    let cpu_agents = vec![
        integration.register_cpu_agent(1).await?,
        integration.register_cpu_agent(2).await?,
    ];
    
    let gpu_agents = vec![
        integration.register_gpu_agent(0).await?,
        integration.register_gpu_agent(1).await?,
    ];
    
    integration.start().await?;
    
    // CPU agents prepare evolution data
    for (i, cpu) in cpu_agents.iter().enumerate() {
        let population_data = create_test_population(100, 64); // 100 individuals, 64 genes
        let job_id = cpu.submit_evolution_job(
            EvolutionTask::Optimize {
                population: population_data,
                fitness_function: FitnessFunction::Rastrigin,
                generations: 50,
            },
            TargetAgent::Gpu(i),
            JobPriority::High,
        ).await?;
        
        println!("CPU {} submitted evolution job {}", i + 1, job_id);
    }
    
    // Wait for GPU processing
    sleep(Duration::from_secs(3)).await;
    
    // Check GPU stats
    for (i, gpu) in gpu_agents.iter().enumerate() {
        let stats = gpu.get_stats().await?;
        assert!(stats.jobs_processed > 0);
        assert!(stats.evolution_generations_computed > 0);
        println!("GPU {} processed {} evolution generations", i, stats.evolution_generations_computed);
    }
    
    // Collect results
    for (i, cpu) in cpu_agents.iter().enumerate() {
        let results = cpu.poll_results(10).await?;
        assert!(!results.is_empty());
        
        for result in results {
            assert_eq!(result.status, JobStatus::Completed);
            let evolution_result: EvolutionResult = bincode::deserialize(&result.data)?;
            assert!(evolution_result.best_fitness > 0.0);
            println!("CPU {} received result with best fitness: {}", i + 1, evolution_result.best_fitness);
        }
    }
    
    integration.shutdown().await?;
    Ok(())
}

#[tokio::test]
async fn test_knowledge_graph_distributed_query() -> anyhow::Result<()> {
    let env = create_test_environment().await?;
    let mut integration = IntegrationManager::new(env.integration_config);
    
    // Setup agents
    let cpu_coordinator = integration.register_cpu_agent(0).await?;
    let gpu_workers = vec![
        integration.register_gpu_agent(0).await?,
        integration.register_gpu_agent(1).await?,
        integration.register_gpu_agent(2).await?,
    ];
    
    integration.start().await?;
    
    // Build distributed knowledge graph
    let graph_chunks = create_test_knowledge_graph(10000); // 10k nodes
    
    // Distribute graph building to GPUs
    for (i, chunk) in graph_chunks.iter().enumerate() {
        let job_id = cpu_coordinator.submit_job(
            JobRequest {
                job_type: JobType::Custom(KNOWLEDGE_GRAPH_BUILD),
                target: TargetAgent::Gpu(i),
                data: bincode::serialize(chunk)?,
                priority: JobPriority::Normal,
                metadata: [("chunk_id".to_string(), i.to_string())].into_iter().collect(),
            }
        ).await?;
        
        println!("Submitted graph chunk {} as job {}", i, job_id);
    }
    
    // Wait for graph construction
    sleep(Duration::from_secs(2)).await;
    
    // Submit complex query
    let query = KnowledgeQuery::MultiHop {
        start_node: "entity_0".to_string(),
        target_node: "entity_9999".to_string(),
        max_hops: 5,
        constraints: vec![QueryConstraint::MinConfidence(0.7)],
    };
    
    let query_job = cpu_coordinator.submit_job(
        JobRequest {
            job_type: JobType::Custom(KNOWLEDGE_GRAPH_QUERY),
            target: TargetAgent::Gpu(0), // Primary GPU coordinates
            data: bincode::serialize(&query)?,
            priority: JobPriority::Critical,
            metadata: Default::default(),
        }
    ).await?;
    
    // Wait for query result
    sleep(Duration::from_millis(500)).await;
    
    let results = cpu_coordinator.poll_results(10).await?;
    let query_result = results.iter().find(|r| r.original_job_id == query_job);
    assert!(query_result.is_some());
    
    let result = query_result.unwrap();
    let paths: Vec<KnowledgePath> = bincode::deserialize(&result.data)?;
    assert!(!paths.is_empty());
    println!("Found {} paths in distributed knowledge graph", paths.len());
    
    integration.shutdown().await?;
    Ok(())
}

#[tokio::test]
async fn test_streaming_data_processing() -> anyhow::Result<()> {
    let env = create_test_environment().await?;
    let mut integration = IntegrationManager::new(env.integration_config);
    
    // Setup streaming pipeline
    let data_producer = integration.register_cpu_agent(1).await?;
    let gpu_processor = integration.register_gpu_agent(0).await?;
    let result_consumer = integration.register_cpu_agent(2).await?;
    
    integration.start().await?;
    
    // Configure streaming
    let stream_config = StreamConfig {
        chunk_size: 1024 * 100, // 100KB chunks
        window_size: 1024 * 500, // 500KB window
        processing_interval: Duration::from_millis(100),
    };
    
    let stream_id = integration.create_stream(
        data_producer.cpu_id(),
        gpu_processor.gpu_id(),
        result_consumer.cpu_id(),
        stream_config,
    ).await?;
    
    // Start producing data
    let producer_handle = tokio::spawn(async move {
        for i in 0..100 {
            let data = generate_sensor_data(i, 1024 * 50); // 50KB of sensor data
            data_producer.submit_stream_data(stream_id, data).await?;
            sleep(Duration::from_millis(50)).await;
        }
        Ok::<(), anyhow::Error>(())
    });
    
    // Monitor GPU processing
    let processed_chunks = Arc::new(AtomicU64::new(0));
    let processed_clone = processed_chunks.clone();
    
    let monitor_handle = tokio::spawn(async move {
        for _ in 0..20 {
            let stats = gpu_processor.get_stats().await?;
            processed_clone.store(stats.stream_chunks_processed, Ordering::Relaxed);
            sleep(Duration::from_millis(250)).await;
        }
        Ok::<(), anyhow::Error>(())
    });
    
    // Consume results
    let mut total_results = 0;
    let start = Instant::now();
    
    while start.elapsed() < Duration::from_secs(10) {
        let results = result_consumer.poll_stream_results(stream_id, 10).await?;
        total_results += results.len();
        
        for result in results {
            // Verify processed data
            assert!(result.data.len() > 0);
            assert_eq!(result.stream_id, stream_id);
        }
        
        sleep(Duration::from_millis(100)).await;
    }
    
    // Wait for tasks
    producer_handle.await??;
    monitor_handle.await??;
    
    // Verify streaming worked
    let chunks_processed = processed_chunks.load(Ordering::Relaxed);
    assert!(chunks_processed > 50);
    assert!(total_results > 50);
    
    println!("Streamed {} chunks, received {} results", chunks_processed, total_results);
    
    integration.close_stream(stream_id).await?;
    integration.shutdown().await?;
    
    Ok(())
}

#[tokio::test]
async fn test_fault_tolerance_and_recovery() -> anyhow::Result<()> {
    let env = create_test_environment().await?;
    let mut integration = IntegrationManager::new(env.integration_config);
    
    integration.enable_fault_tolerance(FaultToleranceConfig {
        max_retries: 3,
        retry_delay: Duration::from_millis(100),
        heartbeat_interval: Duration::from_millis(500),
        agent_timeout: Duration::from_secs(2),
    })?;
    
    let cpu_agent = integration.register_cpu_agent(1).await?;
    let gpu_agent = integration.register_gpu_agent(0).await?;
    
    integration.start().await?;
    
    // Submit job that will fail initially
    let failing_job = FailingJob {
        fail_count: 2, // Fail first 2 attempts
        data: vec![1, 2, 3, 4, 5],
    };
    
    let job_id = cpu_agent.submit_job(
        JobRequest {
            job_type: JobType::Custom(FAILING_JOB_TYPE),
            target: TargetAgent::Gpu(0),
            data: bincode::serialize(&failing_job)?,
            priority: JobPriority::High,
            metadata: Default::default(),
        }
    ).await?;
    
    // Wait for retries and eventual success
    sleep(Duration::from_secs(2)).await;
    
    // Check job succeeded after retries
    let results = cpu_agent.poll_results(10).await?;
    let job_result = results.iter().find(|r| r.original_job_id == job_id);
    assert!(job_result.is_some());
    
    let result = job_result.unwrap();
    assert_eq!(result.status, JobStatus::Completed);
    assert_eq!(result.retry_count, 2);
    
    // Simulate GPU agent crash
    gpu_agent.simulate_crash()?;
    
    // Submit new job
    let new_job_id = cpu_agent.submit_compute_job(
        vec![6, 7, 8],
        TargetAgent::Gpu(0),
        JobPriority::Normal,
    ).await?;
    
    // Wait for agent recovery
    sleep(Duration::from_secs(3)).await;
    
    // Check agent recovered and processed job
    let health_status = integration.get_agent_health_status().await?;
    assert!(health_status.gpu_agents[&0].is_healthy);
    assert!(health_status.gpu_agents[&0].recovered_from_crash);
    
    let new_results = cpu_agent.poll_results(10).await?;
    assert!(new_results.iter().any(|r| r.original_job_id == new_job_id));
    
    integration.shutdown().await?;
    Ok(())
}

#[tokio::test]
async fn test_load_balancing_and_scaling() -> anyhow::Result<()> {
    let env = create_test_environment().await?;
    let mut integration = IntegrationManager::new(env.integration_config);
    
    integration.enable_load_balancing(LoadBalancingConfig {
        strategy: LoadBalancingStrategy::RoundRobin,
        rebalance_interval: Duration::from_secs(1),
        load_threshold: 0.8,
    })?;
    
    // Register multiple agents
    let cpu_agents: Vec<_> = futures::future::try_join_all(
        (0..3).map(|i| integration.register_cpu_agent(i))
    ).await?;
    
    let gpu_agents: Vec<_> = futures::future::try_join_all(
        (0..4).map(|i| integration.register_gpu_agent(i))
    ).await?;
    
    integration.start().await?;
    
    // Submit many jobs
    let total_jobs = 100;
    let mut job_ids = vec![];
    
    for i in 0..total_jobs {
        let cpu_idx = i % cpu_agents.len();
        let data = vec![i as u8; 1024];
        
        let job_id = cpu_agents[cpu_idx].submit_compute_job(
            data,
            TargetAgent::Auto, // Let load balancer decide
            JobPriority::Normal,
        ).await?;
        
        job_ids.push(job_id);
    }
    
    // Wait for processing
    sleep(Duration::from_secs(3)).await;
    
    // Check load distribution
    let load_stats = integration.get_load_statistics().await?;
    
    // Verify jobs were distributed across GPUs
    for i in 0..4 {
        let gpu_load = &load_stats.gpu_loads[&i];
        assert!(gpu_load.jobs_processed > 10); // Each should process some jobs
        assert!(gpu_load.jobs_processed < 40); // But not all jobs
        println!("GPU {} processed {} jobs", i, gpu_load.jobs_processed);
    }
    
    // Verify load balancing worked
    let load_variance = calculate_load_variance(&load_stats.gpu_loads);
    assert!(load_variance < 0.2); // Low variance indicates good balance
    
    integration.shutdown().await?;
    Ok(())
}

// =============================================================================
// Helper Functions
// =============================================================================

fn create_test_population(size: usize, genome_length: usize) -> Vec<Vec<f32>> {
    (0..size)
        .map(|i| {
            (0..genome_length)
                .map(|j| ((i + j) as f32 / size as f32).sin())
                .collect()
        })
        .collect()
}

fn create_test_knowledge_graph(num_nodes: usize) -> Vec<GraphChunk> {
    let nodes_per_chunk = num_nodes / 3;
    let mut chunks = vec![];
    
    for chunk_id in 0..3 {
        let start = chunk_id * nodes_per_chunk;
        let end = if chunk_id == 2 { num_nodes } else { start + nodes_per_chunk };
        
        let nodes: Vec<_> = (start..end)
            .map(|i| KnowledgeNode {
                id: format!("entity_{}", i),
                node_type: if i % 3 == 0 { "Person" } else if i % 3 == 1 { "Place" } else { "Thing" },
                properties: Default::default(),
            })
            .collect();
        
        let edges: Vec<_> = (start..end-1)
            .map(|i| KnowledgeEdge {
                source: format!("entity_{}", i),
                target: format!("entity_{}", i + 1),
                relationship: "connected_to",
                weight: 0.8,
            })
            .collect();
        
        chunks.push(GraphChunk { nodes, edges });
    }
    
    chunks
}

fn generate_sensor_data(batch_id: usize, size: usize) -> Vec<u8> {
    (0..size)
        .map(|i| ((batch_id * size + i) as f32 * 0.1).sin().abs() as u8)
        .collect()
}

fn calculate_load_variance(gpu_loads: &std::collections::HashMap<usize, GpuLoadInfo>) -> f64 {
    let loads: Vec<f64> = gpu_loads.values().map(|l| l.jobs_processed as f64).collect();
    let mean = loads.iter().sum::<f64>() / loads.len() as f64;
    let variance = loads.iter().map(|l| (l - mean).powi(2)).sum::<f64>() / loads.len() as f64;
    variance.sqrt() / mean // Coefficient of variation
}

async fn create_test_environment() -> anyhow::Result<TestEnvironment> {
    let temp_dir = tempdir()?;
    
    let storage_config = SharedStorageConfig {
        base_path: temp_dir.path().to_path_buf(),
        max_job_size: 50 * 1024 * 1024, // 50MB
        cleanup_interval: Duration::from_secs(60),
        job_ttl: Duration::from_secs(300),
        enable_compression: false,
        max_concurrent_jobs: 1000,
        inbox_dir: "inbox".to_string(),
        outbox_dir: "outbox".to_string(),
        processing_dir: "processing".to_string(),
    };
    
    let storage_manager = Arc::new(SharedStorageManager::new(storage_config).await?);
    
    let integration_config = IntegrationConfig {
        storage_manager: storage_manager.clone(),
        gpu_poll_interval: Duration::from_millis(100),
        cpu_poll_interval: Duration::from_millis(100),
        max_batch_size: 32,
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

// Test-specific constants
const KNOWLEDGE_GRAPH_BUILD: u32 = 1001;
const KNOWLEDGE_GRAPH_QUERY: u32 = 1002;
const FAILING_JOB_TYPE: u32 = 1003;

#[derive(serde::Serialize, serde::Deserialize)]
struct FailingJob {
    fail_count: u32,
    data: Vec<u8>,
}