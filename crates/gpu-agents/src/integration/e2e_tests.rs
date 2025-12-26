//! End-to-end tests for complete GPU/CPU agent workflows
//! 
//! Tests real-world scenarios including:
//! - Scientific computing workflows
//! - Machine learning pipelines
//! - Real-time data processing
//! - Multi-stage computations

use super::*;
use shared_storage::*;
use crate::{GpuAgent, GpuEvolutionEngine};
use cpu_agents::{CpuAgent, IoManager, Orchestrator};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::time::{Duration, Instant};
use tempfile::tempdir;
use tokio::time::sleep;

// =============================================================================
// E2E Test Scenarios
// =============================================================================

#[tokio::test]
async fn test_e2e_scientific_computing_workflow() -> anyhow::Result<()> {
    let env = create_test_environment().await?;
    let mut integration = IntegrationManager::new(env.integration_config);
    
    // Setup distributed computing cluster
    let coordinator = integration.register_cpu_agent(0).await?;
    let data_loader = integration.register_cpu_agent(1).await?;
    let result_aggregator = integration.register_cpu_agent(2).await?;
    
    let compute_gpus = vec![
        integration.register_gpu_agent(0).await?,
        integration.register_gpu_agent(1).await?,
        integration.register_gpu_agent(2).await?,
        integration.register_gpu_agent(3).await?,
    ];
    
    integration.start().await?;
    
    // Scientific workflow: Climate simulation
    println!("Starting climate simulation workflow...");
    
    // Step 1: Load climate data
    let climate_data = load_climate_test_data()?;
    let data_chunks = chunk_climate_data(&climate_data, compute_gpus.len());
    
    // Step 2: Distribute initial conditions to GPUs
    let mut init_jobs = vec![];
    for (i, chunk) in data_chunks.iter().enumerate() {
        let job_id = data_loader.submit_job(
            JobRequest {
                job_type: JobType::Custom(CLIMATE_INIT),
                target: TargetAgent::Gpu(i),
                data: bincode::serialize(&chunk)?,
                priority: JobPriority::High,
                metadata: [
                    ("simulation".to_string(), "climate".to_string()),
                    ("chunk_id".to_string(), i.to_string()),
                ].into_iter().collect(),
            }
        ).await?;
        init_jobs.push(job_id);
    }
    
    // Wait for initialization
    sleep(Duration::from_millis(500)).await;
    
    // Step 3: Run simulation timesteps
    let num_timesteps = 100;
    let mut timestep_results = vec![];
    
    for timestep in 0..num_timesteps {
        // Each GPU computes its region
        let mut timestep_jobs = vec![];
        
        for i in 0..compute_gpus.len() {
            let job_id = coordinator.submit_job(
                JobRequest {
                    job_type: JobType::Custom(CLIMATE_TIMESTEP),
                    target: TargetAgent::Gpu(i),
                    data: bincode::serialize(&timestep)?,
                    priority: JobPriority::Normal,
                    metadata: [
                        ("timestep".to_string(), timestep.to_string()),
                        ("gpu_id".to_string(), i.to_string()),
                    ].into_iter().collect(),
                }
            ).await?;
            timestep_jobs.push(job_id);
        }
        
        // Synchronization barrier - wait for all GPUs
        sleep(Duration::from_millis(100)).await;
        
        // Exchange boundary conditions between GPUs
        for i in 0..compute_gpus.len() {
            let left_neighbor = (i + compute_gpus.len() - 1) % compute_gpus.len();
            let right_neighbor = (i + 1) % compute_gpus.len();
            
            coordinator.submit_job(
                JobRequest {
                    job_type: JobType::Custom(BOUNDARY_EXCHANGE),
                    target: TargetAgent::Gpu(i),
                    data: bincode::serialize(&(left_neighbor, right_neighbor))?,
                    priority: JobPriority::High,
                    metadata: Default::default(),
                }
            ).await?;
        }
        
        if timestep % 10 == 0 {
            println!("Completed timestep {}/{}", timestep, num_timesteps);
        }
    }
    
    // Step 4: Collect and aggregate results
    println!("Collecting simulation results...");
    
    for i in 0..compute_gpus.len() {
        coordinator.submit_job(
            JobRequest {
                job_type: JobType::Custom(COLLECT_RESULTS),
                target: TargetAgent::Gpu(i),
                data: vec![],
                priority: JobPriority::High,
                metadata: [("target_cpu".to_string(), result_aggregator.cpu_id().to_string())].into_iter().collect(),
            }
        ).await?;
    }
    
    // Wait for results
    sleep(Duration::from_secs(1)).await;
    
    // Aggregate results
    let results = result_aggregator.poll_results(100).await?;
    assert_eq!(results.len(), compute_gpus.len());
    
    let mut final_state = ClimateState::default();
    for result in results {
        let partial_state: ClimateState = bincode::deserialize(&result.data)?;
        final_state.merge(partial_state);
    }
    
    // Verify simulation results
    assert!(final_state.temperature_field.len() > 0);
    assert!(final_state.average_temperature() > 0.0);
    assert!(final_state.total_energy() > 0.0);
    
    println!("Climate simulation completed successfully!");
    println!("Final average temperature: {:.2}°C", final_state.average_temperature());
    println!("Total energy: {:.2e} J", final_state.total_energy());
    
    integration.shutdown().await?;
    Ok(())
}

#[tokio::test]
async fn test_e2e_ml_training_pipeline() -> anyhow::Result<()> {
    let env = create_test_environment().await?;
    let mut integration = IntegrationManager::new(env.integration_config);
    
    // ML pipeline agents
    let data_preprocessor = integration.register_cpu_agent(1).await?;
    let trainer_coordinator = integration.register_cpu_agent(2).await?;
    let model_evaluator = integration.register_cpu_agent(3).await?;
    
    let training_gpus = vec![
        integration.register_gpu_agent(0).await?,
        integration.register_gpu_agent(1).await?,
    ];
    
    integration.start().await?;
    
    println!("Starting distributed ML training pipeline...");
    
    // Step 1: Load and preprocess data
    let dataset = create_ml_dataset(10000, 128); // 10k samples, 128 features
    let (train_data, val_data) = split_dataset(&dataset, 0.8);
    
    // Distribute data to GPUs
    let data_per_gpu = train_data.len() / training_gpus.len();
    
    for (i, gpu) in training_gpus.iter().enumerate() {
        let start = i * data_per_gpu;
        let end = if i == training_gpus.len() - 1 { train_data.len() } else { start + data_per_gpu };
        
        let gpu_data = &train_data[start..end];
        data_preprocessor.submit_job(
            JobRequest {
                job_type: JobType::Custom(ML_DATA_LOAD),
                target: TargetAgent::Gpu(i),
                data: bincode::serialize(&gpu_data)?,
                priority: JobPriority::High,
                metadata: [("dataset".to_string(), "training".to_string())].into_iter().collect(),
            }
        ).await?;
    }
    
    // Step 2: Initialize model on all GPUs
    let model_config = ModelConfig {
        architecture: "transformer".to_string(),
        layers: 12,
        hidden_dim: 768,
        num_heads: 12,
        learning_rate: 0.001,
    };
    
    for i in 0..training_gpus.len() {
        trainer_coordinator.submit_job(
            JobRequest {
                job_type: JobType::Custom(ML_MODEL_INIT),
                target: TargetAgent::Gpu(i),
                data: bincode::serialize(&model_config)?,
                priority: JobPriority::High,
                metadata: Default::default(),
            }
        ).await?;
    }
    
    sleep(Duration::from_millis(500)).await;
    
    // Step 3: Training loop
    let num_epochs = 10;
    let mut training_metrics = TrainingMetrics::default();
    
    for epoch in 0..num_epochs {
        println!("Training epoch {}/{}", epoch + 1, num_epochs);
        
        // Forward pass on each GPU
        let mut loss_jobs = vec![];
        for i in 0..training_gpus.len() {
            let job_id = trainer_coordinator.submit_job(
                JobRequest {
                    job_type: JobType::Custom(ML_FORWARD_PASS),
                    target: TargetAgent::Gpu(i),
                    data: bincode::serialize(&epoch)?,
                    priority: JobPriority::Normal,
                    metadata: Default::default(),
                }
            ).await?;
            loss_jobs.push(job_id);
        }
        
        // Collect losses
        sleep(Duration::from_millis(200)).await;
        let loss_results = trainer_coordinator.poll_results(training_gpus.len()).await?;
        
        let mut epoch_loss = 0.0;
        for result in loss_results {
            let gpu_loss: f32 = bincode::deserialize(&result.data)?;
            epoch_loss += gpu_loss;
        }
        epoch_loss /= training_gpus.len() as f32;
        training_metrics.train_losses.push(epoch_loss);
        
        // Gradient synchronization (all-reduce)
        trainer_coordinator.submit_job(
            JobRequest {
                job_type: JobType::Custom(ML_GRADIENT_SYNC),
                target: TargetAgent::AllGpus,
                data: vec![],
                priority: JobPriority::High,
                metadata: Default::default(),
            }
        ).await?;
        
        sleep(Duration::from_millis(100)).await;
        
        // Backward pass and optimization
        for i in 0..training_gpus.len() {
            trainer_coordinator.submit_job(
                JobRequest {
                    job_type: JobType::Custom(ML_BACKWARD_PASS),
                    target: TargetAgent::Gpu(i),
                    data: vec![],
                    priority: JobPriority::Normal,
                    metadata: Default::default(),
                }
            ).await?;
        }
        
        // Validation every 2 epochs
        if epoch % 2 == 0 {
            let val_job = model_evaluator.submit_job(
                JobRequest {
                    job_type: JobType::Custom(ML_VALIDATION),
                    target: TargetAgent::Gpu(0), // Use first GPU for validation
                    data: bincode::serialize(&val_data)?,
                    priority: JobPriority::Low,
                    metadata: Default::default(),
                }
            ).await?;
            
            sleep(Duration::from_millis(300)).await;
            
            let val_results = model_evaluator.poll_results(10).await?;
            if let Some(val_result) = val_results.iter().find(|r| r.original_job_id == val_job) {
                let val_metrics: ValidationMetrics = bincode::deserialize(&val_result.data)?;
                training_metrics.val_accuracies.push(val_metrics.accuracy);
                println!("Validation accuracy: {:.2}%", val_metrics.accuracy * 100.0);
            }
        }
        
        println!("Epoch {} loss: {:.4}", epoch + 1, epoch_loss);
    }
    
    // Step 4: Save final model
    let save_job = trainer_coordinator.submit_job(
        JobRequest {
            job_type: JobType::Custom(ML_MODEL_SAVE),
            target: TargetAgent::Gpu(0),
            data: vec![],
            priority: JobPriority::High,
            metadata: [("path".to_string(), "/tmp/model.bin".to_string())].into_iter().collect(),
        }
    ).await?;
    
    sleep(Duration::from_millis(500)).await;
    
    // Verify training completed successfully
    assert!(training_metrics.train_losses.len() == num_epochs);
    assert!(training_metrics.train_losses.last().unwrap() < training_metrics.train_losses.first().unwrap());
    assert!(!training_metrics.val_accuracies.is_empty());
    
    println!("ML training completed successfully!");
    println!("Final training loss: {:.4}", training_metrics.train_losses.last().unwrap());
    println!("Best validation accuracy: {:.2}%", 
             training_metrics.val_accuracies.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap() * 100.0);
    
    integration.shutdown().await?;
    Ok(())
}

#[tokio::test]
async fn test_e2e_realtime_video_processing() -> anyhow::Result<()> {
    let env = create_test_environment().await?;
    let mut integration = IntegrationManager::new(env.integration_config);
    
    // Video processing pipeline
    let video_ingester = integration.register_cpu_agent(1).await?;
    let frame_processor = integration.register_gpu_agent(0).await?;
    let encoder = integration.register_cpu_agent(2).await?;
    
    integration.start().await?;
    
    // Configure real-time constraints
    let rt_config = RealtimeConfig {
        target_fps: 30,
        max_latency_ms: 33, // ~30fps
        frame_buffer_size: 5,
    };
    
    integration.set_realtime_constraints(rt_config)?;
    
    println!("Starting real-time video processing pipeline...");
    
    let stop_flag = Arc::new(AtomicBool::new(false));
    let frames_processed = Arc::new(AtomicU64::new(0));
    let total_latency = Arc::new(AtomicU64::new(0));
    
    // Video ingestion thread
    let stop_ingestion = stop_flag.clone();
    let ingester_handle = tokio::spawn(async move {
        let mut frame_id = 0;
        
        while !stop_ingestion.load(Ordering::Relaxed) {
            // Simulate video frame (1920x1080 RGB)
            let frame = generate_video_frame(frame_id, 1920, 1080);
            let timestamp = Instant::now();
            
            video_ingester.submit_job(
                JobRequest {
                    job_type: JobType::StreamData,
                    target: TargetAgent::Gpu(0),
                    data: frame,
                    priority: JobPriority::Critical, // Real-time priority
                    metadata: [
                        ("frame_id".to_string(), frame_id.to_string()),
                        ("timestamp_us".to_string(), timestamp.elapsed().as_micros().to_string()),
                    ].into_iter().collect(),
                }
            ).await?;
            
            frame_id += 1;
            
            // Maintain 30 FPS input rate
            sleep(Duration::from_micros(33333)).await;
        }
        
        Ok::<usize, anyhow::Error>(frame_id)
    });
    
    // GPU processing monitoring
    let frames_clone = frames_processed.clone();
    let latency_clone = total_latency.clone();
    let stop_monitor = stop_flag.clone();
    
    let monitor_handle = tokio::spawn(async move {
        while !stop_monitor.load(Ordering::Relaxed) {
            let stats = frame_processor.get_stats().await?;
            frames_clone.store(stats.frames_processed, Ordering::Relaxed);
            
            sleep(Duration::from_millis(100)).await;
        }
        Ok::<(), anyhow::Error>(())
    });
    
    // Output encoding
    let stop_encoder = stop_flag.clone();
    let latency_encoder = total_latency.clone();
    
    let encoder_handle = tokio::spawn(async move {
        let mut encoded_frames = 0;
        
        while !stop_encoder.load(Ordering::Relaxed) {
            let results = encoder.poll_results(10).await?;
            
            for result in results {
                if let Some(timestamp_str) = result.metadata.get("timestamp_us") {
                    let start_time: u128 = timestamp_str.parse().unwrap_or(0);
                    let end_time = Instant::now().elapsed().as_micros();
                    let frame_latency = end_time.saturating_sub(start_time);
                    
                    latency_encoder.fetch_add(frame_latency as u64, Ordering::Relaxed);
                    encoded_frames += 1;
                    
                    // Check real-time constraint
                    if frame_latency > 33000 { // 33ms
                        println!("Warning: Frame {} exceeded latency constraint: {}μs", 
                                 result.metadata.get("frame_id").unwrap_or(&"?".to_string()),
                                 frame_latency);
                    }
                }
            }
            
            sleep(Duration::from_millis(10)).await;
        }
        
        Ok::<usize, anyhow::Error>(encoded_frames)
    });
    
    // Run for 5 seconds
    sleep(Duration::from_secs(5)).await;
    stop_flag.store(true, Ordering::Relaxed);
    
    // Wait for tasks to complete
    let total_frames = ingester_handle.await??;
    monitor_handle.await??;
    let encoded_count = encoder_handle.await??;
    
    // Calculate statistics
    let processed = frames_processed.load(Ordering::Relaxed);
    let avg_latency = if encoded_count > 0 {
        total_latency.load(Ordering::Relaxed) / encoded_count as u64
    } else {
        0
    };
    
    println!("Video processing statistics:");
    println!("  Total frames: {}", total_frames);
    println!("  Processed frames: {}", processed);
    println!("  Encoded frames: {}", encoded_count);
    println!("  Average latency: {}μs", avg_latency);
    println!("  Achieved FPS: {:.1}", encoded_count as f64 / 5.0);
    
    // Verify real-time performance
    assert!(processed as f64 / total_frames as f64 > 0.95); // >95% frames processed
    assert!(avg_latency < 40000); // <40ms average latency
    
    integration.shutdown().await?;
    Ok(())
}

// =============================================================================
// Helper Types and Functions
// =============================================================================

#[derive(Default, serde::Serialize, serde::Deserialize)]
struct ClimateState {
    temperature_field: Vec<Vec<f64>>,
    pressure_field: Vec<Vec<f64>>,
    wind_field: Vec<Vec<(f64, f64)>>,
}

impl ClimateState {
    fn merge(&mut self, other: ClimateState) {
        self.temperature_field.extend(other.temperature_field);
        self.pressure_field.extend(other.pressure_field);
        self.wind_field.extend(other.wind_field);
    }
    
    fn average_temperature(&self) -> f64 {
        let sum: f64 = self.temperature_field.iter()
            .flat_map(|row| row.iter())
            .sum();
        let count = self.temperature_field.iter()
            .map(|row| row.len())
            .sum::<usize>();
        
        if count > 0 { sum / count as f64 } else { 0.0 }
    }
    
    fn total_energy(&self) -> f64 {
        // Simplified energy calculation
        self.temperature_field.iter()
            .flat_map(|row| row.iter())
            .map(|&t| t * t)
            .sum::<f64>()
            * 1e6 // Arbitrary scaling
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
struct ModelConfig {
    architecture: String,
    layers: usize,
    hidden_dim: usize,
    num_heads: usize,
    learning_rate: f32,
}

#[derive(Default)]
struct TrainingMetrics {
    train_losses: Vec<f32>,
    val_accuracies: Vec<f32>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct ValidationMetrics {
    accuracy: f32,
    precision: f32,
    recall: f32,
}

fn load_climate_test_data() -> anyhow::Result<ClimateState> {
    let grid_size = 100;
    let mut state = ClimateState::default();
    
    // Generate synthetic climate data
    for i in 0..grid_size {
        let mut temp_row = vec![];
        let mut pressure_row = vec![];
        let mut wind_row = vec![];
        
        for j in 0..grid_size {
            let temp = 15.0 + 10.0 * ((i as f64 / grid_size as f64).sin() * (j as f64 / grid_size as f64).cos());
            let pressure = 1013.25 + 50.0 * ((i + j) as f64 / grid_size as f64).sin();
            let wind_u = 5.0 * (i as f64 / grid_size as f64).cos();
            let wind_v = 5.0 * (j as f64 / grid_size as f64).sin();
            
            temp_row.push(temp);
            pressure_row.push(pressure);
            wind_row.push((wind_u, wind_v));
        }
        
        state.temperature_field.push(temp_row);
        state.pressure_field.push(pressure_row);
        state.wind_field.push(wind_row);
    }
    
    Ok(state)
}

fn chunk_climate_data(state: &ClimateState, num_chunks: usize) -> Vec<ClimateState> {
    let rows_per_chunk = state.temperature_field.len() / num_chunks;
    let mut chunks = vec![];
    
    for i in 0..num_chunks {
        let start = i * rows_per_chunk;
        let end = if i == num_chunks - 1 { state.temperature_field.len() } else { start + rows_per_chunk };
        
        let chunk = ClimateState {
            temperature_field: state.temperature_field[start..end].to_vec(),
            pressure_field: state.pressure_field[start..end].to_vec(),
            wind_field: state.wind_field[start..end].to_vec(),
        };
        
        chunks.push(chunk);
    }
    
    chunks
}

fn create_ml_dataset(num_samples: usize, num_features: usize) -> Vec<(Vec<f32>, f32)> {
    (0..num_samples)
        .map(|i| {
            let features: Vec<f32> = (0..num_features)
                .map(|j| ((i + j) as f32 / num_samples as f32).sin())
                .collect();
            
            // Simple synthetic label
            let label = if features.iter().sum::<f32>() > 0.0 { 1.0 } else { 0.0 };
            
            (features, label)
        })
        .collect()
}

fn split_dataset(dataset: &[(Vec<f32>, f32)], train_ratio: f32) -> (Vec<(Vec<f32>, f32)>, Vec<(Vec<f32>, f32)>) {
    let split_idx = (dataset.len() as f32 * train_ratio) as usize;
    (dataset[..split_idx].to_vec(), dataset[split_idx..].to_vec())
}

fn generate_video_frame(frame_id: usize, width: usize, height: usize) -> Vec<u8> {
    let mut frame = Vec::with_capacity(width * height * 3); // RGB
    
    for y in 0..height {
        for x in 0..width {
            // Generate pattern based on frame_id
            let r = ((x + frame_id) % 256) as u8;
            let g = ((y + frame_id) % 256) as u8;
            let b = ((x * y + frame_id) % 256) as u8;
            
            frame.push(r);
            frame.push(g);
            frame.push(b);
        }
    }
    
    frame
}

async fn create_test_environment() -> anyhow::Result<TestEnvironment> {
    let temp_dir = tempdir()?;
    
    let storage_config = SharedStorageConfig {
        base_path: temp_dir.path().to_path_buf(),
        max_job_size: 100 * 1024 * 1024, // 100MB
        cleanup_interval: Duration::from_secs(60),
        job_ttl: Duration::from_secs(300),
        enable_compression: false,
        max_concurrent_jobs: 10000,
        inbox_dir: "inbox".to_string(),
        outbox_dir: "outbox".to_string(),
        processing_dir: "processing".to_string(),
    };
    
    let storage_manager = Arc::new(SharedStorageManager::new(storage_config).await?);
    
    let integration_config = IntegrationConfig {
        storage_manager: storage_manager.clone(),
        gpu_poll_interval: Duration::from_millis(50),
        cpu_poll_interval: Duration::from_millis(100),
        max_batch_size: 64,
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

// Custom job type constants
const CLIMATE_INIT: u32 = 2001;
const CLIMATE_TIMESTEP: u32 = 2002;
const BOUNDARY_EXCHANGE: u32 = 2003;
const COLLECT_RESULTS: u32 = 2004;

const ML_DATA_LOAD: u32 = 3001;
const ML_MODEL_INIT: u32 = 3002;
const ML_FORWARD_PASS: u32 = 3003;
const ML_GRADIENT_SYNC: u32 = 3004;
const ML_BACKWARD_PASS: u32 = 3005;
const ML_VALIDATION: u32 = 3006;
const ML_MODEL_SAVE: u32 = 3007;