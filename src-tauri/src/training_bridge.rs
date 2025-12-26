//! Training Bridge
//!
//! Integrates rustytorch training with Horizon using hpc-channels.
//!
//! ## Feature Flags
//!
//! - `embedded-training`: When enabled, uses the real `rtx-distributed` crate for
//!   distributed training. This requires CUDA/NCCL and is only available on systems
//!   with NVIDIA GPUs. When disabled, provides mock training data for development.
//!
//! ## Architecture
//!
//! The training bridge wraps either:
//! - Real `MultiGpuTrainer` from rtx-distributed (with embedded-training)
//! - Mock training jobs with simulated progress (without embedded-training)
//!
//! Both implementations expose the same API to the Tauri commands.

use dashmap::DashMap;
use hpc_channels::{channels, TrainingMessage, TrainingConfig as ChannelTrainingConfig, TrainingStatus as ChannelTrainingStatus};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::RwLock;

// Import real rtx-distributed types when feature is enabled
#[cfg(feature = "embedded-training")]
use rtx_distributed::{
    MultiGpuTrainer, TrainingMetrics as RtxTrainingMetrics,
};

/// Bridge to the rustytorch training system.
pub struct TrainingBridge {
    /// Active training jobs (lock-free concurrent access)
    jobs: Arc<DashMap<String, TrainingJob>>,
    /// Job counter for generating IDs
    job_counter: AtomicU64,
    /// Real multi-GPU trainer (when embedded-training feature is enabled)
    #[cfg(feature = "embedded-training")]
    trainer: Arc<RwLock<Option<MultiGpuTrainer>>>,
}

/// A training job.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingJob {
    pub id: String,
    pub name: String,
    pub status: TrainingStatus,
    pub config: TrainingConfig,
    pub progress: TrainingProgress,
    pub metrics: TrainingMetrics,
    pub created_at: u64,
    pub started_at: Option<u64>,
    pub completed_at: Option<u64>,
}

/// Training job status.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum TrainingStatus {
    Queued,
    Running,
    Paused,
    Completed,
    Failed,
    Cancelled,
}

impl From<TrainingStatus> for ChannelTrainingStatus {
    fn from(s: TrainingStatus) -> Self {
        match s {
            TrainingStatus::Queued => ChannelTrainingStatus::Queued,
            TrainingStatus::Running => ChannelTrainingStatus::Running,
            TrainingStatus::Paused => ChannelTrainingStatus::Paused,
            TrainingStatus::Completed => ChannelTrainingStatus::Completed,
            TrainingStatus::Failed => ChannelTrainingStatus::Failed,
            TrainingStatus::Cancelled => ChannelTrainingStatus::Cancelled,
        }
    }
}

/// Training configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub model: String,
    pub dataset: String,
    pub epochs: u32,
    pub batch_size: u32,
    pub learning_rate: f64,
    pub optimizer: String,
    pub distributed: Option<DistributedConfig>,
    pub hyperparameters: HashMap<String, String>,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            model: "transformer".to_string(),
            dataset: "custom".to_string(),
            epochs: 10,
            batch_size: 32,
            learning_rate: 1e-4,
            optimizer: "AdamW".to_string(),
            distributed: None,
            hyperparameters: HashMap::new(),
        }
    }
}

/// Distributed training configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedConfig {
    pub world_size: u32,
    pub strategy: String, // "data_parallel", "model_parallel", "pipeline_parallel"
}

/// Training progress.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TrainingProgress {
    pub current_epoch: u32,
    pub total_epochs: u32,
    pub current_step: u64,
    pub total_steps: u64,
    pub samples_processed: u64,
    pub completion_percentage: f64,
}

/// Training metrics.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TrainingMetrics {
    pub loss: f64,
    pub accuracy: Option<f64>,
    pub learning_rate: f64,
    pub throughput_samples_per_sec: f64,
    pub gpu_utilization: f64,
    pub memory_usage_gb: f64,
    pub epoch_losses: Vec<f64>,
    pub validation_losses: Vec<f64>,
    pub custom_metrics: HashMap<String, f64>,
}

/// Summary of all training jobs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingSummary {
    pub total_jobs: usize,
    pub running_jobs: usize,
    pub completed_jobs: usize,
    pub failed_jobs: usize,
    pub queued_jobs: usize,
}

impl TrainingBridge {
    /// Create a new training bridge (with embedded-training feature).
    #[cfg(feature = "embedded-training")]
    pub fn new() -> Self {
        tracing::info!("TrainingBridge initialized with rtx-distributed (CUDA required)");
        let bridge = Self {
            jobs: Arc::new(DashMap::new()),
            job_counter: AtomicU64::new(0),
            trainer: Arc::new(RwLock::new(None)),
        };

        // Add mock jobs for demo (same as mock implementation)
        Self::add_demo_jobs(bridge.jobs.clone());
        bridge
    }

    /// Create a new training bridge (mock implementation).
    #[cfg(not(feature = "embedded-training"))]
    pub fn new() -> Self {
        tracing::info!("TrainingBridge initialized with mock data (embedded-training feature disabled)");
        let bridge = Self {
            jobs: Arc::new(DashMap::new()),
            job_counter: AtomicU64::new(0),
        };

        // Add mock jobs for demo
        Self::add_demo_jobs(bridge.jobs.clone());
        bridge
    }

    /// Add demo jobs for development/demo purposes.
    fn add_demo_jobs(jobs: Arc<DashMap<String, TrainingJob>>) {
        tokio::spawn(async move {
            // Add a running job
            jobs.insert("job-001".to_string(), TrainingJob {
                id: "job-001".to_string(),
                name: "LLaMA-7B Fine-tune".to_string(),
                status: TrainingStatus::Running,
                config: TrainingConfig {
                    model: "llama-7b".to_string(),
                    dataset: "alpaca-52k".to_string(),
                    epochs: 3,
                    batch_size: 8,
                    learning_rate: 2e-5,
                    optimizer: "AdamW".to_string(),
                    distributed: Some(DistributedConfig {
                        world_size: 8,
                        strategy: "data_parallel".to_string(),
                    }),
                    hyperparameters: HashMap::new(),
                },
                progress: TrainingProgress {
                    current_epoch: 2,
                    total_epochs: 3,
                    current_step: 4500,
                    total_steps: 6000,
                    samples_processed: 36000,
                    completion_percentage: 75.0,
                },
                metrics: TrainingMetrics {
                    loss: 0.342,
                    accuracy: Some(0.89),
                    learning_rate: 1.8e-5,
                    throughput_samples_per_sec: 245.0,
                    gpu_utilization: 0.92,
                    memory_usage_gb: 72.5,
                    epoch_losses: vec![1.24, 0.58, 0.342],
                    validation_losses: vec![1.15, 0.52, 0.38],
                    custom_metrics: HashMap::new(),
                },
                created_at: 1703100000,
                started_at: Some(1703100100),
                completed_at: None,
            });

            // Add a completed job
            jobs.insert("job-002".to_string(), TrainingJob {
                id: "job-002".to_string(),
                name: "Vision Transformer".to_string(),
                status: TrainingStatus::Completed,
                config: TrainingConfig {
                    model: "vit-base".to_string(),
                    dataset: "imagenet-1k".to_string(),
                    epochs: 90,
                    batch_size: 256,
                    learning_rate: 1e-3,
                    optimizer: "AdamW".to_string(),
                    distributed: Some(DistributedConfig {
                        world_size: 8,
                        strategy: "data_parallel".to_string(),
                    }),
                    hyperparameters: HashMap::new(),
                },
                progress: TrainingProgress {
                    current_epoch: 90,
                    total_epochs: 90,
                    current_step: 450000,
                    total_steps: 450000,
                    samples_processed: 115200000,
                    completion_percentage: 100.0,
                },
                metrics: TrainingMetrics {
                    loss: 0.125,
                    accuracy: Some(0.812),
                    learning_rate: 0.0,
                    throughput_samples_per_sec: 3200.0,
                    gpu_utilization: 0.0,
                    memory_usage_gb: 0.0,
                    epoch_losses: vec![2.5, 1.8, 1.2, 0.8, 0.5, 0.3, 0.2, 0.15, 0.125],
                    validation_losses: vec![2.3, 1.6, 1.1, 0.75, 0.45, 0.28, 0.18, 0.14, 0.12],
                    custom_metrics: HashMap::new(),
                },
                created_at: 1702900000,
                started_at: Some(1702900100),
                completed_at: Some(1703000000),
            });
        });
    }

    /// Initialize the real multi-GPU trainer (embedded-training feature only).
    #[cfg(feature = "embedded-training")]
    pub async fn init_trainer(&self, world_size: usize, local_rank: usize) -> Result<(), String> {
        tracing::info!("Initializing MultiGpuTrainer with world_size={}, local_rank={}", world_size, local_rank);

        match MultiGpuTrainer::new(world_size, local_rank).await {
            Ok(trainer) => {
                let mut guard = self.trainer.write().await;
                *guard = Some(trainer);
                tracing::info!("MultiGpuTrainer initialized successfully");
                Ok(())
            }
            Err(e) => {
                tracing::error!("Failed to initialize MultiGpuTrainer: {}", e);
                Err(format!("Failed to initialize trainer: {}", e))
            }
        }
    }

    /// Get real-time metrics from the trainer (embedded-training feature only).
    #[cfg(feature = "embedded-training")]
    pub async fn get_real_metrics(&self) -> Option<RtxTrainingMetrics> {
        let guard = self.trainer.read().await;
        if let Some(ref trainer) = *guard {
            Some(trainer.get_metrics().await)
        } else {
            None
        }
    }

    /// Start a new training job.
    pub async fn start_training(&self, name: String, config: TrainingConfig) -> Result<TrainingJob, String> {
        // Relaxed: independent job ID counter
        let counter = self.job_counter.fetch_add(1, Ordering::Relaxed) + 1;
        let job_id = format!("job-{:03}", counter);

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let job = TrainingJob {
            id: job_id.clone(),
            name,
            status: TrainingStatus::Running,
            config: config.clone(),
            progress: TrainingProgress {
                current_epoch: 0,
                total_epochs: config.epochs,
                current_step: 0,
                total_steps: (config.epochs as u64) * 1000, // Estimate
                samples_processed: 0,
                completion_percentage: 0.0,
            },
            metrics: TrainingMetrics {
                loss: 0.0,
                accuracy: None,
                learning_rate: config.learning_rate,
                throughput_samples_per_sec: 0.0,
                gpu_utilization: 0.0,
                memory_usage_gb: 0.0,
                epoch_losses: vec![],
                validation_losses: vec![],
                custom_metrics: HashMap::new(),
            },
            created_at: now,
            started_at: Some(now),
            completed_at: None,
        };

        self.jobs.insert(job_id.clone(), job.clone());

        // Broadcast training start via channel
        if let Some(tx) = hpc_channels::sender::<TrainingMessage>(channels::TRAINING_START) {
            let channel_config = ChannelTrainingConfig {
                model: config.model,
                dataset: config.dataset,
                epochs: config.epochs,
                batch_size: config.batch_size,
                learning_rate: config.learning_rate,
                optimizer: config.optimizer,
                hyperparameters: config.hyperparameters,
                distributed: config.distributed.map(|d| hpc_channels::DistributedConfig {
                    world_size: d.world_size,
                    strategy: match d.strategy.as_str() {
                        "data_parallel" => hpc_channels::ParallelismStrategy::DataParallel,
                        "model_parallel" => hpc_channels::ParallelismStrategy::ModelParallel,
                        "pipeline_parallel" => hpc_channels::ParallelismStrategy::PipelineParallel,
                        _ => hpc_channels::ParallelismStrategy::DataParallel,
                    },
                }),
            };

            let _ = tx.send(TrainingMessage::Start {
                job_id: job_id.clone(),
                config: channel_config,
            }).await;
        }

        tracing::info!("Started training job: {}", job_id);
        Ok(job)
    }

    /// Get a training job by ID.
    pub async fn get_job(&self, job_id: &str) -> Option<TrainingJob> {
        self.jobs.get(job_id).map(|r| r.clone())
    }

    /// Get all training jobs.
    pub async fn get_all_jobs(&self) -> Vec<TrainingJob> {
        self.jobs.iter().map(|r| r.value().clone()).collect()
    }

    /// Get training summary.
    pub async fn get_summary(&self) -> TrainingSummary {
        TrainingSummary {
            total_jobs: self.jobs.len(),
            running_jobs: self.jobs.iter().filter(|r| r.value().status == TrainingStatus::Running).count(),
            completed_jobs: self.jobs.iter().filter(|r| r.value().status == TrainingStatus::Completed).count(),
            failed_jobs: self.jobs.iter().filter(|r| r.value().status == TrainingStatus::Failed).count(),
            queued_jobs: self.jobs.iter().filter(|r| r.value().status == TrainingStatus::Queued).count(),
        }
    }

    /// Pause a training job.
    pub async fn pause_job(&self, job_id: &str) -> Result<(), String> {
        let mut job_ref = self.jobs.get_mut(job_id).ok_or("Job not found")?;

        if job_ref.status != TrainingStatus::Running {
            return Err("Job is not running".to_string());
        }

        job_ref.status = TrainingStatus::Paused;
        tracing::info!("Paused training job: {}", job_id);
        Ok(())
    }

    /// Resume a training job.
    pub async fn resume_job(&self, job_id: &str) -> Result<(), String> {
        let mut job_ref = self.jobs.get_mut(job_id).ok_or("Job not found")?;

        if job_ref.status != TrainingStatus::Paused {
            return Err("Job is not paused".to_string());
        }

        job_ref.status = TrainingStatus::Running;
        tracing::info!("Resumed training job: {}", job_id);
        Ok(())
    }

    /// Cancel a training job.
    pub async fn cancel_job(&self, job_id: &str) -> Result<(), String> {
        let mut job_ref = self.jobs.get_mut(job_id).ok_or("Job not found")?;

        if job_ref.status == TrainingStatus::Completed || job_ref.status == TrainingStatus::Failed {
            return Err("Job is already finished".to_string());
        }

        job_ref.status = TrainingStatus::Cancelled;
        tracing::info!("Cancelled training job: {}", job_id);
        Ok(())
    }

    /// Simulate progress update (for demo purposes).
    #[allow(dead_code)]
    pub async fn simulate_progress(&self) {
        for mut job_ref in self.jobs.iter_mut() {
            let job = job_ref.value_mut();
            if job.status == TrainingStatus::Running {
                // Increment progress
                job.progress.current_step += 10;
                job.progress.samples_processed += job.config.batch_size as u64 * 10;
                job.progress.completion_percentage =
                    (job.progress.current_step as f64 / job.progress.total_steps as f64) * 100.0;

                // Update epoch if needed
                let steps_per_epoch = job.progress.total_steps / job.config.epochs as u64;
                job.progress.current_epoch = (job.progress.current_step / steps_per_epoch) as u32;

                // Update metrics
                job.metrics.loss = (2.0 - job.progress.completion_percentage / 50.0).max(0.1);
                job.metrics.gpu_utilization = 0.92;
                job.metrics.throughput_samples_per_sec = 250.0;

                // Check if completed
                if job.progress.completion_percentage >= 100.0 {
                    job.status = TrainingStatus::Completed;
                    job.completed_at = Some(
                        std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap()
                            .as_secs()
                    );
                }
            }
        }
    }
}

impl Default for TrainingBridge {
    fn default() -> Self {
        Self::new()
    }
}

/// Start the training message handler task.
#[allow(dead_code)]
pub async fn start_training_handler(bridge: Arc<TrainingBridge>) {
    let (_tx, mut rx) = hpc_channels::channel::<TrainingMessage>(channels::TRAINING_CONTROL);
    let progress_tx = hpc_channels::broadcast::<TrainingMessage>(channels::TRAINING_PROGRESS, 256);

    tracing::info!("Training handler started");

    tokio::spawn(async move {
        while let Some(msg) = rx.recv().await {
            match msg {
                TrainingMessage::Pause { job_id } => {
                    if let Err(e) = bridge.pause_job(&job_id).await {
                        tracing::error!("Failed to pause job {}: {}", job_id, e);
                    }
                }
                TrainingMessage::Resume { job_id } => {
                    if let Err(e) = bridge.resume_job(&job_id).await {
                        tracing::error!("Failed to resume job {}: {}", job_id, e);
                    }
                }
                TrainingMessage::Cancel { job_id } => {
                    if let Err(e) = bridge.cancel_job(&job_id).await {
                        tracing::error!("Failed to cancel job {}: {}", job_id, e);
                    }
                }
                TrainingMessage::GetStatus { job_id } => {
                    if let Some(job) = bridge.get_job(&job_id).await {
                        let _ = progress_tx.send(TrainingMessage::Status {
                            job_id,
                            status: job.status.into(),
                        });
                    }
                }
                _ => {}
            }
        }
    });
}
