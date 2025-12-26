use crate::adapters::InventoryClient;
use crate::config::Config;
use crate::db::JobRepository;
use crate::models::{Job, JobState};
use crate::queue::{FairShareCalculator, PriorityQueue};
use crate::scheduler::{PlacementEngine, PreemptionManager};
use crate::Result;
use sqlx::PgPool;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

/// Core scheduler with EASY backfill algorithm
pub struct Scheduler {
    config: Arc<Config>,
    queue: Arc<RwLock<PriorityQueue>>,
    fair_share: Arc<RwLock<FairShareCalculator>>,
    placement_engine: Arc<PlacementEngine>,
    #[allow(dead_code)]
    preemption_manager: Arc<PreemptionManager>,
    repository: Arc<JobRepository>,
    inventory_client: Arc<InventoryClient>,
}

impl Scheduler {
    pub async fn new(config: Config, pool: PgPool) -> Result<Self> {
        let inventory_client = Arc::new(InventoryClient::new(&config.inventory)?);
        let placement_engine = Arc::new(PlacementEngine::new((*inventory_client).clone()));
        let repository = Arc::new(JobRepository::new(pool));

        Ok(Self {
            config: Arc::new(config.clone()),
            queue: Arc::new(RwLock::new(PriorityQueue::new())),
            fair_share: Arc::new(RwLock::new(FairShareCalculator::new())),
            placement_engine,
            preemption_manager: Arc::new(PreemptionManager::new(config)),
            repository,
            inventory_client,
        })
    }

    /// Submit a new job to the queue
    pub async fn submit_job(&self, mut job: Job) -> Result<Job> {
        // Persist to database
        let persisted = self.repository.create(&job).await?;
        job.id = persisted.id;

        // Add to queue
        let mut queue = self.queue.write().await;
        queue.enqueue(job.clone());

        Ok(job)
    }

    /// Get job by ID
    pub async fn get_job(&self, job_id: Uuid) -> Result<Job> {
        self.repository.get_by_id(job_id).await
    }

    /// List all jobs with optional filtering
    pub async fn list_jobs(&self, state: Option<JobState>) -> Result<Vec<Job>> {
        if let Some(state) = state {
            self.repository.list_by_state(state).await
        } else {
            self.repository.list_all().await
        }
    }

    /// Cancel a job
    pub async fn cancel_job(&self, job_id: Uuid) -> Result<Job> {
        let mut job = self.repository.get_by_id(job_id).await?;

        // Try to remove from queue if still queued
        let mut queue = self.queue.write().await;
        queue.remove(job_id);

        // Transition state
        job.transition_to(JobState::Cancelled)?;

        // Update in database
        self.repository.update(&job).await?;

        // Release resources if running
        if matches!(job.state, JobState::Running | JobState::Scheduled) {
            let _ = self.inventory_client.release_gpus(job_id).await;
        }

        Ok(job)
    }

    /// Main scheduling loop iteration
    pub async fn schedule_next(&self) -> Result<Option<Uuid>> {
        let mut queue = self.queue.write().await;

        // Get next job from queue
        let mut job = match queue.dequeue() {
            Some(j) => j,
            None => return Ok(None),
        };

        // Try to find placement
        match self.placement_engine.find_placement(&job).await {
            Ok(placement) => {
                // Reserve GPUs
                self.inventory_client
                    .reserve_gpus(job.id, &placement.gpu_ids)
                    .await?;

                // Transition to scheduled
                job.transition_to(JobState::Scheduled)?;
                self.repository.update(&job).await?;

                tracing::info!(
                    job_id = %job.id,
                    user_id = %job.user_id,
                    gpus = placement.gpu_ids.len(),
                    nodes = placement.node_ids.len(),
                    score = placement.score,
                    "Job scheduled"
                );

                Ok(Some(job.id))
            }
            Err(e) if e.is_resource_exhausted() => {
                // Re-queue the job
                queue.enqueue(job);

                // Try backfill: look for smaller jobs that can fit
                if self.config.scheduler.enable_backfill {
                    self.try_backfill(&mut queue).await?;
                }

                Ok(None)
            }
            Err(e) => Err(e),
        }
    }

    /// EASY backfill: try to schedule smaller jobs without delaying high-priority jobs
    async fn try_backfill(&self, queue: &mut PriorityQueue) -> Result<usize> {
        let mut backfilled = 0;
        let candidates = queue.all_jobs();

        for job in candidates {
            // Only backfill low/normal priority jobs
            if matches!(job.priority, crate::models::Priority::High) {
                continue;
            }

            // Try placement
            if let Ok(placement) = self.placement_engine.find_placement(&job).await {
                // Remove from queue
                if let Some(mut backfill_job) = queue.remove(job.id) {
                    // Reserve and schedule
                    self.inventory_client
                        .reserve_gpus(backfill_job.id, &placement.gpu_ids)
                        .await?;

                    backfill_job.transition_to(JobState::Scheduled)?;
                    self.repository.update(&backfill_job).await?;

                    backfilled += 1;

                    tracing::debug!(
                        job_id = %backfill_job.id,
                        "Job backfilled"
                    );
                }
            }
        }

        Ok(backfilled)
    }

    /// Get queue statistics
    pub async fn get_queue_stats(&self) -> Result<QueueStats> {
        let queue = self.queue.read().await;

        Ok(QueueStats {
            total: queue.len(),
            urgent_priority: queue.count_by_priority(crate::models::Priority::Urgent),
            high_priority: queue.count_by_priority(crate::models::Priority::High),
            normal_priority: queue.count_by_priority(crate::models::Priority::Normal),
            low_priority: queue.count_by_priority(crate::models::Priority::Low),
        })
    }

    /// Update fair-share usage
    pub async fn record_job_completion(&self, job: &Job) -> Result<()> {
        let mut fair_share = self.fair_share.write().await;

        // Calculate GPU-hours used
        if let Some(started) = job.started_at {
            if let Some(completed) = job.completed_at {
                let duration = completed.signed_duration_since(started);
                let hours = duration.num_seconds() as f64 / 3600.0;

                let gpu_count = job.resources.get_gpu_spec()
                    .map(|s| s.amount)
                    .unwrap_or(0.0);
                let gpu_hours = hours * gpu_count;

                fair_share.record_gpu_usage(&job.user_id, gpu_hours);
            }
        }

        Ok(())
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct QueueStats {
    pub total: usize,
    pub urgent_priority: usize,
    pub high_priority: usize,
    pub normal_priority: usize,
    pub low_priority: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_queue_stats_creation() {
        let stats = QueueStats {
            total: 10,
            urgent_priority: 0,
            high_priority: 3,
            normal_priority: 5,
            low_priority: 2,
        };

        assert_eq!(stats.total, 10);
        assert_eq!(stats.high_priority, 3);
    }
}
