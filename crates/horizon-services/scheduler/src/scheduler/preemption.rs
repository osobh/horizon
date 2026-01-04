use crate::checkpoint::CheckpointManager;
use crate::config::Config;
use crate::models::{Job, JobState};
use crate::Result;
use std::sync::Arc;

/// Manages job preemption logic
pub struct PreemptionManager {
    config: Config,
    checkpoint_manager: Arc<CheckpointManager>,
}

impl PreemptionManager {
    pub fn new(config: Config) -> Self {
        let checkpoint_manager = Arc::new(CheckpointManager::new(&config.checkpoint));

        Self {
            config,
            checkpoint_manager,
        }
    }

    /// Determine if a job should be preempted for a higher-priority job
    pub fn should_preempt(&self, running_job: &Job, queued_job: &Job) -> bool {
        if !self.config.scheduler.enable_preemption {
            return false;
        }

        // Only preempt if queued job has higher priority
        queued_job.priority > running_job.priority
    }

    /// Get list of jobs that can be preempted for a given job
    pub fn find_preemptable_jobs(&self, running_jobs: &[Job], queued_job: &Job) -> Vec<Job> {
        running_jobs
            .iter()
            .filter(|job| self.should_preempt(job, queued_job))
            .cloned()
            .collect()
    }

    /// Preempt a running job (with checkpoint)
    pub async fn preempt_job(&self, job: &mut Job) -> Result<()> {
        // Create checkpoint before preempting
        let checkpoint = self.checkpoint_manager.create_checkpoint(job).await?;

        // Store checkpoint path
        job.checkpoint_path = Some(checkpoint.storage_path);

        // Transition to preempted state
        job.transition_to(JobState::Preempted)?;

        Ok(())
    }

    /// Resume a preempted job from checkpoint
    pub async fn resume_job(&self, job: &mut Job) -> Result<()> {
        if let Some(_checkpoint_path) = &job.checkpoint_path {
            // Load checkpoint
            let _checkpoint = self
                .checkpoint_manager
                .load_checkpoint(&job.id.to_string())
                .await?;

            // Transition back to queued
            job.transition_to(JobState::Queued)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::Priority;

    fn create_job_with_priority(priority: Priority) -> Job {
        Job::builder()
            .user_id("test")
            .gpu_count(4)
            .priority(priority)
            .build()
            .unwrap()
    }

    #[test]
    fn test_should_preempt_higher_priority() {
        let config = Config::from_env().unwrap();
        let manager = PreemptionManager::new(config);

        let low_job = create_job_with_priority(Priority::Low);
        let high_job = create_job_with_priority(Priority::High);

        assert!(manager.should_preempt(&low_job, &high_job));
        assert!(!manager.should_preempt(&high_job, &low_job));
    }

    #[test]
    fn test_find_preemptable_jobs() {
        let config = Config::from_env().unwrap();
        let manager = PreemptionManager::new(config);

        let running = vec![
            create_job_with_priority(Priority::Low),
            create_job_with_priority(Priority::Normal),
            create_job_with_priority(Priority::High),
        ];

        let high_queued = create_job_with_priority(Priority::High);

        let preemptable = manager.find_preemptable_jobs(&running, &high_queued);

        // Should find low and normal priority jobs
        assert!(preemptable.iter().any(|j| j.priority == Priority::Low));
        assert!(preemptable.iter().any(|j| j.priority == Priority::Normal));
        assert!(!preemptable.iter().any(|j| j.priority == Priority::High));
    }
}
