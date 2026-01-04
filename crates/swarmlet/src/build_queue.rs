//! Build Queue Management with Priority
//!
//! This module provides a priority-based queue for build jobs.
//! Jobs are scheduled based on priority level and submission time,
//! with fair scheduling to prevent starvation of low-priority jobs.

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::build_job::BuildJob;

/// Build job priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum BuildPriority {
    /// Critical builds - immediate execution
    Critical = 100,
    /// High priority builds
    High = 75,
    /// Normal priority (default)
    #[default]
    Normal = 50,
    /// Low priority builds (background)
    Low = 25,
    /// Batch jobs - run when idle
    Batch = 10,
}

impl BuildPriority {
    /// Get the base priority value
    pub fn value(&self) -> i32 {
        *self as i32
    }

    /// Parse from string
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "critical" => Some(Self::Critical),
            "high" => Some(Self::High),
            "normal" | "default" => Some(Self::Normal),
            "low" => Some(Self::Low),
            "batch" | "background" => Some(Self::Batch),
            _ => None,
        }
    }
}

/// A queued build job with priority and scheduling metadata
#[derive(Debug, Clone)]
pub struct QueuedBuild {
    /// Build job
    pub job: BuildJob,
    /// Priority level
    pub priority: BuildPriority,
    /// When the job was queued
    pub queued_at: DateTime<Utc>,
    /// Optional deadline
    pub deadline: Option<DateTime<Utc>>,
    /// Number of times this job has been deferred
    pub defer_count: u32,
    /// Group ID for related builds
    pub group_id: Option<String>,
    /// User/tenant ID for fair scheduling
    pub user_id: Option<String>,
}

impl QueuedBuild {
    /// Create a new queued build with default priority
    pub fn new(job: BuildJob) -> Self {
        Self {
            job,
            priority: BuildPriority::default(),
            queued_at: Utc::now(),
            deadline: None,
            defer_count: 0,
            group_id: None,
            user_id: None,
        }
    }

    /// Set priority
    pub fn with_priority(mut self, priority: BuildPriority) -> Self {
        self.priority = priority;
        self
    }

    /// Set deadline
    pub fn with_deadline(mut self, deadline: DateTime<Utc>) -> Self {
        self.deadline = Some(deadline);
        self
    }

    /// Set group ID
    pub fn with_group(mut self, group_id: String) -> Self {
        self.group_id = Some(group_id);
        self
    }

    /// Set user ID
    pub fn with_user(mut self, user_id: String) -> Self {
        self.user_id = Some(user_id);
        self
    }

    /// Calculate effective priority with aging
    /// Priority increases over time to prevent starvation
    pub fn effective_priority(&self) -> i32 {
        let base = self.priority.value();
        let wait_time = Utc::now() - self.queued_at;

        // Add 1 point per minute of waiting (max +30)
        let aging_bonus = (wait_time.num_minutes() as i32).min(30);

        // Add bonus for jobs near deadline
        let deadline_bonus = if let Some(deadline) = self.deadline {
            let time_left = deadline - Utc::now();
            if time_left.num_minutes() < 10 {
                20 // Urgent!
            } else if time_left.num_minutes() < 30 {
                10
            } else if time_left.num_minutes() < 60 {
                5
            } else {
                0
            }
        } else {
            0
        };

        // Add bonus for deferred jobs
        let defer_bonus = (self.defer_count as i32 * 5).min(15);

        base + aging_bonus + deadline_bonus + defer_bonus
    }
}

// Implement ordering for the priority queue (max-heap)
impl PartialEq for QueuedBuild {
    fn eq(&self, other: &Self) -> bool {
        self.job.id == other.job.id
    }
}

impl Eq for QueuedBuild {}

impl PartialOrd for QueuedBuild {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for QueuedBuild {
    fn cmp(&self, other: &Self) -> Ordering {
        // Higher priority first
        let priority_cmp = self.effective_priority().cmp(&other.effective_priority());
        if priority_cmp != Ordering::Equal {
            return priority_cmp;
        }

        // Earlier queued first (FIFO within same priority)
        other.queued_at.cmp(&self.queued_at)
    }
}

/// Queue configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueConfig {
    /// Maximum queue size
    pub max_queue_size: usize,
    /// Maximum builds per user at a time (fair scheduling)
    pub max_per_user: usize,
    /// Maximum build age before force-dequeue (hours)
    pub max_age_hours: u64,
    /// Enable priority aging
    pub enable_aging: bool,
}

impl Default for QueueConfig {
    fn default() -> Self {
        Self {
            max_queue_size: 1000,
            max_per_user: 10,
            max_age_hours: 24,
            enable_aging: true,
        }
    }
}

/// Build queue manager
pub struct BuildQueue {
    /// The priority queue
    queue: Arc<RwLock<BinaryHeap<QueuedBuild>>>,
    /// Index by job ID for O(1) lookup
    index: Arc<RwLock<HashMap<Uuid, QueuedBuild>>>,
    /// Per-user job counts for fair scheduling
    user_counts: Arc<RwLock<HashMap<String, usize>>>,
    /// Configuration
    config: QueueConfig,
}

impl BuildQueue {
    /// Create a new build queue
    pub fn new(config: QueueConfig) -> Self {
        Self {
            queue: Arc::new(RwLock::new(BinaryHeap::new())),
            index: Arc::new(RwLock::new(HashMap::new())),
            user_counts: Arc::new(RwLock::new(HashMap::new())),
            config,
        }
    }

    /// Create with default configuration
    pub fn default_new() -> Self {
        Self::new(QueueConfig::default())
    }

    /// Enqueue a build job
    pub async fn enqueue(&self, queued_build: QueuedBuild) -> Result<(), QueueError> {
        // Check queue size limit
        let current_size = self.queue.read().await.len();
        if current_size >= self.config.max_queue_size {
            return Err(QueueError::QueueFull);
        }

        // Check per-user limit
        if let Some(ref user_id) = queued_build.user_id {
            let user_counts = self.user_counts.read().await;
            if let Some(&count) = user_counts.get(user_id) {
                if count >= self.config.max_per_user {
                    return Err(QueueError::UserLimitExceeded);
                }
            }
        }

        let job_id = queued_build.job.id;

        // Check for duplicate
        {
            let index = self.index.read().await;
            if index.contains_key(&job_id) {
                return Err(QueueError::DuplicateJob);
            }
        }

        // Update user count
        if let Some(ref user_id) = queued_build.user_id {
            let mut user_counts = self.user_counts.write().await;
            *user_counts.entry(user_id.clone()).or_insert(0) += 1;
        }

        // Add to queue and index
        {
            let mut queue = self.queue.write().await;
            let mut index = self.index.write().await;
            index.insert(job_id, queued_build.clone());
            queue.push(queued_build);
        }

        Ok(())
    }

    /// Dequeue the highest priority build
    pub async fn dequeue(&self) -> Option<QueuedBuild> {
        let mut queue = self.queue.write().await;
        let mut index = self.index.write().await;

        if let Some(build) = queue.pop() {
            index.remove(&build.job.id);

            // Update user count
            if let Some(ref user_id) = build.user_id {
                let mut user_counts = self.user_counts.write().await;
                if let Some(count) = user_counts.get_mut(user_id) {
                    *count = count.saturating_sub(1);
                    if *count == 0 {
                        user_counts.remove(user_id);
                    }
                }
            }

            Some(build)
        } else {
            None
        }
    }

    /// Peek at the next build without removing it
    pub async fn peek(&self) -> Option<QueuedBuild> {
        let queue = self.queue.read().await;
        queue.peek().cloned()
    }

    /// Get a specific job by ID
    pub async fn get(&self, job_id: Uuid) -> Option<QueuedBuild> {
        let index = self.index.read().await;
        index.get(&job_id).cloned()
    }

    /// Remove a specific job from the queue
    pub async fn remove(&self, job_id: Uuid) -> Option<QueuedBuild> {
        let mut index = self.index.write().await;

        if let Some(build) = index.remove(&job_id) {
            // Rebuild the queue without this job
            let mut queue = self.queue.write().await;
            let items: Vec<_> = queue.drain().filter(|b| b.job.id != job_id).collect();
            *queue = items.into_iter().collect();

            // Update user count
            if let Some(ref user_id) = build.user_id {
                let mut user_counts = self.user_counts.write().await;
                if let Some(count) = user_counts.get_mut(user_id) {
                    *count = count.saturating_sub(1);
                    if *count == 0 {
                        user_counts.remove(user_id);
                    }
                }
            }

            Some(build)
        } else {
            None
        }
    }

    /// Update priority of a queued job
    pub async fn update_priority(
        &self,
        job_id: Uuid,
        priority: BuildPriority,
    ) -> Result<(), QueueError> {
        let mut index = self.index.write().await;

        if let Some(build) = index.get_mut(&job_id) {
            build.priority = priority;

            // Rebuild queue to reflect new priority
            let mut queue = self.queue.write().await;
            let items: Vec<_> = queue.drain().collect();
            *queue = items
                .into_iter()
                .map(|mut b| {
                    if b.job.id == job_id {
                        b.priority = priority;
                    }
                    b
                })
                .collect();

            Ok(())
        } else {
            Err(QueueError::JobNotFound)
        }
    }

    /// Get current queue length
    pub async fn len(&self) -> usize {
        self.queue.read().await.len()
    }

    /// Check if queue is empty
    pub async fn is_empty(&self) -> bool {
        self.queue.read().await.is_empty()
    }

    /// Get queue status
    pub async fn status(&self) -> QueueStatus {
        let queue = self.queue.read().await;
        let user_counts = self.user_counts.read().await;

        let mut by_priority = HashMap::new();
        for build in queue.iter() {
            *by_priority.entry(build.priority).or_insert(0) += 1;
        }

        QueueStatus {
            total_jobs: queue.len(),
            by_priority,
            active_users: user_counts.len(),
            oldest_job: queue.iter().map(|b| b.queued_at).min(),
        }
    }

    /// Get jobs for a specific user
    pub async fn get_user_jobs(&self, user_id: &str) -> Vec<QueuedBuild> {
        let index = self.index.read().await;
        index
            .values()
            .filter(|b| b.user_id.as_deref() == Some(user_id))
            .cloned()
            .collect()
    }

    /// Clean up expired jobs
    pub async fn cleanup_expired(&self) -> Vec<QueuedBuild> {
        let cutoff = Utc::now() - Duration::hours(self.config.max_age_hours as i64);
        let mut expired = Vec::new();

        let mut queue = self.queue.write().await;
        let mut index = self.index.write().await;
        let mut user_counts = self.user_counts.write().await;

        // Collect expired jobs
        let (keep, remove): (Vec<_>, Vec<_>) = queue.drain().partition(|b| b.queued_at > cutoff);

        for build in remove {
            index.remove(&build.job.id);
            if let Some(ref user_id) = build.user_id {
                if let Some(count) = user_counts.get_mut(user_id) {
                    *count = count.saturating_sub(1);
                    if *count == 0 {
                        user_counts.remove(user_id);
                    }
                }
            }
            expired.push(build);
        }

        *queue = keep.into_iter().collect();
        expired
    }

    /// List all queued jobs (sorted by effective priority)
    pub async fn list_all(&self) -> Vec<QueuedBuild> {
        let queue = self.queue.read().await;
        let mut jobs: Vec<_> = queue.iter().cloned().collect();
        jobs.sort_by(|a, b| b.effective_priority().cmp(&a.effective_priority()));
        jobs
    }
}

/// Queue status information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueStatus {
    /// Total jobs in queue
    pub total_jobs: usize,
    /// Jobs by priority level
    pub by_priority: HashMap<BuildPriority, usize>,
    /// Number of unique users with jobs
    pub active_users: usize,
    /// When the oldest job was queued
    pub oldest_job: Option<DateTime<Utc>>,
}

/// Queue operation errors
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QueueError {
    /// Queue is full
    QueueFull,
    /// User has too many jobs queued
    UserLimitExceeded,
    /// Job is already in queue
    DuplicateJob,
    /// Job not found
    JobNotFound,
}

impl std::fmt::Display for QueueError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::QueueFull => write!(f, "Build queue is full"),
            Self::UserLimitExceeded => write!(f, "User queue limit exceeded"),
            Self::DuplicateJob => write!(f, "Job already in queue"),
            Self::JobNotFound => write!(f, "Job not found in queue"),
        }
    }
}

impl std::error::Error for QueueError {}

/// Shared queue type
pub type SharedBuildQueue = Arc<BuildQueue>;

/// Create a shared build queue
pub fn create_build_queue(config: QueueConfig) -> SharedBuildQueue {
    Arc::new(BuildQueue::new(config))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::build_job::{BuildSource, CargoCommand};
    use std::path::PathBuf;

    fn make_test_job() -> BuildJob {
        BuildJob::new(
            CargoCommand::Build,
            BuildSource::Local {
                path: PathBuf::from("."),
            },
        )
    }

    #[tokio::test]
    async fn test_enqueue_dequeue() {
        let queue = BuildQueue::default_new();

        let job = make_test_job();
        let job_id = job.id;
        let queued = QueuedBuild::new(job);

        queue.enqueue(queued).await.unwrap();
        assert_eq!(queue.len().await, 1);

        let dequeued = queue.dequeue().await.unwrap();
        assert_eq!(dequeued.job.id, job_id);
        assert_eq!(queue.len().await, 0);
    }

    #[tokio::test]
    async fn test_priority_ordering() {
        let queue = BuildQueue::default_new();

        // Add low priority first
        let low = QueuedBuild::new(make_test_job()).with_priority(BuildPriority::Low);
        let low_id = low.job.id;
        queue.enqueue(low).await.unwrap();

        // Add high priority second
        let high = QueuedBuild::new(make_test_job()).with_priority(BuildPriority::High);
        let high_id = high.job.id;
        queue.enqueue(high).await.unwrap();

        // Add normal priority third
        let normal = QueuedBuild::new(make_test_job()).with_priority(BuildPriority::Normal);
        let normal_id = normal.job.id;
        queue.enqueue(normal).await.unwrap();

        // Should dequeue in priority order: high, normal, low
        assert_eq!(queue.dequeue().await.unwrap().job.id, high_id);
        assert_eq!(queue.dequeue().await.unwrap().job.id, normal_id);
        assert_eq!(queue.dequeue().await.unwrap().job.id, low_id);
    }

    #[tokio::test]
    async fn test_remove_job() {
        let queue = BuildQueue::default_new();

        let job = make_test_job();
        let job_id = job.id;
        queue.enqueue(QueuedBuild::new(job)).await.unwrap();

        let removed = queue.remove(job_id).await;
        assert!(removed.is_some());
        assert_eq!(queue.len().await, 0);
    }

    #[tokio::test]
    async fn test_update_priority() {
        let queue = BuildQueue::default_new();

        let job = make_test_job();
        let job_id = job.id;
        queue
            .enqueue(QueuedBuild::new(job).with_priority(BuildPriority::Low))
            .await
            .unwrap();

        queue
            .update_priority(job_id, BuildPriority::Critical)
            .await
            .unwrap();

        let build = queue.get(job_id).await.unwrap();
        assert_eq!(build.priority, BuildPriority::Critical);
    }

    #[tokio::test]
    async fn test_user_limit() {
        let config = QueueConfig {
            max_per_user: 2,
            ..Default::default()
        };
        let queue = BuildQueue::new(config);

        // Add 2 jobs for user1 (should succeed)
        for _ in 0..2 {
            queue
                .enqueue(QueuedBuild::new(make_test_job()).with_user("user1".to_string()))
                .await
                .unwrap();
        }

        // Third job should fail
        let result = queue
            .enqueue(QueuedBuild::new(make_test_job()).with_user("user1".to_string()))
            .await;
        assert_eq!(result, Err(QueueError::UserLimitExceeded));

        // But user2 should still be able to add
        queue
            .enqueue(QueuedBuild::new(make_test_job()).with_user("user2".to_string()))
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn test_duplicate_job() {
        let queue = BuildQueue::default_new();

        let job = make_test_job();
        let queued = QueuedBuild::new(job.clone());

        queue.enqueue(queued.clone()).await.unwrap();

        // Same job ID should fail
        let result = queue.enqueue(QueuedBuild { job, ..queued }).await;
        assert_eq!(result, Err(QueueError::DuplicateJob));
    }

    #[tokio::test]
    async fn test_queue_status() {
        let queue = BuildQueue::default_new();

        queue
            .enqueue(QueuedBuild::new(make_test_job()).with_priority(BuildPriority::High))
            .await
            .unwrap();
        queue
            .enqueue(QueuedBuild::new(make_test_job()).with_priority(BuildPriority::High))
            .await
            .unwrap();
        queue
            .enqueue(QueuedBuild::new(make_test_job()).with_priority(BuildPriority::Low))
            .await
            .unwrap();

        let status = queue.status().await;
        assert_eq!(status.total_jobs, 3);
        assert_eq!(status.by_priority.get(&BuildPriority::High), Some(&2));
        assert_eq!(status.by_priority.get(&BuildPriority::Low), Some(&1));
    }

    #[test]
    fn test_effective_priority_aging() {
        let mut build = QueuedBuild::new(make_test_job());
        build.priority = BuildPriority::Low;

        let base_priority = build.effective_priority();

        // Simulate waiting
        build.queued_at = Utc::now() - Duration::minutes(10);
        let aged_priority = build.effective_priority();

        assert!(aged_priority > base_priority);
    }

    #[test]
    fn test_priority_from_str() {
        assert_eq!(
            BuildPriority::from_str("critical"),
            Some(BuildPriority::Critical)
        );
        assert_eq!(BuildPriority::from_str("HIGH"), Some(BuildPriority::High));
        assert_eq!(
            BuildPriority::from_str("default"),
            Some(BuildPriority::Normal)
        );
        assert_eq!(
            BuildPriority::from_str("background"),
            Some(BuildPriority::Batch)
        );
        assert_eq!(BuildPriority::from_str("invalid"), None);
    }
}
