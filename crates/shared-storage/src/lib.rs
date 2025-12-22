//! Shared storage system for CPU-GPU agent communication
//!
//! This crate provides job-based data exchange between CPU and GPU agents
//! using high-performance NVMe storage at /nvme/gpu/shared/

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

#[cfg(test)]
mod additional_tests;
#[cfg(test)]
mod e2e_tests;
#[cfg(test)]
mod edge_case_tests;
#[cfg(test)]
mod integration_tests;

/// Job priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum JobPriority {
    Low,
    Normal,
    High,
    Critical,
}

/// Job types
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum JobType {
    Compute,
    Training,
    Inference,
    DataProcessing,
    Streaming,
    Custom(String),
}

/// Job status
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JobStatus {
    Pending,
    Processing,
    Completed,
    Failed(String),
    Cancelled,
}

/// A job in the shared storage system
#[derive(Debug, Clone)]
pub struct Job {
    pub id: String,
    pub job_type: JobType,
    pub priority: JobPriority,
    pub status: JobStatus,
    pub data_path: PathBuf,
    pub result_path: Option<PathBuf>,
    pub created_at: SystemTime,
    pub updated_at: SystemTime,
    pub metadata: HashMap<String, String>,
}

/// Configuration for shared storage
#[derive(Debug, Clone)]
pub struct SharedStorageConfig {
    pub base_path: PathBuf,
    pub max_job_size: usize,
    pub cleanup_interval: Duration,
    pub job_ttl: Duration,
}

impl Default for SharedStorageConfig {
    fn default() -> Self {
        Self {
            base_path: PathBuf::from("/nvme/gpu/shared"),
            max_job_size: 1024 * 1024 * 1024,           // 1GB
            cleanup_interval: Duration::from_secs(300), // 5 minutes
            job_ttl: Duration::from_secs(3600),         // 1 hour
        }
    }
}

/// Shared storage manager
pub struct SharedStorageManager {
    config: SharedStorageConfig,
    jobs: Arc<Mutex<HashMap<String, Job>>>,
}

impl SharedStorageManager {
    /// Create a new shared storage manager
    pub fn new(config: SharedStorageConfig) -> Self {
        Self {
            config,
            jobs: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Submit a job to the shared storage
    pub fn submit_job(&self, job: Job) -> Result<String, String> {
        let mut jobs = self.jobs.lock().map_err(|e| e.to_string())?;

        if jobs.contains_key(&job.id) {
            return Err(format!("Job {} already exists", job.id));
        }

        let job_id = job.id.clone();
        jobs.insert(job_id.clone(), job);
        Ok(job_id)
    }

    /// Get a job by ID
    pub fn get_job(&self, job_id: &str) -> Result<Option<Job>, String> {
        let jobs = self.jobs.lock().map_err(|e| e.to_string())?;
        Ok(jobs.get(job_id).cloned())
    }

    /// Update job status
    pub fn update_job_status(&self, job_id: &str, status: JobStatus) -> Result<(), String> {
        let mut jobs = self.jobs.lock().map_err(|e| e.to_string())?;

        match jobs.get_mut(job_id) {
            Some(job) => {
                job.status = status;
                job.updated_at = SystemTime::now();
                Ok(())
            }
            None => Err(format!("Job {} not found", job_id)),
        }
    }

    /// List jobs by status
    pub fn list_jobs_by_status(&self, status: &JobStatus) -> Result<Vec<Job>, String> {
        let jobs = self.jobs.lock().map_err(|e| e.to_string())?;
        Ok(jobs
            .values()
            .filter(|job| &job.status == status)
            .cloned()
            .collect())
    }

    /// Clean up expired jobs
    pub fn cleanup_expired_jobs(&self) -> Result<usize, String> {
        let mut jobs = self.jobs.lock().map_err(|e| e.to_string())?;
        let mut removed = 0;

        jobs.retain(|_, job| {
            let expired = match job.created_at.elapsed() {
                Ok(elapsed) => elapsed > self.config.job_ttl,
                Err(_) => false, // Future timestamps should not be expired
            };

            if expired && matches!(job.status, JobStatus::Completed | JobStatus::Failed(_)) {
                removed += 1;
                false
            } else {
                true
            }
        });

        Ok(removed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    fn create_test_job(id: &str) -> Job {
        Job {
            id: id.to_string(),
            job_type: JobType::Compute,
            priority: JobPriority::Normal,
            status: JobStatus::Pending,
            data_path: PathBuf::from(format!("/tmp/job_{}/data", id)),
            result_path: None,
            created_at: SystemTime::now(),
            updated_at: SystemTime::now(),
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn test_job_priority_ordering() {
        assert!(JobPriority::Critical > JobPriority::High);
        assert!(JobPriority::High > JobPriority::Normal);
        assert!(JobPriority::Normal > JobPriority::Low);
    }

    #[test]
    fn test_job_type_equality() {
        assert_eq!(JobType::Compute, JobType::Compute);
        assert_ne!(JobType::Compute, JobType::Training);
        assert_eq!(
            JobType::Custom("test".to_string()),
            JobType::Custom("test".to_string())
        );
    }

    #[test]
    fn test_job_status_variants() {
        let statuses = vec![
            JobStatus::Pending,
            JobStatus::Processing,
            JobStatus::Completed,
            JobStatus::Failed("error".to_string()),
            JobStatus::Cancelled,
        ];

        for status in statuses {
            match status {
                JobStatus::Failed(msg) => assert!(!msg.is_empty()),
                _ => {}
            }
        }
    }

    #[test]
    fn test_shared_storage_config_default() {
        let config = SharedStorageConfig::default();
        assert_eq!(config.base_path, PathBuf::from("/nvme/gpu/shared"));
        assert_eq!(config.max_job_size, 1024 * 1024 * 1024);
        assert_eq!(config.cleanup_interval, Duration::from_secs(300));
        assert_eq!(config.job_ttl, Duration::from_secs(3600));
    }

    #[test]
    fn test_submit_job_success() {
        let manager = SharedStorageManager::new(SharedStorageConfig::default());
        let job = create_test_job("test1");

        let result = manager.submit_job(job.clone());
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "test1");
    }

    #[test]
    fn test_submit_duplicate_job() {
        let manager = SharedStorageManager::new(SharedStorageConfig::default());
        let job = create_test_job("test1");

        manager.submit_job(job.clone()).unwrap();
        let result = manager.submit_job(job);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("already exists"));
    }

    #[test]
    fn test_get_job_exists() {
        let manager = SharedStorageManager::new(SharedStorageConfig::default());
        let job = create_test_job("test1");

        manager.submit_job(job.clone()).unwrap();
        let retrieved = manager.get_job("test1").unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().id, "test1");
    }

    #[test]
    fn test_get_job_not_exists() {
        let manager = SharedStorageManager::new(SharedStorageConfig::default());
        let retrieved = manager.get_job("nonexistent").unwrap();
        assert!(retrieved.is_none());
    }

    #[test]
    fn test_update_job_status_success() {
        let manager = SharedStorageManager::new(SharedStorageConfig::default());
        let job = create_test_job("test1");

        manager.submit_job(job).unwrap();
        let result = manager.update_job_status("test1", JobStatus::Processing);
        assert!(result.is_ok());

        let updated = manager.get_job("test1").unwrap().unwrap();
        assert_eq!(updated.status, JobStatus::Processing);
    }

    #[test]
    fn test_update_job_status_not_found() {
        let manager = SharedStorageManager::new(SharedStorageConfig::default());
        let result = manager.update_job_status("nonexistent", JobStatus::Processing);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not found"));
    }

    #[test]
    fn test_list_jobs_by_status() {
        let manager = SharedStorageManager::new(SharedStorageConfig::default());

        // Submit multiple jobs
        for i in 0..5 {
            let mut job = create_test_job(&format!("job{}", i));
            if i < 2 {
                job.status = JobStatus::Pending;
            } else if i < 4 {
                job.status = JobStatus::Processing;
            } else {
                job.status = JobStatus::Completed;
            }
            manager.submit_job(job).unwrap();
        }

        let pending = manager.list_jobs_by_status(&JobStatus::Pending).unwrap();
        assert_eq!(pending.len(), 2);

        let processing = manager.list_jobs_by_status(&JobStatus::Processing).unwrap();
        assert_eq!(processing.len(), 2);

        let completed = manager.list_jobs_by_status(&JobStatus::Completed).unwrap();
        assert_eq!(completed.len(), 1);
    }

    #[test]
    fn test_cleanup_expired_jobs() {
        let mut config = SharedStorageConfig::default();
        config.job_ttl = Duration::from_millis(100); // Short TTL for testing

        let manager = SharedStorageManager::new(config);

        // Submit jobs
        let mut job1 = create_test_job("job1");
        job1.status = JobStatus::Completed;
        manager.submit_job(job1).unwrap();

        let mut job2 = create_test_job("job2");
        job2.status = JobStatus::Processing; // Should not be cleaned up
        manager.submit_job(job2).unwrap();

        // Wait for TTL to expire
        thread::sleep(Duration::from_millis(150));

        let removed = manager.cleanup_expired_jobs().unwrap();
        assert_eq!(removed, 1);

        // Check that only the completed job was removed
        assert!(manager.get_job("job1").unwrap().is_none());
        assert!(manager.get_job("job2").unwrap().is_some());
    }

    #[test]
    fn test_job_metadata() {
        let mut job = create_test_job("test1");
        job.metadata
            .insert("key1".to_string(), "value1".to_string());
        job.metadata
            .insert("key2".to_string(), "value2".to_string());

        let manager = SharedStorageManager::new(SharedStorageConfig::default());
        manager.submit_job(job.clone()).unwrap();

        let retrieved = manager.get_job("test1").unwrap().unwrap();
        assert_eq!(retrieved.metadata.len(), 2);
        assert_eq!(retrieved.metadata.get("key1"), Some(&"value1".to_string()));
    }

    #[test]
    fn test_job_result_path() {
        let mut job = create_test_job("test1");
        assert!(job.result_path.is_none());

        job.result_path = Some(PathBuf::from("/tmp/job_test1/result"));

        let manager = SharedStorageManager::new(SharedStorageConfig::default());
        manager.submit_job(job).unwrap();

        let retrieved = manager.get_job("test1").unwrap().unwrap();
        assert!(retrieved.result_path.is_some());
        assert_eq!(
            retrieved.result_path.unwrap(),
            PathBuf::from("/tmp/job_test1/result")
        );
    }

    #[test]
    fn test_concurrent_job_submission() {
        let manager = Arc::new(SharedStorageManager::new(SharedStorageConfig::default()));
        let mut handles = vec![];

        // Spawn multiple threads submitting jobs
        for i in 0..10 {
            let mgr = manager.clone();
            let handle = thread::spawn(move || {
                let job = create_test_job(&format!("concurrent_{}", i));
                mgr.submit_job(job)
            });
            handles.push(handle);
        }

        // Wait for all threads and collect results
        let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();

        // All submissions should succeed
        assert!(results.iter().all(|r| r.is_ok()));

        // Verify all jobs exist
        for i in 0..10 {
            let job = manager.get_job(&format!("concurrent_{}", i)).unwrap();
            assert!(job.is_some());
        }
    }

    #[test]
    fn test_custom_job_types() {
        let custom_types = vec![
            JobType::Custom("MLTraining".to_string()),
            JobType::Custom("DataAnalysis".to_string()),
            JobType::Custom("VideoProcessing".to_string()),
        ];

        for job_type in custom_types {
            let mut job = create_test_job("test");
            job.job_type = job_type.clone();

            match &job.job_type {
                JobType::Custom(name) => assert!(!name.is_empty()),
                _ => panic!("Expected custom job type"),
            }
        }
    }

    #[test]
    fn test_job_priority_in_metadata() {
        let priorities = vec![
            JobPriority::Critical,
            JobPriority::High,
            JobPriority::Normal,
            JobPriority::Low,
        ];

        for (i, priority) in priorities.iter().enumerate() {
            let mut job = create_test_job(&format!("priority_{}", i));
            job.priority = *priority;
            job.metadata
                .insert("priority_value".to_string(), format!("{:?}", priority));

            let manager = SharedStorageManager::new(SharedStorageConfig::default());
            manager.submit_job(job).unwrap();

            let retrieved = manager
                .get_job(&format!("priority_{}", i))
                .unwrap()
                .unwrap();
            assert_eq!(retrieved.priority, *priority);
        }
    }

    #[test]
    fn test_job_timestamps() {
        let manager = SharedStorageManager::new(SharedStorageConfig::default());
        let job = create_test_job("test1");
        let created_at = job.created_at;

        manager.submit_job(job).unwrap();

        // Update status after a small delay
        thread::sleep(Duration::from_millis(10));
        manager
            .update_job_status("test1", JobStatus::Completed)
            .unwrap();

        let updated = manager.get_job("test1").unwrap().unwrap();
        assert_eq!(updated.created_at, created_at);
        assert!(updated.updated_at > created_at);
    }

    #[test]
    fn test_failed_job_with_error_message() {
        let manager = SharedStorageManager::new(SharedStorageConfig::default());
        let job = create_test_job("test1");

        manager.submit_job(job).unwrap();

        let error_msg = "GPU memory allocation failed";
        manager
            .update_job_status("test1", JobStatus::Failed(error_msg.to_string()))
            .unwrap();

        let updated = manager.get_job("test1").unwrap().unwrap();
        match updated.status {
            JobStatus::Failed(msg) => assert_eq!(msg, error_msg),
            _ => panic!("Expected failed status"),
        }
    }

    #[test]
    fn test_job_lifecycle() {
        let manager = SharedStorageManager::new(SharedStorageConfig::default());
        let job = create_test_job("lifecycle");

        // Submit job
        manager.submit_job(job).unwrap();
        let job = manager.get_job("lifecycle").unwrap().unwrap();
        assert_eq!(job.status, JobStatus::Pending);

        // Start processing
        manager
            .update_job_status("lifecycle", JobStatus::Processing)
            .unwrap();
        let job = manager.get_job("lifecycle").unwrap().unwrap();
        assert_eq!(job.status, JobStatus::Processing);

        // Complete job
        manager
            .update_job_status("lifecycle", JobStatus::Completed)
            .unwrap();
        let job = manager.get_job("lifecycle").unwrap().unwrap();
        assert_eq!(job.status, JobStatus::Completed);
    }

    #[test]
    fn test_empty_job_id() {
        let manager = SharedStorageManager::new(SharedStorageConfig::default());
        let mut job = create_test_job("");
        job.id = String::new();

        // Should still accept empty ID
        let result = manager.submit_job(job);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "");
    }

    #[test]
    fn test_very_long_job_id() {
        let manager = SharedStorageManager::new(SharedStorageConfig::default());
        let long_id = "a".repeat(1000);
        let mut job = create_test_job(&long_id);
        job.id = long_id.clone();

        let result = manager.submit_job(job);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), long_id);
    }

    #[test]
    fn test_unicode_in_job_fields() {
        let manager = SharedStorageManager::new(SharedStorageConfig::default());
        let unicode_id = "job_测试_🚀";
        let mut job = create_test_job(unicode_id);
        job.metadata
            .insert("描述".to_string(), "Unicode test 测试".to_string());
        job.data_path = PathBuf::from("/tmp/路径/文件");

        manager.submit_job(job.clone()).unwrap();

        let retrieved = manager.get_job(unicode_id).unwrap().unwrap();
        assert_eq!(retrieved.id, unicode_id);
        assert_eq!(
            retrieved.metadata.get("描述"),
            Some(&"Unicode test 测试".to_string())
        );
    }

    #[test]
    fn test_job_status_transitions() {
        let manager = SharedStorageManager::new(SharedStorageConfig::default());
        let job = create_test_job("transitions");

        manager.submit_job(job).unwrap();

        // Valid transitions
        let valid_transitions = vec![
            (JobStatus::Pending, JobStatus::Processing),
            (JobStatus::Processing, JobStatus::Completed),
            (
                JobStatus::Processing,
                JobStatus::Failed("error".to_string()),
            ),
            (JobStatus::Pending, JobStatus::Cancelled),
            (JobStatus::Processing, JobStatus::Cancelled),
        ];

        for (i, (from, to)) in valid_transitions.iter().enumerate() {
            let job_id = format!("transition_{}", i);
            let mut job = create_test_job(&job_id);
            job.status = from.clone();
            manager.submit_job(job).unwrap();

            let result = manager.update_job_status(&job_id, to.clone());
            assert!(result.is_ok());

            let updated = manager.get_job(&job_id).unwrap().unwrap();
            assert_eq!(&updated.status, to);
        }
    }

    #[test]
    fn test_cleanup_with_future_timestamps() {
        let mut config = SharedStorageConfig::default();
        config.job_ttl = Duration::from_secs(3600);
        let manager = SharedStorageManager::new(config);

        // Create job with future timestamp (should not be cleaned up)
        let mut job = create_test_job("future");
        job.created_at = SystemTime::now() + Duration::from_secs(3600);
        job.status = JobStatus::Completed;

        manager.submit_job(job).unwrap();

        let removed = manager.cleanup_expired_jobs().unwrap();
        assert_eq!(removed, 0);
        assert!(manager.get_job("future").unwrap().is_some());
    }

    #[test]
    fn test_massive_metadata() {
        let manager = SharedStorageManager::new(SharedStorageConfig::default());
        let mut job = create_test_job("massive_metadata");

        // Add 1000 metadata entries
        for i in 0..1000 {
            job.metadata.insert(
                format!("key_{}", i),
                format!("value_{}_with_some_longer_content_to_increase_size", i),
            );
        }

        manager.submit_job(job.clone()).unwrap();

        let retrieved = manager.get_job("massive_metadata").unwrap().unwrap();
        assert_eq!(retrieved.metadata.len(), 1000);
    }

    #[test]
    fn test_job_priority_edge_cases() {
        use std::cmp::Ordering;

        // Test ordering consistency
        let priorities = vec![
            JobPriority::Critical,
            JobPriority::High,
            JobPriority::Normal,
            JobPriority::Low,
        ];

        for i in 0..priorities.len() {
            for j in 0..priorities.len() {
                let ord = priorities[i].cmp(&priorities[j]);
                let expected = match i.cmp(&j) {
                    Ordering::Less => Ordering::Greater, // Critical > High > Normal > Low
                    Ordering::Equal => Ordering::Equal,
                    Ordering::Greater => Ordering::Less,
                };
                assert_eq!(ord, expected);
            }
        }
    }

    #[test]
    fn test_concurrent_cleanup_and_submission() {
        let mut config = SharedStorageConfig::default();
        config.job_ttl = Duration::from_millis(10);
        let manager = Arc::new(SharedStorageManager::new(config));

        let mgr_submit = manager.clone();
        let mgr_cleanup = manager.clone();

        // Submitter thread
        let submit_handle = thread::spawn(move || {
            for i in 0..100 {
                let mut job = create_test_job(&format!("concurrent_{}", i));
                job.status = if i % 2 == 0 {
                    JobStatus::Completed
                } else {
                    JobStatus::Processing
                };
                mgr_submit.submit_job(job).ok();
                thread::sleep(Duration::from_micros(100));
            }
        });

        // Cleanup thread
        let cleanup_handle = thread::spawn(move || {
            let mut total_cleaned = 0;
            for _ in 0..10 {
                thread::sleep(Duration::from_millis(20));
                if let Ok(cleaned) = mgr_cleanup.cleanup_expired_jobs() {
                    total_cleaned += cleaned;
                }
            }
            total_cleaned
        });

        submit_handle.join().unwrap();
        let total_cleaned = cleanup_handle.join().unwrap();

        // Some jobs should have been cleaned
        assert!(total_cleaned > 0);
    }

    #[test]
    fn test_pathological_job_paths() {
        let manager = SharedStorageManager::new(SharedStorageConfig::default());

        let pathological_paths = vec![
            PathBuf::from(""),
            PathBuf::from("/"),
            PathBuf::from("//multiple//slashes//"),
            PathBuf::from("relative/path"),
            PathBuf::from("../../../etc/passwd"),
            PathBuf::from("/tmp/\0null_byte"),
            PathBuf::from("/very/long/path/".repeat(100)),
        ];

        for (i, path) in pathological_paths.iter().enumerate() {
            let mut job = create_test_job(&format!("path_{}", i));
            job.data_path = path.clone();
            job.result_path = Some(path.clone());

            // Should handle any path without crashing
            let result = manager.submit_job(job);
            assert!(result.is_ok());

            let retrieved = manager.get_job(&format!("path_{}", i)).unwrap().unwrap();
            assert_eq!(retrieved.data_path, *path);
        }
    }

    #[test]
    fn test_job_metadata_special_characters() {
        let manager = SharedStorageManager::new(SharedStorageConfig::default());
        let mut job = create_test_job("special_chars");

        // Add metadata with special characters
        let special_pairs = vec![
            ("empty", ""),
            ("spaces", "   "),
            ("newlines", "line1\nline2\nline3"),
            ("tabs", "col1\tcol2\tcol3"),
            ("quotes", r#"She said "Hello""#),
            ("backslashes", r"C:\Windows\System32"),
            ("unicode", "🎉🚀💻🔧"),
            ("control", "\x00\x01\x02\x03"),
        ];

        for (key, value) in &special_pairs {
            job.metadata.insert(key.to_string(), value.to_string());
        }

        manager.submit_job(job).unwrap();

        let retrieved = manager.get_job("special_chars").unwrap().unwrap();
        for (key, expected_value) in &special_pairs {
            assert_eq!(
                retrieved.metadata.get(*key),
                Some(&expected_value.to_string())
            );
        }
    }

    #[test]
    fn test_time_travel_edge_case() {
        let manager = SharedStorageManager::new(SharedStorageConfig::default());
        let mut job = create_test_job("time_travel");

        // Set updated_at before created_at
        job.created_at = SystemTime::now();
        job.updated_at = job.created_at - Duration::from_secs(3600);

        manager.submit_job(job).unwrap();

        // Update status (should fix the timeline)
        manager
            .update_job_status("time_travel", JobStatus::Completed)
            .unwrap();

        let updated = manager.get_job("time_travel").unwrap().unwrap();
        // After update, updated_at should be after created_at
        assert!(updated.updated_at >= updated.created_at);
    }

    #[test]
    fn test_custom_job_type_variations() {
        let manager = SharedStorageManager::new(SharedStorageConfig::default());

        let long_name = "VeryLongCustomJobTypeName".repeat(10);
        let custom_types: Vec<&str> = vec![
            "",
            " ",
            &long_name,
            "Type-With-Dashes",
            "Type_With_Underscores",
            "Type.With.Dots",
            "Type/With/Slashes",
            "类型",
            "🚀SpaceType",
        ];

        for (i, type_name) in custom_types.iter().enumerate() {
            let mut job = create_test_job(&format!("custom_type_{}", i));
            job.job_type = JobType::Custom(type_name.to_string());

            manager.submit_job(job).unwrap();

            let retrieved = manager
                .get_job(&format!("custom_type_{}", i))
                .unwrap()
                .unwrap();
            match retrieved.job_type {
                JobType::Custom(name) => assert_eq!(&name, type_name),
                _ => panic!("Expected custom job type"),
            }
        }
    }
}
