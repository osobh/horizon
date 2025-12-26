//! Edge case tests for shared storage system to enhance coverage to 90%

#[cfg(test)]
mod edge_case_tests {
    use super::super::*;
    use std::sync::{Arc, Barrier};
    use std::thread;
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    // Test edge cases for SharedStorageConfig
    #[test]
    fn test_config_with_extreme_values() {
        let mut config = SharedStorageConfig::default();

        // Test with zero values
        config.max_job_size = 0;
        config.cleanup_interval = Duration::from_secs(0);
        config.job_ttl = Duration::from_secs(0);

        let manager = SharedStorageManager::new(config.clone());
        assert!(manager.submit_job(create_test_job("test")).is_ok());

        // Test with max values
        config.max_job_size = usize::MAX;
        config.cleanup_interval = Duration::from_secs(u64::MAX);
        config.job_ttl = Duration::from_secs(u64::MAX);

        let manager = SharedStorageManager::new(config);
        assert!(manager.submit_job(create_test_job("test2")).is_ok());
    }

    #[test]
    fn test_config_with_non_existent_path() {
        let mut config = SharedStorageConfig::default();
        config.base_path = PathBuf::from("/this/path/definitely/does/not/exist/12345");

        let manager = SharedStorageManager::new(config);
        // Should still work as we're not actually using the filesystem
        assert!(manager.submit_job(create_test_job("test")).is_ok());
    }

    // Test mutex poisoning scenarios
    #[test]
    fn test_mutex_poisoning_recovery() {
        let manager = Arc::new(SharedStorageManager::new(SharedStorageConfig::default()));
        let manager_clone = manager.clone();

        // Spawn a thread that will panic while holding the lock
        let handle = thread::spawn(move || {
            let _jobs = manager_clone.jobs.lock().unwrap();
            panic!("Intentional panic to poison mutex");
        });

        // Wait for panic
        let _ = handle.join();

        // Try to use the manager after mutex poisoning
        let result = manager.submit_job(create_test_job("after_poison"));
        // Should return an error due to poisoned mutex
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("poisoned"));
    }

    // Test concurrent operations with barriers
    #[test]
    fn test_concurrent_operations_with_barriers() {
        let manager = Arc::new(SharedStorageManager::new(SharedStorageConfig::default()));
        let barrier = Arc::new(Barrier::new(10));
        let mut handles = vec![];

        for i in 0..10 {
            let mgr = manager.clone();
            let bar = barrier.clone();

            let handle = thread::spawn(move || {
                // Wait for all threads to be ready
                bar.wait();

                // Perform mixed operations
                let job_id = format!("concurrent_{}", i);
                let job = create_test_job(&job_id);

                // Submit
                mgr.submit_job(job).unwrap();

                // Update
                mgr.update_job_status(&job_id, JobStatus::Processing)
                    .unwrap();

                // Get
                let _ = mgr.get_job(&job_id).unwrap();

                // List
                let _ = mgr.list_jobs_by_status(&JobStatus::Processing).unwrap();

                // Update again
                mgr.update_job_status(&job_id, JobStatus::Completed)
                    .unwrap();
            });

            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // Verify all jobs exist
        for i in 0..10 {
            let job = manager.get_job(&format!("concurrent_{}", i)).unwrap();
            assert!(job.is_some());
            assert_eq!(job.unwrap().status, JobStatus::Completed);
        }
    }

    // Test job with extreme timestamps
    #[test]
    fn test_job_with_extreme_timestamps() {
        let manager = SharedStorageManager::new(SharedStorageConfig::default());

        // Test with UNIX epoch
        let mut job = create_test_job("epoch");
        job.created_at = UNIX_EPOCH;
        job.updated_at = UNIX_EPOCH;
        assert!(manager.submit_job(job).is_ok());

        // Test with far future
        let mut job = create_test_job("future");
        job.created_at = SystemTime::now() + Duration::from_secs(365 * 24 * 3600 * 100); // 100 years
        job.updated_at = job.created_at;
        assert!(manager.submit_job(job).is_ok());

        // Cleanup should handle these gracefully
        let removed = manager.cleanup_expired_jobs().unwrap();
        assert_eq!(removed, 0); // Future jobs shouldn't be cleaned
    }

    // Test job status edge cases
    #[test]
    fn test_job_status_failed_with_empty_message() {
        let manager = SharedStorageManager::new(SharedStorageConfig::default());
        let job = create_test_job("empty_fail");

        manager.submit_job(job).unwrap();
        manager
            .update_job_status("empty_fail", JobStatus::Failed(String::new()))
            .unwrap();

        let retrieved = manager.get_job("empty_fail").unwrap().unwrap();
        match retrieved.status {
            JobStatus::Failed(msg) => assert_eq!(msg, ""),
            _ => panic!("Expected failed status"),
        }
    }

    #[test]
    fn test_job_status_failed_with_very_long_message() {
        let manager = SharedStorageManager::new(SharedStorageConfig::default());
        let job = create_test_job("long_fail");

        let long_message = "Error: ".repeat(10000); // Very long error message

        manager.submit_job(job).unwrap();
        manager
            .update_job_status("long_fail", JobStatus::Failed(long_message.clone()))
            .unwrap();

        let retrieved = manager.get_job("long_fail").unwrap().unwrap();
        match retrieved.status {
            JobStatus::Failed(msg) => assert_eq!(msg, long_message),
            _ => panic!("Expected failed status"),
        }
    }

    // Test duplicate job submission race condition
    #[test]
    fn test_duplicate_job_submission_race() {
        let manager = Arc::new(SharedStorageManager::new(SharedStorageConfig::default()));
        let barrier = Arc::new(Barrier::new(2));

        let mgr1 = manager.clone();
        let bar1 = barrier.clone();
        let handle1 = thread::spawn(move || {
            bar1.wait();
            mgr1.submit_job(create_test_job("duplicate"))
        });

        let mgr2 = manager.clone();
        let bar2 = barrier.clone();
        let handle2 = thread::spawn(move || {
            bar2.wait();
            mgr2.submit_job(create_test_job("duplicate"))
        });

        let result1 = handle1.join().unwrap();
        let result2 = handle2.join().unwrap();

        // One should succeed, one should fail
        assert!(result1.is_ok() != result2.is_ok());
    }

    // Test listing with no jobs
    #[test]
    fn test_list_jobs_empty() {
        let manager = SharedStorageManager::new(SharedStorageConfig::default());

        let pending = manager.list_jobs_by_status(&JobStatus::Pending).unwrap();
        assert_eq!(pending.len(), 0);

        let completed = manager.list_jobs_by_status(&JobStatus::Completed).unwrap();
        assert_eq!(completed.len(), 0);
    }

    // Test cleanup with all statuses
    #[test]
    fn test_cleanup_respects_all_statuses() {
        let mut config = SharedStorageConfig::default();
        config.job_ttl = Duration::from_millis(1); // Very short TTL
        let manager = SharedStorageManager::new(config);

        // Create jobs with all statuses
        let statuses = vec![
            ("pending", JobStatus::Pending),
            ("processing", JobStatus::Processing),
            ("completed", JobStatus::Completed),
            ("failed", JobStatus::Failed("test".to_string())),
            ("cancelled", JobStatus::Cancelled),
        ];

        for (id, status) in &statuses {
            let mut job = create_test_job(id);
            job.status = status.clone();
            manager.submit_job(job).unwrap();
        }

        // Wait for TTL
        thread::sleep(Duration::from_millis(10));

        let removed = manager.cleanup_expired_jobs().unwrap();

        // Only completed and failed should be removed
        assert_eq!(removed, 2);

        // Verify the right jobs were kept
        assert!(manager.get_job("pending").unwrap().is_some());
        assert!(manager.get_job("processing").unwrap().is_some());
        assert!(manager.get_job("completed").unwrap().is_none());
        assert!(manager.get_job("failed").unwrap().is_none());
        assert!(manager.get_job("cancelled").unwrap().is_some());
    }

    // Test updating non-existent job
    #[test]
    fn test_update_nonexistent_job() {
        let manager = SharedStorageManager::new(SharedStorageConfig::default());

        let result = manager.update_job_status("does_not_exist", JobStatus::Completed);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not found"));
    }

    // Test job with all metadata types
    #[test]
    fn test_job_metadata_completeness() {
        let manager = SharedStorageManager::new(SharedStorageConfig::default());
        let mut job = create_test_job("metadata_test");

        // Add various metadata types
        job.metadata
            .insert("bool_true".to_string(), "true".to_string());
        job.metadata
            .insert("bool_false".to_string(), "false".to_string());
        job.metadata
            .insert("int_positive".to_string(), "42".to_string());
        job.metadata
            .insert("int_negative".to_string(), "-42".to_string());
        job.metadata.insert("int_zero".to_string(), "0".to_string());
        job.metadata
            .insert("float".to_string(), "3.14159".to_string());
        job.metadata
            .insert("scientific".to_string(), "1.23e-10".to_string());
        job.metadata
            .insert("hex".to_string(), "0xDEADBEEF".to_string());
        job.metadata
            .insert("binary".to_string(), "0b101010".to_string());
        job.metadata
            .insert("null_char".to_string(), "\0".to_string());

        manager.submit_job(job.clone()).unwrap();

        let retrieved = manager.get_job("metadata_test").unwrap().unwrap();
        assert_eq!(retrieved.metadata.len(), 10);

        // Verify all values are preserved exactly
        for (key, value) in &job.metadata {
            assert_eq!(retrieved.metadata.get(key), Some(value));
        }
    }

    // Test rapid job status transitions
    #[test]
    fn test_rapid_status_transitions() {
        let manager = Arc::new(SharedStorageManager::new(SharedStorageConfig::default()));
        let job = create_test_job("rapid");
        manager.submit_job(job).unwrap();

        let mgr = manager.clone();
        let handle = thread::spawn(move || {
            for _ in 0..100 {
                mgr.update_job_status("rapid", JobStatus::Processing).ok();
                mgr.update_job_status("rapid", JobStatus::Pending).ok();
            }
        });

        // Concurrent reads while updates happen
        for _ in 0..100 {
            let _ = manager.get_job("rapid");
            let _ = manager.list_jobs_by_status(&JobStatus::Processing);
        }

        handle.join().unwrap();

        // Job should still exist and be valid
        assert!(manager.get_job("rapid").unwrap().is_some());
    }

    // Test PathBuf edge cases
    #[test]
    fn test_pathbuf_edge_cases() {
        let manager = SharedStorageManager::new(SharedStorageConfig::default());

        // Test with PathBuf containing only dots
        let mut job = create_test_job("dots");
        job.data_path = PathBuf::from("...");
        job.result_path = Some(PathBuf::from("...."));
        assert!(manager.submit_job(job).is_ok());

        // Test with very nested path
        let mut job = create_test_job("nested");
        let mut nested_path = PathBuf::new();
        for i in 0..100 {
            nested_path.push(format!("level{}", i));
        }
        job.data_path = nested_path;
        assert!(manager.submit_job(job).is_ok());

        // Test with path containing special filesystem characters
        let mut job = create_test_job("special");
        job.data_path = PathBuf::from("/tmp/*?[]{}<>|");
        assert!(manager.submit_job(job).is_ok());
    }

    // Test SystemTime edge cases in cleanup
    #[test]
    fn test_cleanup_with_systemtime_overflow() {
        let mut config = SharedStorageConfig::default();
        // Set TTL to max duration
        config.job_ttl = Duration::from_secs(u64::MAX);
        let manager = SharedStorageManager::new(config);

        let mut job = create_test_job("overflow");
        job.status = JobStatus::Completed;
        // Set created_at to a time that would cause overflow when adding TTL
        job.created_at = SystemTime::now() - Duration::from_secs(1);

        manager.submit_job(job).unwrap();

        // Cleanup should handle overflow gracefully
        let result = manager.cleanup_expired_jobs();
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 0); // Should not remove due to overflow
    }

    // Helper function
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
}
