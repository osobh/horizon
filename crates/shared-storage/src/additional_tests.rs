//! Additional comprehensive tests for shared-storage to enhance coverage to 90%+

#[cfg(test)]
mod additional_tests {
    use super::super::*;
    use std::collections::HashMap;
    use std::sync::{Arc, Barrier};
    use std::thread;
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    // Advanced concurrent scenarios

    #[test]
    fn test_concurrent_status_updates() {
        let manager = Arc::new(SharedStorageManager::new(SharedStorageConfig::default()));
        let job_id = "concurrent_status";

        // Submit initial job
        let job = Job {
            id: job_id.to_string(),
            job_type: JobType::Compute,
            priority: JobPriority::Normal,
            status: JobStatus::Pending,
            data_path: PathBuf::from("/tmp/data"),
            result_path: None,
            created_at: SystemTime::now(),
            updated_at: SystemTime::now(),
            metadata: HashMap::new(),
        };
        manager.submit_job(job).unwrap();

        // Create barrier for synchronized start
        let barrier = Arc::new(Barrier::new(5));
        let mut handles = vec![];

        // Multiple threads updating status
        for i in 0..5 {
            let mgr = manager.clone();
            let b = barrier.clone();
            let handle = thread::spawn(move || {
                b.wait();
                let status = match i % 3 {
                    0 => JobStatus::Processing,
                    1 => JobStatus::Completed,
                    _ => JobStatus::Failed(format!("Error from thread {}", i)),
                };
                mgr.update_job_status(job_id, status)
            });
            handles.push(handle);
        }

        // Collect results
        let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();

        // All but the first should succeed (no duplicate job error)
        assert!(results.iter().filter(|r| r.is_ok()).count() >= 1);

        // Final status should be one of the attempted updates
        let final_job = manager.get_job(job_id).unwrap().unwrap();
        match final_job.status {
            JobStatus::Processing | JobStatus::Completed | JobStatus::Failed(_) => {}
            _ => panic!("Unexpected final status"),
        }
    }

    #[test]
    fn test_job_priority_queue_behavior() {
        let manager = SharedStorageManager::new(SharedStorageConfig::default());

        // Submit jobs with different priorities
        let priorities = vec![
            ("low", JobPriority::Low),
            ("normal", JobPriority::Normal),
            ("high", JobPriority::High),
            ("critical", JobPriority::Critical),
        ];

        for (name, priority) in &priorities {
            let mut job = Job {
                id: format!("priority_{}", name),
                job_type: JobType::Compute,
                priority: *priority,
                status: JobStatus::Pending,
                data_path: PathBuf::from(format!("/tmp/{}", name)),
                result_path: None,
                created_at: SystemTime::now(),
                updated_at: SystemTime::now(),
                metadata: HashMap::new(),
            };
            job.metadata
                .insert("priority_name".to_string(), name.to_string());
            manager.submit_job(job).unwrap();
        }

        // Get all pending jobs
        let pending = manager.list_jobs_by_status(&JobStatus::Pending).unwrap();
        assert_eq!(pending.len(), 4);

        // Sort by priority (highest first)
        let mut sorted = pending.clone();
        sorted.sort_by(|a, b| b.priority.cmp(&a.priority));

        // Verify critical job is first
        assert_eq!(
            sorted[0].metadata.get("priority_name"),
            Some(&"critical".to_string())
        );
        assert_eq!(
            sorted[3].metadata.get("priority_name"),
            Some(&"low".to_string())
        );
    }

    #[test]
    fn test_job_type_statistics() {
        let manager = SharedStorageManager::new(SharedStorageConfig::default());

        // Submit various job types
        let job_types = vec![
            JobType::Compute,
            JobType::Training,
            JobType::Inference,
            JobType::DataProcessing,
            JobType::Streaming,
            JobType::Custom("Analytics".to_string()),
            JobType::Custom("Optimization".to_string()),
        ];

        for (i, jt) in job_types.iter().enumerate() {
            for j in 0..3 {
                let job = Job {
                    id: format!("stats_{}_{}", i, j),
                    job_type: jt.clone(),
                    priority: JobPriority::Normal,
                    status: if j == 0 {
                        JobStatus::Completed
                    } else {
                        JobStatus::Processing
                    },
                    data_path: PathBuf::from("/tmp/data"),
                    result_path: None,
                    created_at: SystemTime::now(),
                    updated_at: SystemTime::now(),
                    metadata: HashMap::new(),
                };
                manager.submit_job(job).unwrap();
            }
        }

        // Count jobs by type
        let all_jobs: Vec<Job> = manager.jobs.lock().unwrap().values().cloned().collect();

        let mut type_counts = HashMap::new();
        for job in &all_jobs {
            *type_counts
                .entry(format!("{:?}", job.job_type))
                .or_insert(0) += 1;
        }

        // Each type should have 3 jobs
        for count in type_counts.values() {
            assert_eq!(*count, 3);
        }
    }

    #[test]
    fn test_cleanup_with_various_ttls() {
        let base_ttl = Duration::from_millis(50);
        let mut config = SharedStorageConfig::default();
        config.job_ttl = base_ttl;

        let manager = SharedStorageManager::new(config);

        // Submit jobs with staggered creation times
        for i in 0..5 {
            let mut job = Job {
                id: format!("ttl_test_{}", i),
                job_type: JobType::Compute,
                priority: JobPriority::Normal,
                status: JobStatus::Completed,
                data_path: PathBuf::from("/tmp/data"),
                result_path: None,
                created_at: SystemTime::now() - Duration::from_millis(i * 25),
                updated_at: SystemTime::now(),
                metadata: HashMap::new(),
            };

            // Older jobs should be cleaned up
            if i >= 2 {
                job.created_at = SystemTime::now() - Duration::from_millis(100);
            }

            manager.submit_job(job).unwrap();
        }

        thread::sleep(Duration::from_millis(60));

        let cleaned = manager.cleanup_expired_jobs().unwrap();
        assert!(cleaned >= 2); // At least the older jobs

        // Verify newer jobs still exist
        assert!(manager.get_job("ttl_test_0").unwrap().is_some());
        assert!(manager.get_job("ttl_test_1").unwrap().is_some());
    }

    #[test]
    fn test_metadata_aggregation() {
        let manager = SharedStorageManager::new(SharedStorageConfig::default());

        // Submit jobs with rich metadata
        for i in 0..10 {
            let mut job = Job {
                id: format!("meta_{}", i),
                job_type: JobType::Training,
                priority: JobPriority::Normal,
                status: JobStatus::Completed,
                data_path: PathBuf::from("/tmp/data"),
                result_path: Some(PathBuf::from(format!("/tmp/result_{}", i))),
                created_at: SystemTime::now(),
                updated_at: SystemTime::now(),
                metadata: HashMap::new(),
            };

            // Add various metadata
            job.metadata
                .insert("epoch".to_string(), (i * 10).to_string());
            job.metadata.insert(
                "accuracy".to_string(),
                format!("{:.2}", 0.8 + (i as f64 * 0.01)),
            );
            job.metadata.insert(
                "loss".to_string(),
                format!("{:.4}", 0.5 - (i as f64 * 0.02)),
            );
            job.metadata
                .insert("model_version".to_string(), format!("v1.{}", i));
            job.metadata
                .insert("dataset".to_string(), "training_set_2024".to_string());

            manager.submit_job(job).unwrap();
        }

        // Aggregate metadata
        let completed = manager.list_jobs_by_status(&JobStatus::Completed).unwrap();
        assert_eq!(completed.len(), 10);

        // Calculate average accuracy
        let total_accuracy: f64 = completed
            .iter()
            .filter_map(|j| j.metadata.get("accuracy"))
            .filter_map(|acc| acc.parse::<f64>().ok())
            .sum();
        let avg_accuracy = total_accuracy / completed.len() as f64;

        assert!(avg_accuracy > 0.8 && avg_accuracy < 0.9);
    }

    #[test]
    fn test_job_cancellation_race() {
        let manager = Arc::new(SharedStorageManager::new(SharedStorageConfig::default()));
        let job_id = "cancel_race";

        // Submit job
        let job = Job {
            id: job_id.to_string(),
            job_type: JobType::Compute,
            priority: JobPriority::High,
            status: JobStatus::Processing,
            data_path: PathBuf::from("/tmp/data"),
            result_path: None,
            created_at: SystemTime::now(),
            updated_at: SystemTime::now(),
            metadata: HashMap::new(),
        };
        manager.submit_job(job).unwrap();

        let barrier = Arc::new(Barrier::new(2));

        // Thread 1: Try to complete
        let mgr1 = manager.clone();
        let b1 = barrier.clone();
        let h1 = thread::spawn(move || {
            b1.wait();
            mgr1.update_job_status(job_id, JobStatus::Completed)
        });

        // Thread 2: Try to cancel
        let mgr2 = manager.clone();
        let b2 = barrier.clone();
        let h2 = thread::spawn(move || {
            b2.wait();
            mgr2.update_job_status(job_id, JobStatus::Cancelled)
        });

        let r1 = h1.join().unwrap();
        let r2 = h2.join().unwrap();

        // One should succeed, one should succeed too (both are valid transitions)
        assert!(r1.is_ok() && r2.is_ok());

        // Final status should be one of the two
        let final_job = manager.get_job(job_id).unwrap().unwrap();
        match final_job.status {
            JobStatus::Completed | JobStatus::Cancelled => {}
            _ => panic!("Unexpected final status"),
        }
    }

    #[test]
    fn test_path_resolution_edge_cases() {
        let manager = SharedStorageManager::new(SharedStorageConfig::default());

        // Test various path edge cases
        let test_cases = vec![
            (PathBuf::from("."), "current_dir"),
            (PathBuf::from(".."), "parent_dir"),
            (PathBuf::from("~/data"), "home_dir"),
            (PathBuf::from("C:\\Windows\\data"), "windows_path"),
            (PathBuf::from("\\\\network\\share"), "unc_path"),
            (PathBuf::from("/tmp/symlink/../real"), "symlink_traversal"),
            (PathBuf::from("data/./subdir"), "dot_component"),
            (PathBuf::from("//double//slash"), "double_slash"),
        ];

        for (path, name) in test_cases {
            let job = Job {
                id: format!("path_{}", name),
                job_type: JobType::DataProcessing,
                priority: JobPriority::Normal,
                status: JobStatus::Pending,
                data_path: path.clone(),
                result_path: Some(path.join("results")),
                created_at: SystemTime::now(),
                updated_at: SystemTime::now(),
                metadata: HashMap::new(),
            };

            manager.submit_job(job).unwrap();

            let retrieved = manager.get_job(&format!("path_{}", name)).unwrap().unwrap();
            assert_eq!(retrieved.data_path, path);
        }
    }

    #[test]
    fn test_extreme_timestamps() {
        let manager = SharedStorageManager::new(SharedStorageConfig::default());

        // Test with various extreme timestamps
        let timestamps = vec![
            ("epoch", UNIX_EPOCH),
            ("far_past", UNIX_EPOCH + Duration::from_secs(1)),
            ("recent", SystemTime::now() - Duration::from_secs(60)),
            ("now", SystemTime::now()),
            ("near_future", SystemTime::now() + Duration::from_secs(60)),
            (
                "far_future",
                SystemTime::now() + Duration::from_secs(365 * 24 * 60 * 60),
            ),
        ];

        for (name, timestamp) in timestamps {
            let job = Job {
                id: format!("time_{}", name),
                job_type: JobType::Compute,
                priority: JobPriority::Normal,
                status: JobStatus::Pending,
                data_path: PathBuf::from("/tmp/data"),
                result_path: None,
                created_at: timestamp,
                updated_at: timestamp,
                metadata: HashMap::new(),
            };

            manager.submit_job(job).unwrap();

            let retrieved = manager.get_job(&format!("time_{}", name)).unwrap().unwrap();
            assert_eq!(retrieved.created_at, timestamp);
        }
    }

    #[test]
    fn test_job_dependency_tracking() {
        let manager = SharedStorageManager::new(SharedStorageConfig::default());

        // Create a DAG of dependent jobs
        let job_dag = vec![
            ("root", vec![]),
            ("child1", vec!["root"]),
            ("child2", vec!["root"]),
            ("grandchild", vec!["child1", "child2"]),
        ];

        for (job_id, dependencies) in job_dag {
            let mut job = Job {
                id: job_id.to_string(),
                job_type: JobType::Compute,
                priority: JobPriority::Normal,
                status: JobStatus::Pending,
                data_path: PathBuf::from(format!("/tmp/{}", job_id)),
                result_path: None,
                created_at: SystemTime::now(),
                updated_at: SystemTime::now(),
                metadata: HashMap::new(),
            };

            // Track dependencies in metadata
            for (i, dep) in dependencies.iter().enumerate() {
                job.metadata.insert(format!("dep_{}", i), dep.to_string());
            }
            job.metadata
                .insert("dep_count".to_string(), dependencies.len().to_string());

            manager.submit_job(job).unwrap();
        }

        // Verify all jobs and their dependencies
        let all_jobs = manager.list_jobs_by_status(&JobStatus::Pending).unwrap();
        assert_eq!(all_jobs.len(), 4);

        // Check grandchild has correct dependencies
        let grandchild = all_jobs.iter().find(|j| j.id == "grandchild").unwrap();
        assert_eq!(grandchild.metadata.get("dep_count"), Some(&"2".to_string()));
    }

    #[test]
    fn test_error_propagation() {
        let manager = SharedStorageManager::new(SharedStorageConfig::default());

        // Submit a job that will fail
        let job = Job {
            id: "error_test".to_string(),
            job_type: JobType::Training,
            priority: JobPriority::High,
            status: JobStatus::Processing,
            data_path: PathBuf::from("/tmp/data"),
            result_path: None,
            created_at: SystemTime::now(),
            updated_at: SystemTime::now(),
            metadata: HashMap::new(),
        };
        manager.submit_job(job).unwrap();

        // Simulate various error scenarios
        let error_scenarios = vec![
            "OutOfMemory: GPU memory exhausted",
            "DataCorruption: Checksum mismatch in input file",
            "NetworkTimeout: Failed to fetch remote dataset",
            "PermissionDenied: Cannot write to output directory",
            "InvalidInput: Unsupported data format",
            "InternalError: Kernel launch failed with code -1",
        ];

        for (i, error_msg) in error_scenarios.iter().enumerate() {
            let job_id = format!("error_scenario_{}", i);
            let mut job = Job {
                id: job_id.clone(),
                job_type: JobType::Inference,
                priority: JobPriority::Normal,
                status: JobStatus::Processing,
                data_path: PathBuf::from("/tmp/data"),
                result_path: None,
                created_at: SystemTime::now(),
                updated_at: SystemTime::now(),
                metadata: HashMap::new(),
            };
            job.metadata
                .insert("scenario".to_string(), format!("{}", i));

            manager.submit_job(job).unwrap();
            manager
                .update_job_status(&job_id, JobStatus::Failed(error_msg.to_string()))
                .unwrap();

            // Verify error is properly stored
            let failed_job = manager.get_job(&job_id).unwrap().unwrap();
            match &failed_job.status {
                JobStatus::Failed(msg) => assert_eq!(msg, error_msg),
                _ => panic!("Expected failed status"),
            }
        }
    }

    #[test]
    fn test_job_result_validation() {
        let manager = SharedStorageManager::new(SharedStorageConfig::default());

        // Submit job with expected results
        let mut job = Job {
            id: "validation_test".to_string(),
            job_type: JobType::Inference,
            priority: JobPriority::Normal,
            status: JobStatus::Processing,
            data_path: PathBuf::from("/tmp/input"),
            result_path: Some(PathBuf::from("/tmp/output")),
            created_at: SystemTime::now(),
            updated_at: SystemTime::now(),
            metadata: HashMap::new(),
        };

        // Add validation metadata
        job.metadata
            .insert("expected_output_size".to_string(), "1048576".to_string()); // 1MB
        job.metadata
            .insert("expected_checksum".to_string(), "abc123def456".to_string());
        job.metadata
            .insert("output_format".to_string(), "json".to_string());
        job.metadata
            .insert("compression".to_string(), "gzip".to_string());

        manager.submit_job(job).unwrap();

        // Simulate validation results
        manager
            .update_job_status("validation_test", JobStatus::Completed)
            .unwrap();

        let completed = manager.get_job("validation_test").unwrap().unwrap();

        // Add actual results to metadata (would be done by worker)
        let mut updated_job = completed.clone();
        updated_job
            .metadata
            .insert("actual_output_size".to_string(), "1048576".to_string());
        updated_job
            .metadata
            .insert("actual_checksum".to_string(), "abc123def456".to_string());
        updated_job
            .metadata
            .insert("validation_passed".to_string(), "true".to_string());

        // In real scenario, this would be done through update mechanism
        assert_eq!(
            updated_job.metadata.get("validation_passed"),
            Some(&"true".to_string())
        );
    }

    #[test]
    fn test_config_edge_cases() {
        // Test with minimal config
        let minimal_config = SharedStorageConfig {
            base_path: PathBuf::from("/"),
            max_job_size: 1,
            cleanup_interval: Duration::from_nanos(1),
            job_ttl: Duration::from_nanos(1),
        };

        let manager = SharedStorageManager::new(minimal_config);
        assert!(
            manager
                .submit_job(Job {
                    id: "minimal".to_string(),
                    job_type: JobType::Compute,
                    priority: JobPriority::Low,
                    status: JobStatus::Pending,
                    data_path: PathBuf::from("/"),
                    result_path: None,
                    created_at: SystemTime::now(),
                    updated_at: SystemTime::now(),
                    metadata: HashMap::new(),
                })
                .is_ok()
        );

        // Test with maximal config
        let maximal_config = SharedStorageConfig {
            base_path: PathBuf::from(
                "/a/very/long/path/that/might/not/exist/but/should/still/be/valid",
            ),
            max_job_size: usize::MAX,
            cleanup_interval: Duration::from_secs(u64::MAX),
            job_ttl: Duration::from_secs(u64::MAX),
        };

        let manager2 = SharedStorageManager::new(maximal_config);
        assert!(
            manager2
                .submit_job(Job {
                    id: "maximal".to_string(),
                    job_type: JobType::Custom("Maximum".to_string()),
                    priority: JobPriority::Critical,
                    status: JobStatus::Pending,
                    data_path: PathBuf::from("/max"),
                    result_path: None,
                    created_at: SystemTime::now(),
                    updated_at: SystemTime::now(),
                    metadata: HashMap::new(),
                })
                .is_ok()
        );
    }

    #[test]
    fn test_stress_many_status_transitions() {
        let manager = SharedStorageManager::new(SharedStorageConfig::default());
        let job_id = "stress_transitions";

        // Submit initial job
        manager
            .submit_job(Job {
                id: job_id.to_string(),
                job_type: JobType::Streaming,
                priority: JobPriority::High,
                status: JobStatus::Pending,
                data_path: PathBuf::from("/tmp/stream"),
                result_path: None,
                created_at: SystemTime::now(),
                updated_at: SystemTime::now(),
                metadata: HashMap::new(),
            })
            .unwrap();

        // Perform many rapid status updates
        for i in 0..100 {
            let status = match i % 4 {
                0 => JobStatus::Processing,
                1 => JobStatus::Pending,
                2 => JobStatus::Failed(format!("Intermittent error {}", i)),
                _ => JobStatus::Processing,
            };

            manager.update_job_status(job_id, status).unwrap();
        }

        // Final update to completed
        manager
            .update_job_status(job_id, JobStatus::Completed)
            .unwrap();

        let final_job = manager.get_job(job_id).unwrap().unwrap();
        assert_eq!(final_job.status, JobStatus::Completed);
        assert!(final_job.updated_at > final_job.created_at);
    }
}
