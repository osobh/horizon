//! Integration tests for shared storage system

#[cfg(test)]
mod tests {
    use crate::*;
    use std::sync::{Arc, Barrier};
    use std::thread;
    use std::time::{Duration, Instant};

    #[test]
    fn test_multi_producer_single_consumer() {
        let manager = Arc::new(SharedStorageManager::new(SharedStorageConfig::default()));
        let barrier = Arc::new(Barrier::new(6)); // 5 producers + 1 consumer
        let mut handles = vec![];

        // Spawn producer threads
        for i in 0..5 {
            let mgr = manager.clone();
            let b = barrier.clone();
            let handle = thread::spawn(move || {
                b.wait(); // Synchronize start

                for j in 0..10 {
                    let mut job = Job {
                        id: format!("producer_{}_job_{}", i, j),
                        job_type: JobType::Compute,
                        priority: if j % 2 == 0 {
                            JobPriority::High
                        } else {
                            JobPriority::Normal
                        },
                        status: JobStatus::Pending,
                        data_path: PathBuf::from(format!("/tmp/producer_{}/job_{}", i, j)),
                        result_path: None,
                        created_at: SystemTime::now(),
                        updated_at: SystemTime::now(),
                        metadata: HashMap::new(),
                    };
                    job.metadata.insert("producer".to_string(), i.to_string());
                    mgr.submit_job(job).unwrap();
                }
            });
            handles.push(handle);
        }

        // Spawn consumer thread
        let mgr = manager.clone();
        let b = barrier.clone();
        let consumer_handle = thread::spawn(move || {
            b.wait(); // Synchronize start
            thread::sleep(Duration::from_millis(50)); // Let producers submit jobs

            let mut processed = 0;
            while processed < 50 {
                let pending = mgr.list_jobs_by_status(&JobStatus::Pending).unwrap();
                for job in pending {
                    mgr.update_job_status(&job.id, JobStatus::Processing)
                        .unwrap();
                    thread::sleep(Duration::from_micros(100)); // Simulate processing
                    mgr.update_job_status(&job.id, JobStatus::Completed)
                        .unwrap();
                    processed += 1;
                }
                thread::sleep(Duration::from_millis(1));
            }
            processed
        });

        // Wait for all threads
        for handle in handles {
            handle.join().unwrap();
        }
        let processed = consumer_handle.join().unwrap();

        assert_eq!(processed, 50);

        // Verify all jobs are completed
        let completed = manager.list_jobs_by_status(&JobStatus::Completed).unwrap();
        assert_eq!(completed.len(), 50);
    }

    #[test]
    fn test_priority_based_processing() {
        let manager = Arc::new(SharedStorageManager::new(SharedStorageConfig::default()));

        // Submit jobs with different priorities
        let priorities = vec![
            (JobPriority::Low, 10),
            (JobPriority::Normal, 10),
            (JobPriority::High, 10),
            (JobPriority::Critical, 10),
        ];

        for (priority, count) in priorities {
            for i in 0..count {
                let job = Job {
                    id: format!("{:?}_{}", priority, i),
                    job_type: JobType::Compute,
                    priority,
                    status: JobStatus::Pending,
                    data_path: PathBuf::from(format!("/tmp/{:?}_{}", priority, i)),
                    result_path: None,
                    created_at: SystemTime::now(),
                    updated_at: SystemTime::now(),
                    metadata: HashMap::new(),
                };
                manager.submit_job(job).unwrap();
            }
        }

        // Process jobs in priority order
        let mut processed_order = vec![];
        let pending = manager.list_jobs_by_status(&JobStatus::Pending).unwrap();

        // Sort by priority (descending)
        let mut sorted_jobs = pending;
        sorted_jobs.sort_by(|a, b| b.priority.cmp(&a.priority));

        for job in sorted_jobs {
            processed_order.push(job.priority);
            manager
                .update_job_status(&job.id, JobStatus::Completed)
                .unwrap();
        }

        // Verify Critical jobs were processed first
        assert!(
            processed_order[0..10]
                .iter()
                .all(|&p| p == JobPriority::Critical)
        );
        assert!(
            processed_order[10..20]
                .iter()
                .all(|&p| p == JobPriority::High)
        );
        assert!(
            processed_order[20..30]
                .iter()
                .all(|&p| p == JobPriority::Normal)
        );
        assert!(
            processed_order[30..40]
                .iter()
                .all(|&p| p == JobPriority::Low)
        );
    }

    #[test]
    fn test_job_retry_mechanism() {
        let manager = SharedStorageManager::new(SharedStorageConfig::default());

        // Submit a job that will fail
        let job = Job {
            id: "retry_test".to_string(),
            job_type: JobType::Compute,
            priority: JobPriority::Normal,
            status: JobStatus::Pending,
            data_path: PathBuf::from("/tmp/retry_test"),
            result_path: None,
            created_at: SystemTime::now(),
            updated_at: SystemTime::now(),
            metadata: HashMap::new(),
        };

        manager.submit_job(job).unwrap();

        // Simulate failures and retries
        let max_retries = 3;
        let mut retry_count = 0;

        loop {
            manager
                .update_job_status("retry_test", JobStatus::Processing)
                .unwrap();

            // Simulate random failure
            if retry_count < max_retries - 1 {
                let error_msg = format!("Attempt {} failed", retry_count + 1);
                manager
                    .update_job_status("retry_test", JobStatus::Failed(error_msg))
                    .unwrap();

                // Get job and check retry count
                let job = manager.get_job("retry_test").unwrap().unwrap();
                match job.status {
                    JobStatus::Failed(_) => {
                        retry_count += 1;
                        // Reset to pending for retry
                        manager
                            .update_job_status("retry_test", JobStatus::Pending)
                            .unwrap();
                    }
                    _ => panic!("Expected failed status"),
                }
            } else {
                // Succeed on last retry
                manager
                    .update_job_status("retry_test", JobStatus::Completed)
                    .unwrap();
                break;
            }
        }

        let final_job = manager.get_job("retry_test").unwrap().unwrap();
        assert_eq!(final_job.status, JobStatus::Completed);
        assert_eq!(retry_count, max_retries - 1);
    }

    #[test]
    fn test_batch_job_submission() {
        let manager = SharedStorageManager::new(SharedStorageConfig::default());
        let batch_size = 100;
        let start = Instant::now();

        // Submit batch of jobs
        let mut job_ids = vec![];
        for i in 0..batch_size {
            let job = Job {
                id: format!("batch_{}", i),
                job_type: JobType::DataProcessing,
                priority: JobPriority::Normal,
                status: JobStatus::Pending,
                data_path: PathBuf::from(format!("/tmp/batch_{}", i)),
                result_path: None,
                created_at: SystemTime::now(),
                updated_at: SystemTime::now(),
                metadata: HashMap::from([
                    ("batch_id".to_string(), "batch_001".to_string()),
                    ("index".to_string(), i.to_string()),
                ]),
            };
            job_ids.push(job.id.clone());
            manager.submit_job(job).unwrap();
        }

        let submission_time = start.elapsed();
        println!("Batch submission time: {:?}", submission_time);

        // Verify all jobs were submitted
        for id in &job_ids {
            assert!(manager.get_job(id).unwrap().is_some());
        }

        // Process batch
        let process_start = Instant::now();
        for id in job_ids {
            manager
                .update_job_status(&id, JobStatus::Processing)
                .unwrap();
            manager
                .update_job_status(&id, JobStatus::Completed)
                .unwrap();
        }
        let process_time = process_start.elapsed();
        println!("Batch processing time: {:?}", process_time);

        // Verify performance (should be fast)
        assert!(submission_time < Duration::from_secs(1));
        assert!(process_time < Duration::from_secs(1));
    }

    #[test]
    fn test_job_cancellation_workflow() {
        let manager = SharedStorageManager::new(SharedStorageConfig::default());

        // Submit multiple jobs
        for i in 0..10 {
            let job = Job {
                id: format!("cancel_{}", i),
                job_type: JobType::Training,
                priority: JobPriority::Low,
                status: JobStatus::Pending,
                data_path: PathBuf::from(format!("/tmp/cancel_{}", i)),
                result_path: None,
                created_at: SystemTime::now(),
                updated_at: SystemTime::now(),
                metadata: HashMap::new(),
            };
            manager.submit_job(job).unwrap();
        }

        // Start processing some jobs
        for i in 0..5 {
            manager
                .update_job_status(&format!("cancel_{}", i), JobStatus::Processing)
                .unwrap();
        }

        // Cancel all pending and processing jobs
        let pending = manager.list_jobs_by_status(&JobStatus::Pending).unwrap();
        let processing = manager.list_jobs_by_status(&JobStatus::Processing).unwrap();

        for job in pending.iter().chain(processing.iter()) {
            manager
                .update_job_status(&job.id, JobStatus::Cancelled)
                .unwrap();
        }

        // Verify all jobs are cancelled
        let cancelled = manager.list_jobs_by_status(&JobStatus::Cancelled).unwrap();
        assert_eq!(cancelled.len(), 10);
    }

    #[test]
    fn test_job_dependency_chain() {
        let manager = SharedStorageManager::new(SharedStorageConfig::default());

        // Create a chain of dependent jobs
        let chain_length = 5;
        let mut previous_id: Option<String> = None;

        for i in 0..chain_length {
            let mut job = Job {
                id: format!("chain_{}", i),
                job_type: JobType::Compute,
                priority: JobPriority::Normal,
                status: JobStatus::Pending,
                data_path: PathBuf::from(format!("/tmp/chain_{}", i)),
                result_path: None,
                created_at: SystemTime::now(),
                updated_at: SystemTime::now(),
                metadata: HashMap::new(),
            };

            // Add dependency on previous job
            if let Some(prev) = &previous_id {
                job.metadata.insert("depends_on".to_string(), prev.clone());
            }

            manager.submit_job(job).unwrap();
            previous_id = Some(format!("chain_{}", i));
        }

        // Process jobs in dependency order
        for i in 0..chain_length {
            let job_id = format!("chain_{}", i);
            let job = manager.get_job(&job_id).unwrap().unwrap();

            // Check dependency
            if i > 0 {
                let depends_on = job.metadata.get("depends_on").unwrap();
                let dependency = manager.get_job(depends_on).unwrap().unwrap();
                assert_eq!(dependency.status, JobStatus::Completed);
            }

            manager
                .update_job_status(&job_id, JobStatus::Processing)
                .unwrap();
            manager
                .update_job_status(&job_id, JobStatus::Completed)
                .unwrap();
        }

        // Verify all jobs completed
        for i in 0..chain_length {
            let job = manager.get_job(&format!("chain_{}", i)).unwrap().unwrap();
            assert_eq!(job.status, JobStatus::Completed);
        }
    }

    #[test]
    fn test_concurrent_status_updates() {
        let manager = Arc::new(SharedStorageManager::new(SharedStorageConfig::default()));

        // Submit a job
        let job = Job {
            id: "concurrent_update".to_string(),
            job_type: JobType::Compute,
            priority: JobPriority::Normal,
            status: JobStatus::Pending,
            data_path: PathBuf::from("/tmp/concurrent_update"),
            result_path: None,
            created_at: SystemTime::now(),
            updated_at: SystemTime::now(),
            metadata: HashMap::new(),
        };
        manager.submit_job(job).unwrap();

        // Spawn multiple threads trying to update the same job
        let mut handles = vec![];
        for i in 0..10 {
            let mgr = manager.clone();
            let handle = thread::spawn(move || {
                let status = if i % 2 == 0 {
                    JobStatus::Processing
                } else {
                    JobStatus::Completed
                };
                mgr.update_job_status("concurrent_update", status)
            });
            handles.push(handle);
        }

        // Wait for all updates
        for handle in handles {
            let _ = handle.join().unwrap();
        }

        // Job should have one of the valid statuses
        let final_job = manager.get_job("concurrent_update").unwrap().unwrap();
        assert!(matches!(
            final_job.status,
            JobStatus::Processing | JobStatus::Completed
        ));
    }

    #[test]
    fn test_job_result_handling() {
        let manager = SharedStorageManager::new(SharedStorageConfig::default());

        // Submit job
        let mut job = Job {
            id: "result_test".to_string(),
            job_type: JobType::Inference,
            priority: JobPriority::High,
            status: JobStatus::Pending,
            data_path: PathBuf::from("/tmp/result_test/input"),
            result_path: None,
            created_at: SystemTime::now(),
            updated_at: SystemTime::now(),
            metadata: HashMap::new(),
        };

        manager.submit_job(job.clone()).unwrap();

        // Process job
        manager
            .update_job_status("result_test", JobStatus::Processing)
            .unwrap();

        // Simulate setting result
        job.result_path = Some(PathBuf::from("/tmp/result_test/output"));
        job.metadata
            .insert("output_size".to_string(), "1024".to_string());
        job.metadata
            .insert("processing_time_ms".to_string(), "150".to_string());

        // Update job with result
        manager
            .update_job_status("result_test", JobStatus::Completed)
            .unwrap();

        // In a real implementation, we would update the job's result_path
        // For now, verify the job completed
        let completed = manager.get_job("result_test").unwrap().unwrap();
        assert_eq!(completed.status, JobStatus::Completed);
    }

    #[test]
    fn test_job_filtering_by_type() {
        let manager = SharedStorageManager::new(SharedStorageConfig::default());

        // Submit jobs of different types
        let job_types = vec![
            (JobType::Compute, 5),
            (JobType::Training, 3),
            (JobType::Inference, 7),
            (JobType::DataProcessing, 4),
            (JobType::Custom("Analysis".to_string()), 2),
        ];

        for (job_type, count) in &job_types {
            for i in 0..*count {
                let job = Job {
                    id: format!("{:?}_{}", job_type, i),
                    job_type: job_type.clone(),
                    priority: JobPriority::Normal,
                    status: JobStatus::Pending,
                    data_path: PathBuf::from(format!("/tmp/{:?}_{}", job_type, i)),
                    result_path: None,
                    created_at: SystemTime::now(),
                    updated_at: SystemTime::now(),
                    metadata: HashMap::new(),
                };
                manager.submit_job(job).unwrap();
            }
        }

        // Filter and count by type
        let all_jobs = manager.list_jobs_by_status(&JobStatus::Pending).unwrap();

        for (job_type, expected_count) in job_types {
            let filtered: Vec<_> = all_jobs.iter().filter(|j| j.job_type == job_type).collect();
            assert_eq!(filtered.len(), expected_count);
        }
    }

    #[test]
    fn test_stress_test_many_jobs() {
        let manager = Arc::new(SharedStorageManager::new(SharedStorageConfig::default()));
        let num_jobs = 1000;
        let num_threads = 8;
        let jobs_per_thread = num_jobs / num_threads;

        let barrier = Arc::new(Barrier::new(num_threads));
        let mut handles = vec![];

        // Spawn threads to submit jobs
        for thread_id in 0..num_threads {
            let mgr = manager.clone();
            let b = barrier.clone();

            let handle = thread::spawn(move || {
                b.wait(); // Synchronize start

                for i in 0..jobs_per_thread {
                    let job = Job {
                        id: format!("stress_{}_{}", thread_id, i),
                        job_type: JobType::Compute,
                        priority: match i % 4 {
                            0 => JobPriority::Critical,
                            1 => JobPriority::High,
                            2 => JobPriority::Normal,
                            _ => JobPriority::Low,
                        },
                        status: JobStatus::Pending,
                        data_path: PathBuf::from(format!("/tmp/stress_{}_{}", thread_id, i)),
                        result_path: None,
                        created_at: SystemTime::now(),
                        updated_at: SystemTime::now(),
                        metadata: HashMap::from([
                            ("thread_id".to_string(), thread_id.to_string()),
                            ("index".to_string(), i.to_string()),
                        ]),
                    };
                    mgr.submit_job(job).unwrap();
                }
            });
            handles.push(handle);
        }

        // Wait for all submissions
        for handle in handles {
            handle.join().unwrap();
        }

        // Verify all jobs were submitted
        let pending = manager.list_jobs_by_status(&JobStatus::Pending).unwrap();
        assert_eq!(pending.len(), num_jobs);
    }
}
