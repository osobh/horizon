//! End-to-end tests for shared storage real-world scenarios

#[cfg(test)]
mod tests {
    use crate::*;
    use std::sync::{Arc, Mutex};
    use std::thread;
    use std::time::{Duration, Instant};

    /// Simulates a GPU agent processing jobs
    struct GpuAgent {
        #[allow(dead_code)]
        id: String,
        manager: Arc<SharedStorageManager>,
        processed_count: Arc<Mutex<usize>>,
    }

    impl GpuAgent {
        fn new(id: String, manager: Arc<SharedStorageManager>) -> Self {
            Self {
                id,
                manager,
                processed_count: Arc::new(Mutex::new(0)),
            }
        }

        fn run(&self, duration: Duration) {
            let start = Instant::now();

            while start.elapsed() < duration {
                // Look for high-priority compute jobs
                let pending = self
                    .manager
                    .list_jobs_by_status(&JobStatus::Pending)
                    .unwrap();

                let compute_jobs: Vec<_> = pending
                    .into_iter()
                    .filter(|j| matches!(j.job_type, JobType::Compute | JobType::Training))
                    .filter(|j| j.priority >= JobPriority::Normal)
                    .take(5) // Batch size
                    .collect();

                for job in compute_jobs {
                    // Claim job
                    if self
                        .manager
                        .update_job_status(&job.id, JobStatus::Processing)
                        .is_ok()
                    {
                        // Simulate GPU processing
                        thread::sleep(Duration::from_millis(10));

                        // Complete job
                        self.manager
                            .update_job_status(&job.id, JobStatus::Completed)
                            .unwrap();

                        let mut count = self.processed_count.lock().unwrap();
                        *count += 1;
                    }
                }

                thread::sleep(Duration::from_millis(1));
            }
        }
    }

    /// Simulates a CPU agent submitting jobs
    struct CpuAgent {
        id: String,
        manager: Arc<SharedStorageManager>,
        submitted_count: Arc<Mutex<usize>>,
    }

    impl CpuAgent {
        fn new(id: String, manager: Arc<SharedStorageManager>) -> Self {
            Self {
                id,
                manager,
                submitted_count: Arc::new(Mutex::new(0)),
            }
        }

        fn run(&self, duration: Duration, job_type: JobType) {
            let start = Instant::now();
            let mut job_counter = 0;

            while start.elapsed() < duration {
                // Submit new job
                let job = Job {
                    id: format!("{}_{}", self.id, job_counter),
                    job_type: job_type.clone(),
                    priority: match job_counter % 10 {
                        0 => JobPriority::Critical,
                        1..=3 => JobPriority::High,
                        _ => JobPriority::Normal,
                    },
                    status: JobStatus::Pending,
                    data_path: PathBuf::from(format!("/tmp/{}/{}", self.id, job_counter)),
                    result_path: None,
                    created_at: SystemTime::now(),
                    updated_at: SystemTime::now(),
                    metadata: HashMap::from([
                        ("agent_id".to_string(), self.id.clone()),
                        ("job_index".to_string(), job_counter.to_string()),
                    ]),
                };

                if self.manager.submit_job(job).is_ok() {
                    let mut count = self.submitted_count.lock().unwrap();
                    *count += 1;
                    job_counter += 1;
                }

                thread::sleep(Duration::from_millis(5));
            }
        }
    }

    #[test]
    fn test_e2e_ml_training_pipeline() {
        let manager = Arc::new(SharedStorageManager::new(SharedStorageConfig::default()));

        // Phase 1: Data preparation jobs
        println!("Phase 1: Data preparation");
        for i in 0..10 {
            let job = Job {
                id: format!("data_prep_{}", i),
                job_type: JobType::DataProcessing,
                priority: JobPriority::High,
                status: JobStatus::Pending,
                data_path: PathBuf::from(format!("/data/raw/batch_{}", i)),
                result_path: None,
                created_at: SystemTime::now(),
                updated_at: SystemTime::now(),
                metadata: HashMap::from([
                    ("phase".to_string(), "preparation".to_string()),
                    ("batch".to_string(), i.to_string()),
                ]),
            };
            manager.submit_job(job).unwrap();
        }

        // Process data prep jobs
        let data_prep_jobs = manager.list_jobs_by_status(&JobStatus::Pending).unwrap();
        for job in data_prep_jobs {
            manager
                .update_job_status(&job.id, JobStatus::Processing)
                .unwrap();
            thread::sleep(Duration::from_millis(5)); // Simulate processing
            manager
                .update_job_status(&job.id, JobStatus::Completed)
                .unwrap();
        }

        // Phase 2: Training jobs
        println!("Phase 2: Model training");
        for epoch in 0..5 {
            let job = Job {
                id: format!("training_epoch_{}", epoch),
                job_type: JobType::Training,
                priority: JobPriority::Critical,
                status: JobStatus::Pending,
                data_path: PathBuf::from("/data/processed"),
                result_path: None,
                created_at: SystemTime::now(),
                updated_at: SystemTime::now(),
                metadata: HashMap::from([
                    ("phase".to_string(), "training".to_string()),
                    ("epoch".to_string(), epoch.to_string()),
                    ("model".to_string(), "transformer_v2".to_string()),
                ]),
            };
            manager.submit_job(job).unwrap();

            // Process epoch
            manager
                .update_job_status(&format!("training_epoch_{}", epoch), JobStatus::Processing)
                .unwrap();
            thread::sleep(Duration::from_millis(20)); // Simulate training
            manager
                .update_job_status(&format!("training_epoch_{}", epoch), JobStatus::Completed)
                .unwrap();
        }

        // Phase 3: Evaluation
        println!("Phase 3: Model evaluation");
        let eval_job = Job {
            id: "model_evaluation".to_string(),
            job_type: JobType::Inference,
            priority: JobPriority::High,
            status: JobStatus::Pending,
            data_path: PathBuf::from("/model/trained"),
            result_path: None,
            created_at: SystemTime::now(),
            updated_at: SystemTime::now(),
            metadata: HashMap::from([
                ("phase".to_string(), "evaluation".to_string()),
                ("model".to_string(), "transformer_v2".to_string()),
            ]),
        };
        manager.submit_job(eval_job).unwrap();

        manager
            .update_job_status("model_evaluation", JobStatus::Processing)
            .unwrap();
        thread::sleep(Duration::from_millis(15));
        manager
            .update_job_status("model_evaluation", JobStatus::Completed)
            .unwrap();

        // Verify all phases completed
        let all_completed = manager.list_jobs_by_status(&JobStatus::Completed).unwrap();
        assert_eq!(all_completed.len(), 16); // 10 prep + 5 training + 1 eval
    }

    #[test]
    fn test_e2e_multi_agent_workload() {
        let manager = Arc::new(SharedStorageManager::new(SharedStorageConfig::default()));
        let test_duration = Duration::from_secs(2);

        // Create GPU agents
        let gpu_agents: Vec<_> = (0..3)
            .map(|i| Arc::new(GpuAgent::new(format!("gpu_{}", i), manager.clone())))
            .collect();

        // Create CPU agents
        let cpu_agents: Vec<_> = (0..5)
            .map(|i| Arc::new(CpuAgent::new(format!("cpu_{}", i), manager.clone())))
            .collect();

        let mut handles = vec![];

        // Start GPU agents
        for agent in &gpu_agents {
            let a = agent.clone();
            let duration = test_duration;
            let handle = thread::spawn(move || {
                a.run(duration);
            });
            handles.push(handle);
        }

        // Start CPU agents with different job types
        let job_types = vec![
            JobType::Compute,
            JobType::Training,
            JobType::Inference,
            JobType::DataProcessing,
            JobType::Custom("Analytics".to_string()),
        ];

        for (i, agent) in cpu_agents.iter().enumerate() {
            let a = agent.clone();
            let job_type = job_types[i].clone();
            let duration = test_duration;
            let handle = thread::spawn(move || {
                a.run(duration, job_type);
            });
            handles.push(handle);
        }

        // Wait for all agents
        for handle in handles {
            handle.join().unwrap();
        }

        // Collect statistics
        let total_submitted: usize = cpu_agents
            .iter()
            .map(|a| *a.submitted_count.lock().unwrap())
            .sum();

        let total_processed: usize = gpu_agents
            .iter()
            .map(|a| *a.processed_count.lock().unwrap())
            .sum();

        println!("Total jobs submitted: {}", total_submitted);
        println!("Total jobs processed: {}", total_processed);

        // Verify reasonable throughput
        assert!(total_submitted > 100);
        assert!(total_processed > 50);

        // Check for unprocessed jobs
        let pending = manager.list_jobs_by_status(&JobStatus::Pending).unwrap();
        let processing = manager.list_jobs_by_status(&JobStatus::Processing).unwrap();
        println!("Remaining pending: {}", pending.len());
        println!("Still processing: {}", processing.len());
    }

    #[test]
    fn test_e2e_streaming_workflow() {
        let manager = Arc::new(SharedStorageManager::new(SharedStorageConfig::default()));

        // Simulate video processing pipeline
        let video_segments = 20;
        let mut segment_jobs = vec![];

        // Phase 1: Submit segment processing jobs
        for i in 0..video_segments {
            let job = Job {
                id: format!("segment_{}", i),
                job_type: JobType::Streaming,
                priority: if i == 0 {
                    JobPriority::Critical
                } else {
                    JobPriority::High
                },
                status: JobStatus::Pending,
                data_path: PathBuf::from(format!("/video/segment_{}.mp4", i)),
                result_path: None,
                created_at: SystemTime::now(),
                updated_at: SystemTime::now(),
                metadata: HashMap::from([
                    ("stream_id".to_string(), "video_001".to_string()),
                    ("segment".to_string(), i.to_string()),
                    ("duration_ms".to_string(), "2000".to_string()),
                ]),
            };
            segment_jobs.push(job.id.clone());
            manager.submit_job(job).unwrap();
        }

        // Phase 2: Process segments in order (simulating streaming constraints)
        for (i, job_id) in segment_jobs.iter().enumerate() {
            // Wait for previous segment if not first
            if i > 0 {
                let prev_job = manager.get_job(&segment_jobs[i - 1]).unwrap().unwrap();
                assert_eq!(prev_job.status, JobStatus::Completed);
            }

            manager
                .update_job_status(job_id, JobStatus::Processing)
                .unwrap();
            thread::sleep(Duration::from_millis(10)); // Simulate processing
            manager
                .update_job_status(job_id, JobStatus::Completed)
                .unwrap();
        }

        // Phase 3: Submit final merge job
        let merge_job = Job {
            id: "merge_segments".to_string(),
            job_type: JobType::DataProcessing,
            priority: JobPriority::Critical,
            status: JobStatus::Pending,
            data_path: PathBuf::from("/video/processed"),
            result_path: None,
            created_at: SystemTime::now(),
            updated_at: SystemTime::now(),
            metadata: HashMap::from([
                ("operation".to_string(), "merge".to_string()),
                ("segments".to_string(), video_segments.to_string()),
            ]),
        };
        manager.submit_job(merge_job).unwrap();

        manager
            .update_job_status("merge_segments", JobStatus::Processing)
            .unwrap();
        thread::sleep(Duration::from_millis(50));
        manager
            .update_job_status("merge_segments", JobStatus::Completed)
            .unwrap();

        // Verify all jobs completed
        let completed = manager.list_jobs_by_status(&JobStatus::Completed).unwrap();
        assert_eq!(completed.len(), video_segments + 1);
    }

    #[test]
    fn test_e2e_fault_recovery_scenario() {
        let manager = Arc::new(SharedStorageManager::new(SharedStorageConfig::default()));

        // Submit critical jobs that must complete
        let critical_jobs = 5;
        for i in 0..critical_jobs {
            let job = Job {
                id: format!("critical_{}", i),
                job_type: JobType::Compute,
                priority: JobPriority::Critical,
                status: JobStatus::Pending,
                data_path: PathBuf::from(format!("/critical/job_{}", i)),
                result_path: None,
                created_at: SystemTime::now(),
                updated_at: SystemTime::now(),
                metadata: HashMap::from([
                    ("resilient".to_string(), "true".to_string()),
                    ("max_retries".to_string(), "3".to_string()),
                ]),
            };
            manager.submit_job(job).unwrap();
        }

        // Simulate processing with random failures
        for i in 0..critical_jobs {
            let job_id = format!("critical_{}", i);
            let mut attempts = 0;
            let max_attempts = 3;

            loop {
                attempts += 1;
                manager
                    .update_job_status(&job_id, JobStatus::Processing)
                    .unwrap();

                // Simulate random failure (50% chance)
                if attempts < max_attempts && i % 2 == 0 {
                    let error = format!("Transient error on attempt {}", attempts);
                    manager
                        .update_job_status(&job_id, JobStatus::Failed(error))
                        .unwrap();

                    // Reset to pending for retry
                    thread::sleep(Duration::from_millis(10)); // Backoff
                    manager
                        .update_job_status(&job_id, JobStatus::Pending)
                        .unwrap();
                } else {
                    // Success
                    manager
                        .update_job_status(&job_id, JobStatus::Completed)
                        .unwrap();
                    break;
                }
            }
        }

        // Verify all critical jobs eventually completed
        for i in 0..critical_jobs {
            let job = manager
                .get_job(&format!("critical_{}", i))
                .unwrap()
                .unwrap();
            assert_eq!(job.status, JobStatus::Completed);
        }
    }

    #[test]
    fn test_e2e_data_pipeline_with_checkpoints() {
        let manager = Arc::new(SharedStorageManager::new(SharedStorageConfig::default()));

        // Large data processing job with checkpoints
        let total_chunks = 100;
        let checkpoint_interval = 10;
        let pipeline_id = "data_pipeline_001";

        // Submit initial job
        let pipeline_job = Job {
            id: pipeline_id.to_string(),
            job_type: JobType::DataProcessing,
            priority: JobPriority::Normal,
            status: JobStatus::Pending,
            data_path: PathBuf::from("/data/large_dataset"),
            result_path: None,
            created_at: SystemTime::now(),
            updated_at: SystemTime::now(),
            metadata: HashMap::from([
                ("total_chunks".to_string(), total_chunks.to_string()),
                (
                    "checkpoint_interval".to_string(),
                    checkpoint_interval.to_string(),
                ),
                ("current_chunk".to_string(), "0".to_string()),
            ]),
        };
        manager.submit_job(pipeline_job).unwrap();

        // Process with checkpoints
        manager
            .update_job_status(pipeline_id, JobStatus::Processing)
            .unwrap();

        for chunk in 0..total_chunks {
            // Simulate chunk processing
            thread::sleep(Duration::from_micros(100));

            // Create checkpoint every N chunks
            if chunk > 0 && chunk % checkpoint_interval == 0 {
                let checkpoint_job = Job {
                    id: format!("{}_checkpoint_{}", pipeline_id, chunk),
                    job_type: JobType::Custom("Checkpoint".to_string()),
                    priority: JobPriority::High,
                    status: JobStatus::Pending,
                    data_path: PathBuf::from(format!("/checkpoints/{}/{}", pipeline_id, chunk)),
                    result_path: None,
                    created_at: SystemTime::now(),
                    updated_at: SystemTime::now(),
                    metadata: HashMap::from([
                        ("parent_job".to_string(), pipeline_id.to_string()),
                        ("chunk".to_string(), chunk.to_string()),
                        (
                            "timestamp".to_string(),
                            SystemTime::now()
                                .duration_since(SystemTime::UNIX_EPOCH)
                                .unwrap()
                                .as_secs()
                                .to_string(),
                        ),
                    ]),
                };
                manager.submit_job(checkpoint_job).unwrap();

                // Process checkpoint immediately
                let checkpoint_id = format!("{}_checkpoint_{}", pipeline_id, chunk);
                manager
                    .update_job_status(&checkpoint_id, JobStatus::Processing)
                    .unwrap();
                manager
                    .update_job_status(&checkpoint_id, JobStatus::Completed)
                    .unwrap();
            }
        }

        // Complete main job
        manager
            .update_job_status(pipeline_id, JobStatus::Completed)
            .unwrap();

        // Verify all checkpoints were created
        let all_jobs = manager.list_jobs_by_status(&JobStatus::Completed).unwrap();
        let checkpoints: Vec<_> = all_jobs
            .iter()
            .filter(|j| j.id.contains("checkpoint"))
            .collect();

        // Checkpoints are created at chunks 10, 20, 30, ..., 90 (not at 0 or 100)
        assert_eq!(checkpoints.len(), (total_chunks / checkpoint_interval) - 1);
    }

    #[test]
    fn test_e2e_distributed_compute_job() {
        let manager = Arc::new(SharedStorageManager::new(SharedStorageConfig::default()));

        // Large compute job that gets split across multiple workers
        let main_job_id = "distributed_compute_001";
        let num_workers = 4;
        let tasks_per_worker = 25;

        // Submit main orchestrator job
        let main_job = Job {
            id: main_job_id.to_string(),
            job_type: JobType::Custom("Orchestrator".to_string()),
            priority: JobPriority::High,
            status: JobStatus::Pending,
            data_path: PathBuf::from("/compute/input"),
            result_path: None,
            created_at: SystemTime::now(),
            updated_at: SystemTime::now(),
            metadata: HashMap::from([
                ("job_type".to_string(), "matrix_multiplication".to_string()),
                ("size".to_string(), "10000x10000".to_string()),
                ("workers".to_string(), num_workers.to_string()),
            ]),
        };
        manager.submit_job(main_job).unwrap();
        manager
            .update_job_status(main_job_id, JobStatus::Processing)
            .unwrap();

        // Create worker jobs
        let mut worker_handles = vec![];
        for worker_id in 0..num_workers {
            let mgr = manager.clone();
            let main_id = main_job_id.to_string();

            let handle = thread::spawn(move || {
                // Submit worker tasks
                for task_id in 0..tasks_per_worker {
                    let job = Job {
                        id: format!("{}_worker_{}_task_{}", main_id, worker_id, task_id),
                        job_type: JobType::Compute,
                        priority: JobPriority::Normal,
                        status: JobStatus::Pending,
                        data_path: PathBuf::from(format!(
                            "/compute/partition_{}/{}",
                            worker_id, task_id
                        )),
                        result_path: None,
                        created_at: SystemTime::now(),
                        updated_at: SystemTime::now(),
                        metadata: HashMap::from([
                            ("parent_job".to_string(), main_id.clone()),
                            ("worker_id".to_string(), worker_id.to_string()),
                            ("task_id".to_string(), task_id.to_string()),
                        ]),
                    };
                    mgr.submit_job(job).unwrap();
                }

                // Process own tasks
                for task_id in 0..tasks_per_worker {
                    let job_id = format!("{}_worker_{}_task_{}", main_id, worker_id, task_id);
                    mgr.update_job_status(&job_id, JobStatus::Processing)
                        .unwrap();
                    thread::sleep(Duration::from_micros(100)); // Simulate computation
                    mgr.update_job_status(&job_id, JobStatus::Completed)
                        .unwrap();
                }
            });
            worker_handles.push(handle);
        }

        // Wait for all workers
        for handle in worker_handles {
            handle.join().unwrap();
        }

        // Verify all tasks completed
        let completed = manager.list_jobs_by_status(&JobStatus::Completed).unwrap();
        let worker_tasks: Vec<_> = completed
            .iter()
            .filter(|j| j.id.contains("worker") && j.id.contains("task"))
            .collect();
        assert_eq!(worker_tasks.len(), num_workers * tasks_per_worker);

        // Complete main job
        manager
            .update_job_status(main_job_id, JobStatus::Completed)
            .unwrap();
    }
}
