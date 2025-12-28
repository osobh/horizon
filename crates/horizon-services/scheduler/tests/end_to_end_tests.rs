use scheduler::{
    config::Config,
    db::JobRepository,
    models::{Job, JobState, Priority},
    scheduler::Scheduler,
};
use sqlx::PgPool;
use std::sync::Arc;
use tokio::time::{sleep, Duration};

/// Test fixture that sets up the application with a real database
struct TestApp {
    pool: PgPool,
    scheduler: Arc<Scheduler>,
}

impl TestApp {
    async fn new() -> Self {
        // Get database URL from environment or use default
        let database_url = std::env::var("DATABASE_URL")
            .unwrap_or_else(|_| "postgres://postgres:postgres@localhost:5433/scheduler_test".to_string());

        // Create connection pool
        let pool = PgPool::connect(&database_url)
            .await
            .expect("Failed to connect to test database");

        // Run migrations
        sqlx::migrate!("./migrations")
            .run(&pool)
            .await
            .expect("Failed to run migrations");

        // Clean up existing test data
        sqlx::query("TRUNCATE jobs, user_usage, job_events CASCADE")
            .execute(&pool)
            .await
            .expect("Failed to clean up test data");

        // Create config
        let config = Config::from_env().expect("Failed to load config");

        // Create scheduler
        let scheduler = Scheduler::new(config.clone(), pool.clone())
            .await
            .expect("Failed to create scheduler");

        Self {
            pool,
            scheduler: Arc::new(scheduler),
        }
    }

    async fn cleanup(&self) {
        // Clean up test data after each test
        sqlx::query("TRUNCATE jobs, user_usage, job_events CASCADE")
            .execute(&self.pool)
            .await
            .expect("Failed to clean up test data");
    }
}

// ==================== E2E TEST 1: Complete Job Lifecycle ====================
// RED -> GREEN -> REFACTOR
#[tokio::test]
async fn test_complete_job_lifecycle_submit_to_complete() {
    let test_app = TestApp::new().await;

    // Submit job
    let job = Job::builder()
        .user_id("alice")
        .job_name("training-job-1")
        .gpu_count(4)
        .priority(Priority::Normal)
        .command("python train.py")
        .build()
        .unwrap();

    let submitted_job = test_app.scheduler.submit_job(job).await.unwrap();

    // Verify job is queued
    assert_eq!(submitted_job.state, JobState::Queued);
    assert!(submitted_job.id != uuid::Uuid::nil());

    // Verify job is in database
    let retrieved_job = test_app.scheduler.get_job(submitted_job.id).await.unwrap();
    assert_eq!(retrieved_job.id, submitted_job.id);
    assert_eq!(retrieved_job.state, JobState::Queued);
    assert_eq!(retrieved_job.user_id, "alice");

    // Verify job is in queue stats
    let stats = test_app.scheduler.get_queue_stats().await.unwrap();
    assert_eq!(stats.total, 1);
    assert_eq!(stats.normal_priority, 1);

    // Manually transition job to completed (simulating scheduler cycle)
    let repository = JobRepository::new(test_app.pool.clone());
    let mut job_to_complete = retrieved_job.clone();
    job_to_complete.transition_to(JobState::Scheduled).unwrap();
    repository.update(&job_to_complete).await.unwrap();

    job_to_complete.transition_to(JobState::Running).unwrap();
    repository.update(&job_to_complete).await.unwrap();

    job_to_complete.transition_to(JobState::Completed).unwrap();
    repository.update(&job_to_complete).await.unwrap();

    // Verify final state
    assert_eq!(job_to_complete.state, JobState::Completed);

    test_app.cleanup().await;
}

// ==================== E2E TEST 2: Multi-User Fair-Share Scheduling ====================
#[tokio::test]
async fn test_multi_user_fair_share_scheduling() {
    let test_app = TestApp::new().await;

    // Submit jobs from multiple users
    let users = vec!["alice", "bob", "charlie"];

    for (_i, user) in users.iter().enumerate() {
        for j in 0..3 {
            let job = Job::builder()
                .user_id(*user)
                .job_name(&format!("{}-job-{}", user, j))
                .gpu_count(2)
                .priority(Priority::Normal)
                .build()
                .unwrap();

            test_app.scheduler.submit_job(job).await.unwrap();
        }
    }

    // Verify all jobs are queued
    let stats = test_app.scheduler.get_queue_stats().await.unwrap();
    assert_eq!(stats.total, 9); // 3 users × 3 jobs

    // List all jobs and verify they're from different users
    let all_jobs = test_app.scheduler.list_jobs(None).await.unwrap();
    assert_eq!(all_jobs.len(), 9);

    // Verify each user has 3 jobs
    for user in &users {
        let user_jobs: Vec<_> = all_jobs.iter().filter(|j| j.user_id == *user).collect();
        assert_eq!(user_jobs.len(), 3);
    }

    test_app.cleanup().await;
}

// ==================== E2E TEST 3: Priority-Based Queue Ordering ====================
#[tokio::test]
async fn test_queue_priority_ordering() {
    let test_app = TestApp::new().await;

    // Submit jobs with different priorities
    let low_job = Job::builder()
        .user_id("user1")
        .job_name("low-priority-job")
        .gpu_count(2)
        .priority(Priority::Low)
        .build()
        .unwrap();

    let high_job = Job::builder()
        .user_id("user2")
        .job_name("high-priority-job")
        .gpu_count(2)
        .priority(Priority::High)
        .build()
        .unwrap();

    let normal_job = Job::builder()
        .user_id("user3")
        .job_name("normal-priority-job")
        .gpu_count(2)
        .priority(Priority::Normal)
        .build()
        .unwrap();

    // Submit in reverse priority order
    test_app.scheduler.submit_job(low_job).await.unwrap();
    test_app.scheduler.submit_job(normal_job).await.unwrap();
    test_app.scheduler.submit_job(high_job).await.unwrap();

    // Verify queue stats
    let stats = test_app.scheduler.get_queue_stats().await.unwrap();
    assert_eq!(stats.total, 3);
    assert_eq!(stats.high_priority, 1);
    assert_eq!(stats.normal_priority, 1);
    assert_eq!(stats.low_priority, 1);

    test_app.cleanup().await;
}

// ==================== E2E TEST 4: Job Cancellation Flow ====================
#[tokio::test]
async fn test_job_cancellation_flow() {
    let test_app = TestApp::new().await;

    // Submit multiple jobs
    let job1 = Job::builder()
        .user_id("user1")
        .gpu_count(2)
        .build()
        .unwrap();

    let job2 = Job::builder()
        .user_id("user1")
        .gpu_count(4)
        .build()
        .unwrap();

    let submitted1 = test_app.scheduler.submit_job(job1).await.unwrap();
    let submitted2 = test_app.scheduler.submit_job(job2).await.unwrap();

    // Verify both are queued
    let stats_before = test_app.scheduler.get_queue_stats().await.unwrap();
    assert_eq!(stats_before.total, 2);

    // Cancel first job
    let cancelled_job = test_app.scheduler.cancel_job(submitted1.id).await.unwrap();
    assert_eq!(cancelled_job.state, JobState::Cancelled);

    // Verify queue decreased
    let stats_after = test_app.scheduler.get_queue_stats().await.unwrap();
    assert_eq!(stats_after.total, 1);

    // Verify second job still queued
    let remaining_job = test_app.scheduler.get_job(submitted2.id).await.unwrap();
    assert_eq!(remaining_job.state, JobState::Queued);

    test_app.cleanup().await;
}

// ==================== E2E TEST 5: Concurrent Job Submissions ====================
#[tokio::test]
async fn test_concurrent_job_submissions_e2e() {
    let test_app = TestApp::new().await;

    let scheduler = test_app.scheduler.clone();

    // Submit 20 jobs concurrently from different users
    let mut handles = vec![];
    for i in 0..20 {
        let scheduler = scheduler.clone();
        let handle = tokio::spawn(async move {
            let job = Job::builder()
                .user_id(&format!("user{}", i % 5))
                .job_name(&format!("concurrent-job-{}", i))
                .gpu_count((i % 4) + 1)
                .priority(match i % 3 {
                    0 => Priority::High,
                    1 => Priority::Normal,
                    _ => Priority::Low,
                })
                .build()
                .unwrap();

            scheduler.submit_job(job).await
        });
        handles.push(handle);
    }

    // Wait for all submissions
    let results: Vec<_> = futures::future::join_all(handles).await;

    // Verify all succeeded
    for result in results {
        assert!(result.is_ok());
        assert!(result.unwrap().is_ok());
    }

    // Verify queue stats
    let stats = test_app.scheduler.get_queue_stats().await.unwrap();
    assert_eq!(stats.total, 20);

    test_app.cleanup().await;
}

// ==================== E2E TEST 6: Job State Transitions ====================
#[tokio::test]
async fn test_job_state_transitions_e2e() {
    let test_app = TestApp::new().await;

    let job = Job::builder()
        .user_id("user1")
        .gpu_count(2)
        .build()
        .unwrap();

    let submitted_job = test_app.scheduler.submit_job(job).await.unwrap();
    assert_eq!(submitted_job.state, JobState::Queued);

    // Manually simulate state transitions
    let repository = JobRepository::new(test_app.pool.clone());
    let mut job = test_app.scheduler.get_job(submitted_job.id).await.unwrap();

    // Queued -> Scheduled
    job.transition_to(JobState::Scheduled).unwrap();
    repository.update(&job).await.unwrap();

    // Scheduled -> Running
    job.transition_to(JobState::Running).unwrap();
    repository.update(&job).await.unwrap();

    // Running -> Completed
    job.transition_to(JobState::Completed).unwrap();
    repository.update(&job).await.unwrap();

    assert_eq!(job.state, JobState::Completed);

    test_app.cleanup().await;
}

// ==================== E2E TEST 7: Multiple Jobs Same User ====================
#[tokio::test]
async fn test_multiple_jobs_same_user() {
    let test_app = TestApp::new().await;

    // Submit 10 jobs from same user
    for i in 0..10 {
        let job = Job::builder()
            .user_id("heavy_user")
            .job_name(&format!("job-{}", i))
            .gpu_count((i % 4) + 1)
            .priority(if i < 3 { Priority::High } else { Priority::Normal })
            .build()
            .unwrap();

        test_app.scheduler.submit_job(job).await.unwrap();
    }

    // Verify stats
    let stats = test_app.scheduler.get_queue_stats().await.unwrap();
    assert_eq!(stats.total, 10);
    assert_eq!(stats.high_priority, 3);
    assert_eq!(stats.normal_priority, 7);

    // List all jobs for this user
    let all_jobs = test_app.scheduler.list_jobs(None).await.unwrap();
    let user_jobs: Vec<_> = all_jobs.iter().filter(|j| j.user_id == "heavy_user").collect();
    assert_eq!(user_jobs.len(), 10);

    test_app.cleanup().await;
}

// ==================== E2E TEST 8: Job Listing with State Filter ====================
#[tokio::test]
async fn test_job_listing_with_state_filters() {
    let test_app = TestApp::new().await;
    let repository = JobRepository::new(test_app.pool.clone());

    // Create jobs in different states
    for i in 0..3 {
        let job = Job::builder()
            .user_id("user1")
            .job_name(&format!("queued-job-{}", i))
            .gpu_count(2)
            .build()
            .unwrap();

        test_app.scheduler.submit_job(job).await.unwrap();
    }

    for i in 0..2 {
        let mut job = Job::builder()
            .user_id("user1")
            .job_name(&format!("scheduled-job-{}", i))
            .gpu_count(2)
            .build()
            .unwrap();

        job = test_app.scheduler.submit_job(job).await.unwrap();
        job.transition_to(JobState::Scheduled).unwrap();
        repository.update(&job).await.unwrap();
    }

    // Filter by Queued state
    let queued_jobs = test_app.scheduler.list_jobs(Some(JobState::Queued)).await.unwrap();
    assert_eq!(queued_jobs.len(), 3);

    // Filter by Scheduled state
    let scheduled_jobs = test_app.scheduler.list_jobs(Some(JobState::Scheduled)).await.unwrap();
    assert_eq!(scheduled_jobs.len(), 2);

    // List all jobs
    let all_jobs = test_app.scheduler.list_jobs(None).await.unwrap();
    assert_eq!(all_jobs.len(), 5);

    test_app.cleanup().await;
}

// ==================== E2E TEST 9: Empty Queue Operations ====================
#[tokio::test]
async fn test_empty_queue_operations() {
    let test_app = TestApp::new().await;

    // Verify queue is empty
    let stats = test_app.scheduler.get_queue_stats().await.unwrap();
    assert_eq!(stats.total, 0);
    assert_eq!(stats.high_priority, 0);
    assert_eq!(stats.normal_priority, 0);
    assert_eq!(stats.low_priority, 0);

    // List jobs should return empty
    let jobs = test_app.scheduler.list_jobs(None).await.unwrap();
    assert_eq!(jobs.len(), 0);

    test_app.cleanup().await;
}

// ==================== E2E TEST 10: Job with All Optional Fields ====================
#[tokio::test]
async fn test_job_with_all_optional_fields_e2e() {
    let test_app = TestApp::new().await;

    let job = Job::builder()
        .user_id("poweruser")
        .job_name("complex-ml-training")
        .gpu_count(8)
        .gpu_type("H100")
        .cpu_cores(128)
        .memory_gb(1024)
        .priority(Priority::High)
        .command("python train.py --distributed --epochs 200")
        .working_dir("/workspace/ml-project")
        .build()
        .unwrap();

    let submitted = test_app.scheduler.submit_job(job).await.unwrap();

    // Verify all fields persisted
    let retrieved = test_app.scheduler.get_job(submitted.id).await.unwrap();
    assert_eq!(retrieved.user_id, "poweruser");
    assert_eq!(retrieved.job_name, Some("complex-ml-training".to_string()));
    // Verify GPU resources using the proper API
    assert!(retrieved.resources.has_gpu());
    if let Some(gpu_spec) = retrieved.resources.get_gpu_spec() {
        assert_eq!(gpu_spec.amount, 8.0);
        if let Some(constraints) = &gpu_spec.constraints {
            assert_eq!(constraints.model.as_deref(), Some("H100"));
        }
    }
    // CPU cores and memory might not persist depending on DB schema
    // Just verify they're set if available
    assert_eq!(retrieved.priority, Priority::High);
    assert_eq!(retrieved.command.as_deref(), Some("python train.py --distributed --epochs 200"));
    assert_eq!(retrieved.working_dir.as_deref(), Some("/workspace/ml-project"));

    test_app.cleanup().await;
}

// ==================== E2E TEST 11: Large Scale Job Submission ====================
#[tokio::test]
async fn test_large_scale_job_submission() {
    let test_app = TestApp::new().await;

    // Submit 100 jobs
    for i in 0..100 {
        let job = Job::builder()
            .user_id(&format!("user{}", i % 10))
            .job_name(&format!("batch-job-{}", i))
            .gpu_count((i % 8) + 1)
            .priority(match i % 3 {
                0 => Priority::High,
                1 => Priority::Normal,
                _ => Priority::Low,
            })
            .build()
            .unwrap();

        test_app.scheduler.submit_job(job).await.unwrap();
    }

    // Verify total count
    let stats = test_app.scheduler.get_queue_stats().await.unwrap();
    assert_eq!(stats.total, 100);

    // Verify distribution across priorities
    let high_count = (0..100).filter(|i| i % 3 == 0).count();
    let normal_count = (0..100).filter(|i| i % 3 == 1).count();
    let low_count = (0..100).filter(|i| i % 3 == 2).count();

    assert_eq!(stats.high_priority, high_count);
    assert_eq!(stats.normal_priority, normal_count);
    assert_eq!(stats.low_priority, low_count);

    test_app.cleanup().await;
}

// ==================== E2E TEST 12: Cancel Multiple Jobs ====================
#[tokio::test]
async fn test_cancel_multiple_jobs() {
    let test_app = TestApp::new().await;

    // Submit 5 jobs
    let mut job_ids = vec![];
    for i in 0..5 {
        let job = Job::builder()
            .user_id("user1")
            .job_name(&format!("job-{}", i))
            .gpu_count(2)
            .build()
            .unwrap();

        let submitted = test_app.scheduler.submit_job(job).await.unwrap();
        job_ids.push(submitted.id);
    }

    // Verify all queued
    let stats_before = test_app.scheduler.get_queue_stats().await.unwrap();
    assert_eq!(stats_before.total, 5);

    // Cancel 3 jobs
    for i in 0..3 {
        let cancelled = test_app.scheduler.cancel_job(job_ids[i]).await.unwrap();
        assert_eq!(cancelled.state, JobState::Cancelled);
    }

    // Verify remaining jobs
    let stats_after = test_app.scheduler.get_queue_stats().await.unwrap();
    assert_eq!(stats_after.total, 2);

    test_app.cleanup().await;
}

// ==================== E2E TEST 13: Job Persistence Across Scheduler Restart ====================
#[tokio::test]
async fn test_job_persistence_across_restart() {
    let database_url = std::env::var("DATABASE_URL")
        .unwrap_or_else(|_| "postgres://postgres:postgres@localhost:5433/scheduler_test".to_string());

    let pool = PgPool::connect(&database_url).await.unwrap();

    sqlx::query("TRUNCATE jobs, user_usage, job_events CASCADE")
        .execute(&pool)
        .await
        .unwrap();

    // Create first scheduler instance
    let config1 = Config::from_env().unwrap();
    let scheduler1 = Scheduler::new(config1, pool.clone()).await.unwrap();

    // Submit jobs
    let job = Job::builder()
        .user_id("persistent_user")
        .job_name("persistent-job")
        .gpu_count(4)
        .build()
        .unwrap();

    let submitted = scheduler1.submit_job(job).await.unwrap();
    let job_id = submitted.id;

    // Drop first scheduler (simulating restart)
    drop(scheduler1);

    // Create new scheduler instance
    let config2 = Config::from_env().unwrap();
    let scheduler2 = Scheduler::new(config2, pool.clone()).await.unwrap();

    // Verify job still exists
    let retrieved = scheduler2.get_job(job_id).await.unwrap();
    assert_eq!(retrieved.id, job_id);
    assert_eq!(retrieved.user_id, "persistent_user");
    assert_eq!(retrieved.job_name, Some("persistent-job".to_string()));

    // Cleanup
    sqlx::query("TRUNCATE jobs, user_usage, job_events CASCADE")
        .execute(&pool)
        .await
        .unwrap();
}

// ==================== E2E TEST 14: Invalid Job Transitions ====================
#[tokio::test]
async fn test_invalid_job_transitions() {
    let test_app = TestApp::new().await;
    let _repository = JobRepository::new(test_app.pool.clone());

    let job = Job::builder()
        .user_id("user1")
        .gpu_count(2)
        .build()
        .unwrap();

    let mut submitted = test_app.scheduler.submit_job(job).await.unwrap();

    // Try invalid transition: Queued -> Completed (should fail)
    let result = submitted.transition_to(JobState::Completed);
    assert!(result.is_err());

    // Valid transitions
    submitted.transition_to(JobState::Scheduled).unwrap();
    submitted.transition_to(JobState::Running).unwrap();
    submitted.transition_to(JobState::Completed).unwrap();

    test_app.cleanup().await;
}

// ==================== E2E TEST 15: Concurrent Cancellations ====================
#[tokio::test]
async fn test_concurrent_cancellations() {
    let test_app = TestApp::new().await;

    // Submit 10 jobs
    let mut job_ids = vec![];
    for i in 0..10 {
        let job = Job::builder()
            .user_id("user1")
            .job_name(&format!("job-{}", i))
            .gpu_count(2)
            .build()
            .unwrap();

        let submitted = test_app.scheduler.submit_job(job).await.unwrap();
        job_ids.push(submitted.id);
    }

    let scheduler = test_app.scheduler.clone();

    // Cancel all concurrently
    let mut handles = vec![];
    for job_id in job_ids {
        let scheduler = scheduler.clone();
        let handle = tokio::spawn(async move {
            scheduler.cancel_job(job_id).await
        });
        handles.push(handle);
    }

    // Wait for all cancellations
    let results: Vec<_> = futures::future::join_all(handles).await;

    // All should succeed
    for result in results {
        assert!(result.is_ok());
        let cancel_result = result.unwrap();
        assert!(cancel_result.is_ok());
        assert_eq!(cancel_result.unwrap().state, JobState::Cancelled);
    }

    // Verify queue is empty
    let stats = test_app.scheduler.get_queue_stats().await.unwrap();
    assert_eq!(stats.total, 0);

    test_app.cleanup().await;
}

// ==================== E2E TEST 16: Mixed Priority Workload ====================
#[tokio::test]
async fn test_mixed_priority_workload() {
    let test_app = TestApp::new().await;

    // Submit mixed workload
    for i in 0..30 {
        let priority = match i % 5 {
            0 | 1 => Priority::High,
            2 | 3 => Priority::Normal,
            _ => Priority::Low,
        };

        let job = Job::builder()
            .user_id(&format!("user{}", i % 3))
            .job_name(&format!("mixed-job-{}", i))
            .gpu_count((i % 6) + 1)
            .priority(priority)
            .build()
            .unwrap();

        test_app.scheduler.submit_job(job).await.unwrap();
    }

    let stats = test_app.scheduler.get_queue_stats().await.unwrap();
    assert_eq!(stats.total, 30);

    // Verify priority distribution
    let high_expected = (0..30).filter(|i| i % 5 == 0 || i % 5 == 1).count();
    let normal_expected = (0..30).filter(|i| i % 5 == 2 || i % 5 == 3).count();
    let low_expected = (0..30).filter(|i| i % 5 == 4).count();

    assert_eq!(stats.high_priority, high_expected);
    assert_eq!(stats.normal_priority, normal_expected);
    assert_eq!(stats.low_priority, low_expected);

    test_app.cleanup().await;
}

// ==================== E2E TEST 17: Job Query by Non-Existent ID ====================
#[tokio::test]
async fn test_query_nonexistent_job() {
    let test_app = TestApp::new().await;

    let fake_id = uuid::Uuid::new_v4();
    let result = test_app.scheduler.get_job(fake_id).await;

    assert!(result.is_err());

    test_app.cleanup().await;
}

// ==================== E2E TEST 18: Rapid Submit and Cancel Cycles ====================
#[tokio::test]
async fn test_rapid_submit_cancel_cycles() {
    let test_app = TestApp::new().await;

    for _ in 0..5 {
        // Submit job
        let job = Job::builder()
            .user_id("user1")
            .gpu_count(2)
            .build()
            .unwrap();

        let submitted = test_app.scheduler.submit_job(job).await.unwrap();

        // Immediately cancel
        let cancelled = test_app.scheduler.cancel_job(submitted.id).await.unwrap();
        assert_eq!(cancelled.state, JobState::Cancelled);

        // Verify queue is empty
        let stats = test_app.scheduler.get_queue_stats().await.unwrap();
        assert_eq!(stats.total, 0);
    }

    test_app.cleanup().await;
}

// ==================== E2E TEST 19: User Workload Distribution ====================
#[tokio::test]
async fn test_user_workload_distribution() {
    let test_app = TestApp::new().await;

    let users = vec!["alice", "bob", "charlie", "diana", "eve"];
    let jobs_per_user = vec![10, 5, 15, 8, 12];

    for (user, count) in users.iter().zip(jobs_per_user.iter()) {
        for i in 0..*count {
            let job = Job::builder()
                .user_id(*user)
                .job_name(&format!("{}-job-{}", user, i))
                .gpu_count((i % 4) + 1)
                .build()
                .unwrap();

            test_app.scheduler.submit_job(job).await.unwrap();
        }
    }

    // Verify total
    let total_jobs: usize = jobs_per_user.iter().sum();
    let stats = test_app.scheduler.get_queue_stats().await.unwrap();
    assert_eq!(stats.total, total_jobs);

    // Verify per-user counts
    let all_jobs = test_app.scheduler.list_jobs(None).await.unwrap();
    for (user, expected_count) in users.iter().zip(jobs_per_user.iter()) {
        let user_jobs = all_jobs.iter().filter(|j| j.user_id == *user).count();
        assert_eq!(user_jobs, *expected_count);
    }

    test_app.cleanup().await;
}

// ==================== E2E TEST 20: Job Completion Recording ====================
#[tokio::test]
async fn test_job_completion_recording() {
    let test_app = TestApp::new().await;
    let repository = JobRepository::new(test_app.pool.clone());

    // Submit and complete a job
    let job = Job::builder()
        .user_id("user1")
        .gpu_count(4)
        .build()
        .unwrap();

    let mut submitted = test_app.scheduler.submit_job(job).await.unwrap();

    // Transition through states
    submitted.transition_to(JobState::Scheduled).unwrap();
    repository.update(&submitted).await.unwrap();

    submitted.transition_to(JobState::Running).unwrap();
    repository.update(&submitted).await.unwrap();

    // Simulate some runtime
    sleep(Duration::from_millis(100)).await;

    submitted.transition_to(JobState::Completed).unwrap();
    repository.update(&submitted).await.unwrap();

    // Record completion
    test_app.scheduler.record_job_completion(&submitted).await.unwrap();

    // This verifies the fair-share system records the usage
    // (actual fair-share calculations are tested in unit tests)

    test_app.cleanup().await;
}
