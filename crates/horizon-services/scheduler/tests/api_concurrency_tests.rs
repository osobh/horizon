use scheduler::models::{Job, Priority};

mod common;
use common::test_app::TestApp;

// ==================== Concurrency Tests ====================

#[tokio::test]
async fn test_concurrent_job_submissions() {
    let test_app = TestApp::new().await;

    let scheduler = test_app.scheduler.clone();

    // Submit 10 jobs concurrently
    let mut handles = vec![];
    for i in 0..10 {
        let scheduler = scheduler.clone();
        let handle = tokio::spawn(async move {
            let job = Job::builder()
                .user_id(&format!("user{}", i))
                .gpu_count(i % 4 + 1)
                .priority(if i % 3 == 0 { Priority::High } else { Priority::Normal })
                .build()
                .unwrap();

            scheduler.submit_job(job).await
        });
        handles.push(handle);
    }

    // Wait for all jobs to complete
    let results: Vec<_> = futures::future::join_all(handles).await;

    // All should succeed
    assert_eq!(results.len(), 10);
    for result in results {
        assert!(result.is_ok());
        assert!(result.unwrap().is_ok());
    }

    // Verify all jobs were created
    let repository = scheduler::db::JobRepository::new(test_app.pool.clone());
    let all_jobs = repository.list_all().await.unwrap();
    assert_eq!(all_jobs.len(), 10);

    test_app.cleanup().await;
}

#[tokio::test]
async fn test_concurrent_reads_and_writes() {
    let test_app = TestApp::new().await;

    // Create initial jobs
    let repository = scheduler::db::JobRepository::new(test_app.pool.clone());
    let mut job_ids = vec![];

    for i in 0..5 {
        let job = Job::builder()
            .user_id(&format!("user{}", i))
            .gpu_count(2)
            .build()
            .unwrap();
        let created = repository.create(&job).await.unwrap();
        job_ids.push(created.id);
    }

    let scheduler = test_app.scheduler.clone();

    // Concurrently read existing jobs and submit new jobs
    let mut handles = vec![];

    // Readers
    for job_id in job_ids.iter() {
        let scheduler = scheduler.clone();
        let job_id = *job_id;
        let handle = tokio::spawn(async move {
            scheduler.get_job(job_id).await
        });
        handles.push(handle);
    }

    // Writers
    for i in 10..15 {
        let scheduler = scheduler.clone();
        let handle = tokio::spawn(async move {
            let job = Job::builder()
                .user_id(&format!("newuser{}", i))
                .gpu_count(1)
                .build()
                .unwrap();
            scheduler.submit_job(job).await
        });
        handles.push(handle);
    }

    // Wait for all operations
    let results: Vec<_> = futures::future::join_all(handles).await;

    // All should succeed
    for result in results {
        assert!(result.is_ok());
        assert!(result.unwrap().is_ok());
    }

    test_app.cleanup().await;
}

#[tokio::test]
async fn test_concurrent_cancellations() {
    let test_app = TestApp::new().await;

    // Create jobs
    let repository = scheduler::db::JobRepository::new(test_app.pool.clone());
    let mut job_ids = vec![];

    for i in 0..5 {
        let job = Job::builder()
            .user_id(&format!("user{}", i))
            .gpu_count(2)
            .build()
            .unwrap();
        let created = repository.create(&job).await.unwrap();
        job_ids.push(created.id);
    }

    let scheduler = test_app.scheduler.clone();

    // Try to cancel all jobs concurrently
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
    }

    test_app.cleanup().await;
}

#[tokio::test]
async fn test_queue_stats_under_load() {
    let test_app = TestApp::new().await;

    let scheduler = test_app.scheduler.clone();

    // Submit jobs while reading stats
    let mut submit_handles = vec![];
    let mut stats_handles = vec![];

    // Submit 20 jobs
    for i in 0..20 {
        let scheduler = scheduler.clone();
        let handle = tokio::spawn(async move {
            let job = Job::builder()
                .user_id(&format!("user{}", i % 5))
                .gpu_count(i % 4 + 1)
                .priority(match i % 3 {
                    0 => Priority::High,
                    1 => Priority::Normal,
                    _ => Priority::Low,
                })
                .build()
                .unwrap();

            scheduler.submit_job(job).await
        });
        submit_handles.push(handle);
    }

    // Read stats concurrently
    for _ in 0..10 {
        let scheduler = scheduler.clone();
        let handle = tokio::spawn(async move {
            scheduler.get_queue_stats().await
        });
        stats_handles.push(handle);
    }

    // Wait for all submissions
    let submit_results: Vec<_> = futures::future::join_all(submit_handles).await;
    for result in submit_results {
        assert!(result.is_ok());
        assert!(result.unwrap().is_ok());
    }

    // Wait for all stats reads
    let stats_results: Vec<_> = futures::future::join_all(stats_handles).await;
    for result in stats_results {
        assert!(result.is_ok());
        assert!(result.unwrap().is_ok());
    }

    // Final stats should show all jobs
    let final_stats = scheduler.get_queue_stats().await.unwrap();
    assert_eq!(final_stats.total, 20);

    test_app.cleanup().await;
}
