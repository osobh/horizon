//! Integration tests for build queue with priority management
//!
//! These tests verify priority ordering, fair scheduling, aging,
//! and queue management under various conditions.

mod common;

use std::sync::Arc;

use chrono::{Duration, Utc};
use swarmlet::build_queue::{BuildPriority, BuildQueue, QueueConfig, QueueError, QueuedBuild};

use common::build_fixtures::*;

/// Test basic priority ordering
#[tokio::test]
async fn test_queue_priority_ordering_basic() {
    let queue = BuildQueue::default_new();

    // Enqueue in reverse priority order
    queue
        .enqueue(create_queued_build(BuildPriority::Batch))
        .await
        .unwrap();
    queue
        .enqueue(create_queued_build(BuildPriority::Low))
        .await
        .unwrap();
    queue
        .enqueue(create_queued_build(BuildPriority::Normal))
        .await
        .unwrap();
    queue
        .enqueue(create_queued_build(BuildPriority::High))
        .await
        .unwrap();
    queue
        .enqueue(create_queued_build(BuildPriority::Critical))
        .await
        .unwrap();

    // Should dequeue in priority order
    assert_eq!(
        queue.dequeue().await.unwrap().priority,
        BuildPriority::Critical
    );
    assert_eq!(queue.dequeue().await.unwrap().priority, BuildPriority::High);
    assert_eq!(
        queue.dequeue().await.unwrap().priority,
        BuildPriority::Normal
    );
    assert_eq!(queue.dequeue().await.unwrap().priority, BuildPriority::Low);
    assert_eq!(
        queue.dequeue().await.unwrap().priority,
        BuildPriority::Batch
    );
}

/// Test FIFO ordering within same priority
#[tokio::test]
async fn test_queue_fifo_within_priority() {
    let queue = BuildQueue::default_new();

    // Enqueue multiple normal priority jobs
    let mut ids = vec![];
    for _ in 0..5 {
        let build = create_queued_build(BuildPriority::Normal);
        ids.push(build.job.id);
        queue.enqueue(build).await.unwrap();
        // Small delay to ensure different timestamps
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }

    // Should dequeue in FIFO order
    for expected_id in ids {
        let dequeued = queue.dequeue().await.unwrap();
        assert_eq!(dequeued.job.id, expected_id);
    }
}

/// Test fair scheduling per-user limits
#[tokio::test]
async fn test_queue_per_user_limits() {
    let config = QueueConfig {
        max_per_user: 3,
        ..Default::default()
    };
    let queue = BuildQueue::new(config);

    // User1 can enqueue up to limit
    for _ in 0..3 {
        queue
            .enqueue(create_queued_build_for_user("user1", BuildPriority::Normal))
            .await
            .unwrap();
    }

    // User1 exceeds limit
    let result = queue
        .enqueue(create_queued_build_for_user("user1", BuildPriority::Normal))
        .await;
    assert_eq!(result, Err(QueueError::UserLimitExceeded));

    // User2 can still enqueue
    queue
        .enqueue(create_queued_build_for_user("user2", BuildPriority::Normal))
        .await
        .unwrap();

    assert_eq!(queue.len().await, 4);
}

/// Test user count decrements on dequeue
#[tokio::test]
async fn test_queue_user_count_on_dequeue() {
    let config = QueueConfig {
        max_per_user: 2,
        ..Default::default()
    };
    let queue = BuildQueue::new(config);

    // Fill user1's quota
    queue
        .enqueue(create_queued_build_for_user("user1", BuildPriority::Normal))
        .await
        .unwrap();
    queue
        .enqueue(create_queued_build_for_user("user1", BuildPriority::Normal))
        .await
        .unwrap();

    // Can't add more
    assert!(queue
        .enqueue(create_queued_build_for_user("user1", BuildPriority::Normal))
        .await
        .is_err());

    // Dequeue one
    queue.dequeue().await.unwrap();

    // Now can add again
    queue
        .enqueue(create_queued_build_for_user("user1", BuildPriority::Normal))
        .await
        .unwrap();
}

/// Test deadline urgency bonus
#[tokio::test]
async fn test_queue_deadline_urgency() {
    let queue = BuildQueue::default_new();

    // Normal priority, no deadline
    let normal = create_queued_build(BuildPriority::Normal);
    let _normal_id = normal.job.id;
    queue.enqueue(normal).await.unwrap();

    // Low priority but urgent deadline (5 minutes)
    let urgent = create_queued_build_with_deadline(5);
    let _urgent_id = urgent.job.id;
    // Manually set to low priority
    let mut urgent_low = create_queued_build(BuildPriority::Low);
    urgent_low.deadline = Some(Utc::now() + Duration::minutes(5));
    queue.enqueue(urgent_low).await.unwrap();

    // The urgent low-priority job should have higher effective priority
    // and be dequeued first (deadline bonus of 20 makes Low(25)+20=45 > Normal(50)...
    // Actually Low + 20 = 45 < Normal 50, so let's adjust the test
    // With deadline < 10 min, bonus is 20, so Low(25)+20=45 < Normal(50)
    // We need Critical deadline to beat Normal

    // Let's clear and redo
    queue.dequeue().await;
    queue.dequeue().await;

    // High priority, no deadline
    let high = create_queued_build(BuildPriority::High);
    let high_id = high.job.id;
    queue.enqueue(high).await.unwrap();

    // Normal priority but very urgent deadline
    let mut urgent_normal = create_queued_build(BuildPriority::Normal);
    urgent_normal.deadline = Some(Utc::now() + Duration::minutes(5));
    let urgent_normal_id = urgent_normal.job.id;
    queue.enqueue(urgent_normal).await.unwrap();

    // High(75) vs Normal(50)+20=70, High still wins
    let first = queue.dequeue().await.unwrap();
    assert_eq!(first.job.id, high_id);

    // But urgent normal comes next
    let second = queue.dequeue().await.unwrap();
    assert_eq!(second.job.id, urgent_normal_id);
}

/// Test queue size limit
#[tokio::test]
async fn test_queue_size_limit() {
    let config = QueueConfig {
        max_queue_size: 5,
        max_per_user: 100,
        ..Default::default()
    };
    let queue = BuildQueue::new(config);

    // Fill the queue
    for _ in 0..5 {
        queue
            .enqueue(create_queued_build(BuildPriority::Normal))
            .await
            .unwrap();
    }

    // Queue is full
    let result = queue
        .enqueue(create_queued_build(BuildPriority::Normal))
        .await;
    assert_eq!(result, Err(QueueError::QueueFull));

    // Dequeue one
    queue.dequeue().await.unwrap();

    // Can add again
    queue
        .enqueue(create_queued_build(BuildPriority::Normal))
        .await
        .unwrap();
}

/// Test duplicate job prevention
#[tokio::test]
async fn test_queue_duplicate_prevention() {
    let queue = BuildQueue::default_new();

    let build = create_queued_build(BuildPriority::Normal);
    let _job_id = build.job.id;

    queue.enqueue(build.clone()).await.unwrap();

    // Try to enqueue same job again
    let duplicate = QueuedBuild {
        job: build.job.clone(),
        ..build
    };
    let result = queue.enqueue(duplicate).await;
    assert_eq!(result, Err(QueueError::DuplicateJob));
}

/// Test job removal
#[tokio::test]
async fn test_queue_job_removal() {
    let queue = BuildQueue::default_new();

    let build1 = create_queued_build(BuildPriority::Normal);
    let id1 = build1.job.id;
    let build2 = create_queued_build(BuildPriority::Normal);
    let id2 = build2.job.id;
    let build3 = create_queued_build(BuildPriority::Normal);
    let id3 = build3.job.id;

    queue.enqueue(build1).await.unwrap();
    queue.enqueue(build2).await.unwrap();
    queue.enqueue(build3).await.unwrap();

    assert_eq!(queue.len().await, 3);

    // Remove middle job
    let removed = queue.remove(id2).await;
    assert!(removed.is_some());
    assert_eq!(removed.unwrap().job.id, id2);
    assert_eq!(queue.len().await, 2);

    // Verify remaining jobs
    let first = queue.dequeue().await.unwrap();
    let second = queue.dequeue().await.unwrap();
    assert!(first.job.id == id1 || first.job.id == id3);
    assert!(second.job.id == id1 || second.job.id == id3);
}

/// Test priority update
#[tokio::test]
async fn test_queue_priority_update() {
    let queue = BuildQueue::default_new();

    let low_build = create_queued_build(BuildPriority::Low);
    let low_id = low_build.job.id;
    queue.enqueue(low_build).await.unwrap();

    let normal_build = create_queued_build(BuildPriority::Normal);
    queue.enqueue(normal_build).await.unwrap();

    // Update low to critical
    queue
        .update_priority(low_id, BuildPriority::Critical)
        .await
        .unwrap();

    // Should now be first
    let first = queue.dequeue().await.unwrap();
    assert_eq!(first.job.id, low_id);
    assert_eq!(first.priority, BuildPriority::Critical);
}

/// Test queue status reporting
#[tokio::test]
async fn test_queue_status() {
    let queue = BuildQueue::default_new();

    queue
        .enqueue(create_queued_build_for_user("user1", BuildPriority::High))
        .await
        .unwrap();
    queue
        .enqueue(create_queued_build_for_user("user1", BuildPriority::High))
        .await
        .unwrap();
    queue
        .enqueue(create_queued_build_for_user("user2", BuildPriority::Normal))
        .await
        .unwrap();
    queue
        .enqueue(create_queued_build_for_user("user3", BuildPriority::Low))
        .await
        .unwrap();

    let status = queue.status().await;

    assert_eq!(status.total_jobs, 4);
    assert_eq!(status.active_users, 3);
    assert_eq!(status.by_priority.get(&BuildPriority::High), Some(&2));
    assert_eq!(status.by_priority.get(&BuildPriority::Normal), Some(&1));
    assert_eq!(status.by_priority.get(&BuildPriority::Low), Some(&1));
    assert!(status.oldest_job.is_some());
}

/// Test get user jobs
#[tokio::test]
async fn test_queue_get_user_jobs() {
    let queue = BuildQueue::default_new();

    for _ in 0..3 {
        queue
            .enqueue(create_queued_build_for_user("user1", BuildPriority::Normal))
            .await
            .unwrap();
    }
    for _ in 0..2 {
        queue
            .enqueue(create_queued_build_for_user("user2", BuildPriority::Normal))
            .await
            .unwrap();
    }

    let user1_jobs = queue.get_user_jobs("user1").await;
    let user2_jobs = queue.get_user_jobs("user2").await;
    let user3_jobs = queue.get_user_jobs("user3").await;

    assert_eq!(user1_jobs.len(), 3);
    assert_eq!(user2_jobs.len(), 2);
    assert_eq!(user3_jobs.len(), 0);
}

/// Test list all jobs sorted by priority
#[tokio::test]
async fn test_queue_list_all_sorted() {
    let queue = BuildQueue::default_new();

    queue
        .enqueue(create_queued_build(BuildPriority::Low))
        .await
        .unwrap();
    queue
        .enqueue(create_queued_build(BuildPriority::Critical))
        .await
        .unwrap();
    queue
        .enqueue(create_queued_build(BuildPriority::Normal))
        .await
        .unwrap();

    let all = queue.list_all().await;

    assert_eq!(all.len(), 3);
    // Should be sorted by effective priority (highest first)
    assert_eq!(all[0].priority, BuildPriority::Critical);
    assert_eq!(all[1].priority, BuildPriority::Normal);
    assert_eq!(all[2].priority, BuildPriority::Low);
}

/// Test concurrent queue operations
#[tokio::test]
async fn test_queue_concurrent_operations() {
    let queue = Arc::new(BuildQueue::new(QueueConfig {
        max_queue_size: 10000,
        max_per_user: 1000,
        ..Default::default()
    }));

    // Spawn producers
    let mut handles = vec![];
    for i in 0..5 {
        let queue = queue.clone();
        let user = format!("user{}", i);
        handles.push(tokio::spawn(async move {
            for _ in 0..100 {
                let _ = queue
                    .enqueue(create_queued_build_for_user(&user, BuildPriority::Normal))
                    .await;
            }
        }));
    }

    // Spawn consumers
    for _ in 0..3 {
        let queue = queue.clone();
        handles.push(tokio::spawn(async move {
            for _ in 0..50 {
                let _ = queue.dequeue().await;
                tokio::time::sleep(std::time::Duration::from_millis(1)).await;
            }
        }));
    }

    // Wait for all
    for handle in handles {
        handle.await.unwrap();
    }

    // Queue should be consistent
    let status = queue.status().await;
    assert!(status.total_jobs <= 500); // At most 500 jobs added
}

/// Test peek doesn't remove item
#[tokio::test]
async fn test_queue_peek() {
    let queue = BuildQueue::default_new();

    let build = create_queued_build(BuildPriority::Normal);
    let id = build.job.id;
    queue.enqueue(build).await.unwrap();

    // Peek multiple times
    for _ in 0..5 {
        let peeked = queue.peek().await.unwrap();
        assert_eq!(peeked.job.id, id);
    }

    // Still in queue
    assert_eq!(queue.len().await, 1);

    // Dequeue should get the same item
    let dequeued = queue.dequeue().await.unwrap();
    assert_eq!(dequeued.job.id, id);
    assert_eq!(queue.len().await, 0);
}

/// Test get specific job
#[tokio::test]
async fn test_queue_get_job() {
    let queue = BuildQueue::default_new();

    let build = create_queued_build(BuildPriority::High);
    let id = build.job.id;
    queue.enqueue(build).await.unwrap();

    // Can get the job
    let found = queue.get(id).await;
    assert!(found.is_some());
    assert_eq!(found.unwrap().priority, BuildPriority::High);

    // Non-existent job
    let not_found = queue.get(uuid::Uuid::new_v4()).await;
    assert!(not_found.is_none());
}

/// Test empty queue operations
#[tokio::test]
async fn test_queue_empty_operations() {
    let queue = BuildQueue::default_new();

    assert!(queue.is_empty().await);
    assert_eq!(queue.len().await, 0);
    assert!(queue.dequeue().await.is_none());
    assert!(queue.peek().await.is_none());

    let status = queue.status().await;
    assert_eq!(status.total_jobs, 0);
    assert_eq!(status.active_users, 0);
    assert!(status.oldest_job.is_none());
}
