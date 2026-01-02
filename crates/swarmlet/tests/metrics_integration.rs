//! Integration tests for build metrics collection
//!
//! These tests verify that metrics are correctly collected, aggregated,
//! and reported across multiple builds.

mod common;

use chrono::Duration;
use swarmlet::build_metrics::{BuildOutcome, CommandType, ProfileType};
use swarmlet::{create_metrics_collector, BuildMetricsCollector};

use common::build_fixtures::*;

/// Test that metrics collector correctly tracks build counts
#[tokio::test]
async fn test_metrics_build_count_tracking() {
    let collector = BuildMetricsCollector::new(100);

    // Record several builds
    for _ in 0..5 {
        collector
            .record_build(create_build_record(BuildOutcome::Success, 30.0))
            .await;
    }
    for _ in 0..3 {
        collector
            .record_build(create_build_record(BuildOutcome::Failed, 20.0))
            .await;
    }

    let stats = collector.get_stats().await;
    assert_eq!(stats.total_builds, 8);
    assert_eq!(stats.successful_builds, 5);
    assert_eq!(stats.failed_builds, 3);
}

/// Test metrics aggregation with diverse build records
#[tokio::test]
async fn test_metrics_diverse_aggregation() {
    let collector = BuildMetricsCollector::new(100);

    // Record diverse builds
    for record in create_diverse_build_records() {
        collector.record_build(record).await;
    }

    let stats = collector.get_stats().await;

    // Check total counts
    assert_eq!(stats.total_builds, 8);

    // Check per-command breakdown
    // 2 explicit Build + 2 from create_build_record_with_cache (which uses Build by default) = 4
    let build_stats = stats.by_command.get(&CommandType::Build).unwrap();
    assert_eq!(build_stats.count, 4); // 4 Build commands

    let test_stats = stats.by_command.get(&CommandType::Test).unwrap();
    assert_eq!(test_stats.count, 2); // 2 Test commands

    // Check that timed out builds are counted
    assert_eq!(stats.timed_out_builds, 1);
}

/// Test cache efficiency tracking
#[tokio::test]
async fn test_metrics_cache_efficiency() {
    let collector = BuildMetricsCollector::new(100);

    // Record builds with varying cache performance
    collector
        .record_build(create_build_record_with_cache(
            BuildOutcome::Success,
            30.0,
            50,
            10,
        ))
        .await;
    collector
        .record_build(create_build_record_with_cache(
            BuildOutcome::Success,
            40.0,
            30,
            20,
        ))
        .await;
    collector
        .record_build(create_build_record_with_cache(
            BuildOutcome::Success,
            20.0,
            20,
            30,
        ))
        .await;

    let stats = collector.get_stats().await;

    // Total cache hits: 50 + 30 + 20 = 100
    assert_eq!(stats.total_cache_hits, 100);
    // Total cache misses: 10 + 20 + 30 = 60
    assert_eq!(stats.total_cache_misses, 60);

    // Check snapshot cache efficiency
    let snapshot = collector.get_snapshot().await;
    let expected_efficiency = 100.0 / 160.0; // 100 hits / (100 hits + 60 misses)
    assert!((snapshot.cache_efficiency - expected_efficiency).abs() < 0.001);
}

/// Test time-windowed summaries
#[tokio::test]
async fn test_metrics_time_window_summary() {
    let collector = BuildMetricsCollector::new(100);

    // Record some builds
    for i in 0..10 {
        let outcome = if i % 3 == 0 {
            BuildOutcome::Failed
        } else {
            BuildOutcome::Success
        };
        collector
            .record_build(create_build_record(outcome, 30.0 + i as f64 * 5.0))
            .await;
    }

    // Get summary for last hour (all builds should be included)
    let summary = collector.get_summary(Duration::hours(1)).await;

    assert_eq!(summary.total_builds, 10);
    // 4 failed (0, 3, 6, 9), 6 success
    assert_eq!(summary.failed_builds, 4);
    assert_eq!(summary.successful_builds, 6);
    assert!((summary.success_rate - 0.6).abs() < 0.001);
}

/// Test that max records limit is enforced
#[tokio::test]
async fn test_metrics_max_records_limit() {
    let max_records = 50;
    let collector = BuildMetricsCollector::new(max_records);

    // Record more than max
    for _ in 0..100 {
        collector
            .record_build(create_build_record(BuildOutcome::Success, 30.0))
            .await;
    }

    // Recent records should be limited
    let records = collector.get_recent_records(1000).await;
    assert_eq!(records.len(), max_records);

    // But total stats should still reflect all builds
    let stats = collector.get_stats().await;
    assert_eq!(stats.total_builds, 100);
}

/// Test metrics snapshot provides comprehensive view
#[tokio::test]
async fn test_metrics_snapshot_comprehensive() {
    let collector = create_metrics_collector(100);

    // Record a mix of builds
    for record in create_build_record_batch(20, 0.8) {
        collector.record_build(record).await;
    }

    let snapshot = collector.get_snapshot().await;

    // Verify all snapshot fields are populated
    assert_eq!(snapshot.stats.total_builds, 20);
    assert!((snapshot.success_rate - 0.8).abs() < 0.001);
    assert!(snapshot.avg_build_time_seconds > 0.0);
    assert!(snapshot.last_hour.total_builds > 0);
    assert!(snapshot.last_24h.total_builds > 0);
}

/// Test duration statistics accuracy
#[tokio::test]
async fn test_metrics_duration_statistics() {
    let collector = BuildMetricsCollector::new(100);

    // Record builds with known durations
    let durations = vec![10.0, 20.0, 30.0, 40.0, 50.0];
    for duration in &durations {
        collector
            .record_build(create_build_record(BuildOutcome::Success, *duration))
            .await;
    }

    let stats = collector.get_stats().await;

    // Average should be 30
    assert!((stats.avg_duration_seconds - 30.0).abs() < 0.001);
    // Min should be 10
    assert!((stats.min_duration_seconds - 10.0).abs() < 0.001);
    // Max should be 50
    assert!((stats.max_duration_seconds - 50.0).abs() < 0.001);
}

/// Test resource usage aggregation
#[tokio::test]
async fn test_metrics_resource_usage_aggregation() {
    let collector = BuildMetricsCollector::new(100);

    // Record builds with resource usage
    for _ in 0..5 {
        let mut record = create_build_record(BuildOutcome::Success, 60.0);
        record.resource_usage.crates_compiled = 20;
        record.resource_usage.cpu_seconds = 50.0;
        collector.record_build(record).await;
    }

    let stats = collector.get_stats().await;

    // Total crates compiled: 5 * 20 = 100
    assert_eq!(stats.total_crates_compiled, 100);
    // Total CPU seconds: 5 * 50 = 250
    assert!((stats.total_cpu_seconds - 250.0).abs() < 0.001);
}

/// Test reset functionality
#[tokio::test]
async fn test_metrics_reset() {
    let collector = BuildMetricsCollector::new(100);

    // Record some builds
    for _ in 0..10 {
        collector
            .record_build(create_build_record(BuildOutcome::Success, 30.0))
            .await;
    }

    assert_eq!(collector.get_stats().await.total_builds, 10);

    // Reset
    collector.reset().await;

    let stats = collector.get_stats().await;
    assert_eq!(stats.total_builds, 0);
    assert_eq!(stats.successful_builds, 0);
    assert!(collector.get_recent_records(100).await.is_empty());
}

/// Test per-profile statistics
#[tokio::test]
async fn test_metrics_per_profile_stats() {
    let collector = BuildMetricsCollector::new(100);

    // Record debug builds
    for _ in 0..5 {
        let mut record = create_build_record(BuildOutcome::Success, 30.0);
        record.profile = ProfileType::Debug;
        collector.record_build(record).await;
    }

    // Record release builds (typically longer)
    for _ in 0..3 {
        let mut record = create_build_record(BuildOutcome::Success, 90.0);
        record.profile = ProfileType::Release;
        collector.record_build(record).await;
    }

    let stats = collector.get_stats().await;

    let debug_stats = stats.by_profile.get(&ProfileType::Debug).unwrap();
    assert_eq!(debug_stats.count, 5);
    assert!((debug_stats.avg_duration_seconds - 30.0).abs() < 0.001);

    let release_stats = stats.by_profile.get(&ProfileType::Release).unwrap();
    assert_eq!(release_stats.count, 3);
    assert!((release_stats.avg_duration_seconds - 90.0).abs() < 0.001);
}

/// Test concurrent metric recording
#[tokio::test]
async fn test_metrics_concurrent_recording() {
    use std::sync::Arc;

    let collector = Arc::new(BuildMetricsCollector::new(1000));

    // Spawn multiple tasks recording metrics concurrently
    let mut handles = vec![];
    for _ in 0..10 {
        let collector = collector.clone();
        handles.push(tokio::spawn(async move {
            for _ in 0..100 {
                collector
                    .record_build(create_build_record(BuildOutcome::Success, 30.0))
                    .await;
            }
        }));
    }

    // Wait for all tasks
    for handle in handles {
        handle.await.unwrap();
    }

    // Should have recorded all builds
    let stats = collector.get_stats().await;
    assert_eq!(stats.total_builds, 1000);
}
