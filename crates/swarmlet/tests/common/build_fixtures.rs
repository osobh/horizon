//! Build job fixtures and helpers for integration tests

use chrono::{Duration, Utc};
use std::path::PathBuf;
use uuid::Uuid;

use swarmlet::build_job::{
    BuildJob, BuildProfile, BuildResourceLimits, BuildResourceUsage, BuildSource, CargoCommand,
};
use swarmlet::build_metrics::{BuildOutcome, BuildRecord, CommandType, ProfileType};
use swarmlet::build_queue::{BuildPriority, QueuedBuild};

/// Create a test build job with sensible defaults
pub fn create_test_job() -> BuildJob {
    BuildJob::new(
        CargoCommand::Build,
        BuildSource::Local {
            path: PathBuf::from("/tmp/test-project"),
        },
    )
}

/// Create a test build job with specific command
pub fn create_test_job_with_command(command: CargoCommand) -> BuildJob {
    BuildJob::new(
        command,
        BuildSource::Local {
            path: PathBuf::from("/tmp/test-project"),
        },
    )
}

/// Create a test build job with specific profile
pub fn create_test_job_with_profile(profile: BuildProfile) -> BuildJob {
    BuildJob::new(
        CargoCommand::Build,
        BuildSource::Local {
            path: PathBuf::from("/tmp/test-project"),
        },
    )
    .with_profile(profile)
}

/// Create a test build job with resource limits
pub fn create_test_job_with_limits(cpu_cores: f32, memory_gb: u64) -> BuildJob {
    let limits = BuildResourceLimits {
        cpu_cores: Some(cpu_cores),
        memory_bytes: Some(memory_gb * 1024 * 1024 * 1024),
        disk_bytes: Some(10 * 1024 * 1024 * 1024),
        timeout_seconds: Some(600),
    };
    BuildJob::new(
        CargoCommand::Build,
        BuildSource::Local {
            path: PathBuf::from("/tmp/test-project"),
        },
    )
    .with_resource_limits(limits)
}

/// Create a queued build with specific priority
pub fn create_queued_build(priority: BuildPriority) -> QueuedBuild {
    QueuedBuild::new(create_test_job()).with_priority(priority)
}

/// Create a queued build with user ID for fair scheduling tests
pub fn create_queued_build_for_user(user_id: &str, priority: BuildPriority) -> QueuedBuild {
    QueuedBuild::new(create_test_job())
        .with_priority(priority)
        .with_user(user_id.to_string())
}

/// Create a queued build with deadline
pub fn create_queued_build_with_deadline(minutes_from_now: i64) -> QueuedBuild {
    let deadline = Utc::now() + Duration::minutes(minutes_from_now);
    QueuedBuild::new(create_test_job())
        .with_priority(BuildPriority::Normal)
        .with_deadline(deadline)
}

/// Create a build record for metrics tests
pub fn create_build_record(outcome: BuildOutcome, duration_secs: f64) -> BuildRecord {
    BuildRecord {
        job_id: Uuid::new_v4(),
        command: CommandType::Build,
        profile: ProfileType::Debug,
        status: outcome,
        started_at: Utc::now() - Duration::seconds(duration_secs as i64),
        completed_at: Utc::now(),
        duration_seconds: duration_secs,
        resource_usage: create_test_resource_usage(duration_secs),
        cache_enabled: true,
        source_cache_hit: false,
        toolchain: "stable".to_string(),
    }
}

/// Create a build record with specific command type
pub fn create_build_record_with_command(
    command: CommandType,
    outcome: BuildOutcome,
    duration_secs: f64,
) -> BuildRecord {
    let mut record = create_build_record(outcome, duration_secs);
    record.command = command;
    record
}

/// Create a build record with cache hit
pub fn create_build_record_with_cache(
    outcome: BuildOutcome,
    duration_secs: f64,
    cache_hits: u32,
    cache_misses: u32,
) -> BuildRecord {
    let mut record = create_build_record(outcome, duration_secs);
    record.resource_usage.cache_hits = cache_hits;
    record.resource_usage.cache_misses = cache_misses;
    record
}

/// Create test resource usage data
pub fn create_test_resource_usage(compile_time: f64) -> BuildResourceUsage {
    BuildResourceUsage {
        cpu_seconds: compile_time * 0.8,
        peak_memory_mb: 512.0,
        disk_read_bytes: 50 * 1024 * 1024,
        disk_write_bytes: 100 * 1024 * 1024,
        compile_time_seconds: compile_time,
        crates_compiled: 25,
        cache_hits: 10,
        cache_misses: 15,
    }
}

/// Create multiple build records for batch testing
pub fn create_build_record_batch(count: usize, success_rate: f64) -> Vec<BuildRecord> {
    let success_count = (count as f64 * success_rate) as usize;
    let mut records = Vec::with_capacity(count);

    for i in 0..count {
        let outcome = if i < success_count {
            BuildOutcome::Success
        } else {
            BuildOutcome::Failed
        };
        let duration = 30.0 + (i as f64 * 5.0); // Varying durations
        records.push(create_build_record(outcome, duration));
    }

    records
}

/// Create a variety of build records for comprehensive testing
pub fn create_diverse_build_records() -> Vec<BuildRecord> {
    vec![
        create_build_record_with_command(CommandType::Build, BuildOutcome::Success, 60.0),
        create_build_record_with_command(CommandType::Test, BuildOutcome::Success, 120.0),
        create_build_record_with_command(CommandType::Build, BuildOutcome::Failed, 45.0),
        create_build_record_with_command(CommandType::Clippy, BuildOutcome::Success, 30.0),
        create_build_record_with_command(CommandType::Check, BuildOutcome::Success, 15.0),
        create_build_record_with_command(CommandType::Test, BuildOutcome::TimedOut, 300.0),
        create_build_record_with_cache(BuildOutcome::Success, 20.0, 50, 5),
        create_build_record_with_cache(BuildOutcome::Success, 80.0, 5, 50),
    ]
}
