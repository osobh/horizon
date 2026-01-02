//! Build Metrics and Statistics Tracking
//!
//! This module provides comprehensive metrics collection and aggregation for build jobs.
//! It tracks success rates, build durations, resource usage, and cache efficiency.

use chrono::{DateTime, Duration, Timelike, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::build_job::{BuildJobStatus, BuildProfile, BuildResourceUsage, CargoCommand};

/// Build metrics collector
pub struct BuildMetricsCollector {
    /// Historical build records
    records: Arc<RwLock<Vec<BuildRecord>>>,
    /// Aggregated statistics
    stats: Arc<RwLock<AggregatedStats>>,
    /// Maximum records to retain
    max_records: usize,
}

/// Record of a completed build for metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildRecord {
    /// Build job ID
    pub job_id: Uuid,
    /// Command type
    pub command: CommandType,
    /// Build profile
    pub profile: ProfileType,
    /// Final status
    pub status: BuildOutcome,
    /// When the build started
    pub started_at: DateTime<Utc>,
    /// When the build completed
    pub completed_at: DateTime<Utc>,
    /// Total duration in seconds
    pub duration_seconds: f64,
    /// Resource usage
    pub resource_usage: BuildResourceUsage,
    /// Whether cache was used
    pub cache_enabled: bool,
    /// Source cache hit (reused cached source)
    pub source_cache_hit: bool,
    /// Toolchain used
    pub toolchain: String,
}

/// Simplified command type for metrics
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum CommandType {
    Build,
    Test,
    Check,
    Clippy,
    Doc,
    Run,
    Bench,
}

impl From<&CargoCommand> for CommandType {
    fn from(cmd: &CargoCommand) -> Self {
        match cmd {
            CargoCommand::Build => CommandType::Build,
            CargoCommand::Test { .. } => CommandType::Test,
            CargoCommand::Check => CommandType::Check,
            CargoCommand::Clippy { .. } => CommandType::Clippy,
            CargoCommand::Doc { .. } => CommandType::Doc,
            CargoCommand::Run { .. } => CommandType::Run,
            CargoCommand::Bench { .. } => CommandType::Bench,
        }
    }
}

/// Simplified profile type for metrics
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum ProfileType {
    Debug,
    Release,
    Custom,
}

impl From<&BuildProfile> for ProfileType {
    fn from(profile: &BuildProfile) -> Self {
        match profile {
            BuildProfile::Debug => ProfileType::Debug,
            BuildProfile::Release => ProfileType::Release,
            BuildProfile::Custom(_) => ProfileType::Custom,
        }
    }
}

/// Build outcome for metrics
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum BuildOutcome {
    Success,
    Failed,
    Cancelled,
    TimedOut,
}

impl From<&BuildJobStatus> for BuildOutcome {
    fn from(status: &BuildJobStatus) -> Self {
        match status {
            BuildJobStatus::Completed => BuildOutcome::Success,
            BuildJobStatus::Failed { .. } => BuildOutcome::Failed,
            BuildJobStatus::Cancelled => BuildOutcome::Cancelled,
            BuildJobStatus::TimedOut => BuildOutcome::TimedOut,
            _ => BuildOutcome::Failed, // Non-terminal statuses shouldn't be recorded
        }
    }
}

/// Aggregated build statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AggregatedStats {
    /// Total builds processed
    pub total_builds: u64,
    /// Successful builds
    pub successful_builds: u64,
    /// Failed builds
    pub failed_builds: u64,
    /// Cancelled builds
    pub cancelled_builds: u64,
    /// Timed out builds
    pub timed_out_builds: u64,
    /// Total build time in seconds
    pub total_build_time_seconds: f64,
    /// Average build duration in seconds
    pub avg_duration_seconds: f64,
    /// Minimum build duration
    pub min_duration_seconds: f64,
    /// Maximum build duration
    pub max_duration_seconds: f64,
    /// Total CPU seconds consumed
    pub total_cpu_seconds: f64,
    /// Total memory used (peak sum across builds) in MB
    pub total_peak_memory_mb: f64,
    /// Total disk bytes read
    pub total_disk_read_bytes: u64,
    /// Total disk bytes written
    pub total_disk_write_bytes: u64,
    /// Total crates compiled
    pub total_crates_compiled: u64,
    /// Total cache hits
    pub total_cache_hits: u64,
    /// Total cache misses
    pub total_cache_misses: u64,
    /// Source cache hits (reused source)
    pub source_cache_hits: u64,
    /// Per-command statistics
    pub by_command: HashMap<CommandType, CommandStats>,
    /// Per-profile statistics
    pub by_profile: HashMap<ProfileType, ProfileStats>,
    /// Hourly build counts (hour of day -> count)
    pub hourly_distribution: [u64; 24],
    /// When stats were last updated
    pub last_updated: Option<DateTime<Utc>>,
}

/// Statistics per command type
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CommandStats {
    pub count: u64,
    pub success_count: u64,
    pub total_duration_seconds: f64,
    pub avg_duration_seconds: f64,
}

/// Statistics per build profile
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProfileStats {
    pub count: u64,
    pub success_count: u64,
    pub total_duration_seconds: f64,
    pub avg_duration_seconds: f64,
}

impl BuildMetricsCollector {
    /// Create a new metrics collector
    pub fn new(max_records: usize) -> Self {
        Self {
            records: Arc::new(RwLock::new(Vec::new())),
            stats: Arc::new(RwLock::new(AggregatedStats::default())),
            max_records,
        }
    }

    /// Create with default settings (1000 records)
    pub fn default_new() -> Self {
        Self::new(1000)
    }

    /// Record a completed build
    pub async fn record_build(&self, record: BuildRecord) {
        // Update records
        {
            let mut records = self.records.write().await;
            records.push(record.clone());

            // Trim old records if necessary
            if records.len() > self.max_records {
                let to_remove = records.len() - self.max_records;
                records.drain(0..to_remove);
            }
        }

        // Update aggregated stats
        self.update_stats(&record).await;
    }

    /// Update aggregated statistics with a new record
    async fn update_stats(&self, record: &BuildRecord) {
        let mut stats = self.stats.write().await;

        stats.total_builds += 1;

        match record.status {
            BuildOutcome::Success => stats.successful_builds += 1,
            BuildOutcome::Failed => stats.failed_builds += 1,
            BuildOutcome::Cancelled => stats.cancelled_builds += 1,
            BuildOutcome::TimedOut => stats.timed_out_builds += 1,
        }

        // Duration stats
        stats.total_build_time_seconds += record.duration_seconds;
        stats.avg_duration_seconds =
            stats.total_build_time_seconds / stats.total_builds as f64;

        if stats.min_duration_seconds == 0.0 || record.duration_seconds < stats.min_duration_seconds
        {
            stats.min_duration_seconds = record.duration_seconds;
        }
        if record.duration_seconds > stats.max_duration_seconds {
            stats.max_duration_seconds = record.duration_seconds;
        }

        // Resource usage
        stats.total_cpu_seconds += record.resource_usage.cpu_seconds;
        stats.total_peak_memory_mb += record.resource_usage.peak_memory_mb as f64;
        stats.total_disk_read_bytes += record.resource_usage.disk_read_bytes;
        stats.total_disk_write_bytes += record.resource_usage.disk_write_bytes;
        stats.total_crates_compiled += record.resource_usage.crates_compiled as u64;
        stats.total_cache_hits += record.resource_usage.cache_hits as u64;
        stats.total_cache_misses += record.resource_usage.cache_misses as u64;

        if record.source_cache_hit {
            stats.source_cache_hits += 1;
        }

        // Per-command stats
        let cmd_stats = stats.by_command.entry(record.command).or_default();
        cmd_stats.count += 1;
        if record.status == BuildOutcome::Success {
            cmd_stats.success_count += 1;
        }
        cmd_stats.total_duration_seconds += record.duration_seconds;
        cmd_stats.avg_duration_seconds = cmd_stats.total_duration_seconds / cmd_stats.count as f64;

        // Per-profile stats
        let profile_stats = stats.by_profile.entry(record.profile).or_default();
        profile_stats.count += 1;
        if record.status == BuildOutcome::Success {
            profile_stats.success_count += 1;
        }
        profile_stats.total_duration_seconds += record.duration_seconds;
        profile_stats.avg_duration_seconds =
            profile_stats.total_duration_seconds / profile_stats.count as f64;

        // Hourly distribution
        let hour = record.started_at.hour() as usize;
        stats.hourly_distribution[hour] += 1;

        stats.last_updated = Some(Utc::now());
    }

    /// Get current aggregated statistics
    pub async fn get_stats(&self) -> AggregatedStats {
        self.stats.read().await.clone()
    }

    /// Get build summary for a time window
    pub async fn get_summary(&self, window: Duration) -> BuildSummary {
        let cutoff = Utc::now() - window;
        let records = self.records.read().await;

        let recent: Vec<_> = records
            .iter()
            .filter(|r| r.completed_at >= cutoff)
            .collect();

        let total = recent.len();
        let successful = recent.iter().filter(|r| r.status == BuildOutcome::Success).count();
        let failed = recent.iter().filter(|r| r.status == BuildOutcome::Failed).count();

        let total_duration: f64 = recent.iter().map(|r| r.duration_seconds).sum();
        let avg_duration = if total > 0 {
            total_duration / total as f64
        } else {
            0.0
        };

        let cache_hits: u64 = recent.iter().map(|r| r.resource_usage.cache_hits as u64).sum();
        let cache_misses: u64 = recent.iter().map(|r| r.resource_usage.cache_misses as u64).sum();
        let cache_hit_rate = if cache_hits + cache_misses > 0 {
            cache_hits as f64 / (cache_hits + cache_misses) as f64
        } else {
            0.0
        };

        BuildSummary {
            window_seconds: window.num_seconds(),
            total_builds: total as u64,
            successful_builds: successful as u64,
            failed_builds: failed as u64,
            success_rate: if total > 0 {
                successful as f64 / total as f64
            } else {
                0.0
            },
            avg_duration_seconds: avg_duration,
            cache_hit_rate,
            as_of: Utc::now(),
        }
    }

    /// Get recent build records
    pub async fn get_recent_records(&self, limit: usize) -> Vec<BuildRecord> {
        let records = self.records.read().await;
        records.iter().rev().take(limit).cloned().collect()
    }

    /// Get records for a specific job
    pub async fn get_record(&self, job_id: Uuid) -> Option<BuildRecord> {
        let records = self.records.read().await;
        records.iter().find(|r| r.job_id == job_id).cloned()
    }

    /// Reset all statistics
    pub async fn reset(&self) {
        let mut records = self.records.write().await;
        let mut stats = self.stats.write().await;
        records.clear();
        *stats = AggregatedStats::default();
    }
}

/// Summary of builds within a time window
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildSummary {
    /// Window size in seconds
    pub window_seconds: i64,
    /// Total builds in window
    pub total_builds: u64,
    /// Successful builds
    pub successful_builds: u64,
    /// Failed builds
    pub failed_builds: u64,
    /// Success rate (0.0 - 1.0)
    pub success_rate: f64,
    /// Average duration in seconds
    pub avg_duration_seconds: f64,
    /// Cache hit rate (0.0 - 1.0)
    pub cache_hit_rate: f64,
    /// When this summary was generated
    pub as_of: DateTime<Utc>,
}

/// Real-time metrics snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsSnapshot {
    /// Current aggregated stats
    pub stats: AggregatedStats,
    /// Last hour summary
    pub last_hour: BuildSummary,
    /// Last 24 hours summary
    pub last_24h: BuildSummary,
    /// Success rate (overall)
    pub success_rate: f64,
    /// Cache efficiency (hit rate)
    pub cache_efficiency: f64,
    /// Average build time
    pub avg_build_time_seconds: f64,
    /// Builds per hour (recent average)
    pub builds_per_hour: f64,
}

impl BuildMetricsCollector {
    /// Get a complete metrics snapshot
    pub async fn get_snapshot(&self) -> MetricsSnapshot {
        let stats = self.get_stats().await;
        let last_hour = self.get_summary(Duration::hours(1)).await;
        let last_24h = self.get_summary(Duration::hours(24)).await;

        let success_rate = if stats.total_builds > 0 {
            stats.successful_builds as f64 / stats.total_builds as f64
        } else {
            0.0
        };

        let cache_efficiency = if stats.total_cache_hits + stats.total_cache_misses > 0 {
            stats.total_cache_hits as f64
                / (stats.total_cache_hits + stats.total_cache_misses) as f64
        } else {
            0.0
        };

        let builds_per_hour = if stats.total_builds > 0 && stats.total_build_time_seconds > 0.0 {
            let hours = stats.total_build_time_seconds / 3600.0;
            if hours > 0.0 {
                stats.total_builds as f64 / hours
            } else {
                0.0
            }
        } else {
            0.0
        };

        MetricsSnapshot {
            stats: stats.clone(),
            last_hour,
            last_24h,
            success_rate,
            cache_efficiency,
            avg_build_time_seconds: stats.avg_duration_seconds,
            builds_per_hour,
        }
    }
}

/// Shared metrics collector type
pub type SharedBuildMetrics = Arc<BuildMetricsCollector>;

/// Create a shared metrics collector
pub fn create_metrics_collector(max_records: usize) -> SharedBuildMetrics {
    Arc::new(BuildMetricsCollector::new(max_records))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_record(status: BuildOutcome, duration: f64) -> BuildRecord {
        BuildRecord {
            job_id: Uuid::new_v4(),
            command: CommandType::Build,
            profile: ProfileType::Debug,
            status,
            started_at: Utc::now() - Duration::seconds(duration as i64),
            completed_at: Utc::now(),
            duration_seconds: duration,
            resource_usage: BuildResourceUsage {
                cpu_seconds: duration * 0.8,
                peak_memory_mb: 512.0,
                disk_read_bytes: 1024 * 1024,
                disk_write_bytes: 512 * 1024,
                compile_time_seconds: duration * 0.9,
                crates_compiled: 10,
                cache_hits: 5,
                cache_misses: 3,
            },
            cache_enabled: true,
            source_cache_hit: false,
            toolchain: "stable".to_string(),
        }
    }

    #[tokio::test]
    async fn test_record_build() {
        let collector = BuildMetricsCollector::new(100);
        let record = make_test_record(BuildOutcome::Success, 60.0);

        collector.record_build(record).await;

        let stats = collector.get_stats().await;
        assert_eq!(stats.total_builds, 1);
        assert_eq!(stats.successful_builds, 1);
    }

    #[tokio::test]
    async fn test_aggregated_stats() {
        let collector = BuildMetricsCollector::new(100);

        // Add several builds
        collector
            .record_build(make_test_record(BuildOutcome::Success, 30.0))
            .await;
        collector
            .record_build(make_test_record(BuildOutcome::Success, 60.0))
            .await;
        collector
            .record_build(make_test_record(BuildOutcome::Failed, 45.0))
            .await;

        let stats = collector.get_stats().await;
        assert_eq!(stats.total_builds, 3);
        assert_eq!(stats.successful_builds, 2);
        assert_eq!(stats.failed_builds, 1);
        assert_eq!(stats.avg_duration_seconds, 45.0);
    }

    #[tokio::test]
    async fn test_max_records() {
        let collector = BuildMetricsCollector::new(5);

        // Add more records than max
        for _ in 0..10 {
            collector
                .record_build(make_test_record(BuildOutcome::Success, 30.0))
                .await;
        }

        let records = collector.get_recent_records(100).await;
        assert_eq!(records.len(), 5);
    }

    #[tokio::test]
    async fn test_summary() {
        let collector = BuildMetricsCollector::new(100);

        collector
            .record_build(make_test_record(BuildOutcome::Success, 30.0))
            .await;
        collector
            .record_build(make_test_record(BuildOutcome::Success, 60.0))
            .await;

        let summary = collector.get_summary(Duration::hours(1)).await;
        assert_eq!(summary.total_builds, 2);
        assert_eq!(summary.success_rate, 1.0);
    }

    #[tokio::test]
    async fn test_command_stats() {
        let collector = BuildMetricsCollector::new(100);

        let mut record = make_test_record(BuildOutcome::Success, 30.0);
        record.command = CommandType::Test;
        collector.record_build(record).await;

        let stats = collector.get_stats().await;
        let test_stats = stats.by_command.get(&CommandType::Test).unwrap();
        assert_eq!(test_stats.count, 1);
        assert_eq!(test_stats.success_count, 1);
    }

    #[tokio::test]
    async fn test_reset() {
        let collector = BuildMetricsCollector::new(100);

        collector
            .record_build(make_test_record(BuildOutcome::Success, 30.0))
            .await;
        assert_eq!(collector.get_stats().await.total_builds, 1);

        collector.reset().await;
        assert_eq!(collector.get_stats().await.total_builds, 0);
    }
}
