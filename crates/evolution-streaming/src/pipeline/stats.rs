//! Pipeline execution statistics

use std::time::Duration;

/// Pipeline execution statistics
#[derive(Debug, Default, Clone)]
pub struct PipelineStats {
    pub cycles_completed: u64,
    pub agents_processed: u64,
    pub mutations_generated: u64,
    pub evaluations_completed: u64,
    pub archive_updates: u64,
    pub total_processing_time: Duration,
    pub average_cycle_time: Duration,
    pub throughput_agents_per_sec: f64,
}
