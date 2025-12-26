//! Evolution cycle result types

use std::time::Duration;

/// Result of a single evolution cycle
#[derive(Debug, Default, Clone)]
pub struct EvolutionCycleResult {
    pub total_time: Duration,
    pub selection_time: Duration,
    pub mutation_time: Duration,
    pub evaluation_time: Duration,
    pub archive_time: Duration,
    pub selected_count: usize,
    pub mutated_count: usize,
    pub evaluated_count: usize,
    pub novel_agents: usize,
}
