//! Progress monitoring for GPU agents benchmarks

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Benchmark phases
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BenchmarkPhase {
    /// Not started yet
    NotStarted,
    /// Initial setup phase
    Initialization,
    /// System validation and GPU detection
    SystemCheck,
    /// Agent scalability testing
    ScalabilityTests,
    /// LLM integration testing
    LlmTests,
    /// Knowledge graph functionality testing  
    KnowledgeGraphTests,
    /// Evolution algorithm testing
    EvolutionTests,
    /// Final report generation
    ReportGeneration,
    /// Benchmark complete
    Complete,
    /// Custom phase for extensibility
    Custom(String),
}

/// Current progress state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressState {
    /// Current benchmark phase
    pub current_phase: BenchmarkPhase,
    /// Progress within current phase (0.0 to 1.0)
    pub phase_progress: f64,
    /// Overall progress across all phases (0.0 to 1.0)
    pub overall_progress: f64,
    /// Current test being executed
    pub current_test: String,
    /// Number of tests completed
    pub tests_completed: usize,
    /// Total number of tests
    pub total_tests: usize,
    /// Time elapsed since benchmark start
    pub elapsed_time: Duration,
    /// Estimated time remaining
    pub estimated_remaining: Duration,
    /// Current operation description
    pub current_operation: String,
}

/// Resource usage snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceSnapshot {
    /// Timestamp when snapshot was taken
    pub timestamp: u64,
    /// CPU usage percentage (0.0 to 100.0)
    pub cpu_usage_percent: f64,
    /// Memory usage in MB
    pub memory_usage_mb: f64,
    /// Total memory in MB
    pub memory_total_mb: f64,
    /// GPU usage percentage (0.0 to 100.0)
    pub gpu_usage_percent: f64,
    /// GPU memory used in MB
    pub gpu_memory_used_mb: f64,
    /// GPU memory total in MB
    pub gpu_memory_total_mb: f64,
    /// GPU temperature in Celsius
    pub gpu_temperature_c: f64,
    /// Disk I/O read rate in MB/s
    pub disk_io_read_mb_s: f64,
    /// Disk I/O write rate in MB/s
    pub disk_io_write_mb_s: f64,
}

/// Progress monitor for tracking benchmark execution
pub struct ProgressMonitor {
    /// Current progress state
    current_state: ProgressState,
    /// Start time of benchmark
    start_time: std::time::Instant,
}

impl ProgressMonitor {
    /// Create a new progress monitor
    pub fn new() -> Result<Self> {
        Ok(Self {
            current_state: ProgressState {
                current_phase: BenchmarkPhase::Initialization,
                phase_progress: 0.0,
                overall_progress: 0.0,
                current_test: "Initializing benchmark suite".to_string(),
                tests_completed: 0,
                total_tests: 0,
                elapsed_time: Duration::ZERO,
                estimated_remaining: Duration::ZERO,
                current_operation: "Starting benchmark".to_string(),
            },
            start_time: std::time::Instant::now(),
        })
    }

    /// Get current progress state
    pub fn get_state(&self) -> &ProgressState {
        &self.current_state
    }

    /// Update current phase
    pub fn set_phase(&mut self, phase: BenchmarkPhase) {
        self.current_state.current_phase = phase;
        self.update_elapsed_time();
    }

    /// Update phase progress
    pub fn set_phase_progress(&mut self, progress: f64) {
        self.current_state.phase_progress = progress.clamp(0.0, 1.0);
        self.update_elapsed_time();
    }

    /// Update overall progress
    pub fn set_overall_progress(&mut self, progress: f64) {
        self.current_state.overall_progress = progress.clamp(0.0, 1.0);
        self.update_elapsed_time();
        self.update_estimated_remaining();
    }

    /// Update current test information
    pub fn set_current_test(&mut self, test_name: &str, completed: usize, total: usize) {
        self.current_state.current_test = test_name.to_string();
        self.current_state.tests_completed = completed;
        self.current_state.total_tests = total;
        self.update_elapsed_time();
    }

    /// Update current operation
    pub fn set_current_operation(&mut self, operation: &str) {
        self.current_state.current_operation = operation.to_string();
        self.update_elapsed_time();
    }

    /// Update elapsed time
    fn update_elapsed_time(&mut self) {
        self.current_state.elapsed_time = self.start_time.elapsed();
    }

    /// Update estimated remaining time
    fn update_estimated_remaining(&mut self) {
        if self.current_state.overall_progress > 0.0 {
            let total_estimated =
                self.current_state.elapsed_time.as_secs_f64() / self.current_state.overall_progress;
            let remaining = total_estimated - self.current_state.elapsed_time.as_secs_f64();
            self.current_state.estimated_remaining = Duration::from_secs_f64(remaining.max(0.0));
        }
    }
}

impl Default for ProgressMonitor {
    fn default() -> Self {
        Self::new()?
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn test_progress_monitor_new() {
        // RED PHASE: Test will fail until ProgressMonitor is properly implemented
        let monitor = ProgressMonitor::new()?;

        assert_eq!(
            monitor.current_state.current_phase,
            BenchmarkPhase::Initialization
        );
        assert_eq!(monitor.current_state.phase_progress, 0.0);
        assert_eq!(monitor.current_state.overall_progress, 0.0);
        assert_eq!(monitor.current_state.tests_completed, 0);
        assert_eq!(monitor.current_state.total_tests, 0);
    }

    #[test]
    fn test_progress_monitor_set_phase() -> Result<(), Box<dyn std::error::Error>> {
        let mut monitor = ProgressMonitor::new()?;

        monitor.set_phase(BenchmarkPhase::SystemCheck);
        assert_eq!(
            monitor.current_state.current_phase,
            BenchmarkPhase::SystemCheck
        );

        monitor.set_phase(BenchmarkPhase::ScalabilityTests);
        assert_eq!(
            monitor.current_state.current_phase,
            BenchmarkPhase::ScalabilityTests
        );
    }

    #[test]
    fn test_progress_monitor_set_progress() -> Result<(), Box<dyn std::error::Error>> {
        let mut monitor = ProgressMonitor::new()?;

        // Test phase progress
        monitor.set_phase_progress(0.5);
        assert_eq!(monitor.current_state.phase_progress, 0.5);

        // Test clamping
        monitor.set_phase_progress(1.5);
        assert_eq!(monitor.current_state.phase_progress, 1.0);

        monitor.set_phase_progress(-0.5);
        assert_eq!(monitor.current_state.phase_progress, 0.0);

        // Test overall progress
        monitor.set_overall_progress(0.75);
        assert_eq!(monitor.current_state.overall_progress, 0.75);
    }

    #[test]
    fn test_progress_monitor_set_current_test() -> Result<(), Box<dyn std::error::Error>> {
        let mut monitor = ProgressMonitor::new()?;

        monitor.set_current_test("Test GPU initialization", 1, 5);
        assert_eq!(
            monitor.current_state.current_test,
            "Test GPU initialization"
        );
        assert_eq!(monitor.current_state.tests_completed, 1);
        assert_eq!(monitor.current_state.total_tests, 5);
    }

    #[test]
    fn test_progress_monitor_set_current_operation() -> Result<(), Box<dyn std::error::Error>> {
        let mut monitor = ProgressMonitor::new()?;

        monitor.set_current_operation("Spawning 1M agents");
        assert_eq!(
            monitor.current_state.current_operation,
            "Spawning 1M agents"
        );
    }

    #[test]
    fn test_progress_monitor_elapsed_time() -> Result<(), Box<dyn std::error::Error>> {
        let mut monitor = ProgressMonitor::new()?;

        // Sleep a tiny bit to ensure time passes
        std::thread::sleep(Duration::from_millis(1));
        monitor.set_phase_progress(0.1);

        assert!(monitor.current_state.elapsed_time > Duration::ZERO);
    }

    #[test]
    fn test_benchmark_phases() {
        // Test that all phases are distinct
        let phases = vec![
            BenchmarkPhase::Initialization,
            BenchmarkPhase::SystemCheck,
            BenchmarkPhase::ScalabilityTests,
            BenchmarkPhase::LlmTests,
            BenchmarkPhase::KnowledgeGraphTests,
            BenchmarkPhase::EvolutionTests,
            BenchmarkPhase::ReportGeneration,
            BenchmarkPhase::Complete,
        ];

        // Each phase should be unique
        for (i, phase1) in phases.iter().enumerate() {
            for (j, phase2) in phases.iter().enumerate() {
                if i != j {
                    assert_ne!(phase1, phase2);
                }
            }
        }
    }

    #[test]
    fn test_resource_snapshot_creation() {
        let snapshot = ResourceSnapshot {
            timestamp: 1234567890,
            cpu_usage_percent: 45.5,
            memory_usage_mb: 2048.0,
            memory_total_mb: 8192.0,
            gpu_usage_percent: 85.2,
            gpu_memory_used_mb: 6144.0,
            gpu_memory_total_mb: 8192.0,
            gpu_temperature_c: 72.5,
            disk_io_read_mb_s: 125.0,
            disk_io_write_mb_s: 75.0,
        };

        assert_eq!(snapshot.timestamp, 1234567890);
        assert_eq!(snapshot.cpu_usage_percent, 45.5);
        assert_eq!(snapshot.gpu_usage_percent, 85.2);
        assert_eq!(snapshot.gpu_temperature_c, 72.5);
    }
}
