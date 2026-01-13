//! Progress writer for benchmark progress logging

use anyhow::Result;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::sync::Mutex;

use super::progress_monitor::BenchmarkPhase;

/// Progress writer that outputs benchmark progress to a log file
pub struct ProgressWriter {
    writer: Mutex<BufWriter<File>>,
}

impl ProgressWriter {
    /// Create a new progress writer
    pub fn new(log_path: &str) -> Result<Self> {
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(log_path)?;

        Ok(Self {
            writer: Mutex::new(BufWriter::new(file)),
        })
    }

    /// Write a log entry with timestamp
    pub fn log(&self, message: &str) -> Result<()> {
        let timestamp = chrono::Local::now().format("%Y-%m-%d %H:%M:%S");
        let entry = format!("[{}] {}\n", timestamp, message);

        let mut writer = self.writer.lock().map_err(|e| anyhow::anyhow!("Lock poisoned: {}", e))?;
        writer.write_all(entry.as_bytes())?;
        writer.flush()?;

        Ok(())
    }

    /// Log phase change
    pub fn log_phase(&self, phase: BenchmarkPhase) -> Result<()> {
        let phase_name = match &phase {
            BenchmarkPhase::NotStarted => "Not Started".to_string(),
            BenchmarkPhase::Initialization => "🚀 Starting GPU Agents Benchmark Suite".to_string(),
            BenchmarkPhase::SystemCheck => {
                "🔍 System Check - Validating GPU availability".to_string()
            }
            BenchmarkPhase::ScalabilityTests => {
                "📈 Phase 1/4 - Agent Scalability Tests".to_string()
            }
            BenchmarkPhase::LlmTests => "🧠 Phase 2/4 - LLM Integration Tests".to_string(),
            BenchmarkPhase::KnowledgeGraphTests => {
                "🕸️ Phase 3/4 - Knowledge Graph Tests".to_string()
            }
            BenchmarkPhase::EvolutionTests => "🧬 Phase 4/4 - Evolution Strategy Tests".to_string(),
            BenchmarkPhase::ReportGeneration => "📊 Report Generation".to_string(),
            BenchmarkPhase::Complete => {
                "✅ Complete - All benchmarks finished successfully!".to_string()
            }
            BenchmarkPhase::Custom(name) => name.clone(),
        };

        self.log(&phase_name)
    }

    /// Log progress update
    pub fn log_progress(&self, percentage: f64) -> Result<()> {
        self.log(&format!("Progress: {:.0}%", percentage * 100.0))
    }

    /// Log GPU usage
    pub fn log_gpu_usage(&self, usage: f64) -> Result<()> {
        self.log(&format!("💻 GPU Usage: {:.0}%", usage))
    }

    /// Log test status
    pub fn log_test(&self, test_name: &str) -> Result<()> {
        self.log(&format!("Testing {}...", test_name))
    }

    /// Log success
    pub fn log_success(&self, message: &str) -> Result<()> {
        self.log(&format!("✓ {}", message))
    }

    /// Log error
    pub fn log_error(&self, message: &str) -> Result<()> {
        self.log(&format!("✗ Error: {}", message))
    }
}
