//! Progress log file parsing for the TUI dashboard

use anyhow::{Context, Result};
use std::fs::File;
use std::io::{BufRead, BufReader, Seek, SeekFrom};
use std::path::Path;

use crate::benchmarks::progress_monitor::{BenchmarkPhase, ProgressState};

/// Parser for benchmark progress log files
#[derive(Debug)]
pub struct ProgressLogParser {
    /// Path to the log file
    log_file_path: String,
    /// Current file position
    file_position: u64,
    /// Cache of recent log entries
    recent_entries: Vec<String>,
}

impl ProgressLogParser {
    /// Create a new progress log parser
    pub fn new(log_file_path: &str) -> Result<Self> {
        Ok(Self {
            log_file_path: log_file_path.to_string(),
            file_position: 0,
            recent_entries: Vec::new(),
        })
    }

    /// Get the latest progress state from the log file
    pub fn get_latest_state(&mut self) -> Result<Option<ProgressState>> {
        if !Path::new(&self.log_file_path).exists() {
            return Ok(None);
        }

        let mut file =
            File::open(&self.log_file_path).context("Failed to open progress log file")?;

        // Seek to our last position
        file.seek(SeekFrom::Start(self.file_position))?;
        let mut reader = BufReader::new(file);

        let mut new_lines = Vec::new();
        let mut line = String::new();

        // Read new content from file
        while reader.read_line(&mut line)? > 0 {
            new_lines.push(line.trim().to_string());
            line.clear();
        }

        // Update file position
        self.file_position = reader.seek(SeekFrom::Current(0))?;

        // Add new lines to recent entries
        self.recent_entries.extend(new_lines);

        // Keep only recent entries (last 100)
        if self.recent_entries.len() > 100 {
            self.recent_entries
                .drain(0..self.recent_entries.len() - 100);
        }

        // Parse the latest state from recent entries
        self.parse_progress_state()
    }

    /// Get recent log entries
    pub fn get_recent_entries(&self, count: usize) -> Result<Vec<String>> {
        let start_idx = if self.recent_entries.len() > count {
            self.recent_entries.len() - count
        } else {
            0
        };

        Ok(self.recent_entries[start_idx..].to_vec())
    }

    /// Parse progress state from log entries
    fn parse_progress_state(&self) -> Result<Option<ProgressState>> {
        if self.recent_entries.is_empty() {
            return Ok(None);
        }

        let current_phase = self.extract_current_phase();
        let overall_progress = self.extract_progress_percentage();
        let (tests_completed, total_tests) = self.extract_test_counts();
        let current_operation = self.extract_current_operation();

        Ok(Some(ProgressState {
            current_phase,
            phase_progress: 0.0, // Will be calculated based on phase
            overall_progress,
            current_test: current_operation.clone(),
            tests_completed,
            total_tests,
            elapsed_time: std::time::Duration::ZERO, // Will be calculated by app
            estimated_remaining: std::time::Duration::ZERO, // Will be calculated by app
            current_operation,
        }))
    }

    /// Extract current benchmark phase from log entries
    fn extract_current_phase(&self) -> BenchmarkPhase {
        // Look for phase indicators in recent entries (newest first)
        for entry in self.recent_entries.iter().rev() {
            if entry.contains("🔍") || entry.contains("System Check") {
                return BenchmarkPhase::SystemCheck;
            }
            if entry.contains("📈") || entry.contains("Scalability Tests") {
                return BenchmarkPhase::ScalabilityTests;
            }
            if entry.contains("🧠") || entry.contains("LLM Integration") {
                return BenchmarkPhase::LlmTests;
            }
            if entry.contains("🕸️") || entry.contains("Knowledge Graph") {
                return BenchmarkPhase::KnowledgeGraphTests;
            }
            if entry.contains("🧬") || entry.contains("Evolution") {
                return BenchmarkPhase::EvolutionTests;
            }
            if entry.contains("📊 Report") || entry.contains("Report Generation") {
                return BenchmarkPhase::ReportGeneration;
            }
            if entry.contains("✅") || entry.contains("Complete") {
                return BenchmarkPhase::Complete;
            }
        }

        BenchmarkPhase::Initialization
    }

    /// Extract progress percentage from log entries
    fn extract_progress_percentage(&self) -> f64 {
        // Look for progress percentages in recent entries
        for entry in self.recent_entries.iter().rev() {
            if let Some(progress_str) = self.extract_percentage_from_line(entry) {
                if let Ok(percentage) = progress_str.parse::<f64>() {
                    return percentage / 100.0; // Convert to 0.0-1.0 range
                }
            }
        }

        0.0
    }

    /// Extract percentage from a log line
    fn extract_percentage_from_line(&self, line: &str) -> Option<String> {
        // Look for patterns like "25%" or "Progress: 75%"
        if let Some(percent_pos) = line.find('%') {
            // Work backwards from % to find the start of the number
            let bytes = line.as_bytes();
            let mut start = percent_pos;

            while start > 0 {
                let prev = start - 1;
                let ch = bytes[prev];
                if ch.is_ascii_digit() || ch == b'.' {
                    start = prev;
                } else {
                    break;
                }
            }

            // Extract the number if we found one
            if start < percent_pos {
                return Some(line[start..percent_pos].to_string());
            }
        }
        None
    }

    /// Extract test counts from log entries
    fn extract_test_counts(&self) -> (usize, usize) {
        // Look for patterns like "3/4 tests" or "Phase 2/4"
        for entry in self.recent_entries.iter().rev() {
            if let Some((completed, total)) = self.extract_counts_from_line(entry) {
                return (completed, total);
            }
        }

        (0, 0)
    }

    /// Extract count pattern from a log line
    fn extract_counts_from_line(&self, line: &str) -> Option<(usize, usize)> {
        // Look for patterns like "2/4" or "Phase 3/4"
        let parts: Vec<&str> = line.split('/').collect();
        if parts.len() == 2 {
            // Extract numbers from both parts
            let completed_str: String = parts[0].chars().filter(|c| c.is_ascii_digit()).collect();
            let total_str: String = parts[1]
                .chars()
                .filter(|c| c.is_ascii_digit())
                .take(2)
                .collect();

            if let (Ok(completed), Ok(total)) =
                (completed_str.parse::<usize>(), total_str.parse::<usize>())
            {
                return Some((completed, total));
            }
        }
        None
    }

    /// Extract current operation from log entries
    fn extract_current_operation(&self) -> String {
        // Get the most recent meaningful log entry
        for entry in self.recent_entries.iter().rev() {
            if !entry.trim().is_empty() && entry.contains(']') {
                // Extract message after timestamp
                if let Some(pos) = entry.find(']') {
                    let operation = entry[pos + 1..].trim();
                    if !operation.is_empty() {
                        return operation.to_string();
                    }
                }
            }
        }

        "No activity detected".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn create_test_log_file(content: &str) -> NamedTempFile {
        let mut file = NamedTempFile::new()?;
        file.write_all(content.as_bytes())?;
        file.flush()?;
        file
    }

    #[test]
    fn test_progress_log_parser_new() {
        // RED PHASE: Test will fail until ProgressLogParser is implemented
        let parser = ProgressLogParser::new("test.log");
        assert!(parser.is_ok());
    }

    #[test]
    fn test_extract_percentage_from_line() -> Result<(), Box<dyn std::error::Error>> {
        let parser = ProgressLogParser::new("test.log")?;

        assert_eq!(
            parser.extract_percentage_from_line("Progress: 75%"),
            Some("75".to_string())
        );
        assert_eq!(
            parser.extract_percentage_from_line("25.5% complete"),
            Some("25.5".to_string())
        );
        assert_eq!(parser.extract_percentage_from_line("No progress"), None);
    }

    #[test]
    fn test_extract_counts_from_line() -> Result<(), Box<dyn std::error::Error>> {
        let parser = ProgressLogParser::new("test.log")?;

        assert_eq!(
            parser.extract_counts_from_line("Phase 2/4 - Testing"),
            Some((2, 4))
        );
        assert_eq!(
            parser.extract_counts_from_line("3/4 tests completed"),
            Some((3, 4))
        );
        assert_eq!(parser.extract_counts_from_line("No counts here"), None);
    }

    #[test]
    fn test_extract_current_phase() -> Result<(), Box<dyn std::error::Error>> {
        let mut parser = ProgressLogParser::new("test.log")?;

        // Test phase detection
        parser.recent_entries =
            vec!["[2024-01-01 12:00:00] 🔍 System Check - Starting".to_string()];
        assert_eq!(parser.extract_current_phase(), BenchmarkPhase::SystemCheck);

        parser.recent_entries =
            vec!["[2024-01-01 12:00:00] 📈 Phase 1/4 - Agent Scalability Tests".to_string()];
        assert_eq!(
            parser.extract_current_phase(),
            BenchmarkPhase::ScalabilityTests
        );

        parser.recent_entries =
            vec!["[2024-01-01 12:00:00] 🧠 LLM Integration Tests starting".to_string()];
        assert_eq!(parser.extract_current_phase(), BenchmarkPhase::LlmTests);
    }

    #[test]
    fn test_get_latest_state_with_nonexistent_file() -> Result<(), Box<dyn std::error::Error>> {
        let mut parser = ProgressLogParser::new("nonexistent.log")?;
        let state = parser.get_latest_state()?;
        assert!(state.is_none());
    }

    #[test]
    fn test_get_latest_state_with_content() {
        let log_content = r#"[2024-01-01 12:00:00] 🚀 Starting GPU Agents Benchmark Suite
[2024-01-01 12:01:00] 🔍 System Check - Validating GPU availability
[2024-01-01 12:02:00] 📈 Phase 1/4 - Agent Scalability Tests
[2024-01-01 12:03:00] 📊 Progress: 25%
"#;

        let temp_file = create_test_log_file(log_content);
        let mut parser = ProgressLogParser::new(temp_file.path().to_str()?)?;

        let state = parser.get_latest_state()?;
        assert!(state.is_some());

        let state = state?;
        assert_eq!(state.current_phase, BenchmarkPhase::ScalabilityTests);

        // Debug progress extraction
        println!("Overall progress: {}", state.overall_progress);
        println!("Recent entries: {:?}", parser.recent_entries);

        assert_eq!(state.overall_progress, 0.25);
        assert_eq!(state.tests_completed, 1);
        assert_eq!(state.total_tests, 4);
    }

    #[test]
    fn test_get_recent_entries() -> Result<(), Box<dyn std::error::Error>> {
        let mut parser = ProgressLogParser::new("test.log")?;
        parser.recent_entries = vec![
            "Entry 1".to_string(),
            "Entry 2".to_string(),
            "Entry 3".to_string(),
            "Entry 4".to_string(),
        ];

        let recent = parser.get_recent_entries(2)?;
        assert_eq!(recent.len(), 2);
        assert_eq!(recent[0], "Entry 3");
        assert_eq!(recent[1], "Entry 4");

        let all = parser.get_recent_entries(10)?;
        assert_eq!(all.len(), 4);
    }

    #[test]
    fn test_extract_current_operation() -> Result<(), Box<dyn std::error::Error>> {
        let mut parser = ProgressLogParser::new("test.log")?;

        parser.recent_entries = vec![
            "[2024-01-01 12:00:00] 📈 Testing 1M agents spawn rate".to_string(),
            "[2024-01-01 12:01:00] 💻 GPU Usage: 85%".to_string(),
        ];

        let operation = parser.extract_current_operation();
        assert_eq!(operation, "💻 GPU Usage: 85%");
    }
}
