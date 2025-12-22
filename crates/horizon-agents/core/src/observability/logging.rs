use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

impl LogLevel {
    pub fn as_str(&self) -> &'static str {
        match self {
            LogLevel::Trace => "TRACE",
            LogLevel::Debug => "DEBUG",
            LogLevel::Info => "INFO",
            LogLevel::Warn => "WARN",
            LogLevel::Error => "ERROR",
        }
    }

    pub fn should_log(&self, min_level: LogLevel) -> bool {
        use LogLevel::*;
        let self_priority = match self {
            Trace => 0,
            Debug => 1,
            Info => 2,
            Warn => 3,
            Error => 4,
        };

        let min_priority = match min_level {
            Trace => 0,
            Debug => 1,
            Info => 2,
            Warn => 3,
            Error => 4,
        };

        self_priority >= min_priority
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    pub level: LogLevel,
    pub message: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub context: HashMap<String, String>,
}

impl LogEntry {
    pub fn new(level: LogLevel, message: String) -> Self {
        Self {
            level,
            message,
            timestamp: chrono::Utc::now(),
            context: HashMap::new(),
        }
    }

    pub fn with_context(mut self, key: String, value: String) -> Self {
        self.context.insert(key, value);
        self
    }

    pub fn format(&self) -> String {
        let ctx = if self.context.is_empty() {
            String::new()
        } else {
            let pairs: Vec<String> = self
                .context
                .iter()
                .map(|(k, v)| format!("{}={}", k, v))
                .collect();
            format!(" [{}]", pairs.join(", "))
        };

        format!(
            "[{}] {} - {}{}",
            self.timestamp.format("%Y-%m-%d %H:%M:%S%.3f"),
            self.level.as_str(),
            self.message,
            ctx
        )
    }
}

pub struct StructuredLogger {
    entries: Vec<LogEntry>,
    max_entries: usize,
    min_level: LogLevel,
}

impl StructuredLogger {
    pub fn new(max_entries: usize, min_level: LogLevel) -> Self {
        Self {
            entries: Vec::new(),
            max_entries,
            min_level,
        }
    }

    pub fn log(&mut self, entry: LogEntry) {
        if !entry.level.should_log(self.min_level) {
            return;
        }

        if self.entries.len() >= self.max_entries {
            self.entries.remove(0);
        }

        self.entries.push(entry);
    }

    pub fn trace(&mut self, message: String) {
        self.log(LogEntry::new(LogLevel::Trace, message));
    }

    pub fn debug(&mut self, message: String) {
        self.log(LogEntry::new(LogLevel::Debug, message));
    }

    pub fn info(&mut self, message: String) {
        self.log(LogEntry::new(LogLevel::Info, message));
    }

    pub fn warn(&mut self, message: String) {
        self.log(LogEntry::new(LogLevel::Warn, message));
    }

    pub fn error(&mut self, message: String) {
        self.log(LogEntry::new(LogLevel::Error, message));
    }

    pub fn get_entries(&self, level: Option<LogLevel>) -> Vec<&LogEntry> {
        if let Some(lvl) = level {
            self.entries.iter().filter(|e| e.level == lvl).collect()
        } else {
            self.entries.iter().collect()
        }
    }

    pub fn get_recent(&self, limit: usize) -> Vec<&LogEntry> {
        let start = if self.entries.len() > limit {
            self.entries.len() - limit
        } else {
            0
        };

        self.entries[start..].iter().rev().collect()
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }

    pub fn set_min_level(&mut self, level: LogLevel) {
        self.min_level = level;
    }

    pub fn size(&self) -> usize {
        self.entries.len()
    }
}

impl Default for StructuredLogger {
    fn default() -> Self {
        Self::new(1000, LogLevel::Info)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_level_as_str() {
        assert_eq!(LogLevel::Trace.as_str(), "TRACE");
        assert_eq!(LogLevel::Debug.as_str(), "DEBUG");
        assert_eq!(LogLevel::Info.as_str(), "INFO");
        assert_eq!(LogLevel::Warn.as_str(), "WARN");
        assert_eq!(LogLevel::Error.as_str(), "ERROR");
    }

    #[test]
    fn test_log_level_should_log() {
        assert!(LogLevel::Error.should_log(LogLevel::Info));
        assert!(LogLevel::Warn.should_log(LogLevel::Info));
        assert!(LogLevel::Info.should_log(LogLevel::Info));
        assert!(!LogLevel::Debug.should_log(LogLevel::Info));
        assert!(!LogLevel::Trace.should_log(LogLevel::Info));
    }

    #[test]
    fn test_log_entry_creation() {
        let entry = LogEntry::new(LogLevel::Info, "Test message".to_string());
        assert_eq!(entry.level, LogLevel::Info);
        assert_eq!(entry.message, "Test message");
        assert!(entry.context.is_empty());
    }

    #[test]
    fn test_log_entry_with_context() {
        let entry = LogEntry::new(LogLevel::Info, "Test message".to_string())
            .with_context("key1".to_string(), "value1".to_string())
            .with_context("key2".to_string(), "value2".to_string());

        assert_eq!(entry.context.len(), 2);
        assert_eq!(entry.context.get("key1").unwrap(), "value1");
    }

    #[test]
    fn test_log_entry_format() {
        let entry = LogEntry::new(LogLevel::Info, "Test message".to_string());
        let formatted = entry.format();

        assert!(formatted.contains("INFO"));
        assert!(formatted.contains("Test message"));
    }

    #[test]
    fn test_log_entry_format_with_context() {
        let entry = LogEntry::new(LogLevel::Info, "Test message".to_string())
            .with_context("user".to_string(), "alice".to_string());

        let formatted = entry.format();
        assert!(formatted.contains("user=alice"));
    }

    #[test]
    fn test_structured_logger_creation() {
        let logger = StructuredLogger::new(100, LogLevel::Info);
        assert_eq!(logger.size(), 0);
        assert_eq!(logger.min_level, LogLevel::Info);
    }

    #[test]
    fn test_structured_logger_log() {
        let mut logger = StructuredLogger::new(100, LogLevel::Info);

        let entry = LogEntry::new(LogLevel::Info, "Test".to_string());
        logger.log(entry);

        assert_eq!(logger.size(), 1);
    }

    #[test]
    fn test_structured_logger_min_level_filter() {
        let mut logger = StructuredLogger::new(100, LogLevel::Warn);

        logger.debug("Debug message".to_string());
        logger.info("Info message".to_string());
        logger.warn("Warn message".to_string());
        logger.error("Error message".to_string());

        assert_eq!(logger.size(), 2); // Only warn and error
    }

    #[test]
    fn test_structured_logger_trace() {
        let mut logger = StructuredLogger::new(100, LogLevel::Trace);
        logger.trace("Trace message".to_string());

        let entries = logger.get_entries(Some(LogLevel::Trace));
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].message, "Trace message");
    }

    #[test]
    fn test_structured_logger_debug() {
        let mut logger = StructuredLogger::new(100, LogLevel::Debug);
        logger.debug("Debug message".to_string());

        let entries = logger.get_entries(Some(LogLevel::Debug));
        assert_eq!(entries.len(), 1);
    }

    #[test]
    fn test_structured_logger_info() {
        let mut logger = StructuredLogger::new(100, LogLevel::Info);
        logger.info("Info message".to_string());

        let entries = logger.get_entries(Some(LogLevel::Info));
        assert_eq!(entries.len(), 1);
    }

    #[test]
    fn test_structured_logger_warn() {
        let mut logger = StructuredLogger::new(100, LogLevel::Info);
        logger.warn("Warn message".to_string());

        let entries = logger.get_entries(Some(LogLevel::Warn));
        assert_eq!(entries.len(), 1);
    }

    #[test]
    fn test_structured_logger_error() {
        let mut logger = StructuredLogger::new(100, LogLevel::Info);
        logger.error("Error message".to_string());

        let entries = logger.get_entries(Some(LogLevel::Error));
        assert_eq!(entries.len(), 1);
    }

    #[test]
    fn test_structured_logger_get_entries() {
        let mut logger = StructuredLogger::new(100, LogLevel::Info);

        logger.info("Info 1".to_string());
        logger.warn("Warn 1".to_string());
        logger.info("Info 2".to_string());

        let all = logger.get_entries(None);
        assert_eq!(all.len(), 3);

        let info_only = logger.get_entries(Some(LogLevel::Info));
        assert_eq!(info_only.len(), 2);

        let warn_only = logger.get_entries(Some(LogLevel::Warn));
        assert_eq!(warn_only.len(), 1);
    }

    #[test]
    fn test_structured_logger_get_recent() {
        let mut logger = StructuredLogger::new(100, LogLevel::Info);

        for i in 0..5 {
            logger.info(format!("Message {}", i));
        }

        let recent = logger.get_recent(3);
        assert_eq!(recent.len(), 3);
        assert_eq!(recent[0].message, "Message 4");
        assert_eq!(recent[1].message, "Message 3");
        assert_eq!(recent[2].message, "Message 2");
    }

    #[test]
    fn test_structured_logger_eviction() {
        let mut logger = StructuredLogger::new(2, LogLevel::Info);

        logger.info("Message 1".to_string());
        logger.info("Message 2".to_string());
        logger.info("Message 3".to_string());

        assert_eq!(logger.size(), 2);
        let entries = logger.get_entries(None);
        assert_eq!(entries[0].message, "Message 2");
        assert_eq!(entries[1].message, "Message 3");
    }

    #[test]
    fn test_structured_logger_clear() {
        let mut logger = StructuredLogger::new(100, LogLevel::Info);

        logger.info("Message".to_string());
        assert_eq!(logger.size(), 1);

        logger.clear();
        assert_eq!(logger.size(), 0);
    }

    #[test]
    fn test_structured_logger_set_min_level() {
        let mut logger = StructuredLogger::new(100, LogLevel::Info);

        logger.debug("Debug 1".to_string());
        assert_eq!(logger.size(), 0);

        logger.set_min_level(LogLevel::Debug);
        logger.debug("Debug 2".to_string());
        assert_eq!(logger.size(), 1);
    }
}
