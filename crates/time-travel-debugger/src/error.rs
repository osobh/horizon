use thiserror::Error;
use uuid::Uuid;

/// Errors that can occur in the time-travel debugger
#[derive(Error, Debug, Clone, PartialEq)]
pub enum TimeDebuggerError {
    #[error("Snapshot not found: {id}")]
    SnapshotNotFound { id: Uuid },

    #[error("Invalid timestamp: {timestamp}")]
    InvalidTimestamp { timestamp: i64 },

    #[error("Event log is empty")]
    EmptyEventLog,

    #[error("Cannot navigate to index {index}: out of bounds")]
    NavigationOutOfBounds { index: usize },

    #[error("Breakpoint not found: {id}")]
    BreakpointNotFound { id: Uuid },

    #[error("Session not found: {id}")]
    SessionNotFound { id: Uuid },

    #[error("State comparison failed: {reason}")]
    ComparisonFailed { reason: String },

    #[error("Serialization error: {message}")]
    SerializationError { message: String },

    #[error("Deserialization error: {message}")]
    DeserializationError { message: String },

    #[error("IO error: {message}")]
    IoError { message: String },

    #[error("Event replay failed at index {index}: {reason}")]
    ReplayFailed { index: usize, reason: String },

    #[error("Invalid state diff: {reason}")]
    InvalidDiff { reason: String },

    #[error("Concurrent access violation: {operation}")]
    ConcurrentAccess { operation: String },
}

impl From<serde_json::Error> for TimeDebuggerError {
    #[cold]
    fn from(err: serde_json::Error) -> Self {
        // Syntax errors, EOF errors, and data errors occur during deserialization
        if err.is_data() || err.is_syntax() || err.is_eof() {
            TimeDebuggerError::DeserializationError {
                message: err.to_string(),
            }
        } else {
            // IO errors during serialization/deserialization can be either,
            // but we'll treat them as serialization errors for simplicity
            TimeDebuggerError::SerializationError {
                message: err.to_string(),
            }
        }
    }
}

impl From<std::io::Error> for TimeDebuggerError {
    #[cold]
    fn from(err: std::io::Error) -> Self {
        TimeDebuggerError::IoError {
            message: err.to_string(),
        }
    }
}

pub type Result<T> = std::result::Result<T, TimeDebuggerError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let id = Uuid::new_v4();
        let error = TimeDebuggerError::SnapshotNotFound { id };
        assert!(error.to_string().contains(&id.to_string()));
    }

    #[test]
    fn test_error_equality() {
        let id = Uuid::new_v4();
        let error1 = TimeDebuggerError::SnapshotNotFound { id };
        let error2 = TimeDebuggerError::SnapshotNotFound { id };
        assert_eq!(error1, error2);
    }

    #[test]
    fn test_serde_error_conversion() {
        let json_err = serde_json::from_str::<serde_json::Value>("invalid json");
        assert!(json_err.is_err());

        let converted: TimeDebuggerError = json_err.unwrap_err().into();
        match converted {
            TimeDebuggerError::DeserializationError { .. } => {}
            _ => panic!("Expected DeserializationError"),
        }
    }

    #[test]
    fn test_io_error_conversion() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let converted: TimeDebuggerError = io_err.into();

        match converted {
            TimeDebuggerError::IoError { message } => {
                assert!(message.contains("file not found"));
            }
            _ => panic!("Expected IoError"),
        }
    }
}
