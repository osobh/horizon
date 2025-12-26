/// Errors that can occur when working with time-series databases
#[derive(thiserror::Error, Debug)]
pub enum TsdbError {
    /// Connection error to the time-series database
    #[error("Connection error: {0}")]
    Connection(String),

    /// Query execution error
    #[error("Query error: {0}")]
    Query(String),

    /// Write operation error
    #[error("Write error: {0}")]
    Write(String),

    /// Data parsing error
    #[error("Parse error: {0}")]
    Parse(String),

    /// Invalid configuration
    #[error("Configuration error: {0}")]
    Config(String),

    /// HTTP request error (boxed to keep enum small)
    #[error("HTTP error: {0}")]
    Http(#[source] Box<reqwest::Error>),

    /// JSON serialization/deserialization error
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Time range validation error
    #[error("Invalid time range: {0}")]
    InvalidTimeRange(String),

    /// Missing required field
    #[error("Missing field: {0}")]
    MissingField(String),
}

impl From<reqwest::Error> for TsdbError {
    fn from(err: reqwest::Error) -> Self {
        TsdbError::Http(Box::new(err))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = TsdbError::Connection("timeout".to_string());
        assert_eq!(err.to_string(), "Connection error: timeout");
    }

    #[tokio::test]
    async fn test_error_from_reqwest() {
        // Test conversion from reqwest::Error to TsdbError
        // Use an actual request that fails to get a real reqwest::Error
        let client = reqwest::Client::new();
        let result = client.get("http://invalid-url-12345:99999").send().await;

        assert!(result.is_err());
        let reqwest_err = result.unwrap_err();
        let tsdb_err: TsdbError = reqwest_err.into();
        assert!(matches!(tsdb_err, TsdbError::Http(_)));
    }
}
