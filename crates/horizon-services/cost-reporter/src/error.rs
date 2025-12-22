//! Error handling for the cost reporter service.

pub use hpc_error::{HpcError, Result};

/// Extension trait for reporter-specific error construction
pub trait ReporterErrorExt {
    /// Creates an invalid time range error
    fn invalid_time_range(start: impl Into<String>, end: impl Into<String>) -> HpcError {
        HpcError::invalid_input(
            "time_range",
            format!("Invalid time range: start={}, end={}", start.into(), end.into()),
        )
    }

    /// Creates an invalid query error
    fn invalid_query(reason: impl Into<String>) -> HpcError {
        HpcError::invalid_input("query", reason)
    }

    /// Creates a report generation error
    fn report_generation_error(reason: impl Into<String>) -> HpcError {
        HpcError::internal(format!("Report generation failed: {}", reason.into()))
    }

    /// Creates an export error
    fn export_error(reason: impl Into<String>) -> HpcError {
        HpcError::internal(format!("Export failed: {}", reason.into()))
    }

    /// Creates a report not found error
    fn report_not_found(id: impl Into<String>) -> HpcError {
        HpcError::not_found("report", id)
    }

    /// Creates an invalid period error
    fn invalid_period(reason: impl Into<String>) -> HpcError {
        HpcError::invalid_input("period", reason)
    }

    /// Creates an insufficient data error
    fn insufficient_data(required: usize) -> HpcError {
        HpcError::invalid_input(
            "data",
            format!("Insufficient data for forecast: need at least {} data points", required),
        )
    }
}

impl ReporterErrorExt for HpcError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_invalid_time_range() {
        let err = HpcError::invalid_time_range("2024-01-01", "2023-12-31");
        assert!(err.to_string().contains("time_range"));
    }

    #[test]
    fn test_invalid_query() {
        let err = HpcError::invalid_query("missing required field");
        assert!(err.to_string().contains("query"));
    }

    #[test]
    fn test_report_not_found() {
        let err = HpcError::report_not_found("report-123");
        assert!(err.to_string().contains("report"));
        assert!(err.to_string().contains("not found"));
    }

    #[test]
    fn test_insufficient_data() {
        let err = HpcError::insufficient_data(30);
        assert!(err.to_string().contains("30"));
    }

    #[test]
    fn test_invalid_period() {
        let err = HpcError::invalid_period("invalid");
        assert!(err.to_string().contains("period"));
    }
}
