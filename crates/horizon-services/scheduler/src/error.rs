//! Error handling for the scheduler service.
//!
//! Uses `hpc_error::HpcError` as the unified error type with feature-gated
//! conversions for sqlx, axum, and reqwest.

pub use hpc_error::{HpcError, Result};

/// Extension trait for scheduler-specific error construction
pub trait SchedulerErrorExt {
    /// Creates a job not found error
    fn job_not_found(job_id: uuid::Uuid) -> HpcError {
        HpcError::not_found("job", job_id.to_string())
    }

    /// Creates a checkpoint not found error
    fn checkpoint_not_found(checkpoint_id: impl Into<String>) -> HpcError {
        HpcError::not_found("checkpoint", checkpoint_id)
    }

    /// Creates an invalid state transition error
    fn invalid_state_transition(from: impl Into<String>, to: impl Into<String>) -> HpcError {
        HpcError::invalid_input(
            "state_transition",
            format!("Invalid state transition from {} to {}", from.into(), to.into()),
        )
    }

    /// Creates an insufficient resources error
    fn insufficient_resources(required: usize, available: usize) -> HpcError {
        HpcError::resource_exhausted(
            format!("required {}, available {}", required, available),
            format!("{} resources", required),
        )
    }

    /// Creates an invalid job configuration error
    fn invalid_job_config(reason: impl Into<String>) -> HpcError {
        HpcError::invalid_input("job_config", reason)
    }

    /// Creates a validation error
    fn validation(reason: impl Into<String>) -> HpcError {
        HpcError::invalid_input("validation", reason)
    }

    /// Creates a checkpoint too large error
    fn checkpoint_too_large(size_gb: u64, max_gb: u64) -> HpcError {
        HpcError::resource_exhausted(
            format!("checkpoint size {} GB", size_gb),
            format!("{} GB", max_gb),
        )
    }

    /// Creates a no container for job error
    fn no_container_for_job(job_id: uuid::Uuid) -> HpcError {
        HpcError::not_found("container", format!("No container for job: {}", job_id))
    }

    /// Creates a no checkpoint for job error
    fn no_checkpoint_for_job(job_id: uuid::Uuid) -> HpcError {
        HpcError::not_found("checkpoint", format!("No checkpoint for job: {}", job_id))
    }

    /// Creates a storage full error
    fn storage_full() -> HpcError {
        HpcError::resource_exhausted("storage", "0 bytes")
    }

    /// Creates a preemption failed error
    fn preemption_failed(reason: impl Into<String>) -> HpcError {
        HpcError::internal(format!("Preemption failed: {}", reason.into()))
    }
}

impl SchedulerErrorExt for HpcError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_job_not_found() {
        let job_id = uuid::Uuid::new_v4();
        let err = HpcError::job_not_found(job_id);
        assert!(err.to_string().contains("job"));
        assert!(err.to_string().contains("not found"));
    }

    #[test]
    fn test_invalid_state_transition() {
        let err = HpcError::invalid_state_transition("queued", "completed");
        assert!(err.to_string().contains("queued"));
        assert!(err.to_string().contains("completed"));
    }

    #[test]
    fn test_insufficient_resources() {
        let err = HpcError::insufficient_resources(8, 4);
        assert!(err.to_string().contains("8"));
        assert!(err.to_string().contains("4"));
    }

    #[test]
    fn test_validation_error() {
        let err = HpcError::validation("GPU count must be positive");
        assert!(err.to_string().contains("GPU count"));
    }

    #[test]
    fn test_checkpoint_too_large() {
        let err = HpcError::checkpoint_too_large(100, 50);
        assert!(err.to_string().contains("100"));
        assert!(err.to_string().contains("50"));
    }

    #[test]
    fn test_storage_full() {
        let err = HpcError::storage_full();
        assert!(err.to_string().contains("storage"));
    }
}
