//! Schemax - Protobuf schema definitions for Horizon
//!
//! This crate provides the protobuf message definitions and gRPC service
//! definitions for the Horizon GPU capacity management system. It includes:
//!
//! - Common messages (health checks, timestamps)
//! - Telemetry messages (GPU, CPU, NIC metrics)
//! - Helper functions for working with timestamps and messages
//! - Builder patterns for convenient message creation
//! - Validation helpers
//!
//! # Examples
//!
//! Creating a GPU metric with the builder:
//!
//! ```
//! use hpc_types::telemetry_helpers::GpuMetricBuilder;
//!
//! let metric = GpuMetricBuilder::new("host-001", "gpu-0")
//!     .utilization(87.5)
//!     .memory(64.5, 80.0)
//!     .temperature(68.5)
//!     .power(350.0)
//!     .build();
//! ```
//!
//! Creating a metric batch:
//!
//! ```
//! use hpc_types::telemetry_helpers::{GpuMetricBuilder, MetricBatchBuilder};
//!
//! let gpu = GpuMetricBuilder::new("host-001", "gpu-0")
//!     .utilization(87.5)
//!     .build();
//!
//! let batch = MetricBatchBuilder::new()
//!     .add_gpu_metric(gpu)
//!     .build();
//! ```

use std::time::{Duration, SystemTime, UNIX_EPOCH};

// Helper modules
pub mod common_helpers;
pub mod telemetry_helpers;

// Include generated protobuf code
pub mod common {
    pub mod v1 {
        tonic::include_proto!("horizon.common.v1");
    }
}

pub mod telemetry {
    pub mod v1 {
        tonic::include_proto!("horizon.telemetry.v1");
    }
}

// Re-export commonly used types for convenience
pub use common::v1::{HealthCheckRequest, HealthCheckResponse, Timestamp};
pub use telemetry::v1::{CpuMetric, GpuMetric, MetricBatch, NicMetric, PublishResponse};

/// Creates a new Timestamp from seconds and nanoseconds
///
/// # Arguments
///
/// * `seconds` - Number of seconds since Unix epoch
/// * `nanos` - Nanosecond offset (0-999,999,999)
///
/// # Examples
///
/// ```
/// use hpc_types::create_timestamp;
///
/// let ts = create_timestamp(1234567890, 123456789);
/// assert_eq!(ts.seconds, 1234567890);
/// assert_eq!(ts.nanos, 123456789);
/// ```
pub fn create_timestamp(seconds: i64, nanos: i32) -> Timestamp {
    Timestamp { seconds, nanos }
}

/// Converts a SystemTime to a protobuf Timestamp
///
/// # Arguments
///
/// * `time` - The SystemTime to convert
///
/// # Examples
///
/// ```
/// use std::time::{SystemTime, UNIX_EPOCH, Duration};
/// use hpc_types::timestamp_from_system_time;
///
/// let time = UNIX_EPOCH + Duration::from_secs(1234567890);
/// let ts = timestamp_from_system_time(time);
/// assert_eq!(ts.seconds, 1234567890);
/// ```
pub fn timestamp_from_system_time(time: SystemTime) -> Timestamp {
    let duration = time
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::from_secs(0));

    Timestamp {
        seconds: duration.as_secs() as i64,
        nanos: duration.subsec_nanos() as i32,
    }
}

/// Converts a protobuf Timestamp to SystemTime
///
/// # Arguments
///
/// * `timestamp` - The protobuf Timestamp to convert
///
/// # Examples
///
/// ```
/// use hpc_types::{create_timestamp, timestamp_to_system_time};
/// use std::time::{UNIX_EPOCH, Duration};
///
/// let ts = create_timestamp(1234567890, 500_000_000);
/// let time = timestamp_to_system_time(&ts);
/// let duration = time.duration_since(UNIX_EPOCH).unwrap();
/// assert_eq!(duration.as_secs(), 1234567890);
/// assert_eq!(duration.subsec_nanos(), 500_000_000);
/// ```
pub fn timestamp_to_system_time(timestamp: &Timestamp) -> SystemTime {
    UNIX_EPOCH
        + Duration::from_secs(timestamp.seconds as u64)
        + Duration::from_nanos(timestamp.nanos as u64)
}

/// Creates the current timestamp
///
/// # Examples
///
/// ```
/// use hpc_types::current_timestamp;
///
/// let ts = current_timestamp();
/// assert!(ts.seconds > 0);
/// ```
pub fn current_timestamp() -> Timestamp {
    timestamp_from_system_time(SystemTime::now())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_timestamp() {
        let ts = create_timestamp(1234567890, 123456789);
        assert_eq!(ts.seconds, 1234567890);
        assert_eq!(ts.nanos, 123456789);
    }

    #[test]
    fn test_timestamp_conversion() {
        let time = UNIX_EPOCH + Duration::from_secs(1234567890);
        let ts = timestamp_from_system_time(time);
        assert_eq!(ts.seconds, 1234567890);

        let recovered = timestamp_to_system_time(&ts);
        assert_eq!(
            recovered.duration_since(UNIX_EPOCH).unwrap().as_secs(),
            1234567890
        );
    }

    #[test]
    fn test_current_timestamp() {
        let ts = current_timestamp();
        assert!(ts.seconds > 1_600_000_000); // After 2020
        assert!(ts.nanos >= 0 && ts.nanos < 1_000_000_000);
    }
}
