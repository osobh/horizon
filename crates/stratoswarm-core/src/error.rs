//! Error types for stratoswarm-core channel infrastructure.
//!
//! This module defines all error types used throughout the channel system,
//! providing clear error messages and proper error propagation.

use std::time::Duration;
use thiserror::Error;

/// Errors that can occur during channel operations.
#[derive(Error, Debug, Clone)]
pub enum ChannelError {
    /// Channel not found in registry
    #[error("Channel '{0}' not found in registry")]
    ChannelNotFound(String),

    /// Failed to send message (channel closed or full)
    #[error("Failed to send message on channel '{channel}': {reason}")]
    SendFailed {
        /// Channel name
        channel: String,
        /// Reason for failure
        reason: String,
    },

    /// Failed to receive message (channel closed)
    #[error("Failed to receive message on channel '{0}': channel closed")]
    ReceiveFailed(String),

    /// Request timeout
    #[error("Request timed out after {timeout:?}")]
    Timeout {
        /// Timeout duration
        timeout: Duration,
    },

    /// Channel already registered
    #[error("Channel '{0}' already registered")]
    ChannelAlreadyExists(String),

    /// Invalid buffer size
    #[error("Invalid buffer size: {0}. Must be greater than 0")]
    InvalidBufferSize(usize),

    /// Broadcast channel error
    #[error("Broadcast channel error: {0}")]
    BroadcastError(String),
}

/// Result type alias for channel operations.
pub type Result<T> = std::result::Result<T, ChannelError>;
