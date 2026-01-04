//! Channel communication patterns.
//!
//! This module provides high-level communication patterns built on top of
//! the channel infrastructure, including request/response with timeout.

use crate::error::{ChannelError, Result};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tokio::sync::oneshot;
use tokio::time::timeout;

/// A responder for sending a single response back to a request.
///
/// This type wraps a oneshot channel sender and provides a clean API
/// for responding to requests.
#[derive(Debug)]
pub struct Responder<T> {
    sender: oneshot::Sender<T>,
}

impl<T> Responder<T> {
    /// Create a new responder from a oneshot sender.
    #[must_use]
    pub fn new(sender: oneshot::Sender<T>) -> Self {
        Self { sender }
    }

    /// Send a response back to the requester.
    ///
    /// # Errors
    ///
    /// Returns an error if the requester has dropped the receiver.
    pub fn respond(self, response: T) -> Result<()> {
        self.sender
            .send(response)
            .map_err(|_| ChannelError::SendFailed {
                channel: "response".to_string(),
                reason: "requester dropped receiver".to_string(),
            })
    }
}

/// A request that expects a response.
///
/// This type combines a request payload with a responder channel,
/// enabling request/response communication patterns.
#[derive(Debug)]
pub struct Request<Req, Resp> {
    /// The request payload
    pub payload: Req,
    /// The responder for sending back a response
    pub responder: Responder<Resp>,
}

impl<Req, Resp> Request<Req, Resp> {
    /// Create a new request with a responder.
    #[must_use]
    pub fn new(payload: Req, responder: Responder<Resp>) -> Self {
        Self { payload, responder }
    }

    /// Respond to this request.
    ///
    /// # Errors
    ///
    /// Returns an error if the requester has dropped the receiver.
    pub fn respond(self, response: Resp) -> Result<()> {
        self.responder.respond(response)
    }
}

/// Send a request and wait for a response with a timeout.
///
/// This function creates a oneshot channel, sends the request with a responder,
/// and waits for the response with the specified timeout.
///
/// # Errors
///
/// Returns `ChannelError::Timeout` if the response doesn't arrive in time.
/// Returns `ChannelError::SendFailed` if sending the request fails.
/// Returns `ChannelError::ReceiveFailed` if the responder drops without responding.
///
/// # Example
///
/// ```rust,no_run
/// use stratoswarm_core::channels::patterns::{request_with_timeout, Request};
/// use tokio::sync::mpsc;
/// use std::time::Duration;
///
/// #[tokio::main]
/// async fn main() {
///     let (tx, mut rx) = mpsc::channel::<Request<String, i32>>(10);
///
///     // Spawn handler
///     tokio::spawn(async move {
///         if let Some(request) = rx.recv().await {
///             request.respond(42).unwrap();
///         }
///     });
///
///     // Send request with timeout
///     let response = request_with_timeout(
///         &tx,
///         "hello".to_string(),
///         Duration::from_secs(5)
///     ).await.unwrap();
///
///     assert_eq!(response, 42);
/// }
/// ```
pub async fn request_with_timeout<Req, Resp>(
    sender: &tokio::sync::mpsc::Sender<Request<Req, Resp>>,
    request: Req,
    timeout_duration: Duration,
) -> Result<Resp> {
    let (resp_tx, resp_rx) = oneshot::channel();
    let responder = Responder::new(resp_tx);
    let req = Request::new(request, responder);

    // Send the request
    sender
        .send(req)
        .await
        .map_err(|_| ChannelError::SendFailed {
            channel: "request".to_string(),
            reason: "channel closed".to_string(),
        })?;

    // Wait for response with timeout
    timeout(timeout_duration, resp_rx)
        .await
        .map_err(|_| ChannelError::Timeout {
            timeout: timeout_duration,
        })?
        .map_err(|_| ChannelError::ReceiveFailed("response".to_string()))
}

/// Response type for query operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResponse<T> {
    /// Whether the query succeeded
    pub success: bool,
    /// Optional result data
    pub data: Option<T>,
    /// Optional error message
    pub error: Option<String>,
}

impl<T> QueryResponse<T> {
    /// Create a successful response.
    #[must_use]
    pub fn success(data: T) -> Self {
        Self {
            success: true,
            data: Some(data),
            error: None,
        }
    }

    /// Create an error response.
    #[must_use]
    pub fn error(message: String) -> Self {
        Self {
            success: false,
            data: None,
            error: Some(message),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::mpsc;

    #[tokio::test]
    async fn test_request_with_timeout_success() {
        let (tx, mut rx) = mpsc::channel::<Request<String, String>>(10);

        // Spawn a handler that responds immediately
        tokio::spawn(async move {
            if let Some(request) = rx.recv().await {
                request.respond("world".to_string()).unwrap();
            }
        });

        let response = request_with_timeout(&tx, "hello".to_string(), Duration::from_secs(1))
            .await
            .unwrap();

        assert_eq!(response, "world");
    }

    #[tokio::test]
    async fn test_request_with_timeout_timeout() {
        let (tx, mut _rx) = mpsc::channel::<Request<String, String>>(10);

        // Don't respond, let it timeout
        let result =
            request_with_timeout(&tx, "hello".to_string(), Duration::from_millis(100)).await;

        assert!(matches!(result, Err(ChannelError::Timeout { .. })));
    }

    #[tokio::test]
    async fn test_request_with_timeout_channel_closed() {
        let (tx, rx) = mpsc::channel::<Request<String, String>>(10);

        // Drop receiver immediately
        drop(rx);

        let result = request_with_timeout(&tx, "hello".to_string(), Duration::from_secs(1)).await;

        assert!(matches!(result, Err(ChannelError::SendFailed { .. })));
    }

    #[tokio::test]
    async fn test_query_response_success() {
        let response = QueryResponse::success(42);
        assert!(response.success);
        assert_eq!(response.data, Some(42));
        assert!(response.error.is_none());
    }

    #[tokio::test]
    async fn test_query_response_error() {
        let response: QueryResponse<i32> = QueryResponse::error("test error".to_string());
        assert!(!response.success);
        assert!(response.data.is_none());
        assert_eq!(response.error, Some("test error".to_string()));
    }
}
