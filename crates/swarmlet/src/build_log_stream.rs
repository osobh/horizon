//! WebSocket log streaming for build jobs
//!
//! This module provides real-time streaming of build logs over WebSocket
//! connections, allowing clients to receive build output as it happens
//! rather than polling the REST API.

use crate::build_job::{BuildJobStatus, BuildLogEntry};
use futures::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{broadcast, RwLock};
use tracing::{debug, info, warn};
use uuid::Uuid;
use warp::ws::{Message, WebSocket};

/// Maximum number of messages to buffer before dropping old ones
const CHANNEL_CAPACITY: usize = 1024;

/// Message sent over WebSocket to clients
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum BuildLogMessage {
    /// Connection established
    Connected { build_id: String },
    /// Build log entry
    LogEntry(BuildLogEntry),
    /// Build status update
    StatusUpdate { status: BuildJobStatus },
    /// Build completed
    Completed { exit_code: i32, duration_secs: f64 },
    /// Error occurred
    Error { message: String },
    /// Heartbeat (keep-alive)
    Heartbeat,
}

/// Manages WebSocket streaming for build logs
pub struct BuildLogStreamer {
    /// Broadcast channels per build ID
    channels: Arc<RwLock<HashMap<Uuid, broadcast::Sender<BuildLogMessage>>>>,
}

impl Default for BuildLogStreamer {
    fn default() -> Self {
        Self::new()
    }
}

impl BuildLogStreamer {
    /// Create a new build log streamer
    pub fn new() -> Self {
        Self {
            channels: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Get or create a broadcast channel for a build
    pub async fn get_channel(&self, build_id: Uuid) -> broadcast::Sender<BuildLogMessage> {
        let mut channels = self.channels.write().await;

        if let Some(sender) = channels.get(&build_id) {
            sender.clone()
        } else {
            let (sender, _) = broadcast::channel(CHANNEL_CAPACITY);
            channels.insert(build_id, sender.clone());
            debug!("Created broadcast channel for build {}", build_id);
            sender
        }
    }

    /// Broadcast a log entry to all subscribers of a build
    pub async fn broadcast_log(&self, build_id: Uuid, entry: BuildLogEntry) {
        let channels = self.channels.read().await;
        if let Some(sender) = channels.get(&build_id) {
            // Ignore send errors (no subscribers)
            let _ = sender.send(BuildLogMessage::LogEntry(entry));
        }
    }

    /// Broadcast a status update to all subscribers of a build
    pub async fn broadcast_status(&self, build_id: Uuid, status: BuildJobStatus) {
        let channels = self.channels.read().await;
        if let Some(sender) = channels.get(&build_id) {
            let _ = sender.send(BuildLogMessage::StatusUpdate { status });
        }
    }

    /// Broadcast build completion to all subscribers
    pub async fn broadcast_completed(&self, build_id: Uuid, exit_code: i32, duration_secs: f64) {
        let channels = self.channels.read().await;
        if let Some(sender) = channels.get(&build_id) {
            let _ = sender.send(BuildLogMessage::Completed {
                exit_code,
                duration_secs,
            });
        }
    }

    /// Broadcast an error to all subscribers
    pub async fn broadcast_error(&self, build_id: Uuid, message: String) {
        let channels = self.channels.read().await;
        if let Some(sender) = channels.get(&build_id) {
            let _ = sender.send(BuildLogMessage::Error { message });
        }
    }

    /// Handle a WebSocket connection for a build
    ///
    /// This method streams all log messages to the client until the build
    /// completes or the connection is closed.
    pub async fn handle_connection(&self, build_id: Uuid, ws: WebSocket) {
        let (mut ws_tx, mut ws_rx) = ws.split();

        // Get or create the broadcast channel for this build
        let sender = self.get_channel(build_id).await;
        let mut receiver = sender.subscribe();

        info!(
            "WebSocket client connected for build {} (subscribers: {})",
            build_id,
            sender.receiver_count()
        );

        // Send connection confirmation
        let connected_msg = BuildLogMessage::Connected {
            build_id: build_id.to_string(),
        };
        if let Ok(json) = serde_json::to_string(&connected_msg) {
            if ws_tx.send(Message::text(json)).await.is_err() {
                return;
            }
        }

        // Create heartbeat interval
        let mut heartbeat_interval = tokio::time::interval(std::time::Duration::from_secs(30));

        // Stream messages to the client
        loop {
            tokio::select! {
                // Handle incoming WebSocket messages
                result = ws_rx.next() => {
                    match result {
                        Some(Ok(msg)) => {
                            if msg.is_close() {
                                debug!("WebSocket close received for build {}", build_id);
                                break;
                            }
                            // Ignore other incoming messages (we're mostly sending)
                        }
                        Some(Err(e)) => {
                            debug!("WebSocket receive error: {}", e);
                            break;
                        }
                        None => {
                            debug!("WebSocket stream ended for build {}", build_id);
                            break;
                        }
                    }
                }

                // Send heartbeat periodically
                _ = heartbeat_interval.tick() => {
                    if let Ok(json) = serde_json::to_string(&BuildLogMessage::Heartbeat) {
                        if ws_tx.send(Message::text(json)).await.is_err() {
                            break;
                        }
                    }
                }

                // Receive broadcast messages
                result = receiver.recv() => {
                    match result {
                        Ok(msg) => {
                            let is_terminal = matches!(
                                msg,
                                BuildLogMessage::Completed { .. } | BuildLogMessage::Error { .. }
                            );

                            if let Ok(json) = serde_json::to_string(&msg) {
                                if ws_tx.send(Message::text(json)).await.is_err() {
                                    break;
                                }
                            }

                            // Close connection after terminal message
                            if is_terminal {
                                debug!("Build {} reached terminal state, closing WebSocket", build_id);
                                break;
                            }
                        }
                        Err(broadcast::error::RecvError::Lagged(n)) => {
                            warn!("WebSocket client lagged {} messages for build {}", n, build_id);
                            // Continue - we'll catch up with newer messages
                        }
                        Err(broadcast::error::RecvError::Closed) => {
                            debug!("Broadcast channel closed for build {}", build_id);
                            break;
                        }
                    }
                }
            }
        }

        // Send close message
        let _ = ws_tx.close().await;

        info!(
            "WebSocket client disconnected from build {} (remaining: {})",
            build_id,
            sender.receiver_count().saturating_sub(1)
        );
    }

    /// Clean up the channel for a build (call after build completes)
    pub async fn cleanup_channel(&self, build_id: Uuid) {
        let mut channels = self.channels.write().await;
        if channels.remove(&build_id).is_some() {
            debug!("Cleaned up broadcast channel for build {}", build_id);
        }
    }

    /// Get the number of subscribers for a build
    pub async fn subscriber_count(&self, build_id: Uuid) -> usize {
        let channels = self.channels.read().await;
        channels
            .get(&build_id)
            .map(|s| s.receiver_count())
            .unwrap_or(0)
    }

    /// Get the number of active builds with channels
    pub async fn active_channel_count(&self) -> usize {
        let channels = self.channels.read().await;
        channels.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::build_job::LogStream;
    use chrono::Utc;

    #[tokio::test]
    async fn test_broadcast_log() {
        let streamer = BuildLogStreamer::new();
        let build_id = Uuid::new_v4();

        // Get channel and subscribe
        let sender = streamer.get_channel(build_id).await;
        let mut receiver = sender.subscribe();

        // Broadcast a log entry
        let entry = BuildLogEntry {
            timestamp: Utc::now(),
            stream: LogStream::Stdout,
            message: "Hello, world!".to_string(),
        };
        streamer.broadcast_log(build_id, entry.clone()).await;

        // Receive and verify
        let msg = receiver.recv().await.unwrap();
        if let BuildLogMessage::LogEntry(received) = msg {
            assert_eq!(received.message, "Hello, world!");
        } else {
            panic!("Expected LogEntry message");
        }
    }

    #[tokio::test]
    async fn test_broadcast_status() {
        let streamer = BuildLogStreamer::new();
        let build_id = Uuid::new_v4();

        let sender = streamer.get_channel(build_id).await;
        let mut receiver = sender.subscribe();

        streamer
            .broadcast_status(build_id, BuildJobStatus::Building)
            .await;

        let msg = receiver.recv().await.unwrap();
        if let BuildLogMessage::StatusUpdate { status } = msg {
            assert_eq!(status, BuildJobStatus::Building);
        } else {
            panic!("Expected StatusUpdate message");
        }
    }

    #[tokio::test]
    async fn test_multiple_subscribers() {
        let streamer = BuildLogStreamer::new();
        let build_id = Uuid::new_v4();

        let sender = streamer.get_channel(build_id).await;
        let mut receiver1 = sender.subscribe();
        let mut receiver2 = sender.subscribe();

        assert_eq!(streamer.subscriber_count(build_id).await, 2);

        streamer.broadcast_completed(build_id, 0, 10.5).await;

        // Both receivers should get the message
        let msg1 = receiver1.recv().await.unwrap();
        let msg2 = receiver2.recv().await.unwrap();

        assert!(matches!(msg1, BuildLogMessage::Completed { .. }));
        assert!(matches!(msg2, BuildLogMessage::Completed { .. }));
    }

    #[tokio::test]
    async fn test_cleanup_channel() {
        let streamer = BuildLogStreamer::new();
        let build_id = Uuid::new_v4();

        // Create channel
        let _ = streamer.get_channel(build_id).await;
        assert_eq!(streamer.active_channel_count().await, 1);

        // Cleanup
        streamer.cleanup_channel(build_id).await;
        assert_eq!(streamer.active_channel_count().await, 0);
    }

    #[tokio::test]
    async fn test_message_serialization() {
        let msg = BuildLogMessage::Connected {
            build_id: "test-123".to_string(),
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("connected"));
        assert!(json.contains("test-123"));

        let entry = BuildLogEntry {
            timestamp: Utc::now(),
            stream: LogStream::Stderr,
            message: "error occurred".to_string(),
        };
        let msg = BuildLogMessage::LogEntry(entry);
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("log_entry"));
        assert!(json.contains("error occurred"));
    }
}
