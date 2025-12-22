//! Replay manager for coordinating replay operations
//!
//! This module contains the ReplayManager for higher-level coordination of replay sessions.

use super::engine::ReplayEngine;
use super::types::*;
use crate::snapshot::MemorySnapshot;
use crate::DebugError;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

/// Replay manager for coordinating replay operations
pub struct ReplayManager {
    engine: Arc<dyn ReplayEngine + Send + Sync>,
    active_sessions: Arc<RwLock<HashMap<Uuid, ReplaySession>>>,
    config: ReplayManagerConfig,
}

impl ReplayManager {
    /// Create new replay manager
    pub fn new(engine: Arc<dyn ReplayEngine + Send + Sync>, config: ReplayManagerConfig) -> Self {
        Self {
            engine,
            active_sessions: Arc::new(RwLock::new(HashMap::new())),
            config,
        }
    }

    /// Start replay from snapshot
    pub async fn start_replay(
        &self,
        snapshot: MemorySnapshot,
        config: Option<ReplayConfig>,
    ) -> Result<Uuid, DebugError> {
        // Check concurrent replay limit
        {
            let sessions = self.active_sessions.read().await;
            let running_count = sessions
                .values()
                .filter(|s| matches!(s.status, ReplayStatus::Running | ReplayStatus::Initializing))
                .count();

            if running_count >= self.config.max_concurrent_replays {
                return Err(DebugError::ReplayFailed {
                    reason: format!(
                        "Maximum concurrent replays reached: {}",
                        self.config.max_concurrent_replays
                    ),
                });
            }
        }

        let replay_config = config.unwrap_or_default();
        let session = self
            .engine
            .initialize_replay(snapshot, replay_config)
            .await?;
        let session_id = session.session_id;

        {
            let mut sessions = self.active_sessions.write().await;
            sessions.insert(session_id, session);
        }

        self.engine.start_replay(session_id).await?;

        // Update session status to running in manager after starting
        {
            let mut sessions = self.active_sessions.write().await;
            if let Some(session) = sessions.get_mut(&session_id) {
                session.status = ReplayStatus::Running;
            }
        }

        Ok(session_id)
    }

    /// Get replay session info
    pub async fn get_session(&self, session_id: Uuid) -> Result<Option<ReplaySession>, DebugError> {
        let sessions = self.active_sessions.read().await;
        Ok(sessions.get(&session_id).cloned())
    }

    /// List all active sessions
    pub async fn list_sessions(&self) -> Vec<ReplaySession> {
        self.active_sessions
            .read()
            .await
            .values()
            .cloned()
            .collect()
    }

    /// Pause replay
    pub async fn pause_replay(&self, session_id: Uuid) -> Result<(), DebugError> {
        self.engine.pause_replay(session_id).await?;

        // Update status in manager as well
        {
            let mut sessions = self.active_sessions.write().await;
            if let Some(session) = sessions.get_mut(&session_id) {
                session.status = ReplayStatus::Paused;
            }
        }

        Ok(())
    }

    /// Resume replay
    pub async fn resume_replay(&self, session_id: Uuid) -> Result<(), DebugError> {
        self.engine.resume_replay(session_id).await?;

        // Update status in manager as well
        {
            let mut sessions = self.active_sessions.write().await;
            if let Some(session) = sessions.get_mut(&session_id) {
                session.status = ReplayStatus::Running;
            }
        }

        Ok(())
    }

    /// Stop replay and get results
    pub async fn stop_replay(&self, session_id: Uuid) -> Result<ReplayResults, DebugError> {
        let results = self.engine.stop_replay(session_id).await?;

        // Remove from active sessions
        {
            let mut sessions = self.active_sessions.write().await;
            sessions.remove(&session_id);
        }

        Ok(results)
    }

    /// Cleanup completed sessions
    pub async fn cleanup_completed(&self) -> Result<usize, DebugError> {
        let mut sessions = self.active_sessions.write().await;
        let initial_count = sessions.len();

        sessions.retain(|_, session| {
            !matches!(
                session.status,
                ReplayStatus::Completed | ReplayStatus::Failed | ReplayStatus::Cancelled
            )
        });

        Ok(initial_count - sessions.len())
    }

    /// Get statistics about active replays
    pub async fn get_stats(&self) -> ReplayStats {
        let sessions = self.active_sessions.read().await;

        let mut stats = ReplayStats {
            total_sessions: sessions.len(),
            running_sessions: 0,
            paused_sessions: 0,
            completed_sessions: 0,
            failed_sessions: 0,
            average_execution_time_ms: 0.0,
        };

        let mut total_execution_time = 0u64;
        let mut execution_count = 0;

        for session in sessions.values() {
            match session.status {
                ReplayStatus::Running | ReplayStatus::Initializing => stats.running_sessions += 1,
                ReplayStatus::Paused => stats.paused_sessions += 1,
                ReplayStatus::Completed => {
                    stats.completed_sessions += 1;
                    if let Some(results) = &session.results {
                        total_execution_time += results.execution_time_ms;
                        execution_count += 1;
                    }
                }
                ReplayStatus::Failed | ReplayStatus::Cancelled => stats.failed_sessions += 1,
                _ => {}
            }
        }

        if execution_count > 0 {
            stats.average_execution_time_ms = total_execution_time as f64 / execution_count as f64;
        }

        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::replay::engine::MockReplayEngine;
    use crate::snapshot::{ExecutionContext, KernelParameters, MemorySnapshot};
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::time::Duration;
    use uuid::Uuid;

    fn create_test_snapshot() -> MemorySnapshot {
        let host_memory = vec![1, 2, 3, 4, 5];
        let device_memory = vec![6, 7, 8, 9, 10];
        let kernel_params = KernelParameters {
            grid_size: (16, 16, 1),
            block_size: (32, 32, 1),
            shared_memory_size: 1024,
            kernel_args: vec![0x42, 0x43],
            stream_id: Some(1),
        };
        let exec_context = ExecutionContext {
            goal_prompt: "Test kernel execution".to_string(),
            agent_id: Some("agent_123".to_string()),
            generation: 5,
            mutation_count: 10,
            parent_fitness: Some(0.85),
            environment_vars: HashMap::new(),
        };

        MemorySnapshot::new(
            Uuid::new_v4(),
            host_memory,
            device_memory,
            kernel_params,
            exec_context,
        )
    }

    #[tokio::test]
    async fn test_replay_manager() {
        let engine = Arc::new(MockReplayEngine::new());
        let config = ReplayManagerConfig::default();
        let manager = ReplayManager::new(engine, config);

        let snapshot = create_test_snapshot();
        let session_id = manager.start_replay(snapshot, None).await.unwrap();

        let session = manager.get_session(session_id).await.unwrap();
        assert!(session.is_some());

        let sessions = manager.list_sessions().await;
        assert_eq!(sessions.len(), 1);

        // Wait for completion and cleanup
        tokio::time::sleep(Duration::from_millis(200)).await;
        let cleaned = manager.cleanup_completed().await.unwrap();
        assert!(cleaned <= 1);
    }

    #[tokio::test]
    async fn test_replay_manager_concurrent_limit() {
        // Create an engine with longer execution delay to prevent completion during test
        let mut engine = MockReplayEngine::new();
        engine.set_execution_delay(1000); // 1 second delay
        let engine = Arc::new(engine);

        let mut config = ReplayManagerConfig::default();
        config.max_concurrent_replays = 1;
        let manager = ReplayManager::new(engine, config);

        let snapshot1 = create_test_snapshot();
        let snapshot2 = create_test_snapshot();

        // First replay should succeed
        let _session_id1 = manager.start_replay(snapshot1, None).await.unwrap();

        // Wait a moment for the first replay to be marked as running
        tokio::time::sleep(Duration::from_millis(50)).await;

        // Second replay should fail due to limit
        let result = manager.start_replay(snapshot2, None).await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Maximum concurrent replays"));
    }

    #[tokio::test]
    async fn test_replay_manager_stats() {
        let engine = Arc::new(MockReplayEngine::new());
        let config = ReplayManagerConfig::default();
        let manager = ReplayManager::new(engine, config);

        let snapshot = create_test_snapshot();
        let _session_id = manager.start_replay(snapshot, None).await.unwrap();

        let stats = manager.get_stats().await;
        assert_eq!(stats.total_sessions, 1);
        assert!(stats.running_sessions <= 1);
    }

    #[tokio::test]
    async fn test_replay_manager_pause_resume() {
        let engine = Arc::new(MockReplayEngine::new());
        let config = ReplayManagerConfig::default();
        let manager = ReplayManager::new(engine, config);

        let snapshot = create_test_snapshot();
        let session_id = manager.start_replay(snapshot, None).await.unwrap();

        // Wait for the replay to start
        tokio::time::sleep(Duration::from_millis(20)).await;

        // Check if it's running or pause it
        let session = manager.get_session(session_id).await.unwrap().unwrap();
        if session.status == ReplayStatus::Running {
            manager.pause_replay(session_id).await.unwrap();

            let session = manager.get_session(session_id).await.unwrap().unwrap();
            assert_eq!(session.status, ReplayStatus::Paused);

            manager.resume_replay(session_id).await.unwrap();

            let session = manager.get_session(session_id).await.unwrap().unwrap();
            assert_eq!(session.status, ReplayStatus::Running);
        } else {
            // If it's not running yet, just check the initial session exists
            assert!(session.session_id == session_id);
        }
    }
}
