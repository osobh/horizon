use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::{AgentError, Result};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RollbackPoint {
    pub id: Uuid,
    pub action_id: String,
    pub state_snapshot: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl RollbackPoint {
    pub fn new(action_id: String, state_snapshot: String) -> Self {
        Self {
            id: Uuid::new_v4(),
            action_id,
            state_snapshot,
            timestamp: chrono::Utc::now(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RollbackOperation {
    pub id: Uuid,
    pub rollback_point_id: Uuid,
    pub status: RollbackStatus,
    pub error: Option<String>,
    pub started_at: chrono::DateTime<chrono::Utc>,
    pub completed_at: Option<chrono::DateTime<chrono::Utc>>,
}

impl RollbackOperation {
    pub fn new(rollback_point_id: Uuid) -> Self {
        Self {
            id: Uuid::new_v4(),
            rollback_point_id,
            status: RollbackStatus::Pending,
            error: None,
            started_at: chrono::Utc::now(),
            completed_at: None,
        }
    }

    pub fn mark_in_progress(&mut self) {
        self.status = RollbackStatus::InProgress;
    }

    pub fn mark_success(&mut self) {
        self.status = RollbackStatus::Success;
        self.completed_at = Some(chrono::Utc::now());
    }

    pub fn mark_failed(&mut self, error: String) {
        self.status = RollbackStatus::Failed;
        self.error = Some(error);
        self.completed_at = Some(chrono::Utc::now());
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RollbackStatus {
    Pending,
    InProgress,
    Success,
    Failed,
}

pub struct RollbackManager {
    rollback_points: Vec<RollbackPoint>,
    operations: Vec<RollbackOperation>,
    max_rollback_points: usize,
}

impl RollbackManager {
    pub fn new(max_rollback_points: usize) -> Result<Self> {
        if max_rollback_points == 0 {
            return Err(AgentError::InvalidConfiguration(
                "Max rollback points must be greater than 0".to_string(),
            ));
        }

        Ok(Self {
            rollback_points: Vec::new(),
            operations: Vec::new(),
            max_rollback_points,
        })
    }

    pub fn create_rollback_point(
        &mut self,
        action_id: String,
        state_snapshot: String,
    ) -> Result<Uuid> {
        if self.rollback_points.len() >= self.max_rollback_points {
            // Remove oldest rollback point
            self.rollback_points.remove(0);
        }

        let point = RollbackPoint::new(action_id, state_snapshot);
        let id = point.id;
        self.rollback_points.push(point);
        Ok(id)
    }

    pub fn get_rollback_point(&self, id: Uuid) -> Result<&RollbackPoint> {
        self.rollback_points
            .iter()
            .find(|p| p.id == id)
            .ok_or_else(|| AgentError::RollbackFailed("Rollback point not found".to_string()))
    }

    pub fn initiate_rollback(&mut self, rollback_point_id: Uuid) -> Result<Uuid> {
        // Verify rollback point exists
        self.get_rollback_point(rollback_point_id)?;

        let operation = RollbackOperation::new(rollback_point_id);
        let id = operation.id;
        self.operations.push(operation);
        Ok(id)
    }

    pub fn execute_rollback(&mut self, operation_id: Uuid) -> Result<String> {
        // First, get the rollback point ID and validate the operation
        let rollback_point_id = {
            let operation = self
                .operations
                .iter_mut()
                .find(|op| op.id == operation_id)
                .ok_or_else(|| AgentError::RollbackFailed("Operation not found".to_string()))?;

            if operation.status != RollbackStatus::Pending {
                return Err(AgentError::RollbackFailed(format!(
                    "Operation already in status: {:?}",
                    operation.status
                )));
            }

            operation.mark_in_progress();
            operation.rollback_point_id
        };

        // Now get the rollback point
        let point = self.get_rollback_point(rollback_point_id)?;

        // In a real implementation, this would restore the state
        // For now, we just return the snapshot and mark success
        let snapshot = point.state_snapshot.clone();

        // Mark operation as success
        let operation = self
            .operations
            .iter_mut()
            .find(|op| op.id == operation_id)
            .ok_or_else(|| AgentError::RollbackFailed("Operation not found".to_string()))?;
        operation.mark_success();

        Ok(snapshot)
    }

    pub fn mark_rollback_failed(&mut self, operation_id: Uuid, error: String) -> Result<()> {
        let operation = self
            .operations
            .iter_mut()
            .find(|op| op.id == operation_id)
            .ok_or_else(|| AgentError::RollbackFailed("Operation not found".to_string()))?;

        operation.mark_failed(error);
        Ok(())
    }

    pub fn get_recent_rollbacks(&self, limit: usize) -> Vec<&RollbackOperation> {
        let start = if self.operations.len() > limit {
            self.operations.len() - limit
        } else {
            0
        };

        self.operations[start..].iter().rev().collect()
    }

    pub fn cleanup_old_points(&mut self, max_age_hours: i64) {
        let cutoff = chrono::Utc::now() - chrono::Duration::hours(max_age_hours);

        self.rollback_points
            .retain(|point| point.timestamp > cutoff);
    }

    pub fn get_rollback_point_count(&self) -> usize {
        self.rollback_points.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rollback_point_creation() {
        let point = RollbackPoint::new("action1".to_string(), "state1".to_string());
        assert_eq!(point.action_id, "action1");
        assert_eq!(point.state_snapshot, "state1");
    }

    #[test]
    fn test_rollback_operation_creation() {
        let point_id = Uuid::new_v4();
        let op = RollbackOperation::new(point_id);
        assert_eq!(op.rollback_point_id, point_id);
        assert_eq!(op.status, RollbackStatus::Pending);
        assert!(op.error.is_none());
        assert!(op.completed_at.is_none());
    }

    #[test]
    fn test_rollback_operation_lifecycle() {
        let point_id = Uuid::new_v4();
        let mut op = RollbackOperation::new(point_id);

        op.mark_in_progress();
        assert_eq!(op.status, RollbackStatus::InProgress);

        op.mark_success();
        assert_eq!(op.status, RollbackStatus::Success);
        assert!(op.completed_at.is_some());
    }

    #[test]
    fn test_rollback_operation_failure() {
        let point_id = Uuid::new_v4();
        let mut op = RollbackOperation::new(point_id);

        op.mark_failed("Test error".to_string());
        assert_eq!(op.status, RollbackStatus::Failed);
        assert_eq!(op.error.as_ref().unwrap(), "Test error");
        assert!(op.completed_at.is_some());
    }

    #[test]
    fn test_rollback_manager_creation() {
        let manager = RollbackManager::new(10);
        assert!(manager.is_ok());

        let manager = manager.unwrap();
        assert_eq!(manager.get_rollback_point_count(), 0);
    }

    #[test]
    fn test_rollback_manager_zero_capacity() {
        let manager = RollbackManager::new(0);
        assert!(manager.is_err());
    }

    #[test]
    fn test_rollback_manager_create_point() {
        let mut manager = RollbackManager::new(10).unwrap();

        let id = manager
            .create_rollback_point("action1".to_string(), "state1".to_string())
            .unwrap();

        assert_eq!(manager.get_rollback_point_count(), 1);

        let point = manager.get_rollback_point(id).unwrap();
        assert_eq!(point.action_id, "action1");
    }

    #[test]
    fn test_rollback_manager_eviction() {
        let mut manager = RollbackManager::new(2).unwrap();

        let id1 = manager
            .create_rollback_point("action1".to_string(), "state1".to_string())
            .unwrap();
        manager
            .create_rollback_point("action2".to_string(), "state2".to_string())
            .unwrap();
        manager
            .create_rollback_point("action3".to_string(), "state3".to_string())
            .unwrap();

        assert_eq!(manager.get_rollback_point_count(), 2);
        assert!(manager.get_rollback_point(id1).is_err());
    }

    #[test]
    fn test_rollback_manager_initiate_rollback() {
        let mut manager = RollbackManager::new(10).unwrap();

        let point_id = manager
            .create_rollback_point("action1".to_string(), "state1".to_string())
            .unwrap();

        let op_id = manager.initiate_rollback(point_id).unwrap();
        assert_eq!(manager.operations.len(), 1);

        let op = &manager.operations[0];
        assert_eq!(op.id, op_id);
        assert_eq!(op.rollback_point_id, point_id);
    }

    #[test]
    fn test_rollback_manager_initiate_rollback_nonexistent() {
        let mut manager = RollbackManager::new(10).unwrap();

        let result = manager.initiate_rollback(Uuid::new_v4());
        assert!(result.is_err());
    }

    #[test]
    fn test_rollback_manager_execute_rollback() {
        let mut manager = RollbackManager::new(10).unwrap();

        let point_id = manager
            .create_rollback_point("action1".to_string(), "state1".to_string())
            .unwrap();

        let op_id = manager.initiate_rollback(point_id).unwrap();
        let snapshot = manager.execute_rollback(op_id).unwrap();

        assert_eq!(snapshot, "state1");

        let op = &manager.operations[0];
        assert_eq!(op.status, RollbackStatus::Success);
    }

    #[test]
    fn test_rollback_manager_execute_rollback_nonexistent() {
        let mut manager = RollbackManager::new(10).unwrap();

        let result = manager.execute_rollback(Uuid::new_v4());
        assert!(result.is_err());
    }

    #[test]
    fn test_rollback_manager_execute_rollback_twice() {
        let mut manager = RollbackManager::new(10).unwrap();

        let point_id = manager
            .create_rollback_point("action1".to_string(), "state1".to_string())
            .unwrap();

        let op_id = manager.initiate_rollback(point_id).unwrap();
        manager.execute_rollback(op_id).unwrap();

        let result = manager.execute_rollback(op_id);
        assert!(result.is_err());
    }

    #[test]
    fn test_rollback_manager_mark_failed() {
        let mut manager = RollbackManager::new(10).unwrap();

        let point_id = manager
            .create_rollback_point("action1".to_string(), "state1".to_string())
            .unwrap();

        let op_id = manager.initiate_rollback(point_id).unwrap();

        manager
            .mark_rollback_failed(op_id, "Test error".to_string())
            .unwrap();

        let op = &manager.operations[0];
        assert_eq!(op.status, RollbackStatus::Failed);
        assert_eq!(op.error.as_ref().unwrap(), "Test error");
    }

    #[test]
    fn test_rollback_manager_get_recent_rollbacks() {
        let mut manager = RollbackManager::new(10).unwrap();

        for i in 0..5 {
            let point_id = manager
                .create_rollback_point(format!("action{}", i), format!("state{}", i))
                .unwrap();
            manager.initiate_rollback(point_id).unwrap();
        }

        let recent = manager.get_recent_rollbacks(3);
        assert_eq!(recent.len(), 3);
    }

    #[test]
    fn test_rollback_manager_cleanup_old_points() {
        let mut manager = RollbackManager::new(10).unwrap();

        manager
            .create_rollback_point("action1".to_string(), "state1".to_string())
            .unwrap();

        assert_eq!(manager.get_rollback_point_count(), 1);

        manager.cleanup_old_points(0);
        assert_eq!(manager.get_rollback_point_count(), 0);
    }
}
