//! Application state for the API
//!
//! Holds shared state across all API handlers.

use crate::migration::{MigrationExecutor, MigrationPlanner};
use crate::service::SubnetManager;
use tokio::sync::RwLock;

/// Application state shared across handlers
pub struct AppState {
    /// Subnet manager service
    pub manager: RwLock<SubnetManager>,
    /// Migration planner
    pub migration_planner: RwLock<MigrationPlanner>,
    /// Migration executor
    pub migration_executor: RwLock<MigrationExecutor>,
}

impl AppState {
    /// Create new application state with default configuration
    pub fn new() -> Self {
        Self {
            manager: RwLock::new(SubnetManager::new()),
            migration_planner: RwLock::new(MigrationPlanner::new()),
            migration_executor: RwLock::new(MigrationExecutor::new()),
        }
    }

    /// Create with custom manager
    pub fn with_manager(manager: SubnetManager) -> Self {
        Self {
            manager: RwLock::new(manager),
            migration_planner: RwLock::new(MigrationPlanner::new()),
            migration_executor: RwLock::new(MigrationExecutor::new()),
        }
    }

    /// Create with all custom components
    pub fn with_components(
        manager: SubnetManager,
        planner: MigrationPlanner,
        executor: MigrationExecutor,
    ) -> Self {
        Self {
            manager: RwLock::new(manager),
            migration_planner: RwLock::new(planner),
            migration_executor: RwLock::new(executor),
        }
    }
}

impl Default for AppState {
    fn default() -> Self {
        Self::new()
    }
}
