//! Migration execution with dual-stack support
//!
//! Executes migrations using a state machine approach for reliable,
//! resumable, and rollback-capable migrations.

use super::{Migration, MigrationPlan, MigrationReason, MigrationStatus};
use crate::{Error, Result};
use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::Ipv4Addr;
use std::sync::Arc;
use uuid::Uuid;

/// Steps in the migration state machine
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MigrationStep {
    /// Initial state, not started
    NotStarted,
    /// Allocating target IP
    AllocatingIp,
    /// Generating WireGuard configuration
    GeneratingConfig,
    /// Enabling dual-stack (node on both IPs)
    EnablingDualStack,
    /// Propagating new config to peers
    PropagatingToPeers,
    /// Verifying connectivity on new address
    VerifyingConnectivity,
    /// Cutting over to new address
    CuttingOver,
    /// Cleaning up old resources
    CleaningUp,
    /// Migration completed
    Completed,
    /// Rolling back changes
    RollingBack,
    /// Rollback completed
    RolledBack,
    /// Migration failed
    Failed,
}

impl MigrationStep {
    /// Get the next step in the normal flow
    pub fn next(&self) -> Option<MigrationStep> {
        match self {
            MigrationStep::NotStarted => Some(MigrationStep::AllocatingIp),
            MigrationStep::AllocatingIp => Some(MigrationStep::GeneratingConfig),
            MigrationStep::GeneratingConfig => Some(MigrationStep::EnablingDualStack),
            MigrationStep::EnablingDualStack => Some(MigrationStep::PropagatingToPeers),
            MigrationStep::PropagatingToPeers => Some(MigrationStep::VerifyingConnectivity),
            MigrationStep::VerifyingConnectivity => Some(MigrationStep::CuttingOver),
            MigrationStep::CuttingOver => Some(MigrationStep::CleaningUp),
            MigrationStep::CleaningUp => Some(MigrationStep::Completed),
            _ => None,
        }
    }

    /// Check if this step is reversible
    pub fn is_reversible(&self) -> bool {
        matches!(
            self,
            MigrationStep::AllocatingIp
                | MigrationStep::GeneratingConfig
                | MigrationStep::EnablingDualStack
                | MigrationStep::PropagatingToPeers
                | MigrationStep::VerifyingConnectivity
        )
    }

    /// Check if this is a terminal state
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            MigrationStep::Completed | MigrationStep::RolledBack | MigrationStep::Failed
        )
    }

    /// Get rollback step for current step
    pub fn rollback_step(&self) -> MigrationStep {
        if self.is_reversible() {
            MigrationStep::RollingBack
        } else {
            MigrationStep::Failed
        }
    }
}

/// Migration execution options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationOptions {
    /// Timeout for each step in seconds
    pub step_timeout_secs: u64,
    /// Number of connectivity verification attempts
    pub verify_attempts: u32,
    /// Delay between verification attempts in milliseconds
    pub verify_delay_ms: u64,
    /// Whether to auto-rollback on failure
    pub auto_rollback: bool,
    /// Whether to wait for peer acknowledgment
    pub wait_for_peers: bool,
    /// Minimum peers that must acknowledge (percentage)
    pub min_peer_ack_percent: u8,
}

impl Default for MigrationOptions {
    fn default() -> Self {
        Self {
            step_timeout_secs: 60,
            verify_attempts: 5,
            verify_delay_ms: 2000,
            auto_rollback: true,
            wait_for_peers: true,
            min_peer_ack_percent: 80,
        }
    }
}

/// Progress tracking for a migration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationProgress {
    /// Migration ID
    pub migration_id: Uuid,
    /// Current step
    pub current_step: MigrationStep,
    /// Step started at
    pub step_started_at: Option<DateTime<Utc>>,
    /// Steps completed
    pub steps_completed: Vec<CompletedStep>,
    /// Overall progress (0-100)
    pub progress_percent: u8,
    /// Peers notified count
    pub peers_notified: usize,
    /// Peers acknowledged count
    pub peers_acknowledged: usize,
    /// Connectivity verified
    pub connectivity_verified: bool,
    /// Last update time
    pub updated_at: DateTime<Utc>,
}

impl MigrationProgress {
    /// Create new progress tracker
    pub fn new(migration_id: Uuid) -> Self {
        Self {
            migration_id,
            current_step: MigrationStep::NotStarted,
            step_started_at: None,
            steps_completed: Vec::new(),
            progress_percent: 0,
            peers_notified: 0,
            peers_acknowledged: 0,
            connectivity_verified: false,
            updated_at: Utc::now(),
        }
    }

    /// Update to next step
    pub fn advance_to(&mut self, step: MigrationStep) {
        if let Some(start) = self.step_started_at {
            self.steps_completed.push(CompletedStep {
                step: self.current_step,
                started_at: start,
                completed_at: Utc::now(),
                success: true,
                error: None,
            });
        }

        self.current_step = step;
        self.step_started_at = Some(Utc::now());
        self.updated_at = Utc::now();
        self.update_progress();
    }

    /// Record step failure
    pub fn record_failure(&mut self, error: String) {
        if let Some(start) = self.step_started_at {
            self.steps_completed.push(CompletedStep {
                step: self.current_step,
                started_at: start,
                completed_at: Utc::now(),
                success: false,
                error: Some(error),
            });
        }
        self.updated_at = Utc::now();
    }

    /// Update progress percentage
    fn update_progress(&mut self) {
        let total_steps = 8; // NotStarted to Completed
        let current = match self.current_step {
            MigrationStep::NotStarted => 0,
            MigrationStep::AllocatingIp => 1,
            MigrationStep::GeneratingConfig => 2,
            MigrationStep::EnablingDualStack => 3,
            MigrationStep::PropagatingToPeers => 4,
            MigrationStep::VerifyingConnectivity => 5,
            MigrationStep::CuttingOver => 6,
            MigrationStep::CleaningUp => 7,
            MigrationStep::Completed => 8,
            _ => self.progress_percent as usize / (100 / total_steps),
        };
        self.progress_percent = ((current * 100) / total_steps) as u8;
    }
}

/// Record of a completed step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletedStep {
    /// The step
    pub step: MigrationStep,
    /// When step started
    pub started_at: DateTime<Utc>,
    /// When step completed
    pub completed_at: DateTime<Utc>,
    /// Whether step succeeded
    pub success: bool,
    /// Error message if failed
    pub error: Option<String>,
}

impl CompletedStep {
    /// Get step duration
    pub fn duration(&self) -> chrono::Duration {
        self.completed_at - self.started_at
    }
}

/// State machine for a single migration
#[derive(Debug, Clone)]
pub struct MigrationStateMachine {
    /// The migration
    pub migration: Migration,
    /// Current step
    pub step: MigrationStep,
    /// Progress tracker
    pub progress: MigrationProgress,
    /// Execution options
    pub options: MigrationOptions,
    /// Rollback data (for reversing changes)
    pub rollback_data: RollbackData,
    /// Step retry count
    pub step_retries: u32,
    /// Max retries per step
    pub max_step_retries: u32,
}

/// Data needed for rollback
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RollbackData {
    /// Target IP that was allocated
    pub allocated_ip: Option<Ipv4Addr>,
    /// WireGuard config that was generated
    pub generated_config: Option<String>,
    /// Peers that were notified
    pub notified_peers: Vec<Uuid>,
    /// Original peer configs (for restoration)
    pub original_peer_configs: HashMap<Uuid, String>,
    /// Whether dual-stack was enabled
    pub dual_stack_enabled: bool,
}

impl MigrationStateMachine {
    /// Create a new state machine from a migration plan
    pub fn new(plan: MigrationPlan, options: MigrationOptions) -> Self {
        let progress = MigrationProgress::new(plan.migration.id);

        Self {
            migration: plan.migration,
            step: MigrationStep::NotStarted,
            progress,
            options,
            rollback_data: RollbackData::default(),
            step_retries: 0,
            max_step_retries: 3,
        }
    }

    /// Start the migration
    pub fn start(&mut self) -> Result<()> {
        if self.step != MigrationStep::NotStarted {
            return Err(Error::MigrationFailed(
                "Migration already started".to_string(),
            ));
        }

        self.migration.start();
        self.advance_to(MigrationStep::AllocatingIp)?;
        Ok(())
    }

    /// Advance to next step
    pub fn advance(&mut self) -> Result<()> {
        let next_step = self.step.next().ok_or_else(|| {
            Error::MigrationFailed(format!("No next step from {:?}", self.step))
        })?;

        self.advance_to(next_step)
    }

    /// Advance to a specific step
    pub fn advance_to(&mut self, step: MigrationStep) -> Result<()> {
        self.step = step;
        self.step_retries = 0;
        self.progress.advance_to(step);

        // Update migration status based on step
        self.migration.current_step = Some(step);
        match step {
            MigrationStep::EnablingDualStack
            | MigrationStep::PropagatingToPeers
            | MigrationStep::VerifyingConnectivity => {
                self.migration.status = MigrationStatus::DualStack;
            }
            MigrationStep::Completed => {
                self.migration.complete();
            }
            MigrationStep::RollingBack => {
                self.migration.status = MigrationStatus::RollingBack;
            }
            MigrationStep::RolledBack => {
                self.migration.status = MigrationStatus::RolledBack;
            }
            MigrationStep::Failed => {
                self.migration.status = MigrationStatus::Failed;
            }
            _ => {}
        }

        Ok(())
    }

    /// Record step completion with data
    pub fn complete_step(&mut self, step: MigrationStep) -> Result<()> {
        if self.step != step {
            return Err(Error::MigrationFailed(format!(
                "Expected step {:?}, got {:?}",
                self.step, step
            )));
        }
        self.advance()
    }

    /// Record step failure
    pub fn fail_step(&mut self, error: String) -> Result<()> {
        self.step_retries += 1;
        self.progress.record_failure(error.clone());

        if self.step_retries >= self.max_step_retries {
            self.migration.record_retry(error.clone());

            if self.options.auto_rollback && self.step.is_reversible() {
                return self.start_rollback(error);
            } else {
                self.migration.fail(error);
                self.step = MigrationStep::Failed;
            }
        }

        Ok(())
    }

    /// Start rollback
    pub fn start_rollback(&mut self, reason: String) -> Result<()> {
        if !self.step.is_reversible() && !matches!(self.step, MigrationStep::RollingBack) {
            return Err(Error::MigrationFailed(format!(
                "Cannot rollback from step {:?}",
                self.step
            )));
        }

        self.migration.last_error = Some(reason);
        self.advance_to(MigrationStep::RollingBack)?;
        Ok(())
    }

    /// Complete rollback
    pub fn complete_rollback(&mut self) -> Result<()> {
        self.advance_to(MigrationStep::RolledBack)?;
        Ok(())
    }

    /// Check if migration is complete
    pub fn is_complete(&self) -> bool {
        self.step.is_terminal()
    }

    /// Check if migration succeeded
    pub fn is_success(&self) -> bool {
        self.step == MigrationStep::Completed
    }

    /// Record allocated IP for rollback
    pub fn record_allocated_ip(&mut self, ip: Ipv4Addr) {
        self.rollback_data.allocated_ip = Some(ip);
        self.migration.target_ip = Some(ip);
    }

    /// Record generated config for rollback
    pub fn record_generated_config(&mut self, config: String) {
        self.rollback_data.generated_config = Some(config);
    }

    /// Record that dual-stack was enabled
    pub fn record_dual_stack_enabled(&mut self) {
        self.rollback_data.dual_stack_enabled = true;
    }

    /// Record notified peer
    pub fn record_peer_notified(&mut self, peer_id: Uuid) {
        self.rollback_data.notified_peers.push(peer_id);
        self.progress.peers_notified = self.rollback_data.notified_peers.len();
    }

    /// Record peer acknowledgment
    pub fn record_peer_ack(&mut self) {
        self.progress.peers_acknowledged += 1;
    }

    /// Record connectivity verified
    pub fn record_connectivity_verified(&mut self) {
        self.progress.connectivity_verified = true;
    }

    /// Get current progress
    pub fn progress(&self) -> &MigrationProgress {
        &self.progress
    }

    /// Get rollback data
    pub fn rollback_data(&self) -> &RollbackData {
        &self.rollback_data
    }
}

/// Handle for tracking a running migration
#[derive(Debug, Clone)]
pub struct MigrationHandle {
    /// Migration ID
    pub id: Uuid,
    /// Node being migrated
    pub node_id: Uuid,
    /// Source subnet
    pub source_subnet_id: Uuid,
    /// Target subnet
    pub target_subnet_id: Uuid,
    /// Created at
    pub created_at: DateTime<Utc>,
}

impl From<&Migration> for MigrationHandle {
    fn from(m: &Migration) -> Self {
        Self {
            id: m.id,
            node_id: m.node_id,
            source_subnet_id: m.source_subnet_id,
            target_subnet_id: m.target_subnet_id,
            created_at: m.created_at,
        }
    }
}

/// Migration executor service
pub struct MigrationExecutor {
    /// Active migrations by ID
    active_migrations: Arc<RwLock<HashMap<Uuid, MigrationStateMachine>>>,
    /// Migration history
    history: Arc<RwLock<Vec<Migration>>>,
    /// Default options
    default_options: MigrationOptions,
    /// Maximum concurrent migrations
    max_concurrent: usize,
}

impl MigrationExecutor {
    /// Create a new migration executor
    pub fn new() -> Self {
        Self {
            active_migrations: Arc::new(RwLock::new(HashMap::new())),
            history: Arc::new(RwLock::new(Vec::new())),
            default_options: MigrationOptions::default(),
            max_concurrent: 10,
        }
    }

    /// Create with custom configuration
    pub fn with_config(default_options: MigrationOptions, max_concurrent: usize) -> Self {
        Self {
            active_migrations: Arc::new(RwLock::new(HashMap::new())),
            history: Arc::new(RwLock::new(Vec::new())),
            default_options,
            max_concurrent,
        }
    }

    /// Submit a migration for execution
    pub fn submit(&self, plan: MigrationPlan) -> Result<MigrationHandle> {
        let active = self.active_migrations.read();
        if active.len() >= self.max_concurrent {
            return Err(Error::MigrationFailed(format!(
                "Maximum concurrent migrations ({}) reached",
                self.max_concurrent
            )));
        }

        // Check if node is already being migrated
        if active
            .values()
            .any(|m| m.migration.node_id == plan.migration.node_id)
        {
            return Err(Error::MigrationInProgress(plan.migration.node_id));
        }
        drop(active);

        let handle = MigrationHandle::from(&plan.migration);
        let state_machine = MigrationStateMachine::new(plan, self.default_options.clone());

        self.active_migrations
            .write()
            .insert(handle.id, state_machine);

        Ok(handle)
    }

    /// Start a submitted migration
    pub fn start(&self, migration_id: Uuid) -> Result<()> {
        let mut active = self.active_migrations.write();
        let sm = active
            .get_mut(&migration_id)
            .ok_or_else(|| Error::MigrationNotFound(migration_id))?;

        sm.start()
    }

    /// Get migration progress
    pub fn get_progress(&self, migration_id: Uuid) -> Option<MigrationProgress> {
        self.active_migrations
            .read()
            .get(&migration_id)
            .map(|sm| sm.progress.clone())
    }

    /// Get migration state machine (for step execution)
    pub fn get_state(&self, migration_id: Uuid) -> Option<MigrationStateMachine> {
        self.active_migrations.read().get(&migration_id).cloned()
    }

    /// Update migration state
    pub fn update_state(&self, migration_id: Uuid, state: MigrationStateMachine) -> Result<()> {
        let mut active = self.active_migrations.write();
        if !active.contains_key(&migration_id) {
            return Err(Error::MigrationNotFound(migration_id));
        }

        // If migration is complete, move to history
        if state.is_complete() {
            active.remove(&migration_id);
            self.history.write().push(state.migration);
        } else {
            active.insert(migration_id, state);
        }

        Ok(())
    }

    /// Advance migration to next step
    pub fn advance_step(&self, migration_id: Uuid) -> Result<MigrationStep> {
        let mut active = self.active_migrations.write();
        let sm = active
            .get_mut(&migration_id)
            .ok_or_else(|| Error::MigrationNotFound(migration_id))?;

        sm.advance()?;
        let new_step = sm.step;

        // If complete, move to history
        if sm.is_complete() {
            let migration = sm.migration.clone();
            active.remove(&migration_id);
            self.history.write().push(migration);
        }

        Ok(new_step)
    }

    /// Record step failure
    pub fn fail_step(&self, migration_id: Uuid, error: String) -> Result<()> {
        let mut active = self.active_migrations.write();
        let sm = active
            .get_mut(&migration_id)
            .ok_or_else(|| Error::MigrationNotFound(migration_id))?;

        sm.fail_step(error)?;

        // If terminal state reached, move to history
        if sm.is_complete() {
            let migration = sm.migration.clone();
            active.remove(&migration_id);
            self.history.write().push(migration);
        }

        Ok(())
    }

    /// Cancel a migration
    pub fn cancel(&self, migration_id: Uuid) -> Result<()> {
        let mut active = self.active_migrations.write();
        let sm = active
            .get_mut(&migration_id)
            .ok_or_else(|| Error::MigrationNotFound(migration_id))?;

        if !sm.migration.status.can_cancel() {
            return Err(Error::MigrationFailed(format!(
                "Cannot cancel migration in state {:?}",
                sm.migration.status
            )));
        }

        if sm.step.is_reversible() {
            sm.start_rollback("Cancelled by user".to_string())?;
        } else {
            sm.migration.status = MigrationStatus::Cancelled;
            let migration = sm.migration.clone();
            active.remove(&migration_id);
            self.history.write().push(migration);
        }

        Ok(())
    }

    /// Rollback a migration
    pub fn rollback(&self, migration_id: Uuid, reason: String) -> Result<()> {
        let mut active = self.active_migrations.write();
        let sm = active
            .get_mut(&migration_id)
            .ok_or_else(|| Error::MigrationNotFound(migration_id))?;

        if !sm.migration.status.can_rollback() {
            return Err(Error::MigrationFailed(format!(
                "Cannot rollback migration in state {:?}",
                sm.migration.status
            )));
        }

        sm.start_rollback(reason)
    }

    /// Get all active migrations
    pub fn active_migrations(&self) -> Vec<MigrationHandle> {
        self.active_migrations
            .read()
            .values()
            .map(|sm| MigrationHandle::from(&sm.migration))
            .collect()
    }

    /// Get active migration count
    pub fn active_count(&self) -> usize {
        self.active_migrations.read().len()
    }

    /// Get migration history
    pub fn history(&self) -> Vec<Migration> {
        self.history.read().clone()
    }

    /// Clear history
    pub fn clear_history(&self) {
        self.history.write().clear();
    }
}

impl Default for MigrationExecutor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::migration::planner::MigrationPlanner;
    use crate::models::{Subnet, SubnetAssignment, SubnetPurpose, SubnetStatus};

    fn create_test_subnet(name: &str, cidr: &str) -> Subnet {
        Subnet {
            id: Uuid::new_v4(),
            name: name.to_string(),
            description: None,
            cidr: cidr.parse().unwrap(),
            purpose: SubnetPurpose::Tenant,
            status: SubnetStatus::Active,
            tenant_id: Some(Uuid::new_v4()),
            node_type: None,
            region: None,
            resource_pool_id: None,
            wg_interface: format!("wg-{}", name),
            wg_listen_port: 51820,
            wg_public_key: Some("pub-key".to_string()),
            wg_private_key: Some("priv-key".to_string()),
            max_nodes: Some(100),
            current_nodes: 10,
            template_id: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            created_by: None,
            metadata: None,
        }
    }

    fn create_test_assignment(subnet_id: Uuid) -> SubnetAssignment {
        SubnetAssignment {
            id: Uuid::new_v4(),
            node_id: Uuid::new_v4(),
            subnet_id,
            assigned_ip: Ipv4Addr::new(10, 100, 0, 10),
            wg_public_key: "node-pub-key".to_string(),
            assigned_at: Utc::now(),
            assignment_method: "manual".to_string(),
            policy_id: None,
            is_migration_temp: false,
        }
    }

    fn create_test_plan() -> MigrationPlan {
        let mut planner = MigrationPlanner::new();
        let source = create_test_subnet("source", "10.100.0.0/24");
        let target = create_test_subnet("target", "10.101.0.0/24");
        let assignment = create_test_assignment(source.id);
        let target_ip = Ipv4Addr::new(10, 101, 0, 20);

        planner
            .plan(
                assignment.node_id,
                &source,
                &target,
                &assignment,
                target_ip,
                MigrationReason::Manual,
            )
            .unwrap()
    }

    #[test]
    fn test_step_progression() {
        assert_eq!(
            MigrationStep::NotStarted.next(),
            Some(MigrationStep::AllocatingIp)
        );
        assert_eq!(
            MigrationStep::AllocatingIp.next(),
            Some(MigrationStep::GeneratingConfig)
        );
        assert_eq!(
            MigrationStep::CleaningUp.next(),
            Some(MigrationStep::Completed)
        );
        assert_eq!(MigrationStep::Completed.next(), None);
    }

    #[test]
    fn test_step_reversibility() {
        assert!(MigrationStep::AllocatingIp.is_reversible());
        assert!(MigrationStep::EnablingDualStack.is_reversible());
        assert!(!MigrationStep::CuttingOver.is_reversible());
        assert!(!MigrationStep::Completed.is_reversible());
    }

    #[test]
    fn test_state_machine_creation() {
        let plan = create_test_plan();
        let sm = MigrationStateMachine::new(plan, MigrationOptions::default());

        assert_eq!(sm.step, MigrationStep::NotStarted);
        assert!(!sm.is_complete());
    }

    #[test]
    fn test_state_machine_start() {
        let plan = create_test_plan();
        let mut sm = MigrationStateMachine::new(plan, MigrationOptions::default());

        sm.start().unwrap();

        assert_eq!(sm.step, MigrationStep::AllocatingIp);
        assert_eq!(sm.migration.status, MigrationStatus::InProgress);
        assert!(sm.migration.started_at.is_some());
    }

    #[test]
    fn test_state_machine_advance() {
        let plan = create_test_plan();
        let mut sm = MigrationStateMachine::new(plan, MigrationOptions::default());

        sm.start().unwrap();
        sm.advance().unwrap();

        assert_eq!(sm.step, MigrationStep::GeneratingConfig);
    }

    #[test]
    fn test_state_machine_complete() {
        let plan = create_test_plan();
        let mut sm = MigrationStateMachine::new(plan, MigrationOptions::default());

        sm.start().unwrap();

        // Advance through all steps
        while sm.step.next().is_some() {
            sm.advance().unwrap();
        }

        assert!(sm.is_complete());
        assert!(sm.is_success());
        assert_eq!(sm.migration.status, MigrationStatus::Completed);
    }

    #[test]
    fn test_state_machine_rollback() {
        let plan = create_test_plan();
        let mut sm = MigrationStateMachine::new(plan, MigrationOptions::default());

        sm.start().unwrap();
        sm.advance().unwrap(); // GeneratingConfig
        sm.advance().unwrap(); // EnablingDualStack

        sm.start_rollback("Test rollback".to_string()).unwrap();
        assert_eq!(sm.step, MigrationStep::RollingBack);
        assert_eq!(sm.migration.status, MigrationStatus::RollingBack);

        sm.complete_rollback().unwrap();
        assert_eq!(sm.step, MigrationStep::RolledBack);
        assert!(sm.is_complete());
    }

    #[test]
    fn test_state_machine_auto_rollback() {
        let plan = create_test_plan();
        let mut options = MigrationOptions::default();
        options.auto_rollback = true;
        let mut sm = MigrationStateMachine::new(plan, options);
        sm.max_step_retries = 2;

        sm.start().unwrap();
        sm.advance().unwrap(); // GeneratingConfig

        // Fail step multiple times to trigger rollback
        sm.fail_step("Error 1".to_string()).unwrap();
        sm.fail_step("Error 2".to_string()).unwrap(); // Should trigger rollback

        assert_eq!(sm.step, MigrationStep::RollingBack);
    }

    #[test]
    fn test_progress_tracking() {
        let plan = create_test_plan();
        let mut sm = MigrationStateMachine::new(plan, MigrationOptions::default());

        sm.start().unwrap();
        assert_eq!(sm.progress.progress_percent, 12); // ~1/8

        sm.advance().unwrap();
        sm.advance().unwrap();
        assert!(sm.progress.progress_percent > 25);

        sm.record_peer_notified(Uuid::new_v4());
        assert_eq!(sm.progress.peers_notified, 1);

        sm.record_peer_ack();
        assert_eq!(sm.progress.peers_acknowledged, 1);

        sm.record_connectivity_verified();
        assert!(sm.progress.connectivity_verified);
    }

    #[test]
    fn test_rollback_data() {
        let plan = create_test_plan();
        let mut sm = MigrationStateMachine::new(plan, MigrationOptions::default());

        sm.start().unwrap();
        sm.record_allocated_ip(Ipv4Addr::new(10, 101, 0, 20));
        sm.record_generated_config("test config".to_string());
        sm.record_dual_stack_enabled();
        sm.record_peer_notified(Uuid::new_v4());

        let data = sm.rollback_data();
        assert_eq!(data.allocated_ip, Some(Ipv4Addr::new(10, 101, 0, 20)));
        assert_eq!(data.generated_config, Some("test config".to_string()));
        assert!(data.dual_stack_enabled);
        assert_eq!(data.notified_peers.len(), 1);
    }

    #[test]
    fn test_executor_submit() {
        let executor = MigrationExecutor::new();
        let plan = create_test_plan();

        let handle = executor.submit(plan).unwrap();

        assert_eq!(executor.active_count(), 1);
        assert!(executor.get_progress(handle.id).is_some());
    }

    #[test]
    fn test_executor_start() {
        let executor = MigrationExecutor::new();
        let plan = create_test_plan();
        let handle = executor.submit(plan).unwrap();

        executor.start(handle.id).unwrap();

        let progress = executor.get_progress(handle.id).unwrap();
        assert_eq!(progress.current_step, MigrationStep::AllocatingIp);
    }

    #[test]
    fn test_executor_advance() {
        let executor = MigrationExecutor::new();
        let plan = create_test_plan();
        let handle = executor.submit(plan).unwrap();

        executor.start(handle.id).unwrap();
        let step = executor.advance_step(handle.id).unwrap();

        assert_eq!(step, MigrationStep::GeneratingConfig);
    }

    #[test]
    fn test_executor_cancel() {
        let executor = MigrationExecutor::new();
        let plan = create_test_plan();
        let handle = executor.submit(plan).unwrap();

        executor.start(handle.id).unwrap();
        executor.cancel(handle.id).unwrap();

        // Migration should be rolling back
        let state = executor.get_state(handle.id);
        assert!(state.is_some());
        assert_eq!(state.unwrap().step, MigrationStep::RollingBack);
    }

    #[test]
    fn test_executor_max_concurrent() {
        let executor = MigrationExecutor::with_config(MigrationOptions::default(), 2);

        let plan1 = create_test_plan();
        let plan2 = create_test_plan();
        let plan3 = create_test_plan();

        executor.submit(plan1).unwrap();
        executor.submit(plan2).unwrap();

        // Third should fail
        let result = executor.submit(plan3);
        assert!(result.is_err());
    }

    #[test]
    fn test_executor_duplicate_node() {
        let executor = MigrationExecutor::new();
        let plan1 = create_test_plan();
        let mut plan2 = create_test_plan();
        plan2.migration.node_id = plan1.migration.node_id;

        executor.submit(plan1).unwrap();

        // Same node should fail
        let result = executor.submit(plan2);
        assert!(result.is_err());
    }

    #[test]
    fn test_executor_history() {
        let executor = MigrationExecutor::new();
        let plan = create_test_plan();
        let handle = executor.submit(plan).unwrap();

        executor.start(handle.id).unwrap();

        // Complete migration
        while executor.get_state(handle.id).is_some() {
            let _ = executor.advance_step(handle.id);
        }

        assert_eq!(executor.active_count(), 0);
        assert_eq!(executor.history().len(), 1);
        assert_eq!(executor.history()[0].status, MigrationStatus::Completed);
    }
}
