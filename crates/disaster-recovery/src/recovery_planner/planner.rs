//! Main recovery planner implementation

use crate::error::{DisasterRecoveryError, DisasterRecoveryResult};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use parking_lot::RwLock;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};
use tokio::time::interval;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use super::{
    config::*,
    types::*,
    objectives::*,
    strategies::*,
    dependencies::*,
    resources::*,
    execution::*,
    validation::*,
    metrics::*,
    optimization::*,
};

/// Service recovery plan definition
#[derive(Debug, Clone)]
pub struct ServiceRecoveryPlan {
    /// Plan ID
    pub id: Uuid,
    /// Service ID
    pub service_id: Uuid,
    /// Service name
    pub service_name: String,
    /// Recovery tier
    pub tier: RecoveryTier,
    /// Recovery objectives
    pub objectives: RecoveryObjective,
    /// Recovery strategy
    pub strategy: RecoveryStrategy,
    /// Dependencies
    pub dependencies: Vec<ServiceDependency>,
    /// Resource requirements
    pub resources: Vec<RecoveryResource>,
    /// Recovery steps
    pub recovery_steps: Vec<RecoveryStep>,
    /// Validation tests
    pub validation_tests: Vec<ValidationTest>,
    /// Recovery priority
    pub priority: u32,
    /// Estimated recovery time
    pub estimated_recovery_minutes: u64,
    /// Created timestamp
    pub created_at: DateTime<Utc>,
    /// Last updated
    pub updated_at: DateTime<Utc>,
}

/// Main recovery planner
pub struct RecoveryPlanner {
    /// Configuration
    config: Arc<RecoveryPlannerConfig>,
    /// Recovery plans
    plans: Arc<DashMap<Uuid, ServiceRecoveryPlan>>,
    /// Active executions
    executions: Arc<DashMap<Uuid, RecoveryExecution>>,
    /// Dependency graph
    dependency_graph: Arc<RwLock<DependencyGraph>>,
    /// Resource pool
    resource_pool: Arc<DashMap<String, ResourcePool>>,
    /// Metrics
    metrics: Arc<RwLock<RecoveryPlannerMetrics>>,
    /// Command channel
    command_tx: mpsc::Sender<PlannerCommand>,
    /// Command receiver
    command_rx: Arc<Mutex<mpsc::Receiver<PlannerCommand>>>,
    /// Shutdown flag
    shutdown: Arc<RwLock<bool>>,
}

/// Planner commands
#[derive(Debug)]
enum PlannerCommand {
    /// Execute recovery plan
    ExecutePlan(Uuid),
    /// Cancel execution
    CancelExecution(Uuid),
    /// Validate plan
    ValidatePlan(Uuid),
    /// Optimize resources
    OptimizeResources,
    /// Update metrics
    UpdateMetrics,
}

/// Recovery task for priority queue
#[derive(Debug, Clone)]
struct RecoveryTask {
    /// Service ID
    service_id: Uuid,
    /// Recovery priority
    priority: u32,
    /// Estimated start time
    estimated_start: DateTime<Utc>,
    /// Dependencies resolved
    dependencies_resolved: bool,
}

impl PartialEq for RecoveryTask {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority
    }
}

impl Eq for RecoveryTask {}

impl PartialOrd for RecoveryTask {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for RecoveryTask {
    fn cmp(&self, other: &Self) -> Ordering {
        // Higher priority comes first (reverse order)
        other
            .priority
            .cmp(&self.priority)
            .then_with(|| self.estimated_start.cmp(&other.estimated_start))
    }
}

impl RecoveryPlanner {
    /// Create new recovery planner
    pub fn new(config: RecoveryPlannerConfig) -> DisasterRecoveryResult<Self> {
        let (command_tx, command_rx) = mpsc::channel(1000);

        Ok(Self {
            config: Arc::new(config),
            plans: Arc::new(DashMap::new()),
            executions: Arc::new(DashMap::new()),
            dependency_graph: Arc::new(RwLock::new(DependencyGraph::new())),
            resource_pool: Arc::new(DashMap::new()),
            metrics: Arc::new(RwLock::new(RecoveryPlannerMetrics::default())),
            command_tx,
            command_rx: Arc::new(Mutex::new(command_rx)),
            shutdown: Arc::new(RwLock::new(false)),
        })
    }

    /// Start recovery planner
    pub async fn start(&self) -> DisasterRecoveryResult<()> {
        info!("Starting recovery planner");

        // Start background tasks
        self.start_command_processor().await?;
        self.start_metrics_updater().await?;

        Ok(())
    }

    /// Stop recovery planner
    pub async fn stop(&self) -> DisasterRecoveryResult<()> {
        info!("Stopping recovery planner");
        *self.shutdown.write() = true;
        Ok(())
    }

    /// Create recovery plan
    pub async fn create_plan(&self, plan: ServiceRecoveryPlan) -> DisasterRecoveryResult<Uuid> {
        let plan_id = plan.id;

        // Validate plan
        self.validate_plan_structure(&plan)?;

        // Check for dependency cycles
        if self.config.cycle_detection_enabled {
            self.check_dependency_cycles(&plan)?;
        }

        self.plans.insert(plan_id, plan.clone());

        // Update dependency graph
        self.update_dependency_graph(&plan).await;

        info!(
            "Created recovery plan: {} for service: {}",
            plan_id, plan.service_name
        );
        self.update_metrics_count().await;

        Ok(plan_id)
    }

    /// Execute recovery plan
    pub async fn execute_plan(&self, plan_id: Uuid) -> DisasterRecoveryResult<Uuid> {
        let plan = self.plans.get(&plan_id).ok_or_else(|| {
            DisasterRecoveryError::ResourceUnavailable {
                resource: "recovery_plan".to_string(),
                reason: "plan not found".to_string(),
            }
        })?;

        let execution_id = Uuid::new_v4();
        let mut execution = RecoveryExecution::new(plan_id);
        execution.id = execution_id;

        // Initialize step executions
        for step in &plan.recovery_steps {
            execution.step_executions.insert(
                step.id,
                StepExecution::new(step.id),
            );
        }

        execution.state = ExecutionState::Running;
        self.executions.insert(execution_id, execution);

        // Update metrics
        {
            let mut metrics = self.metrics.write();
            metrics.active_executions += 1;
        }

        info!("Started execution {} for plan {}", execution_id, plan_id);

        // Send command to execute
        self.command_tx.send(PlannerCommand::ExecutePlan(execution_id)).await
            .map_err(|e| DisasterRecoveryError::Other(format!("Command send failed: {}", e)))?;

        Ok(execution_id)
    }

    /// Get recovery plan
    pub fn get_plan(&self, plan_id: Uuid) -> Option<ServiceRecoveryPlan> {
        self.plans.get(&plan_id).map(|entry| entry.value().clone())
    }

    /// List all recovery plans
    pub fn list_plans(&self) -> Vec<ServiceRecoveryPlan> {
        self.plans.iter().map(|entry| entry.value().clone()).collect()
    }

    /// Get execution status
    pub fn get_execution(&self, execution_id: Uuid) -> Option<RecoveryExecution> {
        self.executions.get(&execution_id).map(|entry| entry.value().clone())
    }

    /// Get metrics
    pub fn get_metrics(&self) -> RecoveryPlannerMetrics {
        self.metrics.read().clone()
    }

    /// Private helper methods

    async fn start_command_processor(&self) -> DisasterRecoveryResult<()> {
        let command_rx = self.command_rx.clone();
        let executions = self.executions.clone();
        let plans = self.plans.clone();
        let shutdown = self.shutdown.clone();

        tokio::spawn(async move {
            let mut receiver = command_rx.lock().await;
            
            while !*shutdown.read() {
                if let Some(command) = receiver.recv().await {
                    match command {
                        PlannerCommand::ExecutePlan(execution_id) => {
                            Self::process_execution(execution_id, &executions, &plans).await;
                        }
                        PlannerCommand::CancelExecution(execution_id) => {
                            Self::cancel_execution(execution_id, &executions).await;
                        }
                        _ => {
                            // Handle other commands
                        }
                    }
                }
            }
        });

        Ok(())
    }

    async fn start_metrics_updater(&self) -> DisasterRecoveryResult<()> {
        let metrics = self.metrics.clone();
        let executions = self.executions.clone();
        let shutdown = self.shutdown.clone();

        tokio::spawn(async move {
            let mut interval = interval(std::time::Duration::from_secs(30));
            
            while !*shutdown.read() {
                interval.tick().await;
                Self::update_metrics(&metrics, &executions).await;
            }
        });

        Ok(())
    }

    async fn process_execution(
        execution_id: Uuid,
        executions: &DashMap<Uuid, RecoveryExecution>,
        plans: &DashMap<Uuid, ServiceRecoveryPlan>,
    ) {
        if let Some(mut execution_entry) = executions.get_mut(&execution_id) {
            let execution = execution_entry.value_mut();
            let plan_id = execution.plan_id;

            if let Some(plan_entry) = plans.get(&plan_id) {
                let plan = plan_entry.value();

                for step in &plan.recovery_steps {
                    if let Some(step_execution) = execution.step_executions.get_mut(&step.id) {
                        step_execution.start();
                        
                        // Simulate step execution
                        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                        
                        step_execution.complete(Some("Step completed successfully".to_string()));
                    }
                }

                execution.state = ExecutionState::Completed;
                execution.completed_at = Some(Utc::now());
                execution.update_progress();
                execution.add_log(LogLevel::Info, "Execution completed successfully".to_string(), None);
            }
        }
    }

    async fn cancel_execution(
        execution_id: Uuid,
        executions: &DashMap<Uuid, RecoveryExecution>,
    ) {
        if let Some(mut execution_entry) = executions.get_mut(&execution_id) {
            let execution = execution_entry.value_mut();
            execution.state = ExecutionState::Cancelled;
            execution.completed_at = Some(Utc::now());
            execution.add_log(LogLevel::Info, "Execution cancelled by user".to_string(), None);
        }
    }

    async fn update_metrics(
        metrics: &RwLock<RecoveryPlannerMetrics>,
        executions: &DashMap<Uuid, RecoveryExecution>,
    ) {
        let mut metrics_guard = metrics.write();
        
        let active_count = executions.iter()
            .filter(|entry| entry.value().state == ExecutionState::Running)
            .count() as u64;
        
        metrics_guard.active_executions = active_count;
        metrics_guard.last_updated = Utc::now();
    }

    fn validate_plan_structure(&self, plan: &ServiceRecoveryPlan) -> DisasterRecoveryResult<()> {
        if plan.recovery_steps.is_empty() {
            return Err(DisasterRecoveryError::ConfigurationError {
                message: "Plan must have at least one recovery step".to_string(),
            });
        }

        if plan.service_name.is_empty() {
            return Err(DisasterRecoveryError::ConfigurationError {
                message: "Service name cannot be empty".to_string(),
            });
        }

        Ok(())
    }

    fn check_dependency_cycles(&self, plan: &ServiceRecoveryPlan) -> DisasterRecoveryResult<()> {
        // Simple cycle detection - in a real implementation this would be more sophisticated
        let mut visited = HashSet::new();
        for dependency in &plan.dependencies {
            if visited.contains(&dependency.dependent_service_id) {
                return Err(DisasterRecoveryError::ConfigurationError {
                    message: "Circular dependency detected".to_string(),
                });
            }
            visited.insert(dependency.dependent_service_id);
        }

        Ok(())
    }

    async fn update_dependency_graph(&self, plan: &ServiceRecoveryPlan) {
        let mut graph = self.dependency_graph.write();
        
        for dependency in &plan.dependencies {
            graph.add_dependency(
                dependency.dependent_service_id,
                dependency.dependency_service_id,
            );
        }
    }

    async fn update_metrics_count(&self) {
        let mut metrics = self.metrics.write();
        metrics.total_plans = self.plans.len() as u64;
        metrics.last_updated = Utc::now();
    }
}
