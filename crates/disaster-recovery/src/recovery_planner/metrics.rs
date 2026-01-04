//! Recovery planning metrics and monitoring

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Recovery planner metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryPlannerMetrics {
    /// Total plans created
    pub total_plans: u64,
    /// Active executions
    pub active_executions: u64,
    /// Successful executions
    pub successful_executions: u64,
    /// Failed executions
    pub failed_executions: u64,
    /// Average execution time in minutes
    pub average_execution_time_minutes: f64,
    /// Resource utilization by type
    pub resource_utilization: HashMap<String, f64>,
    /// Last updated timestamp
    pub last_updated: DateTime<Utc>,
}

impl Default for RecoveryPlannerMetrics {
    fn default() -> Self {
        Self {
            total_plans: 0,
            active_executions: 0,
            successful_executions: 0,
            failed_executions: 0,
            average_execution_time_minutes: 0.0,
            resource_utilization: HashMap::new(),
            last_updated: Utc::now(),
        }
    }
}

impl RecoveryPlannerMetrics {
    /// Calculate success rate
    pub fn success_rate(&self) -> f64 {
        let total_completed = self.successful_executions + self.failed_executions;
        if total_completed == 0 {
            0.0
        } else {
            self.successful_executions as f64 / total_completed as f64
        }
    }

    /// Update execution metrics
    pub fn update_execution_metrics(&mut self, success: bool, duration_minutes: f64) {
        if success {
            self.successful_executions += 1;
        } else {
            self.failed_executions += 1;
        }

        // Update average execution time
        let total_executions = self.successful_executions + self.failed_executions;
        self.average_execution_time_minutes = (self.average_execution_time_minutes
            * (total_executions - 1) as f64
            + duration_minutes)
            / total_executions as f64;

        self.last_updated = Utc::now();
    }

    /// Update resource utilization
    pub fn update_resource_utilization(&mut self, resource_type: String, utilization: f64) {
        self.resource_utilization.insert(resource_type, utilization);
        self.last_updated = Utc::now();
    }
}

/// Recovery timeline for planning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryTimeline {
    /// Timeline ID
    pub id: Uuid,
    /// Plan ID this timeline belongs to
    pub plan_id: Uuid,
    /// Recovery phases
    pub phases: Vec<RecoveryPhase>,
    /// Total estimated duration
    pub total_duration_minutes: u64,
    /// Critical path steps
    pub critical_path: Vec<Uuid>,
    /// Parallel execution opportunities
    pub parallel_groups: Vec<Vec<Uuid>>,
}

/// Recovery phase definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryPhase {
    /// Phase ID
    pub id: Uuid,
    /// Phase name
    pub name: String,
    /// Phase description
    pub description: String,
    /// Steps in this phase
    pub steps: Vec<Uuid>,
    /// Estimated duration
    pub estimated_duration_minutes: u64,
    /// Prerequisites from other phases
    pub phase_prerequisites: Vec<Uuid>,
}

impl RecoveryTimeline {
    /// Create new recovery timeline
    pub fn new(plan_id: Uuid) -> Self {
        Self {
            id: Uuid::new_v4(),
            plan_id,
            phases: Vec::new(),
            total_duration_minutes: 0,
            critical_path: Vec::new(),
            parallel_groups: Vec::new(),
        }
    }

    /// Add recovery phase
    pub fn add_phase(&mut self, phase: RecoveryPhase) {
        self.total_duration_minutes += phase.estimated_duration_minutes;
        self.phases.push(phase);
    }

    /// Calculate critical path
    pub fn calculate_critical_path(&mut self) {
        // Simplified critical path calculation
        // In a real implementation, this would use proper CPM algorithm
        for phase in &self.phases {
            self.critical_path.extend(phase.steps.iter().cloned());
        }
    }

    /// Get total timeline duration
    pub fn total_duration(&self) -> u64 {
        self.total_duration_minutes
    }

    /// Get phases that can run in parallel
    pub fn parallel_phases(&self) -> Vec<Vec<&RecoveryPhase>> {
        let mut parallel_groups = Vec::new();
        let mut current_group = Vec::new();

        for phase in &self.phases {
            if phase.phase_prerequisites.is_empty() || current_group.is_empty() {
                current_group.push(phase);
            } else {
                if !current_group.is_empty() {
                    parallel_groups.push(current_group);
                }
                current_group = vec![phase];
            }
        }

        if !current_group.is_empty() {
            parallel_groups.push(current_group);
        }

        parallel_groups
    }
}
