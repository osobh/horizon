//! Migration planning and validation
//!
//! Validates migration feasibility and creates migration plans.

use super::Migration;
use crate::models::{Subnet, SubnetAssignment, SubnetStatus};
use crate::{Error, Result};
use chrono::{DateTime, Timelike, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::net::Ipv4Addr;
use uuid::Uuid;

/// Reason for migration
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MigrationReason {
    /// Manual migration requested by user
    Manual,
    /// Policy change triggered reassignment
    PolicyChange,
    /// Subnet is being decommissioned
    SubnetDecommission,
    /// Load balancing across subnets
    LoadBalancing,
    /// Node type changed
    NodeTypeChange,
    /// Tenant change
    TenantChange,
    /// Geographic relocation
    GeographicMove,
    /// Resource pool change
    ResourcePoolChange,
    /// Maintenance window
    Maintenance,
    /// Recovery from failure
    Recovery,
}

impl std::fmt::Display for MigrationReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MigrationReason::Manual => write!(f, "Manual"),
            MigrationReason::PolicyChange => write!(f, "Policy Change"),
            MigrationReason::SubnetDecommission => write!(f, "Subnet Decommission"),
            MigrationReason::LoadBalancing => write!(f, "Load Balancing"),
            MigrationReason::NodeTypeChange => write!(f, "Node Type Change"),
            MigrationReason::TenantChange => write!(f, "Tenant Change"),
            MigrationReason::GeographicMove => write!(f, "Geographic Move"),
            MigrationReason::ResourcePoolChange => write!(f, "Resource Pool Change"),
            MigrationReason::Maintenance => write!(f, "Maintenance"),
            MigrationReason::Recovery => write!(f, "Recovery"),
        }
    }
}

/// Constraints for migration planning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationConstraint {
    /// Maximum concurrent migrations
    pub max_concurrent: usize,
    /// Maximum migrations per subnet
    pub max_per_subnet: usize,
    /// Minimum time between migrations for same node (seconds)
    pub min_interval_secs: u64,
    /// Allowed time window (start hour, end hour in UTC)
    pub maintenance_window: Option<(u8, u8)>,
    /// Nodes to exclude from migration
    pub excluded_nodes: HashSet<Uuid>,
    /// Subnets to exclude as targets
    pub excluded_target_subnets: HashSet<Uuid>,
    /// Whether to allow cross-purpose migrations
    pub allow_cross_purpose: bool,
}

impl Default for MigrationConstraint {
    fn default() -> Self {
        Self {
            max_concurrent: 10,
            max_per_subnet: 5,
            min_interval_secs: 300, // 5 minutes
            maintenance_window: None,
            excluded_nodes: HashSet::new(),
            excluded_target_subnets: HashSet::new(),
            allow_cross_purpose: false,
        }
    }
}

impl MigrationConstraint {
    /// Check if current time is within maintenance window
    pub fn is_in_maintenance_window(&self) -> bool {
        match self.maintenance_window {
            Some((start, end)) => {
                let now = Utc::now();
                let hour = now.time().hour() as u8;
                if start <= end {
                    hour >= start && hour < end
                } else {
                    // Window spans midnight
                    hour >= start || hour < end
                }
            }
            None => true, // No window means always allowed
        }
    }
}

/// Validation result for a migration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationValidation {
    /// Whether migration is valid
    pub valid: bool,
    /// Issues found during validation
    pub issues: Vec<ValidationIssue>,
    /// Warnings (migration can proceed but with caveats)
    pub warnings: Vec<String>,
    /// Estimated duration in seconds
    pub estimated_duration_secs: Option<u64>,
}

impl MigrationValidation {
    /// Create a valid result
    pub fn valid() -> Self {
        Self {
            valid: true,
            issues: Vec::new(),
            warnings: Vec::new(),
            estimated_duration_secs: None,
        }
    }

    /// Create an invalid result with issues
    pub fn invalid(issues: Vec<ValidationIssue>) -> Self {
        Self {
            valid: false,
            issues,
            warnings: Vec::new(),
            estimated_duration_secs: None,
        }
    }

    /// Add a warning
    pub fn with_warning(mut self, warning: impl Into<String>) -> Self {
        self.warnings.push(warning.into());
        self
    }

    /// Set estimated duration
    pub fn with_duration(mut self, secs: u64) -> Self {
        self.estimated_duration_secs = Some(secs);
        self
    }
}

/// Issue preventing migration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationIssue {
    /// Issue code
    pub code: String,
    /// Human-readable message
    pub message: String,
    /// Severity
    pub severity: IssueSeverity,
}

impl ValidationIssue {
    pub fn new(
        code: impl Into<String>,
        message: impl Into<String>,
        severity: IssueSeverity,
    ) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
            severity,
        }
    }

    pub fn error(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self::new(code, message, IssueSeverity::Error)
    }

    pub fn warning(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self::new(code, message, IssueSeverity::Warning)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IssueSeverity {
    Error,
    Warning,
    Info,
}

/// A migration plan for a single node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationPlan {
    /// Migration record
    pub migration: Migration,
    /// Source subnet info
    pub source_subnet: SubnetInfo,
    /// Target subnet info
    pub target_subnet: SubnetInfo,
    /// Allocated target IP
    pub target_ip: Ipv4Addr,
    /// Estimated steps and durations
    pub steps: Vec<PlannedStep>,
    /// Total estimated duration in seconds
    pub estimated_duration_secs: u64,
    /// Dependencies (other migrations that must complete first)
    pub dependencies: Vec<Uuid>,
    /// Plan created at
    pub created_at: DateTime<Utc>,
}

/// Minimal subnet info for migration planning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubnetInfo {
    pub id: Uuid,
    pub name: String,
    pub cidr: String,
    pub current_nodes: i32,
    pub max_nodes: Option<i32>,
}

impl From<&Subnet> for SubnetInfo {
    fn from(s: &Subnet) -> Self {
        Self {
            id: s.id,
            name: s.name.clone(),
            cidr: s.cidr.to_string(),
            current_nodes: s.current_nodes,
            max_nodes: s.max_nodes,
        }
    }
}

/// A planned migration step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlannedStep {
    /// Step name
    pub name: String,
    /// Estimated duration in seconds
    pub estimated_secs: u64,
    /// Whether step is reversible
    pub reversible: bool,
}

/// Bulk migration plan for multiple nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BulkMigrationPlan {
    /// Individual migration plans
    pub migrations: Vec<MigrationPlan>,
    /// Execution order (groups of parallel migrations)
    pub execution_order: Vec<Vec<Uuid>>,
    /// Total estimated duration in seconds
    pub total_estimated_secs: u64,
    /// Plan created at
    pub created_at: DateTime<Utc>,
}

impl BulkMigrationPlan {
    /// Get total number of migrations
    pub fn total_migrations(&self) -> usize {
        self.migrations.len()
    }

    /// Get number of execution waves
    pub fn wave_count(&self) -> usize {
        self.execution_order.len()
    }
}

/// Migration planner
pub struct MigrationPlanner {
    /// Planning constraints
    constraints: MigrationConstraint,
    /// Recent migrations for interval checking
    recent_migrations: HashMap<Uuid, DateTime<Utc>>,
    /// Currently active migrations
    active_migrations: HashMap<Uuid, Migration>,
}

impl MigrationPlanner {
    /// Create a new migration planner
    pub fn new() -> Self {
        Self {
            constraints: MigrationConstraint::default(),
            recent_migrations: HashMap::new(),
            active_migrations: HashMap::new(),
        }
    }

    /// Create with custom constraints
    pub fn with_constraints(constraints: MigrationConstraint) -> Self {
        Self {
            constraints,
            recent_migrations: HashMap::new(),
            active_migrations: HashMap::new(),
        }
    }

    /// Get current constraints
    pub fn constraints(&self) -> &MigrationConstraint {
        &self.constraints
    }

    /// Update constraints
    pub fn set_constraints(&mut self, constraints: MigrationConstraint) {
        self.constraints = constraints;
    }

    /// Record a migration start
    pub fn record_migration_start(&mut self, migration: &Migration) {
        self.active_migrations
            .insert(migration.id, migration.clone());
    }

    /// Record a migration completion
    pub fn record_migration_complete(&mut self, migration_id: Uuid, node_id: Uuid) {
        self.active_migrations.remove(&migration_id);
        self.recent_migrations.insert(node_id, Utc::now());
    }

    /// Validate a single migration
    pub fn validate(
        &self,
        node_id: Uuid,
        source_subnet: &Subnet,
        target_subnet: &Subnet,
        _current_assignment: &SubnetAssignment,
    ) -> MigrationValidation {
        let mut issues = Vec::new();
        let mut warnings = Vec::new();

        // Check if node is excluded
        if self.constraints.excluded_nodes.contains(&node_id) {
            issues.push(ValidationIssue::error(
                "NODE_EXCLUDED",
                "Node is excluded from migration",
            ));
        }

        // Check if target subnet is excluded
        if self
            .constraints
            .excluded_target_subnets
            .contains(&target_subnet.id)
        {
            issues.push(ValidationIssue::error(
                "TARGET_EXCLUDED",
                "Target subnet is excluded from migration",
            ));
        }

        // Check source subnet status
        if source_subnet.status == SubnetStatus::Archived {
            issues.push(ValidationIssue::error(
                "SOURCE_ARCHIVED",
                "Source subnet is archived",
            ));
        }

        // Check target subnet status
        if target_subnet.status != SubnetStatus::Active {
            issues.push(ValidationIssue::error(
                "TARGET_NOT_ACTIVE",
                format!(
                    "Target subnet is {:?}, must be Active",
                    target_subnet.status
                ),
            ));
        }

        // Check target subnet capacity
        if !target_subnet.can_accept_nodes() {
            issues.push(ValidationIssue::error(
                "TARGET_FULL",
                "Target subnet cannot accept more nodes",
            ));
        }

        // Check cross-purpose migration
        if !self.constraints.allow_cross_purpose && source_subnet.purpose != target_subnet.purpose {
            issues.push(ValidationIssue::error(
                "CROSS_PURPOSE",
                format!(
                    "Cross-purpose migration not allowed: {:?} -> {:?}",
                    source_subnet.purpose, target_subnet.purpose
                ),
            ));
        }

        // Check maintenance window
        if !self.constraints.is_in_maintenance_window() {
            warnings.push("Migration outside maintenance window".to_string());
        }

        // Check recent migration interval
        if let Some(last_migration) = self.recent_migrations.get(&node_id) {
            let elapsed = (Utc::now() - *last_migration).num_seconds() as u64;
            if elapsed < self.constraints.min_interval_secs {
                issues.push(ValidationIssue::error(
                    "TOO_SOON",
                    format!(
                        "Node was migrated {}s ago, minimum interval is {}s",
                        elapsed, self.constraints.min_interval_secs
                    ),
                ));
            }
        }

        // Check concurrent migration limits
        let active_count = self.active_migrations.len();
        if active_count >= self.constraints.max_concurrent {
            issues.push(ValidationIssue::error(
                "MAX_CONCURRENT",
                format!(
                    "Maximum concurrent migrations ({}) reached",
                    self.constraints.max_concurrent
                ),
            ));
        }

        // Check per-subnet limits
        let target_active = self
            .active_migrations
            .values()
            .filter(|m| m.target_subnet_id == target_subnet.id)
            .count();
        if target_active >= self.constraints.max_per_subnet {
            issues.push(ValidationIssue::error(
                "MAX_PER_SUBNET",
                format!(
                    "Maximum migrations per subnet ({}) reached for target",
                    self.constraints.max_per_subnet
                ),
            ));
        }

        // Check if node is already being migrated
        let already_migrating = self
            .active_migrations
            .values()
            .any(|m| m.node_id == node_id);
        if already_migrating {
            issues.push(ValidationIssue::error(
                "ALREADY_MIGRATING",
                "Node is already being migrated",
            ));
        }

        // Check same subnet migration
        if source_subnet.id == target_subnet.id {
            issues.push(ValidationIssue::error(
                "SAME_SUBNET",
                "Cannot migrate to the same subnet",
            ));
        }

        if issues.is_empty() {
            let mut result = MigrationValidation::valid();
            for warning in warnings {
                result = result.with_warning(warning);
            }
            // Estimate duration based on typical migration time
            result.with_duration(60) // 60 seconds estimated
        } else {
            MigrationValidation::invalid(issues)
        }
    }

    /// Plan a single migration
    pub fn plan(
        &mut self,
        node_id: Uuid,
        source_subnet: &Subnet,
        target_subnet: &Subnet,
        current_assignment: &SubnetAssignment,
        target_ip: Ipv4Addr,
        reason: MigrationReason,
    ) -> Result<MigrationPlan> {
        // Validate first
        let validation = self.validate(node_id, source_subnet, target_subnet, current_assignment);

        if !validation.valid {
            let errors: Vec<String> = validation
                .issues
                .iter()
                .filter(|i| i.severity == IssueSeverity::Error)
                .map(|i| i.message.clone())
                .collect();
            return Err(Error::MigrationFailed(errors.join("; ")));
        }

        let migration = Migration::new(
            node_id,
            source_subnet.id,
            target_subnet.id,
            current_assignment.assigned_ip,
            reason,
        )
        .with_target_ip(target_ip);

        let steps = vec![
            PlannedStep {
                name: "Allocate target IP".to_string(),
                estimated_secs: 1,
                reversible: true,
            },
            PlannedStep {
                name: "Generate WireGuard config".to_string(),
                estimated_secs: 2,
                reversible: true,
            },
            PlannedStep {
                name: "Enable dual-stack".to_string(),
                estimated_secs: 10,
                reversible: true,
            },
            PlannedStep {
                name: "Propagate to peers".to_string(),
                estimated_secs: 15,
                reversible: true,
            },
            PlannedStep {
                name: "Verify connectivity".to_string(),
                estimated_secs: 20,
                reversible: true,
            },
            PlannedStep {
                name: "Cutover".to_string(),
                estimated_secs: 5,
                reversible: false,
            },
            PlannedStep {
                name: "Cleanup".to_string(),
                estimated_secs: 5,
                reversible: false,
            },
        ];

        let estimated_duration: u64 = steps.iter().map(|s| s.estimated_secs).sum();

        Ok(MigrationPlan {
            migration,
            source_subnet: source_subnet.into(),
            target_subnet: target_subnet.into(),
            target_ip,
            steps,
            estimated_duration_secs: estimated_duration,
            dependencies: Vec::new(),
            created_at: Utc::now(),
        })
    }

    /// Plan bulk migration of multiple nodes
    pub fn plan_bulk(
        &mut self,
        nodes: &[(Uuid, SubnetAssignment, Ipv4Addr)], // (node_id, current_assignment, target_ip)
        source_subnet: &Subnet,
        target_subnet: &Subnet,
        reason: MigrationReason,
    ) -> Result<BulkMigrationPlan> {
        let mut migrations = Vec::new();
        let mut errors = Vec::new();

        for (node_id, assignment, target_ip) in nodes {
            match self.plan(
                *node_id,
                source_subnet,
                target_subnet,
                assignment,
                *target_ip,
                reason.clone(),
            ) {
                Ok(plan) => migrations.push(plan),
                Err(e) => errors.push(format!("Node {}: {}", node_id, e)),
            }
        }

        if migrations.is_empty() && !errors.is_empty() {
            return Err(Error::MigrationFailed(errors.join("; ")));
        }

        // Group migrations into waves based on max_concurrent
        let mut execution_order = Vec::new();
        let max_concurrent = self.constraints.max_concurrent;

        for chunk in migrations.chunks(max_concurrent) {
            let wave: Vec<Uuid> = chunk.iter().map(|p| p.migration.id).collect();
            execution_order.push(wave);
        }

        // Calculate total duration (waves are sequential)
        let wave_duration = migrations
            .first()
            .map(|m| m.estimated_duration_secs)
            .unwrap_or(60);
        let total_estimated = wave_duration * execution_order.len() as u64;

        Ok(BulkMigrationPlan {
            migrations,
            execution_order,
            total_estimated_secs: total_estimated,
            created_at: Utc::now(),
        })
    }

    /// Get active migration count
    pub fn active_count(&self) -> usize {
        self.active_migrations.len()
    }

    /// Get active migrations for a subnet
    pub fn active_for_subnet(&self, subnet_id: Uuid) -> Vec<&Migration> {
        self.active_migrations
            .values()
            .filter(|m| m.source_subnet_id == subnet_id || m.target_subnet_id == subnet_id)
            .collect()
    }

    /// Clear stale records (for testing or maintenance)
    pub fn clear_records(&mut self) {
        self.recent_migrations.clear();
        self.active_migrations.clear();
    }
}

impl Default for MigrationPlanner {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::migration::MigrationStatus;
    use crate::models::{SubnetPurpose, SubnetStatus};

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

    #[test]
    fn test_validate_success() {
        let planner = MigrationPlanner::new();
        let source = create_test_subnet("source", "10.100.0.0/24");
        let target = create_test_subnet("target", "10.101.0.0/24");
        let assignment = create_test_assignment(source.id);

        let validation = planner.validate(assignment.node_id, &source, &target, &assignment);

        assert!(validation.valid);
        assert!(validation.issues.is_empty());
    }

    #[test]
    fn test_validate_same_subnet() {
        let planner = MigrationPlanner::new();
        let subnet = create_test_subnet("subnet", "10.100.0.0/24");
        let assignment = create_test_assignment(subnet.id);

        let validation = planner.validate(assignment.node_id, &subnet, &subnet, &assignment);

        assert!(!validation.valid);
        assert!(validation.issues.iter().any(|i| i.code == "SAME_SUBNET"));
    }

    #[test]
    fn test_validate_target_not_active() {
        let planner = MigrationPlanner::new();
        let source = create_test_subnet("source", "10.100.0.0/24");
        let mut target = create_test_subnet("target", "10.101.0.0/24");
        target.status = SubnetStatus::Draining;
        let assignment = create_test_assignment(source.id);

        let validation = planner.validate(assignment.node_id, &source, &target, &assignment);

        assert!(!validation.valid);
        assert!(validation
            .issues
            .iter()
            .any(|i| i.code == "TARGET_NOT_ACTIVE"));
    }

    #[test]
    fn test_validate_target_full() {
        let planner = MigrationPlanner::new();
        let source = create_test_subnet("source", "10.100.0.0/24");
        let mut target = create_test_subnet("target", "10.101.0.0/24");
        target.max_nodes = Some(10);
        target.current_nodes = 10;
        let assignment = create_test_assignment(source.id);

        let validation = planner.validate(assignment.node_id, &source, &target, &assignment);

        assert!(!validation.valid);
        assert!(validation.issues.iter().any(|i| i.code == "TARGET_FULL"));
    }

    #[test]
    fn test_validate_cross_purpose_blocked() {
        let planner = MigrationPlanner::new();
        let source = create_test_subnet("source", "10.100.0.0/24");
        let mut target = create_test_subnet("target", "10.112.0.0/24");
        target.purpose = SubnetPurpose::NodeType;
        let assignment = create_test_assignment(source.id);

        let validation = planner.validate(assignment.node_id, &source, &target, &assignment);

        assert!(!validation.valid);
        assert!(validation.issues.iter().any(|i| i.code == "CROSS_PURPOSE"));
    }

    #[test]
    fn test_validate_cross_purpose_allowed() {
        let mut constraints = MigrationConstraint::default();
        constraints.allow_cross_purpose = true;
        let planner = MigrationPlanner::with_constraints(constraints);

        let source = create_test_subnet("source", "10.100.0.0/24");
        let mut target = create_test_subnet("target", "10.112.0.0/24");
        target.purpose = SubnetPurpose::NodeType;
        let assignment = create_test_assignment(source.id);

        let validation = planner.validate(assignment.node_id, &source, &target, &assignment);

        assert!(validation.valid);
    }

    #[test]
    fn test_validate_node_excluded() {
        let node_id = Uuid::new_v4();
        let mut constraints = MigrationConstraint::default();
        constraints.excluded_nodes.insert(node_id);
        let planner = MigrationPlanner::with_constraints(constraints);

        let source = create_test_subnet("source", "10.100.0.0/24");
        let target = create_test_subnet("target", "10.101.0.0/24");
        let mut assignment = create_test_assignment(source.id);
        assignment.node_id = node_id;

        let validation = planner.validate(node_id, &source, &target, &assignment);

        assert!(!validation.valid);
        assert!(validation.issues.iter().any(|i| i.code == "NODE_EXCLUDED"));
    }

    #[test]
    fn test_plan_success() {
        let mut planner = MigrationPlanner::new();
        let source = create_test_subnet("source", "10.100.0.0/24");
        let target = create_test_subnet("target", "10.101.0.0/24");
        let assignment = create_test_assignment(source.id);
        let target_ip = Ipv4Addr::new(10, 101, 0, 20);

        let plan = planner
            .plan(
                assignment.node_id,
                &source,
                &target,
                &assignment,
                target_ip,
                MigrationReason::Manual,
            )
            .unwrap();

        assert_eq!(plan.target_ip, target_ip);
        assert_eq!(plan.migration.status, MigrationStatus::Pending);
        assert!(!plan.steps.is_empty());
        assert!(plan.estimated_duration_secs > 0);
    }

    #[test]
    fn test_plan_bulk() {
        let mut planner = MigrationPlanner::new();
        let source = create_test_subnet("source", "10.100.0.0/24");
        let target = create_test_subnet("target", "10.101.0.0/24");

        let nodes: Vec<(Uuid, SubnetAssignment, Ipv4Addr)> = (0..15)
            .map(|i| {
                let mut assignment = create_test_assignment(source.id);
                assignment.node_id = Uuid::new_v4();
                assignment.assigned_ip = Ipv4Addr::new(10, 100, 0, 10 + i);
                let target_ip = Ipv4Addr::new(10, 101, 0, 10 + i);
                (assignment.node_id, assignment, target_ip)
            })
            .collect();

        let bulk_plan = planner
            .plan_bulk(
                &nodes,
                &source,
                &target,
                MigrationReason::SubnetDecommission,
            )
            .unwrap();

        assert_eq!(bulk_plan.total_migrations(), 15);
        // With max_concurrent=10, should have 2 waves
        assert_eq!(bulk_plan.wave_count(), 2);
    }

    #[test]
    fn test_active_migration_tracking() {
        let mut planner = MigrationPlanner::new();
        let source = create_test_subnet("source", "10.100.0.0/24");
        let target = create_test_subnet("target", "10.101.0.0/24");
        let assignment = create_test_assignment(source.id);

        let migration = Migration::new(
            assignment.node_id,
            source.id,
            target.id,
            assignment.assigned_ip,
            MigrationReason::Manual,
        );

        planner.record_migration_start(&migration);
        assert_eq!(planner.active_count(), 1);
        assert_eq!(planner.active_for_subnet(source.id).len(), 1);

        planner.record_migration_complete(migration.id, migration.node_id);
        assert_eq!(planner.active_count(), 0);
    }

    #[test]
    fn test_maintenance_window() {
        let mut constraints = MigrationConstraint::default();
        // Set window to current hour to ensure we're in it
        let current_hour = Utc::now().time().hour() as u8;
        constraints.maintenance_window = Some((current_hour, (current_hour + 2) % 24));

        assert!(constraints.is_in_maintenance_window());

        // Set window to a different time
        constraints.maintenance_window = Some(((current_hour + 12) % 24, (current_hour + 14) % 24));
        assert!(!constraints.is_in_maintenance_window());
    }

    #[test]
    fn test_reason_display() {
        assert_eq!(MigrationReason::Manual.to_string(), "Manual");
        assert_eq!(MigrationReason::PolicyChange.to_string(), "Policy Change");
        assert_eq!(
            MigrationReason::SubnetDecommission.to_string(),
            "Subnet Decommission"
        );
    }
}
