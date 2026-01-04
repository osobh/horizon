//! Data Transfer Objects for the REST API
//!
//! Request and response types for API endpoints.

use crate::migration::{MigrationReason, MigrationStatus, MigrationStep};
use crate::models::{NodeType, Region, RouteDirection, SubnetPurpose, SubnetStatus};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::net::Ipv4Addr;
use uuid::Uuid;

// ============================================================================
// Subnet DTOs
// ============================================================================

/// Request to create a new subnet (API)
#[derive(Debug, Clone, Deserialize)]
pub struct CreateSubnetDto {
    /// Human-readable name
    pub name: String,
    /// Optional description
    pub description: Option<String>,
    /// Purpose of the subnet (required unless template_id is provided)
    pub purpose: Option<SubnetPurpose>,
    /// CIDR block (optional - auto-allocated if not provided)
    pub cidr: Option<String>,
    /// Prefix length for auto-allocation (required if cidr not provided)
    pub prefix_len: Option<u8>,
    /// WireGuard listen port
    pub wg_listen_port: Option<u16>,
    /// Maximum nodes (optional)
    pub max_nodes: Option<i32>,
    /// Dimension-specific fields
    pub tenant_id: Option<Uuid>,
    pub node_type: Option<NodeType>,
    pub region: Option<Region>,
    pub resource_pool_id: Option<Uuid>,
    /// Template to create from (optional)
    pub template_id: Option<Uuid>,
}

/// Request to update a subnet
#[derive(Debug, Clone, Deserialize)]
pub struct UpdateSubnetRequest {
    /// New name (optional)
    pub name: Option<String>,
    /// New description (optional)
    pub description: Option<String>,
    /// New status (optional)
    pub status: Option<SubnetStatus>,
    /// New max nodes (optional)
    pub max_nodes: Option<i32>,
}

/// Subnet response
#[derive(Debug, Clone, Serialize)]
pub struct SubnetResponse {
    pub id: Uuid,
    pub name: String,
    pub description: Option<String>,
    pub cidr: String,
    pub purpose: SubnetPurpose,
    pub status: SubnetStatus,
    pub tenant_id: Option<Uuid>,
    pub node_type: Option<NodeType>,
    pub region: Option<Region>,
    pub resource_pool_id: Option<Uuid>,
    pub wg_interface: String,
    pub wg_listen_port: u16,
    pub wg_public_key: Option<String>,
    pub max_nodes: Option<i32>,
    pub current_nodes: i32,
    pub available_ips: usize,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// List subnets query parameters
#[derive(Debug, Clone, Deserialize, Default)]
pub struct ListSubnetsQuery {
    /// Filter by purpose
    pub purpose: Option<SubnetPurpose>,
    /// Filter by status
    pub status: Option<SubnetStatus>,
    /// Filter by tenant
    pub tenant_id: Option<Uuid>,
    /// Filter by node type
    pub node_type: Option<NodeType>,
    /// Pagination offset
    pub offset: Option<usize>,
    /// Pagination limit
    pub limit: Option<usize>,
}

// ============================================================================
// Node Assignment DTOs
// ============================================================================

/// Request to assign a node to a subnet (API)
#[derive(Debug, Clone, Deserialize)]
pub struct AssignNodeDto {
    /// Node ID to assign
    pub node_id: Uuid,
    /// Specific IP to assign (optional - auto-allocated if not provided)
    pub ip_address: Option<Ipv4Addr>,
    /// Node's WireGuard public key
    pub wg_public_key: String,
    /// Assignment method description
    pub assignment_method: Option<String>,
}

/// Request to evaluate policy for a node
#[derive(Debug, Clone, Deserialize)]
pub struct EvaluatePolicyRequest {
    /// Node attributes for evaluation
    pub node_type: Option<NodeType>,
    pub tenant_id: Option<Uuid>,
    pub region: Option<String>,
    pub hostname: Option<String>,
    pub labels: Option<std::collections::HashMap<String, String>>,
    pub gpu_count: Option<u32>,
    pub gpu_memory_gb: Option<u32>,
    pub cpu_cores: Option<u32>,
    pub memory_gb: Option<u32>,
}

/// Node assignment response
#[derive(Debug, Clone, Serialize)]
pub struct NodeAssignmentResponse {
    pub id: Uuid,
    pub node_id: Uuid,
    pub subnet_id: Uuid,
    pub assigned_ip: Ipv4Addr,
    pub wg_public_key: String,
    pub assigned_at: DateTime<Utc>,
    pub assignment_method: String,
    pub policy_id: Option<Uuid>,
}

/// Policy evaluation response
#[derive(Debug, Clone, Serialize)]
pub struct PolicyEvaluationResponse {
    pub matched: bool,
    pub subnet_id: Option<Uuid>,
    pub subnet_name: Option<String>,
    pub policy_id: Option<Uuid>,
    pub policy_name: Option<String>,
}

// ============================================================================
// Policy DTOs
// ============================================================================

/// Request to create a policy
#[derive(Debug, Clone, Deserialize)]
pub struct CreatePolicyRequest {
    /// Policy name
    pub name: String,
    /// Description
    pub description: Option<String>,
    /// Priority (higher = evaluated first)
    pub priority: i32,
    /// Target subnet ID
    pub target_subnet_id: Uuid,
    /// Policy rules
    pub rules: Vec<PolicyRuleDto>,
    /// Whether policy is enabled
    pub enabled: Option<bool>,
}

/// Policy rule DTO
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyRuleDto {
    /// Attribute to match
    pub attribute: String,
    /// Operator
    pub operator: String,
    /// Value to match
    pub value: serde_json::Value,
}

/// Request to update a policy
#[derive(Debug, Clone, Deserialize)]
pub struct UpdatePolicyRequest {
    /// New name
    pub name: Option<String>,
    /// New description
    pub description: Option<String>,
    /// New priority
    pub priority: Option<i32>,
    /// New enabled status
    pub enabled: Option<bool>,
    /// New rules (replaces all existing)
    pub rules: Option<Vec<PolicyRuleDto>>,
}

/// Policy response
#[derive(Debug, Clone, Serialize)]
pub struct PolicyResponse {
    pub id: Uuid,
    pub name: String,
    pub description: Option<String>,
    pub priority: i32,
    pub enabled: bool,
    pub target_subnet_id: Uuid,
    pub rules: Vec<PolicyRuleDto>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

// ============================================================================
// Migration DTOs
// ============================================================================

/// Request to start a migration
#[derive(Debug, Clone, Deserialize)]
pub struct StartMigrationRequest {
    /// Node to migrate
    pub node_id: Uuid,
    /// Target subnet
    pub target_subnet_id: Uuid,
    /// Reason for migration
    pub reason: MigrationReason,
    /// Priority (optional)
    pub priority: Option<i32>,
}

/// Request for bulk migration
#[derive(Debug, Clone, Deserialize)]
pub struct BulkMigrationRequest {
    /// Nodes to migrate
    pub node_ids: Vec<Uuid>,
    /// Target subnet
    pub target_subnet_id: Uuid,
    /// Reason for migration
    pub reason: MigrationReason,
}

/// Migration response
#[derive(Debug, Clone, Serialize)]
pub struct MigrationResponse {
    pub id: Uuid,
    pub node_id: Uuid,
    pub source_subnet_id: Uuid,
    pub target_subnet_id: Uuid,
    pub source_ip: Ipv4Addr,
    pub target_ip: Option<Ipv4Addr>,
    pub status: MigrationStatus,
    pub reason: MigrationReason,
    pub current_step: Option<MigrationStep>,
    pub progress_percent: Option<u8>,
    pub created_at: DateTime<Utc>,
    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
    pub last_error: Option<String>,
}

/// Bulk migration response
#[derive(Debug, Clone, Serialize)]
pub struct BulkMigrationResponse {
    pub migrations: Vec<MigrationResponse>,
    pub total: usize,
    pub wave_count: usize,
    pub estimated_duration_secs: u64,
}

/// Migration progress response
#[derive(Debug, Clone, Serialize)]
pub struct MigrationProgressResponse {
    pub migration_id: Uuid,
    pub current_step: MigrationStep,
    pub progress_percent: u8,
    pub peers_notified: usize,
    pub peers_acknowledged: usize,
    pub connectivity_verified: bool,
    pub steps_completed: Vec<CompletedStepDto>,
}

/// Completed step DTO
#[derive(Debug, Clone, Serialize)]
pub struct CompletedStepDto {
    pub step: MigrationStep,
    pub started_at: DateTime<Utc>,
    pub completed_at: DateTime<Utc>,
    pub success: bool,
    pub error: Option<String>,
}

// ============================================================================
// Route DTOs
// ============================================================================

/// Request to create a cross-subnet route (API)
#[derive(Debug, Clone, Deserialize)]
pub struct CreateRouteDto {
    /// Source subnet ID
    pub source_subnet_id: Uuid,
    /// Destination subnet ID
    pub destination_subnet_id: Uuid,
    /// Direction
    pub direction: RouteDirection,
    /// Allowed ports (optional)
    pub allowed_ports: Option<Vec<PortRangeDto>>,
    /// Allowed protocols (optional)
    pub allowed_protocols: Option<Vec<String>>,
    /// Description
    pub description: Option<String>,
}

/// Port range DTO
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortRangeDto {
    pub start: u16,
    pub end: u16,
}

/// Route response
#[derive(Debug, Clone, Serialize)]
pub struct RouteResponse {
    pub id: Uuid,
    pub source_subnet_id: Uuid,
    pub source_subnet_name: Option<String>,
    pub destination_subnet_id: Uuid,
    pub destination_subnet_name: Option<String>,
    pub direction: RouteDirection,
    pub allowed_ports: Option<Vec<PortRangeDto>>,
    pub allowed_protocols: Option<Vec<String>>,
    pub status: String,
    pub created_at: DateTime<Utc>,
}

// ============================================================================
// Template DTOs
// ============================================================================

/// Template response
#[derive(Debug, Clone, Serialize)]
pub struct TemplateResponse {
    pub id: Uuid,
    pub name: String,
    pub description: Option<String>,
    pub purpose: SubnetPurpose,
    pub default_prefix_len: u8,
    pub default_max_nodes: Option<i32>,
    pub wg_listen_port: u16,
    pub is_system: bool,
}

/// Create subnet from template request
#[derive(Debug, Clone, Deserialize)]
pub struct CreateFromTemplateRequest {
    /// Template ID
    pub template_id: Uuid,
    /// Subnet name
    pub name: String,
    /// Optional description
    pub description: Option<String>,
    /// Override CIDR (optional)
    pub cidr: Option<String>,
    /// Override max nodes (optional)
    pub max_nodes: Option<i32>,
    /// Dimension-specific overrides
    pub tenant_id: Option<Uuid>,
    pub node_type: Option<NodeType>,
    pub region: Option<Region>,
    pub resource_pool_id: Option<Uuid>,
}

// ============================================================================
// Stats DTOs
// ============================================================================

/// Subnet statistics response
#[derive(Debug, Clone, Serialize)]
pub struct SubnetStatsResponse {
    pub subnet_id: Uuid,
    pub total_capacity: usize,
    pub allocated_ips: usize,
    pub available_ips: usize,
    pub utilization_percent: f64,
}

/// Manager statistics response
#[derive(Debug, Clone, Serialize)]
pub struct ManagerStatsResponse {
    pub total_subnets: usize,
    pub active_subnets: usize,
    pub total_nodes: usize,
    pub total_policies: usize,
    pub active_migrations: usize,
    pub subnets_by_purpose: std::collections::HashMap<String, usize>,
    pub subnets_by_status: std::collections::HashMap<String, usize>,
}

/// Migration statistics response
#[derive(Debug, Clone, Serialize)]
pub struct MigrationStatsResponse {
    pub total: usize,
    pub pending: usize,
    pub in_progress: usize,
    pub completed: usize,
    pub failed: usize,
    pub cancelled: usize,
    pub avg_duration_secs: Option<f64>,
    pub success_rate: Option<f64>,
}

// ============================================================================
// Common DTOs
// ============================================================================

/// Paginated list response
#[derive(Debug, Clone, Serialize)]
pub struct PaginatedResponse<T> {
    pub items: Vec<T>,
    pub total: usize,
    pub offset: usize,
    pub limit: usize,
}

/// API error response
#[derive(Debug, Clone, Serialize)]
pub struct ApiError {
    pub code: String,
    pub message: String,
    pub details: Option<serde_json::Value>,
}

impl ApiError {
    pub fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
            details: None,
        }
    }

    pub fn with_details(mut self, details: serde_json::Value) -> Self {
        self.details = Some(details);
        self
    }

    pub fn not_found(resource: &str, id: &str) -> Self {
        Self::new(
            "NOT_FOUND",
            format!("{} with id '{}' not found", resource, id),
        )
    }

    pub fn bad_request(message: impl Into<String>) -> Self {
        Self::new("BAD_REQUEST", message)
    }

    pub fn conflict(message: impl Into<String>) -> Self {
        Self::new("CONFLICT", message)
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self::new("INTERNAL_ERROR", message)
    }
}

/// Success response wrapper
#[derive(Debug, Clone, Serialize)]
pub struct SuccessResponse<T> {
    pub success: bool,
    pub data: T,
}

impl<T> SuccessResponse<T> {
    pub fn new(data: T) -> Self {
        Self {
            success: true,
            data,
        }
    }
}

/// Empty success response
#[derive(Debug, Clone, Serialize)]
pub struct EmptyResponse {
    pub success: bool,
    pub message: String,
}

impl EmptyResponse {
    pub fn ok(message: impl Into<String>) -> Self {
        Self {
            success: true,
            message: message.into(),
        }
    }
}
