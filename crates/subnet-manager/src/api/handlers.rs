//! REST API handlers
//!
//! Implements handlers for subnet, policy, migration, and route endpoints.

use super::dto::*;
use super::state::AppState;
use crate::models::{CrossSubnetRoute, RouteStatus, Subnet};
use crate::policy_engine::NodeAttributes;
use crate::service::{AssignNodeRequest, CreateSubnetRequest};
use crate::Error;
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::IntoResponse,
    Json,
};
use std::sync::Arc;
use uuid::Uuid;

/// Convert internal error to API response
fn error_response(err: Error) -> (StatusCode, Json<ApiError>) {
    let (status, error) = match &err {
        Error::SubnetNotFound(id) => (
            StatusCode::NOT_FOUND,
            ApiError::not_found("Subnet", &id.to_string()),
        ),
        Error::PolicyNotFound(id) => (
            StatusCode::NOT_FOUND,
            ApiError::not_found("Policy", &id.to_string()),
        ),
        Error::MigrationNotFound(id) => (
            StatusCode::NOT_FOUND,
            ApiError::not_found("Migration", &id.to_string()),
        ),
        Error::TemplateNotFound(id) => (
            StatusCode::NOT_FOUND,
            ApiError::not_found("Template", &id.to_string()),
        ),
        Error::NodeNotAssigned(id) => (
            StatusCode::NOT_FOUND,
            ApiError::not_found("Node assignment", &id.to_string()),
        ),
        Error::SubnetAlreadyExists(name) => (
            StatusCode::CONFLICT,
            ApiError::conflict(format!("Subnet '{}' already exists", name)),
        ),
        Error::NodeAlreadyAssigned(node_id, subnet_id) => (
            StatusCode::CONFLICT,
            ApiError::conflict(format!(
                "Node {} is already assigned to subnet {}",
                node_id, subnet_id
            )),
        ),
        Error::IpAlreadyAllocated(ip) => (
            StatusCode::CONFLICT,
            ApiError::conflict(format!("IP {} is already allocated", ip)),
        ),
        Error::MigrationInProgress(node_id) => (
            StatusCode::CONFLICT,
            ApiError::conflict(format!("Migration already in progress for node {}", node_id)),
        ),
        Error::SubnetNotEmpty(count) => (
            StatusCode::CONFLICT,
            ApiError::conflict(format!("Subnet is not empty, contains {} nodes", count)),
        ),
        Error::InvalidCidr(msg) => (StatusCode::BAD_REQUEST, ApiError::bad_request(msg.clone())),
        Error::InvalidArgument(msg) => {
            (StatusCode::BAD_REQUEST, ApiError::bad_request(msg.clone()))
        }
        Error::NoAvailableIps(subnet_id) => (
            StatusCode::CONFLICT,
            ApiError::conflict(format!("No available IPs in subnet {}", subnet_id)),
        ),
        Error::MigrationFailed(msg) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            ApiError::internal(msg.clone()),
        ),
        _ => (
            StatusCode::INTERNAL_SERVER_ERROR,
            ApiError::internal(err.to_string()),
        ),
    };

    (status, Json(error))
}

// ============================================================================
// Subnet Handlers
// ============================================================================

/// List all subnets
pub async fn list_subnets(
    State(state): State<Arc<AppState>>,
    Query(query): Query<ListSubnetsQuery>,
) -> impl IntoResponse {
    let manager = state.manager.read().await;
    let mut subnets: Vec<SubnetResponse> = manager
        .list_subnets()
        .iter()
        .filter(|s| {
            query.purpose.map_or(true, |p| s.purpose == p)
                && query.status.map_or(true, |st| s.status == st)
                && query.tenant_id.map_or(true, |t| s.tenant_id == Some(t))
                && query.node_type.map_or(true, |nt| s.node_type == Some(nt))
        })
        .map(|s| subnet_to_response(s))
        .collect();

    let total = subnets.len();
    let offset = query.offset.unwrap_or(0);
    let limit = query.limit.unwrap_or(100).min(1000);

    subnets = subnets.into_iter().skip(offset).take(limit).collect();

    Json(PaginatedResponse {
        items: subnets,
        total,
        offset,
        limit,
    })
}

/// Get a single subnet
pub async fn get_subnet(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> Result<Json<SubnetResponse>, (StatusCode, Json<ApiError>)> {
    let manager = state.manager.read().await;
    let subnet = manager
        .get_subnet(id)
        .ok_or_else(|| error_response(Error::SubnetNotFound(id)))?;
    Ok(Json(subnet_to_response(&subnet)))
}

/// Create a new subnet
pub async fn create_subnet(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateSubnetDto>,
) -> Result<(StatusCode, Json<SubnetResponse>), (StatusCode, Json<ApiError>)> {
    let manager = state.manager.write().await;

    // If template provided, create from template
    if let Some(template_id) = req.template_id {
        let subnet = manager
            .create_from_template(template_id, req.name.clone(), None)
            .map_err(error_response)?;
        return Ok((StatusCode::CREATED, Json(subnet_to_response(&subnet))));
    }

    // Purpose is required when not using a template
    let purpose = req.purpose.ok_or_else(|| {
        (
            StatusCode::BAD_REQUEST,
            Json(ApiError::bad_request(
                "purpose is required when not using a template",
            )),
        )
    })?;

    // Build the create request
    let create_req = CreateSubnetRequest {
        name: req.name,
        description: req.description,
        purpose,
        cidr: req.cidr.and_then(|s| s.parse().ok()),
        prefix_length: req.prefix_len,
        max_nodes: req.max_nodes,
        tenant_id: req.tenant_id,
        node_type: req.node_type,
        region: req.region,
        resource_pool_id: req.resource_pool_id,
        template_id: None,
        created_by: None,
    };

    let subnet = manager.create_subnet(create_req).map_err(error_response)?;
    Ok((StatusCode::CREATED, Json(subnet_to_response(&subnet))))
}

/// Update a subnet
pub async fn update_subnet(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
    Json(req): Json<UpdateSubnetRequest>,
) -> Result<Json<SubnetResponse>, (StatusCode, Json<ApiError>)> {
    let manager = state.manager.write().await;

    // Get current subnet
    let subnet = manager
        .get_subnet(id)
        .ok_or_else(|| error_response(Error::SubnetNotFound(id)))?;

    // Update status if provided
    if let Some(status) = req.status {
        manager
            .set_subnet_status(id, status)
            .map_err(error_response)?;
    }

    // Re-fetch updated subnet
    let updated = manager
        .get_subnet(id)
        .ok_or_else(|| error_response(Error::SubnetNotFound(id)))?;

    Ok(Json(subnet_to_response(&updated)))
}

/// Delete a subnet
pub async fn delete_subnet(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> Result<Json<EmptyResponse>, (StatusCode, Json<ApiError>)> {
    let manager = state.manager.write().await;
    manager.delete_subnet(id).map_err(error_response)?;
    Ok(Json(EmptyResponse::ok("Subnet deleted")))
}

/// Get subnet statistics
pub async fn get_subnet_stats(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> Result<Json<SubnetStatsResponse>, (StatusCode, Json<ApiError>)> {
    let manager = state.manager.read().await;
    let subnet = manager
        .get_subnet(id)
        .ok_or_else(|| error_response(Error::SubnetNotFound(id)))?;

    let total_capacity = subnet.cidr.hosts().count();
    let allocated = subnet.current_nodes as usize;
    let available = total_capacity.saturating_sub(allocated);
    let utilization = if total_capacity > 0 {
        (allocated as f64 / total_capacity as f64) * 100.0
    } else {
        0.0
    };

    Ok(Json(SubnetStatsResponse {
        subnet_id: id,
        total_capacity,
        allocated_ips: allocated,
        available_ips: available,
        utilization_percent: utilization,
    }))
}

/// List nodes in a subnet
pub async fn list_subnet_nodes(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> Result<Json<Vec<NodeAssignmentResponse>>, (StatusCode, Json<ApiError>)> {
    let manager = state.manager.read().await;

    // Verify subnet exists
    let _ = manager
        .get_subnet(id)
        .ok_or_else(|| error_response(Error::SubnetNotFound(id)))?;

    let assignments: Vec<NodeAssignmentResponse> = manager
        .get_subnet_assignments(id)
        .into_iter()
        .map(|a| NodeAssignmentResponse {
            id: a.id,
            node_id: a.node_id,
            subnet_id: a.subnet_id,
            assigned_ip: a.assigned_ip,
            wg_public_key: a.wg_public_key,
            assigned_at: a.assigned_at,
            assignment_method: a.assignment_method,
            policy_id: a.policy_id,
        })
        .collect();

    Ok(Json(assignments))
}

/// Assign a node to a subnet
pub async fn assign_node(
    State(state): State<Arc<AppState>>,
    Path(subnet_id): Path<Uuid>,
    Json(req): Json<AssignNodeDto>,
) -> Result<(StatusCode, Json<NodeAssignmentResponse>), (StatusCode, Json<ApiError>)> {
    let manager = state.manager.write().await;

    let assign_req = AssignNodeRequest {
        node_id: req.node_id,
        wg_public_key: req.wg_public_key,
        subnet_id: Some(subnet_id),
        attributes: None,
        method: req.assignment_method.unwrap_or_else(|| "api".to_string()),
    };

    let assignment = manager.assign_node(assign_req).map_err(error_response)?;

    Ok((
        StatusCode::CREATED,
        Json(NodeAssignmentResponse {
            id: assignment.id,
            node_id: assignment.node_id,
            subnet_id: assignment.subnet_id,
            assigned_ip: assignment.assigned_ip,
            wg_public_key: assignment.wg_public_key,
            assigned_at: assignment.assigned_at,
            assignment_method: assignment.assignment_method,
            policy_id: assignment.policy_id,
        }),
    ))
}

/// Remove a node from a subnet
pub async fn unassign_node(
    State(state): State<Arc<AppState>>,
    Path((subnet_id, node_id)): Path<(Uuid, Uuid)>,
) -> Result<Json<EmptyResponse>, (StatusCode, Json<ApiError>)> {
    let manager = state.manager.write().await;

    // Verify it's in the right subnet
    let assignment = manager
        .get_assignment(node_id)
        .ok_or_else(|| error_response(Error::NodeNotAssigned(node_id)))?;

    if assignment.subnet_id != subnet_id {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ApiError::bad_request(format!(
                "Node {} is not in subnet {}",
                node_id, subnet_id
            ))),
        ));
    }

    manager.unassign_node(node_id).map_err(error_response)?;
    Ok(Json(EmptyResponse::ok("Node unassigned")))
}

// ============================================================================
// Policy Handlers
// ============================================================================

/// List all policies
pub async fn list_policies(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let manager = state.manager.read().await;
    let policies: Vec<PolicyResponse> = manager
        .list_policies()
        .iter()
        .map(|p| PolicyResponse {
            id: p.id,
            name: p.name.clone(),
            description: p.description.clone(),
            priority: p.priority,
            enabled: p.enabled,
            target_subnet_id: p.target_subnet_id,
            rules: p
                .rules
                .iter()
                .map(|r| PolicyRuleDto {
                    attribute: format!("{:?}", r.attribute),
                    operator: format!("{:?}", r.operator),
                    value: serde_json::to_value(&r.value).unwrap_or_default(),
                })
                .collect(),
            created_at: p.created_at,
            updated_at: p.updated_at,
        })
        .collect();

    Json(policies)
}

/// Get a policy
pub async fn get_policy(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> Result<Json<PolicyResponse>, (StatusCode, Json<ApiError>)> {
    let manager = state.manager.read().await;
    let policy = manager
        .get_policy(id)
        .ok_or_else(|| error_response(Error::PolicyNotFound(id)))?;

    Ok(Json(PolicyResponse {
        id: policy.id,
        name: policy.name.clone(),
        description: policy.description.clone(),
        priority: policy.priority,
        enabled: policy.enabled,
        target_subnet_id: policy.target_subnet_id,
        rules: policy
            .rules
            .iter()
            .map(|r| PolicyRuleDto {
                attribute: format!("{:?}", r.attribute),
                operator: format!("{:?}", r.operator),
                value: serde_json::to_value(&r.value).unwrap_or_default(),
            })
            .collect(),
        created_at: policy.created_at,
        updated_at: policy.updated_at,
    }))
}

/// Evaluate policy for a node (dry run)
pub async fn evaluate_policy(
    State(state): State<Arc<AppState>>,
    Json(req): Json<EvaluatePolicyRequest>,
) -> impl IntoResponse {
    let manager = state.manager.read().await;

    let mut attrs = NodeAttributes::default();
    if let Some(nt) = req.node_type {
        attrs.node_type = Some(nt);
    }
    if let Some(tid) = req.tenant_id {
        attrs.tenant_id = Some(tid);
    }
    if let Some(region) = req.region {
        attrs.region = Some(region);
    }
    if let Some(hostname) = req.hostname {
        attrs.hostname = Some(hostname);
    }
    if let Some(labels) = req.labels {
        // Convert HashMap to Vec<String> (format: "key=value")
        attrs.labels = labels.into_iter().map(|(k, v)| format!("{}={}", k, v)).collect();
    }
    if let Some(gpu) = req.gpu_count {
        attrs.gpu_count = Some(gpu as i32);
    }
    if let Some(mem) = req.gpu_memory_gb {
        attrs.gpu_memory_gb = Some(mem as i32);
    }
    if let Some(cpu) = req.cpu_cores {
        attrs.cpu_cores = Some(cpu as i32);
    }
    if let Some(mem) = req.memory_gb {
        attrs.ram_gb = Some(mem as i32);
    }

    let result = manager.evaluate_policies(&attrs);

    if let Some(subnet_id) = result.subnet_id {
        let subnet = manager.get_subnet(subnet_id);
        let policy_info = result.matched_policy.as_ref();
        Json(PolicyEvaluationResponse {
            matched: true,
            subnet_id: Some(subnet_id),
            subnet_name: subnet.map(|s| s.name.clone()),
            policy_id: policy_info.map(|p| p.policy_id),
            policy_name: policy_info.map(|p| p.policy_name.clone()),
        })
    } else {
        Json(PolicyEvaluationResponse {
            matched: false,
            subnet_id: None,
            subnet_name: None,
            policy_id: None,
            policy_name: None,
        })
    }
}

// ============================================================================
// Migration Handlers
// ============================================================================

/// List migrations
pub async fn list_migrations(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let executor = state.migration_executor.read().await;

    let active: Vec<MigrationResponse> = executor
        .active_migrations()
        .into_iter()
        .map(|h| MigrationResponse {
            id: h.id,
            node_id: h.node_id,
            source_subnet_id: h.source_subnet_id,
            target_subnet_id: h.target_subnet_id,
            source_ip: std::net::Ipv4Addr::new(0, 0, 0, 0),
            target_ip: None,
            status: crate::migration::MigrationStatus::InProgress,
            reason: crate::migration::MigrationReason::Manual,
            current_step: None,
            progress_percent: None,
            created_at: h.created_at,
            started_at: None,
            completed_at: None,
            last_error: None,
        })
        .collect();

    Json(active)
}

/// Get migration status
pub async fn get_migration(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> Result<Json<MigrationProgressResponse>, (StatusCode, Json<ApiError>)> {
    let executor = state.migration_executor.read().await;

    let progress = executor.get_progress(id).ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            Json(ApiError::not_found("Migration", &id.to_string())),
        )
    })?;

    Ok(Json(MigrationProgressResponse {
        migration_id: progress.migration_id,
        current_step: progress.current_step,
        progress_percent: progress.progress_percent,
        peers_notified: progress.peers_notified,
        peers_acknowledged: progress.peers_acknowledged,
        connectivity_verified: progress.connectivity_verified,
        steps_completed: progress
            .steps_completed
            .into_iter()
            .map(|s| CompletedStepDto {
                step: s.step,
                started_at: s.started_at,
                completed_at: s.completed_at,
                success: s.success,
                error: s.error,
            })
            .collect(),
    }))
}

/// Start a migration
pub async fn start_migration(
    State(state): State<Arc<AppState>>,
    Json(req): Json<StartMigrationRequest>,
) -> Result<(StatusCode, Json<MigrationResponse>), (StatusCode, Json<ApiError>)> {
    let manager = state.manager.read().await;

    // Get current assignment
    let assignment = manager
        .get_assignment(req.node_id)
        .ok_or_else(|| error_response(Error::NodeNotAssigned(req.node_id)))?;

    // Get source and target subnets
    let source_subnet = manager
        .get_subnet(assignment.subnet_id)
        .ok_or_else(|| error_response(Error::SubnetNotFound(assignment.subnet_id)))?;
    let target_subnet = manager
        .get_subnet(req.target_subnet_id)
        .ok_or_else(|| error_response(Error::SubnetNotFound(req.target_subnet_id)))?;

    // Plan the migration
    let mut planner = state.migration_planner.write().await;

    // Allocate target IP
    let target_ip = target_subnet.cidr.hosts().next().ok_or_else(|| {
        (
            StatusCode::CONFLICT,
            Json(ApiError::conflict("No available IPs in target subnet")),
        )
    })?;

    let plan = planner
        .plan(
            req.node_id,
            &source_subnet,
            &target_subnet,
            &assignment,
            target_ip,
            req.reason.clone(),
        )
        .map_err(|e| {
            (
                StatusCode::BAD_REQUEST,
                Json(ApiError::bad_request(e.to_string())),
            )
        })?;

    // Submit to executor
    let executor = state.migration_executor.read().await;
    let handle = executor.submit(plan.clone()).map_err(|e| {
        (
            StatusCode::CONFLICT,
            Json(ApiError::conflict(e.to_string())),
        )
    })?;

    // Start the migration
    executor.start(handle.id).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiError::internal(e.to_string())),
        )
    })?;

    Ok((
        StatusCode::ACCEPTED,
        Json(MigrationResponse {
            id: handle.id,
            node_id: req.node_id,
            source_subnet_id: assignment.subnet_id,
            target_subnet_id: req.target_subnet_id,
            source_ip: assignment.assigned_ip,
            target_ip: Some(target_ip),
            status: crate::migration::MigrationStatus::InProgress,
            reason: req.reason,
            current_step: Some(crate::migration::MigrationStep::AllocatingIp),
            progress_percent: Some(0),
            created_at: chrono::Utc::now(),
            started_at: Some(chrono::Utc::now()),
            completed_at: None,
            last_error: None,
        }),
    ))
}

/// Cancel a migration
pub async fn cancel_migration(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> Result<Json<EmptyResponse>, (StatusCode, Json<ApiError>)> {
    let executor = state.migration_executor.read().await;
    executor.cancel(id).map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            Json(ApiError::bad_request(e.to_string())),
        )
    })?;
    Ok(Json(EmptyResponse::ok("Migration cancelled")))
}

/// Get migration statistics
pub async fn get_migration_stats(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let executor = state.migration_executor.read().await;
    let history = executor.history();
    let stats = crate::migration::MigrationStats::from_migrations(&history);

    Json(MigrationStatsResponse {
        total: stats.total,
        pending: stats.pending,
        in_progress: stats.in_progress,
        completed: stats.completed,
        failed: stats.failed,
        cancelled: stats.cancelled,
        avg_duration_secs: stats.avg_duration_secs,
        success_rate: stats.success_rate,
    })
}

// ============================================================================
// Route Handlers
// ============================================================================

/// List cross-subnet routes
pub async fn list_routes(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let manager = state.manager.read().await;

    // Get all subnets and their routes
    let mut routes = Vec::new();
    for subnet in manager.list_subnets() {
        for route in manager.get_routes_for_subnet(subnet.id) {
            let status = RouteStatus::from_route(&route);
            routes.push(RouteResponse {
                id: route.id,
                source_subnet_id: route.source_subnet_id,
                source_subnet_name: None,
                destination_subnet_id: route.destination_subnet_id,
                destination_subnet_name: None,
                direction: route.direction,
                allowed_ports: route.allowed_ports.map(|ports| {
                    ports
                        .into_iter()
                        .map(|p| PortRangeDto {
                            start: p.start,
                            end: p.end,
                        })
                        .collect()
                }),
                allowed_protocols: route.allowed_protocols,
                status: format!("{:?}", status),
                created_at: route.created_at,
            });
        }
    }

    Json(routes)
}

/// Create a cross-subnet route
pub async fn create_route(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateRouteDto>,
) -> Result<(StatusCode, Json<RouteResponse>), (StatusCode, Json<ApiError>)> {
    let manager = state.manager.write().await;

    let mut route = CrossSubnetRoute::new(req.source_subnet_id, req.destination_subnet_id);
    route.direction = req.direction;
    route.allowed_protocols = req.allowed_protocols;
    route.description = req.description;

    let status = RouteStatus::from_route(&route);
    let response = RouteResponse {
        id: route.id,
        source_subnet_id: route.source_subnet_id,
        source_subnet_name: None,
        destination_subnet_id: route.destination_subnet_id,
        destination_subnet_name: None,
        direction: route.direction,
        allowed_ports: None,
        allowed_protocols: route.allowed_protocols.clone(),
        status: format!("{:?}", status),
        created_at: route.created_at,
    };

    manager.create_route(route).map_err(error_response)?;

    Ok((StatusCode::CREATED, Json(response)))
}

/// Delete a route
pub async fn delete_route(
    State(state): State<Arc<AppState>>,
    Path(route_id): Path<Uuid>,
) -> Result<Json<EmptyResponse>, (StatusCode, Json<ApiError>)> {
    let manager = state.manager.write().await;
    manager.delete_route(route_id).map_err(error_response)?;
    Ok(Json(EmptyResponse::ok("Route deleted")))
}

// ============================================================================
// Template Handlers
// ============================================================================

/// List templates
pub async fn list_templates(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let manager = state.manager.read().await;
    let templates: Vec<TemplateResponse> = manager
        .list_templates()
        .iter()
        .map(|t| TemplateResponse {
            id: t.id,
            name: t.name.clone(),
            description: t.description.clone(),
            purpose: t.purpose,
            default_prefix_len: t.prefix_length,
            default_max_nodes: t.defaults.max_nodes,
            wg_listen_port: t.defaults.wg_listen_port,
            is_system: t.is_system,
        })
        .collect();

    Json(templates)
}

// ============================================================================
// Stats Handlers
// ============================================================================

/// Get overall manager statistics
pub async fn get_manager_stats(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let manager = state.manager.read().await;
    let stats = manager.get_stats();

    let mut by_purpose: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let mut by_status: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let mut active_count = 0usize;

    for subnet in manager.list_subnets() {
        *by_purpose
            .entry(format!("{:?}", subnet.purpose))
            .or_default() += 1;
        *by_status
            .entry(format!("{:?}", subnet.status))
            .or_default() += 1;
        if subnet.status == crate::models::SubnetStatus::Active {
            active_count += 1;
        }
    }

    Json(ManagerStatsResponse {
        total_subnets: stats.total_subnets,
        active_subnets: active_count,
        total_nodes: stats.total_nodes,
        total_policies: stats.total_policies,
        active_migrations: state.migration_executor.read().await.active_count(),
        subnets_by_purpose: by_purpose,
        subnets_by_status: by_status,
    })
}

// ============================================================================
// Helper Functions
// ============================================================================

fn subnet_to_response(subnet: &Subnet) -> SubnetResponse {
    SubnetResponse {
        id: subnet.id,
        name: subnet.name.clone(),
        description: subnet.description.clone(),
        cidr: subnet.cidr.to_string(),
        purpose: subnet.purpose,
        status: subnet.status,
        tenant_id: subnet.tenant_id,
        node_type: subnet.node_type,
        region: subnet.region.clone(),
        resource_pool_id: subnet.resource_pool_id,
        wg_interface: subnet.wg_interface.clone(),
        wg_listen_port: subnet.wg_listen_port,
        wg_public_key: subnet.wg_public_key.clone(),
        max_nodes: subnet.max_nodes,
        current_nodes: subnet.current_nodes,
        available_ips: subnet.available_ips(),
        created_at: subnet.created_at,
        updated_at: subnet.updated_at,
    }
}
