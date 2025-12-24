//! Subnet Manager Service
//!
//! Core orchestration service for subnet management, providing:
//! - Subnet CRUD operations
//! - Policy-based node assignment
//! - Cross-subnet routing
//! - IP allocation

use crate::allocator::{CidrAllocator, IpAllocator, SubnetAllocator, SubnetIpAllocator};
use crate::models::{
    AssignmentPolicy, CrossSubnetRoute, NodeType, Region, Subnet, SubnetAssignment,
    SubnetPurpose, SubnetStatus, SubnetTemplate,
};
use crate::policy_engine::{NodeAttributes, PolicyEngine, PolicyEvaluator};
use crate::{Error, Result};
use chrono::Utc;
use dashmap::DashMap;
use ipnet::Ipv4Net;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

/// Configuration for SubnetManager
#[derive(Debug, Clone)]
pub struct SubnetManagerConfig {
    /// Default WireGuard listen port (auto-incremented for new subnets)
    pub default_wg_port: u16,
    /// Enable automatic cleanup of expired assignments
    pub auto_cleanup: bool,
    /// Maximum subnets per purpose
    pub max_subnets_per_purpose: Option<usize>,
}

impl Default for SubnetManagerConfig {
    fn default() -> Self {
        Self {
            default_wg_port: 51820,
            auto_cleanup: true,
            max_subnets_per_purpose: None,
        }
    }
}

/// Request to create a new subnet
#[derive(Debug, Clone)]
pub struct CreateSubnetRequest {
    /// Subnet name
    pub name: String,
    /// Description
    pub description: Option<String>,
    /// Purpose of the subnet
    pub purpose: SubnetPurpose,
    /// Optional specific CIDR (auto-allocated if None)
    pub cidr: Option<Ipv4Net>,
    /// Optional prefix length (uses purpose default if None)
    pub prefix_length: Option<u8>,
    /// Tenant ID (for tenant subnets)
    pub tenant_id: Option<Uuid>,
    /// Node type (for node type subnets)
    pub node_type: Option<NodeType>,
    /// Region (for geographic subnets)
    pub region: Option<Region>,
    /// Resource pool ID (for pool subnets)
    pub resource_pool_id: Option<Uuid>,
    /// Maximum nodes
    pub max_nodes: Option<i32>,
    /// Template to use
    pub template_id: Option<Uuid>,
    /// Created by user
    pub created_by: Option<Uuid>,
}

/// Request to assign a node to a subnet
#[derive(Debug, Clone)]
pub struct AssignNodeRequest {
    /// Node ID
    pub node_id: Uuid,
    /// Node's WireGuard public key
    pub wg_public_key: String,
    /// Specific subnet ID (for manual assignment)
    pub subnet_id: Option<Uuid>,
    /// Node attributes (for policy-based assignment)
    pub attributes: Option<NodeAttributes>,
    /// Assignment method description
    pub method: String,
}

/// Subnet Manager - core orchestration service
pub struct SubnetManager {
    /// Configuration
    config: SubnetManagerConfig,
    /// Subnets indexed by ID
    subnets: DashMap<Uuid, Subnet>,
    /// Subnet allocators by purpose
    subnet_allocators: RwLock<HashMap<SubnetPurpose, SubnetAllocator>>,
    /// IP allocators per subnet
    ip_allocators: DashMap<Uuid, Arc<RwLock<SubnetIpAllocator>>>,
    /// Node assignments indexed by node ID
    assignments: DashMap<Uuid, SubnetAssignment>,
    /// Assignments indexed by subnet ID -> node IDs
    subnet_nodes: DashMap<Uuid, Vec<Uuid>>,
    /// Cross-subnet routes
    routes: DashMap<Uuid, CrossSubnetRoute>,
    /// Templates indexed by ID
    templates: DashMap<Uuid, SubnetTemplate>,
    /// Policy engine
    policy_engine: RwLock<PolicyEngine>,
    /// Next WireGuard port
    next_wg_port: RwLock<u16>,
}

impl SubnetManager {
    /// Create a new SubnetManager with default configuration
    pub fn new() -> Self {
        Self::with_config(SubnetManagerConfig::default())
    }

    /// Create a new SubnetManager with custom configuration
    pub fn with_config(config: SubnetManagerConfig) -> Self {
        // Initialize subnet allocators for each purpose
        let mut subnet_allocators = HashMap::new();
        subnet_allocators.insert(SubnetPurpose::Tenant, SubnetAllocator::for_tenants());
        subnet_allocators.insert(SubnetPurpose::NodeType, SubnetAllocator::for_node_types());
        subnet_allocators.insert(SubnetPurpose::Geographic, SubnetAllocator::for_geographic());
        subnet_allocators.insert(
            SubnetPurpose::ResourcePool,
            SubnetAllocator::for_resource_pools(),
        );

        let next_wg_port = config.default_wg_port;

        // Load system templates
        let templates = DashMap::new();
        for template in SubnetTemplate::system_templates() {
            templates.insert(template.id, template);
        }

        Self {
            config,
            subnets: DashMap::new(),
            subnet_allocators: RwLock::new(subnet_allocators),
            ip_allocators: DashMap::new(),
            assignments: DashMap::new(),
            subnet_nodes: DashMap::new(),
            routes: DashMap::new(),
            templates,
            policy_engine: RwLock::new(PolicyEngine::new()),
            next_wg_port: RwLock::new(next_wg_port),
        }
    }

    // ==================== Subnet Operations ====================

    /// Create a new subnet
    pub fn create_subnet(&self, request: CreateSubnetRequest) -> Result<Subnet> {
        // Determine CIDR
        let cidr = if let Some(cidr) = request.cidr {
            // Reserve the specific CIDR
            let mut allocators = self.subnet_allocators.write();
            let allocator = allocators
                .get_mut(&request.purpose)
                .ok_or_else(|| Error::Internal("Missing allocator".to_string()))?;
            allocator.reserve(cidr)?;
            cidr
        } else {
            // Allocate a new CIDR
            let prefix = request
                .prefix_length
                .unwrap_or_else(|| request.purpose.default_prefix());
            let mut allocators = self.subnet_allocators.write();
            let allocator = allocators
                .get_mut(&request.purpose)
                .ok_or_else(|| Error::Internal("Missing allocator".to_string()))?;
            allocator.allocate(prefix)?
        };

        // Get next WireGuard port
        let wg_port = {
            let mut port = self.next_wg_port.write();
            let current = *port;
            *port += 1;
            current
        };

        // Create subnet
        let mut subnet = Subnet::new(&request.name, cidr, request.purpose, wg_port);
        subnet.description = request.description;
        subnet.tenant_id = request.tenant_id;
        subnet.node_type = request.node_type;
        subnet.region = request.region;
        subnet.resource_pool_id = request.resource_pool_id;
        subnet.max_nodes = request.max_nodes;
        subnet.template_id = request.template_id;
        subnet.created_by = request.created_by;
        subnet.status = SubnetStatus::Active;

        // Create IP allocator for this subnet
        let ip_allocator = SubnetIpAllocator::new(cidr);
        self.ip_allocators
            .insert(subnet.id, Arc::new(RwLock::new(ip_allocator)));

        // Initialize subnet nodes list
        self.subnet_nodes.insert(subnet.id, Vec::new());

        // Add to policy engine
        {
            let mut engine = self.policy_engine.write();
            engine.add_subnet(subnet.clone());
        }

        // Store subnet
        let subnet_id = subnet.id;
        self.subnets.insert(subnet_id, subnet.clone());

        tracing::info!(
            subnet_id = %subnet_id,
            name = %request.name,
            cidr = %cidr,
            purpose = ?request.purpose,
            "Created subnet"
        );

        Ok(subnet)
    }

    /// Get a subnet by ID
    pub fn get_subnet(&self, subnet_id: Uuid) -> Option<Subnet> {
        self.subnets.get(&subnet_id).map(|s| s.clone())
    }

    /// List all subnets
    pub fn list_subnets(&self) -> Vec<Subnet> {
        self.subnets.iter().map(|s| s.clone()).collect()
    }

    /// List subnets by purpose
    pub fn list_subnets_by_purpose(&self, purpose: SubnetPurpose) -> Vec<Subnet> {
        self.subnets
            .iter()
            .filter(|s| s.purpose == purpose)
            .map(|s| s.clone())
            .collect()
    }

    /// Delete a subnet
    pub fn delete_subnet(&self, subnet_id: Uuid) -> Result<()> {
        // Check if subnet exists
        let subnet = self
            .subnets
            .get(&subnet_id)
            .ok_or(Error::SubnetNotFound(subnet_id))?;

        // Check if subnet is empty
        if let Some(nodes) = self.subnet_nodes.get(&subnet_id) {
            if !nodes.is_empty() {
                return Err(Error::SubnetNotEmpty(nodes.len()));
            }
        }

        // Release CIDR
        {
            let mut allocators = self.subnet_allocators.write();
            if let Some(allocator) = allocators.get_mut(&subnet.purpose) {
                let _ = allocator.release(subnet.cidr);
            }
        }

        // Remove from policy engine
        {
            let mut engine = self.policy_engine.write();
            engine.remove_subnet(subnet_id);
        }

        // Remove IP allocator
        self.ip_allocators.remove(&subnet_id);

        // Remove subnet nodes list
        self.subnet_nodes.remove(&subnet_id);

        // Remove any routes involving this subnet
        let route_ids: Vec<Uuid> = self
            .routes
            .iter()
            .filter(|r| r.source_subnet_id == subnet_id || r.destination_subnet_id == subnet_id)
            .map(|r| r.id)
            .collect();
        for route_id in route_ids {
            self.routes.remove(&route_id);
        }

        // Remove subnet
        drop(subnet);
        self.subnets.remove(&subnet_id);

        tracing::info!(subnet_id = %subnet_id, "Deleted subnet");

        Ok(())
    }

    /// Update subnet status
    pub fn set_subnet_status(&self, subnet_id: Uuid, status: SubnetStatus) -> Result<()> {
        let mut subnet = self
            .subnets
            .get_mut(&subnet_id)
            .ok_or(Error::SubnetNotFound(subnet_id))?;

        subnet.status = status;
        subnet.updated_at = Utc::now();

        Ok(())
    }

    // ==================== Node Assignment ====================

    /// Assign a node to a subnet
    pub fn assign_node(&self, request: AssignNodeRequest) -> Result<SubnetAssignment> {
        // Check if node is already assigned
        if let Some(existing) = self.assignments.get(&request.node_id) {
            return Err(Error::NodeAlreadyAssigned(
                request.node_id,
                existing.subnet_id,
            ));
        }

        // Determine target subnet
        let subnet_id = if let Some(id) = request.subnet_id {
            // Manual assignment
            id
        } else if let Some(ref attrs) = request.attributes {
            // Policy-based assignment
            let engine = self.policy_engine.read();
            engine.evaluate(attrs)?
        } else {
            return Err(Error::InvalidArgument(
                "Either subnet_id or attributes must be provided".to_string(),
            ));
        };

        // Verify subnet exists and can accept nodes
        let mut subnet = self
            .subnets
            .get_mut(&subnet_id)
            .ok_or(Error::SubnetNotFound(subnet_id))?;

        if !subnet.can_accept_nodes() {
            return Err(Error::InvalidSubnetState {
                state: format!("{:?}", subnet.status),
            });
        }

        // Allocate IP address
        let ip_allocator = self
            .ip_allocators
            .get(&subnet_id)
            .ok_or(Error::Internal("Missing IP allocator".to_string()))?;

        let assigned_ip = {
            let mut allocator = ip_allocator.write();
            allocator
                .allocate()
                .ok_or(Error::NoAvailableIps(subnet_id))?
        };

        // Create assignment
        let assignment = SubnetAssignment {
            id: Uuid::new_v4(),
            node_id: request.node_id,
            subnet_id,
            assigned_ip,
            wg_public_key: request.wg_public_key,
            assigned_at: Utc::now(),
            assignment_method: request.method,
            policy_id: None,
            is_migration_temp: false,
        };

        // Update subnet node count
        subnet.current_nodes += 1;
        subnet.updated_at = Utc::now();

        // Store assignment
        self.assignments.insert(request.node_id, assignment.clone());

        // Update subnet nodes list
        if let Some(mut nodes) = self.subnet_nodes.get_mut(&subnet_id) {
            nodes.push(request.node_id);
        }

        tracing::info!(
            node_id = %request.node_id,
            subnet_id = %subnet_id,
            assigned_ip = %assigned_ip,
            "Assigned node to subnet"
        );

        Ok(assignment)
    }

    /// Remove a node from its subnet
    pub fn unassign_node(&self, node_id: Uuid) -> Result<()> {
        // Get and remove assignment
        let (_, assignment) = self
            .assignments
            .remove(&node_id)
            .ok_or(Error::NodeNotAssigned(node_id))?;

        // Release IP address
        if let Some(ip_allocator) = self.ip_allocators.get(&assignment.subnet_id) {
            let mut allocator = ip_allocator.write();
            allocator.release(assignment.assigned_ip);
        }

        // Update subnet node count
        if let Some(mut subnet) = self.subnets.get_mut(&assignment.subnet_id) {
            subnet.current_nodes = (subnet.current_nodes - 1).max(0);
            subnet.updated_at = Utc::now();
        }

        // Remove from subnet nodes list
        if let Some(mut nodes) = self.subnet_nodes.get_mut(&assignment.subnet_id) {
            nodes.retain(|&id| id != node_id);
        }

        tracing::info!(
            node_id = %node_id,
            subnet_id = %assignment.subnet_id,
            "Unassigned node from subnet"
        );

        Ok(())
    }

    /// Get a node's assignment
    pub fn get_assignment(&self, node_id: Uuid) -> Option<SubnetAssignment> {
        self.assignments.get(&node_id).map(|a| a.clone())
    }

    /// Get all nodes in a subnet
    pub fn get_subnet_nodes(&self, subnet_id: Uuid) -> Vec<Uuid> {
        self.subnet_nodes
            .get(&subnet_id)
            .map(|nodes| nodes.clone())
            .unwrap_or_default()
    }

    /// Get all assignments in a subnet
    pub fn get_subnet_assignments(&self, subnet_id: Uuid) -> Vec<SubnetAssignment> {
        self.get_subnet_nodes(subnet_id)
            .iter()
            .filter_map(|node_id| self.assignments.get(node_id).map(|a| a.clone()))
            .collect()
    }

    // ==================== Policy Management ====================

    /// Add an assignment policy
    pub fn add_policy(&self, policy: AssignmentPolicy) -> Result<()> {
        // Verify target subnet exists
        if !self.subnets.contains_key(&policy.target_subnet_id) {
            return Err(Error::SubnetNotFound(policy.target_subnet_id));
        }

        let mut engine = self.policy_engine.write();
        engine.add_policy(policy);
        Ok(())
    }

    /// Remove a policy
    pub fn remove_policy(&self, policy_id: Uuid) -> Result<()> {
        let mut engine = self.policy_engine.write();
        engine
            .remove_policy(policy_id)
            .ok_or(Error::PolicyNotFound(policy_id))?;
        Ok(())
    }

    /// Get a policy
    pub fn get_policy(&self, policy_id: Uuid) -> Option<AssignmentPolicy> {
        let engine = self.policy_engine.read();
        engine.get_policy(policy_id).cloned()
    }

    /// List all policies
    pub fn list_policies(&self) -> Vec<AssignmentPolicy> {
        let engine = self.policy_engine.read();
        engine.policies().into_iter().cloned().collect()
    }

    /// Evaluate policies for a node (dry run)
    pub fn evaluate_policies(
        &self,
        attributes: &NodeAttributes,
    ) -> crate::policy_engine::EvaluationResult {
        let engine = self.policy_engine.read();
        engine.evaluate_dry_run(attributes)
    }

    // ==================== Route Management ====================

    /// Create a cross-subnet route
    pub fn create_route(&self, route: CrossSubnetRoute) -> Result<()> {
        // Verify source subnet exists
        if !self.subnets.contains_key(&route.source_subnet_id) {
            return Err(Error::SubnetNotFound(route.source_subnet_id));
        }

        // Verify destination subnet exists
        if !self.subnets.contains_key(&route.destination_subnet_id) {
            return Err(Error::SubnetNotFound(route.destination_subnet_id));
        }

        // Check for self-route
        if route.source_subnet_id == route.destination_subnet_id {
            return Err(Error::SelfRoute);
        }

        // Check for duplicate
        for existing in self.routes.iter() {
            if existing.source_subnet_id == route.source_subnet_id
                && existing.destination_subnet_id == route.destination_subnet_id
            {
                return Err(Error::RouteAlreadyExists(
                    route.source_subnet_id,
                    route.destination_subnet_id,
                ));
            }
        }

        self.routes.insert(route.id, route);
        Ok(())
    }

    /// Delete a route
    pub fn delete_route(&self, route_id: Uuid) -> Result<()> {
        self.routes
            .remove(&route_id)
            .ok_or(Error::MigrationNotFound(route_id))?;
        Ok(())
    }

    /// Get routes for a subnet
    pub fn get_routes_for_subnet(&self, subnet_id: Uuid) -> Vec<CrossSubnetRoute> {
        self.routes
            .iter()
            .filter(|r| r.source_subnet_id == subnet_id || r.destination_subnet_id == subnet_id)
            .map(|r| r.clone())
            .collect()
    }

    // ==================== Template Management ====================

    /// Get a template by ID
    pub fn get_template(&self, template_id: Uuid) -> Option<SubnetTemplate> {
        self.templates.get(&template_id).map(|t| t.clone())
    }

    /// Get a template by name
    pub fn get_template_by_name(&self, name: &str) -> Option<SubnetTemplate> {
        self.templates
            .iter()
            .find(|t| t.name == name)
            .map(|t| t.clone())
    }

    /// List all templates
    pub fn list_templates(&self) -> Vec<SubnetTemplate> {
        self.templates.iter().map(|t| t.clone()).collect()
    }

    /// Create subnet from template
    pub fn create_from_template(
        &self,
        template_id: Uuid,
        name: String,
        overrides: Option<CreateSubnetRequest>,
    ) -> Result<Subnet> {
        let template = self
            .get_template(template_id)
            .ok_or(Error::TemplateNotFound(template_id))?;

        let mut request = overrides.unwrap_or(CreateSubnetRequest {
            name: name.clone(),
            description: template.description.clone(),
            purpose: template.purpose,
            cidr: None,
            prefix_length: Some(template.prefix_length),
            tenant_id: None,
            node_type: template.defaults.node_type,
            region: None,
            resource_pool_id: None,
            max_nodes: template.defaults.max_nodes,
            template_id: Some(template_id),
            created_by: None,
        });

        request.name = name;
        request.template_id = Some(template_id);

        self.create_subnet(request)
    }

    // ==================== Statistics ====================

    /// Get subnet statistics
    pub fn get_stats(&self) -> SubnetManagerStats {
        let mut stats = SubnetManagerStats::default();

        for subnet in self.subnets.iter() {
            stats.total_subnets += 1;
            stats.total_nodes += subnet.current_nodes as usize;

            match subnet.purpose {
                SubnetPurpose::Tenant => stats.tenant_subnets += 1,
                SubnetPurpose::NodeType => stats.node_type_subnets += 1,
                SubnetPurpose::Geographic => stats.geographic_subnets += 1,
                SubnetPurpose::ResourcePool => stats.resource_pool_subnets += 1,
            }
        }

        stats.total_policies = {
            let engine = self.policy_engine.read();
            engine.policies().len()
        };

        stats.total_routes = self.routes.len();

        stats
    }
}

impl Default for SubnetManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for the subnet manager
#[derive(Debug, Clone, Default)]
pub struct SubnetManagerStats {
    pub total_subnets: usize,
    pub tenant_subnets: usize,
    pub node_type_subnets: usize,
    pub geographic_subnets: usize,
    pub resource_pool_subnets: usize,
    pub total_nodes: usize,
    pub total_policies: usize,
    pub total_routes: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_subnet() {
        let manager = SubnetManager::new();

        let request = CreateSubnetRequest {
            name: "Test Subnet".to_string(),
            description: Some("A test subnet".to_string()),
            purpose: SubnetPurpose::Tenant,
            cidr: None,
            prefix_length: Some(24),
            tenant_id: Some(Uuid::new_v4()),
            node_type: None,
            region: None,
            resource_pool_id: None,
            max_nodes: Some(100),
            template_id: None,
            created_by: None,
        };

        let subnet = manager.create_subnet(request).unwrap();

        assert_eq!(subnet.name, "Test Subnet");
        assert_eq!(subnet.purpose, SubnetPurpose::Tenant);
        assert_eq!(subnet.status, SubnetStatus::Active);
        assert!(subnet.can_accept_nodes());
    }

    #[test]
    fn test_assign_node() {
        let manager = SubnetManager::new();

        // Create subnet
        let subnet = manager
            .create_subnet(CreateSubnetRequest {
                name: "Test".to_string(),
                description: None,
                purpose: SubnetPurpose::NodeType,
                cidr: None,
                prefix_length: Some(24),
                tenant_id: None,
                node_type: Some(NodeType::DataCenter),
                region: None,
                resource_pool_id: None,
                max_nodes: None,
                template_id: None,
                created_by: None,
            })
            .unwrap();

        // Assign node
        let node_id = Uuid::new_v4();
        let assignment = manager
            .assign_node(AssignNodeRequest {
                node_id,
                wg_public_key: "test-key".to_string(),
                subnet_id: Some(subnet.id),
                attributes: None,
                method: "manual".to_string(),
            })
            .unwrap();

        assert_eq!(assignment.node_id, node_id);
        assert_eq!(assignment.subnet_id, subnet.id);

        // Verify assignment
        let retrieved = manager.get_assignment(node_id).unwrap();
        assert_eq!(retrieved.assigned_ip, assignment.assigned_ip);
    }

    #[test]
    fn test_policy_based_assignment() {
        let manager = SubnetManager::new();

        // Create subnet
        let subnet = manager
            .create_subnet(CreateSubnetRequest {
                name: "DC Subnet".to_string(),
                description: None,
                purpose: SubnetPurpose::NodeType,
                cidr: None,
                prefix_length: Some(24),
                tenant_id: None,
                node_type: Some(NodeType::DataCenter),
                region: None,
                resource_pool_id: None,
                max_nodes: None,
                template_id: None,
                created_by: None,
            })
            .unwrap();

        // Create policy
        let policy = AssignmentPolicy::new("DC Policy", subnet.id, 100)
            .with_rule(crate::models::PolicyRule::node_type_equals(
                NodeType::DataCenter,
            ));
        manager.add_policy(policy).unwrap();

        // Assign using policy
        let node_id = Uuid::new_v4();
        let attrs = NodeAttributes::new().with_node_type(NodeType::DataCenter);

        let assignment = manager
            .assign_node(AssignNodeRequest {
                node_id,
                wg_public_key: "test-key".to_string(),
                subnet_id: None,
                attributes: Some(attrs),
                method: "policy".to_string(),
            })
            .unwrap();

        assert_eq!(assignment.subnet_id, subnet.id);
    }

    #[test]
    fn test_unassign_node() {
        let manager = SubnetManager::new();

        let subnet = manager
            .create_subnet(CreateSubnetRequest {
                name: "Test".to_string(),
                description: None,
                purpose: SubnetPurpose::Tenant,
                cidr: None,
                prefix_length: Some(24),
                tenant_id: None,
                node_type: None,
                region: None,
                resource_pool_id: None,
                max_nodes: None,
                template_id: None,
                created_by: None,
            })
            .unwrap();

        let node_id = Uuid::new_v4();
        manager
            .assign_node(AssignNodeRequest {
                node_id,
                wg_public_key: "test-key".to_string(),
                subnet_id: Some(subnet.id),
                attributes: None,
                method: "manual".to_string(),
            })
            .unwrap();

        // Unassign
        manager.unassign_node(node_id).unwrap();

        // Verify removed
        assert!(manager.get_assignment(node_id).is_none());
    }

    #[test]
    fn test_delete_empty_subnet() {
        let manager = SubnetManager::new();

        let subnet = manager
            .create_subnet(CreateSubnetRequest {
                name: "Test".to_string(),
                description: None,
                purpose: SubnetPurpose::Tenant,
                cidr: None,
                prefix_length: Some(24),
                tenant_id: None,
                node_type: None,
                region: None,
                resource_pool_id: None,
                max_nodes: None,
                template_id: None,
                created_by: None,
            })
            .unwrap();

        manager.delete_subnet(subnet.id).unwrap();
        assert!(manager.get_subnet(subnet.id).is_none());
    }

    #[test]
    fn test_cannot_delete_non_empty_subnet() {
        let manager = SubnetManager::new();

        let subnet = manager
            .create_subnet(CreateSubnetRequest {
                name: "Test".to_string(),
                description: None,
                purpose: SubnetPurpose::Tenant,
                cidr: None,
                prefix_length: Some(24),
                tenant_id: None,
                node_type: None,
                region: None,
                resource_pool_id: None,
                max_nodes: None,
                template_id: None,
                created_by: None,
            })
            .unwrap();

        manager
            .assign_node(AssignNodeRequest {
                node_id: Uuid::new_v4(),
                wg_public_key: "test-key".to_string(),
                subnet_id: Some(subnet.id),
                attributes: None,
                method: "manual".to_string(),
            })
            .unwrap();

        let result = manager.delete_subnet(subnet.id);
        assert!(matches!(result, Err(Error::SubnetNotEmpty(_))));
    }

    #[test]
    fn test_create_from_template() {
        let manager = SubnetManager::new();

        // Get system template
        let template = manager.get_template_by_name("nodetype-datacenter").unwrap();

        let subnet = manager
            .create_from_template(template.id, "My DC Subnet".to_string(), None)
            .unwrap();

        assert_eq!(subnet.name, "My DC Subnet");
        assert_eq!(subnet.purpose, SubnetPurpose::NodeType);
        assert_eq!(subnet.template_id, Some(template.id));
    }

    #[test]
    fn test_stats() {
        let manager = SubnetManager::new();

        manager
            .create_subnet(CreateSubnetRequest {
                name: "Tenant1".to_string(),
                description: None,
                purpose: SubnetPurpose::Tenant,
                cidr: None,
                prefix_length: Some(24),
                tenant_id: None,
                node_type: None,
                region: None,
                resource_pool_id: None,
                max_nodes: None,
                template_id: None,
                created_by: None,
            })
            .unwrap();

        manager
            .create_subnet(CreateSubnetRequest {
                name: "DC1".to_string(),
                description: None,
                purpose: SubnetPurpose::NodeType,
                cidr: None,
                prefix_length: Some(24),
                tenant_id: None,
                node_type: Some(NodeType::DataCenter),
                region: None,
                resource_pool_id: None,
                max_nodes: None,
                template_id: None,
                created_by: None,
            })
            .unwrap();

        let stats = manager.get_stats();
        assert_eq!(stats.total_subnets, 2);
        assert_eq!(stats.tenant_subnets, 1);
        assert_eq!(stats.node_type_subnets, 1);
    }
}
