//! Agent permission and authorization system
//!
//! The PermissionSystem provides role-based access control (RBAC) for agents,
//! managing their capabilities and resource access permissions.

use chrono::{DateTime, Utc};
use dashmap::DashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{debug, info, warn};
use uuid::Uuid;

use crate::{GovernanceError, Result};
use stratoswarm_agent_core::agent::AgentId;

/// Permission system for managing agent permissions and roles
pub struct PermissionSystem {
    roles: DashMap<Uuid, Role>,
    agent_roles: DashMap<AgentId, Vec<Uuid>>,
    agent_permissions: DashMap<AgentId, Vec<Permission>>,
    permission_cache: DashMap<(AgentId, Permission), (bool, DateTime<Utc>)>,
    cache_ttl_seconds: u64,
}

/// Role definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Role {
    pub id: Uuid,
    pub name: String,
    pub description: String,
    pub permissions: Vec<Permission>,
    pub parent_role: Option<Uuid>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Individual permissions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Permission {
    // Resource permissions
    ReadData,
    WriteData,
    DeleteData,

    // System permissions
    CreateAgent,
    ModifyAgent,
    TerminateAgent,

    // Evolution permissions
    RequestEvolution,
    ApproveEvolution,

    // Coordination permissions
    Coordinate,
    ShareResources,

    // Administrative permissions
    ManageRoles,
    ManagePermissions,
    ViewAuditLog,

    // GPU permissions
    UseGPU,
    AllocateGPU,

    // Network permissions
    NetworkAccess,
    ExternalAPIAccess,
}

impl Permission {
    /// Get all available permissions
    pub fn all() -> Vec<Permission> {
        vec![
            Permission::ReadData,
            Permission::WriteData,
            Permission::DeleteData,
            Permission::CreateAgent,
            Permission::ModifyAgent,
            Permission::TerminateAgent,
            Permission::RequestEvolution,
            Permission::ApproveEvolution,
            Permission::Coordinate,
            Permission::ShareResources,
            Permission::ManageRoles,
            Permission::ManagePermissions,
            Permission::ViewAuditLog,
            Permission::UseGPU,
            Permission::AllocateGPU,
            Permission::NetworkAccess,
            Permission::ExternalAPIAccess,
        ]
    }

    /// Check if this is an administrative permission
    pub fn is_admin(&self) -> bool {
        matches!(
            self,
            Permission::ManageRoles
                | Permission::ManagePermissions
                | Permission::ApproveEvolution
                | Permission::TerminateAgent
        )
    }
}

impl PermissionSystem {
    /// Create a new permission system
    pub fn new() -> Self {
        let mut system = Self {
            roles: DashMap::new(),
            agent_roles: DashMap::new(),
            agent_permissions: DashMap::new(),
            permission_cache: DashMap::new(),
            cache_ttl_seconds: 300, // 5 minute cache
        };

        // Initialize default roles
        system.initialize_default_roles();

        system
    }

    /// Initialize default roles
    fn initialize_default_roles(&mut self) {
        // Admin role
        let admin_role = Role {
            id: Uuid::new_v4(),
            name: "admin".to_string(),
            description: "Full system administrator".to_string(),
            permissions: Permission::all(),
            parent_role: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };
        self.roles.insert(admin_role.id, admin_role);

        // Basic agent role
        let basic_role = Role {
            id: Uuid::new_v4(),
            name: "basic_agent".to_string(),
            description: "Basic agent with limited permissions".to_string(),
            permissions: vec![
                Permission::ReadData,
                Permission::RequestEvolution,
                Permission::Coordinate,
            ],
            parent_role: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };
        self.roles.insert(basic_role.id, basic_role);

        // Worker agent role
        let worker_role = Role {
            id: Uuid::new_v4(),
            name: "worker_agent".to_string(),
            description: "Worker agent with resource access".to_string(),
            permissions: vec![
                Permission::ReadData,
                Permission::WriteData,
                Permission::UseGPU,
                Permission::NetworkAccess,
            ],
            parent_role: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };
        self.roles.insert(worker_role.id, worker_role);
    }

    /// Register a new agent in the permission system
    pub async fn register_agent(&self, agent_id: &AgentId) -> Result<()> {
        info!("Registering agent in permission system: {:?}", agent_id);

        // Assign default basic role
        let basic_role = self
            .roles
            .iter()
            .find(|entry| entry.value().name == "basic_agent")
            .map(|entry| entry.key().clone())
            .ok_or_else(|| GovernanceError::InternalError("Basic role not found".to_string()))?;

        self.agent_roles.insert(agent_id.clone(), vec![basic_role]);
        self.agent_permissions.insert(agent_id.clone(), vec![]);

        Ok(())
    }

    /// Check if an agent has a specific permission
    pub async fn check_permission(
        &self,
        agent_id: &AgentId,
        permission: &Permission,
    ) -> Result<bool> {
        debug!(
            "Checking permission {:?} for agent {:?}",
            permission, agent_id
        );

        // Check cache first
        if let Some((cached_result, cached_at)) =
            self.permission_cache.get(&(agent_id.clone(), *permission))
        {
            if cached_at.timestamp() + self.cache_ttl_seconds as i64 > Utc::now().timestamp() {
                return Ok(*cached_result);
            }
        }

        // Check direct permissions
        if let Some(permissions) = self.agent_permissions.get(agent_id) {
            if permissions.contains(permission) {
                self.cache_permission(agent_id, permission, true);
                return Ok(true);
            }
        }

        // Check role permissions
        if let Some(role_ids) = self.agent_roles.get(agent_id) {
            for role_id in role_ids.iter() {
                if self.role_has_permission(role_id, permission)? {
                    self.cache_permission(agent_id, permission, true);
                    return Ok(true);
                }
            }
        }

        self.cache_permission(agent_id, permission, false);
        Ok(false)
    }

    /// Check if a role has a specific permission (including inherited)
    fn role_has_permission(&self, role_id: &Uuid, permission: &Permission) -> Result<bool> {
        let role = self
            .roles
            .get(role_id)
            .ok_or_else(|| GovernanceError::InternalError("Role not found".to_string()))?;

        // Check direct permissions
        if role.permissions.contains(permission) {
            return Ok(true);
        }

        // Check parent role permissions
        if let Some(parent_id) = role.parent_role {
            return self.role_has_permission(&parent_id, permission);
        }

        Ok(false)
    }

    /// Cache a permission check result
    fn cache_permission(&self, agent_id: &AgentId, permission: &Permission, result: bool) {
        self.permission_cache
            .insert((agent_id.clone(), *permission), (result, Utc::now()));
    }

    /// Grant a permission to an agent
    pub async fn grant_permission(&self, agent_id: &AgentId, permission: Permission) -> Result<()> {
        info!(
            "Granting permission {:?} to agent {:?}",
            permission, agent_id
        );

        let mut permissions = self
            .agent_permissions
            .entry(agent_id.clone())
            .or_insert_with(Vec::new);

        if !permissions.contains(&permission) {
            permissions.push(permission);
        }

        // Invalidate cache
        self.permission_cache
            .remove(&(agent_id.clone(), permission));

        Ok(())
    }

    /// Revoke a permission from an agent
    pub async fn revoke_permission(
        &self,
        agent_id: &AgentId,
        permission: Permission,
    ) -> Result<()> {
        info!(
            "Revoking permission {:?} from agent {:?}",
            permission, agent_id
        );

        if let Some(mut permissions) = self.agent_permissions.get_mut(agent_id) {
            permissions.retain(|&p| p != permission);
        }

        // Invalidate cache
        self.permission_cache
            .remove(&(agent_id.clone(), permission));

        Ok(())
    }

    /// Assign a role to an agent
    pub async fn assign_role(&self, agent_id: &AgentId, role_id: Uuid) -> Result<()> {
        info!("Assigning role {:?} to agent {:?}", role_id, agent_id);

        // Verify role exists
        if !self.roles.contains_key(&role_id) {
            return Err(GovernanceError::InternalError("Role not found".to_string()));
        }

        let mut roles = self
            .agent_roles
            .entry(agent_id.clone())
            .or_insert_with(Vec::new);

        if !roles.contains(&role_id) {
            roles.push(role_id);
        }

        // Invalidate all cached permissions for this agent
        self.invalidate_agent_cache(agent_id);

        Ok(())
    }

    /// Remove a role from an agent
    pub async fn remove_role(&self, agent_id: &AgentId, role_id: Uuid) -> Result<()> {
        info!("Removing role {:?} from agent {:?}", role_id, agent_id);

        if let Some(mut roles) = self.agent_roles.get_mut(agent_id) {
            roles.retain(|&r| r != role_id);
        }

        // Invalidate all cached permissions for this agent
        self.invalidate_agent_cache(agent_id);

        Ok(())
    }

    /// Create a new role
    pub async fn create_role(&self, role: Role) -> Result<Uuid> {
        info!("Creating new role: {}", role.name);

        let id = if role.id == Uuid::nil() {
            Uuid::new_v4()
        } else {
            role.id
        };

        let mut new_role = role;
        new_role.id = id;
        new_role.created_at = Utc::now();
        new_role.updated_at = Utc::now();

        self.roles.insert(id, new_role);

        Ok(id)
    }

    /// Update a role
    pub async fn update_role(&self, role_id: Uuid, updates: RoleUpdate) -> Result<()> {
        let mut role = self
            .roles
            .get_mut(&role_id)
            .ok_or_else(|| GovernanceError::InternalError("Role not found".to_string()))?;

        if let Some(name) = updates.name {
            role.name = name;
        }
        if let Some(description) = updates.description {
            role.description = description;
        }
        if let Some(permissions) = updates.permissions {
            role.permissions = permissions;
        }
        if let Some(parent_role) = updates.parent_role {
            role.parent_role = Some(parent_role);
        }

        role.updated_at = Utc::now();

        // Invalidate cache for all agents with this role
        for entry in self.agent_roles.iter() {
            if entry.value().contains(&role_id) {
                self.invalidate_agent_cache(entry.key());
            }
        }

        Ok(())
    }

    /// Delete a role
    pub async fn delete_role(&self, role_id: Uuid) -> Result<()> {
        // Prevent deletion of default roles
        if let Some(role) = self.roles.get(&role_id) {
            if role.name == "admin" || role.name == "basic_agent" {
                return Err(GovernanceError::PermissionDenied(
                    "Cannot delete default roles".to_string(),
                ));
            }
        }

        self.roles.remove(&role_id);

        // Remove role from all agents
        for mut entry in self.agent_roles.iter_mut() {
            entry.value_mut().retain(|&r| r != role_id);
        }

        Ok(())
    }

    /// Check if a permission can be granted to an agent
    pub async fn can_grant_permission(
        &self,
        agent_id: &AgentId,
        permission: &Permission,
    ) -> Result<bool> {
        // Check if the agent already has the permission
        if self.check_permission(agent_id, permission).await? {
            return Ok(false); // Already has it
        }

        // Check if it's an admin permission and agent has admin role
        if permission.is_admin() {
            let has_admin_role = self
                .agent_roles
                .get(agent_id)
                .map(|roles| {
                    roles.iter().any(|role_id| {
                        self.roles
                            .get(role_id)
                            .map(|role| role.name == "admin")
                            .unwrap_or(false)
                    })
                })
                .unwrap_or(false);

            if !has_admin_role {
                return Ok(false);
            }
        }

        Ok(true)
    }

    /// Get all permissions for an agent
    pub async fn get_agent_permissions(&self, agent_id: &AgentId) -> Result<Vec<Permission>> {
        let mut all_permissions = Vec::new();

        // Direct permissions
        if let Some(permissions) = self.agent_permissions.get(agent_id) {
            all_permissions.extend(permissions.iter().cloned());
        }

        // Role permissions
        if let Some(role_ids) = self.agent_roles.get(agent_id) {
            for role_id in role_ids.iter() {
                if let Some(role) = self.roles.get(role_id) {
                    all_permissions.extend(role.permissions.iter().cloned());
                }
            }
        }

        // Remove duplicates
        all_permissions.sort();
        all_permissions.dedup();

        Ok(all_permissions)
    }

    /// Get all roles for an agent
    pub async fn get_agent_roles(&self, agent_id: &AgentId) -> Result<Vec<Role>> {
        let role_ids = self
            .agent_roles
            .get(agent_id)
            .map(|entry| entry.clone())
            .unwrap_or_default();

        let mut roles = Vec::new();
        for role_id in role_ids {
            if let Some(role) = self.roles.get(&role_id) {
                roles.push(role.clone());
            }
        }

        Ok(roles)
    }

    /// Invalidate all cached permissions for an agent
    fn invalidate_agent_cache(&self, agent_id: &AgentId) {
        let keys_to_remove: Vec<_> = self
            .permission_cache
            .iter()
            .filter(|entry| entry.key().0 == *agent_id)
            .map(|entry| entry.key().clone())
            .collect();

        for key in keys_to_remove {
            self.permission_cache.remove(&key);
        }
    }

    /// Get a role by name
    pub async fn get_role_by_name(&self, name: &str) -> Result<Role> {
        self.roles
            .iter()
            .find(|entry| entry.value().name == name)
            .map(|entry| entry.value().clone())
            .ok_or_else(|| GovernanceError::InternalError(format!("Role '{}' not found", name)))
    }
}

/// Role update structure
#[derive(Debug, Clone)]
pub struct RoleUpdate {
    pub name: Option<String>,
    pub description: Option<String>,
    pub permissions: Option<Vec<Permission>>,
    pub parent_role: Option<Uuid>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::test;

    #[test]
    async fn test_permission_system_creation() {
        let system = PermissionSystem::new();

        // Should have default roles
        assert!(system.roles.len() >= 3); // admin, basic_agent, worker_agent
    }

    #[test]
    async fn test_agent_registration() {
        let system = PermissionSystem::new();
        let agent_id = AgentId::new();

        system.register_agent(&agent_id).await?;

        // Should have basic role assigned
        let roles = system.get_agent_roles(&agent_id).await?;
        assert_eq!(roles.len(), 1);
        assert_eq!(roles[0].name, "basic_agent");
    }

    #[test]
    async fn test_permission_check_basic() {
        let system = PermissionSystem::new();
        let agent_id = AgentId::new();
        system.register_agent(&agent_id).await?;

        // Basic agent should have ReadData permission
        assert!(system
            .check_permission(&agent_id, &Permission::ReadData)
            .await
            ?);

        // Basic agent should not have WriteData permission
        assert!(!system
            .check_permission(&agent_id, &Permission::WriteData)
            .await
            .unwrap());
    }

    #[test]
    async fn test_grant_permission() {
        let system = PermissionSystem::new();
        let agent_id = AgentId::new();
        system.register_agent(&agent_id).await?;

        // Grant WriteData permission
        system
            .grant_permission(&agent_id, Permission::WriteData)
            .await
            ?;

        // Check permission
        assert!(system
            .check_permission(&agent_id, &Permission::WriteData)
            .await
            .unwrap());
    }

    #[test]
    async fn test_revoke_permission() {
        let system = PermissionSystem::new();
        let agent_id = AgentId::new();
        system.register_agent(&agent_id).await?;

        // Grant and then revoke permission
        system
            .grant_permission(&agent_id, Permission::WriteData)
            .await
            ?;
        assert!(system
            .check_permission(&agent_id, &Permission::WriteData)
            .await
            .unwrap());

        system
            .revoke_permission(&agent_id, Permission::WriteData)
            .await
            .unwrap();
        assert!(!system
            .check_permission(&agent_id, &Permission::WriteData)
            .await
            .unwrap());
    }

    #[test]
    async fn test_role_assignment() {
        let system = PermissionSystem::new();
        let agent_id = AgentId::new();
        system.register_agent(&agent_id).await?;

        // Get worker role
        let worker_role = system.get_role_by_name("worker_agent").await?;

        // Assign worker role
        system.assign_role(&agent_id, worker_role.id).await?;

        // Should now have worker permissions
        assert!(system
            .check_permission(&agent_id, &Permission::UseGPU)
            .await
            .unwrap());
        assert!(system
            .check_permission(&agent_id, &Permission::NetworkAccess)
            .await
            .unwrap());
    }

    #[test]
    async fn test_role_removal() {
        let system = PermissionSystem::new();
        let agent_id = AgentId::new();
        system.register_agent(&agent_id).await?;

        let worker_role = system.get_role_by_name("worker_agent").await?;
        system.assign_role(&agent_id, worker_role.id).await?;

        // Remove the role
        system.remove_role(&agent_id, worker_role.id).await?;

        // Should no longer have worker permissions
        assert!(!system
            .check_permission(&agent_id, &Permission::UseGPU)
            .await
            .unwrap());
    }

    #[test]
    async fn test_create_custom_role() {
        let system = PermissionSystem::new();

        let custom_role = Role {
            id: Uuid::nil(),
            name: "custom_role".to_string(),
            description: "A custom role".to_string(),
            permissions: vec![Permission::ReadData, Permission::Coordinate],
            parent_role: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };

        let role_id = system.create_role(custom_role).await?;
        assert_ne!(role_id, Uuid::nil());

        // Verify role was created
        let roles: Vec<_> = system.roles.iter().map(|e| e.value().clone()).collect();
        assert!(roles.iter().any(|r| r.name == "custom_role"));
    }

    #[test]
    async fn test_update_role() {
        let system = PermissionSystem::new();

        let role = Role {
            id: Uuid::nil(),
            name: "test_role".to_string(),
            description: "Test role".to_string(),
            permissions: vec![Permission::ReadData],
            parent_role: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };

        let role_id = system.create_role(role).await?;

        // Update the role
        let update = RoleUpdate {
            name: Some("updated_role".to_string()),
            description: None,
            permissions: Some(vec![Permission::ReadData, Permission::WriteData]),
            parent_role: None,
        };

        system.update_role(role_id, update).await?;

        // Verify updates
        let updated_role = system.roles.get(&role_id)?;
        assert_eq!(updated_role.name, "updated_role");
        assert_eq!(updated_role.permissions.len(), 2);
    }

    #[test]
    async fn test_delete_role() {
        let system = PermissionSystem::new();

        let role = Role {
            id: Uuid::nil(),
            name: "deletable_role".to_string(),
            description: "A role that can be deleted".to_string(),
            permissions: vec![],
            parent_role: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };

        let role_id = system.create_role(role).await?;
        system.delete_role(role_id).await?;

        // Verify deletion
        assert!(!system.roles.contains_key(&role_id));
    }

    #[test]
    async fn test_cannot_delete_default_roles() {
        let system = PermissionSystem::new();

        let admin_role = system.get_role_by_name("admin").await?;
        let result = system.delete_role(admin_role.id).await;

        assert!(matches!(result, Err(GovernanceError::PermissionDenied(_))));
    }

    #[test]
    async fn test_permission_caching() {
        let system = PermissionSystem::new();
        let agent_id = AgentId::new();
        system.register_agent(&agent_id).await?;

        // First check should miss cache
        assert!(system
            .check_permission(&agent_id, &Permission::ReadData)
            .await
            ?);

        // Second check should hit cache
        assert_eq!(system.permission_cache.len(), 1);
        assert!(system
            .check_permission(&agent_id, &Permission::ReadData)
            .await
            .unwrap());
    }

    #[test]
    async fn test_cache_invalidation_on_permission_change() {
        let system = PermissionSystem::new();
        let agent_id = AgentId::new();
        system.register_agent(&agent_id).await?;

        // Cache a permission check
        system
            .check_permission(&agent_id, &Permission::WriteData)
            .await
            ?;
        assert_eq!(system.permission_cache.len(), 1);

        // Grant the permission
        system
            .grant_permission(&agent_id, Permission::WriteData)
            .await
            .unwrap();

        // Cache should be invalidated
        assert!(!system
            .permission_cache
            .contains_key(&(agent_id.clone(), Permission::WriteData)));
    }

    #[test]
    async fn test_admin_permission_check() {
        let system = PermissionSystem::new();

        assert!(Permission::ManageRoles.is_admin());
        assert!(Permission::TerminateAgent.is_admin());
        assert!(!Permission::ReadData.is_admin());
    }

    #[test]
    async fn test_can_grant_permission() {
        let system = PermissionSystem::new();
        let agent_id = AgentId::new();
        system.register_agent(&agent_id).await?;

        // Should be able to grant non-admin permissions
        assert!(system
            .can_grant_permission(&agent_id, &Permission::WriteData)
            .await
            ?);

        // Should not be able to grant admin permissions without admin role
        assert!(!system
            .can_grant_permission(&agent_id, &Permission::ManageRoles)
            .await
            .unwrap());
    }

    #[test]
    async fn test_get_all_agent_permissions() {
        let system = PermissionSystem::new();
        let agent_id = AgentId::new();
        system.register_agent(&agent_id).await?;

        // Grant additional permission
        system
            .grant_permission(&agent_id, Permission::WriteData)
            .await
            ?;

        let permissions = system.get_agent_permissions(&agent_id).await?;

        // Should have basic role permissions + granted permission
        assert!(permissions.contains(&Permission::ReadData)); // From basic role
        assert!(permissions.contains(&Permission::WriteData)); // Granted directly
    }

    #[test]
    async fn test_role_inheritance() {
        let system = PermissionSystem::new();

        // Create parent role
        let parent_role = Role {
            id: Uuid::new_v4(),
            name: "parent_role".to_string(),
            description: "Parent role".to_string(),
            permissions: vec![Permission::ReadData],
            parent_role: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };
        let parent_id = system.create_role(parent_role).await?;

        // Create child role
        let child_role = Role {
            id: Uuid::new_v4(),
            name: "child_role".to_string(),
            description: "Child role".to_string(),
            permissions: vec![Permission::WriteData],
            parent_role: Some(parent_id),
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };
        let child_id = system.create_role(child_role).await?;

        // Assign child role to agent
        let agent_id = AgentId::new();
        system.register_agent(&agent_id).await?;
        system.assign_role(&agent_id, child_id).await?;

        // Should have both parent and child permissions
        assert!(system
            .check_permission(&agent_id, &Permission::ReadData)
            .await
            .unwrap());
        assert!(system
            .check_permission(&agent_id, &Permission::WriteData)
            .await
            .unwrap());
    }

    #[test]
    async fn test_multiple_roles() {
        let system = PermissionSystem::new();
        let agent_id = AgentId::new();
        system.register_agent(&agent_id).await?;

        // Assign multiple roles
        let worker_role = system.get_role_by_name("worker_agent").await?;
        system.assign_role(&agent_id, worker_role.id).await?;

        // Should have permissions from both roles
        let permissions = system.get_agent_permissions(&agent_id).await?;
        assert!(permissions.contains(&Permission::ReadData)); // From basic
        assert!(permissions.contains(&Permission::UseGPU)); // From worker
        assert!(permissions.contains(&Permission::RequestEvolution)); // From basic
        assert!(permissions.contains(&Permission::NetworkAccess)); // From worker
    }

    #[test]
    async fn test_permission_deduplication() {
        let system = PermissionSystem::new();
        let agent_id = AgentId::new();
        system.register_agent(&agent_id).await?;

        // Grant a permission that's already in the role
        system
            .grant_permission(&agent_id, Permission::ReadData)
            .await
            ?;

        let permissions = system.get_agent_permissions(&agent_id).await?;

        // Should only appear once
        let read_count = permissions
            .iter()
            .filter(|&&p| p == Permission::ReadData)
            .count();
        assert_eq!(read_count, 1);
    }
}
