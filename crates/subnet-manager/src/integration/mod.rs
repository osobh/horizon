//! Platform integration module
//!
//! Provides integration points for cluster-mesh, hpc-channels, and
//! other platform components.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                     SubnetMeshOrchestrator                       │
//! │  ┌─────────────────┐    ┌─────────────────┐   ┌──────────────┐  │
//! │  │ ClusterNodeInfo │───►│ NodeClassMapper │──►│ NodeAttrs    │  │
//! │  │ (from mesh)     │    │                 │   │ (for policy) │  │
//! │  └─────────────────┘    └─────────────────┘   └──────┬───────┘  │
//! │                                                       │          │
//! │                    ┌──────────────────────────────────┘          │
//! │                    ▼                                             │
//! │  ┌─────────────────────┐    ┌─────────────────┐                 │
//! │  │ PolicyEngine        │───►│ SubnetManager   │                 │
//! │  │ evaluate(attrs)     │    │ allocate_ip()   │                 │
//! │  └─────────────────────┘    └────────┬────────┘                 │
//! │                                       │                          │
//! │                    ┌──────────────────┘                          │
//! │                    ▼                                             │
//! │  ┌─────────────────────┐    ┌─────────────────┐                 │
//! │  │ NodeSubnetInfo      │◄──│ EventPublisher  │                 │
//! │  │ (returned to mesh)  │    │ publish event   │                 │
//! │  └─────────────────────┘    └─────────────────┘                 │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Usage
//!
//! ```rust,ignore
//! use subnet_manager::integration::{
//!     SubnetMeshOrchestrator, ClusterNodeInfo, NodeClass, NodeClassMapper,
//! };
//! use std::sync::Arc;
//!
//! // Create orchestrator
//! let orchestrator = SubnetMeshOrchestrator::new(
//!     Arc::clone(&subnet_manager),
//!     Arc::clone(&policy_engine),
//!     Arc::clone(&event_publisher),
//! );
//!
//! // When a node joins
//! let node = ClusterNodeInfo::new(node_id, "gpu-server-01", NodeClass::DataCenter)
//!     .with_wg_key(public_key);
//!
//! let subnet_info = orchestrator.on_node_joined(&node).await?;
//! println!("Node assigned to {} with IP {}", subnet_info.subnet_name, subnet_info.assigned_ip);
//! ```
//!
//! # Subnet Affinity for Job Scheduling
//!
//! ```rust,ignore
//! use subnet_manager::integration::SubnetAffinity;
//!
//! // Require nodes in a specific subnet
//! let affinity = SubnetAffinity::Required(subnet_id);
//!
//! // Prefer certain subnets with weights
//! let affinity = SubnetAffinity::Preferred(vec![
//!     (gpu_subnet_id, 1.0),   // First preference
//!     (cpu_subnet_id, 0.5),   // Second preference
//! ]);
//!
//! // Same subnet as another job's nodes
//! let affinity = SubnetAffinity::SameAs(other_node_id);
//!
//! // Check if a node satisfies affinity
//! if orchestrator.satisfies_affinity(node_id, &affinity) {
//!     // Node can be used for this job
//! }
//! ```

pub mod mesh_orchestrator;
pub mod node_mapper;

use async_trait::async_trait;
use uuid::Uuid;

// Re-export main types
pub use mesh_orchestrator::{
    NodeSubnetInfo, OrchestrationError, OrchestratorConfig, SubnetAffinity,
    SubnetMeshOrchestrator,
};

pub use node_mapper::{
    ClusterNodeInfo, HardwareInfo, NetworkInfo, NodeClass, NodeClassMapper,
};

/// Trait for mesh integration callbacks.
///
/// Implement this trait to integrate cluster-mesh node lifecycle
/// with subnet-manager's automatic assignment.
///
/// # Example
///
/// ```rust,ignore
/// use subnet_manager::integration::{MeshCallbacks, ClusterNodeInfo, NodeSubnetInfo};
///
/// struct MyMeshIntegration {
///     orchestrator: Arc<SubnetMeshOrchestrator>,
/// }
///
/// #[async_trait]
/// impl MeshCallbacks for MyMeshIntegration {
///     async fn on_node_joined(&self, node: &ClusterNodeInfo) -> Result<NodeSubnetInfo, OrchestrationError> {
///         self.orchestrator.on_node_joined(node).await
///     }
///
///     async fn on_node_left(&self, node_id: Uuid) -> Result<(), OrchestrationError> {
///         self.orchestrator.on_node_left(node_id).await
///     }
/// }
/// ```
#[async_trait]
pub trait MeshCallbacks: Send + Sync {
    /// Called when a node joins the mesh.
    ///
    /// Should evaluate policies and assign the node to an appropriate subnet.
    /// Returns the subnet assignment info that can be stored on the node.
    async fn on_node_joined(&self, node: &ClusterNodeInfo) -> Result<NodeSubnetInfo, OrchestrationError>;

    /// Called when a node leaves the mesh.
    ///
    /// Should release the node's subnet assignment and cleanup resources.
    async fn on_node_left(&self, node_id: Uuid) -> Result<(), OrchestrationError>;

    /// Called when a node's attributes change (optional).
    ///
    /// May trigger a subnet migration if policies no longer match.
    /// Default implementation does nothing.
    async fn on_node_updated(&self, _node: &ClusterNodeInfo) -> Result<Option<NodeSubnetInfo>, OrchestrationError> {
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_exports() {
        // Verify all expected types are exported
        let _: fn() -> NodeClassMapper = NodeClassMapper::new;
    }
}
