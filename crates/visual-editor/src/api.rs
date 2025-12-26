use async_graphql::{Context, EmptySubscription, InputObject, Object, Schema, SimpleObject, ID};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

/// GraphQL schema type
pub type GraphQLSchema = Schema<Query, Mutation, EmptySubscription>;

/// Node types in the topology
#[derive(Debug, Clone, Serialize, Deserialize, SimpleObject)]
pub struct Node {
    pub id: ID,
    pub node_type: String,
    pub name: String,
    pub position: Position,
    pub properties: serde_json::Value,
}

/// Position of a node
#[derive(Debug, Clone, Serialize, Deserialize, SimpleObject, InputObject)]
#[graphql(input_name = "PositionInput")]
pub struct Position {
    pub x: f64,
    pub y: f64,
}

/// Edge between nodes
#[derive(Debug, Clone, Serialize, Deserialize, SimpleObject)]
pub struct Edge {
    pub id: ID,
    pub source: ID,
    pub target: ID,
    pub bandwidth: i32,
    pub latency: Option<f64>,
    pub properties: serde_json::Value,
}

/// Topology representation
#[derive(Debug, Clone, Serialize, Deserialize, SimpleObject)]
pub struct Topology {
    pub id: ID,
    pub name: String,
    pub description: Option<String>,
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
    pub created_at: String,
    pub updated_at: String,
}

/// Input for creating a topology
#[derive(Debug, InputObject)]
pub struct CreateTopologyInput {
    pub name: String,
    pub description: Option<String>,
}

/// Input for adding a node
#[derive(Debug, InputObject)]
pub struct AddNodeInput {
    pub topology_id: ID,
    pub node_type: String,
    pub name: String,
    pub position: Position,
    pub properties: Option<serde_json::Value>,
}

/// Input for adding an edge
#[derive(Debug, InputObject)]
pub struct AddEdgeInput {
    pub topology_id: ID,
    pub source: ID,
    pub target: ID,
    pub bandwidth: i32,
    pub latency: Option<f64>,
    pub properties: Option<serde_json::Value>,
}

/// Storage for topologies (in-memory for now)
#[derive(Default)]
pub struct TopologyStore {
    topologies: Arc<RwLock<Vec<Topology>>>,
}

/// GraphQL Query root
pub struct Query;

#[Object]
impl Query {
    /// Get all topologies
    async fn topologies(&self, ctx: &Context<'_>) -> Vec<Topology> {
        let store = ctx.data::<TopologyStore>().unwrap();
        store.topologies.read().await.clone()
    }

    /// Get a specific topology by ID
    async fn topology(&self, ctx: &Context<'_>, id: ID) -> Option<Topology> {
        let store = ctx.data::<TopologyStore>().unwrap();
        store
            .topologies
            .read()
            .await
            .iter()
            .find(|t| t.id == id)
            .cloned()
    }

    /// Search topologies by name
    async fn search_topologies(&self, ctx: &Context<'_>, query: String) -> Vec<Topology> {
        let store = ctx.data::<TopologyStore>().unwrap();
        store
            .topologies
            .read()
            .await
            .iter()
            .filter(|t| t.name.to_lowercase().contains(&query.to_lowercase()))
            .cloned()
            .collect()
    }
}

/// GraphQL Mutation root
pub struct Mutation;

#[Object]
impl Mutation {
    /// Create a new topology
    async fn create_topology(&self, ctx: &Context<'_>, input: CreateTopologyInput) -> Topology {
        let store = ctx.data::<TopologyStore>().unwrap();
        let now = Utc::now();

        let topology = Topology {
            id: ID::from(Uuid::new_v4().to_string()),
            name: input.name,
            description: input.description,
            nodes: vec![],
            edges: vec![],
            created_at: now.to_rfc3339(),
            updated_at: now.to_rfc3339(),
        };

        store.topologies.write().await.push(topology.clone());
        topology
    }

    /// Add a node to a topology
    async fn add_node(&self, ctx: &Context<'_>, input: AddNodeInput) -> Option<Node> {
        let store = ctx.data::<TopologyStore>().unwrap();
        let mut topologies = store.topologies.write().await;

        if let Some(topology) = topologies.iter_mut().find(|t| t.id == input.topology_id) {
            let node = Node {
                id: ID::from(Uuid::new_v4().to_string()),
                node_type: input.node_type,
                name: input.name,
                position: Position {
                    x: input.position.x,
                    y: input.position.y,
                },
                properties: input.properties.unwrap_or(serde_json::json!({})),
            };

            topology.nodes.push(node.clone());
            topology.updated_at = Utc::now().to_rfc3339();
            Some(node)
        } else {
            None
        }
    }

    /// Add an edge to a topology
    async fn add_edge(&self, ctx: &Context<'_>, input: AddEdgeInput) -> Option<Edge> {
        let store = ctx.data::<TopologyStore>().unwrap();
        let mut topologies = store.topologies.write().await;

        if let Some(topology) = topologies.iter_mut().find(|t| t.id == input.topology_id) {
            // Verify that both nodes exist
            let source_exists = topology.nodes.iter().any(|n| n.id == input.source);
            let target_exists = topology.nodes.iter().any(|n| n.id == input.target);

            if source_exists && target_exists {
                let edge = Edge {
                    id: ID::from(Uuid::new_v4().to_string()),
                    source: input.source,
                    target: input.target,
                    bandwidth: input.bandwidth,
                    latency: input.latency,
                    properties: input.properties.unwrap_or(serde_json::json!({})),
                };

                topology.edges.push(edge.clone());
                topology.updated_at = Utc::now().to_rfc3339();
                Some(edge)
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Delete a topology
    async fn delete_topology(&self, ctx: &Context<'_>, id: ID) -> bool {
        let store = ctx.data::<TopologyStore>().unwrap();
        let mut topologies = store.topologies.write().await;
        let initial_len = topologies.len();
        topologies.retain(|t| t.id != id);
        topologies.len() < initial_len
    }

    /// Remove a node from a topology
    async fn remove_node(&self, ctx: &Context<'_>, topology_id: ID, node_id: ID) -> bool {
        let store = ctx.data::<TopologyStore>().unwrap();
        let mut topologies = store.topologies.write().await;

        if let Some(topology) = topologies.iter_mut().find(|t| t.id == topology_id) {
            let initial_len = topology.nodes.len();
            topology.nodes.retain(|n| n.id != node_id);

            // Also remove edges connected to this node
            topology
                .edges
                .retain(|e| e.source != node_id && e.target != node_id);

            if topology.nodes.len() < initial_len {
                topology.updated_at = Utc::now().to_rfc3339();
                true
            } else {
                false
            }
        } else {
            false
        }
    }
}

/// Create the GraphQL schema
pub fn create_graphql_schema() -> GraphQLSchema {
    let store = TopologyStore::default();

    Schema::build(Query, Mutation, EmptySubscription)
        .data(store)
        .finish()
}
