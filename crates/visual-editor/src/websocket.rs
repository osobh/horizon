use axum::extract::ws::{Message, WebSocket};
use futures::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{broadcast, RwLock};
use uuid::Uuid;

// Use these for future error handling
#[allow(unused_imports)]
use crate::error::{Result, VisualEditorError};

/// WebSocket message types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum WebSocketMessage {
    /// Client connected
    Connected { client_id: String },

    /// Client disconnected
    Disconnected { client_id: String },

    /// Topology update
    TopologyUpdate {
        topology_id: String,
        data: serde_json::Value,
    },

    /// Node added
    NodeAdded {
        topology_id: String,
        node: serde_json::Value,
    },

    /// Node removed
    NodeRemoved {
        topology_id: String,
        node_id: String,
    },

    /// Edge added
    EdgeAdded {
        topology_id: String,
        edge: serde_json::Value,
    },

    /// Edge removed
    EdgeRemoved {
        topology_id: String,
        edge_id: String,
    },

    /// Cursor position update
    CursorUpdate { client_id: String, x: f64, y: f64 },

    /// Selection update
    SelectionUpdate {
        client_id: String,
        selected_ids: Vec<String>,
    },

    /// Error message
    Error { message: String },

    // =============================================================================
    // Real-time Infrastructure Intelligence Messages
    // =============================================================================

    /// Real-time GPU utilization metrics
    GpuMetrics {
        timestamp: String,
        node_id: String,
        gpu_id: u32,
        utilization_percent: f64,
        memory_used_gb: f64,
        memory_total_gb: f64,
        temperature_celsius: f64,
        power_watts: f64,
        clock_mhz: u32,
    },

    /// Swarmlet status update
    SwarmletStatusUpdate {
        swarmlet_id: String,
        status: String, // joining, active, draining, offline, error
        cpu_utilization: f64,
        memory_utilization: f64,
        gpu_utilization: f64,
        container_count: u32,
        last_heartbeat: String,
    },

    /// Container deployment status update
    DeploymentStatusUpdate {
        deployment_id: String,
        status: String, // pending, starting, running, scaling, stopping, stopped, failed
        running_replicas: u32,
        desired_replicas: u32,
        health_status: String,
        resource_usage: serde_json::Value,
    },

    /// Real-time deployment logs
    DeploymentLogs {
        deployment_id: String,
        container_id: String,
        timestamp: String,
        level: String, // info, warn, error, debug
        message: String,
        source: String, // stdout, stderr
    },

    /// System alert notification
    SystemAlert {
        alert_id: String,
        severity: String, // info, warning, critical, emergency
        category: String, // resource, security, cost, health
        title: String,
        message: String,
        timestamp: String,
        affected_resources: Vec<String>,
        actions: Vec<String>,
    },

    /// Cost tracking update
    CostUpdate {
        timestamp: String,
        hourly_cost: f64,
        daily_cost: f64,
        monthly_projection: f64,
        budget_utilization_percent: f64,
        cost_breakdown: serde_json::Value,
    },

    /// Security compliance update
    SecurityUpdate {
        timestamp: String,
        overall_score: f64,
        critical_vulnerabilities: u32,
        high_vulnerabilities: u32,
        policy_violations: u32,
        recommendations: Vec<String>,
    },

    /// Pipeline execution update
    PipelineExecutionUpdate {
        pipeline_id: String,
        execution_id: String,
        status: String, // running, success, failed, cancelled
        current_stage: Option<String>,
        completed_stages: Vec<String>,
        failed_stages: Vec<String>,
        estimated_completion: Option<String>,
    },

    /// Resource allocation change
    ResourceAllocationUpdate {
        timestamp: String,
        node_id: String,
        resource_type: String, // cpu, memory, gpu, storage, network
        total: f64,
        allocated: f64,
        utilization_percent: f64,
        available: f64,
    },

    /// Network topology change
    TopologyChangeNotification {
        change_type: String, // node_added, node_removed, connection_added, connection_removed, status_changed
        affected_node_ids: Vec<String>,
        connection_id: Option<String>,
        details: serde_json::Value,
        timestamp: String,
    },

    /// Performance metrics update
    PerformanceMetrics {
        timestamp: String,
        node_id: String,
        metrics: serde_json::Value, // CPU, memory, network, storage metrics
        alerts: Vec<String>,
        recommendations: Vec<String>,
    },

    /// Capacity planning alert
    CapacityAlert {
        alert_id: String,
        resource_type: String,
        current_utilization: f64,
        projected_exhaustion: Option<String>,
        recommended_actions: Vec<String>,
        severity: String,
        timestamp: String,
    },

    // =============================================================================
    // Agent XP and Evolution Messages
    // =============================================================================

    /// Agent XP gained notification
    AgentXPGained {
        agent_id: String,
        xp_amount: u64,
        reason: String,
        category: String,
        new_total_xp: u64,
        new_current_xp: u64,
        new_level: u32,
        leveled_up: bool,
        ready_to_evolve: bool,
        timestamp: String,
    },

    /// Agent evolution completed notification
    AgentEvolved {
        agent_id: String,
        previous_level: u32,
        new_level: u32,
        xp_at_evolution: u64,
        evolution_timestamp: String,
    },

    /// System XP statistics update
    SystemXPUpdate {
        total_agents: u32,
        total_xp_awarded: u64,
        average_level: f64,
        highest_level: u32,
        agents_ready_to_evolve: u32,
        xp_gained_24h: u64,
        timestamp: String,
    },
}

/// WebSocket connection handler
pub struct WebSocketHandler {
    /// Active client connections
    clients: Arc<RwLock<HashMap<String, broadcast::Sender<WebSocketMessage>>>>,
    /// Broadcast channel for messages
    broadcast_tx: broadcast::Sender<WebSocketMessage>,
}

impl WebSocketHandler {
    /// Create a new WebSocket handler
    pub fn new() -> Self {
        let (broadcast_tx, _) = broadcast::channel(1024);

        Self {
            clients: Arc::new(RwLock::new(HashMap::new())),
            broadcast_tx,
        }
    }

    /// Add a new client connection
    pub async fn add_client(&self) -> String {
        let client_id = Uuid::new_v4().to_string();
        let mut clients = self.clients.write().await;
        clients.insert(client_id.clone(), self.broadcast_tx.clone());
        client_id
    }

    /// Remove a client connection
    pub async fn remove_client(&self, client_id: &str) {
        let mut clients = self.clients.write().await;
        clients.remove(client_id);

        // Broadcast disconnection
        let msg = WebSocketMessage::Disconnected {
            client_id: client_id.to_string(),
        };
        let _ = self.broadcast_tx.send(msg);
    }

    /// Broadcast a message to all connected clients
    pub async fn broadcast(&self, message: WebSocketMessage) {
        let _ = self.broadcast_tx.send(message);
    }

    /// Handle a WebSocket connection
    pub async fn handle_connection(&self, ws: WebSocket) {
        let (mut sender, mut receiver) = ws.split();
        let client_id = self.add_client().await;

        // Subscribe to broadcast messages
        let mut broadcast_rx = self.broadcast_tx.subscribe();

        // Send connection message
        let connect_msg = WebSocketMessage::Connected {
            client_id: client_id.clone(),
        };
        if let Ok(msg) = serde_json::to_string(&connect_msg) {
            let _ = sender.send(Message::Text(msg)).await;
        }

        // Broadcast connection to other clients
        self.broadcast(connect_msg).await;

        let handler = self.clone();
        let client_id_clone = client_id.clone();

        // Spawn task to handle incoming messages
        let mut recv_task = tokio::spawn(async move {
            while let Some(Ok(msg)) = receiver.next().await {
                if let Message::Text(text) = msg {
                    if let Ok(ws_msg) = serde_json::from_str::<WebSocketMessage>(&text) {
                        handler.handle_message(&client_id_clone, ws_msg).await;
                    }
                }
            }
        });

        // Spawn task to handle outgoing messages
        let mut send_task = tokio::spawn(async move {
            while let Ok(msg) = broadcast_rx.recv().await {
                if let Ok(text) = serde_json::to_string(&msg) {
                    if sender.send(Message::Text(text)).await.is_err() {
                        break;
                    }
                }
            }
        });

        // Wait for either task to complete
        tokio::select! {
            _ = &mut recv_task => {},
            _ = &mut send_task => {},
        }

        // Clean up
        self.remove_client(&client_id).await;
        recv_task.abort();
        send_task.abort();
    }

    /// Handle an incoming WebSocket message
    async fn handle_message(&self, _client_id: &str, message: WebSocketMessage) {
        match message {
            WebSocketMessage::TopologyUpdate { .. }
            | WebSocketMessage::NodeAdded { .. }
            | WebSocketMessage::NodeRemoved { .. }
            | WebSocketMessage::EdgeAdded { .. }
            | WebSocketMessage::EdgeRemoved { .. }
            | WebSocketMessage::CursorUpdate { .. }
            | WebSocketMessage::SelectionUpdate { .. }
            // Real-time infrastructure intelligence messages
            | WebSocketMessage::GpuMetrics { .. }
            | WebSocketMessage::SwarmletStatusUpdate { .. }
            | WebSocketMessage::DeploymentStatusUpdate { .. }
            | WebSocketMessage::DeploymentLogs { .. }
            | WebSocketMessage::SystemAlert { .. }
            | WebSocketMessage::CostUpdate { .. }
            | WebSocketMessage::SecurityUpdate { .. }
            | WebSocketMessage::PipelineExecutionUpdate { .. }
            | WebSocketMessage::ResourceAllocationUpdate { .. }
            | WebSocketMessage::TopologyChangeNotification { .. }
            | WebSocketMessage::PerformanceMetrics { .. }
            | WebSocketMessage::CapacityAlert { .. }
            // Agent XP and evolution messages
            | WebSocketMessage::AgentXPGained { .. }
            | WebSocketMessage::AgentEvolved { .. }
            | WebSocketMessage::SystemXPUpdate { .. } => {
                // Broadcast to all clients
                self.broadcast(message).await;
            }
            _ => {
                // Handle other message types if needed
            }
        }
    }

    /// Broadcast GPU metrics to all connected clients
    pub async fn broadcast_gpu_metrics(
        &self,
        node_id: String,
        gpu_id: u32,
        utilization_percent: f64,
        memory_used_gb: f64,
        memory_total_gb: f64,
        temperature_celsius: f64,
        power_watts: f64,
        clock_mhz: u32,
    ) {
        let message = WebSocketMessage::GpuMetrics {
            timestamp: chrono::Utc::now().to_rfc3339(),
            node_id,
            gpu_id,
            utilization_percent,
            memory_used_gb,
            memory_total_gb,
            temperature_celsius,
            power_watts,
            clock_mhz,
        };
        self.broadcast(message).await;
    }

    /// Broadcast swarmlet status update
    pub async fn broadcast_swarmlet_status(
        &self,
        swarmlet_id: String,
        status: String,
        cpu_utilization: f64,
        memory_utilization: f64,
        gpu_utilization: f64,
        container_count: u32,
    ) {
        let message = WebSocketMessage::SwarmletStatusUpdate {
            swarmlet_id,
            status,
            cpu_utilization,
            memory_utilization,
            gpu_utilization,
            container_count,
            last_heartbeat: chrono::Utc::now().to_rfc3339(),
        };
        self.broadcast(message).await;
    }

    /// Broadcast deployment status update
    pub async fn broadcast_deployment_status(
        &self,
        deployment_id: String,
        status: String,
        running_replicas: u32,
        desired_replicas: u32,
        health_status: String,
        resource_usage: serde_json::Value,
    ) {
        let message = WebSocketMessage::DeploymentStatusUpdate {
            deployment_id,
            status,
            running_replicas,
            desired_replicas,
            health_status,
            resource_usage,
        };
        self.broadcast(message).await;
    }

    /// Broadcast deployment logs
    pub async fn broadcast_deployment_logs(
        &self,
        deployment_id: String,
        container_id: String,
        level: String,
        message_text: String,
        source: String,
    ) {
        let message = WebSocketMessage::DeploymentLogs {
            deployment_id,
            container_id,
            timestamp: chrono::Utc::now().to_rfc3339(),
            level,
            message: message_text,
            source,
        };
        self.broadcast(message).await;
    }

    /// Broadcast system alert
    pub async fn broadcast_system_alert(
        &self,
        alert_id: String,
        severity: String,
        category: String,
        title: String,
        message_text: String,
        affected_resources: Vec<String>,
        actions: Vec<String>,
    ) {
        let message = WebSocketMessage::SystemAlert {
            alert_id,
            severity,
            category,
            title,
            message: message_text,
            timestamp: chrono::Utc::now().to_rfc3339(),
            affected_resources,
            actions,
        };
        self.broadcast(message).await;
    }

    /// Broadcast cost update
    pub async fn broadcast_cost_update(
        &self,
        hourly_cost: f64,
        daily_cost: f64,
        monthly_projection: f64,
        budget_utilization_percent: f64,
        cost_breakdown: serde_json::Value,
    ) {
        let message = WebSocketMessage::CostUpdate {
            timestamp: chrono::Utc::now().to_rfc3339(),
            hourly_cost,
            daily_cost,
            monthly_projection,
            budget_utilization_percent,
            cost_breakdown,
        };
        self.broadcast(message).await;
    }

    /// Broadcast security update
    pub async fn broadcast_security_update(
        &self,
        overall_score: f64,
        critical_vulnerabilities: u32,
        high_vulnerabilities: u32,
        policy_violations: u32,
        recommendations: Vec<String>,
    ) {
        let message = WebSocketMessage::SecurityUpdate {
            timestamp: chrono::Utc::now().to_rfc3339(),
            overall_score,
            critical_vulnerabilities,
            high_vulnerabilities,
            policy_violations,
            recommendations,
        };
        self.broadcast(message).await;
    }

    /// Broadcast pipeline execution update
    pub async fn broadcast_pipeline_execution(
        &self,
        pipeline_id: String,
        execution_id: String,
        status: String,
        current_stage: Option<String>,
        completed_stages: Vec<String>,
        failed_stages: Vec<String>,
        estimated_completion: Option<String>,
    ) {
        let message = WebSocketMessage::PipelineExecutionUpdate {
            pipeline_id,
            execution_id,
            status,
            current_stage,
            completed_stages,
            failed_stages,
            estimated_completion,
        };
        self.broadcast(message).await;
    }

    /// Broadcast resource allocation update
    pub async fn broadcast_resource_allocation(
        &self,
        node_id: String,
        resource_type: String,
        total: f64,
        allocated: f64,
        utilization_percent: f64,
        available: f64,
    ) {
        let message = WebSocketMessage::ResourceAllocationUpdate {
            timestamp: chrono::Utc::now().to_rfc3339(),
            node_id,
            resource_type,
            total,
            allocated,
            utilization_percent,
            available,
        };
        self.broadcast(message).await;
    }

    /// Broadcast topology change notification
    pub async fn broadcast_topology_change(
        &self,
        change_type: String,
        affected_node_ids: Vec<String>,
        connection_id: Option<String>,
        details: serde_json::Value,
    ) {
        let message = WebSocketMessage::TopologyChangeNotification {
            change_type,
            affected_node_ids,
            connection_id,
            details,
            timestamp: chrono::Utc::now().to_rfc3339(),
        };
        self.broadcast(message).await;
    }

    /// Broadcast performance metrics
    pub async fn broadcast_performance_metrics(
        &self,
        node_id: String,
        metrics: serde_json::Value,
        alerts: Vec<String>,
        recommendations: Vec<String>,
    ) {
        let message = WebSocketMessage::PerformanceMetrics {
            timestamp: chrono::Utc::now().to_rfc3339(),
            node_id,
            metrics,
            alerts,
            recommendations,
        };
        self.broadcast(message).await;
    }

    /// Broadcast capacity alert
    pub async fn broadcast_capacity_alert(
        &self,
        alert_id: String,
        resource_type: String,
        current_utilization: f64,
        projected_exhaustion: Option<String>,
        recommended_actions: Vec<String>,
        severity: String,
    ) {
        let message = WebSocketMessage::CapacityAlert {
            alert_id,
            resource_type,
            current_utilization,
            projected_exhaustion,
            recommended_actions,
            severity,
            timestamp: chrono::Utc::now().to_rfc3339(),
        };
        self.broadcast(message).await;
    }

    // =============================================================================
    // Agent XP and Evolution Broadcasting Methods
    // =============================================================================

    /// Broadcast agent XP gained event
    pub async fn broadcast_agent_xp_gained(
        &self,
        agent_id: String,
        xp_amount: u64,
        reason: String,
        category: String,
        new_total_xp: u64,
        new_current_xp: u64,
        new_level: u32,
        leveled_up: bool,
        ready_to_evolve: bool,
    ) {
        let message = WebSocketMessage::AgentXPGained {
            agent_id,
            xp_amount,
            reason,
            category,
            new_total_xp,
            new_current_xp,
            new_level,
            leveled_up,
            ready_to_evolve,
            timestamp: chrono::Utc::now().to_rfc3339(),
        };
        self.broadcast(message).await;
    }

    /// Broadcast agent evolution event
    pub async fn broadcast_agent_evolved(
        &self,
        agent_id: String,
        previous_level: u32,
        new_level: u32,
        xp_at_evolution: u64,
    ) {
        let message = WebSocketMessage::AgentEvolved {
            agent_id,
            previous_level,
            new_level,
            xp_at_evolution,
            evolution_timestamp: chrono::Utc::now().to_rfc3339(),
        };
        self.broadcast(message).await;
    }

    /// Broadcast system XP statistics update
    pub async fn broadcast_system_xp_update(
        &self,
        total_agents: u32,
        total_xp_awarded: u64,
        average_level: f64,
        highest_level: u32,
        agents_ready_to_evolve: u32,
        xp_gained_24h: u64,
    ) {
        let message = WebSocketMessage::SystemXPUpdate {
            total_agents,
            total_xp_awarded,
            average_level,
            highest_level,
            agents_ready_to_evolve,
            xp_gained_24h,
            timestamp: chrono::Utc::now().to_rfc3339(),
        };
        self.broadcast(message).await;
    }
}

impl Clone for WebSocketHandler {
    fn clone(&self) -> Self {
        Self {
            clients: self.clients.clone(),
            broadcast_tx: self.broadcast_tx.clone(),
        }
    }
}
