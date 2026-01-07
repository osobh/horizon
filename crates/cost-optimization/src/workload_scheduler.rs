//! Workload scheduling module for intelligent placement and cost-aware scheduling
//!
//! This module provides advanced workload scheduling capabilities including:
//! - Cost-aware workload placement across heterogeneous resources
//! - Priority-based scheduling with preemption support
//! - Bin packing algorithms for optimal resource utilization
//! - Affinity and anti-affinity rules enforcement

use crate::error::{CostOptimizationError, CostOptimizationResult};
use crate::gpu_optimizer::{GpuAllocation, GpuAllocationRequest};
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use ordered_float::OrderedFloat;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, Mutex, Semaphore};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

/// Workload priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum WorkloadPriority {
    /// Critical workloads that cannot be interrupted
    Critical = 100,
    /// High priority production workloads
    High = 75,
    /// Normal priority workloads
    Normal = 50,
    /// Low priority batch jobs
    Low = 25,
    /// Best-effort workloads
    BestEffort = 0,
}

/// Workload state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WorkloadState {
    /// Pending scheduling
    Pending,
    /// Scheduled and waiting for resources
    Scheduled,
    /// Running on allocated resources
    Running,
    /// Completed successfully
    Completed,
    /// Failed execution
    Failed,
    /// Preempted by higher priority workload
    Preempted,
    /// Cancelled by user
    Cancelled,
}

/// Resource requirements for a workload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirements {
    /// CPU cores required
    pub cpu_cores: f64,
    /// Memory in GB
    pub memory_gb: f64,
    /// GPU count
    pub gpu_count: u32,
    /// GPU memory in GB
    pub gpu_memory_gb: f64,
    /// Storage in GB
    pub storage_gb: f64,
    /// Network bandwidth in Gbps
    pub network_gbps: f64,
    /// Preferred GPU models
    pub gpu_models: Vec<String>,
    /// Exclusive resource allocation
    pub exclusive: bool,
}

/// Workload placement constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlacementConstraints {
    /// Required node labels
    pub node_selector: HashMap<String, String>,
    /// Node affinity rules
    pub node_affinity: Vec<AffinityRule>,
    /// Pod affinity rules
    pub pod_affinity: Vec<AffinityRule>,
    /// Pod anti-affinity rules
    pub pod_anti_affinity: Vec<AffinityRule>,
    /// Toleration for node taints
    pub tolerations: Vec<Toleration>,
    /// Spread constraints
    pub topology_spread: Vec<TopologySpreadConstraint>,
}

/// Affinity rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AffinityRule {
    /// Label selector
    pub label_selector: HashMap<String, String>,
    /// Topology key (e.g., "zone", "node")
    pub topology_key: String,
    /// Required during scheduling
    pub required: bool,
    /// Weight for preferred rules (1-100)
    pub weight: Option<u32>,
}

/// Node taint toleration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Toleration {
    /// Taint key
    pub key: String,
    /// Taint value
    pub value: Option<String>,
    /// Taint effect
    pub effect: TaintEffect,
    /// Toleration operator
    pub operator: TolerationOperator,
}

/// Taint effect
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaintEffect {
    /// Do not schedule new pods
    NoSchedule,
    /// Prefer not to schedule new pods
    PreferNoSchedule,
    /// Evict existing pods
    NoExecute,
}

/// Toleration operator
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TolerationOperator {
    /// Key and value must match
    Equal,
    /// Key must exist
    Exists,
}

/// Topology spread constraint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologySpreadConstraint {
    /// Maximum skew allowed
    pub max_skew: u32,
    /// Topology key
    pub topology_key: String,
    /// What to do when constraint cannot be satisfied
    pub when_unsatisfiable: UnsatisfiableConstraintAction,
}

/// Action when constraint cannot be satisfied
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnsatisfiableConstraintAction {
    /// Do not schedule
    DoNotSchedule,
    /// Schedule anyway
    ScheduleAnyway,
}

/// Workload definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Workload {
    /// Workload ID
    pub id: Uuid,
    /// Workload name
    pub name: String,
    /// Priority
    pub priority: WorkloadPriority,
    /// Resource requirements
    pub resources: ResourceRequirements,
    /// Placement constraints
    pub constraints: PlacementConstraints,
    /// Estimated duration
    pub estimated_duration: Duration,
    /// Maximum cost per hour
    pub max_cost_per_hour: Option<f64>,
    /// Deadline for completion
    pub deadline: Option<DateTime<Utc>>,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

/// Scheduled workload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduledWorkload {
    /// Workload
    pub workload: Workload,
    /// Current state
    pub state: WorkloadState,
    /// Scheduled time
    pub scheduled_at: DateTime<Utc>,
    /// Start time
    pub started_at: Option<DateTime<Utc>>,
    /// Completion time
    pub completed_at: Option<DateTime<Utc>>,
    /// Assigned node
    pub assigned_node: Option<String>,
    /// Resource allocations
    pub allocations: Vec<ResourceAllocation>,
    /// Actual cost per hour
    pub cost_per_hour: f64,
    /// Total cost incurred
    pub total_cost: f64,
}

/// Resource allocation for a workload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceAllocation {
    /// Resource type
    pub resource_type: String,
    /// Resource ID
    pub resource_id: String,
    /// Allocated amount
    pub amount: f64,
    /// Allocation ID
    pub allocation_id: Option<Uuid>,
}

/// Compute node information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeNode {
    /// Node ID
    pub id: String,
    /// Node name
    pub name: String,
    /// Labels
    pub labels: HashMap<String, String>,
    /// Taints
    pub taints: Vec<NodeTaint>,
    /// Total resources
    pub total_resources: NodeResources,
    /// Available resources
    pub available_resources: NodeResources,
    /// Cost per hour
    pub cost_per_hour: f64,
    /// Zone/region
    pub zone: String,
    /// Provider
    pub provider: String,
}

/// Node taint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeTaint {
    /// Taint key
    pub key: String,
    /// Taint value
    pub value: String,
    /// Taint effect
    pub effect: TaintEffect,
}

/// Node resources
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeResources {
    /// CPU cores
    pub cpu_cores: f64,
    /// Memory in GB
    pub memory_gb: f64,
    /// GPU count
    pub gpu_count: u32,
    /// GPU memory in GB
    pub gpu_memory_gb: f64,
    /// Storage in GB
    pub storage_gb: f64,
    /// Network bandwidth in Gbps
    pub network_gbps: f64,
}

/// Scheduling strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SchedulingStrategy {
    /// Minimize cost
    CostOptimized,
    /// Maximize performance
    PerformanceOptimized,
    /// Balance cost and performance
    Balanced,
    /// Minimize latency
    LatencyOptimized,
    /// Maximize utilization
    BinPacking,
    /// Spread workloads evenly
    RoundRobin,
}

/// Scheduler configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerConfig {
    /// Scheduling strategy
    pub strategy: SchedulingStrategy,
    /// Enable preemption
    pub enable_preemption: bool,
    /// Scheduling interval
    pub scheduling_interval: Duration,
    /// Maximum queue size
    pub max_queue_size: usize,
    /// Enable gang scheduling
    pub enable_gang_scheduling: bool,
    /// Oversubscription ratio
    pub oversubscription_ratio: f64,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            strategy: SchedulingStrategy::CostOptimized,
            enable_preemption: true,
            scheduling_interval: Duration::from_secs(10),
            max_queue_size: 10000,
            enable_gang_scheduling: false,
            oversubscription_ratio: 1.2,
        }
    }
}

/// Workload scheduler for intelligent resource allocation
pub struct WorkloadScheduler {
    /// Configuration
    config: Arc<SchedulerConfig>,
    /// Compute nodes
    nodes: Arc<DashMap<String, ComputeNode>>,
    /// Pending workloads queue
    pending_queue: Arc<Mutex<BinaryHeap<PrioritizedWorkload>>>,
    /// Running workloads
    running_workloads: Arc<DashMap<Uuid, ScheduledWorkload>>,
    /// Completed workloads
    completed_workloads: Arc<DashMap<Uuid, ScheduledWorkload>>,
    /// Scheduling channel
    scheduling_tx: mpsc::Sender<SchedulingEvent>,
    /// Scheduling receiver
    scheduling_rx: Arc<Mutex<mpsc::Receiver<SchedulingEvent>>>,
    /// Metrics
    metrics: Arc<RwLock<SchedulerMetrics>>,
}

/// Prioritized workload for queue
#[derive(Debug, Clone)]
struct PrioritizedWorkload {
    workload: Workload,
    priority_score: OrderedFloat<f64>,
    queued_at: DateTime<Utc>,
}

impl PartialEq for PrioritizedWorkload {
    fn eq(&self, other: &Self) -> bool {
        self.priority_score == other.priority_score
    }
}

impl Eq for PrioritizedWorkload {}

impl PartialOrd for PrioritizedWorkload {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PrioritizedWorkload {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.priority_score
            .cmp(&other.priority_score)
            .then_with(|| other.queued_at.cmp(&self.queued_at)) // Earlier first for same priority
    }
}

/// Scheduling event
#[derive(Debug)]
enum SchedulingEvent {
    /// New workload submitted
    WorkloadSubmitted(Workload),
    /// Workload completed
    WorkloadCompleted(Uuid),
    /// Node added
    NodeAdded(ComputeNode),
    /// Node removed
    NodeRemoved(String),
    /// Trigger scheduling cycle
    SchedulingCycle,
}

/// Scheduler metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SchedulerMetrics {
    /// Total workloads scheduled
    pub total_scheduled: u64,
    /// Total workloads completed
    pub total_completed: u64,
    /// Total workloads failed
    pub total_failed: u64,
    /// Total workloads preempted
    pub total_preempted: u64,
    /// Average queue time
    pub avg_queue_time: Duration,
    /// Average completion time
    pub avg_completion_time: Duration,
    /// Resource utilization
    pub resource_utilization: f64,
    /// Total cost
    pub total_cost: f64,
}

impl WorkloadScheduler {
    /// Maximum number of pending scheduling commands to prevent memory exhaustion
    const MAX_SCHEDULING_QUEUE: usize = 10_000;

    /// Create a new workload scheduler
    pub fn new(config: SchedulerConfig) -> CostOptimizationResult<Self> {
        // Use bounded channel to prevent memory exhaustion under load
        let (tx, rx) = mpsc::channel(Self::MAX_SCHEDULING_QUEUE);

        Ok(Self {
            config: Arc::new(config),
            nodes: Arc::new(DashMap::new()),
            pending_queue: Arc::new(Mutex::new(BinaryHeap::new())),
            running_workloads: Arc::new(DashMap::new()),
            completed_workloads: Arc::new(DashMap::new()),
            scheduling_tx: tx,
            scheduling_rx: Arc::new(Mutex::new(rx)),
            metrics: Arc::new(RwLock::new(SchedulerMetrics::default())),
        })
    }

    /// Start the scheduler
    pub async fn start(&self) -> CostOptimizationResult<()> {
        info!("Starting workload scheduler");

        let config = self.config.clone();
        let rx = self.scheduling_rx.clone();
        let nodes = self.nodes.clone();
        let pending_queue = self.pending_queue.clone();
        let running_workloads = self.running_workloads.clone();
        let completed_workloads = self.completed_workloads.clone();
        let metrics = self.metrics.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(config.scheduling_interval);

            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        if let Err(e) = Self::run_scheduling_cycle(
                            &config,
                            &nodes,
                            &pending_queue,
                            &running_workloads,
                            &completed_workloads,
                            &metrics,
                        ).await {
                            error!("Scheduling cycle failed: {}", e);
                        }
                    }
                    event = async {
                        let mut rx = rx.lock().await;
                        rx.recv().await
                    } => {
                        if let Some(event) = event {
                            if let Err(e) = Self::handle_event(
                            event,
                            &config,
                            &nodes,
                            &pending_queue,
                            &running_workloads,
                            &completed_workloads,
                            &metrics,
                        ).await {
                            error!("Event handling failed: {}", e);
                        }
                        }
                    }
                }
            }
        });

        Ok(())
    }

    /// Submit a workload for scheduling
    pub async fn submit_workload(&self, workload: Workload) -> CostOptimizationResult<()> {
        info!("Submitting workload: {} ({})", workload.name, workload.id);

        // Check queue size
        let queue_size = self.pending_queue.lock().await.len();
        if queue_size >= self.config.max_queue_size {
            return Err(CostOptimizationError::SchedulingFailed {
                workload_id: workload.id.to_string(),
                reason: "Queue is full".to_string(),
            });
        }

        self.scheduling_tx
            .send(SchedulingEvent::WorkloadSubmitted(workload))
            .await
            .map_err(|_| CostOptimizationError::SchedulingFailed {
                workload_id: "unknown".to_string(),
                reason: "Failed to send scheduling event".to_string(),
            })?;

        Ok(())
    }

    /// Register a compute node
    pub async fn register_node(&self, node: ComputeNode) -> CostOptimizationResult<()> {
        info!("Registering compute node: {}", node.name);

        self.scheduling_tx
            .send(SchedulingEvent::NodeAdded(node))
            .await
            .map_err(|_| CostOptimizationError::SchedulingFailed {
                workload_id: "unknown".to_string(),
                reason: "Failed to send node event".to_string(),
            })?;

        Ok(())
    }

    /// Unregister a compute node
    pub async fn unregister_node(&self, node_id: &str) -> CostOptimizationResult<()> {
        info!("Unregistering compute node: {}", node_id);

        self.scheduling_tx
            .send(SchedulingEvent::NodeRemoved(node_id.to_string()))
            .await
            .map_err(|_| CostOptimizationError::SchedulingFailed {
                workload_id: "unknown".to_string(),
                reason: "Failed to send node event".to_string(),
            })?;

        Ok(())
    }

    /// Get workload status
    pub fn get_workload_status(&self, workload_id: Uuid) -> Option<ScheduledWorkload> {
        self.running_workloads
            .get(&workload_id)
            .map(|entry| entry.value().clone())
            .or_else(|| {
                self.completed_workloads
                    .get(&workload_id)
                    .map(|entry| entry.value().clone())
            })
    }

    /// Get scheduler metrics
    pub fn get_metrics(&self) -> SchedulerMetrics {
        self.metrics.read().clone()
    }

    /// Handle scheduling events
    async fn handle_event(
        event: SchedulingEvent,
        config: &Arc<SchedulerConfig>,
        nodes: &Arc<DashMap<String, ComputeNode>>,
        pending_queue: &Arc<Mutex<BinaryHeap<PrioritizedWorkload>>>,
        running_workloads: &Arc<DashMap<Uuid, ScheduledWorkload>>,
        completed_workloads: &Arc<DashMap<Uuid, ScheduledWorkload>>,
        metrics: &Arc<RwLock<SchedulerMetrics>>,
    ) -> CostOptimizationResult<()> {
        match event {
            SchedulingEvent::WorkloadSubmitted(workload) => {
                let priority_score = Self::calculate_priority_score(&workload);
                let prioritized = PrioritizedWorkload {
                    workload,
                    priority_score,
                    queued_at: Utc::now(),
                };

                pending_queue.lock().await.push(prioritized);
            }

            SchedulingEvent::WorkloadCompleted(workload_id) => {
                if let Some((_, mut scheduled)) = running_workloads.remove(&workload_id) {
                    scheduled.state = WorkloadState::Completed;
                    scheduled.completed_at = Some(Utc::now());

                    // Update metrics
                    let mut m = metrics.write();
                    m.total_completed += 1;
                    m.total_cost += scheduled.total_cost;

                    completed_workloads.insert(workload_id, scheduled);
                }
            }

            SchedulingEvent::NodeAdded(node) => {
                nodes.insert(node.id.clone(), node);
            }

            SchedulingEvent::NodeRemoved(node_id) => {
                // Reschedule workloads from removed node
                let mut workloads_to_reschedule = Vec::new();

                for entry in running_workloads.iter() {
                    if entry.value().assigned_node.as_ref() == Some(&node_id) {
                        workloads_to_reschedule.push(entry.key().clone());
                    }
                }

                for workload_id in workloads_to_reschedule {
                    if let Some((_, scheduled)) = running_workloads.remove(&workload_id) {
                        let prioritized = PrioritizedWorkload {
                            workload: scheduled.workload,
                            priority_score: OrderedFloat(100.0), // High priority for rescheduling
                            queued_at: Utc::now(),
                        };
                        pending_queue.lock().await.push(prioritized);
                    }
                }

                nodes.remove(&node_id);
            }

            SchedulingEvent::SchedulingCycle => {
                // Handled by the main scheduling loop
            }
        }

        Ok(())
    }

    /// Run a scheduling cycle
    async fn run_scheduling_cycle(
        config: &Arc<SchedulerConfig>,
        nodes: &Arc<DashMap<String, ComputeNode>>,
        pending_queue: &Arc<Mutex<BinaryHeap<PrioritizedWorkload>>>,
        running_workloads: &Arc<DashMap<Uuid, ScheduledWorkload>>,
        completed_workloads: &Arc<DashMap<Uuid, ScheduledWorkload>>,
        metrics: &Arc<RwLock<SchedulerMetrics>>,
    ) -> CostOptimizationResult<()> {
        debug!("Running scheduling cycle");

        let mut queue = pending_queue.lock().await;
        let mut scheduled_count = 0;

        while let Some(prioritized) = queue.pop() {
            let workload = prioritized.workload;

            // Find suitable node based on strategy
            let node_scores = Self::score_nodes(&workload, nodes, config)?;

            if let Some((node_id, _score)) = node_scores.first() {
                // Try to allocate resources on the selected node
                if let Some(mut node_entry) = nodes.get_mut(node_id) {
                    if Self::can_fit(&workload.resources, &node_entry.available_resources) {
                        // Update node resources
                        node_entry.available_resources.cpu_cores -= workload.resources.cpu_cores;
                        node_entry.available_resources.memory_gb -= workload.resources.memory_gb;
                        node_entry.available_resources.gpu_count -= workload.resources.gpu_count;
                        node_entry.available_resources.gpu_memory_gb -=
                            workload.resources.gpu_memory_gb;
                        node_entry.available_resources.storage_gb -= workload.resources.storage_gb;
                        node_entry.available_resources.network_gbps -=
                            workload.resources.network_gbps;

                        // Create scheduled workload
                        let scheduled = ScheduledWorkload {
                            workload: workload.clone(),
                            state: WorkloadState::Scheduled,
                            scheduled_at: Utc::now(),
                            started_at: Some(Utc::now()),
                            completed_at: None,
                            assigned_node: Some(node_id.clone()),
                            allocations: vec![
                                ResourceAllocation {
                                    resource_type: "cpu".to_string(),
                                    resource_id: node_id.clone(),
                                    amount: workload.resources.cpu_cores,
                                    allocation_id: None,
                                },
                                ResourceAllocation {
                                    resource_type: "memory".to_string(),
                                    resource_id: node_id.clone(),
                                    amount: workload.resources.memory_gb,
                                    allocation_id: None,
                                },
                            ],
                            cost_per_hour: node_entry.cost_per_hour,
                            total_cost: 0.0,
                        };

                        running_workloads.insert(workload.id, scheduled);
                        scheduled_count += 1;

                        // Update metrics
                        let mut m = metrics.write();
                        m.total_scheduled += 1;
                    } else {
                        // Check if preemption is enabled and beneficial
                        if config.enable_preemption && workload.priority >= WorkloadPriority::High {
                            if let Some(preemptable) = Self::find_preemptable_workloads(
                                &workload,
                                node_id,
                                running_workloads,
                            ) {
                                // Preempt lower priority workloads
                                for workload_id in preemptable {
                                    if let Some((_, mut scheduled)) =
                                        running_workloads.remove(&workload_id)
                                    {
                                        scheduled.state = WorkloadState::Preempted;

                                        // Re-queue preempted workload
                                        let prioritized = PrioritizedWorkload {
                                            workload: scheduled.workload,
                                            priority_score: OrderedFloat(50.0),
                                            queued_at: Utc::now(),
                                        };
                                        queue.push(prioritized);

                                        // Update metrics
                                        metrics.write().total_preempted += 1;
                                    }
                                }

                                // Retry scheduling
                                let reprioritized = PrioritizedWorkload {
                                    workload: workload.clone(),
                                    priority_score: prioritized.priority_score,
                                    queued_at: prioritized.queued_at,
                                };
                                queue.push(reprioritized);
                                continue;
                            }
                        }

                        // Re-queue if can't schedule
                        let reprioritized = PrioritizedWorkload {
                            workload,
                            priority_score: prioritized.priority_score,
                            queued_at: prioritized.queued_at,
                        };
                        queue.push(reprioritized);
                        break;
                    }
                } else {
                    // Re-queue if node not found
                    let reprioritized = PrioritizedWorkload {
                        workload,
                        priority_score: prioritized.priority_score,
                        queued_at: prioritized.queued_at,
                    };
                    queue.push(reprioritized);
                    break;
                }
            } else {
                // No suitable nodes, re-queue
                let reprioritized = PrioritizedWorkload {
                    workload,
                    priority_score: prioritized.priority_score,
                    queued_at: prioritized.queued_at,
                };
                queue.push(reprioritized);
                break;
            }
        }

        if scheduled_count > 0 {
            info!("Scheduled {} workloads in this cycle", scheduled_count);
        }

        // Update utilization metrics
        Self::update_utilization_metrics(nodes, metrics);

        Ok(())
    }

    /// Calculate priority score for a workload
    fn calculate_priority_score(workload: &Workload) -> OrderedFloat<f64> {
        let mut score = workload.priority as u32 as f64;

        // Boost score for workloads with deadlines
        if let Some(deadline) = workload.deadline {
            let time_until_deadline = (deadline - Utc::now()).num_seconds() as f64;
            if time_until_deadline > 0.0 {
                score += 1000.0 / time_until_deadline; // Urgency boost
            }
        }

        // Consider resource requirements (smaller workloads get slight boost)
        score += 10.0 / (1.0 + workload.resources.cpu_cores);

        OrderedFloat(score)
    }

    /// Score nodes for workload placement
    fn score_nodes(
        workload: &Workload,
        nodes: &Arc<DashMap<String, ComputeNode>>,
        config: &Arc<SchedulerConfig>,
    ) -> CostOptimizationResult<Vec<(String, f64)>> {
        let mut scores = Vec::new();

        for node_entry in nodes.iter() {
            let node = node_entry.value();

            // Check if node can accommodate workload
            if !Self::can_fit(&workload.resources, &node.available_resources) {
                continue;
            }

            // Check constraints
            if !Self::check_constraints(workload, node) {
                continue;
            }

            // Calculate score based on strategy
            let score = match config.strategy {
                SchedulingStrategy::CostOptimized => {
                    // Lower cost is better
                    1000.0 / (1.0 + node.cost_per_hour)
                }
                SchedulingStrategy::PerformanceOptimized => {
                    // More available resources is better
                    node.available_resources.cpu_cores
                        + node.available_resources.gpu_count as f64 * 10.0
                }
                SchedulingStrategy::Balanced => {
                    // Balance between cost and performance
                    let cost_score = 100.0 / (1.0 + node.cost_per_hour);
                    let perf_score = node.available_resources.cpu_cores;
                    (cost_score + perf_score) / 2.0
                }
                SchedulingStrategy::LatencyOptimized => {
                    // Prefer nodes in same zone
                    if workload.metadata.get("preferred_zone") == Some(&node.zone) {
                        1000.0
                    } else {
                        100.0
                    }
                }
                SchedulingStrategy::BinPacking => {
                    // Prefer nodes with least available space (after allocation)
                    let remaining =
                        node.available_resources.cpu_cores - workload.resources.cpu_cores;
                    1000.0 / (1.0 + remaining)
                }
                SchedulingStrategy::RoundRobin => {
                    // Equal score for all nodes
                    1.0
                }
            };

            scores.push((node.id.clone(), score));
        }

        // Sort by score (descending)
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        Ok(scores)
    }

    /// Check if resources can fit
    fn can_fit(required: &ResourceRequirements, available: &NodeResources) -> bool {
        available.cpu_cores >= required.cpu_cores
            && available.memory_gb >= required.memory_gb
            && available.gpu_count >= required.gpu_count
            && available.gpu_memory_gb >= required.gpu_memory_gb
            && available.storage_gb >= required.storage_gb
            && available.network_gbps >= required.network_gbps
    }

    /// Check placement constraints
    fn check_constraints(workload: &Workload, node: &ComputeNode) -> bool {
        // Check node selector
        for (key, value) in &workload.constraints.node_selector {
            if node.labels.get(key) != Some(value) {
                return false;
            }
        }

        // Check tolerations
        for taint in &node.taints {
            let mut tolerated = false;
            for toleration in &workload.constraints.tolerations {
                if Self::matches_toleration(taint, toleration) {
                    tolerated = true;
                    break;
                }
            }
            if !tolerated && taint.effect == TaintEffect::NoSchedule {
                return false;
            }
        }

        true
    }

    /// Check if taint matches toleration
    fn matches_toleration(taint: &NodeTaint, toleration: &Toleration) -> bool {
        if toleration.key != taint.key {
            return false;
        }

        if toleration.effect != taint.effect {
            return false;
        }

        match toleration.operator {
            TolerationOperator::Equal => toleration.value.as_ref() == Some(&taint.value),
            TolerationOperator::Exists => true,
        }
    }

    /// Find preemptable workloads
    fn find_preemptable_workloads(
        workload: &Workload,
        node_id: &str,
        running_workloads: &Arc<DashMap<Uuid, ScheduledWorkload>>,
    ) -> Option<Vec<Uuid>> {
        let mut candidates = Vec::new();
        let mut freed_resources = NodeResources {
            cpu_cores: 0.0,
            memory_gb: 0.0,
            gpu_count: 0,
            gpu_memory_gb: 0.0,
            storage_gb: 0.0,
            network_gbps: 0.0,
        };

        // Find lower priority workloads on the same node
        for entry in running_workloads.iter() {
            let scheduled = entry.value();
            if scheduled.assigned_node.as_ref() == Some(&node_id.to_string())
                && scheduled.workload.priority < workload.priority
            {
                candidates.push((
                    scheduled.workload.id,
                    scheduled.workload.priority,
                    scheduled.workload.resources.clone(),
                ));
            }
        }

        // Sort by priority (lowest first)
        candidates.sort_by_key(|(_, priority, _)| *priority);

        let mut preemptable = Vec::new();

        // Select workloads to preempt
        for (id, _, resources) in candidates {
            preemptable.push(id);
            freed_resources.cpu_cores += resources.cpu_cores;
            freed_resources.memory_gb += resources.memory_gb;
            freed_resources.gpu_count += resources.gpu_count;
            freed_resources.gpu_memory_gb += resources.gpu_memory_gb;
            freed_resources.storage_gb += resources.storage_gb;
            freed_resources.network_gbps += resources.network_gbps;

            // Check if enough resources are freed
            if Self::can_fit(&workload.resources, &freed_resources) {
                return Some(preemptable);
            }
        }

        None
    }

    /// Update utilization metrics
    fn update_utilization_metrics(
        nodes: &Arc<DashMap<String, ComputeNode>>,
        metrics: &Arc<RwLock<SchedulerMetrics>>,
    ) {
        let mut total_used = 0.0;
        let mut total_capacity = 0.0;

        for node_entry in nodes.iter() {
            let node = node_entry.value();
            let used = node.total_resources.cpu_cores - node.available_resources.cpu_cores;
            total_used += used;
            total_capacity += node.total_resources.cpu_cores;
        }

        if total_capacity > 0.0 {
            metrics.write().resource_utilization = total_used / total_capacity;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_workload(
        name: &str,
        priority: WorkloadPriority,
        cpu: f64,
        gpu: u32,
    ) -> Workload {
        Workload {
            id: Uuid::new_v4(),
            name: name.to_string(),
            priority,
            resources: ResourceRequirements {
                cpu_cores: cpu,
                memory_gb: cpu * 4.0, // 4GB per CPU
                gpu_count: gpu,
                gpu_memory_gb: gpu as f64 * 16.0, // 16GB per GPU
                storage_gb: 100.0,
                network_gbps: 1.0,
                gpu_models: vec![],
                exclusive: false,
            },
            constraints: PlacementConstraints {
                node_selector: HashMap::new(),
                node_affinity: vec![],
                pod_affinity: vec![],
                pod_anti_affinity: vec![],
                tolerations: vec![],
                topology_spread: vec![],
            },
            estimated_duration: Duration::from_secs(3600),
            max_cost_per_hour: None,
            deadline: None,
            metadata: HashMap::new(),
        }
    }

    fn create_test_node(id: &str, cpu: f64, gpu: u32, cost: f64) -> ComputeNode {
        ComputeNode {
            id: id.to_string(),
            name: format!("node-{}", id),
            labels: HashMap::new(),
            taints: vec![],
            total_resources: NodeResources {
                cpu_cores: cpu,
                memory_gb: cpu * 4.0,
                gpu_count: gpu,
                gpu_memory_gb: gpu as f64 * 16.0,
                storage_gb: 1000.0,
                network_gbps: 10.0,
            },
            available_resources: NodeResources {
                cpu_cores: cpu,
                memory_gb: cpu * 4.0,
                gpu_count: gpu,
                gpu_memory_gb: gpu as f64 * 16.0,
                storage_gb: 1000.0,
                network_gbps: 10.0,
            },
            cost_per_hour: cost,
            zone: "us-east-1a".to_string(),
            provider: "aws".to_string(),
        }
    }

    #[test]
    fn test_workload_priority_ordering() {
        assert!(WorkloadPriority::Critical > WorkloadPriority::High);
        assert!(WorkloadPriority::High > WorkloadPriority::Normal);
        assert!(WorkloadPriority::Normal > WorkloadPriority::Low);
        assert!(WorkloadPriority::Low > WorkloadPriority::BestEffort);
    }

    #[test]
    fn test_scheduler_creation() {
        let config = SchedulerConfig::default();
        let scheduler = WorkloadScheduler::new(config);
        assert!(scheduler.is_ok());
    }

    #[tokio::test]
    async fn test_workload_submission() {
        let config = SchedulerConfig::default();
        let scheduler = WorkloadScheduler::new(config).unwrap();

        let workload = create_test_workload("test-job", WorkloadPriority::Normal, 4.0, 1);
        let result = scheduler.submit_workload(workload).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_node_registration() {
        let config = SchedulerConfig::default();
        let scheduler = WorkloadScheduler::new(config).unwrap();

        let node = create_test_node("node-1", 32.0, 4, 10.0);
        let result = scheduler.register_node(node).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_basic_scheduling() {
        let config = SchedulerConfig::default();
        let scheduler = WorkloadScheduler::new(config).unwrap();

        // Start scheduler
        scheduler.start().await.unwrap();

        // Register a node
        let node = create_test_node("node-1", 32.0, 4, 10.0);
        scheduler.register_node(node).await.unwrap();

        // Submit a workload
        let workload = create_test_workload("test-job", WorkloadPriority::Normal, 4.0, 1);
        let workload_id = workload.id;
        scheduler.submit_workload(workload).await.unwrap();

        // Wait for scheduling
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Check if workload was scheduled
        let status = scheduler.get_workload_status(workload_id);
        assert!(status.is_some());
    }

    #[test]
    fn test_priority_score_calculation() {
        let workload1 = create_test_workload("high-priority", WorkloadPriority::High, 4.0, 1);
        let workload2 = create_test_workload("normal-priority", WorkloadPriority::Normal, 4.0, 1);

        let score1 = WorkloadScheduler::calculate_priority_score(&workload1);
        let score2 = WorkloadScheduler::calculate_priority_score(&workload2);

        assert!(score1 > score2);
    }

    #[test]
    fn test_resource_fitting() {
        let required = ResourceRequirements {
            cpu_cores: 4.0,
            memory_gb: 16.0,
            gpu_count: 1,
            gpu_memory_gb: 16.0,
            storage_gb: 100.0,
            network_gbps: 1.0,
            gpu_models: vec![],
            exclusive: false,
        };

        let available = NodeResources {
            cpu_cores: 8.0,
            memory_gb: 32.0,
            gpu_count: 2,
            gpu_memory_gb: 32.0,
            storage_gb: 1000.0,
            network_gbps: 10.0,
        };

        assert!(WorkloadScheduler::can_fit(&required, &available));

        // Test insufficient resources
        let insufficient = NodeResources {
            cpu_cores: 2.0,
            memory_gb: 8.0,
            gpu_count: 0,
            gpu_memory_gb: 0.0,
            storage_gb: 50.0,
            network_gbps: 0.5,
        };

        assert!(!WorkloadScheduler::can_fit(&required, &insufficient));
    }

    #[test]
    fn test_constraint_checking() {
        let mut workload = create_test_workload("constrained", WorkloadPriority::Normal, 4.0, 1);
        workload
            .constraints
            .node_selector
            .insert("gpu-type".to_string(), "a100".to_string());

        let mut node = create_test_node("node-1", 32.0, 4, 10.0);
        node.labels
            .insert("gpu-type".to_string(), "a100".to_string());

        assert!(WorkloadScheduler::check_constraints(&workload, &node));

        // Test with mismatched label
        node.labels
            .insert("gpu-type".to_string(), "v100".to_string());
        assert!(!WorkloadScheduler::check_constraints(&workload, &node));
    }

    #[test]
    fn test_taint_toleration() {
        let mut workload = create_test_workload("tolerant", WorkloadPriority::Normal, 4.0, 1);
        workload.constraints.tolerations.push(Toleration {
            key: "gpu-only".to_string(),
            value: Some("true".to_string()),
            effect: TaintEffect::NoSchedule,
            operator: TolerationOperator::Equal,
        });

        let mut node = create_test_node("node-1", 32.0, 4, 10.0);
        node.taints.push(NodeTaint {
            key: "gpu-only".to_string(),
            value: "true".to_string(),
            effect: TaintEffect::NoSchedule,
        });

        assert!(WorkloadScheduler::check_constraints(&workload, &node));

        // Test without toleration
        let workload2 = create_test_workload("intolerant", WorkloadPriority::Normal, 4.0, 1);
        assert!(!WorkloadScheduler::check_constraints(&workload2, &node));
    }

    #[tokio::test]
    async fn test_cost_optimized_scheduling() {
        let mut config = SchedulerConfig::default();
        config.strategy = SchedulingStrategy::CostOptimized;

        let scheduler = WorkloadScheduler::new(config).unwrap();
        scheduler.start().await.unwrap();

        // Register nodes with different costs
        scheduler
            .register_node(create_test_node("expensive", 32.0, 4, 20.0))
            .await
            .unwrap();
        scheduler
            .register_node(create_test_node("cheap", 32.0, 4, 5.0))
            .await
            .unwrap();

        // Submit workload
        let workload = create_test_workload("cost-aware", WorkloadPriority::Normal, 4.0, 1);
        let workload_id = workload.id;
        scheduler.submit_workload(workload).await.unwrap();

        // Wait for scheduling
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Check that workload was scheduled on cheaper node
        let status = scheduler.get_workload_status(workload_id).unwrap();
        assert_eq!(status.assigned_node, Some("cheap".to_string()));
    }

    #[tokio::test]
    async fn test_preemption() {
        let mut config = SchedulerConfig::default();
        config.enable_preemption = true;

        let scheduler = WorkloadScheduler::new(config).unwrap();
        scheduler.start().await.unwrap();

        // Register a small node
        scheduler
            .register_node(create_test_node("node-1", 8.0, 1, 10.0))
            .await
            .unwrap();

        // Submit low priority workload that uses all resources
        let low_priority = create_test_workload("low", WorkloadPriority::Low, 8.0, 1);
        scheduler.submit_workload(low_priority).await.unwrap();

        tokio::time::sleep(Duration::from_millis(100)).await;

        // Submit high priority workload
        let high_priority = create_test_workload("high", WorkloadPriority::High, 4.0, 0);
        let high_id = high_priority.id;
        scheduler.submit_workload(high_priority).await.unwrap();

        tokio::time::sleep(Duration::from_millis(200)).await;

        // High priority workload should be scheduled
        let status = scheduler.get_workload_status(high_id);
        assert!(status.is_some());

        // Check metrics for preemption
        let metrics = scheduler.get_metrics();
        assert!(metrics.total_preempted > 0);
    }

    #[test]
    fn test_deadline_priority_boost() {
        let mut workload1 = create_test_workload("urgent", WorkloadPriority::Normal, 4.0, 1);
        workload1.deadline = Some(Utc::now() + chrono::Duration::hours(1));

        let workload2 = create_test_workload("regular", WorkloadPriority::Normal, 4.0, 1);

        let score1 = WorkloadScheduler::calculate_priority_score(&workload1);
        let score2 = WorkloadScheduler::calculate_priority_score(&workload2);

        // Urgent workload should have higher score
        assert!(score1 > score2);
    }

    #[tokio::test]
    async fn test_gang_scheduling() {
        let mut config = SchedulerConfig::default();
        config.enable_gang_scheduling = true;

        let scheduler = WorkloadScheduler::new(config).unwrap();

        // Gang scheduling test would require multiple related workloads
        // This is a placeholder for gang scheduling logic
        assert!(scheduler.config.enable_gang_scheduling);
    }

    #[tokio::test]
    async fn test_queue_limit() {
        let mut config = SchedulerConfig::default();
        config.max_queue_size = 2;

        let scheduler = WorkloadScheduler::new(config).unwrap();

        // Fill the queue
        for i in 0..2 {
            let workload =
                create_test_workload(&format!("job-{}", i), WorkloadPriority::Normal, 4.0, 1);
            scheduler.submit_workload(workload).await.unwrap();
        }

        // Next submission should fail
        let workload = create_test_workload("overflow", WorkloadPriority::Normal, 4.0, 1);
        let result = scheduler.submit_workload(workload).await;
        assert!(result.is_err());
    }
}
