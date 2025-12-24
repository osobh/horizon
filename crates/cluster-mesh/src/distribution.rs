//! Work distribution and scheduling
//!
//! This module handles intelligent work distribution across heterogeneous nodes,
//! considering capabilities, constraints, and locality.

use crate::{ClusterMeshError, ClusterNode, Job, NodeCapabilities, NodeClass, NodeStatus, Result};
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};
use uuid::Uuid;

/// Job requirements for scheduling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobRequirements {
    pub cpu_cores: Option<u32>,
    pub memory_gb: Option<f32>,
    pub gpu_count: Option<u32>,
    pub gpu_memory_gb: Option<f32>,
    pub storage_gb: Option<f32>,
    pub network_bandwidth_mbps: Option<f32>,
    pub requires_gpu_direct: bool,
    pub node_affinity: Option<NodeAffinity>,
    pub anti_affinity: Option<Vec<Uuid>>,
    pub locality_preference: LocalityPreference,
    pub max_latency_ms: Option<f32>,
    pub battery_safe: bool,
}

/// Node affinity rules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeAffinity {
    RequiredClass(NodeClass),
    PreferredClass(NodeClass),
    RequiredNodes(Vec<Uuid>),
    PreferredNodes(Vec<Uuid>),
}

/// Locality preferences for data/compute placement
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LocalityPreference {
    None,
    DataLocal,
    RackLocal,
    ZoneLocal,
    RegionLocal,
}

/// Scheduling policy
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SchedulingPolicy {
    BestFit,
    FirstFit,
    RoundRobin,
    PowerAware,
    CostOptimized,
    LatencyOptimized,
}

/// Scheduled job information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduledJob {
    pub job: Job,
    pub node_id: Uuid,
    pub scheduled_at: DateTime<Utc>,
    pub estimated_completion: Option<DateTime<Utc>>,
    pub actual_start: Option<DateTime<Utc>>,
    pub actual_completion: Option<DateTime<Utc>>,
    pub status: JobStatus,
}

/// Job execution status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum JobStatus {
    Queued,
    Scheduled,
    Running,
    Completed,
    Failed(String),
    Cancelled,
    Migrating,
}

/// Work distributor
pub struct WorkDistributor {
    policy: RwLock<SchedulingPolicy>,
    job_queue: Arc<Mutex<BinaryHeap<PrioritizedJob>>>,
    scheduled_jobs: Arc<DashMap<Uuid, ScheduledJob>>,
    node_loads: Arc<DashMap<Uuid, NodeLoad>>,
    migration_threshold: f32,
}

/// Node load information
#[derive(Debug, Clone, Default)]
struct NodeLoad {
    cpu_usage: f32,
    memory_usage: f32,
    gpu_usage: f32,
    active_jobs: Vec<Uuid>,
    last_updated: Option<DateTime<Utc>>,
}

/// Prioritized job for the heap
#[derive(Debug, Clone)]
struct PrioritizedJob {
    job: Job,
    score: i32,
}

impl PartialEq for PrioritizedJob {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}

impl Eq for PrioritizedJob {}

impl PartialOrd for PrioritizedJob {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PrioritizedJob {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score.cmp(&other.score)
    }
}

impl WorkDistributor {
    /// Create a new work distributor
    pub fn new() -> Self {
        Self {
            policy: RwLock::new(SchedulingPolicy::BestFit),
            job_queue: Arc::new(Mutex::new(BinaryHeap::new())),
            scheduled_jobs: Arc::new(DashMap::new()),
            node_loads: Arc::new(DashMap::new()),
            migration_threshold: 0.8, // 80% load threshold for migration
        }
    }

    /// Start the distribution service
    pub async fn start_distribution(&self) -> Result<()> {
        // Start background tasks for job scheduling and load balancing
        tokio::spawn(self.clone().scheduling_loop());
        tokio::spawn(self.clone().load_balancing_loop());
        Ok(())
    }

    /// Clone for async spawning
    fn clone(&self) -> Self {
        Self {
            policy: RwLock::new(SchedulingPolicy::BestFit),
            job_queue: self.job_queue.clone(),
            scheduled_jobs: self.scheduled_jobs.clone(),
            node_loads: self.node_loads.clone(),
            migration_threshold: self.migration_threshold,
        }
    }

    /// Main scheduling loop
    async fn scheduling_loop(self) {
        let mut interval = tokio::time::interval(tokio::time::Duration::from_millis(100));

        loop {
            interval.tick().await;

            // Process pending jobs
            if let Ok(mut queue) = self.job_queue.try_lock() {
                while let Some(prioritized_job) = queue.pop() {
                    // This would attempt to schedule the job
                    // For now, we'll just log it
                    tracing::debug!("Processing job: {}", prioritized_job.job.id);
                }
            }
        }
    }

    /// Load balancing loop
    async fn load_balancing_loop(self) {
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(30));

        loop {
            interval.tick().await;

            // Check for overloaded nodes and rebalance
            if let Err(e) = self.check_and_rebalance().await {
                tracing::error!("Load balancing error: {}", e);
            }
        }
    }

    /// Select a node for a job
    pub async fn select_node<'a>(
        &self,
        job: &Job,
        nodes: &'a [ClusterNode],
    ) -> Result<&'a ClusterNode> {
        let policy = self.policy.read().await;

        // Filter eligible nodes
        let eligible_nodes: Vec<&ClusterNode> = nodes
            .iter()
            .filter(|n| n.status == NodeStatus::Online)
            .filter(|n| {
                self.meets_requirements(&job.requirements, &n.capabilities, n)
                    .unwrap_or(false)
            })
            .collect();

        if eligible_nodes.is_empty() {
            return Err(ClusterMeshError::NoSuitableNode);
        }

        // Select based on policy
        match *policy {
            SchedulingPolicy::BestFit => self.select_best_fit(job, &eligible_nodes).await,
            SchedulingPolicy::FirstFit => Ok(eligible_nodes[0]),
            SchedulingPolicy::RoundRobin => self.select_round_robin(&eligible_nodes).await,
            SchedulingPolicy::PowerAware => self.select_power_aware(job, &eligible_nodes).await,
            SchedulingPolicy::CostOptimized => {
                self.select_cost_optimized(job, &eligible_nodes).await
            }
            SchedulingPolicy::LatencyOptimized => {
                self.select_latency_optimized(job, &eligible_nodes).await
            }
        }
    }

    /// Check if a node meets job requirements
    fn meets_requirements(
        &self,
        reqs: &JobRequirements,
        caps: &NodeCapabilities,
        node: &ClusterNode,
    ) -> Result<bool> {
        // Check CPU requirements
        if let Some(req_cores) = reqs.cpu_cores {
            if caps.cpu_cores < req_cores {
                return Ok(false);
            }
        }

        // Check memory requirements
        if let Some(req_memory) = reqs.memory_gb {
            if caps.memory_gb < req_memory {
                return Ok(false);
            }
        }

        // Check GPU requirements
        if let Some(req_gpu_count) = reqs.gpu_count {
            if caps.gpu_count < req_gpu_count {
                return Ok(false);
            }
        }

        if let Some(req_gpu_memory) = reqs.gpu_memory_gb {
            if caps.gpu_memory_gb.unwrap_or(0.0) < req_gpu_memory {
                return Ok(false);
            }
        }

        // Check storage requirements
        if let Some(req_storage) = reqs.storage_gb {
            if caps.storage_gb < req_storage {
                return Ok(false);
            }
        }

        // Check network requirements
        if let Some(req_bandwidth) = reqs.network_bandwidth_mbps {
            if caps.network_bandwidth_mbps < req_bandwidth {
                return Ok(false);
            }
        }

        // Check GPU Direct requirement
        if reqs.requires_gpu_direct && !caps.supports_gpu_direct {
            return Ok(false);
        }

        // Check battery safety
        if !reqs.battery_safe && caps.battery_powered {
            return Ok(false);
        }

        // Check node affinity
        if let Some(ref affinity) = reqs.node_affinity {
            match affinity {
                NodeAffinity::RequiredClass(ref class) => {
                    if node.class != *class {
                        return Ok(false);
                    }
                }
                NodeAffinity::RequiredNodes(ref nodes) => {
                    if !nodes.contains(&node.id) {
                        return Ok(false);
                    }
                }
                _ => {} // Preferences handled in scoring
            }
        }

        // Check anti-affinity
        if let Some(ref anti_affinity) = reqs.anti_affinity {
            if anti_affinity.contains(&node.id) {
                return Ok(false);
            }
        }

        Ok(true)
    }

    /// Select best fit node
    async fn select_best_fit<'a>(
        &self,
        job: &Job,
        nodes: &[&'a ClusterNode],
    ) -> Result<&'a ClusterNode> {
        let mut best_node = nodes[0];
        let mut best_score = self.calculate_fit_score(job, best_node).await?;

        for node in nodes.iter().skip(1) {
            let score = self.calculate_fit_score(job, node).await?;
            if score > best_score {
                best_score = score;
                best_node = node;
            }
        }

        Ok(best_node)
    }

    /// Calculate fit score for a job on a node
    async fn calculate_fit_score(&self, job: &Job, node: &ClusterNode) -> Result<f32> {
        let mut score = 100.0;

        // Penalize based on current load
        if let Some(load) = self.node_loads.get(&node.id) {
            score -= load.cpu_usage * 20.0;
            score -= load.memory_usage * 20.0;
            score -= load.gpu_usage * 20.0;
        }

        // Bonus for matching preferred affinity
        if let Some(NodeAffinity::PreferredClass(ref class)) = job.requirements.node_affinity {
            if node.class == *class {
                score += 20.0;
            }
        }

        // Bonus for locality
        match job.requirements.locality_preference {
            LocalityPreference::DataLocal => {
                // Would check data locality
                score += 10.0;
            }
            _ => {}
        }

        Ok(score)
    }

    /// Round-robin selection
    async fn select_round_robin<'a>(&self, nodes: &[&'a ClusterNode]) -> Result<&'a ClusterNode> {
        // Simple round-robin based on job count
        let mut min_jobs = usize::MAX;
        let mut selected = nodes[0];

        for node in nodes {
            let job_count = self
                .node_loads
                .get(&node.id)
                .map(|entry| entry.active_jobs.len())
                .unwrap_or(0);

            if job_count < min_jobs {
                min_jobs = job_count;
                selected = node;
            }
        }

        Ok(selected)
    }

    /// Power-aware selection
    async fn select_power_aware<'a>(
        &self,
        _job: &Job,
        nodes: &[&'a ClusterNode],
    ) -> Result<&'a ClusterNode> {
        // Prefer non-battery nodes
        for node in nodes {
            if !node.capabilities.battery_powered {
                return Ok(node);
            }
        }

        // If all battery-powered, select one with best power status
        // For now, just return first
        Ok(nodes[0])
    }

    /// Cost-optimized selection
    async fn select_cost_optimized<'a>(
        &self,
        _job: &Job,
        nodes: &[&'a ClusterNode],
    ) -> Result<&'a ClusterNode> {
        // Would consider spot pricing, energy costs, etc.
        // For now, prefer smaller nodes
        let mut selected = nodes[0];
        let mut min_cores = selected.capabilities.cpu_cores;

        for node in nodes {
            if node.capabilities.cpu_cores < min_cores {
                min_cores = node.capabilities.cpu_cores;
                selected = node;
            }
        }

        Ok(selected)
    }

    /// Latency-optimized selection
    async fn select_latency_optimized<'a>(
        &self,
        job: &Job,
        nodes: &[&'a ClusterNode],
    ) -> Result<&'a ClusterNode> {
        // Select node with lowest network latency
        let mut selected = nodes[0];
        let mut min_latency = selected.network.latency_ms;

        for node in nodes {
            if let Some(max_latency) = job.requirements.max_latency_ms {
                if node.network.latency_ms > max_latency {
                    continue;
                }
            }

            if node.network.latency_ms < min_latency {
                min_latency = node.network.latency_ms;
                selected = node;
            }
        }

        Ok(selected)
    }

    /// Schedule a job on a specific node
    pub async fn schedule_on_node(&self, job: Job, node_id: Uuid) -> Result<Uuid> {
        let scheduled_job = ScheduledJob {
            job: job.clone(),
            node_id,
            scheduled_at: Utc::now(),
            estimated_completion: None,
            actual_start: None,
            actual_completion: None,
            status: JobStatus::Scheduled,
        };

        // Add to scheduled jobs
        self.scheduled_jobs.insert(job.id, scheduled_job);

        // Update node load
        self.node_loads
            .entry(node_id)
            .or_default()
            .active_jobs
            .push(job.id);
        if let Some(mut load) = self.node_loads.get_mut(&node_id) {
            load.last_updated = Some(Utc::now());
        }

        Ok(job.id)
    }

    /// Migrate work from a node
    pub async fn migrate_work_from_node(&self, node_id: Uuid) -> Result<()> {
        let jobs_to_migrate: Vec<Uuid> = self
            .scheduled_jobs
            .iter()
            .filter(|entry| entry.node_id == node_id && entry.status == JobStatus::Running)
            .map(|entry| entry.job.id)
            .collect();

        for job_id in jobs_to_migrate {
            self.migrate_job(job_id).await?;
        }

        Ok(())
    }

    /// Migrate a specific job
    async fn migrate_job(&self, job_id: Uuid) -> Result<()> {
        if let Some(mut scheduled_job) = self.scheduled_jobs.get_mut(&job_id) {
            scheduled_job.status = JobStatus::Migrating;

            // In a real implementation, this would:
            // 1. Checkpoint the job state
            // 2. Find a new node
            // 3. Transfer state to new node
            // 4. Resume execution

            tracing::info!(
                "Migrating job {} from node {}",
                job_id,
                scheduled_job.node_id
            );
        }

        Ok(())
    }

    /// Check and rebalance load
    async fn check_and_rebalance(&self) -> Result<()> {
        for entry in self.node_loads.iter() {
            let node_id = entry.key();
            let load = entry.value();
            let total_usage = (load.cpu_usage + load.memory_usage + load.gpu_usage) / 3.0;

            if total_usage > self.migration_threshold {
                tracing::warn!(
                    "Node {} is overloaded ({}% usage), considering migration",
                    node_id,
                    total_usage * 100.0
                );

                // Would trigger job migration here
            }
        }

        Ok(())
    }

    /// Update node load information
    pub async fn update_node_load(
        &self,
        node_id: Uuid,
        cpu_usage: f32,
        memory_usage: f32,
        gpu_usage: f32,
    ) -> Result<()> {
        self.node_loads
            .entry(node_id)
            .and_modify(|load| {
                load.cpu_usage = cpu_usage;
                load.memory_usage = memory_usage;
                load.gpu_usage = gpu_usage;
                load.last_updated = Some(Utc::now());
            })
            .or_insert_with(|| NodeLoad {
                cpu_usage,
                memory_usage,
                gpu_usage,
                active_jobs: Vec::new(),
                last_updated: Some(Utc::now()),
            });

        Ok(())
    }

    /// Get scheduling statistics
    pub async fn get_statistics(&self) -> SchedulingStatistics {
        let queue = self.job_queue.lock().await;

        let mut stats = SchedulingStatistics::default();
        stats.queued_jobs = queue.len();
        stats.total_jobs = self.scheduled_jobs.len();

        for entry in self.scheduled_jobs.iter() {
            match entry.value().status {
                JobStatus::Running => stats.running_jobs += 1,
                JobStatus::Completed => stats.completed_jobs += 1,
                JobStatus::Failed(_) => stats.failed_jobs += 1,
                _ => {}
            }
        }

        stats
    }
}

impl Default for WorkDistributor {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for JobRequirements {
    fn default() -> Self {
        Self {
            cpu_cores: None,
            memory_gb: None,
            gpu_count: None,
            gpu_memory_gb: None,
            storage_gb: None,
            network_bandwidth_mbps: None,
            requires_gpu_direct: false,
            node_affinity: None,
            anti_affinity: None,
            locality_preference: LocalityPreference::None,
            max_latency_ms: None,
            battery_safe: true,
        }
    }
}

/// Scheduling statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SchedulingStatistics {
    pub total_jobs: usize,
    pub queued_jobs: usize,
    pub running_jobs: usize,
    pub completed_jobs: usize,
    pub failed_jobs: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discovery::{HardwareProfile, NatType, NetworkCharacteristics};
    use crate::{
        classification::{NodeClass, Schedule, GPU},
        JobPriority,
    };

    fn create_test_node(id: Uuid, cores: u32, memory: f32, gpu_count: u32) -> ClusterNode {
        ClusterNode {
            id,
            hostname: format!("node-{}", id),
            class: NodeClass::Workstation {
                gpu: if gpu_count > 0 {
                    Some(GPU {
                        model: "Test GPU".to_string(),
                        memory_gb: 8.0,
                        compute_capability: (7, 5),
                    })
                } else {
                    None
                },
                schedule: Schedule::AlwaysOn,
            },
            hardware: HardwareProfile {
                cpu_model: "Test CPU".to_string(),
                cpu_cores: cores,
                memory_gb: memory,
                storage_gb: 1000.0,
                gpus: vec![],
            },
            network: NetworkCharacteristics {
                bandwidth_mbps: 1000.0,
                latency_ms: 1.0,
                jitter_ms: 0.1,
                packet_loss: 0.0,
                nat_type: NatType::None,
            },
            status: NodeStatus::Online,
            capabilities: NodeCapabilities {
                cpu_cores: cores,
                memory_gb: memory,
                gpu_count,
                gpu_memory_gb: if gpu_count > 0 { Some(8.0) } else { None },
                storage_gb: 1000.0,
                network_bandwidth_mbps: 1000.0,
                supports_gpu_direct: gpu_count > 0,
                battery_powered: false,
                thermal_constraints: None,
            },
            last_heartbeat: Utc::now(),
        }
    }

    fn create_test_job(cpu: u32, memory: f32, gpu: u32) -> Job {
        Job {
            id: Uuid::new_v4(),
            name: "Test Job".to_string(),
            requirements: JobRequirements {
                cpu_cores: Some(cpu),
                memory_gb: Some(memory),
                gpu_count: Some(gpu),
                gpu_memory_gb: if gpu > 0 { Some(4.0) } else { None },
                storage_gb: Some(100.0),
                network_bandwidth_mbps: Some(100.0),
                requires_gpu_direct: false,
                node_affinity: None,
                anti_affinity: None,
                locality_preference: LocalityPreference::None,
                max_latency_ms: None,
                battery_safe: true,
            },
            priority: JobPriority::Normal,
            submitted_at: Utc::now(),
        }
    }

    #[tokio::test]
    async fn test_meets_requirements() {
        let distributor = WorkDistributor::new();
        let node = create_test_node(Uuid::new_v4(), 16, 32.0, 1);

        // Job that fits
        let job = create_test_job(8, 16.0, 0);
        assert!(distributor
            .meets_requirements(&job.requirements, &node.capabilities, &node)
            .unwrap());

        // Job that needs too much CPU
        let job = create_test_job(32, 16.0, 0);
        assert!(!distributor
            .meets_requirements(&job.requirements, &node.capabilities, &node)
            .unwrap());

        // Job that needs GPU
        let job = create_test_job(8, 16.0, 2);
        assert!(!distributor
            .meets_requirements(&job.requirements, &node.capabilities, &node)
            .unwrap());
    }

    #[tokio::test]
    async fn test_node_selection() {
        let distributor = WorkDistributor::new();

        let nodes = vec![
            create_test_node(Uuid::new_v4(), 8, 16.0, 0),
            create_test_node(Uuid::new_v4(), 16, 32.0, 1),
            create_test_node(Uuid::new_v4(), 32, 64.0, 2),
        ];

        // Small job should work on any node
        let job = create_test_job(4, 8.0, 0);
        let selected = distributor.select_node(&job, &nodes).await.unwrap();
        assert!(selected.capabilities.cpu_cores >= 4);

        // GPU job should select GPU node
        let job = create_test_job(8, 16.0, 1);
        let selected = distributor.select_node(&job, &nodes).await.unwrap();
        assert!(selected.capabilities.gpu_count >= 1);

        // Large job should select large node
        let job = create_test_job(24, 48.0, 0);
        let selected = distributor.select_node(&job, &nodes).await.unwrap();
        assert_eq!(selected.capabilities.cpu_cores, 32);
    }

    #[tokio::test]
    async fn test_scheduling_policies() {
        let distributor = WorkDistributor::new();

        let nodes = vec![
            create_test_node(Uuid::new_v4(), 8, 16.0, 0),
            create_test_node(Uuid::new_v4(), 16, 32.0, 0),
        ];

        let job = create_test_job(4, 8.0, 0);

        // Test FirstFit
        *distributor.policy.write().await = SchedulingPolicy::FirstFit;
        let selected = distributor.select_node(&job, &nodes).await.unwrap();
        assert_eq!(selected.id, nodes[0].id);

        // Test CostOptimized (prefers smaller nodes)
        *distributor.policy.write().await = SchedulingPolicy::CostOptimized;
        let selected = distributor.select_node(&job, &nodes).await.unwrap();
        assert_eq!(selected.capabilities.cpu_cores, 8);
    }

    #[tokio::test]
    async fn test_job_scheduling() {
        let distributor = WorkDistributor::new();
        let node_id = Uuid::new_v4();
        let job = create_test_job(4, 8.0, 0);

        let job_id = distributor
            .schedule_on_node(job.clone(), node_id)
            .await
            .unwrap();
        assert_eq!(job_id, job.id);

        // Check job is scheduled
        assert!(distributor.scheduled_jobs.contains_key(&job_id));

        // Check node load is updated
        let load = distributor.node_loads.get(&node_id).unwrap();
        assert!(load.active_jobs.contains(&job_id));
    }

    #[tokio::test]
    async fn test_work_migration() {
        let distributor = WorkDistributor::new();
        let node_id = Uuid::new_v4();

        // Schedule some jobs
        for _i in 0..3 {
            let job = create_test_job(2, 4.0, 0);
            distributor.schedule_on_node(job, node_id).await.unwrap();
        }

        // Migrate work from node
        distributor.migrate_work_from_node(node_id).await.unwrap();

        // In a real implementation, jobs would be rescheduled
    }

    #[tokio::test]
    async fn test_load_update() {
        let distributor = WorkDistributor::new();
        let node_id = Uuid::new_v4();

        distributor
            .update_node_load(node_id, 0.5, 0.6, 0.3)
            .await
            .unwrap();

        let load = distributor.node_loads.get(&node_id).unwrap();
        assert_eq!(load.cpu_usage, 0.5);
        assert_eq!(load.memory_usage, 0.6);
        assert_eq!(load.gpu_usage, 0.3);
    }

    #[test]
    fn test_job_priority_heap() {
        let mut heap = BinaryHeap::new();

        heap.push(PrioritizedJob {
            job: create_test_job(1, 1.0, 0),
            score: 10,
        });

        heap.push(PrioritizedJob {
            job: create_test_job(2, 2.0, 0),
            score: 20,
        });

        heap.push(PrioritizedJob {
            job: create_test_job(3, 3.0, 0),
            score: 15,
        });

        // Should pop in order of score
        assert_eq!(heap.pop().unwrap().score, 20);
        assert_eq!(heap.pop().unwrap().score, 15);
        assert_eq!(heap.pop().unwrap().score, 10);
    }
}

#[cfg(test)]
#[path = "distribution_tests.rs"]
mod distribution_coverage_tests;
