//! GPU optimization module for allocation strategies, scheduling, and utilization optimization
//!
//! This module provides advanced GPU resource management including:
//! - Multi-GPU allocation strategies (best-fit, worst-fit, round-robin)
//! - GPU sharing and time-slicing mechanisms
//! - Dynamic GPU scheduling based on workload requirements
//! - GPU memory management and fragmentation prevention

use crate::error::{CostOptimizationError, CostOptimizationResult};
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use ordered_float::OrderedFloat;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{Mutex, Semaphore};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

/// GPU device information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuDevice {
    /// Device ID (e.g., "GPU-0", "GPU-1")
    pub device_id: String,
    /// Device model name
    pub model: String,
    /// Total memory in GB
    pub total_memory: f64,
    /// Available memory in GB
    pub available_memory: f64,
    /// Compute capability (e.g., 8.6 for A100)
    pub compute_capability: f32,
    /// Current utilization percentage
    pub utilization: f64,
    /// Power limit in watts
    pub power_limit: u32,
    /// Current temperature in Celsius
    pub temperature: u32,
    /// PCI bus ID
    pub pci_bus_id: String,
    /// NUMA node affinity
    pub numa_node: Option<u32>,
}

/// GPU allocation request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuAllocationRequest {
    /// Request ID
    pub request_id: Uuid,
    /// Workload ID
    pub workload_id: String,
    /// Required memory in GB
    pub memory_required: f64,
    /// Required compute units (0-100)
    pub compute_required: f64,
    /// Preferred GPU models
    pub preferred_models: Vec<String>,
    /// Exclusive allocation required
    pub exclusive: bool,
    /// Maximum allocation duration
    pub duration: Duration,
    /// Priority (higher is more important)
    pub priority: u32,
    /// Affinity requirements
    pub affinity: GpuAffinity,
}

/// GPU affinity requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuAffinity {
    /// Prefer GPUs on same NUMA node
    pub numa_aware: bool,
    /// Prefer GPUs with NVLink connections
    pub nvlink_required: bool,
    /// Required minimum compute capability
    pub min_compute_capability: Option<f32>,
    /// Anti-affinity workload IDs
    pub anti_affinity: Vec<String>,
}

/// GPU allocation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuAllocation {
    /// Allocation ID
    pub allocation_id: Uuid,
    /// Request ID
    pub request_id: Uuid,
    /// Allocated GPU device IDs
    pub device_ids: Vec<String>,
    /// Allocation timestamp
    pub allocated_at: DateTime<Utc>,
    /// Expiration timestamp
    pub expires_at: DateTime<Utc>,
    /// Memory allocated per device
    pub memory_per_device: HashMap<String, f64>,
    /// Compute allocated per device
    pub compute_per_device: HashMap<String, f64>,
    /// Allocation cost per hour
    pub cost_per_hour: f64,
}

/// GPU allocation strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AllocationStrategy {
    /// Best-fit: minimize fragmentation
    BestFit,
    /// Worst-fit: maximize remaining space
    WorstFit,
    /// First-fit: use first available
    FirstFit,
    /// Round-robin: distribute evenly
    RoundRobin,
    /// Power-aware: minimize power consumption
    PowerAware,
    /// Cost-aware: minimize cost
    CostAware,
}

/// GPU sharing policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuSharingPolicy {
    /// Enable Multi-Instance GPU (MIG)
    pub enable_mig: bool,
    /// Enable time-slicing
    pub enable_time_slicing: bool,
    /// Maximum concurrent allocations per GPU
    pub max_sharing: u32,
    /// Minimum memory per allocation
    pub min_memory_per_allocation: f64,
    /// Memory oversubscription ratio
    pub memory_oversubscription: f64,
}

impl Default for GpuSharingPolicy {
    fn default() -> Self {
        Self {
            enable_mig: true,
            enable_time_slicing: true,
            max_sharing: 4,
            min_memory_per_allocation: 4.0, // 4GB minimum
            memory_oversubscription: 1.2,   // 20% oversubscription
        }
    }
}

/// GPU optimizer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuOptimizerConfig {
    /// Allocation strategy
    pub strategy: AllocationStrategy,
    /// GPU sharing policy
    pub sharing_policy: GpuSharingPolicy,
    /// Enable predictive scaling
    pub enable_predictive_scaling: bool,
    /// Rebalancing interval
    pub rebalance_interval: Duration,
    /// Cost per GPU hour by model
    pub gpu_costs: HashMap<String, f64>,
    /// Default cost per GPU hour
    pub default_gpu_cost: f64,
}

impl Default for GpuOptimizerConfig {
    fn default() -> Self {
        let mut gpu_costs = HashMap::new();
        gpu_costs.insert("A100".to_string(), 3.06);
        gpu_costs.insert("V100".to_string(), 2.48);
        gpu_costs.insert("T4".to_string(), 0.526);
        gpu_costs.insert("A10G".to_string(), 1.212);

        Self {
            strategy: AllocationStrategy::BestFit,
            sharing_policy: GpuSharingPolicy::default(),
            enable_predictive_scaling: true,
            rebalance_interval: Duration::from_secs(300),
            gpu_costs,
            default_gpu_cost: 2.0,
        }
    }
}

/// GPU device state
#[derive(Debug)]
struct GpuDeviceState {
    /// Device information
    device: GpuDevice,
    /// Active allocations
    allocations: HashMap<Uuid, GpuAllocation>,
    /// Allocation semaphore for time-slicing
    time_slice_semaphore: Arc<Semaphore>,
    /// Last allocation time
    last_allocation: DateTime<Utc>,
}

/// GPU optimizer for intelligent resource allocation
pub struct GpuOptimizer {
    /// Configuration
    config: Arc<GpuOptimizerConfig>,
    /// GPU devices by ID
    devices: Arc<DashMap<String, Arc<Mutex<GpuDeviceState>>>>,
    /// Active allocations by ID
    allocations: Arc<DashMap<Uuid, GpuAllocation>>,
    /// Pending requests queue
    pending_requests: Arc<Mutex<BinaryHeap<PrioritizedRequest>>>,
    /// Round-robin state
    round_robin_index: Arc<RwLock<usize>>,
    /// Metrics
    metrics: Arc<RwLock<GpuOptimizerMetrics>>,
}

/// Prioritized allocation request
#[derive(Debug, Clone)]
struct PrioritizedRequest {
    request: GpuAllocationRequest,
    priority_score: OrderedFloat<f64>,
}

impl PartialEq for PrioritizedRequest {
    fn eq(&self, other: &Self) -> bool {
        self.priority_score == other.priority_score
    }
}

impl Eq for PrioritizedRequest {}

impl PartialOrd for PrioritizedRequest {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PrioritizedRequest {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.priority_score.cmp(&other.priority_score)
    }
}

/// GPU optimizer metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GpuOptimizerMetrics {
    /// Total allocations
    pub total_allocations: u64,
    /// Failed allocations
    pub failed_allocations: u64,
    /// Average utilization
    pub avg_utilization: f64,
    /// Peak utilization
    pub peak_utilization: f64,
    /// Total cost
    pub total_cost: f64,
    /// Fragmentation ratio
    pub fragmentation_ratio: f64,
}

impl GpuOptimizer {
    /// Create a new GPU optimizer
    pub fn new(config: GpuOptimizerConfig) -> CostOptimizationResult<Self> {
        Ok(Self {
            config: Arc::new(config),
            devices: Arc::new(DashMap::new()),
            allocations: Arc::new(DashMap::new()),
            pending_requests: Arc::new(Mutex::new(BinaryHeap::new())),
            round_robin_index: Arc::new(RwLock::new(0)),
            metrics: Arc::new(RwLock::new(GpuOptimizerMetrics::default())),
        })
    }

    /// Register a GPU device
    pub async fn register_device(&self, device: GpuDevice) -> CostOptimizationResult<()> {
        info!("Registering GPU device: {}", device.device_id);

        let max_sharing = self.config.sharing_policy.max_sharing as usize;
        let state = GpuDeviceState {
            device: device.clone(),
            allocations: HashMap::new(),
            time_slice_semaphore: Arc::new(Semaphore::new(max_sharing)),
            last_allocation: Utc::now(),
        };

        self.devices
            .insert(device.device_id.clone(), Arc::new(Mutex::new(state)));

        Ok(())
    }

    /// Unregister a GPU device
    pub async fn unregister_device(&self, device_id: &str) -> CostOptimizationResult<()> {
        info!("Unregistering GPU device: {}", device_id);

        // Check for active allocations
        if let Some(device_state) = self.devices.get(device_id) {
            let state = device_state.lock().await;
            if !state.allocations.is_empty() {
                return Err(CostOptimizationError::GpuUnavailable {
                    gpu_id: device_id.to_string(),
                });
            }
        }

        self.devices.remove(device_id);
        Ok(())
    }

    /// Request GPU allocation
    pub async fn allocate(
        &self,
        request: GpuAllocationRequest,
    ) -> CostOptimizationResult<GpuAllocation> {
        info!("Processing GPU allocation request: {}", request.request_id);

        // Try immediate allocation
        match self.try_allocate(&request).await {
            Ok(allocation) => {
                self.update_metrics_on_allocation(&allocation);
                Ok(allocation)
            }
            Err(e) => {
                // Queue request if immediate allocation fails
                warn!("Immediate allocation failed, queuing request: {}", e);
                self.queue_request(request.clone()).await;

                // Try once more after queuing
                self.process_pending_requests().await?;

                self.allocations
                    .get(&request.request_id)
                    .map(|entry| entry.value().clone())
                    .ok_or_else(|| CostOptimizationError::AllocationFailed {
                        reason: "No available GPUs for allocation".to_string(),
                    })
            }
        }
    }

    /// Try to allocate GPUs based on strategy
    async fn try_allocate(
        &self,
        request: &GpuAllocationRequest,
    ) -> CostOptimizationResult<GpuAllocation> {
        match self.config.strategy {
            AllocationStrategy::BestFit => self.allocate_best_fit(request).await,
            AllocationStrategy::WorstFit => self.allocate_worst_fit(request).await,
            AllocationStrategy::FirstFit => self.allocate_first_fit(request).await,
            AllocationStrategy::RoundRobin => self.allocate_round_robin(request).await,
            AllocationStrategy::PowerAware => self.allocate_power_aware(request).await,
            AllocationStrategy::CostAware => self.allocate_cost_aware(request).await,
        }
    }

    /// Best-fit allocation strategy
    async fn allocate_best_fit(
        &self,
        request: &GpuAllocationRequest,
    ) -> CostOptimizationResult<GpuAllocation> {
        let mut candidates = Vec::new();

        for device_entry in self.devices.iter() {
            let device_state = device_entry.lock().await;

            if self.can_accommodate(&device_state, request) {
                let remaining = device_state.device.available_memory - request.memory_required;
                candidates.push((device_entry.key().clone(), remaining));
            }
        }

        // Sort by least remaining space (best fit)
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        if let Some((device_id, _)) = candidates.first() {
            self.create_allocation(vec![device_id.clone()], request)
                .await
        } else {
            Err(CostOptimizationError::NoAvailableResources {
                resource_type: "GPU".to_string(),
            })
        }
    }

    /// Worst-fit allocation strategy
    async fn allocate_worst_fit(
        &self,
        request: &GpuAllocationRequest,
    ) -> CostOptimizationResult<GpuAllocation> {
        let mut candidates = Vec::new();

        for device_entry in self.devices.iter() {
            let device_state = device_entry.lock().await;

            if self.can_accommodate(&device_state, request) {
                let remaining = device_state.device.available_memory - request.memory_required;
                candidates.push((device_entry.key().clone(), remaining));
            }
        }

        // Sort by most remaining space (worst fit)
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        if let Some((device_id, _)) = candidates.first() {
            self.create_allocation(vec![device_id.clone()], request)
                .await
        } else {
            Err(CostOptimizationError::NoAvailableResources {
                resource_type: "GPU".to_string(),
            })
        }
    }

    /// First-fit allocation strategy
    async fn allocate_first_fit(
        &self,
        request: &GpuAllocationRequest,
    ) -> CostOptimizationResult<GpuAllocation> {
        for device_entry in self.devices.iter() {
            let device_state = device_entry.lock().await;

            if self.can_accommodate(&device_state, request) {
                return self
                    .create_allocation(vec![device_entry.key().clone()], request)
                    .await;
            }
        }

        Err(CostOptimizationError::NoAvailableResources {
            resource_type: "GPU".to_string(),
        })
    }

    /// Round-robin allocation strategy
    async fn allocate_round_robin(
        &self,
        request: &GpuAllocationRequest,
    ) -> CostOptimizationResult<GpuAllocation> {
        let device_count = self.devices.len();
        if device_count == 0 {
            return Err(CostOptimizationError::NoAvailableResources {
                resource_type: "GPU".to_string(),
            });
        }

        let start_index = *self.round_robin_index.read() % device_count;
        let mut current_index = start_index;

        loop {
            if let Some(device_entry) = self.devices.iter().nth(current_index) {
                let device_state = device_entry.lock().await;

                if self.can_accommodate(&device_state, request) {
                    *self.round_robin_index.write() = (current_index + 1) % device_count;
                    return self
                        .create_allocation(vec![device_entry.key().clone()], request)
                        .await;
                }
            }

            current_index = (current_index + 1) % device_count;
            if current_index == start_index {
                break;
            }
        }

        Err(CostOptimizationError::NoAvailableResources {
            resource_type: "GPU".to_string(),
        })
    }

    /// Power-aware allocation strategy
    async fn allocate_power_aware(
        &self,
        request: &GpuAllocationRequest,
    ) -> CostOptimizationResult<GpuAllocation> {
        let mut candidates = Vec::new();

        for device_entry in self.devices.iter() {
            let device_state = device_entry.lock().await;

            if self.can_accommodate(&device_state, request) {
                // Score based on power efficiency (lower temperature = better)
                let power_score = device_state.device.temperature as f64;
                candidates.push((device_entry.key().clone(), power_score));
            }
        }

        // Sort by lowest power/temperature
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        if let Some((device_id, _)) = candidates.first() {
            self.create_allocation(vec![device_id.clone()], request)
                .await
        } else {
            Err(CostOptimizationError::NoAvailableResources {
                resource_type: "GPU".to_string(),
            })
        }
    }

    /// Cost-aware allocation strategy
    async fn allocate_cost_aware(
        &self,
        request: &GpuAllocationRequest,
    ) -> CostOptimizationResult<GpuAllocation> {
        let mut candidates = Vec::new();

        for device_entry in self.devices.iter() {
            let device_state = device_entry.lock().await;

            if self.can_accommodate(&device_state, request) {
                let cost = self.calculate_gpu_cost(&device_state.device);
                candidates.push((device_entry.key().clone(), cost));
            }
        }

        // Sort by lowest cost
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        if let Some((device_id, _)) = candidates.first() {
            self.create_allocation(vec![device_id.clone()], request)
                .await
        } else {
            Err(CostOptimizationError::NoAvailableResources {
                resource_type: "GPU".to_string(),
            })
        }
    }

    /// Check if a device can accommodate a request
    fn can_accommodate(
        &self,
        device_state: &GpuDeviceState,
        request: &GpuAllocationRequest,
    ) -> bool {
        // Check memory availability
        let memory_available = device_state.device.available_memory >= request.memory_required;

        // Check compute availability
        let compute_available =
            (100.0 - device_state.device.utilization) >= request.compute_required;

        // Check exclusive requirements
        let exclusive_ok = !request.exclusive || device_state.allocations.is_empty();

        // Check sharing limits
        let sharing_ok =
            device_state.allocations.len() < self.config.sharing_policy.max_sharing as usize;

        // Check model preferences
        let model_ok = request.preferred_models.is_empty()
            || request
                .preferred_models
                .contains(&device_state.device.model);

        // Check compute capability
        let compute_cap_ok = request
            .affinity
            .min_compute_capability
            .map(|min| device_state.device.compute_capability >= min)
            .unwrap_or(true);

        memory_available
            && compute_available
            && exclusive_ok
            && sharing_ok
            && model_ok
            && compute_cap_ok
    }

    /// Create an allocation
    async fn create_allocation(
        &self,
        device_ids: Vec<String>,
        request: &GpuAllocationRequest,
    ) -> CostOptimizationResult<GpuAllocation> {
        let allocation_id = Uuid::new_v4();
        let now = Utc::now();
        let expires_at = now + chrono::Duration::from_std(request.duration)?;

        let mut memory_per_device = HashMap::new();
        let mut compute_per_device = HashMap::new();
        let mut total_cost = 0.0;

        // Allocate resources on each device
        for device_id in &device_ids {
            if let Some(device_entry) = self.devices.get(device_id) {
                let mut device_state = device_entry.lock().await;

                // Update device state
                device_state.device.available_memory -= request.memory_required;
                device_state.device.utilization += request.compute_required;

                memory_per_device.insert(device_id.clone(), request.memory_required);
                compute_per_device.insert(device_id.clone(), request.compute_required);

                // Calculate cost
                total_cost += self.calculate_gpu_cost(&device_state.device);

                // Track allocation
                let allocation = GpuAllocation {
                    allocation_id,
                    request_id: request.request_id,
                    device_ids: device_ids.clone(),
                    allocated_at: now,
                    expires_at,
                    memory_per_device: memory_per_device.clone(),
                    compute_per_device: compute_per_device.clone(),
                    cost_per_hour: total_cost,
                };

                device_state
                    .allocations
                    .insert(allocation_id, allocation.clone());
                device_state.last_allocation = now;
            }
        }

        let allocation = GpuAllocation {
            allocation_id,
            request_id: request.request_id,
            device_ids,
            allocated_at: now,
            expires_at,
            memory_per_device,
            compute_per_device,
            cost_per_hour: total_cost,
        };

        self.allocations.insert(allocation_id, allocation.clone());

        info!(
            "Created GPU allocation: {} for request: {}",
            allocation_id, request.request_id
        );

        Ok(allocation)
    }

    /// Release a GPU allocation
    pub async fn release(&self, allocation_id: Uuid) -> CostOptimizationResult<()> {
        info!("Releasing GPU allocation: {}", allocation_id);

        let allocation = self
            .allocations
            .remove(&allocation_id)
            .map(|(_, v)| v)
            .ok_or_else(|| CostOptimizationError::AllocationFailed {
                reason: format!("Allocation {} not found", allocation_id),
            })?;

        // Release resources on each device
        for device_id in &allocation.device_ids {
            if let Some(device_entry) = self.devices.get(device_id) {
                let mut device_state = device_entry.lock().await;

                // Update device state
                if let Some(memory) = allocation.memory_per_device.get(device_id) {
                    device_state.device.available_memory += memory;
                }

                if let Some(compute) = allocation.compute_per_device.get(device_id) {
                    device_state.device.utilization -= compute;
                }

                device_state.allocations.remove(&allocation_id);
            }
        }

        // Process any pending requests
        self.process_pending_requests().await?;

        Ok(())
    }

    /// Queue a pending request
    async fn queue_request(&self, request: GpuAllocationRequest) {
        let priority_score = OrderedFloat(
            (request.priority as f64) * 1000.0 + (1.0 / request.memory_required) * 10.0,
        );

        let prioritized = PrioritizedRequest {
            request,
            priority_score,
        };

        self.pending_requests.lock().await.push(prioritized);
    }

    /// Process pending allocation requests
    async fn process_pending_requests(&self) -> CostOptimizationResult<()> {
        let mut pending = self.pending_requests.lock().await;
        let mut processed = Vec::new();

        while let Some(prioritized) = pending.pop() {
            match self.try_allocate(&prioritized.request).await {
                Ok(allocation) => {
                    info!(
                        "Processed pending request: {}",
                        prioritized.request.request_id
                    );
                    self.update_metrics_on_allocation(&allocation);
                    processed.push(prioritized.request.request_id);
                }
                Err(_) => {
                    // Re-queue if still can't allocate
                    pending.push(prioritized);
                    break;
                }
            }
        }

        Ok(())
    }

    /// Calculate GPU cost per hour
    fn calculate_gpu_cost(&self, device: &GpuDevice) -> f64 {
        self.config
            .gpu_costs
            .get(&device.model)
            .copied()
            .unwrap_or(self.config.default_gpu_cost)
    }

    /// Update metrics on allocation
    fn update_metrics_on_allocation(&self, allocation: &GpuAllocation) {
        let mut metrics = self.metrics.write();
        metrics.total_allocations += 1;
        metrics.total_cost += allocation.cost_per_hour;

        // Update utilization metrics
        let total_utilization: f64 = self
            .devices
            .iter()
            .map(|entry| {
                entry
                    .value()
                    .try_lock()
                    .map(|state| state.device.utilization)
                    .unwrap_or(0.0)
            })
            .sum();

        let device_count = self.devices.len() as f64;
        if device_count > 0.0 {
            metrics.avg_utilization = total_utilization / device_count;
            metrics.peak_utilization = metrics.peak_utilization.max(metrics.avg_utilization);
        }
    }

    /// Get current optimizer metrics
    pub fn get_metrics(&self) -> GpuOptimizerMetrics {
        self.metrics.read().clone()
    }

    /// Get device status
    pub async fn get_device_status(&self, device_id: &str) -> Option<GpuDevice> {
        self.devices
            .get(device_id)
            .and_then(|entry| entry.try_lock().ok().map(|state| state.device.clone()))
    }

    /// Get all device statuses
    pub async fn get_all_devices(&self) -> Vec<GpuDevice> {
        let mut devices = Vec::new();

        for entry in self.devices.iter() {
            if let Ok(state) = entry.try_lock() {
                devices.push(state.device.clone());
            }
        }

        devices
    }

    /// Rebalance GPU allocations
    pub async fn rebalance(&self) -> CostOptimizationResult<()> {
        info!("Rebalancing GPU allocations");

        // Collect current allocations
        let mut allocations_to_migrate = Vec::new();

        for device_entry in self.devices.iter() {
            let device_state = device_entry.lock().await;

            // Check if device is overutilized
            if device_state.device.utilization > 90.0 {
                for (_, allocation) in &device_state.allocations {
                    allocations_to_migrate.push(allocation.clone());
                }
            }
        }

        // Try to migrate allocations to less utilized devices
        for allocation in allocations_to_migrate {
            debug!(
                "Attempting to migrate allocation: {}",
                allocation.allocation_id
            );
            // Migration logic would go here
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_device(id: &str, memory: f64, utilization: f64) -> GpuDevice {
        GpuDevice {
            device_id: id.to_string(),
            model: "A100".to_string(),
            total_memory: memory,
            available_memory: memory,
            compute_capability: 8.6,
            utilization,
            power_limit: 400,
            temperature: 65,
            pci_bus_id: format!("0000:00:{:02x}.0", id.parse::<u8>().unwrap_or(0)),
            numa_node: Some(0),
        }
    }

    fn create_test_request(memory: f64, compute: f64, exclusive: bool) -> GpuAllocationRequest {
        GpuAllocationRequest {
            request_id: Uuid::new_v4(),
            workload_id: "test-workload".to_string(),
            memory_required: memory,
            compute_required: compute,
            preferred_models: vec![],
            exclusive,
            duration: Duration::from_secs(3600),
            priority: 1,
            affinity: GpuAffinity {
                numa_aware: false,
                nvlink_required: false,
                min_compute_capability: None,
                anti_affinity: vec![],
            },
        }
    }

    #[tokio::test]
    async fn test_gpu_optimizer_creation() {
        let config = GpuOptimizerConfig::default();
        let optimizer = GpuOptimizer::new(config);
        assert!(optimizer.is_ok());
    }

    #[tokio::test]
    async fn test_device_registration() {
        let config = GpuOptimizerConfig::default();
        let optimizer = GpuOptimizer::new(config).unwrap();

        let device = create_test_device("0", 80.0, 0.0);
        let result = optimizer.register_device(device).await;
        assert!(result.is_ok());

        let devices = optimizer.get_all_devices().await;
        assert_eq!(devices.len(), 1);
    }

    #[tokio::test]
    async fn test_simple_allocation() {
        let config = GpuOptimizerConfig::default();
        let optimizer = GpuOptimizer::new(config).unwrap();

        // Register a device
        let device = create_test_device("0", 80.0, 0.0);
        optimizer.register_device(device).await?;

        // Request allocation
        let request = create_test_request(20.0, 25.0, false);
        let result = optimizer.allocate(request).await;
        assert!(result.is_ok());

        let allocation = result.unwrap();
        assert_eq!(allocation.device_ids.len(), 1);
        assert_eq!(allocation.device_ids[0], "0");
    }

    #[tokio::test]
    async fn test_exclusive_allocation() {
        let config = GpuOptimizerConfig::default();
        let optimizer = GpuOptimizer::new(config).unwrap();

        // Register a device
        let device = create_test_device("0", 80.0, 0.0);
        optimizer.register_device(device).await?;

        // First allocation (non-exclusive)
        let request1 = create_test_request(20.0, 25.0, false);
        let allocation1 = optimizer.allocate(request1).await?;

        // Second allocation (exclusive) should fail
        let request2 = create_test_request(20.0, 25.0, true);
        let result2 = optimizer.allocate(request2).await;
        assert!(result2.is_err());

        // Release first allocation
        optimizer.release(allocation1.allocation_id).await.unwrap();

        // Now exclusive allocation should succeed
        let request3 = create_test_request(20.0, 25.0, true);
        let result3 = optimizer.allocate(request3).await;
        assert!(result3.is_ok());
    }

    #[tokio::test]
    async fn test_best_fit_strategy() {
        let mut config = GpuOptimizerConfig::default();
        config.strategy = AllocationStrategy::BestFit;
        let optimizer = GpuOptimizer::new(config).unwrap();

        // Register devices with different available memory
        optimizer
            .register_device(create_test_device("0", 40.0, 0.0))
            .await
            ?;
        optimizer
            .register_device(create_test_device("1", 80.0, 0.0))
            .await
            .unwrap();
        optimizer
            .register_device(create_test_device("2", 30.0, 0.0))
            .await
            .unwrap();

        // Request that fits best in device "2"
        let request = create_test_request(25.0, 10.0, false);
        let allocation = optimizer.allocate(request).await.unwrap();

        // Should allocate to device "2" (least remaining space)
        assert_eq!(allocation.device_ids[0], "2");
    }

    #[tokio::test]
    async fn test_worst_fit_strategy() {
        let mut config = GpuOptimizerConfig::default();
        config.strategy = AllocationStrategy::WorstFit;
        let optimizer = GpuOptimizer::new(config).unwrap();

        // Register devices with different available memory
        optimizer
            .register_device(create_test_device("0", 40.0, 0.0))
            .await
            ?;
        optimizer
            .register_device(create_test_device("1", 80.0, 0.0))
            .await
            .unwrap();
        optimizer
            .register_device(create_test_device("2", 30.0, 0.0))
            .await
            .unwrap();

        // Request
        let request = create_test_request(20.0, 10.0, false);
        let allocation = optimizer.allocate(request).await.unwrap();

        // Should allocate to device "1" (most remaining space)
        assert_eq!(allocation.device_ids[0], "1");
    }

    #[tokio::test]
    async fn test_round_robin_strategy() {
        let mut config = GpuOptimizerConfig::default();
        config.strategy = AllocationStrategy::RoundRobin;
        let optimizer = GpuOptimizer::new(config).unwrap();

        // Register multiple devices
        for i in 0..3 {
            optimizer
                .register_device(create_test_device(&i.to_string(), 80.0, 0.0))
                .await
                ?;
        }

        // Make multiple allocations
        let mut allocations = Vec::new();
        for _ in 0..3 {
            let request = create_test_request(10.0, 10.0, false);
            let allocation = optimizer.allocate(request).await.unwrap();
            allocations.push(allocation);
        }

        // Check that allocations are distributed
        let device_ids: Vec<_> = allocations
            .iter()
            .map(|a| a.device_ids[0].clone())
            .collect();

        assert!(device_ids.contains(&"0".to_string()));
        assert!(device_ids.contains(&"1".to_string()));
        assert!(device_ids.contains(&"2".to_string()));
    }

    #[tokio::test]
    async fn test_gpu_sharing() {
        let mut config = GpuOptimizerConfig::default();
        config.sharing_policy.max_sharing = 3;
        let optimizer = GpuOptimizer::new(config).unwrap();

        // Register a single device
        optimizer
            .register_device(create_test_device("0", 80.0, 0.0))
            .await
            ?;

        // Make multiple small allocations
        let mut allocations = Vec::new();
        for _ in 0..3 {
            let request = create_test_request(20.0, 20.0, false);
            let allocation = optimizer.allocate(request).await.unwrap();
            allocations.push(allocation);
        }

        // All should be on the same device
        for allocation in &allocations {
            assert_eq!(allocation.device_ids[0], "0");
        }

        // Fourth allocation should fail (exceeds max_sharing)
        let request = create_test_request(10.0, 10.0, false);
        let result = optimizer.allocate(request).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_allocation_release() {
        let config = GpuOptimizerConfig::default();
        let optimizer = GpuOptimizer::new(config).unwrap();

        // Register device
        optimizer
            .register_device(create_test_device("0", 80.0, 0.0))
            .await
            ?;

        // Make allocation
        let request = create_test_request(40.0, 50.0, false);
        let allocation = optimizer.allocate(request).await.unwrap();

        // Check device state
        let device = optimizer.get_device_status("0").await.unwrap();
        assert_eq!(device.available_memory, 40.0);
        assert_eq!(device.utilization, 50.0);

        // Release allocation
        optimizer.release(allocation.allocation_id).await.unwrap();

        // Check device state restored
        let device = optimizer.get_device_status("0").await.unwrap();
        assert_eq!(device.available_memory, 80.0);
        assert_eq!(device.utilization, 0.0);
    }

    #[tokio::test]
    async fn test_cost_aware_allocation() {
        let mut config = GpuOptimizerConfig::default();
        config.strategy = AllocationStrategy::CostAware;

        // Set different costs
        config.gpu_costs.insert("A100".to_string(), 3.0);
        config.gpu_costs.insert("T4".to_string(), 0.5);

        let optimizer = GpuOptimizer::new(config)?;

        // Register devices with different models
        let mut device1 = create_test_device("0", 80.0, 0.0);
        device1.model = "A100".to_string();
        optimizer.register_device(device1).await.unwrap();

        let mut device2 = create_test_device("1", 16.0, 0.0);
        device2.model = "T4".to_string();
        optimizer.register_device(device2).await.unwrap();

        // Request that fits in both
        let request = create_test_request(10.0, 20.0, false);
        let allocation = optimizer.allocate(request).await.unwrap();

        // Should allocate to cheaper T4
        assert_eq!(allocation.device_ids[0], "1");
        assert_eq!(allocation.cost_per_hour, 0.5);
    }

    #[tokio::test]
    async fn test_model_preference() {
        let config = GpuOptimizerConfig::default();
        let optimizer = GpuOptimizer::new(config).unwrap();

        // Register different GPU models
        let mut device1 = create_test_device("0", 80.0, 0.0);
        device1.model = "A100".to_string();
        optimizer.register_device(device1).await?;

        let mut device2 = create_test_device("1", 32.0, 0.0);
        device2.model = "V100".to_string();
        optimizer.register_device(device2).await.unwrap();

        // Request with model preference
        let mut request = create_test_request(20.0, 30.0, false);
        request.preferred_models = vec!["V100".to_string()];

        let allocation = optimizer.allocate(request).await.unwrap();
        assert_eq!(allocation.device_ids[0], "1");
    }

    #[tokio::test]
    async fn test_metrics_tracking() {
        let config = GpuOptimizerConfig::default();
        let optimizer = GpuOptimizer::new(config).unwrap();

        // Register device
        optimizer
            .register_device(create_test_device("0", 80.0, 0.0))
            .await
            ?;

        // Make allocations
        for i in 0..3 {
            let request = create_test_request(20.0, 20.0 + i as f64 * 10.0, false);
            optimizer.allocate(request).await.unwrap();
        }

        let metrics = optimizer.get_metrics();
        assert_eq!(metrics.total_allocations, 3);
        assert!(metrics.total_cost > 0.0);
        assert!(metrics.avg_utilization > 0.0);
    }

    #[tokio::test]
    async fn test_compute_capability_requirement() {
        let config = GpuOptimizerConfig::default();
        let optimizer = GpuOptimizer::new(config).unwrap();

        // Register devices with different compute capabilities
        let mut device1 = create_test_device("0", 80.0, 0.0);
        device1.compute_capability = 7.5;
        optimizer.register_device(device1).await?;

        let mut device2 = create_test_device("1", 80.0, 0.0);
        device2.compute_capability = 8.6;
        optimizer.register_device(device2).await.unwrap();

        // Request requiring high compute capability
        let mut request = create_test_request(20.0, 30.0, false);
        request.affinity.min_compute_capability = Some(8.0);

        let allocation = optimizer.allocate(request).await.unwrap();
        assert_eq!(allocation.device_ids[0], "1");
    }
}
