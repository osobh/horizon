//! Message types for stratoswarm channel infrastructure.
//!
//! This module defines all message types that flow through the system channels.
//! Messages use zero-copy `Bytes` buffers where appropriate for performance.

use bytes::Bytes;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// GPU command messages for GPU agent communication.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GpuCommand {
    /// Launch a kernel on the GPU
    LaunchKernel {
        /// Unique kernel identifier
        kernel_id: String,
        /// Grid dimensions (x, y, z)
        grid_dim: (u32, u32, u32),
        /// Block dimensions (x, y, z)
        block_dim: (u32, u32, u32),
        /// Kernel parameters as zero-copy buffer
        params: Bytes,
    },
    /// Transfer data to GPU device
    TransferToDevice {
        /// Destination buffer ID on device
        buffer_id: String,
        /// Data to transfer (zero-copy)
        data: Bytes,
        /// Offset in destination buffer
        offset: usize,
    },
    /// Synchronize GPU execution
    Synchronize {
        /// Stream ID to synchronize (None for all streams)
        stream_id: Option<u32>,
    },
    /// Transfer data from GPU device
    TransferFromDevice {
        /// Source buffer ID on device
        buffer_id: String,
        /// Size of data to transfer
        size: usize,
        /// Offset in source buffer
        offset: usize,
    },
}

/// Evolution engine messages for evolutionary algorithm control.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EvolutionMessage {
    /// Perform one evolution step
    Step {
        /// Generation number
        generation: u64,
    },
    /// Evaluate fitness of individuals
    EvaluateFitness {
        /// Individual IDs to evaluate
        individual_ids: Vec<Uuid>,
    },
    /// Perform selection
    Selection {
        /// Selection strategy
        strategy: SelectionStrategy,
        /// Number of individuals to select
        count: usize,
    },
    /// Perform mutation
    Mutation {
        /// Individual IDs to mutate
        individual_ids: Vec<Uuid>,
        /// Mutation rate (0.0 to 1.0)
        rate: f64,
    },
    /// Get best individual
    GetBest {
        /// Number of best individuals to retrieve
        count: usize,
    },
}

/// Selection strategies for evolution.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum SelectionStrategy {
    /// Tournament selection
    Tournament {
        /// Tournament size
        size: usize,
    },
    /// Roulette wheel selection
    Roulette,
    /// Rank-based selection
    Rank,
    /// Elitist selection (top N)
    Elitist,
}

/// System events broadcast to all subscribers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SystemEvent {
    /// Agent spawned
    AgentSpawned {
        /// Agent unique ID
        agent_id: Uuid,
        /// Agent type
        agent_type: String,
        /// Timestamp (milliseconds since epoch)
        timestamp: u64,
    },
    /// Fitness improved
    FitnessImproved {
        /// Individual ID
        individual_id: Uuid,
        /// Old fitness value
        old_fitness: f64,
        /// New fitness value
        new_fitness: f64,
        /// Timestamp
        timestamp: u64,
    },
    /// GPU utilization update
    GpuUtilization {
        /// GPU device ID
        device_id: u32,
        /// Utilization percentage (0.0 to 100.0)
        utilization: f64,
        /// Timestamp
        timestamp: u64,
    },
    /// Memory pressure warning
    MemoryPressure {
        /// Memory usage percentage (0.0 to 100.0)
        usage_percent: f64,
        /// Available bytes
        available_bytes: u64,
        /// Timestamp
        timestamp: u64,
    },
    /// Kernel execution completed
    KernelCompleted {
        /// Kernel unique ID
        kernel_id: String,
        /// Execution duration in microseconds
        duration_us: u64,
        /// Success status
        success: bool,
        /// Timestamp
        timestamp: u64,
    },
    /// Error occurred
    Error {
        /// Error message
        message: String,
        /// Error source component
        source: String,
        /// Timestamp
        timestamp: u64,
    },
}

/// Cost optimization messages.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CostMessage {
    /// Query current cost
    QueryCost,
    /// Report cost update
    CostUpdate {
        /// Total cost in cents
        total_cents: u64,
        /// Cost breakdown by resource
        breakdown: Vec<(String, u64)>,
    },
    /// Optimize for cost target
    OptimizeFor {
        /// Target cost in cents
        target_cents: u64,
    },
}

/// Efficiency intelligence messages.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EfficiencyMessage {
    /// Query efficiency metrics
    QueryMetrics,
    /// Report efficiency update
    EfficiencyUpdate {
        /// CPU efficiency (0.0 to 1.0)
        cpu_efficiency: f64,
        /// GPU efficiency (0.0 to 1.0)
        gpu_efficiency: f64,
        /// Memory efficiency (0.0 to 1.0)
        memory_efficiency: f64,
    },
    /// Recommend optimization
    RecommendOptimization {
        /// Resource type
        resource: String,
    },
}

/// Scheduler messages for workload scheduling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SchedulerMessage {
    /// Schedule a task
    ScheduleTask {
        /// Task unique ID
        task_id: Uuid,
        /// Task priority (higher = more urgent)
        priority: u32,
        /// Resource requirements
        resources: ResourceRequirements,
    },
    /// Cancel a scheduled task
    CancelTask {
        /// Task ID to cancel
        task_id: Uuid,
    },
    /// Query task status
    QueryStatus {
        /// Task ID to query
        task_id: Uuid,
    },
}

/// Resource requirements for tasks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirements {
    /// CPU cores required
    pub cpu_cores: u32,
    /// Memory in megabytes
    pub memory_mb: u64,
    /// GPU required
    pub gpu_required: bool,
    /// GPU memory in megabytes
    pub gpu_memory_mb: u64,
}

/// Governor messages for resource governance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GovernorMessage {
    /// Request resource allocation
    RequestAllocation {
        /// Requester ID
        requester_id: Uuid,
        /// Resources requested
        resources: ResourceRequirements,
    },
    /// Release resource allocation
    ReleaseAllocation {
        /// Allocation ID
        allocation_id: Uuid,
    },
    /// Query available resources
    QueryAvailable,
}

/// Knowledge graph messages.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KnowledgeMessage {
    /// Store knowledge
    Store {
        /// Key for storage
        key: String,
        /// Value to store (zero-copy)
        value: Bytes,
    },
    /// Retrieve knowledge
    Retrieve {
        /// Key to retrieve
        key: String,
    },
    /// Query knowledge with pattern
    Query {
        /// Query pattern
        pattern: String,
        /// Maximum results
        limit: usize,
    },
}

/// Consensus messages for distributed agreement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsensusMessage {
    /// Propose a value
    Propose {
        /// Proposal ID
        proposal_id: Uuid,
        /// Proposed value
        value: Bytes,
    },
    /// Vote on a proposal
    Vote {
        /// Proposal ID
        proposal_id: Uuid,
        /// Vote (true = accept, false = reject)
        vote: bool,
    },
    /// Commit a decided value
    Commit {
        /// Proposal ID
        proposal_id: Uuid,
    },
}
