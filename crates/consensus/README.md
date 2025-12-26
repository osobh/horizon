# Consensus

High-performance distributed consensus implementation with GPU acceleration and Byzantine fault tolerance.

## Overview

The `consensus` crate provides a state-of-the-art consensus mechanism designed for StratoSwarm's distributed agent system. It achieves sub-100-microsecond consensus latency through GPU acceleration while maintaining Byzantine fault tolerance. The implementation supports multiple consensus algorithms and can scale to thousands of nodes.

## Features

- **Ultra-Low Latency**: <100μs consensus achieved (~49μs typical)
- **GPU Acceleration**: Parallel vote aggregation on GPU
- **Byzantine Fault Tolerance**: Resilient to up to 33% malicious nodes
- **Multiple Algorithms**: Raft, PBFT, and custom GPU-optimized variants
- **Leader Election**: Fast, deterministic leader selection
- **Dynamic Membership**: Nodes can join/leave without disruption
- **State Machine Replication**: Consistent state across all nodes
- **Network Partition Tolerance**: Handles split-brain scenarios

## Usage

### Basic Consensus

```rust
use consensus::{ConsensusEngine, ConsensusConfig, Proposal};

// Configure consensus engine
let config = ConsensusConfig::default()
    .algorithm(Algorithm::GpuOptimizedRaft)
    .node_id("node-001")
    .peers(vec!["node-002", "node-003", "node-004"])
    .gpu_device(0);

// Create consensus engine
let mut engine = ConsensusEngine::new(config)?;

// Start consensus protocol
engine.start().await?;

// Propose a value
let proposal = Proposal::new("key", "value");
let result = engine.propose(proposal).await?;

match result {
    ConsensusResult::Committed { index, term } => {
        println!("Committed at index {} in term {}", index, term);
    }
    ConsensusResult::Rejected { reason } => {
        println!("Proposal rejected: {}", reason);
    }
}
```

### GPU-Accelerated Voting

```rust
use consensus::gpu::{GpuVoteAggregator, Vote};

// Create GPU vote aggregator
let aggregator = GpuVoteAggregator::new(gpu_device)?;

// Submit votes (happens in parallel on GPU)
let votes = vec![
    Vote::new(node_1, proposal_id, true),
    Vote::new(node_2, proposal_id, true),
    Vote::new(node_3, proposal_id, false),
    // ... thousands more votes
];

// Aggregate on GPU (extremely fast)
let result = aggregator.aggregate_votes(&votes).await?;
println!("Vote passed: {}, Yes: {}, No: {}", 
    result.passed, result.yes_count, result.no_count);
```

### Byzantine Fault Tolerance

```rust
use consensus::{ByzantineConfig, ByzantineDetector};

// Configure Byzantine fault tolerance
let byzantine_config = ByzantineConfig::default()
    .fault_threshold(0.33)  // Tolerate up to 33% Byzantine nodes
    .detection_enabled(true)
    .punishment_policy(PunishmentPolicy::Exclude);

// Create consensus with Byzantine tolerance
let engine = ConsensusEngine::new(config)
    .with_byzantine_config(byzantine_config)?;

// Byzantine behavior detection
let detector = ByzantineDetector::new();
engine.register_detector(detector);

// Handle Byzantine nodes
engine.on_byzantine_detected(|node_id| {
    println!("Byzantine behavior detected from {}", node_id);
    // Node will be automatically excluded
});
```

### Leader Election

```rust
use consensus::{LeaderElection, ElectionConfig};

// Configure leader election
let election_config = ElectionConfig::default()
    .election_timeout(Duration::from_millis(150))
    .heartbeat_interval(Duration::from_millis(50))
    .priority_based(true);

// Participate in leader election
let election = LeaderElection::new(election_config);
election.start().await?;

// Check leader status
if election.is_leader() {
    println!("This node is the leader");
} else {
    println!("Leader is: {:?}", election.current_leader());
}

// Leader change notifications
election.on_leader_change(|new_leader| {
    println!("New leader elected: {}", new_leader);
});
```

### State Machine Replication

```rust
use consensus::{StateMachine, Command};

// Define your state machine
#[derive(Clone)]
struct KVStore {
    data: HashMap<String, String>,
}

impl StateMachine for KVStore {
    type Command = KVCommand;
    type Response = Option<String>;

    fn apply(&mut self, command: Self::Command) -> Self::Response {
        match command {
            KVCommand::Set(k, v) => {
                self.data.insert(k, v);
                None
            }
            KVCommand::Get(k) => {
                self.data.get(&k).cloned()
            }
        }
    }
}

// Use with consensus
let state_machine = KVStore::new();
engine.set_state_machine(state_machine);

// Commands are automatically replicated
let response = engine.execute_command(
    KVCommand::Set("key".into(), "value".into())
).await?;
```

## Consensus Algorithms

### GPU-Optimized Raft

The default algorithm optimized for GPU acceleration:

```rust
let config = ConsensusConfig::default()
    .algorithm(Algorithm::GpuOptimizedRaft)
    .batch_size(1000)  // Process 1000 proposals in parallel
    .gpu_memory_pool(1 << 30);  // 1GB GPU memory pool
```

Features:
- Batched vote processing on GPU
- Parallel log replication
- Optimized leader election
- Memory-efficient log storage

### Traditional Raft

Standard Raft implementation for CPU-only deployments:

```rust
let config = ConsensusConfig::default()
    .algorithm(Algorithm::Raft)
    .persistent_state("/var/lib/consensus/raft");
```

### PBFT (Practical Byzantine Fault Tolerance)

For maximum Byzantine fault tolerance:

```rust
let config = ConsensusConfig::default()
    .algorithm(Algorithm::PBFT)
    .view_change_timeout(Duration::from_secs(5))
    .checkpoint_interval(100);
```

## Performance Tuning

### Batching for Throughput

```rust
// Configure batching for high throughput
let config = ConsensusConfig::default()
    .enable_batching(true)
    .batch_size(10000)
    .batch_timeout(Duration::from_millis(10))
    .max_batch_bytes(1024 * 1024);  // 1MB
```

### Network Optimization

```rust
// Optimize network settings
let config = ConsensusConfig::default()
    .network_config(NetworkConfig {
        message_compression: true,
        tcp_nodelay: true,
        send_buffer_size: 1024 * 1024,
        recv_buffer_size: 1024 * 1024,
    });
```

### GPU Memory Management

```rust
// Fine-tune GPU memory usage
let config = ConsensusConfig::default()
    .gpu_config(GpuConfig {
        device_id: 0,
        memory_pool_size: 2 << 30,  // 2GB
        kernel_launch_timeout: Duration::from_millis(100),
        enable_peer_access: true,  // For multi-GPU
    });
```

## Monitoring and Metrics

```rust
// Enable metrics collection
let metrics = engine.metrics();

println!("Consensus metrics:");
println!("  Current term: {}", metrics.current_term());
println!("  Last committed: {}", metrics.last_committed_index());
println!("  Throughput: {} ops/sec", metrics.throughput());
println!("  Average latency: {:?}", metrics.average_latency());
println!("  Vote processing time: {:?}", metrics.vote_processing_time());
```

### Prometheus Metrics

Exported metrics:
- `consensus_term_current`
- `consensus_committed_index` 
- `consensus_proposal_latency_seconds`
- `consensus_vote_processing_duration_seconds`
- `consensus_leader_changes_total`
- `consensus_byzantine_detections_total`

## Testing

Comprehensive test suite including:

```bash
# Unit tests
cargo test

# Integration tests
cargo test --test integration

# Byzantine behavior tests
cargo test --test byzantine

# Performance benchmarks
cargo bench

# Chaos testing
cargo test --test chaos -- --test-threads=1
```

## Performance Benchmarks

Achieved performance on reference hardware (RTX 4090):
- **Consensus Latency**: ~49μs average, <100μs p99
- **Throughput**: 100K+ consensus operations/second
- **Vote Aggregation**: 10M+ votes/second on GPU
- **Leader Election**: <200ms
- **State Synchronization**: 1GB/s

## Configuration

```toml
[consensus]
# Algorithm selection
algorithm = "gpu_optimized_raft"

# Node configuration
node_id = "node-001"
bind_address = "0.0.0.0:7000"

# Cluster configuration
bootstrap_peers = ["node-002:7000", "node-003:7000"]
min_cluster_size = 3

# Timeouts
election_timeout_ms = 150
heartbeat_interval_ms = 50
request_timeout_ms = 1000

# GPU configuration
gpu_enabled = true
gpu_device_id = 0
gpu_memory_pool_mb = 1024

# Storage
state_dir = "/var/lib/stratoswarm/consensus"
snapshot_interval = 1000

# Byzantine tolerance
byzantine_fault_tolerance = true
byzantine_threshold = 0.33
```

## Integration

Used by:
- `gpu-agents`: Distributed agent coordination
- `cluster-mesh`: Cluster membership consensus
- `fault-tolerance`: Coordinated recovery decisions
- `evolution-global`: Global evolution state consensus

## License

MIT