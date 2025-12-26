# Fault Tolerance

Checkpoint and recovery system ensuring resilience for StratoSwarm deployments.

## Overview

The `fault-tolerance` crate provides comprehensive failure detection, checkpointing, and recovery mechanisms for StratoSwarm. It ensures that agent computations, evolution progress, and system state can survive hardware failures, network partitions, and software crashes. The system is designed for minimal overhead during normal operation while providing fast recovery when failures occur.

## Features

- **Automatic Checkpointing**: Periodic state snapshots with configurable intervals
- **Coordinated Recovery**: Distributed consensus on recovery actions
- **Failure Detection**: Fast detection of node, agent, and service failures
- **State Replication**: Multi-node state replication for durability
- **Incremental Checkpoints**: Only save changed state to minimize overhead
- **Point-in-Time Recovery**: Restore to any previous checkpoint
- **Failure Prediction**: ML-based failure prediction and preemptive migration
- **Zero Downtime Recovery**: Hot standby and instant failover

## Usage

### Basic Checkpointing

```rust
use fault_tolerance::{Checkpoint, CheckpointManager};

// Create checkpoint manager
let manager = CheckpointManager::new()
    .interval(Duration::from_secs(300))  // 5 minutes
    .retention(Duration::from_days(7))   // Keep for 7 days
    .compression(CompressionType::Zstd);

// Register state to checkpoint
manager.register_state("agent_state", &agent);
manager.register_state("evolution_progress", &evolution);

// Start automatic checkpointing
manager.start().await?;

// Manual checkpoint
let checkpoint = manager.create_checkpoint("manual_backup").await?;
println!("Checkpoint created: {}", checkpoint.id());
```

### Coordinated Recovery

```rust
use fault_tolerance::{RecoveryCoordinator, RecoveryStrategy};

// Create recovery coordinator
let coordinator = RecoveryCoordinator::new()
    .strategy(RecoveryStrategy::FastestNode)
    .quorum_size(3);

// Detect failure and coordinate recovery
coordinator.on_failure(|failure| async move {
    match failure.failure_type() {
        FailureType::NodeCrash(node_id) => {
            // Redistribute work from failed node
            coordinator.redistribute_work(node_id).await?;
        }
        FailureType::AgentFailure(agent_id) => {
            // Restart agent from last checkpoint
            coordinator.recover_agent(agent_id).await?;
        }
        FailureType::NetworkPartition => {
            // Handle split-brain scenario
            coordinator.resolve_partition().await?;
        }
    }
    Ok(())
});
```

### Failure Detection

```rust
use fault_tolerance::{FailureDetector, HeartbeatConfig};

// Configure failure detection
let detector = FailureDetector::new()
    .heartbeat_interval(Duration::from_secs(5))
    .failure_threshold(3)  // 3 missed heartbeats
    .detection_window(Duration::from_secs(30));

// Monitor components
detector.monitor_node(node_id);
detector.monitor_agent(agent_id);
detector.monitor_service("gpu-consensus");

// React to failures
let mut failures = detector.failure_stream();
while let Some(failure) = failures.next().await {
    println!("Detected failure: {:?}", failure);
    // Trigger recovery
}
```

### State Replication

```rust
use fault_tolerance::{StateReplicator, ReplicationPolicy};

// Configure state replication
let replicator = StateReplicator::new()
    .replication_factor(3)
    .policy(ReplicationPolicy::GeoDistributed)
    .consistency_level(ConsistencyLevel::Quorum);

// Replicate critical state
replicator.replicate("critical_state", &state).await?;

// Read with consistency guarantee
let state = replicator.read_consistent("critical_state").await?;
```

### Incremental Checkpoints

```rust
use fault_tolerance::{IncrementalCheckpoint, DeltaTracker};

// Enable incremental checkpointing
let mut tracker = DeltaTracker::new();
tracker.track_changes(&mut agent_state);

// Create incremental checkpoint
let checkpoint = IncrementalCheckpoint::new()
    .base_checkpoint(last_checkpoint_id)
    .capture_deltas(&tracker)?;

// Much smaller than full checkpoint
println!("Checkpoint size: {} bytes", checkpoint.size());
```

### Point-in-Time Recovery

```rust
use fault_tolerance::{TimelineManager, RecoveryPoint};

// Create timeline manager
let timeline = TimelineManager::new();

// List available recovery points
let points = timeline.list_recovery_points(
    TimeRange::last_hours(24)
)?;

// Restore to specific point
let target_point = points.iter()
    .find(|p| p.timestamp() < failure_time)
    .unwrap();

timeline.restore_to_point(target_point).await?;
```

## Failure Scenarios

### Node Failures

```rust
// Automatic handling of node failures
let node_handler = NodeFailureHandler::new()
    .migration_strategy(MigrationStrategy::LoadBalanced)
    .preserve_gpu_affinity(true);

// Register handler
coordinator.register_handler(node_handler);
```

### Network Partitions

```rust
// Split-brain resolution
let partition_resolver = PartitionResolver::new()
    .strategy(ResolutionStrategy::Raft)
    .leader_election_timeout(Duration::from_secs(10));

// Automatic resolution
coordinator.register_handler(partition_resolver);
```

### Cascading Failures

```rust
// Circuit breaker for cascading failures
use fault_tolerance::CircuitBreaker;

let circuit_breaker = CircuitBreaker::new()
    .failure_threshold(5)
    .reset_timeout(Duration::from_secs(60))
    .half_open_requests(3);

// Wrap risky operations
let result = circuit_breaker.call(async {
    risky_operation().await
}).await?;
```

## Failure Prediction

```rust
use fault_tolerance::{FailurePredictor, PredictionModel};

// ML-based failure prediction
let predictor = FailurePredictor::new()
    .model(PredictionModel::GradientBoost)
    .features(vec![
        "cpu_temperature",
        "memory_pressure", 
        "error_rate",
        "response_time"
    ]);

// Train on historical data
predictor.train(&historical_data).await?;

// Predict failures
let predictions = predictor.predict_failures(
    Duration::from_hours(1)  // 1 hour ahead
)?;

for prediction in predictions {
    if prediction.probability > 0.8 {
        // Preemptive migration
        coordinator.migrate_preemptively(
            prediction.component_id
        ).await?;
    }
}
```

## Configuration

```toml
[fault_tolerance]
# Checkpointing
checkpoint_interval = "5m"
checkpoint_retention = "7d"
checkpoint_compression = "zstd"
checkpoint_storage = "/mnt/nvme/checkpoints"

# Failure detection
heartbeat_interval = "5s"
failure_threshold = 3
detection_window = "30s"

# Recovery
recovery_strategy = "fastest_node"
recovery_timeout = "5m"
max_recovery_attempts = 3

# Replication
replication_factor = 3
consistency_level = "quorum"
geo_distributed = true

# Prediction
prediction_enabled = true
prediction_model = "gradient_boost"
prediction_interval = "1m"
```

## Performance Impact

Minimal overhead during normal operation:
- **Checkpoint Overhead**: <1% CPU for incremental
- **Heartbeat Traffic**: <1KB/s per monitored component
- **Recovery Time**: <5s for hot standby failover
- **Prediction Overhead**: <0.5% CPU when enabled

## Testing

Comprehensive failure injection testing:

```bash
# Run all tests
cargo test

# Failure injection tests
cargo test --test failure_injection

# Chaos testing
cargo test --test chaos -- --test-threads=1

# Performance tests
cargo bench recovery_time
```

### Test Scenarios

- Single node failures
- Multiple simultaneous failures
- Network partitions
- Byzantine failures
- Cascading failures
- Recovery during high load

## Integration

Used throughout StratoSwarm:
- `agent-core`: Agent state checkpointing
- `evolution-engines`: Evolution progress recovery
- `gpu-agents`: GPU computation checkpointing
- `cluster-mesh`: Node failure handling

## Monitoring

Built-in metrics for fault tolerance:

```rust
// Prometheus metrics
fault_tolerance_checkpoints_total
fault_tolerance_checkpoint_duration_seconds
fault_tolerance_recoveries_total
fault_tolerance_recovery_duration_seconds
fault_tolerance_failures_detected_total
fault_tolerance_prediction_accuracy
```

## Best Practices

1. **Checkpoint Frequency**: Balance between overhead and recovery granularity
2. **State Size**: Keep checkpointed state minimal
3. **Geographic Distribution**: Replicate across failure domains
4. **Testing**: Regularly test recovery procedures
5. **Monitoring**: Alert on checkpoint failures

## Examples

See `examples/` directory:

```bash
# Basic checkpointing
cargo run --example checkpointing

# Failure recovery
cargo run --example recovery

# Chaos testing
cargo run --example chaos_test
```

## Recovery Guarantees

- **RPO (Recovery Point Objective)**: Max 5 minutes data loss
- **RTO (Recovery Time Objective)**: <30 seconds for hot standby
- **Durability**: 99.999999% with 3x replication
- **Availability**: 99.99% with automatic failover

## License

MIT