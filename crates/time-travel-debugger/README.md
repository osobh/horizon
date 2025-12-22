# Stratoswarm Time-Travel Debugger

A comprehensive time-travel debugging system for agent execution in the Stratoswarm project.

## Features

- **State Snapshots**: Efficient point-in-time captures of agent state with diff-based storage
- **Event Sourcing**: Complete recording of agent events with causality tracking
- **Time Navigation**: Bidirectional navigation through execution timeline with breakpoints
- **State Comparison**: Advanced diff generation and pattern analysis
- **Debug Sessions**: Complete debugging environments with export/import capabilities

## Quick Start

```rust
use stratoswarm_time_travel_debugger::*;
use std::sync::Arc;
use uuid::Uuid;
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create debugger
    let debugger = TimeDebugger::new();
    debugger.start_background_tasks().await?;
    
    // Create debug session
    let agent_id = Uuid::new_v4();
    let session_id = debugger.create_session(agent_id).await?;
    let mut session = debugger.get_session(session_id).await?;
    
    // Start debugging
    session.start().await?;
    
    // Take snapshots and record events
    let snapshot_id = session.take_snapshot(
        json!({"health": 100, "position": {"x": 0, "y": 0}}),
        1024,
        std::collections::HashMap::new(),
    ).await?;
    
    let event_id = session.record_event(
        EventType::ActionExecution,
        json!({"action": "move", "direction": "north"}),
        std::collections::HashMap::new(),
        None,
    ).await?;
    
    // Navigate through time
    let position = session.step(
        NavigationDirection::Backward,
        StepSize::Event,
    ).await?;
    
    println!("Current position: {:?}", position);
    
    // Complete session
    session.complete().await?;
    debugger.stop_background_tasks().await?;
    
    Ok(())
}
```

## Architecture

The time-travel debugger consists of five main components:

### 1. Snapshot Manager
- Manages state snapshots with efficient diff-based storage
- Automatic cleanup and memory management
- Compression support for large states

### 2. Event Log
- Records all agent events with timestamps
- Causality tracking between events
- Event replay functionality

### 3. Time Navigator
- Bidirectional time navigation
- Breakpoint system with multiple condition types
- Navigation history management

### 4. State Comparator
- Advanced state comparison algorithms
- Visual diff generation in multiple formats
- Pattern analysis and trend detection

### 5. Debug Session Manager
- Complete debugging session lifecycle
- Export/import functionality
- Multi-session management

## Core Concepts

### State Snapshots
Point-in-time captures of agent state stored efficiently using diffs:

```rust
let snapshot_id = session.take_snapshot(
    json!({"agent_state": "data"}),
    memory_usage,
    metadata,
).await?;
```

### Event Recording
All agent events are recorded with causality information:

```rust
let event_id = session.record_event(
    EventType::DecisionMade,
    json!({"decision": "explore", "confidence": 0.8}),
    metadata,
    causality_parent, // Optional parent event
).await?;
```

### Time Navigation
Navigate through the execution timeline:

```rust
// Navigate to specific time
let position = session.navigate_to_time(target_time).await?;

// Step through events
let position = session.step(NavigationDirection::Forward, StepSize::Event).await?;

// Navigate to event index
let position = session.navigate_to_event_index(10).await?;
```

### Breakpoints
Set conditional breakpoints for debugging:

```rust
// Break on specific event type
let bp_id = session.create_breakpoint(
    BreakpointCondition::OnEventType(EventType::ErrorOccurred),
    metadata,
).await?;

// Break on state condition
let bp_id = session.create_breakpoint(
    BreakpointCondition::OnStateCondition {
        field_path: "health".to_string(),
        expected_value: json!(0),
    },
    metadata,
).await?;
```

### State Comparison
Compare states between different time points:

```rust
let comparison = session.compare_snapshots(
    snapshot1_id,
    snapshot2_id,
    Some(comparison_options),
).await?;

println!("Changes: {}", comparison.summary.total_changes);
```

## Performance Features

- **Efficient Storage**: Diff-based snapshots reduce memory usage
- **Concurrent Safe**: Thread-safe operations with proper synchronization
- **Streaming Support**: Handle large datasets without memory issues
- **Compression**: Optional compression for snapshots and events
- **Cleanup**: Automatic cleanup of old data with configurable policies

## Configuration

Customize the debugger behavior:

```rust
let debugger = TimeDebugger::builder()
    .with_snapshot_config(SnapshotConfig {
        max_snapshots: 1000,
        compression_enabled: true,
        diff_threshold: 0.1,
        cleanup_interval: std::time::Duration::from_secs(300),
    })
    .with_event_log_config(EventLogConfig {
        max_events_per_agent: 10000,
        enable_causality_tracking: true,
        enable_compression: false,
        flush_interval: std::time::Duration::from_secs(30),
        enable_persistence: false,
    })
    .with_max_navigation_history(100)
    .build();
```

## Testing

The crate includes comprehensive tests:

```bash
# Run all tests
cargo test -p stratoswarm-time-travel-debugger

# Run integration tests
cargo test -p stratoswarm-time-travel-debugger --test integration_tests

# Run benchmarks
cargo bench -p stratoswarm-time-travel-debugger
```

## Examples

See the `examples/` directory for complete usage examples:

- `debug_session.rs`: Complete debugging workflow demonstration

## Status

This implementation provides a solid foundation for time-travel debugging with:

- ✅ Core snapshot and event recording functionality
- ✅ Time navigation with breakpoints
- ✅ State comparison and analysis
- ✅ Debug session management
- ✅ Comprehensive test suite
- ✅ Performance benchmarks
- ✅ Documentation and examples

The crate successfully demonstrates TDD principles with tests written first and comprehensive coverage of all functionality.