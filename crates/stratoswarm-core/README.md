# Stratoswarm Core

High-performance channel infrastructure for distributed agent systems in Stratoswarm.

## Overview

Stratoswarm Core provides the foundational communication infrastructure for the stratoswarm distributed agent system. It implements a zero-copy, type-safe channel architecture for inter-component communication.

## Features

- **Zero-copy data transfer** using `bytes::Bytes`
- **Type-safe channels** for different message types
- **Point-to-point messaging** (mpsc) for directed communication
- **Broadcast channels** for event distribution
- **Request/response patterns** with configurable timeouts
- **Backpressure control** for GPU operations
- **Comprehensive test coverage** with TDD approach

## Architecture

### Channel Types

#### Point-to-Point Channels (mpsc + broadcast)

These channels provide directed communication with multi-subscriber support:

- **GPU Channel** (buffer: 100) - GPU commands with backpressure control
- **Evolution Channel** (buffer: 10000) - High-throughput evolutionary algorithm messages

#### Request/Response Channels (mpsc)

These channels use the request/response pattern for synchronous communication:

- **Cost Channel** (buffer: 1000) - Cost optimization requests/responses
- **Efficiency Channel** (buffer: 1000) - Efficiency intelligence queries
- **Scheduler Channel** (buffer: 1000) - Task scheduling operations
- **Governor Channel** (buffer: 1000) - Resource allocation governance
- **Knowledge Channel** (buffer: 1000) - Knowledge graph operations

#### Broadcast Channels

These channels distribute events to multiple subscribers:

- **Events Channel** (buffer: 1000) - System events (agent spawned, fitness improved, etc.)

## Usage

### Basic Channel Communication

```rust
use stratoswarm_core::channels::{ChannelRegistry, GpuCommand};
use bytes::Bytes;

#[tokio::main]
async fn main() {
    // Create the registry
    let registry = ChannelRegistry::new();

    // Get senders and receivers
    let gpu_tx = registry.gpu_sender();
    let mut gpu_rx = registry.subscribe_gpu();

    // Send a GPU command with zero-copy data
    let data = Bytes::from(vec![1, 2, 3, 4]);
    let cmd = GpuCommand::TransferToDevice {
        buffer_id: "my_buffer".to_string(),
        data,
        offset: 0,
    };

    gpu_tx.send(cmd).await.unwrap();

    // Receive the command
    let received = gpu_rx.recv().await.unwrap();
}
```

### Request/Response Pattern

```rust
use stratoswarm_core::channels::{ChannelRegistry, CostMessage, request_with_timeout};
use std::time::Duration;

#[tokio::main]
async fn main() {
    let registry = ChannelRegistry::new();
    let cost_tx = registry.cost_sender();
    let cost_rx = registry.subscribe_cost();

    // Spawn a handler
    tokio::spawn(async move {
        loop {
            if let Some(request) = cost_rx.lock().await.recv().await {
                let response = CostMessage::CostUpdate {
                    total_cents: 1234,
                    breakdown: vec![],
                };
                request.respond(response).unwrap();
            }
        }
    });

    // Send a request with timeout
    let response = request_with_timeout(
        &cost_tx,
        CostMessage::QueryCost,
        Duration::from_secs(5)
    ).await.unwrap();
}
```

### Broadcasting Events

```rust
use stratoswarm_core::channels::{ChannelRegistry, SystemEvent};
use uuid::Uuid;

#[tokio::main]
async fn main() {
    let registry = ChannelRegistry::new();

    // Multiple subscribers
    let mut sub1 = registry.subscribe_events();
    let mut sub2 = registry.subscribe_events();

    // Broadcast an event
    let event = SystemEvent::AgentSpawned {
        agent_id: Uuid::new_v4(),
        agent_type: "TestAgent".to_string(),
        timestamp: 1000,
    };

    registry.event_sender().send(event).unwrap();

    // All subscribers receive the event
    let _ = sub1.recv().await.unwrap();
    let _ = sub2.recv().await.unwrap();
}
```

## Performance Characteristics

- **Zero-copy transfers**: Uses `bytes::Bytes` for efficient data sharing
- **Bounded channels**: Provides backpressure to prevent memory exhaustion
- **Lock-free**: Built on tokio's lock-free async channels
- **Type-safe**: Compile-time verification of message types

## Safety

This crate contains no unsafe code and relies entirely on Rust's type system and tokio's well-tested async primitives for safety guarantees.

## Testing

The crate follows strict TDD principles with comprehensive test coverage:

- Channel creation and message passing
- Broadcast to multiple subscribers
- Request/response with timeout (success and timeout cases)
- Backpressure behavior on bounded channels
- Graceful shutdown when channels close
- Zero-copy bytes transfer verification
- Concurrent senders and receivers

Run tests with:

```bash
cargo test -p stratoswarm-core
```

## License

MIT
