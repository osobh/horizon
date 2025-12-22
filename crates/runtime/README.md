# Runtime

GPU container runtime for isolated agent execution with personality-driven behavior.

## Overview

The `runtime` crate provides a sophisticated container runtime specifically designed for GPU-accelerated autonomous agents. It extends traditional container concepts with agent personalities, evolution capabilities, and hardware-aware scheduling, enabling containers that can learn, adapt, and optimize their own behavior.

## Features

- **Agent Personalities**: 5 distinct personality types that influence agent behavior
  - Conservative: Stability-focused, careful resource usage
  - Aggressive: Performance-focused, willing to take risks
  - Balanced: Adaptive middle ground
  - Explorer: Innovation-focused, tries new approaches
  - Cooperative: Team-oriented, optimizes for collective success
- **Container Lifecycle Management**: Complete lifecycle from creation to termination
- **Resource Isolation**: CPU, memory, and GPU resource quotas
- **Hardware Affinity**: Pin containers to specific CPUs, GPUs, or NUMA nodes
- **Evolution Support**: Containers can modify their own behavior over time
- **Secure Execution**: Sandboxed environment with capability restrictions

## Usage

### Basic Container Operations

```rust
use exorust_runtime::{Container, ContainerConfig, Personality};

// Create container configuration
let config = ContainerConfig::builder()
    .name("agent-001")
    .image("stratoswarm/base-agent:latest")
    .personality(Personality::Explorer)
    .cpu_limit(2.0)
    .memory_limit(4 * 1024 * 1024 * 1024) // 4GB
    .gpu_count(1)
    .build()?;

// Create and start container
let container = Container::create(config).await?;
container.start().await?;

// Get container status
let status = container.status().await?;
println!("Container state: {:?}", status.state);

// Stop and remove
container.stop().await?;
container.remove().await?;
```

### Personality-Driven Behavior

```rust
use exorust_runtime::{Personality, PersonalityTraits};

// Get personality traits
let traits = Personality::Aggressive.traits();
println!("Risk tolerance: {}", traits.risk_tolerance);
println!("Exploration rate: {}", traits.exploration_rate);

// Personality influences resource allocation
let resources = container.allocate_resources(&workload)?;
// Aggressive personality may over-provision for performance
// Conservative personality may under-provision for safety
```

### Hardware Affinity

```rust
use exorust_runtime::{HardwareAffinity, NumaNode};

// Pin to specific hardware
let affinity = HardwareAffinity::builder()
    .cpu_cores(vec![0, 1, 2, 3])
    .gpu_device(0)
    .numa_node(NumaNode(0))
    .prefer_local_memory(true)
    .build();

config.set_hardware_affinity(affinity);
```

### Evolution Integration

```rust
use exorust_runtime::{EvolutionConfig, MutationRate};

// Enable evolution for container
let evolution = EvolutionConfig::builder()
    .mutation_rate(MutationRate::Adaptive)
    .fitness_function(|metrics| {
        // Calculate fitness based on performance metrics
        metrics.throughput / metrics.resource_usage
    })
    .generation_interval(Duration::from_secs(3600))
    .build();

container.enable_evolution(evolution).await?;
```

## Architecture

The runtime is organized into key modules:

- `container.rs`: Core container implementation
- `personality.rs`: Agent personality system
- `isolation.rs`: Resource isolation and sandboxing
- `lifecycle.rs`: Container state management
- `secure_runtime.rs`: Security enforcement

## Personality System

Each personality type has distinct characteristics:

| Personality | Risk Tolerance | Cooperation | Exploration | Efficiency Focus | Stability |
|------------|----------------|-------------|-------------|------------------|-----------|
| Conservative | 0.2 | 0.6 | 0.3 | 0.7 | 0.9 |
| Aggressive | 0.9 | 0.3 | 0.8 | 0.9 | 0.4 |
| Balanced | 0.5 | 0.7 | 0.5 | 0.6 | 0.7 |
| Explorer | 0.7 | 0.5 | 0.95 | 0.4 | 0.3 |
| Cooperative | 0.4 | 0.95 | 0.4 | 0.5 | 0.6 |

## Resource Management

The runtime enforces strict resource limits:

```rust
// CPU limits (in cores)
config.cpu_limit(2.5); // 2.5 cores

// Memory limits (in bytes)
config.memory_limit(8 * 1024 * 1024 * 1024); // 8GB

// GPU assignment
config.gpu_devices(vec![0, 1]); // GPUs 0 and 1

// Network bandwidth (in Mbps)
config.network_bandwidth(1000); // 1Gbps

// Disk I/O (in IOPS)
config.disk_iops(10000);
```

## Security Features

- **Capability Dropping**: Remove unnecessary Linux capabilities
- **Seccomp Profiles**: Restrict system calls
- **Namespace Isolation**: Full namespace separation
- **Read-Only Root**: Immutable container filesystem
- **User Namespaces**: Run as non-root inside container

## Performance

- **Container Spawn Time**: <1ms for cached images
- **Resource Allocation**: O(1) for CPU/memory
- **Personality Decision**: <10μs per decision
- **Evolution Overhead**: <1% CPU usage

## Testing

```bash
# Run all tests
cargo test

# Run integration tests
cargo test --test integration

# Run personality tests
cargo test personality

# Benchmark performance
cargo bench
```

## Coverage

Current test coverage: 84.4% (Good, approaching 90% target)

Well-tested areas:
- Container lifecycle management
- Personality system (92% coverage)
- Resource allocation
- Basic isolation

Areas needing additional tests:
- Complex evolution scenarios
- Hardware affinity edge cases
- Security profile validation

## Integration

The runtime is integrated throughout StratoSwarm:
- `agent-core`: Provides container execution for agents
- `evolution-engines`: Evolution algorithms for container optimization
- `zero-config`: Automatic container configuration
- `gpu-agents`: GPU-accelerated agent execution

## Configuration

Environment variables:

```bash
# Default container runtime
STRATOSWARM_RUNTIME_ENGINE=nvidia-docker

# Container image registry
STRATOSWARM_REGISTRY=registry.stratoswarm.io

# Evolution parameters
STRATOSWARM_EVOLUTION_ENABLED=true
STRATOSWARM_EVOLUTION_RATE=0.1
```

## Future Enhancements

- WebAssembly support for cross-platform agents
- Live migration between nodes
- Checkpoint/restore functionality
- Advanced scheduling algorithms

## License

MIT