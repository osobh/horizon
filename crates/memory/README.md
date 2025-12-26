# Memory

GPU memory allocation and management system for the StratoSwarm 5-tier memory hierarchy.

## Overview

The `memory` crate provides a sophisticated memory management system designed for GPU-accelerated computing environments. It implements efficient allocation strategies and supports the full 5-tier memory hierarchy that is fundamental to StratoSwarm's architecture.

## Features

- **GPU Memory Management**: Direct integration with CUDA for GPU memory allocation
- **Memory Pool Architecture**: Pre-allocated pools for reduced allocation overhead
- **5-Tier Hierarchy Support**:
  - Tier 1: GPU Memory (fastest, smallest)
  - Tier 2: CPU Memory
  - Tier 3: NVMe Storage
  - Tier 4: SSD Storage
  - Tier 5: HDD Storage (slowest, largest)
- **Mock Implementation**: CPU-based mock for testing without GPU hardware
- **Async Operations**: Full async/await support for non-blocking operations
- **Safety**: Safe abstractions over unsafe GPU memory operations

## Usage

```rust
use exorust_memory::{MemoryPool, Allocator, AllocationStrategy};

// Create a memory pool with 1GB capacity
let pool = MemoryPool::new(1 << 30, AllocationStrategy::BestFit)?;

// Allocate memory
let allocation = pool.allocate(1024 * 1024).await?; // 1MB

// Use the allocation
allocation.write(&data).await?;
let result = allocation.read().await?;

// Memory is automatically freed when allocation is dropped
```

### GPU-Specific Operations

```rust
#[cfg(feature = "cuda")]
use exorust_memory::gpu::{GpuAllocator, GpuMemory};

// Create GPU allocator
let gpu_allocator = GpuAllocator::new(0)?; // Device 0

// Allocate GPU memory
let gpu_mem = gpu_allocator.allocate(1024 * 1024)?;

// Transfer data to GPU
gpu_mem.copy_from_host(&host_data)?;

// Transfer back to host
let mut result = vec![0u8; size];
gpu_mem.copy_to_host(&mut result)?;
```

## Architecture

The crate consists of three main components:

- `allocator.rs`: Core allocation strategies and management
- `pool.rs`: Memory pool implementation with configurable strategies
- `error.rs`: Comprehensive error types for memory operations

## Allocation Strategies

The crate supports multiple allocation strategies:

- **BestFit**: Finds the smallest suitable block (minimal fragmentation)
- **FirstFit**: Uses the first suitable block (fastest allocation)
- **WorstFit**: Uses the largest suitable block (reduces small fragments)

## Performance Characteristics

- **Allocation Speed**: O(log n) for BestFit, O(n) worst case
- **Deallocation**: O(log n) with coalescing
- **Memory Overhead**: <1% for pools larger than 100MB
- **GPU Transfer**: Limited by PCIe bandwidth (~16GB/s for PCIe 4.0)

## Configuration

Environment variables for tuning:

```bash
# Set default pool size (bytes)
STRATOSWARM_MEMORY_POOL_SIZE=2147483648  # 2GB

# Enable memory debugging
STRATOSWARM_MEMORY_DEBUG=1

# Set allocation strategy (best_fit, first_fit, worst_fit)
STRATOSWARM_MEMORY_STRATEGY=best_fit
```

## Testing

The crate includes both unit and integration tests:

```bash
# Run all tests (including mock GPU tests)
cargo test

# Run only CPU tests
cargo test --no-default-features

# Run with real GPU (requires CUDA)
cargo test --features cuda
```

## Coverage

Current test coverage: 90%+ (Excellent)

Well-tested areas:
- All allocation strategies
- Pool management and fragmentation
- Error handling paths
- Mock GPU implementation

## Safety Considerations

- All GPU operations are wrapped in safe Rust abstractions
- Automatic cleanup on drop prevents memory leaks
- Bounds checking on all memory operations
- Thread-safe pool operations with internal synchronization

## Integration

This crate is used throughout StratoSwarm:
- `agent-core`: Agent memory management
- `runtime`: Container memory isolation
- `gpu-agents`: GPU computation memory
- `zero-config`: Analysis memory buffers

## License

MIT