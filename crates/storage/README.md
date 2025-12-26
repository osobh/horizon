# Storage

NVMe-optimized storage layer with knowledge graph support for StratoSwarm.

## Overview

The `storage` crate provides high-performance storage capabilities optimized for NVMe devices, with specialized support for knowledge graph operations. It implements efficient data structures and I/O patterns designed to maximize the performance of modern storage hardware while supporting the complex data relationships required by intelligent agents.

## Features

- **NVMe Optimization**: Direct I/O and optimized access patterns for NVMe SSDs
- **Knowledge Graph Storage**: Efficient graph storage using Compressed Sparse Row (CSR) format
- **GPU Cache Integration**: Direct GPU memory caching for hot data
- **Memory-Mapped Files**: Zero-copy access to large datasets
- **Write-Ahead Logging**: Durability guarantees with minimal performance impact
- **Multi-Format Support**: Flexible graph storage formats for different use cases

## Usage

### Basic Storage Operations

```rust
use exorust_storage::{NvmeStore, StorageConfig};

// Create NVMe-optimized storage
let config = StorageConfig::default()
    .with_direct_io(true)
    .with_io_depth(32);
    
let store = NvmeStore::new("/mnt/nvme0", config)?;

// Write data with NVMe optimization
store.write_aligned(key, &data).await?;

// Read with zero-copy when possible
let data = store.read_direct(key).await?;
```

### Knowledge Graph Operations

```rust
use exorust_storage::{GraphStorage, GraphFormat};

// Create graph storage
let graph_store = GraphStorage::new(store, GraphFormat::CSR)?;

// Add nodes and edges
graph_store.add_node(node_id, properties)?;
graph_store.add_edge(from_id, to_id, edge_properties)?;

// Efficient graph queries
let neighbors = graph_store.get_neighbors(node_id)?;
let path = graph_store.shortest_path(start_id, end_id)?;
```

### GPU Cache Integration

```rust
use exorust_storage::GpuCache;

// Create GPU-backed cache
let gpu_cache = GpuCache::new(gpu_device, cache_size)?;

// Cache hot data on GPU
gpu_cache.cache(key, &data)?;

// Direct GPU access
let gpu_ptr = gpu_cache.get_device_ptr(key)?;
```

## Architecture

The crate is organized into specialized modules:

- `nvme.rs`: NVMe-specific optimizations and direct I/O
- `graph.rs`: Core graph storage abstractions
- `graph_csr.rs`: Compressed Sparse Row implementation
- `graph_storage.rs`: High-level graph operations
- `gpu_cache.rs`: GPU memory caching layer
- `memory.rs`: Memory-mapped file support
- `benchmarks.rs`: Performance benchmarking utilities

## Performance Optimizations

### NVMe Optimizations
- **4KB Aligned I/O**: Matches NVMe page size
- **Queue Depth Management**: Configurable parallelism
- **Direct I/O**: Bypasses OS cache for predictable latency
- **Batched Operations**: Amortizes command overhead

### Graph Storage Optimizations
- **CSR Format**: Optimal for sparse graphs
- **Edge Locality**: Edges stored contiguously per vertex
- **Compressed Pointers**: Reduced memory footprint
- **SIMD-Friendly Layout**: Enables vectorized operations

## Benchmarks

Run performance benchmarks:

```bash
cargo bench

# Specific benchmarks
cargo bench --bench nvme_throughput
cargo bench --bench graph_traversal
cargo bench --bench gpu_cache
```

Typical performance metrics:
- **Sequential Read**: 7GB/s (PCIe 4.0 NVMe)
- **Random 4KB Read**: 1M+ IOPS
- **Graph Traversal**: 10M+ edges/second
- **GPU Cache Hit**: <100ns latency

## Configuration

Fine-tune storage behavior:

```rust
let config = StorageConfig::builder()
    .io_depth(64)                    // Parallel I/O operations
    .direct_io(true)                  // Bypass OS cache
    .compression(CompressionType::LZ4) // Fast compression
    .cache_size(1 << 30)              // 1GB cache
    .prefetch_distance(16)            // Read-ahead pages
    .build();
```

## Testing

Comprehensive test coverage includes:

```bash
# Unit tests
cargo test

# Integration tests with real NVMe
cargo test --features integration

# Stress tests
cargo test --features stress -- --test-threads=1
```

## Coverage

Current test coverage: 90%+ (Excellent)

Well-tested areas:
- All storage operations
- Graph algorithms
- Error handling
- Concurrent access
- GPU cache operations

## Safety and Reliability

- **ACID Compliance**: Write-ahead logging ensures durability
- **Checksums**: Data integrity verification
- **Graceful Degradation**: Falls back to regular I/O if direct I/O unavailable
- **Resource Limits**: Prevents memory exhaustion

## Integration Points

Used extensively throughout StratoSwarm:
- `agent-core`: Agent knowledge persistence
- `gpu-agents`: GPU computation data
- `knowledge-graph`: Graph operation backend
- `evolution-engines`: Evolution state storage

## License

MIT