# GPU Agent Storage Configuration

## Overview

The GPU agent storage tier provides high-performance persistent storage for GPU agents, knowledge graphs, and swarm data. It uses the `/magikdev/gpu` path by default for production deployments.

## Storage Paths

### Production Environment
```rust
// Default production paths
base_path: /magikdev/gpu
cache_path: /magikdev/gpu/cache
wal_path: /magikdev/gpu/wal
nvme_storage: /magikdev/gpu/nvme_storage
```

### Development Environment
```rust
// Development paths (local directory)
base_path: ./gpu_storage
cache_path: ./gpu_storage/cache
wal_path: ./gpu_storage/wal
```

## Configuration

### Basic Configuration
```rust
use gpu_agents::{GpuStorageConfig, GpuAgentStorage};

// Production configuration (uses /magikdev/gpu)
let config = GpuStorageConfig::production();
let storage = GpuAgentStorage::new(config)?;

// Development configuration (uses local directory)
let config = GpuStorageConfig::development();
let storage = GpuAgentStorage::new(config)?;

// Custom path configuration
let config = GpuStorageConfig::with_base_path("/custom/gpu/path");
let storage = GpuAgentStorage::new(config)?;
```

### Advanced Configuration
```rust
let config = GpuStorageConfig {
    base_path: PathBuf::from("/magikdev/gpu"),
    cache_path: PathBuf::from("/magikdev/gpu/cache"),
    wal_path: PathBuf::from("/magikdev/gpu/wal"),
    enable_gpu_cache: true,      // Enable GPU memory caching
    cache_size_mb: 2048,         // 2GB GPU cache
    enable_compression: true,     // Enable data compression
    sync_interval_ms: 50,        // Sync every 50ms
};
```

## Usage Examples

### Storing GPU Agent Data
```rust
use gpu_agents::{GpuAgentData, GpuAgentStorage, GpuStorageConfig};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize storage
    let config = GpuStorageConfig::production();
    let storage = GpuAgentStorage::new(config)?;
    
    // Create agent data
    let agent = GpuAgentData {
        id: "agent_001".to_string(),
        state: vec![0.1; 256],      // Neural network state
        memory: vec![0.0; 128],     // Agent memory
        generation: 1,
        fitness: 0.85,
        metadata: HashMap::new(),
    };
    
    // Store agent
    storage.store_agent(&agent.id, &agent).await?;
    
    // Retrieve agent
    let retrieved = storage.retrieve_agent(&agent.id).await?;
    
    Ok(())
}
```

### Storing Knowledge Graphs
```rust
use gpu_agents::{GpuKnowledgeGraph, StorageGraphNode, StorageGraphEdge};

// Create knowledge graph
let graph = GpuKnowledgeGraph {
    nodes: vec![
        StorageGraphNode {
            id: "node_1".to_string(),
            embedding: vec![0.1; 768],
            metadata: HashMap::new(),
        },
    ],
    edges: vec![
        StorageGraphEdge {
            source: "node_1".to_string(),
            target: "node_2".to_string(),
            weight: 0.8,
            edge_type: "similarity".to_string(),
        },
    ],
};

// Store graph
storage.store_knowledge_graph("graph_001", &graph).await?;

// Retrieve graph
let retrieved = storage.retrieve_knowledge_graph("graph_001").await?;
```

### GPU Memory Mapping
```rust
use gpu_agents::SwarmData;

// Create swarm data
let swarm_data = SwarmData {
    agent_count: 10_000,
    state_dimension: 256,
    data: vec![0.0f32; 10_000 * 256],
};

// Map to GPU memory for zero-copy access
let gpu_handle = storage.map_to_gpu_memory("swarm_001", &swarm_data).await?;

println!("Mapped {} bytes to GPU memory", gpu_handle.size_bytes());
```

### Cache Management
```rust
// Pre-warm cache with frequently accessed agents
let hot_agents = vec!["agent_001", "agent_002", "agent_003"];
for agent_id in &hot_agents {
    storage.cache_agent(agent_id).await?;
}

// Get cache statistics
let stats = storage.cache_stats().await?;
println!("Cached agents: {}", stats.cached_agents);
println!("Cache hits: {}", stats.cache_hits);
println!("Cache misses: {}", stats.cache_misses);

// Retrieve from cache (fast path)
let agent = storage.retrieve_agent_cached("agent_001").await?;
```

## Performance Considerations

1. **NVMe Optimization**: The storage layer uses NVMe-optimized I/O patterns for maximum throughput
2. **GPU Cache**: Frequently accessed data is cached in GPU memory for microsecond-latency access
3. **Compression**: Optional compression reduces storage footprint for large agent swarms
4. **Write-Ahead Logging**: Ensures data durability with minimal performance impact

## Directory Structure

```
/magikdev/gpu/
├── nvme_storage/      # NVMe-optimized storage files
├── cache/             # GPU cache data
├── wal/               # Write-ahead logs
├── agent/             # Agent state storage
├── graph/             # Knowledge graph storage
└── swarm/             # Swarm data storage
```

## Environment Variables

- `EXORUST_GPU_STORAGE_PATH`: Override default storage path
- `EXORUST_GPU_CACHE_SIZE_MB`: Set GPU cache size in MB
- `EXORUST_STORAGE_COMPRESSION`: Enable/disable compression (true/false)

## Best Practices

1. **Use Production Config**: Always use `GpuStorageConfig::production()` for deployed systems
2. **Monitor Cache Hit Rate**: Aim for >90% cache hit rate for hot data
3. **Batch Operations**: Group storage operations for better throughput
4. **Regular Sync**: Let the storage layer handle sync intervals automatically
5. **Path Permissions**: Ensure `/magikdev/gpu` has appropriate read/write permissions

## Troubleshooting

### Permission Denied
```bash
# Ensure directory exists and has correct permissions
sudo mkdir -p /magikdev/gpu
sudo chown $USER:$USER /magikdev/gpu
chmod 755 /magikdev/gpu
```

### Storage Full
```bash
# Check disk usage
df -h /magikdev/gpu

# Clean old data
find /magikdev/gpu -name "*.old" -mtime +7 -delete
```

### Performance Issues
- Check cache hit rate using `storage.cache_stats()`
- Increase cache size if hit rate is low
- Enable compression for large datasets
- Verify NVMe device performance with `nvme-cli`