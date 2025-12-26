# Knowledge Graph

GPU-accelerated graph operations for semantic knowledge representation and agent memory.

## Overview

The `knowledge-graph` crate provides a high-performance graph database optimized for GPU acceleration. It serves as the semantic memory system for StratoSwarm agents, enabling them to store, query, and reason over complex relationships. The crate leverages GPU parallelism for graph algorithms while maintaining compatibility with CPU-only deployments.

## Features

- **GPU-Accelerated Algorithms**: Parallel BFS, DFS, PageRank, community detection
- **Semantic Queries**: Natural language graph queries via embeddings
- **Evolution Tracking**: Track how knowledge evolves over time
- **Pattern Mining**: Discover recurring patterns and relationships
- **Memory Integration**: Seamless integration with agent memory systems
- **Scalable Storage**: Efficient storage for billions of edges
- **Real-time Updates**: Lock-free concurrent updates
- **GPU Kernels**: Custom PTX kernels for maximum performance

## Usage

### Basic Graph Operations

```rust
use knowledge_graph::{KnowledgeGraph, Node, Edge, NodeType};

// Create knowledge graph
let mut graph = KnowledgeGraph::new()
    .with_gpu_device(0)
    .with_index_type(IndexType::Semantic);

// Add nodes
let agent_node = Node::new("agent_001")
    .with_type(NodeType::Agent)
    .with_properties(json!({
        "name": "Explorer",
        "capabilities": ["synthesis", "analysis"]
    }));
graph.add_node(agent_node)?;

// Add relationships
let edge = Edge::new("agent_001", "knowledge_001")
    .with_type("discovered")
    .with_weight(0.95)
    .with_timestamp(Utc::now());
graph.add_edge(edge)?;

// Query neighbors
let neighbors = graph.neighbors("agent_001", 2)?; // 2-hop neighbors
```

### Semantic Queries

```rust
use knowledge_graph::{SemanticQuery, QueryBuilder};

// Natural language query
let query = SemanticQuery::from_text(
    "Find all agents that discovered optimization patterns"
)?;

let results = graph.semantic_search(query).await?;

// Structured query with embeddings
let query = QueryBuilder::new()
    .match_nodes(NodeType::Agent)
    .connected_to(NodeType::Pattern)
    .with_edge_type("discovered")
    .where_property("pattern.type", "optimization")
    .limit(10)
    .build();

let results = graph.execute_query(query)?;
```

### GPU-Accelerated Algorithms

```rust
use knowledge_graph::algorithms::{PageRank, CommunityDetection, ShortestPath};

// PageRank computation on GPU
let pagerank = PageRank::new()
    .damping_factor(0.85)
    .iterations(100)
    .tolerance(1e-6);

let scores = graph.compute_pagerank_gpu(pagerank).await?;

// Community detection
let communities = graph.detect_communities_gpu(
    CommunityDetection::Louvain { resolution: 1.0 }
).await?;

// Shortest path with GPU acceleration
let path = graph.shortest_path_gpu("start_node", "end_node").await?;
```

### Evolution Tracking

```rust
use knowledge_graph::{EvolutionTracker, TemporalQuery};

// Enable evolution tracking
let mut tracker = EvolutionTracker::new(&graph);
tracker.enable_versioning();

// Track changes over time
graph.add_node_with_tracking(new_node, &tracker)?;

// Query historical state
let historical = TemporalQuery::new()
    .at_timestamp(past_timestamp)
    .node("agent_001");

let past_state = tracker.query_temporal(historical)?;

// Analyze evolution patterns
let evolution_patterns = tracker.analyze_evolution(
    "agent_001",
    TimeRange::last_days(30)
)?;
```

### Pattern Mining

```rust
use knowledge_graph::patterns::{PatternMiner, FrequentPattern};

// Mine frequent subgraphs
let miner = PatternMiner::new()
    .min_support(0.1)  // 10% support threshold
    .max_pattern_size(5)
    .use_gpu(true);

let patterns = miner.mine_patterns(&graph).await?;

// Find specific patterns
let synthesis_patterns = miner.find_patterns()
    .involving_edge_type("synthesized")
    .with_min_frequency(10)
    .execute(&graph)?;
```

### Memory Integration

```rust
use knowledge_graph::memory::{MemoryGraph, MemoryType};

// Create memory-integrated graph
let memory_graph = MemoryGraph::new(graph)
    .with_memory_types(vec![
        MemoryType::Episodic,
        MemoryType::Semantic,
        MemoryType::Procedural
    ]);

// Store agent memories
memory_graph.store_memory(
    agent_id,
    MemoryType::Episodic,
    json!({
        "event": "discovered_optimization",
        "context": context_data,
        "timestamp": Utc::now()
    })
)?;

// Retrieve relevant memories
let memories = memory_graph.recall(
    agent_id,
    "optimization techniques",
    limit: 10
)?;
```

## GPU Kernels

The crate includes optimized GPU kernels:

### Graph Traversal Kernels
- `bfs_kernel.ptx`: Parallel breadth-first search
- `pagerank_kernel.ptx`: PageRank computation
- `community_kernel.ptx`: Community detection
- `path_kernel.ptx`: Shortest path algorithms

### Custom Kernel Development

```rust
use knowledge_graph::kernels::{KernelBuilder, GraphKernel};

// Build custom kernel
let kernel = KernelBuilder::new()
    .source(include_str!("custom_algorithm.cu"))
    .function_name("custom_graph_algorithm")
    .block_size(256)
    .build()?;

// Execute on graph
let result = graph.execute_custom_kernel(kernel, params)?;
```

## Performance Optimization

### Batch Operations

```rust
// Batch node additions for better performance
let nodes = vec![node1, node2, node3, /* ... */];
graph.add_nodes_batch(nodes)?;

// Batch edge additions
let edges = vec![edge1, edge2, edge3, /* ... */];
graph.add_edges_batch(edges)?;
```

### Index Configuration

```rust
// Configure indices for query optimization
graph.create_index(IndexType::BTree, "node.type")?;
graph.create_index(IndexType::Hash, "edge.timestamp")?;
graph.create_index(IndexType::Semantic, "node.embedding")?;
```

### Memory Management

```rust
// Pre-allocate GPU memory
graph.reserve_gpu_memory(
    nodes: 1_000_000,
    edges: 10_000_000
)?;

// Enable memory pooling
graph.enable_memory_pooling(
    pool_size_mb: 1024,
    growth_factor: 1.5
)?;
```

## Benchmarks

Run performance benchmarks:

```bash
# All benchmarks
cargo bench

# Specific benchmarks
cargo bench --bench graph_traversal
cargo bench --bench pagerank_gpu
cargo bench --bench semantic_search
```

Typical performance:
- **BFS Traversal**: 10M+ edges/second on GPU
- **PageRank**: 100M+ edge updates/second
- **Semantic Search**: <10ms for 1M nodes
- **Pattern Mining**: 1000+ patterns/second

## Storage Backend

The graph supports multiple storage backends:

```rust
// In-memory storage (default)
let graph = KnowledgeGraph::new();

// Persistent storage with NVMe optimization
let graph = KnowledgeGraph::with_storage(
    StorageBackend::NVMe("/mnt/nvme/graph.db")
)?;

// Distributed storage
let graph = KnowledgeGraph::with_storage(
    StorageBackend::Distributed(cluster_config)
)?;
```

## Configuration

```toml
[knowledge_graph]
# GPU settings
gpu_enabled = true
gpu_device_id = 0
gpu_memory_pool_mb = 2048

# Storage settings
storage_backend = "nvme"
storage_path = "/mnt/nvme/knowledge.db"
cache_size_mb = 1024

# Query settings
max_query_time_ms = 5000
default_limit = 1000
enable_query_cache = true

# Semantic search
embedding_model = "all-MiniLM-L6-v2"
embedding_cache_size = 10000
similarity_threshold = 0.7
```

## Testing

Comprehensive test coverage:

```bash
# Unit tests
cargo test

# Integration tests
cargo test --test integration

# GPU tests (requires CUDA)
cargo test --features gpu_required

# Scaling tests
cargo test --test scaling -- --test-threads=1
```

## Coverage

Current test coverage: 85% (Good)

Well-tested areas:
- Basic graph operations
- Query execution
- Pattern mining algorithms
- Memory integration

GPU-specific tests:
- Kernel execution
- Memory transfers
- Algorithm correctness

## Examples

See the `examples/` directory:

```bash
# Basic usage
cargo run --example basic_graph

# Semantic search
cargo run --example semantic_search

# Evolution tracking
cargo run --example evolution_tracking

# GPU algorithms
cargo run --example gpu_algorithms
```

## Integration

Core component for:
- `agent-core`: Agent semantic memory
- `gpu-agents`: GPU-accelerated reasoning
- `synthesis`: Pattern-based code generation
- `evolution-engines`: Knowledge evolution tracking

## Future Enhancements

- Graph neural network integration
- Distributed graph processing
- Advanced reasoning capabilities
- Quantum-inspired algorithms
- Natural language graph construction

## License

MIT